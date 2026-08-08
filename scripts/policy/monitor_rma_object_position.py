#!/usr/bin/env python3
"""Continuously display and log the RMA visual head's object-position prediction.

This program connects only to the configured RealSense camera. It does not
connect to or command the Franka arm or gripper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import RealSenseRGBCamera, load_bundle_config  # noqa: E402
from validate_rma_object_position import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_MODEL,
    RMAObjectPositionPredictor,
    _resolve_metadata_path,
)


class RunningXYZStatistics:
    def __init__(self) -> None:
        self.count = 0
        self.mean = np.zeros(3, dtype=np.float64)
        self._m2 = np.zeros(3, dtype=np.float64)

    def update(self, value: np.ndarray) -> None:
        point = np.asarray(value, dtype=np.float64).reshape(3)
        self.count += 1
        delta = point - self.mean
        self.mean += delta / self.count
        self._m2 += delta * (point - self.mean)

    @property
    def std(self) -> np.ndarray:
        if self.count < 2:
            return np.zeros(3, dtype=np.float64)
        return np.sqrt(self._m2 / self.count)


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _draw_live_overlay(
    cv2,
    rgb: np.ndarray,
    position: np.ndarray,
    ema: np.ndarray,
    contact: np.ndarray,
    inference_ms: float,
    controls: dict[str, bool | float | None],
) -> np.ndarray:
    """Return a BGR preview; do not modify the RGB array sent to the model."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    panel_height = min(bgr.shape[0], 96)
    cv2.rectangle(overlay, (0, 0), (bgr.shape[1] - 1, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.68, bgr, 0.32, 0.0, bgr)

    exposure = controls.get("exposure")
    gain = controls.get("gain")
    auto = controls.get("auto_exposure")
    exposure_text = "n/a" if exposure is None else f"{float(exposure):g}"
    gain_text = "n/a" if gain is None else f"{float(gain):g}"
    auto_text = "n/a" if auto is None else ("on" if auto else "off")
    lines = [
        (
            f"Pred XYZ mm: {position[0] * 1000:+.1f} "
            f"{position[1] * 1000:+.1f} {position[2] * 1000:+.1f}",
            (80, 255, 80),
        ),
        (
            f"EMA  XYZ mm: {ema[0] * 1000:+.1f} "
            f"{ema[1] * 1000:+.1f} {ema[2] * 1000:+.1f}",
            (0, 220, 255),
        ),
        (f"Contact L/R: {contact[0]:.3f} {contact[1]:.3f}", (255, 220, 80)),
        (f"Infer: {inference_ms:.1f} ms", (230, 230, 230)),
        (f"Exposure: {exposure_text}  Gain: {gain_text}  Auto: {auto_text}", (230, 230, 230)),
    ]
    for line_index, (line, color) in enumerate(lines):
        cv2.putText(
            bgr,
            line,
            (5, 15 + 18 * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )
    return bgr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="RMA Student TorchScript")
    parser.add_argument("--metadata", help="Metadata JSON; defaults to MODEL with .json suffix")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Deployment config supplying the camera serial and crop contract",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or cuda:0 (default: auto)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Number of frames; 0 runs until Ctrl+C (default: 0)",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N frames (default: 1)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA weight for the newest position, in (0,1] (default: 0.2)",
    )
    parser.add_argument(
        "--save-rgb-every",
        type=int,
        default=0,
        help="Save every Nth processed RGB frame; 0 disables image saving",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory; defaults to runs/TIMESTAMP_rma_object_position_live",
    )
    parser.add_argument(
        "--exposure",
        type=float,
        help="Manual RealSense color exposure value (for example: 80); disables auto exposure",
    )
    parser.add_argument(
        "--gain",
        type=float,
        help="Manual RealSense color gain value (for example: 64); disables auto exposure",
    )
    parser.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Explicitly enable RealSense color auto exposure",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the processed model input with live predictions (default: enabled)",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=3.0,
        help="Preview window scale relative to the model input (default: 3.0)",
    )
    parser.add_argument(
        "--window-name",
        default="RMA object position - model input",
        help="OpenCV preview window title",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.frames < 0:
        raise ValueError("--frames must be non-negative")
    if args.print_every < 1:
        raise ValueError("--print-every must be positive")
    if args.save_rgb_every < 0:
        raise ValueError("--save-rgb-every must be non-negative")
    if not math.isfinite(args.ema_alpha) or not 0.0 < args.ema_alpha <= 1.0:
        raise ValueError("--ema-alpha must be finite and in (0,1]")
    if args.exposure is not None and not math.isfinite(args.exposure):
        raise ValueError("--exposure must be finite")
    if args.gain is not None and not math.isfinite(args.gain):
        raise ValueError("--gain must be finite")
    if args.auto_exposure and (args.exposure is not None or args.gain is not None):
        raise ValueError("--auto-exposure cannot be combined with --exposure or --gain")
    if not math.isfinite(args.display_scale) or args.display_scale <= 0.0:
        raise ValueError("--display-scale must be finite and positive")

    model_path = Path(args.model).expanduser().resolve()
    metadata_path = _resolve_metadata_path(model_path, args.metadata)
    config_path = Path(args.config).expanduser().resolve()
    config = load_bundle_config(config_path)
    device = _resolve_device(args.device)
    predictor = RMAObjectPositionPredictor(model_path, metadata_path, device)

    # Warm TorchScript/CUDA before opening the camera so initialization does
    # not pollute the first live inference timing sample.
    dummy = np.zeros((1, predictor.height, predictor.width, 3), dtype=np.uint8)
    predictor.predict(dummy, batch_size=1)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rma_object_position_live"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    if args.save_rgb_every:
        rgb_dir.mkdir(exist_ok=True)

    fieldnames = [
        "frame",
        "wall_time_iso",
        "pred_x_m",
        "pred_y_m",
        "pred_z_m",
        "ema_x_m",
        "ema_y_m",
        "ema_z_m",
        "contact_left_probability",
        "contact_right_probability",
        "camera_read_ms",
        "inference_ms",
        "auto_exposure",
        "exposure",
        "gain",
    ]
    statistics = RunningXYZStatistics()
    ema: np.ndarray | None = None
    cv2 = None
    if args.display:
        import cv2 as cv2_module

        cv2 = cv2_module

    camera = RealSenseRGBCamera(
        config.camera,
        output_width=predictor.width,
        output_height=predictor.height,
        auto_exposure=True if args.auto_exposure else None,
        exposure=args.exposure,
        gain=args.gain,
    )
    controls = camera.get_color_controls()
    metadata = {
        "model": str(model_path),
        "metadata": str(metadata_path),
        "config": str(config_path),
        "device": device,
        "camera_serial": config.camera.serial,
        "camera_input": [config.camera.width, config.camera.height, config.camera.fps],
        "model_input": [predictor.height, predictor.width, 3],
        "crop": [
            config.camera.crop_left,
            config.camera.crop_top,
            config.camera.crop_width,
            config.camera.crop_height,
        ],
        "requested_camera_controls": {
            "auto_exposure": True if args.auto_exposure else None,
            "exposure": args.exposure,
            "gain": args.gain,
        },
        "actual_camera_controls": controls,
        "display": args.display,
        "display_scale": args.display_scale,
        "position_frame": "robot_root",
        "position_units": "metres",
        "ground_truth_available": False,
    }
    (output_dir / "session.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print("RMA live visual position monitor")
    print(f"Device: {device}")
    print(f"Camera: RealSense {config.camera.serial}")
    print(
        "Color controls: "
        f"auto_exposure={controls['auto_exposure']}, "
        f"exposure={controls['exposure']}, gain={controls['gain']}"
    )
    print("Output: cube-center XYZ in robot_root metres (prediction only; no ground truth)")
    print(f"Logs: {output_dir}")
    if args.display:
        print("Press q/Esc in the preview window, or Ctrl+C in the terminal, to stop.", flush=True)
    else:
        print("Press Ctrl+C to stop.", flush=True)

    interrupted = False
    stopped_by_window = False
    frame_index = 0
    try:
        with (output_dir / "predictions.csv").open(
            "w", encoding="utf-8", newline="", buffering=1
        ) as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            while args.frames == 0 or frame_index < args.frames:
                camera_started = time.perf_counter_ns()
                rgb = camera.read()
                camera_read_ms = (time.perf_counter_ns() - camera_started) / 1e6

                inference_started = time.perf_counter_ns()
                positions, contacts = predictor.predict(rgb[None], batch_size=1)
                inference_ms = (time.perf_counter_ns() - inference_started) / 1e6
                position = positions[0].astype(np.float64)
                contact = contacts[0].astype(np.float64)
                ema = position.copy() if ema is None else (
                    args.ema_alpha * position + (1.0 - args.ema_alpha) * ema
                )
                statistics.update(position)

                writer.writerow(
                    {
                        "frame": frame_index,
                        "wall_time_iso": datetime.now().astimezone().isoformat(),
                        "pred_x_m": float(position[0]),
                        "pred_y_m": float(position[1]),
                        "pred_z_m": float(position[2]),
                        "ema_x_m": float(ema[0]),
                        "ema_y_m": float(ema[1]),
                        "ema_z_m": float(ema[2]),
                        "contact_left_probability": float(contact[0]),
                        "contact_right_probability": float(contact[1]),
                        "camera_read_ms": camera_read_ms,
                        "inference_ms": inference_ms,
                        "auto_exposure": controls["auto_exposure"],
                        "exposure": controls["exposure"],
                        "gain": controls["gain"],
                    }
                )

                if args.save_rgb_every and frame_index % args.save_rgb_every == 0:
                    from PIL import Image

                    Image.fromarray(rgb).save(rgb_dir / f"frame_{frame_index:06d}.png")

                if frame_index % args.print_every == 0:
                    print(
                        f"frame={frame_index:06d} "
                        f"xyz_m=[{position[0]:+.4f}, {position[1]:+.4f}, {position[2]:+.4f}] "
                        f"ema_m=[{ema[0]:+.4f}, {ema[1]:+.4f}, {ema[2]:+.4f}] "
                        f"contact=[{contact[0]:.3f}, {contact[1]:.3f}] "
                        f"camera={camera_read_ms:.2f}ms infer={inference_ms:.2f}ms",
                        flush=True,
                    )
                frame_index += 1

                if cv2 is not None:
                    preview = _draw_live_overlay(
                        cv2,
                        rgb,
                        position,
                        ema,
                        contact,
                        inference_ms,
                        controls,
                    )
                    if args.display_scale != 1.0:
                        preview = cv2.resize(
                            preview,
                            None,
                            fx=args.display_scale,
                            fy=args.display_scale,
                            interpolation=cv2.INTER_NEAREST,
                        )
                    cv2.imshow(args.window_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    window_closed = False
                    try:
                        window_closed = (
                            cv2.getWindowProperty(args.window_name, cv2.WND_PROP_VISIBLE) < 1
                        )
                    except cv2.error:
                        window_closed = True
                    if key in (ord("q"), ord("Q"), 27) or window_closed:
                        stopped_by_window = True
                        print("Stopping live monitor from preview window...", flush=True)
                        break
    except KeyboardInterrupt:
        interrupted = True
        print("\nStopping live monitor...", flush=True)
    finally:
        camera.close()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except cv2.error:
                pass

    summary = {
        **metadata,
        "frame_count": statistics.count,
        "prediction_mean_m": statistics.mean.tolist(),
        "prediction_std_m": statistics.std.tolist(),
        "stopped_by_keyboard_interrupt": interrupted,
        "stopped_by_preview_window": stopped_by_window,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        "Prediction mean XYZ m: "
        f"[{statistics.mean[0]:+.4f}, {statistics.mean[1]:+.4f}, {statistics.mean[2]:+.4f}]"
    )
    print(
        "Prediction std XYZ mm: "
        f"[{statistics.std[0] * 1000:.2f}, {statistics.std[1] * 1000:.2f}, "
        f"{statistics.std[2] * 1000:.2f}]"
    )
    print(f"Results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
