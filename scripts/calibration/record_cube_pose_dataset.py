#!/usr/bin/env python3
"""Record raw RealSense RGB bursts for offline cube-position evaluation.

The program opens only the D435 color stream. It never connects to or commands
the Franka.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_live_pose_module() -> Any:
    path = REPO_ROOT / "scripts" / "calibration" / "live_apriltag_cube_pose.py"
    spec = importlib.util.spec_from_file_location("_live_apriltag_cube_pose", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LIVE_POSE = _load_live_pose_module()
DEFAULT_CAMERA_CONFIG = LIVE_POSE.DEFAULT_CAMERA_CONFIG


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--burst-frames", type=int, default=60)
    parser.add_argument("--max-positions", type=int, default=0, help="0 means record until q/Ctrl-C")
    parser.add_argument("--headless", action="store_true", help="Record bursts without an OpenCV window")
    parser.add_argument(
        "--settle-s",
        type=float,
        default=2.0,
        help="Delay before each headless burst so the cube can settle",
    )
    parser.add_argument(
        "--output-dir",
        help="Defaults to REPO_ROOT/eval_dataset",
    )
    return parser


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write image: {path}")


def _draw_status(
    bgr: np.ndarray,
    position_index: int,
    captured_in_burst: int,
    burst_frames: int,
    total_frames: int,
) -> None:
    lines = [
        "s: save burst   q/Esc: quit",
        f"next position: {position_index:03d}",
        f"burst: {captured_in_burst}/{burst_frames}",
        f"saved frames: {total_frames}",
    ]
    y = 24
    for line in lines:
        cv2.putText(bgr, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main() -> int:
    args = build_parser().parse_args()
    if args.burst_frames <= 0:
        raise ValueError("--burst-frames must be positive")
    if args.max_positions < 0:
        raise ValueError("--max-positions must be non-negative")
    if args.settle_s < 0.0:
        raise ValueError("--settle-s must be non-negative")

    camera_config = LIVE_POSE.load_eye_to_hand_config(args.camera_config).camera
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "eval_dataset"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    stream_config = rs.config()
    stream_config.enable_device(camera_config.serial)
    stream_config.enable_stream(
        rs.stream.color,
        camera_config.width,
        camera_config.height,
        rs.format.rgb8,
        camera_config.fps,
    )

    profile = pipeline.start(stream_config)
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    metadata = {
        "kind": "cube_pose_dataset",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "camera_config": {
            "serial": camera_config.serial,
            "width": camera_config.width,
            "height": camera_config.height,
            "fps": camera_config.fps,
            "warmup_frames": camera_config.warmup_frames,
        },
        "camera_intrinsics": LIVE_POSE._intrinsics_dict(intrinsics),
        "manifest": "manifest.csv",
        "notes": [
            "Images are raw RealSense RGB frames.",
            "Each position_* directory is intended to contain one static cube placement.",
        ],
    }
    _atomic_json(output_dir / "metadata.json", metadata)

    fieldnames = [
        "position_id",
        "frame_index",
        "image_path",
        "host_time_s",
        "camera_time_ms",
    ]
    position_index = 0
    total_saved = 0
    capture_remaining = 0
    captured_in_burst = 0
    global_frame_index = 0

    print("[record] camera only; no Franka connection or motion commands")
    print(f"[record] dataset: {output_dir}")
    print(f"[record] burst_frames={args.burst_frames}")
    if args.headless:
        print("[record] headless mode: records --max-positions bursts, one after each settle delay")
    else:
        print("[record] keys: s save current placement burst, q/Esc quit")

    try:
        for _ in range(max(0, camera_config.warmup_frames)):
            pipeline.wait_for_frames()

        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            if args.headless:
                if args.max_positions <= 0:
                    raise ValueError("--headless requires --max-positions")
                for position_index in range(args.max_positions):
                    if args.settle_s:
                        print(f"[record] position_{position_index:03d}: settling {args.settle_s:.1f}s")
                        time.sleep(args.settle_s)
                    for frame_index in range(args.burst_frames):
                        frames = pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        if not color_frame:
                            continue
                        rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                        rel_path = Path("positions") / f"position_{position_index:03d}" / "rgb" / f"frame_{frame_index:06d}.png"
                        _save_rgb(output_dir / rel_path, rgb)
                        writer.writerow(
                            {
                                "position_id": f"position_{position_index:03d}",
                                "frame_index": frame_index,
                                "image_path": str(rel_path),
                                "host_time_s": time.time(),
                                "camera_time_ms": float(color_frame.get_timestamp()),
                            }
                        )
                        total_saved += 1
                    handle.flush()
                    print(f"[record] position_{position_index:03d}: saved {args.burst_frames} frames")
                return 0

            while args.max_positions == 0 or position_index < args.max_positions:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                if capture_remaining > 0:
                    position_id = f"position_{position_index:03d}"
                    rel_path = Path("positions") / position_id / "rgb" / f"frame_{captured_in_burst:06d}.png"
                    _save_rgb(output_dir / rel_path, rgb)
                    writer.writerow(
                        {
                            "position_id": position_id,
                            "frame_index": captured_in_burst,
                            "image_path": str(rel_path),
                            "host_time_s": time.time(),
                            "camera_time_ms": float(color_frame.get_timestamp()),
                        }
                    )
                    total_saved += 1
                    captured_in_burst += 1
                    capture_remaining -= 1
                    if capture_remaining == 0:
                        handle.flush()
                        print(f"[record] {position_id}: saved {captured_in_burst} frames")
                        position_index += 1
                        captured_in_burst = 0

                _draw_status(
                    bgr,
                    position_index,
                    captured_in_burst,
                    args.burst_frames,
                    total_saved,
                )
                cv2.imshow("Record cube pose dataset", bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and capture_remaining == 0:
                    capture_remaining = args.burst_frames
                    captured_in_burst = 0
                    print(f"[record] position_{position_index:03d}: capturing {args.burst_frames} frames")
                global_frame_index += 1
    except KeyboardInterrupt:
        print("\n[record] interrupted")
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[record] manifest: {manifest_path}")
    print(f"[record] saved frames: {total_saved}")
    return 0 if total_saved > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
