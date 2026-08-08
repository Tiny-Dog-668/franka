#!/usr/bin/env python3
"""Validate the RMA student's image-based object XYZ prediction.

The deployed RMA policy only returns actions from ``forward``.  Its visual
adaptation head nevertheless predicts normalized object XYZ internally.  This
script reads that head without connecting to or moving the robot, converts the
prediction back to robot-root metres, and optionally compares it with measured
ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (  # noqa: E402
    BundleCameraConfig,
    RealSenseRGBCamera,
    _apply_camera_crop,
    load_bundle_config,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0802_dr_simactuator_gpu.json"
DEFAULT_MODEL = REPO_ROOT / "checkpoint" / "0802_DR_gpu" / "rma_student_dr_sim2real_gpu.pt"
POSITION_KEYS = (
    "rma_cube_pos",
    "cube_pos_local",
    "pre_cube_pos_local",
    "cube_position",
    "object_position",
    "object_pose",
)


@dataclass
class Sample:
    name: str
    rgb: np.ndarray
    ground_truth: np.ndarray | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_metadata_path(model_path: Path, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else model_path.with_suffix(".json")


def _triplet(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Expected three finite XYZ values, got {result.tolist()}")
    return result


class RMAObjectPositionPredictor:
    """Expose position/contact predictions from v5 and portable v6 exports."""

    def __init__(self, model_path: Path, metadata_path: Path, device: str) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("kind") != "tacex_rma_student_torchscript":
            raise ValueError("Model metadata is not a TacEx RMA student export")
        components = self.metadata.get("actor_contract", {}).get("object_pose_components")
        if components != "position_xyz_only":
            raise ValueError(
                "This validator requires actor_contract.object_pose_components="
                f"'position_xyz_only', got {components!r}"
            )
        expected_hash = self.metadata.get("torchscript_sha256")
        actual_hash = _sha256(model_path)
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                "TorchScript SHA-256 differs from metadata; refusing to evaluate a mismatched model"
            )

        signature = self.metadata.get("input_signature", {}).get("wrist_rgb")
        if not isinstance(signature, list) or len(signature) != 3:
            raise ValueError("Metadata is missing input_signature.wrist_rgb[H,W,C]")
        self.height, self.width, channels = (int(value) for value in signature)
        if channels != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {channels}")

        normalization = self.metadata.get("normalization", {})
        self.position_center = _triplet(normalization.get("cube_position_center", []))
        self.position_scale = _triplet(normalization.get("cube_position_scale", []))
        if np.any(self.position_scale <= 0.0):
            raise ValueError("cube_position_scale must contain positive values")

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}) but torch.cuda.is_available() is false")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.exported_methods = set(self.model._c._method_names())

    def _predict_adaptation(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if "predict_adaptation" in self.exported_methods:
            outputs = self.model.predict_adaptation(rgb)
        else:
            # Older traced exports retained the relevant submodules but did not
            # export predict_adaptation as a public TorchScript method.
            image = rgb.to(torch.float32).permute(0, 3, 1, 2) / 255.0
            image = (image - self.model.image_mean) / self.model.image_std
            outputs = self.model.adaptation_head(self.model.vision_encoder(image))

        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise RuntimeError("RMA adaptation head did not return its expected two tensors")
        by_dim = {int(tensor.shape[-1]): tensor for tensor in outputs}
        if set(by_dim) != {2, 3}:
            shapes = [tuple(tensor.shape) for tensor in outputs]
            raise RuntimeError(f"Expected position[3] and contact[2], got {shapes}")
        return by_dim[3], by_dim[2]

    def predict(self, images: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        images = np.asarray(images, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected image batch [N,{self.height},{self.width},3], got {images.shape}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        positions: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                writable = np.array(images[start : start + batch_size], copy=True, order="C")
                tensor = torch.as_tensor(writable, dtype=torch.uint8, device=self.device)
                normalized, contact_logits = self._predict_adaptation(tensor)
                position = normalized * torch.as_tensor(
                    self.position_scale, dtype=normalized.dtype, device=self.device
                ) + torch.as_tensor(
                    self.position_center, dtype=normalized.dtype, device=self.device
                )
                positions.append(position.detach().cpu().numpy())
                contacts.append(torch.sigmoid(contact_logits).detach().cpu().numpy())
        return np.concatenate(positions), np.concatenate(contacts)


def _prepare_image(
    rgb: np.ndarray,
    camera: BundleCameraConfig,
    output_width: int,
    output_height: int,
    force_raw: bool,
) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {rgb.shape}")
    if not force_raw and rgb.shape[:2] == (output_height, output_width):
        return np.array(rgb, copy=True, order="C")
    return _apply_camera_crop(rgb, camera, output_width, output_height)


def _ground_truth_from_npz(data: np.lib.npyio.NpzFile, count: int) -> np.ndarray | None:
    for key in POSITION_KEYS:
        if key not in data.files:
            continue
        values = np.asarray(data[key], dtype=np.float32)
        if key == "object_pose" and values.shape[-1] >= 3:
            values = values[..., :3]
        values = values.reshape(-1, 3)
        if len(values) == 1 and count > 1:
            values = np.repeat(values, count, axis=0)
        if len(values) != count:
            raise ValueError(f"NPZ key {key!r} has {len(values)} poses for {count} images")
        return values
    return None


def _samples_from_input(
    paths: list[str],
    camera: BundleCameraConfig,
    width: int,
    height: int,
    force_raw: bool,
    fixed_ground_truth: np.ndarray | None,
) -> list[Sample]:
    samples: list[Sample] = []
    expanded: list[Path] = []
    image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npz"}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            expanded.extend(
                child for child in sorted(path.rglob("*")) if child.suffix.lower() in image_suffixes
            )
        else:
            expanded.append(path)

    for path in expanded:
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                if "wrist_rgb" not in data.files:
                    raise KeyError(f"NPZ does not contain wrist_rgb: {path}")
                images = np.asarray(data["wrist_rgb"], dtype=np.uint8)
                if images.ndim == 3:
                    images = images[None]
                elif images.ndim > 4 and images.shape[-1] == 3:
                    images = images.reshape(-1, *images.shape[-3:])
                if images.ndim != 4:
                    raise ValueError(
                        f"NPZ wrist_rgb must end in [H,W,3], got {images.shape}: {path}"
                    )
                embedded_ground_truth = _ground_truth_from_npz(data, len(images))
                for index, image in enumerate(images):
                    ground_truth = (
                        fixed_ground_truth
                        if fixed_ground_truth is not None
                        else None if embedded_ground_truth is None else embedded_ground_truth[index]
                    )
                    samples.append(
                        Sample(
                            name=f"{path.name}:{index}",
                            rgb=_prepare_image(image, camera, width, height, force_raw),
                            ground_truth=ground_truth,
                        )
                    )
        else:
            image = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            samples.append(
                Sample(
                    name=str(path),
                    rgb=_prepare_image(image, camera, width, height, force_raw),
                    ground_truth=fixed_ground_truth,
                )
            )
    return samples


def _samples_from_manifest(
    path: Path,
    camera: BundleCameraConfig,
    width: int,
    height: int,
    force_raw: bool,
) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            raise ValueError("Manifest must contain an 'image' column")
        xyz_columns = ("x_m", "y_m", "z_m") if "x_m" in reader.fieldnames else ("x", "y", "z")
        if not all(column in reader.fieldnames for column in xyz_columns):
            raise ValueError("Manifest must contain x_m,y_m,z_m (or x,y,z) columns")
        for row_number, row in enumerate(reader, start=2):
            image_path = Path(row["image"]).expanduser()
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest row {row_number} image not found: {image_path}")
            ground_truth = _triplet(float(row[column]) for column in xyz_columns)
            image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            samples.append(
                Sample(
                    name=str(image_path.resolve()),
                    rgb=_prepare_image(image, camera, width, height, force_raw),
                    ground_truth=ground_truth,
                )
            )
    return samples


def _samples_from_realsense(
    camera_config: BundleCameraConfig,
    width: int,
    height: int,
    frame_count: int,
    ground_truth: np.ndarray | None,
) -> list[Sample]:
    if frame_count <= 0:
        raise ValueError("--frames must be positive")
    camera = RealSenseRGBCamera(camera_config, output_width=width, output_height=height)
    try:
        return [
            Sample(name=f"realsense:{index:04d}", rgb=camera.read(), ground_truth=ground_truth)
            for index in range(frame_count)
        ]
    finally:
        camera.close()


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else math.nan


def _build_results(
    samples: list[Sample],
    positions: np.ndarray,
    contacts: np.ndarray,
    threshold_m: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    errors: list[np.ndarray] = []
    for sample, position, contact in zip(samples, positions, contacts):
        row: dict[str, object] = {
            "sample": sample.name,
            "pred_x_m": float(position[0]),
            "pred_y_m": float(position[1]),
            "pred_z_m": float(position[2]),
            "contact_left_probability": float(contact[0]),
            "contact_right_probability": float(contact[1]),
        }
        if sample.ground_truth is not None:
            error = position - sample.ground_truth
            error_3d = float(np.linalg.norm(error))
            errors.append(error)
            row.update(
                {
                    "gt_x_m": float(sample.ground_truth[0]),
                    "gt_y_m": float(sample.ground_truth[1]),
                    "gt_z_m": float(sample.ground_truth[2]),
                    "error_x_m": float(error[0]),
                    "error_y_m": float(error[1]),
                    "error_z_m": float(error[2]),
                    "error_3d_m": error_3d,
                    "within_threshold": error_3d <= threshold_m,
                }
            )
        rows.append(row)

    prediction_std = np.std(positions, axis=0)
    summary: dict[str, object] = {
        "sample_count": len(samples),
        "ground_truth_count": len(errors),
        "position_frame": "robot_root",
        "pose_components": "position_xyz_only",
        "prediction_mean_m": np.mean(positions, axis=0).tolist(),
        "prediction_std_m": prediction_std.tolist(),
        "threshold_mm": threshold_m * 1000.0,
        "verdict": "NO_GROUND_TRUTH",
    }
    if errors:
        error_array = np.asarray(errors, dtype=np.float64)
        norms = np.linalg.norm(error_array, axis=1)
        rmse_xyz = np.sqrt(np.mean(np.square(error_array), axis=0))
        rmse_3d = float(np.sqrt(np.mean(np.square(norms))))
        summary.update(
            {
                "mae_xyz_mm": (1000.0 * np.mean(np.abs(error_array), axis=0)).tolist(),
                "rmse_xyz_mm": (1000.0 * rmse_xyz).tolist(),
                "rmse_3d_mm": 1000.0 * rmse_3d,
                "median_error_3d_mm": 1000.0 * _percentile(norms, 50.0),
                "p95_error_3d_mm": 1000.0 * _percentile(norms, 95.0),
                "max_error_3d_mm": 1000.0 * float(np.max(norms)),
                "within_threshold_fraction": float(np.mean(norms <= threshold_m)),
                "verdict": "PASS" if rmse_3d <= threshold_m else "FAIL",
            }
        )
    return rows, summary


def _write_outputs(
    output_dir: Path,
    samples: list[Sample],
    rows: list[dict[str, object]],
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    for index, (sample, row) in enumerate(zip(samples, rows)):
        relative_path = Path("rgb") / f"sample_{index:04d}.png"
        Image.fromarray(sample.rgb).save(output_dir / relative_path)
        row["rgb_path"] = str(relative_path)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _print_summary(summary: dict[str, object], output_dir: Path) -> None:
    mean = np.asarray(summary["prediction_mean_m"]) * 1000.0
    std = np.asarray(summary["prediction_std_m"]) * 1000.0
    print(f"samples: {summary['sample_count']} (with ground truth: {summary['ground_truth_count']})")
    print(f"frame/components: {summary['position_frame']} / {summary['pose_components']}")
    print(f"prediction mean XYZ: [{mean[0]:.2f}, {mean[1]:.2f}, {mean[2]:.2f}] mm")
    print(f"prediction jitter std: [{std[0]:.2f}, {std[1]:.2f}, {std[2]:.2f}] mm")
    if summary["verdict"] == "NO_GROUND_TRUTH":
        print("verdict: NO_GROUND_TRUTH (prediction only; correctness was not evaluated)")
    else:
        mae = np.asarray(summary["mae_xyz_mm"])
        rmse = np.asarray(summary["rmse_xyz_mm"])
        print(f"MAE XYZ: [{mae[0]:.2f}, {mae[1]:.2f}, {mae[2]:.2f}] mm")
        print(f"RMSE XYZ: [{rmse[0]:.2f}, {rmse[1]:.2f}, {rmse[2]:.2f}] mm")
        print(
            f"3D error RMSE/median/p95/max: {summary['rmse_3d_mm']:.2f} / "
            f"{summary['median_error_3d_mm']:.2f} / {summary['p95_error_3d_mm']:.2f} / "
            f"{summary['max_error_3d_mm']:.2f} mm"
        )
        print(
            f"within {summary['threshold_mm']:.2f} mm: "
            f"{100.0 * summary['within_threshold_fraction']:.1f}%"
        )
        print(f"verdict (3D RMSE <= threshold): {summary['verdict']}")
    print(f"results: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="RMA Student TorchScript path")
    parser.add_argument("--metadata", help="Metadata JSON; defaults to MODEL with .json suffix")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Deployment JSON supplying the exact camera serial/crop contract",
    )
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu or cuda:0")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", help="Image, NPZ, or directory inputs")
    source.add_argument("--manifest", help="CSV with image,x_m,y_m,z_m columns")
    source.add_argument("--realsense", action="store_true", help="Capture directly from RealSense")
    parser.add_argument(
        "--ground-truth",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Fixed measured object XYZ in robot_root metres",
    )
    parser.add_argument("--frames", type=int, default=30, help="RealSense frame count (default: 30)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--raw-inputs",
        action="store_true",
        help="Apply the config crop even when input images already have model resolution",
    )
    parser.add_argument(
        "--max-error-mm",
        type=float,
        default=10.0,
        help="3D RMSE pass threshold in mm (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        help="Result directory (default: runs/TIMESTAMP_rma_object_position_validation)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.max_error_mm) or args.max_error_mm <= 0.0:
        raise ValueError("--max-error-mm must be a finite positive value")

    model_path = Path(args.model).expanduser().resolve()
    metadata_path = _resolve_metadata_path(model_path, args.metadata)
    config = load_bundle_config(Path(args.config).expanduser().resolve())
    predictor = RMAObjectPositionPredictor(model_path, metadata_path, args.device)
    fixed_ground_truth = None if args.ground_truth is None else _triplet(args.ground_truth)

    if args.input:
        samples = _samples_from_input(
            args.input,
            config.camera,
            predictor.width,
            predictor.height,
            args.raw_inputs,
            fixed_ground_truth,
        )
    elif args.manifest:
        if fixed_ground_truth is not None:
            raise ValueError("--ground-truth cannot be combined with --manifest")
        samples = _samples_from_manifest(
            Path(args.manifest).expanduser().resolve(),
            config.camera,
            predictor.width,
            predictor.height,
            args.raw_inputs,
        )
    else:
        samples = _samples_from_realsense(
            config.camera,
            predictor.width,
            predictor.height,
            args.frames,
            fixed_ground_truth,
        )
    if not samples:
        raise RuntimeError("No validation samples were found")

    images = np.stack([sample.rgb for sample in samples])
    positions, contacts = predictor.predict(images, args.batch_size)
    rows, summary = _build_results(
        samples, positions, contacts, threshold_m=args.max_error_mm / 1000.0
    )
    summary.update(
        {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "device": args.device,
        }
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_rma_object_position_validation"
    )
    _write_outputs(output_dir, samples, rows, summary)
    _print_summary(summary, output_dir)
    return 0 if summary["verdict"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
