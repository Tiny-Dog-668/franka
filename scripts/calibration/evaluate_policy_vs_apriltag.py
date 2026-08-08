#!/usr/bin/env python3
"""Evaluate policy cube-position predictions against offline AprilTag poses."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import defaultdict
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


def _parse_scales(value: str) -> list[float]:
    scales = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not scales or any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("--detection-scales must contain positive finite values")
    return scales


def _camera_matrix(metadata: dict[str, Any]) -> np.ndarray:
    intrinsics = metadata.get("camera_intrinsics", {})
    required = ("fx", "fy", "cx", "cy")
    if not all(key in intrinsics for key in required):
        raise ValueError("Dataset metadata.json is missing camera_intrinsics fx/fy/cx/cy")
    return np.asarray(
        [
            [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
            [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _distortion(metadata: dict[str, Any]) -> np.ndarray:
    coefficients = metadata.get("camera_intrinsics", {}).get("distortion_coefficients", [])
    if not coefficients:
        return np.zeros(5, dtype=np.float64)
    return np.asarray(coefficients, dtype=np.float64).reshape(-1)


def _gray_variants(rgb: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    variants = [("gray", gray)]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(("clahe", clahe.apply(gray)))
    kernel = np.asarray([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    variants.append(("sharp", cv2.filter2D(gray, -1, kernel)))
    return variants


def _corner_area(corners: np.ndarray) -> float:
    return float(abs(cv2.contourArea(np.asarray(corners, dtype=np.float32).reshape(4, 2))))


def _make_detector(family: str) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(LIVE_POSE.APRILTAG_FAMILIES[family])
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def _detect_tag_pose(
    rgb: np.ndarray,
    detector: Any,
    marker_id: int,
    scales: list[float],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_length_m: float,
    max_reprojection_px: float,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for variant_name, gray in _gray_variants(rgb):
        for scale in scales:
            if scale == 1.0:
                detection_gray = gray
            else:
                detection_gray = cv2.resize(
                    gray,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            corners, ids, _ = detector.detectMarkers(detection_gray)
            if ids is None:
                continue
            matches = np.flatnonzero(ids.reshape(-1) == marker_id)
            for match in matches:
                raw_corners = np.asarray(corners[int(match)], dtype=np.float64) / scale
                try:
                    camera_t_tag, rvec, reprojection = LIVE_POSE.solve_tag_pose(
                        raw_corners,
                        camera_matrix,
                        distortion,
                        marker_length_m,
                    )
                except RuntimeError:
                    continue
                if reprojection > max_reprojection_px:
                    continue
                candidate = {
                    "camera_T_tag": camera_t_tag,
                    "rvec": rvec,
                    "reprojection_rms_px": reprojection,
                    "area_px2": _corner_area(raw_corners),
                    "detection_scale": scale,
                    "variant": variant_name,
                }
                if best is None or reprojection < best["reprojection_rms_px"]:
                    best = candidate
    return best


def _read_manifest(dataset_dir: Path, manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    if "image_path" not in rows[0]:
        raise ValueError("Manifest must contain image_path")
    for row in rows:
        image_path = Path(row["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = dataset_dir / image_path
        row["_absolute_image_path"] = str(image_path.resolve())
        if "position_id" not in row or not row["position_id"]:
            row["position_id"] = image_path.parent.parent.name
    return rows


def _filter_rows(
    rows: list[dict[str, str]],
    frame_stride: int,
    max_frames_per_position: int,
) -> list[dict[str, str]]:
    if frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if max_frames_per_position < 0:
        raise ValueError("--max-frames-per-position must be non-negative")

    counts: dict[str, int] = defaultdict(int)
    filtered: list[dict[str, str]] = []
    for row in rows:
        try:
            frame_index = int(row.get("frame_index", len(filtered)))
        except ValueError:
            frame_index = len(filtered)
        if frame_index % frame_stride != 0:
            continue
        position_id = row["position_id"]
        if max_frames_per_position and counts[position_id] >= max_frames_per_position:
            continue
        filtered.append(row)
        counts[position_id] += 1
    return filtered


def _xyz(prefix: str, values: np.ndarray | None) -> dict[str, float]:
    if values is None:
        return {
            f"{prefix}_x_m": math.nan,
            f"{prefix}_y_m": math.nan,
            f"{prefix}_z_m": math.nan,
        }
    return {
        f"{prefix}_x_m": float(values[0]),
        f"{prefix}_y_m": float(values[1]),
        f"{prefix}_z_m": float(values[2]),
    }


def _stats(values: np.ndarray) -> dict[str, Any]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean_m": np.mean(values, axis=0).tolist(),
        "median_m": np.median(values, axis=0).tolist(),
        "std_m": np.std(values, axis=0).tolist(),
    }


def _error_stats(errors: list[np.ndarray]) -> dict[str, Any]:
    if not errors:
        return {"count": 0}
    array = np.asarray(errors, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    return {
        "count": int(len(array)),
        "bias_mean_m": np.mean(array, axis=0).tolist(),
        "bias_median_m": np.median(array, axis=0).tolist(),
        "mae_xyz_mm": (1000.0 * np.mean(np.abs(array), axis=0)).tolist(),
        "rmse_xyz_mm": (1000.0 * np.sqrt(np.mean(np.square(array), axis=0))).tolist(),
        "rmse_3d_mm": float(1000.0 * np.sqrt(np.mean(np.square(norms)))),
        "median_3d_mm": float(1000.0 * np.median(norms)),
        "p95_3d_mm": float(1000.0 * np.percentile(norms, 95.0)),
        "max_3d_mm": float(1000.0 * np.max(norms)),
    }


def _fit_affine(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any] | None:
    if len(predicted) < 4:
        return None
    design = np.concatenate([predicted, np.ones((len(predicted), 1))], axis=1)
    if np.linalg.matrix_rank(design) < 4:
        return None
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    matrix = coefficients[:3, :].T
    offset = coefficients[3, :]
    corrected = predicted @ matrix.T + offset
    residual = corrected - target
    norms = np.linalg.norm(residual, axis=1)
    return {
        "matrix": matrix.tolist(),
        "offset_m": offset.tolist(),
        "residual_rmse_3d_mm": float(1000.0 * np.sqrt(np.mean(np.square(norms)))),
        "residual_median_3d_mm": float(1000.0 * np.median(norms)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", help="Dataset directory from record_cube_pose_dataset.py")
    parser.add_argument("--manifest", help="CSV manifest; defaults to DATASET/manifest.csv")
    parser.add_argument("--calibration-report", default=str(LIVE_POSE.DEFAULT_CALIBRATION_REPORT))
    parser.add_argument("--policy-config", default=str(LIVE_POSE.DEFAULT_POLICY_CONFIG))
    parser.add_argument("--policy-model", help="TorchScript model path; defaults to policy config")
    parser.add_argument("--policy-metadata", help="Policy metadata JSON; defaults to policy config")
    parser.add_argument("--policy-device", default="auto")
    parser.add_argument("--family", choices=sorted(LIVE_POSE.APRILTAG_FAMILIES), default="tag36h11")
    parser.add_argument("--id", type=int, default=2, dest="marker_id")
    parser.add_argument("--marker-length-m", type=float, default=0.038)
    parser.add_argument(
        "--tag-to-object",
        nargs=3,
        type=float,
        default=(0.0, 0.0, -0.025),
        metavar=("X_M", "Y_M", "Z_M"),
    )
    parser.add_argument("--base-height-offset-m", type=float, default=0.020)
    parser.add_argument("--detection-scales", default="1,2,3,4")
    parser.add_argument("--max-reprojection-px", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=64, help="Policy inference batch size")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Evaluate every Nth frame within each recorded position",
    )
    parser.add_argument(
        "--max-frames-per-position",
        type=int,
        default=0,
        help="0 means no per-position cap after frame-stride filtering",
    )
    parser.add_argument("--output-dir", help="Defaults to DATASET/evaluation_TIMESTAMP")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dataset is None and args.manifest is None:
        raise ValueError("Provide DATASET or --manifest")
    if args.marker_length_m <= 0.0:
        raise ValueError("--marker-length-m must be positive")
    if args.max_reprojection_px <= 0.0:
        raise ValueError("--max-reprojection-px must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not math.isfinite(args.base_height_offset_m):
        raise ValueError("--base-height-offset-m must be finite")

    dataset_dir = Path(args.dataset).expanduser().resolve() if args.dataset else Path(args.manifest).expanduser().resolve().parent
    manifest_path = Path(args.manifest).expanduser().resolve() if args.manifest else dataset_dir / "manifest.csv"
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = _read_manifest(dataset_dir, manifest_path)
    original_row_count = len(rows)
    rows = _filter_rows(rows, args.frame_stride, args.max_frames_per_position)
    if not rows:
        raise ValueError("No rows remain after frame filtering")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else dataset_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix = _camera_matrix(metadata)
    distortion = _distortion(metadata)
    base_t_camera = LIVE_POSE._load_accepted_base_t_camera(
        Path(args.calibration_report).expanduser().resolve()
    )
    tag_to_object = LIVE_POSE._finite_triplet("--tag-to-object", args.tag_to_object)
    base_height_offset = np.asarray([0.0, 0.0, args.base_height_offset_m], dtype=np.float64)
    detector = _make_detector(args.family)
    scales = _parse_scales(args.detection_scales)

    policy_config = LIVE_POSE._load_policy_config(Path(args.policy_config).expanduser().resolve())
    model_path, policy_metadata_path, device = LIVE_POSE._resolve_policy_paths(args, policy_config)
    predictor = LIVE_POSE.RMAPolicyCubePredictor(model_path, policy_metadata_path, device)
    policy_camera_config = policy_config.get("camera", {})

    print(f"[eval] dataset: {dataset_dir}", flush=True)
    print(f"[eval] frames: {len(rows)} of {original_row_count}", flush=True)
    print(f"[eval] policy device: {device}", flush=True)

    rgbs: list[np.ndarray] = []
    policy_inputs: list[np.ndarray] = []
    for index, manifest_row in enumerate(rows, start=1):
        image_path = Path(manifest_row["_absolute_image_path"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgbs.append(rgb)
        policy_inputs.append(
            LIVE_POSE._policy_crop(
                rgb,
                policy_camera_config,
                predictor.width,
                predictor.height,
            )
        )
        if index % 100 == 0 or index == len(rows):
            print(f"[eval] loaded {index}/{len(rows)} images", flush=True)

    print("[eval] running batched policy prediction", flush=True)
    policy_positions, policy_contacts = predictor.predict_batch(
        np.stack(policy_inputs), batch_size=args.batch_size
    )

    frame_rows: list[dict[str, Any]] = []
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_errors: list[np.ndarray] = []

    print("[eval] running offline AprilTag detection", flush=True)
    for index, (manifest_row, rgb, policy_position, policy_contact) in enumerate(
        zip(rows, rgbs, policy_positions, policy_contacts),
        start=1,
    ):
        image_path = Path(manifest_row["_absolute_image_path"])

        detection = _detect_tag_pose(
            rgb,
            detector,
            args.marker_id,
            scales,
            camera_matrix,
            distortion,
            args.marker_length_m,
            args.max_reprojection_px,
        )

        tag_position: np.ndarray | None = None
        raw_tag_position: np.ndarray | None = None
        reprojection = math.nan
        area = math.nan
        detection_scale = math.nan
        detection_variant = ""
        if detection is not None:
            pose = LIVE_POSE.object_pose_in_base(
                base_t_camera,
                detection["camera_T_tag"],
                tag_to_object,
            )
            raw_tag_position = pose[:3, 3]
            tag_position = raw_tag_position + base_height_offset
            reprojection = float(detection["reprojection_rms_px"])
            area = float(detection["area_px2"])
            detection_scale = float(detection["detection_scale"])
            detection_variant = str(detection["variant"])

        error = None if tag_position is None else policy_position - tag_position
        if error is not None:
            frame_errors.append(error)

        result = {
            "position_id": manifest_row["position_id"],
            "frame_index": manifest_row.get("frame_index", ""),
            "image_path": str(image_path),
            "tag_found": detection is not None,
            "tag_reprojection_rms_px": reprojection,
            "tag_area_px2": area,
            "tag_detection_scale": detection_scale,
            "tag_detection_variant": detection_variant,
            "policy_contact_left": float(policy_contact[0]),
            "policy_contact_right": float(policy_contact[1]),
            **_xyz("policy_pred", policy_position),
            **_xyz("apriltag", tag_position),
            **_xyz("apriltag_raw", raw_tag_position),
            **_xyz("error_policy_minus_apriltag", error),
            "error_3d_m": math.nan if error is None else float(np.linalg.norm(error)),
        }
        frame_rows.append(result)
        by_position[manifest_row["position_id"]].append(result)
        if index % 25 == 0 or index == len(rows):
            print(f"[eval] processed {index}/{len(rows)}", flush=True)

    position_rows: list[dict[str, Any]] = []
    position_errors: list[np.ndarray] = []
    position_predicted: list[np.ndarray] = []
    position_targets: list[np.ndarray] = []
    for position_id, group in sorted(by_position.items()):
        policy_values = np.asarray(
            [[row["policy_pred_x_m"], row["policy_pred_y_m"], row["policy_pred_z_m"]] for row in group],
            dtype=np.float64,
        )
        tag_values = np.asarray(
            [
                [row["apriltag_x_m"], row["apriltag_y_m"], row["apriltag_z_m"]]
                for row in group
                if row["tag_found"]
            ],
            dtype=np.float64,
        )
        policy_median = np.median(policy_values, axis=0)
        tag_median = None if len(tag_values) == 0 else np.median(tag_values, axis=0)
        error = None if tag_median is None else policy_median - tag_median
        if error is not None:
            position_errors.append(error)
            position_predicted.append(policy_median)
            position_targets.append(tag_median)
        position_rows.append(
            {
                "position_id": position_id,
                "frame_count": len(group),
                "tag_detection_count": int(sum(1 for row in group if row["tag_found"])),
                "tag_detection_fraction": float(sum(1 for row in group if row["tag_found"]) / len(group)),
                **_xyz("policy_pred_median", policy_median),
                **_xyz("apriltag_median", tag_median),
                **_xyz("error_policy_minus_apriltag", error),
                "error_3d_m": math.nan if error is None else float(np.linalg.norm(error)),
            }
        )

    target_minus_prediction = [-error for error in position_errors]
    correction: dict[str, Any] = {
        "recommended_bias_from_position_medians_m": (
            np.median(np.asarray(target_minus_prediction), axis=0).tolist()
            if target_minus_prediction
            else None
        ),
        "meaning": "corrected_policy_xyz = policy_xyz + recommended_bias",
    }
    if position_predicted:
        predicted_array = np.asarray(position_predicted, dtype=np.float64)
        target_array = np.asarray(position_targets, dtype=np.float64)
        correction["affine_from_position_medians"] = _fit_affine(predicted_array, target_array)

    summary = {
        "kind": "policy_vs_apriltag_evaluation",
        "dataset": str(dataset_dir),
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "frame_count": len(frame_rows),
        "original_frame_count": original_row_count,
        "frame_stride": args.frame_stride,
        "max_frames_per_position": args.max_frames_per_position,
        "position_count": len(position_rows),
        "tag_detection_count": int(sum(1 for row in frame_rows if row["tag_found"])),
        "tag_detection_fraction": float(sum(1 for row in frame_rows if row["tag_found"]) / len(frame_rows)),
        "frame_error_policy_minus_apriltag": _error_stats(frame_errors),
        "position_error_policy_minus_apriltag": _error_stats(position_errors),
        "policy_prediction": {
            "model": str(model_path),
            "metadata": str(policy_metadata_path),
            "policy_config": str(Path(args.policy_config).expanduser().resolve()),
            "device": device,
            "stats": _stats(
                np.asarray(
                    [[row["policy_pred_x_m"], row["policy_pred_y_m"], row["policy_pred_z_m"]] for row in frame_rows],
                    dtype=np.float64,
                )
            ),
        },
        "apriltag": {
            "family": args.family,
            "id": args.marker_id,
            "marker_length_m": args.marker_length_m,
            "tag_to_object_translation_m": tag_to_object.tolist(),
            "base_height_offset_m": args.base_height_offset_m,
            "max_reprojection_px": args.max_reprojection_px,
            "detection_scales": scales,
            "calibration_report": str(Path(args.calibration_report).expanduser().resolve()),
            "base_T_camera": base_t_camera.tolist(),
        },
        "correction": correction,
    }

    _write_csv(output_dir / "frame_predictions.csv", frame_rows)
    _write_csv(output_dir / "position_summary.csv", position_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    position_stats = summary["position_error_policy_minus_apriltag"]
    print(f"[eval] tag detections: {summary['tag_detection_count']}/{summary['frame_count']}")
    if position_stats["count"]:
        bias = np.asarray(position_stats["bias_median_m"], dtype=np.float64) * 1000.0
        print(
            "[eval] median policy-tag bias: "
            f"[{bias[0]:+.1f}, {bias[1]:+.1f}, {bias[2]:+.1f}] mm"
        )
        print(f"[eval] position median 3D error: {position_stats['median_3d_mm']:.1f} mm")
    print(f"[eval] frame csv: {output_dir / 'frame_predictions.csv'}")
    print(f"[eval] position csv: {output_dir / 'position_summary.csv'}")
    print(f"[eval] summary: {output_dir / 'summary.json'}")
    return 0 if summary["tag_detection_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
