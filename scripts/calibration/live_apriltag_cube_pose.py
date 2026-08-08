#!/usr/bin/env python3
"""Detect the cube AprilTag and estimate its pose in the Franka base frame.

This program is read-only: it opens only the RealSense color stream and never
connects to or commands the robot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HAND_EYE_PATH = REPO_ROOT / "franka_sim2real" / "calibration" / "hand_eye.py"
_HAND_EYE_SPEC = importlib.util.spec_from_file_location("_hand_eye_calibration", _HAND_EYE_PATH)
if _HAND_EYE_SPEC is None or _HAND_EYE_SPEC.loader is None:
    raise ImportError(f"Could not load hand-eye helpers from {_HAND_EYE_PATH}")
_HAND_EYE = importlib.util.module_from_spec(_HAND_EYE_SPEC)
sys.modules[_HAND_EYE_SPEC.name] = _HAND_EYE
_HAND_EYE_SPEC.loader.exec_module(_HAND_EYE)

load_eye_to_hand_config = _HAND_EYE.load_eye_to_hand_config
make_transform = _HAND_EYE.make_transform
rotation_to_quaternion_xyzw = _HAND_EYE.rotation_to_quaternion_xyzw

DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs" / "eye_to_hand_d435_215322076207.json"
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0802_dr_simactuator_gpu.json"
DEFAULT_CALIBRATION_REPORT = (
    REPO_ROOT
    / "runs"
    / "20260802_204111_eye_to_hand"
    / "reports"
    / "20260802_205349"
    / "calibration_report.json"
)
APRILTAG_FAMILIES = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
for _family_name, _opencv_name in (
    ("aruco4x4_50", "DICT_4X4_50"),
    ("aruco4x4_100", "DICT_4X4_100"),
    ("aruco4x4_250", "DICT_4X4_250"),
    ("aruco4x4_1000", "DICT_4X4_1000"),
    ("aruco5x5_50", "DICT_5X5_50"),
    ("aruco5x5_100", "DICT_5X5_100"),
    ("aruco5x5_250", "DICT_5X5_250"),
    ("aruco5x5_1000", "DICT_5X5_1000"),
    ("aruco6x6_50", "DICT_6X6_50"),
    ("aruco6x6_100", "DICT_6X6_100"),
    ("aruco6x6_250", "DICT_6X6_250"),
    ("aruco6x6_1000", "DICT_6X6_1000"),
    ("aruco7x7_50", "DICT_7X7_50"),
    ("aruco7x7_100", "DICT_7X7_100"),
    ("aruco7x7_250", "DICT_7X7_250"),
    ("aruco7x7_1000", "DICT_7X7_1000"),
    ("aruco_original", "DICT_ARUCO_ORIGINAL"),
):
    if hasattr(cv2.aruco, _opencv_name):
        APRILTAG_FAMILIES[_family_name] = getattr(cv2.aruco, _opencv_name)


def _load_accepted_base_t_camera(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Eye-to-hand report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    solve = report.get("solve", {})
    if solve.get("accepted") is not True:
        raise ValueError(f"Eye-to-hand report is not accepted: {path}")
    transform = np.asarray(solve.get("T_base_color"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"Eye-to-hand report has an invalid T_base_color: {path}")
    return transform


def _camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _distortion(intrinsics: Any) -> np.ndarray:
    coefficients = np.asarray(intrinsics.coeffs, dtype=np.float64).reshape(-1)
    model_name = str(intrinsics.model).split(".")[-1]
    if np.any(np.abs(coefficients) > 1e-12) and model_name not in {
        "brown_conrady",
        "modified_brown_conrady",
    }:
        raise RuntimeError(
            f"Unsupported non-zero RealSense distortion model for solvePnP: {model_name}"
        )
    return coefficients


def _tag_object_points(marker_length_m: float) -> np.ndarray:
    half = 0.5 * marker_length_m
    # OpenCV ArUco corner order: top-left, top-right, bottom-right, bottom-left.
    # The tag +Z axis points outward from the printed face.
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, distortion
    )
    residual = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))


def solve_tag_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_length_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return camera_T_tag, rvec and four-corner reprojection RMS."""
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    object_points = _tag_object_points(marker_length_m)
    result = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    success, rvecs, tvecs = result[:3]
    if not success or len(rvecs) == 0:
        raise RuntimeError("SOLVEPNP_IPPE_SQUARE did not return a pose")

    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if float(tvec[2, 0]) <= 0.0:
            continue
        # IPPE supplies the two square-pose initializations. Refine both on the
        # original corner residuals before choosing the lower-error solution;
        # this materially reduces planar-pose tilt bias in the cube-center
        # offset along tag -Z.
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec,
            tvec,
        )
        error = _reprojection_rms(
            object_points, image_points, camera_matrix, distortion, rvec, tvec
        )
        candidates.append((error, rvec, tvec))
    if not candidates:
        raise RuntimeError("All square-PnP solutions placed the tag behind the camera")
    error, rvec, tvec = min(candidates, key=lambda item: item[0])
    rotation, _ = cv2.Rodrigues(rvec)
    camera_t_tag = make_transform(rotation, tvec.reshape(3))
    return camera_t_tag, rvec, error


def _transform_translation(transform: np.ndarray, translation: np.ndarray) -> np.ndarray:
    point = np.concatenate([np.asarray(translation, dtype=np.float64), [1.0]])
    return (transform @ point)[:3]


def object_pose_in_base(
    base_t_camera: np.ndarray,
    camera_t_tag: np.ndarray,
    tag_to_object_translation_m: np.ndarray,
) -> np.ndarray:
    tag_t_object = np.eye(4, dtype=np.float64)
    tag_t_object[:3, 3] = tag_to_object_translation_m
    return base_t_camera @ camera_t_tag @ tag_t_object


def _draw_text(
    image: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (10, 24),
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, color in lines:
        cv2.putText(image, text, (x + 1, y + 1), font, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), font, 0.52, color, 1, cv2.LINE_AA)
        y += 22


def _xyz_text(label: str, xyz_m: np.ndarray) -> str:
    xyz_mm = 1000.0 * np.asarray(xyz_m)
    return f"{label}: [{xyz_mm[0]:+.1f}, {xyz_mm[1]:+.1f}, {xyz_mm[2]:+.1f}] mm"


def _finite_triplet(name: str, values: list[float]) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} requires three finite values")
    return result


def _summary_statistics(values: list[np.ndarray]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean_m": np.mean(array, axis=0).tolist(),
        "std_m": np.std(array, axis=0).tolist(),
        "median_m": np.median(array, axis=0).tolist(),
    }


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model).split(".")[-1],
        "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Policy config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_policy_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, str]:
    model_config = config.get("model", {})
    model_value = args.policy_model or model_config.get("model_path")
    if not model_value:
        raise ValueError("Policy model path must be supplied by --policy-model or policy config")
    model_path = Path(model_value).expanduser().resolve()

    metadata_value = args.policy_metadata or model_config.get("metadata_path")
    metadata_path = (
        Path(metadata_value).expanduser().resolve()
        if metadata_value
        else model_path.with_suffix(".json")
    )

    device = args.policy_device
    if device == "auto":
        device = str(model_config.get("device") or "cpu")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Trained policy prediction requires torch. Activate the project "
                "environment or run with --no-policy-prediction."
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
    return model_path, metadata_path, device


def _triplet(name: str, values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).reshape(-1)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} requires three finite values")
    return result


def _policy_crop(rgb: np.ndarray, camera_config: dict[str, Any], width: int, height: int) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")

    if camera_config.get("enable_crop", True):
        crop_width = camera_config.get("crop_width")
        crop_height = camera_config.get("crop_height")
        if crop_width is not None and crop_height is not None:
            src_height, src_width = image.shape[:2]
            left = max(0, min(int(camera_config.get("crop_left", 0)), max(src_width - 1, 0)))
            top = max(0, min(int(camera_config.get("crop_top", 0)), max(src_height - 1, 0)))
            right = min(left + max(1, int(crop_width)), src_width)
            bottom = min(top + max(1, int(crop_height)), src_height)
            image = image[top:bottom, left:right]
            if image.size == 0:
                raise ValueError("Policy camera crop produced an empty image")

    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.array(image, copy=True, order="C")


class RMAPolicyCubePredictor:
    """Read cube XYZ from the RMA student's visual adaptation head."""

    def __init__(self, model_path: Path, metadata_path: Path, device: str) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Policy model not found: {model_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Policy metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("kind") != "tacex_rma_student_torchscript":
            raise ValueError("Policy metadata is not a TacEx RMA student export")

        expected_hash = self.metadata.get("torchscript_sha256")
        if expected_hash and _sha256(model_path) != expected_hash:
            raise ValueError("TorchScript SHA-256 differs from policy metadata")

        signature = self.metadata.get("input_signature", {}).get("wrist_rgb")
        if not isinstance(signature, list) or len(signature) != 3:
            raise ValueError("Policy metadata is missing input_signature.wrist_rgb")
        self.height, self.width, channels = (int(value) for value in signature)
        if channels != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {channels}")

        normalization = self.metadata.get("normalization", {})
        self.position_center = _triplet(
            "normalization.cube_position_center",
            normalization.get("cube_position_center", []),
        )
        self.position_scale = _triplet(
            "normalization.cube_position_scale",
            normalization.get("cube_position_scale", []),
        )
        if np.any(self.position_scale <= 0.0):
            raise ValueError("normalization.cube_position_scale must be positive")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Trained policy prediction requires torch. Activate the project "
                "environment or run with --no-policy-prediction."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested ({device}) but torch.cuda.is_available() is false")
        self.model = torch.jit.load(str(model_path), map_location=self.device).eval()
        self.exported_methods = set(self.model._c._method_names())
        self._position_scale = torch.as_tensor(
            self.position_scale, dtype=torch.float32, device=self.device
        )
        self._position_center = torch.as_tensor(
            self.position_center, dtype=torch.float32, device=self.device
        )

    def _predict_adaptation(self, rgb_tensor: Any) -> tuple[Any, Any]:
        if "predict_adaptation" in self.exported_methods:
            outputs = self.model.predict_adaptation(rgb_tensor)
        else:
            image = rgb_tensor.to(self.torch.float32).permute(0, 3, 1, 2) / 255.0
            image = (image - self.model.image_mean) / self.model.image_std
            outputs = self.model.adaptation_head(self.model.vision_encoder(image))
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise RuntimeError("RMA adaptation head returned an unexpected output")
        by_dim = {int(tensor.shape[-1]): tensor for tensor in outputs}
        if set(by_dim) != {2, 3}:
            shapes = [tuple(tensor.shape) for tensor in outputs]
            raise RuntimeError(f"Expected position[3] and contact[2], got {shapes}")
        return by_dim[3], by_dim[2]

    def predict(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(rgb, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if image.shape != expected:
            raise ValueError(f"Expected RGB image {expected}, got {image.shape}")
        with self.torch.inference_mode():
            batch = image.reshape(1, self.height, self.width, 3)
            tensor = self.torch.as_tensor(batch, dtype=self.torch.uint8, device=self.device)
            normalized, contact_logits = self._predict_adaptation(tensor)
            position = normalized * self._position_scale + self._position_center
            contact = self.torch.sigmoid(contact_logits)
        return (
            position[0].detach().cpu().numpy().astype(np.float64),
            contact[0].detach().cpu().numpy().astype(np.float64),
        )

    def predict_batch(self, images: np.ndarray, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        images = np.asarray(images, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected RGB image batch [N,{self.height},{self.width},3], got {images.shape}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        positions: list[np.ndarray] = []
        contacts: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = np.array(images[start : start + batch_size], copy=True, order="C")
                tensor = self.torch.as_tensor(batch, dtype=self.torch.uint8, device=self.device)
                normalized, contact_logits = self._predict_adaptation(tensor)
                position = normalized * self._position_scale + self._position_center
                contact = self.torch.sigmoid(contact_logits)
                positions.append(position.detach().cpu().numpy().astype(np.float64))
                contacts.append(contact.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(positions, axis=0), np.concatenate(contacts, axis=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--calibration-report", default=str(DEFAULT_CALIBRATION_REPORT))
    parser.add_argument(
        "--policy-config",
        default=str(DEFAULT_POLICY_CONFIG),
        help="Deployment config supplying the policy model path and camera crop contract",
    )
    parser.add_argument("--policy-model", help="TorchScript model path; defaults to policy config")
    parser.add_argument("--policy-metadata", help="Policy metadata JSON; defaults to policy config")
    parser.add_argument(
        "--policy-device",
        default="auto",
        help="Inference device. 'auto' uses policy config model.device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--no-policy-prediction",
        action="store_true",
        help="Disable the trained policy cube-position prediction overlay",
    )
    parser.add_argument("--family", choices=sorted(APRILTAG_FAMILIES), default="tag36h11")
    parser.add_argument("--id", type=int, default=2, dest="marker_id")
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=0.038,
        help="Outer black-square edge length, not the 50 mm paper size",
    )
    parser.add_argument(
        "--tag-to-object",
        nargs=3,
        type=float,
        default=(0.0, 0.0, -0.025),
        metavar=("X_M", "Y_M", "Z_M"),
        help="Object center expressed in the tag frame; default is a centered 50 mm cube face",
    )
    parser.add_argument(
        "--ground-truth",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Independently measured cube-center XYZ in the Franka base frame",
    )
    parser.add_argument(
        "--base-height-offset-m",
        type=float,
        default=0.020,
        help="Z offset added to calibrated cube XYZ to align with the policy/training base height",
    )
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument(
        "--detection-scale",
        type=float,
        default=4.0,
        help="Upscale used only for detecting the small, oblique tag; pose still uses raw pixels",
    )
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Ctrl-C")
    parser.add_argument("--print-hz", type=float, default=2.0)
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", help="Defaults to a timestamped directory under runs/")
    return parser


def _validate_color_control_args(
    auto_exposure: bool,
    exposure: float | None,
    gain: float | None,
) -> None:
    if exposure is not None and not math.isfinite(exposure):
        raise ValueError("--exposure must be finite")
    if gain is not None and not math.isfinite(gain):
        raise ValueError("--gain must be finite")
    if auto_exposure and (exposure is not None or gain is not None):
        raise ValueError("--auto-exposure cannot be combined with --exposure or --gain")


def _set_color_option(sensor: Any, option: Any, value: float, label: str) -> None:
    if not sensor.supports(option):
        raise RuntimeError(f"The selected RealSense color sensor does not support {label}")
    option_range = sensor.get_option_range(option)
    numeric_value = float(value)
    if numeric_value < option_range.min or numeric_value > option_range.max:
        raise ValueError(
            f"RealSense {label} {numeric_value:g} is outside the supported range "
            f"[{option_range.min:g}, {option_range.max:g}]"
        )
    sensor.set_option(option, numeric_value)


def _configure_color_controls(
    rs: Any,
    sensor: Any,
    *,
    auto_exposure: bool,
    exposure: float | None,
    gain: float | None,
) -> dict[str, bool | float | None]:
    manual = exposure is not None or gain is not None
    if manual:
        _set_color_option(sensor, rs.option.enable_auto_exposure, 0.0, "auto exposure")
    elif auto_exposure:
        _set_color_option(sensor, rs.option.enable_auto_exposure, 1.0, "auto exposure")
    if exposure is not None:
        _set_color_option(sensor, rs.option.exposure, exposure, "exposure")
    if gain is not None:
        _set_color_option(sensor, rs.option.gain, gain, "gain")

    def get(option: Any) -> float | None:
        if not sensor.supports(option):
            return None
        return float(sensor.get_option(option))

    effective_auto = get(rs.option.enable_auto_exposure)
    return {
        "auto_exposure": (
            None if effective_auto is None else bool(round(effective_auto))
        ),
        "exposure": get(rs.option.exposure),
        "gain": get(rs.option.gain),
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.marker_length_m <= 0.0:
        raise ValueError("--marker-length-m must be positive")
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing-window must be positive")
    if not math.isfinite(args.detection_scale) or not 1.0 <= args.detection_scale <= 4.0:
        raise ValueError("--detection-scale must be finite and in [1, 4]")
    if args.max_reprojection_px <= 0.0 or args.print_hz <= 0.0 or args.max_frames < 0:
        raise ValueError("reprojection threshold and print rate must be positive; max-frames >= 0")
    if not math.isfinite(args.base_height_offset_m):
        raise ValueError("--base-height-offset-m must be finite")
    _validate_color_control_args(args.auto_exposure, args.exposure, args.gain)

    camera_config = load_eye_to_hand_config(args.camera_config).camera
    tag_to_object = _finite_triplet("--tag-to-object", args.tag_to_object)
    calibrated_offset = np.asarray([0.0, 0.0, args.base_height_offset_m], dtype=np.float64)
    ground_truth = (
        None if args.ground_truth is None else _finite_triplet("--ground-truth", args.ground_truth)
    )
    base_t_camera = _load_accepted_base_t_camera(
        Path(args.calibration_report).expanduser().resolve()
    )

    policy_config = _load_policy_config(Path(args.policy_config).expanduser().resolve())
    policy_camera_config = policy_config.get("camera", {})
    predictor: RMAPolicyCubePredictor | None = None
    policy_model_path: Path | None = None
    policy_metadata_path: Path | None = None
    policy_device = ""
    if not args.no_policy_prediction:
        policy_model_path, policy_metadata_path, policy_device = _resolve_policy_paths(
            args, policy_config
        )
        predictor = RMAPolicyCubePredictor(policy_model_path, policy_metadata_path, policy_device)
        warmup_rgb = np.zeros((predictor.height, predictor.width, 3), dtype=np.uint8)
        predictor.predict(warmup_rgb)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_live_apriltag_cube_pose"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "detections.csv"

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
    try:
        color_sensor = profile.get_device().first_color_sensor()
        color_controls = _configure_color_controls(
            rs,
            color_sensor,
            auto_exposure=args.auto_exposure,
            exposure=args.exposure,
            gain=args.gain,
        )
    except BaseException:
        pipeline.stop()
        raise
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = stream.get_intrinsics()
    camera_matrix = _camera_matrix(intrinsics)
    distortion = _distortion(intrinsics)

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[args.family])
    parameters = cv2.aruco.DetectorParameters()
    # The tag is only about 30 px wide and is strongly foreshortened on the
    # cube face in the current view. AprilTag line refinement on a 3x detection
    # image recovers it reliably; corners are mapped back to raw 640x480 pixels
    # before PnP so camera intrinsics remain unchanged.
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    fieldnames = [
        "host_time_s",
        "camera_time_ms",
        "frame_index",
        "reprojection_rms_px",
        "camera_tag_x_m",
        "camera_tag_y_m",
        "camera_tag_z_m",
        "camera_tag_qx",
        "camera_tag_qy",
        "camera_tag_qz",
        "camera_tag_qw",
    ]
    fieldnames.extend(
        [
            "calibrated_cube_x_m",
            "calibrated_cube_y_m",
            "calibrated_cube_z_m",
            "calibrated_cube_qx",
            "calibrated_cube_qy",
            "calibrated_cube_qz",
            "calibrated_cube_qw",
            "calibrated_error_3d_m",
            "calibrated_raw_cube_x_m",
            "calibrated_raw_cube_y_m",
            "calibrated_raw_cube_z_m",
            "policy_pred_cube_x_m",
            "policy_pred_cube_y_m",
            "policy_pred_cube_z_m",
            "policy_pred_contact_left",
            "policy_pred_contact_right",
            "policy_pred_vs_calibrated_error_3d_m",
            "policy_pred_vs_ground_truth_error_3d_m",
        ]
    )

    calibrated_buffer: deque[np.ndarray] = deque(maxlen=args.smoothing_window)
    prediction_buffer: deque[np.ndarray] = deque(maxlen=args.smoothing_window)
    all_calibrated_positions: list[np.ndarray] = []
    all_prediction_positions: list[np.ndarray] = []
    all_calibrated_errors: list[float] = []
    all_prediction_errors: list[float] = []
    all_prediction_vs_calibrated_errors: list[float] = []
    frame_count = 0
    detection_count = 0
    rejected_reprojection_count = 0
    start_time = time.monotonic()
    previous_print_time = 0.0
    latest_display: np.ndarray | None = None

    print("[AprilTag] camera only; no Franka connection or motion commands")
    print(f"[AprilTag] family={args.family}, id={args.marker_id}, marker_length={args.marker_length_m:.3f} m")
    print(f"[AprilTag] tag_to_object={tag_to_object.tolist()} m")
    print(f"[AprilTag] calibrated base_height_offset={args.base_height_offset_m:.3f} m")
    print(f"[AprilTag] calibrated report={Path(args.calibration_report).expanduser().resolve()}")
    print(
        "[AprilTag] color controls: "
        f"auto_exposure={color_controls['auto_exposure']}, "
        f"exposure={color_controls['exposure']}, gain={color_controls['gain']}"
    )
    if predictor is not None:
        print(f"[AprilTag] policy model={policy_model_path}")
        print(f"[AprilTag] policy metadata={policy_metadata_path}")
        print(f"[AprilTag] policy device={policy_device}")
    if ground_truth is not None:
        print(f"[AprilTag] ground_truth={ground_truth.tolist()} m")
    print("[AprilTag] keys: q/Esc quit, s snapshot, c clear smoothing")

    try:
        for _ in range(max(0, camera_config.warmup_frames)):
            pipeline.wait_for_frames()
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            while args.max_frames == 0 or frame_count < args.max_frames:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_count += 1
                rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                if args.detection_scale == 1.0:
                    detection_gray = gray
                else:
                    detection_gray = cv2.resize(
                        gray,
                        None,
                        fx=args.detection_scale,
                        fy=args.detection_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                corners, ids, _ = detector.detectMarkers(detection_gray)
                if args.detection_scale != 1.0:
                    corners = [corner / args.detection_scale for corner in corners]
                fps = frame_count / max(time.monotonic() - start_time, 1e-9)
                lines: list[tuple[str, tuple[int, int, int]]] = [
                    (f"AprilTag {args.family} ID {args.marker_id} | {fps:.1f} FPS", (255, 255, 255))
                ]
                policy_position: np.ndarray | None = None
                policy_contact: np.ndarray | None = None
                policy_median: np.ndarray | None = None
                if predictor is not None:
                    try:
                        policy_rgb = _policy_crop(
                            rgb,
                            policy_camera_config,
                            predictor.width,
                            predictor.height,
                        )
                        policy_position, policy_contact = predictor.predict(policy_rgb)
                        prediction_buffer.append(policy_position.copy())
                        all_prediction_positions.append(policy_position.copy())
                        policy_median = np.median(np.asarray(prediction_buffer), axis=0)
                        lines.append(
                            (_xyz_text("policy_pred_p_cube median", policy_median), (255, 0, 255))
                        )
                    except RuntimeError as exc:
                        lines.append((f"POLICY PRED FAILED: {exc}", (0, 0, 255)))
                row: dict[str, Any] | None = None
                selected_index = None
                if ids is not None:
                    matches = np.flatnonzero(ids.reshape(-1) == args.marker_id)
                    if len(matches):
                        selected_index = int(matches[0])

                if selected_index is None:
                    lines.append(("TAG NOT FOUND", (0, 0, 255)))
                else:
                    selected_corners = corners[selected_index]
                    cv2.polylines(
                        display,
                        [np.rint(selected_corners.reshape(4, 2)).astype(np.int32)],
                        True,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    try:
                        camera_t_tag, rvec, reprojection = solve_tag_pose(
                            selected_corners,
                            camera_matrix,
                            distortion,
                            args.marker_length_m,
                        )
                    except RuntimeError as exc:
                        lines.append((f"POSE FAILED: {exc}", (0, 0, 255)))
                    else:
                        tvec = camera_t_tag[:3, 3].reshape(3, 1)
                        cv2.drawFrameAxes(
                            display,
                            camera_matrix,
                            distortion,
                            rvec,
                            tvec,
                            0.030,
                            2,
                        )
                        quality_ok = reprojection <= args.max_reprojection_px
                        quality_color = (0, 255, 0) if quality_ok else (0, 165, 255)
                        lines.append((_xyz_text("camera_p_tag", camera_t_tag[:3, 3]), (255, 255, 0)))
                        lines.append((f"reprojection RMS: {reprojection:.3f} px", quality_color))
                        if quality_ok:
                            detection_count += 1
                            row = {
                                "host_time_s": time.time(),
                                "camera_time_ms": float(color_frame.get_timestamp()),
                                "frame_index": frame_count,
                                "reprojection_rms_px": reprojection,
                                "camera_tag_x_m": float(camera_t_tag[0, 3]),
                                "camera_tag_y_m": float(camera_t_tag[1, 3]),
                                "camera_tag_z_m": float(camera_t_tag[2, 3]),
                            }
                            camera_tag_quaternion = rotation_to_quaternion_xyzw(
                                camera_t_tag[:3, :3]
                            )
                            row.update(
                                {
                                    "camera_tag_qx": float(camera_tag_quaternion[0]),
                                    "camera_tag_qy": float(camera_tag_quaternion[1]),
                                    "camera_tag_qz": float(camera_tag_quaternion[2]),
                                    "camera_tag_qw": float(camera_tag_quaternion[3]),
                                }
                            )
                            pose = object_pose_in_base(
                                base_t_camera, camera_t_tag, tag_to_object
                            )
                            raw_position = pose[:3, 3]
                            position = raw_position + calibrated_offset
                            calibrated_buffer.append(position.copy())
                            all_calibrated_positions.append(position.copy())
                            calibrated_median = np.median(np.asarray(calibrated_buffer), axis=0)
                            calibrated_color = (255, 128, 0)
                            lines.append(
                                (_xyz_text("calibrated_p_cube median", calibrated_median), calibrated_color)
                            )
                            calibrated_error = math.nan
                            object_quaternion = rotation_to_quaternion_xyzw(pose[:3, :3])
                            if ground_truth is not None:
                                calibrated_error = float(np.linalg.norm(calibrated_median - ground_truth))
                                all_calibrated_errors.append(float(np.linalg.norm(position - ground_truth)))
                                lines.append(
                                    (f"calibrated error: {1000.0 * calibrated_error:.1f} mm", calibrated_color)
                                )

                            prediction_vs_calibrated_error = math.nan
                            prediction_vs_ground_truth_error = math.nan
                            if policy_position is not None and policy_contact is not None:
                                prediction_vs_calibrated_error = float(
                                    np.linalg.norm(policy_position - position)
                                )
                                all_prediction_vs_calibrated_errors.append(
                                    prediction_vs_calibrated_error
                                )
                                comparison_error = (
                                    np.linalg.norm(policy_median - calibrated_median)
                                    if policy_median is not None
                                    else prediction_vs_calibrated_error
                                )
                                lines.append(
                                    (
                                        f"policy_pred vs tag: {1000.0 * comparison_error:.1f} mm",
                                        (255, 0, 255),
                                    )
                                )
                                if ground_truth is not None:
                                    prediction_vs_ground_truth_error = float(
                                        np.linalg.norm(policy_position - ground_truth)
                                    )
                                    all_prediction_errors.append(prediction_vs_ground_truth_error)
                                    lines.append(
                                        (
                                            f"policy_pred error: {1000.0 * prediction_vs_ground_truth_error:.1f} mm",
                                            (255, 0, 255),
                                        )
                                    )

                            row.update(
                                {
                                    "calibrated_cube_x_m": float(position[0]),
                                    "calibrated_cube_y_m": float(position[1]),
                                    "calibrated_cube_z_m": float(position[2]),
                                    "calibrated_cube_qx": float(object_quaternion[0]),
                                    "calibrated_cube_qy": float(object_quaternion[1]),
                                    "calibrated_cube_qz": float(object_quaternion[2]),
                                    "calibrated_cube_qw": float(object_quaternion[3]),
                                    "calibrated_error_3d_m": calibrated_error,
                                    "calibrated_raw_cube_x_m": float(raw_position[0]),
                                    "calibrated_raw_cube_y_m": float(raw_position[1]),
                                    "calibrated_raw_cube_z_m": float(raw_position[2]),
                                    "policy_pred_cube_x_m": math.nan
                                    if policy_position is None
                                    else float(policy_position[0]),
                                    "policy_pred_cube_y_m": math.nan
                                    if policy_position is None
                                    else float(policy_position[1]),
                                    "policy_pred_cube_z_m": math.nan
                                    if policy_position is None
                                    else float(policy_position[2]),
                                    "policy_pred_contact_left": math.nan
                                    if policy_contact is None
                                    else float(policy_contact[0]),
                                    "policy_pred_contact_right": math.nan
                                    if policy_contact is None
                                    else float(policy_contact[1]),
                                    "policy_pred_vs_calibrated_error_3d_m": prediction_vs_calibrated_error,
                                    "policy_pred_vs_ground_truth_error_3d_m": prediction_vs_ground_truth_error,
                                }
                            )
                            writer.writerow(row)
                            csv_file.flush()
                        else:
                            rejected_reprojection_count += 1
                            lines.append(("POSE REJECTED BY REPROJECTION THRESHOLD", quality_color))

                _draw_text(display, lines)
                latest_display = display
                now = time.monotonic()
                if row is not None and now - previous_print_time >= 1.0 / args.print_hz:
                    print(" | ".join(text for text, _ in lines[1:]))
                    previous_print_time = now

                if not args.headless:
                    cv2.imshow("AprilTag cube pose (camera only)", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if key == ord("c"):
                        calibrated_buffer.clear()
                        prediction_buffer.clear()
                        print("[AprilTag] smoothing buffers cleared")
                    if key == ord("s"):
                        snapshot = snapshots_dir / f"frame_{frame_count:06d}.png"
                        cv2.imwrite(str(snapshot), display)
                        print(f"[AprilTag] snapshot: {snapshot}")
    except KeyboardInterrupt:
        print("\n[AprilTag] interrupted")
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    summary: dict[str, Any] = {
        "kind": "live_apriltag_cube_pose",
        "camera_only_no_robot_connection": True,
        "camera_serial": camera_config.serial,
        "camera_intrinsics": _intrinsics_dict(intrinsics),
        "camera_color_controls": color_controls,
        "tag": {
            "family": args.family,
            "id": args.marker_id,
            "marker_length_m": args.marker_length_m,
            "tag_to_object_translation_m": tag_to_object.tolist(),
        },
        "frame_count": frame_count,
        "accepted_detection_count": detection_count,
        "rejected_reprojection_count": rejected_reprojection_count,
        "max_reprojection_px": args.max_reprojection_px,
        "detection_scale": args.detection_scale,
        "ground_truth_m": None if ground_truth is None else ground_truth.tolist(),
        "calibrated": {
            **_summary_statistics(all_calibrated_positions),
            "base_T_camera": base_t_camera.tolist(),
            "calibration_report": str(Path(args.calibration_report).expanduser().resolve()),
            "base_height_offset_m": args.base_height_offset_m,
            "position_is_offset_adjusted": True,
        },
        "policy_prediction": {
            **_summary_statistics(all_prediction_positions),
            "enabled": predictor is not None,
            "policy_config": str(Path(args.policy_config).expanduser().resolve()),
            "model": None if policy_model_path is None else str(policy_model_path),
            "metadata": None if policy_metadata_path is None else str(policy_metadata_path),
            "device": policy_device or None,
        },
        "detections_csv": str(csv_path),
    }
    if ground_truth is not None and all_calibrated_errors:
        errors = np.asarray(all_calibrated_errors, dtype=np.float64)
        summary["calibrated"]["error_3d_rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
        summary["calibrated"]["error_3d_median_m"] = float(np.median(errors))
        summary["calibrated"]["error_3d_p95_m"] = float(np.percentile(errors, 95.0))
    if ground_truth is not None and all_prediction_errors:
        errors = np.asarray(all_prediction_errors, dtype=np.float64)
        summary["policy_prediction"]["error_3d_rmse_m"] = float(np.sqrt(np.mean(np.square(errors))))
        summary["policy_prediction"]["error_3d_median_m"] = float(np.median(errors))
        summary["policy_prediction"]["error_3d_p95_m"] = float(np.percentile(errors, 95.0))
    if all_prediction_vs_calibrated_errors:
        errors = np.asarray(all_prediction_vs_calibrated_errors, dtype=np.float64)
        summary["policy_prediction"]["vs_calibrated_error_3d_rmse_m"] = float(
            np.sqrt(np.mean(np.square(errors)))
        )
        summary["policy_prediction"]["vs_calibrated_error_3d_median_m"] = float(
            np.median(errors)
        )
        summary["policy_prediction"]["vs_calibrated_error_3d_p95_m"] = float(
            np.percentile(errors, 95.0)
        )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if latest_display is not None:
        cv2.imwrite(str(output_dir / "last_frame.png"), latest_display)
    print(f"[AprilTag] detections: {csv_path}")
    print(f"[AprilTag] summary: {summary_path}")
    return 0 if detection_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
