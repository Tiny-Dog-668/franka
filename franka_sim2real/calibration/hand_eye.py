from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 1


@dataclass
class CameraConfig:
    serial: str = "215322076207"
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30
    burst_frames: int = 10


@dataclass
class BoardConfig:
    inner_corners_columns: int = 4
    inner_corners_rows: int = 4
    square_size_m: float = 0.030

    @property
    def pattern_size(self) -> tuple[int, int]:
        return self.inner_corners_columns, self.inner_corners_rows


@dataclass
class RobotConfig:
    ip: str = "172.16.0.2"
    realtime: str = "ignore"
    speed: float = 0.02
    settle_time_s: float = 1.0
    workspace_minimum_m: list[float] = field(
        default_factory=lambda: [0.2, -0.3, 0.05]
    )
    workspace_maximum_m: list[float] = field(
        default_factory=lambda: [0.65, 0.3, 0.45]
    )
    max_segment_translation_m: float = 0.010
    max_segment_rotation_deg: float = 5.0
    tool_envelope_radius_m: float = 0.220


@dataclass
class QualityConfig:
    max_reprojection_rms_px: float = 0.8
    max_joint_velocity_rad_s: float = 0.02
    max_capture_tcp_drift_m: float = 0.0005
    max_capture_tcp_drift_deg: float = 0.2
    max_target_translation_error_m: float = 0.002
    max_target_rotation_error_deg: float = 1.0
    max_gripper_width_drift_m: float = 0.0005
    consensus_translation_m: float = 0.005
    consensus_rotation_deg: float = 1.0
    validation_translation_m: float = 0.005
    validation_rotation_deg: float = 1.0
    required_calibration_samples: int = 25
    required_validation_samples: int = 5
    minimum_valid_methods: int = 3


@dataclass
class EyeToHandConfig:
    schema_version: int = SCHEMA_VERSION
    camera: CameraConfig = field(default_factory=CameraConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EyeToHandConfig":
        top_level = dict(data)
        camera = CameraConfig(**top_level.pop("camera", {}))
        board = BoardConfig(**top_level.pop("board", {}))
        robot = RobotConfig(**top_level.pop("robot", {}))
        quality = QualityConfig(**top_level.pop("quality", {}))
        config = cls(
            camera=camera,
            board=board,
            robot=robot,
            quality=quality,
            **top_level,
        )
        validate_config(config)
        return config

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryPose:
    sample_id: str
    phase: str
    group: str
    translation_offset_base_m: tuple[float, float, float]
    rpy_offset_tool_deg: tuple[float, float, float]
    hub: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "phase": self.phase,
            "group": self.group,
            "translation_offset_base_m": list(self.translation_offset_base_m),
            "rpy_offset_tool_deg": list(self.rpy_offset_tool_deg),
            "hub": self.hub,
        }


def load_eye_to_hand_config(path: str | Path) -> EyeToHandConfig:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return EyeToHandConfig.from_dict(json.load(handle))


def validate_config(config: EyeToHandConfig) -> None:
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported eye-to-hand config schema {config.schema_version}; "
            f"expected {SCHEMA_VERSION}"
        )
    if not config.camera.serial.strip():
        raise ValueError("camera.serial must be explicit and non-empty")
    for name in ("width", "height", "fps", "burst_frames"):
        if int(getattr(config.camera, name)) < 1:
            raise ValueError(f"camera.{name} must be positive")
    if config.camera.warmup_frames < 0:
        raise ValueError("camera.warmup_frames must be non-negative")
    if config.board.inner_corners_columns < 2 or config.board.inner_corners_rows < 2:
        raise ValueError("board inner corner counts must both be at least 2")
    if not math.isfinite(config.board.square_size_m) or config.board.square_size_m <= 0:
        raise ValueError("board.square_size_m must be finite and positive")
    if config.robot.realtime not in {"ignore", "enforce"}:
        raise ValueError("robot.realtime must be 'ignore' or 'enforce'")
    if not 0.0 < config.robot.speed <= 1.0:
        raise ValueError("robot.speed must be in (0, 1]")
    if config.robot.settle_time_s < 0.0:
        raise ValueError("robot.settle_time_s must be non-negative")
    for name in (
        "max_segment_translation_m",
        "max_segment_rotation_deg",
        "tool_envelope_radius_m",
    ):
        if not math.isfinite(getattr(config.robot, name)) or getattr(config.robot, name) <= 0:
            raise ValueError(f"robot.{name} must be finite and positive")
    minimum = _finite_vector(
        "robot.workspace_minimum_m", config.robot.workspace_minimum_m, 3
    )
    maximum = _finite_vector(
        "robot.workspace_maximum_m", config.robot.workspace_maximum_m, 3
    )
    if np.any(minimum >= maximum):
        raise ValueError("robot workspace minimum must be lower than maximum")
    for name, value in asdict(config.quality).items():
        if isinstance(value, int):
            if value < 1:
                raise ValueError(f"quality.{name} must be positive")
        elif not math.isfinite(value) or value <= 0:
            raise ValueError(f"quality.{name} must be finite and positive")
    calibration = build_trajectory("calibration")
    validation = build_trajectory("validation")
    if len(calibration) != config.quality.required_calibration_samples:
        raise ValueError("required calibration sample count does not match trajectory")
    if len(validation) != config.quality.required_validation_samples:
        raise ValueError("required validation sample count does not match trajectory")


def build_trajectory(phase: str) -> list[TrajectoryPose]:
    if phase == "calibration":
        specs: list[
            tuple[str, tuple[float, float, float], tuple[float, float, float], str]
        ] = [
            ("center", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "center"),
            ("translation_x_pos", (0.05, 0.0, 0.0), (0.0, 0.0, 0.0), "center"),
            ("translation_x_neg", (-0.05, 0.0, 0.0), (0.0, 0.0, 0.0), "center"),
            ("translation_y_pos", (0.0, 0.05, 0.0), (0.0, 0.0, 0.0), "center"),
            ("translation_y_neg", (0.0, -0.05, 0.0), (0.0, 0.0, 0.0), "center"),
            ("raised_center", (0.0, 0.0, 0.03), (0.0, 0.0, 0.0), "raised"),
            ("roll_pos_15", (0.0, 0.0, 0.03), (15.0, 0.0, 0.0), "raised"),
            ("roll_neg_15", (0.0, 0.0, 0.03), (-15.0, 0.0, 0.0), "raised"),
            ("roll_pos_25", (0.0, 0.0, 0.03), (25.0, 0.0, 0.0), "raised"),
            ("roll_neg_25", (0.0, 0.0, 0.03), (-25.0, 0.0, 0.0), "raised"),
            ("pitch_pos_15", (0.0, 0.0, 0.03), (0.0, 15.0, 0.0), "raised"),
            ("pitch_neg_15", (0.0, 0.0, 0.03), (0.0, -15.0, 0.0), "raised"),
            ("pitch_pos_25", (0.0, 0.0, 0.03), (0.0, 25.0, 0.0), "raised"),
            ("pitch_neg_25", (0.0, 0.0, 0.03), (0.0, -25.0, 0.0), "raised"),
            ("yaw_pos_15", (0.0, 0.0, 0.03), (0.0, 0.0, 15.0), "raised"),
            ("yaw_neg_15", (0.0, 0.0, 0.03), (0.0, 0.0, -15.0), "raised"),
            ("roll_pitch_pp", (0.0, 0.0, 0.03), (20.0, 20.0, 0.0), "raised"),
            ("roll_pitch_pn", (0.0, 0.0, 0.03), (20.0, -20.0, 0.0), "raised"),
            ("roll_pitch_np", (0.0, 0.0, 0.03), (-20.0, 20.0, 0.0), "raised"),
            ("roll_pitch_nn", (0.0, 0.0, 0.03), (-20.0, -20.0, 0.0), "raised"),
            ("x_pos_pitch", (0.04, 0.0, 0.03), (0.0, 20.0, 0.0), "raised"),
            ("x_neg_pitch", (-0.04, 0.0, 0.03), (0.0, -20.0, 0.0), "raised"),
            ("y_pos_roll", (0.0, 0.04, 0.03), (-20.0, 0.0, 0.0), "raised"),
            ("y_neg_roll", (0.0, -0.04, 0.03), (20.0, 0.0, 0.0), "raised"),
            (
                "xyz_rpy_combo",
                (0.03, 0.03, 0.03),
                (10.0, -10.0, 10.0),
                "raised",
            ),
        ]
    elif phase == "validation":
        specs = [
            ("holdout_xy_pp", (0.03, 0.03, 0.03), (10.0, -10.0, 0.0), "raised"),
            ("holdout_xy_np", (-0.03, 0.03, 0.03), (-10.0, -10.0, 0.0), "raised"),
            ("holdout_xy_pn", (0.03, -0.03, 0.03), (10.0, 10.0, 0.0), "raised"),
            ("holdout_xy_nn", (-0.03, -0.03, 0.03), (-10.0, 10.0, 0.0), "raised"),
            ("holdout_rpy", (0.0, 0.0, 0.03), (12.0, 12.0, 10.0), "raised"),
        ]
    else:
        raise ValueError("phase must be 'calibration' or 'validation'")

    return [
        TrajectoryPose(
            sample_id=f"{phase}_{index:03d}_{name}",
            phase=phase,
            group=name,
            translation_offset_base_m=translation,
            rpy_offset_tool_deg=rpy,
            hub=hub,
        )
        for index, (name, translation, rpy, hub) in enumerate(specs)
    ]


def _finite_vector(name: str, value: Sequence[float], length: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def make_transform(rotation: Sequence[Sequence[float]], translation: Sequence[float]) -> np.ndarray:
    rotation_array = np.asarray(rotation, dtype=np.float64)
    translation_array = _finite_vector("translation", translation, 3)
    if rotation_array.shape != (3, 3) or not np.all(np.isfinite(rotation_array)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_array
    transform[:3, 3] = translation_array
    return transform


def invert_transform(transform: Sequence[Sequence[float]]) -> np.ndarray:
    transform_array = np.asarray(transform, dtype=np.float64)
    if transform_array.shape != (4, 4) or not np.all(np.isfinite(transform_array)):
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = transform_array[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform_array[:3, 3]
    return result


def quaternion_xyzw_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    q = _finite_vector("quaternion_xyzw", quaternion, 4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_to_quaternion_xyzw(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.asarray(
            [
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
                0.25 * s,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(max(0.0, 1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2
            q = np.asarray(
                [
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[2, 1] - matrix[1, 2]) / s,
                ]
            )
        elif index == 1:
            s = math.sqrt(max(0.0, 1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2
            q = np.asarray(
                [
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    (matrix[0, 2] - matrix[2, 0]) / s,
                ]
            )
        else:
            s = math.sqrt(max(0.0, 1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2
            q = np.asarray(
                [
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                    (matrix[1, 0] - matrix[0, 1]) / s,
                ]
            )
    q /= np.linalg.norm(q)
    return q


def rpy_degrees_to_rotation(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = np.radians(_finite_vector("rpy_deg", rpy_deg, 3))
    rx = np.asarray(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]],
        dtype=np.float64,
    )
    ry = np.asarray(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]],
        dtype=np.float64,
    )
    rz = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def transform_from_pose(
    translation: Sequence[float], quaternion_xyzw: Sequence[float]
) -> np.ndarray:
    return make_transform(quaternion_xyzw_to_rotation(quaternion_xyzw), translation)


def target_transform(center_base_to_gripper: np.ndarray, pose: TrajectoryPose) -> np.ndarray:
    result = np.asarray(center_base_to_gripper, dtype=np.float64).copy()
    if result.shape != (4, 4):
        raise ValueError("center_base_to_gripper must be 4x4")
    result[:3, 3] += np.asarray(pose.translation_offset_base_m, dtype=np.float64)
    result[:3, :3] = (
        center_base_to_gripper[:3, :3]
        @ rpy_degrees_to_rotation(pose.rpy_offset_tool_deg)
    )
    return result


def hub_transform(center_base_to_gripper: np.ndarray, hub: str) -> np.ndarray:
    result = np.asarray(center_base_to_gripper, dtype=np.float64).copy()
    if hub == "raised":
        result[2, 3] += 0.03
    elif hub != "center":
        raise ValueError(f"Unknown trajectory hub: {hub}")
    return result


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = np.asarray(rotation_a).T @ np.asarray(rotation_b)
    cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def transform_error(transform_a: np.ndarray, transform_b: np.ndarray) -> tuple[float, float]:
    translation = float(
        np.linalg.norm(np.asarray(transform_a)[:3, 3] - np.asarray(transform_b)[:3, 3])
    )
    rotation = rotation_error_deg(
        np.asarray(transform_a)[:3, :3], np.asarray(transform_b)[:3, :3]
    )
    return translation, rotation


def interpolate_transforms(
    start: np.ndarray,
    target: np.ndarray,
    max_translation_m: float,
    max_rotation_deg: float,
) -> list[np.ndarray]:
    start_array = np.asarray(start, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    translation_distance, rotation_distance = transform_error(start_array, target_array)
    steps = max(
        1,
        math.ceil(translation_distance / max_translation_m),
        math.ceil(rotation_distance / max_rotation_deg),
    )
    relative_rotation = start_array[:3, :3].T @ target_array[:3, :3]
    relative_rvec, _ = cv2.Rodrigues(relative_rotation)
    results = []
    for index in range(1, steps + 1):
        alpha = index / steps
        rotation_delta, _ = cv2.Rodrigues(relative_rvec * alpha)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = start_array[:3, :3] @ rotation_delta
        transform[:3, 3] = (
            start_array[:3, 3]
            + alpha * (target_array[:3, 3] - start_array[:3, 3])
        )
        results.append(transform)
    results[-1] = target_array.copy()
    return results


def validate_target_transform(
    transform: np.ndarray,
    config: EyeToHandConfig,
) -> list[str]:
    translation = np.asarray(transform, dtype=np.float64)[:3, 3]
    minimum = np.asarray(config.robot.workspace_minimum_m, dtype=np.float64)
    maximum = np.asarray(config.robot.workspace_maximum_m, dtype=np.float64)
    failures = []
    if np.any(translation < minimum) or np.any(translation > maximum):
        failures.append(
            f"TCP translation {translation.tolist()} lies outside workspace "
            f"{minimum.tolist()}..{maximum.tolist()}"
        )
    lowest_envelope_z = float(translation[2] - config.robot.tool_envelope_radius_m)
    if lowest_envelope_z < minimum[2]:
        failures.append(
            f"tool envelope lowest z {lowest_envelope_z:.6f} m is below "
            f"configured minimum {minimum[2]:.6f} m"
        )
    return failures


def checkerboard_object_points(board: BoardConfig) -> np.ndarray:
    points = np.zeros(
        (board.inner_corners_rows * board.inner_corners_columns, 3),
        dtype=np.float64,
    )
    grid = np.mgrid[
        0 : board.inner_corners_columns,
        0 : board.inner_corners_rows,
    ].T.reshape(-1, 2)
    points[:, :2] = grid * board.square_size_m
    return points


def detect_checkerboard(
    rgb: np.ndarray,
    board: BoardConfig,
) -> tuple[np.ndarray | None, float]:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("checkerboard image must be RGB HWC")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # At steep perspective angles findChessboardCornersSB can extrapolate a
    # complete row or column onto the dark table cloth. The classic detector
    # plus sub-pixel refinement is more stable for the physical board. Keep SB
    # as a fallback for fronto-parallel or low-contrast views.
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(
        gray, board.pattern_size, flags=classic_flags
    )
    if found and corners is not None:
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        )
        object_points_2d = checkerboard_object_points(board)[:, :2]
        refinements = []
        for half_window in range(2, 12):
            refined = cv2.cornerSubPix(
                gray,
                np.asarray(corners, dtype=np.float32).copy(),
                (half_window, half_window),
                (-1, -1),
                criteria,
            )
            refined_2d = np.asarray(refined, dtype=np.float64).reshape(-1, 2)
            homography, _ = cv2.findHomography(object_points_2d, refined_2d, 0)
            if homography is None:
                continue
            projected = cv2.perspectiveTransform(
                object_points_2d.reshape(-1, 1, 2),
                homography,
            ).reshape(-1, 2)
            residual = projected - refined_2d
            homography_rms_px = float(
                np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
            )
            refinements.append((homography_rms_px, half_window, refined_2d))
        if refinements:
            _, _, best_corners = min(
                refinements,
                key=lambda item: (item[0], item[1]),
            )
            return best_corners, sharpness

    sb_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    found, corners = cv2.findChessboardCornersSB(
        gray, board.pattern_size, flags=sb_flags
    )
    if not found or corners is None:
        return None, sharpness
    return np.asarray(corners, dtype=np.float64).reshape(-1, 2), sharpness


def estimate_target_to_camera(
    corners_px: np.ndarray,
    camera_matrix: Sequence[Sequence[float]],
    distortion_coefficients: Sequence[float],
    board: BoardConfig,
) -> dict[str, Any]:
    image_points = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    object_points = checkerboard_object_points(board)
    if image_points.shape[0] != object_points.shape[0]:
        raise ValueError(
            f"Expected {object_points.shape[0]} checkerboard corners, got {image_points.shape[0]}"
        )
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1, 1)
    result = cv2.solvePnPGeneric(
        object_points,
        image_points,
        matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result[0]:
        raise RuntimeError("SOLVEPNP_IPPE failed")
    rvecs, tvecs = result[1], result[2]
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            matrix,
            distortion,
            np.asarray(rvec, dtype=np.float64),
            np.asarray(tvec, dtype=np.float64),
        )
        rotation, _ = cv2.Rodrigues(refined_rvec)
        camera_points = (rotation @ object_points.T + refined_tvec.reshape(3, 1)).T
        if float(np.min(camera_points[:, 2])) <= 0.0:
            continue
        projected, _ = cv2.projectPoints(
            object_points, refined_rvec, refined_tvec, matrix, distortion
        )
        residual = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        transform = make_transform(rotation, refined_tvec.reshape(3))
        candidates.append((rms, refined_rvec, refined_tvec, transform))
    if not candidates:
        raise RuntimeError("PnP returned no positive-depth checkerboard pose")
    rms, rvec, tvec, transform = min(candidates, key=lambda item: item[0])
    return {
        "method": "SOLVEPNP_IPPE+RefineLM",
        "reprojection_rms_px": rms,
        "rvec": np.asarray(rvec).reshape(3).tolist(),
        "tvec_m": np.asarray(tvec).reshape(3).tolist(),
        "target_to_camera": transform.tolist(),
        "candidate_count": len(candidates),
    }


def checkerboard_corner_symmetries(
    corners_px: np.ndarray,
    board: BoardConfig,
) -> list[tuple[str, np.ndarray]]:
    rows = board.inner_corners_rows
    columns = board.inner_corners_columns
    grid = np.asarray(corners_px, dtype=np.float64).reshape(rows, columns, 2)
    variants: list[tuple[str, np.ndarray]] = [
        ("identity", grid),
        ("rotate_180", np.rot90(grid, 2)),
        ("flip_vertical", np.flipud(grid)),
        ("flip_horizontal", np.fliplr(grid)),
    ]
    if rows == columns:
        variants.extend(
            [
                ("rotate_90", np.rot90(grid, 1)),
                ("rotate_270", np.rot90(grid, 3)),
                ("transpose", np.transpose(grid, (1, 0, 2))),
                (
                    "anti_transpose",
                    np.flipud(np.fliplr(np.transpose(grid, (1, 0, 2)))),
                ),
            ]
        )
    return [
        (name, np.ascontiguousarray(candidate).reshape(-1, 2))
        for name, candidate in variants
    ]


def estimate_target_to_camera_candidates(
    corners_px: np.ndarray,
    camera_matrix: Sequence[Sequence[float]],
    distortion_coefficients: Sequence[float],
    board: BoardConfig,
) -> list[dict[str, Any]]:
    candidates = []
    for symmetry_index, (symmetry, reordered) in enumerate(
        checkerboard_corner_symmetries(corners_px, board)
    ):
        try:
            result = estimate_target_to_camera(
                reordered,
                camera_matrix,
                distortion_coefficients,
                board,
            )
            candidates.append(
                {
                    **result,
                    "symmetry_index": symmetry_index,
                    "symmetry": symmetry,
                }
            )
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("PnP failed for every checkerboard symmetry")
    return candidates


def draw_checkerboard_overlay(
    rgb: np.ndarray,
    corners_px: np.ndarray,
    board: BoardConfig,
) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    corners = np.asarray(corners_px, dtype=np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(bgr, board.pattern_size, corners, True)
    for index, point in enumerate(corners.reshape(-1, 2)):
        x, y = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.putText(
            bgr,
            str(index),
            (x + 3, y - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 255) if index == 0 else (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _mean_rotation(rotations: Iterable[np.ndarray]) -> np.ndarray:
    quaternions = [rotation_to_quaternion_xyzw(rotation) for rotation in rotations]
    reference = quaternions[0]
    aligned = [q if float(np.dot(q, reference)) >= 0.0 else -q for q in quaternions]
    accumulator = sum(np.outer(q, q) for q in aligned)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    quaternion = eigenvectors[:, int(np.argmax(eigenvalues))]
    if float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    return quaternion_xyzw_to_rotation(quaternion)


def robust_mean_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("At least one transform is required")
    translations = np.stack([transform[:3, 3] for transform in transforms])
    return make_transform(
        _mean_rotation([transform[:3, :3] for transform in transforms]),
        np.median(translations, axis=0),
    )


def _candidate_metrics(
    base_to_camera: np.ndarray,
    samples: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    gripper_to_targets = []
    for sample in samples:
        base_to_gripper = np.asarray(sample["base_to_gripper"], dtype=np.float64)
        target_to_camera = np.asarray(sample["target_to_camera"], dtype=np.float64)
        gripper_to_targets.append(
            invert_transform(base_to_gripper) @ base_to_camera @ target_to_camera
        )
    reference = robust_mean_transform(gripper_to_targets)
    residuals = []
    for sample, transform in zip(samples, gripper_to_targets):
        translation_m, rotation_deg = transform_error(reference, transform)
        residuals.append(
            {
                "sample_id": sample["sample_id"],
                "translation_m": translation_m,
                "rotation_deg": rotation_deg,
            }
        )
    translations = np.asarray([entry["translation_m"] for entry in residuals])
    rotations = np.asarray([entry["rotation_deg"] for entry in residuals])
    metrics = {
        "median_translation_m": float(np.median(translations)),
        "max_translation_m": float(np.max(translations)),
        "median_rotation_deg": float(np.median(rotations)),
        "max_rotation_deg": float(np.max(rotations)),
        "residuals": residuals,
    }
    return metrics, reference


def _validate_sample(sample: dict[str, Any]) -> None:
    for key in ("sample_id", "base_to_gripper", "target_to_camera"):
        if key not in sample:
            raise ValueError(f"Hand-eye sample is missing {key}")
    for key in ("base_to_gripper", "target_to_camera"):
        transform = np.asarray(sample[key], dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"Sample {sample['sample_id']} has invalid {key}")


def _fit_base_to_camera_for_assignments(
    samples: Sequence[dict[str, Any]],
    assignments: Sequence[int],
    method: int,
) -> np.ndarray:
    gripper_to_base = [
        invert_transform(np.asarray(sample["base_to_gripper"], dtype=np.float64))
        for sample in samples
    ]
    target_to_camera = [
        np.asarray(
            sample["target_to_camera_candidates"][assignment]["target_to_camera"],
            dtype=np.float64,
        )
        for sample, assignment in zip(samples, assignments)
    ]
    rotation, translation = cv2.calibrateHandEye(
        [transform[:3, :3] for transform in gripper_to_base],
        [transform[:3, 3].reshape(3, 1) for transform in gripper_to_base],
        [transform[:3, :3] for transform in target_to_camera],
        [transform[:3, 3].reshape(3, 1) for transform in target_to_camera],
        method=method,
    )
    result = make_transform(rotation, np.asarray(translation).reshape(3))
    if not np.all(np.isfinite(result)):
        raise ValueError("Symmetry seed returned NaN or Inf")
    return result


def _assigned_gripper_to_targets(
    samples: Sequence[dict[str, Any]],
    assignments: Sequence[int],
    base_to_camera: np.ndarray,
) -> list[np.ndarray]:
    return [
        invert_transform(np.asarray(sample["base_to_gripper"], dtype=np.float64))
        @ base_to_camera
        @ np.asarray(
            sample["target_to_camera_candidates"][assignment]["target_to_camera"],
            dtype=np.float64,
        )
        for sample, assignment in zip(samples, assignments)
    ]


def resolve_checkerboard_symmetry(
    samples: Sequence[dict[str, Any]],
    quality: QualityConfig,
    max_iterations: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(samples) < 3:
        raise ValueError("At least 3 samples are required to resolve checkerboard symmetry")
    candidate_count = len(samples[0].get("target_to_camera_candidates", []))
    if candidate_count < 2:
        raise ValueError("Samples do not contain checkerboard symmetry candidates")
    for sample in samples:
        if len(sample.get("target_to_camera_candidates", [])) != candidate_count:
            raise ValueError("All samples must contain the same symmetry candidates")

    seed_methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    }
    solutions = []
    for seed_index in range(candidate_count):
        for method_name, method in seed_methods.items():
            assignments = [seed_index] * len(samples)
            try:
                for iteration in range(max_iterations):
                    base_to_camera = _fit_base_to_camera_for_assignments(
                        samples, assignments, method
                    )
                    transforms = _assigned_gripper_to_targets(
                        samples, assignments, base_to_camera
                    )
                    reference = robust_mean_transform(transforms)
                    updated = []
                    for sample in samples:
                        gripper_to_base = invert_transform(
                            np.asarray(sample["base_to_gripper"], dtype=np.float64)
                        )
                        scores = []
                        for candidate_index, candidate in enumerate(
                            sample["target_to_camera_candidates"]
                        ):
                            gripper_to_target = (
                                gripper_to_base
                                @ base_to_camera
                                @ np.asarray(candidate["target_to_camera"])
                            )
                            translation_m, rotation_deg = transform_error(
                                reference, gripper_to_target
                            )
                            score = (
                                translation_m / quality.consensus_translation_m
                                + rotation_deg / quality.consensus_rotation_deg
                            )
                            scores.append((score, candidate_index))
                        updated.append(min(scores)[1])
                    if updated == assignments:
                        break
                    assignments = updated

                base_to_camera = _fit_base_to_camera_for_assignments(
                    samples, assignments, method
                )
                transforms = _assigned_gripper_to_targets(
                    samples, assignments, base_to_camera
                )
                reference = robust_mean_transform(transforms)
                residuals = [
                    transform_error(reference, transform)
                    for transform in transforms
                ]
                median_translation_m = float(
                    np.median([residual[0] for residual in residuals])
                )
                median_rotation_deg = float(
                    np.median([residual[1] for residual in residuals])
                )
                max_translation_m = float(max(residual[0] for residual in residuals))
                max_rotation_deg = float(max(residual[1] for residual in residuals))
                score = (
                    median_translation_m / quality.consensus_translation_m
                    + median_rotation_deg / quality.consensus_rotation_deg
                )
                solutions.append(
                    {
                        "score": score,
                        "seed_index": seed_index,
                        "seed_method": method_name,
                        "iterations": iteration + 1,
                        "assignments": assignments,
                        "base_to_camera": base_to_camera,
                        "gripper_to_target_reference": reference,
                        "median_translation_m": median_translation_m,
                        "median_rotation_deg": median_rotation_deg,
                        "max_translation_m": max_translation_m,
                        "max_rotation_deg": max_rotation_deg,
                    }
                )
            except Exception:
                continue
    if not solutions:
        raise RuntimeError("No checkerboard symmetry seed converged")
    best = min(
        solutions,
        key=lambda solution: (
            solution["score"],
            solution["max_translation_m"],
            solution["max_rotation_deg"],
            solution["seed_method"],
            solution["seed_index"],
        ),
    )
    resolved = []
    assignment_report = []
    for sample, assignment in zip(samples, best["assignments"]):
        candidate = sample["target_to_camera_candidates"][assignment]
        entry = dict(sample)
        entry["target_to_camera"] = candidate["target_to_camera"]
        entry["reprojection_rms_px"] = candidate["reprojection_rms_px"]
        entry["resolved_symmetry"] = candidate["symmetry"]
        entry["resolved_symmetry_index"] = candidate["symmetry_index"]
        resolved.append(entry)
        assignment_report.append(
            {
                "sample_id": sample["sample_id"],
                "symmetry": candidate["symmetry"],
                "symmetry_index": candidate["symmetry_index"],
                "reprojection_rms_px": candidate["reprojection_rms_px"],
            }
        )
    diagnostics = {
        "required": True,
        "reason": (
            "Checkerboard corner ordering can change under rotations or reflections; "
            "robot motion consistency resolves the physical target frame."
        ),
        "seed_method": best["seed_method"],
        "seed_index": best["seed_index"],
        "iterations": best["iterations"],
        "score": best["score"],
        "median_translation_m": best["median_translation_m"],
        "median_rotation_deg": best["median_rotation_deg"],
        "max_translation_m": best["max_translation_m"],
        "max_rotation_deg": best["max_rotation_deg"],
        "diagnostic_base_to_camera": best["base_to_camera"].tolist(),
        "gripper_to_target_reference": best[
            "gripper_to_target_reference"
        ].tolist(),
        "assignments": assignment_report,
        "converged_seed_count": len(solutions),
    }
    return resolved, diagnostics


def resolve_validation_symmetry(
    samples: Sequence[dict[str, Any]],
    base_to_camera: Sequence[Sequence[float]],
    gripper_to_target_reference: Sequence[Sequence[float]],
    quality: QualityConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base_to_camera_array = np.asarray(base_to_camera, dtype=np.float64)
    reference = np.asarray(gripper_to_target_reference, dtype=np.float64)
    resolved = []
    diagnostics = []
    for sample in samples:
        gripper_to_base = invert_transform(
            np.asarray(sample["base_to_gripper"], dtype=np.float64)
        )
        scores = []
        for candidate_index, candidate in enumerate(
            sample["target_to_camera_candidates"]
        ):
            gripper_to_target = (
                gripper_to_base
                @ base_to_camera_array
                @ np.asarray(candidate["target_to_camera"])
            )
            translation_m, rotation_deg = transform_error(
                reference, gripper_to_target
            )
            score = (
                translation_m / quality.validation_translation_m
                + rotation_deg / quality.validation_rotation_deg
            )
            scores.append(
                (
                    score,
                    candidate_index,
                    translation_m,
                    rotation_deg,
                )
            )
        _, assignment, translation_m, rotation_deg = min(scores)
        candidate = sample["target_to_camera_candidates"][assignment]
        entry = dict(sample)
        entry["target_to_camera"] = candidate["target_to_camera"]
        entry["reprojection_rms_px"] = candidate["reprojection_rms_px"]
        entry["resolved_symmetry"] = candidate["symmetry"]
        entry["resolved_symmetry_index"] = candidate["symmetry_index"]
        resolved.append(entry)
        diagnostics.append(
            {
                "sample_id": sample["sample_id"],
                "symmetry": candidate["symmetry"],
                "symmetry_index": candidate["symmetry_index"],
                "seed_translation_m": translation_m,
                "seed_rotation_deg": rotation_deg,
            }
        )
    return resolved, diagnostics


def solve_eye_to_hand(
    calibration_samples: Sequence[dict[str, Any]],
    validation_samples: Sequence[dict[str, Any]],
    quality: QualityConfig,
    *,
    use_available_samples: bool = False,
) -> dict[str, Any]:
    for sample in [*calibration_samples, *validation_samples]:
        _validate_sample(sample)

    methods = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    gripper_to_base = [
        invert_transform(np.asarray(sample["base_to_gripper"], dtype=np.float64))
        for sample in calibration_samples
    ]
    target_to_camera = [
        np.asarray(sample["target_to_camera"], dtype=np.float64)
        for sample in calibration_samples
    ]
    candidates: dict[str, dict[str, Any]] = {}
    for name, method in methods.items():
        try:
            rotation, translation = cv2.calibrateHandEye(
                [transform[:3, :3] for transform in gripper_to_base],
                [transform[:3, 3].reshape(3, 1) for transform in gripper_to_base],
                [transform[:3, :3] for transform in target_to_camera],
                [transform[:3, 3].reshape(3, 1) for transform in target_to_camera],
                method=method,
            )
            base_to_camera = make_transform(rotation, np.asarray(translation).reshape(3))
            if not np.all(np.isfinite(base_to_camera)):
                raise ValueError("method returned NaN or Inf")
            determinant = float(np.linalg.det(base_to_camera[:3, :3]))
            if abs(determinant - 1.0) > 1e-3:
                raise ValueError(f"rotation determinant is {determinant}")
            metrics, gripper_to_target = _candidate_metrics(
                base_to_camera, calibration_samples
            )
            score = (
                metrics["median_translation_m"] / quality.consensus_translation_m
                + metrics["median_rotation_deg"] / quality.consensus_rotation_deg
            )
            candidates[name] = {
                "valid": True,
                "base_to_camera": base_to_camera.tolist(),
                "camera_to_base": invert_transform(base_to_camera).tolist(),
                "base_to_camera_quaternion_xyzw": rotation_to_quaternion_xyzw(
                    base_to_camera[:3, :3]
                ).tolist(),
                "score": float(score),
                "calibration_metrics": metrics,
                "gripper_to_target_reference": gripper_to_target.tolist(),
            }
        except Exception as exc:
            candidates[name] = {"valid": False, "error": str(exc)}

    valid_names = [name for name, result in candidates.items() if result["valid"]]
    consensus: dict[str, list[str]] = {}
    for name in valid_names:
        base_to_camera = np.asarray(candidates[name]["base_to_camera"])
        peers = []
        for peer_name in valid_names:
            peer = np.asarray(candidates[peer_name]["base_to_camera"])
            translation_m, rotation_deg = transform_error(base_to_camera, peer)
            if (
                translation_m <= quality.consensus_translation_m
                and rotation_deg <= quality.consensus_rotation_deg
            ):
                peers.append(peer_name)
        consensus[name] = peers

    selected_name: str | None = None
    if valid_names:
        selected_name = min(
            valid_names,
            key=lambda name: (
                -len(consensus[name]),
                candidates[name]["score"],
                name,
            ),
        )
    consensus_names = consensus.get(selected_name, []) if selected_name else []
    method_pass = (
        len(valid_names) >= quality.minimum_valid_methods
        and len(consensus_names) >= quality.minimum_valid_methods
    )

    validation_results = []
    required_calibration_samples = (
        3 if use_available_samples else quality.required_calibration_samples
    )
    required_validation_samples = (
        0 if use_available_samples else quality.required_validation_samples
    )
    dataset_complete = (
        len(calibration_samples) >= quality.required_calibration_samples
        and len(validation_samples) >= quality.required_validation_samples
    )
    validation_pass = len(validation_samples) >= required_validation_samples
    if selected_name is not None:
        selected = candidates[selected_name]
        base_to_camera = np.asarray(selected["base_to_camera"])
        reference = np.asarray(selected["gripper_to_target_reference"])
        for sample in validation_samples:
            gripper_to_target = (
                invert_transform(np.asarray(sample["base_to_gripper"]))
                @ base_to_camera
                @ np.asarray(sample["target_to_camera"])
            )
            translation_m, rotation_deg = transform_error(reference, gripper_to_target)
            reprojection = float(sample.get("reprojection_rms_px", math.inf))
            passed = (
                translation_m <= quality.validation_translation_m
                and rotation_deg <= quality.validation_rotation_deg
                and reprojection <= quality.max_reprojection_rms_px
            )
            validation_results.append(
                {
                    "sample_id": sample["sample_id"],
                    "translation_m": translation_m,
                    "rotation_deg": rotation_deg,
                    "reprojection_rms_px": reprojection,
                    "passed": passed,
                }
            )
        validation_pass = validation_pass and all(
            result["passed"] for result in validation_results
        )
    else:
        validation_pass = False

    reprojection_failures = [
        sample["sample_id"]
        for sample in [*calibration_samples, *validation_samples]
        if float(sample.get("reprojection_rms_px", math.inf))
        > quality.max_reprojection_rms_px
    ]
    sample_count_pass = (
        len(calibration_samples) >= required_calibration_samples
        and len(validation_samples) >= required_validation_samples
    )
    reprojection_pass = not reprojection_failures
    accepted = bool(
        selected_name
        and sample_count_pass
        and reprojection_pass
        and method_pass
        and validation_pass
    )
    selected_transform = (
        candidates[selected_name]["base_to_camera"]
        if accepted and selected_name is not None
        else None
    )
    selected_inverse = (
        candidates[selected_name]["camera_to_base"]
        if accepted and selected_name is not None
        else None
    )
    selected_quaternion = (
        candidates[selected_name]["base_to_camera_quaternion_xyzw"]
        if accepted and selected_name is not None
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "fit_mode": "available" if use_available_samples else "complete",
        "accepted": accepted,
        "provisional": bool(accepted and not dataset_complete),
        "deployment_ready": bool(accepted and dataset_complete),
        "frame_convention": {
            "result": "T_base_color transforms RGB color optical-frame points into Franka base-frame points",
            "equation": "T_gripper_base * T_base_color * T_color_target = T_gripper_target",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "translation_unit": "meter",
            "quaternion_order": "xyzw",
            "depth_note": (
                "Depth points require RealSense depth-to-color composition or "
                "alignment before applying T_base_color."
            ),
        },
        "checks": {
            "sample_count_pass": sample_count_pass,
            "dataset_complete": dataset_complete,
            "reprojection_pass": reprojection_pass,
            "method_consensus_pass": method_pass,
            "validation_pass": validation_pass,
            "validation_available": bool(validation_samples),
            "calibration_sample_count": len(calibration_samples),
            "validation_sample_count": len(validation_samples),
            "required_calibration_sample_count": required_calibration_samples,
            "required_validation_sample_count": required_validation_samples,
            "configured_calibration_sample_count": quality.required_calibration_samples,
            "configured_validation_sample_count": quality.required_validation_samples,
            "reprojection_failures": reprojection_failures,
            "valid_method_count": len(valid_names),
            "selected_consensus_methods": consensus_names,
        },
        "selected_method": selected_name,
        "T_base_color": selected_transform,
        "T_color_base": selected_inverse,
        "base_to_color_quaternion_xyzw": selected_quaternion,
        "diagnostic_selected_candidate": (
            candidates[selected_name] if selected_name is not None else None
        ),
        "methods": candidates,
        "method_consensus": consensus,
        "validation": validation_results,
    }
