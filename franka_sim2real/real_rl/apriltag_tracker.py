from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import AprilTagConfig


@dataclass(frozen=True)
class TagPose:
    sequence: int
    capture_timestamp: float
    camera_timestamp_ms: float | None
    object_position_base_m: np.ndarray
    reprojection_rms_px: float

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (time.monotonic() if now is None else now) - self.capture_timestamp)

    @property
    def host_monotonic_s(self) -> float:
        """兼容旧调用方；Replay v2 的契约名称是 capture_timestamp。"""
        return self.capture_timestamp


def load_base_t_camera(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    solve = payload.get("solve", {})
    if solve.get("accepted") is not True:
        raise ValueError(f"Eye-to-hand calibration is not accepted: {path}")
    transform = np.asarray(solve.get("T_base_color"), dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"Invalid T_base_color in {path}")
    return transform


def camera_calibration(intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(
        [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64,
    )
    distortion = np.asarray(intrinsics["distortion_coefficients"], dtype=np.float64)
    model = str(intrinsics.get("distortion_model", "none"))
    if np.any(np.abs(distortion) > 1e-12) and model not in {
        "brown_conrady", "modified_brown_conrady",
    }:
        raise ValueError(f"Unsupported RealSense distortion model: {model}")
    return matrix, distortion


def _tag_points(length_m: float) -> np.ndarray:
    half = 0.5 * length_m
    return np.asarray(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]],
        dtype=np.float64,
    )


def solve_tag_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_length_m: float,
) -> tuple[np.ndarray, float]:
    points = _tag_points(marker_length_m)
    image = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    result = cv2.solvePnPGeneric(
        points, image, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    success, rvecs, tvecs = result[:3]
    if not success:
        raise RuntimeError("AprilTag PnP failed")
    candidates: list[tuple[float, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        if tvec[2, 0] <= 0.0:
            continue
        rvec, tvec = cv2.solvePnPRefineLM(points, image, camera_matrix, distortion, rvec, tvec)
        projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, distortion)
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image) ** 2, axis=1))))
        rotation, _ = cv2.Rodrigues(rvec)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = tvec.reshape(3)
        candidates.append((error, transform))
    if not candidates:
        raise RuntimeError("AprilTag PnP placed all candidates behind the camera")
    error, transform = min(candidates, key=lambda value: value[0])
    return transform, error


class AprilTagTracker:
    """Latest-only detector consuming raw frames from the policy RealSense."""

    def __init__(self, camera: Any, config: AprilTagConfig) -> None:
        config.validate()
        if not hasattr(camera, "read_raw_packet") or not hasattr(camera, "camera_intrinsics"):
            raise ValueError("Real-RL collect requires a RealSense camera with raw frame packets")
        self.camera = camera
        self.config = config
        self.base_t_camera = load_base_t_camera(config.calibration_report)
        self.camera_matrix, self.distortion = camera_calibration(camera.camera_intrinsics())
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._positions: deque[np.ndarray] = deque(maxlen=config.smoothing_window)
        self._lock = threading.Lock()
        self._latest: TagPose | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._condition = threading.Condition(self._lock)
        self._thread = threading.Thread(target=self._loop, name="real-rl-apriltag", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        last_sequence = -1
        try:
            while True:
                with self._lock:
                    if self._closing:
                        return
                packet = self.camera.read_raw_packet()
                if packet.sequence == last_sequence:
                    time.sleep(0.002)
                    continue
                last_sequence = packet.sequence
                gray = cv2.cvtColor(packet.raw_rgb, cv2.COLOR_RGB2GRAY)
                scale = self.config.detection_scale
                detected = gray if scale == 1.0 else cv2.resize(
                    gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                corners, ids, _rejected = self.detector.detectMarkers(detected)
                if scale != 1.0:
                    corners = [value / scale for value in corners]
                if ids is None:
                    continue
                matches = np.flatnonzero(ids.reshape(-1) == self.config.marker_id)
                if len(matches) == 0:
                    continue
                camera_t_tag, error = solve_tag_pose(
                    corners[int(matches[0])], self.camera_matrix, self.distortion,
                    self.config.marker_length_m,
                )
                if error > self.config.max_reprojection_px:
                    continue
                tag_t_object = np.eye(4, dtype=np.float64)
                tag_t_object[:3, 3] = np.asarray(self.config.tag_to_object_m, dtype=np.float64)
                base_t_object = self.base_t_camera @ camera_t_tag @ tag_t_object
                position = base_t_object[:3, 3].copy()
                position[2] += self.config.height_offset_m
                self._positions.append(position)
                smoothed = np.median(np.stack(tuple(self._positions)), axis=0)
                pose = TagPose(
                    sequence=packet.sequence,
                    capture_timestamp=packet.capture_timestamp,
                    camera_timestamp_ms=packet.camera_timestamp_ms,
                    object_position_base_m=smoothed.astype(np.float64),
                    reprojection_rms_px=error,
                )
                with self._condition:
                    self._latest = pose
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                if not self._closing:
                    self._error = exc
                    self._condition.notify_all()

    def check(self) -> None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"AprilTag tracker failed: {self._error}") from self._error

    def latest(self, *, allow_stale: bool = False) -> TagPose | None:
        self.check()
        with self._lock:
            pose = self._latest
        if pose is None:
            return None
        if not allow_stale and pose.age_s() > self.config.max_pose_age_s:
            return None
        return pose

    def wait_for_initial_object_z_in_base(self) -> float:
        deadline = time.monotonic() + self.config.preflight_timeout_s
        positions: list[float] = []
        last_sequence = -1
        while len(positions) < self.config.preflight_valid_detections:
            self.check()
            pose = self.latest()
            if pose is not None and pose.sequence != last_sequence:
                positions.append(float(pose.object_position_base_m[2]))
                last_sequence = pose.sequence
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    "AprilTag preflight timed out: "
                    f"{len(positions)}/{self.config.preflight_valid_detections} valid detections"
                )
            with self._condition:
                self._condition.wait(timeout=min(0.05, remaining))
        return float(np.median(np.asarray(positions, dtype=np.float64)))

    def wait_for_initial_object_z(self) -> float:
        """兼容旧调用方；新方法名显式标注了 base 坐标系。"""
        return self.wait_for_initial_object_z_in_base()

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("AprilTag tracker did not stop")

    def __enter__(self) -> "AprilTagTracker":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
