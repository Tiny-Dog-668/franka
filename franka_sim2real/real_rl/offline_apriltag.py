from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .apriltag_tracker import camera_calibration, load_base_t_camera, solve_tag_pose
from .config import RealRLConfig
from .replay_buffer import ReplayBuffer, _pack, _unpack
from .reward import compute_progress_reward


@dataclass(frozen=True)
class OfflineTagPose:
    boundary: int
    sequence: int
    capture_timestamp: float
    camera_timestamp_ms: float | None
    object_position_base_m: np.ndarray
    reprojection_rms_px: float
    corners_px: np.ndarray
    search_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "detected": True,
            "sequence": self.sequence,
            "capture_timestamp": self.capture_timestamp,
            "camera_timestamp_ms": self.camera_timestamp_ms,
            "object_position_base_m": self.object_position_base_m.tolist(),
            "reprojection_rms_px": self.reprojection_rms_px,
            "corners_px": self.corners_px.tolist(),
            "search_mode": self.search_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OfflineTagPose":
        if payload.get("detected") is not True:
            raise ValueError("Cannot construct OfflineTagPose from a missed detection")
        return cls(
            boundary=int(payload["boundary"]),
            sequence=int(payload["sequence"]),
            capture_timestamp=float(payload["capture_timestamp"]),
            camera_timestamp_ms=(
                None
                if payload.get("camera_timestamp_ms") is None
                else float(payload["camera_timestamp_ms"])
            ),
            object_position_base_m=np.asarray(
                payload["object_position_base_m"], dtype=np.float64
            ).reshape(3),
            reprojection_rms_px=float(payload["reprojection_rms_px"]),
            corners_px=np.asarray(payload["corners_px"], dtype=np.float64).reshape(4, 2),
            search_mode=str(payload["search_mode"]),
        )


def _load_saved_detections(
    run_dir: Path,
) -> tuple[dict[int, OfflineTagPose | None], list[dict[str, Any]]]:
    path = run_dir / "offline_apriltag_labels.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved offline AprilTag labels: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    poses: dict[int, OfflineTagPose | None] = {}
    for item in records:
        boundary = int(item["boundary"])
        poses[boundary] = (
            OfflineTagPose.from_dict(item) if item.get("detected") is True else None
        )
    if not poses:
        raise RuntimeError(f"Saved AprilTag label file is empty: {path}")
    return poses, records


def _equal_training_value(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (bytes, bytearray, memoryview)) or isinstance(
        right, (bytes, bytearray, memoryview)
    ):
        return bytes(left) == bytes(right)
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12))
    return left == right


class OfflineAprilTagDetector:
    """Sequential raw-frame detector with enlarged ROI and full-frame fallback."""

    def __init__(self, config: RealRLConfig, intrinsics: dict[str, Any]) -> None:
        self.config = config
        self.tag = config.apriltag
        self.base_t_camera = load_base_t_camera(self.tag.calibration_report)
        self.camera_matrix, self.distortion = camera_calibration(intrinsics)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self._previous_corners: np.ndarray | None = None

    def _roi(self, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if self._previous_corners is None:
            return None
        height, width = shape
        points = self._previous_corners.reshape(4, 2)
        center = points.mean(axis=0)
        span = np.maximum(points.max(axis=0) - points.min(axis=0), 8.0)
        half = 0.5 * span * self.tag.offline_roi_expansion
        x0 = max(0, int(np.floor(center[0] - half[0])))
        y0 = max(0, int(np.floor(center[1] - half[1])))
        x1 = min(width, int(np.ceil(center[0] + half[0])))
        y1 = min(height, int(np.ceil(center[1] + half[1])))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        return x0, y0, x1, y1

    def _corners(
        self, gray: np.ndarray, origin: tuple[int, int]
    ) -> np.ndarray | None:
        scale = self.tag.detection_scale
        detected = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        corners, ids, _rejected = self.detector.detectMarkers(detected)
        if ids is None:
            return None
        matches = np.flatnonzero(ids.reshape(-1) == self.tag.marker_id)
        if len(matches) == 0:
            return None
        result = np.asarray(corners[int(matches[0])], dtype=np.float64).reshape(4, 2)
        if scale != 1.0:
            result /= scale
        result += np.asarray(origin, dtype=np.float64)
        return result

    def detect(
        self,
        boundary: int,
        image_bgr: np.ndarray,
        metadata: dict[str, Any],
    ) -> OfflineTagPose | None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners: np.ndarray | None = None
        mode = "full_frame"
        roi = self._roi(gray.shape)
        if roi is not None:
            x0, y0, x1, y1 = roi
            corners = self._corners(gray[y0:y1, x0:x1], (x0, y0))
            mode = "roi"
        if corners is None:
            corners = self._corners(gray, (0, 0))
            mode = "full_frame" if roi is None else "full_frame_fallback"
        if corners is None:
            return None
        camera_t_tag, error = solve_tag_pose(
            corners,
            self.camera_matrix,
            self.distortion,
            self.tag.marker_length_m,
        )
        if error > self.tag.max_reprojection_px:
            return None
        tag_t_object = np.eye(4, dtype=np.float64)
        tag_t_object[:3, 3] = np.asarray(self.tag.tag_to_object_m, dtype=np.float64)
        base_t_object = self.base_t_camera @ camera_t_tag @ tag_t_object
        position = base_t_object[:3, 3].copy()
        position[2] += self.tag.height_offset_m
        self._previous_corners = corners.copy()
        return OfflineTagPose(
            boundary=boundary,
            sequence=int(metadata["sequence"]),
            capture_timestamp=float(metadata["capture_timestamp"]),
            camera_timestamp_ms=(
                None
                if metadata.get("camera_timestamp_ms") is None
                else float(metadata["camera_timestamp_ms"])
            ),
            object_position_base_m=position,
            reprojection_rms_px=float(error),
            corners_px=corners,
            search_mode=mode,
        )


def _load_boundaries(run_dir: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    rollout_path = run_dir / "rollout.jsonl"
    if not rollout_path.is_file():
        raise FileNotFoundError(f"Missing rollout metadata: {rollout_path}")
    records = [
        json.loads(line)
        for line in rollout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    boundaries: dict[int, dict[str, Any]] = {}
    intrinsics: dict[str, Any] | None = None
    for record in records:
        step = int(record["step_index"])
        model_input = record.get("model_input", {})
        offline_boundary = model_input.get("offline_apriltag_boundary", {})
        relpath = offline_boundary.get("raw_rgb_path") or model_input.get("raw_rgb_path")
        metadata = offline_boundary.get("camera_frame") or model_input.get("camera_frame")
        if relpath and isinstance(metadata, dict):
            boundaries[step] = {"path": run_dir / relpath, "metadata": metadata}
            if isinstance(metadata.get("camera_intrinsics"), dict):
                intrinsics = dict(metadata["camera_intrinsics"])
        next_relpath = model_input.get("next_raw_rgb_path")
        next_metadata = model_input.get("next_boundary_camera_frame")
        if next_relpath and isinstance(next_metadata, dict):
            boundaries[step + 1] = {
                "path": run_dir / next_relpath,
                "metadata": next_metadata,
            }
            if isinstance(next_metadata.get("camera_intrinsics"), dict):
                intrinsics = dict(next_metadata["camera_intrinsics"])
    if not boundaries:
        raise RuntimeError(
            f"{run_dir} has no raw boundary frames; older 224x224-only runs cannot be relabeled"
        )
    if intrinsics is None:
        raise RuntimeError(f"{run_dir} raw frames do not contain RealSense intrinsics")
    return boundaries, intrinsics


def _offline_success_boundary(
    poses: dict[int, OfflineTagPose | None], initial_z: float, config: RealRLConfig
) -> int | None:
    streak = 0
    for boundary in sorted(poses):
        pose = poses[boundary]
        if (
            pose is not None
            and float(pose.object_position_base_m[2] - initial_z)
            > config.reward.success_height_m
        ):
            streak += 1
        else:
            streak = 0
        if streak >= config.reward.success_consecutive_detections:
            return boundary
    return None


def label_episode_from_run(
    run_dir: str | Path,
    config: RealRLConfig,
    *,
    reuse_saved_detections: bool = False,
) -> dict[str, Any]:
    """Detect every raw boundary frame, then atomically relabel one replay episode."""

    resolved_run = Path(run_dir).expanduser().resolve()
    if reuse_saved_detections:
        poses, label_records = _load_saved_detections(resolved_run)
    else:
        boundaries, intrinsics = _load_boundaries(resolved_run)
        detector = OfflineAprilTagDetector(config, intrinsics)
        poses = {}
        label_records = []
        for boundary in sorted(boundaries):
            item = boundaries[boundary]
            image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read raw boundary frame: {item['path']}")
            pose = detector.detect(boundary, image, item["metadata"])
            poses[boundary] = pose
            label_records.append(
                pose.to_dict()
                if pose is not None
                else {
                    "boundary": boundary,
                    "detected": False,
                    "sequence": item["metadata"].get("sequence"),
                    "capture_timestamp": item["metadata"].get("capture_timestamp"),
                }
            )

    episode_id = resolved_run.name
    with ReplayBuffer(config.replay.path, create=False) as replay:
        rows = replay.connection.execute(
            "SELECT * FROM transitions WHERE episode_id=? ORDER BY rollout_step,id",
            (episode_id,),
        ).fetchall()
        if not rows:
            raise RuntimeError(f"Replay contains no transitions for episode {episode_id}")
        initial_z = float(rows[0]["initial_object_z_in_base"])
        success_boundary = _offline_success_boundary(poses, initial_z, config)
        trainable = 0
        reasons: dict[str, int] = {}
        pending_updates: list[tuple[Any, tuple[Any, ...], bool]] = []
        for row in rows:
            step = int(row["rollout_step"])
            pose_t = poses.get(step)
            pose_next = poses.get(step + 1)
            accepted = bool(row["policy_action_accepted"])
            action_timestamp = row["action_timestamp"]
            reason = "ok"
            if success_boundary is not None and step >= success_boundary:
                reason = "after_offline_success"
            elif not accepted:
                reason = "policy_action_not_accepted"
            elif pose_t is None:
                reason = "offline_tag_missing_t"
            elif pose_next is None:
                reason = "offline_tag_missing_t1"
            elif action_timestamp is None:
                reason = "action_timestamp_missing"
            elif pose_t.capture_timestamp > float(action_timestamp):
                reason = "tag_t_after_action"
            elif float(action_timestamp) >= pose_next.capture_timestamp:
                reason = "tag_t1_not_after_action"

            valid = reason == "ok"
            is_success = bool(valid and success_boundary == step + 1)
            original_truncated = bool(row["truncated"])
            terminated = is_success
            # Preserve the collector's physical time-limit boundary even when
            # Tag labeling fails, and make repeated relabeling idempotent.
            truncated = bool(original_truncated and not terminated)
            reward = None
            position_t = None if pose_t is None else pose_t.object_position_base_m
            position_next = None if pose_next is None else pose_next.object_position_base_m
            ee_t = _unpack(row["ee_position_t"], 3)
            ee_next = _unpack(row["ee_position_next"], 3)
            assert ee_t is not None and ee_next is not None
            relative_t = None if position_t is None else position_t - ee_t
            relative_next = None if position_next is None else position_next - ee_next
            height_t = None if position_t is None else float(position_t[2] - initial_z)
            height_next = None if position_next is None else float(position_next[2] - initial_z)
            distance_t = None if relative_t is None else float(np.linalg.norm(relative_t))
            distance_next = (
                None if relative_next is None else float(np.linalg.norm(relative_next))
            )
            if valid:
                unit_action = _unpack(row["sac_unit_action"], 3)
                assert unit_action is not None
                assert distance_t is not None and distance_next is not None
                assert height_t is not None and height_next is not None
                reward = compute_progress_reward(
                    distance_t,
                    distance_next,
                    height_t,
                    height_next,
                    unit_action,
                    is_success,
                    config.reward,
                )
                trainable += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            packed_position_t = _pack(position_t, 3, "object_position_t")
            packed_position_next = _pack(position_next, 3, "object_position_next")
            packed_relative_t = _pack(relative_t, 3, "object_relative_to_ee_t")
            packed_relative_next = _pack(
                relative_next, 3, "object_relative_to_ee_next"
            )
            terminal_reason = (
                "success" if terminated else ("time_limit" if truncated else None)
            )
            values = (
                None if reward is None else reward.reward,
                None if reward is None else reward.reach_reward,
                None if reward is None else reward.lift_reward,
                None if reward is None else reward.success_reward,
                None if reward is None else reward.residual_penalty,
                int(valid), int(valid), reason,
                packed_position_t, packed_position_next,
                packed_relative_t, packed_relative_next,
                height_t, height_next, distance_t, distance_next,
                None if pose_t is None else pose_t.sequence,
                None if pose_next is None else pose_next.sequence,
                None if pose_t is None else 0.0,
                None if pose_next is None else 0.0,
                None if pose_t is None else pose_t.reprojection_rms_px,
                None if pose_next is None else pose_next.reprojection_rms_px,
                None if pose_t is None else pose_t.capture_timestamp,
                None if pose_next is None else pose_next.capture_timestamp,
                int(terminated or truncated), int(terminated), int(truncated),
                int(is_success), terminal_reason,
            )
            training_values = {
                "reward": values[0],
                "reach_reward": values[1],
                "lift_reward": values[2],
                "success_reward": values[3],
                "residual_penalty": values[4],
                "reward_valid": values[5],
                "trainable": values[6],
                "trainable_reason": values[7],
                "object_relative_to_ee_t": packed_relative_t,
                "object_relative_to_ee_next": packed_relative_next,
                "object_height_t": height_t,
                "object_height_next": height_next,
                "done": values[24],
                "terminated": values[25],
                "truncated": values[26],
                "success": values[27],
                "terminal_reason": terminal_reason,
            }
            training_changed = any(
                not _equal_training_value(row[name], value)
                for name, value in training_values.items()
            )
            pending_updates.append((row, values, training_changed))

        changed_transitions = sum(changed for _row, _values, changed in pending_updates)
        previous_revision = replay.max_label_revision()
        label_revision = previous_revision + 1 if changed_transitions else previous_revision
        for row, values, training_changed in pending_updates:
            row_revision = label_revision if training_changed else int(row["label_revision"])
            replay.connection.execute(
                """UPDATE transitions SET
                reward=?, reach_reward=?, lift_reward=?, success_reward=?, residual_penalty=?,
                reward_valid=?, trainable=?, trainable_reason=?,
                object_position_t=?, object_position_next=?,
                object_relative_to_ee_t=?, object_relative_to_ee_next=?,
                object_height_t=?, object_height_next=?, distance_t=?, distance_next=?,
                tag_sequence_t=?, tag_sequence_next=?, tag_age_t=?, tag_age_next=?,
                reprojection_error_t=?, reprojection_error_next=?,
                tag_capture_timestamp_t=?, tag_capture_timestamp_next=?,
                done=?, terminated=?, truncated=?, success=?, terminal_reason=?,
                label_revision=? WHERE id=?""",
                values + (row_revision, int(row["id"])),
            )
        replay.connection.commit()

    detected = sum(pose is not None for pose in poses.values())
    modes: dict[str, int] = {}
    for pose in poses.values():
        if pose is not None:
            modes[pose.search_mode] = modes.get(pose.search_mode, 0) + 1
    report = {
        "episode_id": episode_id,
        "run_dir": str(resolved_run),
        "replay_path": str(config.replay.path),
        "boundary_frames": len(poses),
        "detected_frames": detected,
        "detection_rate": detected / len(poses),
        "search_modes": modes,
        "transitions": len(rows),
        "trainable_transitions": trainable,
        "trainable_reasons": reasons,
        "success_boundary": success_boundary,
        "reused_saved_detections": bool(reuse_saved_detections),
        "changed_transitions": changed_transitions,
        "label_revision": label_revision,
        "reward_contract": {
            "success_height_m": config.reward.success_height_m,
            "success_consecutive_detections": (
                config.reward.success_consecutive_detections
            ),
            "k_reach": config.reward.k_reach,
            "k_lift": config.reward.k_lift,
            "k_success": config.reward.k_success,
            "k_action": config.reward.k_action,
        },
    }
    with (resolved_run / "offline_apriltag_labels.jsonl").open("w", encoding="utf-8") as handle:
        for item in label_records:
            handle.write(json.dumps(item) + "\n")
    (resolved_run / "offline_apriltag_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
