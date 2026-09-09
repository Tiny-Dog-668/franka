from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .apriltag_tracker import AprilTagTracker, TagPose
from .config import RealRLConfig
from .replay_buffer import ReplayTransition, ReplayWriter
from .runtime import RealRLDeploySettings


@dataclass(frozen=True)
class _BoundaryPoint:
    step: int
    state: np.ndarray
    base_action: np.ndarray
    ee_position: np.ndarray
    tag_pose: TagPose | None
    tag_age_s: float | None
    object_position: np.ndarray | None
    relative: np.ndarray | None
    object_height_in_base: float | None
    distance: float | None


@dataclass(frozen=True)
class _PendingAction:
    point: _BoundaryPoint
    unit_action: np.ndarray
    residual_normalized: np.ndarray
    residual_m: np.ndarray
    executed_action: np.ndarray | None
    accepted: bool
    action_timestamp: float | None
    pre_safety_action: np.ndarray
    post_safety_action: np.ndarray
    safety_intervened: bool
    intervention_magnitude: float
    checkpoint_sha256: str | None


class RealRLCollector:
    """Capture transition skeletons; rewards are labeled after control stops."""

    def __init__(
        self,
        settings: RealRLDeploySettings,
        camera: Any,
        run_dir: Path,
        *,
        tracker_factory: Callable[[Any, Any], AprilTagTracker] = AprilTagTracker,
        writer_factory: Callable[..., ReplayWriter] = ReplayWriter,
    ) -> None:
        self.settings = settings
        self.config: RealRLConfig = settings.config
        self.run_dir = run_dir.resolve()
        self.episode_id = run_dir.name
        self.tracker = tracker_factory(camera, self.config.apriltag)
        self._tracker_closed = False
        try:
            self.initial_object_z_in_base = self.tracker.wait_for_initial_object_z_in_base()
        finally:
            # Detection is intentionally absent from the real-time episode.
            # Preflight establishes the stationary height baseline; raw
            # boundary frames are labeled after the robot worker stops.
            self.tracker.close()
            self._tracker_closed = True
        self.writer = writer_factory(
            self.config.replay.path, queue_size=self.config.replay.writer_queue_size
        )
        self._pending: _PendingAction | None = None
        self._current: _BoundaryPoint | None = None
        self._success_streak = 0
        self._last_success_sequence: int | None = None
        self.transitions_submitted = 0
        self.success = False

    def _point(self, step: int, observation: Any, result: Any) -> _BoundaryPoint:
        info = result.inference_info.get("real_rl")
        if not isinstance(info, dict):
            raise RuntimeError("Real-RL policy result is missing runtime state")
        state = np.asarray(info["state"], dtype=np.float32)
        base = np.asarray(info["base_action"], dtype=np.float32)
        physical_tcp = result.observation_metadata.get(
            "physical_tool_tcp_translation_m",
            observation.metadata.get("physical_tool_tcp_translation_m"),
        )
        if physical_tcp is None:
            raise RuntimeError("Robot observation lacks physical_tool_tcp_translation_m")
        ee = np.asarray(physical_tcp, dtype=np.float32).reshape(3)
        pose = None
        object_position: np.ndarray | None = None
        relative: np.ndarray | None = None
        height: float | None = None
        distance: float | None = None
        age: float | None = None
        if pose is not None:
            object_position = np.asarray(pose.object_position_base_m, dtype=np.float32).reshape(3)
            relative = object_position - ee
            # 高度始终在已标定的 Franka base 坐标系中定义。
            height = float(object_position[2] - self.initial_object_z_in_base)
            distance = float(np.linalg.norm(relative))
            age = float(pose.age_s())
        return _BoundaryPoint(
            step=step,
            state=state,
            base_action=base,
            ee_position=ee,
            tag_pose=pose,
            tag_age_s=age,
            object_position=object_position,
            relative=relative,
            object_height_in_base=height,
            distance=distance,
        )

    def _update_success(self, point: _BoundaryPoint) -> bool:
        pose = point.tag_pose
        if pose is None or point.object_height_in_base is None:
            self._success_streak = 0
            return False
        if pose.sequence == self._last_success_sequence:
            return self._success_streak >= self.config.reward.success_consecutive_detections
        self._last_success_sequence = pose.sequence
        if point.object_height_in_base > self.config.reward.success_height_m:
            self._success_streak += 1
        else:
            self._success_streak = 0
        return self._success_streak >= self.config.reward.success_consecutive_detections

    def observe_boundary(
        self,
        step: int,
        observation: Any,
        result: Any,
        *,
        truncate: bool = False,
    ) -> bool:
        self.writer.check()
        point = self._point(step, observation, result)
        success = False
        self._current = point
        if self._pending is None:
            return False
        pending = self._pending
        previous = pending.point
        trainable, trainable_reason = self._trainability(pending, point)
        reward_valid = False
        reward = None
        terminated = bool(success)
        truncated = bool(truncate and not terminated)
        transition = ReplayTransition(
            episode_id=self.episode_id,
            step_id=previous.step,
            run_dir=str(self.run_dir),
            rollout_step=previous.step,
            state=previous.state,
            next_state=point.state,
            base_action=previous.base_action,
            next_base_action=point.base_action,
            sac_unit_action=pending.unit_action,
            residual_action_normalized=pending.residual_normalized,
            residual_action_m=pending.residual_m,
            executed_action=pending.executed_action,
            policy_action_accepted=pending.accepted,
            action_timestamp=pending.action_timestamp,
            pre_safety_action=pending.pre_safety_action,
            post_safety_action=pending.post_safety_action,
            safety_intervened=pending.safety_intervened,
            intervention_magnitude=pending.intervention_magnitude,
            reward=None if reward is None else reward.reward,
            reach_reward=None if reward is None else reward.reach_reward,
            lift_reward=None if reward is None else reward.lift_reward,
            success_reward=None if reward is None else reward.success_reward,
            residual_penalty=None if reward is None else reward.residual_penalty,
            reward_valid=reward_valid,
            trainable=trainable,
            trainable_reason=trainable_reason,
            object_position_t=previous.object_position,
            object_position_next=point.object_position,
            ee_position_t=previous.ee_position,
            ee_position_next=point.ee_position,
            object_relative_to_ee_t=previous.relative,
            object_relative_to_ee_next=point.relative,
            initial_object_z_in_base=self.initial_object_z_in_base,
            object_height_t=previous.object_height_in_base,
            object_height_next=point.object_height_in_base,
            distance_t=previous.distance,
            distance_next=point.distance,
            tag_sequence_t=None if previous.tag_pose is None else previous.tag_pose.sequence,
            tag_sequence_next=None if point.tag_pose is None else point.tag_pose.sequence,
            tag_age_t=previous.tag_age_s,
            tag_age_next=point.tag_age_s,
            reprojection_error_t=(
                None if previous.tag_pose is None else previous.tag_pose.reprojection_rms_px
            ),
            reprojection_error_next=(
                None if point.tag_pose is None else point.tag_pose.reprojection_rms_px
            ),
            tag_capture_timestamp_t=(
                None if previous.tag_pose is None else previous.tag_pose.capture_timestamp
            ),
            tag_capture_timestamp_next=(
                None if point.tag_pose is None else point.tag_pose.capture_timestamp
            ),
            done=terminated or truncated,
            terminated=terminated,
            truncated=truncated,
            success=terminated,
            terminal_reason="success" if terminated else ("time_limit" if truncated else None),
            base_model_sha256=str(result.inference_info["real_rl"]["base_model_sha256"]),
            residual_checkpoint_sha256=pending.checkpoint_sha256,
        )
        self.writer.submit(transition)
        self.transitions_submitted += 1
        self._pending = None
        self.success = self.success or success
        return success

    @staticmethod
    def _trainability(
        pending: _PendingAction, point: _BoundaryPoint
    ) -> tuple[bool, str]:
        if not pending.accepted:
            return False, "policy_action_not_accepted"
        return False, "offline_apriltag_pending"

    def record_action(
        self,
        result: Any,
        accepted: bool,
        *,
        action_timestamp: float | None = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("observe_boundary must be called before record_action")
        info = result.inference_info["real_rl"]
        pre_safety = np.asarray(result.raw_action, dtype=np.float32).reshape(4)
        post_safety = np.asarray(result.executed_action, dtype=np.float32).reshape(4)
        intervention_magnitude = float(np.linalg.norm(post_safety - pre_safety))
        self._pending = _PendingAction(
            point=self._current,
            unit_action=np.asarray(info["sac_unit_action"], dtype=np.float32),
            residual_normalized=np.asarray(info["residual_action_normalized"], dtype=np.float32),
            residual_m=np.asarray(info["residual_action_m"], dtype=np.float32),
            executed_action=(
                np.asarray(result.executed_action, dtype=np.float32) if accepted else None
            ),
            accepted=bool(accepted),
            action_timestamp=(float(action_timestamp) if accepted else None),
            pre_safety_action=pre_safety,
            post_safety_action=post_safety,
            safety_intervened=intervention_magnitude > 1e-6,
            intervention_magnitude=intervention_magnitude,
            checkpoint_sha256=info.get("checkpoint_sha256"),
        )

    def close(self) -> None:
        errors: list[str] = []
        try:
            self.writer.close()
        except Exception as exc:
            errors.append(f"Replay writer: {exc}")
        if not self._tracker_closed:
            try:
                self.tracker.close()
                self._tracker_closed = True
            except Exception as exc:
                errors.append(f"AprilTag tracker: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def report(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "initial_object_z_in_base": self.initial_object_z_in_base,
            "transitions_submitted": self.transitions_submitted,
            "transitions_written": self.writer.written,
            "success": self.success,
            "labeling_mode": "offline_per_frame",
            "replay_path": str(self.config.replay.path),
        }
