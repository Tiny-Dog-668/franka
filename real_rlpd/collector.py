from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from franka_sim2real.real_rl.apriltag_tracker import AprilTagTracker

from .action import expert_residual_target
from .expert_dataset import ExpertEpisodeDataset
from .replay import ReplayBuffer, ReplayRole, Transition
from .runtime import RLPDDeploySettings


@dataclass
class _Boundary:
    step: int
    state: np.ndarray
    ee_position: np.ndarray


@dataclass
class _Pending:
    boundary: _Boundary
    action: np.ndarray
    base_limited_action: np.ndarray
    executed_action: np.ndarray
    accepted: bool
    action_timestamp: float | None
    checkpoint_sha256: str | None


class RLPDCollector:
    """把 streaming boundary/action 配对写入独立 offline/online Replay。"""

    def __init__(
        self,
        settings: RLPDDeploySettings,
        camera: Any,
        run_dir: Path,
        role: ReplayRole,
        contract: dict[str, Any],
    ) -> None:
        if settings.mode == "expert_shadow" and role != "offline":
            raise ValueError("Expert shadow collection must write the offline replay")
        if settings.mode == "checkpoint" and role != "online":
            raise ValueError("Checkpoint collection must write the online replay")
        self.settings = settings
        self.role = role
        self.run_dir = Path(run_dir).resolve()
        self.episode_id = self.run_dir.name
        path = (
            settings.config.offline_replay_path
            if role == "offline" else settings.config.online_replay_path
        )
        self.replay = ReplayBuffer(path, role)
        try:
            self.replay.assert_contract(contract)
        except BaseException:
            self.replay.close()
            raise
        try:
            self.tracker = AprilTagTracker(camera, settings.config.apriltag)
            self._tracker_closed = False
            try:
                self.initial_object_z = self.tracker.wait_for_initial_object_z_in_base()
            finally:
                # AprilTag is a stationary preflight only. Reward frames are
                # labeled after control, so no detector thread runs at 30 Hz.
                self.tracker.close()
                self._tracker_closed = True
        except BaseException:
            self.replay.close()
            raise
        self.expert_dataset = (
            ExpertEpisodeDataset(
                self.run_dir,
                action_scales=contract["action_scales"],
                initial_object_z=self.initial_object_z,
                collection_provenance={
                    "shadow_policy_kind": contract["policy_kind"],
                    "shadow_model_sha256": contract["model_sha256"],
                    "shadow_metadata_sha256": contract["metadata_sha256"],
                    "note": "provenance only; not part of the reusable expert contract",
                },
            )
            if role == "offline"
            else None
        )
        self._current: _Boundary | None = None
        self._pending: _Pending | None = None
        self._transitions: list[Transition] = []
        self.written = 0
        self._closed = False

    @staticmethod
    def _boundary(step: int, observation: Any, result: Any) -> _Boundary:
        info = result.inference_info.get("rlpd")
        if not isinstance(info, dict):
            raise RuntimeError("RLPD collector did not receive policy state")
        state = np.asarray(info["state"], dtype=np.float32).reshape(-1)
        physical_tcp = result.observation_metadata.get(
            "physical_tool_tcp_translation_m",
            observation.metadata.get("physical_tool_tcp_translation_m"),
        )
        if physical_tcp is None:
            raise RuntimeError("Robot observation lacks physical_tool_tcp_translation_m")
        return _Boundary(
            step=int(step),
            state=state,
            ee_position=np.asarray(physical_tcp, dtype=np.float32).reshape(3),
        )

    def observe_boundary(
        self,
        step: int,
        observation: Any,
        result: Any,
        *,
        truncate: bool = False,
    ) -> bool:
        point = self._boundary(step, observation, result)
        if self.expert_dataset is not None:
            self.expert_dataset.observe_boundary(
                step, observation, result, truncate=truncate
            )
        if self._pending is not None:
            pending = self._pending
            accepted = pending.accepted
            self._transitions.append(Transition(
                episode_id=self.episode_id,
                step_id=pending.boundary.step,
                state=pending.boundary.state,
                next_state=point.state,
                action=pending.action,
                base_limited_action=pending.base_limited_action,
                executed_action=pending.executed_action,
                ee_position=pending.boundary.ee_position,
                next_ee_position=point.ee_position,
                initial_object_z=self.initial_object_z,
                privileged=None,
                next_privileged=None,
                reward=None,
                terminated=False,
                truncated=bool(truncate),
                success=False,
                accepted=accepted,
                trainable=False,
                trainable_reason=("offline_apriltag_pending" if accepted else "action_not_accepted"),
                action_timestamp=pending.action_timestamp,
                run_dir=str(self.run_dir),
                checkpoint_sha256=pending.checkpoint_sha256,
            ))
            self._pending = None
        self._current = point
        return False

    def record_action(
        self,
        result: Any,
        accepted: bool,
        *,
        action_timestamp: float | None = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("observe_boundary must precede record_action")
        info = result.inference_info["rlpd"]
        base_limited = np.asarray(
            info["base_limited_action"], dtype=np.float32
        ).reshape(4)
        executed = np.asarray(result.executed_action, dtype=np.float32).reshape(4)
        action = (
            expert_residual_target(
                base_limited, executed, self.settings.commissioning_limit
            )
            if self.role == "offline"
            else np.asarray(info["unit_residual_action"], dtype=np.float32).reshape(4)
        )
        self._pending = _Pending(
            boundary=self._current,
            action=action,
            base_limited_action=base_limited,
            executed_action=executed,
            accepted=bool(accepted),
            action_timestamp=float(action_timestamp) if accepted and action_timestamp is not None else None,
            checkpoint_sha256=info.get("checkpoint_sha256"),
        )
        if self.expert_dataset is not None:
            self.expert_dataset.record_action(
                result, accepted, action_timestamp=action_timestamp
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[str] = []
        expert_report: dict[str, Any] | None = None
        if self.expert_dataset is not None:
            try:
                expert_report = self.expert_dataset.close()
            except Exception as exc:
                errors.append(f"Expert source dataset: {exc}")
        try:
            self.replay.append_many(self._transitions)
            self.written += len(self._transitions)
            self._transitions.clear()
        except Exception as exc:
            errors.append(f"Replay write: {exc}")
        try:
            self.replay.close()
        except Exception as exc:
            errors.append(f"Replay close: {exc}")
        if not self._tracker_closed:
            try:
                self.tracker.close()
            except Exception as exc:
                errors.append(f"AprilTag: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
        self._expert_report = expert_report

    def report(self) -> dict[str, Any]:
        report = {
            "episode_id": self.episode_id,
            "role": self.role,
            "transitions_written": self.written,
            "initial_object_z": self.initial_object_z,
        }
        expert_report = getattr(self, "_expert_report", None)
        if expert_report is not None:
            report["expert_source"] = expert_report
        return report
