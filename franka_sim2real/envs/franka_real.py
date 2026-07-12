from __future__ import annotations

import math
import time
from typing import Any

from franky import Affine, CartesianMotion, ControlException, Gripper, RealtimeConfig, ReferenceType, Robot

from ..config import Sim2RealConfig
from ..metrics import compute_goal_metrics
from ..types import RobotAction, RobotObservation
from .base import BaseFrankaEnv


def _active_error_names(errors: Any) -> list[str]:
    names = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        value = getattr(errors, name)
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def _mode_name(mode: Any) -> str:
    return str(mode).split(".")[-1]


class RealFrankaEnv(BaseFrankaEnv):
    backend_name = "real"

    def __init__(self, config: Sim2RealConfig) -> None:
        self.config = config
        realtime_config = (
            RealtimeConfig.Ignore
            if config.backend.realtime == "ignore"
            else RealtimeConfig.Enforce
        )
        self.robot = Robot(config.backend.robot_ip, realtime_config=realtime_config)
        self.robot.relative_dynamics_factor = config.control.speed
        self.gripper = Gripper(config.backend.robot_ip) if config.backend.enable_gripper else None
        self._gripper_future = None
        self._queued_gripper_command: tuple[float, float] | None = None
        self._cached_gripper_width: float | None = None
        self._cached_gripper_max_width: float | None = None
        self._cached_gripper_is_grasped: bool | None = None

    def _cache_gripper_state(self) -> None:
        if self.gripper is None:
            return
        state = self.gripper.state
        self._cached_gripper_width = float(state.width)
        self._cached_gripper_max_width = float(state.max_width)
        self._cached_gripper_is_grasped = bool(state.is_grasped)

    def _start_async_gripper_command(
        self,
        width: float,
        speed: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        if self.gripper is None:
            return
        self._gripper_future = self.gripper.move_async(width, speed)
        # Use the active target as the best low-latency estimate until the
        # command finishes and an actual gripper state can be read.
        self._cached_gripper_width = width
        if info is not None:
            info["gripper_command_async"] = True
            info["gripper_command_success"] = None

    def _poll_gripper_command(self, info: dict[str, Any] | None = None) -> bool:
        if self._gripper_future is None:
            return True
        if not self._gripper_future.wait(0.0):
            if info is not None:
                info["gripper_command_pending"] = True
            return False

        success = bool(self._gripper_future.get())
        self._gripper_future = None
        self._cache_gripper_state()
        if info is not None:
            info["previous_gripper_command_completed"] = True
            info["previous_gripper_command_success"] = success
        if self._queued_gripper_command is not None:
            width, speed = self._queued_gripper_command
            self._queued_gripper_command = None
            self._start_async_gripper_command(width, speed, info)
            if info is not None:
                info["queued_gripper_command_started"] = True
            return False
        return True

    def reset(self, episode_index: int = 0) -> RobotObservation:
        reset_start = time.perf_counter()
        if self.config.backend.auto_recover and self.robot.has_errors:
            self.robot.recover_from_errors()

        if self.gripper is not None:
            self._cache_gripper_state()
            if (
                self.config.backend.auto_gripper_homing
                and self._cached_gripper_max_width == 0.0
            ):
                self.gripper.homing()
                self._cache_gripper_state()

        observation = self._observation()
        observation.metadata["timing"] = {
            "reset_ms": (time.perf_counter() - reset_start) * 1000.0,
        }
        return observation

    def _observation(self) -> RobotObservation:
        observe_start = time.perf_counter()
        state = self.robot.state
        pose = self.robot.current_pose.end_effector_pose
        metadata = {
            "current_errors": _active_error_names(state.current_errors),
            "last_motion_errors": _active_error_names(state.last_motion_errors),
            "timing": {},
        }
        observation = RobotObservation(
            joint_positions=state.q.tolist(),
            joint_velocities=state.dq.tolist(),
            tcp_translation=pose.translation.tolist(),
            tcp_quaternion=pose.quaternion.tolist(),
            external_wrench=state.O_F_ext_hat_K.tolist(),
            robot_mode=_mode_name(state.robot_mode),
            has_errors=self.robot.has_errors,
            is_in_control=self.robot.is_in_control,
            control_command_success_rate=float(state.control_command_success_rate),
            goal_translation=list(self.config.goal.target_translation),
            goal_yaw_deg=self.config.goal.target_yaw_deg,
            metadata=metadata,
        )
        if self.gripper is not None:
            if not self.config.backend.async_gripper_commands or self._poll_gripper_command():
                self._cache_gripper_state()
            observation.gripper_width = self._cached_gripper_width
            observation.gripper_max_width = self._cached_gripper_max_width
            observation.gripper_is_grasped = self._cached_gripper_is_grasped
        observation.metadata["timing"]["state_read_ms"] = (time.perf_counter() - observe_start) * 1000.0
        return observation

    def step(self, action: RobotAction) -> tuple[RobotObservation, float, bool, dict[str, Any]]:
        info: dict[str, Any] = {}
        step_start = time.perf_counter()
        arm_move_ms = 0.0
        gripper_move_ms = 0.0
        settle_ms = 0.0

        if self.config.backend.async_gripper_commands:
            self._poll_gripper_command(info)

        if action.is_arm_command():
            yaw_rad = math.radians(action.yaw_deg)
            motion = CartesianMotion(
                Affine(
                    [action.dx, action.dy, action.dz],
                    [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)],
                ),
                ReferenceType.Relative,
                action.speed if action.speed is not None else self.config.control.speed,
            )
            try:
                arm_start = time.perf_counter()
                self.robot.move(motion)
                arm_move_ms = (time.perf_counter() - arm_start) * 1000.0
            except ControlException as exc:
                observation = self._observation()
                metrics = compute_goal_metrics(observation, self.config.goal)
                info.update(metrics)
                info["arm_command_failed"] = True
                info["exception"] = str(exc)
                info["timing"] = {
                    "arm_move_ms": arm_move_ms,
                    "gripper_move_ms": gripper_move_ms,
                    "settle_ms": settle_ms,
                    "step_total_ms": (time.perf_counter() - step_start) * 1000.0,
                }
                return observation, float(metrics["reward"]), True, info

        if action.gripper_width is not None and self.gripper is not None:
            if self.config.backend.async_gripper_commands:
                self._poll_gripper_command(info)
            if self._cached_gripper_max_width is None:
                self._cache_gripper_state()
            if self._cached_gripper_max_width == 0.0 and self.config.backend.auto_gripper_homing:
                self.gripper.homing()
                self._cache_gripper_state()
            info["gripper_target_width"] = action.gripper_width
            current_gripper_width = float(self._cached_gripper_width or 0.0)
            info["gripper_width_before_command"] = current_gripper_width
            gripper_delta = abs(action.gripper_width - current_gripper_width)
            info["gripper_command_delta_m"] = gripper_delta
            if gripper_delta <= self.config.control.gripper_command_tolerance_m:
                info["gripper_command_skipped"] = True
                info["gripper_command_success"] = True
            else:
                gripper_start = time.perf_counter()
                gripper_speed = (
                    action.gripper_speed
                    if action.gripper_speed is not None
                    else self.config.control.gripper_speed
                )
                if self.config.backend.async_gripper_commands:
                    if self._gripper_future is None:
                        self._start_async_gripper_command(
                            action.gripper_width,
                            gripper_speed,
                            info,
                        )
                    else:
                        self._queued_gripper_command = (
                            action.gripper_width,
                            gripper_speed,
                        )
                        info["gripper_command_queued"] = True
                        info["queued_gripper_target_width"] = action.gripper_width
                        info["gripper_command_success"] = None
                else:
                    info["gripper_command_success"] = self.gripper.move(
                        action.gripper_width,
                        gripper_speed,
                    )
                    self._cache_gripper_state()
                gripper_move_ms = (time.perf_counter() - gripper_start) * 1000.0

        settle_start = time.perf_counter()
        time.sleep(self.config.backend.settle_time_s)
        settle_ms = (time.perf_counter() - settle_start) * 1000.0
        observation = self._observation()
        metrics = compute_goal_metrics(observation, self.config.goal)
        info.update(metrics)
        info["timing"] = {
            "arm_move_ms": arm_move_ms,
            "gripper_move_ms": gripper_move_ms,
            "settle_ms": settle_ms,
            "state_read_ms": observation.metadata.get("timing", {}).get("state_read_ms", 0.0),
            "step_total_ms": (time.perf_counter() - step_start) * 1000.0,
        }
        return observation, float(metrics["reward"]), bool(metrics["success"]), info

    def close(self) -> None:
        if self.gripper is None:
            return
        deadline = time.monotonic() + 2.0
        while self._gripper_future is not None or self._queued_gripper_command is not None:
            if self._gripper_future is None and self._queued_gripper_command is not None:
                width, speed = self._queued_gripper_command
                self._queued_gripper_command = None
                self._start_async_gripper_command(width, speed)

            remaining = max(0.0, deadline - time.monotonic())
            if remaining == 0.0 or not self._gripper_future.wait(remaining):
                self.gripper.stop()
                break
            self._gripper_future.get()
            self._gripper_future = None
            self._cache_gripper_state()
        self._gripper_future = None
        self._queued_gripper_command = None
