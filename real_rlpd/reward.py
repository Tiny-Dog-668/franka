from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from franka_sim2real.real_rl.config import (
    RewardConfig,
    X040ObservableAbsoluteRewardConfig,
)


APRILTAG_PROGRESS_REWARD = "apriltag_reach_lift_success"
APRILTAG_X040_OBSERVABLE_ABSOLUTE_REWARD = "apriltag_x040_observable_absolute_v1"


@dataclass(frozen=True)
class RewardInput:
    object_relative_to_ee: np.ndarray
    next_object_relative_to_ee: np.ndarray
    object_height: float
    next_object_height: float
    action: np.ndarray
    success: bool
    executed_action: np.ndarray | None = None
    previous_executed_action: np.ndarray | None = None
    next_tool_tcp_position: np.ndarray | None = None
    dropped: bool = False


@dataclass(frozen=True)
class RewardOutput:
    total: float
    reach: float
    lift: float
    success: float
    action_penalty: float
    action_magnitude_penalty: float = 0.0
    action_rate_penalty: float = 0.0
    table_clearance_penalty: float = 0.0
    drop_penalty: float = 0.0


class RewardFunction(Protocol):
    def compute(self, value: RewardInput) -> RewardOutput:
        ...


class AprilTagProgressReward:
    """当前抓取任务 reward；AprilTag 仅提供离线 privileged 几何。"""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(self, value: RewardInput) -> RewardOutput:
        reach = self.config.k_reach * (
            float(np.linalg.norm(value.object_relative_to_ee))
            - float(np.linalg.norm(value.next_object_relative_to_ee))
        )
        lift = self.config.k_lift * (
            float(value.next_object_height) - float(value.object_height)
        )
        success = self.config.k_success if value.success else 0.0
        action_penalty = self.config.k_action * float(
            np.dot(value.action, value.action)
        )
        return RewardOutput(
            total=float(reach + lift + success - action_penalty),
            reach=float(reach),
            lift=float(lift),
            success=float(success),
            action_penalty=float(action_penalty),
        )


class AprilTagX040ObservableAbsoluteReward:
    """0912 仿真 reward 的真机可观测子集，不使用模型辅助 contact head。"""

    def __init__(self, config: X040ObservableAbsoluteRewardConfig) -> None:
        self.config = config

    @staticmethod
    def _vector(value: np.ndarray | None, size: int, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} is required by the X040 observable reward")
        result = np.asarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite {size}D vector")
        return result

    def compute(self, value: RewardInput) -> RewardOutput:
        relative_to_tool = self._vector(
            value.next_object_relative_to_ee, 3, "next_object_relative_to_ee"
        )
        grasp_offset = np.asarray(
            self.config.grasp_center_offset_from_tool_tcp_m, dtype=np.float64
        )
        reach_distance = float(np.linalg.norm(relative_to_tool - grasp_offset))
        reach = self.config.reach_weight * (
            1.0 - float(np.tanh(reach_distance / self.config.reach_sigma_m))
        )
        lift_progress = float(np.clip(
            value.next_object_height / self.config.success_height_m,
            0.0,
            1.0,
        ))
        lift = self.config.lift_weight * lift_progress
        success = self.config.success_weight if value.success else 0.0

        executed = self._vector(value.executed_action, 4, "executed_action")
        previous = self._vector(
            value.previous_executed_action, 4, "previous_executed_action"
        )
        magnitude_penalty = self.config.action_magnitude_weight * float(
            np.mean(np.square(executed))
        )
        rate_penalty = self.config.action_rate_weight * float(
            np.mean(np.square(executed - previous))
        )

        tool_tcp = self._vector(
            value.next_tool_tcp_position, 3, "next_tool_tcp_position"
        )
        clearance = float(tool_tcp[2] - self.config.table_height_base_m)
        table_penalty = (
            self.config.table_clearance_penalty
            if clearance < self.config.table_clearance_min_m
            else 0.0
        )
        drop_penalty = self.config.drop_penalty if value.dropped else 0.0
        action_penalty = magnitude_penalty + rate_penalty
        return RewardOutput(
            total=float(
                reach
                + lift
                + success
                - action_penalty
                + table_penalty
                + drop_penalty
            ),
            reach=float(reach),
            lift=float(lift),
            success=float(success),
            action_penalty=float(action_penalty),
            action_magnitude_penalty=float(magnitude_penalty),
            action_rate_penalty=float(rate_penalty),
            table_clearance_penalty=float(table_penalty),
            drop_penalty=float(drop_penalty),
        )


def build_reward(
    kind: str,
    config: RewardConfig | X040ObservableAbsoluteRewardConfig,
) -> RewardFunction:
    if kind == APRILTAG_PROGRESS_REWARD:
        if not isinstance(config, RewardConfig):
            raise TypeError("Legacy AprilTag progress reward requires RewardConfig")
        return AprilTagProgressReward(config)
    if kind == APRILTAG_X040_OBSERVABLE_ABSOLUTE_REWARD:
        if not isinstance(config, X040ObservableAbsoluteRewardConfig):
            raise TypeError(
                "X040 observable absolute reward requires "
                "X040ObservableAbsoluteRewardConfig"
            )
        return AprilTagX040ObservableAbsoluteReward(config)
    raise ValueError(f"Unsupported RLPD reward kind: {kind!r}")
