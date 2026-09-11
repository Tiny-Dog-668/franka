from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from franka_sim2real.real_rl.config import RewardConfig


APRILTAG_PROGRESS_REWARD = "apriltag_reach_lift_success"


@dataclass(frozen=True)
class RewardInput:
    object_relative_to_ee: np.ndarray
    next_object_relative_to_ee: np.ndarray
    object_height: float
    next_object_height: float
    action: np.ndarray
    success: bool


@dataclass(frozen=True)
class RewardOutput:
    total: float
    reach: float
    lift: float
    success: float
    action_penalty: float


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


def build_reward(kind: str, config: RewardConfig) -> RewardFunction:
    if kind == APRILTAG_PROGRESS_REWARD:
        return AprilTagProgressReward(config)
    raise ValueError(f"Unsupported RLPD reward kind: {kind!r}")
