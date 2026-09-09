from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RewardConfig


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reach_reward: float
    lift_reward: float
    success_reward: float
    residual_penalty: float


def compute_progress_reward(
    distance_t: float,
    distance_t1: float,
    height_t: float,
    height_t1: float,
    unit_action: np.ndarray,
    success: bool,
    config: RewardConfig,
) -> RewardResult:
    config.validate()
    action = np.asarray(unit_action, dtype=np.float64).reshape(-1)
    if action.shape != (3,) or not np.all(np.isfinite(action)):
        raise ValueError("unit_action must contain three finite values")
    values = np.asarray([distance_t, distance_t1, height_t, height_t1], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Reward distances and heights must be finite")
    reach = config.k_reach * (float(distance_t) - float(distance_t1))
    lift = config.k_lift * (float(height_t1) - float(height_t))
    success_reward = config.k_success if success else 0.0
    penalty = config.k_action * float(np.dot(action, action))
    return RewardResult(
        reward=float(reach + lift + success_reward - penalty),
        reach_reward=float(reach),
        lift_reward=float(lift),
        success_reward=float(success_reward),
        residual_penalty=float(penalty),
    )
