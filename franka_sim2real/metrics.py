from __future__ import annotations

import math

import numpy as np

from .config import GoalConfig
from .types import RobotObservation


def shortest_angle_deg(delta_deg: float) -> float:
    wrapped = (delta_deg + 180.0) % 360.0 - 180.0
    return wrapped


def compute_goal_metrics(observation: RobotObservation, goal: GoalConfig) -> dict[str, float | bool]:
    position_error = float(
        np.linalg.norm(
            np.array(observation.tcp_translation, dtype=float)
            - np.array(goal.target_translation, dtype=float)
        )
    )
    yaw_error_deg = abs(shortest_angle_deg(observation.tcp_yaw_deg - goal.target_yaw_deg))
    success = (
        position_error <= goal.translation_tolerance_m
        and yaw_error_deg <= goal.yaw_tolerance_deg
    )
    reward = -position_error - math.radians(yaw_error_deg) * 0.1
    return {
        "position_error_m": position_error,
        "yaw_error_deg": yaw_error_deg,
        "success": success,
        "reward": reward,
    }
