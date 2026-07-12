from __future__ import annotations

import math


def _yaw_deg_from_quaternion(quaternion):
    x, y, z, w = quaternion
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


def predict(observation_dict, step_index, episode_index, config_dict):
    goal = observation_dict.get("goal_translation") or config_dict["goal"]["target_translation"]
    tcp = observation_dict["tcp_translation"]
    yaw_goal = config_dict["goal"]["target_yaw_deg"]
    yaw_now = _yaw_deg_from_quaternion(observation_dict["tcp_quaternion"])

    return {
        "dx": 0.5 * (goal[0] - tcp[0]),
        "dy": 0.5 * (goal[1] - tcp[1]),
        "dz": 0.5 * (goal[2] - tcp[2]),
        "yaw_deg": 0.5 * (yaw_goal - yaw_now),
    }
