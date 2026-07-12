from __future__ import annotations

import math

import numpy as np

from ..config import Sim2RealConfig
from ..metrics import compute_goal_metrics
from ..types import RobotAction, RobotObservation
from .base import BaseFrankaEnv


class MockFrankaSimEnv(BaseFrankaEnv):
    backend_name = "sim"

    def __init__(self, config: Sim2RealConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.sim.seed)
        self.translation = np.array(config.sim.initial_translation, dtype=float)
        self.yaw_deg = float(config.sim.initial_yaw_deg)
        self.gripper_width = float(config.sim.gripper_max_width)

    def reset(self, episode_index: int = 0) -> RobotObservation:
        self.translation = np.array(self.config.sim.initial_translation, dtype=float)
        self.translation += self.rng.normal(
            0.0,
            self.config.sim.observation_noise_std,
            size=3,
        )
        self.yaw_deg = float(self.config.sim.initial_yaw_deg)
        self.gripper_width = float(self.config.sim.gripper_max_width)
        return self._observation()

    def _quaternion_from_yaw_deg(self, yaw_deg: float) -> list[float]:
        yaw_rad = math.radians(yaw_deg)
        return [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]

    def _observation(self) -> RobotObservation:
        return RobotObservation(
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            tcp_translation=self.translation.tolist(),
            tcp_quaternion=self._quaternion_from_yaw_deg(self.yaw_deg),
            external_wrench=[0.0] * 6,
            robot_mode="Simulated",
            has_errors=False,
            is_in_control=True,
            control_command_success_rate=1.0,
            gripper_width=self.gripper_width,
            gripper_max_width=self.config.sim.gripper_max_width,
            gripper_is_grasped=False,
            goal_translation=list(self.config.goal.target_translation),
            goal_yaw_deg=self.config.goal.target_yaw_deg,
        )

    def step(self, action: RobotAction) -> tuple[RobotObservation, float, bool, dict[str, float | bool]]:
        delta = np.array([action.dx, action.dy, action.dz], dtype=float)
        delta += self.rng.normal(0.0, self.config.sim.action_noise_std, size=3)
        self.translation += delta
        self.translation = np.array(
            self.config.control.workspace.clamp_translation(self.translation.tolist()),
            dtype=float,
        )

        yaw_noise = self.rng.normal(0.0, self.config.sim.yaw_noise_deg)
        self.yaw_deg += action.yaw_deg + yaw_noise

        if action.gripper_width is not None:
            self.gripper_width = float(
                np.clip(action.gripper_width, 0.0, self.config.sim.gripper_max_width)
            )

        observation = self._observation()
        metrics = compute_goal_metrics(observation, self.config.goal)
        return observation, float(metrics["reward"]), bool(metrics["success"]), metrics
