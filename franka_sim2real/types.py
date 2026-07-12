from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import atan2, degrees
from typing import Any


@dataclass
class WorkspaceLimits:
    minimum: list[float] = field(default_factory=lambda: [0.2, -0.3, 0.05])
    maximum: list[float] = field(default_factory=lambda: [0.65, 0.3, 0.45])

    def clamp_translation(self, translation: list[float]) -> list[float]:
        return [
            min(max(value, self.minimum[idx]), self.maximum[idx])
            for idx, value in enumerate(translation)
        ]


@dataclass
class RobotAction:
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    yaw_deg: float = 0.0
    speed: float | None = None
    gripper_width: float | None = None
    gripper_speed: float | None = None
    gripper_force: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_arm_command(self) -> bool:
        return any(value != 0.0 for value in (self.dx, self.dy, self.dz, self.yaw_deg))

    def is_gripper_command(self) -> bool:
        return self.gripper_width is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotObservation:
    joint_positions: list[float]
    joint_velocities: list[float]
    tcp_translation: list[float]
    tcp_quaternion: list[float]
    external_wrench: list[float]
    robot_mode: str
    has_errors: bool
    is_in_control: bool
    control_command_success_rate: float
    gripper_width: float | None = None
    gripper_max_width: float | None = None
    gripper_is_grasped: bool | None = None
    goal_translation: list[float] | None = None
    goal_yaw_deg: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tcp_yaw_deg(self) -> float:
        x, y, z, w = self.tcp_quaternion
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return degrees(atan2(siny_cosp, cosy_cosp))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
