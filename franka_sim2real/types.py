from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import atan2, degrees, sqrt
import sys
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


def _fmt_vector(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):+.{precision}f}" for value in values) + "]"


def external_wrench_norms(observation: RobotObservation) -> tuple[float, float]:
    wrench = [float(value) for value in observation.external_wrench]
    if len(wrench) != 6:
        return float("nan"), float("nan")
    force_norm = sqrt(sum(value * value for value in wrench[:3]))
    torque_norm = sqrt(sum(value * value for value in wrench[3:]))
    return force_norm, torque_norm


def format_error_wrench_report(
    observation: RobotObservation,
    *,
    context: str = "Franka error",
) -> str:
    """Format the external wrench observed when the robot reports an error."""
    wrench = [float(value) for value in observation.external_wrench]
    force = wrench[:3] if len(wrench) >= 3 else []
    torque = wrench[3:6] if len(wrench) >= 6 else []
    force_norm, torque_norm = external_wrench_norms(observation)
    current_errors = observation.metadata.get("current_errors")
    last_motion_errors = observation.metadata.get("last_motion_errors")
    worker_error = observation.metadata.get("worker_error_message")
    worker_error_code = observation.metadata.get("worker_error_code")

    lines = [
        "=" * 72,
        f"{context}: robot state at event",
        f"Robot mode: {observation.robot_mode}",
        f"Has errors: {observation.has_errors}",
        f"Is in control: {observation.is_in_control}",
        f"Control command success rate: {observation.control_command_success_rate:.3f}",
        f"TCP translation [m]: {_fmt_vector(observation.tcp_translation, precision=4)}",
        f"External force  [N] : {_fmt_vector(force)} |F|={force_norm:.3f} N",
        f"External torque [Nm]: {_fmt_vector(torque)} |M|={torque_norm:.3f} Nm",
    ]
    if current_errors is not None:
        lines.append(f"Current errors: {current_errors if current_errors else 'none'}")
    if last_motion_errors is not None:
        lines.append(f"Last motion errors: {last_motion_errors if last_motion_errors else 'none'}")
    worker_status = observation.metadata.get("worker_status")
    if worker_status:
        lines.append(f"Worker status: {worker_status}")
    for key, label in (
        ("latched_action_generation", "Latched action generation"),
        ("ik_tick_count", "IK tick count"),
        ("control_cycle_count", "Control cycle count"),
    ):
        value = observation.metadata.get(key)
        if value is not None:
            lines.append(f"{label}: {value}")
    if worker_error_code is not None:
        lines.append(f"Worker error code: {worker_error_code}")
    if worker_error:
        lines.append(f"Worker error message: {worker_error}")
    lines.append("=" * 72)
    return "\n".join(lines)


def print_error_wrench_report(
    observation: RobotObservation,
    *,
    context: str = "Franka error",
) -> None:
    print(format_error_wrench_report(observation, context=context), file=sys.stderr, flush=True)
