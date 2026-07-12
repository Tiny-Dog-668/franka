from __future__ import annotations

from .config import ControlConfig
from .types import RobotAction, RobotObservation


def clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def apply_safety_limits(
    action: RobotAction,
    control: ControlConfig,
    observation: RobotObservation | None = None,
) -> RobotAction:
    dx = clamp(action.dx, -control.max_dx, control.max_dx)
    dy = clamp(action.dy, -control.max_dy, control.max_dy)
    dz = clamp(action.dz, -control.max_dz, control.max_dz)
    yaw_deg = clamp(action.yaw_deg, -control.max_yaw_deg, control.max_yaw_deg)

    if observation is not None:
        desired_translation = [
            observation.tcp_translation[0] + dx,
            observation.tcp_translation[1] + dy,
            observation.tcp_translation[2] + dz,
        ]
        clamped_translation = control.workspace.clamp_translation(desired_translation)
        dx = clamped_translation[0] - observation.tcp_translation[0]
        dy = clamped_translation[1] - observation.tcp_translation[1]
        dz = clamped_translation[2] - observation.tcp_translation[2]

    gripper_width = action.gripper_width
    if gripper_width is not None:
        max_width = control.fallback_gripper_max_width
        if observation is not None and observation.gripper_max_width not in (None, 0.0):
            max_width = float(observation.gripper_max_width)
        gripper_width = clamp(gripper_width, 0.0, max_width)

    speed = clamp(action.speed if action.speed is not None else control.speed, 0.0, 1.0)
    gripper_speed = max(
        action.gripper_speed if action.gripper_speed is not None else control.gripper_speed,
        0.001,
    )
    gripper_force = max(
        action.gripper_force if action.gripper_force is not None else control.gripper_force,
        0.1,
    )

    return RobotAction(
        dx=dx,
        dy=dy,
        dz=dz,
        yaw_deg=yaw_deg,
        speed=speed,
        gripper_width=gripper_width,
        gripper_speed=gripper_speed,
        gripper_force=gripper_force,
        metadata=dict(action.metadata),
    )
