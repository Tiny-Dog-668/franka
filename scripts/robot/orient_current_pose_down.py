#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from franky import Affine, CartesianMotion, ControlException, RealtimeConfig, ReferenceType, Robot


DOWN_QUATERNION_XYZW = [1.0, 0.0, 0.0, 0.0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep the current TCP translation and rotate the Franka TCP to point vertically down."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.03,
        help="Relative dynamics factor in the range [0, 1].",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    realtime_config = (
        RealtimeConfig.Ignore if args.realtime == "ignore" else RealtimeConfig.Enforce
    )
    robot = Robot(args.ip, realtime_config=realtime_config)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = args.speed

    current_pose = robot.current_pose.end_effector_pose
    translation = current_pose.translation.tolist()
    quaternion = current_pose.quaternion.tolist()

    print("Current pose:")
    print(f"  translation: {translation}")
    print(f"  quaternion:  {quaternion}")
    print("Target pose:")
    print(f"  translation: {translation}")
    print(f"  quaternion:  {DOWN_QUATERNION_XYZW}")
    print(f"  speed: {args.speed}")
    input("确认工作空间安全后按 Enter 原地旋转到竖直向下，Ctrl-C 取消...")

    motion = CartesianMotion(
        Affine(translation, DOWN_QUATERNION_XYZW),
        ReferenceType.Absolute,
        args.speed,
    )
    try:
        robot.move(motion)
    except ControlException as exc:
        print(f"Motion failed: {exc}")
        return 1

    print("TCP orientation is now commanded to vertical down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
