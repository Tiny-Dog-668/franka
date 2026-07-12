#!/usr/bin/env python3
"""
Read the robot's current pose and print it in a form that can be copied into
/home/td/franka/scripts/robot/go_to_zero_pose.py or a config file.
"""

from __future__ import annotations

import argparse
import os

from franky import Gripper, RealtimeConfig, Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the current Franka pose and print zero-pose constants."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode.",
    )
    parser.add_argument(
        "--include-gripper",
        action="store_true",
        help="Also read the current gripper width.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    realtime_config = (
        RealtimeConfig.Ignore if args.realtime == "ignore" else RealtimeConfig.Enforce
    )

    robot = Robot(args.ip, realtime_config=realtime_config)
    pose = robot.current_pose.end_effector_pose
    state = robot.state

    translation = pose.translation.tolist()
    quaternion = pose.quaternion.tolist()

    print("Current Franka pose:")
    print(f"  tcp_translation: {translation}")
    print(f"  tcp_quaternion:  {quaternion}")
    print(f"  joint_positions: {state.q.tolist()}")

    print("\nCopyable constants:")
    print("TARGET_TRANSLATION = [")
    for value in translation:
        print(f"    {value},")
    print("]")
    print()
    print("TARGET_QUATERNION = [")
    for value in quaternion:
        print(f"    {value},")
    print("]")

    if args.include_gripper:
        gripper = Gripper(args.ip)
        print()
        print(f"TARGET_GRIPPER_WIDTH = {gripper.width}")
        print(f"Current gripper max width = {gripper.max_width}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
