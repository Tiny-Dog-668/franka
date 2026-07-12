#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os

from franky import JointMotion, RealtimeConfig, Robot


# This joint pose was captured earlier in this workspace with TCP z ~= 0.286 m.
# It is used as a joint-space recovery target when Cartesian motion is rejected
# near singular poses.
SAFE_JOINT_POSITION = [
    -2.5358874335035484,
    0.5039265391136142,
    0.2882268240939081,
    -1.4756666805883107,
    -0.1739286195968113,
    1.8799285323489334,
    0.2502478689500724,
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move Franka to a previously observed higher joint-space recovery pose."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.01,
        help="Relative dynamics factor. Keep this small for recovery.",
    )
    parser.add_argument(
        "--max-step-rad",
        type=float,
        default=0.15,
        help="Maximum absolute joint change per staged segment in radians.",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Execute only the first staged segment toward the safe pose.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.speed <= 0.0:
        print("--speed must be positive.")
        return 1
    if args.max_step_rad <= 0.0:
        print("--max-step-rad must be positive.")
        return 1

    robot = Robot(args.ip, realtime_config=RealtimeConfig.Ignore)
    robot.recover_from_errors()
    robot.relative_dynamics_factor = args.speed

    current_q = [float(value) for value in robot.current_joint_state.position]
    delta_q = [
        target - current
        for target, current in zip(SAFE_JOINT_POSITION, current_q)
    ]

    print("Current q:")
    print(current_q)
    print("Target safe q:")
    print(SAFE_JOINT_POSITION)
    print("Delta q:")
    print(delta_q)
    print(f"speed: {args.speed}")
    max_delta = max(abs(value) for value in delta_q)
    step_count = max(1, math.ceil(max_delta / args.max_step_rad))
    if args.single_step:
        step_count = 1
    print(f"max_step_rad: {args.max_step_rad}")
    print(f"planned staged segments: {step_count}")
    print("This is a staged joint-space recovery motion to a previously observed TCP z ~= 0.286 m pose.")
    try:
        input("确认工作空间、线缆、桌面和夹爪附近安全后按 Enter 执行，Ctrl-C 取消...")
    except KeyboardInterrupt:
        print("\n已取消，没有发送运动命令。")
        return 130

    for step_index in range(1, step_count + 1):
        if args.single_step:
            fraction = min(args.max_step_rad / max_delta, 1.0) if max_delta > 0.0 else 1.0
        else:
            fraction = step_index / step_count
        target_q = [
            float(current + fraction * delta)
            for current, delta in zip(current_q, delta_q)
        ]
        print(f"Segment {step_index}/{step_count}: target q={target_q}")
        try:
            robot.move(JointMotion(target_q, relative_dynamics_factor=args.speed))
        except KeyboardInterrupt:
            print("\n运动中收到 Ctrl-C，正在请求 robot.stop()...")
            robot.stop()
            print("已发送 stop。请读取 robot state 确认当前位置和错误状态。")
            return 130

    if args.single_step:
        print("Finished first staged recovery segment.")
    else:
        print("Reached safe joint recovery target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
