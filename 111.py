from __future__ import annotations

import argparse
import os

from franky import JointMotion, RealtimeConfig, Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 JointMotion 小步调整 panda_joint4，帮助机械臂退出 Cartesian singular 附近。"
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address.",
    )
    parser.add_argument(
        "--q4-delta",
        type=float,
        default=-0.2,
        help="Relative change for panda_joint4 in rad. Negative bends the elbow more.",
    )
    parser.add_argument(
        "--q4-target",
        type=float,
        help="Absolute target for panda_joint4 in rad. Overrides --q4-delta.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.02,
        help="Relative dynamics factor.",
    )
    parser.add_argument(
        "--min-q4",
        type=float,
        default=-2.5,
        help="Lower safety clamp for panda_joint4 in rad.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    robot = Robot(args.ip, realtime_config=RealtimeConfig.Ignore)

    robot.recover_from_errors()
    robot.relative_dynamics_factor = args.speed

    q = list(robot.current_joint_state.position)
    print("current q:", q)

    q_safe = q.copy()
    if args.q4_target is None:
        q_safe[3] = q[3] + args.q4_delta
    else:
        q_safe[3] = args.q4_target
    q_safe[3] = max(q_safe[3], args.min_q4)

    print("target q:", q_safe)
    print("delta q:", [target - current for target, current in zip(q_safe, q)])
    print(f"speed: {args.speed}")

    try:
        input("确认工作空间安全后按 Enter 执行关节运动，Ctrl-C 取消...")
    except KeyboardInterrupt:
        print("\n已取消，没有发送运动命令。")
        return 130

    robot.move(JointMotion(q_safe, relative_dynamics_factor=args.speed))
    print("JointMotion finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
