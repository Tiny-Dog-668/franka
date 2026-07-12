#!/usr/bin/env python3
"""
Read current Franka state without commanding any motion.

Usage:
  source /home/td/franka/.venv/bin/activate
  python3 /home/td/franka/scripts/robot/read_franka_state.py --ip 172.16.0.2
  python3 /home/td/franka/scripts/robot/read_franka_state.py --ip 172.16.0.2 --count 0 --interval 1.0
"""

import argparse
import os
import time

from franky import RealtimeConfig, Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read Franka state with franky")
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP"),
        help="Robot IP address. Falls back to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of samples to print. Use 0 for continuous mode.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Delay between samples in seconds.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode. 'ignore' is recommended for read-only checks.",
    )
    return parser


def active_error_names(errors) -> list[str]:
    names = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        value = getattr(errors, name)
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def mode_name(mode) -> str:
    return str(mode).split(".")[-1]


def print_state(robot: Robot, sample_idx: int) -> None:
    state = robot.state
    pose = robot.current_pose.end_effector_pose
    translation = pose.translation.tolist()
    quaternion = pose.quaternion.tolist()
    joint_positions = state.q.tolist()
    joint_velocities = state.dq.tolist()

    current_errors = active_error_names(state.current_errors)
    last_motion_errors = active_error_names(state.last_motion_errors)

    print("=" * 72)
    print(f"Sample #{sample_idx}")
    print(f"Robot mode: {mode_name(state.robot_mode)}")
    print(f"Has errors: {robot.has_errors}")
    print(f"Is in control: {robot.is_in_control}")
    print(f"Control command success rate: {state.control_command_success_rate:.3f}")
    print(f"Joint positions q [rad]: {joint_positions}")
    print(f"Joint velocities dq [rad/s]: {joint_velocities}")
    print(f"TCP translation [m]: {translation}")
    print(f"TCP quaternion [x, y, z, w]: {quaternion}")
    print(f"External wrench O_F_ext_hat_K: {state.O_F_ext_hat_K.tolist()}")
    print(f"Current errors: {current_errors if current_errors else 'none'}")
    print(f"Last motion errors: {last_motion_errors if last_motion_errors else 'none'}")


def main() -> int:
    args = build_parser().parse_args()
    if not args.ip:
        print("Please pass --ip <robot_ip> or set FRANKA_ROBOT_IP.")
        return 1

    realtime_config = (
        RealtimeConfig.Ignore
        if args.realtime == "ignore"
        else RealtimeConfig.Enforce
    )

    robot = Robot(args.ip, realtime_config=realtime_config)
    print(f"Connected to Franka at {args.ip}")
    print(f"Realtime mode: {args.realtime}")

    sample_idx = 1
    while args.count == 0 or sample_idx <= args.count:
        print_state(robot, sample_idx)
        sample_idx += 1
        if args.count == 0 or sample_idx <= args.count:
            time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
