#!/usr/bin/env python3
"""Move Franka to the saved policy rollout initial pose."""

from __future__ import annotations

import argparse
import math
import os
import time

from franky import ControlException, Gripper, JointMotion, RealtimeConfig, Robot


# Initial joint configuration from the simulation environment. JointMotion is
# used so the real robot starts from the same joint branch as the policy saw in
# simulation; the resulting real TCP pose is measured and printed afterward.
SIM_INITIAL_JOINT_DEGREES = [
    -17.649,
    -6.608,
    17.296,
    -130.236,
    2.348,
    123.894,
    43.211,
]

TARGET_JOINT_POSITION = [
    -0.308033159684479,
    -0.115331356971785,
    0.301872147424939,
    -2.273047004627335,
    0.040980330836827,
    2.162358223465855,
    0.754174223079270,
]

SIM_INITIAL_FINGER_JOINT_POSITION = 0.02
TARGET_GRIPPER_WIDTH = 2.0 * SIM_INITIAL_FINGER_JOINT_POSITION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move Franka to the simulation-aligned policy initial joint pose."
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
        "--max-step-rad",
        type=float,
        default=0.10,
        help="Maximum absolute joint change per staged segment in radians.",
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Execute only the first staged segment toward the saved pose.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.2,
        help="Seconds to wait before verifying final joint position and velocity.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode.",
    )
    parser.add_argument(
        "--skip-gripper",
        action="store_true",
        help="Do not move the gripper to the saved width.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.05,
        help="Gripper speed in m/s.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.speed <= 1.0:
        print("--speed must be in the range (0, 1].")
        return 1
    if args.max_step_rad <= 0.0:
        print("--max-step-rad must be positive.")
        return 1
    if args.settle_time < 0.0:
        print("--settle-time must be non-negative.")
        return 1

    realtime_config = (
        RealtimeConfig.Ignore if args.realtime == "ignore" else RealtimeConfig.Enforce
    )

    print("Saved policy initial pose:")
    print(f"  joint position [deg]: {SIM_INITIAL_JOINT_DEGREES}")
    print(f"  joint position [rad]: {TARGET_JOINT_POSITION}")
    if not args.skip_gripper:
        print(
            "  gripper: "
            f"finger1={SIM_INITIAL_FINGER_JOINT_POSITION:.6f} m, "
            f"finger2={SIM_INITIAL_FINGER_JOINT_POSITION:.6f} m, "
            f"total_width={TARGET_GRIPPER_WIDTH:.6f} m"
        )
    print(f"  speed: {args.speed}")
    print(f"  realtime: {args.realtime}")

    robot = Robot(args.ip, realtime_config=realtime_config)
    robot.relative_dynamics_factor = args.speed
    print(f"Connected to Franka arm at {args.ip}")

    current_q = [float(value) for value in robot.current_joint_state.position]
    delta_q = [
        target - current
        for target, current in zip(TARGET_JOINT_POSITION, current_q)
    ]
    max_delta = max(abs(value) for value in delta_q)
    full_step_count = max(1, math.ceil(max_delta / args.max_step_rad))
    step_count = 1 if args.single_step else full_step_count

    print(f"Current joint position: {current_q}")
    print(f"Joint delta: {delta_q}")
    print(f"max_step_rad: {args.max_step_rad}")
    print(f"planned staged segments: {full_step_count}")
    if args.single_step:
        print("single-step mode: only the first staged segment will execute")

    try:
        input("确认工作空间、线缆和桌面安全后按 Enter 执行，Ctrl-C 取消...")
    except KeyboardInterrupt:
        print("\nCancelled; no motion command was sent.")
        return 130

    if robot.has_errors:
        print("Robot has an active error; attempting recovery before the confirmed motion.")
        robot.recover_from_errors()

    for step_index in range(1, step_count + 1):
        fraction = step_index / full_step_count
        target_q = [
            current + fraction * delta
            for current, delta in zip(current_q, delta_q)
        ]
        print(f"Segment {step_index}/{full_step_count}: target q={target_q}")
        try:
            robot.move(JointMotion(target_q, relative_dynamics_factor=args.speed))
        except KeyboardInterrupt:
            print("\nMotion interrupted; requesting robot.stop()...")
            robot.stop()
            return 130
        except ControlException as exc:
            print(f"Arm motion failed at segment {step_index}: {exc}")
            return 1

    if args.single_step and full_step_count > 1:
        print("Finished the first staged segment; the full initial pose was not reached.")
        return 0

    time.sleep(args.settle_time)
    state = robot.state
    actual_q = [float(value) for value in state.q]
    actual_dq = [float(value) for value in state.dq]
    joint_errors = [
        actual - target
        for actual, target in zip(actual_q, TARGET_JOINT_POSITION)
    ]
    pose = robot.current_pose.end_effector_pose
    print("Arm reached saved policy initial joint pose.")
    print(f"Actual joint position [rad]: {actual_q}")
    print(f"Joint position error [rad]: {joint_errors}")
    print(f"Actual joint velocity [rad/s]: {actual_dq}")
    print(f"Max absolute joint position error [rad]: {max(abs(x) for x in joint_errors):.9f}")
    print(f"Max absolute joint velocity [rad/s]: {max(abs(x) for x in actual_dq):.9f}")
    print(f"Actual TCP translation: {pose.translation.tolist()}")
    print(f"Actual TCP quaternion: {pose.quaternion.tolist()}")

    if not args.skip_gripper:
        gripper = Gripper(args.ip)
        print(
            f"Connected to gripper at {args.ip} "
            f"(width={gripper.width:.6f} m, max_width={gripper.max_width:.6f} m)."
        )
        success = gripper.move(TARGET_GRIPPER_WIDTH, args.gripper_speed)
        if not success:
            print("Gripper move reported failure.")
            return 1
        print(f"Gripper moved to {gripper.width:.6f} m.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
