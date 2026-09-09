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

# This is deliberately a transit-only setting, not the deployment impedance.
# The policy runner configures its own impedance when streaming starts.
DEFAULT_TRANSIT_JOINT_IMPEDANCE = [1500.0, 1500.0, 1500.0, 1200.0, 1200.0, 1000.0, 1000.0]


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
        help="Maximum absolute joint change per segment in explicit --staged mode.",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help=(
            "Use legacy multiple JointMotion segments. The default is one continuous "
            "JointMotion to avoid trajectory restart discontinuities."
        ),
    )
    parser.add_argument(
        "--single-step",
        action="store_true",
        help="Execute only the first explicit staged segment toward the saved pose.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.2,
        help="Minimum seconds to wait before verifying final joint position and velocity.",
    )
    parser.add_argument(
        "--settle-timeout",
        type=float,
        default=2.0,
        help="Maximum seconds to wait for the arm to meet the final-state tolerances.",
    )
    parser.add_argument(
        "--position-tolerance-rad",
        type=float,
        default=0.008,
        help="Required maximum absolute final joint-position error in radians.",
    )
    parser.add_argument(
        "--velocity-tolerance-rad-s",
        type=float,
        default=0.02,
        help="Required maximum absolute final joint velocity in rad/s.",
    )
    parser.add_argument(
        "--transit-joint-impedance",
        type=float,
        nargs=7,
        default=None,
        metavar=("K1", "K2", "K3", "K4", "K5", "K6", "K7"),
        help=(
            "Set a known joint stiffness only while moving to the initial pose. "
            "Use the printed recommended values after a low-impedance policy run."
        ),
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


def gripper_state_failure(
    width_m: float,
    max_width_m: float,
    target_width_m: float,
) -> str | None:
    """在发送回零夹爪指令前拒绝无效或未 homing 的状态。"""
    values = {
        "width": width_m,
        "max_width": max_width_m,
        "target_width": target_width_m,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            return f"gripper {name} is not finite ({value!r})"
    if max_width_m <= 0.0:
        return "gripper max_width is zero; the gripper is not homed or has no valid range"
    if target_width_m < 0.0 or target_width_m > max_width_m:
        return (
            f"saved target width {target_width_m:.6f} m is outside the reported "
            f"range [0, {max_width_m:.6f}] m"
        )
    if width_m < 0.0 or width_m > max_width_m:
        return (
            f"reported gripper width {width_m:.6f} m is outside the reported "
            f"range [0, {max_width_m:.6f}] m"
        )
    return None


def wait_for_stationary(
    robot: Robot,
    velocity_tolerance_rad_s: float,
    timeout_s: float,
) -> tuple[bool, list[float]]:
    """等待实测关节速度归零，拒绝在运动过程中启动下一条独立轨迹。"""
    deadline = time.monotonic() + timeout_s
    last_velocity: list[float] = []
    while True:
        state = robot.state
        last_velocity = [float(value) for value in state.dq]
        if max(abs(value) for value in last_velocity) <= velocity_tolerance_rad_s:
            return True, last_velocity
        if time.monotonic() >= deadline:
            return False, last_velocity
        time.sleep(0.02)


def motion_failure_hint(error: ControlException) -> str | None:
    """为 libfranka 的常见关节轨迹保护错误给出不放宽保护的处理建议。"""
    detail = str(error)
    if (
        "joint_motion_generator_velocity_discontinuity" in detail
        or "joint_motion_generator_acceleration_discontinuity" in detail
        or "Motion finished commanded, but the robot is still moving" in detail
    ):
        return (
            "FCI 检测到关节轨迹速度/加速度不连续，已触发反射保护。"
            "不要立即重试、不要提高 speed 或放宽限位。确认机械臂静止并完成错误恢复后，"
            "使用默认连续模式；若刚结束低阻抗 streaming，请显式传入脚本打印的 "
            "--transit-joint-impedance 参数。"
        )
    return None


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.speed <= 1.0:
        print("--speed must be in the range (0, 1].")
        return 1
    if args.max_step_rad <= 0.0:
        print("--max-step-rad must be positive.")
        return 1
    if args.settle_time < 0.0 or args.settle_timeout < args.settle_time:
        print("Require 0 <= --settle-time <= --settle-timeout.")
        return 1
    if args.position_tolerance_rad <= 0.0 or args.velocity_tolerance_rad_s <= 0.0:
        print("Final-state tolerances must be positive.")
        return 1
    if args.transit_joint_impedance is not None and any(value <= 0.0 for value in args.transit_joint_impedance):
        print("--transit-joint-impedance values must all be positive.")
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
    if args.transit_joint_impedance is None:
        print(
            "  low-impedance recovery recommendation: "
            "--transit-joint-impedance "
            + " ".join(str(int(value)) for value in DEFAULT_TRANSIT_JOINT_IMPEDANCE)
        )

    robot = Robot(args.ip, realtime_config=realtime_config)
    robot.relative_dynamics_factor = args.speed
    print(f"Connected to Franka arm at {args.ip}")

    gripper: Gripper | None = None
    if not args.skip_gripper:
        try:
            gripper = Gripper(args.ip)
            gripper_width = float(gripper.width)
            gripper_max_width = float(gripper.max_width)
        except Exception as exc:
            print(
                "Failed to read gripper state; no arm or gripper motion command was sent: "
                f"{exc}"
            )
            return 3
        print(
            f"Connected to gripper at {args.ip} "
            f"(width={gripper_width:.6f} m, max_width={gripper_max_width:.6f} m)."
        )
        failure = gripper_state_failure(
            gripper_width,
            gripper_max_width,
            TARGET_GRIPPER_WIDTH,
        )
        if failure is not None:
            print("Gripper safety gate failed; no arm or gripper motion command was sent.")
            print(f"  {failure}")
            print(
                "Clear the gripper, then run: "
                f"python scripts/robot/minimal_franka_move.py --ip {args.ip} "
                "--gripper-homing"
            )
            print("Use --skip-gripper only when intentionally positioning the arm alone.")
            return 3

    current_q = [float(value) for value in robot.current_joint_state.position]
    delta_q = [
        target - current
        for target, current in zip(TARGET_JOINT_POSITION, current_q)
    ]
    max_delta = max(abs(value) for value in delta_q)
    full_step_count = max(1, math.ceil(max_delta / args.max_step_rad))
    use_staged_motion = args.staged or args.single_step
    step_count = 1 if args.single_step else full_step_count

    print(f"Current joint position: {current_q}")
    print(f"Joint delta: {delta_q}")
    if use_staged_motion:
        print("Motion mode: explicit staged JointMotion")
        print(f"max_step_rad: {args.max_step_rad}")
        print(f"planned staged segments: {full_step_count}")
    else:
        print("Motion mode: one continuous JointMotion (default)")
        print(
            "Legacy staged plan would use "
            f"{full_step_count} segments at max_step_rad={args.max_step_rad}; it is disabled "
            "to avoid restarting the motion generator between segments."
        )
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

    if args.transit_joint_impedance is not None:
        robot.set_joint_impedance(args.transit_joint_impedance)
        print(
            "Set transit joint impedance [Nm/rad]: "
            f"{[float(value) for value in args.transit_joint_impedance]}"
        )

    stationary, entry_velocity = wait_for_stationary(
        robot,
        args.velocity_tolerance_rad_s,
        args.settle_timeout,
    )
    if not stationary:
        print("Arm is still moving before the requested recovery trajectory; no motion command was sent.")
        print(f"Last joint velocity [rad/s]: {entry_velocity}")
        return 2

    if use_staged_motion:
        targets = []
        for step_index in range(1, step_count + 1):
            fraction = step_index / full_step_count
            targets.append([
                current + fraction * delta
                for current, delta in zip(current_q, delta_q)
            ])
    else:
        targets = [list(TARGET_JOINT_POSITION)]

    for step_index, target_q in enumerate(targets, start=1):
        if use_staged_motion:
            print(f"Segment {step_index}/{full_step_count}: target q={target_q}")
        else:
            print(f"Continuous target q={target_q}")
        try:
            robot.move(JointMotion(target_q, relative_dynamics_factor=args.speed))
        except KeyboardInterrupt:
            print("\nMotion interrupted; requesting robot.stop()...")
            robot.stop()
            return 130
        except ControlException as exc:
            print(f"Arm motion failed at segment {step_index}: {exc}")
            hint = motion_failure_hint(exc)
            if hint is not None:
                print(f"中文提示：{hint}")
            return 1

        if use_staged_motion and step_index < len(targets):
            stationary, intersegment_velocity = wait_for_stationary(
                robot,
                args.velocity_tolerance_rad_s,
                args.settle_timeout,
            )
            if not stationary:
                print(
                    "Arm did not become stationary after the staged segment; "
                    "the next segment will not be sent."
                )
                print(f"Last joint velocity [rad/s]: {intersegment_velocity}")
                return 2

    if args.single_step and full_step_count > 1:
        print("Finished the first staged segment; the full initial pose was not reached.")
        return 0

    deadline = time.monotonic() + args.settle_timeout
    time.sleep(args.settle_time)
    while True:
        state = robot.state
        actual_q = [float(value) for value in state.q]
        actual_dq = [float(value) for value in state.dq]
        joint_errors = [
            actual - target
            for actual, target in zip(actual_q, TARGET_JOINT_POSITION)
        ]
        max_position_error = max(abs(value) for value in joint_errors)
        max_velocity = max(abs(value) for value in actual_dq)
        if (
            max_position_error <= args.position_tolerance_rad
            and max_velocity <= args.velocity_tolerance_rad_s
        ):
            break
        if time.monotonic() >= deadline:
            print("Arm trajectory completed, but the actual final state is outside tolerance.")
            print(f"Max absolute joint position error [rad]: {max_position_error:.9f}")
            print(f"Max absolute joint velocity [rad/s]: {max_velocity:.9f}")
            print(
                "No policy should be started from this state. If this followed a low-impedance "
                "streaming run, retry with --transit-joint-impedance."
            )
            return 2
        time.sleep(0.02)

    pose = robot.current_pose.end_effector_pose
    print("Arm reached the saved policy initial joint pose within tolerance.")
    print(f"Actual joint position [rad]: {actual_q}")
    print(f"Joint position error [rad]: {joint_errors}")
    print(f"Actual joint velocity [rad/s]: {actual_dq}")
    print(f"Max absolute joint position error [rad]: {max_position_error:.9f}")
    print(f"Max absolute joint velocity [rad/s]: {max_velocity:.9f}")
    print(f"Actual TCP translation: {pose.translation.tolist()}")
    print(f"Actual TCP quaternion: {pose.quaternion.tolist()}")

    if not args.skip_gripper:
        assert gripper is not None
        success = gripper.move(TARGET_GRIPPER_WIDTH, args.gripper_speed)
        if not success:
            print("Gripper move reported failure.")
            return 1
        print(f"Gripper moved to {gripper.width:.6f} m.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
