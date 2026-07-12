#!/usr/bin/env python3
"""
Minimal Franka motion example using franky.

Usage:
  source /home/td/franka/.venv/bin/activate
  python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2

This script performs one small relative Cartesian motion.
"""

import argparse
import math
import os

from franky import (
    Affine,
    CartesianMotion,
    ControlException,
    Gripper,
    RealtimeConfig,
    ReferenceType,
    Robot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Franka motion test with franky")
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP"),
        help="Robot IP address. Falls back to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        default=0.0,
        help="Relative X motion in meters.",
    )
    parser.add_argument(
        "--dy",
        type=float,
        default=0.0,
        help="Relative Y motion in meters.",
    )
    parser.add_argument(
        "--dz",
        type=float,
        default=0.0,
        help="Relative Z motion in meters.",
    )
    parser.add_argument(
        "--yaw",
        type=float,
        default=0.0,
        help="Relative yaw rotation in degrees about Z.",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=0.0,
        help="Relative roll rotation in degrees about X.",
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=0.0,
        help="Relative pitch rotation in degrees about Y.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.05,
        help="Relative dynamics factor in the range [0, 1].",
    )
    gripper_group = parser.add_mutually_exclusive_group()
    gripper_group.add_argument(
        "--gripper-open",
        action="store_true",
        help="Open the gripper fully.",
    )
    gripper_group.add_argument(
        "--gripper-width",
        type=float,
        help="Move the gripper to the requested width in meters.",
    )
    gripper_group.add_argument(
        "--grasp-width",
        type=float,
        help="Close the gripper until grasping at the requested width in meters.",
    )
    gripper_group.add_argument(
        "--gripper-homing",
        action="store_true",
        help="Run gripper homing.",
    )
    parser.add_argument(
        "--gripper-speed",
        type=float,
        default=0.05,
        help="Gripper speed in meters per second.",
    )
    parser.add_argument(
        "--gripper-force",
        type=float,
        default=20.0,
        help="Gripper force in newtons for grasp actions.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode. Use 'ignore' on a normal desktop without RT privileges.",
    )
    parser.add_argument(
        "--async-move",
        action="store_true",
        help="Send the arm motion asynchronously and return without waiting for completion.",
    )
    return parser


def rpy_quaternion_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[float]:
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def run_gripper_action(args: argparse.Namespace) -> int:
    gripper = Gripper(args.ip)
    print(
        f"Connected to gripper at {args.ip} "
        f"(width={gripper.width:.3f} m, max_width={gripper.max_width:.3f} m)."
    )

    if args.gripper_homing:
        print("Running gripper homing.")
        success = gripper.homing()
    elif args.gripper_open:
        print(f"Opening gripper at speed {args.gripper_speed:.3f} m/s.")
        success = gripper.open(args.gripper_speed)
    elif args.gripper_width is not None:
        print(
            f"Moving gripper to width={args.gripper_width:.3f} m "
            f"at speed {args.gripper_speed:.3f} m/s."
        )
        success = gripper.move(args.gripper_width, args.gripper_speed)
    elif args.grasp_width is not None:
        print(
            f"Grasping to width={args.grasp_width:.3f} m "
            f"at speed {args.gripper_speed:.3f} m/s with force {args.gripper_force:.1f} N."
        )
        success = gripper.grasp(args.grasp_width, args.gripper_speed, args.gripper_force)
    else:
        return 0

    if not success:
        print("Gripper command reported failure.")
        return 1

    print(
        f"Gripper action finished. width={gripper.width:.3f} m, "
        f"is_grasped={gripper.is_grasped}"
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if not args.ip:
        print("Please pass --ip <robot_ip> or set FRANKA_ROBOT_IP.")
        return 1

    arm_requested = any(
        value != 0.0
        for value in (args.dx, args.dy, args.dz, args.roll, args.pitch, args.yaw)
    )
    gripper_requested = any(
        (
            args.gripper_open,
            args.gripper_width is not None,
            args.grasp_width is not None,
            args.gripper_homing,
        )
    )

    if not arm_requested and not gripper_requested:
        print(
            "No action requested. Pass arm motion options "
            "(--dx/--dy/--dz/--roll/--pitch/--yaw) "
            "and/or a gripper option."
        )
        return 1

    if arm_requested:
        print(
            "Planned arm move: "
            f"dx={args.dx:.3f} m, dy={args.dy:.3f} m, dz={args.dz:.3f} m, "
            f"roll={args.roll:.2f} deg, pitch={args.pitch:.2f} deg, "
            f"yaw={args.yaw:.2f} deg at speed factor {args.speed:.2f}."
        )
        print(f"Realtime mode: {args.realtime}")
    if args.gripper_homing:
        print("Planned gripper action: homing")
    elif args.gripper_open:
        print(f"Planned gripper action: open at {args.gripper_speed:.3f} m/s")
    elif args.gripper_width is not None:
        print(
            "Planned gripper action: "
            f"move to width={args.gripper_width:.3f} m at {args.gripper_speed:.3f} m/s"
        )
    elif args.grasp_width is not None:
        print(
            "Planned gripper action: "
            f"grasp width={args.grasp_width:.3f} m at {args.gripper_speed:.3f} m/s "
            f"with force {args.gripper_force:.1f} N"
        )

    input("Make sure the workspace is clear, then press Enter to continue...")

    if arm_requested:
        realtime_config = (
            RealtimeConfig.Ignore
            if args.realtime == "ignore"
            else RealtimeConfig.Enforce
        )
        robot = Robot(args.ip, realtime_config=realtime_config)
        robot.relative_dynamics_factor = args.speed
        quaternion = rpy_quaternion_xyzw(args.roll, args.pitch, args.yaw)
        print(f"Connected to Franka arm at {args.ip}")

        motion = CartesianMotion(
            Affine([args.dx, args.dy, args.dz], quaternion),
            ReferenceType.Relative,
            args.speed,
        )
        try:
            robot.move(motion, asynchronous=args.async_move)
        except KeyboardInterrupt:
            print("\nMotion interrupted by Ctrl-C. Requesting robot.stop()...")
            robot.stop()
            print("Stop requested. Read robot state before sending another motion.")
            return 130
        except ControlException as exc:
            print(f"Motion failed: {exc}")
            if "communication_constraints_violation" in str(exc):
                print(
                    "Hint: the robot started the realtime motion loop, but the PC/network could not "
                    "keep the FCI timing constraints. `--realtime ignore` only skips the startup "
                    "permission check; it does not remove the 1 kHz communication requirement."
                )
            return 1

        print("Arm motion finished.")

    if gripper_requested:
        try:
            return run_gripper_action(args)
        except Exception as exc:
            print(f"Gripper action failed: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
