#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from franka_sim2real.config import Sim2RealConfig
from franka_sim2real.e2e_bundle import BundleCameraConfig, RealSenseRGBCamera
from franka_sim2real.envs.franka_real import RealFrankaEnv
from franka_sim2real.types import RobotAction, RobotObservation


@dataclass
class DiagnosticCase:
    name: str
    action: RobotAction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose Franka action directions by executing small signed motions and saving before/after observations."
    )
    parser.add_argument("--robot-ip", default="172.16.0.2", help="Franka robot IP.")
    parser.add_argument("--realtime", choices=("ignore", "enforce"), default="ignore")
    parser.add_argument("--speed", type=float, default=0.08, help="Relative dynamics factor for diagnostic motions.")
    parser.add_argument("--dx", type=float, default=0.002, help="Signed diagnostic delta for x in meters.")
    parser.add_argument("--dy", type=float, default=0.002, help="Signed diagnostic delta for y in meters.")
    parser.add_argument("--dz", type=float, default=0.002, help="Signed diagnostic delta for z in meters.")
    parser.add_argument("--yaw", type=float, default=2.0, help="Signed diagnostic yaw delta in degrees.")
    parser.add_argument("--settle-time", type=float, default=0.2, help="Settle time after each step.")
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=20.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--cases",
        nargs="*",
        default=["dx+", "dx-", "dy+", "dy-", "dz+", "dz-", "yaw+", "yaw-"],
        help="Subset of cases to run. Choices: dx+, dx-, dy+, dy-, dz+, dz-, yaw+, yaw-",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs"),
        help="Base directory for diagnostic outputs.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip initial confirmation prompt.")
    return parser


def make_cases(args: argparse.Namespace) -> list[DiagnosticCase]:
    all_cases = {
        "dx+": DiagnosticCase("dx+", RobotAction(dx=args.dx, speed=args.speed)),
        "dx-": DiagnosticCase("dx-", RobotAction(dx=-args.dx, speed=args.speed)),
        "dy+": DiagnosticCase("dy+", RobotAction(dy=args.dy, speed=args.speed)),
        "dy-": DiagnosticCase("dy-", RobotAction(dy=-args.dy, speed=args.speed)),
        "dz+": DiagnosticCase("dz+", RobotAction(dz=args.dz, speed=args.speed)),
        "dz-": DiagnosticCase("dz-", RobotAction(dz=-args.dz, speed=args.speed)),
        "yaw+": DiagnosticCase("yaw+", RobotAction(yaw_deg=args.yaw, speed=args.speed)),
        "yaw-": DiagnosticCase("yaw-", RobotAction(yaw_deg=-args.yaw, speed=args.speed)),
    }
    invalid = [name for name in args.cases if name not in all_cases]
    if invalid:
        raise ValueError(f"Unsupported cases: {invalid}")
    return [all_cases[name] for name in args.cases]


def make_env(args: argparse.Namespace) -> RealFrankaEnv:
    env_config = Sim2RealConfig.from_dict(
        {
            "backend": {
                "kind": "real",
                "robot_ip": args.robot_ip,
                "realtime": args.realtime,
                "enable_gripper": True,
                "auto_recover": False,
                "auto_gripper_homing": True,
                "settle_time_s": args.settle_time,
            },
            "control": {
                "speed": args.speed,
                "gripper_speed": args.gripper_speed,
                "gripper_force": args.gripper_force,
                "max_dx": abs(args.dx),
                "max_dy": abs(args.dy),
                "max_dz": abs(args.dz),
                "max_yaw_deg": abs(args.yaw),
                "fallback_gripper_max_width": 0.08,
                "workspace": {
                    "minimum": [0.2, -0.3, 0.05],
                    "maximum": [0.65, 0.3, 0.45],
                },
            },
        }
    )
    return RealFrankaEnv(env_config)


def save_rgb(path: Path, rgb) -> None:
    Image.fromarray(rgb).save(path)


def observation_summary(observation: RobotObservation) -> dict:
    return {
        "tcp_translation": observation.tcp_translation,
        "tcp_quaternion": observation.tcp_quaternion,
        "tcp_yaw_deg": observation.tcp_yaw_deg,
        "gripper_width": observation.gripper_width,
        "joint_positions": observation.joint_positions,
        "joint_velocities": observation.joint_velocities,
    }


def main() -> int:
    args = build_parser().parse_args()
    cases = make_cases(args)

    run_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_direction_diagnosis"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.yes:
        print("This will execute small real robot motions for direction diagnosis.")
        print("Cases:", ", ".join(case.name for case in cases))
        confirm = input("Type YES to continue: ")
        if confirm != "YES":
            print("Aborted.")
            return 1

    camera = RealSenseRGBCamera(
        BundleCameraConfig(
            source="realsense",
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
    )
    env = make_env(args)
    results: list[dict] = []

    try:
        for index, case in enumerate(cases):
            case_dir = run_dir / f"{index:02d}_{case.name}"
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"\nCase {index + 1}/{len(cases)}: {case.name}")
            print(json.dumps(asdict(case.action), indent=2))
            confirm = input("Press Enter to capture BEFORE state, then type EXECUTE to run this case: ")
            if confirm.strip():
                print("Waiting for explicit EXECUTE...")
            confirm = input("Type EXECUTE to run this case, or just press Enter to skip: ")
            if confirm != "EXECUTE":
                print("Skipped.")
                continue

            before_rgb = camera.read()
            before_obs = env.reset()
            save_rgb(case_dir / "before.png", before_rgb)

            after_obs, reward, done, info = env.step(case.action)
            after_rgb = camera.read()
            save_rgb(case_dir / "after.png", after_rgb)

            delta_translation = [
                after_obs.tcp_translation[i] - before_obs.tcp_translation[i]
                for i in range(3)
            ]
            delta_yaw_deg = after_obs.tcp_yaw_deg - before_obs.tcp_yaw_deg
            delta_gripper = None
            if before_obs.gripper_width is not None and after_obs.gripper_width is not None:
                delta_gripper = after_obs.gripper_width - before_obs.gripper_width

            record = {
                "case": case.name,
                "commanded_action": asdict(case.action),
                "before": observation_summary(before_obs),
                "after": observation_summary(after_obs),
                "delta_translation": delta_translation,
                "delta_yaw_deg": delta_yaw_deg,
                "delta_gripper_width": delta_gripper,
                "reward": reward,
                "done": done,
                "info": info,
            }
            with (case_dir / "result.json").open("w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            results.append(record)

            print(
                json.dumps(
                    {
                        "case": case.name,
                        "delta_translation": delta_translation,
                        "delta_yaw_deg": delta_yaw_deg,
                        "delta_gripper_width": delta_gripper,
                    },
                    indent=2,
                )
            )

        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved diagnostic results to: {run_dir}")
        return 0
    finally:
        camera.close()
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
