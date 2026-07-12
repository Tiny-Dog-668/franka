#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import load_bundle_config, run_bundle_deploy

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0711.json"


def build_parser(default_config: str | Path = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an exported cube-grasp TorchScript policy on Franka."
    )
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Deployment config path.",
    )
    parser.add_argument("--robot-ip", help="Override robot_ip from the config.")
    parser.add_argument("--steps", type=int, help="Override runner.steps from the config.")
    parser.add_argument("--image", help="Use a static RGB image instead of live RealSense input.")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Run observation, image preprocessing, inference, and action mapping without motion.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Execute all requested real-robot steps without the first-step confirmation.",
    )
    parser.add_argument(
        "--confirm-each-step",
        action="store_true",
        help="Require confirmation before every real-robot policy step.",
    )
    return parser


def main(default_config: str | Path = DEFAULT_CONFIG) -> int:
    args = build_parser(default_config).parse_args()
    config = load_bundle_config(args.config)

    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("--steps must be at least 1")
        config.runner.steps = args.steps
    if args.image:
        config.camera.source = "image"
        config.camera.image_path = args.image

    metadata = json.loads(Path(config.model.metadata_path).read_text(encoding="utf-8"))
    print(f"Policy task: {metadata.get('task', 'unknown')}")
    print(f"Config: {args.config}")
    print(f"Model: {config.model.model_path}")
    print("Model inputs: action_history[4], proprio_obs[15], wrist_rgb[224,224,3]")
    print("Model outputs: [dx, dy, dz, gripper]")
    print(
        "History: "
        f"source={config.model.history_source}, "
        f"scale={config.model.history_scale}, "
        f"delay_steps={config.model.history_delay_steps}"
    )
    if args.preview_only:
        print("Mode: preview only; arm, gripper, and automatic gripper homing are disabled.")
    else:
        print("Mode: REAL ROBOT MOTION")

    confirmed_session = {"armed": bool(args.yes)}

    def confirm_first_step(step_index, robot_action, observation):
        if confirmed_session["armed"]:
            return True

        print(f"Step {step_index} proposed real-robot action:")
        print(
            json.dumps(
                {
                    "dx_m": robot_action.dx,
                    "dy_m": robot_action.dy,
                    "dz_m": robot_action.dz,
                    "gripper_width_m": robot_action.gripper_width,
                    "speed": robot_action.speed,
                    "current_tcp_m": observation.tcp_translation,
                    "current_gripper_width_m": observation.gripper_width,
                },
                indent=2,
            )
        )
        prompt = (
            "Type y/yes to execute this step: "
            if args.confirm_each_step
            else "Type y/yes to execute this step and all remaining requested steps: "
        )
        confirm = input(prompt)
        confirmed = confirm.strip().lower() in {"y", "yes"}
        if confirmed and not args.confirm_each_step:
            confirmed_session["armed"] = True
        return confirmed

    summary = run_bundle_deploy(
        config,
        execute_motion=not args.preview_only,
        confirm_step_callback=(
            None if args.preview_only or args.yes else confirm_first_step
        ),
    )

    print(f"Run saved to: {summary['run_dir']}")
    print(f"Recorded steps: {summary.get('num_steps', 0)}")
    if summary.get("steps"):
        last_step = summary["steps"][-1]
        print(
            json.dumps(
                {
                    "raw_action": last_step.get("raw_action"),
                    "clipped_action": last_step.get("clipped_action"),
                    "robot_action": last_step.get("robot_action"),
                    "motion_executed": last_step.get("info", {}).get("motion_executed"),
                    "observation_after": last_step.get("observation_after"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
