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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bundled end-to-end sim2real policy on Franka.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "e2e_bundle_real_example.json"),
        help="Path to the bundle deployment JSON config.",
    )
    parser.add_argument(
        "--robot-ip",
        help="Override robot_ip from the config.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override runner.steps from the config.",
    )
    parser.add_argument(
        "--image",
        help="Use a static RGB image instead of live RealSense input.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Run the full observation/inference/action-mapping chain but do not execute robot motion.",
    )
    parser.add_argument(
        "--confirm-step",
        action="store_true",
        help="Ask for an extra confirmation before each step is executed.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_bundle_config(args.config)

    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.steps is not None:
        config.runner.steps = args.steps
    if args.image:
        config.camera.source = "image"
        config.camera.image_path = args.image

    if args.preview_only:
        print("Running the TorchScript bundle in preview-only mode on the real Franka setup.")
    else:
        print("Running the TorchScript bundle on the real Franka robot.")
    if config.camera.source == "image":
        print(f"Camera source: static image ({config.camera.image_path})")

    confirmed_session = {"armed": False}

    def confirm_step(step_index, robot_action, observation):
        if confirmed_session["armed"]:
            return True
        print(f"Step {step_index} proposed action:")
        print(
            json.dumps(
                {
                    "dx": robot_action.dx,
                    "dy": robot_action.dy,
                    "dz": robot_action.dz,
                    "yaw_deg": robot_action.yaw_deg,
                    "gripper_width": robot_action.gripper_width,
                    "speed": robot_action.speed,
                },
                indent=2,
            )
        )
        print("Current TCP:", [round(value, 6) for value in observation.tcp_translation])
        print("Current gripper_width:", observation.gripper_width)
        confirm = input("Type y/yes to run this step and continue remaining steps automatically: ")
        confirmed = confirm.strip().lower() in {"y", "yes"}
        if confirmed:
            confirmed_session["armed"] = True
        return confirmed

    summary = run_bundle_deploy(
        config,
        execute_motion=not args.preview_only,
        confirm_step_callback=confirm_step if args.confirm_step and not args.preview_only else None,
    )
    print(f"Run saved to: {summary['run_dir']}")
    print(f"Executed steps: {summary.get('num_steps', 0)}")
    if summary.get("steps"):
        last_step = summary["steps"][-1]
        last_info = last_step.get("info", {})
        timing = last_info.get("timing", {})
        observation = last_step.get("observation", {})
        print(
            json.dumps(
                {
                    "motion_executed": last_info.get("motion_executed"),
                    "success": last_info.get("success"),
                    "reward": last_info.get("reward"),
                    "position_error_m": last_info.get("position_error_m"),
                    "yaw_error_deg": last_info.get("yaw_error_deg"),
                    "tcp_translation": observation.get("tcp_translation"),
                    "gripper_width": observation.get("gripper_width"),
                    "timing_ms": {
                        "arm_move": timing.get("arm_move_ms"),
                        "gripper_move": timing.get("gripper_move_ms"),
                        "policy_infer": timing.get("policy_infer_ms"),
                        "step_total": timing.get("step_total_ms"),
                    },
                },
                indent=2,
            )
        )
        print("Full rollout is stored in rollout.jsonl and summary.json under the run directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
