#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (
    load_bundle_config,
    run_bundle_deploy,
    validate_bundle_artifacts,
)

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
    parser.add_argument(
        "--model",
        help="Override the TorchScript model path from the deployment config.",
    )
    parser.add_argument(
        "--metadata",
        help=(
            "Override the model metadata JSON path; when --model is supplied, "
            "the default is MODEL with a .json suffix."
        ),
    )
    parser.add_argument(
        "--device",
        help="Override the inference device from the config, for example cpu or cuda:0.",
    )
    parser.add_argument("--robot-ip", help="Override robot_ip from the config.")
    parser.add_argument("--steps", type=int, help="Override runner.steps from the config.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run TorchScript policy inference on the first CUDA GPU (cuda:0).",
    )
    parser.add_argument("--image", help="Use a static RGB image instead of live RealSense input.")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Run observation, image preprocessing, inference, and action mapping without motion.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate hashes, metadata contract, model loading, and dummy inference "
            "without hardware."
        ),
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
    parser.add_argument(
        "--control-mode",
        choices=("blocking", "streaming"),
        help="Override the Bundle control mode.",
    )
    parser.add_argument(
        "--allow-full-scale",
        action="store_true",
        help="Unlock streaming normalized actions from commissioning limit to +/-1.0.",
    )
    parser.add_argument(
        "--commissioning-limit",
        type=float,
        help=(
            "Temporarily lower the streaming normalized action limit for a hardware "
            "diagnostic run; it cannot raise the configured limit."
        ),
    )
    parser.add_argument(
        "--action-limit",
        type=float,
        help=(
            "Temporarily set the streaming normalized action limit in (0, 1]. "
            "Unlike --commissioning-limit, this may raise the configured limit; "
            "it cannot be combined with --yes."
        ),
    )
    parser.add_argument(
        "--streaming-check",
        action="store_true",
        help="Hold the current joints for two seconds and test 60 Hz communication only.",
    )
    parser.add_argument(
        "--cube-position-root",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help=(
            "Bypass RMA visual position prediction and use a fixed cube-center XYZ "
            "in robot_root metres for the whole session."
        ),
    )
    parser.add_argument(
        "--rma-position-source",
        choices=("vision", "oracle"),
        help="Select the RMA cube-position source.",
    )
    parser.add_argument(
        "--rma-contact-source",
        choices=("vision", "zero"),
        help="Select visual contact probabilities or constant [0,0] contact.",
    )
    return parser


def main(default_config: str | Path = DEFAULT_CONFIG) -> int:
    args = build_parser(default_config).parse_args()
    config = load_bundle_config(args.config)

    if args.gpu and args.device:
        raise ValueError("--gpu cannot be combined with --device")
    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        config.model.model_path = str(model_path)
        if not args.metadata:
            config.model.metadata_path = str(model_path.with_suffix(".json"))
    if args.metadata:
        config.model.metadata_path = str(Path(args.metadata).expanduser().resolve())
    if args.device:
        config.model.device = args.device
    if args.gpu:
        config.model.device = "cuda:0"
    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.steps is not None:
        if args.steps < 1:
            raise ValueError("--steps must be at least 1")
        config.runner.steps = args.steps
    if args.image:
        config.camera.source = "image"
        config.camera.image_path = args.image
    if args.control_mode:
        config.control_mode = args.control_mode
    if args.streaming_check:
        config.control_mode = "streaming"
    if args.rma_position_source:
        config.model.rma_position_source = args.rma_position_source
        if args.rma_position_source == "vision" and args.cube_position_root is None:
            config.model.rma_oracle_cube_position_root = None
    if args.cube_position_root is not None:
        config.model.rma_position_source = "oracle"
        config.model.rma_oracle_cube_position_root = list(args.cube_position_root)
    if args.rma_contact_source:
        config.model.rma_contact_source = args.rma_contact_source

    is_streaming = config.control_mode == "streaming"
    if args.action_limit is not None and args.commissioning_limit is not None:
        raise ValueError("--action-limit cannot be combined with --commissioning-limit")
    if args.action_limit is not None and args.allow_full_scale:
        raise ValueError("--action-limit cannot be combined with --allow-full-scale")
    if args.commissioning_limit is not None:
        if not is_streaming:
            raise ValueError("--commissioning-limit is only valid with streaming control")
        if not 0.0 < args.commissioning_limit <= config.streaming.commissioning_action_limit:
            raise ValueError(
                "--commissioning-limit must be positive and no greater than the "
                f"configured limit {config.streaming.commissioning_action_limit:g}"
            )
        config.streaming.commissioning_action_limit = args.commissioning_limit
    if args.action_limit is not None:
        if not is_streaming:
            raise ValueError("--action-limit is only valid with streaming control")
        if not 0.0 < args.action_limit <= 1.0:
            raise ValueError("--action-limit must be finite and in (0, 1]")
        config.streaming.commissioning_action_limit = args.action_limit
    if is_streaming and args.confirm_each_step:
        raise ValueError("Streaming control does not support --confirm-each-step")
    if args.allow_full_scale and not is_streaming:
        raise ValueError("--allow-full-scale is only valid with streaming control")
    if args.allow_full_scale and args.yes:
        raise ValueError("Full-scale streaming cannot be combined with --yes")
    if args.action_limit is not None and args.yes:
        raise ValueError("--action-limit cannot be combined with --yes")
    if args.streaming_check and args.preview_only:
        raise ValueError("--streaming-check cannot be combined with --preview-only")
    if args.streaming_check and args.validate_only:
        raise ValueError("--streaming-check cannot be combined with --validate-only")

    metadata = json.loads(Path(config.model.metadata_path).read_text(encoding="utf-8"))
    policy_name = metadata.get("task") or metadata.get("kind", "unknown")
    print(f"Policy task: {policy_name}")
    print(f"Config: {args.config}")
    print(f"Model: {config.model.model_path}")
    print(f"Inference device: {config.model.device}")
    print(
        "TorchScript inference optimization: "
        f"{'enabled' if config.model.optimize_for_inference else 'disabled'}"
    )
    input_order = metadata.get("input_order", ["action_history", "proprio_obs", "wrist_rgb"])
    input_signature = metadata.get("input_signature", {})
    print(
        "Model inputs: "
        + ", ".join(
            (
                f"{name}[{','.join(str(value) for value in input_signature[name])}]"
                if name in input_signature
                else str(name)
            )
            for name in input_order
        )
    )
    print("Model outputs: [dx, dy, dz, gripper]")
    print(f"Control mode: {config.control_mode}")
    print(
        "History: "
        f"source={config.model.history_source}, "
        f"scale={config.model.history_scale}, "
        f"delay_steps={config.model.history_delay_steps}"
    )
    if (
        config.model.rma_position_source != "vision"
        or config.model.rma_contact_source != "vision"
    ):
        print(
            "RMA actor input override: "
            f"position_source={config.model.rma_position_source}, "
            f"cube_position_root={config.model.rma_oracle_cube_position_root}, "
            f"contact_source={config.model.rma_contact_source}"
        )
    if args.validate_only:
        report = validate_bundle_artifacts(config)
        print("Validation: PASS")
        print(json.dumps(report, indent=2))
        return 0
    if args.preview_only:
        print("Mode: preview only; arm, gripper, and automatic gripper homing are disabled.")
    else:
        print("Mode: REAL ROBOT MOTION")
        if is_streaming:
            action_limit = (
                1.0
                if args.allow_full_scale
                else config.streaming.commissioning_action_limit
            )
            print(
                "Streaming: "
                f"policy={config.streaming.policy_frequency_hz:g} Hz, "
                f"IK={config.streaming.ik_frequency_hz:g} Hz, "
                f"normalized action limit=+/-{action_limit:g}"
            )
            print(f"Streaming backend: {config.streaming.backend}")
            collision_behavior = config.streaming.collision_behavior
            if collision_behavior is None:
                print(
                    "Collision behavior: unchanged; active controller thresholds "
                    "cannot be read back"
                )
            else:
                print(
                    "Collision behavior: explicitly configured before control "
                    "(Cartesian order: Fx, Fy, Fz, Mx, My, Mz)"
                )
                print(
                    "  joint contact/collision [Nm]: "
                    f"{collision_behavior.lower_torque_thresholds} / "
                    f"{collision_behavior.upper_torque_thresholds}"
                )
                print(
                    "  Cartesian contact/collision [N,N,N,Nm,Nm,Nm]: "
                    f"{collision_behavior.lower_force_thresholds} / "
                    f"{collision_behavior.upper_force_thresholds}"
                )

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
        if is_streaming:
            prompt = "Type y/yes to start this streaming session: "
        else:
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
        allow_full_scale=args.allow_full_scale,
        streaming_check=args.streaming_check,
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
