#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import load_bundle_config, run_bundle_deploy

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0402.json"
DEFAULT_MODEL_DIR = REPO_ROOT / "exported_0402"
MODEL_PATTERN = re.compile(r"policy_actor_e2e_agent_(\d+)\.pt$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exported_0402 on the real Franka while recording per-step observations and actions."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Base deployment config path.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing exported_0402 model .pt/.json pairs.",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Use a specific checkpoint step such as 150000 or 200000. Defaults to the latest available step.",
    )
    parser.add_argument("--robot-ip", help="Override robot_ip from the config.")
    parser.add_argument("--steps", type=int, help="Override runner.steps from the config.")
    parser.add_argument("--image", help="Use a static RGB image instead of live RealSense input.")
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


def _resolve_model_pair(model_dir: Path, step: int | None) -> tuple[Path, Path, int]:
    candidates: list[tuple[int, Path, Path]] = []
    for model_path in sorted(model_dir.glob("policy_actor_e2e_agent_*.pt")):
        match = MODEL_PATTERN.match(model_path.name)
        if match is None:
            continue
        current_step = int(match.group(1))
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.is_file():
            candidates.append((current_step, model_path, metadata_path))

    if not candidates:
        raise FileNotFoundError(f"No exported model pairs were found under: {model_dir}")

    if step is None:
        current_step, model_path, metadata_path = max(candidates, key=lambda item: item[0])
        return model_path, metadata_path, current_step

    for current_step, model_path, metadata_path in candidates:
        if current_step == step:
            return model_path, metadata_path, current_step

    available = ", ".join(str(current_step) for current_step, _, _ in candidates)
    raise FileNotFoundError(
        f"Could not find exported_0402 step {step}. Available steps: {available}"
    )


def _configure_action_adapter(config, metadata: dict) -> None:
    output_signature = metadata.get("output_signature", {})
    action_dim = int(output_signature.get("mean_actions", [0])[0])
    current_lengths = [
        len(config.action_adapter.labels),
        len(config.action_adapter.scales),
        len(config.action_adapter.clip_low),
        len(config.action_adapter.clip_high),
    ]

    if current_lengths == [action_dim, action_dim, action_dim, action_dim]:
        return

    if action_dim == 4:
        config.action_adapter.labels = ["dx", "dy", "dz", "gripper"]
        config.action_adapter.scales = [0.01, 0.01, 0.01, 0.005]
        config.action_adapter.clip_low = [-1.0, -1.0, -1.0, -1.0]
        config.action_adapter.clip_high = [1.0, 1.0, 1.0, 1.0]
        return

    if action_dim == 5:
        config.action_adapter.labels = ["dx", "dy", "dz", "yaw_deg", "gripper"]
        config.action_adapter.scales = [0.01, 0.01, 0.01, 0.0, 0.005]
        config.action_adapter.clip_low = [-1.0, -1.0, -1.0, -1.0, -1.0]
        config.action_adapter.clip_high = [1.0, 1.0, 1.0, 1.0, 1.0]
        return

    raise ValueError(f"Unsupported exported_0402 action dim: {action_dim}")


def main() -> int:
    args = build_parser().parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_path, metadata_path, step = _resolve_model_pair(model_dir, args.step)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config = load_bundle_config(args.config)
    config.model.model_path = str(model_path)
    config.model.metadata_path = str(metadata_path)
    config.runner.run_name = f"{config.runner.run_name}_record_agent_{step}"
    _configure_action_adapter(config, metadata)

    if args.robot_ip:
        config.robot_ip = args.robot_ip
    if args.steps is not None:
        config.runner.steps = args.steps
    if args.image:
        config.camera.source = "image"
        config.camera.image_path = args.image

    if args.preview_only:
        print("Recording exported_0402 in preview-only mode on the real Franka setup.")
    else:
        print("Recording exported_0402 on the real Franka robot.")
    print(f"Selected model step: {step}")
    print(f"Model path: {config.model.model_path}")
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
        save_step_data=True,
    )
    print(f"Run saved to: {summary['run_dir']}")
    print(f"Executed steps: {summary.get('num_steps', 0)}")
    print("Recorded files:")
    print("  - rollout.jsonl")
    print("  - summary.json")
    print("  - rgb/step_XXXX.png")
    print("  - step_data/step_XXXX.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
