#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (  # noqa: E402
    load_bundle_config,
    run_bundle_deploy,
    validate_bundle_artifacts,
)
from franka_sim2real.real_rl.config import load_real_rl_config  # noqa: E402
from franka_sim2real.real_rl.apriltag_tracker import load_base_t_camera  # noqa: E402
from franka_sim2real.real_rl.offline_apriltag import label_episode_from_run  # noqa: E402
from franka_sim2real.real_rl.runtime import (  # noqa: E402
    RealRLDeploySettings,
    sha256_file,
)
from franka_sim2real.real_rl.replay_buffer import (  # noqa: E402
    ReplayBuffer,
    real_rl_replay_contract,
)
from franka_sim2real.real_rl.trainer import train_from_replay  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "real_residual_sac_0814.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Franka real-world Residual SAC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate artifacts without robot/camera")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate.add_argument("--device", default="cuda:0")
    validate.add_argument("--checkpoint", type=Path)

    collect = subparsers.add_parser("collect", help="Run one real-robot collection episode")
    collect.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    collect.add_argument("--device", default="cuda:0")
    collect.add_argument("--steps", type=int, default=300)
    collect.add_argument("--auto-gelsight", action="store_true")
    collect.add_argument("--preview-only", action="store_true")
    collect.add_argument("--enable-real-rl-control", action="store_true")
    collect.add_argument(
        "--save-debug-artifacts",
        action="store_true",
        help=(
            "Also save model RGB, GelSight PNGs, and per-step image NPZ files. "
            "By default collect keeps only lossless raw boundary RGB plus Replay/log metadata."
        ),
    )
    collect.add_argument(
        "--defer-offline-label",
        action="store_true",
        help="Return after artifact writing and run the AprilTag label command later",
    )
    collect.add_argument(
        "--allow-checkpoint-fallback-collect",
        action="store_true",
        help="Explicitly continue base-only collection after checkpoint validation fails",
    )
    mode = collect.add_mutually_exclusive_group(required=True)
    mode.add_argument("--zero-residual", action="store_true")
    mode.add_argument("--warmup-random-residual", action="store_true")
    mode.add_argument("--checkpoint", type=Path)

    train = subparsers.add_parser("train", help="Offline SAC updates from persistent Replay")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--utd-ratio", type=float)
    train.add_argument("--bootstrap-updates", type=int)
    train.add_argument("--checkpoint", type=Path, help="Optional checkpoint to resume")

    label = subparsers.add_parser(
        "label", help="Offline per-frame AprilTag labeling for one collected run"
    )
    label.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    label.add_argument("--run-dir", type=Path, required=True)
    label.add_argument(
        "--reuse-existing-detections",
        action="store_true",
        help=(
            "Recompute rewards/terminals from offline_apriltag_labels.jsonl "
            "without running image detection again"
        ),
    )
    return parser


def _base_config(real_config, device: str, steps: int | None = None, auto_gelsight: bool = False):
    config = load_bundle_config(real_config.base_policy_config)
    config.model.device = device
    config.runner.log_dir = str(real_config.log_dir)
    if steps is not None:
        if steps < 1:
            raise ValueError("--steps must be at least 1")
        config.runner.steps = steps
    if auto_gelsight:
        config.tactile_camera.auto_discover = True
    if config.control_mode != "streaming" or config.streaming.backend != "server9_joint_position":
        raise ValueError("Real-RL requires server9_joint_position streaming")
    if config.model.optimize_for_inference:
        raise ValueError("Real-RL requires model.optimize_for_inference=false")
    if config.model.rma_position_source != "vision" or config.model.rma_contact_source != "vision":
        raise ValueError("Real-RL cannot use RMA observation overrides")
    scales = list(config.action_adapter.scales[:3])
    expected = real_config.residual.action_scale_m
    if scales != [expected, expected, expected]:
        raise ValueError(
            f"Real-RL action scale contract requires XYZ={[expected] * 3}, got {scales}"
        )
    return config


def _settings(args, real_config, *, validation: bool = False) -> RealRLDeploySettings:
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint is not None:
        mode = "checkpoint"
        checkpoint = checkpoint.expanduser().resolve()
    elif getattr(args, "warmup_random_residual", False):
        mode = "warmup_random"
    else:
        mode = "zero"
    settings = RealRLDeploySettings(
        config=real_config,
        mode=mode,
        checkpoint_path=checkpoint,
        stochastic=not validation and not getattr(args, "preview_only", False),
        seed=real_config.seed,
        allow_checkpoint_fallback_collect=getattr(
            args, "allow_checkpoint_fallback_collect", False
        ),
    )
    settings.validate()
    return settings


def validate_command(args: argparse.Namespace) -> int:
    real_config = load_real_rl_config(args.config)
    base = _base_config(real_config, args.device)
    settings = _settings(args, real_config, validation=True)
    load_base_t_camera(real_config.apriltag.calibration_report)
    report = validate_bundle_artifacts(base, real_rl_settings=settings)
    runtime_report = report.get("real_rl")
    if isinstance(runtime_report, dict) and runtime_report.get("episode_start_refused"):
        raise ValueError(
            "Residual checkpoint validation failed: "
            + str(runtime_report.get("checkpoint_fallback"))
        )
    runtime_input = report.get("rma_actor_input", {}).get("real_rl")
    if isinstance(runtime_input, dict) and "state" in runtime_input:
        runtime_input["state_dim"] = len(runtime_input.pop("state"))
    report["real_rl_config"] = str(Path(args.config).resolve())
    report["replay_path"] = str(real_config.replay.path)
    report["log_dir"] = str(real_config.log_dir)
    report["calibration_report"] = str(real_config.apriltag.calibration_report)
    print("Validation: PASS")
    print(json.dumps(report, indent=2))
    return 0


def collect_command(args: argparse.Namespace) -> int:
    if not args.preview_only and not args.enable_real_rl_control:
        raise ValueError(
            "Real motion requires --enable-real-rl-control; run --preview-only first"
        )
    real_config = load_real_rl_config(args.config)
    base = _base_config(real_config, args.device, args.steps, args.auto_gelsight)
    if not args.save_debug_artifacts:
        # Real-RL training consumes the persistent Replay state/action tensors;
        # offline AprilTag only needs the full-resolution raw boundary frames.
        base.camera.save_rgb = False
        base.tactile_camera.save_rgb = False
    settings = _settings(args, real_config)
    base_sha = sha256_file(Path(base.model.model_path).expanduser().resolve())
    expected_contract = real_rl_replay_contract(
        base_model_sha256=base_sha,
        max_residual_m=real_config.residual.max_residual_m,
        action_scale_m=real_config.residual.action_scale_m,
    )
    with ReplayBuffer(real_config.replay.path) as replay:
        existing_contract = replay.get_meta("real_rl_contract")
        if existing_contract is None:
            replay.set_meta("real_rl_contract", expected_contract)
        elif existing_contract != expected_contract:
            raise ValueError(
                f"Replay Buffer contract mismatch: {existing_contract!r}"
            )
    print("Real-RL collect configuration:")
    print(json.dumps({
        "mode": settings.mode,
        "checkpoint": None if settings.checkpoint_path is None else str(settings.checkpoint_path),
        "max_residual_m": real_config.residual.max_residual_m,
        "warmup_std_m": real_config.residual.warmup_std_m,
        "warmup_cap_m": real_config.residual.warmup_cap_m,
        "warmup_correlation": real_config.residual.warmup_correlation,
        "allow_checkpoint_fallback_collect": settings.allow_checkpoint_fallback_collect,
        "replay": str(real_config.replay.path),
        "log_dir": str(real_config.log_dir),
        "steps": base.runner.steps,
        "preview_only": args.preview_only,
        "artifact_profile": (
            "debug_full" if args.save_debug_artifacts else "compact_lossless_raw"
        ),
        "offline_label_deferred": args.defer_offline_label,
    }, indent=2))

    def confirm(_step, action, observation) -> bool:
        print(json.dumps({
            "proposed_robot_delta_m": [action.dx, action.dy, action.dz],
            "gripper_width_m": action.gripper_width,
            "current_tcp_m": observation.tcp_translation,
        }, indent=2))
        answer = input("Type y/yes to start this Real-RL streaming episode: ")
        return answer.strip().lower() in {"y", "yes"}

    summary = run_bundle_deploy(
        base,
        execute_motion=not args.preview_only,
        confirm_step_callback=None if args.preview_only else confirm,
        save_step_data=args.save_debug_artifacts,
        real_rl_settings=settings,
    )
    print(f"Run saved to: {summary['run_dir']}")
    print(json.dumps(summary.get("timing", {}).get("real_rl_collection", {}), indent=2))
    if args.defer_offline_label:
        print("Offline AprilTag labeling deferred. Run:")
        print(
            f"  .venv/bin/python scripts/real_rl/run_residual_sac.py label "
            f"--config {args.config} --run-dir {summary['run_dir']}"
        )
        return 0
    offline_report = label_episode_from_run(summary["run_dir"], real_config)
    print("Offline AprilTag labeling:")
    print(json.dumps(offline_report, indent=2))
    return 0


def label_command(args: argparse.Namespace) -> int:
    real_config = load_real_rl_config(args.config)
    report = label_episode_from_run(
        args.run_dir,
        real_config,
        reuse_saved_detections=args.reuse_existing_detections,
    )
    print(json.dumps(report, indent=2))
    return 0


def train_command(args: argparse.Namespace) -> int:
    real_config = load_real_rl_config(args.config)
    base = _base_config(real_config, args.device)
    base_sha = sha256_file(Path(base.model.model_path).expanduser().resolve())
    report = train_from_replay(
        real_config,
        base_model_sha256=base_sha,
        device=args.device,
        utd_ratio=args.utd_ratio,
        bootstrap_updates=args.bootstrap_updates,
        checkpoint_path=(None if args.checkpoint is None else args.checkpoint.expanduser().resolve()),
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            return validate_command(args)
        if args.command == "collect":
            return collect_command(args)
        if args.command == "label":
            return label_command(args)
        return train_command(args)
    except KeyboardInterrupt:
        print("\nStopped by operator; no further robot commands will be sent.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Real-RL {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
