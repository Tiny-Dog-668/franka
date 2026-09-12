#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (  # noqa: E402
    load_bundle_config,
    run_bundle_deploy,
    validate_bundle_artifacts,
)
from franka_sim2real.real_rl.apriltag_tracker import load_base_t_camera  # noqa: E402
from real_rlpd.config import RLPDConfig, load_config  # noqa: E402
from real_rlpd.labeler import label_episode  # noqa: E402
from real_rlpd.replay import clone_for_reward_relabel  # noqa: E402
from real_rlpd.runtime import RLPDDeploySettings  # noqa: E402
from real_rlpd.trainer import train  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "real_rlpd_0823.json"
EXPECTED_ACTION_SCALES = [0.05, 0.05, 0.05, 0.01]


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RLPD-style residual RL for server9 Franka streaming"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate policy/RLPD artifacts offline")
    _common_config(validate)
    validate.add_argument("--checkpoint", type=Path)

    streaming_check = commands.add_parser(
        "streaming-check", help="Run the existing zero-motion server9 timing check"
    )
    _common_config(streaming_check)
    streaming_check.add_argument("--enable-streaming-check", action="store_true")

    expert = commands.add_parser(
        "collect-expert", help="Collect one full-takeover 4D expert episode"
    )
    _common_config(expert)
    expert.add_argument("--steps", type=int, default=300)
    expert.add_argument("--auto-gelsight", action="store_true")
    expert.add_argument("--preview-only", action="store_true")
    expert.add_argument("--enable-expert-control", action="store_true")
    expert.add_argument("--save-debug-artifacts", action="store_true")
    expert.add_argument("--defer-offline-label", action="store_true")

    online = commands.add_parser(
        "collect-online", help="Collect one checkpoint-controlled online episode"
    )
    _common_config(online)
    online.add_argument("--checkpoint", type=Path, required=True)
    online.add_argument("--steps", type=int, default=300)
    online.add_argument("--auto-gelsight", action="store_true")
    online.add_argument("--preview-only", action="store_true")
    online.add_argument("--enable-rlpd-control", action="store_true")
    online.add_argument("--stochastic", action="store_true")
    online.add_argument("--enable-stochastic-control", action="store_true")
    online.add_argument("--save-debug-artifacts", action="store_true")
    online.add_argument("--defer-offline-label", action="store_true")

    label = commands.add_parser("label", help="Offline AprilTag label one episode")
    label.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    label.add_argument("--run-dir", type=Path, required=True)
    label.add_argument("--role", choices=("offline", "online"), required=True)

    rebuild = commands.add_parser(
        "rebuild-reward-replay",
        help="Clone a compatible Replay and relabel it under a new reward contract",
    )
    _common_config(rebuild)
    rebuild.add_argument("--source-replay", type=Path, required=True)
    rebuild.add_argument("--role", choices=("offline", "online"), required=True)

    training = commands.add_parser(
        "train", help="Run offline pretraining or between-episode online updates"
    )
    _common_config(training)
    training.add_argument("--checkpoint", type=Path)
    training.add_argument("--groups", type=int)
    training.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print training metrics every N update groups (default: 10)",
    )
    return parser


def _base_config(
    config: RLPDConfig,
    device: str,
    *,
    steps: int | None = None,
    auto_gelsight: bool = False,
    data_role: str = "rollout",
) -> Any:
    base = load_bundle_config(config.base_policy_config)
    base.model.device = device
    if data_role not in {"expert", "rollout"}:
        raise ValueError("RLPD data_role must be expert or rollout")
    base.runner.log_dir = str(
        config.expert_data_dir if data_role == "expert" else config.rollout_data_dir
    )
    base.runner.run_name = f"{data_role}_{base.runner.run_name}"
    if steps is not None:
        if steps < 1:
            raise ValueError("--steps must be at least 1")
        base.runner.steps = int(steps)
    if auto_gelsight:
        base.tactile_camera.auto_discover = True
    if base.control_mode != "streaming" or base.streaming.backend != "server9_joint_position":
        raise ValueError("RLPD requires server9_joint_position streaming")
    if not abs(float(base.streaming.policy_frequency_hz) - 30.0) < 1e-6:
        raise ValueError("RLPD collection contract requires a 30 Hz policy loop")
    if base.model.optimize_for_inference:
        raise ValueError("RLPD feature extraction requires model.optimize_for_inference=false")
    if base.model.rma_position_source != "vision" or base.model.rma_contact_source != "vision":
        raise ValueError("RLPD cannot use RMA oracle observation overrides")
    if list(base.action_adapter.scales) != EXPECTED_ACTION_SCALES:
        raise ValueError(
            f"RLPD action scale contract requires {EXPECTED_ACTION_SCALES}, "
            f"got {list(base.action_adapter.scales)}"
        )
    if list(base.action_adapter.clip_low) != [-1.0] * 4 or list(
        base.action_adapter.clip_high
    ) != [1.0] * 4:
        raise ValueError("RLPD requires the normalized [-1,1] 4D action contract")
    return base


def _settings(
    config: RLPDConfig,
    base: Any,
    *,
    checkpoint: Path | None = None,
    stochastic: bool = False,
    enable_stochastic_control: bool = False,
) -> RLPDDeploySettings:
    settings = RLPDDeploySettings(
        config=config,
        mode="expert_shadow" if checkpoint is None else "checkpoint",
        commissioning_limit=float(base.streaming.commissioning_action_limit),
        checkpoint_path=(
            None if checkpoint is None else checkpoint.expanduser().resolve()
        ),
        stochastic=bool(stochastic),
        enable_stochastic_control=bool(enable_stochastic_control),
    )
    settings.validate()
    return settings


def _validate_and_contract(
    config: RLPDConfig,
    base: Any,
    *,
    checkpoint: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = _settings(config, base, checkpoint=checkpoint)
    report = validate_bundle_artifacts(base, rlpd_settings=settings)
    runtime = report.get("rlpd")
    if not isinstance(runtime, dict):
        raise RuntimeError("Bundle validation did not construct an RLPD runtime")
    if runtime.get("episode_start_refused"):
        raise ValueError("RLPD checkpoint rejected: " + str(runtime.get("checkpoint_error")))
    contract = runtime.get("contract")
    if not isinstance(contract, dict):
        raise RuntimeError("RLPD validation did not return a replay contract")
    return report, contract


def validate_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = _base_config(config, args.device)
    load_base_t_camera(config.apriltag.calibration_report)
    report, _contract = _validate_and_contract(
        config, base, checkpoint=args.checkpoint
    )
    rlpd_input = report.get("rma_actor_input", {}).get("rlpd")
    if isinstance(rlpd_input, dict) and "state" in rlpd_input:
        rlpd_input["state_dim"] = len(rlpd_input.pop("state"))
    report["rlpd_config"] = str(Path(args.config).resolve())
    report["offline_replay"] = str(config.offline_replay_path)
    report["online_replay"] = str(config.online_replay_path)
    report["expert_data_dir"] = str(config.expert_data_dir)
    report["rollout_data_dir"] = str(config.rollout_data_dir)
    print("Validation: PASS")
    print(json.dumps(report, indent=2))
    return 0


def streaming_check_command(args: argparse.Namespace) -> int:
    if not args.enable_streaming_check:
        raise ValueError("Streaming check requires --enable-streaming-check")
    config = load_config(args.config)
    base = _base_config(config, args.device)
    summary = run_bundle_deploy(
        base,
        execute_motion=True,
        streaming_check=True,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _confirm(label: str):
    def confirm(_step: int, action: Any, observation: Any) -> bool:
        print(json.dumps({
            "mode": label,
            "first_proposed_robot_delta_m": [action.dx, action.dy, action.dz],
            "first_gripper_width_m": action.gripper_width,
            "current_tcp_m": observation.tcp_translation,
        }, indent=2))
        answer = input(f"Type y/yes to start this {label} episode: ")
        return answer.strip().lower() in {"y", "yes"}

    return confirm


def _offline_label_skip_reason(summary: dict[str, Any]) -> str | None:
    """Explain why a completed collection has no transition to label."""

    timing = summary.get("timing")
    if isinstance(timing, dict) and bool(timing.get("cancelled_by_user")):
        return "the operator cancelled before the control session started"
    collection = timing.get("rlpd_collection") if isinstance(timing, dict) else None
    transition_count = (
        collection.get("transitions_written")
        if isinstance(collection, dict)
        else None
    )
    if transition_count is not None and int(transition_count) < 1:
        return "the episode contains no RLPD transitions"
    step_count = summary.get("num_steps")
    if step_count is not None and int(step_count) < 1:
        return "the episode contains no recorded policy steps"
    return None


def _collect(args: argparse.Namespace, *, expert: bool) -> int:
    enable = args.enable_expert_control if expert else args.enable_rlpd_control
    if not args.preview_only and not enable:
        flag = "--enable-expert-control" if expert else "--enable-rlpd-control"
        raise ValueError(f"Real motion requires {flag}; run --preview-only first")
    if not expert and args.enable_stochastic_control and not args.stochastic:
        raise ValueError("--enable-stochastic-control requires --stochastic")
    config = load_config(args.config)
    base = _base_config(
        config,
        args.device,
        steps=args.steps,
        auto_gelsight=args.auto_gelsight,
        data_role="expert" if expert else "rollout",
    )
    if (
        not args.preview_only
        and float(base.workspace["minimum"][2]) < 0.01
    ):
        raise ValueError(
            "RLPD real motion requires workspace.minimum.z >= 0.01 m; "
            f"selected base config has {base.workspace['minimum'][2]!r}. "
            "Resolve the base-policy workspace contract before motion."
        )
    if not args.save_debug_artifacts:
        base.camera.save_rgb = False
        base.tactile_camera.save_rgb = False
    settings = _settings(
        config,
        base,
        checkpoint=None if expert else args.checkpoint,
        stochastic=False if expert or args.preview_only else args.stochastic,
        enable_stochastic_control=False if expert else args.enable_stochastic_control,
    )
    _report, contract = _validate_and_contract(
        config,
        base,
        checkpoint=None if expert else args.checkpoint,
    )
    print(json.dumps({
        "mode": "expert_full_takeover" if expert else "rlpd_checkpoint",
        "checkpoint": None if expert else str(args.checkpoint.expanduser().resolve()),
        "stochastic": settings.stochastic,
        "steps": base.runner.steps,
        "preview_only": args.preview_only,
        "replay": str(config.offline_replay_path if expert else config.online_replay_path),
        "run_root": str(
            config.expert_data_dir if expert else config.rollout_data_dir
        ),
        "expert_source_is_policy_independent": bool(expert),
        "contract": contract,
    }, indent=2))
    summary = run_bundle_deploy(
        base,
        execute_motion=not args.preview_only,
        confirm_step_callback=(
            None
            if args.preview_only
            else _confirm("RLPD expert" if expert else "RLPD checkpoint")
        ),
        save_step_data=args.save_debug_artifacts,
        rlpd_settings=settings,
        rlpd_expert=expert,
    )
    print(f"Run saved to: {summary['run_dir']}")
    if args.preview_only:
        return 0
    skip_reason = _offline_label_skip_reason(summary)
    if skip_reason is not None:
        print(f"Offline label skipped: {skip_reason}.")
        return 0
    role = "offline" if expert else "online"
    if args.defer_offline_label:
        print("Offline label deferred. Run:")
        print(
            f"  .venv/bin/python scripts/real_rlpd/run_rlpd.py label "
            f"--config {args.config} --role {role} --run-dir {summary['run_dir']}"
        )
        return 0
    print(json.dumps(label_episode(summary["run_dir"], config, role), indent=2))
    return 0


def label_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(json.dumps(label_episode(args.run_dir, config, args.role), indent=2))
    return 0


def rebuild_reward_replay_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = _base_config(config, args.device)
    _report, contract = _validate_and_contract(config, base)
    target = (
        config.offline_replay_path
        if args.role == "offline"
        else config.online_replay_path
    )
    report = clone_for_reward_relabel(
        args.source_replay,
        target,
        args.role,
        contract,
    )
    labels = []
    failures = []
    skipped = []
    for run_dir in report["episodes"]:
        resolved_run = Path(run_dir)
        if not resolved_run.is_dir() or not any(
            (resolved_run / "raw_rgb").glob("boundary_*.png")
        ):
            skipped.append({
                "run_dir": run_dir,
                "reason": "raw AprilTag boundary frames are unavailable",
            })
            continue
        try:
            labels.append(label_episode(run_dir, config, args.role))
        except Exception as exc:
            failures.append({
                "run_dir": run_dir,
                "error": f"{type(exc).__name__}: {exc}",
            })
    report["labels"] = labels
    report["failures"] = failures
    report["skipped"] = skipped
    report["trainable_transitions"] = sum(
        int(item["trainable_transitions"]) for item in labels
    )
    print(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError(
            f"reward Replay was cloned, but {len(failures)} episode(s) failed relabeling; "
            "inspect the report and rerun those episodes with the label command"
        )
    return 0


def train_command(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    base = _base_config(config, args.device)
    _report, contract = _validate_and_contract(config, base)
    checkpoint = (
        None if args.checkpoint is None else args.checkpoint.expanduser().resolve()
    )
    print(json.dumps(train(
        config,
        contract,
        device=args.device,
        checkpoint_path=checkpoint,
        groups=args.groups,
        progress_interval=args.progress_interval,
    ), indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "validate":
            return validate_command(args)
        if args.command == "streaming-check":
            return streaming_check_command(args)
        if args.command == "collect-expert":
            return _collect(args, expert=True)
        if args.command == "collect-online":
            return _collect(args, expert=False)
        if args.command == "label":
            return label_command(args)
        if args.command == "rebuild-reward-replay":
            return rebuild_reward_replay_command(args)
        return train_command(args)
    except KeyboardInterrupt:
        print("\nStopped by operator; no further robot commands will be sent.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"RLPD {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
