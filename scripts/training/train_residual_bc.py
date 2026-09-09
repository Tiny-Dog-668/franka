#!/usr/bin/env python3
"""Train a frozen-feature XYZ Residual BC head from Franka HIL runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_sim2real.residual_bc import (  # noqa: E402
    EpisodeSplit,
    FrozenActorFeatureExtractor,
    ResidualMLPConfig,
    checkpoint_payload,
    discover_run_dirs,
    evaluate_model,
    extract_feature_matrix,
    group_counts,
    scan_accepted_samples,
    seed_everything,
    sha256_file,
    split_episodes,
    train_residual_model,
    write_json,
    atomic_torch_save,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an independent XYZ Residual BC MLP from accepted HIL step_data. "
            "The exported base policy remains frozen."
        )
    )
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--run-pattern",
        default="*_e2e_bundle_real_exported_0814_gelsight",
        help="Glob below --runs-root; ignored when --run-dir is supplied.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help="Explicit run directory; repeat to select multiple episodes.",
    )
    parser.add_argument(
        "--validation-run",
        action="append",
        default=[],
        help="Episode basename/path assigned to validation; repeatable.",
    )
    parser.add_argument(
        "--test-run",
        action="append",
        default=[],
        help="Episode basename/path assigned to test; repeatable.",
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=positive_int, default=100)
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument(
        "--feature-batch-size",
        type=positive_int,
        default=32,
        help=(
            "Number of NPZ samples loaded together. Frozen policy inference still "
            "uses batch size one to match deployment numerics."
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=positive_float, default=3e-4)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=1e-4)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--output-scale", type=positive_float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-balanced-sampling",
        action="store_true",
        help="Disable inverse-frequency balancing of normal/hold/moving samples.",
    )
    parser.add_argument(
        "--base-action-tolerance",
        type=positive_float,
        default=5e-4,
        help="Reject data if the frozen model cannot reproduce logged base_action.",
    )
    parser.add_argument(
        "--max-samples-per-run",
        type=positive_int,
        default=None,
        help="Debug/smoke-test limit applied after accepted-action filtering.",
    )
    return parser


def _validate_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {value}")
    if device.type == "cuda":
        torch.empty(1, device=device)
    return device


def _prepare_output_dir(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _episode_names(paths: Sequence[Path]) -> list[str]:
    return [path.name for path in paths]


def _print_split(split: EpisodeSplit) -> None:
    print("Episode split:")
    print("  train:      " + ", ".join(_episode_names(split.train)))
    print("  validation: " + ", ".join(_episode_names(split.validation)))
    print("  test:       " + (", ".join(_episode_names(split.test)) or "<none>"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    device = _validate_device(args.device)
    if not args.base_model.is_file():
        raise ValueError(f"Base model does not exist: {args.base_model}")
    _prepare_output_dir(args.output_dir)
    seed_everything(args.seed)

    run_dirs = discover_run_dirs(args.runs_root, args.run_pattern, args.run_dir)
    split = split_episodes(run_dirs, args.validation_run, args.test_run)
    _print_split(split)

    samples_by_split = {
        "train": scan_accepted_samples(
            split.train, max_samples_per_run=args.max_samples_per_run
        ),
        "validation": scan_accepted_samples(
            split.validation, max_samples_per_run=args.max_samples_per_run
        ),
    }
    if split.test:
        samples_by_split["test"] = scan_accepted_samples(
            split.test, max_samples_per_run=args.max_samples_per_run
        )
    for name, samples in samples_by_split.items():
        print(f"{name}: {len(samples)} accepted samples {group_counts(samples)}")

    print(f"Loading frozen base model on {device}: {args.base_model}")
    scripted_model = torch.jit.load(str(args.base_model), map_location=device)
    extractor = FrozenActorFeatureExtractor(scripted_model, expected_dim=1043)
    matrices = {}
    for name, samples in samples_by_split.items():
        print(f"Extracting frozen actor features for {name}...")
        matrices[name] = extract_feature_matrix(
            samples,
            extractor,
            device=device,
            batch_size=args.feature_batch_size,
            workers=args.workers,
            base_action_tolerance=args.base_action_tolerance,
        )
        print(
            f"  {name}: feature_shape={tuple(matrices[name].features.shape)}, "
            f"max_base_action_error={matrices[name].max_base_action_error:.3g}"
        )
    del extractor, scripted_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model_config = ResidualMLPConfig(output_scale=args.output_scale)
    base_sha256 = sha256_file(args.base_model)
    best_checkpoint_path = args.output_dir / "residual_bc_best.pt"

    def save_best(model, epoch, validation_loss, validation_metrics) -> None:
        atomic_torch_save(
            checkpoint_payload(
                model,
                base_model_path=args.base_model,
                base_model_sha256=base_sha256,
                split=split,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_metrics=validation_metrics,
            ),
            best_checkpoint_path,
        )

    print(f"Training Residual MLP on {device}...")
    model, history, best_epoch = train_residual_model(
        matrices["train"],
        matrices["validation"],
        config=model_config,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        balanced_sampling=not args.no_balanced_sampling,
        seed=args.seed,
        on_best=save_best,
    )
    validation_loss, validation_metrics = evaluate_model(
        model,
        matrices["validation"],
        device=device,
        batch_size=args.batch_size,
    )
    test_loss = None
    test_metrics = None
    if "test" in matrices:
        test_loss, test_metrics = evaluate_model(
            model, matrices["test"], device=device, batch_size=args.batch_size
        )

    model = model.cpu().eval()
    scripted = torch.jit.script(model)
    torch.jit.save(scripted, str(args.output_dir / "residual_bc_best.ts"))
    write_json(args.output_dir / "training_history.json", history)
    metadata = {
        "kind": "franka_hil_residual_bc",
        "format_version": 1,
        "base_model_path": str(args.base_model.resolve()),
        "base_model_sha256": base_sha256,
        "model_config": {
            "actor_feature_dim": model_config.actor_feature_dim,
            "base_action_dim": model_config.base_action_dim,
            "hidden_dims": list(model_config.hidden_dims),
            "output_dim": model_config.output_dim,
            "output_scale": model_config.output_scale,
        },
        "input_contract": "actor_feature_1043_plus_base_action_4",
        "output_contract": "prelimit_normalized_residual_xyz_3",
        "gripper_source": "base_policy",
        "policy_action_filter": "policy_action_accepted == true",
        "train_episodes": _episode_names(split.train),
        "validation_episodes": _episode_names(split.validation),
        "test_episodes": _episode_names(split.test),
        "sample_counts": {
            name: len(matrix) for name, matrix in matrices.items()
        },
        "group_counts": {
            name: group_counts(samples) for name, samples in samples_by_split.items()
        },
        "best_epoch": best_epoch,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "balanced_sampling": not args.no_balanced_sampling,
        "seed": args.seed,
        "max_base_action_error": {
            name: matrix.max_base_action_error for name, matrix in matrices.items()
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"TorchScript head: {args.output_dir / 'residual_bc_best.ts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
