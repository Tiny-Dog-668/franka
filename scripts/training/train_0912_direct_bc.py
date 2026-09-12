#!/usr/bin/env python3
"""Train and export the 0912 frozen-encoder direct-action BC policy."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from real_rlpd.direct_bc import (  # noqa: E402
    ACTION_LIMIT,
    DirectBCArtifact,
    DirectBCConfig,
    FrozenEncoderDirectBCPolicy,
    action_group_counts,
    atomic_torch_save,
    checkpoint_payload,
    evaluate,
    load_direct_bc_data,
    seed_everything,
    select_episodes,
    sha256_file,
    split_episodes,
    train_direct_bc,
)


DEFAULT_REPLAY = REPO_ROOT / "real_rlpd_data/0912/derived/offline_expert_observable_absolute_v1.sqlite3"
DEFAULT_BASE_MODEL = REPO_ROOT / "checkpoint/0912/gelsight_x040_three_frame_student_0100000.pt"
DEFAULT_BASE_METADATA = REPO_ROOT / "checkpoint/0912/gelsight_x040_three_frame_student_0100000.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoint/0912_direct_bc"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train 0912 frozen-encoder four-dimensional Direct BC"
    )
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--base-metadata", type=Path, default=DEFAULT_BASE_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--batch-size", type=_positive_int, default=256)
    parser.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=_positive_int, default=15)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Opt in to inverse-frequency hold/XYZ/gripper action-group sampling.",
    )
    return parser


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"CUDA device requested but CUDA is unavailable: {value}")
    torch.empty(1, device=device)
    return device


def _prepare_output(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and any(resolved.iterdir()):
        raise ValueError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _zeros(device: torch.device, batch: int = 1) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((batch, 3, 224, 224, 3), dtype=torch.uint8, device=device),
        torch.zeros((batch, 15), dtype=torch.float32, device=device),
        torch.zeros((batch, 4), dtype=torch.float32, device=device),
        torch.zeros((batch, 96, 128, 3), dtype=torch.uint8, device=device),
        torch.zeros((batch, 96, 128, 3), dtype=torch.uint8, device=device),
        torch.zeros((batch, 96, 128, 3), dtype=torch.uint8, device=device),
        torch.zeros((batch, 96, 128, 3), dtype=torch.uint8, device=device),
    )


def main() -> int:
    args = build_parser().parse_args()
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be finite and non-negative")
    device = _device(args.device)
    output_dir = _prepare_output(args.output_dir)
    base_model = args.base_model.expanduser().resolve()
    base_metadata_path = args.base_metadata.expanduser().resolve()
    if not base_model.is_file() or not base_metadata_path.is_file():
        raise FileNotFoundError("0912 base model or metadata is missing")
    seed_everything(args.seed)

    data, replay_contract = load_direct_bc_data(
        args.replay,
        base_model_path=base_model,
    )
    split = split_episodes(data, args.seed)
    matrices = {
        "train": select_episodes(data, split.train),
        "validation": select_episodes(data, split.validation),
        "test": select_episodes(data, split.test),
    }
    print(json.dumps({
        "replay": str(args.replay.expanduser().resolve()),
        "transitions": len(data),
        "split_episodes": {
            "train": list(split.train),
            "validation": list(split.validation),
            "test": list(split.test),
        },
        "split_samples": {name: len(value) for name, value in matrices.items()},
        "action_groups": {
            name: action_group_counts(value.actions)
            for name, value in matrices.items()
        },
    }, indent=2))

    config = DirectBCConfig()
    checkpoint_path = output_dir / "direct_bc_best.pt"

    def save_best(head, normalizer, epoch, metrics) -> None:
        atomic_torch_save(
            checkpoint_payload(
                head,
                normalizer,
                config,
                split,
                replay_contract,
                base_model,
                epoch,
                metrics,
            ),
            checkpoint_path,
        )

    head, normalizer, history, best_epoch = train_direct_bc(
        matrices["train"],
        matrices["validation"],
        config=config,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        balanced_sampling=args.balanced_sampling,
        on_best=save_best,
    )
    metrics = {
        name: evaluate(
            head,
            matrix,
            normalizer,
            device=device,
            batch_size=args.batch_size,
        )
        for name, matrix in matrices.items()
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )

    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    base = torch.jit.load(str(base_model), map_location="cpu").eval()
    artifact = DirectBCArtifact(head.cpu().eval(), normalizer, ACTION_LIMIT).eval()
    eager = FrozenEncoderDirectBCPolicy(base, artifact).eval()
    example = _zeros(torch.device("cpu"), 1)
    with torch.inference_mode():
        traced = torch.jit.trace(eager, example, strict=True)
        cpu_errors: dict[str, list[float]] = {}
        for batch in (1, 8):
            values = _zeros(torch.device("cpu"), batch)
            expected = eager(*values)
            actual = traced(*values)
            cpu_errors[f"cpu_batch_{batch}_max_abs_error"] = [
                float(torch.max(torch.abs(left - right)))
                for left, right in zip(expected, actual)
            ]
    model_path = output_dir / "direct_bc_policy.pt"
    torch.jit.save(traced, str(model_path))

    cuda_errors = None
    if torch.cuda.is_available():
        cuda_model = torch.jit.load(str(model_path), map_location="cuda:0").eval()
        cpu_model = torch.jit.load(str(model_path), map_location="cpu").eval()
        values_cpu = _zeros(torch.device("cpu"), 1)
        values_cuda = tuple(value.cuda() for value in values_cpu)
        with torch.inference_mode():
            cpu_output = cpu_model(*values_cpu)
            cuda_output = cuda_model(*values_cuda)
        cuda_errors = [
            float(torch.max(torch.abs(left - right.cpu())))
            for left, right in zip(cpu_output, cuda_output)
        ]

    metadata = copy.deepcopy(base_metadata)
    metadata.update({
        "torchscript_sha256": sha256_file(model_path),
        "deployment_variant": "frozen_encoder_direct_bc",
        "base_policy_model_path": str(base_model),
        "base_policy_model_sha256": sha256_file(base_model),
        "direct_bc": {
            "format_version": 1,
            "input": "frozen_actor_feature_1043",
            "output": "commissioning_limited_direct_action_4",
            "action_limit": ACTION_LIMIT,
            "loss": "smooth_l1_on_expert_executed_action_div_0.1",
            "hidden_dims": list(config.hidden_dims),
            "balanced_action_group_sampling": args.balanced_sampling,
            "replay_path": str(args.replay.expanduser().resolve()),
            "replay_contract": replay_contract,
            "best_epoch": best_epoch,
            "metrics": metrics,
            "split": {
                "train": list(split.train),
                "validation": list(split.validation),
                "test": list(split.test),
            },
        },
        "validation": cpu_errors,
        "cuda_validation": cuda_errors is not None,
        "direct_bc_cuda_max_abs_error": cuda_errors,
    })
    metadata["student_model_contract"]["action_head"] = [1043, 256, 256, 4]
    metadata_path = output_dir / "direct_bc_policy.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "best_epoch": best_epoch,
        "metrics": metrics,
        "cpu_validation": cpu_errors,
        "cuda_validation_max_abs_error": cuda_errors,
        "checkpoint": str(checkpoint_path),
        "torchscript": str(model_path),
        "metadata": str(metadata_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
