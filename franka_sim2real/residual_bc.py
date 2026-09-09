"""Offline Residual BC training utilities for Franka HIL rollouts.

The base policy is always frozen.  Its exported TorchScript encoders are used
to reproduce the exact feature vector consumed by the deployed action head;
only a small XYZ residual MLP is optimized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, WeightedRandomSampler

from .policy_features import extract_actor_features, validate_actor_feature_contract


GROUP_NORMAL = 0
GROUP_HOLD = 1
GROUP_MOVING = 2
GROUP_NAMES = {
    GROUP_NORMAL: "normal",
    GROUP_HOLD: "intervention_hold",
    GROUP_MOVING: "intervention_moving",
}

REQUIRED_INPUT_KEYS = (
    "wrist_rgb",
    "proprio_obs",
    "action_history",
    "gsmini_left_rgb",
    "gsmini_right_rgb",
    "gsmini_left_reference_rgb",
    "gsmini_right_reference_rgb",
    "base_action",
    "residual_target_xyz",
    "intervention",
    "policy_action_accepted",
)


@dataclass(frozen=True)
class ResidualSample:
    path: Path
    episode_id: str
    step_id: int
    group: int


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    test: tuple[Path, ...]


@dataclass(frozen=True)
class FeatureMatrix:
    features: torch.Tensor
    base_actions: torch.Tensor
    targets: torch.Tensor
    groups: torch.Tensor
    episode_ids: tuple[str, ...]
    step_ids: tuple[int, ...]
    max_base_action_error: float

    def __len__(self) -> int:
        return int(self.targets.shape[0])


@dataclass(frozen=True)
class ResidualMLPConfig:
    actor_feature_dim: int = 1043
    base_action_dim: int = 4
    hidden_dims: tuple[int, ...] = (256, 128)
    output_dim: int = 3
    output_scale: float = 1.2


class ResidualMLP(nn.Module):
    """Predict a bounded pre-limit normalized XYZ residual."""

    def __init__(self, config: ResidualMLPConfig = ResidualMLPConfig()) -> None:
        super().__init__()
        if config.actor_feature_dim <= 0 or config.base_action_dim <= 0:
            raise ValueError("Residual MLP input dimensions must be positive")
        if not config.hidden_dims or any(value <= 0 for value in config.hidden_dims):
            raise ValueError("Residual MLP hidden dimensions must be positive")
        if config.output_dim != 3:
            raise ValueError("Residual BC v1 must output exactly XYZ (3 values)")
        if not math.isfinite(config.output_scale) or config.output_scale <= 0.0:
            raise ValueError("Residual output scale must be finite and positive")
        self.config = config
        dimensions = [
            config.actor_feature_dim + config.base_action_dim,
            *config.hidden_dims,
            config.output_dim,
        ]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)
        self.output_scale = float(config.output_scale)

    def forward(self, actor_features: torch.Tensor, base_action: torch.Tensor) -> torch.Tensor:
        if actor_features.shape[:-1] != base_action.shape[:-1]:
            raise ValueError("Actor features and base action batch shapes must match")
        value = torch.cat((actor_features, base_action), dim=-1)
        return torch.tanh(self.network(value)) * self.output_scale


class FrozenActorFeatureExtractor(nn.Module):
    """Reproduce the 0814 GelSight Student's action-head input features."""

    def __init__(self, model: torch.jit.ScriptModule, expected_dim: int = 1043) -> None:
        super().__init__()
        validate_actor_feature_contract(model, expected_dim)
        self.model = model.eval()
        self.expected_dim = int(expected_dim)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return extract_actor_features(
            self.model,
            wrist_rgb,
            proprio_obs,
            action_history,
            left_current,
            right_current,
            left_reference,
            right_reference,
            expected_dim=self.expected_dim,
        )


class HILStepDataset(Dataset[dict[str, Any]]):
    def __init__(self, samples: Sequence[ResidualSample]) -> None:
        self.samples = tuple(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample.path, allow_pickle=False) as payload:
            return {
                "wrist_rgb": np.asarray(payload["wrist_rgb"], dtype=np.uint8),
                "proprio_obs": np.asarray(payload["proprio_obs"], dtype=np.float32),
                "action_history": np.asarray(payload["action_history"], dtype=np.float32),
                "left_current": np.asarray(payload["gsmini_left_rgb"], dtype=np.uint8),
                "right_current": np.asarray(payload["gsmini_right_rgb"], dtype=np.uint8),
                "left_reference": np.asarray(
                    payload["gsmini_left_reference_rgb"], dtype=np.uint8
                ),
                "right_reference": np.asarray(
                    payload["gsmini_right_reference_rgb"], dtype=np.uint8
                ),
                "base_action": np.asarray(payload["base_action"], dtype=np.float32),
                "target": np.asarray(payload["residual_target_xyz"], dtype=np.float32),
                "group": np.int64(sample.group),
                "sample_index": np.int64(index),
            }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_run_dirs(
    runs_root: Path,
    pattern: str,
    explicit_run_dirs: Sequence[Path] = (),
) -> tuple[Path, ...]:
    candidates = list(explicit_run_dirs) if explicit_run_dirs else list(runs_root.glob(pattern))
    usable: list[Path] = []
    seen: set[Path] = set()
    for candidate in sorted(candidates, key=lambda path: path.name):
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not candidate.is_dir():
            raise ValueError(f"Run directory does not exist: {candidate}")
        if any((candidate / "step_data").glob("*.npz")):
            usable.append(candidate)
    if not usable:
        raise ValueError(
            f"No run directories with step_data/*.npz matched {runs_root / pattern}"
        )
    return tuple(usable)


def _select_named_runs(run_dirs: Sequence[Path], names: Sequence[str]) -> set[Path]:
    selected: set[Path] = set()
    by_name = {path.name: path for path in run_dirs}
    by_resolved = {path.resolve(): path for path in run_dirs}
    for value in names:
        candidate = Path(value)
        match = by_name.get(value) or by_name.get(candidate.name)
        if match is None and candidate.exists():
            match = by_resolved.get(candidate.resolve())
        if match is None:
            raise ValueError(f"Requested split episode is not in the discovered data: {value}")
        selected.add(match)
    return selected


def split_episodes(
    run_dirs: Sequence[Path],
    validation_runs: Sequence[str] = (),
    test_runs: Sequence[str] = (),
) -> EpisodeSplit:
    ordered = tuple(sorted(run_dirs, key=lambda path: path.name))
    if validation_runs or test_runs:
        validation = _select_named_runs(ordered, validation_runs)
        test = _select_named_runs(ordered, test_runs)
        overlap = validation & test
        if overlap:
            raise ValueError(
                "Episodes cannot be both validation and test: "
                + ", ".join(sorted(path.name for path in overlap))
            )
        train = tuple(path for path in ordered if path not in validation and path not in test)
        if not train or not validation:
            raise ValueError("Explicit split requires at least one train and validation episode")
        return EpisodeSplit(
            train=train,
            validation=tuple(path for path in ordered if path in validation),
            test=tuple(path for path in ordered if path in test),
        )
    if len(ordered) < 2:
        raise ValueError("At least two episodes are required for an episode-level split")
    if len(ordered) == 2:
        return EpisodeSplit(train=ordered[:1], validation=ordered[1:], test=())
    return EpisodeSplit(train=ordered[:-2], validation=ordered[-2:-1], test=ordered[-1:])


def _scalar_bool(payload: np.lib.npyio.NpzFile, key: str, path: Path) -> bool:
    value = np.asarray(payload[key])
    if value.shape != ():
        raise ValueError(f"{path}: {key} must be a scalar, got {value.shape}")
    return bool(value.item())


def scan_accepted_samples(
    run_dirs: Sequence[Path],
    *,
    max_samples_per_run: int | None = None,
) -> tuple[ResidualSample, ...]:
    if max_samples_per_run is not None and max_samples_per_run <= 0:
        raise ValueError("max_samples_per_run must be positive")
    samples: list[ResidualSample] = []
    for run_dir in run_dirs:
        accepted_in_run = 0
        for path in sorted((run_dir / "step_data").glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                missing = [key for key in REQUIRED_INPUT_KEYS if key not in payload.files]
                if missing:
                    raise ValueError(f"{path}: missing required keys {missing}")
                if not _scalar_bool(payload, "policy_action_accepted", path):
                    continue
                intervention = _scalar_bool(payload, "intervention", path)
                base_action = np.asarray(payload["base_action"], dtype=np.float32)
                target = np.asarray(payload["residual_target_xyz"], dtype=np.float32)
                if base_action.shape != (4,) or target.shape != (3,):
                    raise ValueError(
                        f"{path}: expected base_action (4,) and target (3,), got "
                        f"{base_action.shape} and {target.shape}"
                    )
                if not np.all(np.isfinite(base_action)) or not np.all(np.isfinite(target)):
                    raise ValueError(f"{path}: base action and residual target must be finite")
                if intervention:
                    if "human_action" not in payload.files:
                        raise ValueError(f"{path}: intervention sample is missing human_action")
                    human_action = np.asarray(payload["human_action"], dtype=np.float32)
                    if human_action.shape != (4,) or not np.all(np.isfinite(human_action)):
                        raise ValueError(f"{path}: human_action must contain four finite values")
                    expected = human_action[:3] - base_action[:3]
                    if not np.allclose(target, expected, rtol=0.0, atol=1e-6):
                        raise ValueError(f"{path}: residual target does not match human-base")
                    group = (
                        GROUP_HOLD
                        if float(np.linalg.norm(human_action[:3])) <= 1e-7
                        else GROUP_MOVING
                    )
                else:
                    if not np.allclose(target, 0.0, rtol=0.0, atol=1e-7):
                        raise ValueError(f"{path}: non-intervention target must be zero")
                    group = GROUP_NORMAL
                episode_id = (
                    str(np.asarray(payload["episode_id"]).item())
                    if "episode_id" in payload.files
                    else run_dir.name
                )
                step_id = (
                    int(np.asarray(payload["step_id"]).item())
                    if "step_id" in payload.files
                    else int(path.stem.rsplit("_", 1)[-1])
                )
            samples.append(
                ResidualSample(
                    path=path,
                    episode_id=episode_id,
                    step_id=step_id,
                    group=group,
                )
            )
            accepted_in_run += 1
            if max_samples_per_run is not None and accepted_in_run >= max_samples_per_run:
                break
    if not samples:
        raise ValueError("The selected episodes contain no accepted HIL samples")
    return tuple(samples)


def extract_feature_matrix(
    samples: Sequence[ResidualSample],
    extractor: FrozenActorFeatureExtractor,
    *,
    device: torch.device,
    batch_size: int,
    workers: int = 0,
    base_action_tolerance: float = 5e-4,
) -> FeatureMatrix:
    if batch_size <= 0 or workers < 0:
        raise ValueError("Feature batch size must be positive and workers non-negative")
    if base_action_tolerance <= 0.0 or not math.isfinite(base_action_tolerance):
        raise ValueError("Base action tolerance must be finite and positive")
    loader = DataLoader(
        HILStepDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    feature_parts: list[torch.Tensor] = []
    base_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    group_parts: list[torch.Tensor] = []
    max_error = 0.0
    extractor.to(device).eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: batch[key].to(device, non_blocking=True)
                for key in (
                    "wrist_rgb",
                    "proprio_obs",
                    "action_history",
                    "left_current",
                    "right_current",
                    "left_reference",
                    "right_reference",
                )
            }
            # Deployment invokes the policy one timestep at a time. CUDA CNN
            # kernels can produce action differences around 2e-3 when the same
            # images are evaluated in a larger batch. Keep batched NPZ loading,
            # but reproduce the deployed batch-size-one feature distribution.
            features_in_batch: list[torch.Tensor] = []
            recomputed_in_batch: list[torch.Tensor] = []
            for row in range(int(tensors["wrist_rgb"].shape[0])):
                row_slice = slice(row, row + 1)
                row_features, row_recomputed = extractor(
                    tensors["wrist_rgb"][row_slice],
                    tensors["proprio_obs"][row_slice],
                    tensors["action_history"][row_slice],
                    tensors["left_current"][row_slice],
                    tensors["right_current"][row_slice],
                    tensors["left_reference"][row_slice],
                    tensors["right_reference"][row_slice],
                )
                features_in_batch.append(row_features)
                recomputed_in_batch.append(row_recomputed)
            features = torch.cat(features_in_batch, dim=0)
            recomputed = torch.cat(recomputed_in_batch, dim=0)
            base_action = batch["base_action"].to(device, non_blocking=True)
            error = float(torch.max(torch.abs(recomputed - base_action)).item())
            max_error = max(max_error, error)
            feature_parts.append(features.detach().cpu())
            base_parts.append(base_action.detach().cpu())
            target_parts.append(batch["target"].detach().cpu())
            group_parts.append(batch["group"].detach().cpu())
    if max_error > base_action_tolerance:
        raise ValueError(
            "Frozen base model does not reproduce the logged base_action: "
            f"max_abs_error={max_error:.6g}, tolerance={base_action_tolerance:.6g}. "
            "Use the exact policy checkpoint that collected these episodes."
        )
    return FeatureMatrix(
        features=torch.cat(feature_parts, dim=0).float().contiguous(),
        base_actions=torch.cat(base_parts, dim=0).float().contiguous(),
        targets=torch.cat(target_parts, dim=0).float().contiguous(),
        groups=torch.cat(group_parts, dim=0).long().contiguous(),
        episode_ids=tuple(sample.episode_id for sample in samples),
        step_ids=tuple(sample.step_id for sample in samples),
        max_base_action_error=max_error,
    )


def balanced_sample_weights(groups: torch.Tensor) -> torch.Tensor:
    values = groups.detach().cpu().long().reshape(-1)
    if values.numel() == 0:
        raise ValueError("Cannot balance an empty training set")
    counts = torch.bincount(values, minlength=len(GROUP_NAMES)).float()
    present = counts > 0
    weights_per_group = torch.zeros_like(counts)
    weights_per_group[present] = 1.0 / counts[present]
    return weights_per_group[values].double()


def make_training_loader(
    matrix: FeatureMatrix,
    *,
    batch_size: int,
    balanced: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        matrix.features,
        matrix.base_actions,
        matrix.targets,
        matrix.groups,
    )
    generator = torch.Generator().manual_seed(seed)
    if balanced:
        sampler = WeightedRandomSampler(
            balanced_sample_weights(matrix.groups),
            num_samples=len(matrix),
            replacement=True,
            generator=generator,
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def regression_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
) -> dict[str, Any]:
    prediction = prediction.detach().cpu().float()
    target = target.detach().cpu().float()
    groups = groups.detach().cpu().long()
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("Prediction and target must both have shape [N,3]")
    error = prediction - target

    def summarize(mask: torch.Tensor) -> dict[str, Any]:
        selected = error[mask]
        if selected.numel() == 0:
            return {"count": 0, "mae": None, "rmse": None, "mae_xyz": None}
        return {
            "count": int(selected.shape[0]),
            "mae": float(selected.abs().mean().item()),
            "rmse": float(torch.sqrt(torch.mean(selected.square())).item()),
            "mae_xyz": [float(value) for value in selected.abs().mean(dim=0).tolist()],
        }

    result: dict[str, Any] = {"overall": summarize(torch.ones(len(error), dtype=torch.bool))}
    for group, name in GROUP_NAMES.items():
        result[name] = summarize(groups == group)
    normal_mask = groups == GROUP_NORMAL
    result["normal_prediction_norm_mean"] = (
        float(torch.linalg.vector_norm(prediction[normal_mask], dim=-1).mean().item())
        if torch.any(normal_mask)
        else None
    )
    return result


def evaluate_model(
    model: ResidualMLP,
    matrix: FeatureMatrix,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    loader = DataLoader(
        TensorDataset(matrix.features, matrix.base_actions, matrix.targets, matrix.groups),
        batch_size=batch_size,
        shuffle=False,
    )
    criterion = nn.SmoothL1Loss(reduction="sum")
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    groups: list[torch.Tensor] = []
    total_loss = 0.0
    model.eval()
    with torch.inference_mode():
        for features, base_action, target, group in loader:
            prediction = model(features.to(device), base_action.to(device))
            total_loss += float(criterion(prediction, target.to(device)).item())
            predictions.append(prediction.cpu())
            targets.append(target)
            groups.append(group)
    target_tensor = torch.cat(targets)
    loss = total_loss / float(target_tensor.numel())
    return loss, regression_metrics(
        torch.cat(predictions), target_tensor, torch.cat(groups)
    )


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    model: ResidualMLP,
    *,
    base_model_path: Path,
    base_model_sha256: str,
    split: EpisodeSplit,
    epoch: int,
    validation_loss: float,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "franka_hil_residual_bc",
        "format_version": 1,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": asdict(model.config),
        "input_contract": "actor_feature_1043_plus_base_action_4",
        "output_contract": "prelimit_normalized_residual_xyz_3",
        "gripper_source": "base_policy",
        "base_model_path": str(base_model_path.resolve()),
        "base_model_sha256": base_model_sha256,
        "train_episodes": [path.name for path in split.train],
        "validation_episodes": [path.name for path in split.validation],
        "test_episodes": [path.name for path in split.test],
        "epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "validation_metrics": validation_metrics,
    }


def train_residual_model(
    train: FeatureMatrix,
    validation: FeatureMatrix,
    *,
    config: ResidualMLPConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    balanced_sampling: bool,
    seed: int,
    on_best: Any | None = None,
) -> tuple[ResidualMLP, list[dict[str, Any]], int]:
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError("epochs, batch_size, and patience must be positive")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    model = ResidualMLP(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss()
    loader = make_training_loader(
        train,
        batch_size=batch_size,
        balanced=balanced_sampling,
        seed=seed,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_elements = 0
        for features, base_action, target, _group in loader:
            features = features.to(device)
            base_action = base_action.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features, base_action)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(target.numel())
            train_elements += int(target.numel())
        validation_loss, validation_metrics = evaluate_model(
            model, validation, device=device, batch_size=batch_size
        )
        record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / float(train_elements),
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
        }
        history.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            if on_best is not None:
                on_best(model, epoch, validation_loss, validation_metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Residual BC training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def group_counts(samples: Iterable[ResidualSample]) -> dict[str, int]:
    counts = {name: 0 for name in GROUP_NAMES.values()}
    for sample in samples:
        counts[GROUP_NAMES[sample.group]] += 1
    return counts


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
