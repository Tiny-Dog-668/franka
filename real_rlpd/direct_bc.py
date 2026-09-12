"""Frozen-encoder direct-action behavior cloning for the 0912 policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


FEATURE_DIM = 1043
ACTION_DIM = 4
ACTION_LIMIT = 0.1
ADAPTER_ID = "gelsight_reference_progress_three_frame_v1"
BASE_KIND = "tacex_rma_gelsight_x040_dr_three_frame_student_torchscript"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DirectBCData:
    features: np.ndarray
    actions: np.ndarray
    episode_ids: tuple[str, ...]
    success: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class EpisodeSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class DirectBCConfig:
    feature_dim: int = FEATURE_DIM
    hidden_dims: tuple[int, ...] = (256, 256)
    action_dim: int = ACTION_DIM
    action_limit: float = ACTION_LIMIT
    normalizer_clip: float = 10.0

    def validate(self) -> None:
        if self.feature_dim != FEATURE_DIM or self.action_dim != ACTION_DIM:
            raise ValueError("0912 Direct BC requires feature[1043] and action[4]")
        if not self.hidden_dims or any(value <= 0 for value in self.hidden_dims):
            raise ValueError("Direct BC hidden dimensions must be positive")
        if not math.isclose(self.action_limit, ACTION_LIMIT, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("0912 Direct BC action_limit must remain 0.1")
        if not math.isfinite(self.normalizer_clip) or self.normalizer_clip <= 0.0:
            raise ValueError("Direct BC normalizer_clip must be finite and positive")


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    std: np.ndarray
    clip: float

    @classmethod
    def fit(cls, features: np.ndarray, clip: float) -> "FeatureNormalizer":
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FEATURE_DIM:
            raise ValueError(f"Direct BC features must be [N,{FEATURE_DIM}]")
        if not np.all(np.isfinite(values)):
            raise ValueError("Direct BC features contain NaN or Inf")
        return cls(
            mean=values.mean(0).astype(np.float32),
            std=np.maximum(values.std(0), 1e-6).astype(np.float32),
            clip=float(clip),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        return np.clip(
            (np.asarray(features, dtype=np.float32) - self.mean) / self.std,
            -self.clip,
            self.clip,
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"mean": self.mean, "std": self.std, "clip": self.clip}


class DirectBCHead(nn.Module):
    """Predict unit direct actions; the artifact applies the fixed 0.1 limit."""

    def __init__(self, config: DirectBCConfig = DirectBCConfig()) -> None:
        super().__init__()
        config.validate()
        dimensions = [config.feature_dim, *config.hidden_dims, config.action_dim]
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_dim, output_dim), nn.SiLU()))
        output = nn.Linear(dimensions[-2], dimensions[-1])
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.network = nn.Sequential(*layers)

    def forward(self, normalized_features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.network(normalized_features))


class DirectBCArtifact(nn.Module):
    """Deployable feature-to-commissioning-limited direct action head."""

    def __init__(
        self,
        head: DirectBCHead,
        normalizer: FeatureNormalizer,
        action_limit: float = ACTION_LIMIT,
    ) -> None:
        super().__init__()
        self.head = head
        self.register_buffer("feature_mean", torch.as_tensor(normalizer.mean))
        self.register_buffer("feature_std", torch.as_tensor(normalizer.std))
        self.normalizer_clip = float(normalizer.clip)
        self.action_limit = float(action_limit)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = torch.clamp(
            (features - self.feature_mean) / self.feature_std,
            -self.normalizer_clip,
            self.normalizer_clip,
        )
        return self.head(normalized) * self.action_limit


class FrozenEncoderDirectBCPolicy(nn.Module):
    """Full policy: frozen 0912 encoders plus a direct-action BC head."""

    def __init__(
        self,
        base_model: torch.jit.ScriptModule,
        artifact: DirectBCArtifact,
    ) -> None:
        super().__init__()
        self.base_model = base_model.eval()
        self.artifact = artifact.eval()
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.artifact.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        wrist_rgb_history: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        left_current: torch.Tensor,
        right_current: torch.Tensor,
        left_reference: torch.Tensor,
        right_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual = self.base_model.encode_visual_features(wrist_rgb_history)
        left, right, contact_logits = self.base_model.tactile_encoder(
            left_current,
            right_current,
            left_reference,
            right_reference,
        )
        features = torch.cat(
            (
                visual,
                left,
                right,
                self.base_model.normalizer.normalize_proprio(proprio_obs),
                self.base_model.normalizer.normalize_history(action_history),
            ),
            dim=-1,
        )
        action = self.artifact(features)
        cube_position_root = self.base_model.normalizer.denormalize_position(
            self.base_model.position_head(visual)
        )
        return action, torch.sigmoid(contact_logits), cube_position_root


def _read_meta(connection: sqlite3.Connection, key: str) -> Any | None:
    row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def load_direct_bc_data(
    replay_path: str | Path,
    *,
    base_model_path: str | Path,
) -> tuple[DirectBCData, dict[str, Any]]:
    path = Path(replay_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        contract = _read_meta(connection, "contract")
        if not isinstance(contract, dict):
            raise ValueError("Direct BC replay is missing its policy contract")
        expected = {
            "adapter_id": ADAPTER_ID,
            "policy_kind": BASE_KIND,
            "feature_dim": FEATURE_DIM,
            "action_dim": ACTION_DIM,
            "model_sha256": sha256_file(base_model_path),
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(
                    f"Direct BC replay contract {key} mismatch: "
                    f"expected {value!r}, got {contract.get(key)!r}"
                )
        rows = connection.execute(
            """SELECT episode_id,state,executed_action,success
            FROM transitions WHERE trainable=1 ORDER BY id"""
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError("Direct BC replay has no trainable transitions")
    features = np.stack([
        np.frombuffer(row["state"], dtype="<f4", count=FEATURE_DIM).copy()
        for row in rows
    ])
    actions = np.stack([
        np.frombuffer(row["executed_action"], dtype="<f4").copy()
        for row in rows
    ])
    if features.shape != (len(rows), FEATURE_DIM):
        raise RuntimeError(f"Direct BC feature matrix has invalid shape {features.shape}")
    if actions.shape != (len(rows), ACTION_DIM):
        raise RuntimeError(f"Direct BC action matrix has invalid shape {actions.shape}")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(actions)):
        raise ValueError("Direct BC replay contains NaN or Inf")
    if np.any(np.abs(actions) > ACTION_LIMIT + 1e-5):
        raise ValueError("Direct BC expert actions exceed the 0.1 commissioning limit")
    return DirectBCData(
        features=features.astype(np.float32, copy=False),
        actions=actions.astype(np.float32, copy=False),
        episode_ids=tuple(str(row["episode_id"]) for row in rows),
        success=np.asarray([bool(row["success"]) for row in rows]),
    ), contract


def split_episodes(data: DirectBCData, seed: int) -> EpisodeSplit:
    per_episode_success: dict[str, bool] = {}
    for episode_id, success in zip(data.episode_ids, data.success):
        per_episode_success[episode_id] = per_episode_success.get(episode_id, False) or bool(success)
    successful = sorted(name for name, value in per_episode_success.items() if value)
    unsuccessful = sorted(name for name, value in per_episode_success.items() if not value)
    if len(successful) < 3 or len(unsuccessful) < 3:
        raise ValueError(
            "Direct BC requires at least three successful and three unsuccessful episodes "
            "for an episode-level train/validation/test split"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(successful)
    rng.shuffle(unsuccessful)
    validation = (successful[0], unsuccessful[0])
    test = (successful[1], unsuccessful[1])
    held_out = set((*validation, *test))
    train = tuple(sorted(name for name in per_episode_success if name not in held_out))
    return EpisodeSplit(train=train, validation=validation, test=test)


def select_episodes(data: DirectBCData, episode_ids: tuple[str, ...]) -> DirectBCData:
    selected = set(episode_ids)
    mask = np.asarray([name in selected for name in data.episode_ids], dtype=bool)
    if not np.any(mask):
        raise ValueError("Direct BC episode split selected no transitions")
    return DirectBCData(
        features=data.features[mask],
        actions=data.actions[mask],
        episode_ids=tuple(name for name, keep in zip(data.episode_ids, mask) if keep),
        success=data.success[mask],
    )


def _action_groups(actions: np.ndarray) -> np.ndarray:
    moving_xyz = np.linalg.norm(actions[:, :3], axis=1) > 1e-6
    moving_gripper = np.abs(actions[:, 3]) > 1e-6
    return moving_xyz.astype(np.int64) + 2 * moving_gripper.astype(np.int64)


def action_group_counts(actions: np.ndarray) -> dict[str, int]:
    labels = ("hold", "xyz_only", "gripper_only", "xyz_and_gripper")
    groups = _action_groups(actions)
    return {label: int(np.sum(groups == index)) for index, label in enumerate(labels)}


def _loader(
    features: np.ndarray,
    target_unit: np.ndarray,
    actions: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    balanced: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(target_unit),
    )
    if not balanced:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    groups = _action_groups(actions)
    counts = np.bincount(groups, minlength=4)
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def evaluate(
    head: DirectBCHead,
    data: DirectBCData,
    normalizer: FeatureNormalizer,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    features = torch.from_numpy(normalizer.transform(data.features))
    targets = torch.from_numpy(data.actions / ACTION_LIMIT)
    predictions: list[np.ndarray] = []
    head.eval()
    with torch.inference_mode():
        for (batch,) in DataLoader(TensorDataset(features), batch_size=batch_size):
            predictions.append(head(batch.to(device)).cpu().numpy())
    predicted_unit = np.concatenate(predictions)
    target_unit = targets.numpy()
    error = predicted_unit - target_unit
    predicted_action = predicted_unit * ACTION_LIMIT
    nonzero = np.abs(data.actions) > 1e-6
    sign_accuracy = []
    for axis in range(ACTION_DIM):
        mask = nonzero[:, axis]
        sign_accuracy.append(
            None
            if not np.any(mask)
            else float(np.mean(np.sign(predicted_action[mask, axis]) == np.sign(data.actions[mask, axis])))
        )
    return {
        "smooth_l1_unit": float(np.mean(np.where(
            np.abs(error) < 1.0,
            0.5 * error * error,
            np.abs(error) - 0.5,
        ))),
        "mae_unit_per_axis": np.mean(np.abs(error), axis=0).tolist(),
        "mae_action_per_axis": np.mean(np.abs(predicted_action - data.actions), axis=0).tolist(),
        "target_mean_action": np.mean(data.actions, axis=0).tolist(),
        "predicted_mean_action": np.mean(predicted_action, axis=0).tolist(),
        "nonzero_target_sign_accuracy_per_axis": sign_accuracy,
        "predicted_abs_ge_0_095_fraction": float(np.mean(np.any(np.abs(predicted_action) >= 0.095, axis=1))),
    }


def train_direct_bc(
    train: DirectBCData,
    validation: DirectBCData,
    *,
    config: DirectBCConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    balanced_sampling: bool,
    on_best: Callable[[DirectBCHead, FeatureNormalizer, int, dict[str, Any]], None] | None = None,
) -> tuple[DirectBCHead, FeatureNormalizer, list[dict[str, Any]], int]:
    config.validate()
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("Direct BC epochs, batch_size, and patience must be positive")
    normalizer = FeatureNormalizer.fit(train.features, config.normalizer_clip)
    normalized = normalizer.transform(train.features)
    target_unit = (train.actions / config.action_limit).astype(np.float32)
    loader = _loader(
        normalized,
        target_unit,
        train.actions,
        batch_size=batch_size,
        seed=seed,
        balanced=balanced_sampling,
    )
    head = DirectBCHead(config).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.SmoothL1Loss()
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        head.train()
        loss_sum = 0.0
        elements = 0
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * int(target.numel())
            elements += int(target.numel())
        metrics = evaluate(
            head,
            validation,
            normalizer,
            device=device,
            batch_size=batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / elements,
            "validation_loss": metrics["smooth_l1_unit"],
            "validation_metrics": metrics,
        }
        history.append(record)
        print(
            f"Direct BC epoch {epoch}/{epochs} | train={record['train_loss']:.6f} "
            f"| validation={record['validation_loss']:.6f}",
            flush=True,
        )
        if record["validation_loss"] < best_loss:
            best_loss = float(record["validation_loss"])
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            if on_best is not None:
                on_best(head, normalizer, epoch, metrics)
        else:
            stale += 1
            if stale >= patience:
                print(f"Direct BC early stop at epoch {epoch}; best_epoch={best_epoch}")
                break
    if best_state is None:
        raise RuntimeError("Direct BC training did not produce a checkpoint")
    head.load_state_dict(best_state)
    return head, normalizer, history, best_epoch


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    head: DirectBCHead,
    normalizer: FeatureNormalizer,
    config: DirectBCConfig,
    split: EpisodeSplit,
    replay_contract: dict[str, Any],
    base_model_path: Path,
    epoch: int,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "kind": "franka_0912_frozen_encoder_direct_bc",
        "format_version": 1,
        "model_config": asdict(config),
        "head": head.state_dict(),
        "normalizer": normalizer.to_dict(),
        "split": asdict(split),
        "replay_contract": replay_contract,
        "base_model_path": str(base_model_path.resolve()),
        "base_model_sha256": sha256_file(base_model_path),
        "best_epoch": int(epoch),
        "validation_metrics": validation_metrics,
    }
