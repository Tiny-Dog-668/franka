from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RealRLConfig
from .replay_buffer import ReplayBuffer, real_rl_replay_contract
from .residual_sac import DeterministicActorArtifact, FrozenNormalizer, ResidualSAC


def _torch_load(path: Path, device: str | torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _validate_checkpoint(
    payload: dict[str, Any],
    base_sha: str,
    residual_config: Any | None = None,
) -> None:
    if payload.get("kind") != "franka_real_residual_sac" or payload.get("format_version") != 2:
        raise ValueError("Residual SAC checkpoint kind/version is incompatible")
    if payload.get("base_model_sha256") != base_sha:
        raise ValueError("Residual SAC checkpoint belongs to a different base model")
    if payload.get("input_contract") != (
        "empirical_norm_policy_feature_1043_plus_fixed_base_action_4"
    ):
        raise ValueError("Residual SAC actor input contract is incompatible")
    if payload.get("critic_privileged_contract") != "object_relative_xyz_3_plus_height_1":
        raise ValueError("Residual SAC privileged Critic contract is incompatible")
    if payload.get("output_contract") != "unit_xyz_3_times_configured_max_residual_m":
        raise ValueError("Residual SAC output contract is incompatible")
    if payload.get("gripper_source") != "base_policy":
        raise ValueError("Residual SAC checkpoint must preserve the base gripper")
    if residual_config is not None and payload.get("residual_config") != asdict(residual_config):
        raise ValueError("Residual SAC checkpoint residual mapping differs from config")


def gradient_updates_for(new_trainable_count: int, utd_ratio: float) -> int:
    if new_trainable_count < 0:
        raise ValueError("new_trainable_count must be non-negative")
    if not 1.0 <= float(utd_ratio) <= 4.0:
        raise ValueError("UTD ratio must be in [1,4]")
    return int(math.ceil(new_trainable_count * float(utd_ratio)))


def planned_gradient_updates(
    *,
    bootstrap: bool,
    bootstrap_updates: int,
    new_trainable_count: int,
    utd_ratio: float,
) -> int:
    if bootstrap_updates < 1:
        raise ValueError("bootstrap_updates must be positive")
    return (
        int(bootstrap_updates)
        if bootstrap
        else gradient_updates_for(new_trainable_count, utd_ratio)
    )


def stratified_batch_indices(
    rng: np.random.Generator,
    success: np.ndarray,
    *,
    batch_size: int,
    success_samples_per_batch: int,
) -> np.ndarray:
    """Sample a batch with an exact success quota whenever successes exist."""

    flags = np.asarray(success, dtype=np.bool_).reshape(-1)
    if flags.size < 1:
        raise ValueError("Cannot sample an empty Replay")
    if batch_size < 1 or not 0 <= success_samples_per_batch < batch_size:
        raise ValueError("Invalid stratified Replay batch settings")
    success_pool = np.flatnonzero(flags)
    if success_samples_per_batch == 0 or success_pool.size == 0:
        return rng.integers(0, flags.size, size=batch_size)
    non_success_pool = np.flatnonzero(~flags)
    if non_success_pool.size == 0:
        return rng.choice(success_pool, size=batch_size, replace=True)
    success_indices = rng.choice(
        success_pool,
        size=success_samples_per_batch,
        replace=success_pool.size < success_samples_per_batch,
    )
    non_success_count = batch_size - success_samples_per_batch
    non_success_indices = rng.choice(
        non_success_pool,
        size=non_success_count,
        replace=non_success_pool.size < non_success_count,
    )
    indices = np.concatenate((success_indices, non_success_indices)).astype(
        np.int64, copy=False
    )
    rng.shuffle(indices)
    return indices


def _training_diagnostics(
    learner: ResidualSAC,
    data: dict[str, np.ndarray],
    *,
    max_residual_m: float,
    chunk_size: int = 512,
) -> dict[str, Any]:
    """Summarize Q ordering and deterministic residual magnitudes."""

    count = int(data["state"].shape[0])
    behavior_q_parts: list[np.ndarray] = []
    actor_q_parts: list[np.ndarray] = []
    actor_action_parts: list[np.ndarray] = []
    learner.actor.eval()
    learner.q1.eval()
    learner.q2.eval()
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            state = torch.as_tensor(
                data["state"][start:stop], dtype=torch.float32, device=learner.device
            )
            privileged = torch.as_tensor(
                data["privileged"][start:stop],
                dtype=torch.float32,
                device=learner.device,
            )
            behavior_action = torch.as_tensor(
                data["sac_unit_action"][start:stop],
                dtype=torch.float32,
                device=learner.device,
            )
            state = learner.normalizer.state_tensor(state)
            privileged = learner.normalizer.privileged_tensor(privileged)
            actor_action = learner.actor(state)
            behavior_q = torch.minimum(
                learner.q1(state, privileged, behavior_action),
                learner.q2(state, privileged, behavior_action),
            )
            actor_q = torch.minimum(
                learner.q1(state, privileged, actor_action),
                learner.q2(state, privileged, actor_action),
            )
            behavior_q_parts.append(behavior_q.squeeze(-1).cpu().numpy())
            actor_q_parts.append(actor_q.squeeze(-1).cpu().numpy())
            actor_action_parts.append(actor_action.cpu().numpy())

    behavior_q = np.concatenate(behavior_q_parts)
    actor_q = np.concatenate(actor_q_parts)
    actor_action = np.concatenate(actor_action_parts)
    success = np.asarray(data["success"], dtype=np.bool_)
    non_success = ~success
    abs_action_mm = np.abs(actor_action) * float(max_residual_m) * 1000.0
    diagnostics: dict[str, Any] = {
        "success_transitions": int(success.sum()),
        "success_fraction": float(success.mean()),
        "behavior_q_mean": float(behavior_q.mean()),
        "actor_q_mean": float(actor_q.mean()),
        "actor_abs_residual_p95_mm_xyz": np.percentile(
            abs_action_mm, 95, axis=0
        ).tolist(),
        "actor_abs_residual_max_mm_xyz": abs_action_mm.max(axis=0).tolist(),
        "actor_near_limit_axis_fraction": float((np.abs(actor_action) >= 0.9).mean()),
    }
    if success.any() and non_success.any():
        success_behavior_q = behavior_q[success]
        success_actor_q = actor_q[success]
        diagnostics.update({
            "behavior_q_success_mean": float(success_behavior_q.mean()),
            "behavior_q_non_success_mean": float(behavior_q[non_success].mean()),
            "behavior_q_success_gap": float(
                success_behavior_q.mean() - behavior_q[non_success].mean()
            ),
            "actor_q_success_mean": float(success_actor_q.mean()),
            "actor_q_non_success_mean": float(actor_q[non_success].mean()),
            "actor_q_success_gap": float(
                success_actor_q.mean() - actor_q[non_success].mean()
            ),
            "behavior_q_success_percentile_mean": float(np.mean([
                np.mean(behavior_q[non_success] < value)
                for value in success_behavior_q
            ])),
        })
        random_action_count = 256
        diagnostic_rng = np.random.default_rng(0)
        success_state = torch.as_tensor(
            data["state"][success], dtype=torch.float32, device=learner.device
        )
        success_privileged = torch.as_tensor(
            data["privileged"][success], dtype=torch.float32, device=learner.device
        )
        random_action = torch.as_tensor(
            diagnostic_rng.uniform(
                -1.0,
                1.0,
                size=(int(success.sum()), random_action_count, 3),
            ),
            dtype=torch.float32,
            device=learner.device,
        )
        with torch.no_grad():
            success_state = learner.normalizer.state_tensor(success_state)
            success_privileged = learner.normalizer.privileged_tensor(
                success_privileged
            )
            repeated_state = success_state[:, None, :].expand(
                -1, random_action_count, -1
            ).reshape(-1, success_state.shape[-1])
            repeated_privileged = success_privileged[:, None, :].expand(
                -1, random_action_count, -1
            ).reshape(-1, success_privileged.shape[-1])
            flat_random_action = random_action.reshape(-1, 3)
            random_q = torch.minimum(
                learner.q1(repeated_state, repeated_privileged, flat_random_action),
                learner.q2(repeated_state, repeated_privileged, flat_random_action),
            ).reshape(int(success.sum()), random_action_count).cpu().numpy()
        diagnostics.update({
            "success_state_behavior_minus_actor_q_mean": float(
                (success_behavior_q - success_actor_q).mean()
            ),
            "success_state_behavior_action_q_percentile_vs_uniform": float(
                (random_q < success_behavior_q[:, None]).mean()
            ),
            "success_state_actor_action_q_percentile_vs_uniform": float(
                (random_q < success_actor_q[:, None]).mean()
            ),
        })
    return diagnostics


def train_from_replay(
    config: RealRLConfig,
    *,
    base_model_sha256: str,
    device: str,
    utd_ratio: float | None = None,
    bootstrap_updates: int | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    ratio = config.sac.utd_ratio if utd_ratio is None else float(utd_ratio)
    if not 1.0 <= ratio <= 4.0:
        raise ValueError("UTD ratio must be in [1,4]")
    bootstrap = (
        config.sac.bootstrap_updates
        if bootstrap_updates is None else int(bootstrap_updates)
    )
    if bootstrap < 1:
        raise ValueError("bootstrap_updates must be positive")
    latest = config.checkpoint_dir / "latest.pt"
    resume = checkpoint_path or (latest if latest.is_file() else None)
    with ReplayBuffer(config.replay.path, create=False) as replay:
        frozen_normalizer_meta = replay.get_meta("frozen_normalizer")
        if resume is None and frozen_normalizer_meta is not None:
            raise RuntimeError(
                "Replay already has a frozen normalizer but its learner checkpoint is "
                "unavailable; refusing to refit it. Restore the recorded checkpoint: "
                + str(frozen_normalizer_meta.get("checkpoint"))
            )
        contract = replay.get_meta("real_rl_contract")
        expected_contract = real_rl_replay_contract(
            base_model_sha256=base_model_sha256,
            max_residual_m=config.residual.max_residual_m,
            action_scale_m=config.residual.action_scale_m,
        )
        if contract != expected_contract:
            raise ValueError(
                f"Replay contract does not match training config: {contract!r}"
            )
        replay_shas = replay.base_model_sha256_values()
        if replay_shas and replay_shas != {base_model_sha256}:
            raise ValueError(f"Replay mixes incompatible base models: {sorted(replay_shas)}")
        trainable_count = replay.trainable_count()
        if trainable_count < config.sac.minimum_trainable_replay:
            raise ValueError(
                f"Need at least {config.sac.minimum_trainable_replay} trainable transitions; "
                f"got {trainable_count}"
            )
        data = replay.load_trainable()
        if not data:
            raise ValueError("Replay has no trainable transitions")
        if not np.allclose(
            data["residual_action_m"],
            data["sac_unit_action"] * config.residual.max_residual_m,
            atol=1e-7,
        ) or not np.allclose(
            data["residual_action_normalized"],
            data["residual_action_m"] / config.residual.action_scale_m,
            atol=1e-7,
        ):
            raise ValueError("Replay residual action mapping differs from the configured contract")

        if resume is not None:
            payload = _torch_load(Path(resume), device)
            _validate_checkpoint(payload, base_model_sha256, config.residual)
            normalizer = FrozenNormalizer.from_dict(payload["normalizer"])
            if frozen_normalizer_meta is not None:
                expected_normalizer_meta = {
                    "sample_count": normalizer.sample_count,
                    "min_replay_id": normalizer.min_replay_id,
                    "max_replay_id": normalizer.max_replay_id,
                    "contract_hash": normalizer.contract_hash,
                }
                actual_normalizer_meta = {
                    key: frozen_normalizer_meta.get(key)
                    for key in expected_normalizer_meta
                }
                if actual_normalizer_meta != expected_normalizer_meta:
                    raise ValueError(
                        "Checkpoint normalizer does not match the Replay frozen normalizer"
                    )
            high_watermark = int(payload.get("replay_high_watermark", 0))
            label_revision_watermark = int(
                payload.get("replay_label_revision", 0)
            )
        else:
            count = config.sac.normalizer_samples
            normalizer = FrozenNormalizer.fit(
                data["state"][:count], data["privileged"][:count], data["id"][:count],
                clip=config.sac.normalizer_clip,
            )
            high_watermark = 0
            label_revision_watermark = 0

        changes = replay.trainable_change_counts(
            after_id=high_watermark,
            after_label_revision=label_revision_watermark,
        )
        new_count = changes["total"]
        replay_high_watermark = replay.max_id()
        replay_label_revision = replay.max_label_revision()
        if resume is not None and new_count == 0:
            return {
                "updated": False,
                "trainable_replay": trainable_count,
                "new_trainable_transitions": 0,
                "new_id_trainable_transitions": 0,
                "relabeled_transitions": 0,
                "replay_high_watermark": high_watermark,
                "replay_label_revision": label_revision_watermark,
            }

        learner = ResidualSAC(config.sac, normalizer, device)
        if resume is not None:
            learner.restore(payload)
        schedule = "bootstrap" if resume is None else "utd"
        updates = planned_gradient_updates(
            bootstrap=resume is None,
            bootstrap_updates=bootstrap,
            new_trainable_count=new_count,
            utd_ratio=ratio,
        )
        rng = np.random.default_rng(config.seed + learner.update_count)
        last_metrics: dict[str, float] = {}
        metric_sums: dict[str, float] = {}
        sample_count = int(data["state"].shape[0])
        for local_update in range(updates):
            indices = stratified_batch_indices(
                rng,
                data["success"],
                batch_size=config.sac.batch_size,
                success_samples_per_batch=config.sac.success_samples_per_batch,
            )
            batch = {
                name: values[indices]
                for name, values in data.items()
                if isinstance(values, np.ndarray) and values.shape[0] == sample_count
            }
            last_metrics = learner.update(batch)
            for name, value in last_metrics.items():
                metric_sums[name] = metric_sums.get(name, 0.0) + float(value)
            if learner.update_count % config.sac.snapshot_interval_updates == 0:
                snapshot = config.checkpoint_dir / f"step_{learner.update_count:08d}.pt"
                periodic_payload = learner.checkpoint(
                    base_model_sha256=base_model_sha256,
                    replay_high_watermark=replay_high_watermark,
                    replay_label_revision=replay_label_revision,
                )
                periodic_payload["residual_config"] = asdict(config.residual)
                _atomic_torch_save(periodic_payload, snapshot)

        checkpoint = learner.checkpoint(
            base_model_sha256=base_model_sha256,
            replay_high_watermark=replay_high_watermark,
            replay_label_revision=replay_label_revision,
        )
        checkpoint["residual_config"] = asdict(config.residual)
        _atomic_torch_save(checkpoint, latest)
        artifact = DeterministicActorArtifact(learner.actor, normalizer).to(learner.device).eval()
        traced = torch.jit.trace(
            artifact, torch.zeros((1, 1047), dtype=torch.float32, device=learner.device)
        )
        artifact_path = config.checkpoint_dir / "actor_latest.ts"
        temporary_artifact = artifact_path.with_name(f".{artifact_path.name}.tmp-{os.getpid()}")
        torch.jit.save(traced, str(temporary_artifact))
        os.replace(temporary_artifact, artifact_path)
        replay.set_meta("frozen_normalizer", {
            "sample_count": normalizer.sample_count,
            "min_replay_id": normalizer.min_replay_id,
            "max_replay_id": normalizer.max_replay_id,
            "contract_hash": normalizer.contract_hash,
            "checkpoint": str(latest),
        })
        return {
            "updated": True,
            "trainable_replay": trainable_count,
            "new_trainable_transitions": new_count,
            "new_id_trainable_transitions": changes["new_id"],
            "relabeled_transitions": changes["relabeled"],
            "update_schedule": schedule,
            "bootstrap_updates": bootstrap,
            "utd_ratio": ratio,
            "gradient_updates": updates,
            "total_updates": learner.update_count,
            "replay_high_watermark": replay_high_watermark,
            "replay_label_revision": replay_label_revision,
            "success_samples_per_batch": config.sac.success_samples_per_batch,
            "normalizer_samples": normalizer.sample_count,
            "checkpoint": str(latest),
            "actor_artifact": str(artifact_path),
            "metrics": last_metrics,
            "metrics_mean": {
                name: value / updates for name, value in metric_sums.items()
            },
            "diagnostics": _training_diagnostics(
                learner,
                data,
                max_residual_m=config.residual.max_residual_m,
            ),
        }


__all__ = [
    "train_from_replay", "gradient_updates_for", "planned_gradient_updates",
    "stratified_batch_indices",
    "_torch_load", "_validate_checkpoint"
]
