from __future__ import annotations

import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import RLPDConfig
from .learner import DeterministicActorArtifact, FrozenNormalizer, RLPDLearner, STATE_DIM
from .replay import ReplayBuffer


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_training_progress(
    *,
    completed: int,
    total: int,
    update_group: int,
    elapsed_s: float,
    metrics: dict[str, float],
) -> str:
    fraction = completed / total
    eta_s = elapsed_s * (total - completed) / completed
    fields = [
        f"RLPD train {completed}/{total} ({100.0 * fraction:.1f}%)",
        f"update_group={update_group}",
        f"elapsed={_format_duration(elapsed_s)}",
        f"eta={_format_duration(eta_s)}",
    ]
    for name in (
        "critic_loss",
        "q_mean",
        "actor_loss",
        "temperature",
        "entropy",
    ):
        if name in metrics:
            fields.append(f"{name}={float(metrics[name]):.6g}")
    return " | ".join(fields)


def _atomic_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path, device: str) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _sample(
    rng: np.random.Generator,
    data: dict[str, np.ndarray],
    count: int,
) -> dict[str, np.ndarray]:
    size = int(data["state"].shape[0])
    indices = rng.integers(0, size, size=count)
    return {name: values[indices] for name, values in data.items() if values.shape[0] == size}


def mixed_batch(
    rng: np.random.Generator,
    offline: dict[str, np.ndarray],
    online: dict[str, np.ndarray],
    *,
    total: int,
    offline_ratio: float,
) -> dict[str, np.ndarray]:
    if not offline:
        raise ValueError("RLPD requires offline expert data")
    offline_count = total if not online else int(round(total * offline_ratio))
    offline_count = min(max(0, offline_count), total)
    online_count = total - offline_count
    parts = [_sample(rng, offline, offline_count)] if offline_count else []
    if online_count:
        parts.append(_sample(rng, online, online_count))
    names = set.intersection(*(set(part) for part in parts))
    combined = {name: np.concatenate([part[name] for part in parts], axis=0) for name in names}
    order = rng.permutation(total)
    return {name: values[order] for name, values in combined.items()}


def train(
    config: RLPDConfig,
    contract: dict[str, Any],
    *,
    device: str,
    checkpoint_path: Path | None = None,
    groups: int | None = None,
    progress_interval: int = 10,
) -> dict[str, Any]:
    config.validate()
    latest = config.checkpoint_dir / "latest.pt"
    resume_path = checkpoint_path or (latest if latest.is_file() else None)
    with ReplayBuffer(config.offline_replay_path, "offline", create=False) as offline_replay:
        offline_replay.assert_contract(contract)
        offline = offline_replay.load()
    if not offline or len(offline["state"]) < config.algorithm.minimum_offline_transitions:
        count = 0 if not offline else len(offline["state"])
        raise ValueError(
            f"Need {config.algorithm.minimum_offline_transitions} offline transitions, got {count}"
        )
    with ReplayBuffer(config.online_replay_path, "online") as online_replay:
        online_replay.assert_contract(contract)
        online = online_replay.load()
        online_high_watermark = online_replay.max_id()

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if resume_path is None:
        normalizer = FrozenNormalizer.fit(
            offline["state"], offline["privileged"], config.algorithm.normalizer_clip
        )
        learner = RLPDLearner(config.algorithm, normalizer, device)
        group_count = config.algorithm.offline_pretrain_groups if groups is None else int(groups)
        schedule = "offline_pretrain"
        training_online: dict[str, np.ndarray] = {}
        checkpoint_high_watermark = 0
    else:
        payload = _torch_load(Path(resume_path), device)
        if payload.get("kind") != "franka_real_rlpd_residual" or payload.get("contract") != contract:
            raise ValueError("RLPD checkpoint contract does not match the selected base policy")
        if payload.get("algorithm") != asdict(config.algorithm):
            raise ValueError("RLPD checkpoint algorithm config does not match this run")
        normalizer = FrozenNormalizer.from_dict(payload["normalizer"])
        learner = RLPDLearner(config.algorithm, normalizer, device)
        learner.restore(payload)
        previous = int(payload.get("replay_high_watermark", 0))
        with ReplayBuffer(config.online_replay_path, "online", create=False) as replay:
            new_online = int(replay.connection.execute(
                "SELECT COUNT(*) FROM transitions WHERE trainable=1 AND id>?",
                (previous,),
            ).fetchone()[0])
        group_count = new_online if groups is None else int(groups)
        schedule = "online_episode_updates"
        if group_count == 0:
            return {"updated": False, "reason": "no_new_online_transitions"}
        training_online = online
        checkpoint_high_watermark = online_high_watermark
    if group_count < 1:
        raise ValueError("Training groups must be positive")
    if isinstance(progress_interval, bool) or int(progress_interval) < 1:
        raise ValueError("Training progress interval must be a positive integer")
    progress_interval = int(progress_interval)

    rng = np.random.default_rng(config.seed + learner.update_groups)
    group_size = config.algorithm.batch_size * config.algorithm.utd_ratio
    sums: dict[str, float] = {}
    last_metrics: dict[str, float] = {}
    started_at = time.monotonic()
    print(
        "RLPD training started: "
        f"schedule={schedule}, groups={group_count}, "
        f"start_update_group={learner.update_groups}, "
        f"offline_transitions={len(offline['state'])}, "
        f"online_transitions={0 if not online else len(online['state'])}, "
        f"device={device}, progress_interval={progress_interval}",
        flush=True,
    )
    for index in range(group_count):
        batch = mixed_batch(
            rng, offline, training_online,
            total=group_size,
            offline_ratio=config.algorithm.offline_ratio,
        )
        last_metrics = learner.update_group(batch, rng)
        for name, value in last_metrics.items():
            sums[name] = sums.get(name, 0.0) + float(value)
        completed = index + 1
        if (
            completed == 1
            or completed % progress_interval == 0
            or completed == group_count
        ):
            print(
                _format_training_progress(
                    completed=completed,
                    total=group_count,
                    update_group=learner.update_groups,
                    elapsed_s=time.monotonic() - started_at,
                    metrics=last_metrics,
                ),
                flush=True,
            )
        if learner.update_groups % config.algorithm.snapshot_interval_groups == 0:
            _atomic_save(
                learner.checkpoint(contract, checkpoint_high_watermark),
                config.checkpoint_dir / f"group_{learner.update_groups:08d}.pt",
            )

    _atomic_save(learner.checkpoint(contract, checkpoint_high_watermark), latest)
    artifact = DeterministicActorArtifact(learner.actor, normalizer).to(learner.device).eval()
    traced = torch.jit.trace(
        artifact,
        torch.zeros((1, STATE_DIM), dtype=torch.float32, device=learner.device),
    )
    artifact_path = config.checkpoint_dir / "actor_latest.ts"
    temporary = artifact_path.with_name(f".{artifact_path.name}.tmp-{os.getpid()}")
    torch.jit.save(traced, str(temporary))
    os.replace(temporary, artifact_path)
    return {
        "updated": True,
        "schedule": schedule,
        "groups": group_count,
        "utd_ratio": config.algorithm.utd_ratio,
        "critic_updates": group_count * config.algorithm.utd_ratio,
        "offline_transitions": len(offline["state"]),
        "online_transitions": 0 if not online else len(online["state"]),
        "checkpoint": str(latest),
        "actor_artifact": str(artifact_path),
        "metrics": last_metrics,
        "metrics_mean": {name: value / group_count for name, value in sums.items()},
    }
