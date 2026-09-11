from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from franka_sim2real.real_rl.config import AprilTagConfig, RewardConfig
from .reward import APRILTAG_PROGRESS_REWARD


REPO_ROOT = Path(__file__).resolve().parents[1]


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AlgorithmConfig:
    hidden_dims: tuple[int, int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    batch_size: int = 256
    utd_ratio: int = 20
    offline_ratio: float = 0.5
    num_qs: int = 10
    num_min_qs: int = 2
    critic_layer_norm: bool = True
    backup_entropy: bool = True
    target_entropy: float = -2.0
    initial_temperature: float = 1.0
    initial_log_std: float = -3.0
    minimum_offline_transitions: int = 1000
    offline_pretrain_groups: int = 1000
    snapshot_interval_groups: int = 250
    normalizer_clip: float = 10.0

    def validate(self) -> None:
        if tuple(self.hidden_dims) != (256, 256):
            raise ValueError("RLPD hidden_dims must be [256,256]")
        for name in (
            "actor_lr", "critic_lr", "temperature_lr", "initial_temperature",
            "normalizer_clip",
        ):
            _positive(f"algorithm.{name}", getattr(self, name))
        if not 0.0 < self.discount <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("algorithm discount and tau must be in (0,1]")
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("algorithm.batch_size must be a positive even integer")
        if not 1 <= self.utd_ratio <= 20:
            raise ValueError("algorithm.utd_ratio must be in [1,20]")
        if not 0.0 <= self.offline_ratio <= 1.0:
            raise ValueError("algorithm.offline_ratio must be in [0,1]")
        if self.num_qs < 2 or not 1 <= self.num_min_qs <= self.num_qs:
            raise ValueError("algorithm requires 2 <= num_qs and 1 <= num_min_qs <= num_qs")
        if self.minimum_offline_transitions < self.batch_size:
            raise ValueError("minimum_offline_transitions must cover one batch")
        if self.offline_pretrain_groups < 1 or self.snapshot_interval_groups < 1:
            raise ValueError("pretrain and snapshot group counts must be positive")


@dataclass(frozen=True)
class ResidualConfig:
    action_dim: int = 4
    composition: str = "post_commissioning"
    span_multiplier: float = 2.0

    def validate(self) -> None:
        if self.action_dim != 4:
            raise ValueError("RLPD residual action must be XYZ plus gripper (4D)")
        if self.composition != "post_commissioning":
            raise ValueError("RLPD residual composition must be post_commissioning")
        if self.span_multiplier != 2.0:
            raise ValueError("Full takeover requires residual span_multiplier=2.0")


@dataclass(frozen=True)
class TeleopConfig:
    xyz_speed_m_s: float = 0.05
    gripper_speed_m_s: float = 0.03

    def validate(self) -> None:
        _positive("teleop.xyz_speed_m_s", self.xyz_speed_m_s)
        _positive("teleop.gripper_speed_m_s", self.gripper_speed_m_s)


@dataclass(frozen=True)
class RLPDConfig:
    base_policy_config: Path
    offline_replay_path: Path
    online_replay_path: Path
    checkpoint_dir: Path
    expert_data_dir: Path
    rollout_data_dir: Path
    apriltag: AprilTagConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    reward_kind: str = APRILTAG_PROGRESS_REWARD
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    teleop: TeleopConfig = field(default_factory=TeleopConfig)
    seed: int = 17
    schema_version: int = 2

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("RLPD config schema_version must be 2")
        if not self.base_policy_config.is_file():
            raise ValueError(f"Base policy config does not exist: {self.base_policy_config}")
        if self.offline_replay_path == self.online_replay_path:
            raise ValueError("Offline and online replay paths must differ")
        if self.expert_data_dir == self.rollout_data_dir:
            raise ValueError("Expert source data and rollout data directories must differ")
        for parent, child in (
            (self.expert_data_dir, self.rollout_data_dir),
            (self.rollout_data_dir, self.expert_data_dir),
        ):
            try:
                child.relative_to(parent)
            except ValueError:
                continue
            raise ValueError("Expert source data and rollout directories must not overlap")
        self.apriltag.validate()
        self.reward.validate()
        if self.reward_kind != APRILTAG_PROGRESS_REWARD:
            raise ValueError(f"Unsupported RLPD reward_kind: {self.reward_kind!r}")
        self.algorithm.validate()
        self.residual.validate()
        self.teleop.validate()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def load_config(path: str | Path) -> RLPDConfig:
    payload: dict[str, Any] = json.loads(Path(path).expanduser().resolve().read_text())
    april = dict(payload["apriltag"])
    april["calibration_report"] = _path(april["calibration_report"])
    if "tag_to_object_m" in april:
        april["tag_to_object_m"] = tuple(float(v) for v in april["tag_to_object_m"])
    algorithm = dict(payload.get("algorithm", {}))
    if "hidden_dims" in algorithm:
        algorithm["hidden_dims"] = tuple(int(v) for v in algorithm["hidden_dims"])
    replay = payload["replay"]
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError(
            "RLPD schema v2 requires separate data.expert_dir and data.rollout_dir"
        )
    result = RLPDConfig(
        schema_version=int(payload.get("schema_version", 1)),
        base_policy_config=_path(payload["base_policy_config"]),
        offline_replay_path=_path(replay["offline_path"]),
        online_replay_path=_path(replay["online_path"]),
        checkpoint_dir=_path(payload["checkpoint_dir"]),
        expert_data_dir=_path(data["expert_dir"]),
        rollout_data_dir=_path(data["rollout_dir"]),
        apriltag=AprilTagConfig(**april),
        reward=RewardConfig(**payload.get("reward", {})),
        reward_kind=str(payload.get("reward_kind", APRILTAG_PROGRESS_REWARD)),
        algorithm=AlgorithmConfig(**algorithm),
        residual=ResidualConfig(**payload.get("residual", {})),
        teleop=TeleopConfig(**payload.get("teleop", {})),
        seed=int(payload.get("seed", 17)),
    )
    result.validate()
    return result
