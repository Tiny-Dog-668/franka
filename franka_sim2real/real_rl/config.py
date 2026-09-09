from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def _positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AprilTagConfig:
    calibration_report: Path
    family: str = "tag36h11"
    marker_id: int = 2
    marker_length_m: float = 0.04
    tag_to_object_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    height_offset_m: float = 0.0
    detection_scale: float = 3.0
    max_reprojection_px: float = 0.8
    max_pose_age_s: float = 0.2
    smoothing_window: int = 1
    preflight_valid_detections: int = 15
    preflight_timeout_s: float = 15.0
    offline_roi_expansion: float = 3.0

    def validate(self) -> None:
        if self.family != "tag36h11" or self.marker_id != 2:
            raise ValueError("Real-RL v2 requires AprilTag tag36h11 ID=2")
        _positive("apriltag.marker_length_m", self.marker_length_m)
        _positive("apriltag.detection_scale", self.detection_scale)
        _positive("apriltag.max_reprojection_px", self.max_reprojection_px)
        _positive("apriltag.max_pose_age_s", self.max_pose_age_s)
        _positive("apriltag.preflight_timeout_s", self.preflight_timeout_s)
        _positive("apriltag.offline_roi_expansion", self.offline_roi_expansion)
        if self.smoothing_window < 1 or self.preflight_valid_detections < 1:
            raise ValueError("AprilTag window and preflight count must be positive")
        if len(self.tag_to_object_m) != 3 or not all(math.isfinite(v) for v in self.tag_to_object_m):
            raise ValueError("apriltag.tag_to_object_m must contain three finite values")
        if not math.isfinite(self.height_offset_m):
            raise ValueError("apriltag.height_offset_m must be finite")
        if not self.calibration_report.is_file():
            raise ValueError(f"Accepted eye-to-hand report does not exist: {self.calibration_report}")


@dataclass(frozen=True)
class RewardConfig:
    k_reach: float = 10.0
    k_lift: float = 20.0
    k_success: float = 10.0
    k_action: float = 0.01
    success_height_m: float = 0.03
    success_consecutive_detections: int = 3

    def validate(self) -> None:
        for name in ("k_reach", "k_lift", "k_success", "k_action"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"reward.{name} must be finite and non-negative")
        _positive("reward.success_height_m", self.success_height_m)
        if self.success_consecutive_detections < 1:
            raise ValueError("reward.success_consecutive_detections must be positive")


@dataclass(frozen=True)
class ResidualConfig:
    max_residual_m: float = 0.002
    action_scale_m: float = 0.05
    warmup_std_m: float = 0.0005
    warmup_cap_m: float = 0.001
    warmup_correlation: float = 0.9

    def validate(self) -> None:
        for name in ("max_residual_m", "action_scale_m", "warmup_std_m", "warmup_cap_m"):
            _positive(f"residual.{name}", float(getattr(self, name)))
        if self.max_residual_m > 0.003:
            raise ValueError("residual.max_residual_m may not exceed 3 mm")
        if self.warmup_cap_m > self.max_residual_m:
            raise ValueError("residual.warmup_cap_m may not exceed max_residual_m")
        if not math.isfinite(self.warmup_correlation) or not 0.0 <= self.warmup_correlation < 1.0:
            raise ValueError("residual.warmup_correlation must be finite and in [0,1)")


@dataclass(frozen=True)
class SACConfig:
    hidden_dims: tuple[int, int] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    success_samples_per_batch: int = 1
    target_entropy: float = -3.0
    initial_log_std: float = -3.0
    minimum_trainable_replay: int = 1000
    normalizer_samples: int = 1000
    normalizer_clip: float = 10.0
    utd_ratio: float = 2.0
    bootstrap_updates: int = 1000
    snapshot_interval_updates: int = 5000

    def validate(self) -> None:
        if tuple(self.hidden_dims) != (256, 256):
            raise ValueError("Residual SAC v2 hidden_dims must be [256,256]")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("sac.gamma and sac.tau must be in (0,1]")
        for name in ("actor_lr", "critic_lr", "alpha_lr", "normalizer_clip"):
            _positive(f"sac.{name}", float(getattr(self, name)))
        if self.batch_size < 1 or self.minimum_trainable_replay < self.batch_size:
            raise ValueError("SAC replay minimum must be at least one batch")
        if not 0 <= self.success_samples_per_batch < self.batch_size:
            raise ValueError(
                "sac.success_samples_per_batch must be in [0,batch_size)"
            )
        if self.normalizer_samples != 1000:
            raise ValueError("Real-RL v2 freezes the normalizer on exactly 1000 samples")
        if self.minimum_trainable_replay < self.normalizer_samples:
            raise ValueError("SAC minimum_trainable_replay must cover all normalizer samples")
        if not 1.0 <= self.utd_ratio <= 4.0:
            raise ValueError("sac.utd_ratio must be in [1,4]")
        if self.bootstrap_updates < 1:
            raise ValueError("sac.bootstrap_updates must be positive")
        if self.snapshot_interval_updates < 1:
            raise ValueError("sac.snapshot_interval_updates must be positive")


@dataclass(frozen=True)
class ReplayConfig:
    path: Path
    writer_queue_size: int = 256

    def validate(self) -> None:
        if self.writer_queue_size < 1:
            raise ValueError("replay.writer_queue_size must be positive")


@dataclass(frozen=True)
class RealRLConfig:
    base_policy_config: Path
    checkpoint_dir: Path
    log_dir: Path
    replay: ReplayConfig
    apriltag: AprilTagConfig
    reward: RewardConfig = field(default_factory=RewardConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    seed: int = 17
    schema_version: int = 2

    def validate(self) -> None:
        if self.schema_version != 2:
            raise ValueError("Real-RL config schema_version must be 2")
        if not self.base_policy_config.is_file():
            raise ValueError(f"Base policy config does not exist: {self.base_policy_config}")
        self.replay.validate()
        self.apriltag.validate()
        self.reward.validate()
        self.residual.validate()
        self.sac.validate()
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")


def load_real_rl_config(path: str | Path) -> RealRLConfig:
    config_path = Path(path).expanduser().resolve()
    payload: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    april = dict(payload["apriltag"])
    april["calibration_report"] = _path(april["calibration_report"])
    if "tag_to_object_m" in april:
        april["tag_to_object_m"] = tuple(float(v) for v in april["tag_to_object_m"])
    sac = dict(payload.get("sac", {}))
    if "hidden_dims" in sac:
        sac["hidden_dims"] = tuple(int(v) for v in sac["hidden_dims"])
    config = RealRLConfig(
        schema_version=int(payload.get("schema_version", 2)),
        base_policy_config=_path(payload["base_policy_config"]),
        checkpoint_dir=_path(payload["checkpoint_dir"]),
        log_dir=_path(payload.get("log_dir", "real_rl_logs")),
        replay=ReplayConfig(
            path=_path(payload["replay"]["path"]),
            writer_queue_size=int(payload["replay"].get("writer_queue_size", 256)),
        ),
        apriltag=AprilTagConfig(**april),
        reward=RewardConfig(**payload.get("reward", {})),
        residual=ResidualConfig(**payload.get("residual", {})),
        sac=SACConfig(**sac),
        seed=int(payload.get("seed", 17)),
    )
    config.validate()
    return config
