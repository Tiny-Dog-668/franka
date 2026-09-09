from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .config import RealRLConfig
from .residual_sac import FrozenNormalizer, GaussianActor
from .trainer import _torch_load, _validate_checkpoint


CollectMode = Literal["zero", "warmup_random", "checkpoint"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RealRLDeploySettings:
    config: RealRLConfig
    mode: CollectMode
    checkpoint_path: Path | None = None
    stochastic: bool = True
    seed: int = 17
    allow_checkpoint_fallback_collect: bool = False

    def validate(self) -> None:
        self.config.validate()
        if self.mode not in {"zero", "warmup_random", "checkpoint"}:
            raise ValueError(f"Unsupported Real-RL collect mode: {self.mode}")
        if self.mode == "checkpoint" and self.checkpoint_path is None:
            raise ValueError("Checkpoint collect mode requires checkpoint_path")
        if self.mode != "checkpoint" and self.checkpoint_path is not None:
            raise ValueError("Warmup modes cannot load a checkpoint")
        if self.allow_checkpoint_fallback_collect and self.mode != "checkpoint":
            raise ValueError("Checkpoint fallback opt-in is only valid with checkpoint mode")


class RealRLPolicyRuntime:
    """XYZ-only residual action selector; failures fall back to exactly zero."""

    def __init__(
        self,
        settings: RealRLDeploySettings,
        *,
        base_model_path: Path,
        base_kind: str,
        device: torch.device,
    ) -> None:
        settings.validate()
        if base_kind != "tacex_rma_gelsight_size_buckets_student_torchscript":
            raise ValueError(f"Real-RL v2 requires the 0814 GelSight Student, got {base_kind!r}")
        self.settings = settings
        self.device = device
        self.base_model_sha256 = sha256_file(base_model_path)
        self.checkpoint_sha256: str | None = None
        self.fallback_reason: str | None = None
        self.actor: GaussianActor | None = None
        self.normalizer: FrozenNormalizer | None = None
        self._rng = np.random.default_rng(settings.seed)
        self._warmup_previous_m: np.ndarray | None = None
        if settings.mode == "checkpoint":
            assert settings.checkpoint_path is not None
            try:
                path = settings.checkpoint_path.expanduser().resolve()
                if not path.is_file():
                    raise FileNotFoundError(path)
                payload = _torch_load(path, device)
                _validate_checkpoint(
                    payload, self.base_model_sha256, settings.config.residual
                )
                self.normalizer = FrozenNormalizer.from_dict(payload["normalizer"])
                self.actor = GaussianActor(float(payload["sac_config"]["initial_log_std"])).to(device)
                self.actor.load_state_dict(payload["actor"])
                self.actor.eval()
                self.checkpoint_sha256 = sha256_file(path)
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    "REAL-RL CHECKPOINT REJECTED: residual forced to zero because "
                    + self.fallback_reason,
                    RuntimeWarning,
                    stacklevel=2,
                )

    @property
    def requires_episode_refusal(self) -> bool:
        return bool(
            self.settings.mode == "checkpoint"
            and self.fallback_reason is not None
            and not self.settings.allow_checkpoint_fallback_collect
        )

    def reset_episode(self) -> None:
        """在 episode 边界重置有时间相关性的 warmup 探索。"""
        self._rng = np.random.default_rng(self.settings.seed)
        self._warmup_previous_m = None

    def apply(
        self,
        actor_features: torch.Tensor,
        base_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if tuple(actor_features.shape) != (1, 1043) or tuple(base_action.shape) != (1, 4):
            raise RuntimeError("Real-RL requires actor_features[1,1043] and base_action[1,4]")
        state = torch.cat((actor_features, base_action), dim=-1)
        residual_config = self.settings.config.residual
        if self.settings.mode == "warmup_random":
            innovation = self._rng.normal(0.0, residual_config.warmup_std_m, size=3)
            if self._warmup_previous_m is None:
                physical = innovation
            else:
                rho = float(residual_config.warmup_correlation)
                physical = (
                    rho * self._warmup_previous_m
                    + math.sqrt(1.0 - rho * rho) * innovation
                )
            physical = np.clip(
                physical, -residual_config.warmup_cap_m, residual_config.warmup_cap_m
            )
            self._warmup_previous_m = physical.copy()
            unit_numpy = physical / residual_config.max_residual_m
            unit_action = torch.as_tensor(
                unit_numpy, dtype=torch.float32, device=self.device
            ).reshape(1, 3)
        elif self.actor is not None and self.normalizer is not None:
            normalized = self.normalizer.state_tensor(state)
            with torch.inference_mode():
                unit_action, _log_prob = self.actor.sample(
                    normalized, deterministic=not self.settings.stochastic
                )
        else:
            unit_action = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
        unit_action = torch.clamp(unit_action, -1.0, 1.0)
        residual_m = unit_action * float(residual_config.max_residual_m)
        residual_normalized = residual_m / float(residual_config.action_scale_m)
        maximum_normalized = residual_config.max_residual_m / residual_config.action_scale_m
        residual_normalized = torch.clamp(
            residual_normalized, -maximum_normalized, maximum_normalized
        )
        final = base_action.clone()
        final[:, :3] += residual_normalized
        if not torch.isfinite(final).all():
            raise RuntimeError("Real-RL combined action contains NaN or Inf")
        info = {
            "state": state.detach().cpu().reshape(-1).tolist(),
            "base_action": base_action.detach().cpu().reshape(-1).tolist(),
            "sac_unit_action": unit_action.detach().cpu().reshape(-1).tolist(),
            "residual_action_m": residual_m.detach().cpu().reshape(-1).tolist(),
            "residual_action_normalized": residual_normalized.detach().cpu().reshape(-1).tolist(),
            "final_action": final.detach().cpu().reshape(-1).tolist(),
            "mode": self.settings.mode,
            "stochastic": bool(self.settings.stochastic),
            "fallback": self.fallback_reason is not None,
            "fallback_reason": self.fallback_reason,
            "checkpoint_sha256": self.checkpoint_sha256,
            "base_model_sha256": self.base_model_sha256,
            "max_residual_m": float(residual_config.max_residual_m),
        }
        return final, info

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.settings.mode,
            "base_model_sha256": self.base_model_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_fallback": self.fallback_reason,
            "checkpoint_fallback_collect_allowed": bool(
                self.settings.allow_checkpoint_fallback_collect
            ),
            "episode_start_refused": self.requires_episode_refusal,
            "max_residual_m": self.settings.config.residual.max_residual_m,
            "warmup_std_m": self.settings.config.residual.warmup_std_m,
            "warmup_cap_m": self.settings.config.residual.warmup_cap_m,
            "warmup_correlation": self.settings.config.residual.warmup_correlation,
            "warmup_noise": "temporally_correlated_gaussian_ar1",
            "actor_input_dim": 1047,
            "actor_output_dim": 3,
        }
