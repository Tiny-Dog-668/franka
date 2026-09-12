from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .action import combine_post_limit
from .adapters import ACTION_DIM, GelSightPolicyAdapter, sha256_file
from .config import RLPDConfig
from .learner import FrozenNormalizer, GaussianActor


DeployMode = Literal["expert_shadow", "checkpoint"]


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@dataclass(frozen=True)
class RLPDDeploySettings:
    config: RLPDConfig
    mode: DeployMode
    commissioning_limit: float
    checkpoint_path: Path | None = None
    stochastic: bool = False
    enable_stochastic_control: bool = False

    def validate(self) -> None:
        self.config.validate()
        if self.mode not in {"expert_shadow", "checkpoint"}:
            raise ValueError(f"Unsupported RLPD mode: {self.mode}")
        if self.mode == "checkpoint" and self.checkpoint_path is None:
            raise ValueError("Checkpoint mode requires checkpoint_path")
        if self.mode == "expert_shadow" and self.checkpoint_path is not None:
            raise ValueError("Expert shadow mode cannot load a checkpoint")
        if self.stochastic and not self.enable_stochastic_control:
            raise ValueError("Stochastic RLPD control requires explicit enable_stochastic_control")
        if self.enable_stochastic_control and not self.stochastic:
            raise ValueError("enable_stochastic_control requires stochastic=True")


class RLPDPolicyRuntime:
    def __init__(
        self,
        settings: RLPDDeploySettings,
        *,
        model: torch.jit.ScriptModule,
        metadata: dict[str, Any],
        model_path: Path,
        device: torch.device,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.device = device
        self.adapter = GelSightPolicyAdapter(model, metadata, model_path)
        self.actor: GaussianActor | None = None
        self.normalizer: FrozenNormalizer | None = None
        self.checkpoint_sha256: str | None = None
        self.fallback_reason: str | None = None
        self._prepared_features: torch.Tensor | None = None
        if settings.mode == "checkpoint":
            try:
                assert settings.checkpoint_path is not None
                path = settings.checkpoint_path.expanduser().resolve()
                payload = _torch_load(path, device)
                self._validate_checkpoint(payload)
                self.normalizer = FrozenNormalizer.from_dict(payload["normalizer"])
                self.actor = GaussianActor(float(payload["algorithm"]["initial_log_std"])).to(device)
                self.actor.load_state_dict(payload["actor"])
                self.actor.eval()
                self.checkpoint_sha256 = sha256_file(path)
            except Exception as exc:
                self.fallback_reason = f"{type(exc).__name__}: {exc}"
                warnings.warn(
                    "RLPD CHECKPOINT REJECTED: episode start will be refused; " + self.fallback_reason,
                    RuntimeWarning,
                    stacklevel=2,
                )

    @property
    def requires_episode_refusal(self) -> bool:
        return self.settings.mode == "checkpoint" and self.fallback_reason is not None

    def contract(self) -> dict[str, Any]:
        base = self.adapter.contract.to_dict()
        reward_contract = json.loads(json.dumps(asdict(self.settings.config.reward)))
        base.update({
            "state_contract": "policy_feature_1043_plus_base_limited_action_4",
            "action_contract": "unit_residual_4_times_2x_commissioning_limit",
            "composition": "post_commissioning_then_final_existing_safety_clip",
            "commissioning_limit": float(self.settings.commissioning_limit),
            "action_scales": [0.05, 0.05, 0.05, 0.01],
            "reward_kind": self.settings.config.reward_kind,
            "reward": reward_contract,
        })
        return base

    def _validate_checkpoint(self, payload: dict[str, Any]) -> None:
        if payload.get("kind") != "franka_real_rlpd_residual" or payload.get("format_version") != 1:
            raise ValueError("RLPD checkpoint kind/version is incompatible")
        if payload.get("contract") != self.contract():
            raise ValueError("RLPD checkpoint belongs to another policy or action contract")

    def prepare(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        tactile: dict[str, torch.Tensor],
        base_action: torch.Tensor,
    ) -> None:
        self._prepared_features = self.adapter.extract(
            wrist_rgb, proprio_obs, action_history, tactile, base_action
        ).detach()

    def apply_post_limit(self, base_limited: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        if self._prepared_features is None:
            raise RuntimeError("RLPD runtime was not prepared by the current policy tick")
        base = torch.as_tensor(base_limited, dtype=torch.float32, device=self.device).reshape(1, ACTION_DIM)
        state = torch.cat((self._prepared_features, base), dim=-1)
        if self.actor is None or self.normalizer is None:
            unit = torch.zeros((1, ACTION_DIM), dtype=torch.float32, device=self.device)
        else:
            with torch.inference_mode():
                unit, _ = self.actor.sample(
                    self.normalizer.state_tensor(state),
                    deterministic=not self.settings.stochastic,
                )
        unit_np = torch.clamp(unit, -1.0, 1.0).cpu().numpy().reshape(ACTION_DIM)
        candidate, residual = combine_post_limit(
            base_limited, unit_np, self.settings.commissioning_limit
        )
        info = {
            "state": state.cpu().numpy().reshape(-1).tolist(),
            "base_limited_action": np.asarray(base_limited, dtype=np.float32).tolist(),
            "unit_residual_action": unit_np.tolist(),
            "residual_normalized_action": residual.tolist(),
            "candidate_action": candidate.tolist(),
            "mode": self.settings.mode,
            "stochastic": self.settings.stochastic,
            "checkpoint_sha256": self.checkpoint_sha256,
            "fallback_reason": self.fallback_reason,
            "contract": self.contract(),
        }
        self._prepared_features = None
        return candidate, info

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.settings.mode,
            "stochastic": self.settings.stochastic,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_error": self.fallback_reason,
            "episode_start_refused": self.requires_episode_refusal,
            "contract": self.contract(),
        }
