from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import SACConfig
from .replay_buffer import PRIVILEGED_DIM, STATE_DIM


ACTION_DIM = 3
POLICY_FEATURE_DIM = 1043
BASE_ACTION_DIM = 4
if POLICY_FEATURE_DIM + BASE_ACTION_DIM != STATE_DIM:
    raise RuntimeError("Residual SAC actor state dimensions are inconsistent")
NORMALIZER_CONTRACT = (
    "policy_feature_1043_empirical;base_action_4_fixed_no_zscore;"
    "privileged_4_empirical;clip_normalized"
)
NORMALIZER_CONTRACT_HASH = hashlib.sha256(NORMALIZER_CONTRACT.encode("utf-8")).hexdigest()
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class FrozenNormalizer:
    policy_feature_mean: np.ndarray
    policy_feature_std: np.ndarray
    privileged_mean: np.ndarray
    privileged_std: np.ndarray
    clip: float
    sample_count: int
    min_replay_id: int
    max_replay_id: int
    contract_hash: str

    def validate(self) -> None:
        for name, size in (
            ("policy_feature_mean", POLICY_FEATURE_DIM),
            ("policy_feature_std", POLICY_FEATURE_DIM),
            ("privileged_mean", PRIVILEGED_DIM), ("privileged_std", PRIVILEGED_DIM),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Frozen normalizer {name} must contain {size} finite values")
            if name.endswith("std") and np.any(value <= 0.0):
                raise ValueError(f"Frozen normalizer {name} must be positive")
        if self.sample_count != 1000:
            raise ValueError("Frozen normalizer must use exactly the first 1000 trainable samples")
        if self.min_replay_id < 1 or self.max_replay_id < self.min_replay_id:
            raise ValueError("Frozen normalizer Replay ID range is invalid")
        if self.contract_hash != NORMALIZER_CONTRACT_HASH:
            raise ValueError("Frozen normalizer contract hash is incompatible")
        if not math.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("Frozen normalizer clip must be finite and positive")

    @classmethod
    def fit(
        cls,
        state: np.ndarray,
        privileged: np.ndarray,
        replay_ids: np.ndarray,
        *,
        clip: float = 10.0,
    ) -> "FrozenNormalizer":
        states = np.asarray(state, dtype=np.float64)
        priv = np.asarray(privileged, dtype=np.float64)
        ids = np.asarray(replay_ids, dtype=np.int64)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"Normalizer state must be [N,{STATE_DIM}]")
        if priv.shape != (states.shape[0], PRIVILEGED_DIM):
            raise ValueError(f"Normalizer privileged state must be [N,{PRIVILEGED_DIM}]")
        if ids.shape != (states.shape[0],) or states.shape[0] < 1:
            raise ValueError("Normalizer Replay IDs do not match samples")
        if np.any(np.diff(ids) <= 0):
            raise ValueError("Normalizer Replay IDs must be strictly increasing")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(priv)):
            raise ValueError("Normalizer data contains NaN or Inf")
        policy_features = states[:, :POLICY_FEATURE_DIM]
        policy_feature_std = np.maximum(policy_features.std(axis=0), 1e-6)
        privileged_std = np.maximum(priv.std(axis=0), 1e-6)
        result = cls(
            policy_feature_mean=policy_features.mean(axis=0).astype(np.float32),
            policy_feature_std=policy_feature_std.astype(np.float32),
            privileged_mean=priv.mean(axis=0).astype(np.float32),
            privileged_std=privileged_std.astype(np.float32),
            clip=float(clip),
            sample_count=int(states.shape[0]),
            min_replay_id=int(ids.min()),
            max_replay_id=int(ids.max()),
            contract_hash=NORMALIZER_CONTRACT_HASH,
        )
        result.validate()
        return result

    def state_tensor(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != STATE_DIM:
            raise ValueError(f"Actor state must end in {STATE_DIM} values")
        mean = torch.as_tensor(
            self.policy_feature_mean, dtype=value.dtype, device=value.device
        )
        std = torch.as_tensor(
            self.policy_feature_std, dtype=value.dtype, device=value.device
        )
        features = torch.clamp(
            (value[..., :POLICY_FEATURE_DIM] - mean) / std, -self.clip, self.clip
        )
        # Base action 已处于固定的归一化 policy action contract；不从 Replay 拟合或做 z-score。
        base_action = torch.clamp(value[..., POLICY_FEATURE_DIM:], -1.0, 1.0)
        return torch.cat((features, base_action), dim=-1)

    def privileged_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.privileged_mean, dtype=value.dtype, device=value.device)
        std = torch.as_tensor(self.privileged_std, dtype=value.dtype, device=value.device)
        return torch.clamp((value - mean) / std, -self.clip, self.clip)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_feature_mean": self.policy_feature_mean,
            "policy_feature_std": self.policy_feature_std,
            "privileged_mean": self.privileged_mean,
            "privileged_std": self.privileged_std,
            "clip": self.clip,
            "sample_count": self.sample_count,
            "min_replay_id": self.min_replay_id,
            "max_replay_id": self.max_replay_id,
            "contract_hash": self.contract_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenNormalizer":
        result = cls(
            policy_feature_mean=np.asarray(payload["policy_feature_mean"], dtype=np.float32),
            policy_feature_std=np.asarray(payload["policy_feature_std"], dtype=np.float32),
            privileged_mean=np.asarray(payload["privileged_mean"], dtype=np.float32),
            privileged_std=np.asarray(payload["privileged_std"], dtype=np.float32),
            clip=float(payload["clip"]),
            sample_count=int(payload["sample_count"]),
            min_replay_id=int(payload["min_replay_id"]),
            max_replay_id=int(payload["max_replay_id"]),
            contract_hash=str(payload["contract_hash"]),
        )
        result.validate()
        return result


class GaussianActor(nn.Module):
    def __init__(self, initial_log_std: float = -3.0) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, ACTION_DIM)
        self.log_std = nn.Linear(256, ACTION_DIM)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, float(initial_log_std))

    def statistics(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        return self.mean(hidden), torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def sample(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self.statistics(state)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        pre_tanh = distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = distribution.log_prob(pre_tanh) - torch.log(
            torch.clamp(1.0 - action.square(), min=1e-6)
        )
        return action, log_prob.sum(dim=-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.statistics(state)
        return torch.tanh(mean)


class PrivilegedQ(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_DIM + PRIVILEGED_DIM + ACTION_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(
        self, state: torch.Tensor, privileged: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((state, privileged, action), dim=-1))


class ResidualSAC:
    def __init__(
        self,
        config: SACConfig,
        normalizer: FrozenNormalizer,
        device: str | torch.device,
    ) -> None:
        config.validate()
        self.config = config
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.actor = GaussianActor(config.initial_log_std).to(self.device)
        self.q1 = PrivilegedQ().to(self.device)
        self.q2 = PrivilegedQ().to(self.device)
        self.target_q1 = PrivilegedQ().to(self.device)
        self.target_q2 = PrivilegedQ().to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=config.critic_lr
        )
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
        self.update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensor = lambda name: torch.as_tensor(batch[name], dtype=torch.float32, device=self.device)
        state = self.normalizer.state_tensor(tensor("state"))
        next_state = self.normalizer.state_tensor(tensor("next_state"))
        privileged = self.normalizer.privileged_tensor(tensor("privileged"))
        next_privileged = self.normalizer.privileged_tensor(tensor("next_privileged"))
        action = tensor("sac_unit_action")
        reward = tensor("reward").reshape(-1, 1)
        terminated = tensor("terminated").reshape(-1, 1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            assert next_log_prob is not None
            next_q = torch.minimum(
                self.target_q1(next_state, next_privileged, next_action),
                self.target_q2(next_state, next_privileged, next_action),
            ) - self.alpha.detach() * next_log_prob
            target = reward + self.config.gamma * (1.0 - terminated) * next_q

        q1 = self.q1(state, privileged, action)
        q2 = self.q2(state, privileged, action)
        critic_loss = torch.nn.functional.mse_loss(q1, target) + torch.nn.functional.mse_loss(q2, target)
        self.q_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.q_optimizer.step()

        sampled_action, log_prob = self.actor.sample(state)
        assert log_prob is not None
        q_pi = torch.minimum(
            self.q1(state, privileged, sampled_action),
            self.q2(state, privileged, sampled_action),
        )
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob.detach() + self.config.target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_parameter.lerp_(parameter, self.config.tau)
            for target_parameter, parameter in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_parameter.lerp_(parameter, self.config.tau)
        self.update_count += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q_mean": float(q_pi.detach().mean().cpu()),
        }

    def checkpoint(
        self,
        *,
        base_model_sha256: str,
        replay_high_watermark: int,
        replay_label_revision: int = 0,
    ) -> dict[str, Any]:
        return {
            "kind": "franka_real_residual_sac",
            "format_version": 2,
            "input_contract": "empirical_norm_policy_feature_1043_plus_fixed_base_action_4",
            "critic_privileged_contract": "object_relative_xyz_3_plus_height_1",
            "output_contract": "unit_xyz_3_times_configured_max_residual_m",
            "gripper_source": "base_policy",
            "base_model_sha256": base_model_sha256,
            "sac_config": asdict(self.config),
            "normalizer": self.normalizer.to_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(), "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self.update_count,
            "replay_high_watermark": int(replay_high_watermark),
            "replay_label_revision": int(replay_label_revision),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
            "numpy_rng_state": np.random.get_state(),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.q1.load_state_dict(payload["q1"]); self.q2.load_state_dict(payload["q2"])
        self.target_q1.load_state_dict(payload["target_q1"])
        self.target_q2.load_state_dict(payload["target_q2"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.q_optimizer.load_state_dict(payload["q_optimizer"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(payload["log_alpha"], device=self.device))
        self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        self.update_count = int(payload["update_count"])
        if "torch_rng_state" in payload:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_state = payload.get("cuda_rng_state_all")
        if cuda_state is not None and torch.cuda.is_available():
            # ``_torch_load(..., map_location="cuda:0")`` also moves the
            # serialized CUDA RNG ByteTensors onto the GPU. PyTorch's RNG
            # restore API requires those state buffers to reside on CPU.
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_state])
        if "numpy_rng_state" in payload:
            np.random.set_state(payload["numpy_rng_state"])


class DeterministicActorArtifact(nn.Module):
    def __init__(self, actor: GaussianActor, normalizer: FrozenNormalizer) -> None:
        super().__init__()
        self.actor = actor.eval()
        self.register_buffer(
            "policy_feature_mean", torch.as_tensor(normalizer.policy_feature_mean)
        )
        self.register_buffer(
            "policy_feature_std", torch.as_tensor(normalizer.policy_feature_std)
        )
        self.clip = float(normalizer.clip)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.clamp(
            (state[..., :POLICY_FEATURE_DIM] - self.policy_feature_mean)
            / self.policy_feature_std,
            -self.clip,
            self.clip,
        )
        base_action = torch.clamp(state[..., POLICY_FEATURE_DIM:], -1.0, 1.0)
        normalized = torch.cat((features, base_action), dim=-1)
        return self.actor(normalized)
