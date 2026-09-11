from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .adapters import ACTION_DIM, FEATURE_DIM
from .config import AlgorithmConfig


STATE_DIM = FEATURE_DIM + ACTION_DIM
PRIVILEGED_DIM = 4
LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


@dataclass(frozen=True)
class FrozenNormalizer:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    privileged_mean: np.ndarray
    privileged_std: np.ndarray
    clip: float
    sample_count: int

    @classmethod
    def fit(cls, states: np.ndarray, privileged: np.ndarray, clip: float) -> "FrozenNormalizer":
        state = np.asarray(states, dtype=np.float64)
        priv = np.asarray(privileged, dtype=np.float64)
        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            raise ValueError(f"states must be [N,{STATE_DIM}]")
        if priv.shape != (state.shape[0], PRIVILEGED_DIM) or state.shape[0] < 1:
            raise ValueError(f"privileged must be [N,{PRIVILEGED_DIM}]")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(priv)):
            raise ValueError("Normalizer input contains NaN or Inf")
        features = state[:, :FEATURE_DIM]
        return cls(
            feature_mean=features.mean(0).astype(np.float32),
            feature_std=np.maximum(features.std(0), 1e-6).astype(np.float32),
            privileged_mean=priv.mean(0).astype(np.float32),
            privileged_std=np.maximum(priv.std(0), 1e-6).astype(np.float32),
            clip=float(clip),
            sample_count=int(state.shape[0]),
        )

    def state_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.feature_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.feature_std, device=value.device, dtype=value.dtype)
        features = torch.clamp((value[..., :FEATURE_DIM] - mean) / std, -self.clip, self.clip)
        base = torch.clamp(value[..., FEATURE_DIM:], -1.0, 1.0)
        return torch.cat((features, base), dim=-1)

    def privileged_tensor(self, value: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.privileged_mean, device=value.device, dtype=value.dtype)
        std = torch.as_tensor(self.privileged_std, device=value.device, dtype=value.dtype)
        return torch.clamp((value - mean) / std, -self.clip, self.clip)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean,
            "feature_std": self.feature_std,
            "privileged_mean": self.privileged_mean,
            "privileged_std": self.privileged_std,
            "clip": self.clip,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrozenNormalizer":
        result = cls(
            feature_mean=np.asarray(value["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(value["feature_std"], dtype=np.float32),
            privileged_mean=np.asarray(value["privileged_mean"], dtype=np.float32),
            privileged_std=np.asarray(value["privileged_std"], dtype=np.float32),
            clip=float(value["clip"]),
            sample_count=int(value["sample_count"]),
        )
        if result.feature_mean.shape != (FEATURE_DIM,) or result.feature_std.shape != (FEATURE_DIM,):
            raise ValueError("RLPD checkpoint feature normalizer shape is incompatible")
        if result.privileged_mean.shape != (PRIVILEGED_DIM,) or result.privileged_std.shape != (PRIVILEGED_DIM,):
            raise ValueError("RLPD checkpoint privileged normalizer shape is incompatible")
        if (
            not np.all(np.isfinite(result.feature_mean))
            or not np.all(np.isfinite(result.feature_std))
            or not np.all(np.isfinite(result.privileged_mean))
            or not np.all(np.isfinite(result.privileged_std))
            or np.any(result.feature_std <= 0.0)
            or np.any(result.privileged_std <= 0.0)
            or not np.isfinite(result.clip)
            or result.clip <= 0.0
            or result.sample_count < 1
        ):
            raise ValueError("RLPD checkpoint normalizer contains invalid values")
        return result


class GaussianActor(nn.Module):
    def __init__(self, initial_log_std: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(STATE_DIM, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, ACTION_DIM)
        self.log_std = nn.Linear(256, ACTION_DIM)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, float(initial_log_std))

    def distribution(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(state)
        return self.mean(hidden), torch.clamp(self.log_std(hidden), LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        mean, log_std = self.distribution(state)
        if deterministic:
            return torch.tanh(mean), None
        dist = torch.distributions.Normal(mean, log_std.exp())
        before_tanh = dist.rsample()
        action = torch.tanh(before_tanh)
        log_prob = dist.log_prob(before_tanh) - torch.log(torch.clamp(1.0 - action.square(), min=1e-6))
        return action, log_prob.sum(-1, keepdim=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.distribution(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, layer_norm: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        width = STATE_DIM + PRIVILEGED_DIM + ACTION_DIM
        for output in (256, 256):
            layers.append(nn.Linear(width, output))
            if layer_norm:
                layers.append(nn.LayerNorm(output))
            layers.append(nn.ReLU())
            width = output
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, privileged: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, privileged, action), dim=-1))


class CriticEnsemble(nn.Module):
    def __init__(self, count: int, layer_norm: bool) -> None:
        super().__init__()
        self.qs = nn.ModuleList(Critic(layer_norm) for _ in range(count))

    def forward(self, state: torch.Tensor, privileged: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(state, privileged, action) for q in self.qs], dim=0)


class RLPDLearner:
    def __init__(self, config: AlgorithmConfig, normalizer: FrozenNormalizer, device: str | torch.device) -> None:
        config.validate()
        self.config = config
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.actor = GaussianActor(config.initial_log_std).to(self.device)
        self.critic = CriticEnsemble(config.num_qs, config.critic_layer_norm).to(self.device)
        self.target_critic = CriticEnsemble(config.num_qs, config.critic_layer_norm).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.log_temperature = torch.tensor(
            np.log(config.initial_temperature), device=self.device, dtype=torch.float32, requires_grad=True
        )
        self.temperature_optimizer = torch.optim.Adam([self.log_temperature], lr=config.temperature_lr)
        self.update_groups = 0
        self.critic_updates = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def _tensor(self, batch: dict[str, np.ndarray], name: str) -> torch.Tensor:
        return torch.as_tensor(batch[name], dtype=torch.float32, device=self.device)

    def _critic_update(self, batch: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, float]:
        state = self.normalizer.state_tensor(self._tensor(batch, "state"))
        next_state = self.normalizer.state_tensor(self._tensor(batch, "next_state"))
        privileged = self.normalizer.privileged_tensor(self._tensor(batch, "privileged"))
        next_privileged = self.normalizer.privileged_tensor(self._tensor(batch, "next_privileged"))
        action = self._tensor(batch, "action")
        reward = self._tensor(batch, "reward").reshape(-1, 1)
        terminated = self._tensor(batch, "terminated").reshape(-1, 1)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            assert next_log_prob is not None
            selected = rng.choice(self.config.num_qs, size=self.config.num_min_qs, replace=False)
            selected_tensor = torch.as_tensor(
                selected, dtype=torch.long, device=self.device
            )
            target_qs = self.target_critic(
                next_state, next_privileged, next_action
            ).index_select(0, selected_tensor)
            next_q = target_qs.min(dim=0).values
            if self.config.backup_entropy:
                next_q = next_q - self.temperature.detach() * next_log_prob
            target = reward + self.config.discount * (1.0 - terminated) * next_q
        predicted = self.critic(state, privileged, action)
        loss = ((predicted - target.unsqueeze(0)) ** 2).mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.config.tau)
        self.critic_updates += 1
        return {"critic_loss": float(loss.detach().cpu()), "q_mean": float(predicted.mean().detach().cpu())}

    def _actor_temperature_update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        state = self.normalizer.state_tensor(self._tensor(batch, "state"))
        privileged = self.normalizer.privileged_tensor(self._tensor(batch, "privileged"))
        action, log_prob = self.actor.sample(state)
        assert log_prob is not None
        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        try:
            qs = self.critic(state, privileged, action)
            actor_loss = (
                self.temperature.detach() * log_prob - qs.mean(dim=0)
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
        temperature_loss = -(
            self.log_temperature * (log_prob.detach() + self.config.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
            "entropy": float((-log_prob).mean().detach().cpu()),
        }

    def update_group(self, batch: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, float]:
        total = int(batch["state"].shape[0])
        expected = self.config.batch_size * self.config.utd_ratio
        if total != expected:
            raise ValueError(f"RLPD update group requires {expected} samples, got {total}")
        order = rng.permutation(total)
        critic_sums: dict[str, float] = {}
        last: dict[str, np.ndarray] | None = None
        for index in range(self.config.utd_ratio):
            selected = order[index * self.config.batch_size:(index + 1) * self.config.batch_size]
            last = {name: values[selected] for name, values in batch.items()}
            metrics = self._critic_update(last, rng)
            for name, value in metrics.items():
                critic_sums[name] = critic_sums.get(name, 0.0) + value
        assert last is not None
        metrics = {name: value / self.config.utd_ratio for name, value in critic_sums.items()}
        metrics.update(self._actor_temperature_update(last))
        self.update_groups += 1
        return metrics

    def checkpoint(self, contract: dict[str, Any], replay_high_watermark: int) -> dict[str, Any]:
        return {
            "kind": "franka_real_rlpd_residual",
            "format_version": 1,
            "contract": contract,
            "algorithm": asdict(self.config),
            "normalizer": self.normalizer.to_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_groups": self.update_groups,
            "critic_updates": self.critic_updates,
            "replay_high_watermark": int(replay_high_watermark),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        }

    def restore(self, payload: dict[str, Any]) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        with torch.no_grad():
            self.log_temperature.copy_(torch.as_tensor(payload["log_temperature"], device=self.device))
        self.temperature_optimizer.load_state_dict(payload["temperature_optimizer"])
        self.update_groups = int(payload["update_groups"])
        self.critic_updates = int(payload["critic_updates"])
        if payload.get("torch_rng_state") is not None:
            torch.set_rng_state(payload["torch_rng_state"].cpu())
        cuda_states = payload.get("cuda_rng_state_all")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


class DeterministicActorArtifact(nn.Module):
    def __init__(self, actor: GaussianActor, normalizer: FrozenNormalizer) -> None:
        super().__init__()
        self.actor = actor.eval()
        self.register_buffer("feature_mean", torch.as_tensor(normalizer.feature_mean))
        self.register_buffer("feature_std", torch.as_tensor(normalizer.feature_std))
        self.clip = float(normalizer.clip)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.clamp(
            (state[..., :FEATURE_DIM] - self.feature_mean) / self.feature_std,
            -self.clip,
            self.clip,
        )
        normalized = torch.cat((features, torch.clamp(state[..., FEATURE_DIM:], -1.0, 1.0)), -1)
        return self.actor(normalized)
