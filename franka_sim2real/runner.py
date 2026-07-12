from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Sim2RealConfig
from .envs import make_env
from .policies import PolicyStepContext, build_policy, policy_output_to_action
from .safety import apply_safety_limits


@dataclass
class EpisodeSummary:
    episode_index: int
    steps: int
    total_reward: float
    success: bool
    final_position_error_m: float
    final_yaw_error_deg: float


def _make_run_dir(config: Sim2RealConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.runner.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = Path(__file__).resolve().parents[1] / log_dir
    run_dir = log_dir / f"{timestamp}_{config.runner.run_name}_{config.backend.kind}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_validation(config: Sim2RealConfig) -> dict[str, Any]:
    run_dir = _make_run_dir(config)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    env = make_env(config)
    policy = build_policy(config.policy)
    episode_summaries: list[EpisodeSummary] = []

    with (run_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode_index in range(config.runner.episodes):
            observation = env.reset(episode_index=episode_index)
            total_reward = 0.0
            last_info: dict[str, Any] = {}
            success = False
            executed_steps = 0

            for step_index in range(config.runner.max_steps):
                context = PolicyStepContext(
                    episode_index=episode_index,
                    step_index=step_index,
                    config=config,
                )
                raw_output = policy.act(observation, context)
                action = policy_output_to_action(raw_output, observation, config)
                action = apply_safety_limits(action, config.control, observation)
                next_observation, reward, done, info = env.step(action)
                total_reward += reward
                executed_steps = step_index + 1
                last_info = info
                success = bool(info.get("success", done))

                record = {
                    "episode_index": episode_index,
                    "step_index": step_index,
                    "action": action.to_dict(),
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "observation": next_observation.to_dict(),
                }
                handle.write(json.dumps(record) + "\n")
                observation = next_observation

                if done:
                    break

            episode_summaries.append(
                EpisodeSummary(
                    episode_index=episode_index,
                    steps=executed_steps,
                    total_reward=total_reward,
                    success=success,
                    final_position_error_m=float(last_info.get("position_error_m", 0.0)),
                    final_yaw_error_deg=float(last_info.get("yaw_error_deg", 0.0)),
                )
            )

    env.close()

    success_rate = (
        sum(1 for summary in episode_summaries if summary.success) / len(episode_summaries)
        if episode_summaries
        else 0.0
    )
    summary = {
        "backend": config.backend.kind,
        "run_dir": str(run_dir),
        "episodes": [asdict(summary) for summary in episode_summaries],
        "success_rate": success_rate,
        "mean_total_reward": (
            sum(summary.total_reward for summary in episode_summaries) / len(episode_summaries)
            if episode_summaries
            else 0.0
        ),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
