from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .types import WorkspaceLimits


@dataclass
class BackendConfig:
    kind: str = "sim"
    robot_ip: str = "172.16.0.2"
    realtime: str = "ignore"
    enable_gripper: bool = True
    auto_recover: bool = False
    auto_gripper_homing: bool = False
    async_gripper_commands: bool = False
    settle_time_s: float = 0.2


@dataclass
class RunnerConfig:
    episodes: int = 2
    max_steps: int = 20
    log_dir: str = "runs"
    run_name: str = "sim2real_validation"


@dataclass
class ControlConfig:
    speed: float = 0.03
    gripper_speed: float = 0.03
    gripper_force: float = 20.0
    gripper_command_tolerance_m: float = 0.0001
    max_dx: float = 0.01
    max_dy: float = 0.01
    max_dz: float = 0.01
    max_yaw_deg: float = 10.0
    fallback_gripper_max_width: float = 0.08
    workspace: WorkspaceLimits = field(default_factory=WorkspaceLimits)


@dataclass
class GoalConfig:
    target_translation: list[float] = field(default_factory=lambda: [0.45, 0.0, 0.18])
    target_yaw_deg: float = 0.0
    translation_tolerance_m: float = 0.01
    yaw_tolerance_deg: float = 5.0


@dataclass
class PolicyConfig:
    kind: str = "goal_tracking"
    action_mode: str = "normalized"
    seed: int = 7
    callable: str | None = None
    checkpoint_path: str | None = None
    device: str = "cpu"
    observation_keys: list[str] = field(default_factory=list)
    output_key: str | None = None
    expects_batch_dim: bool = True
    scripted_actions: list[Any] = field(default_factory=list)


@dataclass
class SimConfig:
    initial_translation: list[float] = field(default_factory=lambda: [0.38, 0.05, 0.13])
    initial_yaw_deg: float = 0.0
    observation_noise_std: float = 0.001
    action_noise_std: float = 0.0005
    yaw_noise_deg: float = 0.5
    gripper_max_width: float = 0.056
    seed: int = 7


@dataclass
class Sim2RealConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    goal: GoalConfig = field(default_factory=GoalConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    sim: SimConfig = field(default_factory=SimConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Sim2RealConfig":
        backend = BackendConfig(**data.get("backend", {}))
        runner = RunnerConfig(**data.get("runner", {}))

        control_raw = dict(data.get("control", {}))
        workspace_raw = control_raw.pop("workspace", {})
        workspace = WorkspaceLimits(**workspace_raw)
        control = ControlConfig(workspace=workspace, **control_raw)

        goal = GoalConfig(**data.get("goal", {}))
        policy = PolicyConfig(**data.get("policy", {}))
        sim = SimConfig(**data.get("sim", {}))
        return cls(
            backend=backend,
            runner=runner,
            control=control,
            goal=goal,
            policy=policy,
            sim=sim,
        )


def load_config(path: str | Path) -> Sim2RealConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Sim2RealConfig.from_dict(data)
