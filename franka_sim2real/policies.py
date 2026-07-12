from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import PolicyConfig, Sim2RealConfig
from .types import RobotAction, RobotObservation


@dataclass
class PolicyStepContext:
    episode_index: int
    step_index: int
    config: Sim2RealConfig


class Policy(Protocol):
    def act(self, observation: RobotObservation, context: PolicyStepContext) -> Any:
        ...


class ZeroPolicy:
    def act(self, observation: RobotObservation, context: PolicyStepContext) -> dict[str, float]:
        return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw_deg": 0.0}


class RandomPolicy:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, observation: RobotObservation, context: PolicyStepContext) -> np.ndarray:
        dims = 5 if observation.gripper_max_width is not None else 4
        return self.rng.uniform(-1.0, 1.0, size=dims)


class GoalTrackingPolicy:
    def act(self, observation: RobotObservation, context: PolicyStepContext) -> dict[str, float]:
        target = observation.goal_translation or context.config.goal.target_translation
        dx = target[0] - observation.tcp_translation[0]
        dy = target[1] - observation.tcp_translation[1]
        dz = target[2] - observation.tcp_translation[2]
        yaw_error = context.config.goal.target_yaw_deg - observation.tcp_yaw_deg
        return {
            "dx": dx * 0.5,
            "dy": dy * 0.5,
            "dz": dz * 0.5,
            "yaw_deg": yaw_error * 0.5,
        }


class ScriptedPolicy:
    def __init__(self, scripted_actions: list[Any]) -> None:
        self.scripted_actions = scripted_actions or []

    def act(self, observation: RobotObservation, context: PolicyStepContext) -> Any:
        if not self.scripted_actions:
            return {"dx": 0.0, "dy": 0.0, "dz": 0.0, "yaw_deg": 0.0}
        index = min(context.step_index, len(self.scripted_actions) - 1)
        return self.scripted_actions[index]


class PythonCallablePolicy:
    def __init__(self, callable_path: str) -> None:
        module_name, function_name = callable_path.split(":", 1)
        module = importlib.import_module(module_name)
        self.function = getattr(module, function_name)

    def act(self, observation: RobotObservation, context: PolicyStepContext) -> Any:
        return self.function(observation.to_dict(), context.step_index, context.episode_index, context.config.to_dict())


def _default_observation_vector(observation: RobotObservation) -> list[float]:
    vector = []
    vector.extend(observation.joint_positions)
    vector.extend(observation.joint_velocities)
    vector.extend(observation.tcp_translation)
    vector.extend(observation.tcp_quaternion)
    vector.extend(observation.external_wrench)
    if observation.gripper_width is not None:
        vector.append(observation.gripper_width)
    if observation.gripper_max_width is not None:
        vector.append(observation.gripper_max_width)
    if observation.gripper_is_grasped is not None:
        vector.append(float(observation.gripper_is_grasped))
    return vector


def _extract_observation_field(observation: RobotObservation, key: str) -> list[float]:
    value = getattr(observation, key)
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(item) for item in value]
    if isinstance(value, bool):
        return [float(value)]
    if isinstance(value, (int, float)):
        return [float(value)]
    raise TypeError(f"Unsupported observation field for vectorization: {key}={type(value)!r}")


def observation_to_vector(
    observation: RobotObservation,
    observation_keys: list[str] | None = None,
) -> np.ndarray:
    if not observation_keys:
        return np.asarray(_default_observation_vector(observation), dtype=np.float32)

    vector: list[float] = []
    for key in observation_keys:
        vector.extend(_extract_observation_field(observation, key))
    return np.asarray(vector, dtype=np.float32)


class TorchScriptPolicy:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        observation_keys: list[str] | None = None,
        output_key: str | None = None,
        expects_batch_dim: bool = True,
    ) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "torch is not installed in the current environment. "
                "Install PyTorch before using policy.kind='torchscript'."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.model = torch.jit.load(self.checkpoint_path, map_location=self.device)
        self.model.eval()
        self.observation_keys = observation_keys or []
        self.output_key = output_key
        self.expects_batch_dim = expects_batch_dim

    def act(self, observation: RobotObservation, context: PolicyStepContext) -> Any:
        obs_vec = observation_to_vector(observation, self.observation_keys)
        obs_tensor = self.torch.as_tensor(obs_vec, dtype=self.torch.float32, device=self.device)
        if self.expects_batch_dim:
            obs_tensor = obs_tensor.unsqueeze(0)

        with self.torch.no_grad():
            output = self.model(obs_tensor)

        if self.output_key is not None:
            output = output[self.output_key]

        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()

        if isinstance(output, np.ndarray) and output.ndim > 1:
            output = output[0]
        return output


def build_policy(policy_config: PolicyConfig) -> Policy:
    if policy_config.kind == "zero":
        return ZeroPolicy()
    if policy_config.kind == "random":
        return RandomPolicy(policy_config.seed)
    if policy_config.kind == "goal_tracking":
        return GoalTrackingPolicy()
    if policy_config.kind == "scripted":
        return ScriptedPolicy(policy_config.scripted_actions)
    if policy_config.kind == "python_callable":
        if not policy_config.callable:
            raise ValueError("policy.callable must be set for python_callable policies.")
        return PythonCallablePolicy(policy_config.callable)
    if policy_config.kind == "torchscript":
        if not policy_config.checkpoint_path:
            raise ValueError("policy.checkpoint_path must be set for torchscript policies.")
        return TorchScriptPolicy(
            checkpoint_path=policy_config.checkpoint_path,
            device=policy_config.device,
            observation_keys=policy_config.observation_keys,
            output_key=policy_config.output_key,
            expects_batch_dim=policy_config.expects_batch_dim,
        )
    raise ValueError(f"Unsupported policy kind: {policy_config.kind}")


def _available_gripper_width(observation: RobotObservation, config: Sim2RealConfig) -> float:
    if observation.gripper_max_width not in (None, 0.0):
        return float(observation.gripper_max_width)
    return config.control.fallback_gripper_max_width


def policy_output_to_action(
    output: Any,
    observation: RobotObservation,
    config: Sim2RealConfig,
) -> RobotAction:
    if isinstance(output, RobotAction):
        return output

    if isinstance(output, dict):
        if "action" in output:
            output = output["action"]
        else:
            return RobotAction(
                dx=float(output.get("dx", 0.0)),
                dy=float(output.get("dy", 0.0)),
                dz=float(output.get("dz", 0.0)),
                yaw_deg=float(output.get("yaw_deg", output.get("yaw", 0.0))),
                speed=float(output["speed"]) if "speed" in output else None,
                gripper_width=float(output["gripper_width"]) if "gripper_width" in output else None,
                gripper_speed=float(output["gripper_speed"]) if "gripper_speed" in output else None,
                gripper_force=float(output["gripper_force"]) if "gripper_force" in output else None,
            )

    if isinstance(output, (list, tuple, np.ndarray)):
        values = [float(value) for value in np.asarray(output).reshape(-1).tolist()]
        if len(values) not in (4, 5):
            raise ValueError(f"Expected action length 4 or 5, got {len(values)}")

        if config.policy.action_mode == "normalized":
            action = RobotAction(
                dx=values[0] * config.control.max_dx,
                dy=values[1] * config.control.max_dy,
                dz=values[2] * config.control.max_dz,
                yaw_deg=values[3] * config.control.max_yaw_deg,
                speed=config.control.speed,
            )
            if len(values) == 5:
                max_width = _available_gripper_width(observation, config)
                action.gripper_width = ((values[4] + 1.0) / 2.0) * max_width
                action.gripper_speed = config.control.gripper_speed
                action.gripper_force = config.control.gripper_force
            return action

        action = RobotAction(
            dx=values[0],
            dy=values[1],
            dz=values[2],
            yaw_deg=values[3],
            speed=config.control.speed,
        )
        if len(values) == 5:
            action.gripper_width = values[4]
            action.gripper_speed = config.control.gripper_speed
            action.gripper_force = config.control.gripper_force
        return action

    raise TypeError(f"Unsupported policy output type: {type(output)!r}")
