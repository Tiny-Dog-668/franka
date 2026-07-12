from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import Sim2RealConfig
from .envs.franka_real import RealFrankaEnv
from .safety import apply_safety_limits
from .types import RobotAction, RobotObservation

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "deploy_bundle_e2e"


@dataclass
class BundleCameraConfig:
    source: str = "realsense"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 0
    image_path: str | None = None
    save_rgb: bool = True
    enable_crop: bool = True
    crop_left: int = 0
    crop_top: int = 0
    crop_width: int | None = None
    crop_height: int | None = None


@dataclass
class BundleActionAdapterConfig:
    labels: list[str] = field(default_factory=lambda: ["dx", "dy", "dz", "yaw_deg", "gripper"])
    scales: list[float] = field(default_factory=lambda: [0.005, 0.005, 0.005, 5.0, 1.0])
    clip_low: list[float] = field(default_factory=lambda: [-1.0, -1.0, -1.0, -1.0, -1.0])
    clip_high: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])
    gripper_mode: str = "binary"
    gripper_binary_threshold: float = 0.0


@dataclass
class BundleRunnerConfig:
    steps: int = 20
    log_dir: str = "runs"
    run_name: str = "e2e_bundle_real"


@dataclass
class BundleModelConfig:
    model_path: str = str(BUNDLE_DIR / "policy_actor_e2e.pt")
    metadata_path: str = str(BUNDLE_DIR / "policy_actor_e2e.json")
    device: str = "cpu"
    history_source: str = "clipped_action"


@dataclass
class BundleDeployConfig:
    robot_ip: str = "172.16.0.2"
    realtime: str = "ignore"
    speed: float = 0.03
    gripper_speed: float = 0.03
    gripper_force: float = 20.0
    gripper_command_tolerance_m: float = 0.0001
    settle_time_s: float = 0.2
    auto_recover: bool = False
    auto_gripper_homing: bool = True
    async_gripper_commands: bool = False
    camera: BundleCameraConfig = field(default_factory=BundleCameraConfig)
    action_adapter: BundleActionAdapterConfig = field(default_factory=BundleActionAdapterConfig)
    runner: BundleRunnerConfig = field(default_factory=BundleRunnerConfig)
    model: BundleModelConfig = field(default_factory=BundleModelConfig)
    workspace: dict[str, list[float]] = field(
        default_factory=lambda: {
            "minimum": [0.2, -0.3, 0.05],
            "maximum": [0.65, 0.3, 0.45],
        }
    )
    fallback_gripper_max_width: float = 0.056

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BundleDeployConfig":
        camera = BundleCameraConfig(**data.get("camera", {}))
        action_adapter = BundleActionAdapterConfig(**data.get("action_adapter", {}))
        runner = BundleRunnerConfig(**data.get("runner", {}))
        model = BundleModelConfig(**data.get("model", {}))
        top_level = dict(data)
        top_level.pop("camera", None)
        top_level.pop("action_adapter", None)
        top_level.pop("runner", None)
        top_level.pop("model", None)
        return cls(
            camera=camera,
            action_adapter=action_adapter,
            runner=runner,
            model=model,
            **top_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bundle_config(path: str | Path) -> BundleDeployConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return BundleDeployConfig.from_dict(json.load(handle))


def _apply_camera_crop(
    rgb: np.ndarray,
    camera_config: BundleCameraConfig,
    output_width: int | None = None,
    output_height: int | None = None,
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    target_size = (
        int(output_width) if output_width is not None else camera_config.width,
        int(output_height) if output_height is not None else camera_config.height,
    )

    if not camera_config.enable_crop:
        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            pil_image = Image.fromarray(image)
            return np.asarray(pil_image.resize(target_size, Image.BILINEAR), dtype=np.uint8)
        return image

    if camera_config.crop_width is None or camera_config.crop_height is None:
        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            pil_image = Image.fromarray(image)
            return np.asarray(pil_image.resize(target_size, Image.BILINEAR), dtype=np.uint8)
        return image

    src_height, src_width = image.shape[:2]
    left = max(0, min(int(camera_config.crop_left), max(src_width - 1, 0)))
    top = max(0, min(int(camera_config.crop_top), max(src_height - 1, 0)))
    crop_width = max(1, int(camera_config.crop_width))
    crop_height = max(1, int(camera_config.crop_height))
    right = min(left + crop_width, src_width)
    bottom = min(top + crop_height, src_height)

    cropped = image[top:bottom, left:right]
    if cropped.size == 0:
        raise ValueError(
            "Camera crop produced an empty image. "
            f"Got left={left}, top={top}, width={crop_width}, height={crop_height} "
            f"for source size {src_width}x{src_height}."
        )

    if cropped.shape[1] != target_size[0] or cropped.shape[0] != target_size[1]:
        pil_image = Image.fromarray(cropped)
        cropped = np.asarray(pil_image.resize(target_size, Image.BILINEAR), dtype=np.uint8)
    return cropped


class RealSenseRGBCamera:
    def __init__(
        self,
        camera_config: BundleCameraConfig,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.camera_config = camera_config
        self.output_width = output_width
        self.output_height = output_height
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if camera_config.serial:
            self.config.enable_device(camera_config.serial)
        self.config.enable_stream(
            rs.stream.color,
            camera_config.width,
            camera_config.height,
            rs.format.rgb8,
            camera_config.fps,
        )
        self.pipeline.start(self.config)
        for _ in range(max(0, int(camera_config.warmup_frames))):
            self.pipeline.wait_for_frames()

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame from RealSense.")
        rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
        return _apply_camera_crop(
            rgb,
            self.camera_config,
            output_width=self.output_width,
            output_height=self.output_height,
        )

    def close(self) -> None:
        self.pipeline.stop()


class StaticRGBCamera:
    def __init__(
        self,
        image_path: str,
        camera_config: BundleCameraConfig,
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        image = Image.open(image_path).convert("RGB")
        self.image = _apply_camera_crop(
            np.asarray(image, dtype=np.uint8),
            camera_config,
            output_width=output_width,
            output_height=output_height,
        )

    def read(self) -> np.ndarray:
        return self.image.copy()

    def close(self) -> None:
        return None


class BundleTorchScriptPolicy:
    def __init__(self, model_path: str, metadata_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

        signature = self.metadata["input_signature"]
        output_signature = self.metadata.get("output_signature", {})
        self.history_dim = int(signature["action_history"][0])
        self.proprio_dim = int(signature["proprio_obs"][0])
        self.rgb_height, self.rgb_width, _ = signature["wrist_rgb"]
        self.action_dim = int(output_signature.get("mean_actions", [self.history_dim])[0])

    def predict(
        self,
        action_history: np.ndarray,
        proprio_obs: np.ndarray,
        wrist_rgb: np.ndarray,
    ) -> np.ndarray:
        action_tensor = torch.as_tensor(action_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        proprio_tensor = torch.as_tensor(proprio_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        # RealSense/Pillow arrays can be backed by a read-only buffer. PyTorch
        # warns because a tensor created from that buffer could otherwise have
        # undefined behavior if it were ever mutated.
        writable_rgb = np.array(wrist_rgb, dtype=np.uint8, order="C", copy=True)
        rgb_tensor = torch.as_tensor(writable_rgb, dtype=torch.uint8, device=self.device).unsqueeze(0)

        with torch.inference_mode():
            output = self.model(action_tensor, proprio_tensor, rgb_tensor)

        output = output.detach().cpu().numpy().reshape(-1)
        return output


def build_bundle_inputs(
    observation: RobotObservation,
    previous_action_history: np.ndarray,
    proprio_dim: int,
    history_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    proprio = np.asarray(
        observation.joint_positions
        + observation.joint_velocities
        + [observation.gripper_width if observation.gripper_width is not None else 0.0],
        dtype=np.float32,
    )
    if proprio.shape[0] != proprio_dim:
        raise ValueError(f"Expected proprio_obs with {proprio_dim} dims, got {proprio.shape[0]}")

    action_history = np.asarray(previous_action_history, dtype=np.float32).reshape(-1)
    if action_history.shape[0] != history_dim:
        raise ValueError(f"Expected action_history with {history_dim} dims, got {action_history.shape[0]}")
    return action_history, proprio


def _action_adapter_dim(config: BundleDeployConfig) -> int:
    lengths = {
        "labels": len(config.action_adapter.labels),
        "scales": len(config.action_adapter.scales),
        "clip_low": len(config.action_adapter.clip_low),
        "clip_high": len(config.action_adapter.clip_high),
    }
    dims = set(lengths.values())
    if len(dims) != 1:
        length_summary = ", ".join(f"{name}={value}" for name, value in lengths.items())
        raise ValueError(f"action_adapter fields must have the same length, got {length_summary}")
    return lengths["labels"]


def _action_scale_lookup(config: BundleDeployConfig) -> dict[str, float]:
    return {
        label: abs(float(scale))
        for label, scale in zip(config.action_adapter.labels, config.action_adapter.scales)
    }


def _runtime_control_dict(config: BundleDeployConfig) -> dict[str, Any]:
    scale_lookup = _action_scale_lookup(config)
    return {
        "speed": config.speed,
        "gripper_speed": config.gripper_speed,
        "gripper_force": config.gripper_force,
        "gripper_command_tolerance_m": config.gripper_command_tolerance_m,
        "max_dx": scale_lookup.get("dx", 0.0),
        "max_dy": scale_lookup.get("dy", 0.0),
        "max_dz": scale_lookup.get("dz", 0.0),
        "max_yaw_deg": scale_lookup.get("yaw_deg", 0.0),
        "fallback_gripper_max_width": config.fallback_gripper_max_width,
        "workspace": config.workspace,
    }


def _validate_bundle_action_dims(bundle: BundleTorchScriptPolicy, config: BundleDeployConfig) -> None:
    adapter_dim = _action_adapter_dim(config)
    if bundle.action_dim != adapter_dim:
        raise ValueError(
            f"Model output dim {bundle.action_dim} does not match action_adapter dim {adapter_dim}"
        )
    if config.model.history_source in {"raw_action", "clipped_action"} and bundle.history_dim != bundle.action_dim:
        raise ValueError(
            "model.history_source requires action_history dim to match model output dim, "
            f"got history_dim={bundle.history_dim}, output_dim={bundle.action_dim}"
        )


def clip_policy_action(
    raw_action: np.ndarray,
    config: BundleDeployConfig,
) -> np.ndarray:
    action_dim = _action_adapter_dim(config)
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    clip_low = np.asarray(config.action_adapter.clip_low, dtype=np.float32).reshape(-1)
    clip_high = np.asarray(config.action_adapter.clip_high, dtype=np.float32).reshape(-1)
    if raw.shape[0] != action_dim:
        raise ValueError(f"Expected {action_dim} raw action values, got {raw.shape[0]}")
    if clip_low.shape[0] != raw.shape[0] or clip_high.shape[0] != raw.shape[0]:
        raise ValueError(f"clip_low and clip_high must both have {action_dim} values.")
    return np.clip(raw, clip_low, clip_high)


def raw_action_to_robot_action(
    raw_action: np.ndarray,
    observation: RobotObservation,
    config: BundleDeployConfig,
    gripper_reference_width: float | None = None,
) -> RobotAction:
    clipped = clip_policy_action(raw_action, config)
    values = clipped.tolist()
    clipped_by_label = dict(zip(config.action_adapter.labels, values))

    scaled: dict[str, float] = {}
    for label, scale, value in zip(config.action_adapter.labels, config.action_adapter.scales, values):
        scaled[label] = float(value) * float(scale)

    action = RobotAction(
        dx=scaled.get("dx", 0.0),
        dy=-scaled.get("dy", 0.0),
        dz=-scaled.get("dz", 0.0),
        yaw_deg=-scaled.get("yaw_deg", 0.0),
        speed=config.speed,
        gripper_speed=config.gripper_speed,
        gripper_force=config.gripper_force,
        metadata={
            "raw_action": np.asarray(raw_action, dtype=np.float32).reshape(-1).tolist(),
            "clipped_action": values,
            "scaled_action": scaled,
        },
    )

    if "gripper" in scaled:
        current_width = (
            float(gripper_reference_width)
            if gripper_reference_width is not None
            else (observation.gripper_width or 0.0)
        )
        max_width = observation.gripper_max_width or config.fallback_gripper_max_width
        raw_gripper = float(clipped_by_label["gripper"])

        if config.action_adapter.gripper_mode == "absolute_width":
            action.gripper_width = ((raw_gripper + 1.0) / 2.0) * max_width
        elif config.action_adapter.gripper_mode == "delta_width":
            action.gripper_width = current_width + scaled["gripper"]
        elif config.action_adapter.gripper_mode == "binary":
            action.gripper_width = (
                max_width
                if raw_gripper >= config.action_adapter.gripper_binary_threshold
                else 0.0
            )
        else:
            raise ValueError(
                f"Unsupported gripper_mode: {config.action_adapter.gripper_mode}"
            )

    runtime_config = Sim2RealConfig.from_dict(
        {
            "backend": {
                "kind": "real",
                "robot_ip": config.robot_ip,
                "realtime": config.realtime,
                "enable_gripper": True,
                "auto_recover": config.auto_recover,
                "auto_gripper_homing": config.auto_gripper_homing,
                "async_gripper_commands": config.async_gripper_commands,
                "settle_time_s": config.settle_time_s,
            },
            "control": _runtime_control_dict(config),
        }
    )
    return apply_safety_limits(action, runtime_config.control, observation)


def _make_camera(
    camera_config: BundleCameraConfig,
    output_width: int | None = None,
    output_height: int | None = None,
):
    if camera_config.source == "realsense":
        return RealSenseRGBCamera(
            camera_config,
            output_width=output_width,
            output_height=output_height,
        )
    if camera_config.source == "image":
        if not camera_config.image_path:
            raise ValueError("camera.image_path must be set when source='image'")
        return StaticRGBCamera(
            camera_config.image_path,
            camera_config,
            output_width=output_width,
            output_height=output_height,
        )
    raise ValueError(f"Unsupported camera source: {camera_config.source}")


def _make_run_dir(config: BundleDeployConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(config.runner.log_dir).expanduser()
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    run_dir = log_dir / f"{timestamp}_{config.runner.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_bundle_deploy(
    config: BundleDeployConfig,
    execute_motion: bool = True,
    confirm_step_callback: Any | None = None,
    save_step_data: bool = False,
) -> dict[str, Any]:
    run_dir = _make_run_dir(config)
    (run_dir / "rgb").mkdir(exist_ok=True)
    if save_step_data:
        (run_dir / "step_data").mkdir(exist_ok=True)
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2)

    env_config = Sim2RealConfig.from_dict(
        {
            "backend": {
                "kind": "real",
                "robot_ip": config.robot_ip,
                "realtime": config.realtime,
                "enable_gripper": True,
                "auto_recover": config.auto_recover,
                "auto_gripper_homing": config.auto_gripper_homing if execute_motion else False,
                "async_gripper_commands": config.async_gripper_commands if execute_motion else False,
                "settle_time_s": config.settle_time_s,
            },
            "control": _runtime_control_dict(config),
        }
    )

    bundle = BundleTorchScriptPolicy(
        model_path=config.model.model_path,
        metadata_path=config.model.metadata_path,
        device=config.model.device,
    )
    _validate_bundle_action_dims(bundle, config)
    env = RealFrankaEnv(env_config)
    camera = _make_camera(
        config.camera,
        output_width=bundle.rgb_width,
        output_height=bundle.rgb_height,
    )

    previous_action_history = np.zeros(bundle.history_dim, dtype=np.float32)
    observation = env.reset()
    desired_gripper_width = observation.gripper_width
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "steps": [],
    }

    with (run_dir / "rollout.jsonl").open("w", encoding="utf-8") as log_file:
        for step_index in range(config.runner.steps):
            step_timing: dict[str, float] = {}

            image_start = time.perf_counter()
            wrist_rgb = camera.read()
            step_timing["image_read_ms"] = (time.perf_counter() - image_start) * 1000.0
            if wrist_rgb.shape[:2] != (bundle.rgb_height, bundle.rgb_width):
                raise ValueError(
                    f"Expected RGB frame {(bundle.rgb_height, bundle.rgb_width)}, got {wrist_rgb.shape[:2]}"
                )
            rgb_path = run_dir / "rgb" / f"step_{step_index:04d}.png"
            rgb_relpath = str(rgb_path.relative_to(run_dir))
            if config.camera.save_rgb or save_step_data:
                Image.fromarray(wrist_rgb).save(rgb_path)

            pack_start = time.perf_counter()
            action_history, proprio = build_bundle_inputs(
                observation,
                previous_action_history,
                proprio_dim=bundle.proprio_dim,
                history_dim=bundle.history_dim,
            )
            step_timing["input_pack_ms"] = (time.perf_counter() - pack_start) * 1000.0

            infer_start = time.perf_counter()
            raw_action = bundle.predict(action_history, proprio, wrist_rgb)
            step_timing["policy_infer_ms"] = (time.perf_counter() - infer_start) * 1000.0

            clip_start = time.perf_counter()
            clipped_action = clip_policy_action(raw_action, config)
            robot_action = raw_action_to_robot_action(
                raw_action,
                observation,
                config,
                gripper_reference_width=(
                    desired_gripper_width if config.async_gripper_commands else None
                ),
            )
            if config.async_gripper_commands and robot_action.gripper_width is not None:
                desired_gripper_width = robot_action.gripper_width
            step_timing["action_map_ms"] = (time.perf_counter() - clip_start) * 1000.0

            step_data_relpath: str | None = None
            if save_step_data:
                step_data_path = run_dir / "step_data" / f"step_{step_index:04d}.npz"
                np.savez_compressed(
                    step_data_path,
                    action_history=np.asarray(action_history, dtype=np.float32),
                    proprio_obs=np.asarray(proprio, dtype=np.float32),
                    wrist_rgb=np.asarray(wrist_rgb, dtype=np.uint8),
                    raw_action=np.asarray(raw_action, dtype=np.float32),
                    clipped_action=np.asarray(clipped_action, dtype=np.float32),
                )
                step_data_relpath = str(step_data_path.relative_to(run_dir))

            info: dict[str, Any]
            if not execute_motion:
                next_observation = observation
                reward = 0.0
                done = True
                info = {
                    "preview_only": True,
                    "motion_executed": False,
                    "timing": {
                        "arm_move_ms": 0.0,
                        "gripper_move_ms": 0.0,
                        "settle_ms": 0.0,
                        "state_read_ms": observation.metadata.get("timing", {}).get("state_read_ms", 0.0),
                        "step_total_ms": 0.0,
                    },
                }
            else:
                if confirm_step_callback is not None:
                    should_execute = bool(confirm_step_callback(step_index, robot_action, observation))
                    if not should_execute:
                        next_observation = observation
                        reward = 0.0
                        done = True
                        info = {
                            "preview_only": True,
                            "motion_executed": False,
                            "step_cancelled_by_user": True,
                            "timing": {
                                "arm_move_ms": 0.0,
                                "gripper_move_ms": 0.0,
                                "settle_ms": 0.0,
                                "state_read_ms": observation.metadata.get("timing", {}).get("state_read_ms", 0.0),
                                "step_total_ms": 0.0,
                            },
                        }
                    else:
                        next_observation, reward, done, info = env.step(robot_action)
                        info["motion_executed"] = not bool(info.get("arm_command_failed"))
                else:
                    next_observation, reward, done, info = env.step(robot_action)
                    info["motion_executed"] = not bool(info.get("arm_command_failed"))

            if "timing" not in info:
                info["timing"] = {}
            info["timing"].update(step_timing)

            record = {
                "step_index": step_index,
                "observation_before": observation.to_dict(),
                "model_input": {
                    "action_history": np.asarray(action_history, dtype=np.float32).tolist(),
                    "proprio_obs": np.asarray(proprio, dtype=np.float32).tolist(),
                    "rgb_path": rgb_relpath,
                    "rgb_shape": list(wrist_rgb.shape),
                    "step_data_path": step_data_relpath,
                },
                "raw_action": raw_action.tolist(),
                "clipped_action": clipped_action.tolist(),
                "robot_action": robot_action.to_dict(),
                "reward": reward,
                "done": done,
                "info": info,
                "observation": next_observation.to_dict(),
                "observation_after": next_observation.to_dict(),
            }
            log_file.write(json.dumps(record) + "\n")
            summary["steps"].append(record)

            observation = next_observation
            if config.model.history_source == "raw_action":
                previous_action_history = np.asarray(raw_action, dtype=np.float32)
            elif config.model.history_source == "clipped_action":
                previous_action_history = clipped_action.astype(np.float32)
            elif config.model.history_source == "zeros":
                previous_action_history = np.zeros(bundle.history_dim, dtype=np.float32)
            else:
                raise ValueError(
                    "model.history_source must be one of: raw_action, clipped_action, zeros"
                )

            if done:
                break

    camera.close()
    env.close()
    summary["num_steps"] = len(summary["steps"])
    summary["save_step_data"] = save_step_data
    if summary["steps"]:
        summary["last_info"] = summary["steps"][-1]["info"]
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
