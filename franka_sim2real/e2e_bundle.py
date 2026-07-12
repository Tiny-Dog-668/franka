from __future__ import annotations

import json
import math
import sys
import time
import warnings
from collections import deque
from contextlib import contextmanager
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
    history_scale: float = 1.0
    history_delay_steps: int = 1


@dataclass
class BundleInitialStateConfig:
    enforce: bool = False
    joint_positions: list[float] | None = None
    joint_position_tolerance_rad: float = 0.01
    max_abs_joint_velocity_rad_s: float = 0.02
    tcp_translation: list[float] | None = None
    tcp_translation_tolerance_m: float = 0.005
    tcp_quaternion_xyzw: list[float] | None = None
    tcp_orientation_tolerance_deg: float = 2.0
    gripper_width_m: float | None = None
    gripper_width_tolerance_m: float = 0.002
    minimum_gripper_max_width_m: float | None = None
    required_robot_mode: str | None = None
    require_gripper_not_grasped: bool = False
    require_no_robot_errors: bool = True


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
    initial_state: BundleInitialStateConfig = field(default_factory=BundleInitialStateConfig)
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
        initial_state = BundleInitialStateConfig(**data.get("initial_state", {}))
        top_level = dict(data)
        top_level.pop("camera", None)
        top_level.pop("action_adapter", None)
        top_level.pop("runner", None)
        top_level.pop("model", None)
        top_level.pop("initial_state", None)
        return cls(
            camera=camera,
            action_adapter=action_adapter,
            runner=runner,
            model=model,
            initial_state=initial_state,
            **top_level,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_bundle_config(path: str | Path) -> BundleDeployConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return BundleDeployConfig.from_dict(json.load(handle))


class ActionHistoryBuffer:
    """Track policy history with the same scale and delay used during training."""

    VALID_SOURCES = {"raw_action", "clipped_action", "zeros"}

    def __init__(
        self,
        history_dim: int,
        source: str,
        scale: float = 1.0,
        delay_steps: int = 1,
    ) -> None:
        if isinstance(history_dim, bool) or not isinstance(history_dim, int) or history_dim < 1:
            raise ValueError("history_dim must be a positive integer")
        if source not in self.VALID_SOURCES:
            valid = ", ".join(sorted(self.VALID_SOURCES))
            raise ValueError(f"model.history_source must be one of: {valid}")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise ValueError("model.history_scale must be finite and positive")
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("model.history_scale must be finite and positive")
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 1:
            raise ValueError("model.history_delay_steps must be a positive integer")

        self.history_dim = history_dim
        self.source = source
        self.scale = scale
        self.delay_steps = delay_steps
        zeros = np.zeros(history_dim, dtype=np.float32)
        self._queue: deque[np.ndarray] = deque(
            (zeros.copy() for _ in range(delay_steps)),
            maxlen=delay_steps,
        )

    def current(self) -> np.ndarray:
        return self._queue[0].copy()

    def update(self, raw_action: np.ndarray, clipped_action: np.ndarray) -> None:
        if self.source == "zeros":
            candidate = np.zeros(self.history_dim, dtype=np.float32)
        elif self.source == "raw_action":
            candidate = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        else:
            candidate = np.asarray(clipped_action, dtype=np.float32).reshape(-1)

        if candidate.shape[0] != self.history_dim:
            raise ValueError(
                f"Expected history action with {self.history_dim} values, got {candidate.shape[0]}"
            )
        if not np.all(np.isfinite(candidate)):
            raise ValueError("History action contains NaN or Inf")
        self._queue.append(np.asarray(candidate * self.scale, dtype=np.float32).copy())


def _validate_vector(name: str, values: list[float] | None, expected_dim: int) -> np.ndarray | None:
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape[0] != expected_dim:
        raise ValueError(f"initial_state.{name} must contain exactly {expected_dim} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"initial_state.{name} contains NaN or Inf")
    return vector


def _validate_nonnegative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"initial_state.{name} must be finite and non-negative")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"initial_state.{name} must be finite and non-negative")
    return value


def _quaternion_error_deg(actual_xyzw: np.ndarray, expected_xyzw: np.ndarray) -> float:
    actual_norm = float(np.linalg.norm(actual_xyzw))
    expected_norm = float(np.linalg.norm(expected_xyzw))
    if actual_norm <= 1e-12 or expected_norm <= 1e-12:
        raise ValueError("TCP quaternion norm must be non-zero")
    dot = float(np.dot(actual_xyzw / actual_norm, expected_xyzw / expected_norm))
    dot = min(1.0, max(-1.0, abs(dot)))
    return math.degrees(2.0 * math.acos(dot))


def evaluate_initial_state(
    observation: RobotObservation,
    config: BundleInitialStateConfig,
) -> dict[str, Any]:
    """Evaluate the real robot state before opening the camera or running policy inference."""

    expected_joint_positions = _validate_vector("joint_positions", config.joint_positions, 7)
    expected_tcp_translation = _validate_vector("tcp_translation", config.tcp_translation, 3)
    expected_tcp_quaternion = _validate_vector("tcp_quaternion_xyzw", config.tcp_quaternion_xyzw, 4)
    joint_tolerance = _validate_nonnegative_finite(
        "joint_position_tolerance_rad", config.joint_position_tolerance_rad
    )
    velocity_tolerance = _validate_nonnegative_finite(
        "max_abs_joint_velocity_rad_s", config.max_abs_joint_velocity_rad_s
    )
    translation_tolerance = _validate_nonnegative_finite(
        "tcp_translation_tolerance_m", config.tcp_translation_tolerance_m
    )
    orientation_tolerance = _validate_nonnegative_finite(
        "tcp_orientation_tolerance_deg", config.tcp_orientation_tolerance_deg
    )
    gripper_tolerance = _validate_nonnegative_finite(
        "gripper_width_tolerance_m", config.gripper_width_tolerance_m
    )
    if config.gripper_width_m is not None:
        expected_gripper_width = _validate_nonnegative_finite(
            "gripper_width_m", config.gripper_width_m
        )
    else:
        expected_gripper_width = None
    if config.minimum_gripper_max_width_m is not None:
        minimum_gripper_max_width = _validate_nonnegative_finite(
            "minimum_gripper_max_width_m", config.minimum_gripper_max_width_m
        )
    else:
        minimum_gripper_max_width = None
    if config.required_robot_mode is not None and not str(config.required_robot_mode).strip():
        raise ValueError("initial_state.required_robot_mode must be a non-empty string or null")

    report: dict[str, Any] = {
        "enforced": bool(config.enforce),
        "passed": True,
        "checks": {},
        "failures": [],
    }

    def add_check(name: str, passed: bool, details: dict[str, Any], failure: str) -> None:
        report["checks"][name] = {"passed": bool(passed), **details}
        if not passed:
            report["passed"] = False
            report["failures"].append(failure)

    if config.require_no_robot_errors:
        current_errors = observation.metadata.get("current_errors", [])
        passed = not bool(observation.has_errors) and not bool(current_errors)
        add_check(
            "robot_errors",
            passed,
            {
                "has_errors": bool(observation.has_errors),
                "current_errors": current_errors,
                "last_motion_errors": observation.metadata.get("last_motion_errors", []),
            },
            "robot reports an active error",
        )

    if config.required_robot_mode is not None:
        actual_mode = str(observation.robot_mode)
        expected_mode = str(config.required_robot_mode)
        add_check(
            "robot_mode",
            actual_mode == expected_mode,
            {"actual": actual_mode, "expected": expected_mode},
            f"robot mode is {actual_mode!r}, expected {expected_mode!r}",
        )

    actual_joint_positions = _validate_vector("actual_joint_positions", observation.joint_positions, 7)
    if expected_joint_positions is not None:
        max_error = float(np.max(np.abs(actual_joint_positions - expected_joint_positions)))
        add_check(
            "joint_positions",
            max_error <= joint_tolerance,
            {
                "actual_rad": actual_joint_positions.tolist(),
                "expected_rad": expected_joint_positions.tolist(),
                "max_abs_error_rad": max_error,
                "tolerance_rad": joint_tolerance,
            },
            f"joint position max error {max_error:.6f} rad exceeds {joint_tolerance:.6f} rad",
        )

    actual_joint_velocities = _validate_vector("actual_joint_velocities", observation.joint_velocities, 7)
    max_velocity = float(np.max(np.abs(actual_joint_velocities)))
    add_check(
        "joint_velocities",
        max_velocity <= velocity_tolerance,
        {
            "actual_rad_s": actual_joint_velocities.tolist(),
            "max_abs_rad_s": max_velocity,
            "tolerance_rad_s": velocity_tolerance,
        },
        f"joint velocity {max_velocity:.6f} rad/s exceeds {velocity_tolerance:.6f} rad/s",
    )

    actual_tcp_translation = _validate_vector("actual_tcp_translation", observation.tcp_translation, 3)
    if expected_tcp_translation is not None:
        translation_error = float(np.linalg.norm(actual_tcp_translation - expected_tcp_translation))
        add_check(
            "tcp_translation",
            translation_error <= translation_tolerance,
            {
                "actual_m": actual_tcp_translation.tolist(),
                "expected_m": expected_tcp_translation.tolist(),
                "error_norm_m": translation_error,
                "tolerance_m": translation_tolerance,
            },
            f"TCP translation error {translation_error:.6f} m exceeds {translation_tolerance:.6f} m",
        )

    actual_tcp_quaternion = _validate_vector("actual_tcp_quaternion", observation.tcp_quaternion, 4)
    if expected_tcp_quaternion is not None:
        orientation_error = _quaternion_error_deg(actual_tcp_quaternion, expected_tcp_quaternion)
        add_check(
            "tcp_orientation",
            orientation_error <= orientation_tolerance,
            {
                "actual_xyzw": actual_tcp_quaternion.tolist(),
                "expected_xyzw": expected_tcp_quaternion.tolist(),
                "error_deg": orientation_error,
                "tolerance_deg": orientation_tolerance,
            },
            f"TCP orientation error {orientation_error:.3f} deg exceeds {orientation_tolerance:.3f} deg",
        )

    if expected_gripper_width is not None:
        if observation.gripper_width is None:
            add_check(
                "gripper_width",
                False,
                {
                    "actual_m": None,
                    "expected_m": expected_gripper_width,
                    "tolerance_m": gripper_tolerance,
                },
                "gripper width is unavailable",
            )
        else:
            actual_gripper_width = float(observation.gripper_width)
            gripper_error = abs(actual_gripper_width - expected_gripper_width)
            add_check(
                "gripper_width",
                gripper_error <= gripper_tolerance,
                {
                    "actual_m": actual_gripper_width,
                    "expected_m": expected_gripper_width,
                    "abs_error_m": gripper_error,
                    "tolerance_m": gripper_tolerance,
                },
                f"gripper width error {gripper_error:.6f} m exceeds {gripper_tolerance:.6f} m",
            )

    if minimum_gripper_max_width is not None:
        if observation.gripper_max_width is None:
            add_check(
                "gripper_max_width",
                False,
                {"actual_m": None, "minimum_m": minimum_gripper_max_width},
                "gripper maximum width is unavailable",
            )
        else:
            actual_max_width = float(observation.gripper_max_width)
            add_check(
                "gripper_max_width",
                actual_max_width >= minimum_gripper_max_width,
                {"actual_m": actual_max_width, "minimum_m": minimum_gripper_max_width},
                f"gripper maximum width {actual_max_width:.6f} m is below {minimum_gripper_max_width:.6f} m",
            )

    if config.require_gripper_not_grasped:
        is_grasped = observation.gripper_is_grasped
        add_check(
            "gripper_not_grasped",
            is_grasped is False,
            {"is_grasped": is_grasped},
            "gripper grasp state is unavailable or reports an existing grasp",
        )

    return report


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
    ActionHistoryBuffer(
        history_dim=bundle.history_dim,
        source=config.model.history_source,
        scale=config.model.history_scale,
        delay_steps=config.model.history_delay_steps,
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


@contextmanager
def _managed_deploy_resources(env: RealFrankaEnv, camera: Any):
    """Close the robot environment first and the camera second on every exit path."""

    try:
        yield
    finally:
        active_exception = sys.exc_info()[0] is not None
        close_errors: list[str] = []
        for name, resource in (("robot environment", env), ("camera", camera)):
            try:
                resource.close()
            except Exception as exc:  # pragma: no cover - hardware cleanup failure
                close_errors.append(f"{name}: {exc}")

        if close_errors:
            message = "Failed to close deployment resources cleanly: " + "; ".join(close_errors)
            if active_exception:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
            else:
                raise RuntimeError(message)


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
    action_history_buffer = ActionHistoryBuffer(
        history_dim=bundle.history_dim,
        source=config.model.history_source,
        scale=config.model.history_scale,
        delay_steps=config.model.history_delay_steps,
    )

    env = RealFrankaEnv(env_config)
    try:
        observation = env.reset()
        initial_state_report = evaluate_initial_state(observation, config.initial_state)
    except Exception:
        env.close()
        raise

    initial_state_report_path = run_dir / "initial_state_check.json"
    with initial_state_report_path.open("w", encoding="utf-8") as handle:
        json.dump(initial_state_report, handle, indent=2)

    if config.initial_state.enforce and not initial_state_report["passed"]:
        env.close()
        failure_lines = "\n".join(
            f"  - {failure}" for failure in initial_state_report["failures"]
        )
        raise RuntimeError(
            "Initial-state safety check failed; no policy motion command was sent.\n"
            f"{failure_lines}\n"
            "Move the robot and gripper back to the configured policy initial state, then retry.\n"
            "Suggested command:\n"
            f"  .venv/bin/python scripts/robot/go_to_zero_pose.py --ip {config.robot_ip} "
            "--realtime ignore --speed 0.1 --max-step-rad 0.1 --gripper-speed 0.03"
        )

    try:
        camera = _make_camera(
            config.camera,
            output_width=bundle.rgb_width,
            output_height=bundle.rgb_height,
        )
    except Exception:
        env.close()
        raise

    desired_gripper_width = observation.gripper_width
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "initial_state_check": initial_state_report,
        "steps": [],
    }

    with _managed_deploy_resources(env, camera), (run_dir / "rollout.jsonl").open(
        "w", encoding="utf-8"
    ) as log_file:
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
                action_history_buffer.current(),
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
            action_history_buffer.update(raw_action, clipped_action)

            if done:
                break

    summary["num_steps"] = len(summary["steps"])
    summary["save_step_data"] = save_step_data
    if summary["steps"]:
        summary["last_info"] = summary["steps"][-1]["info"]
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
