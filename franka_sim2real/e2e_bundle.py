from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
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
from .types import RobotAction, RobotObservation, print_error_wrench_report

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
class BundleTactileCameraConfig:
    enabled: bool = False
    left_device: int | str = 0
    right_device: int | str = 6
    width: int = 3280
    height: int = 2464
    fps: int = 25
    warmup_frames: int = 5
    first_frame_timeout_s: float = 10.0
    save_rgb: bool = True


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
    history_scale: float | list[float] = 1.0
    history_delay_steps: int = 1
    enforce_policy_contract: bool = False
    # CPU inference latency is not monotonic in thread count, so real-time
    # deployments should pin it rather than inherit torch's host default.
    # None keeps torch's default.
    torch_num_threads: int | None = None
    # Apply TorchScript's inference-only graph optimization after loading. This
    # folds and pre-packs inference operators without changing the saved model.
    optimize_for_inference: bool = False
    # Optional RMA ablation inputs. "oracle" bypasses the visual position
    # prediction and supplies a fixed cube-center XYZ in robot_root metres.
    rma_position_source: str = "vision"
    rma_contact_source: str = "vision"
    rma_oracle_cube_position_root: list[float] | None = None


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
class BundleCollisionBehaviorConfig:
    """Explicit libfranka contact and collision thresholds.

    Cartesian entries are ordered as Fx, Fy, Fz, Mx, My, Mz. The first three
    are forces in N and the last three are torques in Nm.
    """

    lower_torque_thresholds: list[float] = field(default_factory=lambda: [20.0] * 7)
    upper_torque_thresholds: list[float] = field(default_factory=lambda: [40.0] * 7)
    lower_force_thresholds: list[float] = field(default_factory=lambda: [10.0] * 6)
    upper_force_thresholds: list[float] = field(default_factory=lambda: [20.0] * 6)


@dataclass
class BundleStreamingConfig:
    backend: str = "async_position"
    server9_worker_path: str = "dist/franka_server9/franka_server9_streaming_worker"
    # Pin the native libfranka process to the CPU that owns the FCI NIC IRQ.
    # None keeps the operating system's default scheduler affinity.
    server9_control_cpu: int | None = None
    policy_frequency_hz: float = 30.0
    ik_frequency_hz: float = 60.0
    dls_lambda: float = 0.01
    commissioning_action_limit: float = 0.1
    maximum_joint_velocities: list[float] = field(
        default_factory=lambda: [0.655, 0.655, 0.655, 0.655, 1.315, 1.315, 1.315]
    )
    # None preserves the robot's current Desk-configured joint impedance.
    # A supplied array is applied once before Robot::control starts.
    joint_impedance: list[float] | None = None
    # None leaves the controller's current collision behavior untouched. When
    # supplied, server9 applies all four arrays before Robot::control starts so
    # the active thresholds are deterministic and auditable in the run config.
    collision_behavior: BundleCollisionBehaviorConfig | None = None
    # Per-joint limits used by the native 1 kHz trajectory generator.  These
    # default to libfranka's official maxima for legacy configurations.
    maximum_joint_accelerations: list[float] = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    )
    maximum_joint_jerks: list[float] = field(
        default_factory=lambda: [5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0, 5000.0]
    )
    maximum_joint_target_delta_rad: float = 0.02
    joint_limit_margin_rad: float = 0.05
    policy_watchdog_s: float = 0.1
    control_watchdog_s: float = 0.05
    # "joint_position_pursuit" chases a reference a fixed joint delta ahead of
    # the measured position. libfranka::limitRate then sees an implied velocity
    # of delta/period, which saturates maximum_joint_velocities for any usable
    # delta, so the joint speed no longer depends on how large the policy action
    # was. "sim_actuator_velocity" instead reproduces the Isaac Lab implicit-PD
    # steady state, qd = (stiffness / damping) * dq_ik, so the joint speed is
    # proportional to the action the way it was during training.
    control_law: str = "joint_position_pursuit"
    # The simulator's stiffness/damping ratio. FRANKA_PANDA_HIGH_PD_CFG uses
    # 400/80, so a saturated action settles at 5 * 0.05 m = 0.25 m/s of TCP
    # speed, which is 15% of the Panda's 1.7 m/s Cartesian limit.
    reference_velocity_gain: float = 5.0
    # Reference limiting on the DLS delta before the gain, in the spirit of
    # serl_franka_controllers. It guards against a large pose error demanding an
    # unbounded velocity and must stay wide enough not to bind at the trained
    # action scale, or it discards the DLS magnitude the way the per-tick joint
    # clamp does.
    maximum_ik_reference_delta_rad: float = 0.25


@dataclass
class BundleDeployConfig:
    robot_ip: str = "172.16.0.2"
    realtime: str = "ignore"
    control_mode: str = "blocking"
    speed: float = 0.03
    gripper_speed: float = 0.03
    gripper_force: float = 20.0
    gripper_command_tolerance_m: float = 0.0001
    settle_time_s: float = 0.2
    auto_recover: bool = False
    auto_gripper_homing: bool = True
    async_gripper_commands: bool = False
    camera: BundleCameraConfig = field(default_factory=BundleCameraConfig)
    tactile_camera: BundleTactileCameraConfig = field(
        default_factory=BundleTactileCameraConfig
    )
    action_adapter: BundleActionAdapterConfig = field(default_factory=BundleActionAdapterConfig)
    runner: BundleRunnerConfig = field(default_factory=BundleRunnerConfig)
    model: BundleModelConfig = field(default_factory=BundleModelConfig)
    initial_state: BundleInitialStateConfig = field(default_factory=BundleInitialStateConfig)
    streaming: BundleStreamingConfig = field(default_factory=BundleStreamingConfig)
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
        tactile_camera = BundleTactileCameraConfig(**data.get("tactile_camera", {}))
        action_adapter = BundleActionAdapterConfig(**data.get("action_adapter", {}))
        runner = BundleRunnerConfig(**data.get("runner", {}))
        model = BundleModelConfig(**data.get("model", {}))
        initial_state = BundleInitialStateConfig(**data.get("initial_state", {}))
        streaming_data = dict(data.get("streaming", {}))
        collision_behavior_data = streaming_data.pop("collision_behavior", None)
        if collision_behavior_data is None:
            collision_behavior = None
        elif isinstance(collision_behavior_data, dict):
            collision_behavior = BundleCollisionBehaviorConfig(**collision_behavior_data)
        else:
            raise ValueError("streaming.collision_behavior must be an object or null")
        streaming = BundleStreamingConfig(
            collision_behavior=collision_behavior,
            **streaming_data,
        )
        top_level = dict(data)
        top_level.pop("camera", None)
        top_level.pop("tactile_camera", None)
        top_level.pop("action_adapter", None)
        top_level.pop("runner", None)
        top_level.pop("model", None)
        top_level.pop("initial_state", None)
        top_level.pop("streaming", None)
        return cls(
            camera=camera,
            tactile_camera=tactile_camera,
            action_adapter=action_adapter,
            runner=runner,
            model=model,
            initial_state=initial_state,
            streaming=streaming,
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

    VALID_SOURCES = {"raw_action", "clipped_action", "processed_action", "zeros"}

    def __init__(
        self,
        history_dim: int,
        source: str,
        scale: float | list[float] = 1.0,
        delay_steps: int = 1,
        processed_action_scale: float | list[float] | None = None,
    ) -> None:
        if isinstance(history_dim, bool) or not isinstance(history_dim, int) or history_dim < 1:
            raise ValueError("history_dim must be a positive integer")
        if source not in self.VALID_SOURCES:
            valid = ", ".join(sorted(self.VALID_SOURCES))
            raise ValueError(f"model.history_source must be one of: {valid}")
        if isinstance(scale, bool):
            raise ValueError(
                "model.history_scale must be a finite positive number or per-dimension list"
            )
        if isinstance(scale, (int, float)):
            scale_vector = np.full(history_dim, float(scale), dtype=np.float32)
        elif isinstance(scale, list):
            scale_vector = np.asarray(scale, dtype=np.float32).reshape(-1)
            if scale_vector.shape[0] != history_dim:
                raise ValueError(
                    "model.history_scale list must contain exactly "
                    f"{history_dim} values, got {scale_vector.shape[0]}"
                )
        else:
            raise ValueError(
                "model.history_scale must be a finite positive number or per-dimension list"
            )
        if not np.all(np.isfinite(scale_vector)) or np.any(scale_vector <= 0.0):
            raise ValueError(
                "model.history_scale must be a finite positive number or per-dimension list"
            )
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 1:
            raise ValueError("model.history_delay_steps must be a positive integer")

        self.history_dim = history_dim
        self.source = source
        self.scale = scale_vector
        self.delay_steps = delay_steps
        self.processed_action_scale: np.ndarray | None = None
        if source == "processed_action":
            if processed_action_scale is None:
                raise ValueError(
                    "processed_action history requires per-dimension action scales"
                )
            raw_scales = np.asarray(processed_action_scale, dtype=np.float32).reshape(-1)
            if raw_scales.shape != (history_dim,):
                raise ValueError(
                    "processed_action_scale must contain exactly "
                    f"{history_dim} values, got {raw_scales.shape[0]}"
                )
            if not np.all(np.isfinite(raw_scales)) or np.any(raw_scales <= 0.0):
                raise ValueError("processed_action_scale must contain finite positive values")
            self.processed_action_scale = raw_scales
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
        elif self.source == "processed_action":
            assert self.processed_action_scale is not None
            candidate = (
                np.asarray(clipped_action, dtype=np.float32).reshape(-1)
                * self.processed_action_scale
            )
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
        *,
        auto_exposure: bool | None = None,
        exposure: float | None = None,
        gain: float | None = None,
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
        manual_controls = exposure is not None or gain is not None
        if auto_exposure is True and manual_controls:
            raise ValueError("Manual --exposure/--gain cannot be combined with auto exposure")

        self._started = False
        try:
            profile = self.pipeline.start(self.config)
            self._started = True
            self.color_sensor = profile.get_device().first_color_sensor()

            # Manual values must be set before camera warmup. Otherwise the
            # warmup frames are captured using auto exposure and the image may
            # keep changing during the first predictions.
            if manual_controls:
                self._set_color_option(rs.option.enable_auto_exposure, 0.0, "auto exposure")
            elif auto_exposure is not None:
                self._set_color_option(
                    rs.option.enable_auto_exposure,
                    1.0 if auto_exposure else 0.0,
                    "auto exposure",
                )
            if exposure is not None:
                self._set_color_option(rs.option.exposure, exposure, "exposure")
            if gain is not None:
                self._set_color_option(rs.option.gain, gain, "gain")

            for _ in range(max(0, int(camera_config.warmup_frames))):
                self.pipeline.wait_for_frames()
        except BaseException:
            if self._started:
                self.pipeline.stop()
                self._started = False
            raise

    def _set_color_option(self, option: Any, value: float, label: str) -> None:
        if not math.isfinite(float(value)):
            raise ValueError(f"RealSense {label} must be finite, got {value!r}")
        if not self.color_sensor.supports(option):
            raise RuntimeError(f"The selected RealSense color sensor does not support {label}")
        option_range = self.color_sensor.get_option_range(option)
        numeric_value = float(value)
        if numeric_value < option_range.min or numeric_value > option_range.max:
            raise ValueError(
                f"RealSense {label} {numeric_value:g} is outside the supported range "
                f"[{option_range.min:g}, {option_range.max:g}]"
            )
        self.color_sensor.set_option(option, numeric_value)

    def get_color_controls(self) -> dict[str, bool | float | None]:
        def get_option(option: Any) -> float | None:
            if not self.color_sensor.supports(option):
                return None
            return float(self.color_sensor.get_option(option))

        auto_exposure = get_option(self.rs.option.enable_auto_exposure)
        return {
            "auto_exposure": None if auto_exposure is None else bool(round(auto_exposure)),
            "exposure": get_option(self.rs.option.exposure),
            "gain": get_option(self.rs.option.gain),
        }

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
        if self._started:
            self.pipeline.stop()
            self._started = False


class LatestFrameCamera:
    """Continuously capture frames so policy ticks never wait for camera I/O."""

    def __init__(self, camera: Any, first_frame_timeout_s: float = 2.0) -> None:
        self.camera = camera
        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="realsense-latest-frame",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + first_frame_timeout_s
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out waiting for the first RealSense frame")
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                if self._closing:
                    return
            try:
                frame = self.camera.read()
            except BaseException as exc:
                with self._condition:
                    if not self._closing:
                        self._error = exc
                        self._condition.notify_all()
                return
            with self._condition:
                if self._closing:
                    return
                self._latest = np.asarray(frame, dtype=np.uint8).copy()
                self._condition.notify_all()

    def read(self) -> np.ndarray:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error
            if self._latest is None:
                raise RuntimeError("No RealSense frame is available")
            return self._latest.copy()

    def close(self) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            self._condition.notify_all()
        # Stopping the pipeline releases wait_for_frames in the capture thread.
        self.camera.close()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("RealSense capture thread did not stop")


class GelSightPairCamera:
    """Continuously capture a left/right GelSight pair without blocking policy ticks."""

    def __init__(
        self,
        config: BundleTactileCameraConfig,
        left_shape: tuple[int, int, int],
        right_shape: tuple[int, int, int],
    ) -> None:
        if left_shape[2] != 3 or right_shape[2] != 3:
            raise ValueError("GelSight model inputs must be HxWx3 RGB images")
        if config.left_device == config.right_device:
            raise ValueError("tactile_camera left_device and right_device must differ")
        if config.warmup_frames < 1:
            raise ValueError("tactile_camera.warmup_frames must be positive")
        if config.first_frame_timeout_s <= 0.0:
            raise ValueError("tactile_camera.first_frame_timeout_s must be positive")

        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for GelSight camera capture") from exc

        self.cv2 = cv2
        self.config = config
        self.left_shape = left_shape
        self.right_shape = right_shape
        self._condition = threading.Condition()
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._captures: list[Any] = []
        try:
            self._captures = [
                self._open(config.left_device),
                self._open(config.right_device),
            ]
        except Exception:
            for capture in self._captures:
                capture.release()
            raise

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="gelsight-pair-latest-frame",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + config.first_frame_timeout_s
        startup_error: BaseException | None = None
        with self._condition:
            while self._latest is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    startup_error = RuntimeError(
                        "Timed out waiting for the first GelSight frame pair"
                    )
                    break
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                startup_error = RuntimeError(
                    f"GelSight pair capture failed: {self._error}"
                )
        if startup_error is not None:
            cause = self._error
            self.close()
            if cause is not None:
                raise startup_error from cause
            raise startup_error

    def _open(self, device: int | str) -> Any:
        capture = self.cv2.VideoCapture(device, self.cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open GelSight camera {device!r}")
        capture.set(
            self.cv2.CAP_PROP_FOURCC,
            self.cv2.VideoWriter_fourcc(*"MJPG"),
        )
        capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(self.cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(self.cv2.CAP_PROP_BUFFERSIZE, 1)
        actual = (
            int(capture.get(self.cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(self.cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if actual != (self.config.width, self.config.height):
            capture.release()
            raise RuntimeError(
                f"GelSight camera {device!r} negotiated {actual[0]}x{actual[1]}, "
                f"expected {self.config.width}x{self.config.height}"
            )
        return capture

    def _resize_rgb(
        self,
        bgr: np.ndarray,
        shape: tuple[int, int, int],
    ) -> np.ndarray:
        height, width, _ = shape
        resized = self.cv2.resize(
            bgr,
            (width, height),
            interpolation=self.cv2.INTER_AREA,
        )
        return self.cv2.cvtColor(resized, self.cv2.COLOR_BGR2RGB)

    def _capture_loop(self) -> None:
        captured_pairs = 0
        while True:
            with self._condition:
                if self._closing:
                    return
            try:
                grabbed = [capture.grab() for capture in self._captures]
                frames: list[np.ndarray] = []
                for capture, ok in zip(self._captures, grabbed):
                    retrieved, frame = capture.retrieve() if ok else (False, None)
                    if not retrieved or frame is None:
                        raise RuntimeError("failed to retrieve a GelSight frame")
                    frames.append(frame)
                pair = (
                    self._resize_rgb(frames[0], self.left_shape),
                    self._resize_rgb(frames[1], self.right_shape),
                )
                captured_pairs += 1
            except BaseException as exc:
                with self._condition:
                    if not self._closing:
                        self._error = exc
                        self._condition.notify_all()
                return
            if captured_pairs < self.config.warmup_frames:
                continue
            with self._condition:
                if self._closing:
                    return
                self._latest = pair
                self._condition.notify_all()

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"GelSight pair capture failed: {self._error}") from self._error
            if self._latest is None:
                raise RuntimeError("No GelSight frame pair is available")
            return self._latest[0].copy(), self._latest[1].copy()

    def close(self) -> None:
        with self._condition:
            if self._closing:
                return
            self._closing = True
            self._condition.notify_all()
        for capture in self._captures:
            capture.release()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                raise RuntimeError("GelSight capture thread did not stop")


class PolicyCameraRig:
    def __init__(self, wrist_camera: Any, tactile_camera: GelSightPairCamera) -> None:
        self.wrist_camera = wrist_camera
        self.tactile_camera = tactile_camera

    def read(self) -> np.ndarray:
        return self.wrist_camera.read()

    def read_tactile(self) -> tuple[np.ndarray, np.ndarray]:
        return self.tactile_camera.read()

    def close(self) -> None:
        errors: list[str] = []
        for name, camera in (
            ("wrist camera", self.wrist_camera),
            ("GelSight cameras", self.tactile_camera),
        ):
            try:
                camera.close()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))


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
    def __init__(
        self,
        model_path: str,
        metadata_path: str,
        device: str = "cpu",
        torch_num_threads: int | None = None,
        optimize_for_inference: bool = False,
        rma_position_source: str = "vision",
        rma_contact_source: str = "vision",
        rma_oracle_cube_position_root: list[float] | None = None,
    ) -> None:
        if torch_num_threads is not None:
            if isinstance(torch_num_threads, bool) or not isinstance(torch_num_threads, int):
                raise ValueError("model.torch_num_threads must be a positive integer or null")
            if torch_num_threads < 1:
                raise ValueError("model.torch_num_threads must be a positive integer or null")
            torch.set_num_threads(torch_num_threads)
        if not isinstance(optimize_for_inference, bool):
            raise ValueError("model.optimize_for_inference must be a boolean")
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()
        if optimize_for_inference:
            self.model = torch.jit.optimize_for_inference(self.model)

        signature = self.metadata["input_signature"]
        output_signature = self.metadata.get("output_signature", {})
        self.history_dim = int(signature["action_history"][0])
        self.proprio_dim = int(signature["proprio_obs"][0])
        self.rgb_height, self.rgb_width, _ = signature["wrist_rgb"]
        self.gelsight_input_shapes: dict[str, tuple[int, int, int]] = {}
        for name in ("gsmini_left_rgb", "gsmini_right_rgb"):
            if name in signature:
                shape = tuple(int(value) for value in signature[name])
                if len(shape) != 3 or shape[2] != 3:
                    raise ValueError(f"{name} must have an HxWx3 input signature")
                self.gelsight_input_shapes[name] = shape
        if len(self.gelsight_input_shapes) not in (0, 2):
            raise ValueError(
                "TorchScript metadata must provide both gsmini_left_rgb and "
                "gsmini_right_rgb, or neither"
            )
        self.has_gelsight_inputs = bool(self.gelsight_input_shapes)
        self.action_dim = int(output_signature.get("mean_actions", [self.history_dim])[0])
        self.kind = str(self.metadata.get("kind", "legacy_e2e_torchscript"))
        self.is_tacex_rma_student = self.kind == "tacex_rma_student_torchscript"
        self.rma_position_source = str(rma_position_source)
        self.rma_contact_source = str(rma_contact_source)
        self.rma_oracle_cube_position_root: tuple[float, float, float] | None = None
        self.last_inference_info: dict[str, Any] = {}
        if self.rma_position_source not in {"vision", "oracle"}:
            raise ValueError("model.rma_position_source must be 'vision' or 'oracle'")
        if self.rma_contact_source not in {"vision", "zero"}:
            raise ValueError("model.rma_contact_source must be 'vision' or 'zero'")
        override_enabled = (
            self.rma_position_source != "vision" or self.rma_contact_source != "vision"
        )
        if override_enabled and not self.is_tacex_rma_student:
            raise ValueError("RMA position/contact overrides require a TacEx RMA Student model")
        if self.rma_position_source == "oracle":
            values = np.asarray(rma_oracle_cube_position_root, dtype=np.float64).reshape(-1)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(
                    "model.rma_oracle_cube_position_root must contain three finite "
                    "robot_root coordinates"
                )
            self.rma_oracle_cube_position_root = tuple(float(value) for value in values)
        elif rma_oracle_cube_position_root is not None:
            raise ValueError(
                "model.rma_oracle_cube_position_root requires rma_position_source='oracle'"
            )
        if override_enabled and not hasattr(self.model, "actor_core"):
            raise ValueError("RMA TorchScript does not expose actor_core for input ablation")
        default_order = ["action_history", "proprio_obs", "wrist_rgb"]
        if self.has_gelsight_inputs:
            default_order.extend(["gsmini_left_rgb", "gsmini_right_rgb"])
        self.input_order = list(self.metadata.get("input_order", default_order))
        if sorted(self.input_order) != sorted(default_order):
            raise ValueError(
                "TorchScript metadata input_order must contain exactly "
                + ", ".join(default_order)
            )
        expected_tacex_order = ["wrist_rgb", "proprio_obs", "action_history"]
        if self.has_gelsight_inputs:
            expected_tacex_order.extend(["gsmini_left_rgb", "gsmini_right_rgb"])
        if self.is_tacex_rma_student and self.input_order != expected_tacex_order:
            raise ValueError(
                f"TacEx RMA Student requires input_order {expected_tacex_order}"
            )

    def _predict_adaptation(
        self,
        wrist_rgb: torch.Tensor,
        tactile_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.has_gelsight_inputs:
            return self.model.predict_adaptation(
                wrist_rgb,
                tactile_tensors["gsmini_left_rgb"],
                tactile_tensors["gsmini_right_rgb"],
            )
        return self.model.predict_adaptation(wrist_rgb)

    def predict(
        self,
        action_history: np.ndarray,
        proprio_obs: np.ndarray,
        wrist_rgb: np.ndarray,
        gsmini_left_rgb: np.ndarray | None = None,
        gsmini_right_rgb: np.ndarray | None = None,
        *,
        collect_rma_debug: bool = False,
    ) -> np.ndarray:
        action_tensor = torch.as_tensor(action_history, dtype=torch.float32, device=self.device).unsqueeze(0)
        proprio_tensor = torch.as_tensor(proprio_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        needs_vision = (
            self.rma_position_source == "vision" or self.rma_contact_source == "vision"
        )
        rgb_tensor: torch.Tensor | None = None
        if needs_vision or not self.is_tacex_rma_student:
            # RealSense/Pillow arrays can be backed by a read-only buffer. PyTorch
            # warns because a tensor created from that buffer could otherwise have
            # undefined behavior if it were ever mutated.
            writable_rgb = np.array(wrist_rgb, dtype=np.uint8, order="C", copy=True)
            rgb_tensor = torch.as_tensor(
                writable_rgb, dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
        tactile_tensors: dict[str, torch.Tensor] = {}
        tactile_values = {
            "gsmini_left_rgb": gsmini_left_rgb,
            "gsmini_right_rgb": gsmini_right_rgb,
        }
        for name, shape in self.gelsight_input_shapes.items():
            value = tactile_values[name]
            if value is None:
                raise ValueError(f"Model requires {name}, but no image was supplied")
            writable = np.array(value, dtype=np.uint8, order="C", copy=True)
            if writable.shape != shape:
                raise ValueError(f"Expected {name} shape {shape}, got {writable.shape}")
            tactile_tensors[name] = torch.as_tensor(
                writable, dtype=torch.uint8, device=self.device
            ).unsqueeze(0)

        override_enabled = (
            self.rma_position_source != "vision" or self.rma_contact_source != "vision"
        )
        with torch.inference_mode():
            if not override_enabled:
                tensors = {
                    "action_history": action_tensor,
                    "proprio_obs": proprio_tensor,
                    "wrist_rgb": rgb_tensor,
                    **tactile_tensors,
                }
                output = self.model(*(tensors[name] for name in self.input_order))
                self.last_inference_info = {
                    "rma_position_source": "vision",
                    "rma_contact_source": "vision",
                    "vision_bypassed": False,
                } if self.is_tacex_rma_student else {}
                if collect_rma_debug and self.is_tacex_rma_student:
                    if not hasattr(self.model, "predict_adaptation"):
                        raise RuntimeError(
                            "TacEx RMA Student model does not expose predict_adaptation"
                        )
                    assert rgb_tensor is not None
                    normalized_position, contact_logits = self._predict_adaptation(
                        rgb_tensor, tactile_tensors
                    )
                    normalization = self.metadata["normalization"]
                    center = torch.tensor(
                        normalization["cube_position_center"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scale = torch.tensor(
                        normalization["cube_position_scale"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    cube_position_root = normalized_position * scale + center
                    contact_state = torch.sigmoid(contact_logits)
                    self.last_inference_info.update(
                        {
                            "cube_position_root": cube_position_root.detach()
                            .cpu()
                            .reshape(-1)
                            .tolist(),
                            "contact_state": contact_state.detach().cpu().reshape(-1).tolist(),
                        }
                    )
            else:
                normalized_position: torch.Tensor | None = None
                contact_logits: torch.Tensor | None = None
                if needs_vision:
                    assert rgb_tensor is not None
                    normalized_position, contact_logits = self._predict_adaptation(
                        rgb_tensor, tactile_tensors
                    )

                if self.rma_position_source == "oracle":
                    assert self.rma_oracle_cube_position_root is not None
                    cube_position = torch.tensor(
                        [self.rma_oracle_cube_position_root],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    position_is_normalized = False
                    cube_position_root = cube_position
                else:
                    assert normalized_position is not None
                    cube_position = normalized_position
                    position_is_normalized = True
                    normalization = self.metadata["normalization"]
                    center = torch.tensor(
                        normalization["cube_position_center"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    scale = torch.tensor(
                        normalization["cube_position_scale"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    cube_position_root = normalized_position * scale + center

                if self.rma_contact_source == "zero":
                    contact_state = torch.zeros((1, 2), dtype=torch.float32, device=self.device)
                else:
                    assert contact_logits is not None
                    contact_state = torch.sigmoid(contact_logits)

                output = self.model.actor_core(
                    proprio_tensor,
                    action_tensor,
                    cube_position,
                    contact_state,
                    position_is_normalized,
                )
                self.last_inference_info = {
                    "rma_position_source": self.rma_position_source,
                    "rma_contact_source": self.rma_contact_source,
                    "cube_position_root": cube_position_root.detach().cpu().reshape(-1).tolist(),
                    "contact_state": contact_state.detach().cpu().reshape(-1).tolist(),
                    "vision_bypassed": not needs_vision,
                }

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
    history_buffer = ActionHistoryBuffer(
        history_dim=bundle.history_dim,
        source=config.model.history_source,
        scale=config.model.history_scale,
        delay_steps=config.model.history_delay_steps,
        processed_action_scale=config.action_adapter.scales,
    )
    if config.model.enforce_policy_contract:
        if bundle.is_tacex_rma_student:
            _validate_tacex_rma_student_contract(bundle, config, history_buffer.scale)
        else:
            _validate_policy_contract(bundle, config, history_buffer.scale)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_policy_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    contract = bundle.metadata.get("policy_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "model.enforce_policy_contract is true, but metadata.policy_contract is missing"
        )

    expected_sha256 = bundle.metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(
            "model.enforce_policy_contract is true, but metadata.torchscript_sha256 is missing"
        )
    actual_sha256 = _sha256_file(bundle.model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "TorchScript SHA-256 does not match metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    expected_source = contract.get("deployment_history_source")
    if expected_source != config.model.history_source:
        raise ValueError(
            "Policy contract requires model.history_source="
            f"{expected_source!r}, got {config.model.history_source!r}"
        )

    expected_history_scales = np.asarray(
        contract.get("deployment_history_scales", []), dtype=np.float32
    ).reshape(-1)
    if (
        expected_history_scales.shape != history_scale_vector.shape
        or not np.allclose(expected_history_scales, history_scale_vector, rtol=0.0, atol=1e-8)
    ):
        raise ValueError(
            "Policy contract history scales do not match model.history_scale: "
            f"expected {expected_history_scales.tolist()}, "
            f"got {history_scale_vector.tolist()}"
        )

    expected_delay = contract.get("deployment_history_delay_steps")
    if expected_delay != config.model.history_delay_steps:
        raise ValueError(
            "Policy contract requires model.history_delay_steps="
            f"{expected_delay}, got {config.model.history_delay_steps}"
        )

    expected_action_scales = np.asarray(
        contract.get("action_scales", []), dtype=np.float32
    ).reshape(-1)
    configured_action_scales = np.asarray(
        config.action_adapter.scales, dtype=np.float32
    ).reshape(-1)
    if (
        expected_action_scales.shape != configured_action_scales.shape
        or not np.allclose(
            expected_action_scales,
            configured_action_scales,
            rtol=0.0,
            atol=1e-8,
        )
    ):
        raise ValueError(
            "Policy contract action scales do not match action_adapter.scales: "
            f"expected {expected_action_scales.tolist()}, "
            f"got {configured_action_scales.tolist()}"
        )

    expected_gripper_mode = contract.get("gripper_control_mode")
    if (
        expected_gripper_mode == "total_width_delta_cached_target"
        and config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(
            "Policy contract requires action_adapter.gripper_mode='delta_width', "
            f"got {config.action_adapter.gripper_mode!r}"
        )

    expected_camera_shape = (
        int(contract.get("camera_height", -1)),
        int(contract.get("camera_width", -1)),
    )
    actual_camera_shape = (bundle.rgb_height, bundle.rgb_width)
    if expected_camera_shape != actual_camera_shape:
        raise ValueError(
            "Policy contract camera shape does not match the model input signature: "
            f"expected {expected_camera_shape}, got {actual_camera_shape}"
        )


def _validate_tacex_rma_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """Fail closed on the deployment-relevant RMA Student v5-v7 contract.

    The RMA exporter intentionally keeps its JSON small and does not embed the
    simulator's action/timing manifest.  These checks mirror the current TacEx
    RMA Student environment rather than pretending that the legacy E2E contract
    is applicable.
    """
    metadata = bundle.metadata
    version = metadata.get("version")
    if version not in (5, 6, 7):
        raise ValueError("TacEx RMA Student deployment requires metadata version 5, 6, or 7")
    if torch.device(config.model.device).type == "cuda" and version not in (6, 7):
        raise ValueError(
            "TacEx RMA Student CUDA inference requires portable metadata version 6 or 7"
        )
    if version in (6, 7):
        cuda_validation = metadata.get("cuda_validation")
        cuda_validation_atol = metadata.get("cuda_validation_atol")
        if not isinstance(cuda_validation, dict) or cuda_validation.get("available") is not True:
            raise ValueError(
                "TacEx RMA Student v6/v7 metadata requires successful CUDA validation"
            )
        cuda_max_abs_error = cuda_validation.get("max_abs_error")
        if (
            isinstance(cuda_max_abs_error, bool)
            or not isinstance(cuda_max_abs_error, (int, float))
            or not math.isfinite(float(cuda_max_abs_error))
            or float(cuda_max_abs_error) < 0.0
            or isinstance(cuda_validation_atol, bool)
            or not isinstance(cuda_validation_atol, (int, float))
            or not math.isfinite(float(cuda_validation_atol))
            or float(cuda_validation_atol) <= 0.0
            or float(cuda_max_abs_error) > float(cuda_validation_atol)
        ):
            raise ValueError(
                "TacEx RMA Student v6/v7 metadata has invalid CUDA validation error/tolerance"
            )
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("TacEx RMA Student metadata is missing torchscript_sha256")
    actual_sha256 = _sha256_file(bundle.model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "TorchScript SHA-256 does not match metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if bundle.action_dim != 4 or bundle.history_dim != 4 or bundle.proprio_dim != 15:
        raise ValueError("TacEx RMA Student requires action_history[4], proprio_obs[15], actions[4]")
    if (bundle.rgb_height, bundle.rgb_width) != (224, 224):
        raise ValueError("TacEx RMA Student requires wrist_rgb[224,224,3]")
    if version == 7:
        expected_tactile_shapes = {
            "gsmini_left_rgb": (96, 128, 3),
            "gsmini_right_rgb": (96, 128, 3),
        }
        if bundle.gelsight_input_shapes != expected_tactile_shapes:
            raise ValueError(
                "TacEx RMA Student v7 requires left/right GelSight RGB inputs "
                "with shape [96,128,3]"
            )
        if not config.tactile_camera.enabled:
            raise ValueError(
                "TacEx RMA Student v7 requires tactile_camera.enabled=true"
            )
    elif bundle.has_gelsight_inputs:
        raise ValueError("TacEx RMA Student v5/v6 must not declare GelSight inputs")
    if config.model.history_source != "processed_action":
        raise ValueError(
            "TacEx RMA Student requires model.history_source='processed_action'"
        )
    if not np.allclose(history_scale_vector, np.ones(4, dtype=np.float32), rtol=0.0, atol=1e-8):
        raise ValueError(
            "TacEx RMA Student history is already in physical units; "
            "model.history_scale must be [1, 1, 1, 1]"
        )
    if config.model.history_delay_steps != 1:
        raise ValueError("TacEx RMA Student requires model.history_delay_steps=1")
    if config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]:
        raise ValueError("TacEx RMA Student requires action labels [dx, dy, dz, gripper]")
    expected_action_scales = np.asarray([0.05, 0.05, 0.05, 0.01], dtype=np.float32)
    configured_action_scales = np.asarray(config.action_adapter.scales, dtype=np.float32).reshape(-1)
    if (
        configured_action_scales.shape != expected_action_scales.shape
        or not np.allclose(
            configured_action_scales, expected_action_scales, rtol=0.0, atol=1e-8
        )
    ):
        raise ValueError(
            "TacEx RMA Student requires action scales [0.05, 0.05, 0.05, 0.01]"
        )
    if (
        config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
        or config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(
            "TacEx RMA Student requires normalized [-1,1] actions and delta_width gripper"
        )
    camera = config.camera
    if (
        camera.width,
        camera.height,
        camera.fps,
        camera.crop_left,
        camera.crop_top,
        camera.crop_width,
        camera.crop_height,
    ) != (640, 480, 30, 100, 34, 400, 398):
        raise ValueError(
            "TacEx RMA Student requires D435 640x480@30 crop (100,34,400,398)"
        )
    if not camera.enable_crop:
        raise ValueError("TacEx RMA Student requires calibrated camera crop enabled")


def validate_bundle_artifacts(config: BundleDeployConfig) -> dict[str, Any]:
    """Validate and exercise a bundle without connecting to robot or camera hardware."""
    bundle = BundleTorchScriptPolicy(
        model_path=config.model.model_path,
        metadata_path=config.model.metadata_path,
        device=config.model.device,
        torch_num_threads=config.model.torch_num_threads,
        optimize_for_inference=config.model.optimize_for_inference,
        rma_position_source=config.model.rma_position_source,
        rma_contact_source=config.model.rma_contact_source,
        rma_oracle_cube_position_root=config.model.rma_oracle_cube_position_root,
    )
    _validate_bundle_action_dims(bundle, config)
    streaming_report: dict[str, Any] | None = None
    if config.control_mode == "streaming":
        from .streaming import validate_streaming_contract

        streaming_report = validate_streaming_contract(bundle, config)
    elif config.control_mode != "blocking":
        raise ValueError("control_mode must be 'blocking' or 'streaming'")

    action_history = np.zeros(bundle.history_dim, dtype=np.float32)
    proprio = np.zeros(bundle.proprio_dim, dtype=np.float32)
    wrist_rgb = np.zeros(
        (bundle.rgb_height, bundle.rgb_width, 3),
        dtype=np.uint8,
    )
    tactile_zeros = {
        name: np.zeros(shape, dtype=np.uint8)
        for name, shape in bundle.gelsight_input_shapes.items()
    }
    output = bundle.predict(
        action_history,
        proprio,
        wrist_rgb,
        tactile_zeros.get("gsmini_left_rgb"),
        tactile_zeros.get("gsmini_right_rgb"),
    )
    if output.shape != (bundle.action_dim,):
        raise ValueError(
            f"Expected model output shape {(bundle.action_dim,)}, got {output.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Model smoke-test output contains NaN or Inf")

    return {
        "model_path": str(bundle.model_path),
        "metadata_path": str(bundle.metadata_path),
        "torchscript_sha256": _sha256_file(bundle.model_path),
        "task": bundle.metadata.get("task"),
        "input_signature": {
            "action_history": [bundle.history_dim],
            "proprio_obs": [bundle.proprio_dim],
            "wrist_rgb": [bundle.rgb_height, bundle.rgb_width, 3],
            **{
                name: list(shape)
                for name, shape in bundle.gelsight_input_shapes.items()
            },
        },
        "output_signature": {"mean_actions": [bundle.action_dim]},
        "smoke_test_output": output.tolist(),
        "rma_actor_input": dict(bundle.last_inference_info),
        "policy_contract_enforced": config.model.enforce_policy_contract,
        "streaming_contract": streaming_report,
    }


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
    *,
    tactile_config: BundleTactileCameraConfig | None = None,
    gelsight_input_shapes: dict[str, tuple[int, int, int]] | None = None,
):
    if camera_config.source == "realsense":
        wrist_camera = LatestFrameCamera(
            RealSenseRGBCamera(
                camera_config,
                output_width=output_width,
                output_height=output_height,
            )
        )
    elif camera_config.source == "image":
        if not camera_config.image_path:
            raise ValueError("camera.image_path must be set when source='image'")
        wrist_camera = StaticRGBCamera(
            camera_config.image_path,
            camera_config,
            output_width=output_width,
            output_height=output_height,
        )
    else:
        raise ValueError(f"Unsupported camera source: {camera_config.source}")

    shapes = gelsight_input_shapes or {}
    if not shapes:
        if tactile_config is not None and tactile_config.enabled:
            wrist_camera.close()
            raise ValueError(
                "tactile_camera is enabled, but the model has no GelSight inputs"
            )
        return wrist_camera
    if tactile_config is None or not tactile_config.enabled:
        wrist_camera.close()
        raise ValueError("Model requires GelSight inputs, but tactile_camera is disabled")
    try:
        tactile_camera = GelSightPairCamera(
            tactile_config,
            shapes["gsmini_left_rgb"],
            shapes["gsmini_right_rgb"],
        )
    except Exception:
        wrist_camera.close()
        raise
    return PolicyCameraRig(wrist_camera, tactile_camera)


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
    allow_full_scale: bool = False,
    streaming_check: bool = False,
) -> dict[str, Any]:
    if config.control_mode == "streaming" or streaming_check:
        from .streaming import run_streaming_bundle_deploy

        return run_streaming_bundle_deploy(
            config,
            execute_motion=execute_motion,
            confirm_session_callback=confirm_step_callback,
            save_step_data=save_step_data,
            allow_full_scale=allow_full_scale,
            streaming_check=streaming_check,
        )

    if config.control_mode != "blocking":
        raise ValueError("control_mode must be 'blocking' or 'streaming'")
    if allow_full_scale:
        raise ValueError("allow_full_scale is only valid for streaming control")
    run_dir = _make_run_dir(config)
    (run_dir / "rgb").mkdir(exist_ok=True)
    if config.tactile_camera.enabled:
        (run_dir / "tactile_rgb").mkdir(exist_ok=True)
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
        torch_num_threads=config.model.torch_num_threads,
        optimize_for_inference=config.model.optimize_for_inference,
        rma_position_source=config.model.rma_position_source,
        rma_contact_source=config.model.rma_contact_source,
        rma_oracle_cube_position_root=config.model.rma_oracle_cube_position_root,
    )
    _validate_bundle_action_dims(bundle, config)
    action_history_buffer = ActionHistoryBuffer(
        history_dim=bundle.history_dim,
        source=config.model.history_source,
        scale=config.model.history_scale,
        delay_steps=config.model.history_delay_steps,
        processed_action_scale=config.action_adapter.scales,
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
            tactile_config=config.tactile_camera,
            gelsight_input_shapes=getattr(bundle, "gelsight_input_shapes", {}),
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

            tactile_images: dict[str, np.ndarray] = {}
            tactile_relpaths: dict[str, str] = {}
            if bundle.has_gelsight_inputs:
                if not hasattr(camera, "read_tactile"):
                    raise RuntimeError(
                        "Model requires GelSight inputs, but camera rig has none"
                    )
                left_tactile, right_tactile = camera.read_tactile()
                tactile_images = {
                    "gsmini_left_rgb": np.asarray(left_tactile, dtype=np.uint8),
                    "gsmini_right_rgb": np.asarray(right_tactile, dtype=np.uint8),
                }
                for name, tactile_image in tactile_images.items():
                    tactile_path = run_dir / "tactile_rgb" / f"{name}_step_{step_index:04d}.png"
                    tactile_relpaths[name] = str(tactile_path.relative_to(run_dir))
                    if config.tactile_camera.save_rgb or save_step_data:
                        Image.fromarray(tactile_image).save(tactile_path)

            pack_start = time.perf_counter()
            action_history, proprio = build_bundle_inputs(
                observation,
                action_history_buffer.current(),
                proprio_dim=bundle.proprio_dim,
                history_dim=bundle.history_dim,
            )
            step_timing["input_pack_ms"] = (time.perf_counter() - pack_start) * 1000.0

            infer_start = time.perf_counter()
            raw_action = bundle.predict(
                action_history,
                proprio,
                wrist_rgb,
                tactile_images.get("gsmini_left_rgb"),
                tactile_images.get("gsmini_right_rgb"),
            )
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
                    **tactile_images,
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
                    "tactile_rgb_paths": tactile_relpaths,
                    "tactile_rgb_shapes": {
                        name: list(image.shape)
                        for name, image in tactile_images.items()
                    },
                    "step_data_path": step_data_relpath,
                    "rma_actor_input": dict(bundle.last_inference_info),
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

            if observation.has_errors or bool(info.get("arm_command_failed")):
                print_error_wrench_report(
                    observation,
                    context=f"Policy step {step_index}",
                )

            if done:
                break

    summary["num_steps"] = len(summary["steps"])
    summary["save_step_data"] = save_step_data
    if summary["steps"]:
        summary["last_info"] = summary["steps"][-1]["info"]
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
