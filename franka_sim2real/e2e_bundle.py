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
from .gelsight_devices import select_gelsight_pair
from .hil import HILSettings
from .policy_features import extract_actor_features, validate_actor_feature_contract
from .residual_runtime import ResidualDeploySettings, ResidualPolicyRuntime
from .real_rl.runtime import RealRLDeploySettings, RealRLPolicyRuntime
from .safety import apply_safety_limits
from .types import RobotAction, RobotObservation, print_error_wrench_report
from real_rlpd.runtime import RLPDDeploySettings, RLPDPolicyRuntime

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = REPO_ROOT / "deploy_bundle_e2e"


@dataclass(frozen=True)
class CameraFramePacket:
    sequence: int
    capture_timestamp: float
    camera_timestamp_ms: float | None
    raw_rgb: np.ndarray
    policy_rgb: np.ndarray

    @property
    def host_monotonic_s(self) -> float:
        """兼容 Real-RL v2 时间戳契约之前的调用方。"""
        return self.capture_timestamp


@dataclass
class BundleCameraConfig:
    source: str = "realsense"
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 0
    auto_exposure: bool | None = None
    exposure: float | None = None
    gain: float | None = None
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
    auto_discover: bool = False
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
    # XY RMA Student v8 在运行时需要左右指尖接触证据。真机夹爪只有一个
    # grasp 标志，"gripper_is_grasped" 是显式近似：true 映射成
    # [threshold, threshold]，绝不从腕部 wrench 推断。
    rma_contact_force_source: str = "unsupported"


@dataclass
class BundleInitialStateConfig:
    enforce: bool = False
    joint_positions: list[float] | None = None
    joint_position_tolerance_rad: float = 0.01
    max_abs_joint_velocity_rad_s: float = 0.02
    tcp_translation: list[float] | None = None
    tcp_translation_tolerance_m: float = 0.005
    # O_T_EE reported by libfranka is already O_T_F @ F_T_EE.  Record the
    # F_T_EE translation independently so a tool swap cannot silently remove
    # or double the simulation TCP offset.
    flange_to_tcp_translation_m: list[float] | None = None
    flange_to_tcp_translation_tolerance_m: float = 0.001
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
    # ``joint`` uses Franka's joint impedance controller. ``cartesian`` uses
    # its Cartesian impedance controller, whose stiffness axes are the
    # configured stiffness frame.
    impedance_mode: str = "joint"
    # A supplied array is applied once before Robot::control starts in joint
    # impedance mode. None preserves the Desk-configured value.
    joint_impedance: list[float] | None = None
    # [Kx, Ky, Kz, Kroll, Kpitch, Kyaw]. Required in Cartesian mode; XYZ is
    # N/m and rotation is Nm/rad. Kz is element 2.
    cartesian_impedance: list[float] | None = None
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
    # Offset from libfranka's configured EE to the physical tool point that
    # must remain in the workspace. It is distinct from the nominal O_T_EE.
    tool_tcp_offset_ee_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
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
    flange_to_tcp_translation_tolerance = _validate_nonnegative_finite(
        "flange_to_tcp_translation_tolerance_m",
        config.flange_to_tcp_translation_tolerance_m,
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

    expected_flange_to_tcp_translation = _validate_vector(
        "flange_to_tcp_translation_m", config.flange_to_tcp_translation_m, 3
    )
    if expected_flange_to_tcp_translation is not None:
        actual_flange_to_tcp_translation = _validate_vector(
            "actual_flange_to_tcp_translation",
            observation.metadata.get("flange_to_tcp_translation_m"),
            3,
        )
        if actual_flange_to_tcp_translation is None:
            add_check(
                "flange_to_tcp_translation",
                False,
                {
                    "actual_m": None,
                    "expected_m": expected_flange_to_tcp_translation.tolist(),
                    "tolerance_m": flange_to_tcp_translation_tolerance,
                },
                "active flange-to-TCP transform is unavailable",
            )
        else:
            flange_to_tcp_error = float(
                np.linalg.norm(actual_flange_to_tcp_translation - expected_flange_to_tcp_translation)
            )
            add_check(
                "flange_to_tcp_translation",
                flange_to_tcp_error <= flange_to_tcp_translation_tolerance,
                {
                    "actual_m": actual_flange_to_tcp_translation.tolist(),
                    "expected_m": expected_flange_to_tcp_translation.tolist(),
                    "error_norm_m": flange_to_tcp_error,
                    "tolerance_m": flange_to_tcp_translation_tolerance,
                },
                "flange-to-TCP translation error "
                f"{flange_to_tcp_error:.6f} m exceeds "
                f"{flange_to_tcp_translation_tolerance:.6f} m",
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
            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intrinsics = stream.get_intrinsics()
            self.intrinsics = {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
                "distortion_model": str(intrinsics.model).split(".")[-1],
                "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
            }

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

    def read_packet(self) -> tuple[np.ndarray, np.ndarray, float | None]:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame from RealSense.")
        rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8).copy()
        policy_rgb = _apply_camera_crop(
            rgb,
            self.camera_config,
            output_width=self.output_width,
            output_height=self.output_height,
        )
        return rgb, np.array(policy_rgb, dtype=np.uint8, copy=True), float(color_frame.get_timestamp())

    def read(self) -> np.ndarray:
        _raw, policy_rgb, _timestamp = self.read_packet()
        return policy_rgb

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
        self._latest_packet: CameraFramePacket | None = None
        self._sequence = 0
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
                if hasattr(self.camera, "read_packet"):
                    raw, frame, camera_timestamp_ms = self.camera.read_packet()
                else:
                    frame = self.camera.read()
                    raw = frame
                    camera_timestamp_ms = None
                capture_timestamp = time.monotonic()
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
                self._sequence += 1
                self._latest_packet = CameraFramePacket(
                    sequence=self._sequence,
                    capture_timestamp=capture_timestamp,
                    camera_timestamp_ms=camera_timestamp_ms,
                    raw_rgb=np.asarray(raw, dtype=np.uint8).copy(),
                    policy_rgb=self._latest.copy(),
                )
                self._condition.notify_all()

    def read(self) -> np.ndarray:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error
            if self._latest is None:
                raise RuntimeError("No RealSense frame is available")
            return self._latest.copy()

    def read_policy_packet(self) -> tuple[np.ndarray, dict[str, Any]]:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error
            if self._latest_packet is None:
                raise RuntimeError("No RealSense frame packet is available")
            packet = self._latest_packet
            return packet.policy_rgb.copy(), {
                "sequence": packet.sequence,
                "capture_timestamp": packet.capture_timestamp,
                "host_monotonic_s": packet.host_monotonic_s,
                "camera_timestamp_ms": packet.camera_timestamp_ms,
            }

    def read_policy_packet_with_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return policy and raw pixels from one atomic camera packet."""

        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error
            if self._latest_packet is None:
                raise RuntimeError("No RealSense frame packet is available")
            packet = self._latest_packet
            return packet.policy_rgb.copy(), packet.raw_rgb.copy(), {
                "sequence": packet.sequence,
                "capture_timestamp": packet.capture_timestamp,
                "host_monotonic_s": packet.host_monotonic_s,
                "camera_timestamp_ms": packet.camera_timestamp_ms,
                "camera_intrinsics": self.camera_intrinsics(),
            }

    def read_raw_packet(self) -> CameraFramePacket:
        with self._condition:
            if self._error is not None:
                raise RuntimeError(f"RealSense capture failed: {self._error}") from self._error
            if self._latest_packet is None:
                raise RuntimeError("No RealSense frame packet is available")
            packet = self._latest_packet
            return CameraFramePacket(
                packet.sequence,
                packet.capture_timestamp,
                packet.camera_timestamp_ms,
                packet.raw_rgb.copy(),
                packet.policy_rgb.copy(),
            )

    def camera_intrinsics(self) -> dict[str, Any]:
        value = getattr(self.camera, "intrinsics", None)
        if not isinstance(value, dict):
            raise RuntimeError("RealSense intrinsics are unavailable")
        return dict(value)

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


def resolve_gelsight_devices(
    config: BundleTactileCameraConfig,
    selector: Any = select_gelsight_pair,
) -> tuple[int | str, int | str]:
    """Resolve optional runtime discovery and return the concrete left/right pair."""

    if not isinstance(config.auto_discover, bool):
        raise ValueError("tactile_camera.auto_discover must be a boolean")
    if config.auto_discover:
        left_spec, right_spec = selector()
        config.left_device = left_spec.cam_id
        config.right_device = right_spec.cam_id
        print("Automatically discovered two GelSight primary image streams:")
        for side, spec in (("left", left_spec), ("right", right_spec)):
            serial = f", serial={spec.serial}" if spec.serial else ""
            print(
                f"  {side}: /dev/video{spec.cam_id} "
                f"({spec.label}{serial})"
            )
    if config.left_device == config.right_device:
        raise ValueError("tactile_camera left_device and right_device must differ")
    return config.left_device, config.right_device


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
        left_device, right_device = resolve_gelsight_devices(config)
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
        # OpenCV VideoCapture.release() must never race with grab()/retrieve().
        # UVC/OpenCV can otherwise segfault in native code while closing a
        # two-camera GelSight session.
        self._capture_lock = threading.Lock()
        self._latest: tuple[np.ndarray, np.ndarray] | None = None
        self._error: BaseException | None = None
        self._closing = False
        self._captures: list[Any] = []
        try:
            self._captures = [
                self._open(left_device),
                self._open(right_device),
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
                with self._capture_lock:
                    # ``close`` may have begun while this thread was waiting
                    # for the capture lock. Never touch released captures.
                    with self._condition:
                        if self._closing:
                            return
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
        # Releasing a VideoCapture while _capture_loop is in grab/retrieve is
        # an OpenCV native-code use-after-free. If a driver call is stuck, fail
        # without an unsafe release instead of crashing the Python process.
        if not self._capture_lock.acquire(timeout=2.0):
            raise RuntimeError(
                "Timed out waiting for GelSight capture thread before releasing cameras"
            )
        try:
            for capture in self._captures:
                capture.release()
        finally:
            self._capture_lock.release()
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

    def read_policy_packet(self) -> tuple[np.ndarray, dict[str, Any]]:
        return self.wrist_camera.read_policy_packet()

    def read_policy_packet_with_raw(
        self,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        return self.wrist_camera.read_policy_packet_with_raw()

    def read_raw_packet(self) -> CameraFramePacket:
        return self.wrist_camera.read_raw_packet()

    def camera_intrinsics(self) -> dict[str, Any]:
        return self.wrist_camera.camera_intrinsics()

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


class WristRGBHistoryBuffer:
    """Maintain the exact oldest-to-newest RGB history required by TacEx.

    The three-frame TacEx policies initialise their simulator history by
    repeating the first post-reset frame.  Doing the same here avoids a
    synthetic black history at the first two real-robot policy ticks.
    """

    def __init__(self, frames: int, image_shape: tuple[int, int, int]) -> None:
        if frames < 1:
            raise ValueError("wrist RGB history must contain at least one frame")
        if len(image_shape) != 3 or image_shape[2] != 3:
            raise ValueError("wrist RGB history image shape must be HxWx3")
        self.frames = int(frames)
        self.image_shape = tuple(int(value) for value in image_shape)
        self._images: deque[np.ndarray] = deque(maxlen=self.frames)

    def update(self, image: np.ndarray) -> np.ndarray:
        frame = np.asarray(image, dtype=np.uint8)
        if frame.shape != self.image_shape:
            raise ValueError(
                f"Expected RGB frame {self.image_shape}, got {frame.shape}"
            )
        frame = np.array(frame, dtype=np.uint8, order="C", copy=True)
        if not self._images:
            self._images.extend(frame.copy() for _ in range(self.frames))
        else:
            self._images.append(frame)
        return self.current()

    def current(self) -> np.ndarray:
        if len(self._images) != self.frames:
            raise RuntimeError("RGB history is not initialized; call update(frame) first")
        return np.stack(tuple(self._images), axis=0)


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
        residual_settings: ResidualDeploySettings | None = None,
        real_rl_settings: RealRLDeploySettings | None = None,
        rlpd_settings: RLPDDeploySettings | None = None,
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

        signature = dict(self.metadata["input_signature"])
        declared_input_order = self.metadata.get("input_order", ())
        tactile_input_names = (
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        )
        # The first three-frame exporter used a compact ``tactile_rgb`` shape
        # entry while its TorchScript input_order already named all four camera
        # tensors. Expand that unambiguous legacy form here. New exports write
        # the four names directly.
        if (
            "tactile_rgb" in signature
            and all(name in declared_input_order for name in tactile_input_names)
            and not any(name in signature for name in tactile_input_names)
        ):
            tactile_shape = signature["tactile_rgb"]
            if not isinstance(tactile_shape, list) or len(tactile_shape) != 3:
                raise ValueError("tactile_rgb must have input signature [H,W,3]")
            signature.update({name: tactile_shape for name in tactile_input_names})
        output_signature = self.metadata.get("output_signature", {})
        self.history_dim = int(signature["action_history"][0])
        self.proprio_dim = int(signature["proprio_obs"][0])
        has_single_rgb = "wrist_rgb" in signature
        has_rgb_history = "wrist_rgb_history" in signature
        if has_single_rgb == has_rgb_history:
            raise ValueError(
                "TorchScript metadata must provide exactly one of wrist_rgb or wrist_rgb_history"
            )
        self.uses_wrist_rgb_history = has_rgb_history
        self.rgb_history_frames = 1
        if has_rgb_history:
            history_shape = tuple(int(value) for value in signature["wrist_rgb_history"])
            if len(history_shape) != 4 or history_shape[0] < 1 or history_shape[-1] != 3:
                raise ValueError("wrist_rgb_history must have input signature [T,H,W,3]")
            self.rgb_history_frames, self.rgb_height, self.rgb_width, _ = history_shape
            self.rgb_input_name = "wrist_rgb_history"
        else:
            rgb_shape = tuple(int(value) for value in signature["wrist_rgb"])
            if len(rgb_shape) != 3 or rgb_shape[-1] != 3:
                raise ValueError("wrist_rgb must have input signature [H,W,3]")
            self.rgb_height, self.rgb_width, _ = rgb_shape
            self.rgb_input_name = "wrist_rgb"
        self.rgb_input_shape = (
            (self.rgb_history_frames, self.rgb_height, self.rgb_width, 3)
            if self.uses_wrist_rgb_history
            else (self.rgb_height, self.rgb_width, 3)
        )
        self.gelsight_input_shapes: dict[str, tuple[int, int, int]] = {}
        gelsight_current_names = ("gsmini_left_rgb", "gsmini_right_rgb")
        gelsight_reference_names = (
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        )
        for name in (*gelsight_current_names, *gelsight_reference_names):
            if name in signature:
                shape = tuple(int(value) for value in signature[name])
                if len(shape) != 3 or shape[2] != 3:
                    raise ValueError(f"{name} must have an HxWx3 input signature")
                self.gelsight_input_shapes[name] = shape
        current_inputs = [name for name in gelsight_current_names if name in signature]
        reference_inputs = [name for name in gelsight_reference_names if name in signature]
        if len(current_inputs) not in (0, 2) or len(reference_inputs) not in (0, 2):
            raise ValueError(
                "TorchScript metadata must provide complete left/right GelSight "
                "current and reference image pairs"
            )
        if reference_inputs and not current_inputs:
            raise ValueError("GelSight reference inputs require current GelSight inputs")
        self.has_gelsight_inputs = bool(current_inputs)
        self.has_gelsight_reference_inputs = bool(reference_inputs)
        if self.has_gelsight_reference_inputs:
            for current, reference in zip(gelsight_current_names, gelsight_reference_names):
                if self.gelsight_input_shapes[current] != self.gelsight_input_shapes[reference]:
                    raise ValueError(
                        f"{reference} must have the same shape as {current}"
                    )
        self.action_dim = int(output_signature.get("mean_actions", [self.history_dim])[0])
        self.kind = str(self.metadata.get("kind", "legacy_e2e_torchscript"))
        self.is_tacex_rma_student = self.kind == "tacex_rma_student_torchscript"
        self.is_tacex_rma_xy_student = self.kind == "tacex_rma_xy_student_torchscript"
        self.is_tacex_rma_direct_action_student = (
            self.kind == "tacex_rma_direct_action_student_torchscript"
        )
        self.is_tacex_rma_x040_wide_direct_action_student = (
            self.kind == "tacex_rma_x040_wide_direct_action_torchscript"
        )
        self.is_tacex_rma_x040_wide_three_frame_direct_action_student = (
            self.kind == "tacex_rma_x040_wide_three_frame_direct_action_torchscript"
        )
        self.is_tacex_rma_gelsight_size_buckets_student = (
            self.kind == "tacex_rma_gelsight_size_buckets_student_torchscript"
        )
        self.is_tacex_rma_gelsight_size_buckets_progress_student = (
            self.kind
            == "tacex_rma_gelsight_size_buckets_progress_student_torchscript"
        )
        self.is_tacex_rma_gelsight_x040_three_frame_student = (
            self.kind == "tacex_rma_gelsight_x040_dr_three_frame_student_torchscript"
        )
        self.contact_force_dim = 0
        self.contact_force_threshold_n: float | None = None
        if "contact_force_n" in signature:
            shape = tuple(int(value) for value in signature["contact_force_n"])
            if shape != (2,):
                raise ValueError("contact_force_n must have input signature [2]")
            threshold = signature.get("contact_force_threshold_n")
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or float(threshold) <= 0.0
            ):
                raise ValueError("contact_force_threshold_n must be finite and positive")
            self.contact_force_dim = 2
            self.contact_force_threshold_n = float(threshold)
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
        default_order = ["action_history", "proprio_obs", self.rgb_input_name]
        if self.has_gelsight_inputs:
            default_order.extend(["gsmini_left_rgb", "gsmini_right_rgb"])
        if self.has_gelsight_reference_inputs:
            default_order.extend(gelsight_reference_names)
        if self.contact_force_dim:
            default_order.append("contact_force_n")
        self.input_order = list(self.metadata.get("input_order", default_order))
        if sorted(self.input_order) != sorted(default_order):
            raise ValueError(
                "TorchScript metadata input_order must contain exactly "
                + ", ".join(default_order)
            )
        expected_tacex_order = [self.rgb_input_name, "proprio_obs", "action_history"]
        if self.has_gelsight_inputs:
            expected_tacex_order.extend(["gsmini_left_rgb", "gsmini_right_rgb"])
        if (
            self.is_tacex_rma_student
            or self.is_tacex_rma_direct_action_student
            or self.is_tacex_rma_x040_wide_direct_action_student
            or self.is_tacex_rma_x040_wide_three_frame_direct_action_student
        ) and self.input_order != expected_tacex_order:
            raise ValueError(
                "TacEx RMA Student requires input_order "
                f"{expected_tacex_order}"
            )
        if (
            self.is_tacex_rma_gelsight_size_buckets_student
            or self.is_tacex_rma_gelsight_size_buckets_progress_student
            or self.is_tacex_rma_gelsight_x040_three_frame_student
        ):
            expected_reference_order = [
                self.rgb_input_name,
                "proprio_obs",
                "action_history",
                "gsmini_left_rgb",
                "gsmini_right_rgb",
                "gsmini_left_reference_rgb",
                "gsmini_right_reference_rgb",
            ]
            if self.input_order != expected_reference_order:
                raise ValueError(
                    "TacEx GelSight reference Student requires input_order "
                    f"{expected_reference_order}"
                )
        if self.is_tacex_rma_xy_student:
            expected_xy_order = [
                "wrist_rgb",
                "proprio_obs",
                "action_history",
                "contact_force_n",
            ]
            if self.input_order != expected_xy_order:
                raise ValueError(
                    f"TacEx RMA XY Student requires input_order {expected_xy_order}"
                )
        self.residual_runtime: ResidualPolicyRuntime | None = None
        self.real_rl_runtime: RealRLPolicyRuntime | None = None
        self.rlpd_runtime: RLPDPolicyRuntime | None = None
        if residual_settings is not None and real_rl_settings is not None:
            raise ValueError("Residual BC and Real-RL are mutually exclusive")
        if rlpd_settings is not None and (
            residual_settings is not None or real_rl_settings is not None
        ):
            raise ValueError("RLPD, Residual BC, and legacy Real-RL are mutually exclusive")
        if residual_settings is not None:
            if override_enabled:
                raise ValueError("Residual BC cannot be combined with RMA input overrides")
            required_methods = ("encode_visual", "tactile_encoder", "normalizer", "action_head")
            missing_methods = [name for name in required_methods if not hasattr(self.model, name)]
            if missing_methods:
                raise ValueError(
                    "Base TorchScript does not expose Residual BC feature methods: "
                    + ", ".join(missing_methods)
                )
            self.residual_runtime = ResidualPolicyRuntime(
                residual_settings,
                base_model_path=self.model_path,
                base_kind=self.kind,
                device=self.device,
            )
        if real_rl_settings is not None:
            if override_enabled:
                raise ValueError("Real-RL cannot be combined with RMA input overrides")
            validate_actor_feature_contract(self.model)
            self.real_rl_runtime = RealRLPolicyRuntime(
                real_rl_settings,
                base_model_path=self.model_path,
                base_kind=self.kind,
                device=self.device,
            )
        if rlpd_settings is not None:
            if override_enabled:
                raise ValueError("RLPD cannot be combined with RMA input overrides")
            self.rlpd_runtime = RLPDPolicyRuntime(
                rlpd_settings,
                model=self.model,
                metadata=self.metadata,
                model_path=self.model_path,
                device=self.device,
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

    def _predict_gelsight_auxiliary(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        tactile_tensors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the GelSight export's auxiliary position/contact outputs.

        These outputs are diagnostics only.  The policy action always comes
        from the regular deployed ``forward`` method, so enabling progress
        logging cannot alter the command sent to the robot.
        """
        if not hasattr(self.model, "forward_with_auxiliary"):
            raise RuntimeError(
                "TacEx GelSight export does not expose forward_with_auxiliary"
            )
        outputs = self.model.forward_with_auxiliary(
            wrist_rgb,
            proprio_obs,
            action_history,
            tactile_tensors["gsmini_left_rgb"],
            tactile_tensors["gsmini_right_rgb"],
            tactile_tensors["gsmini_left_reference_rgb"],
            tactile_tensors["gsmini_right_reference_rgb"],
        )
        if not isinstance(outputs, tuple) or len(outputs) != 4:
            raise RuntimeError(
                "GelSight forward_with_auxiliary returned an unexpected output"
            )
        _actions, _normalized_position, contact_logits, _heatmap = outputs
        if tuple(contact_logits.shape) != (int(wrist_rgb.shape[0]), 2):
            raise RuntimeError(
                "Expected GelSight auxiliary contact logits [N,2], got "
                f"{tuple(contact_logits.shape)}"
            )
        return _normalized_position, contact_logits

    def _apply_residual_bc(
        self,
        base_action: torch.Tensor,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        tactile_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        runtime = self.residual_runtime
        if runtime is None:
            return base_action
        required_tactile = (
            "gsmini_left_rgb",
            "gsmini_right_rgb",
            "gsmini_left_reference_rgb",
            "gsmini_right_reference_rgb",
        )
        missing = [name for name in required_tactile if name not in tactile_tensors]
        if missing:
            raise RuntimeError("Residual BC is missing GelSight inputs: " + ", ".join(missing))
        actor_features, recomputed_base = extract_actor_features(
            self.model,
            wrist_rgb,
            proprio_obs,
            action_history,
            tactile_tensors["gsmini_left_rgb"],
            tactile_tensors["gsmini_right_rgb"],
            tactile_tensors["gsmini_left_reference_rgb"],
            tactile_tensors["gsmini_right_reference_rgb"],
        )
        base_error = float(torch.max(torch.abs(recomputed_base - base_action)).item())
        if base_error > 5e-4:
            raise RuntimeError(
                "Residual feature path does not reproduce the base action: "
                f"max_abs_error={base_error:.6g}"
            )
        final, predicted, applied = runtime.apply(actor_features, base_action)
        self.last_inference_info["residual_bc"] = {
            "base_action": base_action.detach().cpu().reshape(-1).tolist(),
            "predicted_residual_xyz": predicted.detach().cpu().reshape(-1).tolist(),
            "applied_residual_xyz": applied.detach().cpu().reshape(-1).tolist(),
            "final_action": final.detach().cpu().reshape(-1).tolist(),
            "scale": float(runtime.settings.scale),
            "max_abs": float(runtime.settings.max_abs),
            "model_sha256": runtime.model_sha256,
            "base_recompute_max_abs_error": base_error,
        }
        return final

    def _apply_real_rl(
        self,
        base_action: torch.Tensor,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        tactile_tensors: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        runtime = self.real_rl_runtime
        if runtime is None:
            return base_action
        required = (
            "gsmini_left_rgb", "gsmini_right_rgb",
            "gsmini_left_reference_rgb", "gsmini_right_reference_rgb",
        )
        missing = [name for name in required if name not in tactile_tensors]
        if missing:
            raise RuntimeError("Real-RL is missing GelSight inputs: " + ", ".join(missing))
        features, recomputed = extract_actor_features(
            self.model,
            wrist_rgb,
            proprio_obs,
            action_history,
            tactile_tensors["gsmini_left_rgb"],
            tactile_tensors["gsmini_right_rgb"],
            tactile_tensors["gsmini_left_reference_rgb"],
            tactile_tensors["gsmini_right_reference_rgb"],
        )
        base_error = float(torch.max(torch.abs(recomputed - base_action)).item())
        if base_error > 5e-4:
            raise RuntimeError(
                "Real-RL feature path does not reproduce the base action: "
                f"max_abs_error={base_error:.6g}"
            )
        final, info = runtime.apply(features, base_action)
        info["base_recompute_max_abs_error"] = base_error
        self.last_inference_info["real_rl"] = info
        return final

    def predict(
        self,
        action_history: np.ndarray,
        proprio_obs: np.ndarray,
        wrist_rgb: np.ndarray,
        gsmini_left_rgb: np.ndarray | None = None,
        gsmini_right_rgb: np.ndarray | None = None,
        gsmini_left_reference_rgb: np.ndarray | None = None,
        gsmini_right_reference_rgb: np.ndarray | None = None,
        contact_force_n: np.ndarray | None = None,
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
            expected_rgb_shape = self.rgb_input_shape
            if writable_rgb.shape != expected_rgb_shape:
                raise ValueError(
                    f"Expected {self.rgb_input_name} shape {expected_rgb_shape}, "
                    f"got {writable_rgb.shape}"
                )
            rgb_tensor = torch.as_tensor(
                writable_rgb, dtype=torch.uint8, device=self.device
            ).unsqueeze(0)
        tactile_tensors: dict[str, torch.Tensor] = {}
        tactile_values = {
            "gsmini_left_rgb": gsmini_left_rgb,
            "gsmini_right_rgb": gsmini_right_rgb,
            "gsmini_left_reference_rgb": gsmini_left_reference_rgb,
            "gsmini_right_reference_rgb": gsmini_right_reference_rgb,
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
        contact_force_tensor: torch.Tensor | None = None
        if self.contact_force_dim:
            if contact_force_n is None:
                raise ValueError("Model requires contact_force_n, but none was supplied")
            force = np.asarray(contact_force_n, dtype=np.float32).reshape(-1)
            if force.shape != (self.contact_force_dim,) or not np.all(np.isfinite(force)):
                raise ValueError(
                    f"contact_force_n must contain {self.contact_force_dim} finite values"
                )
            contact_force_tensor = torch.as_tensor(
                force, dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        override_enabled = (
            self.rma_position_source != "vision" or self.rma_contact_source != "vision"
        )
        with torch.inference_mode():
            if not override_enabled:
                tensors = {
                    "action_history": action_tensor,
                    "proprio_obs": proprio_tensor,
                    self.rgb_input_name: rgb_tensor,
                    **tactile_tensors,
                }
                if contact_force_tensor is not None:
                    tensors["contact_force_n"] = contact_force_tensor
                output = self.model(*(tensors[name] for name in self.input_order))
                self.last_inference_info = {
                    "rma_position_source": "vision",
                    "rma_contact_source": "vision",
                    "vision_bypassed": False,
                } if self.is_tacex_rma_student else {}
                if self.is_tacex_rma_xy_student:
                    assert contact_force_tensor is not None
                    self.last_inference_info = {
                        "rma_contact_force_n": contact_force_tensor.detach()
                        .cpu()
                        .reshape(-1)
                        .tolist(),
                        "rma_grasped": bool(
                            torch.all(
                                contact_force_tensor >= float(self.contact_force_threshold_n)
                            ).item()
                        ),
                    }
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
                elif (
                    collect_rma_debug
                    and (
                        self.is_tacex_rma_gelsight_size_buckets_student
                        or self.is_tacex_rma_gelsight_size_buckets_progress_student
                    )
                    and hasattr(self.model, "forward_with_auxiliary")
                ):
                    assert rgb_tensor is not None
                    _normalized_position, contact_logits = self._predict_gelsight_auxiliary(
                        rgb_tensor,
                        proprio_tensor,
                        action_tensor,
                        tactile_tensors,
                    )
                    contact_state = torch.sigmoid(contact_logits)
                    contact_active = contact_logits >= 0.0
                    self.last_inference_info.update(
                        {
                            "contact_state": contact_state.detach()
                            .cpu()
                            .reshape(-1)
                            .tolist(),
                            "contact_active": contact_active.detach()
                            .cpu()
                            .reshape(-1)
                            .tolist(),
                            "rma_contact_source": "gelsight_auxiliary_logits",
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

        if self.is_tacex_rma_gelsight_x040_three_frame_student:
            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError(
                    "TacEx GelSight X040 three-frame Student returned an unexpected output"
                )
            action, contact_probability, cube_position_root = output
            batch = int(action_tensor.shape[0])
            if (
                tuple(action.shape) != (batch, 4)
                or tuple(contact_probability.shape) != (batch, 2)
                or tuple(cube_position_root.shape) != (batch, 3)
                or not torch.isfinite(action).all()
                or not torch.isfinite(contact_probability).all()
                or not torch.isfinite(cube_position_root).all()
            ):
                raise RuntimeError(
                    "TacEx GelSight X040 three-frame Student output contract is invalid"
                )
            if collect_rma_debug:
                self.last_inference_info = {
                    "contact_probability": contact_probability.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                    "cube_position_root": cube_position_root.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                }
            output = action
        if self.rlpd_runtime is not None:
            assert rgb_tensor is not None
            self.rlpd_runtime.prepare(
                rgb_tensor,
                proprio_tensor,
                action_tensor,
                tactile_tensors,
                output,
            )
        if self.residual_runtime is not None:
            assert rgb_tensor is not None
            output = self._apply_residual_bc(
                output,
                rgb_tensor,
                proprio_tensor,
                action_tensor,
                tactile_tensors,
            )
        if self.real_rl_runtime is not None:
            assert rgb_tensor is not None
            output = self._apply_real_rl(
                output,
                rgb_tensor,
                proprio_tensor,
                action_tensor,
                tactile_tensors,
            )
        output = output.detach().cpu().numpy().reshape(-1)
        return output

    def apply_rlpd_post_limit(self, base_limited: np.ndarray) -> np.ndarray:
        if self.rlpd_runtime is None:
            return np.asarray(base_limited, dtype=np.float32)
        candidate, info = self.rlpd_runtime.apply_post_limit(base_limited)
        self.last_inference_info["rlpd"] = info
        return candidate


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


def build_rma_contact_force_input(
    observation: RobotObservation,
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
) -> np.ndarray | None:
    """构造 XY RMA Student 可选的双侧接触力输入。

    真机没有独立的左右指尖力读数。腕部外力包含机械臂/桌面接触，无法区分
    两根手指，不能替代这里的输入。夹爪标志近似只复现 Actor 的
    ``双方均 >= threshold`` 二值特征。
    """
    if not getattr(bundle, "contact_force_dim", 0):
        return None
    source = config.model.rma_contact_force_source
    if source == "gripper_is_grasped":
        if observation.gripper_is_grasped is None:
            raise RuntimeError(
                "RMA XY contact input requires gripper_is_grasped, but the gripper state "
                "is unavailable"
            )
        assert bundle.contact_force_threshold_n is not None
        value = bundle.contact_force_threshold_n if observation.gripper_is_grasped else 0.0
        return np.full(bundle.contact_force_dim, value, dtype=np.float32)
    if source == "zeros":
        return np.zeros(bundle.contact_force_dim, dtype=np.float32)
    raise ValueError(
        "TacEx RMA XY Student requires model.rma_contact_force_source to be "
        "'gripper_is_grasped' or 'zeros'"
    )


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
        if bundle.is_tacex_rma_gelsight_x040_three_frame_student:
            _validate_tacex_rma_gelsight_x040_three_frame_student_contract(
                bundle, config, history_buffer.scale
            )
        elif bundle.is_tacex_rma_gelsight_size_buckets_progress_student:
            _validate_tacex_rma_gelsight_size_buckets_progress_student_contract(
                bundle, config, history_buffer.scale
            )
        elif bundle.is_tacex_rma_gelsight_size_buckets_student:
            _validate_tacex_rma_gelsight_size_buckets_student_contract(
                bundle, config, history_buffer.scale
            )
        elif bundle.is_tacex_rma_student:
            _validate_tacex_rma_student_contract(bundle, config, history_buffer.scale)
        elif bundle.is_tacex_rma_xy_student:
            _validate_tacex_rma_xy_student_contract(bundle, config, history_buffer.scale)
        elif bundle.is_tacex_rma_direct_action_student:
            _validate_tacex_rma_direct_action_student_contract(
                bundle, config, history_buffer.scale
            )
        elif bundle.is_tacex_rma_x040_wide_direct_action_student:
            _validate_tacex_rma_x040_wide_direct_action_student_contract(
                bundle, config, history_buffer.scale
            )
        elif bundle.is_tacex_rma_x040_wide_three_frame_direct_action_student:
            _validate_tacex_rma_x040_wide_three_frame_direct_action_student_contract(
                bundle, config, history_buffer.scale
            )
        else:
            _validate_policy_contract(bundle, config, history_buffer.scale)


def reject_evaluation_only_motion(
    bundle: BundleTorchScriptPolicy,
    execute_motion: bool,
    rlpd_settings: RLPDDeploySettings | None = None,
) -> None:
    """Reject exports whose recorded behavior is not approved for direct motion."""

    if (
        execute_motion
        and getattr(bundle, "is_tacex_rma_x040_wide_three_frame_direct_action_student", False) is True
        and bundle.metadata.get("legacy_appearance_evaluation_only") is True
    ):
        raise ValueError(
            "This TorchScript export is marked legacy_appearance_evaluation_only by TacEx; "
            "it may be used with --validate-only or --preview-only, but cannot command "
            "the real robot. Re-export a deployment-approved checkpoint first."
        )
    if (
        execute_motion
        and getattr(
            bundle,
            "is_tacex_rma_gelsight_size_buckets_progress_student",
            False,
        ) is True
        and rlpd_settings is None
    ):
        raise ValueError(
            "The 0911 Progress Student is marked motion_authorization='rlpd_only' "
            "because its offline behavior audit rejected direct motion. It may be used "
            "with --validate-only/--preview-only, RLPD expert takeover, or a validated "
            "RLPD checkpoint; direct base-policy motion is disabled."
        )


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


def _validate_tacex_rma_gelsight_size_buckets_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """Fail closed on fixed-size GelSight reference-delta Students."""
    policy_name = "TacEx GelSight Size-Buckets Reference Student"
    metadata = bundle.metadata
    version = metadata.get("version")
    if version not in (1, 2):
        raise ValueError(f"{policy_name} deployment requires metadata version 1 or 2")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"{policy_name} metadata is missing torchscript_sha256")
    if _sha256_file(bundle.model_path) != expected_sha256:
        raise ValueError(f"{policy_name} TorchScript SHA-256 does not match metadata")
    expected_tactile_shapes = {
        "gsmini_left_rgb": (96, 128, 3),
        "gsmini_right_rgb": (96, 128, 3),
        "gsmini_left_reference_rgb": (96, 128, 3),
        "gsmini_right_reference_rgb": (96, 128, 3),
    }
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
        or bundle.gelsight_input_shapes != expected_tactile_shapes
        or not bundle.has_gelsight_reference_inputs
        or bundle.uses_wrist_rgb_history
        or (bundle.rgb_height, bundle.rgb_width) != (224, 224)
    ):
        raise ValueError(
            f"{policy_name} requires wrist_rgb[224,224,3], proprio_obs[15], "
            "action_history[4], two current and two reference GelSight inputs, "
            "and actions[4]"
        )
    if metadata.get("input_order") != [
        "wrist_rgb",
        "proprio_obs",
        "action_history",
        "gsmini_left_rgb",
        "gsmini_right_rgb",
        "gsmini_left_reference_rgb",
        "gsmini_right_reference_rgb",
    ]:
        raise ValueError(f"{policy_name} input_order is inconsistent")
    if metadata.get("tactile_delta") != (
        "signed_float32_current_minus_reference_div_255"
    ):
        raise ValueError(f"{policy_name} tactile-delta convention is inconsistent")
    if version == 1:
        if metadata.get("contact_to_actor") != "hard_binary_logit_ge_0":
            raise ValueError(f"{policy_name} contact-to-Actor convention is inconsistent")
    else:
        if metadata.get("contact_to_actor") != (
            "continuous_tactile_features; logits_are_auxiliary_only"
        ):
            raise ValueError(f"{policy_name} v2 contact-to-Actor convention is inconsistent")
        model_contract = metadata.get("student_model_contract")
        if not isinstance(model_contract, dict) or (
            model_contract.get("runtime_input_order") != metadata.get("input_order")
            or model_contract.get("actor_feature_dim") != 1043
            or model_contract.get("runtime_privileged_inputs") != []
        ):
            raise ValueError(f"{policy_name} v2 Student model contract is inconsistent")
        deployment_contract = metadata.get("deployment_contract")
        if not isinstance(deployment_contract, dict):
            raise ValueError(f"{policy_name} v2 deployment contract is missing")
        frequency = deployment_contract.get("policy_frequency_hz")
        episode_length = deployment_contract.get("episode_length_s")
        max_steps = deployment_contract.get("max_episode_length_steps")
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, (int, float))
            or not math.isclose(float(frequency), 30.0, rel_tol=0.0, abs_tol=1e-8)
            or isinstance(episode_length, bool)
            or not isinstance(episode_length, (int, float))
            or not math.isclose(float(episode_length), 5.0, rel_tol=0.0, abs_tol=1e-8)
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps != 150
        ):
            raise ValueError(f"{policy_name} v2 deployment step contract is inconsistent")
    validation = metadata.get("validation")
    if not isinstance(validation, dict) or any(
        not isinstance(validation.get(name), (int, float))
        or not math.isfinite(float(validation[name]))
        or float(validation[name]) > 1.0e-5
        for name in ("cpu_batch_1_max_abs_error", "cpu_batch_8_max_abs_error")
    ):
        raise ValueError(f"{policy_name} CPU TorchScript validation is invalid")
    if torch.device(config.model.device).type == "cuda" and metadata.get("cuda_validation") is not True:
        raise ValueError(f"{policy_name} CUDA deployment requires successful CUDA validation")
    if not config.tactile_camera.enabled:
        raise ValueError(f"{policy_name} requires tactile_camera.enabled=true")
    _validate_tacex_rma_x040_wide_real_runtime(
        config, history_scale_vector, policy_name
    )


def _validate_tacex_rma_gelsight_size_buckets_progress_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """Validate the isolated 0911 terminal-success Progress Student profile."""

    policy_name = "TacEx GelSight Size-Buckets Progress Student"
    _validate_tacex_rma_gelsight_size_buckets_student_contract(
        bundle, config, history_scale_vector
    )
    metadata = bundle.metadata
    expected_task = (
        "TacEx-Sim2Real-Cube-Real-Alignment-RMA-GelSight-Size-Buckets-"
        "Progress-Student-DR-v0"
    )
    if metadata.get("task") != expected_task:
        raise ValueError(f"{policy_name} task provenance is inconsistent")
    if metadata.get("behavior_profile") != "gelsight_size_buckets_progress_rlpd_base_v1":
        raise ValueError(f"{policy_name} behavior profile is inconsistent")
    if metadata.get("motion_authorization") != "rlpd_only":
        raise ValueError(f"{policy_name} must remain restricted to RLPD motion")

    source = metadata.get("source_export")
    if not isinstance(source, dict) or source != {
        "kind": "tacex_rma_gelsight_size_buckets_student_torchscript",
        "version": 2,
        "metadata_filename": "gelsight_reference_student_student_100000.json",
        "metadata_sha256": "71ea77b146f6ec9b5e4ed05c8298ad49eccf6cf11413820d62dfe5afb744e564",
    }:
        raise ValueError(f"{policy_name} source export provenance is inconsistent")
    source_metadata_path = bundle.metadata_path.with_name(source["metadata_filename"])
    if (
        not source_metadata_path.is_file()
        or _sha256_file(source_metadata_path) != source["metadata_sha256"]
    ):
        raise ValueError(f"{policy_name} source metadata file/hash is inconsistent")

    progress = metadata.get("progress_training_contract")
    expected_progress = {
        "reward": "signed_reach_lift_contact_progress",
        "success": "once_on_confirmed_terminal_success",
        "success_lift_delta_m": 0.035,
        "success_hold_steps": 5,
        "action_magnitude_penalty_weight": 0.05,
        "excess_contact_force_threshold_n": 15.0,
        "excess_contact_force_quadratic_weight": 5.0,
    }
    if progress != expected_progress:
        raise ValueError(f"{policy_name} progress reward/success contract is inconsistent")
    deployment = metadata.get("deployment_contract")
    if (
        not isinstance(deployment, dict)
        or deployment.get("success_stop")
        != "external_apriltag_or_operator_required"
        or config.runner.steps > int(deployment.get("max_episode_length_steps", 0))
    ):
        raise ValueError(
            f"{policy_name} requires an external/operator success stop and at most "
            "150 policy steps"
        )

    geometry = metadata.get("gelsight_geometry")
    if not isinstance(geometry, dict) or geometry != {
        "version": 3,
        "center_offset_hand_m": [0.0, 0.0, 0.1392],
        "lowest_point_offset_hand_m": [0.0, 0.0, 0.1563],
        "flange_to_ee_translation_m": [0.0, 0.0, 0.1034],
        "tool_tcp_offset_ee_m": [0.0, 0.0, 0.0529],
    }:
        raise ValueError(f"{policy_name} GelSight geometry contract is inconsistent")
    if not np.allclose(
        np.asarray(config.tool_tcp_offset_ee_m, dtype=np.float64),
        [0.0, 0.0, 0.0529],
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            f"{policy_name} requires tool_tcp_offset_ee_m=[0, 0, 0.0529] "
            "for the GelSight v7 lowest-point workspace check"
        )
    audit = metadata.get("behavioral_validation")
    if (
        not isinstance(audit, dict)
        or audit.get("status") != "direct_motion_rejected"
        or audit.get("approved_runtime") != "rlpd_residual_only"
    ):
        raise ValueError(f"{policy_name} behavior audit restriction is missing")
    cuda_details = metadata.get("cuda_validation_details")
    cuda_error = (
        cuda_details.get("cpu_cuda_max_abs_error")
        if isinstance(cuda_details, dict)
        else None
    )
    cuda_tolerance = (
        cuda_details.get("tolerance") if isinstance(cuda_details, dict) else None
    )
    if (
        not isinstance(cuda_details, dict)
        or cuda_details.get("device") != "cuda:0"
        or not isinstance(cuda_details.get("synthetic_cases"), int)
        or cuda_details["synthetic_cases"] < 1
        or isinstance(cuda_error, bool)
        or not isinstance(cuda_error, (int, float))
        or not math.isfinite(float(cuda_error))
        or float(cuda_error) < 0.0
        or isinstance(cuda_tolerance, bool)
        or not isinstance(cuda_tolerance, (int, float))
        or not math.isfinite(float(cuda_tolerance))
        or float(cuda_tolerance) <= 0.0
        or float(cuda_error) > float(cuda_tolerance)
    ):
        raise ValueError(f"{policy_name} CUDA validation details are inconsistent")


def _validate_tacex_rma_gelsight_x040_three_frame_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """Fail closed on supported 0814/0815 X040 three-frame GelSight Students."""

    policy_name = "TacEx GelSight X040 Three-Frame Student"
    metadata = bundle.metadata
    if metadata.get("version") != 1:
        raise ValueError(f"{policy_name} deployment requires metadata version 1")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"{policy_name} metadata is missing torchscript_sha256")
    if _sha256_file(bundle.model_path) != expected_sha256:
        raise ValueError(f"{policy_name} TorchScript SHA-256 does not match metadata")
    expected_tactile_shapes = {
        "gsmini_left_rgb": (96, 128, 3),
        "gsmini_right_rgb": (96, 128, 3),
        "gsmini_left_reference_rgb": (96, 128, 3),
        "gsmini_right_reference_rgb": (96, 128, 3),
    }
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
        or bundle.gelsight_input_shapes != expected_tactile_shapes
        or not bundle.has_gelsight_reference_inputs
        or not bundle.uses_wrist_rgb_history
        or bundle.rgb_history_frames != 3
        or (bundle.rgb_height, bundle.rgb_width) != (224, 224)
    ):
        raise ValueError(
            f"{policy_name} requires wrist_rgb_history[3,224,224,3], "
            "proprio_obs[15], action_history[4], two current and two reference "
            "GelSight inputs, and actions[4]"
        )
    expected_input_order = [
        "wrist_rgb_history",
        "proprio_obs",
        "action_history",
        "gsmini_left_rgb",
        "gsmini_right_rgb",
        "gsmini_left_reference_rgb",
        "gsmini_right_reference_rgb",
    ]
    if metadata.get("input_order") != expected_input_order:
        raise ValueError(f"{policy_name} input_order is inconsistent")
    if metadata.get("output_signature") != {
        "action": [4],
        "left_right_contact_probability": [2],
        "cube_position_root_m": [3],
    }:
        raise ValueError(f"{policy_name} output signature is inconsistent")
    if metadata.get("input_signature", {}).get("tactile_delta") != (
        "signed_float32_current_minus_reference_div_255"
    ):
        raise ValueError(f"{policy_name} tactile-delta convention is inconsistent")
    model_contract = metadata.get("student_model_contract")
    if not isinstance(model_contract, dict) or (
        model_contract.get("runtime_input_order") != expected_input_order
        or model_contract.get("runtime_output") != metadata.get("output_signature")
        or model_contract.get("actor_feature_dim") != 1043
        or model_contract.get("runtime_privileged_inputs") != []
    ):
        raise ValueError(f"{policy_name} model contract is inconsistent")
    model_version = model_contract.get("model_version")
    if model_version not in (2, 3):
        raise ValueError(f"{policy_name} model contract version is unsupported")
    if (
        model_version == 3
        and model_contract.get("position_normalization")
        != "x040_wide_robot_root_xyz"
    ):
        raise ValueError(f"{policy_name} v3 position-normalization is inconsistent")
    validation = metadata.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"{policy_name} CPU TorchScript validation is missing")
    for name in ("cpu_batch_1_max_abs_error", "cpu_batch_8_max_abs_error"):
        values = validation.get(name)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) > 1.0e-5
                for value in values
            )
        ):
            raise ValueError(f"{policy_name} CPU TorchScript validation is invalid")
    if torch.device(config.model.device).type == "cuda" and metadata.get("cuda_validation") is not True:
        raise ValueError(f"{policy_name} CUDA deployment requires successful CUDA validation")
    if not config.tactile_camera.enabled:
        raise ValueError(f"{policy_name} requires tactile_camera.enabled=true")
    if config.control_mode == "streaming" and not math.isclose(
        config.streaming.policy_frequency_hz, 30.0, rel_tol=0.0, abs_tol=1.0e-8
    ):
        raise ValueError(f"{policy_name} requires streaming.policy_frequency_hz=30")
    # The corrected GelSight geometry places the lowest centered fingertip
    # 0.1613 m along panda_hand +Z. libfranka's configured O_T_EE origin is
    # already 0.1034 m along the same axis from panda_hand, so the workspace
    # point must use the remaining 0.0579 m. This offset only changes safety
    # validation; IK continues to command the configured O_T_EE frame.
    expected_tool_tcp_offset = np.asarray([0.0, 0.0, 0.0579], dtype=np.float64)
    configured_tool_tcp_offset = np.asarray(
        config.tool_tcp_offset_ee_m, dtype=np.float64
    ).reshape(-1)
    if (
        configured_tool_tcp_offset.shape != expected_tool_tcp_offset.shape
        or not np.allclose(
            configured_tool_tcp_offset,
            expected_tool_tcp_offset,
            rtol=0.0,
            atol=1.0e-9,
        )
    ):
        raise ValueError(
            f"{policy_name} requires tool_tcp_offset_ee_m=[0, 0, 0.0579] "
            "for the corrected 161.3 mm panda_hand-to-lowest-point geometry"
        )
    _validate_tacex_rma_x040_wide_real_runtime(
        config, history_scale_vector, policy_name
    )


def _validate_tacex_rma_direct_action_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """对 0809 Direct-Action Visual Student 的部署契约执行失败即拒绝的校验。"""
    metadata = bundle.metadata
    if metadata.get("version") != 1:
        raise ValueError("TacEx RMA Direct-Action Student deployment requires metadata version 1")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("TacEx RMA Direct-Action Student metadata is missing torchscript_sha256")
    actual_sha256 = _sha256_file(bundle.model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "TorchScript SHA-256 does not match metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if torch.device(config.model.device).type == "cuda":
        cuda_validation = metadata.get("cuda_validation")
        cuda_validation_atol = metadata.get("cuda_validation_atol")
        cuda_max_abs_error = (
            cuda_validation.get("max_abs_error") if isinstance(cuda_validation, dict) else None
        )
        if (
            not isinstance(cuda_validation, dict)
            or cuda_validation.get("available") is not True
            or isinstance(cuda_max_abs_error, bool)
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
                "TacEx RMA Direct-Action Student CUDA deployment requires successful "
                "CUDA validation metadata"
            )
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
    ):
        raise ValueError(
            "TacEx RMA Direct-Action Student requires wrist_rgb[224,224,3], "
            "proprio_obs[15], action_history[4], no contact_force_n, and actions[4]"
        )
    if bundle.has_gelsight_inputs:
        raise ValueError("TacEx RMA Direct-Action Student must not declare GelSight inputs")
    if (bundle.rgb_height, bundle.rgb_width) != (224, 224):
        raise ValueError("TacEx RMA Direct-Action Student requires wrist_rgb[224,224,3]")
    model_contract = metadata.get("model_contract")
    if not isinstance(model_contract, dict):
        raise ValueError("TacEx RMA Direct-Action Student metadata is missing model_contract")
    if (
        model_contract.get("model_version") != 1
        or model_contract.get("input_order") != ["wrist_rgb", "proprio_obs", "action_history"]
        or model_contract.get("input_signature") != metadata.get("input_signature")
        or model_contract.get("output_signature") != metadata.get("output_signature")
        or model_contract.get("feature_dim") != 531
        or model_contract.get("activation") != "ELU_then_tanh"
    ):
        raise ValueError("TacEx RMA Direct-Action Student model_contract is inconsistent")
    if metadata.get("runtime_privileged_inputs") != []:
        raise ValueError("TacEx RMA Direct-Action Student must not require privileged inputs")
    if config.model.history_source != "processed_action":
        raise ValueError(
            "TacEx RMA Direct-Action Student requires "
            "model.history_source='processed_action'"
        )
    if not np.allclose(history_scale_vector, np.ones(4, dtype=np.float32), rtol=0.0, atol=1e-8):
        raise ValueError(
            "TacEx RMA Direct-Action Student history is already in physical units; "
            "model.history_scale must be [1, 1, 1, 1]"
        )
    if config.model.history_delay_steps != 1:
        raise ValueError(
            "TacEx RMA Direct-Action Student requires model.history_delay_steps=1"
        )
    if config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]:
        raise ValueError(
            "TacEx RMA Direct-Action Student requires action labels [dx, dy, dz, gripper]"
        )
    expected_action_scales = np.asarray([0.05, 0.05, 0.05, 0.01], dtype=np.float32)
    configured_action_scales = np.asarray(config.action_adapter.scales, dtype=np.float32).reshape(-1)
    if (
        configured_action_scales.shape != expected_action_scales.shape
        or not np.allclose(
            configured_action_scales, expected_action_scales, rtol=0.0, atol=1e-8
        )
    ):
        raise ValueError(
            "TacEx RMA Direct-Action Student requires action scales "
            "[0.05, 0.05, 0.05, 0.01]"
        )
    if (
        config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
        or config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(
            "TacEx RMA Direct-Action Student requires normalized [-1,1] actions "
            "and delta_width gripper"
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
    ) != (640, 480, 30, 100, 34, 400, 398) or not camera.enable_crop:
        raise ValueError(
            "TacEx RMA Direct-Action Student requires D435 640x480@30 crop "
            "(100,34,400,398)"
        )

def _validate_tacex_rma_x040_wide_direct_action_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """对 X040-Wide 三输入 Direct-Action Student 执行训练契约校验。"""
    metadata = bundle.metadata
    policy_name = "TacEx RMA X040-Wide Direct-Action Student"
    # 0811 size-change exports intentionally use the compact TacEx exporter
    # metadata.  They have the same real-robot action/camera contract as the
    # earlier X040-Wide policy but no training-only position-head manifest.
    if "model_contract" not in metadata:
        _validate_tacex_rma_x040_wide_compact_contract(
            bundle, config, history_scale_vector, policy_name
        )
        return
    if metadata.get("version") != 1:
        raise ValueError(f"{policy_name} deployment requires metadata version 1")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"{policy_name} metadata is missing torchscript_sha256")
    actual_sha256 = _sha256_file(bundle.model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "TorchScript SHA-256 does not match metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if torch.device(config.model.device).type == "cuda":
        cuda_validation = metadata.get("cuda_validation")
        cuda_validation_atol = metadata.get("cuda_validation_atol")
        cuda_max_abs_error = (
            cuda_validation.get("max_abs_error") if isinstance(cuda_validation, dict) else None
        )
        if (
            not isinstance(cuda_validation, dict)
            or cuda_validation.get("available") is not True
            or isinstance(cuda_max_abs_error, bool)
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
                f"{policy_name} CUDA deployment requires successful CUDA validation metadata"
            )
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
        or bundle.has_gelsight_inputs
        or (bundle.rgb_height, bundle.rgb_width) != (224, 224)
    ):
        raise ValueError(
            f"{policy_name} requires wrist_rgb[224,224,3], proprio_obs[15], "
            "action_history[4], no contact_force_n/GelSight inputs, and actions[4]"
        )
    expected_normalization = {
        "joint_lower": [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        "joint_upper": [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        "joint_velocity_scale": [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
        "gripper_width_center_scale": [0.04, 0.04],
        "history_scale": [0.025, 0.025, 0.025, 0.005],
        "cube_position_center": [0.4, 0.0, 0.026],
        "cube_position_scale": [0.08, 0.1, 0.1],
        "gripper_position_center": [0.5, 0.0, 0.175],
        "gripper_position_scale": [0.1, 0.1, 0.15],
        "target_position_center": [-0.1, 0.0, -0.149],
        "target_position_scale": [0.18, 0.1, 0.15],
        "position_frame": "robot_root",
    }
    if metadata.get("normalization") != expected_normalization:
        raise ValueError(f"{policy_name} normalization contract is inconsistent")
    model_contract = metadata.get("model_contract")
    if not isinstance(model_contract, dict) or (
        model_contract.get("model_version") != 1
        or model_contract.get("input_order") != ["wrist_rgb", "proprio_obs", "action_history"]
        or model_contract.get("input_signature") != metadata.get("input_signature")
        or model_contract.get("output_signature") != metadata.get("output_signature")
        or model_contract.get("feature_dim") != 531
        or model_contract.get("training_only_label") != "normalized_cube_position_root[3]"
        or model_contract.get("activation") != "ELU_then_action_tanh"
        or model_contract.get("position_head") != [512, 256, 128, 3]
    ):
        raise ValueError(f"{policy_name} model_contract is inconsistent")
    expected_environment = {
        "profile": "rma_x040_wide_xyz_no_contact_v1",
        "action_dim": 4,
        "action_scales": [0.05, 0.05, 0.05, 0.01],
        "gripper_control_mode": "total_width_delta_cached_target",
        "cube_nominal_position_root_m": [0.4, 0.0, 0.026],
        "cube_reset_half_range_xy_m": [0.08, 0.1],
        "cube_position_curriculum_enabled": False,
        "cube_position_curriculum_force_full_range": True,
        "tcp_table_clearance_min_z_m": 0.011,
        "position_frame": "robot_root",
        "teacher_actor_feature_dim": 28,
        "teacher_contact_input": "none",
    }
    if metadata.get("environment_contract") != expected_environment:
        raise ValueError(f"{policy_name} environment contract is inconsistent")
    expected_initial_joints = np.asarray([
        -0.3077768694457032, -0.11490349419892411, 0.30181493015057165,
        -2.2731998141397707, 0.040613268755509774, 2.162421075317457,
        0.7543247225501507,
    ], dtype=np.float64)
    if (
        not np.allclose(
            np.asarray(config.initial_state.joint_positions, dtype=np.float64),
            expected_initial_joints,
            rtol=0.0,
            atol=1e-9,
        )
        or abs(config.initial_state.gripper_width_m - 0.040001507848501205) > 1e-8
    ):
        raise ValueError(f"{policy_name} requires the X040-Wide training initial state")
    if config.model.history_source != "processed_action":
        raise ValueError(f"{policy_name} requires model.history_source='processed_action'")
    if not np.allclose(history_scale_vector, np.ones(4, dtype=np.float32), rtol=0.0, atol=1e-8):
        raise ValueError(f"{policy_name} requires model.history_scale=[1, 1, 1, 1]")
    if config.model.history_delay_steps != 1:
        raise ValueError(f"{policy_name} requires model.history_delay_steps=1")
    expected_action_scales = np.asarray([0.05, 0.05, 0.05, 0.01], dtype=np.float32)
    if (
        config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]
        or not np.allclose(
            np.asarray(config.action_adapter.scales, dtype=np.float32).reshape(-1),
            expected_action_scales,
            rtol=0.0,
            atol=1e-8,
        )
        or config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
        or config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(f"{policy_name} action adapter is inconsistent")
    camera = config.camera
    if (
        camera.width,
        camera.height,
        camera.fps,
        camera.crop_left,
        camera.crop_top,
        camera.crop_width,
        camera.crop_height,
        camera.enable_crop,
    ) != (640, 480, 30, 100, 34, 400, 398, True):
        raise ValueError(
            f"{policy_name} requires D435 640x480@30 crop (100,34,400,398)"
        )
    x_min, x_max = 0.32, 0.48
    y_min, y_max = -0.10, 0.10
    if (
        config.workspace["minimum"][0] > x_min
        or config.workspace["maximum"][0] < x_max
        or config.workspace["minimum"][1] > y_min
        or config.workspace["maximum"][1] < y_max
        or config.workspace["minimum"][2] > 0.026
        or config.workspace["maximum"][2] < 0.026
    ):
        raise ValueError(
            f"{policy_name} workspace must contain the trained cube reset range "
            "x=[0.32,0.48], y=[-0.10,0.10], z=0.026"
        )
    if metadata.get("runtime_privileged_inputs") != []:
        raise ValueError(f"{policy_name} must not require runtime privileged inputs")


def _validate_tacex_rma_x040_wide_compact_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
    policy_name: str,
) -> None:
    """Validate the compact metadata produced by the current TacEx exporters."""

    metadata = bundle.metadata
    if metadata.get("version") != 1:
        raise ValueError(f"{policy_name} deployment requires metadata version 1")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"{policy_name} metadata is missing torchscript_sha256")
    if _sha256_file(bundle.model_path) != expected_sha256:
        raise ValueError(f"{policy_name} TorchScript SHA-256 does not match metadata")
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
        or bundle.has_gelsight_inputs
        or bundle.uses_wrist_rgb_history
        or (bundle.rgb_height, bundle.rgb_width) != (224, 224)
    ):
        raise ValueError(
            f"{policy_name} requires wrist_rgb[224,224,3], proprio_obs[15], "
            "action_history[4], no auxiliary runtime inputs, and actions[4]"
        )
    if metadata.get("input_order") != ["wrist_rgb", "proprio_obs", "action_history"]:
        raise ValueError(f"{policy_name} input_order is inconsistent")
    if metadata.get("runtime_privileged_inputs") != []:
        raise ValueError(f"{policy_name} must not require runtime privileged inputs")
    _validate_tacex_rma_x040_wide_real_runtime(
        config, history_scale_vector, policy_name
    )


def _validate_tacex_rma_x040_wide_three_frame_direct_action_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """Fail closed on three-frame TacEx visual students before robot motion."""

    policy_name = "TacEx RMA X040-Wide Three-Frame Direct-Action Student"
    metadata = bundle.metadata
    if metadata.get("version") != 1:
        raise ValueError(f"{policy_name} deployment requires metadata version 1")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(f"{policy_name} metadata is missing torchscript_sha256")
    if _sha256_file(bundle.model_path) != expected_sha256:
        raise ValueError(f"{policy_name} TorchScript SHA-256 does not match metadata")
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 0
        or bundle.has_gelsight_inputs
        or not bundle.uses_wrist_rgb_history
        or bundle.rgb_history_frames != 3
        or (bundle.rgb_height, bundle.rgb_width) != (224, 224)
    ):
        raise ValueError(
            f"{policy_name} requires wrist_rgb_history[3,224,224,3], "
            "proprio_obs[15], action_history[4], and actions[4]"
        )
    if metadata.get("input_order") != [
        "wrist_rgb_history", "proprio_obs", "action_history"
    ]:
        raise ValueError(f"{policy_name} input_order is inconsistent")
    if metadata.get("runtime_privileged_inputs") != []:
        raise ValueError(f"{policy_name} must not require runtime privileged inputs")
    contract = metadata.get("student_environment_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"{policy_name} metadata is missing student_environment_contract")
    if contract.get("action_scales") != [0.05, 0.05, 0.05, 0.01]:
        raise ValueError(f"{policy_name} action scales are inconsistent")
    history_contract = contract.get("wrist_rgb_history")
    if not isinstance(history_contract, dict) or (
        history_contract.get("shape") != [3, 224, 224, 3]
        or history_contract.get("order") != "oldest_to_newest"
        or history_contract.get("stride_policy_steps") != 1
        or history_contract.get("reset_fill") != "repeat_first_post_reset_frame"
        or history_contract.get("policy_frequency_hz") != 30
    ):
        raise ValueError(f"{policy_name} RGB-history contract is inconsistent")
    _validate_tacex_rma_x040_wide_real_runtime(
        config, history_scale_vector, policy_name
    )


def _validate_tacex_rma_x040_wide_real_runtime(
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
    policy_name: str,
) -> None:
    """The deployment values shared by current X040-Wide TacEx policies."""

    expected_initial_joints = np.asarray([
        -0.3077768694457032, -0.11490349419892411, 0.30181493015057165,
        -2.2731998141397707, 0.040613268755509774, 2.162421075317457,
        0.7543247225501507,
    ], dtype=np.float64)
    if (
        not np.allclose(np.asarray(config.initial_state.joint_positions, dtype=np.float64),
                        expected_initial_joints, rtol=0.0, atol=1e-9)
        or abs(config.initial_state.gripper_width_m - 0.040001507848501205) > 1e-8
    ):
        raise ValueError(f"{policy_name} requires the X040-Wide training initial state")
    if config.model.history_source != "processed_action":
        raise ValueError(f"{policy_name} requires model.history_source='processed_action'")
    if not np.allclose(history_scale_vector, np.ones(4, dtype=np.float32), rtol=0.0, atol=1e-8):
        raise ValueError(f"{policy_name} requires model.history_scale=[1, 1, 1, 1]")
    if config.model.history_delay_steps != 1:
        raise ValueError(f"{policy_name} requires model.history_delay_steps=1")
    if (
        config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]
        or not np.allclose(np.asarray(config.action_adapter.scales, dtype=np.float32),
                           [0.05, 0.05, 0.05, 0.01], rtol=0.0, atol=1e-8)
        or config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
        or config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(f"{policy_name} action adapter is inconsistent")
    camera = config.camera
    if (camera.width, camera.height, camera.fps, camera.crop_left, camera.crop_top,
        camera.crop_width, camera.crop_height, camera.enable_crop) != (640, 480, 30, 100, 34, 400, 398, True):
        raise ValueError(f"{policy_name} requires D435 640x480@30 crop (100,34,400,398)")


def _validate_tacex_rma_xy_student_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
    history_scale_vector: np.ndarray,
) -> None:
    """对 0809 XY Visual Student 的部署契约执行失败即拒绝的校验。"""
    metadata = bundle.metadata
    if metadata.get("version") != 8:
        raise ValueError("TacEx RMA XY Student deployment requires metadata version 8")
    cuda_validation = metadata.get("cuda_validation")
    cuda_validation_atol = metadata.get("cuda_validation_atol")
    cuda_max_abs_error = cuda_validation.get("max_abs_error") if isinstance(cuda_validation, dict) else None
    if (
        not isinstance(cuda_validation, dict)
        or cuda_validation.get("available") is not True
        or isinstance(cuda_max_abs_error, bool)
        or not isinstance(cuda_max_abs_error, (int, float))
        or not math.isfinite(float(cuda_max_abs_error))
        or float(cuda_max_abs_error) < 0.0
        or isinstance(cuda_validation_atol, bool)
        or not isinstance(cuda_validation_atol, (int, float))
        or not math.isfinite(float(cuda_validation_atol))
        or float(cuda_validation_atol) <= 0.0
        or float(cuda_max_abs_error) > float(cuda_validation_atol)
    ):
        raise ValueError("TacEx RMA XY Student metadata has invalid CUDA validation")
    expected_sha256 = metadata.get("torchscript_sha256")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError("TacEx RMA XY Student metadata is missing torchscript_sha256")
    actual_sha256 = _sha256_file(bundle.model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "TorchScript SHA-256 does not match metadata: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    if (
        bundle.action_dim != 4
        or bundle.history_dim != 4
        or bundle.proprio_dim != 15
        or bundle.contact_force_dim != 2
        or bundle.contact_force_threshold_n != 1.0
    ):
        raise ValueError(
            "TacEx RMA XY Student requires action_history[4], proprio_obs[15], "
            "contact_force_n[2] at 1 N, and actions[4]"
        )
    if bundle.has_gelsight_inputs:
        raise ValueError("TacEx RMA XY Student must not declare GelSight inputs")
    if (bundle.rgb_height, bundle.rgb_width) != (224, 224):
        raise ValueError("TacEx RMA XY Student requires wrist_rgb[224,224,3]")
    if config.model.rma_contact_force_source not in {"gripper_is_grasped", "zeros"}:
        raise ValueError(
            "TacEx RMA XY Student requires model.rma_contact_force_source to be "
            "'gripper_is_grasped' or 'zeros'"
        )
    if config.model.history_source != "processed_action":
        raise ValueError("TacEx RMA XY Student requires model.history_source='processed_action'")
    if not np.allclose(history_scale_vector, np.ones(4, dtype=np.float32), rtol=0.0, atol=1e-8):
        raise ValueError(
            "TacEx RMA XY Student history is already in physical units; "
            "model.history_scale must be [1, 1, 1, 1]"
        )
    if config.model.history_delay_steps != 1:
        raise ValueError("TacEx RMA XY Student requires model.history_delay_steps=1")
    if config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]:
        raise ValueError("TacEx RMA XY Student requires action labels [dx, dy, dz, gripper]")
    expected_action_scales = np.asarray([0.05, 0.05, 0.05, 0.01], dtype=np.float32)
    if not np.allclose(
        np.asarray(config.action_adapter.scales, dtype=np.float32).reshape(-1),
        expected_action_scales,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("TacEx RMA XY Student requires action scales [0.05, 0.05, 0.05, 0.01]")
    if (
        config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
        or config.action_adapter.gripper_mode != "delta_width"
    ):
        raise ValueError(
            "TacEx RMA XY Student requires normalized [-1,1] actions and delta_width gripper"
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
    ) != (640, 480, 30, 100, 34, 400, 398) or not camera.enable_crop:
        raise ValueError("TacEx RMA XY Student requires D435 640x480@30 crop (100,34,400,398)")


def validate_bundle_artifacts(
    config: BundleDeployConfig,
    residual_settings: ResidualDeploySettings | None = None,
    real_rl_settings: RealRLDeploySettings | None = None,
    rlpd_settings: RLPDDeploySettings | None = None,
) -> dict[str, Any]:
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
        residual_settings=residual_settings,
        real_rl_settings=real_rl_settings,
        rlpd_settings=rlpd_settings,
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
    wrist_rgb = np.zeros(bundle.rgb_input_shape, dtype=np.uint8)
    tactile_zeros = {
        name: np.zeros(shape, dtype=np.uint8)
        for name, shape in bundle.gelsight_input_shapes.items()
    }
    contact_force_zeros = (
        np.zeros(bundle.contact_force_dim, dtype=np.float32)
        if bundle.contact_force_dim
        else None
    )
    output = bundle.predict(
        action_history,
        proprio,
        wrist_rgb,
        tactile_zeros.get("gsmini_left_rgb"),
        tactile_zeros.get("gsmini_right_rgb"),
        tactile_zeros.get("gsmini_left_reference_rgb"),
        tactile_zeros.get("gsmini_right_reference_rgb"),
        contact_force_zeros,
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
            bundle.rgb_input_name: list(bundle.rgb_input_shape),
            **{
                name: list(shape)
                for name, shape in bundle.gelsight_input_shapes.items()
            },
            **({"contact_force_n": [bundle.contact_force_dim]} if bundle.contact_force_dim else {}),
        },
        "output_signature": {"mean_actions": [bundle.action_dim]},
        "smoke_test_output": output.tolist(),
        "rma_actor_input": dict(bundle.last_inference_info),
        "rgb_history": {
            "frames": bundle.rgb_history_frames,
            "order": "oldest_to_newest" if bundle.uses_wrist_rgb_history else None,
        },
        "rma_contact_force_source": (
            config.model.rma_contact_force_source if bundle.contact_force_dim else None
        ),
        "policy_contract_enforced": config.model.enforce_policy_contract,
        "streaming_contract": streaming_report,
        "residual_bc": (
            None
            if bundle.residual_runtime is None
            else bundle.residual_runtime.report()
        ),
        "real_rl": (
            None
            if bundle.real_rl_runtime is None
            else bundle.real_rl_runtime.report()
        ),
        "rlpd": None if bundle.rlpd_runtime is None else bundle.rlpd_runtime.report(),
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
        color_camera = RealSenseRGBCamera(
            camera_config,
            output_width=output_width,
            output_height=output_height,
            auto_exposure=camera_config.auto_exposure,
            exposure=camera_config.exposure,
            gain=camera_config.gain,
        )
        controls = color_camera.get_color_controls()
        print(
            "RealSense color controls: "
            f"auto_exposure={controls['auto_exposure']}, "
            f"exposure={controls['exposure']}, gain={controls['gain']}"
        )
        wrist_camera = LatestFrameCamera(color_camera)
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


def capture_gelsight_reference_frames(
    camera: Any,
    bundle: BundleTorchScriptPolicy,
) -> dict[str, np.ndarray]:
    """Capture the fixed post-reset tactile baseline required by the 0813 Student.

    The simulator stores the first valid GelSight image after every reset and
    keeps it unchanged for the episode.  Real deployment has one rollout per
    process, so we take exactly one left/right pair after camera warm-up and
    before generating the first policy action.
    """
    if not getattr(bundle, "has_gelsight_reference_inputs", False):
        return {}
    if not hasattr(camera, "read_tactile"):
        raise RuntimeError("Model requires GelSight reference inputs, but camera rig has none")
    left, right = camera.read_tactile()
    references = {
        "gsmini_left_reference_rgb": np.asarray(left, dtype=np.uint8),
        "gsmini_right_reference_rgb": np.asarray(right, dtype=np.uint8),
    }
    for name, value in references.items():
        expected = bundle.gelsight_input_shapes[name]
        if value.shape != expected:
            raise ValueError(
                f"Expected {name} shape {expected}, got {value.shape}"
            )
        references[name] = np.array(value, dtype=np.uint8, order="C", copy=True)
    bundle._deployment_tactile_references = references
    return references


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
    hil_settings: HILSettings | None = None,
    residual_settings: ResidualDeploySettings | None = None,
    real_rl_settings: RealRLDeploySettings | None = None,
    rlpd_settings: RLPDDeploySettings | None = None,
    rlpd_expert: bool = False,
) -> dict[str, Any]:
    hil_enabled = bool(hil_settings is not None and hil_settings.enabled)
    if hil_settings is not None:
        hil_settings.validate()
    if config.control_mode == "streaming" or streaming_check:
        from .streaming import run_streaming_bundle_deploy

        return run_streaming_bundle_deploy(
            config,
            execute_motion=execute_motion,
            confirm_session_callback=confirm_step_callback,
            save_step_data=save_step_data,
            allow_full_scale=allow_full_scale,
            streaming_check=streaming_check,
            hil_settings=hil_settings,
            residual_settings=residual_settings,
            real_rl_settings=real_rl_settings,
            rlpd_settings=rlpd_settings,
            rlpd_expert=rlpd_expert,
        )

    if config.control_mode != "blocking":
        raise ValueError("control_mode must be 'blocking' or 'streaming'")
    if allow_full_scale:
        raise ValueError("allow_full_scale is only valid for streaming control")
    if hil_enabled:
        raise ValueError("--hil is only valid with server9 streaming control")
    if residual_settings is not None:
        raise ValueError("Residual BC is only valid with server9 streaming control")
    if real_rl_settings is not None:
        raise ValueError("Real-RL is only valid with server9 streaming control")
    if rlpd_settings is not None or rlpd_expert:
        raise ValueError("RLPD is only valid with server9 streaming control")
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
    reject_evaluation_only_motion(bundle, execute_motion)
    action_history_buffer = ActionHistoryBuffer(
        history_dim=bundle.history_dim,
        source=config.model.history_source,
        scale=config.model.history_scale,
        delay_steps=config.model.history_delay_steps,
        processed_action_scale=config.action_adapter.scales,
    )
    rgb_history_buffer = (
        WristRGBHistoryBuffer(
            bundle.rgb_history_frames,
            (bundle.rgb_height, bundle.rgb_width, 3),
        )
        if getattr(bundle, "uses_wrist_rgb_history", False) is True
        else None
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

    try:
        tactile_references = capture_gelsight_reference_frames(camera, bundle)
        if tactile_references:
            reference_dir = run_dir / "tactile_reference_rgb"
            reference_dir.mkdir(exist_ok=True)
            for name, image in tactile_references.items():
                Image.fromarray(image).save(reference_dir / f"{name}.png")
    except Exception:
        camera.close()
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
            model_rgb = (
                rgb_history_buffer.update(wrist_rgb)
                if rgb_history_buffer is not None
                else wrist_rgb
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
            contact_force_n = build_rma_contact_force_input(observation, bundle, config)
            raw_action = bundle.predict(
                action_history,
                proprio,
                model_rgb,
                tactile_images.get("gsmini_left_rgb"),
                tactile_images.get("gsmini_right_rgb"),
                tactile_references.get("gsmini_left_reference_rgb"),
                tactile_references.get("gsmini_right_reference_rgb"),
                contact_force_n,
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
                    **{bundle.rgb_input_name: np.asarray(model_rgb, dtype=np.uint8)},
                    **(
                        {"contact_force_n": np.asarray(contact_force_n, dtype=np.float32)}
                        if contact_force_n is not None
                        else {}
                    ),
                    raw_action=np.asarray(raw_action, dtype=np.float32),
                    clipped_action=np.asarray(clipped_action, dtype=np.float32),
                    **tactile_images,
                    **tactile_references,
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
                    "rgb_history_shape": (
                        list(model_rgb.shape) if rgb_history_buffer is not None else None
                    ),
                    "tactile_rgb_paths": tactile_relpaths,
                    "tactile_rgb_shapes": {
                        name: list(image.shape)
                        for name, image in tactile_images.items()
                    },
                    "tactile_reference_rgb_paths": {
                        name: str(Path("tactile_reference_rgb") / f"{name}.png")
                        for name in tactile_references
                    },
                    "step_data_path": step_data_relpath,
                    "contact_force_n": (
                        None if contact_force_n is None else contact_force_n.tolist()
                    ),
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
