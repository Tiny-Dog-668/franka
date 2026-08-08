from __future__ import annotations

import json
import math
import threading
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .e2e_bundle import (
    ActionHistoryBuffer,
    BundleDeployConfig,
    BundleTorchScriptPolicy,
    _make_camera,
    _make_run_dir,
    _validate_bundle_action_dims,
    _validate_tacex_rma_student_contract,
    build_bundle_inputs,
    evaluate_initial_state,
)
from .types import RobotAction, RobotObservation, print_error_wrench_report


PANDA_JOINT_LOWER = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float64,
)
PANDA_JOINT_UPPER = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float64,
)
EXPECTED_PYLIBFRANKA_VERSION = "0.21.1"
# Smallest singular value of the position Jacobian at the 0802 reference pose,
# in m/rad. It converts a Cartesian action scale into the worst-case joint speed
# the velocity envelope has to carry.
_WORST_CASE_MANIPULABILITY = 0.2637


def reshape_column_major(values: Any, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (rows, columns):
        return array.copy()
    if array.size != rows * columns:
        raise ValueError(
            f"Expected {rows * columns} column-major values, got {array.size}"
        )
    return array.reshape((rows, columns), order="F")


def dls_joint_delta(jacobian: Any, pose_error: Any, damping: float) -> np.ndarray:
    jacobian_matrix = reshape_column_major(jacobian, 6, 7)
    error = np.asarray(pose_error, dtype=np.float64).reshape(-1)
    if error.shape != (6,):
        raise ValueError(f"Expected 6D pose error, got {error.shape}")
    if not np.all(np.isfinite(jacobian_matrix)) or not np.all(np.isfinite(error)):
        raise RuntimeError("DLS input contains NaN or Inf")
    if not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("streaming.dls_lambda must be finite and positive")
    regularized = (
        jacobian_matrix @ jacobian_matrix.T
        + float(damping) ** 2 * np.eye(6, dtype=np.float64)
    )
    delta = jacobian_matrix.T @ np.linalg.solve(regularized, error)
    if not np.all(np.isfinite(delta)):
        raise RuntimeError("DLS result contains NaN or Inf")
    return delta


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    trace_value = float(np.trace(rotation))
    angle = math.acos(float(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)))
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise RuntimeError("Unable to compute orientation error")
        return angle * axis / norm
    return angle * skew / (2.0 * math.sin(angle))


def pose_error(current_pose: Any, target_pose: Any) -> np.ndarray:
    current = reshape_column_major(current_pose, 4, 4)
    target = reshape_column_major(target_pose, 4, 4)
    error = np.concatenate(
        (
            target[:3, 3] - current[:3, 3],
            _rotation_vector(target[:3, :3] @ current[:3, :3].T),
        )
    )
    if not np.all(np.isfinite(error)):
        raise RuntimeError("Pose error contains NaN or Inf")
    return error


def latch_tcp_target(current_pose: Any, executed_xyz: Any) -> np.ndarray:
    target = reshape_column_major(current_pose, 4, 4)
    xyz = np.asarray(executed_xyz, dtype=np.float64).reshape(-1)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("Executed XYZ action must contain three finite values")
    target[:3, 3] += xyz
    return target


def clip_streaming_action(
    raw_action: Any,
    config: BundleDeployConfig,
    allow_full_scale: bool,
) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw.shape != (4,) or not np.all(np.isfinite(raw)):
        raise ValueError("Streaming policy action must contain four finite values")
    low = np.asarray(config.action_adapter.clip_low, dtype=np.float32)
    high = np.asarray(config.action_adapter.clip_high, dtype=np.float32)
    contract_clipped = np.clip(raw, low, high)
    limit = 1.0 if allow_full_scale else float(config.streaming.commissioning_action_limit)
    if not math.isfinite(limit) or limit <= 0.0 or limit > 1.0:
        raise ValueError(
            "streaming.commissioning_action_limit must be finite and in (0, 1]"
        )
    # The commissioning limit must only reduce the Cartesian speed, never rotate
    # the commanded direction. Clipping XYZ per dimension would collapse a
    # [-0.44, 0.85, -1.00] command into three equal-magnitude components and
    # send the arm somewhere the policy never asked for. Scale XYZ uniformly.
    limited = contract_clipped.copy()
    peak = float(np.max(np.abs(limited[:3])))
    if peak > limit:
        limited[:3] *= limit / peak
    # The gripper is an independent degree of freedom, not part of that
    # direction, so it is limited on its own.
    limited[3] = np.clip(limited[3], -limit, limit)
    return limited.astype(np.float32)


def streaming_robot_action(
    executed_action: np.ndarray,
    config: BundleDeployConfig,
    desired_gripper_width: float,
) -> RobotAction:
    scales = np.asarray(config.action_adapter.scales, dtype=np.float64)
    scaled = np.asarray(executed_action, dtype=np.float64) * scales
    gripper_target = float(np.clip(desired_gripper_width + scaled[3], 0.0, 0.08))
    # XYZ is deliberately in the robot base frame. Unlike the legacy blocking
    # adapter, streaming does not negate Y or Z.
    return RobotAction(
        dx=float(scaled[0]),
        dy=float(scaled[1]),
        dz=float(scaled[2]),
        speed=config.speed,
        gripper_width=gripper_target,
        gripper_speed=config.gripper_speed,
        gripper_force=config.gripper_force,
        metadata={
            "executed_normalized_action": executed_action.tolist(),
            "scaled_action": {
                "dx": float(scaled[0]),
                "dy": float(scaled[1]),
                "dz": float(scaled[2]),
                "gripper": float(scaled[3]),
            },
            "xyz_command_frame": "robot_root",
        },
    )


def _require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Streaming contract requires {name}={expected}, got {actual}")


def validate_streaming_contract(
    bundle: BundleTorchScriptPolicy,
    config: BundleDeployConfig,
) -> dict[str, Any]:
    if config.control_mode != "streaming":
        raise ValueError("Streaming validation requires control_mode='streaming'")
    if config.realtime != "enforce":
        raise ValueError("Streaming control requires realtime='enforce'")
    if not config.model.enforce_policy_contract:
        raise ValueError("Streaming requires model.enforce_policy_contract=true")
    if bundle.action_dim != 4 or bundle.history_dim != 4:
        raise ValueError("Streaming requires a strict 4D action/history Bundle")
    if config.action_adapter.labels != ["dx", "dy", "dz", "gripper"]:
        raise ValueError(
            "Streaming requires action labels ['dx', 'dy', 'dz', 'gripper']"
        )

    stream = config.streaming
    if stream.backend not in {"async_position", "server9_joint_position"}:
        raise ValueError(
            "streaming.backend must be 'async_position' or 'server9_joint_position'"
        )
    if stream.server9_control_cpu is not None and (
        isinstance(stream.server9_control_cpu, bool)
        or not isinstance(stream.server9_control_cpu, int)
        or stream.server9_control_cpu < 0
    ):
        raise ValueError("streaming.server9_control_cpu must be a non-negative integer or null")
    for name in (
        "policy_frequency_hz",
        "ik_frequency_hz",
        "dls_lambda",
        "policy_watchdog_s",
        "control_watchdog_s",
    ):
        value = float(getattr(stream, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"streaming.{name} must be finite and positive")
    if len(stream.maximum_joint_velocities) != 7 or not np.all(
        np.isfinite(stream.maximum_joint_velocities)
    ):
        raise ValueError("streaming.maximum_joint_velocities must have seven finite values")
    if np.any(np.asarray(stream.maximum_joint_velocities) <= 0.0):
        raise ValueError("streaming.maximum_joint_velocities must be positive")
    for name in ("maximum_joint_accelerations", "maximum_joint_jerks"):
        values = np.asarray(getattr(stream, name), dtype=np.float64).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"streaming.{name} must have seven finite positive values")
    if stream.joint_impedance is not None:
        joint_impedance = np.asarray(stream.joint_impedance, dtype=np.float64).reshape(-1)
        if joint_impedance.shape != (7,) or not np.all(np.isfinite(joint_impedance)):
            raise ValueError("streaming.joint_impedance must have seven finite values or be null")
        if np.any(joint_impedance <= 0.0) or np.any(joint_impedance > 14250.0):
            raise ValueError("streaming.joint_impedance values must be in (0, 14250]")
    if (
        not math.isfinite(stream.maximum_joint_target_delta_rad)
        or not (0.0 < stream.maximum_joint_target_delta_rad <= 0.1)
    ):
        raise ValueError(
            "streaming.maximum_joint_target_delta_rad must be finite and in (0, 0.1]"
        )
    if not (0.0 < stream.joint_limit_margin_rad < 0.5):
        raise ValueError("streaming.joint_limit_margin_rad must be in (0, 0.5)")
    if stream.control_law not in {"joint_position_pursuit", "sim_actuator_velocity"}:
        raise ValueError(
            "streaming.control_law must be 'joint_position_pursuit' or 'sim_actuator_velocity'"
        )
    if stream.control_law == "sim_actuator_velocity":
        gain = float(stream.reference_velocity_gain)
        if not math.isfinite(gain) or not (0.0 < gain <= 100.0):
            raise ValueError("streaming.reference_velocity_gain must be finite and in (0, 100]")
        reference_delta = float(stream.maximum_ik_reference_delta_rad)
        if not math.isfinite(reference_delta) or not (0.0 < reference_delta <= 0.5):
            raise ValueError(
                "streaming.maximum_ik_reference_delta_rad must be finite and in (0, 0.5]"
            )
        # The gain turns the DLS delta into a joint velocity, so the velocity
        # envelope has to be able to carry the trained action scale. If it
        # cannot, the command saturates and the law degrades back to the
        # magnitude-blind behaviour it was introduced to remove.
        required_rad_s = gain * max(config.action_adapter.scales[:3]) / _WORST_CASE_MANIPULABILITY
        if min(stream.maximum_joint_velocities) < required_rad_s:
            raise ValueError(
                "streaming.maximum_joint_velocities must each be at least "
                f"{required_rad_s:.3f} rad/s for control_law='sim_actuator_velocity' with "
                f"reference_velocity_gain={gain} and action scale "
                f"{max(config.action_adapter.scales[:3])} m"
            )
    ratio = stream.ik_frequency_hz / stream.policy_frequency_hz
    if not math.isclose(ratio, 2.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Streaming requires exactly two IK ticks per policy action")
    if config.camera.fps != round(stream.policy_frequency_hz):
        raise ValueError("Streaming camera FPS must match policy_frequency_hz")
    if (
        config.action_adapter.clip_low != [-1.0] * 4
        or config.action_adapter.clip_high != [1.0] * 4
    ):
        raise ValueError("Streaming normalized action bounds must be exactly [-1, 1]")

    if getattr(bundle, "is_tacex_rma_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_student_contract(bundle, config, history_scale)
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_student_metadata_version": bundle.metadata.get("version"),
            "rma_student_v5": True,
        }

    contract = bundle.metadata.get("policy_contract")
    if not isinstance(contract, dict):
        raise ValueError("Streaming requires metadata.policy_contract")
    if contract.get("action_dim") != 4:
        raise ValueError("Streaming contract action_dim must be 4")
    if contract.get("xyz_command_frame") != "robot_root":
        raise ValueError("Streaming requires xyz_command_frame='robot_root'")
    if contract.get("privileged_dz_gate") != "disabled":
        raise ValueError("Streaming requires privileged_dz_gate='disabled'")
    if contract.get("gripper_control_mode") != "total_width_delta_cached_target":
        raise ValueError("Streaming requires cached total-width delta gripper control")
    if contract.get("gripper_width_bounds") != [0.0, 0.08]:
        raise ValueError("Streaming contract gripper width bounds must be [0.0, 0.08]")
    if contract.get("gripper_width_delta_updates_per_policy_step") != 1:
        raise ValueError("Streaming requires one gripper target update per policy step")
    if contract.get("action_history") != "per_dimension_processed_action_no_privileged_gate_v3":
        raise ValueError("Streaming contract has an unsupported action_history definition")
    _require_close(contract.get("sim_dt", math.nan), 1.0 / stream.ik_frequency_hz, "sim_dt")
    if contract.get("decimation") != 2:
        raise ValueError("Streaming contract decimation must be 2")
    _require_close(
        contract.get("nominal_policy_frequency_hz", math.nan),
        stream.policy_frequency_hz,
        "nominal_policy_frequency_hz",
    )
    _require_close(
        contract.get("nominal_camera_frequency_hz", math.nan),
        stream.policy_frequency_hz,
        "nominal_camera_frequency_hz",
    )
    episode_limit = int(contract.get("max_episode_length_steps", 150))
    if config.runner.steps > min(150, episode_limit):
        raise ValueError(
            f"Streaming rollout is capped at {min(150, episode_limit)} policy steps"
        )
    return {
        "policy_frequency_hz": stream.policy_frequency_hz,
        "ik_frequency_hz": stream.ik_frequency_hz,
        "ticks_per_action": 2,
        "xyz_command_frame": "robot_root",
        "commissioning_action_limit": stream.commissioning_action_limit,
    }


def validate_pylibfranka_streaming_api(module: Any) -> None:
    version = str(getattr(module, "__version__", "unknown"))
    handler = getattr(module, "AsyncPositionControlHandler", None)
    missing = []
    if handler is None:
        missing.append("AsyncPositionControlHandler")
    elif not hasattr(handler, "read_once"):
        missing.append("AsyncPositionControlHandler.read_once")
    if not hasattr(module, "TargetStatus"):
        missing.append("TargetStatus")
    if version != EXPECTED_PYLIBFRANKA_VERSION or missing:
        details = ", ".join(missing) if missing else "no interfaces"
        raise RuntimeError(
            "Streaming requires the vendored pylibfranka 0.21.1 wheel with the "
            f"async state patch; installed version={version}, missing={details}. "
            "Build/install it with scripts/build_pylibfranka_wheel.sh."
        )


def evaluate_streaming_check_state(
    observation: RobotObservation,
    config: BundleDeployConfig,
) -> dict[str, Any]:
    """Safety gate for a hold-only communication check.

    This deliberately omits the policy-specific reference pose and every
    gripper requirement. It retains the stationary/mode/error checks and adds
    absolute Panda joint and configured workspace bounds.
    """

    hold_config = replace(
        config.initial_state,
        joint_positions=None,
        tcp_translation=None,
        tcp_quaternion_xyzw=None,
        gripper_width_m=None,
        minimum_gripper_max_width_m=None,
        require_gripper_not_grasped=False,
    )
    report = evaluate_initial_state(observation, hold_config)
    report["profile"] = "streaming_check_hold_only"
    report["skipped_policy_checks"] = [
        "reference_joint_positions",
        "reference_tcp_translation",
        "reference_tcp_orientation",
        "gripper",
    ]

    def add_check(name: str, passed: bool, details: dict[str, Any], failure: str) -> None:
        report["checks"][name] = {"passed": bool(passed), **details}
        if not passed:
            report["passed"] = False
            report["failures"].append(failure)

    joints = np.asarray(observation.joint_positions, dtype=np.float64).reshape(-1)
    joint_ok = (
        joints.shape == (7,)
        and np.all(np.isfinite(joints))
        and np.all(joints >= PANDA_JOINT_LOWER)
        and np.all(joints <= PANDA_JOINT_UPPER)
    )
    add_check(
        "panda_joint_limits",
        bool(joint_ok),
        {
            "actual_rad": joints.tolist(),
            "minimum_rad": PANDA_JOINT_LOWER.tolist(),
            "maximum_rad": PANDA_JOINT_UPPER.tolist(),
        },
        "joint state is non-finite or outside Panda limits",
    )

    position = np.asarray(observation.tcp_translation, dtype=np.float64).reshape(-1)
    minimum = np.asarray(config.workspace["minimum"], dtype=np.float64)
    maximum = np.asarray(config.workspace["maximum"], dtype=np.float64)
    workspace_ok = (
        position.shape == (3,)
        and np.all(np.isfinite(position))
        and np.all(position >= minimum)
        and np.all(position <= maximum)
    )
    add_check(
        "workspace",
        bool(workspace_ok),
        {
            "actual_m": position.tolist(),
            "minimum_m": minimum.tolist(),
            "maximum_m": maximum.tolist(),
        },
        "TCP is non-finite or outside the configured workspace",
    )
    return report


def _active_error_names(errors: Any) -> list[str]:
    names: list[str] = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        try:
            value = getattr(errors, name)
        except Exception:
            continue
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def _mode_name(mode: Any) -> str:
    name = str(mode).split(".")[-1]
    return name[1:] if name.startswith("k") else name


def _rotation_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace_value = float(np.trace(matrix))
    if trace_value > 0.0:
        scale = math.sqrt(trace_value + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if not math.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("Robot pose contains an invalid rotation")
    return (quaternion / norm).tolist()


def robot_state_to_observation(
    state: Any,
    gripper_state: Any | None,
    in_control: bool,
) -> RobotObservation:
    pose = reshape_column_major(state.O_T_EE, 4, 4)
    current_errors = _active_error_names(state.current_errors)
    last_errors = _active_error_names(state.last_motion_errors)
    observation = RobotObservation(
        joint_positions=np.asarray(state.q, dtype=np.float64).tolist(),
        joint_velocities=np.asarray(state.dq, dtype=np.float64).tolist(),
        tcp_translation=pose[:3, 3].tolist(),
        tcp_quaternion=_rotation_to_quaternion_xyzw(pose[:3, :3]),
        external_wrench=np.asarray(state.O_F_ext_hat_K, dtype=np.float64).tolist(),
        robot_mode=_mode_name(state.robot_mode),
        has_errors=bool(current_errors),
        is_in_control=in_control,
        control_command_success_rate=float(state.control_command_success_rate),
        metadata={"current_errors": current_errors, "last_motion_errors": last_errors},
    )
    if gripper_state is not None:
        observation.gripper_width = float(gripper_state.width)
        observation.gripper_max_width = float(gripper_state.max_width)
        observation.gripper_is_grasped = bool(gripper_state.is_grasped)
    return observation


class AsyncGripperQueue:
    def __init__(
        self,
        gripper: Any,
        speed: float,
        tolerance: float,
        force: float = 20.0,
    ) -> None:
        self.gripper = gripper
        self.speed = speed
        self.tolerance = tolerance
        self.force = force
        state = gripper.state
        self.cached_state = state
        self.desired_width = float(state.width)
        self._future: Any | None = None
        self._future_kind: str | None = None
        self._future_target: float | None = None
        self._future_start_width: float | None = None
        self._queued_width: float | None = None
        self._control_session_active = False
        self._lock = threading.RLock()

    def mark_control_session_active(self) -> None:
        with self._lock:
            self._control_session_active = True

    def command(self, width: float) -> None:
        with self._lock:
            max_width = float(getattr(self.cached_state, "max_width", 0.08))
            if not math.isfinite(max_width) or max_width <= 0.0:
                max_width = 0.08
            target = float(np.clip(width, 0.0, max_width))
            self.desired_width = target
            self.poll()
            current_width = float(self.cached_state.width)
            if (
                bool(getattr(self.cached_state, "is_grasped", False))
                and target <= current_width + self.tolerance
            ):
                # Keep the active grasp instead of replacing it with another
                # close command that cannot reach the policy's nominal width.
                return
            if abs(target - current_width) <= self.tolerance:
                return
            if self._future is None:
                self._start_move(target, current_width)
            else:
                self._queued_width = target

    def _start_move(self, target: float, current_width: float) -> None:
        self._future = self.gripper.move_async(target, self.speed)
        self._future_kind = "move"
        self._future_target = target
        self._future_start_width = current_width

    def _start_grasp(self, width: float) -> None:
        self._future = self.gripper.grasp_async(width, self.speed, self.force)
        self._future_kind = "grasp"
        self._future_target = width
        self._future_start_width = width

    def _clear_future(self) -> None:
        self._future = None
        self._future_kind = None
        self._future_target = None
        self._future_start_width = None

    def _start_queued_command(self) -> None:
        if self._queued_width is None:
            return
        target = self._queued_width
        self._queued_width = None
        current_width = float(self.cached_state.width)
        if (
            bool(getattr(self.cached_state, "is_grasped", False))
            and target <= current_width + self.tolerance
        ):
            return
        if abs(target - current_width) > self.tolerance:
            self._start_move(target, current_width)

    def poll(self) -> None:
        with self._lock:
            if self._future is None or not self._future.wait(0.0):
                return
            kind = self._future_kind
            target = self._future_target
            start_width = self._future_start_width
            try:
                success = bool(self._future.get())
            except Exception as exc:
                self._clear_future()
                raise RuntimeError(
                    f"Asynchronous gripper {kind or 'command'} raised an exception"
                ) from exc
            self._clear_future()
            self.cached_state = self.gripper.state
            current_width = float(self.cached_state.width)

            if success or (
                kind == "grasp"
                and bool(getattr(self.cached_state, "is_grasped", False))
            ):
                self._start_queued_command()
                return

            blocked_close = (
                kind == "move"
                and target is not None
                and start_width is not None
                and target < start_width - self.tolerance
                and current_width > target + self.tolerance
            )
            if blocked_close:
                # A Franka `move` reports False when an object prevents it
                # from reaching the requested width. Convert that measured
                # contact width into a force-controlled grasp instead of
                # treating the expected obstruction as a communication fault.
                self._start_grasp(current_width)
                return

            raise RuntimeError(
                "Asynchronous gripper command failed: "
                f"kind={kind}, target_width={target}, "
                f"actual_width={current_width:.6f}, "
                f"is_grasped={bool(getattr(self.cached_state, 'is_grasped', False))}"
            )

    def stop(self) -> None:
        with self._lock:
            self._queued_width = None
            try:
                if self._control_session_active or self._future is not None:
                    self.gripper.stop()
            finally:
                self._clear_future()
                self._control_session_active = False


@dataclass
class _PolicyResult:
    raw_action: np.ndarray
    executed_action: np.ndarray
    robot_action: RobotAction
    action_history: np.ndarray
    proprio: np.ndarray
    image: np.ndarray
    tactile_images: dict[str, np.ndarray]
    inference_info: dict[str, Any]
    elapsed_ns: int


def _run_policy_tick(
    bundle: BundleTorchScriptPolicy,
    camera: Any,
    history: ActionHistoryBuffer,
    observation: RobotObservation,
    config: BundleDeployConfig,
    allow_full_scale: bool,
    desired_gripper_width: float,
    clock_ns: Callable[[], int],
    *,
    collect_rma_debug: bool = False,
) -> _PolicyResult:
    started = clock_ns()
    image = camera.read()
    if image.shape != (bundle.rgb_height, bundle.rgb_width, 3):
        raise ValueError(
            f"Expected RGB frame {(bundle.rgb_height, bundle.rgb_width, 3)}, got {image.shape}"
        )
    action_history, proprio = build_bundle_inputs(
        observation,
        history.current(),
        bundle.proprio_dim,
        bundle.history_dim,
    )
    tactile_images: dict[str, np.ndarray] = {}
    if getattr(bundle, "has_gelsight_inputs", False) is True:
        if not hasattr(camera, "read_tactile"):
            raise RuntimeError("Model requires GelSight inputs, but camera rig has none")
        left_tactile, right_tactile = camera.read_tactile()
        tactile_images = {
            "gsmini_left_rgb": np.asarray(left_tactile, dtype=np.uint8),
            "gsmini_right_rgb": np.asarray(right_tactile, dtype=np.uint8),
        }
    predict_kwargs = {"collect_rma_debug": True} if collect_rma_debug else {}
    if tactile_images:
        raw_action = bundle.predict(
            action_history,
            proprio,
            image,
            tactile_images["gsmini_left_rgb"],
            tactile_images["gsmini_right_rgb"],
            **predict_kwargs,
        )
    else:
        raw_action = bundle.predict(
            action_history,
            proprio,
            image,
            **predict_kwargs,
        )
    executed = clip_streaming_action(raw_action, config, allow_full_scale)
    action = streaming_robot_action(executed, config, desired_gripper_width)
    return _PolicyResult(
        raw_action=np.asarray(raw_action, dtype=np.float32),
        executed_action=executed,
        robot_action=action,
        action_history=action_history,
        proprio=proprio,
        image=np.asarray(image, dtype=np.uint8).copy(),
        tactile_images={
            name: value.copy() for name, value in tactile_images.items()
        },
        inference_info=dict(getattr(bundle, "last_inference_info", {})),
        elapsed_ns=clock_ns() - started,
    )


def _sleep_until(
    deadline_ns: int,
    clock_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> int:
    remaining_ns = deadline_ns - clock_ns()
    if remaining_ns > 0:
        sleep(remaining_ns / 1e9)
    return clock_ns() - deadline_ns


def policy_result_is_timely(
    elapsed_ns: int,
    policy_period_ns: int,
    watchdog_ns: int,
    deadline_lateness_ns: int = 0,
) -> bool:
    if elapsed_ns > watchdog_ns:
        raise RuntimeError(f"Policy watchdog exceeded: {elapsed_ns / 1e6:.3f} ms")
    # A pending result is computed during the preceding policy period. If the
    # loop has slipped by a complete period, that result belongs to an expired
    # boundary even when its isolated GPU inference time was short.
    return (
        elapsed_ns <= policy_period_ns
        and max(0, deadline_lateness_ns) < policy_period_ns
    )


def _safe_stop(handler: Any | None, gripper_queue: AsyncGripperQueue | None) -> list[str]:
    errors: list[str] = []
    if handler is not None:
        try:
            handler.stop_control()
        except Exception as exc:
            errors.append(f"arm stop_control: {exc}")
    if gripper_queue is not None:
        try:
            gripper_queue.stop()
        except Exception as exc:
            errors.append(f"gripper stop: {exc}")
    return errors


def _write_streaming_artifacts(
    run_dir: Path,
    config: BundleDeployConfig,
    initial_report: dict[str, Any],
    records: list[dict[str, Any]],
    images: list[np.ndarray],
    control_trace: list[dict[str, Any]],
    timing: dict[str, Any],
    save_step_data: bool,
) -> dict[str, Any]:
    (run_dir / "rgb").mkdir(exist_ok=True)
    if config.tactile_camera.enabled:
        (run_dir / "tactile_rgb").mkdir(exist_ok=True)
    if save_step_data:
        (run_dir / "step_data").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "initial_state_check.json").write_text(
        json.dumps(initial_report, indent=2), encoding="utf-8"
    )
    for index, (record, image) in enumerate(zip(records, images)):
        tactile_images = record.pop("_tactile_images", {})
        rgb_relpath = f"rgb/step_{index:04d}.png"
        record["model_input"]["rgb_path"] = rgb_relpath
        if config.camera.save_rgb or save_step_data:
            Image.fromarray(image).save(run_dir / rgb_relpath)
        tactile_paths: dict[str, str] = {}
        for name, tactile_image in tactile_images.items():
            relpath = f"tactile_rgb/{name}_step_{index:04d}.png"
            tactile_paths[name] = relpath
            if config.tactile_camera.save_rgb or save_step_data:
                Image.fromarray(tactile_image).save(run_dir / relpath)
        record["model_input"]["tactile_rgb_paths"] = tactile_paths
        if save_step_data:
            data_relpath = f"step_data/step_{index:04d}.npz"
            step_payload = {
                "action_history": np.asarray(
                    record["model_input"]["action_history"], dtype=np.float32
                ),
                "proprio_obs": np.asarray(
                    record["model_input"]["proprio_obs"], dtype=np.float32
                ),
                "wrist_rgb": image,
                "raw_action": np.asarray(record["raw_action"], dtype=np.float32),
                "executed_action": np.asarray(
                    record["limited_action"], dtype=np.float32
                ),
                **tactile_images,
            }
            np.savez_compressed(run_dir / data_relpath, **step_payload)
            record["model_input"]["step_data_path"] = data_relpath
    with (run_dir / "rollout.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (run_dir / "control_trace.jsonl").open("w", encoding="utf-8") as handle:
        for tick in control_trace:
            handle.write(json.dumps(tick) + "\n")
    (run_dir / "timing_summary.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8"
    )
    summary = {
        "run_dir": str(run_dir),
        "control_mode": "streaming",
        "initial_state_check": initial_report,
        "num_steps": len(records),
        "num_control_ticks": len(control_trace),
        "steps": records,
        "timing": timing,
        "save_step_data": save_step_data,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_streaming_bundle_deploy(
    config: BundleDeployConfig,
    execute_motion: bool = True,
    confirm_session_callback: Any | None = None,
    save_step_data: bool = False,
    allow_full_scale: bool = False,
    streaming_check: bool = False,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if streaming_check:
        config.control_mode = "streaming"
    if config.streaming.backend == "server9_joint_position":
        from .streaming_server9 import run_server9_streaming_bundle_deploy

        return run_server9_streaming_bundle_deploy(
            config,
            execute_motion=execute_motion,
            confirm_session_callback=confirm_session_callback,
            save_step_data=save_step_data,
            allow_full_scale=allow_full_scale,
            streaming_check=streaming_check,
            clock_ns=clock_ns,
            sleep=sleep,
        )
    run_dir = _make_run_dir(config)
    records: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    control_trace: list[dict[str, Any]] = []
    timing: dict[str, Any] = {
        "policy_deadline_misses": 0,
        "maximum_policy_elapsed_ms": 0.0,
        "maximum_control_lateness_ms": 0.0,
    }

    bundle: BundleTorchScriptPolicy | None = None
    camera: Any | None = None
    gripper_queue: AsyncGripperQueue | None = None
    handler: Any | None = None
    initial_report: dict[str, Any] = {"passed": False, "checks": {}, "failures": []}
    stopped = False
    try:
        if not streaming_check:
            bundle = BundleTorchScriptPolicy(
                config.model.model_path,
                config.model.metadata_path,
                config.model.device,
                torch_num_threads=config.model.torch_num_threads,
                optimize_for_inference=config.model.optimize_for_inference,
                rma_position_source=config.model.rma_position_source,
                rma_contact_source=config.model.rma_contact_source,
                rma_oracle_cube_position_root=config.model.rma_oracle_cube_position_root,
            )
            _validate_bundle_action_dims(bundle, config)
            validate_streaming_contract(bundle, config)

        import pylibfranka

        try:
            from pylibfranka_streaming_patch import install as install_streaming_patch
        except ImportError:
            pass
        else:
            install_streaming_patch(pylibfranka)
        validate_pylibfranka_streaming_api(pylibfranka)
        robot = pylibfranka.Robot(config.robot_ip, pylibfranka.RealtimeConfig.kEnforce)
        model = robot.load_model()
        initial_state = robot.read_once()

        if not streaming_check:
            from franky import Gripper

            gripper_queue = AsyncGripperQueue(
                Gripper(config.robot_ip),
                config.gripper_speed,
                config.gripper_command_tolerance_m,
                config.gripper_force,
            )
        initial_observation = robot_state_to_observation(
            initial_state,
            gripper_queue.cached_state if gripper_queue else None,
            in_control=False,
        )
        initial_report = (
            evaluate_streaming_check_state(initial_observation, config)
            if streaming_check
            else evaluate_initial_state(initial_observation, config.initial_state)
        )
        if config.initial_state.enforce and not initial_report["passed"]:
            failures = "\n".join(f"  - {item}" for item in initial_report["failures"])
            raise RuntimeError(
                "Initial-state safety check failed; no streaming control was started.\n" + failures
            )

        history: ActionHistoryBuffer | None = None
        first_policy: _PolicyResult | None = None
        if not streaming_check:
            assert bundle is not None
            camera = _make_camera(
                config.camera,
                bundle.rgb_width,
                bundle.rgb_height,
                tactile_config=config.tactile_camera,
                gelsight_input_shapes=getattr(bundle, "gelsight_input_shapes", {}),
            )
            history = ActionHistoryBuffer(
                bundle.history_dim,
                config.model.history_source,
                config.model.history_scale,
                config.model.history_delay_steps,
                processed_action_scale=config.action_adapter.scales,
            )
            # Camera warm-up is completed by construction. Exercise the exact
            # TorchScript path once before starting the active control connection.
            tactile_zeros = {
                name: np.zeros(shape, dtype=np.uint8)
                for name, shape in getattr(bundle, "gelsight_input_shapes", {}).items()
            }
            warmup_args = (
                np.zeros(bundle.history_dim, dtype=np.float32),
                np.zeros(bundle.proprio_dim, dtype=np.float32),
                np.zeros((bundle.rgb_height, bundle.rgb_width, 3), dtype=np.uint8),
            )
            if tactile_zeros:
                bundle.predict(
                    *warmup_args,
                    tactile_zeros["gsmini_left_rgb"],
                    tactile_zeros["gsmini_right_rgb"],
                )
            else:
                bundle.predict(*warmup_args)
            first_policy = _run_policy_tick(
                bundle,
                camera,
                history,
                initial_observation,
                config,
                allow_full_scale,
                float(gripper_queue.desired_width if gripper_queue else 0.0),
                clock_ns,
            )

        if not execute_motion:
            if streaming_check:
                raise ValueError("--streaming-check cannot be combined with --preview-only")
            assert bundle is not None and camera is not None and history is not None
            observation = initial_observation
            policy_period_ns = round(1e9 / config.streaming.policy_frequency_hz)
            start_ns = clock_ns()
            for step_index in range(config.runner.steps):
                _sleep_until(start_ns + step_index * policy_period_ns, clock_ns, sleep)
                result = first_policy if step_index == 0 else _run_policy_tick(
                    bundle,
                    camera,
                    history,
                    observation,
                    config,
                    allow_full_scale,
                    float(gripper_queue.desired_width if gripper_queue else 0.0),
                    clock_ns,
                )
                assert result is not None
                timing["maximum_policy_elapsed_ms"] = max(
                    timing["maximum_policy_elapsed_ms"], result.elapsed_ns / 1e6
                )
                timely = policy_result_is_timely(
                    result.elapsed_ns,
                    policy_period_ns,
                    round(config.streaming.policy_watchdog_s * 1e9),
                )
                if timely:
                    history.update(result.raw_action, result.executed_action)
                else:
                    timing["policy_deadline_misses"] += 1
                records.append(
                    _policy_record(step_index, observation, result, timely, False)
                )
                images.append(result.image)
            preview_elapsed_s = max(0.0, (clock_ns() - start_ns) / 1e9)
            timing["preview_only"] = True
            timing["policy_elapsed_s"] = preview_elapsed_s
            timing["effective_policy_frequency_hz"] = (
                (config.runner.steps - 1) / preview_elapsed_s
                if config.runner.steps > 1 and preview_elapsed_s > 0.0
                else None
            )
            cleanup_errors = _safe_stop(None, gripper_queue)
            gripper_queue = None
            stopped = True
            if cleanup_errors:
                raise RuntimeError("; ".join(cleanup_errors))
            return _write_streaming_artifacts(
                run_dir,
                config,
                initial_report,
                records,
                images,
                control_trace,
                timing,
                save_step_data,
            )

        if confirm_session_callback is not None:
            proposed_action = (
                first_policy.robot_action
                if first_policy is not None
                else RobotAction(speed=0.0, metadata={"streaming_check": True})
            )
            if not bool(confirm_session_callback(0, proposed_action, initial_observation)):
                timing["cancelled_by_user"] = True
                cleanup_errors = _safe_stop(None, gripper_queue)
                gripper_queue = None
                stopped = True
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))
                return _write_streaming_artifacts(
                    run_dir,
                    config,
                    initial_report,
                    records,
                    images,
                    control_trace,
                    timing,
                    save_step_data,
                )

        configuration = pylibfranka.AsyncPositionControlHandler.Configuration(
            config.streaming.maximum_joint_velocities,
            1e-4,
        )
        configured = pylibfranka.AsyncPositionControlHandler.configure(robot, configuration)
        if configured.handler is None:
            raise RuntimeError(
                f"Failed to start async position control: {configured.error_message}"
            )
        handler = configured.handler
        if gripper_queue is not None:
            gripper_queue.mark_control_session_active()

        control_period_ns = round(1e9 / config.streaming.ik_frequency_hz)
        policy_period_ns = round(1e9 / config.streaming.policy_frequency_hz)
        total_ticks = (
            round(2.0 * config.streaming.ik_frequency_hz)
            if streaming_check
            else config.runner.steps * 2
        )
        start_ns = clock_ns()
        target_pose: np.ndarray | None = None
        pending_first = first_policy

        for tick_index in range(total_ticks):
            deadline_ns = start_ns + tick_index * control_period_ns
            lateness_ns = _sleep_until(deadline_ns, clock_ns, sleep)
            timing["maximum_control_lateness_ms"] = max(
                timing["maximum_control_lateness_ms"], max(0, lateness_ns) / 1e6
            )
            if lateness_ns > config.streaming.control_watchdog_s * 1e9:
                raise RuntimeError(
                    f"Control watchdog exceeded: {lateness_ns / 1e6:.3f} ms late"
                )

            tick_started_ns = clock_ns()
            state = handler.read_once()
            observation = robot_state_to_observation(
                state,
                gripper_queue.cached_state if gripper_queue else None,
                in_control=True,
            )
            if observation.has_errors:
                print_error_wrench_report(
                    observation,
                    context=f"Streaming tick {tick_index} policy_step {tick_index // 2}",
                )
                raise RuntimeError(
                    "Robot reported errors: " + ", ".join(observation.metadata["current_errors"])
                )
            if observation.robot_mode not in {"Move", "Idle"}:
                print_error_wrench_report(
                    observation,
                    context=f"Streaming tick {tick_index} policy_step {tick_index // 2}",
                )
                raise RuntimeError(f"Robot entered unsafe mode {observation.robot_mode!r}")
            _validate_current_state_limits(state, config)

            policy_step = tick_index // 2
            policy_accepted = False
            if not streaming_check and tick_index % 2 == 0:
                assert bundle is not None and camera is not None and history is not None
                result = pending_first
                pending_first = None
                is_preflight_result = result is not None
                if result is None:
                    result = _run_policy_tick(
                        bundle,
                        camera,
                        history,
                        observation,
                        config,
                        allow_full_scale,
                        float(gripper_queue.desired_width if gripper_queue else 0.0),
                        clock_ns,
                    )
                timing["maximum_policy_elapsed_ms"] = max(
                    timing["maximum_policy_elapsed_ms"], result.elapsed_ns / 1e6
                )
                timely = policy_result_is_timely(
                    result.elapsed_ns,
                    policy_period_ns,
                    round(config.streaming.policy_watchdog_s * 1e9),
                ) if not is_preflight_result else True
                if not timely:
                    timing["policy_deadline_misses"] += 1
                    records.append(_policy_record(policy_step, observation, result, False, True))
                    images.append(result.image)
                else:
                    proposed_target = latch_tcp_target(
                        state.O_T_EE,
                        [result.robot_action.dx, result.robot_action.dy, result.robot_action.dz],
                    )
                    _validate_workspace_target(proposed_target, config)
                    target_pose = proposed_target
                    history.update(result.raw_action, result.executed_action)
                    if gripper_queue is not None and result.robot_action.gripper_width is not None:
                        gripper_queue.command(result.robot_action.gripper_width)
                    records.append(_policy_record(policy_step, observation, result, True, True))
                    images.append(result.image)
                    policy_accepted = True

            if streaming_check:
                q_target = np.asarray(state.q, dtype=np.float64)
            elif target_pose is None:
                q_target = np.asarray(state.q, dtype=np.float64)
            else:
                jacobian = model.zero_jacobian(state)
                delta_q = dls_joint_delta(
                    jacobian,
                    pose_error(state.O_T_EE, target_pose),
                    config.streaming.dls_lambda,
                )
                q_target = np.asarray(state.q, dtype=np.float64) + delta_q
                q_target = _apply_joint_limits(q_target, config.streaming.joint_limit_margin_rad)

            command = handler.set_joint_position_target(
                pylibfranka.AsyncPositionControlHandler.JointPositionTarget(q_target.tolist())
            )
            if not command.was_successful:
                raise RuntimeError(f"Async position target failed: {command.error_message}")
            if gripper_queue is not None:
                gripper_queue.poll()
            tick_elapsed_ns = clock_ns() - tick_started_ns
            if tick_elapsed_ns > config.streaming.control_watchdog_s * 1e9:
                raise RuntimeError(
                    f"Control tick stalled for {tick_elapsed_ns / 1e6:.3f} ms"
                )
            control_trace.append(
                {
                    "tick_index": tick_index,
                    "policy_step": policy_step if not streaming_check else None,
                    "policy_accepted": policy_accepted,
                    "scheduled_ns": deadline_ns,
                    "observed_ns": clock_ns(),
                    "lateness_ns": max(0, lateness_ns),
                    "elapsed_ns": tick_elapsed_ns,
                    "q": np.asarray(state.q, dtype=np.float64).tolist(),
                    "q_target": q_target.tolist(),
                    "command_success_rate": observation.control_command_success_rate,
                }
            )

        stop_errors = _safe_stop(handler, gripper_queue)
        stopped_ns = clock_ns()
        stopped = True
        handler = None
        gripper_queue = None
        if stop_errors:
            raise RuntimeError("; ".join(stop_errors))
        timing["expected_control_ticks"] = total_ticks
        timing["expected_policy_steps"] = 0 if streaming_check else config.runner.steps
        timing["streaming_check"] = streaming_check
        elapsed_s = max(0.0, (stopped_ns - start_ns) / 1e9)
        timing["control_elapsed_s"] = elapsed_s
        timing["scheduled_horizon_s"] = total_ticks / config.streaming.ik_frequency_hz
        timing["effective_control_frequency_hz"] = (
            (total_ticks - 1) / elapsed_s if total_ticks > 1 and elapsed_s > 0.0 else None
        )
        policy_samples = 0 if streaming_check else config.runner.steps
        timing["effective_policy_frequency_hz"] = (
            (policy_samples - 1) / elapsed_s
            if policy_samples > 1 and elapsed_s > 0.0
            else None
        )
        return _write_streaming_artifacts(
            run_dir, config, initial_report, records, images, control_trace, timing, save_step_data
        )
    except BaseException as exc:
        if not stopped:
            cleanup_errors = _safe_stop(handler, gripper_queue)
            handler = None
            gripper_queue = None
            stopped = True
            timing["cleanup_errors"] = cleanup_errors
        timing["aborted"] = True
        timing["exception_type"] = type(exc).__name__
        timing["exception"] = str(exc)
        try:
            _write_streaming_artifacts(
                run_dir,
                config,
                initial_report,
                records,
                images,
                control_trace,
                timing,
                save_step_data,
            )
        except Exception as write_exc:
            warnings.warn(
                f"Failed to write aborted streaming artifacts: {write_exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        raise
    finally:
        if not stopped:
            cleanup_errors = _safe_stop(handler, gripper_queue)
            if cleanup_errors:
                warnings.warn("; ".join(cleanup_errors), RuntimeWarning, stacklevel=2)
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                warnings.warn(f"camera close: {exc}", RuntimeWarning, stacklevel=2)


def _validate_workspace_target(target_pose: np.ndarray, config: BundleDeployConfig) -> None:
    position = target_pose[:3, 3]
    minimum = np.asarray(config.workspace["minimum"], dtype=np.float64)
    maximum = np.asarray(config.workspace["maximum"], dtype=np.float64)
    if (
        not np.all(np.isfinite(position))
        or np.any(position < minimum)
        or np.any(position > maximum)
    ):
        raise RuntimeError(
            f"Latched TCP target {position.tolist()} is outside workspace "
            f"[{minimum.tolist()}, {maximum.tolist()}]"
        )


def _validate_current_state_limits(state: Any, config: BundleDeployConfig) -> None:
    joints = np.asarray(state.q, dtype=np.float64).reshape(-1)
    if (
        joints.shape != (7,)
        or not np.all(np.isfinite(joints))
        or np.any(joints < PANDA_JOINT_LOWER)
        or np.any(joints > PANDA_JOINT_UPPER)
    ):
        raise RuntimeError("Current joint state is non-finite or outside Panda limits")
    pose = reshape_column_major(state.O_T_EE, 4, 4)
    _validate_workspace_target(pose, config)


def _apply_joint_limits(q_target: Any, margin: float) -> np.ndarray:
    target = np.asarray(q_target, dtype=np.float64).reshape(-1)
    if target.shape != (7,) or not np.all(np.isfinite(target)):
        raise RuntimeError("Joint target contains NaN, Inf, or the wrong dimension")
    lower = PANDA_JOINT_LOWER + margin
    upper = PANDA_JOINT_UPPER - margin
    return np.clip(target, lower, upper)


def _policy_record(
    step_index: int,
    observation: RobotObservation,
    result: _PolicyResult,
    accepted: bool,
    motion_enabled: bool,
    *,
    deadline_lateness_ns: int = 0,
    timing_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    record = {
        "step_index": step_index,
        "observation_before": observation.to_dict(),
        "model_input": {
            "action_history": result.action_history.tolist(),
            "proprio_obs": result.proprio.tolist(),
            "rgb_path": None,
            "rgb_shape": list(result.image.shape),
            "tactile_rgb_paths": {},
            "tactile_rgb_shapes": {
                name: list(image.shape)
                for name, image in result.tactile_images.items()
            },
            "step_data_path": None,
            "rma_actor_input": dict(result.inference_info),
        },
        "raw_action": result.raw_action.tolist(),
        "clipped_action": result.executed_action.tolist(),
        "limited_action": result.executed_action.tolist(),
        "executed_action": result.executed_action.tolist() if accepted else None,
        "robot_action": result.robot_action.to_dict(),
        "info": {
            "motion_executed": bool(accepted and motion_enabled),
            "policy_action_accepted": accepted,
            "policy_deadline_miss": not accepted,
            "policy_elapsed_ms": result.elapsed_ns / 1e6,
            "policy_deadline_lateness_ms": max(0, deadline_lateness_ns) / 1e6,
            "timing_ms": dict(timing_ms or {}),
        },
        "observation": observation.to_dict(),
        "observation_after": observation.to_dict(),
    }
    if result.tactile_images:
        record["_tactile_images"] = result.tactile_images
    return record
