from __future__ import annotations

import json
import math
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .e2e_bundle import (
    ActionHistoryBuffer,
    BundleDeployConfig,
    BundleTorchScriptPolicy,
    GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_TASK,
    WristRGBHistoryBuffer,
    _make_camera,
    _make_run_dir,
    _validate_bundle_action_dims,
    capture_gelsight_reference_frames,
    reject_evaluation_only_motion,
    _validate_tacex_rma_direct_action_student_contract,
    _validate_tacex_rma_gelsight_size_buckets_progress_student_contract,
    _validate_tacex_rma_gelsight_size_buckets_student_contract,
    _validate_tacex_rma_gelsight_x040_three_frame_student_contract,
    _validate_tacex_rma_x040_wide_direct_action_student_contract,
    _validate_tacex_rma_x040_wide_three_frame_direct_action_student_contract,
    _validate_tacex_rma_student_contract,
    _validate_tacex_rma_xy_student_contract,
    build_bundle_inputs,
    build_rma_contact_force_input,
    evaluate_initial_state,
)
from .hil import HILInputSnapshot, HILSettings, HILStepData, human_normalized_xyz
from .residual_runtime import ResidualDeploySettings
from .real_rl.runtime import RealRLDeploySettings
from .types import RobotAction, RobotObservation, print_error_wrench_report
from real_rlpd.runtime import RLPDDeploySettings


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


def _save_rgb_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image as a lossless PNG with OpenCV's fast encoder."""

    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"PNG artifact must be an HxWx3 RGB image, got {rgb.shape}")
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise RuntimeError(f"Failed to write PNG artifact: {path}")


def _retain_artifact_arrays(
    record: dict[str, Any],
    result: "_PolicyResult",
    config: BundleDeployConfig,
    save_step_data: bool,
) -> np.ndarray | None:
    """Drop arrays that the selected artifact profile will never write."""

    save_rgb = bool(config.camera.save_rgb or save_step_data)
    save_tactile = bool(config.tactile_camera.save_rgb or save_step_data)
    if not save_step_data:
        record.pop("_model_rgb", None)
    if not save_tactile:
        record.pop("_tactile_images", None)
        record.pop("_tactile_references", None)
    return result.image if save_rgb else None


def reshape_column_major(values: Any, rows: int, columns: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape == (rows, columns):
        return array.copy()
    if array.size != rows * columns:
        raise ValueError(
            f"Expected {rows * columns} column-major values, got {array.size}"
        )
    return array.reshape((rows, columns), order="F")


def physical_tool_tcp_translation(pose: Any, config: BundleDeployConfig) -> np.ndarray:
    """Return the configured physical tool point in robot-root metres."""

    matrix = reshape_column_major(pose, 4, 4)
    offset = np.asarray(config.tool_tcp_offset_ee_m, dtype=np.float64).reshape(-1)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("tool_tcp_offset_ee_m must contain three finite values")
    return matrix[:3, 3] + matrix[:3, :3] @ offset


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
    tool_tcp_offset = np.asarray(config.tool_tcp_offset_ee_m, dtype=np.float64).reshape(-1)
    if tool_tcp_offset.shape != (3,) or not np.all(np.isfinite(tool_tcp_offset)):
        raise ValueError("tool_tcp_offset_ee_m must contain three finite values")

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
    if stream.impedance_mode not in {"joint", "cartesian"}:
        raise ValueError("streaming.impedance_mode must be 'joint' or 'cartesian'")
    if stream.joint_impedance is not None:
        joint_impedance = np.asarray(stream.joint_impedance, dtype=np.float64).reshape(-1)
        if joint_impedance.shape != (7,) or not np.all(np.isfinite(joint_impedance)):
            raise ValueError("streaming.joint_impedance must have seven finite values or be null")
        if np.any(joint_impedance <= 0.0) or np.any(joint_impedance > 14250.0):
            raise ValueError("streaming.joint_impedance values must be in (0, 14250]")
    if stream.cartesian_impedance is not None:
        cartesian_impedance = np.asarray(stream.cartesian_impedance, dtype=np.float64).reshape(-1)
        if cartesian_impedance.shape != (6,) or not np.all(np.isfinite(cartesian_impedance)):
            raise ValueError(
                "streaming.cartesian_impedance must have six finite values or be null"
            )
        if (
            np.any(cartesian_impedance[:3] < 10.0)
            or np.any(cartesian_impedance[:3] > 3000.0)
            or np.any(cartesian_impedance[3:] < 1.0)
            or np.any(cartesian_impedance[3:] > 300.0)
        ):
            raise ValueError(
                "streaming.cartesian_impedance must be [10, 3000] N/m for XYZ "
                "and [1, 300] Nm/rad for rotation"
            )
    if stream.impedance_mode == "cartesian":
        if stream.cartesian_impedance is None:
            raise ValueError(
                "streaming.cartesian_impedance is required when impedance_mode='cartesian'"
            )
        if stream.joint_impedance is not None:
            raise ValueError(
                "streaming.joint_impedance must be null when impedance_mode='cartesian'"
            )
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

    if getattr(bundle, "is_tacex_rma_gelsight_x040_three_frame_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_gelsight_x040_three_frame_student_contract(
            bundle, config, history_scale
        )
        report = {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_gelsight_x040_three_frame_student_metadata_version": bundle.metadata.get(
                "version"
            ),
            "rgb_history": "three_frames_oldest_to_newest",
            "tactile_reference": "first_post_reset_frame_fixed_per_rollout",
        }
        if (
            bundle.metadata.get("task")
            == GELSIGHT_X040_PROGRESS_THREE_FRAME_STUDENT_TASK
        ):
            report["training_episode_length_steps"] = 150
        else:
            report["max_episode_length_steps"] = 150
        return report
    if getattr(bundle, "is_tacex_rma_gelsight_size_buckets_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_gelsight_size_buckets_student_contract(
            bundle, config, history_scale
        )
        deployment_contract = bundle.metadata.get("deployment_contract", {})
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_gelsight_size_buckets_student_metadata_version": bundle.metadata.get(
                "version"
            ),
            "tactile_reference": "first_post_reset_frame_fixed_per_rollout",
            "max_episode_length_steps": deployment_contract.get(
                "max_episode_length_steps", 150
            ),
        }
    if getattr(
        bundle, "is_tacex_rma_gelsight_size_buckets_progress_student", False
    ):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_gelsight_size_buckets_progress_student_contract(
            bundle, config, history_scale
        )
        deployment_contract = bundle.metadata.get("deployment_contract", {})
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_gelsight_size_buckets_progress_student_metadata_version": (
                bundle.metadata.get("version")
            ),
            "behavior_profile": bundle.metadata.get("behavior_profile"),
            "motion_authorization": bundle.metadata.get("motion_authorization"),
            "tactile_reference": "first_post_reset_frame_fixed_per_rollout",
            "training_episode_length_steps": deployment_contract.get(
                "training_episode_length_steps"
            ),
        }
    if getattr(bundle, "is_tacex_rma_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_student_contract(bundle, config, history_scale)
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_student_metadata_version": getattr(bundle, "metadata", {}).get("version"),
            "rma_student_v5": True,
        }
    if getattr(bundle, "is_tacex_rma_xy_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_xy_student_contract(bundle, config, history_scale)
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_xy_student_metadata_version": bundle.metadata.get("version"),
            "rma_contact_force_source": config.model.rma_contact_force_source,
        }
    if getattr(bundle, "is_tacex_rma_direct_action_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_direct_action_student_contract(bundle, config, history_scale)
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_direct_action_student_metadata_version": bundle.metadata.get("version"),
        }
    if getattr(bundle, "is_tacex_rma_x040_wide_direct_action_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_x040_wide_direct_action_student_contract(
            bundle, config, history_scale
        )
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_x040_wide_direct_action_student_metadata_version": bundle.metadata.get(
                "version"
            ),
        }
    if getattr(bundle, "is_tacex_rma_x040_wide_three_frame_direct_action_student", False):
        history_scale = np.asarray(config.model.history_scale, dtype=np.float32).reshape(-1)
        _validate_tacex_rma_x040_wide_three_frame_direct_action_student_contract(
            bundle, config, history_scale
        )
        return {
            "policy_frequency_hz": stream.policy_frequency_hz,
            "ik_frequency_hz": stream.ik_frequency_hz,
            "ticks_per_action": 2,
            "rma_x040_wide_three_frame_direct_action_student_metadata_version": (
                bundle.metadata.get("version")
            ),
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
    return {
        "policy_frequency_hz": stream.policy_frequency_hz,
        "ik_frequency_hz": stream.ik_frequency_hz,
        "ticks_per_action": 2,
        "xyz_command_frame": "robot_root",
        "commissioning_action_limit": stream.commissioning_action_limit,
        "training_episode_length_steps": contract.get("max_episode_length_steps"),
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

    position = np.asarray(
        observation.metadata.get("physical_tool_tcp_translation_m", observation.tcp_translation),
        dtype=np.float64,
    ).reshape(-1)
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
        "physical tool TCP is non-finite or outside the configured workspace",
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
        self._cached_state = state
        self._desired_width = float(state.width)
        self._future: Any | None = None
        self._future_kind: str | None = None
        self._future_target: float | None = None
        self._future_start_width: float | None = None
        self._pending_grasp_width: float | None = None
        self._control_session_active = False
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._closing = False
        self._closed = False
        self._api_call_in_progress = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._loop,
            name="franka-gripper-worker",
            daemon=True,
        )
        self._thread.start()

    @property
    def cached_state(self) -> Any:
        with self._lock:
            return self._cached_state

    @property
    def desired_width(self) -> float:
        with self._lock:
            return self._desired_width

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Asynchronous gripper worker failed: {self._error}") from self._error

    def check(self) -> None:
        with self._lock:
            self._raise_error_locked()
            if self._closed:
                raise RuntimeError("Asynchronous gripper worker is closed")

    def mark_control_session_active(self) -> None:
        with self._lock:
            self._raise_error_locked()
            if self._closed or self._closing:
                raise RuntimeError("Cannot activate a closed gripper worker")
            self._control_session_active = True

    def command(self, width: float) -> None:
        # This is the only method called from the 30 Hz policy path. It must
        # never enter franky/libfranka: it only coalesces the newest target.
        with self._condition:
            self._raise_error_locked()
            if self._closed or self._closing:
                raise RuntimeError("Cannot command a closed gripper worker")
            max_width = float(getattr(self._cached_state, "max_width", 0.08))
            if not math.isfinite(max_width) or max_width <= 0.0:
                max_width = 0.08
            target = float(np.clip(width, 0.0, max_width))
            self._desired_width = target
            self._condition.notify_all()

    def _clear_future_locked(self) -> None:
        self._future = None
        self._future_kind = None
        self._future_target = None
        self._future_start_width = None

    def _next_operation_locked(self) -> tuple[str, float, float] | None:
        if self._pending_grasp_width is not None:
            width = self._pending_grasp_width
            self._pending_grasp_width = None
            return "grasp", width, width
        current_width = float(self._cached_state.width)
        target = self._desired_width
        if (
            bool(getattr(self._cached_state, "is_grasped", False))
            and target <= current_width + self.tolerance
        ):
            # Keep an active force-controlled grasp instead of issuing a move
            # that cannot reach the nominal closed width through the object.
            return None
        if abs(target - current_width) <= self.tolerance:
            return None
        return "move", target, current_width

    def _start_operation(self, operation: tuple[str, float, float]) -> None:
        kind, target, start_width = operation
        with self._condition:
            if self._closing:
                return
            self._api_call_in_progress = True
        try:
            if kind == "grasp":
                future = self.gripper.grasp_async(target, self.speed, self.force)
            else:
                future = self.gripper.move_async(target, self.speed)
        finally:
            with self._condition:
                self._api_call_in_progress = False
                self._condition.notify_all()
        with self._condition:
            self._future = future
            self._future_kind = kind
            self._future_target = target
            self._future_start_width = start_width
            self._condition.notify_all()

    def _complete_future(self, future: Any) -> None:
        with self._condition:
            kind = self._future_kind
            target = self._future_target
            start_width = self._future_start_width
            self._api_call_in_progress = True
        try:
            success = bool(future.get())
            state = self.gripper.state
        except Exception as exc:
            raise RuntimeError(
                f"Asynchronous gripper {kind or 'command'} raised an exception"
            ) from exc
        finally:
            with self._condition:
                self._api_call_in_progress = False
                self._condition.notify_all()

        with self._condition:
            if future is not self._future:
                return
            self._clear_future_locked()
            self._cached_state = state
            current_width = float(state.width)
            if success or (
                kind == "grasp" and bool(getattr(state, "is_grasped", False))
            ):
                self._condition.notify_all()
                return
            blocked_close = (
                kind == "move"
                and target is not None
                and start_width is not None
                and target < start_width - self.tolerance
                and current_width > target + self.tolerance
            )
            if blocked_close:
                # Expected object contact: convert the measured blocked width
                # into a force-controlled grasp on the same worker thread.
                self._pending_grasp_width = current_width
                self._condition.notify_all()
                return
            raise RuntimeError(
                "Asynchronous gripper command failed: "
                f"kind={kind}, target_width={target}, "
                f"actual_width={current_width:.6f}, "
                f"is_grasped={bool(getattr(state, 'is_grasped', False))}"
            )

    def _loop(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closing:
                        return
                    future = self._future
                    operation = (
                        None if future is not None else self._next_operation_locked()
                    )
                    if future is None and operation is None:
                        self._condition.wait(timeout=0.01)
                        continue
                if operation is not None:
                    self._start_operation(operation)
                    continue
                assert future is not None
                if future.wait(0.0):
                    self._complete_future(future)
                else:
                    with self._condition:
                        self._condition.wait(timeout=0.002)
        except BaseException as exc:
            with self._condition:
                if not self._closing:
                    self._error = exc
                self._condition.notify_all()

    def poll(self) -> None:
        # Backward-compatible name used by streaming loops. All Franka Hand
        # calls are owned by the worker; this check is bounded and non-blocking.
        self.check()

    def wait_idle(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                self._raise_error_locked()
                operation = self._next_operation_locked()
                if operation is not None:
                    # Restore a pending grasp consumed by the idle probe.
                    if operation[0] == "grasp":
                        self._pending_grasp_width = operation[1]
                idle = (
                    self._future is None
                    and not self._api_call_in_progress
                    and operation is None
                )
                if idle:
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out waiting for asynchronous gripper worker")
                self._condition.wait(timeout=min(0.02, remaining))

    def stop(self) -> None:
        with self._condition:
            if self._closed:
                return
            should_stop = bool(
                self._control_session_active
                or self._future is not None
                or self._api_call_in_progress
                or self._next_operation_locked() is not None
            )
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            raise RuntimeError("Asynchronous gripper worker did not stop")
        stop_error: BaseException | None = None
        try:
            if should_stop:
                self.gripper.stop()
        except BaseException as exc:
            stop_error = exc
        with self._condition:
            worker_error = self._error
            self._clear_future_locked()
            self._pending_grasp_width = None
            self._control_session_active = False
            self._closed = True
            self._condition.notify_all()
        if stop_error is not None:
            raise RuntimeError(f"Failed to stop gripper: {stop_error}") from stop_error
        if worker_error is not None:
            raise RuntimeError(f"Asynchronous gripper worker failed: {worker_error}") from worker_error


@dataclass
class _PolicyResult:
    raw_action: np.ndarray
    executed_action: np.ndarray
    robot_action: RobotAction
    action_history: np.ndarray
    proprio: np.ndarray
    contact_force_n: np.ndarray | None
    image: np.ndarray
    model_rgb: np.ndarray
    tactile_images: dict[str, np.ndarray]
    tactile_references: dict[str, np.ndarray]
    inference_info: dict[str, Any]
    elapsed_ns: int
    raw_image: np.ndarray | None = None
    hil_step: HILStepData | None = None
    camera_metadata: dict[str, Any] = field(default_factory=dict)
    observation_metadata: dict[str, Any] = field(default_factory=dict)


def _run_policy_tick(
    bundle: BundleTorchScriptPolicy,
    camera: Any,
    history: ActionHistoryBuffer,
    observation: RobotObservation,
    config: BundleDeployConfig,
    allow_full_scale: bool,
    desired_gripper_width: float,
    clock_ns: Callable[[], int],
    rgb_history: WristRGBHistoryBuffer | None = None,
    tactile_references: dict[str, np.ndarray] | None = None,
    *,
    collect_rma_debug: bool = False,
) -> _PolicyResult:
    started = clock_ns()
    if tactile_references is None:
        tactile_references = getattr(bundle, "_deployment_tactile_references", None)
    if rgb_history is None and getattr(bundle, "uses_wrist_rgb_history", False):
        rgb_history = getattr(bundle, "_deployment_rgb_history", None)
        if rgb_history is None:
            rgb_history = WristRGBHistoryBuffer(
                bundle.rgb_history_frames,
                (bundle.rgb_height, bundle.rgb_width, 3),
            )
            bundle._deployment_rgb_history = rgb_history
    camera_metadata: dict[str, Any] = {}
    raw_image: np.ndarray | None = None
    if hasattr(camera, "read_policy_packet_with_raw"):
        image, raw_image, camera_metadata = camera.read_policy_packet_with_raw()
    elif hasattr(camera, "read_policy_packet"):
        image, camera_metadata = camera.read_policy_packet()
    else:
        image = camera.read()
    if image.shape != (bundle.rgb_height, bundle.rgb_width, 3):
        raise ValueError(
            f"Expected RGB frame {(bundle.rgb_height, bundle.rgb_width, 3)}, got {image.shape}"
        )
    model_rgb = rgb_history.update(image) if rgb_history is not None else image
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
    if getattr(bundle, "has_gelsight_reference_inputs", False) is True and not tactile_references:
        raise RuntimeError(
            "Model requires fixed GelSight reference inputs, but the deployment "
            "session did not capture them"
        )
    predict_kwargs = {"collect_rma_debug": True} if collect_rma_debug else {}
    contact_force_n = build_rma_contact_force_input(observation, bundle, config)
    if tactile_images:
        raw_action = bundle.predict(
            action_history,
            proprio,
            model_rgb,
            tactile_images["gsmini_left_rgb"],
            tactile_images["gsmini_right_rgb"],
            gsmini_left_reference_rgb=(tactile_references or {}).get(
                "gsmini_left_reference_rgb"
            ),
            gsmini_right_reference_rgb=(tactile_references or {}).get(
                "gsmini_right_reference_rgb"
            ),
            contact_force_n=contact_force_n,
            **predict_kwargs,
        )
    else:
        predict_args = (action_history, proprio, model_rgb)
        if contact_force_n is None:
            raw_action = bundle.predict(*predict_args, **predict_kwargs)
        else:
            raw_action = bundle.predict(
                *predict_args,
                contact_force_n=contact_force_n,
                **predict_kwargs,
            )
    base_raw_action = np.asarray(raw_action, dtype=np.float32)
    base_limited_action = clip_streaming_action(
        base_raw_action, config, allow_full_scale
    )
    if getattr(bundle, "rlpd_runtime", None) is not None:
        raw_action = bundle.apply_rlpd_post_limit(base_limited_action)
        bundle.last_inference_info["rlpd"]["base_raw_action"] = base_raw_action.tolist()
    executed = clip_streaming_action(raw_action, config, allow_full_scale)
    action = streaming_robot_action(executed, config, desired_gripper_width)
    image_copy = np.asarray(image, dtype=np.uint8).copy()
    model_rgb_copy = (
        image_copy
        if model_rgb is image
        else np.asarray(model_rgb, dtype=np.uint8).copy()
    )
    return _PolicyResult(
        raw_action=np.asarray(raw_action, dtype=np.float32),
        executed_action=executed,
        robot_action=action,
        action_history=action_history,
        proprio=proprio,
        contact_force_n=contact_force_n,
        image=image_copy,
        raw_image=(
            None if raw_image is None else np.asarray(raw_image, dtype=np.uint8).copy()
        ),
        model_rgb=model_rgb_copy,
        tactile_images={
            name: value.copy() for name, value in tactile_images.items()
        },
        # Reference images are immutable for the rollout. Sharing this pair
        # avoids copying the same arrays on every policy boundary.
        tactile_references=dict(tactile_references or {}),
        inference_info=dict(getattr(bundle, "last_inference_info", {})),
        elapsed_ns=clock_ns() - started,
        camera_metadata=camera_metadata,
        observation_metadata=dict(observation.metadata),
    )


def warm_up_bundle_policy(bundle: BundleTorchScriptPolicy) -> dict[str, Any]:
    """Initialize the exact inference path before any policy deadline applies.

    The 0912 Direct BC TorchScript/CUDA graph has two observable lazy-start
    costs on this deployment host: the first all-zero invocation and the first
    non-zero image invocation. Both must happen before the first timed camera
    boundary; otherwise a safe preview aborts at step zero despite subsequent
    inference taking only a few milliseconds.
    """

    pixel_values = [0]
    if (
        getattr(bundle, "metadata", {}).get("deployment_variant")
        == "frozen_encoder_direct_bc"
    ):
        pixel_values.append(127)

    action_history = np.zeros(bundle.history_dim, dtype=np.float32)
    proprio = np.zeros(bundle.proprio_dim, dtype=np.float32)
    contact_force = (
        np.zeros(getattr(bundle, "contact_force_dim", 0), dtype=np.float32)
        if getattr(bundle, "contact_force_dim", 0)
        else None
    )
    elapsed_ms: list[float] = []
    for pixel_value in pixel_values:
        rgb_input_shape = getattr(bundle, "rgb_input_shape", None)
        if rgb_input_shape is None:
            rgb_input_shape = (bundle.rgb_height, bundle.rgb_width, 3)
        wrist_rgb = np.full(rgb_input_shape, pixel_value, dtype=np.uint8)
        tactile = {
            name: np.full(shape, pixel_value, dtype=np.uint8)
            for name, shape in getattr(bundle, "gelsight_input_shapes", {}).items()
        }
        started_ns = time.perf_counter_ns()
        if tactile:
            bundle.predict(
                action_history,
                proprio,
                wrist_rgb,
                tactile["gsmini_left_rgb"],
                tactile["gsmini_right_rgb"],
                gsmini_left_reference_rgb=tactile.get(
                    "gsmini_left_reference_rgb"
                ),
                gsmini_right_reference_rgb=tactile.get(
                    "gsmini_right_reference_rgb"
                ),
                contact_force_n=contact_force,
            )
        elif contact_force is None:
            bundle.predict(action_history, proprio, wrist_rgb)
        else:
            bundle.predict(
                action_history,
                proprio,
                wrist_rgb,
                contact_force_n=contact_force,
            )
        elapsed_ms.append((time.perf_counter_ns() - started_ns) / 1e6)

    report = {
        "iterations": len(pixel_values),
        "pixel_values": pixel_values,
        "elapsed_ms": elapsed_ms,
    }
    if len(pixel_values) > 1:
        print(
            "Policy CUDA warmup completed before control: "
            + ", ".join(f"{value:.1f} ms" for value in elapsed_ms),
            flush=True,
        )
    return report


def apply_hil_action(
    result: _PolicyResult,
    snapshot: HILInputSnapshot,
    settings: HILSettings,
    config: BundleDeployConfig,
    allow_full_scale: bool,
    desired_gripper_width: float,
) -> _PolicyResult:
    """Select a boundary-time human XYZ action while retaining policy gripper control."""

    settings.validate()
    if not settings.enabled:
        raise ValueError("apply_hil_action requires enabled HIL settings")
    base_action = np.asarray(result.raw_action, dtype=np.float32).reshape(-1)
    if base_action.shape != (4,) or not np.all(np.isfinite(base_action)):
        raise ValueError("HIL requires a finite four-dimensional base action")

    human_action: np.ndarray | None = None
    residual_target = np.zeros(3, dtype=np.float32)
    selected_action = base_action.copy()
    if snapshot.intervention:
        human_xyz = human_normalized_xyz(
            snapshot,
            speed_m_s=settings.speed_m_s,
            policy_frequency_hz=config.streaming.policy_frequency_hz,
            action_scales=config.action_adapter.scales,
        )
        human_action = np.concatenate((human_xyz, base_action[3:4])).astype(np.float32)
        residual_target = np.asarray(human_xyz - base_action[:3], dtype=np.float32)
        selected_action = human_action.copy()

    limited_action = clip_streaming_action(selected_action, config, allow_full_scale)
    robot_action = streaming_robot_action(
        limited_action,
        config,
        desired_gripper_width,
    )
    return replace(
        result,
        raw_action=selected_action,
        executed_action=limited_action,
        robot_action=robot_action,
        hil_step=HILStepData(
            intervention=snapshot.intervention,
            base_action=base_action.copy(),
            human_action=None if human_action is None else human_action.copy(),
            residual_target_xyz=residual_target.copy(),
            input_snapshot=snapshot,
        ),
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
    images: list[np.ndarray | None],
    control_trace: list[dict[str, Any]],
    timing: dict[str, Any],
    save_step_data: bool,
) -> dict[str, Any]:
    artifact_started = time.perf_counter()
    artifact_profile = {
        "png_encoder": "opencv_lossless",
        "model_rgb_png": bool(config.camera.save_rgb or save_step_data),
        "tactile_rgb_png": bool(config.tactile_camera.save_rgb or save_step_data),
        "step_data_npz": bool(save_step_data),
        "raw_boundary_rgb_png": any(
            record.get("_raw_image") is not None for record in records
        ),
    }
    (run_dir / "rgb").mkdir(exist_ok=True)
    if config.tactile_camera.enabled:
        (run_dir / "tactile_rgb").mkdir(exist_ok=True)
    if save_step_data:
        (run_dir / "step_data").mkdir(exist_ok=True)
    if any(record.get("_raw_image") is not None for record in records):
        (run_dir / "raw_rgb").mkdir(exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (run_dir / "initial_state_check.json").write_text(
        json.dumps(initial_report, indent=2), encoding="utf-8"
    )
    for index, (record, image) in enumerate(zip(records, images)):
        raw_image = record.pop("_raw_image", None)
        raw_camera_frame = record.pop("_offline_boundary_camera_frame", None)
        next_raw_image = record.pop("_offline_next_raw_image", None)
        next_camera_frame = record.pop("_offline_next_camera_frame", None)
        tactile_images = record.pop("_tactile_images", {})
        tactile_references = record.pop("_tactile_references", {})
        model_rgb = record.pop("_model_rgb", None)
        rgb_relpath = f"rgb/step_{index:04d}.png"
        record["model_input"]["rgb_path"] = rgb_relpath
        if config.camera.save_rgb or save_step_data:
            if image is None:
                raise RuntimeError("RGB artifact was released before it could be written")
            _save_rgb_png(run_dir / rgb_relpath, image)
        if raw_image is not None:
            raw_relpath = f"raw_rgb/boundary_{index:04d}.png"
            _save_rgb_png(run_dir / raw_relpath, raw_image)
            record["model_input"]["raw_rgb_path"] = raw_relpath
            record["model_input"]["raw_rgb_shape"] = list(raw_image.shape)
            record["model_input"]["offline_apriltag_boundary"] = {
                "raw_rgb_path": raw_relpath,
                "raw_rgb_shape": list(raw_image.shape),
                "camera_frame": dict(raw_camera_frame or {}),
            }
        if next_raw_image is not None:
            next_index = index + 1
            next_relpath = f"raw_rgb/boundary_{next_index:04d}.png"
            _save_rgb_png(run_dir / next_relpath, next_raw_image)
            record["model_input"]["next_raw_rgb_path"] = next_relpath
            record["model_input"]["next_raw_rgb_shape"] = list(next_raw_image.shape)
            record["model_input"]["next_boundary_camera_frame"] = dict(
                next_camera_frame or {}
            )
        tactile_paths: dict[str, str] = {}
        for name, tactile_image in tactile_images.items():
            relpath = f"tactile_rgb/{name}_step_{index:04d}.png"
            tactile_paths[name] = relpath
            if config.tactile_camera.save_rgb or save_step_data:
                _save_rgb_png(run_dir / relpath, tactile_image)
        record["model_input"]["tactile_rgb_paths"] = tactile_paths
        reference_paths: dict[str, str] = {}
        if tactile_references:
            reference_dir = run_dir / "tactile_reference_rgb"
            reference_dir.mkdir(exist_ok=True)
            for name, tactile_reference in tactile_references.items():
                relpath = f"tactile_reference_rgb/{name}.png"
                reference_paths[name] = relpath
                output = run_dir / relpath
                if not output.exists() and (config.tactile_camera.save_rgb or save_step_data):
                    _save_rgb_png(output, tactile_reference)
        record["model_input"]["tactile_reference_rgb_paths"] = reference_paths
        if save_step_data:
            if model_rgb is None:
                raise RuntimeError("Step-data RGB was released before it could be written")
            data_relpath = f"step_data/step_{index:04d}.npz"
            step_payload = {
                "action_history": np.asarray(
                    record["model_input"]["action_history"], dtype=np.float32
                ),
                "proprio_obs": np.asarray(
                    record["model_input"]["proprio_obs"], dtype=np.float32
                ),
                **{
                    record["model_input"].get("rgb_input_name", "wrist_rgb"):
                    np.asarray(model_rgb, dtype=np.uint8)
                },
                **(
                    {
                        "contact_force_n": np.asarray(
                            record["model_input"]["contact_force_n"], dtype=np.float32
                        )
                    }
                    if record["model_input"]["contact_force_n"] is not None
                    else {}
                ),
                "raw_action": np.asarray(record["raw_action"], dtype=np.float32),
                "executed_action": np.asarray(
                    record["limited_action"], dtype=np.float32
                ),
                **tactile_images,
                **tactile_references,
            }
            if "intervention" in record:
                step_payload.update(
                    {
                        "episode_id": np.asarray(record["episode_id"]),
                        "step_id": np.asarray(record["step_id"], dtype=np.int64),
                        "intervention": np.asarray(
                            record["intervention"], dtype=np.bool_
                        ),
                        "policy_action_accepted": np.asarray(
                            record["info"]["policy_action_accepted"], dtype=np.bool_
                        ),
                        "base_action": np.asarray(
                            record["base_action"], dtype=np.float32
                        ),
                        "residual_target_xyz": np.asarray(
                            record["residual_target_xyz"], dtype=np.float32
                        ),
                    }
                )
                if record["human_action"] is not None:
                    step_payload["human_action"] = np.asarray(
                        record["human_action"], dtype=np.float32
                    )
            if "predicted_residual_xyz" in record:
                step_payload.update(
                    {
                        "base_action": np.asarray(
                            record["base_action"], dtype=np.float32
                        ),
                        "predicted_residual_xyz": np.asarray(
                            record["predicted_residual_xyz"], dtype=np.float32
                        ),
                        "applied_residual_xyz": np.asarray(
                            record["applied_residual_xyz"], dtype=np.float32
                        ),
                        "residual_scale": np.asarray(
                            record["residual_scale"], dtype=np.float32
                        ),
                        "residual_max_abs": np.asarray(
                            record["residual_max_abs"], dtype=np.float32
                        ),
                    }
                )
            if "sac_unit_action" in record:
                real_rl_info = record["model_input"]["rma_actor_input"]["real_rl"]
                step_payload.update(
                    {
                        "real_rl_state": np.asarray(real_rl_info["state"], dtype=np.float32),
                        "base_action": np.asarray(record["base_action"], dtype=np.float32),
                        "sac_unit_action": np.asarray(record["sac_unit_action"], dtype=np.float32),
                        "residual_action_normalized": np.asarray(
                            record["residual_action_normalized"], dtype=np.float32
                        ),
                        "residual_action_m": np.asarray(
                            record["residual_action_m"], dtype=np.float32
                        ),
                        "policy_action_accepted": np.asarray(
                            record["info"]["policy_action_accepted"], dtype=np.bool_
                        ),
                    }
                )
            np.savez_compressed(run_dir / data_relpath, **step_payload)
            record["model_input"]["step_data_path"] = data_relpath
    with (run_dir / "rollout.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (run_dir / "control_trace.jsonl").open("w", encoding="utf-8") as handle:
        for tick in control_trace:
            handle.write(json.dumps(tick) + "\n")
    timing["artifact_profile"] = artifact_profile
    timing["artifact_write_elapsed_s"] = time.perf_counter() - artifact_started
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
        "artifact_profile": artifact_profile,
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
    hil_settings: HILSettings | None = None,
    residual_settings: ResidualDeploySettings | None = None,
    real_rl_settings: RealRLDeploySettings | None = None,
    rlpd_settings: RLPDDeploySettings | None = None,
    rlpd_expert: bool = False,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    hil_enabled = bool(hil_settings is not None and hil_settings.enabled)
    if hil_settings is not None:
        hil_settings.validate()
    if hil_enabled and streaming_check:
        raise ValueError("--hil cannot be combined with --streaming-check")
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
            hil_settings=hil_settings,
            residual_settings=residual_settings,
            real_rl_settings=real_rl_settings,
            rlpd_settings=rlpd_settings,
            rlpd_expert=rlpd_expert,
            clock_ns=clock_ns,
            sleep=sleep,
        )
    if hil_enabled:
        raise ValueError(
            "--hil currently requires streaming.backend='server9_joint_position'"
        )
    if residual_settings is not None:
        raise ValueError(
            "Residual BC currently requires streaming.backend='server9_joint_position'"
        )
    if real_rl_settings is not None:
        raise ValueError(
            "Real-RL currently requires streaming.backend='server9_joint_position'"
        )
    if rlpd_settings is not None or rlpd_expert:
        raise ValueError("RLPD requires streaming.backend='server9_joint_position'")
    run_dir = _make_run_dir(config)
    records: list[dict[str, Any]] = []
    images: list[np.ndarray | None] = []
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
            reject_evaluation_only_motion(bundle, execute_motion)
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
        rgb_history: WristRGBHistoryBuffer | None = None
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
            capture_gelsight_reference_frames(camera, bundle)
            history = ActionHistoryBuffer(
                bundle.history_dim,
                config.model.history_source,
                config.model.history_scale,
                config.model.history_delay_steps,
                processed_action_scale=config.action_adapter.scales,
            )
            rgb_history = (
                WristRGBHistoryBuffer(
                    bundle.rgb_history_frames,
                    (bundle.rgb_height, bundle.rgb_width, 3),
                )
                if getattr(bundle, "uses_wrist_rgb_history", False) is True
                else None
            )
            # Camera warm-up is completed by construction. Exercise every
            # required TorchScript/CUDA startup path before policy deadlines.
            timing["policy_warmup"] = warm_up_bundle_policy(bundle)
            first_policy = _run_policy_tick(
                bundle,
                camera,
                history,
                initial_observation,
                config,
                allow_full_scale,
                float(gripper_queue.desired_width if gripper_queue else 0.0),
                clock_ns,
                rgb_history,
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
                    rgb_history,
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
                record = _policy_record(step_index, observation, result, timely, False)
                records.append(record)
                images.append(_retain_artifact_arrays(record, result, config, save_step_data))
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
                        rgb_history,
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
                    record = _policy_record(policy_step, observation, result, False, True)
                    records.append(record)
                    images.append(
                        _retain_artifact_arrays(record, result, config, save_step_data)
                    )
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
                    record = _policy_record(policy_step, observation, result, True, True)
                    records.append(record)
                    images.append(
                        _retain_artifact_arrays(record, result, config, save_step_data)
                    )
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
    position = physical_tool_tcp_translation(target_pose, config)
    minimum = np.asarray(config.workspace["minimum"], dtype=np.float64)
    maximum = np.asarray(config.workspace["maximum"], dtype=np.float64)
    if (
        not np.all(np.isfinite(position))
        or np.any(position < minimum)
        or np.any(position > maximum)
    ):
        raise RuntimeError(
            f"Latched physical tool TCP target {position.tolist()} is outside workspace "
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
    episode_id: str | None = None,
    policy_action_accepted: bool | None = None,
    deadline_lateness_ns: int = 0,
    timing_ms: dict[str, float] | None = None,
) -> dict[str, Any]:
    action_accepted = accepted if policy_action_accepted is None else policy_action_accepted
    record = {
        "step_index": step_index,
        "observation_before": observation.to_dict(),
        "model_input": {
            "action_history": result.action_history.tolist(),
            "proprio_obs": result.proprio.tolist(),
            "contact_force_n": (
                None if result.contact_force_n is None else result.contact_force_n.tolist()
            ),
            "rgb_path": None,
            "rgb_shape": list(result.image.shape),
            "rgb_input_name": "wrist_rgb_history" if result.model_rgb.ndim == 4 else "wrist_rgb",
            "rgb_history_shape": list(result.model_rgb.shape) if result.model_rgb.ndim == 4 else None,
            "tactile_rgb_paths": {},
            "tactile_rgb_shapes": {
                name: list(image.shape)
                for name, image in result.tactile_images.items()
            },
            "tactile_reference_rgb_shapes": {
                name: list(image.shape)
                for name, image in result.tactile_references.items()
            },
            "step_data_path": None,
            "rma_actor_input": dict(result.inference_info),
            "camera_frame": dict(result.camera_metadata),
        },
        "raw_action": result.raw_action.tolist(),
        "clipped_action": result.executed_action.tolist(),
        "limited_action": result.executed_action.tolist(),
        "executed_action": result.executed_action.tolist() if action_accepted else None,
        "robot_action": result.robot_action.to_dict(),
        "info": {
            "motion_executed": bool(action_accepted and motion_enabled),
            "policy_action_accepted": action_accepted,
            "policy_deadline_miss": not accepted,
            "policy_elapsed_ms": result.elapsed_ns / 1e6,
            "policy_deadline_lateness_ms": max(0, deadline_lateness_ns) / 1e6,
            "timing_ms": dict(timing_ms or {}),
        },
        "observation": observation.to_dict(),
        "_model_rgb": result.model_rgb,
        "observation_after": observation.to_dict(),
    }
    if (
        "real_rl" in result.inference_info or "rlpd" in result.inference_info
    ) and result.raw_image is not None:
        record["_raw_image"] = result.raw_image
        record["_offline_boundary_camera_frame"] = dict(result.camera_metadata)
    if result.tactile_images:
        record["_tactile_images"] = result.tactile_images
    if result.tactile_references:
        record["_tactile_references"] = result.tactile_references
    if result.hil_step is not None:
        if not episode_id:
            raise ValueError("HIL policy records require a non-empty episode_id")
        hil_step = result.hil_step
        record.update(
            {
                "episode_id": episode_id,
                "step_id": step_index,
                "intervention": hil_step.intervention,
                "base_action": hil_step.base_action.tolist(),
                "human_action": (
                    None
                    if hil_step.human_action is None
                    else hil_step.human_action.tolist()
                ),
                "residual_target_xyz": hil_step.residual_target_xyz.tolist(),
                "hil_input": {
                    "pressed_keys": list(hil_step.input_snapshot.pressed_keys),
                    "direction_xyz": list(hil_step.input_snapshot.direction_xyz),
                    "sampled_monotonic_ns": hil_step.input_snapshot.sampled_monotonic_ns,
                    "focused": hil_step.input_snapshot.focused,
                },
            }
        )
    residual_info = result.inference_info.get("residual_bc")
    if residual_info is not None:
        record.update(
            {
                "base_action": list(residual_info["base_action"]),
                "predicted_residual_xyz": list(
                    residual_info["predicted_residual_xyz"]
                ),
                "applied_residual_xyz": list(
                    residual_info["applied_residual_xyz"]
                ),
                "residual_scale": float(residual_info["scale"]),
                "residual_max_abs": float(residual_info["max_abs"]),
                "residual_model_sha256": str(residual_info["model_sha256"]),
            }
        )
    real_rl_info = result.inference_info.get("real_rl")
    if real_rl_info is not None:
        record.update(
            {
                "base_action": list(real_rl_info["base_action"]),
                "sac_unit_action": list(real_rl_info["sac_unit_action"]),
                "residual_action_normalized": list(
                    real_rl_info["residual_action_normalized"]
                ),
                "residual_action_m": list(real_rl_info["residual_action_m"]),
                "real_rl_mode": str(real_rl_info["mode"]),
                "real_rl_checkpoint_sha256": real_rl_info["checkpoint_sha256"],
                "real_rl_fallback": bool(real_rl_info["fallback"]),
                "real_rl_fallback_reason": real_rl_info["fallback_reason"],
            }
        )
    rlpd_info = result.inference_info.get("rlpd")
    if rlpd_info is not None:
        if not episode_id:
            raise ValueError("RLPD policy records require a non-empty episode_id")
        expert_takeover = bool(rlpd_info.get("expert_takeover", False))
        record.update({
            "episode_id": episode_id,
            "step_id": step_index,
            "base_limited_action": list(rlpd_info["base_limited_action"]),
            "rlpd_mode": str(rlpd_info["mode"]),
            "rlpd_checkpoint_sha256": rlpd_info["checkpoint_sha256"],
            "rlpd_expert_takeover": expert_takeover,
        })
        if expert_takeover:
            # Expert takeover is an absolute 4D command. The shadow runtime's
            # zero residual is deliberately removed by _apply_rlpd_expert_action
            # and must not be serialized as an expert label.
            record.update({
                "expert_requested_action": list(
                    rlpd_info["expert_requested_action"]
                ),
                "expert_limited_action": list(
                    rlpd_info["expert_limited_action"]
                ),
            })
            record["rlpd_expert_input"] = {
                "pressed_keys": list(rlpd_info.get("pressed_keys", ())),
                "sampled_monotonic_ns": rlpd_info.get("sampled_monotonic_ns"),
                "focused": bool(rlpd_info.get("focused", False)),
            }
        else:
            record.update({
                "unit_residual_action": list(rlpd_info["unit_residual_action"]),
                "residual_normalized_action": list(
                    rlpd_info["residual_normalized_action"]
                ),
            })
    return record
