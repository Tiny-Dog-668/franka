#!/usr/bin/env python3
"""Collect Franka/D435 eye-to-hand calibration samples with guarded motion."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.calibration.hand_eye import (
    EyeToHandConfig,
    build_trajectory,
    detect_checkerboard,
    draw_checkerboard_overlay,
    estimate_target_to_camera,
    hub_transform,
    interpolate_transforms,
    load_eye_to_hand_config,
    robust_mean_transform,
    rotation_to_quaternion_xyzw,
    target_transform,
    transform_error,
    transform_from_pose,
    validate_target_transform,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "eye_to_hand_d435_215322076207.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs"
MANIFEST_NAME = "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect guarded eye-to-hand samples. The gripper is read-only; "
            "this program never sends a gripper command."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Calibration config JSON.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Parent directory for a new dataset.",
    )
    parser.add_argument(
        "--phase",
        choices=("calibration", "validation"),
        default="calibration",
        help="Collect the 25 solve poses or the 5 independent validation poses.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume an existing dataset without overwriting accepted samples.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read current state, check camera/corners, and print all targets without motion.",
    )
    parser.add_argument(
        "--camera-check",
        action="store_true",
        help="Check the configured D435 and checkerboard only; do not connect to Franka.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Do not open an OpenCV preview window; overlays are still saved.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "After one phase-level confirmation, collect every pose without "
            "per-pose prompts. Motion and quality checks remain enabled."
        ),
    )
    parser.add_argument(
        "--max-capture-attempts",
        type=int,
        default=3,
        help=(
            "Maximum capture attempts per pose in --continuous mode before "
            "stopping (default: 3)."
        ),
    )
    return parser


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_error_names(errors: Any) -> list[str]:
    names = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        value = getattr(errors, name)
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def _robot_snapshot(robot: Any) -> dict[str, Any]:
    state = robot.state
    pose = robot.current_pose.end_effector_pose
    return {
        "captured_at_unix_s": time.time(),
        "joint_positions_rad": [float(value) for value in state.q],
        "joint_velocities_rad_s": [float(value) for value in state.dq],
        "tcp_translation_m": [float(value) for value in pose.translation],
        "tcp_quaternion_xyzw": [float(value) for value in pose.quaternion],
        "base_to_gripper": transform_from_pose(
            pose.translation.tolist(), pose.quaternion.tolist()
        ).tolist(),
        "robot_mode": str(state.robot_mode).split(".")[-1],
        "has_errors": bool(robot.has_errors),
        "is_in_control": bool(robot.is_in_control),
        "control_command_success_rate": float(state.control_command_success_rate),
        "current_errors": _active_error_names(state.current_errors),
        "last_motion_errors": _active_error_names(state.last_motion_errors),
    }


def _gripper_snapshot(gripper: Any) -> dict[str, Any]:
    state = gripper.state
    return {
        "captured_at_unix_s": time.time(),
        "width_m": float(state.width),
        "max_width_m": float(state.max_width),
        "is_grasped": bool(state.is_grasped),
    }


class RealSenseColorCamera:
    def __init__(self, config: EyeToHandConfig) -> None:
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        stream = rs.config()
        stream.enable_device(config.camera.serial)
        stream.enable_stream(
            rs.stream.color,
            config.camera.width,
            config.camera.height,
            rs.format.rgb8,
            config.camera.fps,
        )
        try:
            self.profile = self.pipeline.start(stream)
        except RuntimeError as exc:
            message = str(exc)
            if "busy" in message.lower() or "VIDIOC_S_FMT" in message:
                raise RuntimeError(
                    "Configured D435 is busy. Close RealSense Viewer and every other "
                    "camera process, then retry."
                ) from exc
            raise
        self._closed = False
        for _ in range(config.camera.warmup_frames):
            self.pipeline.wait_for_frames()

        device = self.profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)
        if serial != config.camera.serial:
            self.close()
            raise RuntimeError(
                f"Connected D435 serial {serial}, expected {config.camera.serial}"
            )
        profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = profile.get_intrinsics()
        self.metadata = {
            "name": device.get_info(rs.camera_info.name),
            "serial": serial,
            "firmware": device.get_info(rs.camera_info.firmware_version),
            "stream": {
                "width": int(profile.width()),
                "height": int(profile.height()),
                "fps": int(profile.fps()),
                "format": str(profile.format()).split(".")[-1],
            },
            "intrinsics": {
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx),
                "cy": float(intrinsics.ppy),
                "camera_matrix": [
                    [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
                    [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_model": str(intrinsics.model).split(".")[-1],
                "distortion_coefficients": [
                    float(value) for value in intrinsics.coeffs
                ],
            },
        }

    def read_burst(self, count: int) -> list[dict[str, Any]]:
        frames = []
        for _ in range(count):
            frameset = self.pipeline.wait_for_frames()
            color = frameset.get_color_frame()
            if not color:
                raise RuntimeError("RealSense returned no color frame")
            frames.append(
                {
                    "rgb": np.asanyarray(color.get_data(), dtype=np.uint8).copy(),
                    "frame_number": int(color.get_frame_number()),
                    "timestamp_ms": float(color.get_timestamp()),
                }
            )
        return frames

    def close(self) -> None:
        if not self._closed:
            self.pipeline.stop()
            self._closed = True


@contextmanager
def _camera_resource(config: EyeToHandConfig):
    camera = RealSenseColorCamera(config)
    try:
        yield camera
    finally:
        camera.close()


def _camera_matrix(camera: RealSenseColorCamera) -> np.ndarray:
    return np.asarray(camera.metadata["intrinsics"]["camera_matrix"], dtype=np.float64)


def _camera_distortion(camera: RealSenseColorCamera) -> np.ndarray:
    return np.asarray(
        camera.metadata["intrinsics"]["distortion_coefficients"], dtype=np.float64
    )


def _preview_board(
    camera: RealSenseColorCamera,
    config: EyeToHandConfig,
    show_window: bool,
) -> bool:
    frames = camera.read_burst(config.camera.burst_frames)
    detections = []
    for frame in frames:
        corners, sharpness = detect_checkerboard(frame["rgb"], config.board)
        if corners is not None:
            pnp = estimate_target_to_camera(
                corners,
                _camera_matrix(camera),
                _camera_distortion(camera),
                config.board,
            )
            detections.append((pnp["reprojection_rms_px"], -sharpness, frame, corners))
    print(
        f"Camera {camera.metadata['serial']}: checkerboard detected in "
        f"{len(detections)}/{len(frames)} frames"
    )
    if not detections:
        return False
    _, _, frame, corners = min(detections, key=lambda item: (item[0], item[1]))
    if show_window:
        overlay = draw_checkerboard_overlay(frame["rgb"], corners, config.board)
        cv2.imshow(
            "Eye-to-hand checkerboard preview",
            cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
        )
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    return True


def _trajectory_records(
    center: np.ndarray,
    phase: str,
    config: EyeToHandConfig,
) -> list[dict[str, Any]]:
    records = []
    for pose in build_trajectory(phase):
        target = target_transform(center, pose)
        hub = hub_transform(center, pose.hub)
        failures = [
            *validate_target_transform(hub, config),
            *validate_target_transform(target, config),
        ]
        records.append(
            {
                **pose.to_dict(),
                "target_base_to_gripper": target.tolist(),
                "hub_base_to_gripper": hub.tolist(),
                "preflight_failures": failures,
            }
        )
    return records


def _print_trajectory(records: list[dict[str, Any]]) -> None:
    for index, record in enumerate(records, start=1):
        target = np.asarray(record["target_base_to_gripper"])
        quaternion = rotation_to_quaternion_xyzw(target[:3, :3])
        print(
            f"[{index:02d}/{len(records):02d}] {record['sample_id']} "
            f"group={record['group']} hub={record['hub']}"
        )
        print(f"  translation_m={target[:3, 3].tolist()}")
        print(f"  quaternion_xyzw={quaternion.tolist()}")
        if record["preflight_failures"]:
            for failure in record["preflight_failures"]:
                print(f"  PREFLIGHT FAILURE: {failure}")


def _move_absolute(robot: Any, target: np.ndarray, config: EyeToHandConfig) -> None:
    from franky import Affine, CartesianMotion, ReferenceType

    current_snapshot = _robot_snapshot(robot)
    current = np.asarray(current_snapshot["base_to_gripper"], dtype=np.float64)
    segments = interpolate_transforms(
        current,
        target,
        config.robot.max_segment_translation_m,
        config.robot.max_segment_rotation_deg,
    )
    preflight_failures = []
    for segment_index, segment in enumerate(segments, start=1):
        failures = validate_target_transform(segment, config)
        if failures:
            preflight_failures.extend(
                f"segment {segment_index}/{len(segments)}: {failure}"
                for failure in failures
            )
    if preflight_failures:
        raise RuntimeError(
            "Entire motion was rejected before its first segment: "
            + "; ".join(preflight_failures)
        )

    for segment in segments:
        if robot.has_errors:
            raise RuntimeError("Robot developed an active error; motion stopped.")
        quaternion = rotation_to_quaternion_xyzw(segment[:3, :3])
        robot.move(
            CartesianMotion(
                Affine(segment[:3, 3].tolist(), quaternion.tolist()),
                ReferenceType.Absolute,
                config.robot.speed,
            )
        )


def _show_overlay(path: Path, show_window: bool) -> None:
    if not show_window:
        return
    image = cv2.imread(str(path))
    if image is None:
        return
    cv2.imshow("Accepted eye-to-hand sample", image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()


def _capture_attempt(
    dataset_dir: Path,
    sample_record: dict[str, Any],
    attempt_index: int,
    robot: Any,
    gripper: Any,
    camera: RealSenseColorCamera,
    baseline_gripper_width_m: float,
    config: EyeToHandConfig,
    show_window: bool,
) -> dict[str, Any]:
    attempt_dir = (
        dataset_dir
        / "samples"
        / sample_record["sample_id"]
        / f"attempt_{attempt_index:03d}"
    )
    attempt_dir.mkdir(parents=True, exist_ok=False)
    expected = np.asarray(sample_record["target_base_to_gripper"], dtype=np.float64)
    robot_before = _robot_snapshot(robot)
    gripper_before = _gripper_snapshot(gripper)
    frames = camera.read_burst(config.camera.burst_frames)
    robot_after = _robot_snapshot(robot)
    gripper_after = _gripper_snapshot(gripper)

    candidates = []
    frame_metadata = []
    for frame_index, frame in enumerate(frames):
        filename = f"rgb_{frame_index:02d}.png"
        if not cv2.imwrite(
            str(attempt_dir / filename),
            cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"Failed to save raw RGB frame: {attempt_dir / filename}")
        corners, sharpness = detect_checkerboard(frame["rgb"], config.board)
        entry: dict[str, Any] = {
            "file": filename,
            "frame_number": frame["frame_number"],
            "timestamp_ms": frame["timestamp_ms"],
            "sharpness_laplacian_variance": sharpness,
            "checkerboard_detected": corners is not None,
        }
        if corners is not None:
            try:
                pnp = estimate_target_to_camera(
                    corners,
                    _camera_matrix(camera),
                    _camera_distortion(camera),
                    config.board,
                )
                entry["pnp"] = pnp
                candidates.append(
                    (
                        float(pnp["reprojection_rms_px"]),
                        -sharpness,
                        frame_index,
                        corners,
                        pnp,
                    )
                )
            except Exception as exc:
                entry["pnp_error"] = str(exc)
        frame_metadata.append(entry)

    failures = []
    before_transform = np.asarray(robot_before["base_to_gripper"])
    after_transform = np.asarray(robot_after["base_to_gripper"])
    capture_drift_m, capture_drift_deg = transform_error(
        before_transform, after_transform
    )
    actual_transform = robust_mean_transform([before_transform, after_transform])
    target_error_m, target_error_deg = transform_error(expected, actual_transform)
    max_joint_velocity = max(
        abs(value)
        for value in [
            *robot_before["joint_velocities_rad_s"],
            *robot_after["joint_velocities_rad_s"],
        ]
    )
    gripper_capture_drift = abs(
        gripper_after["width_m"] - gripper_before["width_m"]
    )
    gripper_session_drift = max(
        abs(gripper_before["width_m"] - baseline_gripper_width_m),
        abs(gripper_after["width_m"] - baseline_gripper_width_m),
    )
    if robot_before["has_errors"] or robot_after["has_errors"]:
        failures.append("robot has active errors")
    if max_joint_velocity > config.quality.max_joint_velocity_rad_s:
        failures.append(
            f"max joint velocity {max_joint_velocity:.6f} rad/s exceeds "
            f"{config.quality.max_joint_velocity_rad_s:.6f}"
        )
    if capture_drift_m > config.quality.max_capture_tcp_drift_m:
        failures.append(
            f"capture TCP drift {capture_drift_m:.6f} m exceeds "
            f"{config.quality.max_capture_tcp_drift_m:.6f}"
        )
    if capture_drift_deg > config.quality.max_capture_tcp_drift_deg:
        failures.append(
            f"capture TCP drift {capture_drift_deg:.6f} deg exceeds "
            f"{config.quality.max_capture_tcp_drift_deg:.6f}"
        )
    if target_error_m > config.quality.max_target_translation_error_m:
        failures.append(
            f"target translation error {target_error_m:.6f} m exceeds "
            f"{config.quality.max_target_translation_error_m:.6f}"
        )
    if target_error_deg > config.quality.max_target_rotation_error_deg:
        failures.append(
            f"target rotation error {target_error_deg:.6f} deg exceeds "
            f"{config.quality.max_target_rotation_error_deg:.6f}"
        )
    if (
        gripper_capture_drift > config.quality.max_gripper_width_drift_m
        or gripper_session_drift > config.quality.max_gripper_width_drift_m
    ):
        failures.append(
            "gripper width drift exceeds "
            f"{config.quality.max_gripper_width_drift_m:.6f} m"
        )
    selected = None
    if not candidates:
        failures.append("checkerboard/PnP failed in every burst frame")
    else:
        rms, _, frame_index, corners, pnp = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        if rms > config.quality.max_reprojection_rms_px:
            failures.append(
                f"reprojection RMS {rms:.6f} px exceeds "
                f"{config.quality.max_reprojection_rms_px:.6f}"
            )
        selected_rgb = f"rgb_{frame_index:02d}.png"
        overlay_path = attempt_dir / "selected_corners.png"
        overlay = draw_checkerboard_overlay(
            frames[frame_index]["rgb"], corners, config.board
        )
        cv2.imwrite(
            str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        )
        selected = {
            "frame_index": frame_index,
            "rgb_file": selected_rgb,
            "overlay_file": overlay_path.name,
            "corners_px": np.asarray(corners).tolist(),
            "pnp": pnp,
        }

    attempt = {
        "attempt_index": attempt_index,
        "accepted": not failures,
        "failures": failures,
        "robot_before": robot_before,
        "robot_after": robot_after,
        "base_to_gripper": actual_transform.tolist(),
        "gripper_before": gripper_before,
        "gripper_after": gripper_after,
        "quality": {
            "capture_tcp_drift_m": capture_drift_m,
            "capture_tcp_drift_deg": capture_drift_deg,
            "target_translation_error_m": target_error_m,
            "target_rotation_error_deg": target_error_deg,
            "max_joint_velocity_rad_s": max_joint_velocity,
            "gripper_capture_drift_m": gripper_capture_drift,
            "gripper_session_drift_m": gripper_session_drift,
        },
        "frames": frame_metadata,
        "selected": selected,
    }
    _atomic_json(attempt_dir / "attempt.json", attempt)
    if selected is not None:
        _show_overlay(attempt_dir / selected["overlay_file"], show_window)
    return {
        "attempt": attempt,
        "attempt_dir": str(attempt_dir.relative_to(dataset_dir)),
    }


def _new_manifest(
    config_path: Path,
    config: EyeToHandConfig,
    center: np.ndarray,
    gripper: dict[str, Any],
    camera: RealSenseColorCamera,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "config_path": str(config_path),
        "config": config.to_dict(),
        "camera": camera.metadata,
        "center_base_to_gripper": center.tolist(),
        "baseline_gripper": gripper,
        "trajectory": {
            phase: _trajectory_records(center, phase, config)
            for phase in ("calibration", "validation")
        },
        "samples": {},
    }


def _validate_resume(
    manifest: dict[str, Any],
    config: EyeToHandConfig,
    camera: RealSenseColorCamera,
    gripper: dict[str, Any],
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset manifest schema")
    if manifest.get("config") != config.to_dict():
        raise ValueError("Resume config does not exactly match the dataset config")
    if manifest["camera"]["serial"] != camera.metadata["serial"]:
        raise ValueError("Resume camera serial does not match the dataset")
    baseline_width = float(manifest["baseline_gripper"]["width_m"])
    drift = abs(float(gripper["width_m"]) - baseline_width)
    if drift > config.quality.max_gripper_width_drift_m:
        raise RuntimeError(
            f"Current gripper width drift {drift:.6f} m exceeds "
            f"{config.quality.max_gripper_width_drift_m:.6f} m; "
            "the checkerboard may have moved."
        )


def _phase_complete(
    manifest: dict[str, Any],
    phase: str,
) -> bool:
    expected = manifest["trajectory"][phase]
    return all(
        manifest["samples"].get(record["sample_id"], {}).get("status") == "accepted"
        for record in expected
    )


def _confirm(prompt: str) -> bool:
    while True:
        choice = input(prompt).strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no", "q", "quit", "a", "abort"}:
            return False
        print("无效输入：请输入 y/yes 继续，或 n/no 取消。")


def _recover_sample_entry(
    dataset_dir: Path,
    sample_id: str,
    phase: str,
    existing: dict[str, Any] | None,
) -> dict[str, Any]:
    entry = existing or {
        "sample_id": sample_id,
        "phase": phase,
        "status": "pending",
        "attempts": [],
    }
    known_attempts = {
        int(attempt["attempt_index"]) for attempt in entry.get("attempts", [])
    }
    sample_dir = dataset_dir / "samples" / sample_id
    for attempt_path in sorted(sample_dir.glob("attempt_*/attempt.json")):
        attempt = _load_json(attempt_path)
        attempt_index = int(attempt["attempt_index"])
        if attempt_index not in known_attempts:
            relative_dir = str(attempt_path.parent.relative_to(dataset_dir))
            entry.setdefault("attempts", []).append(
                {
                    "attempt_index": attempt_index,
                    "accepted": bool(attempt["accepted"]),
                    "attempt_dir": relative_dir,
                    "failures": list(attempt["failures"]),
                }
            )
            known_attempts.add(attempt_index)
        if attempt["accepted"]:
            selected = attempt["selected"]
            if selected is None:
                continue
            relative_dir = attempt_path.parent.relative_to(dataset_dir)
            entry.update(
                {
                    "status": "accepted",
                    "accepted_attempt": attempt_index,
                    "base_to_gripper": attempt["base_to_gripper"],
                    "target_to_camera": selected["pnp"]["target_to_camera"],
                    "reprojection_rms_px": selected["pnp"][
                        "reprojection_rms_px"
                    ],
                    "selected_rgb": str(relative_dir / selected["rgb_file"]),
                    "selected_overlay": str(
                        relative_dir / selected["overlay_file"]
                    ),
                }
            )
    entry["attempts"] = sorted(
        entry.get("attempts", []), key=lambda attempt: int(attempt["attempt_index"])
    )
    if entry["status"] != "accepted" and entry["attempts"]:
        entry["status"] = "rejected"
    return entry


def _collect_phase(
    dataset_dir: Path,
    manifest: dict[str, Any],
    phase: str,
    robot: Any,
    gripper: Any,
    camera: RealSenseColorCamera,
    config: EyeToHandConfig,
    show_window: bool,
    continuous: bool,
    max_capture_attempts: int,
) -> None:
    if phase == "validation" and not _phase_complete(manifest, "calibration"):
        raise RuntimeError(
            "Validation collection requires all 25 calibration samples to be accepted first."
        )
    records = manifest["trajectory"][phase]
    baseline_width = float(manifest["baseline_gripper"]["width_m"])
    for index, record in enumerate(records, start=1):
        existing = manifest["samples"].get(record["sample_id"])
        sample_entry = _recover_sample_entry(
            dataset_dir,
            record["sample_id"],
            phase,
            existing,
        )
        manifest["samples"][record["sample_id"]] = sample_entry
        _atomic_json(dataset_dir / MANIFEST_NAME, manifest)
        if sample_entry.get("status") == "accepted":
            print(f"Skipping accepted sample {record['sample_id']}")
            continue
        target = np.asarray(record["target_base_to_gripper"], dtype=np.float64)
        hub = np.asarray(record["hub_base_to_gripper"], dtype=np.float64)
        print(
            f"\n[{index}/{len(records)}] {record['sample_id']} "
            f"group={record['group']}"
        )
        print(f"Hub TCP: {hub[:3, 3].tolist()}")
        print(f"Target TCP: {target[:3, 3].tolist()}")
        print(
            "Target quaternion xyzw: "
            f"{rotation_to_quaternion_xyzw(target[:3, :3]).tolist()}"
        )
        if not continuous:
            if not _confirm(
                "确认工作空间、标定板夹持、线缆和急停安全，输入 y/yes 执行该姿态: "
            ):
                raise KeyboardInterrupt
        _move_absolute(robot, hub, config)
        _move_absolute(robot, target, config)
        time.sleep(config.robot.settle_time_s)

        attempts_this_run = 0
        while True:
            attempt_index = len(sample_entry["attempts"]) + 1
            attempts_this_run += 1
            captured = _capture_attempt(
                dataset_dir,
                record,
                attempt_index,
                robot,
                gripper,
                camera,
                baseline_width,
                config,
                show_window,
            )
            attempt = captured["attempt"]
            sample_entry["attempts"].append(
                {
                    "attempt_index": attempt_index,
                    "accepted": attempt["accepted"],
                    "attempt_dir": captured["attempt_dir"],
                    "failures": attempt["failures"],
                }
            )
            sample_entry["status"] = (
                "accepted" if attempt["accepted"] else "rejected"
            )
            manifest["samples"][record["sample_id"]] = sample_entry
            _atomic_json(dataset_dir / MANIFEST_NAME, manifest)
            if attempt["accepted"]:
                selected = attempt["selected"]
                assert selected is not None
                attempt_dir = Path(captured["attempt_dir"])
                sample_entry.update(
                    {
                        "status": "accepted",
                        "accepted_attempt": attempt_index,
                        "base_to_gripper": attempt["base_to_gripper"],
                        "target_to_camera": selected["pnp"]["target_to_camera"],
                        "reprojection_rms_px": selected["pnp"][
                            "reprojection_rms_px"
                        ],
                        "selected_rgb": str(attempt_dir / selected["rgb_file"]),
                        "selected_overlay": str(
                            attempt_dir / selected["overlay_file"]
                        ),
                    }
                )
                print(
                    f"Accepted {record['sample_id']}: reprojection RMS "
                    f"{sample_entry['reprojection_rms_px']:.4f} px"
                )
                manifest["samples"][record["sample_id"]] = sample_entry
                _atomic_json(dataset_dir / MANIFEST_NAME, manifest)
                break
            print("Sample rejected:")
            for failure in attempt["failures"]:
                print(f"  - {failure}")
            if continuous:
                if attempts_this_run < max_capture_attempts:
                    print(
                        f"Continuous mode: retrying capture "
                        f"({attempts_this_run + 1}/{max_capture_attempts}) after settling."
                    )
                    time.sleep(config.robot.settle_time_s)
                    continue
                raise RuntimeError(
                    f"{record['sample_id']} failed {max_capture_attempts} capture "
                    "attempts in this run; stopping continuous collection. Resume "
                    "the dataset after correcting the cause."
                )
            while True:
                choice = input("输入 r 重采、s 跳过、a 中止: ").strip().lower()
                if choice in {"r", "retry"}:
                    time.sleep(config.robot.settle_time_s)
                    break
                if choice in {"s", "skip"}:
                    break
                if choice in {"a", "abort", "q", "quit"}:
                    raise KeyboardInterrupt
                print("无效输入：请输入 r（重采）、s（跳过）或 a（中止）。")
            if choice in {"r", "retry"}:
                continue
            break

        manifest["samples"][record["sample_id"]] = sample_entry
        _atomic_json(dataset_dir / MANIFEST_NAME, manifest)
        _move_absolute(robot, hub, config)

    center = np.asarray(manifest["center_base_to_gripper"], dtype=np.float64)
    if continuous or _confirm("本阶段结束。输入 y/yes 将机械臂返回本次标定中心姿态: "):
        _move_absolute(robot, center, config)


def main() -> int:
    args = build_parser().parse_args()
    if args.max_capture_attempts < 1:
        raise ValueError("--max-capture-attempts must be at least 1")
    config_path = Path(args.config).expanduser().resolve()
    config = load_eye_to_hand_config(config_path)
    show_window = not args.no_window and bool(os.environ.get("DISPLAY"))
    if not args.no_window and not show_window:
        print("DISPLAY is unavailable; continuing without an OpenCV window.")

    if args.camera_check:
        with _camera_resource(config) as camera:
            print(json.dumps(camera.metadata, indent=2))
            return 0 if _preview_board(camera, config, show_window) else 2

    from franky import Gripper, RealtimeConfig, Robot

    realtime = (
        RealtimeConfig.Ignore
        if config.robot.realtime == "ignore"
        else RealtimeConfig.Enforce
    )
    robot = Robot(config.robot.ip, realtime_config=realtime)
    robot.relative_dynamics_factor = config.robot.speed
    gripper = Gripper(config.robot.ip)
    motion_started = False
    try:
        robot_state = _robot_snapshot(robot)
        if robot_state["has_errors"]:
            raise RuntimeError(
                "Robot has active errors; recover and verify state before calibration."
            )
        gripper_state = _gripper_snapshot(gripper)
        with _camera_resource(config) as camera:
            print(json.dumps(camera.metadata, indent=2))
            if not _preview_board(camera, config, show_window):
                raise RuntimeError(
                    f"{config.board.inner_corners_columns}x"
                    f"{config.board.inner_corners_rows} checkerboard was not detected. "
                    "Reposition/focus the board before any calibration motion."
                )

            if args.resume:
                dataset_dir = args.resume.expanduser().resolve()
                manifest_path = dataset_dir / MANIFEST_NAME
                manifest = _load_json(manifest_path)
                _validate_resume(manifest, config, camera, gripper_state)
                center = np.asarray(
                    manifest["center_base_to_gripper"], dtype=np.float64
                )
            else:
                if args.phase == "validation":
                    raise ValueError("--phase validation requires --resume DATASET")
                center = np.asarray(robot_state["base_to_gripper"], dtype=np.float64)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_dir = (
                    Path(args.output_dir).expanduser().resolve()
                    / f"{timestamp}_eye_to_hand"
                )
                if not args.dry_run:
                    dataset_dir.mkdir(parents=True, exist_ok=False)
                manifest = _new_manifest(
                    config_path, config, center, gripper_state, camera
                )
                if not args.dry_run:
                    _atomic_json(dataset_dir / "config.json", config.to_dict())
                    _atomic_json(dataset_dir / MANIFEST_NAME, manifest)

            records = manifest["trajectory"][args.phase]
            if args.dry_run:
                records = [
                    *manifest["trajectory"]["calibration"],
                    *manifest["trajectory"]["validation"],
                ]
            _print_trajectory(records)
            failures = [
                failure
                for record in records
                for failure in record["preflight_failures"]
            ]
            if failures:
                raise RuntimeError(
                    "Trajectory preflight failed; no motion was sent:\n  - "
                    + "\n  - ".join(failures)
                )
            print(
                "Gripper is READ-ONLY in this program. "
                f"Baseline width: {gripper_state['width_m']:.6f} m"
            )
            if args.dry_run:
                print("Dry-run passed. No robot or gripper motion command was sent.")
                return 0

            if args.continuous:
                phase_count = len(manifest["trajectory"][args.phase])
                if not _confirm(
                    f"连续模式将自动运动并采集本阶段全部 {phase_count} 个姿态，"
                    "期间不会逐姿态暂停；确认工作空间无人、标定板与线缆牢固、"
                    "急停可用后，输入 y/yes 开始: "
                ):
                    print("Continuous collection cancelled before motion.")
                    return 130

            motion_started = True
            _collect_phase(
                dataset_dir,
                manifest,
                args.phase,
                robot,
                gripper,
                camera,
                config,
                show_window,
                args.continuous,
                args.max_capture_attempts,
            )
            complete = _phase_complete(manifest, args.phase)
            print(f"Dataset: {dataset_dir}")
            print(f"Phase {args.phase} complete: {complete}")
            return 0 if complete else 2
    except KeyboardInterrupt:
        if motion_started:
            print("\nInterrupted; requesting robot.stop().")
            robot.stop()
        return 130
    except Exception:
        if motion_started:
            try:
                robot.stop()
            except Exception as stop_error:
                print(f"Warning: robot.stop() failed: {stop_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
