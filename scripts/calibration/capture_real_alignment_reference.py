#!/usr/bin/env python3
"""Capture a read-only real-scene reference for sim-to-real alignment.

The script never sends an arm or gripper command. It records Franka state,
RealSense RGB frames/settings/intrinsics, and the exact image seen by the
exported policy after crop and resize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import BundleCameraConfig, load_bundle_config

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0711.json"


def _active_error_names(errors: Any) -> list[str]:
    names = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        value = getattr(errors, name)
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def _mode_name(mode: Any) -> str:
    return str(mode).split(".")[-1]


def _robot_snapshot(robot: Any, gripper: Any | None) -> dict[str, Any]:
    state = robot.state
    pose = robot.current_pose.end_effector_pose
    joint_positions = np.asarray(state.q.tolist(), dtype=np.float64)
    result: dict[str, Any] = {
        "joint_positions_rad": joint_positions.tolist(),
        "joint_positions_deg": np.degrees(joint_positions).tolist(),
        "joint_velocities_rad_s": state.dq.tolist(),
        "tcp_translation_m": pose.translation.tolist(),
        "tcp_quaternion_xyzw": pose.quaternion.tolist(),
        "external_wrench": state.O_F_ext_hat_K.tolist(),
        "robot_mode": _mode_name(state.robot_mode),
        "has_errors": bool(robot.has_errors),
        "is_in_control": bool(robot.is_in_control),
        "control_command_success_rate": float(state.control_command_success_rate),
        "current_errors": _active_error_names(state.current_errors),
        "last_motion_errors": _active_error_names(state.last_motion_errors),
    }
    if gripper is not None:
        gripper_state = gripper.state
        result["gripper"] = {
            "width_m": float(gripper_state.width),
            "max_width_m": float(gripper_state.max_width),
            "is_grasped": bool(gripper_state.is_grasped),
        }
    return result


def _robot_drift(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    q_before = np.asarray(before["joint_positions_rad"], dtype=np.float64)
    q_after = np.asarray(after["joint_positions_rad"], dtype=np.float64)
    xyz_before = np.asarray(before["tcp_translation_m"], dtype=np.float64)
    xyz_after = np.asarray(after["tcp_translation_m"], dtype=np.float64)
    return {
        "max_abs_joint_delta_rad": float(np.max(np.abs(q_after - q_before))),
        "tcp_translation_delta_m": float(np.linalg.norm(xyz_after - xyz_before)),
    }


def _crop_bounds(config: BundleCameraConfig, width: int, height: int) -> tuple[int, int, int, int]:
    if not config.enable_crop or config.crop_width is None or config.crop_height is None:
        return 0, 0, width, height
    left = max(0, min(int(config.crop_left), max(width - 1, 0)))
    top = max(0, min(int(config.crop_top), max(height - 1, 0)))
    right = min(width, left + max(1, int(config.crop_width)))
    bottom = min(height, top + max(1, int(config.crop_height)))
    if right <= left or bottom <= top:
        raise ValueError("Configured camera crop is empty.")
    return left, top, right, bottom


def _intrinsics_dict(intrinsics: Any) -> dict[str, Any]:
    fov_x = math.degrees(2.0 * math.atan(intrinsics.width / (2.0 * intrinsics.fx)))
    fov_y = math.degrees(2.0 * math.atan(intrinsics.height / (2.0 * intrinsics.fy)))
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx_px": float(intrinsics.fx),
        "fy_px": float(intrinsics.fy),
        "cx_px": float(intrinsics.ppx),
        "cy_px": float(intrinsics.ppy),
        "camera_matrix_3x3": [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        "horizontal_fov_deg": fov_x,
        "vertical_fov_deg": fov_y,
        "distortion_model": str(intrinsics.model).split(".")[-1],
        "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
    }


def _transformed_intrinsics(
    raw: dict[str, Any],
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    output_width: int,
    output_height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cropped = {
        **raw,
        "width": crop_width,
        "height": crop_height,
        "cx_px": float(raw["cx_px"]) - left,
        "cy_px": float(raw["cy_px"]) - top,
    }
    cropped["horizontal_fov_deg"] = math.degrees(
        2.0 * math.atan(crop_width / (2.0 * float(cropped["fx_px"])))
    )
    cropped["vertical_fov_deg"] = math.degrees(
        2.0 * math.atan(crop_height / (2.0 * float(cropped["fy_px"])))
    )
    cropped["camera_matrix_3x3"] = [
        [cropped["fx_px"], 0.0, cropped["cx_px"]],
        [0.0, cropped["fy_px"], cropped["cy_px"]],
        [0.0, 0.0, 1.0],
    ]

    scale_x = output_width / crop_width
    scale_y = output_height / crop_height
    model = {
        **cropped,
        "width": output_width,
        "height": output_height,
        "fx_px": float(cropped["fx_px"]) * scale_x,
        "fy_px": float(cropped["fy_px"]) * scale_y,
        "cx_px": float(cropped["cx_px"]) * scale_x,
        "cy_px": float(cropped["cy_px"]) * scale_y,
        "resize_scale_x": scale_x,
        "resize_scale_y": scale_y,
    }
    model["camera_matrix_3x3"] = [
        [model["fx_px"], 0.0, model["cx_px"]],
        [0.0, model["fy_px"], model["cy_px"]],
        [0.0, 0.0, 1.0],
    ]
    return cropped, model


def _device_info(device: Any, rs: Any) -> dict[str, str]:
    result = {}
    for key in ("name", "serial_number", "firmware_version", "product_line", "usb_type_descriptor"):
        info = getattr(rs.camera_info, key, None)
        if info is not None and device.supports(info):
            result[key] = device.get_info(info)
    return result


def _sensor_options(sensor: Any, rs: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = (
        "enable_auto_exposure",
        "exposure",
        "gain",
        "enable_auto_white_balance",
        "white_balance",
        "brightness",
        "contrast",
        "saturation",
        "sharpness",
        "gamma",
        "power_line_frequency",
    )
    for name in names:
        option = getattr(rs.option, name, None)
        if option is None or not sensor.supports(option):
            continue
        entry: dict[str, Any] = {"value": float(sensor.get_option(option))}
        try:
            value_range = sensor.get_option_range(option)
            entry["range"] = {
                "minimum": float(value_range.min),
                "maximum": float(value_range.max),
                "step": float(value_range.step),
                "default": float(value_range.default),
            }
        except RuntimeError:
            pass
        result[name] = entry
    return result


def _model_image_size(config: Any) -> tuple[int, int]:
    metadata_path = Path(config.model.metadata_path).expanduser()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    height, width, channels = metadata["input_signature"]["wrist_rgb"]
    if channels != 3:
        raise ValueError(f"Expected a 3-channel wrist_rgb input, got {channels}.")
    return int(width), int(height)


def _capture_realsense(
    camera_config: BundleCameraConfig,
    capture_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    stream_config = rs.config()
    if camera_config.serial:
        stream_config.enable_device(camera_config.serial)
    stream_config.enable_stream(
        rs.stream.color,
        camera_config.width,
        camera_config.height,
        rs.format.rgb8,
        camera_config.fps,
    )
    profile = pipeline.start(stream_config)
    try:
        for _ in range(max(0, int(camera_config.warmup_frames))):
            pipeline.wait_for_frames()

        frames = []
        timestamps_ms = []
        for _ in range(max(1, capture_frames)):
            frameset = pipeline.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to capture a RealSense color frame.")
            frames.append(np.asanyarray(color_frame.get_data(), dtype=np.uint8).copy())
            timestamps_ms.append(float(color_frame.get_timestamp()))

        frame_stack = np.stack(frames)
        median_rgb = np.median(frame_stack, axis=0).astype(np.uint8)
        last_rgb = frames[-1]
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        raw_intrinsics = _intrinsics_dict(stream.get_intrinsics())
        device = profile.get_device()
        color_sensor = device.first_color_sensor()
        frame_means = frame_stack.reshape(len(frames), -1, 3).mean(axis=1)
        mad = np.abs(frame_stack.astype(np.float32) - median_rgb.astype(np.float32)).mean(axis=(1, 2, 3))
        metadata = {
            "device": _device_info(device, rs),
            "stream": {
                "width": int(stream.width()),
                "height": int(stream.height()),
                "fps": int(stream.fps()),
                "format": str(stream.format()).split(".")[-1],
            },
            "intrinsics_raw": raw_intrinsics,
            "color_sensor_options": _sensor_options(color_sensor, rs),
            "capture": {
                "warmup_frames": int(camera_config.warmup_frames),
                "captured_frames": len(frames),
                "frame_timestamps_ms": timestamps_ms,
                "frame_rgb_mean_mean": frame_means.mean(axis=0).tolist(),
                "frame_rgb_mean_std": frame_means.std(axis=0).tolist(),
                "mean_abs_difference_from_median": float(mad.mean()),
                "max_abs_difference_from_median": float(mad.max()),
            },
        }
        return median_rgb, last_rgb, metadata
    finally:
        pipeline.stop()


def _make_comparison(raw: np.ndarray, cropped: np.ndarray, model: np.ndarray) -> Image.Image:
    panels = [
        ("Raw RGB", Image.fromarray(raw)),
        ("Native crop", Image.fromarray(cropped)),
        ("Policy RGB", Image.fromarray(model).resize((480, 480), Image.Resampling.NEAREST)),
    ]
    margin, header, gap = 16, 34, 14
    panel_width = max(image.width for _, image in panels)
    panel_height = max(image.height for _, image in panels)
    canvas = Image.new(
        "RGB",
        (2 * margin + len(panels) * panel_width + (len(panels) - 1) * gap, panel_height + header + 2 * margin),
        (24, 28, 32),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        x = margin + index * (panel_width + gap)
        draw.text((x, margin), label, fill=(235, 235, 235))
        canvas.paste(image, (x, margin + header))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture real RGB, camera calibration, and Franka initial state without motion."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Bundle deployment config.")
    parser.add_argument("--ip", help="Override robot IP from the config or FRANKA_ROBOT_IP.")
    parser.add_argument("--realtime", choices=("ignore", "enforce"), help="Override realtime mode.")
    parser.add_argument("--frames", type=int, default=10, help="Frames used to form the median reference.")
    parser.add_argument("--skip-gripper", action="store_true", help="Do not connect to the gripper.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "runs"), help="Parent output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_bundle_config(config_path)
    robot_ip = args.ip or os.environ.get("FRANKA_ROBOT_IP") or config.robot_ip
    realtime = args.realtime or config.realtime
    if args.frames < 1:
        raise ValueError("--frames must be at least 1.")

    from franky import Gripper, RealtimeConfig, Robot

    realtime_config = RealtimeConfig.Ignore if realtime == "ignore" else RealtimeConfig.Enforce
    print(f"Connecting read-only to Franka at {robot_ip} ...")
    robot = Robot(robot_ip, realtime_config=realtime_config)
    gripper = None if args.skip_gripper else Gripper(robot_ip)
    robot_before = _robot_snapshot(robot, gripper)

    print(
        f"Capturing D435 RGB: {config.camera.width}x{config.camera.height} "
        f"@ {config.camera.fps} FPS ..."
    )
    median_raw, last_raw, camera_metadata = _capture_realsense(config.camera, args.frames)
    robot_after = _robot_snapshot(robot, gripper)

    raw_height, raw_width = median_raw.shape[:2]
    left, top, right, bottom = _crop_bounds(config.camera, raw_width, raw_height)
    cropped = median_raw[top:bottom, left:right]
    model_width, model_height = _model_image_size(config)
    model_rgb = np.asarray(
        Image.fromarray(cropped).resize((model_width, model_height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ).copy()
    last_cropped = last_raw[top:bottom, left:right]
    last_model_rgb = np.asarray(
        Image.fromarray(last_cropped).resize((model_width, model_height), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    ).copy()

    raw_intrinsics = camera_metadata["intrinsics_raw"]
    cropped_intrinsics, model_intrinsics = _transformed_intrinsics(
        raw_intrinsics,
        left,
        top,
        right - left,
        bottom - top,
        model_width,
        model_height,
    )

    timestamp = datetime.now().astimezone()
    run_dir = Path(args.output_dir).expanduser().resolve() / (
        timestamp.strftime("%Y%m%d_%H%M%S") + "_real_alignment_reference"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(median_raw).save(run_dir / "rgb_raw_median.png")
    Image.fromarray(last_raw).save(run_dir / "rgb_raw_last.png")
    Image.fromarray(cropped).save(run_dir / "rgb_crop_median.png")
    Image.fromarray(model_rgb).save(run_dir / "rgb_model_median.png")
    Image.fromarray(last_model_rgb).save(run_dir / "rgb_model_last.png")
    _make_comparison(median_raw, cropped, model_rgb).save(run_dir / "comparison.png")

    metadata = {
        "schema_version": 1,
        "captured_at": timestamp.isoformat(),
        "read_only": True,
        "config_path": str(config_path),
        "robot_ip": robot_ip,
        "realtime": realtime,
        "robot_state_before_capture": robot_before,
        "robot_state_after_capture": robot_after,
        "robot_drift_during_capture": _robot_drift(robot_before, robot_after),
        "camera": {
            **camera_metadata,
            "configured": {
                "serial": config.camera.serial,
                "width": config.camera.width,
                "height": config.camera.height,
                "fps": config.camera.fps,
                "warmup_frames": config.camera.warmup_frames,
            },
            "crop": {
                "enabled": config.camera.enable_crop,
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": right - left,
                "height": bottom - top,
            },
            "intrinsics_cropped": cropped_intrinsics,
            "intrinsics_model_input": model_intrinsics,
        },
        "model_input": {
            "width": model_width,
            "height": model_height,
            "layout": "HWC",
            "color_order": "RGB",
            "dtype": "uint8",
        },
        "files": {
            "raw_median": "rgb_raw_median.png",
            "raw_last": "rgb_raw_last.png",
            "crop_median": "rgb_crop_median.png",
            "model_median": "rgb_model_median.png",
            "model_last": "rgb_model_last.png",
            "comparison": "comparison.png",
        },
        "notes": {
            "tcp_quaternion_order": "x, y, z, w",
            "camera_to_robot_extrinsics": "Not measured; requires hand-eye calibration.",
            "median_frame": "Pixel-wise median of the captured RGB frames after warmup.",
        },
    }
    metadata_path = run_dir / "alignment_reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    drift = metadata["robot_drift_during_capture"]
    print(f"Saved alignment reference to: {run_dir}")
    print(f"  metadata: {metadata_path}")
    print(f"  policy image: {run_dir / 'rgb_model_median.png'}")
    print(f"  comparison: {run_dir / 'comparison.png'}")
    print(f"  max joint drift: {drift['max_abs_joint_delta_rad']:.6g} rad")
    print(f"  TCP translation drift: {drift['tcp_translation_delta_m']:.6g} m")
    print("No robot or gripper motion command was sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
