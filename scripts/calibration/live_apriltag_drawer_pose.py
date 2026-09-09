#!/usr/bin/env python3
"""Estimate drawer key points in the Franka base frame from an AprilTag.

The AprilTag is assumed to be centered in a square paper backing whose top-left
corner is aligned with the drawer's top-left corner.  At zero drawer rotation,
tag +X follows drawer length to the right, tag -Y follows drawer width from the
top edge toward the bottom (image-bottom) edge, and tag -Z points inward.

By default, the four possible tag-aligned orientations are checked in the
camera image so drawer length points right and drawer width points down.  The
drawer itself may extend outside the camera frame; only the tag must be visible.

This program is read-only: it opens only the RealSense color stream and never
connects to or commands the robot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, CALIBRATION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from franka_sim2real.calibration.hand_eye import load_eye_to_hand_config  # noqa: E402
from live_apriltag_cube_pose import (  # noqa: E402
    APRILTAG_FAMILIES,
    DEFAULT_CALIBRATION_REPORT,
    DEFAULT_CAMERA_CONFIG,
    _camera_matrix,
    _configure_color_controls,
    _distortion,
    _draw_text,
    _load_accepted_base_t_camera,
    _validate_color_control_args,
    _xyz_text,
    solve_tag_pose,
)


POINT_NAMES = ("top_center", "bottom_edge_center", "volume_center", "bottom_surface_center")
POINT_COLORS = {
    "top_center": (0, 255, 0),
    "bottom_edge_center": (0, 200, 255),
    "volume_center": (255, 128, 0),
    "bottom_surface_center": (255, 0, 255),
}


def drawer_points_in_tag(
    *,
    marker_paper_size_m: float,
    drawer_width_m: float,
    drawer_length_m: float,
    drawer_height_m: float,
    drawer_rotation_deg: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return drawer reference points expressed in the AprilTag frame.

    ``marker_paper_size_m`` is the outer size of the square white backing.  Its
    top-left corner is assumed flush with the drawer's top-left corner and the
    black AprilTag is assumed centered in that backing.
    """

    dimensions = {
        "marker_paper_size_m": marker_paper_size_m,
        "drawer_width_m": drawer_width_m,
        "drawer_length_m": drawer_length_m,
        "drawer_height_m": drawer_height_m,
    }
    for name, value in dimensions.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
    if not math.isfinite(drawer_rotation_deg):
        raise ValueError("drawer_rotation_deg must be finite")

    angle = math.radians(drawer_rotation_deg)
    # Zero rotation follows the printed tag axes: length=+X and width=-Y.
    length_direction = np.asarray([math.cos(angle), math.sin(angle), 0.0])
    width_direction = np.asarray([math.sin(angle), -math.cos(angle), 0.0])
    inward_direction = np.asarray([0.0, 0.0, -1.0])

    # The tag is centered in its square white backing.  Walking half a backing
    # width along both drawer axes goes from the drawer corner to tag center.
    drawer_top_left = -0.5 * marker_paper_size_m * (
        width_direction + length_direction
    )
    top_center = (
        drawer_top_left
        + 0.5 * drawer_length_m * length_direction
        + 0.5 * drawer_width_m * width_direction
    )
    bottom_edge_center = (
        drawer_top_left
        + 0.5 * drawer_length_m * length_direction
        + drawer_width_m * width_direction
    )

    return {
        "top_left": drawer_top_left,
        "top_right": drawer_top_left + drawer_length_m * length_direction,
        "bottom_right": (
            drawer_top_left
            + drawer_length_m * length_direction
            + drawer_width_m * width_direction
        ),
        "bottom_left": drawer_top_left + drawer_width_m * width_direction,
        "top_center": top_center,
        "bottom_edge_center": bottom_edge_center,
        "volume_center": top_center + 0.5 * drawer_height_m * inward_direction,
        "bottom_surface_center": top_center + drawer_height_m * inward_direction,
    }


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {transform.shape}")
    return (transform @ np.concatenate([point, [1.0]]))[:3]


def project_tag_points(
    points: list[np.ndarray],
    camera_t_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(camera_t_tag[:3, :3])
    projected, _ = cv2.projectPoints(
        np.asarray(points, dtype=np.float64),
        rotation_vector,
        camera_t_tag[:3, 3].reshape(3, 1),
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def image_aligned_drawer_rotation_deg(
    camera_t_tag: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    """Choose a tag-aligned rotation whose length points right and width down.

    The tag backing is square, so its physical placement can differ from the
    canonical decoded AprilTag orientation by 0, 90, 180, or -90 degrees.
    Selecting in image space avoids requiring the user to know that canonical
    orientation and does not require any drawer boundary to be visible.
    """

    # Use a symmetric local derivative around the tag center.  Projecting a
    # one-metre virtual axis is incorrect for this perspective view: it can
    # cross a vanishing point and reverse the apparent axis direction when the
    # tag lies obliquely on the table.
    epsilon_m = 0.010
    basis_pixels = project_tag_points(
        [
            np.asarray([-epsilon_m, 0.0, 0.0]),
            np.asarray([+epsilon_m, 0.0, 0.0]),
            np.asarray([0.0, -epsilon_m, 0.0]),
            np.asarray([0.0, +epsilon_m, 0.0]),
        ],
        camera_t_tag,
        camera_matrix,
        distortion,
    )
    tag_x_pixels = (basis_pixels[1] - basis_pixels[0]) / (2.0 * epsilon_m)
    tag_y_pixels = (basis_pixels[3] - basis_pixels[2]) / (2.0 * epsilon_m)

    def image_vector(tag_vector: np.ndarray) -> np.ndarray:
        return tag_vector[0] * tag_x_pixels + tag_vector[1] * tag_y_pixels

    best_rotation = 0.0
    best_score = -math.inf
    for rotation_deg in (0.0, 90.0, 180.0, -90.0):
        angle = math.radians(rotation_deg)
        length_tag = np.asarray([math.cos(angle), math.sin(angle)])
        width_tag = np.asarray([math.sin(angle), -math.cos(angle)])
        length_pixels = image_vector(length_tag)
        width_pixels = image_vector(width_tag)
        length_norm = float(np.linalg.norm(length_pixels))
        width_norm = float(np.linalg.norm(width_pixels))
        if width_norm <= 1e-12 or length_norm <= 1e-12:
            continue
        # Image +X is right and image +Y is down.
        score = float(length_pixels[0] / length_norm + width_pixels[1] / width_norm)
        if score > best_score:
            best_score = score
            best_rotation = rotation_deg
    if not math.isfinite(best_score):
        raise RuntimeError("Could not determine drawer orientation from the tag projection")
    return best_rotation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--calibration-report", default=str(DEFAULT_CALIBRATION_REPORT))
    parser.add_argument("--family", choices=sorted(APRILTAG_FAMILIES), default="tag36h11")
    parser.add_argument("--id", type=int, default=1, dest="marker_id")
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=0.048,
        help="Outer black-square edge length; default: 0.048 m",
    )
    parser.add_argument(
        "--marker-paper-size-m",
        type=float,
        default=0.060,
        help="Outer white square size aligned to the drawer corner; default: 0.060 m",
    )
    parser.add_argument("--drawer-width-m", type=float, default=0.300)
    parser.add_argument("--drawer-length-m", type=float, default=0.250)
    parser.add_argument("--drawer-height-m", type=float, default=0.130)
    parser.add_argument(
        "--drawer-orientation",
        choices=("image", "tag"),
        default="image",
        help=(
            "image: automatically make length point right and width point down; "
            "tag: use the decoded tag axes directly"
        ),
    )
    parser.add_argument(
        "--drawer-rotation-deg",
        type=float,
        default=0.0,
        help="Extra in-plane rotation after automatic orientation; default: 0",
    )
    parser.add_argument("--smoothing-window", type=int, default=15)
    parser.add_argument("--detection-scale", type=float, default=2.0)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Ctrl-C")
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--gain", type=float)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.marker_length_m) or args.marker_length_m <= 0.0:
        raise ValueError("--marker-length-m must be positive and finite")
    if args.marker_id < 0:
        raise ValueError("--id must be non-negative")
    if args.smoothing_window <= 0:
        raise ValueError("--smoothing-window must be positive")
    if not math.isfinite(args.detection_scale) or not 1.0 <= args.detection_scale <= 4.0:
        raise ValueError("--detection-scale must be finite and in [1, 4]")
    if args.max_reprojection_px <= 0.0 or args.print_hz <= 0.0 or args.max_frames < 0:
        raise ValueError("reprojection threshold and print rate must be positive; max-frames >= 0")
    _validate_color_control_args(args.auto_exposure, args.exposure, args.gain)


def _output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_live_apriltag_drawer_pose"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    return output_dir, snapshots_dir, output_dir / "detections.csv", output_dir / "summary.json"


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    # Validate all physical dimensions before opening the camera.  In image
    # orientation mode the effective rotation is selected for every accepted
    # tag pose below.
    drawer_points_in_tag(
        marker_paper_size_m=args.marker_paper_size_m,
        drawer_width_m=args.drawer_width_m,
        drawer_length_m=args.drawer_length_m,
        drawer_height_m=args.drawer_height_m,
        drawer_rotation_deg=args.drawer_rotation_deg,
    )
    base_t_camera = _load_accepted_base_t_camera(
        Path(args.calibration_report).expanduser().resolve()
    )
    camera_config = load_eye_to_hand_config(
        Path(args.camera_config).expanduser().resolve()
    ).camera
    output_dir, snapshots_dir, csv_path, summary_path = _output_paths(args)

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    stream_config = rs.config()
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
        color_controls = _configure_color_controls(
            rs,
            profile.get_device().first_color_sensor(),
            auto_exposure=args.auto_exposure,
            exposure=args.exposure,
            gain=args.gain,
        )
    except BaseException:
        pipeline.stop()
        raise

    video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video_stream.get_intrinsics()
    camera_matrix = _camera_matrix(intrinsics)
    distortion = _distortion(intrinsics)

    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[args.family])
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    buffers = {
        name: deque(maxlen=args.smoothing_window) for name in POINT_NAMES
    }
    all_positions: dict[str, list[np.ndarray]] = {name: [] for name in POINT_NAMES}
    frame_count = 0
    accepted_count = 0
    rejected_count = 0
    selected_rotations: list[float] = []
    start_time = time.monotonic()
    previous_print_time = 0.0
    latest_display: np.ndarray | None = None

    fieldnames = [
        "host_time_s",
        "camera_time_ms",
        "frame_index",
        "reprojection_rms_px",
        "drawer_rotation_deg",
        "base_tag_x_m",
        "base_tag_y_m",
        "base_tag_z_m",
    ]
    for name in POINT_NAMES:
        fieldnames.extend(
            [f"base_{name}_x_m", f"base_{name}_y_m", f"base_{name}_z_m"]
        )

    print("[drawer] camera only; no Franka connection or motion commands")
    print(
        f"[drawer] family={args.family} id={args.marker_id} "
        f"black_marker={args.marker_length_m:.3f} m paper={args.marker_paper_size_m:.3f} m"
    )
    print(
        f"[drawer] width={args.drawer_width_m:.3f} m length={args.drawer_length_m:.3f} m "
        f"height={args.drawer_height_m:.3f} m rotation={args.drawer_rotation_deg:.1f} deg"
    )
    if args.drawer_orientation == "image":
        print(
            "[drawer] orientation=image: length extends right and width extends down "
            "from the tagged top-left corner"
        )
    else:
        print("[drawer] orientation=tag: length=tag +X, width=tag -Y, inward=tag -Z")
    print(f"[drawer] calibration={Path(args.calibration_report).expanduser().resolve()}")
    print(
        "[drawer] color controls: "
        f"auto_exposure={color_controls['auto_exposure']}, "
        f"exposure={color_controls['exposure']}, gain={color_controls['gain']}"
    )
    print(f"[drawer] output={output_dir}")
    print("[drawer] keys: q/Esc quit, s snapshot, c clear smoothing")

    try:
        for _ in range(max(0, camera_config.warmup_frames)):
            pipeline.wait_for_frames()
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            while args.max_frames == 0 or frame_count < args.max_frames:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_count += 1
                rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
                gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                detection_gray = (
                    gray
                    if args.detection_scale == 1.0
                    else cv2.resize(
                        gray,
                        None,
                        fx=args.detection_scale,
                        fy=args.detection_scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
                corners, ids, _ = detector.detectMarkers(detection_gray)
                if args.detection_scale != 1.0:
                    corners = [corner / args.detection_scale for corner in corners]

                fps = frame_count / max(time.monotonic() - start_time, 1e-9)
                lines: list[tuple[str, tuple[int, int, int]]] = [
                    (f"Drawer from AprilTag {args.family} ID {args.marker_id} | {fps:.1f} FPS", (255, 255, 255))
                ]
                selected_index: int | None = None
                if ids is not None:
                    matches = np.flatnonzero(ids.reshape(-1) == args.marker_id)
                    if len(matches):
                        selected_index = int(matches[0])

                if selected_index is None:
                    lines.append(("TAG NOT FOUND", (0, 0, 255)))
                else:
                    selected_corners = corners[selected_index]
                    cv2.polylines(
                        display,
                        [np.rint(selected_corners.reshape(4, 2)).astype(np.int32)],
                        True,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    try:
                        camera_t_tag, rvec, reprojection = solve_tag_pose(
                            selected_corners,
                            camera_matrix,
                            distortion,
                            args.marker_length_m,
                        )
                    except RuntimeError as exc:
                        lines.append((f"POSE FAILED: {exc}", (0, 0, 255)))
                    else:
                        automatic_rotation = (
                            image_aligned_drawer_rotation_deg(
                                camera_t_tag,
                                camera_matrix,
                                distortion,
                            )
                            if args.drawer_orientation == "image"
                            else 0.0
                        )
                        effective_rotation = automatic_rotation + args.drawer_rotation_deg
                        drawer_points_tag = drawer_points_in_tag(
                            marker_paper_size_m=args.marker_paper_size_m,
                            drawer_width_m=args.drawer_width_m,
                            drawer_length_m=args.drawer_length_m,
                            drawer_height_m=args.drawer_height_m,
                            drawer_rotation_deg=effective_rotation,
                        )
                        cv2.drawFrameAxes(
                            display,
                            camera_matrix,
                            distortion,
                            rvec,
                            camera_t_tag[:3, 3].reshape(3, 1),
                            0.050,
                            2,
                        )
                        quality_ok = reprojection <= args.max_reprojection_px
                        quality_color = (0, 255, 0) if quality_ok else (0, 165, 255)
                        lines.append((f"reprojection RMS: {reprojection:.3f} px", quality_color))
                        lines.append(
                            (
                                f"drawer: length right / width down | rotation={effective_rotation:+.0f} deg",
                                (255, 255, 0),
                            )
                        )

                        outline_names = ("top_left", "top_right", "bottom_right", "bottom_left")
                        outline_pixels = project_tag_points(
                            [drawer_points_tag[name] for name in outline_names],
                            camera_t_tag,
                            camera_matrix,
                            distortion,
                        )
                        if np.all(np.isfinite(outline_pixels)):
                            cv2.polylines(
                                display,
                                [np.rint(outline_pixels).astype(np.int32)],
                                True,
                                (255, 255, 0),
                                2,
                                cv2.LINE_AA,
                            )

                        point_pixels = project_tag_points(
                            [drawer_points_tag[name] for name in POINT_NAMES],
                            camera_t_tag,
                            camera_matrix,
                            distortion,
                        )
                        for name, pixel in zip(POINT_NAMES, point_pixels):
                            if np.all(np.isfinite(pixel)):
                                xy = tuple(np.rint(pixel).astype(int))
                                cv2.circle(display, xy, 5, POINT_COLORS[name], -1, cv2.LINE_AA)
                                cv2.putText(
                                    display,
                                    name,
                                    (xy[0] + 7, xy[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.42,
                                    POINT_COLORS[name],
                                    1,
                                    cv2.LINE_AA,
                                )

                        if quality_ok:
                            accepted_count += 1
                            selected_rotations.append(effective_rotation)
                            base_t_tag = base_t_camera @ camera_t_tag
                            base_tag = base_t_tag[:3, 3]
                            base_positions = {
                                name: transform_point(base_t_tag, drawer_points_tag[name])
                                for name in POINT_NAMES
                            }
                            row: dict[str, Any] = {
                                "host_time_s": time.time(),
                                "camera_time_ms": float(color_frame.get_timestamp()),
                                "frame_index": frame_count,
                                "reprojection_rms_px": reprojection,
                                "drawer_rotation_deg": effective_rotation,
                                "base_tag_x_m": float(base_tag[0]),
                                "base_tag_y_m": float(base_tag[1]),
                                "base_tag_z_m": float(base_tag[2]),
                            }
                            medians: dict[str, np.ndarray] = {}
                            for name, position in base_positions.items():
                                buffers[name].append(position.copy())
                                all_positions[name].append(position.copy())
                                median = np.median(np.asarray(buffers[name]), axis=0)
                                medians[name] = median
                                lines.append((_xyz_text(f"base_{name}", median), POINT_COLORS[name]))
                                for axis, value in zip("xyz", position):
                                    row[f"base_{name}_{axis}_m"] = float(value)
                            writer.writerow(row)

                            now = time.monotonic()
                            if now - previous_print_time >= 1.0 / args.print_hz:
                                previous_print_time = now
                                values = " ".join(
                                    f"{name}=[{median[0]:+.4f},{median[1]:+.4f},{median[2]:+.4f}]m"
                                    for name, median in medians.items()
                                )
                                print(
                                    f"frame={frame_count} rms={reprojection:.3f}px {values}"
                                )
                        else:
                            rejected_count += 1
                            lines.append(("POSE REJECTED BY REPROJECTION LIMIT", (0, 165, 255)))

                _draw_text(display, lines)
                latest_display = display
                if not args.headless:
                    cv2.imshow("AprilTag drawer pose (camera only)", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("c"):
                        for buffer in buffers.values():
                            buffer.clear()
                        print("[drawer] smoothing buffers cleared")
                    if key == ord("s") and latest_display is not None:
                        snapshot = snapshots_dir / f"drawer_{frame_count:06d}.png"
                        cv2.imwrite(str(snapshot), latest_display)
                        print(f"[drawer] snapshot={snapshot}")
    except KeyboardInterrupt:
        print("\n[drawer] interrupted")
    finally:
        pipeline.stop()
        if not args.headless:
            cv2.destroyAllWindows()

    statistics: dict[str, Any] = {}
    for name, positions in all_positions.items():
        if positions:
            values = np.asarray(positions)
            statistics[name] = {
                "count": len(positions),
                "mean_m": np.mean(values, axis=0).tolist(),
                "median_m": np.median(values, axis=0).tolist(),
                "std_m": np.std(values, axis=0).tolist(),
            }
        else:
            statistics[name] = {"count": 0}
    summary = {
        "kind": "live_apriltag_drawer_pose",
        "family": args.family,
        "id": args.marker_id,
        "marker_length_m": args.marker_length_m,
        "marker_paper_size_m": args.marker_paper_size_m,
        "drawer_width_m": args.drawer_width_m,
        "drawer_length_m": args.drawer_length_m,
        "drawer_height_m": args.drawer_height_m,
        "drawer_orientation": args.drawer_orientation,
        "drawer_rotation_deg": args.drawer_rotation_deg,
        "selected_rotation_deg": (
            float(np.median(selected_rotations)) if selected_rotations else None
        ),
        "coordinate_assumption": {
            "length": "toward image right in image orientation mode",
            "width": "toward image down in image orientation mode",
            "inward": "tag -Z",
            "paper_alignment": "paper top-left corner flush with drawer top-left corner",
            "visibility": "only the AprilTag must be visible; drawer points are dimension-based extrapolations",
        },
        "base_T_camera": base_t_camera.tolist(),
        "accepted_detection_count": accepted_count,
        "rejected_reprojection_count": rejected_count,
        "statistics": statistics,
        "detections_csv": str(csv_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[drawer] accepted={accepted_count} rejected={rejected_count}")
    print(f"[drawer] detections={csv_path}")
    print(f"[drawer] summary={summary_path}")
    return 0 if accepted_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
