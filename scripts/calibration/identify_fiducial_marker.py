#!/usr/bin/env python3
"""Identify an ArUco/AprilTag marker from the RealSense color stream.

This script is read-only: it opens only the camera and never connects to the
robot. It tries the common OpenCV ArUco and AprilTag dictionaries and prints
every marker family/ID that is detected.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.calibration.hand_eye import load_eye_to_hand_config  # noqa: E402

DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs" / "eye_to_hand_d435_215322076207.json"


def _available_dictionaries() -> dict[str, int]:
    names = [
        "DICT_4X4_50",
        "DICT_4X4_100",
        "DICT_4X4_250",
        "DICT_4X4_1000",
        "DICT_5X5_50",
        "DICT_5X5_100",
        "DICT_5X5_250",
        "DICT_5X5_1000",
        "DICT_6X6_50",
        "DICT_6X6_100",
        "DICT_6X6_250",
        "DICT_6X6_1000",
        "DICT_7X7_50",
        "DICT_7X7_100",
        "DICT_7X7_250",
        "DICT_7X7_1000",
        "DICT_ARUCO_ORIGINAL",
        "DICT_APRILTAG_16h5",
        "DICT_APRILTAG_25h9",
        "DICT_APRILTAG_36h10",
        "DICT_APRILTAG_36h11",
    ]
    return {name: getattr(cv2.aruco, name) for name in names if hasattr(cv2.aruco, name)}


def _make_detector(dictionary_id: int) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    parameters = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def _detect(detector: Any, image: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None]:
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)
    return corners, ids


def _camera_matrix(intrinsics: Any) -> np.ndarray:
    return np.asarray(
        [
            [float(intrinsics.fx), 0.0, float(intrinsics.ppx)],
            [0.0, float(intrinsics.fy), float(intrinsics.ppy)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _distortion(intrinsics: Any) -> np.ndarray:
    return np.asarray(intrinsics.coeffs, dtype=np.float64).reshape(-1)


def _marker_object_points(marker_length_m: float) -> np.ndarray:
    half = 0.5 * marker_length_m
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def _solve_pose(
    corners: np.ndarray,
    marker_length_m: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, float]:
    object_points = _marker_object_points(marker_length_m)
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    ok, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok or rvecs is None or tvecs is None:
        raise RuntimeError("solvePnPGeneric returned no pose")

    best_index = 0
    if reprojection_errors is not None and len(reprojection_errors):
        best_index = int(np.argmin(np.asarray(reprojection_errors).reshape(-1)))
    rvec = np.asarray(rvecs[best_index], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvecs[best_index], dtype=np.float64).reshape(3)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec.reshape(3, 1), camera_matrix, distortion)
    error = image_points - projected.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return tvec, rms


def _corner_area(corners: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return float(abs(cv2.contourArea(points)))


def _draw_text(
    image: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (10, 24),
) -> None:
    x, y = origin
    for text, color in lines:
        cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 22


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=0.080,
        help="Measured black outer-square edge length. Default: 0.080 m.",
    )
    parser.add_argument("--detection-scale", type=float, default=2.0)
    parser.add_argument("--print-hz", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means run until q/Ctrl-C")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional directory for detections.csv. Disabled when empty.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.marker_length_m <= 0:
        raise ValueError("--marker-length-m must be positive")
    if args.detection_scale <= 0:
        raise ValueError("--detection-scale must be positive")

    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - hardware dependency
        raise RuntimeError("pyrealsense2 is required for RealSense capture") from exc

    camera_config = load_eye_to_hand_config(Path(args.camera_config).expanduser().resolve()).camera
    dictionaries = _available_dictionaries()
    detectors = {name: _make_detector(dictionary_id) for name, dictionary_id in dictionaries.items()}
    if not detectors:
        raise RuntimeError("This OpenCV build exposes no ArUco/AprilTag dictionaries")

    stream_config = rs.config()
    stream_config.enable_device(camera_config.serial)
    stream_config.enable_stream(
        rs.stream.color,
        camera_config.width,
        camera_config.height,
        rs.format.rgb8,
        camera_config.fps,
    )
    pipeline = rs.pipeline()
    profile = pipeline.start(stream_config)
    video_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = video_stream.get_intrinsics()
    camera_matrix = _camera_matrix(intrinsics)
    distortion = _distortion(intrinsics)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    csv_file = None
    writer = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_file = (output_dir / "detections.csv").open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "time_s",
                "frame",
                "family",
                "id",
                "area_px2",
                "camera_x_m",
                "camera_y_m",
                "camera_z_m",
                "reprojection_rms_px",
            ],
        )
        writer.writeheader()

    print("[identify] camera only; no Franka connection or motion commands")
    print(f"[identify] serial={camera_config.serial}")
    print(f"[identify] marker_length_m={args.marker_length_m:.6f}")
    print(f"[identify] dictionaries={len(detectors)}")
    if output_dir is not None:
        print(f"[identify] output={output_dir}")

    start = time.monotonic()
    last_print = 0.0
    frame_count = 0
    try:
        for _ in range(max(0, camera_config.warmup_frames)):
            pipeline.wait_for_frames()

        while args.max_frames == 0 or frame_count < args.max_frames:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame_count += 1
            rgb = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if args.detection_scale == 1.0:
                detection_gray = gray
            else:
                detection_gray = cv2.resize(
                    gray,
                    None,
                    fx=args.detection_scale,
                    fy=args.detection_scale,
                    interpolation=cv2.INTER_CUBIC,
                )
            display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            detections: list[dict[str, Any]] = []
            for family, detector in detectors.items():
                corners_list, ids = _detect(detector, detection_gray)
                if ids is None:
                    continue
                if args.detection_scale != 1.0:
                    corners_list = [corners / args.detection_scale for corners in corners_list]
                for corners, marker_id in zip(corners_list, ids.reshape(-1)):
                    corners = np.asarray(corners, dtype=np.float64).reshape(1, 4, 2)
                    try:
                        tvec, rms = _solve_pose(
                            corners,
                            args.marker_length_m,
                            camera_matrix,
                            distortion,
                        )
                    except RuntimeError:
                        tvec = np.full(3, np.nan)
                        rms = float("nan")
                    detections.append(
                        {
                            "family": family,
                            "id": int(marker_id),
                            "corners": corners,
                            "area": _corner_area(corners),
                            "tvec": tvec,
                            "rms": rms,
                        }
                    )

            detections.sort(key=lambda item: item["area"], reverse=True)
            now = time.monotonic()
            fps = frame_count / max(now - start, 1e-9)
            lines: list[tuple[str, tuple[int, int, int]]] = [
                (f"fiducial auto-id | {fps:.1f} FPS | L={args.marker_length_m * 1000:.1f}mm", (255, 255, 255))
            ]
            if detections:
                for index, detection in enumerate(detections[:5]):
                    color = (0, 255, 0) if index == 0 else (0, 200, 255)
                    corners_i = np.rint(detection["corners"].reshape(4, 2)).astype(np.int32)
                    cv2.polylines(display, [corners_i], True, color, 2, cv2.LINE_AA)
                    tvec = np.asarray(detection["tvec"], dtype=np.float64)
                    lines.append(
                        (
                            f"{detection['family']} id={detection['id']} "
                            f"z={tvec[2] * 1000:+.1f}mm rms={detection['rms']:.3f}",
                            color,
                        )
                    )
                    if writer is not None:
                        writer.writerow(
                            {
                                "time_s": now - start,
                                "frame": frame_count,
                                "family": detection["family"],
                                "id": detection["id"],
                                "area_px2": detection["area"],
                                "camera_x_m": float(tvec[0]),
                                "camera_y_m": float(tvec[1]),
                                "camera_z_m": float(tvec[2]),
                                "reprojection_rms_px": float(detection["rms"]),
                            }
                        )
                if now - last_print >= 1.0 / max(args.print_hz, 1e-9):
                    last_print = now
                    best = detections[0]
                    tvec = np.asarray(best["tvec"], dtype=np.float64)
                    print(
                        f"frame={frame_count} family={best['family']} id={best['id']} "
                        f"area={best['area']:.1f}px2 "
                        f"camera_p_marker=[{tvec[0]:+.4f}, {tvec[1]:+.4f}, {tvec[2]:+.4f}]m "
                        f"rms={best['rms']:.3f}px"
                    )
            else:
                lines.append(("NO MARKER FOUND", (0, 0, 255)))

            _draw_text(display, lines)
            if not args.no_window:
                cv2.imshow("Fiducial marker auto-id", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        pipeline.stop()
        if csv_file is not None:
            csv_file.close()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
