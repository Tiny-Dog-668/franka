#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.gelsight_devices import (
    GelSightDeviceSpec as CameraSpec,
    discover_gelsight_cameras,
)

PREVIEW_SIZE = (640, 480)
WINDOW_NAME = "GelSight Multi Preview"
SENSOR_WIDTH = 3280
SENSOR_HEIGHT = 2464
SENSOR_FPS = 25

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview one or two GelSight Mini cameras.")
    parser.add_argument(
        "--cams",
        type=int,
        nargs="+",
        metavar="CAM",
        help="Optional manual camera IDs, for example: --cams 4 or --cams 4 6",
    )
    parser.add_argument(
        "--save-dir",
        default=".",
        help="Directory where snapshot images will be saved.",
    )
    return parser

def open_camera(cam_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] 无法打开相机 /dev/video{cam_id}")

    # GelSight Mini exposes one native capture mode: MJPEG 3280x2464 @ 25 FPS.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SENSOR_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SENSOR_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, SENSOR_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] 成功打开相机 /dev/video{cam_id} -> {width}x{height} @ {fps:.2f} FPS")
    if (width, height) != (SENSOR_WIDTH, SENSOR_HEIGHT):
        print(
            f"[WARN] /dev/video{cam_id} 未使用预期分辨率 "
            f"{SENSOR_WIDTH}x{SENSOR_HEIGHT}"
        )
    return cap


def _annotate_preview(preview: np.ndarray, label: str) -> np.ndarray:
    canvas = preview.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        label,
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _make_placeholder(label: str) -> np.ndarray:
    image = np.zeros((PREVIEW_SIZE[1], PREVIEW_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(
        image,
        label,
        (10, PREVIEW_SIZE[1] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def _build_grid(previews: list[np.ndarray]) -> np.ndarray:
    cols = min(2, len(previews))
    rows = math.ceil(len(previews) / cols)
    blank = np.zeros((PREVIEW_SIZE[1], PREVIEW_SIZE[0], 3), dtype=np.uint8)

    row_images: list[np.ndarray] = []
    for row_idx in range(rows):
        row_previews = previews[row_idx * cols : (row_idx + 1) * cols]
        if len(row_previews) < cols:
            row_previews = row_previews + [blank.copy() for _ in range(cols - len(row_previews))]
        row_images.append(np.hstack(row_previews))
    return np.vstack(row_images)


def main() -> int:
    args = build_parser().parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.cams:
        camera_specs = [CameraSpec(cam_id=cam_id, label=f"/dev/video{cam_id}") for cam_id in args.cams]
        print(f"[INFO] 使用手动指定的相机列表: {' '.join(str(spec.cam_id) for spec in camera_specs)}")
    else:
        camera_specs = discover_gelsight_cameras()
        if not camera_specs:
            raise RuntimeError("[ERROR] 没有自动发现任何 GelSight 图像流。")
        print(f"[INFO] 自动发现 {len(camera_specs)} 个 GelSight 图像流。")
        for spec in camera_specs:
            print(f"[INFO]   /dev/video{spec.cam_id} -> {spec.label}")

    if len(camera_specs) not in (1, 2):
        discovered = " ".join(f"/dev/video{spec.cam_id}" for spec in camera_specs) or "无"
        raise RuntimeError(
            f"[ERROR] 需要 1 或 2 个 GelSight 图像流，当前为 {len(camera_specs)} 个: {discovered}"
        )

    caps: list[tuple[CameraSpec, cv2.VideoCapture]] = []
    try:
        for spec in camera_specs:
            caps.append((spec, open_camera(spec.cam_id)))
    except Exception:
        for _, cap in caps:
            cap.release()
        raise

    print(f"[INFO] 正在预览相机: {' '.join(f'/dev/video{spec.cam_id}' for spec in camera_specs)}")
    print(f"[INFO] 单路预览尺寸: {PREVIEW_SIZE[0]}x{PREVIEW_SIZE[1]}")
    print("[INFO] 按 ESC 退出，按 s 保存当前拼图预览")

    last_time = time.time()
    frame_count = 0

    try:
        while True:
            previews: list[np.ndarray] = []
            frame_count += 1

            # Trigger all UVC devices before decoding to reduce inter-camera skew.
            grabbed = [cap.grab() for _, cap in caps]
            for (spec, cap), was_grabbed in zip(caps, grabbed):
                ret, frame = cap.retrieve() if was_grabbed else (False, None)
                if not ret or frame is None:
                    print(f"[WARN] /dev/video{spec.cam_id} 读取失败，显示占位图")
                    previews.append(_make_placeholder(f"/dev/video{spec.cam_id} read failed"))
                    continue

                preview = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
                previews.append(_annotate_preview(preview, f"/dev/video{spec.cam_id}"))

            grid = _build_grid(previews)

            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                print(f"[INFO] Mosaic FPS: {fps:.2f}")
                frame_count = 0
                last_time = now

            cv2.imshow(WINDOW_NAME, grid)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[INFO] 退出")
                break
            if key == ord("s"):
                filename = save_dir / f"gelsight_mosaic_{int(time.time())}.png"
                cv2.imwrite(str(filename), grid)
                print(f"[INFO] 已保存: {filename}")
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
