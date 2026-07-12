#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import numpy as np
    from PIL import Image, ImageDraw
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "dependency")
    print(
        f"Missing Python package: {missing}\n"
        "Please run this script from the project virtual environment:\n"
        f"  source {REPO_ROOT / '.venv' / 'bin' / 'activate'}",
        file=sys.stderr,
    )
    raise SystemExit(1)

from franka_sim2real.e2e_bundle import BundleCameraConfig, _apply_camera_crop, load_bundle_config


class RawRealSenseCamera:
    def __init__(self, camera_config: BundleCameraConfig) -> None:
        import pyrealsense2 as rs

        self.camera_config = camera_config
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

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture color frame from RealSense.")
        return np.asanyarray(color_frame.get_data(), dtype=np.uint8)

    def close(self) -> None:
        self.pipeline.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one RealSense frame and save raw/cropped comparison images."
    )
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "e2e_bundle_real_example.json"),
        help="Path to the bundle deployment JSON config.",
    )
    parser.add_argument(
        "--image",
        help="Optional static image path. If omitted, the script always captures from RealSense.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs"),
        help="Directory where preview outputs will be saved.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the comparison image with xdg-open after saving.",
    )
    return parser


def _load_raw_image(camera_config: BundleCameraConfig) -> np.ndarray:
    if camera_config.source == "image":
        if not camera_config.image_path:
            raise ValueError("camera.image_path must be set when source='image'")
        return np.asarray(Image.open(camera_config.image_path).convert("RGB"), dtype=np.uint8)

    if camera_config.source == "realsense":
        camera = RawRealSenseCamera(camera_config)
        try:
            return camera.read()
        finally:
            camera.close()

    raise ValueError(f"Unsupported camera source: {camera_config.source}")


def _make_comparison(raw_rgb: np.ndarray, cropped_rgb: np.ndarray) -> Image.Image:
    raw_image = Image.fromarray(raw_rgb)
    cropped_image = Image.fromarray(cropped_rgb)

    panel_width = max(raw_image.width, cropped_image.width)
    panel_height = max(raw_image.height, cropped_image.height)
    header_height = 36
    gap = 16
    canvas = Image.new("RGB", (panel_width * 2 + gap * 3, panel_height + header_height + gap * 2), (24, 28, 32))
    draw = ImageDraw.Draw(canvas)

    raw_x = gap
    crop_x = gap * 2 + panel_width
    image_y = gap + header_height

    draw.text((raw_x, gap), "Raw frame", fill=(230, 230, 230))
    draw.text((crop_x, gap), "Cropped frame", fill=(230, 230, 230))

    canvas.paste(raw_image, (raw_x, image_y))
    canvas.paste(cropped_image, (crop_x, image_y))
    return canvas


def main() -> int:
    args = build_parser().parse_args()
    config = load_bundle_config(args.config)
    camera_config = config.camera

    if args.image:
        camera_config.source = "image"
        camera_config.image_path = args.image
    else:
        camera_config.source = "realsense"
        camera_config.image_path = None

    raw_rgb = _load_raw_image(camera_config)
    cropped_rgb = _apply_camera_crop(raw_rgb, camera_config)

    run_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_camera_crop_preview"
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw.png"
    cropped_path = run_dir / "cropped.png"
    comparison_path = run_dir / "comparison.png"
    metadata_path = run_dir / "crop_metadata.json"

    Image.fromarray(raw_rgb).save(raw_path)
    Image.fromarray(cropped_rgb).save(cropped_path)
    _make_comparison(raw_rgb, cropped_rgb).save(comparison_path)

    metadata = {
        "camera_source": camera_config.source,
        "image_path": camera_config.image_path,
        "enable_crop": camera_config.enable_crop,
        "source_width": int(raw_rgb.shape[1]),
        "source_height": int(raw_rgb.shape[0]),
        "output_width": int(cropped_rgb.shape[1]),
        "output_height": int(cropped_rgb.shape[0]),
        "crop_left": camera_config.crop_left,
        "crop_top": camera_config.crop_top,
        "crop_width": camera_config.crop_width,
        "crop_height": camera_config.crop_height,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved raw frame to: {raw_path}")
    print(f"Saved cropped frame to: {cropped_path}")
    print(f"Saved comparison image to: {comparison_path}")
    print(f"Saved crop metadata to: {metadata_path}")

    if args.open:
        subprocess.Popen(["xdg-open", str(comparison_path)])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
