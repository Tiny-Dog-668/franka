#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture D435 RGB samples and save fixed-crop preprocessing outputs."
    )
    parser.add_argument("--serial", default="215322076207")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--left", type=int, default=100)
    parser.add_argument("--top", type=int, default=34)
    parser.add_argument("--crop-width", type=int, default=400)
    parser.add_argument("--crop-height", type=int, default=398)
    parser.add_argument("--output-width", type=int, default=224)
    parser.add_argument("--output-height", type=int, default=224)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts"),
        help="Parent directory for the timestamped capture directory.",
    )
    return parser


def transformed_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    left: int,
    top: int,
    crop_width: int,
    crop_height: int,
    output_width: int,
    output_height: int,
) -> dict[str, object]:
    scale_x = output_width / crop_width
    scale_y = output_height / crop_height
    crop_cx = cx - left
    crop_cy = cy - top
    return {
        "cropped": {
            "width": crop_width,
            "height": crop_height,
            "fx_px": fx,
            "fy_px": fy,
            "cx_px": crop_cx,
            "cy_px": crop_cy,
        },
        "model_input": {
            "width": output_width,
            "height": output_height,
            "fx_px": fx * scale_x,
            "fy_px": fy * scale_y,
            "cx_px": crop_cx * scale_x,
            "cy_px": crop_cy * scale_y,
            "resize_scale_x": scale_x,
            "resize_scale_y": scale_y,
        },
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.interval < 0:
        raise ValueError("--interval must be non-negative")

    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    profile = pipeline.start(config)

    timestamp = datetime.now().astimezone()
    run_dir = (
        Path(args.output_dir).expanduser().resolve()
        / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_d435_crop_samples"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    try:
        for _ in range(max(0, args.warmup_frames)):
            pipeline.wait_for_frames()

        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        right = args.left + args.crop_width
        bottom = args.top + args.crop_height
        if args.left < 0 or args.top < 0 or right > 640 or bottom > 480:
            raise ValueError(
                f"Crop [{args.left}:{right}, {args.top}:{bottom}] exceeds 640x480 frame."
            )

        samples: list[dict[str, object]] = []
        for index in range(1, args.count + 1):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError("Failed to capture D435 color frame.")

            raw = np.asanyarray(color_frame.get_data(), dtype=np.uint8).copy()
            crop = raw[args.top:bottom, args.left:right]
            model = np.asarray(
                Image.fromarray(crop).resize(
                    (args.output_width, args.output_height),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )

            raw_image = Image.fromarray(raw)
            preview = raw_image.copy()
            ImageDraw.Draw(preview).rectangle(
                (args.left, args.top, right - 1, bottom - 1),
                outline=(255, 0, 0),
                width=3,
            )

            prefix = f"sample_{index:02d}"
            files = {
                "raw": f"{prefix}_raw_640x480.png",
                "crop_preview": f"{prefix}_crop_preview.png",
                "crop": f"{prefix}_crop_{args.crop_width}x{args.crop_height}.png",
                "model_input": (
                    f"{prefix}_model_{args.output_width}x{args.output_height}.png"
                ),
            }
            raw_image.save(run_dir / files["raw"])
            preview.save(run_dir / files["crop_preview"])
            Image.fromarray(crop).save(run_dir / files["crop"])
            Image.fromarray(model).save(run_dir / files["model_input"])
            samples.append(
                {
                    "index": index,
                    "frame_number": int(color_frame.get_frame_number()),
                    "timestamp_ms": float(color_frame.get_timestamp()),
                    "files": files,
                }
            )
            print(f"Captured sample {index}/{args.count}: {prefix}")
            if index < args.count and args.interval:
                time.sleep(args.interval)

        raw_intrinsics = {
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "fx_px": float(intrinsics.fx),
            "fy_px": float(intrinsics.fy),
            "cx_px": float(intrinsics.ppx),
            "cy_px": float(intrinsics.ppy),
            "distortion_model": str(intrinsics.model).split(".")[-1],
            "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
        }
        metadata = {
            "captured_at": timestamp.isoformat(),
            "camera": {
                "model": "Intel RealSense D435",
                "serial": args.serial,
                "stream": {"width": 640, "height": 480, "fps": 30, "format": "RGB8"},
            },
            "crop": {
                "left": args.left,
                "top": args.top,
                "right_exclusive": right,
                "bottom_exclusive": bottom,
                "width": args.crop_width,
                "height": args.crop_height,
            },
            "resize": {
                "width": args.output_width,
                "height": args.output_height,
                "method": "PIL bilinear",
            },
            "intrinsics_raw": raw_intrinsics,
            "intrinsics_transformed": transformed_intrinsics(
                float(intrinsics.fx),
                float(intrinsics.fy),
                float(intrinsics.ppx),
                float(intrinsics.ppy),
                args.left,
                args.top,
                args.crop_width,
                args.crop_height,
                args.output_width,
                args.output_height,
            ),
            "samples": samples,
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    finally:
        pipeline.stop()

    print(f"Saved capture set to: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
