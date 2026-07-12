#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a recorded step_data .npz file."
    )
    parser.add_argument("path", help="Path to step_data/step_XXXX.npz")
    parser.add_argument(
        "--save-image",
        help="Optional output path for saving wrist_rgb as a PNG image.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full array values for non-image arrays.",
    )
    return parser


def _format_preview(array: np.ndarray, full: bool) -> str:
    if array.ndim >= 2 and array.size > 64 and not full:
        return f"<omitted; use --full to print {array.shape}>"
    return np.array2string(array, precision=6, suppress_small=False, threshold=array.size)


def main() -> int:
    args = build_parser().parse_args()
    npz_path = Path(args.path).expanduser().resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    print(f"path: {npz_path}")
    print(f"keys: {', '.join(data.files)}")

    for key in data.files:
        array = data[key]
        print(f"\n[{key}]")
        print(f"shape: {array.shape}")
        print(f"dtype: {array.dtype}")
        if key == "wrist_rgb":
            print(f"min/max: {array.min()} / {array.max()}")
            print("preview: <image array omitted>")
        else:
            print("values:")
            print(_format_preview(array, args.full))

    if args.save_image:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Pillow is required for --save-image. Run from the project .venv."
            ) from exc

        if "wrist_rgb" not in data.files:
            raise KeyError("This NPZ file does not contain 'wrist_rgb'")
        image_path = Path(args.save_image).expanduser().resolve()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(data["wrist_rgb"]).save(image_path)
        print(f"\nsaved image: {image_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
