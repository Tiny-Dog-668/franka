#!/usr/bin/env python3
"""Offline prototype for canonicalizing real RGB frames toward simulation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SIM_IMAGE = REPO_ROOT / "start_wrist_camera_env0_frame_000_224x224.png"
DEFAULT_MODEL = (
    REPO_ROOT
    / "checkpoint"
    / "exported_0711"
    / "exported"
    / "policy_actor_e2e_best_agent.pt"
)


def _latest_real_frame() -> Path:
    candidates = sorted(
        REPO_ROOT.glob("runs/*_e2e_bundle_real_exported_0711/rgb/step_0000.png")
    )
    if not candidates:
        raise FileNotFoundError("No exported_0711 rollout RGB frames were found.")
    return candidates[-1]


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def _connected_components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index].tolist()
        components.append((x, y, width, height, area))
    return components


def _segment_real_foreground(real_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width = real_rgb.shape[:2]
    hsv = cv2.cvtColor(real_rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    channel_range = real_rgb.max(axis=2).astype(np.int16) - real_rgb.min(axis=2).astype(np.int16)

    neutral_bright = ((saturation < 75) & (value > 105) & (channel_range < 70)).astype(np.uint8)

    # Robot: keep neutral/bright pixels in the upper central workspace. This
    # intentionally excludes the white table edge at the bottom and most rail
    # clutter. Closing fills small gaps between adjacent robot surfaces.
    robot_roi = np.zeros((height, width), dtype=np.uint8)
    # The camera is wrist-mounted and the arm enters through the upper-centre
    # part of the frame. A deliberately tight ROI prevents the white wall on
    # the right and the horizontal aluminium rail from becoming "robot".
    left = int(round(0.24 * width))
    right = int(round(0.67 * width))
    bottom = int(round(0.47 * height))
    robot_roi[:bottom, left:right] = 1
    robot_candidate = (
        (saturation < 65) & (value > 125) & (channel_range < 55)
    ).astype(np.uint8)
    robot_mask = robot_candidate * robot_roi
    robot_mask = cv2.morphologyEx(
        robot_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    robot_mask = cv2.morphologyEx(
        robot_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    # Retain components that are large or touch the upper image boundary. This
    # removes isolated highlights while preserving the articulated arm.
    filtered_robot = np.zeros_like(robot_mask)
    for x, y, component_width, component_height, area in _connected_components(robot_mask):
        if area >= 18 or y <= 3:
            filtered_robot[y : y + component_height, x : x + component_width] |= robot_mask[
                y : y + component_height, x : x + component_width
            ]
    robot_mask = filtered_robot

    # Object: choose a compact neutral/bright component on the table. Components
    # are scored by squareness, brightness, area, and proximity to image center.
    object_roi = np.zeros((height, width), dtype=np.uint8)
    object_roi[min(92, height) : min(195, height), 15 : max(16, width - 15)] = 1
    object_candidates = neutral_bright * object_roi
    object_candidates = cv2.morphologyEx(
        object_candidates,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    best = None
    for x, y, component_width, component_height, area in _connected_components(object_candidates):
        if not (20 <= area <= 650 and 4 <= component_width <= 35 and 4 <= component_height <= 35):
            continue
        aspect = min(component_width, component_height) / max(component_width, component_height)
        center_x = x + 0.5 * component_width
        center_penalty = abs(center_x - 0.5 * width) / width
        score = 2.0 * aspect + min(area, 220) / 220.0 - 0.35 * center_penalty
        if best is None or score > best[0]:
            best = (score, x, y, component_width, component_height)

    object_mask = np.zeros_like(robot_mask)
    if best is not None:
        _, x, y, component_width, component_height = best
        padding = 2
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + component_width + padding)
        bottom = min(height, y + component_height + padding)
        local = object_candidates[top:bottom, left:right]
        object_mask[top:bottom, left:right] = local
        object_mask = cv2.morphologyEx(
            object_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

    # Object wins if the two masks overlap near the table boundary.
    robot_mask[object_mask > 0] = 0
    return robot_mask.astype(bool), object_mask.astype(bool)


def _estimate_sim_palette(sim_rgb: np.ndarray) -> dict[str, np.ndarray]:
    height, width = sim_rgb.shape[:2]
    wall_samples = np.concatenate(
        [
            sim_rgb[5 : max(6, height // 3), 0 : max(1, width // 5)].reshape(-1, 3),
            sim_rgb[5 : max(6, height // 3), -max(1, width // 5) :].reshape(-1, 3),
        ],
        axis=0,
    )
    table_samples = sim_rgb[int(0.72 * height) :, :].reshape(-1, 3)

    bright = sim_rgb.reshape(-1, 3)
    bright = bright[bright.mean(axis=1) > 175]
    robot_color = np.array([222, 222, 216], dtype=np.uint8)

    return {
        "wall": np.median(wall_samples, axis=0).astype(np.uint8),
        "table": np.median(table_samples, axis=0).astype(np.uint8),
        "robot": robot_color.astype(np.uint8),
        "object": np.array([242, 239, 232], dtype=np.uint8),
    }


def _canonical_images(
    real_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    robot_mask: np.ndarray,
    object_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = real_rgb.shape[:2]
    palette = _estimate_sim_palette(sim_rgb)
    table_boundary = int(round(0.42 * height))

    background = np.empty_like(real_rgb)
    background[:table_boundary] = palette["wall"]
    background[table_boundary:] = palette["table"]

    canonical_rgb = background.copy()
    foreground_mask = robot_mask | object_mask
    canonical_rgb[foreground_mask] = real_rgb[foreground_mask]

    canonical_semantic = background.copy()
    canonical_semantic[robot_mask] = palette["robot"]
    canonical_semantic[object_mask] = palette["object"]

    overlay = real_rgb.copy()
    overlay[robot_mask] = (
        0.55 * overlay[robot_mask].astype(np.float32)
        + 0.45 * np.array([255, 50, 50], dtype=np.float32)
    ).astype(np.uint8)
    overlay[object_mask] = (
        0.45 * overlay[object_mask].astype(np.float32)
        + 0.55 * np.array([40, 255, 40], dtype=np.float32)
    ).astype(np.uint8)
    return canonical_rgb, canonical_semantic, overlay


def _foreground_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _sim_target_boxes(sim_rgb: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Locate the bright simulated arm and cube in this fixed reference view."""
    hsv = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] < 55) & (hsv[..., 2] > 215)).astype(np.uint8)
    components = sorted(_connected_components(mask), key=lambda item: item[4], reverse=True)
    robot = next((item for item in components if item[1] < 20 and item[4] > 300), None)
    object_ = next((item for item in components if item[1] > 80 and 20 < item[4] < 700), None)
    if robot is None or object_ is None:
        raise RuntimeError("Could not locate robot/object in the simulation reference image.")
    rx, ry, rw, rh, _ = robot
    ox, oy, ow, oh, _ = object_
    return (rx, ry, rx + rw, ry + rh), (ox, oy, ox + ow, oy + oh)


def _paste_scaled_foreground(
    canvas: np.ndarray,
    source_rgb: np.ndarray,
    source_mask: np.ndarray,
    target_box: tuple[int, int, int, int],
) -> None:
    source_box = _foreground_box(source_mask)
    if source_box is None:
        return
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = target_box
    target_size = (tx1 - tx0, ty1 - ty0)
    crop_rgb = source_rgb[sy0:sy1, sx0:sx1]
    crop_mask = source_mask[sy0:sy1, sx0:sx1].astype(np.uint8)
    resized_rgb = cv2.resize(crop_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(crop_mask, target_size, interpolation=cv2.INTER_NEAREST).astype(bool)
    target = canvas[ty0:ty1, tx0:tx1]
    target[resized_mask] = resized_rgb[resized_mask]


def _geometrically_aligned_image(
    real_rgb: np.ndarray,
    canonical_rgb: np.ndarray,
    sim_rgb: np.ndarray,
    robot_mask: np.ndarray,
    object_mask: np.ndarray,
) -> np.ndarray:
    aligned = canonical_rgb.copy()
    # Remove the foreground at its real-image location, then place it at the
    # scale and position measured from the simulation reference.
    height = aligned.shape[0]
    table_boundary = int(round(0.42 * height))
    palette = _estimate_sim_palette(sim_rgb)
    foreground = robot_mask | object_mask
    ys, _ = np.indices(foreground.shape)
    aligned[foreground & (ys < table_boundary)] = palette["wall"]
    aligned[foreground & (ys >= table_boundary)] = palette["table"]
    robot_box, object_box = _sim_target_boxes(sim_rgb)
    _paste_scaled_foreground(aligned, real_rgb, robot_mask, robot_box)
    _paste_scaled_foreground(aligned, real_rgb, object_mask, object_box)
    return aligned


def _model_inputs_for_frame(real_path: Path) -> tuple[np.ndarray, np.ndarray]:
    match = re.match(r"step_(\d+)\.png$", real_path.name)
    rollout_path = real_path.parents[1] / "rollout.jsonl"
    if match is None or not rollout_path.is_file():
        return np.zeros(4, dtype=np.float32), np.zeros(15, dtype=np.float32)

    step_index = int(match.group(1))
    with rollout_path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    if step_index >= len(records):
        raise IndexError(f"Step {step_index} not found in {rollout_path}")
    model_input = records[step_index]["model_input"]
    return (
        np.asarray(model_input["action_history"], dtype=np.float32),
        np.asarray(model_input["proprio_obs"], dtype=np.float32),
    )


def _extract_features(model, images: list[np.ndarray]) -> dict[str, torch.Tensor]:
    rgb = torch.from_numpy(np.stack(images)).to(torch.uint8)
    x = rgb.float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - model._imgnet_mean) / model._imgnet_std
    with torch.inference_mode():
        raw = model.vision_encoder(x).reshape(len(images), 512)
        states = torch.zeros((len(images), 561), dtype=torch.float32)
        states[:, 49:561] = raw
        scaled = model.state_preprocessor(states)[:, 49:561]
    return {"raw_resnet": raw, "scaled_resnet": scaled}


def _pair_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    cosine = F.cosine_similarity(reference.unsqueeze(0), candidate.unsqueeze(0)).item()
    denominator = 0.5 * (
        torch.linalg.vector_norm(reference) + torch.linalg.vector_norm(candidate)
    )
    relative_l2 = (torch.linalg.vector_norm(reference - candidate) / denominator).item()
    return {"cosine": cosine, "relative_l2": relative_l2}


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment a real rollout frame and compare canonicalized images to simulation."
    )
    parser.add_argument("--real", type=Path, help="Real rollout RGB frame; defaults to latest step_0000.")
    parser.add_argument("--sim", type=Path, default=DEFAULT_SIM_IMAGE, help="Simulation reference image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="TorchScript policy bundle.")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults under runs/.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    real_path = (args.real or _latest_real_frame()).expanduser().resolve()
    sim_path = args.sim.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = REPO_ROOT / "runs" / f"{timestamp}_visual_alignment"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    real_rgb = _load_rgb(real_path)
    sim_rgb = _load_rgb(sim_path)
    if real_rgb.shape != sim_rgb.shape:
        real_rgb = np.asarray(
            Image.fromarray(real_rgb).resize((sim_rgb.shape[1], sim_rgb.shape[0]), Image.BILINEAR),
            dtype=np.uint8,
        ).copy()

    robot_mask, object_mask = _segment_real_foreground(real_rgb)
    canonical_rgb, canonical_semantic, overlay = _canonical_images(
        real_rgb,
        sim_rgb,
        robot_mask,
        object_mask,
    )
    aligned_rgb = _geometrically_aligned_image(
        real_rgb, canonical_rgb, sim_rgb, robot_mask, object_mask
    )

    images = {
        "sim": sim_rgb,
        "real": real_rgb,
        "canonical_rgb": canonical_rgb,
        "aligned_rgb": aligned_rgb,
        "canonical_semantic": canonical_semantic,
    }
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    features = _extract_features(model, list(images.values()))
    action_history, proprio = _model_inputs_for_frame(real_path)
    action_tensor = torch.from_numpy(action_history).unsqueeze(0).repeat(len(images), 1)
    proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).repeat(len(images), 1)
    rgb_tensor = torch.from_numpy(np.stack(list(images.values()))).to(torch.uint8)
    with torch.inference_mode():
        actions = model(action_tensor, proprio_tensor, rgb_tensor).cpu().numpy()

    names = list(images)
    metrics: dict[str, object] = {
        "real_path": str(real_path),
        "sim_path": str(sim_path),
        "model_path": str(model_path),
        "robot_mask_pixels": int(robot_mask.sum()),
        "object_mask_pixels": int(object_mask.sum()),
        "variants": {},
    }
    for index, name in enumerate(names):
        variant = {
            "raw_action": actions[index].astype(float).tolist(),
            "clipped_action": np.clip(actions[index], -1.0, 1.0).astype(float).tolist(),
        }
        if index != 0:
            variant["features_vs_sim"] = {
                key: _pair_metrics(value[0], value[index])
                for key, value in features.items()
            }
            variant["action_l2_vs_sim"] = float(np.linalg.norm(actions[index] - actions[0]))
        metrics["variants"][name] = variant

    Image.fromarray(sim_rgb).save(output_dir / "sim.png")
    Image.fromarray(real_rgb).save(output_dir / "real.png")
    _save_mask(output_dir / "robot_mask.png", robot_mask)
    _save_mask(output_dir / "object_mask.png", object_mask)
    Image.fromarray(overlay).save(output_dir / "mask_overlay.png")
    Image.fromarray(canonical_rgb).save(output_dir / "canonical_rgb.png")
    Image.fromarray(aligned_rgb).save(output_dir / "aligned_rgb.png")
    Image.fromarray(canonical_semantic).save(output_dir / "canonical_semantic.png")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"Real frame: {real_path}")
    print(f"Simulation frame: {sim_path}")
    print(f"Output directory: {output_dir}")
    print(f"Mask pixels: robot={robot_mask.sum()}, object={object_mask.sum()}")
    for name in names:
        variant = metrics["variants"][name]
        if name == "sim":
            print(f"{name:20s} action={np.round(variant['raw_action'], 4).tolist()}")
            continue
        raw_feature = variant["features_vs_sim"]["raw_resnet"]
        scaled_feature = variant["features_vs_sim"]["scaled_resnet"]
        print(
            f"{name:20s} "
            f"raw_cos={raw_feature['cosine']:.4f} "
            f"scaled_cos={scaled_feature['cosine']:.4f} "
            f"action_l2={variant['action_l2_vs_sim']:.4f} "
            f"action={np.round(variant['raw_action'], 4).tolist()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
