"""Standalone smoke test for the bundled TorchScript policy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    Image = None


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BUNDLE_DIR / "policy_actor_e2e.pt"
DEFAULT_METADATA = BUNDLE_DIR / "policy_actor_e2e.json"


def _load_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_vector(raw: str | None, dim: int, name: str) -> torch.Tensor:
    if raw is None:
        return torch.zeros(dim, dtype=torch.float32)
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != dim:
        raise ValueError(f"{name} expects {dim} comma-separated values, got {len(values)}")
    return torch.tensor(values, dtype=torch.float32)


def _make_gradient_rgb(height: int, width: int) -> torch.Tensor:
    y = torch.linspace(0, 255, steps=height, dtype=torch.float32).view(height, 1)
    x = torch.linspace(0, 255, steps=width, dtype=torch.float32).view(1, width)
    red = x.expand(height, width)
    green = y.expand(height, width)
    blue = 0.5 * (red + green)
    return torch.stack([red, green, blue], dim=-1).clamp(0, 255).to(torch.uint8)


def _load_rgb_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    if Image is None:
        raise RuntimeError("Pillow is required for --image. Install it with `pip install pillow`.")
    image = Image.open(image_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height))
    return torch.from_numpy(np.asarray(image, dtype=np.uint8))


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the bundled TorchScript policy.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to the TorchScript model.")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA), help="Path to the sidecar JSON metadata.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device: cpu or cuda[:id].")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the smoke test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used by synthetic inputs.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before timing.")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs.")
    parser.add_argument(
        "--rgb-mode",
        type=str,
        default="gradient",
        choices=["gradient", "random", "zeros"],
        help="Synthetic RGB input mode when --image is not provided.",
    )
    parser.add_argument("--image", type=str, default=None, help="Optional RGB image to feed the policy.")
    parser.add_argument("--action-history", type=str, default=None, help="Comma-separated action history values.")
    parser.add_argument("--proprio-obs", type=str, default=None, help="Comma-separated proprio observation values.")
    args = parser.parse_args()

    model_path = Path(args.model)
    metadata_path = Path(args.metadata)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    metadata = _load_metadata(metadata_path)
    input_signature = metadata.get("input_signature", {})
    if "wrist_rgb" not in input_signature:
        raise RuntimeError("This standalone bundle test expects an end-to-end RGB policy.")

    action_dim = input_signature["action_history"][0]
    proprio_dim = input_signature["proprio_obs"][0]
    rgb_height, rgb_width, _ = input_signature["wrist_rgb"]

    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    model.to(device)

    action_history = _parse_vector(args.action_history, action_dim, "action_history")
    action_history = action_history.unsqueeze(0).repeat(args.batch_size, 1).to(device)
    proprio_obs = _parse_vector(args.proprio_obs, proprio_dim, "proprio_obs")
    proprio_obs = proprio_obs.unsqueeze(0).repeat(args.batch_size, 1).to(device)

    if args.image is not None:
        wrist_rgb = _load_rgb_image(Path(args.image), rgb_height, rgb_width)
    elif args.rgb_mode == "zeros":
        wrist_rgb = torch.zeros((rgb_height, rgb_width, 3), dtype=torch.uint8)
    elif args.rgb_mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        wrist_rgb = torch.randint(0, 256, (rgb_height, rgb_width, 3), dtype=torch.uint8, generator=generator)
    else:
        wrist_rgb = _make_gradient_rgb(rgb_height, rgb_width)
    wrist_rgb = wrist_rgb.unsqueeze(0).repeat(args.batch_size, 1, 1, 1).to(device)

    wrist_rgb_perturbed = wrist_rgb.clone()
    wrist_rgb_perturbed[..., 0] = torch.clamp(wrist_rgb_perturbed[..., 0].to(torch.int16) + 20, 0, 255).to(torch.uint8)

    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"model         : {model_path}")
    print(f"metadata      : {metadata_path}")
    print(f"file_size_mb  : {model_path.stat().st_size / (1024 ** 2):.2f}")
    print(f"num_parameters: {num_params}")
    print(f"device        : {device}")
    print(f"wrist_rgb     : {tuple(wrist_rgb.shape)} {wrist_rgb.dtype}")

    with torch.inference_mode():
        out_a = model(action_history, proprio_obs, wrist_rgb)
        out_b = model(action_history, proprio_obs, wrist_rgb)
        out_c = model(action_history, proprio_obs, wrist_rgb_perturbed)

    finite_ok = bool(torch.isfinite(out_a).all().item())
    repeat_diff = float(torch.max(torch.abs(out_a - out_b)).item())
    perturb_diff = float(torch.max(torch.abs(out_a - out_c)).item())

    print(f"output_shape  : {tuple(out_a.shape)}")
    print(f"finite_output : {finite_ok}")
    print(f"repeat_diff   : {repeat_diff:.8f}")
    print(f"perturb_diff  : {perturb_diff:.8f}")
    print("actions       :")
    print(out_a.detach().cpu())

    for _ in range(args.warmup):
        with torch.inference_mode():
            _ = model(action_history, proprio_obs, wrist_rgb)
    _sync_if_needed(device)

    start = time.perf_counter()
    for _ in range(args.runs):
        with torch.inference_mode():
            _ = model(action_history, proprio_obs, wrist_rgb)
    _sync_if_needed(device)
    avg_latency_ms = 1000.0 * (time.perf_counter() - start) / max(args.runs, 1)
    print(f"avg_latency_ms: {avg_latency_ms:.3f}")

    if not finite_ok:
        raise RuntimeError("Model output contains NaN or Inf.")


if __name__ == "__main__":
    main()
