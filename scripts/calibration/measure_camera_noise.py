#!/usr/bin/env python3
"""测量 RealSense 彩色相机的信号相关噪声模型 sigma(mu)，用于对齐仿真域随机化。

本脚本只打开 D435 彩色流，不连接 Franka，也不发送任何控制指令。

原理是逐像素的时间统计：场景静止时，同一像素在多帧之间的标准差就是该亮度下的传感器
噪声。因此**采集期间场景必须完全静止**，机械臂、手、光源都不能动，否则运动会被误计
入噪声。脚本会检查这一点并在超标时报错退出。

两个刻意的设计选择：

1. 默认在**模型输入域**（crop -> 双线性缩放到 224x224）上测量，而不是 640x480 原始
   分辨率。缩放会对邻域像素做平均，白噪声因此被压低约 1.8 倍；仿真的噪声是加在模型
   输入分辨率上的，所以必须在同一个域里测才有可比性。crop 与缩放直接复用
   ``e2e_bundle`` 的实现，保证和部署逐像素一致。
2. LUT 使用**扣除整帧亮度漂移之后**的标准差。市电工频闪烁和光源热漂移会让整帧一起
   明暗变化，那属于照明变化（仿真里由 brightness 随机化负责），不是传感器噪声。
   报告里同时给出未扣除的数值供对照。

输出的 ``sigma_lut`` 可直接用于仿真：把当前那种"全图同一个 std"的同方差高斯噪声，
换成按像素亮度查表的信号相关噪声。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.e2e_bundle import (  # noqa: E402
    BundleCameraConfig,
    RealSenseRGBCamera,
    load_bundle_config,
)

DEFAULT_BUNDLE_CONFIG = (
    REPO_ROOT / "configs" / "e2e_bundle_real_exported_0803_dr_heatmap.json"
)
DEFAULT_EXPOSURES = (100.0,)
DEFAULT_GAINS = (16.0, 32.0, 64.0)
# TacEx 侧 Sim2RealCubeCameraAlignmentDREnvCfg.wrist_gaussian_noise_std_range 的上界，
# 单位是归一化到 [0, 1] 的像素值。仅用于在报告里做对照，可用 CLI 覆盖。
DEFAULT_SIMULATION_NOISE_STD = 0.006

CSV_FIELDS = [
    "requested_exposure",
    "requested_gain",
    "effective_exposure",
    "effective_gain",
    "channel",
    "bin_center_dn",
    "sigma_dn",
    "sigma_normalized",
    "pixel_count",
]

CHANNEL_LABELS = ("red", "green", "blue")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle-config",
        default=str(DEFAULT_BUNDLE_CONFIG),
        help="部署配置，提供相机串号、分辨率和 crop（默认 0803 heatmap 配置）",
    )
    parser.add_argument(
        "--metadata",
        help="模型 metadata JSON；默认取 bundle config 的 model.metadata_path，"
        "从 input_signature.wrist_rgb 读取模型输入尺寸",
    )
    parser.add_argument(
        "--domain",
        choices=("model", "raw"),
        default="model",
        help="model=crop 并缩放到模型输入尺寸（默认，与仿真加噪的域一致）；raw=原始分辨率",
    )
    parser.add_argument("--exposures", nargs="+", type=float, default=list(DEFAULT_EXPOSURES))
    parser.add_argument("--gains", nargs="+", type=float, default=list(DEFAULT_GAINS))
    parser.add_argument("--frames", type=int, default=200, help="每个工况的统计帧数")
    parser.add_argument(
        "--extra-warmup-frames",
        type=int,
        default=30,
        help="在 bundle config 的 warmup_frames 之外额外丢弃的帧数",
    )
    parser.add_argument("--settle-s", type=float, default=1.0, help="两个工况之间的等待时间")
    parser.add_argument("--bins", type=int, default=32, help="按像素均值分箱的箱数")
    parser.add_argument(
        "--minimum-bin-count",
        type=int,
        default=200,
        help="低于该像素数的箱不参与拟合，LUT 中记为 null",
    )
    parser.add_argument(
        "--dark-threshold-dn",
        type=float,
        default=32.0,
        help="暗区判定阈值；本场景背景接近全黑，暗区噪声是最关键的指标",
    )
    parser.add_argument(
        "--motion-sigma-dn",
        type=float,
        default=12.0,
        help="单像素时间标准差超过该值即视为该像素在动",
    )
    parser.add_argument(
        "--maximum-motion-fraction",
        type=float,
        default=0.01,
        help="运动像素占比超过该值则判定场景不静止，测量作废",
    )
    parser.add_argument(
        "--simulation-noise-std",
        type=float,
        default=DEFAULT_SIMULATION_NOISE_STD,
        help="仿真当前的高斯噪声 std 上界，用于打印实测/仿真的倍率对照",
    )
    parser.add_argument(
        "--maximum-stack-mb",
        type=float,
        default=2048.0,
        help="单个工况帧堆栈的内存上限，超出直接报错而不是把机器拖垮",
    )
    parser.add_argument("--output-dir", help="默认 runs/TIMESTAMP_camera_noise")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.exposures or any(not math.isfinite(value) or value <= 0.0 for value in args.exposures):
        raise ValueError("--exposures 需要一个或多个有限正数")
    if not args.gains or any(not math.isfinite(value) or value < 0.0 for value in args.gains):
        raise ValueError("--gains 需要一个或多个有限非负数")
    if args.frames < 2:
        raise ValueError("--frames 至少为 2，否则无法估计时间方差")
    if args.extra_warmup_frames < 0:
        raise ValueError("--extra-warmup-frames 不能为负")
    if args.settle_s < 0.0:
        raise ValueError("--settle-s 不能为负")
    if args.bins < 2:
        raise ValueError("--bins 至少为 2")
    if args.minimum_bin_count < 1:
        raise ValueError("--minimum-bin-count 必须为正")
    if not 0.0 <= args.dark_threshold_dn <= 255.0:
        raise ValueError("--dark-threshold-dn 必须落在 [0, 255]")
    if args.motion_sigma_dn <= 0.0:
        raise ValueError("--motion-sigma-dn 必须为正")
    if not 0.0 <= args.maximum_motion_fraction <= 1.0:
        raise ValueError("--maximum-motion-fraction 必须落在 [0, 1]")
    if args.simulation_noise_std <= 0.0:
        raise ValueError("--simulation-noise-std 必须为正")
    if args.maximum_stack_mb <= 0.0:
        raise ValueError("--maximum-stack-mb 必须为正")


def _value_slug(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def resolve_model_view_size(
    bundle_config_path: Path,
    metadata_path: Path | None,
) -> tuple[int, int, Path]:
    """从 metadata 的 input_signature.wrist_rgb 读出模型输入尺寸，不硬编码 224。"""
    if metadata_path is None:
        config = json.loads(bundle_config_path.read_text(encoding="utf-8"))
        raw_path = (config.get("model") or {}).get("metadata_path")
        if not raw_path:
            raise ValueError(
                f"{bundle_config_path} 缺少 model.metadata_path，请用 --metadata 显式指定"
            )
        metadata_path = Path(raw_path).expanduser()
    metadata_path = metadata_path.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"模型 metadata 不存在: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = (metadata.get("input_signature") or {}).get("wrist_rgb")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"{metadata_path} 的 input_signature.wrist_rgb 不是 [H, W, C]")
    height, width, channels = (int(value) for value in shape)
    if channels != 3 or height < 1 or width < 1:
        raise ValueError(f"不支持的 wrist_rgb 形状: {shape}")
    return width, height, metadata_path


def frame_means(frames: np.ndarray) -> np.ndarray:
    """每帧的全局平均亮度，用来分离整体照明漂移和逐像素传感器噪声。"""
    return frames.reshape(frames.shape[0], -1).mean(axis=1, dtype=np.float64)


def flicker_weights(means: np.ndarray) -> tuple[np.ndarray, float]:
    """把每帧缩放到共同的平均亮度，抵消工频闪烁和光源热漂移。"""
    reference = float(np.mean(means))
    if reference <= 0.0 or not np.all(means > 0.0):
        raise ValueError("存在平均亮度为零的帧，无法做闪烁补偿；检查曝光设置和镜头遮挡")
    return reference / means, reference


def pixel_statistics(
    frames: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """逐像素的时间均值与标准差（无偏）。

    逐帧累加而不是一次性把整个堆栈转成 float64，避免原始分辨率下的内存峰值。
    """
    count = int(frames.shape[0])
    if count < 2:
        raise ValueError("需要至少 2 帧才能估计时间方差")
    if weights is not None and weights.shape[0] != count:
        raise ValueError("weights 长度必须与帧数一致")

    total = np.zeros(frames.shape[1:], dtype=np.float64)
    total_squared = np.zeros(frames.shape[1:], dtype=np.float64)
    for index in range(count):
        value = frames[index].astype(np.float64)
        if weights is not None:
            value *= float(weights[index])
        total += value
        total_squared += value * value

    mean = total / count
    variance = np.maximum(total_squared / count - mean * mean, 0.0) * count / (count - 1)
    return mean, np.sqrt(variance)


def build_sigma_lut(
    mean: np.ndarray,
    sigma: np.ndarray,
    bins: int,
    minimum_bin_count: int,
) -> dict[str, list[Any]]:
    """按像素均值分箱，取箱内 sigma 的中位数得到 sigma(mu) 查找表。

    用中位数而非平均值：场景里若残留个别微动像素，中位数不会被整体抬高。
    """
    flat_mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    flat_sigma = np.asarray(sigma, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, 255.0, bins + 1)
    bin_index = np.clip(np.digitize(flat_mean, edges[1:-1]), 0, bins - 1)

    centers: list[float] = []
    sigmas: list[float | None] = []
    counts: list[int] = []
    for index in range(bins):
        selected = bin_index == index
        count = int(np.count_nonzero(selected))
        counts.append(count)
        if count < minimum_bin_count:
            centers.append(float(0.5 * (edges[index] + edges[index + 1])))
            sigmas.append(None)
            continue
        centers.append(float(np.median(flat_mean[selected])))
        sigmas.append(float(np.median(flat_sigma[selected])))

    return {
        "bin_center_dn": centers,
        "sigma_dn": sigmas,
        "sigma_normalized": [None if value is None else value / 255.0 for value in sigmas],
        "pixel_count": counts,
    }


def fit_affine_noise_model(lut: dict[str, list[Any]]) -> dict[str, float | int | None]:
    """拟合 sigma^2 = a*mu + b。

    a 是散粒噪声项（正比于增益），b 是读出噪声项（正比于增益平方）。按箱内像素数加权，
    避免只有零星像素的极亮/极暗箱主导拟合结果。
    """
    centers = np.asarray(lut["bin_center_dn"], dtype=np.float64)
    sigmas = np.asarray(
        [np.nan if value is None else value for value in lut["sigma_dn"]],
        dtype=np.float64,
    )
    counts = np.asarray(lut["pixel_count"], dtype=np.float64)
    valid = np.isfinite(sigmas) & (counts > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return {"a": None, "b": None, "rmse_dn": None, "fitted_bin_count": int(np.count_nonzero(valid))}

    mu = centers[valid]
    measured_sigma = sigmas[valid]
    weight = np.sqrt(counts[valid])
    design = np.stack([mu, np.ones_like(mu)], axis=1) * weight[:, None]
    target = (measured_sigma**2) * weight
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    a, b = (float(value) for value in solution)
    predicted_sigma = np.sqrt(np.maximum(a * mu + b, 0.0))
    rmse = float(np.sqrt(np.mean((predicted_sigma - measured_sigma) ** 2)))
    return {"a": a, "b": b, "rmse_dn": rmse, "fitted_bin_count": int(mu.size)}


def _summarize_sigma(sigma: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    values = sigma.reshape(-1) if mask is None else sigma[mask].reshape(-1)
    if values.size == 0:
        return {"pixel_count": 0, "median_dn": float("nan"), "p95_dn": float("nan")}
    median = float(np.median(values))
    p95 = float(np.percentile(values, 95.0))
    return {
        "pixel_count": int(values.size),
        "median_dn": median,
        "median_normalized": median / 255.0,
        "p95_dn": p95,
        "p95_normalized": p95 / 255.0,
    }


def analyze_frame_stack(
    frames: np.ndarray,
    *,
    bins: int,
    minimum_bin_count: int,
    dark_threshold_dn: float,
    motion_sigma_dn: float,
) -> dict[str, Any]:
    """把一个工况的帧堆栈变成完整的噪声报告。纯函数，可脱离硬件单测。"""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"期望 [T, H, W, 3] 的帧堆栈，得到 {frames.shape}")

    means_per_frame = frame_means(frames)
    weights, reference_mean = flicker_weights(means_per_frame)

    raw_mean, raw_sigma = pixel_statistics(frames)
    mean, sigma = pixel_statistics(frames, weights=weights)

    motion_pixel_fraction = float(np.mean(raw_sigma > motion_sigma_dn))
    dark_mask = mean <= dark_threshold_dn

    per_channel = {}
    for index, label in enumerate(CHANNEL_LABELS):
        channel_lut = build_sigma_lut(
            mean[..., index], sigma[..., index], bins, minimum_bin_count
        )
        per_channel[label] = {
            "sigma_lut": channel_lut,
            "affine_fit": fit_affine_noise_model(channel_lut),
            "overall": _summarize_sigma(sigma[..., index]),
        }

    combined_lut = build_sigma_lut(mean, sigma, bins, minimum_bin_count)
    return {
        "frame_count": int(frames.shape[0]),
        "image_shape": [int(value) for value in frames.shape[1:]],
        "illumination": {
            "reference_frame_mean_dn": reference_mean,
            "frame_mean_std_dn": float(np.std(means_per_frame)),
            "frame_mean_min_dn": float(np.min(means_per_frame)),
            "frame_mean_max_dn": float(np.max(means_per_frame)),
        },
        "static_scene_check": {
            "motion_sigma_threshold_dn": float(motion_sigma_dn),
            "motion_pixel_fraction": motion_pixel_fraction,
        },
        "sigma_lut": combined_lut,
        "affine_fit": fit_affine_noise_model(combined_lut),
        "overall": _summarize_sigma(sigma),
        "dark_region": {
            "threshold_dn": float(dark_threshold_dn),
            **_summarize_sigma(sigma, dark_mask),
        },
        "without_flicker_removal": {"overall": _summarize_sigma(raw_sigma)},
        "per_channel": per_channel,
        "_mean_image": mean,
        "_sigma_image": sigma,
    }


def capture_frame_stack(
    camera_config: BundleCameraConfig,
    *,
    exposure: float,
    gain: float,
    frames: int,
    output_width: int | None,
    output_height: int | None,
    extra_warmup_frames: int,
) -> tuple[np.ndarray, dict[str, bool | float | None]]:
    camera = RealSenseRGBCamera(
        camera_config,
        output_width=output_width,
        output_height=output_height,
        auto_exposure=None,
        exposure=exposure,
        gain=gain,
    )
    try:
        controls = camera.get_color_controls()
        for _ in range(extra_warmup_frames):
            camera.read()
        first = camera.read()
        stack = np.empty((frames, *first.shape), dtype=np.uint8)
        stack[0] = first
        for index in range(1, frames):
            stack[index] = camera.read()
    finally:
        camera.close()
    return stack, controls


def _save_mean_and_sigma_images(
    directory: Path,
    slug: str,
    mean: np.ndarray,
    sigma: np.ndarray,
) -> dict[str, str | float]:
    directory.mkdir(parents=True, exist_ok=True)
    mean_path = directory / f"mean_{slug}.png"
    sigma_path = directory / f"sigma_{slug}.png"
    Image.fromarray(np.clip(np.rint(mean), 0.0, 255.0).astype(np.uint8)).save(mean_path)

    # sigma 通常只有几个 DN，直接存会是一张全黑图，按 p99 拉伸后才看得出空间分布。
    scale_reference = float(np.percentile(sigma, 99.0))
    display_scale = 255.0 / scale_reference if scale_reference > 1e-6 else 1.0
    Image.fromarray(
        np.clip(np.rint(sigma * display_scale), 0.0, 255.0).astype(np.uint8)
    ).save(sigma_path)
    return {
        "mean_image": str(mean_path),
        "sigma_image": str(sigma_path),
        "sigma_image_display_scale": display_scale,
    }


def _csv_rows(trial: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = [("all", trial["sigma_lut"])]
    sources.extend(
        (label, section["sigma_lut"]) for label, section in trial["per_channel"].items()
    )
    for channel, lut in sources:
        for center, sigma_dn, sigma_normalized, count in zip(
            lut["bin_center_dn"],
            lut["sigma_dn"],
            lut["sigma_normalized"],
            lut["pixel_count"],
        ):
            rows.append(
                {
                    "requested_exposure": trial["requested_exposure"],
                    "requested_gain": trial["requested_gain"],
                    "effective_exposure": trial["camera_color_controls"].get("exposure"),
                    "effective_gain": trial["camera_color_controls"].get("gain"),
                    "channel": channel,
                    "bin_center_dn": center,
                    "sigma_dn": sigma_dn,
                    "sigma_normalized": sigma_normalized,
                    "pixel_count": count,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_trial_summary(trial: dict[str, Any], simulation_noise_std: float) -> None:
    overall = trial["overall"]
    dark = trial["dark_region"]
    fit = trial["affine_fit"]
    illumination = trial["illumination"]
    controls = trial["camera_color_controls"]
    print(
        f"[noise] exposure={controls.get('exposure')} gain={controls.get('gain')}: "
        f"整图 sigma 中位数 {overall['median_dn']:.2f} DN "
        f"(归一化 {overall['median_normalized']:.5f})，"
        f"p95 {overall['p95_dn']:.2f} DN",
        flush=True,
    )
    if dark["pixel_count"] > 0:
        print(
            f"[noise]   暗区 (mu<={dark['threshold_dn']:g} DN, {dark['pixel_count']} px): "
            f"sigma 中位数 {dark['median_dn']:.2f} DN "
            f"(归一化 {dark['median_normalized']:.5f})",
            flush=True,
        )
    if fit["a"] is not None:
        print(
            f"[noise]   拟合 sigma^2 = {fit['a']:.4g}*mu + {fit['b']:.4g}，"
            f"残差 {fit['rmse_dn']:.3f} DN，参与拟合 {fit['fitted_bin_count']} 箱",
            flush=True,
        )
    print(
        f"[noise]   整帧亮度波动 std {illumination['frame_mean_std_dn']:.3f} DN "
        f"(已在 LUT 中扣除)，运动像素占比 "
        f"{100.0 * trial['static_scene_check']['motion_pixel_fraction']:.3f}%",
        flush=True,
    )
    ratio = overall["median_normalized"] / simulation_noise_std
    print(
        f"[noise]   对照仿真上界 {simulation_noise_std:g}: 实测/仿真 = {ratio:.1f}x",
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    bundle_config_path = Path(args.bundle_config).expanduser().resolve()
    metadata_argument = Path(args.metadata).expanduser() if args.metadata else None
    model_width, model_height, metadata_path = resolve_model_view_size(
        bundle_config_path, metadata_argument
    )
    base_camera_config = load_bundle_config(bundle_config_path).camera
    if base_camera_config.source != "realsense":
        raise ValueError(
            f"{bundle_config_path} 的 camera.source 是 {base_camera_config.source!r}，"
            "本脚本需要 realsense"
        )

    if args.domain == "model":
        camera_config = base_camera_config
        output_width: int | None = model_width
        output_height: int | None = model_height
    else:
        camera_config = replace(base_camera_config, enable_crop=False)
        output_width = None
        output_height = None

    capture_height = output_height if output_height is not None else camera_config.height
    capture_width = output_width if output_width is not None else camera_config.width
    stack_mb = args.frames * capture_height * capture_width * 3 / (1024.0 * 1024.0)
    if stack_mb > args.maximum_stack_mb:
        raise ValueError(
            f"帧堆栈需要 {stack_mb:.0f} MB，超过 --maximum-stack-mb={args.maximum_stack_mb:g}；"
            "请减少 --frames 或改用 --domain model"
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT / "runs" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_camera_noise"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[noise] 仅使用相机，不连接 Franka，不发送任何控制指令", flush=True)
    print(
        f"[noise] 采样域={args.domain} {capture_width}x{capture_height}，"
        f"每工况 {args.frames} 帧，堆栈 {stack_mb:.0f} MB",
        flush=True,
    )
    print("[noise] 采集期间请保持场景完全静止（机械臂、手、光源都不要动）", flush=True)

    trials: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    failures = 0
    non_static = 0
    for exposure in args.exposures:
        for gain in args.gains:
            slug = f"e{_value_slug(exposure)}_g{_value_slug(gain)}"
            print(f"\n[noise] 采集 exposure={exposure:g}, gain={gain:g}", flush=True)
            try:
                frames, controls = capture_frame_stack(
                    camera_config,
                    exposure=exposure,
                    gain=gain,
                    frames=args.frames,
                    output_width=output_width,
                    output_height=output_height,
                    extra_warmup_frames=args.extra_warmup_frames,
                )
            except KeyboardInterrupt:
                print("\n[noise] 已中断", flush=True)
                return 130
            except Exception as error:  # noqa: BLE001 - 单个工况失败不应终止整轮采集
                failures += 1
                print(f"[noise]   采集失败: {error}", flush=True)
                trials.append(
                    {
                        "requested_exposure": exposure,
                        "requested_gain": gain,
                        "status": "failed",
                        "error": str(error),
                    }
                )
                continue

            trial = analyze_frame_stack(
                frames,
                bins=args.bins,
                minimum_bin_count=args.minimum_bin_count,
                dark_threshold_dn=args.dark_threshold_dn,
                motion_sigma_dn=args.motion_sigma_dn,
            )
            mean_image = trial.pop("_mean_image")
            sigma_image = trial.pop("_sigma_image")
            trial.update(
                {
                    "requested_exposure": exposure,
                    "requested_gain": gain,
                    "camera_color_controls": controls,
                    "status": "ok",
                }
            )
            trial["artifacts"] = _save_mean_and_sigma_images(
                output_dir, slug, mean_image, sigma_image
            )

            motion_fraction = trial["static_scene_check"]["motion_pixel_fraction"]
            if motion_fraction > args.maximum_motion_fraction:
                non_static += 1
                trial["static_scene_check"]["passed"] = False
                print(
                    f"[noise]   场景不静止：{100.0 * motion_fraction:.2f}% 像素的时间"
                    f"标准差超过 {args.motion_sigma_dn:g} DN，该工况结果不可用",
                    flush=True,
                )
            else:
                trial["static_scene_check"]["passed"] = True

            if controls.get("auto_exposure"):
                print(
                    "[noise]   警告：相机仍处于自动曝光，测得的波动包含 AE 调节，不是纯噪声",
                    flush=True,
                )

            _print_trial_summary(trial, args.simulation_noise_std)
            trials.append(trial)
            csv_rows.extend(_csv_rows(trial))

            if args.settle_s > 0.0:
                time.sleep(args.settle_s)

    csv_path = output_dir / "noise_lut.csv"
    _write_csv(csv_path, csv_rows)
    report = {
        "kind": "camera_noise_model",
        "camera_only_no_robot_connection": True,
        "bundle_config": str(bundle_config_path),
        "model_metadata": str(metadata_path),
        "domain": args.domain,
        "capture_size": [capture_width, capture_height],
        "camera": {
            "serial": camera_config.serial,
            "width": camera_config.width,
            "height": camera_config.height,
            "fps": camera_config.fps,
            "enable_crop": camera_config.enable_crop,
            "crop_left": camera_config.crop_left,
            "crop_top": camera_config.crop_top,
            "crop_width": camera_config.crop_width,
            "crop_height": camera_config.crop_height,
            "warmup_frames": camera_config.warmup_frames,
        },
        "frames_per_trial": args.frames,
        "bins": args.bins,
        "minimum_bin_count": args.minimum_bin_count,
        "simulation_noise_std_reference": args.simulation_noise_std,
        "trials": trials,
    }
    report_path = output_dir / "noise_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n[noise] LUT CSV: {csv_path}")
    print(f"[noise] 报告: {report_path}")
    if failures:
        print(f"[noise] {failures} 个工况采集失败", flush=True)
    if non_static:
        print(
            f"[noise] {non_static} 个工况因场景运动作废，请清空视野后重测",
            flush=True,
        )
        return 1
    if not csv_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
