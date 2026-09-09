#!/usr/bin/env python3
"""比较仿真与真机的策略输入图像统计，定位视觉 sim2real 的外观差异。

纯离线分析，不连接机器人也不打开相机。

两边都必须是**模型输入域**的图像（crop 之后缩放到策略输入尺寸），否则统计不可比：

| 来源 | 路径 | 说明 |
| --- | --- | --- |
| 真机 rollout | ``runs/<部署运行>/rgb/`` | `e2e_bundle` 落盘的就是模型视角图 |
| 真机静态基准 | ``runs/<时间戳>_camera_noise/mean_*.png`` | 多帧平均，无噪声 |
| 仿真 | ``collect_rma_student_rollouts.py`` 产出的 NPZ 的 ``wrist_rgb`` | 已经过 DR 与内参补偿，就是策略实际看到的图 |

关注三件事，都是全局 brightness 随机化覆盖不到的：

1. **整体亮度与动态范围**：真机场景可能极暗且双峰（近黑背景 + 明亮手臂），
   仿真若渲染出完全不同的亮度分布，冻结 ResNet 的特征就对不上。
2. **通道平衡**：机器人状态指示灯之类的有色光源会造成整体偏色，仿真的白平衡
   随机化幅度通常远小于真实偏色。
3. **分布距离**：亮度直方图的 Wasserstein-1 距离直接给出"仿真整体需要平移多少 DN
   才能对上真机"，比逐项看百分位更容易判断严重程度。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# 与 TacEx 的 DR 管线一致的 ITU-R BT.709 亮度权重。
LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
CHANNEL_LABELS = ("red", "green", "blue")
# 逐块处理，避免把上千张图一次性转成 float64。
CHUNK_SIZE = 32


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        required=True,
        help="真机图像来源：PNG 文件、图像目录，或含 wrist_rgb 的 NPZ",
    )
    parser.add_argument(
        "--sim",
        required=True,
        help="仿真图像来源：同上，通常是 collect_rma_student_rollouts.py 的 episode NPZ",
    )
    parser.add_argument("--npz-key", default="wrist_rgb", help="NPZ 中图像数组的键名")
    parser.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="每一侧最多取多少张（按序均匀抽样），控制内存与耗时",
    )
    parser.add_argument(
        "--dark-threshold-dn",
        type=float,
        default=16.0,
        help="亮度低于该值算暗区；偏色在暗区表现最明显",
    )
    parser.add_argument(
        "--bright-threshold-dn",
        type=float,
        default=120.0,
        help="亮度高于该值算亮区（手臂、方块等）",
    )
    parser.add_argument("--histogram-bins", type=int, default=64, help="对比图里直方图的箱数")
    parser.add_argument("--no-figure", action="store_true", help="只写 JSON，不画对比图")
    parser.add_argument("--output-dir", help="默认 runs/TIMESTAMP_sim_real_image_stats")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_images < 1:
        raise ValueError("--max-images 必须为正")
    if not 0.0 <= args.dark_threshold_dn < args.bright_threshold_dn <= 255.0:
        raise ValueError("需要 0 <= --dark-threshold-dn < --bright-threshold-dn <= 255")
    if args.histogram_bins < 2:
        raise ValueError("--histogram-bins 至少为 2")


def load_image_source(path: Path, npz_key: str, max_images: int) -> tuple[np.ndarray, dict[str, Any]]:
    """从 PNG 文件、图像目录或 NPZ 载入 [N, H, W, 3] uint8 图像。"""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"图像来源不存在: {path}")

    if path.is_dir():
        # 一个 rollout 目录里要么是逐帧 PNG（真机部署产物），要么是逐 episode 的 NPZ
        # （TacEx collect_rma_student_rollouts.py 产物）。后者需要跨 episode 拼接。
        archives = sorted(entry for entry in path.iterdir() if entry.suffix.lower() == ".npz")
        if archives:
            stacked = [_load_npz_images(entry, npz_key) for entry in archives]
            array = np.concatenate(stacked, axis=0)
            selected = _subsample_indices(array.shape[0], max_images)
            images = array[selected]
            source = {
                "kind": "npz_directory",
                "npz_key": npz_key,
                "archive_count": len(archives),
                "available": int(array.shape[0]),
            }
        else:
            files = sorted(
                entry for entry in path.iterdir() if entry.suffix.lower() in IMAGE_SUFFIXES
            )
            if not files:
                raise ValueError(f"目录里既没有图像文件也没有 NPZ: {path}")
            selected = _subsample_indices(len(files), max_images)
            images = np.stack(
                [np.asarray(Image.open(files[i]).convert("RGB")) for i in selected]
            )
            source = {"kind": "directory", "available": len(files)}
    elif path.suffix.lower() == ".npz":
        array = _load_npz_images(path, npz_key)
        selected = _subsample_indices(array.shape[0], max_images)
        images = array[selected]
        source = {"kind": "npz", "npz_key": npz_key, "available": int(array.shape[0])}
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        images = np.asarray(Image.open(path).convert("RGB"))[None]
        source = {"kind": "image", "available": 1}
    else:
        raise ValueError(f"不支持的图像来源后缀: {path.suffix!r}")

    images = np.ascontiguousarray(images)
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{path} 解析出的形状是 {images.shape}，期望 [N, H, W, 3]")
    if images.dtype != np.uint8:
        raise ValueError(f"{path} 的图像 dtype 是 {images.dtype}，期望 uint8")

    source.update({"path": str(path), "used": int(images.shape[0])})
    return images, source


def _load_npz_images(path: Path, npz_key: str) -> np.ndarray:
    with np.load(path) as data:
        if npz_key not in data:
            raise KeyError(f"{path} 中没有键 {npz_key!r}，可用键: {sorted(data.keys())}")
        array = np.asarray(data[npz_key])
    return array[None] if array.ndim == 3 else array


def _subsample_indices(count: int, max_images: int) -> np.ndarray:
    if count <= max_images:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, max_images).astype(np.int64))


def luminance(images: np.ndarray) -> np.ndarray:
    values = images.astype(np.float32)
    return (
        LUMINANCE_WEIGHTS[0] * values[..., 0]
        + LUMINANCE_WEIGHTS[1] * values[..., 1]
        + LUMINANCE_WEIGHTS[2] * values[..., 2]
    )


def percentile_from_histogram(histogram: np.ndarray, percentile: float) -> float:
    """从 256 箱的整数直方图取分位数；亮度已按 DN 取整，1 DN 的分辨率足够。"""
    total = float(histogram.sum())
    if total <= 0.0:
        return float("nan")
    cumulative = np.cumsum(histogram) / total
    return float(np.searchsorted(cumulative, percentile / 100.0))


def wasserstein_distance_dn(histogram_a: np.ndarray, histogram_b: np.ndarray) -> float:
    """两个亮度直方图之间的 Wasserstein-1 距离，单位 DN。

    支撑集是间隔为 1 的整数灰度，所以 W1 就等于两条 CDF 之差的绝对值之和。
    物理含义：把一侧的亮度分布搬成另一侧，平均每个像素需要移动多少 DN。
    """
    cumulative_a = np.cumsum(histogram_a) / max(float(histogram_a.sum()), 1e-12)
    cumulative_b = np.cumsum(histogram_b) / max(float(histogram_b.sum()), 1e-12)
    return float(np.abs(cumulative_a - cumulative_b).sum())


def kolmogorov_smirnov(histogram_a: np.ndarray, histogram_b: np.ndarray) -> float:
    cumulative_a = np.cumsum(histogram_a) / max(float(histogram_a.sum()), 1e-12)
    cumulative_b = np.cumsum(histogram_b) / max(float(histogram_b.sum()), 1e-12)
    return float(np.max(np.abs(cumulative_a - cumulative_b)))


def _channel_ratios(channel_means: list[float]) -> dict[str, float | None]:
    red, green, blue = channel_means
    return {
        "green_over_red": green / red if red > 1e-6 else None,
        "green_over_blue": green / blue if blue > 1e-6 else None,
    }


def image_statistics(
    images: np.ndarray,
    *,
    dark_threshold_dn: float,
    bright_threshold_dn: float,
) -> dict[str, Any]:
    """逐块统计亮度直方图、通道均值和明暗分区的通道平衡。"""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"期望 [N, H, W, 3] 的图像堆栈，得到 {images.shape}")

    histogram = np.zeros(256, dtype=np.float64)
    channel_total = np.zeros(3, dtype=np.float64)
    dark_total = np.zeros(3, dtype=np.float64)
    bright_total = np.zeros(3, dtype=np.float64)
    dark_count = 0
    bright_count = 0
    pixel_count = 0
    mean_image = np.zeros(images.shape[1:], dtype=np.float64)

    for start in range(0, images.shape[0], CHUNK_SIZE):
        chunk = images[start : start + CHUNK_SIZE].astype(np.float32)
        mean_image += chunk.sum(axis=0, dtype=np.float64)
        chunk_luminance = (
            LUMINANCE_WEIGHTS[0] * chunk[..., 0]
            + LUMINANCE_WEIGHTS[1] * chunk[..., 1]
            + LUMINANCE_WEIGHTS[2] * chunk[..., 2]
        )
        histogram += np.bincount(
            np.clip(np.rint(chunk_luminance), 0.0, 255.0).astype(np.int64).reshape(-1),
            minlength=256,
        )
        channel_total += chunk.sum(axis=(0, 1, 2), dtype=np.float64)
        pixel_count += int(chunk_luminance.size)

        dark_mask = chunk_luminance < dark_threshold_dn
        bright_mask = chunk_luminance > bright_threshold_dn
        dark_count += int(np.count_nonzero(dark_mask))
        bright_count += int(np.count_nonzero(bright_mask))
        for index in range(3):
            dark_total[index] += float(chunk[..., index][dark_mask].sum(dtype=np.float64))
            bright_total[index] += float(chunk[..., index][bright_mask].sum(dtype=np.float64))

    mean_image /= images.shape[0]
    overall_channels = (channel_total / max(pixel_count, 1)).tolist()
    dark_channels = (dark_total / max(dark_count, 1)).tolist()
    bright_channels = (bright_total / max(bright_count, 1)).tolist()

    mean_luminance = float(
        sum(weight * value for weight, value in zip(LUMINANCE_WEIGHTS, overall_channels))
    )
    return {
        "image_count": int(images.shape[0]),
        "image_shape": [int(value) for value in images.shape[1:]],
        "luminance": {
            "mean_dn": mean_luminance,
            "percentiles_dn": {
                f"p{percentile:g}": percentile_from_histogram(histogram, percentile)
                for percentile in PERCENTILES
            },
            "fraction_below": {
                f"dn_{threshold}": float(histogram[:threshold].sum() / max(histogram.sum(), 1e-12))
                for threshold in (16, 32, 64, 128, 200)
            },
        },
        "channels": {
            "overall_mean_dn": dict(zip(CHANNEL_LABELS, overall_channels)),
            "overall_ratios": _channel_ratios(overall_channels),
            "dark_region": {
                "threshold_dn": float(dark_threshold_dn),
                "pixel_fraction": dark_count / max(pixel_count, 1),
                "mean_dn": dict(zip(CHANNEL_LABELS, dark_channels)),
                "ratios": _channel_ratios(dark_channels),
            },
            "bright_region": {
                "threshold_dn": float(bright_threshold_dn),
                "pixel_fraction": bright_count / max(pixel_count, 1),
                "mean_dn": dict(zip(CHANNEL_LABELS, bright_channels)),
                "ratios": _channel_ratios(bright_channels),
            },
        },
        "_histogram": histogram,
        "_mean_image": mean_image,
    }


def _ratio_delta(real: float | None, sim: float | None) -> float | None:
    if real is None or sim is None:
        return None
    return sim - real


def compare_statistics(real: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    """把两侧统计量做差；符号统一为 sim - real。"""
    real_percentiles = real["luminance"]["percentiles_dn"]
    sim_percentiles = sim["luminance"]["percentiles_dn"]
    return {
        "luminance_mean_delta_dn": sim["luminance"]["mean_dn"] - real["luminance"]["mean_dn"],
        "luminance_percentile_delta_dn": {
            key: sim_percentiles[key] - real_percentiles[key] for key in real_percentiles
        },
        "luminance_wasserstein_dn": wasserstein_distance_dn(
            real["_histogram"], sim["_histogram"]
        ),
        "luminance_kolmogorov_smirnov": kolmogorov_smirnov(
            real["_histogram"], sim["_histogram"]
        ),
        "channel_mean_delta_dn": {
            label: sim["channels"]["overall_mean_dn"][label]
            - real["channels"]["overall_mean_dn"][label]
            for label in CHANNEL_LABELS
        },
        "dark_region_ratio_delta": {
            key: _ratio_delta(
                real["channels"]["dark_region"]["ratios"][key],
                sim["channels"]["dark_region"]["ratios"][key],
            )
            for key in ("green_over_red", "green_over_blue")
        },
        "bright_region_ratio_delta": {
            key: _ratio_delta(
                real["channels"]["bright_region"]["ratios"][key],
                sim["channels"]["bright_region"]["ratios"][key],
            )
            for key in ("green_over_red", "green_over_blue")
        },
        "dark_pixel_fraction_delta": (
            sim["channels"]["dark_region"]["pixel_fraction"]
            - real["channels"]["dark_region"]["pixel_fraction"]
        ),
    }


def _write_figure(
    path: Path,
    real: dict[str, Any],
    sim: dict[str, Any],
    comparison: dict[str, Any],
    histogram_bins: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0))
    for axis, stats, title in (
        (axes[0][0], real, "real (temporal mean)"),
        (axes[0][1], sim, "sim (temporal mean)"),
    ):
        axis.imshow(np.clip(np.rint(stats["_mean_image"]), 0, 255).astype(np.uint8))
        axis.set_title(f"{title}\nmean luminance {stats['luminance']['mean_dn']:.1f} DN")
        axis.axis("off")

    # 直方图按 DN 分箱后再合并显示，纵轴取对数，否则近黑的那一峰会压掉其余部分。
    edges = np.linspace(0.0, 256.0, histogram_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    group = np.clip(np.digitize(np.arange(256), edges[1:-1]), 0, histogram_bins - 1)
    axis = axes[1][0]
    for stats, label, color in ((real, "real", "tab:blue"), (sim, "sim", "tab:orange")):
        binned = np.bincount(group, weights=stats["_histogram"], minlength=histogram_bins)
        axis.step(centers, binned / binned.sum(), where="mid", label=label, color=color)
    axis.set_yscale("log")
    axis.set_xlabel("luminance (DN)")
    axis.set_ylabel("pixel fraction (log)")
    axis.set_title("luminance histogram")
    axis.legend()
    axis.grid(alpha=0.3)

    axis = axes[1][1]
    for stats, label, color in ((real, "real", "tab:blue"), (sim, "sim", "tab:orange")):
        cumulative = np.cumsum(stats["_histogram"]) / stats["_histogram"].sum()
        axis.plot(np.arange(256), cumulative, label=label, color=color)
    axis.set_xlabel("luminance (DN)")
    axis.set_ylabel("CDF")
    axis.set_title(
        f"luminance CDF\nW1 = {comparison['luminance_wasserstein_dn']:.1f} DN, "
        f"KS = {comparison['luminance_kolmogorov_smirnov']:.3f}"
    )
    axis.legend()
    axis.grid(alpha=0.3)

    axis = axes[1][2]
    positions = np.arange(3)
    for offset, (stats, label, color) in enumerate(
        ((real, "real", "tab:blue"), (sim, "sim", "tab:orange"))
    ):
        dark = [stats["channels"]["dark_region"]["mean_dn"][key] for key in CHANNEL_LABELS]
        bright = [stats["channels"]["bright_region"]["mean_dn"][key] for key in CHANNEL_LABELS]
        axis.bar(positions + 0.2 * offset - 0.1, dark, width=0.2, color=color, label=f"{label} dark")
        axis.bar(
            positions + 3.5 + 0.2 * offset - 0.1,
            bright,
            width=0.2,
            color=color,
            alpha=0.55,
            label=f"{label} bright",
        )
    axis.set_xticks(list(positions) + list(positions + 3.5))
    axis.set_xticklabels([label[0].upper() for label in CHANNEL_LABELS] * 2)
    axis.set_ylabel("channel mean (DN)")
    axis.set_title("channel balance: dark region (left) / bright region (right)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")

    axes[0][2].axis("off")
    summary = [
        "sim - real",
        f"  mean luminance   {comparison['luminance_mean_delta_dn']:+.1f} DN",
        f"  median (p50)     {comparison['luminance_percentile_delta_dn']['p50']:+.0f} DN",
        f"  p90              {comparison['luminance_percentile_delta_dn']['p90']:+.0f} DN",
        f"  W1 distance      {comparison['luminance_wasserstein_dn']:.1f} DN",
        f"  dark pixel frac  {comparison['dark_pixel_fraction_delta']:+.3f}",
        "",
        "channel mean delta (DN)",
    ]
    summary.extend(
        f"  {label:<6} {comparison['channel_mean_delta_dn'][label]:+.1f}"
        for label in CHANNEL_LABELS
    )
    summary.append("")
    summary.append("G/R ratio        real     sim")
    for region, label in (("dark_region", "dark"), ("bright_region", "bright")):
        summary.append(
            f"  {label:<8} "
            f"{_format_ratio(real['channels'][region]['ratios']['green_over_red']):>8} "
            f"{_format_ratio(sim['channels'][region]['ratios']['green_over_red']):>8}"
        )
    axes[0][2].text(
        0.0,
        1.0,
        "\n".join(summary),
        family="monospace",
        fontsize=10,
        va="top",
    )

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def _format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _print_report(real: dict[str, Any], sim: dict[str, Any], comparison: dict[str, Any]) -> None:
    print("\n=== 亮度分布（DN）===", flush=True)
    print(f"{'':>10} {'real':>10} {'sim':>10} {'sim-real':>10}")
    print(
        f"{'mean':>10} {real['luminance']['mean_dn']:10.1f} "
        f"{sim['luminance']['mean_dn']:10.1f} "
        f"{comparison['luminance_mean_delta_dn']:+10.1f}"
    )
    for key in real["luminance"]["percentiles_dn"]:
        print(
            f"{key:>10} {real['luminance']['percentiles_dn'][key]:10.0f} "
            f"{sim['luminance']['percentiles_dn'][key]:10.0f} "
            f"{comparison['luminance_percentile_delta_dn'][key]:+10.0f}"
        )
    print(
        f"\n  Wasserstein-1 = {comparison['luminance_wasserstein_dn']:.1f} DN"
        f"   (仿真整体平均需平移这么多 DN 才能对上真机)"
    )
    print(f"  Kolmogorov-Smirnov = {comparison['luminance_kolmogorov_smirnov']:.3f}")

    print("\n=== 暗区像素占比 ===", flush=True)
    for key in real["luminance"]["fraction_below"]:
        threshold = key.split("_")[1]
        print(
            f"  < {threshold:>3} DN: real {100.0 * real['luminance']['fraction_below'][key]:5.1f}%"
            f"   sim {100.0 * sim['luminance']['fraction_below'][key]:5.1f}%"
        )

    print("\n=== 通道平衡 ===", flush=True)
    for region in ("dark_region", "bright_region"):
        name = "暗区" if region == "dark_region" else "亮区"
        real_region = real["channels"][region]
        sim_region = sim["channels"][region]
        print(
            f"  {name}(占比 real {100.0 * real_region['pixel_fraction']:.1f}% / "
            f"sim {100.0 * sim_region['pixel_fraction']:.1f}%)"
        )
        for source, section in (("real", real_region), ("sim", sim_region)):
            means = section["mean_dn"]
            print(
                f"    {source:<4} R/G/B = {means['red']:7.2f} {means['green']:7.2f} "
                f"{means['blue']:7.2f}   G/R = {_format_ratio(section['ratios']['green_over_red'])}"
                f"   G/B = {_format_ratio(section['ratios']['green_over_blue'])}"
            )


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)

    real_images, real_source = load_image_source(
        Path(args.real), args.npz_key, args.max_images
    )
    sim_images, sim_source = load_image_source(Path(args.sim), args.npz_key, args.max_images)
    if real_images.shape[1:] != sim_images.shape[1:]:
        raise ValueError(
            f"两侧图像尺寸不同：real {real_images.shape[1:]} vs sim {sim_images.shape[1:]}。"
            "两边都必须是模型输入域的图像；真机侧请用 runs/<部署运行>/rgb/ 或 "
            "measure_camera_noise.py 的 mean_*.png"
        )

    print(
        f"[compare] real: {real_source['used']}/{real_source['available']} 张 "
        f"({real_source['kind']}) {real_source['path']}",
        flush=True,
    )
    print(
        f"[compare] sim : {sim_source['used']}/{sim_source['available']} 张 "
        f"({sim_source['kind']}) {sim_source['path']}",
        flush=True,
    )

    real_stats = image_statistics(
        real_images,
        dark_threshold_dn=args.dark_threshold_dn,
        bright_threshold_dn=args.bright_threshold_dn,
    )
    sim_stats = image_statistics(
        sim_images,
        dark_threshold_dn=args.dark_threshold_dn,
        bright_threshold_dn=args.bright_threshold_dn,
    )
    comparison = compare_statistics(real_stats, sim_stats)
    _print_report(real_stats, sim_stats, comparison)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_sim_real_image_stats"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_path = output_dir / "comparison.png"
    if not args.no_figure:
        _write_figure(figure_path, real_stats, sim_stats, comparison, args.histogram_bins)

    serializable = {
        "kind": "sim_real_image_statistics",
        "real_source": real_source,
        "sim_source": sim_source,
        "dark_threshold_dn": args.dark_threshold_dn,
        "bright_threshold_dn": args.bright_threshold_dn,
        "real": {key: value for key, value in real_stats.items() if not key.startswith("_")},
        "sim": {key: value for key, value in sim_stats.items() if not key.startswith("_")},
        "comparison": comparison,
    }
    report_path = output_dir / "image_stats.json"
    report_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n[compare] 报告: {report_path}")
    if not args.no_figure:
        print(f"[compare] 对比图: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
