#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot Franka motion force curves from force_plot_data.csv."
    )
    parser.add_argument(
        "run_dir",
        help="Run directory containing force_plot_data.csv.",
    )
    parser.add_argument(
        "--component",
        choices=("norm", "fx", "fy", "fz"),
        default="norm",
        help="Force component to plot.",
    )
    parser.add_argument(
        "--x",
        choices=("time", "progress"),
        default="progress",
        help="Use motion progress or time on the x-axis.",
    )
    parser.add_argument(
        "--baseline-source",
        choices=("motion", "exp", "settled"),
        default="motion",
        help=(
            "'motion' draws the baseline through the first and last raw motion samples; "
            "'exp' draws an endpoint-aligned exponential baseline through the raw samples; "
            "'settled' draws the baseline through the settled A/B endpoint baselines."
        ),
    )
    parser.add_argument(
        "--exp-tau",
        type=float,
        default=0.2,
        help="Shape parameter for --baseline-source exp. Smaller values rise faster.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for PNG plots. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively after saving.",
    )
    parser.add_argument(
        "--annotate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add motion parameters and summary values to each plot.",
    )
    return parser


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def force_component(values: list[float], component: str) -> float:
    if component == "norm":
        return math.sqrt(sum(value * value for value in values[:3]))
    index = {"fx": 0, "fy": 1, "fz": 2}[component]
    return values[index]


def raw_force(row: dict[str, str]) -> list[float]:
    return [
        as_float(row, "raw_fx_n"),
        as_float(row, "raw_fy_n"),
        as_float(row, "raw_fz_n"),
    ]


def lerp(a: list[float], b: list[float], progress: float) -> list[float]:
    clamped = min(max(progress, 0.0), 1.0)
    return [
        (1.0 - clamped) * a[index] + clamped * b[index]
        for index in range(len(a))
    ]


def exp_lerp(a: list[float], b: list[float], progress: float, tau: float) -> list[float]:
    clamped = min(max(progress, 0.0), 1.0)
    if tau <= 0.0:
        shaped = clamped
    else:
        denom = 1.0 - math.exp(-1.0 / tau)
        shaped = (1.0 - math.exp(-clamped / tau)) / denom if denom != 0.0 else clamped
    return lerp(a, b, shaped)


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(len(a))]


def y_columns(component: str) -> tuple[str, str, str, str]:
    suffix = "force_norm_n" if component == "norm" else f"{component}_n"
    label = "|F| [N]" if component == "norm" else f"{component.upper()} [N]"
    return (
        f"raw_{suffix}",
        f"linear_baseline_{suffix}",
        f"corrected_true_{suffix}",
        label,
    )


def fmt_list(values: list[float] | None, precision: int = 4) -> str:
    if values is None:
        return "-"
    return "[" + ", ".join(f"{value:.{precision}f}" for value in values) + "]"


def annotation_text(summary: dict, record: dict | None, motion_id: str) -> str:
    args = summary.get("args", {})
    command = [
        float(args.get("dx", 0.0) or 0.0),
        float(args.get("dy", 0.0) or 0.0),
        float(args.get("dz", 0.0) or 0.0),
    ]
    if record and record.get("direction") == "back":
        command = [-value for value in command]
        yaw = -float(args.get("yaw", 0.0) or 0.0)
    else:
        yaw = float(args.get("yaw", 0.0) or 0.0)

    if record:
        direction = record.get("direction", "-")
        cycle = record.get("cycle", "-")
        start_tcp = fmt_list(record.get("start_translation"))
        actual_delta = fmt_list(record.get("actual_delta_translation"))
        max_force = record.get("post_interp_max_force_norm_n")
        if max_force is None:
            max_force = record.get("max_force_norm_n")
        max_torque = record.get("max_torque_norm_nm")
        contact = record.get("post_interp_contact")
        if contact is None:
            contact = record.get("contact")
        duration = record.get("duration_s")
    else:
        direction = "-"
        cycle = "-"
        start_tcp = "-"
        actual_delta = "-"
        max_force = None
        max_torque = None
        contact = None
        duration = None

    force_text = "-" if max_force is None else f"{float(max_force):.4f} N"
    torque_text = "-" if max_torque is None else f"{float(max_torque):.4f} Nm"
    duration_text = "-" if duration is None else f"{float(duration):.3f} s"
    contact_text = "-" if contact is None else ("yes" if contact else "no")

    return "\n".join(
        [
            f"motion={motion_id}  direction={direction}  cycle={cycle}  speed={args.get('speed', '-')}",
            f"command delta[m]={fmt_list(command)}  yaw[deg]={yaw:.4f}  baseline_mode={args.get('baseline_mode', '-')}",
            f"start TCP[m]={start_tcp}  actual delta[m]={actual_delta}  duration={duration_text}",
            f"sample_interval={args.get('sample_interval', '-')} s  contact_threshold={args.get('contact_force_threshold', '-')} N",
            f"max |dF|={force_text}  max |dT|={torque_text}  contact={contact_text}",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    data_path = run_dir / "force_plot_data.csv"
    if not data_path.exists():
        print(f"Missing plot data: {data_path}")
        print("Run test_franka_force_during_motion.py again to generate it.")
        return 1

    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required. Install it in the active environment first.")
        return 1

    rows = load_rows(data_path)
    if not rows:
        print(f"No rows in {data_path}")
        return 1
    summary = load_summary(run_dir / "summary.json")
    records_by_motion = {
        record["motion_id"]: record
        for record in summary.get("motion_records", [])
    }

    by_motion: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_motion[row["motion_id"]].append(row)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_col, baseline_col, corrected_col, y_label = y_columns(args.component)
    x_col = "progress" if args.x == "progress" else "time_s"
    x_label = "Motion progress A->B" if args.x == "progress" else "Time [s]"

    for motion_id, motion_rows in sorted(by_motion.items()):
        motion_rows = sorted(motion_rows, key=lambda row: as_float(row, x_col))
        xs = [as_float(row, x_col) for row in motion_rows]
        raw = [as_float(row, raw_col) for row in motion_rows]
        record = records_by_motion.get(motion_id)
        if args.baseline_source in ("motion", "exp"):
            start_force = raw_force(motion_rows[0])
            end_force = raw_force(motion_rows[-1])
            if args.baseline_source == "exp":
                baseline_vectors = [
                    exp_lerp(start_force, end_force, as_float(row, "progress"), args.exp_tau)
                    for row in motion_rows
                ]
                baseline_label = f"exponential endpoint baseline (tau={args.exp_tau:g})"
            else:
                baseline_vectors = [
                    lerp(start_force, end_force, as_float(row, "progress"))
                    for row in motion_rows
                ]
                baseline_label = "motion endpoint baseline"
            corrected_vectors = [
                subtract(raw_force(row), baseline_vector)
                for row, baseline_vector in zip(motion_rows, baseline_vectors)
            ]
            baseline = [
                force_component(values, args.component)
                for values in baseline_vectors
            ]
            corrected = [
                force_component(values, args.component)
                for values in corrected_vectors
            ]
            corrected_label = "residual force"
        else:
            baseline = [as_float(row, baseline_col) for row in motion_rows]
            corrected = [as_float(row, corrected_col) for row in motion_rows]
            baseline_label = "settled A/B baseline"
            corrected_label = "corrected true force"

        fig, ax = plt.subplots(figsize=(11.5, 7.0 if args.annotate else 5.0))
        ax.plot(xs, raw, label="raw measured force", linewidth=1.8)
        ax.plot(xs, baseline, label=baseline_label, linewidth=1.8)
        ax.plot(xs, corrected, label=corrected_label, linewidth=2.2)
        if args.x == "progress" and args.baseline_source in ("motion", "exp"):
            ax.scatter([0.0, 1.0], [raw[0], raw[-1]], s=54, marker="o", color="tab:orange", zorder=5)
            ax.annotate("A raw", (0.0, raw[0]), xytext=(8, 8), textcoords="offset points", fontsize=9)
            ax.annotate("B raw", (1.0, raw[-1]), xytext=(-48, 8), textcoords="offset points", fontsize=9)
            print(f"{motion_id}: A raw={raw[0]:.4f}, B raw={raw[-1]:.4f}")
        elif record and args.x == "progress":
            a_value = force_component(record["start_baseline"], args.component)
            b_baseline = record.get("target_baseline") or record.get("post_interp_target_baseline")
            if b_baseline:
                b_value = force_component(b_baseline, args.component)
                ax.scatter([0.0, 1.0], [a_value, b_value], s=54, marker="o", color="tab:orange", zorder=5)
                ax.annotate("A baseline", (0.0, a_value), xytext=(8, 8), textcoords="offset points", fontsize=9)
                ax.annotate("B baseline", (1.0, b_value), xytext=(-72, 8), textcoords="offset points", fontsize=9)
                print(f"{motion_id}: A baseline={a_value:.4f}, B baseline={b_value:.4f}")
        if xs and raw and args.baseline_source == "settled":
            ax.scatter([xs[0], xs[-1]], [raw[0], raw[-1]], s=34, marker="x", color="tab:blue", zorder=5)
        ax.set_title(f"{motion_id} {args.component}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if args.annotate:
            fig.text(
                0.01,
                0.01,
                annotation_text(summary, record, motion_id),
                ha="left",
                va="bottom",
                fontsize=8.5,
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.45",
                    "facecolor": "white",
                    "edgecolor": "0.75",
                    "alpha": 0.94,
                },
            )
            fig.tight_layout(rect=[0.0, 0.24, 1.0, 1.0])
        else:
            fig.tight_layout()

        output_path = output_dir / f"{motion_id}_{args.component}_{args.x}_{args.baseline_source}.png"
        fig.savefig(output_path, dpi=160)
        print(f"Saved {output_path}")
        if args.show:
            plt.show()
        plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
