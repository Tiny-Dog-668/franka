#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from franky import (
    Affine,
    CartesianMotion,
    ControlException,
    Frame,
    RealtimeConfig,
    ReferenceType,
    Robot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Move the Franka through small relative motions while sampling external "
            "wrench, to estimate motion-induced force/torque measurement artifacts."
        )
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address. Defaults to FRANKA_ROBOT_IP or 172.16.0.2.",
    )
    parser.add_argument("--dx", type=float, default=0.005, help="Relative X motion in meters.")
    parser.add_argument("--dy", type=float, default=0.0, help="Relative Y motion in meters.")
    parser.add_argument("--dz", type=float, default=0.0, help="Relative Z motion in meters.")
    parser.add_argument("--yaw", type=float, default=0.0, help="Relative yaw motion in degrees.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of forward motions. With --return-motion, each cycle also moves back.",
    )
    parser.add_argument(
        "--return-motion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After each forward motion, execute the opposite return motion. Default is one-way only.",
    )
    parser.add_argument("--speed", type=float, default=0.03, help="Relative dynamics factor.")
    parser.add_argument(
        "--motion-frequency",
        type=float,
        default=1.0,
        help="Maximum motion command start frequency in Hz.",
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("start", "linear", "exp"),
        default="linear",
        help=(
            "Baseline compensation mode. 'start' freezes the start baseline; "
            "'linear' post-processes each motion with A/B endpoint baseline interpolation; "
            "'exp' uses an exponential A/B endpoint baseline."
        ),
    )
    parser.add_argument(
        "--baseline-exp-tau",
        type=float,
        default=0.2,
        help="Shape parameter for --baseline-mode exp. Smaller values move toward B faster.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=30,
        help="Number of samples used whenever a local baseline is updated.",
    )
    parser.add_argument("--sample-interval", type=float, default=0.02, help="Sampling interval in seconds.")
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.2,
        help="Print one live wrench line at this interval. Use 0 to disable live printing.",
    )
    parser.add_argument(
        "--raw-output",
        action="store_true",
        help="Print live raw O_F_ext_hat_K force/torque only, without baseline-relative deltas.",
    )
    parser.add_argument(
        "--contact-force-threshold",
        type=float,
        default=6.0,
        help="Mark contact during motion when |dF| exceeds this threshold in newtons.",
    )
    parser.add_argument(
        "--contact-min-samples",
        type=int,
        default=3,
        help="Number of consecutive over-threshold motion samples required for contact.",
    )
    parser.add_argument(
        "--stop-on-contact",
        action="store_true",
        help="Call robot.stop() after contact is latched. Default is to keep moving and record.",
    )
    parser.add_argument("--settle-time", type=float, default=0.5, help="Static dwell between motions.")
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "runs"),
        help="Base directory for diagnostic outputs.",
    )
    parser.add_argument(
        "--auto-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically generate annotated PNG plots after the run.",
    )
    parser.add_argument(
        "--plot-components",
        nargs="+",
        choices=("norm", "fx", "fy", "fz"),
        default=["norm", "fx", "fy", "fz"],
        help="Force components to plot when --auto-plot is enabled.",
    )
    parser.add_argument(
        "--plot-x",
        choices=("progress", "time"),
        default="progress",
        help="X-axis used by automatically generated plots.",
    )
    parser.add_argument(
        "--plot-baseline-source",
        choices=("auto", "motion", "exp", "settled"),
        default="auto",
        help="Baseline source used by automatically generated plots. 'auto' follows --baseline-mode.",
    )
    parser.add_argument(
        "--compare-jacobian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also estimate external wrench from tau_ext_hat_filtered via the model "
            "Jacobian and store comparison columns in samples.csv."
        ),
    )
    parser.add_argument(
        "--jacobian-frame",
        choices=("stiffness", "end_effector", "flange"),
        default="stiffness",
        help=(
            "Frame used for the zero Jacobian. 'stiffness' best matches "
            "O_F_ext_hat_K."
        ),
    )
    parser.add_argument(
        "--jacobian-mode",
        choices=("full", "force"),
        default="full",
        help=(
            "'full' estimates 6D wrench [F,T]. 'force' estimates only [Fx,Fy,Fz] "
            "assuming no external torque at the chosen frame."
        ),
    )
    parser.add_argument(
        "--jacobian-damping",
        type=float,
        default=0.01,
        help="Damping lambda for the least-squares Jacobian inverse.",
    )
    return parser


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def average(samples: list[list[float]]) -> list[float]:
    return [sum(sample[index] for sample in samples) / len(samples) for index in range(6)]


def wrench(robot: Robot) -> list[float]:
    return robot.state.O_F_ext_hat_K.tolist()


def frame_from_name(name: str) -> Frame:
    if name == "stiffness":
        return Frame.Stiffness
    if name == "end_effector":
        return Frame.EndEffector
    if name == "flange":
        return Frame.Flange
    raise ValueError(f"Unsupported Jacobian frame: {name}")


def damped_least_squares(a: np.ndarray, b: np.ndarray, damping: float) -> np.ndarray:
    normal = a.T @ a
    if damping > 0.0:
        normal = normal + (damping * damping) * np.eye(normal.shape[0])
    rhs = a.T @ b
    try:
        return np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(a) @ b


def jacobian_wrench(model, state, frame: Frame, mode: str, damping: float) -> list[float]:
    raw_jacobian = np.asarray(model.zero_jacobian(frame, state), dtype=float)
    if raw_jacobian.shape == (6, 7):
        jacobian = raw_jacobian
    else:
        jacobian = raw_jacobian.reshape(6, 7, order="F")
    tau_ext = np.asarray(state.tau_ext_hat_filtered, dtype=float).reshape(7)
    if mode == "force":
        force = damped_least_squares(jacobian[:3, :].T, tau_ext, damping)
        return [float(force[0]), float(force[1]), float(force[2]), 0.0, 0.0, 0.0]
    estimated = damped_least_squares(jacobian.T, tau_ext, damping)
    return [float(value) for value in estimated]


def tcp_translation(robot: Robot) -> list[float]:
    return robot.current_pose.end_effector_pose.translation.tolist()


def fmt(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value: .{precision}f}" for value in values) + "]"


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


def translation_delta(
    start_translation: list[float] | None,
    end_translation: list[float] | None,
) -> list[float] | None:
    if start_translation is None or end_translation is None:
        return None
    return [
        end_translation[index] - start_translation[index]
        for index in range(3)
    ]


def motion_progress(
    current_translation: list[float] | None,
    start_translation: list[float] | None,
    delta_translation: list[float] | None,
) -> float | None:
    if current_translation is None or start_translation is None:
        return None
    if delta_translation is None:
        return None
    denom = sum(value * value for value in delta_translation)
    if denom <= 1e-12:
        return None
    relative = [
        current_translation[index] - start_translation[index]
        for index in range(3)
    ]
    return min(max(sum(relative[index] * delta_translation[index] for index in range(3)) / denom, 0.0), 1.0)


def live_header(raw_output: bool = False, compare_jacobian: bool = False) -> str:
    if raw_output:
        header = (
            "time[s] | phase        | raw_force[N]             | |rawF|[N] | "
            "raw_torque[Nm]          | |rawT|[Nm] | tcp[m]"
        )
        if compare_jacobian:
            header += " | J_force[N]              | J-O force[N]"
        return header
    header = (
        "time[s] | phase        | raw_force[N]             | raw_torque[Nm]          | "
        "|dF|[N] | d_force[N]               | |dT|[Nm] | d_torque[Nm]            | "
        "baseline | contact"
    )
    if compare_jacobian:
        header += " | |JdF|[N] | J_d_force[N]"
    return header


def make_motion(dx: float, dy: float, dz: float, yaw_deg: float, speed: float) -> CartesianMotion:
    yaw_rad = math.radians(yaw_deg)
    yaw_quaternion = [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]
    return CartesianMotion(
        Affine([dx, dy, dz], yaw_quaternion),
        ReferenceType.Relative,
        speed,
    )


def summarize(rows: list[dict]) -> dict:
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)

    summary = {}
    for phase, phase_rows in by_phase.items():
        force_norms = [row["delta_force_norm_n"] for row in phase_rows]
        torque_norms = [row["delta_torque_norm_nm"] for row in phase_rows]
        summary[phase] = {
            "samples": len(phase_rows),
            "force_norm_mean_n": statistics.fmean(force_norms),
            "force_norm_max_n": max(force_norms),
            "torque_norm_mean_nm": statistics.fmean(torque_norms),
            "torque_norm_max_nm": max(torque_norms),
        }
    return summary


def export_force_plot_data(run_dir: Path, rows: list[dict]) -> Path:
    plot_rows = []
    for row in rows:
        if not row.get("motion_id"):
            continue
        if row.get("post_interp_progress") is None:
            continue

        raw_force = [row["fx_n"], row["fy_n"], row["fz_n"]]
        baseline_force = [
            row["interp_baseline_fx_n"],
            row["interp_baseline_fy_n"],
            row["interp_baseline_fz_n"],
        ]
        corrected_force = [
            row["corrected_fx_n"],
            row["corrected_fy_n"],
            row["corrected_fz_n"],
        ]
        residual_force = [
            row["post_interp_residual_dfx_n"],
            row["post_interp_residual_dfy_n"],
            row["post_interp_residual_dfz_n"],
        ]
        plot_rows.append(
            {
                "motion_id": row["motion_id"],
                "phase": row["phase"],
                "time_s": row["time_s"],
                "progress": row["post_interp_progress"],
                "raw_fx_n": raw_force[0],
                "raw_fy_n": raw_force[1],
                "raw_fz_n": raw_force[2],
                "raw_force_norm_n": vector_norm(raw_force),
                "linear_baseline_fx_n": baseline_force[0],
                "linear_baseline_fy_n": baseline_force[1],
                "linear_baseline_fz_n": baseline_force[2],
                "linear_baseline_force_norm_n": vector_norm(baseline_force),
                "corrected_true_fx_n": corrected_force[0],
                "corrected_true_fy_n": corrected_force[1],
                "corrected_true_fz_n": corrected_force[2],
                "corrected_true_force_norm_n": vector_norm(corrected_force),
                "residual_fx_n": residual_force[0],
                "residual_fy_n": residual_force[1],
                "residual_fz_n": residual_force[2],
                "residual_force_norm_n": vector_norm(residual_force),
            }
        )

    output_path = run_dir / "force_plot_data.csv"
    if plot_rows:
        fieldnames = list(plot_rows[0])
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(plot_rows)
    else:
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            handle.write(
                "motion_id,phase,time_s,progress,raw_fx_n,raw_fy_n,raw_fz_n,"
                "raw_force_norm_n,linear_baseline_fx_n,linear_baseline_fy_n,"
                "linear_baseline_fz_n,linear_baseline_force_norm_n,"
                "corrected_true_fx_n,corrected_true_fy_n,corrected_true_fz_n,"
                "corrected_true_force_norm_n,residual_fx_n,residual_fy_n,"
                "residual_fz_n,residual_force_norm_n\n"
            )
    return output_path


def generate_annotated_plots(args: argparse.Namespace, run_dir: Path) -> list[Path]:
    plot_script = Path(__file__).with_name("plot_franka_force_motion.py")
    output_paths: list[Path] = []
    baseline_source = args.plot_baseline_source
    if baseline_source == "auto":
        baseline_source = "exp" if args.baseline_mode == "exp" else "motion"
    for component in args.plot_components:
        command = [
            sys.executable,
            str(plot_script),
            str(run_dir),
            "--component",
            component,
            "--x",
            args.plot_x,
            "--baseline-source",
            baseline_source,
            "--annotate",
        ]
        if baseline_source == "exp":
            command.extend(["--exp-tau", str(args.baseline_exp_tau)])
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                "Plot generation failed for "
                f"component={component} with return code {result.returncode}."
            )
            continue
        output_paths.extend(sorted((run_dir / "plots").glob(f"*_{component}_{args.plot_x}_{baseline_source}.png")))
    return output_paths


def main() -> int:
    args = build_parser().parse_args()
    if args.cycles < 1:
        print("--cycles must be >= 1.")
        return 1
    if args.sample_interval <= 0.0:
        print("--sample-interval must be > 0.")
        return 1
    if args.print_interval < 0.0:
        print("--print-interval must be >= 0.")
        return 1
    if args.motion_frequency <= 0.0:
        print("--motion-frequency must be > 0.")
        return 1
    if args.baseline_exp_tau <= 0.0:
        print("--baseline-exp-tau must be > 0.")
        return 1
    if args.contact_force_threshold < 0.0:
        print("--contact-force-threshold must be >= 0.")
        return 1
    if args.contact_min_samples < 1:
        print("--contact-min-samples must be >= 1.")
        return 1
    if args.jacobian_damping < 0.0:
        print("--jacobian-damping must be >= 0.")
        return 1
    if not any(value != 0.0 for value in (args.dx, args.dy, args.dz, args.yaw)):
        print("No motion requested. Pass at least one of --dx/--dy/--dz/--yaw.")
        return 1

    realtime_config = (
        RealtimeConfig.Ignore
        if args.realtime == "ignore"
        else RealtimeConfig.Enforce
    )
    robot = Robot(args.ip, realtime_config=realtime_config)
    robot.relative_dynamics_factor = args.speed
    model = None
    jacobian_frame = None
    if args.compare_jacobian:
        model = robot.model
        jacobian_frame = frame_from_name(args.jacobian_frame)

    print(f"Connected to Franka at {args.ip}")
    print(
        "Planned motion: "
        f"dx={args.dx:.4f} m, dy={args.dy:.4f} m, dz={args.dz:.4f} m, "
        f"yaw={args.yaw:.2f} deg, cycles={args.cycles}, speed={args.speed:.3f}, "
        f"motion_frequency={args.motion_frequency:.3f} Hz, "
        f"baseline_mode={args.baseline_mode}"
    )
    print(
        "Contact detection during move_out/move_back: "
        f"|dF| >= {args.contact_force_threshold:.3f} N for "
        f"{args.contact_min_samples} consecutive samples."
    )
    if args.baseline_mode in ("linear", "exp"):
        mode_text = "linear" if args.baseline_mode == "linear" else f"exponential tau={args.baseline_exp_tau:g}"
        print(
            f"{mode_text} baseline compensation is computed after each motion "
            "reaches the next settled endpoint."
        )
    if args.stop_on_contact:
        print("stop_on_contact: enabled")
    if args.compare_jacobian:
        print(
            "Jacobian compare: enabled "
            f"(frame={args.jacobian_frame}, mode={args.jacobian_mode}, "
            f"damping={args.jacobian_damping:g})"
        )

    run_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_force_during_motion"
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    baseline_updates: list[dict] = []
    motion_records: list[dict] = []
    stop_errors: list[dict] = []
    stop_event = threading.Event()
    phase_lock = threading.Lock()
    baseline_lock = threading.Lock()
    motion_lock = threading.Lock()
    print_lock = threading.Lock()
    current_phase = {"name": "pre_static"}
    baseline_state = {
        "values": [0.0] * 6,
        "jacobian_values": [0.0] * 6,
        "version": 0,
        "label": "unset",
    }
    active_motion_index = {"value": None}
    contact_consecutive = {"value": 0}
    stop_requested = {"value": False}
    start_time = time.perf_counter()
    last_print_time = {"value": 0.0}

    def set_phase(name: str) -> None:
        with phase_lock:
            current_phase["name"] = name
        if args.print_interval > 0.0:
            with print_lock:
                print(f"\n--- {name} ---")
                print(live_header(args.raw_output, args.compare_jacobian))
                last_print_time["value"] = 0.0

    def get_phase() -> str:
        with phase_lock:
            return current_phase["name"]

    def collect_baseline(
        label: str,
        endpoint: str | None = None,
    ) -> list[float]:
        set_phase(label)
        samples = []
        jacobian_samples = []
        for _ in range(args.baseline_samples):
            state = robot.state
            samples.append(state.O_F_ext_hat_K.tolist())
            if args.compare_jacobian:
                jacobian_samples.append(
                    jacobian_wrench(
                        model,
                        state,
                        jacobian_frame,
                        args.jacobian_mode,
                        args.jacobian_damping,
                    )
                )
            time.sleep(args.sample_interval)
        new_baseline = average(samples)
        new_jacobian_baseline = (
            average(jacobian_samples) if args.compare_jacobian else [0.0] * 6
        )
        try:
            baseline_translation = tcp_translation(robot)
        except Exception:
            baseline_translation = None
        with baseline_lock:
            baseline_state["values"] = new_baseline
            baseline_state["jacobian_values"] = new_jacobian_baseline
            baseline_state["version"] += 1
            baseline_state["label"] = label
            baseline_version = baseline_state["version"]
        baseline_updates.append(
            {
                "time_s": time.perf_counter() - start_time,
                "label": label,
                "version": baseline_version,
                "baseline": new_baseline,
                "jacobian_baseline": new_jacobian_baseline if args.compare_jacobian else None,
                "endpoint": endpoint,
                "tcp_translation": baseline_translation,
            }
        )
        with print_lock:
            endpoint_text = f" endpoint={endpoint}" if endpoint else ""
            print(
                f"Baseline #{baseline_version} ({label}) "
                f"force[N]={fmt(new_baseline[:3])} torque[Nm]={fmt(new_baseline[3:])}"
                f"{endpoint_text}"
            )
            if args.compare_jacobian:
                print(
                    f"  Jacobian baseline force[N]={fmt(new_jacobian_baseline[:3])} "
                    f"torque[Nm]={fmt(new_jacobian_baseline[3:])}"
                )
        return new_baseline, baseline_translation

    def wait_for_motion_tick(next_start_time: float) -> None:
        remaining_s = next_start_time - time.perf_counter()
        if remaining_s > 0.0:
            set_phase("motion_tick_wait")
            time.sleep(remaining_s)

    def begin_motion(
        cycle: int,
        direction: str,
        phase: str,
        delta_translation: list[float],
        target_endpoint: str,
    ) -> None:
        motion_id = f"{direction}_{cycle}"
        with baseline_lock:
            baseline_version = int(baseline_state["version"])
            baseline_label = str(baseline_state["label"])
            start_baseline = list(baseline_state["values"])
        start_translation = tcp_translation(robot)
        with motion_lock:
            active_motion_index["value"] = len(motion_records)
            contact_consecutive["value"] = 0
            stop_requested["value"] = False
            motion_records.append(
                {
        "motion_id": motion_id,
                    "cycle": cycle,
                    "direction": direction,
                    "phase": phase,
                    "baseline_version": baseline_version,
                    "baseline_label": baseline_label,
                    "baseline_mode": args.baseline_mode,
                    "start_baseline": start_baseline,
                    "target_endpoint": target_endpoint,
                    "target_baseline": None,
                    "post_interp_available": False,
                    "start_time_s": time.perf_counter() - start_time,
                    "end_time_s": None,
                    "duration_s": None,
                    "start_translation": start_translation,
                    "target_translation": None,
                    "actual_delta_translation": None,
                    "delta_translation": delta_translation,
                    "sample_count": 0,
                    "max_force_norm_n": 0.0,
                    "max_force_time_s": None,
                    "max_force_delta_n": None,
                    "max_start_force_norm_n": 0.0,
                    "max_start_force_time_s": None,
                    "max_start_force_delta_n": None,
                    "max_torque_norm_nm": 0.0,
                    "max_torque_time_s": None,
                    "max_torque_delta_nm": None,
                    "post_interp_max_force_norm_n": None,
                    "post_interp_max_force_time_s": None,
                    "post_interp_max_force_delta_n": None,
                    "post_interp_contact": None,
                    "post_interp_contact_time_s": None,
                    "contact": False,
                    "contact_time_s": None,
                    "contact_force_norm_n": None,
                    "contact_torque_norm_nm": None,
                    "contact_delta_force_n": None,
                    "contact_delta_torque_nm": None,
                    "stop_requested": False,
                }
            )
        set_phase(phase)

    def end_motion() -> dict | None:
        with motion_lock:
            index = active_motion_index["value"]
            if index is None:
                return None
            record = motion_records[index]
            end_time_s = time.perf_counter() - start_time
            record["end_time_s"] = end_time_s
            record["duration_s"] = end_time_s - record["start_time_s"]
            active_motion_index["value"] = None
            contact_consecutive["value"] = 0
            stop_requested["value"] = False
            return dict(record)

    def print_motion_record(record: dict | None) -> None:
        if record is None:
            return
        contact = "YES" if record["contact"] else "no"
        with print_lock:
            print(
                "Motion result: "
                f"{record['motion_id']} samples={record['sample_count']} "
                f"max|dF|={record['max_force_norm_n']:.3f} N "
                f"max|dT|={record['max_torque_norm_nm']:.3f} Nm "
                f"contact={contact} source=start"
            )

    def postprocess_motion_interpolation(
        record: dict | None,
        target_baseline: list[float],
        target_translation: list[float] | None,
    ) -> None:
        if record is None:
            return
        actual_delta_translation = translation_delta(
            record["start_translation"],
            target_translation,
        )
        force_over_count = 0
        contact = False
        contact_time = None
        max_force_norm = 0.0
        max_force_time = None
        max_force_delta = None
        for row in rows:
            if row.get("motion_id") != record["motion_id"]:
                continue
            current_translation = None
            if row.get("tcp_x") is not None:
                current_translation = [row["tcp_x"], row["tcp_y"], row["tcp_z"]]
            progress = motion_progress(
                current_translation,
                record["start_translation"],
                actual_delta_translation,
            )
            if progress is None:
                continue
            if args.baseline_mode == "exp":
                interp_baseline = exp_lerp(
                    record["start_baseline"],
                    target_baseline,
                    progress,
                    args.baseline_exp_tau,
                )
            else:
                interp_baseline = lerp(record["start_baseline"], target_baseline, progress)
            sample = [
                row["fx_n"],
                row["fy_n"],
                row["fz_n"],
                row["tx_nm"],
                row["ty_nm"],
                row["tz_nm"],
            ]
            # Compensate only the position-dependent baseline drift from A to s.
            # This keeps the actual A-point wrench level instead of forcing A to zero.
            baseline_shift = subtract(interp_baseline, record["start_baseline"])
            interp_delta = subtract(sample, baseline_shift)
            residual_delta = subtract(sample, interp_baseline)
            interp_force_norm = vector_norm(interp_delta[:3])
            row["post_interp_progress"] = progress
            row["interp_baseline_fx_n"] = interp_baseline[0]
            row["interp_baseline_fy_n"] = interp_baseline[1]
            row["interp_baseline_fz_n"] = interp_baseline[2]
            row["interp_baseline_force_norm_n"] = vector_norm(interp_baseline[:3])
            row["baseline_shift_fx_n"] = baseline_shift[0]
            row["baseline_shift_fy_n"] = baseline_shift[1]
            row["baseline_shift_fz_n"] = baseline_shift[2]
            row["baseline_shift_force_norm_n"] = vector_norm(baseline_shift[:3])
            row["post_interp_dfx_n"] = interp_delta[0]
            row["post_interp_dfy_n"] = interp_delta[1]
            row["post_interp_dfz_n"] = interp_delta[2]
            row["post_interp_delta_force_norm_n"] = interp_force_norm
            row["corrected_fx_n"] = interp_delta[0]
            row["corrected_fy_n"] = interp_delta[1]
            row["corrected_fz_n"] = interp_delta[2]
            row["corrected_force_norm_n"] = interp_force_norm
            row["post_interp_residual_dfx_n"] = residual_delta[0]
            row["post_interp_residual_dfy_n"] = residual_delta[1]
            row["post_interp_residual_dfz_n"] = residual_delta[2]
            row["post_interp_residual_force_norm_n"] = vector_norm(residual_delta[:3])
            if interp_force_norm > max_force_norm:
                max_force_norm = interp_force_norm
                max_force_time = row["time_s"]
                max_force_delta = interp_delta[:3]
            if interp_force_norm >= args.contact_force_threshold:
                force_over_count += 1
            else:
                force_over_count = 0
            if not contact and force_over_count >= args.contact_min_samples:
                contact = True
                contact_time = row["time_s"]

        with motion_lock:
            for stored_record in motion_records:
                if stored_record["motion_id"] == record["motion_id"]:
                    stored_record["post_interp_available"] = True
                    stored_record["target_baseline"] = target_baseline
                    stored_record["target_translation"] = target_translation
                    stored_record["actual_delta_translation"] = actual_delta_translation
                    stored_record["post_interp_target_baseline"] = target_baseline
                    stored_record["post_interp_max_force_norm_n"] = max_force_norm
                    stored_record["post_interp_max_force_time_s"] = max_force_time
                    stored_record["post_interp_max_force_delta_n"] = max_force_delta
                    stored_record["post_interp_contact"] = contact
                    stored_record["post_interp_contact_time_s"] = contact_time
                    break
        with print_lock:
            contact_text = "YES" if contact else "no"
            print(
                "Post-interp result: "
                f"{record['motion_id']} max|dF|={max_force_norm:.3f} N "
                f"contact={contact_text} "
                f"({args.baseline_mode} baseline, position drift compensated, A wrench preserved)"
            )

    def sample_loop() -> None:
        while not stop_event.is_set():
            sample_time = time.perf_counter()
            try:
                state = robot.state
                sample = state.O_F_ext_hat_K.tolist()
                jacobian_sample = None
                if args.compare_jacobian:
                    jacobian_sample = jacobian_wrench(
                        model,
                        state,
                        jacobian_frame,
                        args.jacobian_mode,
                        args.jacobian_damping,
                    )
            except Exception as exc:
                rows.append(
                    {
                        "time_s": sample_time - start_time,
                        "phase": get_phase(),
                        "error": str(exc),
                    }
                )
                time.sleep(args.sample_interval)
                continue

            with baseline_lock:
                baseline = list(baseline_state["values"])
                jacobian_baseline = list(baseline_state["jacobian_values"])
                baseline_version = int(baseline_state["version"])
                baseline_label = str(baseline_state["label"])
            start_delta = subtract(sample, baseline)
            delta = list(start_delta)
            jacobian_delta = None
            jacobian_force_norm = None
            jacobian_force_diff = None
            if jacobian_sample is not None:
                jacobian_delta = subtract(jacobian_sample, jacobian_baseline)
                jacobian_force_norm = vector_norm(jacobian_delta[:3])
                jacobian_force_diff = [
                    jacobian_sample[index] - sample[index]
                    for index in range(3)
                ]
            baseline_source = "start"
            linear_progress = None
            current_translation = None
            delta_force_norm = vector_norm(delta[:3])
            delta_torque_norm = vector_norm(delta[3:])
            raw_force_norm = vector_norm(sample[:3])
            raw_torque_norm = vector_norm(sample[3:])
            phase = get_phase()
            elapsed_s = sample_time - start_time
            motion_id = ""
            contact_candidate = False
            contact_latched = False
            contact_just_latched = False
            stop_now = False
            if phase in ("move_out", "move_back"):
                try:
                    current_translation = tcp_translation(robot)
                except Exception:
                    current_translation = None
                with motion_lock:
                    index = active_motion_index["value"]
                    if index is not None:
                        record = motion_records[index]
                        motion_id = record["motion_id"]
                        linear_progress = motion_progress(
                            current_translation,
                            record["start_translation"],
                            record["delta_translation"],
                        )
                        contact_candidate = delta_force_norm >= args.contact_force_threshold
                        record["sample_count"] += 1
                        start_force_norm = vector_norm(start_delta[:3])
                        if start_force_norm > record["max_start_force_norm_n"]:
                            record["max_start_force_norm_n"] = start_force_norm
                            record["max_start_force_time_s"] = elapsed_s
                            record["max_start_force_delta_n"] = start_delta[:3]
                        if delta_force_norm > record["max_force_norm_n"]:
                            record["max_force_norm_n"] = delta_force_norm
                            record["max_force_time_s"] = elapsed_s
                            record["max_force_delta_n"] = delta[:3]
                        if delta_torque_norm > record["max_torque_norm_nm"]:
                            record["max_torque_norm_nm"] = delta_torque_norm
                            record["max_torque_time_s"] = elapsed_s
                            record["max_torque_delta_nm"] = delta[3:]

                        if contact_candidate:
                            contact_consecutive["value"] += 1
                        else:
                            contact_consecutive["value"] = 0

                        if record["contact"]:
                            contact_latched = True
                        elif contact_consecutive["value"] >= args.contact_min_samples:
                            record["contact"] = True
                            record["contact_time_s"] = elapsed_s
                            record["contact_force_norm_n"] = delta_force_norm
                            record["contact_torque_norm_nm"] = delta_torque_norm
                            record["contact_delta_force_n"] = delta[:3]
                            record["contact_delta_torque_nm"] = delta[3:]
                            contact_latched = True
                            contact_just_latched = True
                            if args.stop_on_contact and not stop_requested["value"]:
                                stop_requested["value"] = True
                                record["stop_requested"] = True
                                stop_now = True

            row = {
                "time_s": elapsed_s,
                "phase": phase,
                "motion_id": motion_id,
                "baseline_version": baseline_version,
                "baseline_label": baseline_label,
                "baseline_source": baseline_source,
                "linear_progress": linear_progress,
                "contact_candidate": contact_candidate,
                "contact_latched": contact_latched,
                "contact_just_latched": contact_just_latched,
                "tcp_x": current_translation[0] if current_translation is not None else None,
                "tcp_y": current_translation[1] if current_translation is not None else None,
                "tcp_z": current_translation[2] if current_translation is not None else None,
                "fx_n": sample[0],
                "fy_n": sample[1],
                "fz_n": sample[2],
                "tx_nm": sample[3],
                "ty_nm": sample[4],
                "tz_nm": sample[5],
                "raw_force_norm_n": raw_force_norm,
                "raw_torque_norm_nm": raw_torque_norm,
                "dfx_n": delta[0],
                "dfy_n": delta[1],
                "dfz_n": delta[2],
                "dtx_nm": delta[3],
                "dty_nm": delta[4],
                "dtz_nm": delta[5],
                "delta_force_norm_n": delta_force_norm,
                "delta_torque_norm_nm": delta_torque_norm,
                "start_dfx_n": start_delta[0],
                "start_dfy_n": start_delta[1],
                "start_dfz_n": start_delta[2],
                "start_delta_force_norm_n": vector_norm(start_delta[:3]),
            }
            if jacobian_sample is not None:
                row.update(
                    {
                        "j_fx_n": jacobian_sample[0],
                        "j_fy_n": jacobian_sample[1],
                        "j_fz_n": jacobian_sample[2],
                        "j_tx_nm": jacobian_sample[3],
                        "j_ty_nm": jacobian_sample[4],
                        "j_tz_nm": jacobian_sample[5],
                        "j_raw_force_norm_n": vector_norm(jacobian_sample[:3]),
                        "j_raw_torque_norm_nm": vector_norm(jacobian_sample[3:]),
                        "j_dfx_n": jacobian_delta[0],
                        "j_dfy_n": jacobian_delta[1],
                        "j_dfz_n": jacobian_delta[2],
                        "j_dtx_nm": jacobian_delta[3],
                        "j_dty_nm": jacobian_delta[4],
                        "j_dtz_nm": jacobian_delta[5],
                        "j_delta_force_norm_n": jacobian_force_norm,
                        "j_delta_torque_norm_nm": vector_norm(jacobian_delta[3:]),
                        "j_minus_o_fx_n": jacobian_force_diff[0],
                        "j_minus_o_fy_n": jacobian_force_diff[1],
                        "j_minus_o_fz_n": jacobian_force_diff[2],
                        "j_minus_o_force_norm_n": vector_norm(jacobian_force_diff),
                    }
                )
            rows.append(row)
            if stop_now:
                try:
                    robot.stop()
                except Exception as exc:
                    stop_errors.append(
                        {
                            "time_s": elapsed_s,
                            "motion_id": motion_id,
                            "error": str(exc),
                        }
                    )
            if (
                args.print_interval > 0.0
                and (
                    sample_time - last_print_time["value"] >= args.print_interval
                    or contact_just_latched
                )
            ):
                contact_text = ""
                if contact_just_latched:
                    contact_text = "CONTACT"
                elif contact_latched:
                    contact_text = "contact"
                elif contact_candidate:
                    contact_text = "candidate"
                with print_lock:
                    if args.raw_output:
                        tcp_text = "-"
                        if current_translation is not None:
                            tcp_text = fmt(current_translation)
                        print(
                            f"{elapsed_s:7.3f} | {phase:12s} | {fmt(sample[:3])} | "
                            f"{raw_force_norm:9.3f} | {fmt(sample[3:])} | "
                            f"{raw_torque_norm:10.3f} | {tcp_text}",
                            end="",
                        )
                        if jacobian_sample is not None:
                            print(
                                f" | {fmt(jacobian_sample[:3])} | "
                                f"{fmt(jacobian_force_diff)}"
                            )
                        else:
                            print()
                    else:
                        print(
                            f"{elapsed_s:7.3f} | {phase:12s} | {fmt(sample[:3])} | "
                            f"{fmt(sample[3:])} | {delta_force_norm:7.3f} | "
                            f"{fmt(delta[:3])} | {delta_torque_norm:8.3f} | "
                            f"{fmt(delta[3:])} | {baseline_source:8s} | {contact_text}",
                            end="",
                        )
                        if jacobian_delta is not None:
                            print(
                                f" | {jacobian_force_norm:8.3f} | "
                                f"{fmt(jacobian_delta[:3])}"
                            )
                        else:
                            print()
                    last_print_time["value"] = sample_time
            time.sleep(args.sample_interval)

    sampler = threading.Thread(target=sample_loop, daemon=True)

    try:
        print("Collecting initial static baseline...")
        collect_baseline("initial_baseline", endpoint="home")
        sampler.start()

        set_phase("pre_static")
        time.sleep(args.settle_time)
        motion_period_s = 1.0 / args.motion_frequency
        next_motion_start = time.perf_counter()
        for cycle in range(args.cycles):
            wait_for_motion_tick(next_motion_start)
            current_endpoint = "home" if cycle == 0 else f"forward_{cycle}"
            next_endpoint = f"forward_{cycle + 1}"
            with print_lock:
                print(f"\nCycle {cycle + 1}/{args.cycles}: move forward")
            collect_baseline(
                f"baseline_static_before_move_forward_{cycle + 1}",
                endpoint=current_endpoint,
            )
            begin_motion(
                cycle + 1,
                "forward",
                "move_out",
                [args.dx, args.dy, args.dz],
                next_endpoint,
            )
            motion_start = time.perf_counter()
            motion_record = None
            try:
                robot.move(make_motion(args.dx, args.dy, args.dz, args.yaw, args.speed))
            finally:
                motion_record = end_motion()
                print_motion_record(motion_record)
            next_motion_start = motion_start + motion_period_s

            set_phase("settle_forward")
            time.sleep(args.settle_time)
            forward_baseline, forward_translation = collect_baseline(
                f"baseline_after_settle_forward_{cycle + 1}",
                endpoint=next_endpoint,
            )
            if args.baseline_mode in ("linear", "exp"):
                postprocess_motion_interpolation(motion_record, forward_baseline, forward_translation)

            if not args.return_motion:
                continue

            wait_for_motion_tick(next_motion_start)
            with print_lock:
                print(f"\nCycle {cycle + 1}/{args.cycles}: move back")
            begin_motion(
                cycle + 1,
                "back",
                "move_back",
                [-args.dx, -args.dy, -args.dz],
                current_endpoint,
            )
            motion_start = time.perf_counter()
            motion_record = None
            try:
                robot.move(make_motion(-args.dx, -args.dy, -args.dz, -args.yaw, args.speed))
            finally:
                motion_record = end_motion()
                print_motion_record(motion_record)
            next_motion_start = motion_start + motion_period_s

            set_phase("settle_back")
            time.sleep(args.settle_time)
            home_baseline, home_translation = collect_baseline(
                f"baseline_after_settle_back_{cycle + 1}",
                endpoint=current_endpoint,
            )
            if args.baseline_mode in ("linear", "exp"):
                postprocess_motion_interpolation(motion_record, home_baseline, home_translation)

        set_phase("post_static")
        time.sleep(args.settle_time)
    except ControlException as exc:
        print(f"Motion failed: {exc}")
        return_code = 1
    finally:
        stop_event.set()
        if sampler.is_alive():
            sampler.join(timeout=2.0)

    return_code = locals().get("return_code", 0)
    summary = summarize([row for row in rows if "error" not in row])

    csv_path = run_dir / "samples.csv"
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    plot_data_path = export_force_plot_data(run_dir, rows)

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "args": vars(args),
                "baseline_updates": baseline_updates,
                "motion_records": motion_records,
                "summary": summary,
                "sample_errors": [row for row in rows if "error" in row],
                "stop_errors": stop_errors,
            },
            handle,
            indent=2,
        )

    plot_paths: list[Path] = []
    if args.auto_plot:
        print("\nGenerating annotated plots...")
        plot_paths = generate_annotated_plots(args, run_dir)

    print("\nSummary by phase:")
    for phase, phase_summary in summary.items():
        print(
            f"{phase:12s} samples={phase_summary['samples']:4d} "
            f"|dF| mean/max={phase_summary['force_norm_mean_n']:.3f}/"
            f"{phase_summary['force_norm_max_n']:.3f} N "
            f"|dT| mean/max={phase_summary['torque_norm_mean_nm']:.3f}/"
            f"{phase_summary['torque_norm_max_nm']:.3f} Nm"
        )
    print("\nMotion contact summary:")
    for record in motion_records:
        contact = "YES" if record["contact"] else "no"
        contact_time = (
            f"{record['contact_time_s']:.3f}s"
            if record["contact_time_s"] is not None
            else "-"
        )
        post_interp = "-"
        if record.get("post_interp_available"):
            post_contact = "YES" if record.get("post_interp_contact") else "no"
            post_interp = (
                f"{record['post_interp_max_force_norm_n']:.3f} N/"
                f"{post_contact}"
            )
        print(
            f"{record['motion_id']:8s} "
            f"max|dF|={record['max_force_norm_n']:.3f} N "
            f"start-max|dF|={record['max_start_force_norm_n']:.3f} N "
            f"post-interp={post_interp} "
            f"max|dT|={record['max_torque_norm_nm']:.3f} Nm "
            f"contact={contact} contact_time={contact_time}"
        )
    print(f"\nSaved samples to: {csv_path}")
    print(f"Saved plot data to: {plot_data_path}")
    print(f"Saved summary to: {summary_path}")
    if args.auto_plot:
        if plot_paths:
            print(f"Saved annotated plots under: {run_dir / 'plots'}")
        else:
            print("No annotated plots were generated.")
    print(
        "Interpretation: compare move_out samples against static baseline phases. "
        "Large no-contact increases during motion are motion-induced artifacts or "
        "dynamic loads, not necessarily external contact. Use --return-motion only "
        "when you explicitly want a back motion."
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
