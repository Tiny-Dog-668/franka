#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from franky import (
    Affine,
    CartesianMotion,
    ControlException,
    RealtimeConfig,
    ReferenceType,
    Robot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
WRENCH_NAMES = ("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm")


@dataclass(frozen=True)
class TestPosition:
    name: str
    offset: tuple[float, float, float]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Franka 6D external wrench responses to small x/y/z "
            "translations at multiple Cartesian positions."
        )
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address. Defaults to FRANKA_ROBOT_IP or 172.16.0.2.",
    )
    parser.add_argument(
        "--position",
        action="append",
        dest="positions",
        default=[],
        metavar="NAME=DX,DY,DZ",
        help=(
            "Relative test position from the starting TCP pose, in meters. "
            "Can be passed multiple times. Example: --position x_plus=0.03,0,0"
        ),
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=("x", "y", "z"),
        default=["x", "y", "z"],
        help="Translation axes to probe at each position.",
    )
    parser.add_argument(
        "--probe-distance",
        type=float,
        default=0.005,
        help="Signed translation distance for each probe motion in meters.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=2,
        help="Number of plus/back cycles per axis at each position.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.03,
        help="Relative dynamics factor for all Cartesian motions.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=30,
        help="Number of static samples used as the local baseline before each axis probe.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.02,
        help="Sampling interval during baselines and motions, in seconds.",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.5,
        help="Static dwell after moving to a test position or finishing a probe.",
    )
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
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Move back to the starting TCP position at the end.",
    )
    parser.add_argument(
        "--abort-on-motion-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the diagnostic after the first rejected motion or position move.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the initial confirmation prompt.",
    )
    return parser


def default_positions() -> list[TestPosition]:
    return [
        TestPosition("center", (0.0, 0.0, 0.0)),
        TestPosition("x_plus", (0.03, 0.0, 0.0)),
        TestPosition("x_minus", (-0.03, 0.0, 0.0)),
        TestPosition("y_plus", (0.0, 0.03, 0.0)),
        TestPosition("y_minus", (0.0, -0.03, 0.0)),
        TestPosition("z_plus", (0.0, 0.0, 0.03)),
    ]


def parse_position(value: str) -> TestPosition:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Invalid position {value!r}; expected NAME=DX,DY,DZ."
        )
    name, offset_text = value.split("=", 1)
    parts = offset_text.split(",")
    if not name:
        raise argparse.ArgumentTypeError("Position name must not be empty.")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Invalid position {value!r}; expected three comma-separated offsets."
        )
    try:
        offset = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid numeric offset in position {value!r}."
        ) from exc
    return TestPosition(name, offset)  # type: ignore[arg-type]


def vector_norm(values: list[float] | tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values))


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [a[index] - b[index] for index in range(len(a))]


def average(samples: list[list[float]]) -> list[float]:
    return [sum(sample[index] for sample in samples) / len(samples) for index in range(6)]


def fmt(values: list[float] | tuple[float, ...], precision: int = 4) -> str:
    return "[" + ", ".join(f"{value: .{precision}f}" for value in values) + "]"


def wrench(robot: Robot) -> list[float]:
    return robot.state.O_F_ext_hat_K.tolist()


def tcp_translation(robot: Robot) -> list[float]:
    return robot.current_pose.end_effector_pose.translation.tolist()


def make_motion(dx: float, dy: float, dz: float, speed: float) -> CartesianMotion:
    return CartesianMotion(
        Affine([dx, dy, dz], [0.0, 0.0, 0.0, 1.0]),
        ReferenceType.Relative,
        speed,
    )


def axis_delta(axis: str, distance: float) -> tuple[float, float, float]:
    if axis == "x":
        return (distance, 0.0, 0.0)
    if axis == "y":
        return (0.0, distance, 0.0)
    return (0.0, 0.0, distance)


def add_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub_vec(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def collect_baseline(robot: Robot, sample_count: int, interval: float) -> tuple[list[float], list[float]]:
    samples = []
    for _ in range(sample_count):
        samples.append(wrench(robot))
        time.sleep(interval)
    return average(samples), tcp_translation(robot)


def summarize_motion(rows: list[dict]) -> dict:
    if not rows:
        return {
            "samples": 0,
            "mean_delta": [0.0] * 6,
            "max_abs_delta": [0.0] * 6,
            "max_force_norm_n": 0.0,
            "max_torque_norm_nm": 0.0,
        }

    deltas = [
        [
            row["dfx_n"],
            row["dfy_n"],
            row["dfz_n"],
            row["dtx_nm"],
            row["dty_nm"],
            row["dtz_nm"],
        ]
        for row in rows
    ]
    mean_delta = [
        statistics.fmean(delta[index] for delta in deltas)
        for index in range(6)
    ]
    max_abs_delta = [
        max(deltas, key=lambda delta: abs(delta[index]))[index]
        for index in range(6)
    ]
    return {
        "samples": len(rows),
        "mean_delta": mean_delta,
        "max_abs_delta": max_abs_delta,
        "max_force_norm_n": max(row["delta_force_norm_n"] for row in rows),
        "max_torque_norm_nm": max(row["delta_torque_norm_nm"] for row in rows),
    }


def run_sampled_motion(
    robot: Robot,
    run_rows: list[dict],
    baseline: list[float],
    metadata: dict,
    delta: tuple[float, float, float],
    sample_interval: float,
) -> dict:
    local_rows: list[dict] = []
    stop_event = threading.Event()
    start_time = time.perf_counter()

    def sample_loop() -> None:
        sample_index = 0
        while not stop_event.is_set():
            sample_time = time.perf_counter() - start_time
            try:
                sample = wrench(robot)
                tcp = tcp_translation(robot)
            except Exception as exc:
                row = {
                    **metadata,
                    "sample_index": sample_index,
                    "motion_time_s": sample_time,
                    "error": str(exc),
                }
                run_rows.append(row)
                local_rows.append(row)
                time.sleep(sample_interval)
                sample_index += 1
                continue

            wrench_delta = subtract(sample, baseline)
            row = {
                **metadata,
                "sample_index": sample_index,
                "motion_time_s": sample_time,
                "tcp_x": tcp[0],
                "tcp_y": tcp[1],
                "tcp_z": tcp[2],
                "fx_n": sample[0],
                "fy_n": sample[1],
                "fz_n": sample[2],
                "tx_nm": sample[3],
                "ty_nm": sample[4],
                "tz_nm": sample[5],
                "dfx_n": wrench_delta[0],
                "dfy_n": wrench_delta[1],
                "dfz_n": wrench_delta[2],
                "dtx_nm": wrench_delta[3],
                "dty_nm": wrench_delta[4],
                "dtz_nm": wrench_delta[5],
                "delta_force_norm_n": vector_norm(wrench_delta[:3]),
                "delta_torque_norm_nm": vector_norm(wrench_delta[3:]),
                "error": "",
            }
            run_rows.append(row)
            local_rows.append(row)
            time.sleep(sample_interval)
            sample_index += 1

    sampler = threading.Thread(target=sample_loop, daemon=True)
    start_tcp = tcp_translation(robot)
    sampler.start()
    error = ""
    try:
        robot.move(make_motion(*delta, args.speed))
    except ControlException as exc:
        error = str(exc)
    finally:
        stop_event.set()
        sampler.join(timeout=max(1.0, sample_interval * 5.0))
    end_tcp = tcp_translation(robot)

    summary = summarize_motion([row for row in local_rows if not row.get("error")])
    summary.update(
        {
            **metadata,
            "command_delta": list(delta),
            "start_tcp": start_tcp,
            "end_tcp": end_tcp,
            "actual_delta": [
                end_tcp[index] - start_tcp[index]
                for index in range(3)
            ],
            "duration_s": time.perf_counter() - start_time,
            "error": error,
        }
    )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "position_name",
        "position_offset_x",
        "position_offset_y",
        "position_offset_z",
        "axis",
        "cycle",
        "direction",
        "motion_id",
        "sample_index",
        "motion_time_s",
        "tcp_x",
        "tcp_y",
        "tcp_z",
        "fx_n",
        "fy_n",
        "fz_n",
        "tx_nm",
        "ty_nm",
        "tz_nm",
        "dfx_n",
        "dfy_n",
        "dfz_n",
        "dtx_nm",
        "dty_nm",
        "dtz_nm",
        "delta_force_norm_n",
        "delta_torque_norm_nm",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def try_relative_move(
    robot: Robot,
    delta: tuple[float, float, float],
    label: str,
) -> tuple[bool, str]:
    if vector_norm(delta) == 0.0:
        return True, ""
    try:
        robot.move(make_motion(*delta, args.speed))
    except ControlException as exc:
        return False, str(exc)
    return True, ""


def main() -> int:
    args = build_parser().parse_args()
    if args.probe_distance <= 0.0:
        print("--probe-distance must be > 0.")
        return 1
    if args.cycles < 1:
        print("--cycles must be >= 1.")
        return 1
    if args.baseline_samples < 1:
        print("--baseline-samples must be >= 1.")
        return 1
    if args.sample_interval <= 0.0:
        print("--sample-interval must be > 0.")
        return 1

    positions = [parse_position(value) for value in args.positions] if args.positions else default_positions()
    if len({position.name for position in positions}) != len(positions):
        print("Position names must be unique.")
        return 1

    print("This diagnostic will move the real robot through multiple relative positions.")
    print(f"Robot IP: {args.ip}")
    print(f"Axes: {', '.join(args.axes)}")
    print(f"Probe distance: {args.probe_distance:.4f} m, cycles per axis: {args.cycles}")
    print("Test positions relative to the starting TCP pose:")
    for position in positions:
        print(f"  {position.name}: {fmt(position.offset)} m")
    if not args.yes:
        confirm = input("Type YES to start the test: ")
        if confirm != "YES":
            print("Aborted.")
            return 1

    realtime_config = (
        RealtimeConfig.Ignore
        if args.realtime == "ignore"
        else RealtimeConfig.Enforce
    )
    robot = Robot(args.ip, realtime_config=realtime_config)
    robot.relative_dynamics_factor = args.speed

    run_dir = Path(args.output_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_wrench_translation_grid"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    records: list[dict] = []
    baselines: list[dict] = []
    run_errors: list[dict] = []
    current_offset = (0.0, 0.0, 0.0)
    start_tcp = tcp_translation(robot)
    exit_code = 0

    print(f"Connected. Start TCP: {fmt(start_tcp)}")
    try:
        for position in positions:
            if exit_code != 0 and args.abort_on_motion_error:
                break
            move_to_position = sub_vec(position.offset, current_offset)
            print(f"\nMoving to position {position.name}: delta={fmt(move_to_position)} m")
            if vector_norm(move_to_position) > 0.0:
                ok, error = try_relative_move(robot, move_to_position, f"position {position.name}")
                if not ok:
                    print(f"  position move error: {error}")
                    run_errors.append(
                        {
                            "stage": "position_move",
                            "position_name": position.name,
                            "delta": list(move_to_position),
                            "error": error,
                        }
                    )
                    exit_code = 1
                    if args.abort_on_motion_error:
                        break
                    continue
                current_offset = position.offset
                time.sleep(args.settle_time)

            for axis in args.axes:
                if exit_code != 0 and args.abort_on_motion_error:
                    break
                print(f"Collecting local baseline at {position.name}, axis {axis}.")
                baseline, baseline_tcp = collect_baseline(
                    robot,
                    args.baseline_samples,
                    args.sample_interval,
                )
                baselines.append(
                    {
                        "position_name": position.name,
                        "position_offset": list(position.offset),
                        "axis": axis,
                        "baseline": baseline,
                        "tcp_translation": baseline_tcp,
                    }
                )
                print(
                    f"Baseline force={fmt(baseline[:3])} N "
                    f"torque={fmt(baseline[3:])} Nm"
                )

                for cycle in range(1, args.cycles + 1):
                    for direction, sign in (("plus", 1.0), ("minus", -1.0)):
                        delta = axis_delta(axis, sign * args.probe_distance)
                        motion_id = f"{position.name}_{axis}_{cycle}_{direction}"
                        print(f"Motion {motion_id}: delta={fmt(delta)} m")
                        metadata = {
                            "position_name": position.name,
                            "position_offset_x": position.offset[0],
                            "position_offset_y": position.offset[1],
                            "position_offset_z": position.offset[2],
                            "axis": axis,
                            "cycle": cycle,
                            "direction": direction,
                            "motion_id": motion_id,
                        }
                        record = run_sampled_motion(
                            robot,
                            rows,
                            baseline,
                            metadata,
                            delta,
                            args.sample_interval,
                        )
                        records.append(record)
                        if not record["error"]:
                            current_offset = add_vec(current_offset, delta)
                        time.sleep(args.settle_time)
                        print(
                            f"  max|dF|={record['max_force_norm_n']:.4f} N, "
                            f"max|dT|={record['max_torque_norm_nm']:.4f} Nm, "
                            f"mean_delta={fmt(record['mean_delta'])}"
                        )
                        if record["error"]:
                            print(f"  motion error: {record['error']}")
                            run_errors.append(
                                {
                                    "stage": "probe_motion",
                                    "position_name": position.name,
                                    "axis": axis,
                                    "cycle": cycle,
                                    "direction": direction,
                                    "motion_id": motion_id,
                                    "delta": list(delta),
                                    "error": record["error"],
                                }
                            )
                            exit_code = 1
                            if args.abort_on_motion_error:
                                break
                    if exit_code != 0 and args.abort_on_motion_error:
                        break

        if args.return_home:
            home_delta = tuple(-value for value in current_offset)
            print(f"\nReturning to start position: delta={fmt(home_delta)} m")
            if vector_norm(home_delta) > 0.0:
                ok, error = try_relative_move(robot, home_delta, "return_home")
                if ok:
                    current_offset = (0.0, 0.0, 0.0)
                    time.sleep(args.settle_time)
                else:
                    print(f"  return-home error: {error}")
                    print("  Robot may still be near the last test pose; recover manually if needed.")
                    run_errors.append(
                        {
                            "stage": "return_home",
                            "delta": list(home_delta),
                            "error": error,
                        }
                    )
                    exit_code = 1
    finally:
        csv_path = run_dir / "samples.csv"
        summary_path = run_dir / "summary.json"
        write_csv(csv_path, rows)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "robot_ip": args.ip,
                    "start_tcp": start_tcp,
                    "final_tcp": tcp_translation(robot),
                    "final_offset_estimate": list(current_offset),
                    "positions": [
                        {"name": position.name, "offset": list(position.offset)}
                        for position in positions
                    ],
                    "axes": args.axes,
                    "probe_distance": args.probe_distance,
                    "cycles": args.cycles,
                    "speed": args.speed,
                    "baseline_samples": args.baseline_samples,
                    "sample_interval": args.sample_interval,
                    "abort_on_motion_error": args.abort_on_motion_error,
                    "run_errors": run_errors,
                    "baselines": baselines,
                    "motion_records": records,
                },
                handle,
                indent=2,
            )
        print(f"\nSaved samples to: {csv_path}")
        print(f"Saved summary to: {summary_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
