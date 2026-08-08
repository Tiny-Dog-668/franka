#!/usr/bin/env python3
"""Record Franka external wrench around a reflex/error trigger.

This script is read-only: it connects to the Franka and samples robot state,
but does not command arm or gripper motion.  It keeps a pre-trigger ring buffer
and, when the requested error appears, records a short post-trigger window.

Example:
  python scripts/robot/record_franka_reflex_wrench.py --ip 172.16.0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from franky import RealtimeConfig, Robot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "franka_reflex_wrench"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Continuously record Franka external wrench and save data around a reflex/error trigger."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP"),
        help="Robot IP address. Falls back to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode. 'ignore' is recommended for read-only logging.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=50.0,
        help="Target state sampling rate in Hz.",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=3.0,
        help="Seconds of samples to keep before the trigger.",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=2.0,
        help="Seconds of samples to record after the trigger.",
    )
    parser.add_argument(
        "--trigger-error",
        default="cartesian_reflex",
        help="Error name that triggers saving. Default: cartesian_reflex.",
    )
    parser.add_argument(
        "--trigger-any-error",
        action="store_true",
        help="Trigger on any current/last motion error, not only --trigger-error.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where trigger event folders are written.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds if no trigger occurs. Use 0 to wait forever.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between live status prints.",
    )
    return parser


def active_error_names(errors: Any) -> list[str]:
    names: list[str] = []
    for name in dir(errors):
        if name.startswith("_"):
            continue
        value = getattr(errors, name)
        if isinstance(value, bool) and value:
            names.append(name)
    return sorted(names)


def mode_name(mode: Any) -> str:
    return str(mode).split(".")[-1]


def vector_norm(values: list[float] | np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def sample_state(robot: Robot, sample_index: int, start_monotonic: float) -> dict[str, Any]:
    now_unix = time.time()
    now_monotonic = time.monotonic()
    state = robot.state
    pose = robot.current_pose.end_effector_pose

    wrench = [float(value) for value in state.O_F_ext_hat_K.tolist()]
    force = wrench[:3]
    torque = wrench[3:]
    current_errors = active_error_names(state.current_errors)
    last_motion_errors = active_error_names(state.last_motion_errors)

    return {
        "sample_index": sample_index,
        "time_unix_s": now_unix,
        "elapsed_s": now_monotonic - start_monotonic,
        "robot_mode": mode_name(state.robot_mode),
        "has_errors": bool(robot.has_errors),
        "is_in_control": bool(robot.is_in_control),
        "control_command_success_rate": float(state.control_command_success_rate),
        "q": [float(value) for value in state.q.tolist()],
        "dq": [float(value) for value in state.dq.tolist()],
        "tcp_translation": [float(value) for value in pose.translation.tolist()],
        "tcp_quaternion_xyzw": [float(value) for value in pose.quaternion.tolist()],
        "wrench": wrench,
        "force_norm_n": vector_norm(force),
        "torque_norm_nm": vector_norm(torque),
        "current_errors": current_errors,
        "last_motion_errors": last_motion_errors,
    }


def sample_triggers(sample: dict[str, Any], trigger_error: str, trigger_any_error: bool) -> bool:
    current_errors = set(sample["current_errors"])
    last_motion_errors = set(sample["last_motion_errors"])
    all_errors = current_errors | last_motion_errors
    if trigger_any_error and all_errors:
        return True
    if trigger_error and trigger_error in all_errors:
        return True
    # A Reflex robot mode can be observed even when the detailed error list is
    # not stable across samples. Treat it as a useful safety net for the default.
    return trigger_error == "cartesian_reflex" and sample["robot_mode"] == "Reflex"


def baseline_from_pretrigger(samples: list[dict[str, Any]], trigger_elapsed_s: float) -> np.ndarray:
    pre = [sample for sample in samples if sample["elapsed_s"] < trigger_elapsed_s]
    if not pre:
        pre = samples
    wrenches = np.asarray([sample["wrench"] for sample in pre], dtype=np.float64)
    return np.median(wrenches, axis=0)


def add_delta_fields(samples: list[dict[str, Any]], baseline: np.ndarray) -> None:
    for sample in samples:
        wrench = np.asarray(sample["wrench"], dtype=np.float64)
        delta = wrench - baseline
        sample["delta_wrench"] = delta.tolist()
        sample["delta_force_norm_n"] = vector_norm(delta[:3])
        sample["delta_torque_norm_nm"] = vector_norm(delta[3:])


def flatten_sample(sample: dict[str, Any], trigger_elapsed_s: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_index": sample["sample_index"],
        "time_unix_s": f"{sample['time_unix_s']:.9f}",
        "elapsed_s": f"{sample['elapsed_s']:.9f}",
        "relative_to_trigger_s": f"{sample['elapsed_s'] - trigger_elapsed_s:.9f}",
        "robot_mode": sample["robot_mode"],
        "has_errors": sample["has_errors"],
        "is_in_control": sample["is_in_control"],
        "control_command_success_rate": f"{sample['control_command_success_rate']:.6f}",
        "force_norm_n": f"{sample['force_norm_n']:.9f}",
        "torque_norm_nm": f"{sample['torque_norm_nm']:.9f}",
        "delta_force_norm_n": f"{sample['delta_force_norm_n']:.9f}",
        "delta_torque_norm_nm": f"{sample['delta_torque_norm_nm']:.9f}",
        "current_errors": "|".join(sample["current_errors"]),
        "last_motion_errors": "|".join(sample["last_motion_errors"]),
    }
    for index, value in enumerate(sample["q"]):
        row[f"q{index}"] = f"{value:.12g}"
    for index, value in enumerate(sample["dq"]):
        row[f"dq{index}"] = f"{value:.12g}"
    for name, value in zip(("tcp_x_m", "tcp_y_m", "tcp_z_m"), sample["tcp_translation"]):
        row[name] = f"{value:.12g}"
    for name, value in zip(("tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw"), sample["tcp_quaternion_xyzw"]):
        row[name] = f"{value:.12g}"
    for name, value in zip(("fx_n", "fy_n", "fz_n", "mx_nm", "my_nm", "mz_nm"), sample["wrench"]):
        row[name] = f"{value:.12g}"
    for name, value in zip(
        ("dfx_n", "dfy_n", "dfz_n", "dmx_nm", "dmy_nm", "dmz_nm"),
        sample["delta_wrench"],
    ):
        row[name] = f"{value:.12g}"
    return row


def summarize(samples: list[dict[str, Any]], baseline: np.ndarray, trigger_index: int) -> dict[str, Any]:
    force_norms = np.asarray([sample["force_norm_n"] for sample in samples], dtype=np.float64)
    torque_norms = np.asarray([sample["torque_norm_nm"] for sample in samples], dtype=np.float64)
    delta_force_norms = np.asarray([sample["delta_force_norm_n"] for sample in samples], dtype=np.float64)
    delta_torque_norms = np.asarray([sample["delta_torque_norm_nm"] for sample in samples], dtype=np.float64)
    trigger_sample = samples[trigger_index]
    return {
        "sample_count": len(samples),
        "trigger_sample_index_in_file": trigger_index,
        "trigger_robot_sample_index": trigger_sample["sample_index"],
        "trigger_elapsed_s": trigger_sample["elapsed_s"],
        "trigger_current_errors": trigger_sample["current_errors"],
        "trigger_last_motion_errors": trigger_sample["last_motion_errors"],
        "baseline_wrench": baseline.tolist(),
        "trigger_wrench": trigger_sample["wrench"],
        "trigger_force_norm_n": trigger_sample["force_norm_n"],
        "trigger_torque_norm_nm": trigger_sample["torque_norm_nm"],
        "trigger_delta_wrench": trigger_sample["delta_wrench"],
        "trigger_delta_force_norm_n": trigger_sample["delta_force_norm_n"],
        "trigger_delta_torque_norm_nm": trigger_sample["delta_torque_norm_nm"],
        "max_force_norm_n": float(np.max(force_norms)),
        "max_torque_norm_nm": float(np.max(torque_norms)),
        "max_delta_force_norm_n": float(np.max(delta_force_norms)),
        "max_delta_torque_norm_nm": float(np.max(delta_torque_norms)),
    }


def write_event(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[dict[str, Any]],
    trigger_sample: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    event_dir = output_dir / f"{timestamp}_reflex_wrench"
    event_dir.mkdir(parents=True, exist_ok=False)

    trigger_elapsed_s = float(trigger_sample["elapsed_s"])
    baseline = baseline_from_pretrigger(samples, trigger_elapsed_s)
    add_delta_fields(samples, baseline)
    trigger_index = next(
        index
        for index, sample in enumerate(samples)
        if sample["sample_index"] == trigger_sample["sample_index"]
    )
    summary = summarize(samples, baseline, trigger_index)
    summary["settings"] = {
        "ip": args.ip,
        "realtime": args.realtime,
        "sample_rate_hz": args.sample_rate_hz,
        "pre_seconds": args.pre_seconds,
        "post_seconds": args.post_seconds,
        "trigger_error": args.trigger_error,
        "trigger_any_error": args.trigger_any_error,
    }

    csv_path = event_dir / "samples.csv"
    rows = [flatten_sample(sample, trigger_elapsed_s) for sample in samples]
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = event_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return event_dir, summary


def main() -> int:
    args = build_parser().parse_args()
    if not args.ip:
        print("Please pass --ip <robot_ip> or set FRANKA_ROBOT_IP.")
        return 1
    if args.sample_rate_hz <= 0.0:
        raise ValueError("--sample-rate-hz must be positive")
    if args.pre_seconds < 0.0 or args.post_seconds < 0.0:
        raise ValueError("--pre-seconds and --post-seconds must be non-negative")
    if args.max_seconds < 0.0:
        raise ValueError("--max-seconds must be non-negative")

    realtime_config = (
        RealtimeConfig.Ignore
        if args.realtime == "ignore"
        else RealtimeConfig.Enforce
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    robot = Robot(args.ip, realtime_config=realtime_config)
    print(f"Connected to Franka at {args.ip}")
    print(f"Realtime mode: {args.realtime}")
    print(
        f"Recording at target {args.sample_rate_hz:g} Hz; "
        f"pre={args.pre_seconds:g}s, post={args.post_seconds:g}s"
    )
    print(
        "Trigger: "
        + ("any error" if args.trigger_any_error else f"error '{args.trigger_error}' or Reflex mode")
    )
    print(f"Output dir: {output_dir}")
    print("Waiting for trigger. Press Ctrl-C to stop without saving.")

    period_s = 1.0 / args.sample_rate_hz
    pre_capacity = max(1, int(math.ceil(args.pre_seconds * args.sample_rate_hz)) + 1)
    post_count = int(math.ceil(args.post_seconds * args.sample_rate_hz))
    ring: deque[dict[str, Any]] = deque(maxlen=pre_capacity)

    start_monotonic = time.monotonic()
    next_sample_time = start_monotonic
    next_print_time = start_monotonic
    sample_index = 0
    triggered = False
    trigger_sample: dict[str, Any] | None = None
    post_remaining = 0

    try:
        while True:
            now = time.monotonic()
            if now < next_sample_time:
                time.sleep(min(next_sample_time - now, 0.01))
                continue
            sample_index += 1
            sample = sample_state(robot, sample_index, start_monotonic)
            ring.append(sample)
            next_sample_time += period_s

            if now >= next_print_time:
                errors = sample["current_errors"] or sample["last_motion_errors"]
                print(
                    f"t={sample['elapsed_s']:7.3f}s mode={sample['robot_mode']:<8} "
                    f"|F|={sample['force_norm_n']:6.2f}N "
                    f"|M|={sample['torque_norm_nm']:6.2f}Nm "
                    f"errors={errors if errors else 'none'}"
                )
                next_print_time = now + max(args.print_interval, period_s)

            if not triggered and sample_triggers(sample, args.trigger_error, args.trigger_any_error):
                triggered = True
                trigger_sample = sample
                post_remaining = post_count
                print(
                    "\nTrigger detected: "
                    f"sample={sample['sample_index']} t={sample['elapsed_s']:.3f}s "
                    f"mode={sample['robot_mode']} current={sample['current_errors']} "
                    f"last={sample['last_motion_errors']}"
                )
                if post_remaining == 0:
                    break
            elif triggered:
                post_remaining -= 1
                if post_remaining <= 0:
                    break

            if (
                not triggered
                and args.max_seconds > 0.0
                and sample["elapsed_s"] >= args.max_seconds
            ):
                print(f"No trigger within {args.max_seconds:g}s; nothing saved.")
                return 2
    except KeyboardInterrupt:
        print("\nStopped by user before trigger save.")
        return 130

    if trigger_sample is None:
        print("Internal error: trigger_sample is missing.")
        return 1

    samples = list(ring)
    event_dir, summary = write_event(output_dir, args, samples, trigger_sample)
    print(f"\nSaved reflex wrench event: {event_dir}")
    print(
        "Trigger raw: "
        f"|F|={summary['trigger_force_norm_n']:.3f} N, "
        f"|M|={summary['trigger_torque_norm_nm']:.3f} Nm"
    )
    print(
        "Trigger delta from pre-window median: "
        f"|dF|={summary['trigger_delta_force_norm_n']:.3f} N, "
        f"|dM|={summary['trigger_delta_torque_norm_nm']:.3f} Nm"
    )
    print(
        "Window max delta: "
        f"|dF|max={summary['max_delta_force_norm_n']:.3f} N, "
        f"|dM|max={summary['max_delta_torque_norm_nm']:.3f} Nm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
