#!/usr/bin/env python3
"""Safely benchmark Franka Hand command and stop latency without moving the arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any


DEFAULT_IP = "172.16.0.2"
DEFAULT_SPEED_M_S = 0.03
DEFAULT_TRAVEL_M = 0.006
DEFAULT_STOP_AFTER_S = 0.05
MAX_SPEED_M_S = 0.05
MAX_TRAVEL_M = 0.010
MAX_STOP_AFTER_S = 0.20
MAX_REPEATS = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Franka Hand move/stop latency without connecting to or "
            "moving the robot arm. The default invocation is dry-run only."
        )
    )
    parser.add_argument(
        "--run-hardware",
        action="store_true",
        help="Connect to the Franka Hand and execute the bounded test.",
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", DEFAULT_IP),
        help="Robot/Hand IP address.",
    )
    parser.add_argument(
        "--mode",
        choices=("stop", "natural", "compare"),
        default="compare",
        help=(
            "stop: interrupt one bounded move; natural: wait for one bounded move; "
            "compare: stop a bounded move, then naturally return to its start width."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=("open", "close"),
        default="open",
        help="Direction of the first bounded move.",
    )
    parser.add_argument(
        "--travel-mm",
        type=float,
        default=DEFAULT_TRAVEL_M * 1000.0,
        help=f"Maximum commanded travel per move (1 to {MAX_TRAVEL_M * 1000:g} mm).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED_M_S,
        help=f"Gripper speed in m/s (maximum {MAX_SPEED_M_S:g} for this diagnostic).",
    )
    parser.add_argument(
        "--stop-after-ms",
        type=float,
        default=DEFAULT_STOP_AFTER_S * 1000.0,
        help=(
            "Delay after move_async returns before stop() is requested "
            f"(10 to {MAX_STOP_AFTER_S * 1000:g} ms)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=f"Number of trials (1 to {MAX_REPEATS}).",
    )
    parser.add_argument(
        "--natural-timeout-s",
        type=float,
        default=5.0,
        help="Timeout for a natural finite move before requesting stop.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Hardware runs otherwise use runs/gripper_latency/.",
    )
    return parser


def validate_parameters(args: argparse.Namespace) -> None:
    travel_m = float(args.travel_mm) / 1000.0
    stop_after_s = float(args.stop_after_ms) / 1000.0
    finite_values = {
        "--travel-mm": travel_m,
        "--speed": float(args.speed),
        "--stop-after-ms": stop_after_s,
        "--natural-timeout-s": float(args.natural_timeout_s),
    }
    for name, value in finite_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if not 0.001 <= travel_m <= MAX_TRAVEL_M:
        raise ValueError(f"--travel-mm must be in [1, {MAX_TRAVEL_M * 1000:g}]")
    if not 0.005 <= args.speed <= MAX_SPEED_M_S:
        raise ValueError(f"--speed must be in [0.005, {MAX_SPEED_M_S:g}]")
    if not 0.010 <= stop_after_s <= MAX_STOP_AFTER_S:
        raise ValueError(
            f"--stop-after-ms must be in [10, {MAX_STOP_AFTER_S * 1000:g}]"
        )
    if args.mode in {"stop", "compare"}:
        nominal_duration_s = travel_m / float(args.speed)
        if stop_after_s >= 0.8 * nominal_duration_s:
            raise ValueError(
                "--stop-after-ms must be below 80% of the nominal move duration; "
                "otherwise stop() may be requested after the bounded move has finished"
            )
    if not 1 <= args.repeats <= MAX_REPEATS:
        raise ValueError(f"--repeats must be in [1, {MAX_REPEATS}]")
    if not 0.5 <= args.natural_timeout_s <= 10.0:
        raise ValueError("--natural-timeout-s must be in [0.5, 10.0]")


def checked_state(gripper: Any) -> dict[str, Any]:
    state = gripper.state
    result = {
        "width_m": float(state.width),
        "max_width_m": float(state.max_width),
        "is_grasped": bool(state.is_grasped),
    }
    if not math.isfinite(result["width_m"]):
        raise RuntimeError("Hand reported a non-finite width")
    if not math.isfinite(result["max_width_m"]) or result["max_width_m"] <= 0.0:
        raise RuntimeError("Hand reported an invalid max_width; homing may be required")
    if not 0.0 <= result["width_m"] <= result["max_width_m"]:
        raise RuntimeError("Hand width is outside its reported physical range")
    return result


def bounded_target(
    width_m: float, max_width_m: float, direction: str, travel_m: float
) -> float:
    sign = 1.0 if direction == "open" else -1.0
    target = width_m + sign * travel_m
    if target < 0.0 or target > max_width_m:
        raise ValueError(
            f"Cannot move {direction} by {travel_m * 1000:.1f} mm from "
            f"{width_m * 1000:.1f} mm; reported range is "
            f"[0, {max_width_m * 1000:.1f}] mm. Choose the other direction "
            "or reduce --travel-mm."
        )
    return target


def _timed_state(gripper: Any) -> tuple[dict[str, Any], float]:
    started_ns = time.monotonic_ns()
    state = checked_state(gripper)
    return state, (time.monotonic_ns() - started_ns) / 1e6


def run_stop_trial(
    gripper: Any,
    *,
    direction: str,
    travel_m: float,
    speed_m_s: float,
    stop_after_s: float,
) -> dict[str, Any]:
    before, before_state_ms = _timed_state(gripper)
    target_m = bounded_target(
        before["width_m"], before["max_width_m"], direction, travel_m
    )
    future: Any | None = None
    stop_requested = False
    try:
        call_started_ns = time.monotonic_ns()
        future = gripper.move_async(target_m, speed_m_s)
        move_async_ms = (time.monotonic_ns() - call_started_ns) / 1e6
        time.sleep(stop_after_s)
        pending_before_stop = not bool(future.wait(0.0))
        if not pending_before_stop:
            raise RuntimeError(
                "Bounded move completed before stop() was requested; reduce "
                "--stop-after-ms or increase --travel-mm within the safety cap"
            )
        stop_requested = True
        stop_started_ns = time.monotonic_ns()
        stop_result = gripper.stop()
        stop_ms = (time.monotonic_ns() - stop_started_ns) / 1e6
        if stop_result is False:
            raise RuntimeError("Franka Hand stop() reported failure")
        after, after_state_ms = _timed_state(gripper)
        return {
            "kind": "stop",
            "direction": direction,
            "before": before,
            "target_width_m": target_m,
            "commanded_travel_m": abs(target_m - before["width_m"]),
            "speed_m_s": speed_m_s,
            "stop_after_ms": stop_after_s * 1000.0,
            "pending_before_stop": pending_before_stop,
            "move_async_call_ms": move_async_ms,
            "stop_call_ms": stop_ms,
            "before_state_read_ms": before_state_ms,
            "after_state_read_ms": after_state_ms,
            "after": after,
            "measured_travel_before_stop_return_m": abs(
                after["width_m"] - before["width_m"]
            ),
        }
    except BaseException:
        if future is not None and not stop_requested:
            try:
                if not bool(future.wait(0.0)):
                    gripper.stop()
            except BaseException as cleanup_error:
                print(
                    f"WARNING: cleanup stop failed: {cleanup_error}",
                    file=sys.stderr,
                )
        raise


def run_natural_trial(
    gripper: Any,
    *,
    target_m: float,
    speed_m_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    before, before_state_ms = _timed_state(gripper)
    if not 0.0 <= target_m <= before["max_width_m"]:
        raise ValueError("Natural-move target is outside the reported Hand range")
    travel_m = abs(target_m - before["width_m"])
    if travel_m > MAX_TRAVEL_M + 1e-9:
        raise ValueError("Natural-move travel exceeds the 10 mm diagnostic cap")
    if travel_m < 1e-6:
        raise ValueError("Natural-move target is already reached")

    future: Any | None = None
    completed = False
    try:
        call_started_ns = time.monotonic_ns()
        future = gripper.move_async(target_m, speed_m_s)
        move_async_ms = (time.monotonic_ns() - call_started_ns) / 1e6
        wait_started_ns = time.monotonic_ns()
        deadline = time.monotonic() + timeout_s
        while not bool(future.wait(0.0)):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Finite Hand move did not finish within {timeout_s:.3f} s"
                )
            time.sleep(0.002)
        completed = True
        completion_ms = (time.monotonic_ns() - wait_started_ns) / 1e6
        success = bool(future.get())
        if not success:
            raise RuntimeError("Franka Hand finite move reported failure")
        after, after_state_ms = _timed_state(gripper)
        return {
            "kind": "natural",
            "before": before,
            "target_width_m": target_m,
            "commanded_travel_m": travel_m,
            "speed_m_s": speed_m_s,
            "nominal_motion_ms": travel_m / speed_m_s * 1000.0,
            "move_async_call_ms": move_async_ms,
            "completion_wait_ms": completion_ms,
            "before_state_read_ms": before_state_ms,
            "after_state_read_ms": after_state_ms,
            "after": after,
            "final_error_m": after["width_m"] - target_m,
        }
    except BaseException:
        if future is not None and not completed:
            try:
                gripper.stop()
            except BaseException as cleanup_error:
                print(
                    f"WARNING: cleanup stop failed: {cleanup_error}",
                    file=sys.stderr,
                )
        raise


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"trial_count": len(trials)}
    for kind, field in (("stop", "stop_call_ms"), ("natural", "completion_wait_ms")):
        values = [float(item[field]) for item in trials if item["kind"] == kind]
        if values:
            summary[kind] = {
                "count": len(values),
                "minimum_ms": min(values),
                "median_ms": statistics.median(values),
                "maximum_ms": max(values),
            }
    return summary


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "gripper_latency" / f"{stamp}_franky.json"


def print_plan(args: argparse.Namespace) -> None:
    travel_m = args.travel_mm / 1000.0
    print("Franka Hand latency diagnostic")
    print("  backend: franky (current production Hand path)")
    print("  arm connection/motion: disabled")
    print(f"  mode: {args.mode}")
    print(f"  first direction: {args.direction}")
    print(f"  bounded travel: {travel_m * 1000:.1f} mm (hard cap 10 mm)")
    print(f"  speed: {args.speed:.3f} m/s (hard cap 0.05 m/s)")
    if args.mode in {"stop", "compare"}:
        print(f"  stop request delay: {args.stop_after_ms:.1f} ms")
    print(f"  repeats: {args.repeats}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_parameters(args)
    except ValueError as exc:
        parser.error(str(exc))

    print_plan(args)
    if not args.run_hardware:
        try:
            version = metadata.version("franky-control")
        except metadata.PackageNotFoundError:
            version = "not installed"
        print(f"  franky-control: {version}")
        print("Dry run only: no Hand connection and no movement were performed.")
        print("Add --run-hardware only after removing objects and clearing the fingers.")
        return 0

    answer = input(
        "Remove all objects, keep hands clear, and type y/yes to run this "
        "gripper-only test: "
    ).strip().lower()
    if answer not in {"y", "yes"}:
        print("Cancelled; no Hand connection or movement was performed.")
        return 1

    from franky import Gripper

    gripper = Gripper(args.ip)
    initial = checked_state(gripper)
    if initial["is_grasped"]:
        raise RuntimeError(
            "Hand reports is_grasped=true; remove the object before latency testing"
        )

    travel_m = args.travel_mm / 1000.0
    stop_after_s = args.stop_after_ms / 1000.0
    trials: list[dict[str, Any]] = []
    for index in range(args.repeats):
        print(f"Running trial {index + 1}/{args.repeats}...")
        start = checked_state(gripper)
        first_target = bounded_target(
            start["width_m"], start["max_width_m"], args.direction, travel_m
        )
        if args.mode in {"stop", "compare"}:
            trial = run_stop_trial(
                gripper,
                direction=args.direction,
                travel_m=travel_m,
                speed_m_s=args.speed,
                stop_after_s=stop_after_s,
            )
            trials.append(trial)
            print(
                f"  stop(): {trial['stop_call_ms']:.2f} ms; "
                f"width {trial['before']['width_m'] * 1000:.2f} -> "
                f"{trial['after']['width_m'] * 1000:.2f} mm"
            )
        if args.mode == "natural":
            trial = run_natural_trial(
                gripper,
                target_m=first_target,
                speed_m_s=args.speed,
                timeout_s=args.natural_timeout_s,
            )
            trials.append(trial)
        elif args.mode == "compare":
            # Return naturally to the exact pre-stop width, keeping this second
            # move no larger than the already bounded first movement.
            current = checked_state(gripper)
            return_travel_m = abs(start["width_m"] - current["width_m"])
            if return_travel_m >= 1e-6:
                trial = run_natural_trial(
                    gripper,
                    target_m=start["width_m"],
                    speed_m_s=args.speed,
                    timeout_s=args.natural_timeout_s,
                )
                trials.append(trial)
                print(
                    f"  natural return: {trial['completion_wait_ms']:.2f} ms "
                    f"for {return_travel_m * 1000:.2f} mm"
                )
            else:
                print("  natural return skipped: stop produced no measurable travel")

    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "backend": "franky",
        "franky_control_version": metadata.version("franky-control"),
        "python": platform.python_version(),
        "robot_ip": args.ip,
        "arm_connection_or_motion": False,
        "parameters": {
            "mode": args.mode,
            "direction": args.direction,
            "travel_m": travel_m,
            "speed_m_s": args.speed,
            "stop_after_s": stop_after_s,
            "repeats": args.repeats,
            "natural_timeout_s": args.natural_timeout_s,
        },
        "initial_state": initial,
        "trials": trials,
        "summary": summarize(trials),
    }
    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved report to: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
