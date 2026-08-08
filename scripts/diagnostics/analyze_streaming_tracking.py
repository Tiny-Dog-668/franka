#!/usr/bin/env python3
"""Measure how faithfully a streaming rollout executed the policy's Cartesian actions.

A sim-faithful controller has two properties that a saturating one does not:

* the achieved TCP displacement is proportional to the commanded one, so the
  slope of achieved-vs-commanded is close to one and the linearity ratio between
  large and small actions is close to the ratio of the actions themselves;
* the achieved direction agrees with the commanded direction.

Isaac Lab's implicit PD (``FRANKA_PANDA_HIGH_PD_CFG``: stiffness 400, damping 80,
robot gravity disabled) re-derives the IK target from the measured pose every
policy step, so the position error stays pinned at the IK delta and the simulator
settles at ``qd = (stiffness / damping) * dq_ik``. A saturated action therefore
moves the simulated TCP at ``5 * action_scale`` per second, which is the
``sim reference`` reported below -- not ``action_scale`` per policy step.

Usage:
    python scripts/diagnostics/analyze_streaming_tracking.py runs/<run_dir> [more_run_dirs...]
    python scripts/diagnostics/analyze_streaming_tracking.py --latest 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _commanded_xyz(record: dict[str, Any]) -> np.ndarray:
    action = record["robot_action"]
    return np.asarray([action["dx"], action["dy"], action["dz"]], dtype=np.float64)


def _unit(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else None


def analyze_run(run_dir: Path) -> dict[str, Any]:
    records = _load_jsonl(run_dir / "rollout.jsonl")
    trace = _load_jsonl(run_dir / "control_trace.jsonl")
    timing = {}
    timing_path = run_dir / "timing_summary.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    config = {}
    config_path = run_dir / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    streaming = config.get("streaming", {})
    adapter = config.get("action_adapter", {})
    scales = adapter.get("scales") or [math.nan] * 4
    report: dict[str, Any] = {
        "run_dir": str(run_dir),
        "control_law": streaming.get("control_law", "joint_position_pursuit"),
        "reference_velocity_gain": streaming.get("reference_velocity_gain"),
        "commissioning_action_limit": streaming.get("commissioning_action_limit"),
        "maximum_joint_target_delta_rad": streaming.get("maximum_joint_target_delta_rad"),
        "maximum_joint_velocities": streaming.get("maximum_joint_velocities"),
        "policy_steps": len(records),
        "control_ticks": len(trace),
        "policy_deadline_misses": timing.get("policy_deadline_misses"),
        "maximum_policy_elapsed_ms": timing.get("maximum_policy_elapsed_ms"),
        "maximum_control_lateness_ms": timing.get("maximum_control_lateness_ms"),
        "aborted": bool(timing.get("aborted", False)),
        "exception": timing.get("exception"),
    }

    # The displacement observed at step k is the result of the action commanded
    # at step k-1, so the two series are offset by one policy step.
    commanded: list[float] = []
    achieved: list[float] = []
    direction_errors_deg: list[float] = []
    for index in range(1, len(records)):
        previous = records[index - 1]
        if previous.get("executed_action") is None:
            continue
        before = np.asarray(
            previous["observation_before"]["tcp_translation"], dtype=np.float64
        )
        after = np.asarray(
            records[index]["observation_before"]["tcp_translation"], dtype=np.float64
        )
        command = _commanded_xyz(previous)
        displacement = after - before
        commanded.append(float(np.linalg.norm(command)))
        achieved.append(float(np.linalg.norm(displacement)))
        unit_command = _unit(command)
        unit_displacement = _unit(displacement)
        if unit_command is not None and unit_displacement is not None:
            cosine = float(np.clip(unit_command @ unit_displacement, -1.0, 1.0))
            direction_errors_deg.append(math.degrees(math.acos(cosine)))

    if commanded:
        command_array = np.asarray(commanded)
        achieved_array = np.asarray(achieved)
        report["commanded_mm_per_step_mean"] = float(command_array.mean() * 1000.0)
        report["achieved_mm_per_step_mean"] = float(achieved_array.mean() * 1000.0)
        usable = command_array > 1e-9
        if usable.any():
            ratios = achieved_array[usable] / command_array[usable]
            report["achieved_over_commanded_median"] = float(np.median(ratios))
            # Least-squares slope through the origin is the tracking gain the
            # controller applied to the policy's Cartesian command.
            report["tracking_slope"] = float(
                (command_array[usable] @ achieved_array[usable])
                / (command_array[usable] @ command_array[usable])
            )
            spread = float(command_array[usable].max() / max(command_array[usable].min(), 1e-12))
            report["commanded_magnitude_spread"] = spread
            if spread < 1.5:
                report["linearity_note"] = (
                    "commanded magnitudes are nearly constant in this run, so linearity "
                    "cannot be measured; the policy saturates its output"
                )
    if direction_errors_deg:
        errors = np.asarray(direction_errors_deg)
        report["direction_error_deg_median"] = float(np.median(errors))
        report["direction_error_deg_p90"] = float(np.percentile(errors, 90))

    if trace:
        deltas = np.asarray(
            [
                np.abs(np.asarray(entry["q_target"]) - np.asarray(entry["q"])).max()
                for entry in trace
            ]
        )
        report["max_joint_reference_delta_rad"] = float(deltas.max())
        clamp = streaming.get("maximum_joint_target_delta_rad")
        if report["control_law"] == "joint_position_pursuit" and clamp:
            saturated = float((deltas >= 0.999 * float(clamp)).mean())
            report["reference_clamp_saturation_fraction"] = saturated
        generations = [entry.get("action_generation") for entry in trace]
        counts: dict[int, int] = {}
        for generation in generations:
            counts[generation] = counts.get(generation, 0) + 1
        latched = sorted(g for g in counts if g)
        report["latched_generations"] = len(latched)
        if latched:
            report["dropped_generations"] = sorted(
                set(range(1, max(latched) + 1)) - set(latched)
            )

    sim_reference_mm = None
    gain = streaming.get("reference_velocity_gain")
    policy_hz = streaming.get("policy_frequency_hz")
    if gain and policy_hz and np.all(np.isfinite(scales[:3])):
        limit = streaming.get("commissioning_action_limit") or 1.0
        sim_reference_mm = float(gain) * max(scales[:3]) * float(limit) / float(policy_hz) * 1000.0
        report["sim_reference_mm_per_step_at_saturation"] = sim_reference_mm
    return report


def _format(report: dict[str, Any]) -> str:
    lines = [f"{Path(report['run_dir']).name}"]
    lines.append(
        f"  control_law={report['control_law']}"
        f"  action_limit={report.get('commissioning_action_limit')}"
        f"  gain={report.get('reference_velocity_gain')}"
    )
    lines.append(
        f"  steps={report['policy_steps']} ticks={report['control_ticks']}"
        f" misses={report.get('policy_deadline_misses')}"
        f" max_policy_ms={_number(report.get('maximum_policy_elapsed_ms'))}"
        f" max_late_ms={_number(report.get('maximum_control_lateness_ms'))}"
    )
    if report["aborted"]:
        lines.append(f"  ABORTED: {report.get('exception')}")
    if "achieved_over_commanded_median" in report:
        lines.append(
            f"  commanded={_number(report.get('commanded_mm_per_step_mean'))} mm/step"
            f"  achieved={_number(report.get('achieved_mm_per_step_mean'))} mm/step"
            f"  ratio={_number(report.get('achieved_over_commanded_median'), 3)}"
            f"  slope={_number(report.get('tracking_slope'), 3)}"
        )
    if "sim_reference_mm_per_step_at_saturation" in report:
        lines.append(
            "  sim reference at this action limit="
            f"{_number(report['sim_reference_mm_per_step_at_saturation'])} mm/step"
        )
    if "direction_error_deg_median" in report:
        lines.append(
            f"  direction error median={_number(report['direction_error_deg_median'], 1)} deg"
            f"  p90={_number(report.get('direction_error_deg_p90'), 1)} deg"
        )
    if "reference_clamp_saturation_fraction" in report:
        fraction = report["reference_clamp_saturation_fraction"]
        lines.append(
            f"  reference clamp saturated on {fraction * 100:.0f}% of IK ticks"
            + ("  <-- the DLS magnitude is being discarded" if fraction > 0.5 else "")
        )
    if report.get("dropped_generations"):
        lines.append(f"  policy actions never latched: {report['dropped_generations']}")
    if "linearity_note" in report:
        lines.append(f"  note: {report['linearity_note']}")
    return "\n".join(lines)


def _number(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", type=Path)
    parser.add_argument(
        "--latest",
        type=int,
        default=0,
        help="analyze the N most recent run directories under runs/",
    )
    parser.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    arguments = parser.parse_args(argv)

    run_dirs = list(arguments.run_dirs)
    if arguments.latest:
        candidates = sorted(
            (path for path in (REPO_ROOT / "runs").iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        run_dirs.extend(candidates[: arguments.latest])
    if not run_dirs:
        parser.error("pass at least one run directory or --latest N")

    reports = [analyze_run(path) for path in run_dirs]
    if arguments.json:
        print(json.dumps(reports, indent=2))
    else:
        print("\n".join(_format(report) for report in reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
