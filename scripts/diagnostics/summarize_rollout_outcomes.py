#!/usr/bin/env python3
"""Summarise reach and grasp outcomes across recorded real-robot rollouts.

Reads ``runs/*/summary.json`` and reports, per run, how close the fingertip
midpoint got to the cube and whether the gripper ever closed on it. Oracle runs
carry the true cube position in ``model_input.rma_actor_input``, so for those
the reach error is measured against ground truth rather than a prediction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"

# Fingertip midpoint sits this far below the reported TCP frame along tool z.
HAND_TO_FINGERTIP_M = 0.1034


def _fingertip(tcp: list[float]) -> tuple[float, float, float]:
    return (tcp[0], tcp[1], tcp[2] - HAND_TO_FINGERTIP_M)


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def summarize(summary_path: Path) -> dict | None:
    try:
        data = json.loads(summary_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    steps = data.get("steps") or []
    if not steps:
        return None

    cube = None
    first = steps[0].get("model_input", {}).get("rma_actor_input")
    if first and first.get("cube_position_root"):
        cube = tuple(first["cube_position_root"])

    widths: list[float] = []
    grasped = False
    reach_errors: list[float] = []
    horizontal_errors: list[float] = []
    tips: list[tuple[float, float, float]] = []

    for step in steps:
        obs = step.get("observation_after") or step.get("observation") or {}
        tcp = obs.get("tcp_translation")
        if tcp:
            tip = _fingertip(tcp)
            tips.append(tip)
            if cube is not None:
                reach_errors.append(_distance(tip, cube))
                horizontal_errors.append(math.dist(tip[:2], cube[:2]))
        width = obs.get("gripper_width")
        if width is not None:
            widths.append(width)
        if obs.get("gripper_is_grasped"):
            grasped = True

    if not tips:
        return None

    lift = None
    if widths and min(widths) < 0.045:
        closed_at = widths.index(min(widths))
        after = [t[2] for t in tips[closed_at:]]
        if after:
            lift = max(after) - tips[closed_at][2]

    return {
        "run": summary_path.parent.name,
        "steps": len(steps),
        "cube": cube,
        "min_reach_mm": min(reach_errors) * 1000 if reach_errors else None,
        "final_reach_mm": reach_errors[-1] * 1000 if reach_errors else None,
        "min_horizontal_mm": min(horizontal_errors) * 1000 if horizontal_errors else None,
        "start_tip": tips[0],
        "final_tip": tips[-1],
        "min_width_mm": min(widths) * 1000 if widths else None,
        "final_width_mm": widths[-1] * 1000 if widths else None,
        "grasped": grasped,
        "lift_mm": lift * 1000 if lift is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*", help="Glob over run directory names.")
    parser.add_argument("--limit", type=int, default=0, help="Only the newest N runs.")
    parser.add_argument("--min-steps", type=int, default=5, help="Skip shorter runs.")
    args = parser.parse_args()

    candidates = sorted(RUNS_DIR.glob(args.pattern), reverse=True)
    rows = []
    for run_dir in candidates:
        summary = run_dir / "summary.json"
        if not summary.is_file():
            continue
        row = summarize(summary)
        if row and row["steps"] >= args.min_steps:
            rows.append(row)
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        print("no matching runs")
        return 1

    header = f"{'run':<62} {'steps':>5} {'reach_mm':>9} {'horiz_mm':>9} {'width_mm':>9} {'lift_mm':>8}  grasp"
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value: float | None, width: int) -> str:
            return f"{value:>{width}.1f}" if value is not None else " " * (width - 1) + "-"

        print(
            f"{row['run'][:62]:<62} {row['steps']:>5} "
            f"{fmt(row['min_reach_mm'], 9)} {fmt(row['min_horizontal_mm'], 9)} "
            f"{fmt(row['min_width_mm'], 9)} {fmt(row['lift_mm'], 8)}  "
            f"{'YES' if row['grasped'] else 'no'}"
        )

    grasps = sum(1 for row in rows if row["grasped"])
    print(f"\n{len(rows)} runs, {grasps} with gripper_is_grasped")
    reaches = [row["min_reach_mm"] for row in rows if row["min_reach_mm"] is not None]
    if reaches:
        reaches.sort()
        print(
            f"reach error vs true cube: best {reaches[0]:.1f} mm, "
            f"median {reaches[len(reaches) // 2]:.1f} mm, worst {reaches[-1]:.1f} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
