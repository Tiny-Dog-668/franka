#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real import load_config, run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Franka sim2real validation.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "sim2real_validation_example.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--backend",
        choices=("sim", "real"),
        help="Override backend.kind from the config file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override runner.episodes from the config file.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Override runner.max_steps from the config file.",
    )
    parser.add_argument(
        "--robot-ip",
        help="Override backend.robot_ip from the config file.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the real-robot confirmation prompt.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.backend:
        config.backend.kind = args.backend
    if args.episodes is not None:
        config.runner.episodes = args.episodes
    if args.max_steps is not None:
        config.runner.max_steps = args.max_steps
    if args.robot_ip:
        config.backend.robot_ip = args.robot_ip

    if config.backend.kind == "real" and not args.yes:
        print("This will command the real Franka robot.")
        confirm = input("Type YES to continue: ")
        if confirm != "YES":
            print("Aborted.")
            return 1

    summary = run_validation(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
