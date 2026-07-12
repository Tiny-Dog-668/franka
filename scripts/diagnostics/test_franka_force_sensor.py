#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from franky import Frame, RealtimeConfig, Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read Franka external wrench and print baseline-relative force/torque changes."
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2"),
        help="Robot IP address. Defaults to FRANKA_ROBOT_IP or 172.16.0.2.",
    )
    parser.add_argument(
        "--baseline-samples",
        type=int,
        default=20,
        help="Number of initial samples used as the zero-force baseline.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of live samples after baseline. Use 0 for continuous mode.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Delay between samples in seconds.",
    )
    parser.add_argument(
        "--realtime",
        choices=("ignore", "enforce"),
        default="ignore",
        help="Real-time scheduling mode. 'ignore' is recommended for read-only checks.",
    )
    parser.add_argument(
        "--threshold-n",
        type=float,
        default=0.5,
        help="Mark samples whose baseline-relative force norm exceeds this value.",
    )
    parser.add_argument(
        "--dynamic-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Slowly update the baseline when no contact is detected.",
    )
    parser.add_argument(
        "--baseline-alpha",
        type=float,
        default=0.02,
        help="Exponential moving average update rate for dynamic baseline.",
    )
    parser.add_argument(
        "--baseline-update-ratio",
        type=float,
        default=0.5,
        help="Update dynamic baseline only when |dF| is below this fraction of --threshold-n.",
    )
    parser.add_argument(
        "--compare-jacobian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also estimate external wrench from tau_ext_hat_filtered via the model "
            "Jacobian and print it beside O_F_ext_hat_K."
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
    parser.add_argument(
        "--show-joint-torques",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Print joint torque residuals tau_ext_hat_filtered and their "
            "baseline-relative deltas. Units are Nm."
        ),
    )
    parser.add_argument(
        "--show-measured-joint-torques",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also print measured joint torques tau_J. Units are Nm.",
    )
    return parser


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


def average(samples: list[list[float]]) -> list[float]:
    return [sum(sample[index] for sample in samples) / len(samples) for index in range(6)]


def vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def fmt(values: list[float], precision: int = 3) -> str:
    return "[" + ", ".join(f"{value: .{precision}f}" for value in values) + "]"


def update_baseline(baseline: list[float], sample: list[float], alpha: float) -> list[float]:
    return [
        (1.0 - alpha) * baseline[index] + alpha * sample[index]
        for index in range(6)
    ]


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 < args.baseline_alpha <= 1.0:
        print("--baseline-alpha must be in (0, 1].")
        return 1
    if not 0.0 <= args.baseline_update_ratio <= 1.0:
        print("--baseline-update-ratio must be in [0, 1].")
        return 1
    if args.jacobian_damping < 0.0:
        print("--jacobian-damping must be >= 0.")
        return 1

    realtime_config = (
        RealtimeConfig.Ignore
        if args.realtime == "ignore"
        else RealtimeConfig.Enforce
    )

    robot = Robot(args.ip, realtime_config=realtime_config)
    model = None
    jacobian_frame = None
    if args.compare_jacobian:
        model = robot.model
        jacobian_frame = frame_from_name(args.jacobian_frame)
    print(f"Connected to Franka at {args.ip}")
    if args.compare_jacobian:
        print(
            "Jacobian compare: enabled "
            f"(frame={args.jacobian_frame}, mode={args.jacobian_mode}, "
            f"damping={args.jacobian_damping:g})"
        )
    print("Keep the end effector unloaded during baseline collection.")

    baseline_samples: list[list[float]] = []
    jacobian_baseline_samples: list[list[float]] = []
    tau_ext_baseline_samples: list[list[float]] = []
    tau_j_baseline_samples: list[list[float]] = []
    for index in range(args.baseline_samples):
        state = robot.state
        sample = state.O_F_ext_hat_K.tolist()
        baseline_samples.append(sample)
        line = f"Baseline {index + 1:02d}/{args.baseline_samples}: {fmt(sample)}"
        if args.show_joint_torques:
            tau_ext_sample = state.tau_ext_hat_filtered.tolist()
            tau_ext_baseline_samples.append(tau_ext_sample)
            line += f" | tau_ext {fmt(tau_ext_sample)}"
        if args.show_measured_joint_torques:
            tau_j_sample = state.tau_J.tolist()
            tau_j_baseline_samples.append(tau_j_sample)
            line += f" | tau_J {fmt(tau_j_sample)}"
        if args.compare_jacobian:
            jacobian_sample = jacobian_wrench(
                model,
                state,
                jacobian_frame,
                args.jacobian_mode,
                args.jacobian_damping,
            )
            jacobian_baseline_samples.append(jacobian_sample)
            line += f" | Jtau {fmt(jacobian_sample)}"
        print(line)
        time.sleep(args.interval)

    baseline = average(baseline_samples)
    jacobian_baseline = (
        average(jacobian_baseline_samples)
        if args.compare_jacobian and jacobian_baseline_samples
        else None
    )
    tau_ext_baseline = (
        average(tau_ext_baseline_samples)
        if args.show_joint_torques and tau_ext_baseline_samples
        else None
    )
    tau_j_baseline = (
        average(tau_j_baseline_samples)
        if args.show_measured_joint_torques and tau_j_baseline_samples
        else None
    )
    print("\nBaseline O_F_ext_hat_K:")
    print(f"  force  [N]:  {fmt(baseline[:3])}")
    print(f"  torque [Nm]: {fmt(baseline[3:])}")
    if jacobian_baseline is not None:
        print("Baseline Jacobian estimate:")
        print(f"  force  [N]:  {fmt(jacobian_baseline[:3])}")
        print(f"  torque [Nm]: {fmt(jacobian_baseline[3:])}")
    if tau_ext_baseline is not None:
        print("Baseline tau_ext_hat_filtered [Nm]:")
        print(f"  {fmt(tau_ext_baseline)}")
    if tau_j_baseline is not None:
        print("Baseline tau_J [Nm]:")
        print(f"  {fmt(tau_j_baseline)}")
    if args.dynamic_baseline:
        update_limit = args.threshold_n * args.baseline_update_ratio
        print(
            "Dynamic baseline: enabled "
            f"(alpha={args.baseline_alpha}, update when |dF| < {update_limit:.3f} N)"
        )
    else:
        print("Dynamic baseline: disabled")
    print("\nLive readings. Press Ctrl-C to stop.")
    if args.compare_jacobian:
        print(
            "sample | O_force[N]               | O_d_force[N]             | "
            "J_force[N]               | J_d_force[N]             | "
            "|O_dF| | |J_dF| | J-O force[N]           | baseline | flag"
        )
    elif args.show_joint_torques:
        print(
            "sample | force[N]                 | |dF|[N] | "
            "tau_ext[Nm]                                           | "
            "d_tau_ext[Nm]                                         | "
            "|d_tau_wrist| | |d_tau_all| | baseline | flag"
        )
    else:
        print(
            "sample | force[N]                 | d_force[N]               | "
            "|dF|[N] | torque[Nm]              | d_torque[Nm]            | baseline | flag"
        )

    sample_index = 1
    try:
        while args.count == 0 or sample_index <= args.count:
            state = robot.state
            sample = state.O_F_ext_hat_K.tolist()
            delta = [sample[index] - baseline[index] for index in range(6)]
            force_norm = vector_norm(delta[:3])
            flag = "CONTACT" if force_norm >= args.threshold_n else ""
            jacobian_sample = None
            jacobian_delta = None
            jacobian_force_norm = None
            tau_ext_sample = None
            tau_ext_delta = None
            tau_ext_wrist_norm = None
            tau_ext_all_norm = None
            tau_j_sample = None
            tau_j_delta = None
            if args.compare_jacobian:
                jacobian_sample = jacobian_wrench(
                    model,
                    state,
                    jacobian_frame,
                    args.jacobian_mode,
                    args.jacobian_damping,
                )
                jacobian_delta = [
                    jacobian_sample[index] - jacobian_baseline[index]
                    for index in range(6)
                ]
                jacobian_force_norm = vector_norm(jacobian_delta[:3])
            if args.show_joint_torques:
                tau_ext_sample = state.tau_ext_hat_filtered.tolist()
                tau_ext_delta = [
                    tau_ext_sample[index] - tau_ext_baseline[index]
                    for index in range(7)
                ]
                tau_ext_wrist_norm = vector_norm(tau_ext_delta[4:7])
                tau_ext_all_norm = vector_norm(tau_ext_delta)
            if args.show_measured_joint_torques:
                tau_j_sample = state.tau_J.tolist()
                tau_j_delta = [
                    tau_j_sample[index] - tau_j_baseline[index]
                    for index in range(7)
                ]
            baseline_status = "fixed"
            if args.dynamic_baseline:
                update_limit = args.threshold_n * args.baseline_update_ratio
                if force_norm < update_limit:
                    baseline = update_baseline(baseline, sample, args.baseline_alpha)
                    if jacobian_sample is not None and jacobian_baseline is not None:
                        jacobian_baseline = update_baseline(
                            jacobian_baseline,
                            jacobian_sample,
                            args.baseline_alpha,
                        )
                    if tau_ext_sample is not None and tau_ext_baseline is not None:
                        tau_ext_baseline = update_baseline(
                            tau_ext_baseline,
                            tau_ext_sample,
                            args.baseline_alpha,
                        )
                    if tau_j_sample is not None and tau_j_baseline is not None:
                        tau_j_baseline = update_baseline(
                            tau_j_baseline,
                            tau_j_sample,
                            args.baseline_alpha,
                        )
                    baseline_status = "update"
                else:
                    baseline_status = "hold"
            if args.compare_jacobian:
                force_diff = [
                    jacobian_sample[index] - sample[index]
                    for index in range(3)
                ]
                print(
                    f"{sample_index:6d} | {fmt(sample[:3])} | {fmt(delta[:3])} | "
                    f"{fmt(jacobian_sample[:3])} | {fmt(jacobian_delta[:3])} | "
                    f"{force_norm:6.3f} | {jacobian_force_norm:6.3f} | "
                    f"{fmt(force_diff)} | {baseline_status:8s} | {flag}"
                )
            elif args.show_joint_torques:
                print(
                    f"{sample_index:6d} | {fmt(sample[:3])} | {force_norm:6.3f} | "
                    f"{fmt(tau_ext_sample)} | {fmt(tau_ext_delta)} | "
                    f"{tau_ext_wrist_norm:12.4f} | {tau_ext_all_norm:11.4f} | "
                    f"{baseline_status:8s} | {flag}"
                )
                if args.show_measured_joint_torques:
                    print(
                        f"       tau_J={fmt(tau_j_sample)} "
                        f"d_tau_J={fmt(tau_j_delta)}"
                    )
            else:
                print(
                    f"{sample_index:6d} | {fmt(sample[:3])} | {fmt(delta[:3])} | "
                    f"{force_norm:6.3f} | {fmt(sample[3:])} | {fmt(delta[3:])} | "
                    f"{baseline_status:8s} | {flag}"
                )
            sample_index += 1
            if args.count == 0 or sample_index <= args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
