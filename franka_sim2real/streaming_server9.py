from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .e2e_bundle import (
    ActionHistoryBuffer,
    BundleDeployConfig,
    BundleTorchScriptPolicy,
    _make_camera,
    _make_run_dir,
    _validate_bundle_action_dims,
    evaluate_initial_state,
)
from .server9_ipc import STATUS_ABORTED, STATUS_RUNNING, Server9Worker, WorkerSnapshot
from .streaming import (
    AsyncGripperQueue,
    _PolicyResult,
    _policy_record,
    _rotation_to_quaternion_xyzw,
    _run_policy_tick,
    _sleep_until,
    _write_streaming_artifacts,
    evaluate_streaming_check_state,
    policy_result_is_timely,
    reshape_column_major,
    validate_streaming_contract,
)
from .types import (
    RobotAction,
    RobotObservation,
    external_wrench_norms,
    print_error_wrench_report,
)


ROBOT_MODE_NAMES = {
    0: "Other", 1: "Idle", 2: "Move", 3: "Guiding", 4: "Reflex",
    5: "UserStopped", 6: "AutomaticErrorRecovery",
}
PROGRESS_INTERVAL_STEPS = 30


def _should_print_streaming_progress(completed_steps: int, total_steps: int) -> bool:
    return completed_steps % PROGRESS_INTERVAL_STEPS == 0 or completed_steps == total_steps


def _print_server9_error_snapshot(
    worker: Server9Worker,
    gripper_queue: AsyncGripperQueue | None,
    *,
    context: str,
    force: bool = False,
) -> bool:
    try:
        snapshot = worker.memory.snapshot()
        observation = snapshot_to_observation(
            snapshot,
            gripper_queue.cached_state if gripper_queue else None,
            in_control=True,
        )
    except Exception as exc:
        print(
            "=" * 72
            + f"\n{context}: failed to read final server9 snapshot: {exc}\n"
            + "=" * 72,
            flush=True,
        )
        return False
    if (
        not force
        and
        not observation.has_errors
        and observation.robot_mode in {"Move", "Idle"}
        and snapshot.status != STATUS_ABORTED
        and not snapshot.error_code
        and not snapshot.error_message
    ):
        return False
    print_error_wrench_report(observation, context=context)
    return True


def _print_streaming_progress(
    completed_steps: int,
    total_steps: int,
    deadline_misses: int,
    observation: RobotObservation | None = None,
    result: _PolicyResult | None = None,
    *,
    accepted: bool | None = None,
    timing_ms: dict[str, float] | None = None,
) -> None:
    if not _should_print_streaming_progress(completed_steps, total_steps):
        return
    accepted_steps = completed_steps - deadline_misses
    lines = [
        "Streaming progress: "
        f"{completed_steps}/{total_steps} steps "
        f"(accepted={accepted_steps}, deadline_misses={deadline_misses})"
    ]
    if observation is not None:
        force_norm, torque_norm = external_wrench_norms(observation)
        gripper = (
            "none"
            if observation.gripper_width is None
            else f"{observation.gripper_width:.4f}m"
        )
        lines.append(
            "  robot: "
            f"mode={observation.robot_mode} "
            f"tcp=[{observation.tcp_translation[0]:+.4f}, "
            f"{observation.tcp_translation[1]:+.4f}, "
            f"{observation.tcp_translation[2]:+.4f}]m "
            f"gripper={gripper} "
            f"|F|={force_norm:.2f}N |M|={torque_norm:.2f}Nm"
        )
    if result is not None:
        rma = result.inference_info
        cube_position = rma.get("cube_position_root")
        contact_state = rma.get("contact_state")
        if cube_position is not None or contact_state is not None:
            cube_text = (
                "none"
                if cube_position is None
                else "["
                + ", ".join(f"{float(value):+.4f}" for value in cube_position)
                + "]m"
            )
            contact_text = (
                "none"
                if contact_state is None
                else "[" + ", ".join(f"{float(value):.3f}" for value in contact_state) + "]"
            )
            lines.append(
                "  policy_obs: "
                f"cube_position_root={cube_text} "
                f"contact_lr={contact_text} "
                f"source=pos:{rma.get('rma_position_source', '?')}/"
                f"contact:{rma.get('rma_contact_source', '?')}"
            )
        else:
            lines.append(
                "  policy_obs: "
                "cube_position_root/contact_state not collected on this step"
            )
        robot_action = result.robot_action
        status = "accepted" if accepted else "held"
        raw = ", ".join(f"{float(value):+.3f}" for value in result.raw_action)
        executed = ", ".join(f"{float(value):+.3f}" for value in result.executed_action)
        gripper_target = (
            "none"
            if robot_action.gripper_width is None
            else f"{robot_action.gripper_width:.4f}m"
        )
        lines.append(
            "  action: "
            f"{status} raw_norm=[{raw}] executed_norm=[{executed}] "
            f"robot_delta=[{robot_action.dx:+.4f}, "
            f"{robot_action.dy:+.4f}, {robot_action.dz:+.4f}]m "
            f"gripper_target={gripper_target}"
        )
    if timing_ms:
        lines.append(
            "  timing_ms: "
            f"policy={timing_ms.get('policy', 0.0):.2f} "
            f"deadline_late={timing_ms.get('deadline_lateness', 0.0):.2f} "
            f"latch_wait={timing_ms.get('latch_wait', 0.0):.2f} "
            f"loop={timing_ms.get('loop_body', 0.0):.2f}"
        )
    print(
        "\n".join(lines),
        flush=True,
    )


def snapshot_to_observation(
    snapshot: WorkerSnapshot, gripper_state: Any | None, *, in_control: bool
) -> RobotObservation:
    pose = reshape_column_major(snapshot.O_T_EE, 4, 4)
    observation = RobotObservation(
        joint_positions=list(snapshot.q),
        joint_velocities=list(snapshot.dq),
        tcp_translation=pose[:3, 3].tolist(),
        tcp_quaternion=_rotation_to_quaternion_xyzw(pose[:3, :3]),
        external_wrench=list(snapshot.external_wrench),
        robot_mode=ROBOT_MODE_NAMES.get(snapshot.robot_mode, f"Unknown({snapshot.robot_mode})"),
        has_errors=snapshot.has_errors,
        is_in_control=in_control,
        control_command_success_rate=snapshot.command_success_rate,
        metadata={
            "server_version": 9,
            "worker_status": snapshot.status_name,
            "worker_error_code": snapshot.error_code,
            "worker_error_message": snapshot.error_message,
            "latched_action_generation": snapshot.latched_action_generation,
            "ik_tick_count": snapshot.ik_tick_count,
            "control_cycle_count": snapshot.control_cycle_count,
        },
    )
    if gripper_state is not None:
        observation.gripper_width = float(gripper_state.width)
        observation.gripper_max_width = float(gripper_state.max_width)
        observation.gripper_is_grasped = bool(gripper_state.is_grasped)
    return observation


def _worker_binary(config: BundleDeployConfig) -> Path:
    configured = Path(config.streaming.server9_worker_path).expanduser()
    return configured if configured.is_absolute() else Path(__file__).resolve().parents[1] / configured


def _wait_running(worker: Server9Worker, timeout_s: float) -> WorkerSnapshot:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = worker.snapshot()
        if snapshot.status == STATUS_RUNNING:
            return snapshot
        time.sleep(0.001)
    raise RuntimeError("server9 worker did not enter active control")


def _wait_generation_ticks(
    worker: Server9Worker,
    generation: int,
    required_ticks: int,
    timeout_s: float,
    sleep: Callable[[float], None],
) -> None:
    """Wait until a latched policy target received all of its 60 Hz DLS ticks."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        # This counter is independent of the bounded diagnostics trace. Using
        # trace entries here used to falsely reject the final action as soon as
        # the old 300-entry trace buffer filled.
        snapshot = worker.snapshot()
        if (
            snapshot.latched_action_generation == generation
            and snapshot.latched_action_tick_count >= required_ticks
        ):
            return
        worker.hold_policy_target()
        sleep(0.001)
    raise RuntimeError(
        "server9 worker did not apply policy action "
        f"{generation} for {required_ticks} IK ticks"
    )


def _audit_fci_log(
    records: list[dict[str, Any]], config: BundleDeployConfig
) -> dict[str, Any]:
    """Reconstruct command derivatives using FCI's fixed 1 ms command period."""
    if not records:
        return {"num_records": 0}
    # libfranka's first Record can contain the default-zero RobotCommand paired
    # with the first nonzero robot state, before the user callback emitted a
    # command. It is initialization metadata, not a trajectory sample.
    discarded_leading_records = 0
    while discarded_leading_records < len(records):
        record = records[discarded_leading_records]
        if max(
            abs(float(command) - float(desired))
            for command, desired in zip(record["q_command"], record["q_d"])
        ) <= 0.1:
            break
        discarded_leading_records += 1
    records = records[discarded_leading_records:]
    if not records:
        return {
            "num_records": 0,
            "discarded_leading_initialization_records": discarded_leading_records,
        }
    commands = np.asarray([record["q_command"] for record in records], dtype=np.float64)
    velocities = np.diff(commands, axis=0) / 1e-3
    accelerations = np.diff(velocities, axis=0) / 1e-3
    jerks = np.diff(accelerations, axis=0) / 1e-3

    def peak(values: np.ndarray) -> dict[str, Any] | None:
        if values.size == 0:
            return None
        flat_index = int(np.argmax(np.abs(values)))
        record_offset, joint = np.unravel_index(flat_index, values.shape)
        return {
            "abs_value": float(abs(values[record_offset, joint])),
            "signed_value": float(values[record_offset, joint]),
            "joint_index": int(joint),
            "record_offset": int(record_offset),
        }

    q_d = np.asarray([record["q_d"] for record in records], dtype=np.float64)
    actual_dq = np.asarray([record["dq"] for record in records], dtype=np.float64)
    state_command_error = commands - q_d
    limits = {
        "velocity": list(config.streaming.maximum_joint_velocities),
        "acceleration": list(config.streaming.maximum_joint_accelerations),
        "jerk": list(config.streaming.maximum_joint_jerks),
    }
    violations: list[dict[str, Any]] = []
    for name, values, configured in (
        ("velocity", velocities, np.asarray(limits["velocity"])),
        ("acceleration", accelerations, np.asarray(limits["acceleration"])),
        ("jerk", jerks, np.asarray(limits["jerk"])),
    ):
        if values.size:
            positions = np.argwhere(np.abs(values) > configured.reshape(1, 7) + 1e-6)
            for record_offset, joint in positions[:20]:
                violations.append({
                    "quantity": name,
                    "record_offset": int(record_offset),
                    "joint_index": int(joint),
                    "value": float(values[record_offset, joint]),
                    "configured_limit": float(configured[joint]),
                })
    return {
        "num_records": len(records),
        "discarded_leading_initialization_records": discarded_leading_records,
        "fixed_command_period_s": 0.001,
        "configured_limits": limits,
        "peak_command_velocity": peak(velocities),
        "peak_command_acceleration": peak(accelerations),
        "peak_command_jerk": peak(jerks),
        "peak_actual_joint_velocity": peak(actual_dq),
        "final_actual_joint_velocity": actual_dq[-1].tolist(),
        "peak_abs_q_command_minus_state_q_d": peak(state_command_error),
        "violations": violations,
    }


def _write_fci_diagnostics(
    run_dir: Path, worker: Server9Worker, config: BundleDeployConfig
) -> None:
    reader = getattr(worker.memory, "fci_log", None)
    if reader is None:
        return
    records = reader()
    if not records:
        return
    with (run_dir / "fci_error_trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (run_dir / "fci_error_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(_audit_fci_log(records, config), handle, indent=2)


def _stop_resources(
    worker: Server9Worker | None,
    gripper_queue: AsyncGripperQueue | None,
    run_dir: Path,
    config: BundleDeployConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    trace: list[dict[str, Any]] = []
    if worker is not None:
        errors.extend(worker.stop_control())
        try:
            _write_fci_diagnostics(run_dir, worker, config)
        except Exception as exc:
            errors.append(f"FCI diagnostics: {exc}")
        try:
            trace = worker.memory.trace()
        except Exception as exc:
            errors.append(f"control trace: {exc}")
    if gripper_queue is not None:
        try:
            gripper_queue.stop()
        except Exception as exc:
            errors.append(f"gripper stop: {exc}")
    return errors, trace


def _preview(
    bundle: BundleTorchScriptPolicy,
    camera: Any,
    history: ActionHistoryBuffer,
    observation: RobotObservation,
    first_policy: _PolicyResult,
    gripper_queue: AsyncGripperQueue | None,
    config: BundleDeployConfig,
    allow_full_scale: bool,
    timing: dict[str, Any],
    records: list[dict[str, Any]],
    images: list[np.ndarray],
    clock_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> None:
    period_ns = round(1e9 / config.streaming.policy_frequency_hz)
    start_ns = clock_ns()
    for step_index in range(config.runner.steps):
        _sleep_until(start_ns + step_index * period_ns, clock_ns, sleep)
        result = first_policy if step_index == 0 else _run_policy_tick(
            bundle, camera, history, observation, config, allow_full_scale,
            float(gripper_queue.desired_width if gripper_queue else 0.0), clock_ns,
        )
        timing["maximum_policy_elapsed_ms"] = max(
            timing["maximum_policy_elapsed_ms"], result.elapsed_ns / 1e6
        )
        timely = policy_result_is_timely(
            result.elapsed_ns, period_ns, round(config.streaming.policy_watchdog_s * 1e9)
        )
        if timely:
            history.update(result.raw_action, result.executed_action)
        else:
            timing["policy_deadline_misses"] += 1
        records.append(_policy_record(step_index, observation, result, timely, False))
        images.append(result.image)
    elapsed_s = max(0.0, (clock_ns() - start_ns) / 1e9)
    timing.update({
        "preview_only": True,
        "policy_elapsed_s": elapsed_s,
        "effective_policy_frequency_hz": (
            (config.runner.steps - 1) / elapsed_s
            if config.runner.steps > 1 and elapsed_s > 0.0 else None
        ),
    })


def run_server9_streaming_bundle_deploy(
    config: BundleDeployConfig,
    execute_motion: bool = True,
    confirm_session_callback: Any | None = None,
    save_step_data: bool = False,
    allow_full_scale: bool = False,
    streaming_check: bool = False,
    *,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    run_dir = _make_run_dir(config)
    records: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    control_trace: list[dict[str, Any]] = []
    timing: dict[str, Any] = {
        "backend": "server9_joint_position",
        "policy_deadline_misses": 0,
        "absolute_policy_deadline_misses": 0,
        "maximum_policy_elapsed_ms": 0.0,
        "maximum_control_lateness_ms": 0.0,
        "maximum_policy_decision_lateness_ms": 0.0,
        "maximum_gripper_command_ms": 0.0,
        "maximum_gripper_poll_ms": 0.0,
        "maximum_latch_wait_ms": 0.0,
        "maximum_loop_body_ms": 0.0,
    }
    bundle: BundleTorchScriptPolicy | None = None
    camera: Any | None = None
    gripper_queue: AsyncGripperQueue | None = None
    worker: Server9Worker | None = None
    initial_report: dict[str, Any] = {"passed": False, "checks": {}, "failures": []}
    stopped = False
    printed_error_report = False
    try:
        if not streaming_check:
            bundle = BundleTorchScriptPolicy(
                config.model.model_path,
                config.model.metadata_path,
                config.model.device,
                torch_num_threads=config.model.torch_num_threads,
                optimize_for_inference=config.model.optimize_for_inference,
                rma_position_source=config.model.rma_position_source,
                rma_contact_source=config.model.rma_contact_source,
                rma_oracle_cube_position_root=config.model.rma_oracle_cube_position_root,
            )
            _validate_bundle_action_dims(bundle, config)
            validate_streaming_contract(bundle, config)

        worker = Server9Worker(config, worker_path=_worker_binary(config))
        initial_snapshot = worker.wait_ready()
        if not streaming_check:
            from franky import Gripper

            gripper_queue = AsyncGripperQueue(
                Gripper(config.robot_ip), config.gripper_speed,
                config.gripper_command_tolerance_m, config.gripper_force,
            )
            worker.set_abort_callback(gripper_queue.stop)
        initial_observation = snapshot_to_observation(
            initial_snapshot, gripper_queue.cached_state if gripper_queue else None,
            in_control=False,
        )
        initial_report = (
            evaluate_streaming_check_state(initial_observation, config)
            if streaming_check
            else evaluate_initial_state(initial_observation, config.initial_state)
        )
        if config.initial_state.enforce and not initial_report["passed"]:
            failures = "\n".join(f"  - {item}" for item in initial_report["failures"])
            raise RuntimeError(
                "Initial-state safety check failed; no streaming control was started.\n" + failures
            )

        history: ActionHistoryBuffer | None = None
        first_policy: _PolicyResult | None = None
        if not streaming_check:
            assert bundle is not None
            camera = _make_camera(
                config.camera,
                bundle.rgb_width,
                bundle.rgb_height,
                tactile_config=config.tactile_camera,
                gelsight_input_shapes=getattr(bundle, "gelsight_input_shapes", {}),
            )
            history = ActionHistoryBuffer(
                bundle.history_dim, config.model.history_source,
                config.model.history_scale, config.model.history_delay_steps,
                processed_action_scale=config.action_adapter.scales,
            )
            tactile_zeros = {
                name: np.zeros(shape, dtype=np.uint8)
                for name, shape in getattr(bundle, "gelsight_input_shapes", {}).items()
            }
            warmup_args = (
                np.zeros(bundle.history_dim, dtype=np.float32),
                np.zeros(bundle.proprio_dim, dtype=np.float32),
                np.zeros((bundle.rgb_height, bundle.rgb_width, 3), dtype=np.uint8),
            )
            if tactile_zeros:
                bundle.predict(
                    *warmup_args,
                    tactile_zeros["gsmini_left_rgb"],
                    tactile_zeros["gsmini_right_rgb"],
                )
            else:
                bundle.predict(*warmup_args)
            # The first D435 frame can exceed one 30 Hz policy period despite
            # camera startup warmup. Consume it before the proposed action and
            # before the FCI control loop; it cannot affect action history or motion.
            camera.read()
            first_policy = _run_policy_tick(
                bundle, camera, history, initial_observation, config, allow_full_scale,
                float(gripper_queue.desired_width if gripper_queue else 0.0), clock_ns,
                collect_rma_debug=_should_print_streaming_progress(1, config.runner.steps),
            )

        if not execute_motion:
            if streaming_check:
                raise ValueError("--streaming-check cannot be combined with --preview-only")
            assert bundle is not None and camera is not None and history is not None
            assert first_policy is not None
            _preview(
                bundle, camera, history, initial_observation, first_policy, gripper_queue,
                config, allow_full_scale, timing, records, images, clock_ns, sleep,
            )
            cleanup_errors, control_trace = _stop_resources(
                worker, gripper_queue, run_dir, config
            )
            stopped = True
            if cleanup_errors:
                raise RuntimeError("; ".join(cleanup_errors))
            return _write_streaming_artifacts(
                run_dir, config, initial_report, records, images, control_trace, timing,
                save_step_data,
            )

        if confirm_session_callback is not None:
            proposed_action = first_policy.robot_action if first_policy else RobotAction(
                speed=0.0, metadata={"streaming_check": True}
            )
            if not bool(confirm_session_callback(0, proposed_action, initial_observation)):
                timing["cancelled_by_user"] = True
                cleanup_errors, control_trace = _stop_resources(
                    worker, gripper_queue, run_dir, config
                )
                stopped = True
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))
                return _write_streaming_artifacts(
                    run_dir, config, initial_report, records, images, control_trace, timing,
                    save_step_data,
                )

        worker.start(streaming_check=streaming_check)
        _wait_running(worker, 2.0)
        if gripper_queue is not None:
            gripper_queue.mark_control_session_active()

        start_ns = clock_ns()
        period_ns = round(1e9 / config.streaming.policy_frequency_hz)
        watchdog_ns = round(config.streaming.policy_watchdog_s * 1e9)
        expected_ticks = (
            round(2.0 * config.streaming.ik_frequency_hz)
            if streaming_check else config.runner.steps * 2
        )
        if streaming_check:
            while worker.snapshot().ik_tick_count < expected_ticks:
                worker.hold_policy_target()
                sleep(0.002)
        else:
            assert bundle is not None and camera is not None and history is not None
            pending_result = first_policy
            last_accepted_generation: int | None = None
            for step_index in range(config.runner.steps):
                loop_started_ns = clock_ns()
                deadline_ns = start_ns + step_index * period_ns
                lateness_ns = _sleep_until(deadline_ns, clock_ns, sleep)
                timing["maximum_control_lateness_ms"] = max(
                    timing["maximum_control_lateness_ms"], max(0, lateness_ns) / 1e6
                )
                snapshot = worker.snapshot()
                observation = snapshot_to_observation(
                    snapshot, gripper_queue.cached_state if gripper_queue else None,
                    in_control=True,
                )
                if observation.has_errors:
                    print_error_wrench_report(
                        observation,
                        context=f"server9 streaming step {step_index}",
                    )
                    printed_error_report = True
                    raise RuntimeError("server9 robot reported errors")
                if observation.robot_mode not in {"Move", "Idle"}:
                    print_error_wrench_report(
                        observation,
                        context=f"server9 streaming step {step_index}",
                    )
                    printed_error_report = True
                    raise RuntimeError(
                        f"server9 robot entered unsafe mode {observation.robot_mode!r}"
                    )
                result = pending_result
                pending_result = None
                is_preflight_result = step_index == 0 and result is not None
                if result is None:
                    result = _run_policy_tick(
                        bundle, camera, history, observation, config, allow_full_scale,
                        float(gripper_queue.desired_width if gripper_queue else 0.0), clock_ns,
                        collect_rma_debug=_should_print_streaming_progress(
                            step_index + 1,
                            config.runner.steps,
                        ),
                    )
                timing["maximum_policy_elapsed_ms"] = max(
                    timing["maximum_policy_elapsed_ms"], result.elapsed_ns / 1e6
                )
                decision_lateness_ns = max(0, clock_ns() - deadline_ns)
                timing["maximum_policy_decision_lateness_ms"] = max(
                    timing["maximum_policy_decision_lateness_ms"],
                    decision_lateness_ns / 1e6,
                )
                absolute_deadline_expired = (
                    not is_preflight_result and decision_lateness_ns >= period_ns
                )
                timely = (
                    True
                    if is_preflight_result
                    else policy_result_is_timely(
                        result.elapsed_ns,
                        period_ns,
                        watchdog_ns,
                        decision_lateness_ns,
                    )
                )
                step_timing_ms: dict[str, float] = {
                    "deadline_lateness": decision_lateness_ns / 1e6,
                    "policy": result.elapsed_ns / 1e6,
                    "gripper_command": 0.0,
                    "gripper_poll": 0.0,
                    "latch_wait": 0.0,
                }
                if timely:
                    worker.send_action(step_index + 1, [
                        result.robot_action.dx, result.robot_action.dy, result.robot_action.dz,
                    ], snapshot.O_T_EE)
                    history.update(result.raw_action, result.executed_action)
                    if gripper_queue is not None and result.robot_action.gripper_width is not None:
                        gripper_started_ns = clock_ns()
                        gripper_queue.command(result.robot_action.gripper_width)
                        step_timing_ms["gripper_command"] = (
                            clock_ns() - gripper_started_ns
                        ) / 1e6
                    if step_index + 1 < config.runner.steps:
                        # The policy result for the next 30 Hz boundary must be
                        # ready before that boundary.  The native worker runs
                        # independently at 1 kHz, so this inference overlaps
                        # the two 60 Hz DLS applications of the current action.
                        next_snapshot = worker.snapshot()
                        next_observation = snapshot_to_observation(
                            next_snapshot,
                            gripper_queue.cached_state if gripper_queue else None,
                            in_control=True,
                        )
                        pending_result = _run_policy_tick(
                            bundle,
                            camera,
                            history,
                            next_observation,
                            config,
                            allow_full_scale,
                            float(gripper_queue.desired_width if gripper_queue else 0.0),
                            clock_ns,
                            collect_rma_debug=_should_print_streaming_progress(
                                step_index + 2,
                                config.runner.steps,
                            ),
                        )
                    latch_started_ns = clock_ns()
                    try:
                        worker.wait_latched(step_index + 1, config.streaming.control_watchdog_s)
                    except RuntimeError:
                        printed_error_report = _print_server9_error_snapshot(
                            worker,
                            gripper_queue,
                            context=(
                                f"server9 latch timeout at step {step_index} "
                                f"waiting action {step_index + 1}"
                            ),
                            force=True,
                        )
                        raise
                    step_timing_ms["latch_wait"] = (
                        clock_ns() - latch_started_ns
                    ) / 1e6
                    last_accepted_generation = step_index + 1
                else:
                    timing["policy_deadline_misses"] += 1
                    if absolute_deadline_expired:
                        timing["absolute_policy_deadline_misses"] += 1
                    worker.hold_policy_target()
                if gripper_queue is not None:
                    gripper_poll_started_ns = clock_ns()
                    gripper_queue.poll()
                    step_timing_ms["gripper_poll"] = (
                        clock_ns() - gripper_poll_started_ns
                    ) / 1e6
                step_timing_ms["loop_body"] = (clock_ns() - loop_started_ns) / 1e6
                timing["maximum_gripper_command_ms"] = max(
                    timing["maximum_gripper_command_ms"],
                    step_timing_ms["gripper_command"],
                )
                timing["maximum_gripper_poll_ms"] = max(
                    timing["maximum_gripper_poll_ms"],
                    step_timing_ms["gripper_poll"],
                )
                timing["maximum_latch_wait_ms"] = max(
                    timing["maximum_latch_wait_ms"], step_timing_ms["latch_wait"]
                )
                timing["maximum_loop_body_ms"] = max(
                    timing["maximum_loop_body_ms"], step_timing_ms["loop_body"]
                )
                records.append(_policy_record(
                    step_index,
                    observation,
                    result,
                    timely,
                    True,
                    deadline_lateness_ns=decision_lateness_ns,
                    timing_ms=step_timing_ms,
                ))
                images.append(result.image)
                _print_streaming_progress(
                    step_index + 1,
                    config.runner.steps,
                    timing["policy_deadline_misses"],
                    observation,
                    result,
                    accepted=timely,
                    timing_ms=step_timing_ms,
                )

            if last_accepted_generation is not None:
                try:
                    _wait_generation_ticks(
                        worker,
                        last_accepted_generation,
                        required_ticks=2,
                        timeout_s=max(
                            config.streaming.control_watchdog_s,
                            2.0 / config.streaming.ik_frequency_hz + 0.02,
                        ),
                        sleep=sleep,
                    )
                except RuntimeError:
                    printed_error_report = _print_server9_error_snapshot(
                        worker,
                        gripper_queue,
                        context=(
                            "server9 final generation wait timeout after action "
                            f"{last_accepted_generation}"
                        ),
                        force=True,
                    )
                    raise

        cleanup_errors, control_trace = _stop_resources(
            worker, gripper_queue, run_dir, config
        )
        stopped_ns = clock_ns()
        stopped = True
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))
        elapsed_s = max(0.0, (stopped_ns - start_ns) / 1e9)
        timing.update({
            "expected_control_ticks": expected_ticks,
            "actual_control_ticks": len(control_trace),
            "expected_policy_steps": 0 if streaming_check else config.runner.steps,
            "streaming_check": streaming_check,
            "control_elapsed_s": elapsed_s,
            "scheduled_horizon_s": expected_ticks / config.streaming.ik_frequency_hz,
            "effective_control_frequency_hz": (
                (len(control_trace) - 1) / elapsed_s
                if len(control_trace) > 1 and elapsed_s > 0.0 else None
            ),
            "effective_policy_frequency_hz": (
                (config.runner.steps - 1) / elapsed_s
                if not streaming_check and config.runner.steps > 1 and elapsed_s > 0.0
                else None
            ),
        })
        return _write_streaming_artifacts(
            run_dir, config, initial_report, records, images, control_trace, timing,
            save_step_data,
        )
    except BaseException as exc:
        if worker is not None and not printed_error_report:
            printed_error_report = _print_server9_error_snapshot(
                worker,
                gripper_queue,
                context="server9 streaming abort",
            )
        if not stopped:
            cleanup_errors, control_trace = _stop_resources(
                worker, gripper_queue, run_dir, config
            )
            stopped = True
            timing["cleanup_errors"] = cleanup_errors
        timing.update({
            "aborted": True, "exception_type": type(exc).__name__, "exception": str(exc),
        })
        try:
            _write_streaming_artifacts(
                run_dir, config, initial_report, records, images, control_trace, timing,
                save_step_data,
            )
        except Exception as write_exc:
            warnings.warn(
                f"Failed to write aborted streaming artifacts: {write_exc}",
                RuntimeWarning, stacklevel=2,
            )
        raise
    finally:
        if not stopped:
            cleanup_errors, _ = _stop_resources(worker, gripper_queue, run_dir, config)
            if cleanup_errors:
                warnings.warn("; ".join(cleanup_errors), RuntimeWarning, stacklevel=2)
        if worker is not None:
            worker.close()
        if camera is not None:
            try:
                camera.close()
            except Exception as exc:
                warnings.warn(f"camera close: {exc}", RuntimeWarning, stacklevel=2)
