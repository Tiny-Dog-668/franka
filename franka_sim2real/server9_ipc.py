from __future__ import annotations

import ctypes
import math
import mmap
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAGIC = 0x46524B4139535452
ABI_VERSION = 11
# Keep enough 60 Hz diagnostics for the full 150-step RMA episode, including
# temporary extra hold ticks while the Python policy loop catches up.
TRACE_CAPACITY = 1024
FCI_LOG_CAPACITY = 1024

CONTROL_LAW_JOINT_POSITION_PURSUIT = 0
CONTROL_LAW_SIM_ACTUATOR_VELOCITY = 1

CONTROL_LAW_IDS = {
    "joint_position_pursuit": CONTROL_LAW_JOINT_POSITION_PURSUIT,
    "sim_actuator_velocity": CONTROL_LAW_SIM_ACTUATOR_VELOCITY,
}

IMPEDANCE_MODE_JOINT = 0
IMPEDANCE_MODE_CARTESIAN = 1

IMPEDANCE_MODE_IDS = {
    "joint": IMPEDANCE_MODE_JOINT,
    "cartesian": IMPEDANCE_MODE_CARTESIAN,
}

COMMAND_WAIT = 0
COMMAND_START = 1
COMMAND_STOP = 2

STATUS_INITIALIZING = 0
STATUS_READY = 1
STATUS_RUNNING = 2
STATUS_STOPPED = 3
STATUS_ABORTED = 4

STATUS_NAMES = {
    STATUS_INITIALIZING: "initializing",
    STATUS_READY: "ready",
    STATUS_RUNNING: "running",
    STATUS_STOPPED: "stopped",
    STATUS_ABORTED: "aborted",
}


class TraceEntry(ctypes.Structure):
    _fields_ = [
        ("tick_index", ctypes.c_uint64),
        ("monotonic_ns", ctypes.c_uint64),
        ("action_generation", ctypes.c_uint64),
        ("q", ctypes.c_double * 7),
        ("q_target", ctypes.c_double * 7),
        ("command_success_rate", ctypes.c_double),
    ]


class FciLogEntry(ctypes.Structure):
    _fields_ = [
        ("q_command", ctypes.c_double * 7),
        ("q_d", ctypes.c_double * 7),
        ("dq_d", ctypes.c_double * 7),
        ("ddq_d", ctypes.c_double * 7),
        ("dq", ctypes.c_double * 7),
        ("command_success_rate", ctypes.c_double),
    ]


class SharedData(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint64),
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("command_seq", ctypes.c_uint64),
        ("parent_heartbeat_ns", ctypes.c_uint64),
        ("policy_heartbeat_ns", ctypes.c_uint64),
        ("action_generation", ctypes.c_uint64),
        ("command", ctypes.c_uint32),
        ("streaming_check", ctypes.c_uint32),
        ("delta_xyz", ctypes.c_double * 3),
        ("target_pose", ctypes.c_double * 16),
        ("maximum_joint_velocities", ctypes.c_double * 7),
        ("joint_impedance", ctypes.c_double * 7),
        ("cartesian_impedance", ctypes.c_double * 6),
        ("impedance_mode", ctypes.c_uint64),
        ("maximum_joint_accelerations", ctypes.c_double * 7),
        ("maximum_joint_jerks", ctypes.c_double * 7),
        ("dls_lambda", ctypes.c_double),
        ("maximum_joint_target_delta_rad", ctypes.c_double),
        ("joint_limit_margin_rad", ctypes.c_double),
        ("policy_watchdog_s", ctypes.c_double),
        ("control_watchdog_s", ctypes.c_double),
        ("workspace_minimum", ctypes.c_double * 3),
        ("workspace_maximum", ctypes.c_double * 3),
        ("tool_tcp_offset_ee", ctypes.c_double * 3),
        ("control_law", ctypes.c_uint64),
        ("reference_velocity_gain", ctypes.c_double),
        ("maximum_ik_reference_delta_rad", ctypes.c_double),
        ("state_seq", ctypes.c_uint64),
        ("worker_heartbeat_ns", ctypes.c_uint64),
        ("control_cycle_count", ctypes.c_uint64),
        ("ik_tick_count", ctypes.c_uint64),
        ("latched_action_generation", ctypes.c_uint64),
        ("latched_action_tick_count", ctypes.c_uint64),
        ("trace_count", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
        ("error_code", ctypes.c_int32),
        ("robot_mode", ctypes.c_uint32),
        ("has_errors", ctypes.c_uint32),
        ("command_success_rate", ctypes.c_double),
        ("q", ctypes.c_double * 7),
        ("dq", ctypes.c_double * 7),
        ("O_T_EE", ctypes.c_double * 16),
        # The active flange-to-end-effector transform configured in the robot.
        # O_T_EE already includes this transform, but exposing it lets a
        # deployment verify that its physical tool TCP matches training.
        ("F_T_EE", ctypes.c_double * 16),
        ("external_wrench", ctypes.c_double * 6),
        ("q_target", ctypes.c_double * 7),
        ("error_message", ctypes.c_char * 256),
        ("trace", TraceEntry * TRACE_CAPACITY),
        ("fci_log_count", ctypes.c_uint64),
        ("fci_log", FciLogEntry * FCI_LOG_CAPACITY),
        ("collision_behavior_enabled", ctypes.c_uint32),
        ("collision_behavior_reserved", ctypes.c_uint32),
        ("lower_torque_thresholds", ctypes.c_double * 7),
        ("upper_torque_thresholds", ctypes.c_double * 7),
        ("lower_force_thresholds", ctypes.c_double * 6),
        ("upper_force_thresholds", ctypes.c_double * 6),
    ]


EXPECTED_LAYOUT = {
    "shared_size": 444024,
    "trace_size": 144,
    "fci_log_size": 288,
    "command_seq": 16,
    "state_seq": 624,
    "trace": 1432,
    "fci_log": 148896,
    "collision_behavior_enabled": 443808,
}


def validate_ctypes_layout() -> None:
    actual = {
        "shared_size": ctypes.sizeof(SharedData),
        "trace_size": ctypes.sizeof(TraceEntry),
        "fci_log_size": ctypes.sizeof(FciLogEntry),
        "command_seq": SharedData.command_seq.offset,
        "state_seq": SharedData.state_seq.offset,
        "trace": SharedData.trace.offset,
        "fci_log": SharedData.fci_log.offset,
        "collision_behavior_enabled": SharedData.collision_behavior_enabled.offset,
    }
    if actual != EXPECTED_LAYOUT:
        raise RuntimeError(f"server9 shared-memory ABI mismatch: {actual!r}")


@dataclass(frozen=True)
class WorkerSnapshot:
    status: int
    error_code: int
    error_message: str
    worker_heartbeat_ns: int
    control_cycle_count: int
    ik_tick_count: int
    latched_action_generation: int
    latched_action_tick_count: int
    trace_count: int
    robot_mode: int
    has_errors: bool
    command_success_rate: float
    q: tuple[float, ...]
    dq: tuple[float, ...]
    O_T_EE: tuple[float, ...]
    F_T_EE: tuple[float, ...]
    external_wrench: tuple[float, ...]
    q_target: tuple[float, ...]

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"unknown({self.status})")


class Server9SharedMemory:
    def __init__(self, config: Any, *, directory: str = "/dev/shm") -> None:
        validate_ctypes_layout()
        descriptor, path = tempfile.mkstemp(prefix="franka_server9_", dir=directory)
        self.path = Path(path)
        self._closed = False
        try:
            os.ftruncate(descriptor, ctypes.sizeof(SharedData))
            self._mapping = mmap.mmap(descriptor, ctypes.sizeof(SharedData))
        finally:
            os.close(descriptor)
        self.shared = SharedData.from_buffer(self._mapping)
        ctypes.memset(ctypes.addressof(self.shared), 0, ctypes.sizeof(self.shared))
        self.shared.magic = MAGIC
        self.shared.abi_version = ABI_VERSION
        self.shared.struct_size = ctypes.sizeof(SharedData)
        self.shared.parent_heartbeat_ns = time.monotonic_ns()
        self.shared.policy_heartbeat_ns = time.monotonic_ns()
        self.shared.command = COMMAND_WAIT
        self.shared.streaming_check = 0
        self._copy_array(
            self.shared.maximum_joint_velocities,
            config.streaming.maximum_joint_velocities,
        )
        joint_impedance = config.streaming.joint_impedance
        self._copy_array(
            self.shared.joint_impedance,
            ([math.nan] * 7 if joint_impedance is None else joint_impedance),
        )
        cartesian_impedance = config.streaming.cartesian_impedance
        self._copy_array(
            self.shared.cartesian_impedance,
            ([math.nan] * 6 if cartesian_impedance is None else cartesian_impedance),
        )
        impedance_mode = str(config.streaming.impedance_mode)
        if impedance_mode not in IMPEDANCE_MODE_IDS:
            valid = ", ".join(sorted(IMPEDANCE_MODE_IDS))
            raise ValueError(f"streaming.impedance_mode must be one of: {valid}")
        self.shared.impedance_mode = IMPEDANCE_MODE_IDS[impedance_mode]
        collision_behavior = config.streaming.collision_behavior
        if collision_behavior is not None:
            lower_torque = self._validated_thresholds(
                "lower_torque_thresholds",
                collision_behavior.lower_torque_thresholds,
                7,
            )
            upper_torque = self._validated_thresholds(
                "upper_torque_thresholds",
                collision_behavior.upper_torque_thresholds,
                7,
            )
            lower_force = self._validated_thresholds(
                "lower_force_thresholds",
                collision_behavior.lower_force_thresholds,
                6,
            )
            upper_force = self._validated_thresholds(
                "upper_force_thresholds",
                collision_behavior.upper_force_thresholds,
                6,
            )
            self._validate_threshold_pair("torque", lower_torque, upper_torque)
            self._validate_threshold_pair("force", lower_force, upper_force)
            self._copy_array(self.shared.lower_torque_thresholds, lower_torque)
            self._copy_array(self.shared.upper_torque_thresholds, upper_torque)
            self._copy_array(self.shared.lower_force_thresholds, lower_force)
            self._copy_array(self.shared.upper_force_thresholds, upper_force)
            self.shared.collision_behavior_enabled = 1
        self._copy_array(
            self.shared.maximum_joint_accelerations,
            config.streaming.maximum_joint_accelerations,
        )
        self._copy_array(
            self.shared.maximum_joint_jerks,
            config.streaming.maximum_joint_jerks,
        )
        self.shared.dls_lambda = config.streaming.dls_lambda
        self.shared.maximum_joint_target_delta_rad = (
            config.streaming.maximum_joint_target_delta_rad
        )
        self.shared.joint_limit_margin_rad = config.streaming.joint_limit_margin_rad
        self.shared.policy_watchdog_s = config.streaming.policy_watchdog_s
        self.shared.control_watchdog_s = config.streaming.control_watchdog_s
        self._copy_array(self.shared.workspace_minimum, config.workspace["minimum"])
        self._copy_array(self.shared.workspace_maximum, config.workspace["maximum"])
        self._copy_array(self.shared.tool_tcp_offset_ee, config.tool_tcp_offset_ee_m)
        control_law = str(config.streaming.control_law)
        if control_law not in CONTROL_LAW_IDS:
            valid = ", ".join(sorted(CONTROL_LAW_IDS))
            raise ValueError(f"streaming.control_law must be one of: {valid}")
        self.shared.control_law = CONTROL_LAW_IDS[control_law]
        self.shared.reference_velocity_gain = float(config.streaming.reference_velocity_gain)
        self.shared.maximum_ik_reference_delta_rad = float(
            config.streaming.maximum_ik_reference_delta_rad
        )

    @staticmethod
    def _copy_array(destination: Any, source: Any) -> None:
        values = list(source)
        if len(values) != len(destination):
            raise ValueError("shared-memory array dimension mismatch")
        for index, value in enumerate(values):
            destination[index] = float(value)

    @staticmethod
    def _validated_thresholds(name: str, source: Any, expected_size: int) -> list[float]:
        values = list(source)
        if len(values) != expected_size:
            raise ValueError(
                f"streaming.collision_behavior.{name} must contain "
                f"{expected_size} values"
            )
        converted: list[float] = []
        for value in values:
            if isinstance(value, bool):
                raise ValueError(
                    f"streaming.collision_behavior.{name} values must be finite and positive"
                )
            converted.append(float(value))
        if any(not math.isfinite(value) or value <= 0.0 for value in converted):
            raise ValueError(
                f"streaming.collision_behavior.{name} values must be finite and positive"
            )
        return converted

    @staticmethod
    def _validate_threshold_pair(
        quantity: str, lower: list[float], upper: list[float]
    ) -> None:
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError(
                "streaming.collision_behavior lower "
                f"{quantity} thresholds must not exceed upper thresholds"
            )

    def heartbeat(self) -> None:
        self.shared.parent_heartbeat_ns = time.monotonic_ns()

    def policy_heartbeat(self) -> None:
        self.shared.policy_heartbeat_ns = time.monotonic_ns()

    def command(
        self,
        value: int,
        *,
        streaming_check: bool | None = None,
        action_generation: int | None = None,
        delta_xyz: Any | None = None,
        target_pose: Any | None = None,
    ) -> None:
        sequence = int(self.shared.command_seq)
        self.shared.command_seq = sequence + 1 if sequence % 2 == 0 else sequence + 2
        if streaming_check is not None:
            self.shared.streaming_check = bool(streaming_check)
        if action_generation is not None:
            self.shared.action_generation = int(action_generation)
        if delta_xyz is not None:
            self._copy_array(self.shared.delta_xyz, delta_xyz)
        if target_pose is not None:
            self._copy_array(self.shared.target_pose, target_pose)
        self.shared.command = int(value)
        self.shared.command_seq += 1

    def snapshot(self) -> WorkerSnapshot:
        for _ in range(10000):
            before = int(self.shared.state_seq)
            if before % 2:
                continue
            blob = bytes(self._mapping)
            after = int(self.shared.state_seq)
            if before == after and after % 2 == 0:
                copied = SharedData.from_buffer_copy(blob)
                message = bytes(copied.error_message).split(b"\0", 1)[0].decode(
                    "utf-8", errors="replace"
                )
                return WorkerSnapshot(
                    status=int(copied.status),
                    error_code=int(copied.error_code),
                    error_message=message,
                    worker_heartbeat_ns=int(copied.worker_heartbeat_ns),
                    control_cycle_count=int(copied.control_cycle_count),
                    ik_tick_count=int(copied.ik_tick_count),
                    latched_action_generation=int(copied.latched_action_generation),
                    latched_action_tick_count=int(copied.latched_action_tick_count),
                    trace_count=int(copied.trace_count),
                    robot_mode=int(copied.robot_mode),
                    has_errors=bool(copied.has_errors),
                    command_success_rate=float(copied.command_success_rate),
                    q=tuple(copied.q),
                    dq=tuple(copied.dq),
                    O_T_EE=tuple(copied.O_T_EE),
                    F_T_EE=tuple(copied.F_T_EE),
                    external_wrench=tuple(copied.external_wrench),
                    q_target=tuple(copied.q_target),
                )
        raise RuntimeError("server9 state seqlock remained busy")

    def trace(self) -> list[dict[str, Any]]:
        count = min(int(self.shared.trace_count), TRACE_CAPACITY)
        return [
            {
                "tick_index": int(entry.tick_index),
                "observed_ns": int(entry.monotonic_ns),
                "action_generation": int(entry.action_generation),
                "policy_step": int(entry.action_generation - 1)
                if entry.action_generation
                else None,
                "q": list(entry.q),
                "q_target": list(entry.q_target),
                "command_success_rate": float(entry.command_success_rate),
            }
            for entry in self.shared.trace[:count]
        ]

    def fci_log(self) -> list[dict[str, Any]]:
        count = min(int(self.shared.fci_log_count), FCI_LOG_CAPACITY)
        return [
            {
                "record_index": index,
                "q_command": list(entry.q_command),
                "q_d": list(entry.q_d),
                "dq_d": list(entry.dq_d),
                "ddq_d": list(entry.ddq_d),
                "dq": list(entry.dq),
                "command_success_rate": float(entry.command_success_rate),
            }
            for index, entry in enumerate(self.shared.fci_log[:count])
        ]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        del self.shared
        self._mapping.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


class Server9Worker:
    def __init__(self, config: Any, *, worker_path: str | Path) -> None:
        path = Path(worker_path).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"server9 streaming worker is missing: {path}. "
                "Build it with scripts/build_franka_server9_worker.sh."
            )
        self.memory = Server9SharedMemory(config)
        self._heartbeat_stop = threading.Event()
        self._abort_callback: Any | None = None
        self._abort_notified = False
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, name="franka-server9-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()
        try:
            worker_command = [str(path), config.robot_ip, str(self.memory.path)]
            control_cpu = config.streaming.server9_control_cpu
            if control_cpu is not None:
                if isinstance(control_cpu, bool) or not isinstance(control_cpu, int) or control_cpu < 0:
                    raise ValueError(
                        "streaming.server9_control_cpu must be a non-negative integer or null"
                    )
                worker_command = ["taskset", "--cpu-list", str(control_cpu), *worker_command]
            self.process = subprocess.Popen(
                worker_command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except BaseException:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=1.0)
            self.memory.close()
            raise
        self._closed = False

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(0.01):
            self.memory.heartbeat()
            # Liveness is independent of timely policy inference.  A late
            # policy result is handled by the native worker's two-tick action
            # lifetime; this heartbeat only tells it that the supervising
            # process has not died.
            self.memory.policy_heartbeat()
            if (
                not self._abort_notified
                and int(self.memory.shared.status) == STATUS_ABORTED
            ):
                self._abort_notified = True
                callback = self._abort_callback
                if callback is not None:
                    try:
                        callback()
                    except Exception:
                        pass

    def set_abort_callback(self, callback: Any) -> None:
        self._abort_callback = callback

    def snapshot(self) -> WorkerSnapshot:
        snapshot = self.memory.snapshot()
        if snapshot.status == STATUS_ABORTED:
            raise RuntimeError(
                "server9 control worker aborted"
                + (f": {snapshot.error_message}" if snapshot.error_message else "")
            )
        return snapshot

    def wait_ready(self, timeout_s: float = 10.0) -> WorkerSnapshot:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            snapshot = self.snapshot()
            if snapshot.status == STATUS_READY:
                return snapshot
            if self.process.poll() is not None:
                break
            time.sleep(0.005)
        stderr = self._read_stderr()
        raise RuntimeError(
            "server9 streaming worker did not become ready"
            + (f": {stderr}" if stderr else "")
        )

    def start(self, *, streaming_check: bool) -> None:
        self.memory.policy_heartbeat()
        self.memory.command(COMMAND_START, streaming_check=streaming_check)

    def send_action(self, generation: int, delta_xyz: Any, anchor_pose: Any) -> None:
        delta = list(delta_xyz)
        target = list(anchor_pose)
        if len(delta) != 3 or len(target) != 16:
            raise ValueError("server9 action needs 3D delta and 4x4 anchor pose")
        target[12] += float(delta[0])
        target[13] += float(delta[1])
        target[14] += float(delta[2])
        self.memory.policy_heartbeat()
        self.memory.command(
            COMMAND_START,
            action_generation=generation,
            delta_xyz=delta,
            target_pose=target,
        )

    def hold_policy_target(self) -> None:
        self.memory.policy_heartbeat()

    def wait_latched(self, generation: int, timeout_s: float) -> WorkerSnapshot:
        deadline = time.monotonic() + timeout_s
        while True:
            snapshot = self.snapshot()
            if snapshot.latched_action_generation >= generation:
                return snapshot
            if time.monotonic() >= deadline:
                break
            time.sleep(0.0005)
        raise RuntimeError(
            "server9 worker did not latch policy action "
            f"{generation} within {timeout_s * 1e3:.1f} ms "
            f"(last latched {snapshot.latched_action_generation}; "
            f"worker_status={snapshot.status_name}; "
            f"worker_error_code={snapshot.error_code}; "
            f"worker_error_message={snapshot.error_message or '<none>'}; "
            f"control_cycles={snapshot.control_cycle_count}; "
            f"ik_ticks={snapshot.ik_tick_count}; "
            f"latched_action_ticks={snapshot.latched_action_tick_count})"
        )

    def _read_stderr(self) -> str:
        if self.process.stderr is None or self.process.poll() is None:
            return ""
        return self.process.stderr.read().strip()

    def stop_control(self) -> list[str]:
        errors: list[str] = []
        if self.process.poll() is None:
            try:
                self.memory.command(COMMAND_STOP)
                self.process.wait(timeout=2.0)
            except Exception as exc:
                errors.append(f"arm stop_control: {exc}")
                self.process.terminate()
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
        if self.process.returncode not in (None, 0):
            try:
                snapshot = self.memory.snapshot()
                detail = snapshot.error_message
            except Exception:
                detail = self._read_stderr()
            if detail:
                errors.append(f"server9 worker: {detail}")
        return errors

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.stop_control()
        self._heartbeat_stop.set()
        self._heartbeat_thread.join(timeout=1.0)
        if self.process.stderr is not None:
            self.process.stderr.close()
        self.memory.close()
