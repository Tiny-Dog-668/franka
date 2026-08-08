from __future__ import annotations

import math
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from franka_sim2real.e2e_bundle import BundleDeployConfig
from franka_sim2real.server9_ipc import (
    COMMAND_START,
    STATUS_READY,
    STATUS_RUNNING,
    Server9SharedMemory,
    WorkerSnapshot,
    validate_ctypes_layout,
)
from franka_sim2real.streaming_server9 import (
    _audit_fci_log,
    _print_streaming_progress,
    _wait_generation_ticks,
    snapshot_to_observation,
)


class Server9IpcTests(unittest.TestCase):
    def test_streaming_progress_prints_every_30_steps_and_at_end(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            _print_streaming_progress(29, 65, 0)
            _print_streaming_progress(30, 65, 1)
            _print_streaming_progress(60, 65, 2)
            _print_streaming_progress(65, 65, 2)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "Streaming progress: 30/65 steps (accepted=29, deadline_misses=1)",
                "Streaming progress: 60/65 steps (accepted=58, deadline_misses=2)",
                "Streaming progress: 65/65 steps (accepted=63, deadline_misses=2)",
            ],
        )

    def test_fci_audit_finds_command_jerk_violation(self) -> None:
        config = BundleDeployConfig()
        config.streaming.maximum_joint_velocities = [1.0] * 7
        config.streaming.maximum_joint_accelerations = [7.0] * 7
        config.streaming.maximum_joint_jerks = [1500.0] * 7
        records = []
        for index, q0 in enumerate([0.0, 0.0, 0.0000015, 0.0000100]):
            q = [0.0] * 7
            q[0] = q0
            records.append({
                "record_index": index,
                "q_command": q,
                "q_d": q,
                "dq_d": [0.0] * 7,
                "ddq_d": [0.0] * 7,
                "dq": [0.0] * 7,
                "command_success_rate": 1.0,
            })
        audit = _audit_fci_log(records, config)
        self.assertEqual(audit["num_records"], 4)
        self.assertTrue(
            any(item["quantity"] == "jerk" for item in audit["violations"])
        )

    def test_wait_generation_ticks_requires_two_matching_dls_ticks(self) -> None:
        class FakeWorker:
            calls = 0

            def snapshot(self):
                self.calls += 1
                return type("Snapshot", (), {
                    "latched_action_generation": 7,
                    "latched_action_tick_count": min(self.calls, 2),
                })()

            def hold_policy_target(self):
                return None

        _wait_generation_ticks(FakeWorker(), 7, 2, 0.1, lambda _: None)

    def test_python_layout_matches_native_contract(self) -> None:
        validate_ctypes_layout()
        binary = (
            Path(__file__).resolve().parents[1]
            / "dist/franka_server9/franka_server9_streaming_worker"
        )
        if binary.exists():
            output = subprocess.check_output([binary, "--self-test"], text=True)
            self.assertIn("PASS", output)

    def test_command_snapshot_and_trace_round_trip(self) -> None:
        config = BundleDeployConfig()
        with tempfile.TemporaryDirectory() as directory:
            memory = Server9SharedMemory(config, directory=directory)
            try:
                memory.command(
                    COMMAND_START,
                    streaming_check=False,
                    action_generation=7,
                    delta_xyz=[0.001, -0.002, 0.003],
                    target_pose=[1.0, 0.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0, 0.0,
                                 0.0, 0.0, 1.0, 0.0,
                                 0.5, 0.0, 0.3, 1.0],
                )
                self.assertEqual(memory.shared.command, COMMAND_START)
                self.assertEqual(memory.shared.action_generation, 7)
                self.assertEqual(list(memory.shared.delta_xyz), [0.001, -0.002, 0.003])
                self.assertEqual(memory.shared.maximum_joint_target_delta_rad, 0.02)
                self.assertTrue(all(math.isnan(value) for value in memory.shared.joint_impedance))
                self.assertEqual(list(memory.shared.maximum_joint_accelerations), [10.0] * 7)
                self.assertEqual(list(memory.shared.maximum_joint_jerks), [5000.0] * 7)
                self.assertEqual(memory.shared.command_seq % 2, 0)

                memory.shared.state_seq = 1
                memory.shared.status = STATUS_READY
                memory.shared.robot_mode = 1
                memory.shared.command_success_rate = 1.0
                memory.shared.O_T_EE[0] = 1.0
                memory.shared.O_T_EE[5] = 1.0
                memory.shared.O_T_EE[10] = 1.0
                memory.shared.O_T_EE[15] = 1.0
                memory.shared.O_T_EE[12] = 0.5
                memory.shared.O_T_EE[14] = 0.3
                memory.shared.state_seq = 2
                snapshot = memory.snapshot()
                self.assertEqual(snapshot.status, STATUS_READY)
                observation = snapshot_to_observation(snapshot, None, in_control=False)
                self.assertEqual(observation.robot_mode, "Idle")
                self.assertEqual(observation.tcp_translation, [0.5, 0.0, 0.3])

                memory.shared.trace_count = 1
                memory.shared.trace[0].tick_index = 3
                memory.shared.trace[0].action_generation = 2
                memory.shared.trace[0].command_success_rate = 0.99
                trace = memory.trace()
                self.assertEqual(trace[0]["tick_index"], 3)
                self.assertEqual(trace[0]["policy_step"], 1)

                memory.shared.fci_log_count = 1
                memory.shared.fci_log[0].q_command[0] = 0.25
                memory.shared.fci_log[0].dq_d[0] = 0.1
                memory.shared.fci_log[0].dq[0] = 0.2
                fci_log = memory.fci_log()
                self.assertEqual(fci_log[0]["q_command"][0], 0.25)
                self.assertEqual(fci_log[0]["dq_d"][0], 0.1)
                self.assertEqual(fci_log[0]["dq"][0], 0.2)
            finally:
                memory.close()

    def test_collision_thresholds_round_trip_to_shared_memory(self) -> None:
        config = BundleDeployConfig.from_dict({
            "streaming": {
                "collision_behavior": {
                    "lower_torque_thresholds": [20.0] * 7,
                    "upper_torque_thresholds": [40.0] * 7,
                    "lower_force_thresholds": [10.0] * 6,
                    "upper_force_thresholds": [20.0, 20.0, 25.0, 20.0, 20.0, 20.0],
                }
            }
        })
        with tempfile.TemporaryDirectory() as directory:
            memory = Server9SharedMemory(config, directory=directory)
            try:
                self.assertEqual(memory.shared.collision_behavior_enabled, 1)
                self.assertEqual(list(memory.shared.lower_torque_thresholds), [20.0] * 7)
                self.assertEqual(list(memory.shared.upper_torque_thresholds), [40.0] * 7)
                self.assertEqual(list(memory.shared.lower_force_thresholds), [10.0] * 6)
                self.assertEqual(
                    list(memory.shared.upper_force_thresholds),
                    [20.0, 20.0, 25.0, 20.0, 20.0, 20.0],
                )
            finally:
                memory.close()

    def test_collision_thresholds_reject_lower_above_upper(self) -> None:
        config = BundleDeployConfig.from_dict({
            "streaming": {
                "collision_behavior": {
                    "lower_force_thresholds": [30.0] * 6,
                    "upper_force_thresholds": [20.0] * 6,
                }
            }
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "lower force thresholds"):
                Server9SharedMemory(config, directory=directory)

    def test_fake_streaming_check_runs_120_ik_ticks_and_stops(self) -> None:
        config = BundleDeployConfig()
        config.control_mode = "streaming"
        config.realtime = "enforce"
        config.streaming.backend = "server9_joint_position"
        config.initial_state.enforce = False

        class FakeMemory:
            def __init__(self, owner):
                self.owner = owner

            def trace(self):
                return [
                    {"tick_index": index, "q": [0.0] * 7, "q_target": [0.0] * 7}
                    for index in range(self.owner.ik_ticks)
                ]

        class FakeWorker:
            instance = None

            def __init__(self, unused_config, worker_path):
                del worker_path
                self.status = STATUS_READY
                self.ik_ticks = 0
                self.stop_calls = 0
                self.memory = FakeMemory(self)
                FakeWorker.instance = self

            def _snapshot(self):
                return WorkerSnapshot(
                    status=self.status,
                    error_code=0,
                    error_message="",
                    worker_heartbeat_ns=0,
                    control_cycle_count=self.ik_ticks * 16,
                    ik_tick_count=self.ik_ticks,
                    latched_action_generation=0,
                    latched_action_tick_count=0,
                    trace_count=self.ik_ticks,
                    robot_mode=1,
                    has_errors=False,
                    command_success_rate=1.0,
                    q=(0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0),
                    dq=(0.0,) * 7,
                    O_T_EE=(1.0, 0.0, 0.0, 0.0,
                            0.0, 1.0, 0.0, 0.0,
                            0.0, 0.0, 1.0, 0.0,
                            0.5, 0.0, 0.3, 1.0),
                    external_wrench=(0.0,) * 6,
                    q_target=(0.0,) * 7,
                )

            def wait_ready(self):
                return self._snapshot()

            def start(self, streaming_check):
                self.status = STATUS_RUNNING
                self.streaming_check = streaming_check

            def snapshot(self):
                if self.status == STATUS_RUNNING and self.ik_ticks < 120:
                    self.ik_ticks += 1
                return self._snapshot()

            def hold_policy_target(self):
                pass

            def stop_control(self):
                self.stop_calls += 1
                return []

            def close(self):
                pass

        class FakeClock:
            now = 0

            def __call__(self):
                return self.now

            def sleep(self, seconds):
                self.now += round(seconds * 1e9)

        from franka_sim2real import streaming_server9

        clock = FakeClock()
        with tempfile.TemporaryDirectory() as directory, patch.object(
            streaming_server9, "Server9Worker", FakeWorker
        ), patch.object(
            streaming_server9, "_make_run_dir", return_value=Path(directory)
        ):
            summary = streaming_server9.run_server9_streaming_bundle_deploy(
                config,
                streaming_check=True,
                clock_ns=clock,
                sleep=clock.sleep,
            )
        self.assertEqual(summary["num_steps"], 0)
        self.assertEqual(summary["num_control_ticks"], 120)
        self.assertTrue(FakeWorker.instance.streaming_check)
        self.assertEqual(FakeWorker.instance.stop_calls, 1)


if __name__ == "__main__":
    unittest.main()
