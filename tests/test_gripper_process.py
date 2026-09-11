from __future__ import annotations

import functools
import multiprocessing as mp
import time
import types
import unittest

from franka_sim2real.gripper_process import ProcessGripperQueue


class _ImmediateFuture:
    def __init__(self, result: bool) -> None:
        self._result = result

    def wait(self, _timeout: float) -> bool:
        return True

    def get(self) -> bool:
        return self._result


class _PendingFuture:
    def wait(self, _timeout: float) -> bool:
        return False


class _ServoGripper:
    def __init__(self, _robot_ip: str, event_queue) -> None:
        self._events = event_queue
        self._state = types.SimpleNamespace(
            width=0.04, max_width=0.08, is_grasped=False
        )

    @property
    def state(self):
        return self._state

    def move_async(self, width: float, speed: float) -> _PendingFuture:
        self._events.put(("move", width, speed))
        return _PendingFuture()

    def stop(self) -> bool:
        self._state = types.SimpleNamespace(
            width=0.042, max_width=0.08, is_grasped=False
        )
        self._events.put(("stop",))
        return True

    def stop_async(self) -> _ImmediateFuture:
        raise AssertionError("servo preemption must use stop(), not stop_async()")


class _SlowStateGripper:
    def __init__(self, _robot_ip: str) -> None:
        self._state = types.SimpleNamespace(
            width=0.04, max_width=0.08, is_grasped=False
        )
        self._state_reads = 0

    @property
    def state(self):
        self._state_reads += 1
        if self._state_reads > 1:
            # Deliberately hold this process's GIL, like the observed native
            # Hand state call. A successful move must not wait for this read
            # before acknowledging the next 1 mm policy step.
            deadline = time.perf_counter() + 0.75
            while time.perf_counter() < deadline:
                pass
        return self._state

    def move_async(self, width: float, _speed: float) -> _ImmediateFuture:
        self._state = types.SimpleNamespace(
            width=width, max_width=0.08, is_grasped=False
        )
        return _ImmediateFuture(True)

    def stop(self) -> None:
        return None


class _BlockedCloseGripper:
    def __init__(self, _robot_ip: str) -> None:
        self._state = types.SimpleNamespace(
            width=0.04, max_width=0.08, is_grasped=False
        )

    @property
    def state(self):
        return self._state

    def move_async(self, _width: float, _speed: float) -> _ImmediateFuture:
        self._state = types.SimpleNamespace(
            width=0.025, max_width=0.08, is_grasped=False
        )
        return _ImmediateFuture(False)

    def grasp_async(
        self, width: float, _speed: float, _force: float
    ) -> _ImmediateFuture:
        self._state = types.SimpleNamespace(
            width=width, max_width=0.08, is_grasped=True
        )
        return _ImmediateFuture(True)

    def stop(self) -> None:
        return None


class _FailedOpenGripper:
    def __init__(self, _robot_ip: str) -> None:
        self._state = types.SimpleNamespace(
            width=0.04, max_width=0.08, is_grasped=False
        )

    @property
    def state(self):
        return self._state

    def move_async(self, _width: float, _speed: float) -> _ImmediateFuture:
        return _ImmediateFuture(False)

    def stop(self) -> None:
        return None


class ProcessGripperQueueTests(unittest.TestCase):
    def _queue(self, factory) -> ProcessGripperQueue:
        return ProcessGripperQueue(
            "test-robot",
            speed=0.2,
            tolerance=1e-4,
            start_method="spawn",
            gripper_factory=factory,
        )

    def test_successful_steps_do_not_query_slow_hand_state(self) -> None:
        queue = self._queue(_SlowStateGripper)
        try:
            command_times = []
            for width_mm in range(41, 61):
                started = time.perf_counter()
                queue.command(width_mm / 1000.0)
                command_times.append(time.perf_counter() - started)
                time.sleep(0.005)
            poll_times = []
            deadline = time.monotonic() + 0.08
            while time.monotonic() < deadline:
                started = time.perf_counter()
                queue.poll()
                poll_times.append(time.perf_counter() - started)
            self.assertLess(max(command_times), 0.02)
            self.assertLess(max(poll_times), 0.02)
            queue.wait_idle(timeout_s=0.5)
            self.assertAlmostEqual(queue.cached_state.width, 0.06)
        finally:
            queue.stop()

    def test_servo_steps_start_one_endpoint_motion_and_stop_on_hold(self) -> None:
        context = mp.get_context("spawn")
        events = context.Queue()
        queue = ProcessGripperQueue(
            "test-robot",
            speed=0.2,
            tolerance=1e-4,
            servo_frequency_hz=30.0,
            start_method="spawn",
            gripper_factory=functools.partial(_ServoGripper, event_queue=events),
        )
        try:
            queue.mark_control_session_active()
            queue.command(0.041)
            kind, endpoint, speed = events.get(timeout=1.0)
            self.assertEqual(kind, "move")
            self.assertAlmostEqual(endpoint, 0.08)
            self.assertAlmostEqual(speed, 0.03)

            queue.command(0.042)
            queue.command(0.042)
            self.assertEqual(events.get(timeout=1.0), ("stop",))
            queue.wait_idle(timeout_s=1.0)
            self.assertAlmostEqual(queue.cached_state.width, 0.042)
            self.assertAlmostEqual(queue.desired_width, 0.042)

            queue.command(0.041)
            kind, endpoint, speed = events.get(timeout=1.0)
            self.assertEqual(kind, "move")
            self.assertAlmostEqual(endpoint, 0.0)
            self.assertAlmostEqual(speed, 0.03)
            queue.hold_servo_target()
            self.assertEqual(events.get(timeout=1.0), ("stop",))
            queue.wait_idle(timeout_s=1.0)
            timing = queue.timing_report()
            self.assertEqual(timing["motion_request_count"], 2)
            self.assertEqual(timing["stop_request_count"], 2)
            self.assertIsNotNone(timing["maximum_publish_to_motion_request_ms"])
            self.assertIsNotNone(timing["maximum_move_async_call_ms"])
            self.assertIsNotNone(timing["maximum_publish_to_stop_request_ms"])
            self.assertIsNotNone(timing["maximum_stop_call_ms"])
            self.assertIsNotNone(timing["maximum_stop_state_read_ms"])
        finally:
            queue.stop()
            events.close()
            events.join_thread()

    def test_blocked_close_is_converted_to_force_grasp(self) -> None:
        queue = self._queue(_BlockedCloseGripper)
        try:
            queue.command(0.0)
            queue.wait_idle(timeout_s=2.0)
            self.assertTrue(queue.cached_state.is_grasped)
            self.assertAlmostEqual(queue.cached_state.width, 0.025)
        finally:
            queue.stop()

    def test_non_contact_failure_is_reported_to_parent(self) -> None:
        queue = self._queue(_FailedOpenGripper)
        queue.command(0.07)
        with self.assertRaisesRegex(RuntimeError, "kind=move"):
            queue.wait_idle(timeout_s=2.0)
        with self.assertRaisesRegex(RuntimeError, "kind=move"):
            queue.stop()


if __name__ == "__main__":
    unittest.main()
