from __future__ import annotations

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
            # Hand state call. It must not affect the policy process.
            deadline = time.perf_counter() + 0.12
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

    def test_slow_hand_state_does_not_stall_policy_process(self) -> None:
        queue = self._queue(_SlowStateGripper)
        try:
            started = time.perf_counter()
            queue.command(0.06)
            command_elapsed = time.perf_counter() - started
            poll_times = []
            deadline = time.monotonic() + 0.08
            while time.monotonic() < deadline:
                started = time.perf_counter()
                queue.poll()
                poll_times.append(time.perf_counter() - started)
            self.assertLess(command_elapsed, 0.02)
            self.assertLess(max(poll_times), 0.02)
            queue.wait_idle(timeout_s=2.0)
            self.assertAlmostEqual(queue.cached_state.width, 0.06)
        finally:
            queue.stop()

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
