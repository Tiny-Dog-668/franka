from __future__ import annotations

import argparse
import importlib.util
import types
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "diagnostics"
    / "test_franka_gripper_latency.py"
)
SPEC = importlib.util.spec_from_file_location("franka_gripper_latency", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def args(**overrides):
    values = {
        "travel_mm": 6.0,
        "speed": 0.03,
        "stop_after_ms": 50.0,
        "natural_timeout_s": 5.0,
        "mode": "compare",
        "repeats": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _StateGripper:
    def __init__(self, width: float = 0.04, max_width: float = 0.08) -> None:
        self.state = types.SimpleNamespace(
            width=width, max_width=max_width, is_grasped=False
        )


class _Future:
    def __init__(self, completed: bool, result: bool = True) -> None:
        self.completed = completed
        self.result = result

    def wait(self, _timeout: float) -> bool:
        return self.completed

    def get(self) -> bool:
        return self.result


class _StopGripper(_StateGripper):
    def __init__(self) -> None:
        super().__init__()
        self.future = _Future(False)
        self.stop_count = 0

    def move_async(self, target: float, _speed: float) -> _Future:
        self.target = target
        return self.future

    def stop(self) -> bool:
        self.stop_count += 1
        self.future.completed = True
        self.state.width = 0.042
        return True


class _NaturalGripper(_StateGripper):
    def move_async(self, target: float, _speed: float) -> _Future:
        self.state.width = target
        return _Future(True)


class FrankaGripperLatencyTests(unittest.TestCase):
    def test_default_parameters_are_valid(self) -> None:
        MODULE.validate_parameters(args())

    def test_rejects_unbounded_travel(self) -> None:
        with self.assertRaisesRegex(ValueError, "travel-mm"):
            MODULE.validate_parameters(args(travel_mm=10.1))

    def test_rejects_stop_after_nominal_completion(self) -> None:
        with self.assertRaisesRegex(ValueError, "80%"):
            MODULE.validate_parameters(args(stop_after_ms=170.0))

    def test_bounded_target_uses_total_gripper_width(self) -> None:
        self.assertAlmostEqual(MODULE.bounded_target(0.04, 0.08, "open", 0.006), 0.046)
        self.assertAlmostEqual(MODULE.bounded_target(0.04, 0.08, "close", 0.006), 0.034)

    def test_bounded_target_rejects_physical_range_overrun(self) -> None:
        with self.assertRaisesRegex(ValueError, "Choose the other direction"):
            MODULE.bounded_target(0.078, 0.08, "open", 0.006)

    def test_checked_state_rejects_unhomed_hand(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "homing"):
            MODULE.checked_state(_StateGripper(max_width=0.0))

    def test_stop_trial_measures_bounded_interrupt(self) -> None:
        gripper = _StopGripper()
        trial = MODULE.run_stop_trial(
            gripper,
            direction="open",
            travel_m=0.006,
            speed_m_s=0.03,
            stop_after_s=0.0,
        )
        self.assertEqual(gripper.stop_count, 1)
        self.assertTrue(trial["pending_before_stop"])
        self.assertAlmostEqual(trial["target_width_m"], 0.046)
        self.assertAlmostEqual(trial["measured_travel_before_stop_return_m"], 0.002)

    def test_natural_trial_measures_finite_completion(self) -> None:
        trial = MODULE.run_natural_trial(
            _NaturalGripper(),
            target_m=0.046,
            speed_m_s=0.03,
            timeout_s=0.5,
        )
        self.assertAlmostEqual(trial["nominal_motion_ms"], 200.0)
        self.assertAlmostEqual(trial["final_error_m"], 0.0)

    def test_summary_keeps_stop_and_natural_metrics_separate(self) -> None:
        summary = MODULE.summarize(
            [
                {"kind": "stop", "stop_call_ms": 900.0},
                {"kind": "stop", "stop_call_ms": 1100.0},
                {"kind": "natural", "completion_wait_ms": 250.0},
            ]
        )
        self.assertEqual(summary["stop"]["median_ms"], 1000.0)
        self.assertEqual(summary["natural"]["maximum_ms"], 250.0)


if __name__ == "__main__":
    unittest.main()
