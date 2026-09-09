from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "robot" / "go_to_zero_pose.py"
SPEC = importlib.util.spec_from_file_location("go_to_zero_pose", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
go_to_zero_pose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = go_to_zero_pose
SPEC.loader.exec_module(go_to_zero_pose)


class GripperStateFailureTests(unittest.TestCase):
    def test_rejects_unhomed_zero_max_width(self) -> None:
        failure = go_to_zero_pose.gripper_state_failure(0.0, 0.0, 0.04)
        self.assertIsNotNone(failure)
        self.assertIn("not homed", failure)

    def test_rejects_target_outside_reported_range(self) -> None:
        failure = go_to_zero_pose.gripper_state_failure(0.02, 0.03, 0.04)
        self.assertIsNotNone(failure)
        self.assertIn("outside", failure)

    def test_accepts_homed_state_and_saved_target(self) -> None:
        self.assertIsNone(go_to_zero_pose.gripper_state_failure(0.0, 0.08, 0.04))

    def test_unhomed_gripper_stops_before_confirmation_or_arm_motion(self) -> None:
        robot = MagicMock()
        gripper = MagicMock()
        gripper.width = 0.0
        gripper.max_width = 0.0
        with (
            patch.object(go_to_zero_pose, "Robot", return_value=robot),
            patch.object(go_to_zero_pose, "Gripper", return_value=gripper),
            patch.object(sys, "argv", ["go_to_zero_pose.py", "--ip", "172.16.0.2"]),
            patch("builtins.input") as confirm,
        ):
            self.assertEqual(go_to_zero_pose.main(), 3)

        confirm.assert_not_called()
        robot.move.assert_not_called()
        gripper.move.assert_not_called()


class RecoveryMotionTests(unittest.TestCase):
    def test_default_uses_one_continuous_joint_motion(self) -> None:
        robot = MagicMock()
        robot.has_errors = False
        robot.current_joint_state.position = [0.0] * 7
        robot.state = SimpleNamespace(
            q=list(go_to_zero_pose.TARGET_JOINT_POSITION),
            dq=[0.0] * 7,
        )
        robot.current_pose.end_effector_pose.translation.tolist.return_value = [0.5, 0.0, 0.3]
        robot.current_pose.end_effector_pose.quaternion.tolist.return_value = [1.0, 0.0, 0.0, 0.0]
        motion = MagicMock()

        with (
            patch.object(go_to_zero_pose, "Robot", return_value=robot),
            patch.object(go_to_zero_pose, "JointMotion", return_value=motion) as joint_motion,
            patch.object(sys, "argv", ["go_to_zero_pose.py", "--skip-gripper"]),
            patch("builtins.input", return_value=""),
            patch.object(go_to_zero_pose.time, "sleep"),
        ):
            self.assertEqual(go_to_zero_pose.main(), 0)

        joint_motion.assert_called_once_with(
            go_to_zero_pose.TARGET_JOINT_POSITION,
            relative_dynamics_factor=0.03,
        )
        robot.move.assert_called_once_with(motion)

    def test_discontinuity_error_has_chinese_non_retry_hint(self) -> None:
        error = go_to_zero_pose.ControlException(
            "joint_motion_generator_acceleration_discontinuity"
        )

        hint = go_to_zero_pose.motion_failure_hint(error)

        self.assertIsNotNone(hint)
        self.assertIn("不要立即重试", hint)


if __name__ == "__main__":
    unittest.main()
