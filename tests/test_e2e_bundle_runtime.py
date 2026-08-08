from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import franka_sim2real.e2e_bundle as e2e_bundle
from franka_sim2real.e2e_bundle import (
    ActionHistoryBuffer,
    BundleInitialStateConfig,
    evaluate_initial_state,
    load_bundle_config,
)
from franka_sim2real.types import RobotObservation


REPO_ROOT = Path(__file__).resolve().parents[1]


def make_observation(
    *,
    joint_positions: list[float] | None = None,
    joint_velocities: list[float] | None = None,
    tcp_translation: list[float] | None = None,
    tcp_quaternion: list[float] | None = None,
    gripper_width: float | None = 0.04,
    has_errors: bool = False,
) -> RobotObservation:
    return RobotObservation(
        joint_positions=joint_positions or [0.0] * 7,
        joint_velocities=joint_velocities or [0.0] * 7,
        tcp_translation=tcp_translation or [0.5, 0.0, 0.3],
        tcp_quaternion=tcp_quaternion or [1.0, 0.0, 0.0, 0.0],
        external_wrench=[0.0] * 6,
        robot_mode="Idle",
        has_errors=has_errors,
        is_in_control=False,
        control_command_success_rate=0.0,
        gripper_width=gripper_width,
        gripper_max_width=0.08,
        gripper_is_grasped=False,
        metadata={"current_errors": [], "last_motion_errors": []},
    )


class ActionHistoryBufferTests(unittest.TestCase):
    def test_rma_processed_action_history_keeps_physical_units(self) -> None:
        history = ActionHistoryBuffer(
            history_dim=4,
            source="processed_action",
            scale=[1.0, 1.0, 1.0, 1.0],
            delay_steps=1,
            processed_action_scale=[0.05, 0.05, 0.05, 0.01],
        )
        executed = np.asarray([0.1, -0.1, 0.05, -0.1], dtype=np.float32)
        history.update(np.ones(4, dtype=np.float32), executed)
        np.testing.assert_allclose(
            history.current(),
            np.asarray([0.005, -0.005, 0.0025, -0.001], dtype=np.float32),
            rtol=0.0,
            atol=1e-8,
        )

    def test_0712_two_step_scaled_clipped_history(self) -> None:
        history = ActionHistoryBuffer(
            history_dim=4,
            source="clipped_action",
            scale=0.05,
            delay_steps=2,
        )
        clipped_0 = np.asarray([1.0, -1.0, 0.5, -0.25], dtype=np.float32)
        clipped_1 = np.asarray([-0.4, 0.6, -0.8, 1.0], dtype=np.float32)

        np.testing.assert_array_equal(history.current(), np.zeros(4, dtype=np.float32))
        history.update(clipped_0 * 9.0, clipped_0)
        np.testing.assert_array_equal(history.current(), np.zeros(4, dtype=np.float32))
        history.update(clipped_1 * 9.0, clipped_1)
        np.testing.assert_allclose(
            history.current(),
            np.asarray([0.05, -0.05, 0.025, -0.0125], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(history.current().dtype, np.float32)

    def test_default_mode_preserves_previous_behavior(self) -> None:
        history = ActionHistoryBuffer(4, "clipped_action")
        clipped = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
        history.update(clipped * 10.0, clipped)
        np.testing.assert_array_equal(history.current(), clipped)

    def test_per_dimension_history_scale(self) -> None:
        history = ActionHistoryBuffer(
            4,
            "clipped_action",
            scale=[0.05, 0.05, 0.05, 0.01],
        )
        clipped = np.asarray([1.0, -1.0, 0.5, -0.25], dtype=np.float32)
        history.update(clipped * 9.0, clipped)
        np.testing.assert_allclose(
            history.current(),
            np.asarray([0.05, -0.05, 0.025, -0.0025], dtype=np.float32),
            rtol=0.0,
            atol=1e-7,
        )

    def test_zeros_mode_stays_zero(self) -> None:
        history = ActionHistoryBuffer(4, "zeros", delay_steps=2)
        history.update(np.ones(4), np.ones(4))
        history.update(np.ones(4), np.ones(4))
        np.testing.assert_array_equal(history.current(), np.zeros(4, dtype=np.float32))

    def test_invalid_history_configuration_is_rejected(self) -> None:
        for scale in (0.0, -0.05, math.nan, math.inf, True, "0.05"):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ActionHistoryBuffer(4, "clipped_action", scale=scale)
        for scale in ([0.05] * 3, [0.05, 0.05, 0.05, 0.0]):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                ActionHistoryBuffer(4, "clipped_action", scale=scale)
        for delay in (0, -1, 1.5, True):
            with self.subTest(delay=delay), self.assertRaises(ValueError):
                ActionHistoryBuffer(4, "clipped_action", delay_steps=delay)
        with self.assertRaises(ValueError):
            ActionHistoryBuffer(4, "unknown")

    def test_history_dimension_is_checked(self) -> None:
        history = ActionHistoryBuffer(4, "clipped_action")
        with self.assertRaisesRegex(ValueError, "Expected history action with 4 values, got 3"):
            history.update(np.ones(4), np.ones(3))


class InitialStateTests(unittest.TestCase):
    def test_matching_state_passes(self) -> None:
        config = BundleInitialStateConfig(
            enforce=True,
            joint_positions=[0.0] * 7,
            joint_position_tolerance_rad=0.01,
            max_abs_joint_velocity_rad_s=0.02,
            tcp_translation=[0.5, 0.0, 0.3],
            tcp_translation_tolerance_m=0.005,
            tcp_quaternion_xyzw=[1.0, 0.0, 0.0, 0.0],
            tcp_orientation_tolerance_deg=1.0,
            gripper_width_m=0.04,
            gripper_width_tolerance_m=0.003,
            minimum_gripper_max_width_m=0.075,
            required_robot_mode="Idle",
            require_gripper_not_grasped=True,
        )
        report = evaluate_initial_state(make_observation(), config)
        self.assertTrue(report["passed"])
        self.assertEqual(report["failures"], [])

    def test_quaternion_sign_is_equivalent(self) -> None:
        config = BundleInitialStateConfig(
            enforce=True,
            tcp_quaternion_xyzw=[-1.0, 0.0, 0.0, 0.0],
            tcp_orientation_tolerance_deg=0.001,
        )
        report = evaluate_initial_state(make_observation(), config)
        self.assertTrue(report["checks"]["tcp_orientation"]["passed"])

    def test_wrong_gripper_width_fails_closed(self) -> None:
        config = BundleInitialStateConfig(
            enforce=True,
            gripper_width_m=0.04,
            gripper_width_tolerance_m=0.003,
        )
        report = evaluate_initial_state(make_observation(gripper_width=0.08), config)
        self.assertFalse(report["passed"])
        self.assertIn("gripper width error", report["failures"][0])

    def test_active_robot_error_fails(self) -> None:
        config = BundleInitialStateConfig(enforce=True, require_no_robot_errors=True)
        observation = make_observation(has_errors=True)
        observation.metadata["current_errors"] = ["joint_position_limits_violation"]
        report = evaluate_initial_state(observation, config)
        self.assertFalse(report["checks"]["robot_errors"]["passed"])


class Exported0712ConfigTests(unittest.TestCase):
    def test_0712_config_uses_safe_scales_and_compatible_history(self) -> None:
        config = load_bundle_config(REPO_ROOT / "configs/e2e_bundle_real_exported_0712.json")
        self.assertEqual(config.action_adapter.scales[:3], [0.005, 0.005, 0.005])
        self.assertEqual(config.action_adapter.scales[3], 0.002)
        self.assertEqual(config.model.history_source, "clipped_action")
        self.assertEqual(config.model.history_scale, 0.05)
        self.assertEqual(config.model.history_delay_steps, 2)
        self.assertTrue(config.initial_state.enforce)
        self.assertAlmostEqual(config.initial_state.gripper_width_m or 0.0, 0.040001507848501205)
        self.assertIn("/checkpoint/0712/exported/", config.model.model_path)

    def test_legacy_config_keeps_original_history_behavior(self) -> None:
        config = load_bundle_config(REPO_ROOT / "configs/e2e_bundle_real_exported_0711.json")
        self.assertEqual(config.model.history_scale, 1.0)
        self.assertEqual(config.model.history_delay_steps, 1)
        self.assertFalse(config.initial_state.enforce)


class Exported0726ConfigTests(unittest.TestCase):
    def test_0726_config_matches_exported_policy_contract(self) -> None:
        config = load_bundle_config(REPO_ROOT / "configs/e2e_bundle_real_exported_0726.json")
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertEqual(config.model.history_source, "clipped_action")
        self.assertEqual(config.model.history_scale, [0.05, 0.05, 0.05, 0.01])
        self.assertEqual(config.model.history_delay_steps, 1)
        self.assertTrue(config.model.enforce_policy_contract)
        self.assertEqual(config.camera.serial, "215322076207")
        self.assertEqual(
            (
                config.camera.crop_left,
                config.camera.crop_top,
                config.camera.crop_width,
                config.camera.crop_height,
            ),
            (80, 0, 480, 480),
        )
        self.assertEqual(config.runner.steps, 1)
        self.assertIn("/checkpoint/0726/exported/", config.model.model_path)


class Exported0808GelSightConfigTests(unittest.TestCase):
    def test_0808_config_enables_the_two_tactile_inputs(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0808_gelsight.json"
        )
        self.assertTrue(config.tactile_camera.enabled)
        self.assertEqual(config.tactile_camera.left_device, 0)
        self.assertEqual(config.tactile_camera.right_device, 6)
        self.assertEqual(
            (config.tactile_camera.width, config.tactile_camera.height),
            (3280, 2464),
        )
        self.assertEqual(config.model.history_source, "processed_action")
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertIn("/checkpoint/0808/", config.model.model_path)


class InitialStateRolloutGateTests(unittest.TestCase):
    def test_failed_initial_state_stops_before_camera_and_policy(self) -> None:
        config = load_bundle_config(REPO_ROOT / "configs/e2e_bundle_real_exported_0712.json")
        observation = make_observation(
            joint_positions=list(config.initial_state.joint_positions or []),
            tcp_translation=list(config.initial_state.tcp_translation or []),
            tcp_quaternion=list(config.initial_state.tcp_quaternion_xyzw or []),
            gripper_width=0.08,
        )
        fake_policy = MagicMock(history_dim=4, action_dim=4, proprio_dim=15)
        fake_policy.rgb_height = 224
        fake_policy.rgb_width = 224
        fake_env = MagicMock()
        fake_env.reset.return_value = observation

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "rollout"
            run_dir.mkdir()
            with (
                patch.object(e2e_bundle, "_make_run_dir", return_value=run_dir),
                patch.object(
                    e2e_bundle,
                    "BundleTorchScriptPolicy",
                    return_value=fake_policy,
                ),
                patch.object(e2e_bundle, "RealFrankaEnv", return_value=fake_env),
                patch.object(e2e_bundle, "_make_camera") as make_camera,
            ):
                with self.assertRaisesRegex(RuntimeError, "Initial-state safety check failed"):
                    e2e_bundle.run_bundle_deploy(config)

            fake_env.reset.assert_called_once_with()
            fake_env.step.assert_not_called()
            fake_env.close.assert_called_once_with()
            make_camera.assert_not_called()
            fake_policy.predict.assert_not_called()

            report = json.loads((run_dir / "initial_state_check.json").read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["gripper_width"]["passed"])


if __name__ == "__main__":
    unittest.main()
