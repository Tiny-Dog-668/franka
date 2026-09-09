from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import franka_sim2real.e2e_bundle as e2e_bundle
import franka_sim2real.streaming as streaming
from franka_sim2real.e2e_bundle import (
    ActionHistoryBuffer,
    BundleCameraConfig,
    BundleInitialStateConfig,
    BundleTorchScriptPolicy,
    build_rma_contact_force_input,
    capture_gelsight_reference_frames,
    evaluate_initial_state,
    load_bundle_config,
    validate_bundle_artifacts,
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

    def test_flange_to_tcp_offset_must_match_the_simulation_contract(self) -> None:
        config = BundleInitialStateConfig(
            enforce=True,
            flange_to_tcp_translation_m=[0.0, 0.0, 0.1034],
            flange_to_tcp_translation_tolerance_m=0.001,
        )
        observation = make_observation()
        observation.metadata["flange_to_tcp_translation_m"] = [0.0, 0.0, 0.1034]
        self.assertTrue(evaluate_initial_state(observation, config)["passed"])

        observation.metadata["flange_to_tcp_translation_m"] = [0.0, 0.0, 0.0]
        report = evaluate_initial_state(observation, config)
        self.assertFalse(report["passed"])
        self.assertIn("flange-to-TCP translation error", report["failures"][-1])


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


class Exported0813GelSightReferenceConfigTests(unittest.TestCase):
    def test_0813_config_and_artifact_expose_fixed_reference_inputs(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0813_gelsight.json"
        )
        config.model.device = "cpu"
        report = validate_bundle_artifacts(config)

        self.assertTrue(config.tactile_camera.enabled)
        self.assertEqual(
            report["input_signature"]["gsmini_left_reference_rgb"], [96, 128, 3]
        )
        self.assertEqual(
            report["input_signature"]["gsmini_right_reference_rgb"], [96, 128, 3]
        )
        self.assertEqual(
            report["streaming_contract"]["tactile_reference"],
            "first_post_reset_frame_fixed_per_rollout",
        )

    def test_reference_capture_copies_one_fixed_tactile_pair(self) -> None:
        bundle = type("Bundle", (), {
            "has_gelsight_reference_inputs": True,
            "gelsight_input_shapes": {
                "gsmini_left_reference_rgb": (96, 128, 3),
                "gsmini_right_reference_rgb": (96, 128, 3),
            },
        })()
        left = np.full((96, 128, 3), 11, dtype=np.uint8)
        right = np.full((96, 128, 3), 22, dtype=np.uint8)
        camera = MagicMock()
        camera.read_tactile.return_value = left, right

        references = capture_gelsight_reference_frames(camera, bundle)
        left[...] = 0
        right[...] = 0

        camera.read_tactile.assert_called_once_with()
        self.assertEqual(int(references["gsmini_left_reference_rgb"][0, 0, 0]), 11)
        self.assertEqual(int(references["gsmini_right_reference_rgb"][0, 0, 0]), 22)
        self.assertIs(bundle._deployment_tactile_references, references)

    def test_streaming_tick_reuses_the_captured_reference_pair(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0813_gelsight.json"
        )
        config.model.device = "cpu"
        bundle = BundleTorchScriptPolicy(
            config.model.model_path, config.model.metadata_path, device="cpu"
        )
        history = ActionHistoryBuffer(
            4, "processed_action", [1.0] * 4, 1, config.action_adapter.scales
        )

        class StaticCamera:
            def __init__(self) -> None:
                self.left = np.full((96, 128, 3), 17, dtype=np.uint8)
                self.right = np.full((96, 128, 3), 23, dtype=np.uint8)

            def read(self) -> np.ndarray:
                return np.zeros((224, 224, 3), dtype=np.uint8)

            def read_tactile(self) -> tuple[np.ndarray, np.ndarray]:
                return self.left.copy(), self.right.copy()

        camera = StaticCamera()
        capture_gelsight_reference_frames(camera, bundle)
        camera.left[...] = 51
        camera.right[...] = 63
        result = streaming._run_policy_tick(
            bundle, camera, history, make_observation(), config, False, 0.04,
            time.monotonic_ns,
        )

        self.assertEqual(int(result.tactile_images["gsmini_left_rgb"][0, 0, 0]), 51)
        self.assertEqual(
            int(result.tactile_references["gsmini_left_reference_rgb"][0, 0, 0]), 17
        )
        self.assertEqual(
            int(result.tactile_references["gsmini_right_reference_rgb"][0, 0, 0]), 23
        )


class Exported0814GelSightReferenceConfigTests(unittest.TestCase):
    def test_0814_v2_config_records_but_does_not_cap_the_training_horizon(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0814_gelsight.json"
        )
        config.model.device = "cpu"

        self.assertEqual(config.runner.steps, 150)
        self.assertEqual(config.tool_tcp_offset_ee_m, [0.0, 0.0, 0.027408])
        self.assertEqual(config.workspace["minimum"][2], 0.01)
        self.assertEqual(
            config.initial_state.flange_to_tcp_translation_m, [0.0, 0.0, 0.1034]
        )
        # The 150-step training horizon remains provenance, not a deployment
        # stop condition for a continuing real-robot policy stream.
        config.runner.steps = 1000
        report = validate_bundle_artifacts(config)
        self.assertEqual(
            report["streaming_contract"]["max_episode_length_steps"], 150
        )


class Exported0823GelSightThreeFrameConfigTests(unittest.TestCase):
    def test_0823_config_matches_corrected_geometry_and_runtime_contract(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0823_gelsight.json"
        )
        self.assertEqual(config.tool_tcp_offset_ee_m, [0.0, 0.0, 0.0579])
        self.assertEqual(config.workspace["minimum"][2], 0.01)
        self.assertEqual(config.runner.steps, 150)
        self.assertEqual(config.runner.log_dir, "real_policy_logs")
        self.assertFalse(config.camera.save_rgb)
        self.assertFalse(config.tactile_camera.save_rgb)
        self.assertEqual(config.model.history_source, "processed_action")
        self.assertEqual(config.model.history_scale, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(config.model.history_delay_steps, 1)
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertIn("/checkpoint/0823/", config.model.model_path)

    def test_0823_artifact_validates_with_three_frame_and_reference_inputs(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0823_gelsight.json"
        )
        config.model.device = "cpu"
        report = validate_bundle_artifacts(config)

        self.assertEqual(report["rgb_history"], {
            "frames": 3,
            "order": "oldest_to_newest",
        })
        self.assertEqual(
            report["input_signature"]["gsmini_left_reference_rgb"],
            [96, 128, 3],
        )
        self.assertEqual(
            report["streaming_contract"][
                "rma_gelsight_x040_three_frame_student_metadata_version"
            ],
            1,
        )

    def test_three_frame_policy_rejects_the_old_short_workspace_offset(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0823_gelsight.json"
        )
        config.model.device = "cpu"
        config.tool_tcp_offset_ee_m = [0.0, 0.0, 0.027408]
        with self.assertRaisesRegex(ValueError, "corrected 161.3 mm"):
            validate_bundle_artifacts(config)


class Exported0809XYConfigTests(unittest.TestCase):
    def test_0809_config_declares_gripper_contact_approximation(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_xy_only.json"
        )
        self.assertEqual(config.model.rma_contact_force_source, "gripper_is_grasped")
        self.assertEqual(config.model.history_source, "processed_action")
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertIn("/checkpoint/0809_xy_only/", config.model.model_path)

    def test_gripper_contact_approximation_maps_only_the_binary_actor_feature(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_xy_only.json"
        )
        bundle = BundleTorchScriptPolicy(
            config.model.model_path,
            config.model.metadata_path,
            device="cpu",
        )
        open_observation = make_observation()
        closed_observation = make_observation()
        closed_observation.gripper_is_grasped = True
        np.testing.assert_array_equal(
            build_rma_contact_force_input(open_observation, bundle, config),
            np.asarray([0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            build_rma_contact_force_input(closed_observation, bundle, config),
            np.asarray([1.0, 1.0], dtype=np.float32),
        )

    def test_0809_model_contract_validates_on_cpu(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_xy_only.json"
        )
        config.model.device = "cpu"
        report = validate_bundle_artifacts(config)
        self.assertEqual(report["input_signature"]["contact_force_n"], [2])
        self.assertEqual(report["rma_contact_force_source"], "gripper_is_grasped")

    def test_0809_streaming_policy_tick_passes_contact_by_keyword(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_xy_only.json"
        )
        config.model.device = "cpu"
        bundle = BundleTorchScriptPolicy(
            config.model.model_path,
            config.model.metadata_path,
            device="cpu",
        )
        history = ActionHistoryBuffer(
            history_dim=4,
            source="processed_action",
            scale=[1.0] * 4,
            delay_steps=1,
            processed_action_scale=config.action_adapter.scales,
        )

        class StaticCamera:
            def read(self) -> np.ndarray:
                return np.zeros((224, 224, 3), dtype=np.uint8)

        result = streaming._run_policy_tick(
            bundle,
            StaticCamera(),
            history,
            make_observation(),
            config,
            allow_full_scale=False,
            desired_gripper_width=0.04,
            clock_ns=time.monotonic_ns,
        )
        np.testing.assert_array_equal(
            result.contact_force_n,
            np.asarray([0.0, 0.0], dtype=np.float32),
        )


class Exported0809DirectActionConfigTests(unittest.TestCase):
    def test_0809_direct_action_config_has_no_contact_force_approximation(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_direct_action.json"
        )
        self.assertEqual(config.model.device, "cuda:0")
        self.assertEqual(config.model.history_source, "processed_action")
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertIn("/checkpoint/0809_direct_action/", config.model.model_path)

    def test_0809_direct_action_model_contract_validates_on_cpu(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_direct_action.json"
        )
        config.model.device = "cpu"
        report = validate_bundle_artifacts(config)
        self.assertNotIn("contact_force_n", report["input_signature"])
        self.assertIsNone(report["rma_contact_force_source"])
        self.assertEqual(
            report["streaming_contract"]["rma_direct_action_student_metadata_version"], 1
        )

    def test_0809_direct_action_streaming_policy_tick_omits_contact_input(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_direct_action.json"
        )
        bundle = BundleTorchScriptPolicy(
            config.model.model_path,
            config.model.metadata_path,
            device="cpu",
        )
        history = ActionHistoryBuffer(
            history_dim=4,
            source="processed_action",
            scale=[1.0] * 4,
            delay_steps=1,
            processed_action_scale=config.action_adapter.scales,
        )

        class StaticCamera:
            def read(self) -> np.ndarray:
                return np.zeros((224, 224, 3), dtype=np.uint8)

        result = streaming._run_policy_tick(
            bundle,
            StaticCamera(),
            history,
            make_observation(),
            config,
            allow_full_scale=False,
            desired_gripper_width=0.04,
            clock_ns=time.monotonic_ns,
        )
        self.assertIsNone(result.contact_force_n)
        self.assertEqual(result.raw_action.shape, (4,))
        self.assertTrue(np.isfinite(result.raw_action).all())


class Exported0809X040WideDirectActionConfigTests(unittest.TestCase):
    def test_0809_x040_wide_direct_action_model_contract_validates_on_cpu(self) -> None:
        config = load_bundle_config(
            REPO_ROOT / "configs/e2e_bundle_real_exported_0809_x040_wide_direct_action.json"
        )
        config.model.device = "cpu"
        report = validate_bundle_artifacts(config)
        self.assertNotIn("contact_force_n", report["input_signature"])
        self.assertIsNone(report["rma_contact_force_source"])
        self.assertEqual(
            report["streaming_contract"][
                "rma_x040_wide_direct_action_student_metadata_version"
            ],
            1,
        )


class RealSenseColorControlTests(unittest.TestCase):
    def test_make_camera_forwards_configured_color_controls(self) -> None:
        config = BundleCameraConfig(
            auto_exposure=False,
            exposure=100.0,
            gain=64.0,
        )
        fake_color_camera = MagicMock()
        fake_color_camera.get_color_controls.return_value = {
            "auto_exposure": False,
            "exposure": 100.0,
            "gain": 64.0,
        }

        with (
            patch.object(
                e2e_bundle,
                "RealSenseRGBCamera",
                return_value=fake_color_camera,
            ) as camera_class,
            patch.object(
                e2e_bundle,
                "LatestFrameCamera",
                side_effect=lambda camera: camera,
            ),
        ):
            camera = e2e_bundle._make_camera(config, 224, 224)

        self.assertIs(camera, fake_color_camera)
        camera_class.assert_called_once_with(
            config,
            output_width=224,
            output_height=224,
            auto_exposure=False,
            exposure=100.0,
            gain=64.0,
        )


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
