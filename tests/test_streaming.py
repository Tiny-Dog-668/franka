from __future__ import annotations

import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from franka_sim2real.e2e_bundle import ActionHistoryBuffer, BundleDeployConfig, load_bundle_config
import franka_sim2real.streaming as streaming
from franka_sim2real.types import RobotObservation


def streaming_config() -> BundleDeployConfig:
    config = BundleDeployConfig()
    config.control_mode = "streaming"
    config.realtime = "enforce"
    config.model.enforce_policy_contract = True
    config.model.history_source = "clipped_action"
    config.model.history_scale = [0.025, 0.025, 0.025, 0.005]
    config.action_adapter.labels = ["dx", "dy", "dz", "gripper"]
    config.action_adapter.scales = [0.025, 0.025, 0.025, 0.005]
    config.action_adapter.clip_low = [-1.0] * 4
    config.action_adapter.clip_high = [1.0] * 4
    config.action_adapter.gripper_mode = "delta_width"
    config.runner.steps = 2
    config.camera.save_rgb = False
    config.initial_state.enforce = False
    return config


def policy_contract() -> dict[str, object]:
    return {
        "action_dim": 4,
        "action_history": "per_dimension_processed_action_no_privileged_gate_v3",
        "xyz_command_frame": "robot_root",
        "privileged_dz_gate": "disabled",
        "gripper_control_mode": "total_width_delta_cached_target",
        "gripper_width_bounds": [0.0, 0.08],
        "gripper_width_delta_updates_per_policy_step": 1,
        "sim_dt": 1.0 / 60.0,
        "decimation": 2,
        "nominal_policy_frequency_hz": 30.0,
        "nominal_camera_frequency_hz": 30.0,
        "max_episode_length_steps": 150,
    }


class NumericStreamingTests(unittest.TestCase):
    def test_column_major_jacobian_and_dls_formula(self) -> None:
        jacobian = np.arange(42, dtype=np.float64).reshape(6, 7) / 50.0
        flat_column_major = jacobian.reshape(-1, order="F")
        error = np.asarray([0.01, -0.02, 0.03, 0.001, -0.002, 0.003])
        expected = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.01**2 * np.eye(6), error
        )
        np.testing.assert_allclose(
            streaming.dls_joint_delta(flat_column_major, error, 0.01),
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_target_is_absolute_and_orientation_is_latched(self) -> None:
        pose = np.eye(4)
        pose[:3, 3] = [0.5, 0.0, 0.3]
        pose[:3, :3] = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = streaming.latch_tcp_target(pose, [0.01, -0.02, 0.03])
        np.testing.assert_allclose(target[:3, 3], [0.51, -0.02, 0.33])
        np.testing.assert_array_equal(target[:3, :3], pose[:3, :3])
        # Reusing this target on two IK ticks does not accumulate the action.
        np.testing.assert_allclose(
            streaming.pose_error(pose, target),
            [0.01, -0.02, 0.03, 0.0, 0.0, 0.0],
            atol=1e-12,
        )

    def test_physical_tool_tcp_rotates_the_gelsight_surface_offset(self) -> None:
        config = streaming_config()
        config.tool_tcp_offset_ee_m = [0.0, 0.0, 0.027408]
        pose = np.eye(4)
        # The deployed reference orientation points the EE local +Z toward
        # the table, so the physical GelSight surface is below O_T_EE.
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = [0.4375, -0.0152, 0.0422]
        np.testing.assert_allclose(
            streaming.physical_tool_tcp_translation(pose, config),
            [0.4375, -0.0152, 0.014792],
            rtol=0.0,
            atol=1e-9,
        )

    def test_commissioning_and_full_scale_history_match_executed_action(self) -> None:
        config = streaming_config()
        raw = np.asarray([0.8, -0.4, 0.2, -1.0], dtype=np.float32)
        commissioning = streaming.clip_streaming_action(raw, config, False)
        full_scale = streaming.clip_streaming_action(raw, config, True)
        # The commissioning limit scales XYZ uniformly, so the peak component
        # lands on the limit and the commanded direction is preserved.
        np.testing.assert_allclose(commissioning, [0.1, -0.05, 0.025, -0.1])
        np.testing.assert_array_equal(full_scale, raw)

        history = ActionHistoryBuffer(4, "clipped_action", config.model.history_scale)
        history.update(raw, commissioning)
        np.testing.assert_allclose(history.current(), commissioning * config.model.history_scale)

    def test_commissioning_limit_preserves_cartesian_direction(self) -> None:
        config = streaming_config()
        raw = np.asarray([-0.44, 0.85, -0.997, 0.0], dtype=np.float32)
        limited = streaming.clip_streaming_action(raw, config, False)
        self.assertAlmostEqual(float(np.max(np.abs(limited[:3]))), 0.1, places=6)
        unit_raw = raw[:3] / np.linalg.norm(raw[:3])
        unit_limited = limited[:3] / np.linalg.norm(limited[:3])
        np.testing.assert_allclose(unit_limited, unit_raw, atol=1e-6)

    def test_commissioning_limit_leaves_small_actions_untouched(self) -> None:
        config = streaming_config()
        raw = np.asarray([0.05, -0.02, 0.01, 0.03], dtype=np.float32)
        np.testing.assert_allclose(
            streaming.clip_streaming_action(raw, config, False), raw
        )

    def test_streaming_xyz_is_base_frame_without_blocking_sign_flip(self) -> None:
        config = streaming_config()
        action = streaming.streaming_robot_action(
            np.asarray([0.1, 0.1, 0.1, 0.0], dtype=np.float32), config, 0.04
        )
        self.assertGreater(action.dy, 0.0)
        self.assertGreater(action.dz, 0.0)
        self.assertEqual(action.metadata["xyz_command_frame"], "robot_root")


class ContractAndSchedulingTests(unittest.TestCase):
    def test_0802_dr_config_uses_rma_physical_history_contract(self) -> None:
        config = load_bundle_config(
            Path(__file__).resolve().parents[1]
            / "configs/e2e_bundle_real_exported_0802_dr.json"
        )
        self.assertEqual(config.action_adapter.scales, [0.05, 0.05, 0.05, 0.01])
        self.assertEqual(config.model.history_source, "processed_action")
        self.assertEqual(config.model.history_scale, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(config.model.history_delay_steps, 1)
        self.assertIsNotNone(config.streaming.joint_impedance)
        self.assertEqual(len(config.streaming.joint_impedance or []), 7)
        self.assertTrue(all(value > 0.0 for value in config.streaming.joint_impedance or []))
        self.assertTrue(config.model.enforce_policy_contract)

    def test_0802_dr_smooth_config_has_explicit_trajectory_limits(self) -> None:
        config = load_bundle_config(
            Path(__file__).resolve().parents[1]
            / "configs/e2e_bundle_real_exported_0802_dr_smooth.json"
        )
        self.assertEqual(
            config.streaming.maximum_joint_accelerations,
            [2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
        )
        self.assertEqual(
            config.streaming.maximum_joint_jerks,
            [100.0, 100.0, 100.0, 100.0, 150.0, 150.0, 150.0],
        )

    def test_simactuator_config_never_saturates_the_reference_limit(self) -> None:
        """A binding limit rotates the Cartesian direction, it does not just slow it down.

        Clamping the DLS delta per joint changes the joint-space direction, which
        maps to an entirely different Cartesian direction. Measured on the 0802
        reference pose, the 0.005 rad per-tick clamp rotates a
        ``[-0.577, 0.577, -0.577]`` command to ``[-0.104, 0.991, 0.088]``: 54.5
        degrees off, with the Z sign flipped. So the reference limit has to stay
        wide enough that it never binds at the trained action scale.
        """
        config = load_bundle_config(
            Path(__file__).resolve().parents[1]
            / "configs/e2e_bundle_real_exported_0802_dr_simactuator.json"
        )
        self.assertEqual(config.streaming.control_law, "sim_actuator_velocity")
        # FRANKA_PANDA_HIGH_PD_CFG is stiffness 400 over damping 80.
        self.assertEqual(config.streaming.reference_velocity_gain, 5.0)

        smallest_singular_value_m_per_rad = 0.2637
        largest_cartesian_scale_m = max(config.action_adapter.scales[:3])
        worst_case_delta_rad = largest_cartesian_scale_m / smallest_singular_value_m_per_rad
        self.assertGreater(
            config.streaming.maximum_ik_reference_delta_rad, worst_case_delta_rad
        )

        required_rad_s = config.streaming.reference_velocity_gain * worst_case_delta_rad
        self.assertTrue(
            all(limit >= required_rad_s for limit in config.streaming.maximum_joint_velocities)
        )

        official_maximum_velocities = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
        official_maximum_accelerations = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
        official_maximum_jerks = [7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0]
        for configured, official in (
            (config.streaming.maximum_joint_velocities, official_maximum_velocities),
            (config.streaming.maximum_joint_accelerations, official_maximum_accelerations),
            (config.streaming.maximum_joint_jerks, official_maximum_jerks),
        ):
            for joint, (value, limit) in enumerate(zip(configured, official)):
                self.assertLessEqual(value, limit, msg=f"joint {joint + 1} exceeds its Panda limit")

    def test_sim_actuator_velocity_requires_a_usable_velocity_envelope(self) -> None:
        config = streaming_config()
        config.streaming.control_law = "sim_actuator_velocity"
        config.streaming.reference_velocity_gain = 5.0
        config.action_adapter.scales = [0.05, 0.05, 0.05, 0.01]
        config.streaming.maximum_joint_velocities = [0.25] * 7
        bundle = types.SimpleNamespace(
            action_dim=4,
            history_dim=4,
            metadata={"policy_contract": policy_contract()},
        )
        with self.assertRaisesRegex(ValueError, "maximum_joint_velocities"):
            streaming.validate_streaming_contract(bundle, config)

        config.streaming.maximum_joint_velocities = [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5]
        streaming.validate_streaming_contract(bundle, config)

    def test_0802_dr_smooth_pins_the_inference_thread_count(self) -> None:
        # A 30 Hz policy has a 33.3 ms budget. This ResNet18 student measures
        # 26 ms mean / 55 ms p95 at torch's default thread count on this host
        # and 18 ms / 22 ms at eight threads, so the count cannot be inherited.
        config = load_bundle_config(
            Path(__file__).resolve().parents[1]
            / "configs/e2e_bundle_real_exported_0802_dr_smooth.json"
        )
        self.assertEqual(config.model.torch_num_threads, 8)

    def test_0802_dr_simactuator_optimizes_torchscript_for_inference(self) -> None:
        config = load_bundle_config(
            Path(__file__).resolve().parents[1]
            / "configs/e2e_bundle_real_exported_0802_dr_simactuator.json"
        )
        self.assertTrue(config.model.optimize_for_inference)

    def test_0801_defaults_to_enforced_streaming(self) -> None:
        config = load_bundle_config(
            Path(__file__).resolve().parents[1] / "configs/e2e_bundle_real_exported_0801.json"
        )
        self.assertEqual(config.control_mode, "streaming")
        self.assertEqual(config.realtime, "enforce")
        self.assertEqual(config.runner.steps, 1)
        self.assertEqual(config.streaming.policy_frequency_hz, 30.0)
        self.assertEqual(config.streaming.ik_frequency_hz, 60.0)
        self.assertEqual(config.streaming.backend, "server9_joint_position")
        self.assertEqual(config.streaming.server9_control_cpu, 21)

    def test_strict_streaming_contract(self) -> None:
        bundle = types.SimpleNamespace(
            action_dim=4,
            history_dim=4,
            metadata={"policy_contract": policy_contract()},
        )
        report = streaming.validate_streaming_contract(bundle, streaming_config())
        self.assertEqual(report["ticks_per_action"], 2)
        bad = streaming_config()
        bad.realtime = "ignore"
        with self.assertRaisesRegex(ValueError, "realtime='enforce'"):
            streaming.validate_streaming_contract(bundle, bad)

    def test_cartesian_impedance_requires_six_axis_stiffness_without_joint_stiffness(self) -> None:
        bundle = types.SimpleNamespace(
            action_dim=4,
            history_dim=4,
            metadata={"policy_contract": policy_contract()},
        )
        config = streaming_config()
        config.streaming.impedance_mode = "cartesian"
        config.streaming.joint_impedance = None
        config.streaming.cartesian_impedance = [1000.0, 1000.0, 500.0, 30.0, 30.0, 30.0]
        streaming.validate_streaming_contract(bundle, config)

        config.streaming.cartesian_impedance = None
        with self.assertRaisesRegex(ValueError, "cartesian_impedance is required"):
            streaming.validate_streaming_contract(bundle, config)

    def test_rma_streaming_contract_allows_longer_operator_selected_run(self) -> None:
        bundle = types.SimpleNamespace(
            action_dim=4,
            history_dim=4,
            is_tacex_rma_student=True,
        )
        config = streaming_config()
        config.runner.steps = 250
        with patch.object(streaming, "_validate_tacex_rma_student_contract"):
            report = streaming.validate_streaming_contract(bundle, config)
        self.assertTrue(report["rma_student_v5"])

    def test_absolute_deadlines_do_not_accumulate_drift(self) -> None:
        class FakeClock:
            now = 1_000_000

            def __call__(self) -> int:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.now += round(seconds * 1e9)

        clock = FakeClock()
        start = clock()
        period = 16_666_667
        for tick in range(300):
            lateness = streaming._sleep_until(start + tick * period, clock, clock.sleep)
            self.assertEqual(lateness, 0)
        self.assertEqual(clock(), start + 299 * period)

    def test_cleanup_stops_arm_and_gripper(self) -> None:
        arm = types.SimpleNamespace(stopped=False)
        arm.stop_control = lambda: setattr(arm, "stopped", True)
        gripper = types.SimpleNamespace(stopped=False)
        gripper.stop = lambda: setattr(gripper, "stopped", True)
        self.assertEqual(streaming._safe_stop(arm, gripper), [])
        self.assertTrue(arm.stopped)
        self.assertTrue(gripper.stopped)

    def test_gripper_queue_does_not_send_stop_before_control_session(self) -> None:
        state = types.SimpleNamespace(width=0.04, max_width=0.08, is_grasped=False)
        gripper = types.SimpleNamespace(state=state, stop_calls=0)
        gripper.stop = lambda: setattr(gripper, "stop_calls", gripper.stop_calls + 1)
        queue = streaming.AsyncGripperQueue(gripper, speed=0.03, tolerance=1e-4)
        queue.stop()
        self.assertEqual(gripper.stop_calls, 0)
        active_queue = streaming.AsyncGripperQueue(
            gripper, speed=0.03, tolerance=1e-4
        )
        active_queue.mark_control_session_active()
        active_queue.stop()
        self.assertEqual(gripper.stop_calls, 1)

    def test_gripper_queue_converts_blocked_close_to_grasp(self) -> None:
        class FakeFuture:
            def __init__(self, result: bool) -> None:
                self.result = result
                self.ready = threading.Event()

            def wait(self, timeout: float) -> bool:
                return self.ready.wait(timeout)

            def get(self) -> bool:
                return self.result

        class FakeGripper:
            def __init__(self) -> None:
                self.state = types.SimpleNamespace(
                    width=0.04, max_width=0.0798, is_grasped=False
                )
                self.move_future = FakeFuture(False)
                self.grasp_future = FakeFuture(True)
                self.move_calls: list[tuple[float, float]] = []
                self.grasp_calls: list[tuple[float, float, float]] = []
                self.move_started = threading.Event()
                self.grasp_started = threading.Event()

            def move_async(self, width: float, speed: float) -> FakeFuture:
                self.move_calls.append((width, speed))
                self.move_started.set()
                return self.move_future

            def grasp_async(
                self, width: float, speed: float, force: float
            ) -> FakeFuture:
                self.grasp_calls.append((width, speed, force))
                self.grasp_started.set()
                return self.grasp_future

            def stop(self) -> None:
                return None

        gripper = FakeGripper()
        queue = streaming.AsyncGripperQueue(
            gripper, speed=0.03, tolerance=1e-4, force=20.0
        )
        queue.command(0.0)
        queue.command(0.0)
        self.assertTrue(gripper.move_started.wait(1.0))
        self.assertEqual(gripper.move_calls, [(0.0, 0.03)])

        gripper.state = types.SimpleNamespace(
            width=0.0516, max_width=0.0798, is_grasped=False
        )
        gripper.move_future.ready.set()
        self.assertTrue(gripper.grasp_started.wait(1.0))
        self.assertEqual(gripper.grasp_calls, [(0.0516, 0.03, 20.0)])

        gripper.state = types.SimpleNamespace(
            width=0.0515, max_width=0.0798, is_grasped=True
        )
        gripper.grasp_future.ready.set()
        queue.wait_idle()
        queue.command(0.0)
        queue.wait_idle()
        self.assertEqual(gripper.move_calls, [(0.0, 0.03)])
        queue.stop()

    def test_gripper_queue_reports_non_contact_move_failure(self) -> None:
        class FakeFuture:
            def wait(self, timeout: float) -> bool:
                return True

            def get(self) -> bool:
                return False

        state = types.SimpleNamespace(width=0.04, max_width=0.0798, is_grasped=False)
        gripper = types.SimpleNamespace(state=state)
        gripper.move_async = lambda width, speed: FakeFuture()
        gripper.stop = lambda: None
        queue = streaming.AsyncGripperQueue(gripper, speed=0.03, tolerance=1e-4)
        queue.command(0.07)
        with self.assertRaisesRegex(RuntimeError, "kind=move"):
            queue.wait_idle()
        with self.assertRaisesRegex(RuntimeError, "kind=move"):
            queue.stop()

    def test_gripper_queue_never_calls_hand_api_on_command_thread(self) -> None:
        class PendingFuture:
            def wait(self, timeout: float) -> bool:
                return False

        main_thread = threading.get_ident()
        api_thread: list[int] = []
        started = threading.Event()
        state = types.SimpleNamespace(width=0.04, max_width=0.08, is_grasped=False)

        class FakeGripper:
            @property
            def state(self):
                return state

            def move_async(self, _width, _speed):
                api_thread.append(threading.get_ident())
                started.set()
                return PendingFuture()

            def stop(self):
                return None

        queue = streaming.AsyncGripperQueue(
            FakeGripper(), speed=0.1, tolerance=1e-4
        )
        started_at = time.perf_counter()
        queue.command(0.08)
        elapsed = time.perf_counter() - started_at
        self.assertLess(elapsed, 0.02)
        self.assertTrue(started.wait(1.0))
        self.assertNotEqual(api_thread, [main_thread])
        queue.poll()
        queue.stop()

    def test_policy_deadline_miss_holds_and_watchdog_aborts(self) -> None:
        period = 33_333_333
        self.assertTrue(streaming.policy_result_is_timely(period, period, 100_000_000))
        self.assertFalse(
            streaming.policy_result_is_timely(period + 1, period, 100_000_000)
        )
        self.assertFalse(
            streaming.policy_result_is_timely(
                1_000_000, period, 100_000_000, period
            )
        )
        with self.assertRaisesRegex(RuntimeError, "Policy watchdog exceeded"):
            streaming.policy_result_is_timely(100_000_001, period, 100_000_000)

    def test_direct_bc_warmup_covers_zero_and_nonzero_cuda_paths(self) -> None:
        class FakeBundle:
            metadata = {"deployment_variant": "frozen_encoder_direct_bc"}
            history_dim = 4
            proprio_dim = 15
            contact_force_dim = 0
            rgb_input_shape = (3, 8, 8, 3)
            gelsight_input_shapes = {
                "gsmini_left_rgb": (4, 6, 3),
                "gsmini_right_rgb": (4, 6, 3),
                "gsmini_left_reference_rgb": (4, 6, 3),
                "gsmini_right_reference_rgb": (4, 6, 3),
            }

            def __init__(self) -> None:
                self.pixel_values: list[int] = []

            def predict(self, _history, _proprio, wrist_rgb, *_args, **_kwargs):
                self.pixel_values.append(int(wrist_rgb[0, 0, 0, 0]))
                return np.zeros(4, dtype=np.float32)

        bundle = FakeBundle()
        report = streaming.warm_up_bundle_policy(bundle)
        self.assertEqual(bundle.pixel_values, [0, 127])
        self.assertEqual(report["iterations"], 2)
        self.assertEqual(report["pixel_values"], [0, 127])

    def test_existing_policy_keeps_single_zero_warmup(self) -> None:
        class FakeBundle:
            metadata = {}
            history_dim = 4
            proprio_dim = 15
            contact_force_dim = 0
            rgb_input_shape = (8, 8, 3)
            gelsight_input_shapes: dict[str, tuple[int, int, int]] = {}

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, *_args, **_kwargs):
                self.calls += 1
                return np.zeros(4, dtype=np.float32)

        bundle = FakeBundle()
        report = streaming.warm_up_bundle_policy(bundle)
        self.assertEqual(bundle.calls, 1)
        self.assertEqual(report["pixel_values"], [0])

    def test_runtime_rejects_unpatched_pylibfranka(self) -> None:
        module = types.SimpleNamespace(
            __version__="0.21.1",
            AsyncPositionControlHandler=type("Handler", (), {}),
        )
        with self.assertRaisesRegex(RuntimeError, "async state patch"):
            streaming.validate_pylibfranka_streaming_api(module)

    def test_streaming_check_uses_hold_only_gate_without_gripper(self) -> None:
        config = load_bundle_config(
            Path(__file__).resolve().parents[1] / "configs/e2e_bundle_real_exported_0801.json"
        )
        joints = list(config.initial_state.joint_positions)
        joints[0] += 0.012
        tcp = list(config.initial_state.tcp_translation)
        tcp[0] += 0.0082
        observation = RobotObservation(
            joint_positions=joints,
            joint_velocities=[0.0] * 7,
            tcp_translation=tcp,
            tcp_quaternion=[0.0, 0.0, 0.0, 1.0],
            external_wrench=[0.0] * 6,
            robot_mode="Idle",
            has_errors=False,
            is_in_control=False,
            control_command_success_rate=1.0,
            gripper_width=None,
            gripper_max_width=None,
            gripper_is_grasped=None,
        )
        strict = streaming.evaluate_initial_state(observation, config.initial_state)
        self.assertFalse(strict["passed"])
        hold_only = streaming.evaluate_streaming_check_state(observation, config)
        self.assertTrue(hold_only["passed"], hold_only["failures"])
        self.assertEqual(hold_only["profile"], "streaming_check_hold_only")
        self.assertNotIn("gripper_width", hold_only["checks"])

        observation.tcp_translation = [0.7, 0.0, 0.3]
        outside = streaming.evaluate_streaming_check_state(observation, config)
        self.assertFalse(outside["checks"]["workspace"]["passed"])


class StreamingArtifactTests(unittest.TestCase):
    def test_fast_png_writer_preserves_rgb_values_losslessly(self) -> None:
        image = np.asarray(
            [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [13, 29, 47]]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            streaming._save_rgb_png(path, image)
            decoded = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(decoded, image)

    def test_compact_profile_releases_duplicate_model_and_tactile_arrays(self) -> None:
        config = streaming_config()
        config.camera.save_rgb = False
        config.tactile_camera.save_rgb = False
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        record = {
            "_model_rgb": image,
            "_tactile_images": {"left": image},
            "_tactile_references": {"left_reference": image},
            "_raw_image": image,
        }
        result = types.SimpleNamespace(image=image)
        retained = streaming._retain_artifact_arrays(
            record, result, config, save_step_data=False
        )
        self.assertIsNone(retained)
        self.assertNotIn("_model_rgb", record)
        self.assertNotIn("_tactile_images", record)
        self.assertNotIn("_tactile_references", record)
        self.assertIn("_raw_image", record)


class FakeStreamingIntegrationTests(unittest.TestCase):
    def test_two_policy_steps_generate_four_control_ticks_and_stop(self) -> None:
        config = streaming_config()

        class Errors:
            pass

        class State:
            q = np.asarray([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
            dq = np.zeros(7)
            O_T_EE = np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).reshape(-1, order="F")
            O_F_ext_hat_K = np.zeros(6)
            current_errors = Errors()
            last_motion_errors = Errors()
            control_command_success_rate = 1.0
            robot_mode = "RobotMode.Idle"

        state = State()

        class FakeModel:
            def zero_jacobian(self, unused_state):
                jacobian = np.zeros((6, 7))
                jacobian[:6, :6] = np.eye(6)
                return jacobian.reshape(-1, order="F")

        class FakeRobot:
            def __init__(self, *unused_args):
                pass

            def load_model(self):
                return FakeModel()

            def read_once(self):
                return state

        class Result:
            was_successful = True
            error_message = ""

        class Handler:
            def __init__(self):
                self.targets = []
                self.stopped = False

            def read_once(self):
                return state

            def set_joint_position_target(self, target):
                self.targets.append(target.joint_positions)
                return Result()

            def stop_control(self):
                self.stopped = True

        handler = Handler()

        class Async:
            class Configuration:
                def __init__(self, velocities, tolerance):
                    self.velocities = velocities
                    self.tolerance = tolerance

            class JointPositionTarget:
                def __init__(self, joints):
                    self.joint_positions = joints

            @staticmethod
            def configure(unused_robot, unused_config):
                return types.SimpleNamespace(handler=handler, error_message="")

            def read_once(self):
                pass

        fake_pylibfranka = types.SimpleNamespace(
            __version__="0.21.1",
            AsyncPositionControlHandler=Async,
            TargetStatus=object,
            Robot=FakeRobot,
            RealtimeConfig=types.SimpleNamespace(kEnforce=object()),
        )

        class GripperState:
            width = 0.04
            max_width = 0.08
            is_grasped = False

        class FakeGripper:
            state = GripperState()
            stopped = False

            def stop(self):
                self.stopped = True

        fake_franky = types.SimpleNamespace(Gripper=lambda unused_ip: FakeGripper())
        fake_sidecar = types.SimpleNamespace(install=lambda unused_module: None)

        class FakePolicy:
            action_dim = 4
            history_dim = 4
            proprio_dim = 15
            rgb_height = 8
            rgb_width = 8
            metadata = {"policy_contract": policy_contract()}

            def __init__(self, *unused_args, **unused_kwargs):
                pass

            def predict(self, unused_history, unused_proprio, unused_image):
                return np.asarray([0.1, -0.1, 0.1, 0.0], dtype=np.float32)

        class FakeCamera:
            def read(self):
                return np.zeros((8, 8, 3), dtype=np.uint8)

            def close(self):
                pass

        class FakeClock:
            now = 0

            def __call__(self):
                return self.now

            def sleep(self, seconds):
                self.now += round(seconds * 1e9)

        clock = FakeClock()
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            sys.modules,
            {
                "pylibfranka": fake_pylibfranka,
                "franky": fake_franky,
                "pylibfranka_streaming_patch": fake_sidecar,
            },
        ), patch.object(streaming, "BundleTorchScriptPolicy", FakePolicy), patch.object(
            streaming, "_validate_bundle_action_dims", lambda unused_bundle, unused_config: None
        ), patch.object(streaming, "_make_camera", return_value=FakeCamera()), patch.object(
            streaming, "_make_run_dir", return_value=Path(temp_dir)
        ):
            summary = streaming.run_streaming_bundle_deploy(
                config, clock_ns=clock, sleep=clock.sleep
            )

        self.assertEqual(summary["num_steps"], 2)
        self.assertEqual(summary["num_control_ticks"], 4)
        self.assertEqual(len(handler.targets), 4)
        self.assertTrue(handler.stopped)
        np.testing.assert_allclose(
            summary["steps"][1]["model_input"]["action_history"],
            [0.0025, -0.0025, 0.0025, 0.0],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()
