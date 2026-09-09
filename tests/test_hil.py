from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import franka_sim2real.hil as hil_module
from franka_sim2real.e2e_bundle import (
    ActionHistoryBuffer,
    BundleActionAdapterConfig,
    BundleDeployConfig,
)
from franka_sim2real.hil import (
    HILInputSnapshot,
    HILKeyState,
    HILSettings,
    human_normalized_xyz,
)
from franka_sim2real.streaming import (
    _PolicyResult,
    _policy_record,
    _write_streaming_artifacts,
    apply_hil_action,
)
from franka_sim2real.streaming_server9 import _preview
from franka_sim2real.types import RobotAction, RobotObservation


def _config() -> BundleDeployConfig:
    config = BundleDeployConfig()
    config.control_mode = "streaming"
    config.streaming.backend = "server9_joint_position"
    config.streaming.policy_frequency_hz = 30.0
    config.streaming.commissioning_action_limit = 0.1
    config.action_adapter = BundleActionAdapterConfig(
        labels=["dx", "dy", "dz", "gripper"],
        scales=[0.05, 0.05, 0.05, 0.01],
        clip_low=[-1.0] * 4,
        clip_high=[1.0] * 4,
        gripper_mode="delta_width",
    )
    return config


def _snapshot(
    *,
    intervention: bool,
    direction: tuple[float, float, float],
) -> HILInputSnapshot:
    return HILInputSnapshot(
        intervention=intervention,
        direction_xyz=direction,
        pressed_keys=("space", "w") if intervention else (),
        sampled_monotonic_ns=123,
        focused=True,
    )


def _result(raw_action: list[float] | None = None) -> _PolicyResult:
    raw = np.asarray(raw_action or [0.08, -0.02, 0.04, 0.6], dtype=np.float32)
    return _PolicyResult(
        raw_action=raw,
        executed_action=np.clip(raw, -0.1, 0.1).astype(np.float32),
        robot_action=RobotAction(),
        action_history=np.zeros(4, dtype=np.float32),
        proprio=np.zeros(15, dtype=np.float32),
        contact_force_n=None,
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        model_rgb=np.zeros((4, 5, 3), dtype=np.uint8),
        tactile_images={},
        tactile_references={},
        inference_info={},
        elapsed_ns=1_000_000,
    )


def _observation() -> RobotObservation:
    return RobotObservation(
        joint_positions=[0.0] * 7,
        joint_velocities=[0.0] * 7,
        tcp_translation=[0.4, 0.0, 0.2],
        tcp_quaternion=[0.0, 0.0, 0.0, 1.0],
        external_wrench=[0.0] * 6,
        robot_mode="Idle",
        has_errors=False,
        is_in_control=False,
        control_command_success_rate=1.0,
        gripper_width=0.04,
    )


class HILKeyStateTests(unittest.TestCase):
    def test_import_does_not_eagerly_load_pygame(self) -> None:
        self.assertNotIn("pygame", hil_module.__dict__)

    def test_space_press_release_and_focus_loss(self) -> None:
        state = HILKeyState()
        state.set_focus(True)
        state.set_key("space", True)
        state.set_key("w", True)
        self.assertTrue(state.snapshot(1).intervention)
        self.assertEqual(state.snapshot(1).direction_xyz, (1.0, 0.0, 0.0))

        state.set_key("space", False)
        self.assertFalse(state.snapshot(2).intervention)
        state.set_key("space", True)
        state.set_focus(False)
        lost = state.snapshot(3)
        self.assertFalse(lost.intervention)
        self.assertEqual(lost.pressed_keys, ())
        self.assertEqual(lost.direction_xyz, (0.0, 0.0, 0.0))

    def test_opposite_keys_cancel_and_diagonal_is_normalized(self) -> None:
        state = HILKeyState()
        state.set_focus(True)
        state.set_key("w", True)
        state.set_key("s", True)
        self.assertEqual(state.snapshot(1).direction_xyz, (0.0, 0.0, 0.0))

        state.set_key("s", False)
        state.set_key("a", True)
        diagonal = np.asarray(state.snapshot(2).direction_xyz)
        self.assertAlmostEqual(float(np.linalg.norm(diagonal)), 1.0)
        np.testing.assert_allclose(diagonal, [2**-0.5, 2**-0.5, 0.0])

    def test_space_without_direction_is_zero_xyz_intervention(self) -> None:
        state = HILKeyState()
        state.set_focus(True)
        state.set_key("space", True)
        snapshot = state.snapshot(1)
        self.assertTrue(snapshot.intervention)
        self.assertEqual(snapshot.direction_xyz, (0.0, 0.0, 0.0))


class HILActionTests(unittest.TestCase):
    def test_speed_is_converted_to_prelimit_normalized_action(self) -> None:
        value = human_normalized_xyz(
            _snapshot(intervention=True, direction=(1.0, 0.0, 0.0)),
            speed_m_s=0.05,
            policy_frequency_hz=30.0,
            action_scales=[0.05, 0.05, 0.05, 0.01],
        )
        np.testing.assert_allclose(value, [1.0 / 30.0, 0.0, 0.0], atol=1e-7)

    def test_intervention_replaces_xyz_keeps_base_gripper_and_computes_residual(self) -> None:
        base = _result([0.8, -0.2, 0.4, 0.6])
        selected = apply_hil_action(
            base,
            _snapshot(intervention=True, direction=(1.0, 0.0, 0.0)),
            HILSettings(enabled=True, speed_m_s=0.05),
            _config(),
            allow_full_scale=False,
            desired_gripper_width=0.04,
        )

        assert selected.hil_step is not None and selected.hil_step.human_action is not None
        np.testing.assert_allclose(
            selected.hil_step.human_action,
            [1.0 / 30.0, 0.0, 0.0, 0.6],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            selected.hil_step.residual_target_xyz,
            [1.0 / 30.0 - 0.8, 0.2, -0.4],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            selected.executed_action,
            [1.0 / 30.0, 0.0, 0.0, 0.1],
            atol=1e-7,
        )
        self.assertAlmostEqual(selected.robot_action.dx, 0.05 / 30.0)
        self.assertAlmostEqual(selected.robot_action.gripper_width, 0.041)

    def test_nonintervention_uses_base_and_zero_residual(self) -> None:
        base = _result()
        selected = apply_hil_action(
            base,
            _snapshot(intervention=False, direction=(1.0, 0.0, 0.0)),
            HILSettings(enabled=True),
            _config(),
            allow_full_scale=False,
            desired_gripper_width=0.04,
        )
        assert selected.hil_step is not None
        self.assertIsNone(selected.hil_step.human_action)
        np.testing.assert_array_equal(selected.hil_step.residual_target_xyz, np.zeros(3))
        np.testing.assert_array_equal(selected.raw_action, base.raw_action)

    def test_history_uses_accepted_limited_human_action(self) -> None:
        selected = apply_hil_action(
            _result([0.8, -0.2, 0.4, 0.6]),
            _snapshot(intervention=True, direction=(1.0, 0.0, 0.0)),
            HILSettings(enabled=True),
            _config(),
            allow_full_scale=False,
            desired_gripper_width=0.04,
        )
        history = ActionHistoryBuffer(
            4,
            "processed_action",
            processed_action_scale=[0.05, 0.05, 0.05, 0.01],
        )
        history.update(selected.raw_action, selected.executed_action)
        np.testing.assert_allclose(
            history.current(),
            [0.05 / 30.0, 0.0, 0.0, 0.001],
            atol=1e-7,
        )

        unchanged = ActionHistoryBuffer(
            4,
            "processed_action",
            processed_action_scale=[0.05, 0.05, 0.05, 0.01],
        )
        # A deadline miss follows the deployment loop's existing hold path:
        # no ActionHistoryBuffer.update call is made.
        np.testing.assert_array_equal(unchanged.current(), np.zeros(4))

    def test_settings_reject_nonfinite_or_nonpositive_speed(self) -> None:
        for speed in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                HILSettings(enabled=True, speed_m_s=speed).validate()


class HILLoggingTests(unittest.TestCase):
    def test_jsonl_and_npz_include_hil_labels_and_deadline_acceptance(self) -> None:
        selected = apply_hil_action(
            _result([0.8, -0.2, 0.4, 0.6]),
            _snapshot(intervention=True, direction=(1.0, 0.0, 0.0)),
            HILSettings(enabled=True),
            _config(),
            allow_full_scale=False,
            desired_gripper_width=0.04,
        )
        record = _policy_record(
            7,
            _observation(),
            selected,
            accepted=False,
            motion_enabled=True,
            episode_id="episode_hil",
        )
        self.assertIsNone(record["executed_action"])
        self.assertFalse(record["info"]["policy_action_accepted"])

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_streaming_artifacts(
                run_dir,
                _config(),
                {"passed": True},
                [record],
                [selected.image],
                [],
                {},
                save_step_data=True,
            )
            logged = json.loads((run_dir / "rollout.jsonl").read_text())
            self.assertEqual(logged["episode_id"], "episode_hil")
            self.assertEqual(logged["step_id"], 7)
            self.assertTrue(logged["intervention"])
            self.assertIsNotNone(logged["human_action"])
            self.assertIn("base_action", logged)
            self.assertIn("residual_target_xyz", logged)
            with np.load(run_dir / "step_data" / "step_0000.npz") as payload:
                self.assertEqual(str(payload["episode_id"]), "episode_hil")
                self.assertEqual(int(payload["step_id"]), 7)
                self.assertFalse(bool(payload["policy_action_accepted"]))
                self.assertIn("base_action", payload.files)
                self.assertIn("human_action", payload.files)
                self.assertIn("residual_target_xyz", payload.files)

    def test_normal_hil_step_omits_human_npz_array_and_no_hil_log_is_compatible(self) -> None:
        base = _result()
        no_hil = _policy_record(
            0, _observation(), base, accepted=True, motion_enabled=False
        )
        self.assertNotIn("intervention", no_hil)
        self.assertNotIn("base_action", no_hil)

        selected = apply_hil_action(
            base,
            _snapshot(intervention=False, direction=(0.0, 0.0, 0.0)),
            HILSettings(enabled=True),
            _config(),
            allow_full_scale=False,
            desired_gripper_width=0.04,
        )
        record = _policy_record(
            0,
            _observation(),
            selected,
            accepted=True,
            motion_enabled=False,
            episode_id="episode_hil",
        )
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            _write_streaming_artifacts(
                run_dir,
                _config(),
                {"passed": True},
                [record],
                [selected.image],
                [],
                {},
                save_step_data=True,
            )
            with np.load(run_dir / "step_data" / "step_0000.npz") as payload:
                self.assertNotIn("human_action", payload.files)
                np.testing.assert_array_equal(payload["residual_target_xyz"], np.zeros(3))


class HILPreviewControlFlowTests(unittest.TestCase):
    def test_policy_runs_during_continuous_intervention_and_sees_human_history(self) -> None:
        config = _config()
        config.runner.steps = 3
        history = ActionHistoryBuffer(
            4,
            "processed_action",
            processed_action_scale=config.action_adapter.scales,
        )

        class FakeBundle:
            rgb_width = 5
            rgb_height = 4
            history_dim = 4
            proprio_dim = 15
            contact_force_dim = 0
            last_inference_info = {}

            def __init__(self) -> None:
                self.histories: list[np.ndarray] = []

            def predict(self, action_history, proprio, image, **kwargs):
                self.histories.append(np.asarray(action_history).copy())
                return np.asarray([0.8, -0.2, 0.4, 0.6], dtype=np.float32)

        class FakeCamera:
            def read(self):
                return np.zeros((4, 5, 3), dtype=np.uint8)

        class FakeKeyboard:
            def sample(self):
                return _snapshot(intervention=True, direction=(1.0, 0.0, 0.0))

        bundle = FakeBundle()
        records: list[dict] = []
        images: list[np.ndarray] = []
        timing = {"maximum_policy_elapsed_ms": 0.0, "policy_deadline_misses": 0}
        _preview(
            bundle,
            FakeCamera(),
            history,
            _observation(),
            _result([0.8, -0.2, 0.4, 0.6]),
            None,
            config,
            False,
            timing,
            records,
            images,
            clock_ns=lambda: 0,
            sleep=lambda _: None,
            hil_settings=HILSettings(enabled=True),
            hil_keyboard=FakeKeyboard(),
            episode_id="episode_hil",
        )

        # The first base action was precomputed, then the policy still ran for
        # both remaining steps while Space stayed held.
        self.assertEqual(len(bundle.histories) + 1, config.runner.steps)
        self.assertEqual(len(records), config.runner.steps)
        self.assertTrue(all(record["intervention"] for record in records))
        self.assertTrue(
            all(not record["info"]["policy_action_accepted"] for record in records)
        )
        self.assertTrue(all(record["executed_action"] is None for record in records))
        self.assertTrue(all(not record["info"]["policy_deadline_miss"] for record in records))
        expected_history = np.asarray([0.05 / 30.0, 0.0, 0.0, 0.001])
        np.testing.assert_allclose(bundle.histories[0], expected_history, atol=1e-7)
        np.testing.assert_allclose(bundle.histories[1], expected_history, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
