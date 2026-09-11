from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from real_rlpd.action import combine_post_limit, expert_residual_target
from real_rlpd.config import AlgorithmConfig, load_config
from real_rlpd.collector import RLPDCollector
from real_rlpd.expert_dataset import ExpertEpisodeDataset
from real_rlpd.learner import (
    ACTION_DIM,
    PRIVILEGED_DIM,
    STATE_DIM,
    FrozenNormalizer,
    RLPDLearner,
)
from real_rlpd.replay import ReplayBuffer, Transition
from real_rlpd.reward import RewardInput, build_reward
from real_rlpd.runtime import RLPDDeploySettings
from real_rlpd.teleop import ExpertKeyState, ExpertSnapshot, expert_raw_action
from real_rlpd.trainer import mixed_batch
from scripts.real_rlpd.run_rlpd import _base_config
from franka_sim2real.e2e_bundle import load_bundle_config, validate_bundle_artifacts
from franka_sim2real.streaming import _PolicyResult
from franka_sim2real.streaming_server9 import (
    _apply_rlpd_expert_action,
    _print_streaming_progress,
)
from franka_sim2real.types import RobotAction


REPO_ROOT = Path(__file__).resolve().parents[1]


def _transition(step: int, *, trainable: bool = True) -> Transition:
    state = np.full(STATE_DIM, step * 0.01, dtype=np.float32)
    return Transition(
        episode_id="episode",
        step_id=step,
        state=state,
        next_state=state + 0.001,
        action=np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        base_limited_action=np.zeros(ACTION_DIM, dtype=np.float32),
        executed_action=np.zeros(ACTION_DIM, dtype=np.float32),
        ee_position=np.asarray([0.4, 0.0, 0.2], dtype=np.float32),
        next_ee_position=np.asarray([0.401, 0.0, 0.2], dtype=np.float32),
        initial_object_z=0.02,
        privileged=np.zeros(PRIVILEGED_DIM, dtype=np.float32) if trainable else None,
        next_privileged=np.ones(PRIVILEGED_DIM, dtype=np.float32) if trainable else None,
        reward=1.0 if trainable else None,
        terminated=False,
        truncated=False,
        success=False,
        accepted=True,
        trainable=trainable,
        trainable_reason="ok" if trainable else "offline_apriltag_pending",
        action_timestamp=1.0,
        run_dir="/tmp/episode",
    )


class RLPDConfigTests(unittest.TestCase):
    def test_0823_uses_corrected_tool_offset_and_separate_data_roots(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0823.json")
        base = load_bundle_config(config.base_policy_config)
        self.assertEqual(config.schema_version, 2)
        self.assertEqual(base.tool_tcp_offset_ee_m, [0.0, 0.0, 0.0579])
        self.assertEqual(base.workspace["minimum"][2], 0.01)
        self.assertEqual(config.expert_data_dir, REPO_ROOT / "real_rlpd_data" / "expert")
        self.assertEqual(
            config.rollout_data_dir,
            REPO_ROOT / "real_rlpd_data" / "rollout" / "0823",
        )
        self.assertNotEqual(config.expert_data_dir, config.rollout_data_dir)
        expert_base = _base_config(config, "cpu", data_role="expert")
        rollout_base = _base_config(config, "cpu", data_role="rollout")
        self.assertEqual(Path(expert_base.runner.log_dir), config.expert_data_dir)
        self.assertEqual(Path(rollout_base.runner.log_dir), config.rollout_data_dir)
        self.assertTrue(expert_base.runner.run_name.startswith("expert_"))
        self.assertTrue(rollout_base.runner.run_name.startswith("rollout_"))

    def test_expert_and_rollout_roots_must_not_overlap(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0823.json")
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            replace(
                config,
                rollout_data_dir=config.expert_data_dir / "rollout",
            ).validate()

    def test_0911_progress_uses_its_own_policy_and_replay_contract(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0911_progress.json")
        base = _base_config(config, "cpu")
        settings = RLPDDeploySettings(
            config,
            "expert_shadow",
            base.streaming.commissioning_action_limit,
        )
        report = validate_bundle_artifacts(base, rlpd_settings=settings)
        contract = report["rlpd"]["contract"]

        self.assertEqual(config.reward.success_height_m, 0.035)
        self.assertEqual(config.reward.success_consecutive_detections, 5)
        self.assertEqual(
            config.rollout_data_dir,
            REPO_ROOT / "real_rlpd_data" / "rollout" / "0911_progress",
        )
        self.assertEqual(
            contract["adapter_id"], "gelsight_reference_progress_single_frame_v1"
        )
        self.assertEqual(
            contract["policy_kind"],
            "tacex_rma_gelsight_size_buckets_progress_student_torchscript",
        )


class RLPDActionTests(unittest.TestCase):
    def test_expert_target_exactly_reconstructs_full_takeover(self) -> None:
        base = np.asarray([0.1, -0.1, 0.04, -0.03], dtype=np.float32)
        expert = np.asarray([-0.1, 0.06, -0.1, 0.1], dtype=np.float32)
        unit = expert_residual_target(base, expert, 0.1)
        candidate, residual = combine_post_limit(base, unit, 0.1)
        np.testing.assert_allclose(candidate, expert, atol=1e-7)
        np.testing.assert_allclose(residual, expert - base, atol=1e-7)

    def test_keyboard_diagonal_is_normalized_and_gripper_is_independent(self) -> None:
        state = ExpertKeyState()
        state.set_focus(True)
        for key in ("w", "a", "u"):
            state.set_key(key, True)
        snapshot = state.snapshot()
        np.testing.assert_allclose(snapshot.xyz_direction[:2], [2 ** -0.5] * 2)
        action = expert_raw_action(
            snapshot,
            xyz_speed_m_s=0.05,
            gripper_speed_m_s=0.03,
            policy_frequency_hz=30.0,
            action_scales=[0.05, 0.05, 0.05, 0.01],
        )
        np.testing.assert_allclose(action[:2], [2 ** -0.5 / 30.0] * 2, rtol=1e-6)
        self.assertAlmostEqual(float(action[3]), 0.1, places=6)

    def test_server9_expert_action_replaces_shadow_policy_in_all_four_axes(self) -> None:
        rlpd_config = load_config(REPO_ROOT / "configs" / "real_rlpd_0814.json")
        base_config = load_bundle_config(rlpd_config.base_policy_config)
        settings = RLPDDeploySettings(
            rlpd_config,
            "expert_shadow",
            base_config.streaming.commissioning_action_limit,
        )
        base_limited = np.asarray([0.1, -0.1, 0.04, -0.03], dtype=np.float32)
        result = _PolicyResult(
            raw_action=base_limited.copy(),
            executed_action=base_limited.copy(),
            robot_action=RobotAction(),
            action_history=np.zeros(4, dtype=np.float32),
            proprio=np.zeros(15, dtype=np.float32),
            contact_force_n=None,
            image=np.zeros((1, 1, 3), dtype=np.uint8),
            model_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
            tactile_images={},
            tactile_references={},
            inference_info={
                "rlpd": {
                    "base_limited_action": base_limited.tolist(),
                    "unit_residual_action": [0.0] * 4,
                    "residual_normalized_action": [0.0] * 4,
                    "mode": "expert_shadow",
                    "stochastic": False,
                    "checkpoint_sha256": None,
                }
            },
            elapsed_ns=1,
        )

        class Keyboard:
            @staticmethod
            def sample() -> ExpertSnapshot:
                return ExpertSnapshot((1.0, 0.0, 0.0), -1.0, ("i", "w"), True, 123)

        takeover = _apply_rlpd_expert_action(
            result, Keyboard(), settings, base_config, desired_gripper_width=0.04
        )
        expected_x = min(
            rlpd_config.teleop.xyz_speed_m_s
            / base_config.streaming.policy_frequency_hz
            / base_config.action_adapter.scales[0],
            base_config.streaming.commissioning_action_limit,
        )
        np.testing.assert_allclose(
            takeover.executed_action,
            [expected_x, 0.0, 0.0, -0.1],
            atol=1e-7,
        )
        info = takeover.inference_info["rlpd"]
        self.assertNotIn("unit_residual_action", info)
        np.testing.assert_allclose(
            info["expert_limited_action"], takeover.executed_action, atol=1e-7
        )

        with mock.patch("builtins.print") as print_mock:
            _print_streaming_progress(30, 30, 0, result=takeover, accepted=True)
        output = print_mock.call_args.args[0]
        self.assertIn("expert_takeover=True", output)
        self.assertIn("requested_norm=", output)


class ExpertDatasetTests(unittest.TestCase):
    def test_source_dataset_excludes_base_and_residual_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "expert_episode"
            dataset = ExpertEpisodeDataset(
                root,
                action_scales=[0.05, 0.05, 0.05, 0.01],
                initial_object_z=0.02,
                collection_provenance={"shadow_policy_kind": "test"},
            )

            def result(step: int, *, expert: bool = False) -> _PolicyResult:
                info = {
                    "expert_takeover": True,
                    "pressed_keys": ["w"],
                    "sampled_monotonic_ns": 123 + step,
                    "focused": True,
                } if expert else {}
                return _PolicyResult(
                    raw_action=np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                    executed_action=np.asarray([0.1, 0.0, 0.0, 0.0], dtype=np.float32),
                    robot_action=RobotAction(),
                    action_history=np.zeros(4, dtype=np.float32),
                    proprio=np.zeros(15, dtype=np.float32),
                    contact_force_n=None,
                    image=np.full((8, 9, 3), step, dtype=np.uint8),
                    model_rgb=np.full((3, 8, 9, 3), step, dtype=np.uint8),
                    tactile_images={
                        "gsmini_left_rgb": np.full((4, 5, 3), step, dtype=np.uint8),
                        "gsmini_right_rgb": np.full((4, 5, 3), step, dtype=np.uint8),
                    },
                    tactile_references={
                        "gsmini_left_reference_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                        "gsmini_right_reference_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
                    },
                    inference_info={"rlpd": info},
                    elapsed_ns=1,
                    camera_metadata={"sequence": step},
                )

            observation = mock.Mock()
            observation.to_dict.return_value = {"joint_positions": [0.0] * 7}
            first = result(0, expert=True)
            dataset.observe_boundary(0, observation, first)
            dataset.record_action(first, True, action_timestamp=1.25)
            dataset.observe_boundary(1, observation, result(1), truncate=True)
            report = dataset.close()

            self.assertEqual(report["complete_transitions"], 1)
            manifest = json.loads((root / "expert_manifest.json").read_text())
            self.assertEqual(manifest["kind"], "franka_policy_independent_expert_episode")
            self.assertEqual(
                manifest["derived_fields_excluded"],
                ["base_policy_feature", "base_policy_action", "residual_action"],
            )
            action = json.loads((root / "expert_actions.jsonl").read_text())
            self.assertNotIn("base_limited_action", action)
            self.assertNotIn("unit_residual_action", action)
            np.testing.assert_allclose(
                action["expert_action_physical_delta_m"], [0.005, 0.0, 0.0, 0.0]
            )
            boundary = np.load(root / "expert_observations" / "boundary_0000.npz")
            self.assertIn("wrist_rgb", boundary.files)
            self.assertIn("gsmini_left_rgb", boundary.files)
            self.assertIn("gsmini_right_rgb", boundary.files)


class RLPDReplayTests(unittest.TestCase):
    def test_policy_contract_and_trainable_filter_are_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "offline.sqlite3"
            contract = {"policy": "0814", "action_dim": 4}
            with ReplayBuffer(path, "offline") as replay:
                replay.assert_contract(contract)
                replay.append(_transition(0))
                replay.append(_transition(1, trainable=False))
                self.assertEqual(replay.count(), 2)
                self.assertEqual(replay.count(trainable_only=True), 1)
            with ReplayBuffer(path, "offline", create=False) as replay:
                replay.assert_contract(contract)
                self.assertEqual(replay.load()["state"].shape, (1, STATE_DIM))
                with self.assertRaises(ValueError):
                    replay.assert_contract({"policy": "0823", "action_dim": 4})

    def test_collector_rejects_replay_contract_before_apriltag_preflight(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0814.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = replace(config, offline_replay_path=root / "offline.sqlite3")
            settings = RLPDDeploySettings(config, "expert_shadow", 0.1)
            with ReplayBuffer(config.offline_replay_path, "offline") as replay:
                replay.assert_contract({"reward": {"success_height_m": 0.015}})
            with mock.patch("real_rlpd.collector.AprilTagTracker") as tracker:
                with self.assertRaisesRegex(ValueError, "replay contract mismatch"):
                    RLPDCollector(
                        settings,
                        camera=object(),
                        run_dir=root / "episode",
                        role="offline",
                        contract={"reward": {"success_height_m": 0.05}},
                    )
            tracker.assert_not_called()

    def test_mixed_batch_is_exactly_half_offline_when_online_exists(self) -> None:
        offline = {
            "state": np.zeros((8, 1), dtype=np.float32),
            "reward": np.zeros(8, dtype=np.float32),
        }
        online = {
            "state": np.ones((8, 1), dtype=np.float32),
            "reward": np.ones(8, dtype=np.float32),
        }
        batch = mixed_batch(
            np.random.default_rng(4), offline, online, total=10, offline_ratio=0.5
        )
        self.assertEqual(int(batch["state"].sum()), 5)

    def test_configured_reward_interface_preserves_current_task_formula(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0814.json")
        self.assertEqual(config.reward.success_height_m, 0.05)
        self.assertEqual(config.reward.success_consecutive_detections, 3)
        reward = build_reward(config.reward_kind, config.reward).compute(RewardInput(
            object_relative_to_ee=np.asarray([0.10, 0.0, 0.0]),
            next_object_relative_to_ee=np.asarray([0.08, 0.0, 0.0]),
            object_height=0.0,
            next_object_height=0.01,
            action=np.asarray([1.0, 0.0, 0.0, 0.0]),
            success=True,
        ))
        self.assertAlmostEqual(reward.reach, 0.2)
        self.assertAlmostEqual(reward.lift, 0.2)
        self.assertAlmostEqual(reward.success, 10.0)
        self.assertAlmostEqual(reward.action_penalty, 0.01)
        self.assertAlmostEqual(reward.total, 10.39)


class RLPDLearnerTests(unittest.TestCase):
    def test_ensemble_update_uses_one_actor_update_per_utd_group(self) -> None:
        rng = np.random.default_rng(8)
        states = rng.normal(size=(8, STATE_DIM)).astype(np.float32)
        privileged = rng.normal(size=(8, PRIVILEGED_DIM)).astype(np.float32)
        normalizer = FrozenNormalizer.fit(states, privileged, 10.0)
        algorithm = replace(
            AlgorithmConfig(),
            batch_size=2,
            utd_ratio=2,
            num_qs=3,
            num_min_qs=2,
            minimum_offline_transitions=2,
        )
        learner = RLPDLearner(algorithm, normalizer, "cpu")
        count = algorithm.batch_size * algorithm.utd_ratio
        batch = {
            "state": states[:count],
            "next_state": states[1:count + 1],
            "privileged": privileged[:count],
            "next_privileged": privileged[1:count + 1],
            "action": rng.uniform(-1, 1, (count, ACTION_DIM)).astype(np.float32),
            "reward": rng.normal(size=count).astype(np.float32),
            "terminated": np.zeros(count, dtype=np.float32),
        }
        metrics = learner.update_group(batch, rng)
        self.assertEqual(learner.update_groups, 1)
        self.assertEqual(learner.critic_updates, 2)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_stochastic_deployment_needs_explicit_enable(self) -> None:
        config = load_config(REPO_ROOT / "configs" / "real_rlpd_0814.json")
        settings = RLPDDeploySettings(
            config=config,
            mode="checkpoint",
            commissioning_limit=0.1,
            checkpoint_path=Path("dummy.pt"),
            stochastic=True,
        )
        with self.assertRaises(ValueError):
            settings.validate()


if __name__ == "__main__":
    unittest.main()
