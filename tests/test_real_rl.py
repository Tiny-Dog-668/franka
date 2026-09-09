from __future__ import annotations

import tempfile
import unittest
import importlib.util
import contextlib
import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import cv2

from franka_sim2real.real_rl.config import load_real_rl_config
from franka_sim2real.real_rl.apriltag_tracker import TagPose, solve_tag_pose
from franka_sim2real.real_rl.collector import RealRLCollector
from franka_sim2real.real_rl.offline_apriltag import (
    OfflineAprilTagDetector,
    OfflineTagPose,
    label_episode_from_run,
)
from franka_sim2real.real_rl.replay_buffer import (
    ReplayBuffer,
    ReplayTransition,
    ReplayWriter,
)
from franka_sim2real.real_rl.residual_sac import (
    FrozenNormalizer,
    GaussianActor,
    PrivilegedQ,
    ResidualSAC,
)
from franka_sim2real.real_rl.reward import compute_progress_reward
from franka_sim2real.real_rl.runtime import RealRLDeploySettings, RealRLPolicyRuntime
from franka_sim2real.real_rl.trainer import (
    gradient_updates_for,
    planned_gradient_updates,
    stratified_batch_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def transition(step: int = 0, *, accepted: bool = True, valid: bool = True) -> ReplayTransition:
    state = np.full(1047, step * 0.01, dtype=np.float32)
    next_state = state + 0.001
    object_t = np.asarray([0.5, 0.0, 0.02], dtype=np.float32)
    object_next = np.asarray([0.499, 0.0, 0.021], dtype=np.float32)
    ee_t = np.asarray([0.48, 0.0, 0.02], dtype=np.float32)
    ee_next = np.asarray([0.481, 0.0, 0.02], dtype=np.float32)
    relative_t = object_t - ee_t
    relative_next = object_next - ee_next
    return ReplayTransition(
        episode_id="episode", step_id=step, run_dir="/tmp/run", rollout_step=step,
        state=state, next_state=next_state,
        base_action=np.zeros(4, dtype=np.float32),
        next_base_action=np.zeros(4, dtype=np.float32),
        sac_unit_action=np.asarray([0.1, -0.2, 0.0], dtype=np.float32),
        residual_action_normalized=np.asarray([0.004, -0.008, 0.0], dtype=np.float32),
        residual_action_m=np.asarray([0.0002, -0.0004, 0.0], dtype=np.float32),
        executed_action=np.zeros(4, dtype=np.float32) if accepted else None,
        policy_action_accepted=accepted,
        action_timestamp=1.05 if accepted else None,
        pre_safety_action=np.zeros(4, dtype=np.float32),
        post_safety_action=np.zeros(4, dtype=np.float32),
        safety_intervened=False,
        intervention_magnitude=0.0,
        reward=0.1 if valid else None,
        reach_reward=0.08 if valid else None,
        lift_reward=0.02 if valid else None,
        success_reward=0.0 if valid else None,
        residual_penalty=0.0005 if valid else None,
        reward_valid=valid,
        trainable=valid,
        trainable_reason="ok" if valid else "policy_action_not_accepted",
        object_position_t=object_t if valid else None,
        object_position_next=object_next if valid else None,
        ee_position_t=ee_t, ee_position_next=ee_next,
        object_relative_to_ee_t=relative_t if valid else None,
        object_relative_to_ee_next=relative_next if valid else None,
        initial_object_z_in_base=0.02,
        object_height_t=0.0 if valid else None,
        object_height_next=0.001 if valid else None,
        distance_t=float(np.linalg.norm(relative_t)) if valid else None,
        distance_next=float(np.linalg.norm(relative_next)) if valid else None,
        tag_sequence_t=step if valid else None,
        tag_sequence_next=step + 1 if valid else None,
        tag_age_t=0.01 if valid else None, tag_age_next=0.01 if valid else None,
        reprojection_error_t=0.2 if valid else None,
        reprojection_error_next=0.2 if valid else None,
        tag_capture_timestamp_t=1.0 if valid else None,
        tag_capture_timestamp_next=1.1 if valid else None,
        done=False, terminated=False, truncated=False, success=False,
        terminal_reason=None, base_model_sha256="base-sha",
        residual_checkpoint_sha256=None,
    )


class RealRLRewardTests(unittest.TestCase):
    def test_progress_lift_success_and_penalty(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        result = compute_progress_reward(
            0.10, 0.08, 0.0, 0.01, np.asarray([1.0, 0.0, 0.0]), True, config.reward
        )
        self.assertAlmostEqual(result.reach_reward, 0.2)
        self.assertAlmostEqual(result.lift_reward, 0.2)
        self.assertAlmostEqual(result.success_reward, 10.0)
        self.assertAlmostEqual(result.residual_penalty, 0.01)
        self.assertAlmostEqual(result.reward, 10.39)

    def test_apriltag_pose_solver_matches_synthetic_projection(self) -> None:
        length = 0.04
        half = length / 2.0
        points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        matrix = np.asarray([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1.0]])
        distortion = np.zeros(5)
        rvec = np.asarray([[0.1], [-0.05], [0.02]])
        tvec = np.asarray([[0.01], [-0.02], [0.5]])
        corners, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
        transform, error = solve_tag_pose(corners, matrix, distortion, length)
        np.testing.assert_allclose(transform[:3, 3], tvec.reshape(3), atol=1e-5)
        self.assertLess(error, 1e-4)

    def test_offline_detector_uses_roi_after_full_frame_lock(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        detector = OfflineAprilTagDetector(config, {
            "fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0,
            "distortion_coefficients": [0.0] * 5,
            "distortion_model": "none",
        })
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
            config.apriltag.marker_id,
            100,
        )
        poses = []
        for boundary, x in enumerate((270, 280)):
            image = np.full((480, 640, 3), 255, dtype=np.uint8)
            image[190:290, x:x + 100] = marker[:, :, None]
            poses.append(detector.detect(boundary, image, {
                "sequence": boundary + 1,
                "capture_timestamp": float(boundary),
                "camera_timestamp_ms": None,
            }))
        self.assertIsNotNone(poses[0])
        self.assertIsNotNone(poses[1])
        self.assertEqual(poses[0].search_mode, "full_frame")
        self.assertEqual(poses[1].search_mode, "roi")


class ReplayTests(unittest.TestCase):
    def test_wal_append_reopen_and_valid_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.sqlite3"
            with ReplayBuffer(path) as replay:
                replay.append(transition(0))
                replay.append(transition(1, accepted=False, valid=False))
                self.assertEqual(replay.count(), 2)
                self.assertEqual(replay.valid_count(), 1)
                self.assertEqual(replay.trainable_count(), 1)
                self.assertEqual(replay.get_meta("schema_version"), 2)
            with ReplayBuffer(path, create=False) as replay:
                data = replay.load_valid()
                self.assertEqual(data["state"].shape, (1, 1047))
                self.assertEqual(data["privileged"].shape, (1, 4))
                self.assertTrue(data["trainable"][0])
                self.assertEqual(data["trainable_reason"][0], "ok")
                self.assertEqual(data["pre_safety_action"].shape, (1, 4))
                self.assertEqual(replay.max_id(), 2)
                self.assertEqual(replay.max_label_revision(), 0)
                np.testing.assert_array_equal(data["label_revision"], [0])

    def test_relabel_revision_counts_old_transition_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.sqlite3"
            with ReplayBuffer(path) as replay:
                replay.append(transition(0))
                replay.append(transition(1))
                replay.connection.execute(
                    "UPDATE transitions SET reward=?, label_revision=? WHERE id=?",
                    (12.0, 3, 1),
                )
                replay.connection.commit()
                self.assertEqual(
                    replay.trainable_change_counts(
                        after_id=2, after_label_revision=0
                    ),
                    {"new_id": 0, "relabeled": 1, "total": 1},
                )
                self.assertEqual(replay.max_label_revision(), 3)

    def test_background_writer_flushes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "replay.sqlite3"
            writer = ReplayWriter(path, queue_size=2)
            writer.submit(transition())
            writer.close()
            self.assertEqual(writer.written, 1)
            with ReplayBuffer(path, create=False) as replay:
                self.assertEqual(replay.count(), 1)


class SACTests(unittest.TestCase):
    def test_actor_starts_at_zero_and_privileged_q_contract(self) -> None:
        actor = GaussianActor(-3.0)
        state = torch.randn(5, 1047)
        torch.testing.assert_close(actor(state), torch.zeros(5, 3))
        action, log_prob = actor.sample(state)
        self.assertEqual(action.shape, (5, 3))
        self.assertEqual(log_prob.shape, (5, 1))
        q = PrivilegedQ()(state, torch.zeros(5, 4), action)
        self.assertEqual(q.shape, (5, 1))

    def test_sac_update_with_asymmetric_critic(self) -> None:
        real_config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        rng = np.random.default_rng(3)
        fit_state = rng.normal(size=(1000, 1047)).astype(np.float32)
        fit_priv = rng.normal(size=(1000, 4)).astype(np.float32)
        normalizer = FrozenNormalizer.fit(fit_state, fit_priv, np.arange(1, 1001))
        learner = ResidualSAC(real_config.sac, normalizer, "cpu")
        count = real_config.sac.batch_size
        batch = {
            "state": fit_state[:count], "next_state": fit_state[1:count + 1],
            "privileged": fit_priv[:count], "next_privileged": fit_priv[1:count + 1],
            "sac_unit_action": rng.uniform(-1, 1, (count, 3)).astype(np.float32),
            "reward": rng.normal(size=count).astype(np.float32),
            "terminated": np.zeros(count, dtype=np.float32),
        }
        metrics = learner.update(batch)
        self.assertEqual(learner.update_count, 1)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_normalizer_is_locked_to_first_1000_and_checkpoint_restores(self) -> None:
        real_config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        states = np.zeros((1000, 1047), dtype=np.float32)
        privileged = np.zeros((1000, 4), dtype=np.float32)
        normalizer = FrozenNormalizer.fit(states, privileged, np.arange(1, 1001))
        self.assertEqual(normalizer.sample_count, 1000)
        with self.assertRaises(ValueError):
            FrozenNormalizer.fit(states[:999], privileged[:999], np.arange(1, 1000))
        learner = ResidualSAC(real_config.sac, normalizer, "cpu")
        with torch.no_grad():
            learner.actor.mean.bias.fill_(0.125)
        payload = learner.checkpoint(
            base_model_sha256="base",
            replay_high_watermark=55,
            replay_label_revision=7,
        )
        restored = ResidualSAC(real_config.sac, normalizer, "cpu")
        restored.restore(payload)
        torch.testing.assert_close(
            restored.actor(torch.zeros(1, 1047)), learner.actor(torch.zeros(1, 1047))
        )
        self.assertEqual(payload["replay_high_watermark"], 55)
        self.assertEqual(payload["replay_label_revision"], 7)

    def test_checkpoint_restore_moves_cuda_rng_states_to_cpu(self) -> None:
        real_config = load_real_rl_config(
            REPO_ROOT / "configs/real_residual_sac_0814.json"
        )
        normalizer = FrozenNormalizer.fit(
            np.zeros((1000, 1047), dtype=np.float32),
            np.zeros((1000, 4), dtype=np.float32),
            np.arange(1, 1001),
        )
        learner = ResidualSAC(real_config.sac, normalizer, "cpu")
        payload = learner.checkpoint(
            base_model_sha256="base-sha",
            replay_high_watermark=17,
        )

        class FakeMappedCudaState:
            def __init__(self) -> None:
                self.cpu_state = torch.zeros(8, dtype=torch.uint8)

            def cpu(self) -> torch.Tensor:
                return self.cpu_state

        mapped_state = FakeMappedCudaState()
        payload["cuda_rng_state_all"] = [mapped_state]
        restored = ResidualSAC(real_config.sac, normalizer, "cpu")
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.set_rng_state_all"
        ) as set_rng_state_all:
            restored.restore(payload)
        set_rng_state_all.assert_called_once()
        restored_states = set_rng_state_all.call_args.args[0]
        self.assertEqual(len(restored_states), 1)
        self.assertEqual(restored_states[0].device.type, "cpu")
        torch.testing.assert_close(restored_states[0], mapped_state.cpu_state)

    def test_normalizer_zscores_features_but_not_fixed_contract_base_action(self) -> None:
        states = np.zeros((1000, 1047), dtype=np.float32)
        states[:, :1043] = np.arange(1000, dtype=np.float32)[:, None]
        states[:, 1043:] = np.linspace(-0.8, 0.8, 1000, dtype=np.float32)[:, None]
        privileged = np.zeros((1000, 4), dtype=np.float32)
        normalizer = FrozenNormalizer.fit(states, privileged, np.arange(1, 1001))
        value = torch.zeros(1, 1047)
        value[:, 1043:] = torch.tensor([[0.2, -0.3, 0.4, 0.5]])
        normalized = normalizer.state_tensor(value)
        torch.testing.assert_close(normalized[:, 1043:], value[:, 1043:])
        self.assertEqual(normalizer.policy_feature_mean.shape, (1043,))

    def test_utd_uses_new_valid_count_and_is_bounded(self) -> None:
        self.assertEqual(gradient_updates_for(1000, 2.0), 2000)
        self.assertEqual(gradient_updates_for(3, 1.5), 5)
        with self.assertRaises(ValueError):
            gradient_updates_for(10, 4.1)
        self.assertEqual(
            planned_gradient_updates(
                bootstrap=True, bootstrap_updates=777,
                new_trainable_count=1000, utd_ratio=2.0,
            ),
            777,
        )
        self.assertEqual(
            planned_gradient_updates(
                bootstrap=False, bootstrap_updates=777,
                new_trainable_count=3, utd_ratio=2.0,
            ),
            6,
        )

    def test_success_stratified_batch_guarantees_quota(self) -> None:
        success = np.zeros(1000, dtype=np.bool_)
        success[[7, 88]] = True
        indices = stratified_batch_indices(
            np.random.default_rng(9),
            success,
            batch_size=256,
            success_samples_per_batch=3,
        )
        self.assertEqual(indices.shape, (256,))
        self.assertEqual(int(success[indices].sum()), 3)

    def test_success_stratified_batch_falls_back_without_successes(self) -> None:
        success = np.zeros(1000, dtype=np.bool_)
        indices = stratified_batch_indices(
            np.random.default_rng(9),
            success,
            batch_size=256,
            success_samples_per_batch=1,
        )
        self.assertEqual(indices.shape, (256,))
        self.assertTrue(np.all((0 <= indices) & (indices < 1000)))


class RuntimeTests(unittest.TestCase):
    def test_random_warmup_is_bounded_and_preserves_gripper(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base.pt"
            base.write_bytes(b"base")
            runtime = RealRLPolicyRuntime(
                RealRLDeploySettings(config, "warmup_random", seed=5),
                base_model_path=base,
                base_kind="tacex_rma_gelsight_size_buckets_student_torchscript",
                device=torch.device("cpu"),
            )
            base_action = torch.tensor([[0.2, -0.1, 0.3, 0.75]])
            final, info = runtime.apply(torch.zeros(1, 1043), base_action)
            residual_m = np.asarray(info["residual_action_m"])
            self.assertTrue(np.all(np.abs(residual_m) <= 0.00100001))
            self.assertLessEqual(np.max(np.abs(final[0, :3].numpy() - base_action[0, :3].numpy())), 0.02 + 1e-6)
            self.assertEqual(float(final[0, 3]), 0.75)
            _final2, info2 = runtime.apply(torch.zeros(1, 1043), base_action)
            rng = np.random.default_rng(5)
            first = np.clip(
                rng.normal(0.0, config.residual.warmup_std_m, 3),
                -config.residual.warmup_cap_m, config.residual.warmup_cap_m,
            )
            innovation = rng.normal(0.0, config.residual.warmup_std_m, 3)
            rho = config.residual.warmup_correlation
            expected_second = np.clip(
                rho * first + np.sqrt(1.0 - rho * rho) * innovation,
                -config.residual.warmup_cap_m, config.residual.warmup_cap_m,
            )
            np.testing.assert_allclose(info["residual_action_m"], first, atol=1e-8)
            np.testing.assert_allclose(info2["residual_action_m"], expected_second, atol=1e-8)

    def test_missing_checkpoint_falls_back_to_zero(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base.pt"; base.write_bytes(b"base")
            runtime = RealRLPolicyRuntime(
                RealRLDeploySettings(config, "checkpoint", Path(temporary) / "missing.pt"),
                base_model_path=base,
                base_kind="tacex_rma_gelsight_size_buckets_student_torchscript",
                device=torch.device("cpu"),
            )
            base_action = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
            final, info = runtime.apply(torch.zeros(1, 1043), base_action)
            torch.testing.assert_close(final, base_action)
            self.assertTrue(info["fallback"])
            self.assertTrue(runtime.requires_episode_refusal)

            allowed = RealRLPolicyRuntime(
                RealRLDeploySettings(
                    config, "checkpoint", Path(temporary) / "missing.pt",
                    allow_checkpoint_fallback_collect=True,
                ),
                base_model_path=base,
                base_kind="tacex_rma_gelsight_size_buckets_student_torchscript",
                device=torch.device("cpu"),
            )
            self.assertFalse(allowed.requires_episode_refusal)


class _FakeTracker:
    def __init__(self, _camera, _config) -> None:
        self.pose = TagPose(1, 1.0, None, np.asarray([0.5, 0.0, 0.02]), 0.2)
        self.closed = False

    def wait_for_initial_object_z_in_base(self) -> float:
        return 0.02

    def latest(self):
        return self.pose

    def check(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeWriter:
    def __init__(self, _path, queue_size=0) -> None:
        self.items = []
        self.written = 0

    def check(self) -> None:
        return None

    def submit(self, value) -> None:
        self.items.append(value)
        self.written += 1

    def close(self) -> None:
        return None


class CollectorTests(unittest.TestCase):
    @staticmethod
    def result(step: int):
        return SimpleNamespace(
            inference_info={
                "real_rl": {
                    "state": np.full(1047, step, dtype=np.float32).tolist(),
                    "base_action": [0.0, 0.0, 0.0, 0.25],
                    "sac_unit_action": [0.1, 0.0, 0.0],
                    "residual_action_normalized": [0.004, 0.0, 0.0],
                    "residual_action_m": [0.0002, 0.0, 0.0],
                    "checkpoint_sha256": None,
                    "base_model_sha256": "base-sha",
                }
            },
            observation_metadata={"physical_tool_tcp_translation_m": [0.48, 0.0, 0.02]},
            raw_action=np.asarray([0.2, 0.0, 0.0, 0.25], dtype=np.float32),
            executed_action=np.zeros(4, dtype=np.float32),
        )

    def test_accepted_transition_waits_for_offline_labeling(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        writer = _FakeWriter(None)
        collector = RealRLCollector(
            RealRLDeploySettings(config, "zero"),
            object(),
            Path("/tmp/test-real-rl-run"),
            tracker_factory=_FakeTracker,
            writer_factory=lambda *_args, **_kwargs: writer,
        )
        observation = SimpleNamespace(metadata={})
        collector.observe_boundary(0, observation, self.result(0))
        collector.record_action(self.result(0), True, action_timestamp=1.05)
        collector.observe_boundary(1, observation, self.result(1))
        self.assertEqual(len(writer.items), 1)
        self.assertFalse(writer.items[0].reward_valid)
        self.assertFalse(writer.items[0].trainable)
        self.assertEqual(writer.items[0].trainable_reason, "offline_apriltag_pending")
        self.assertIsNone(writer.items[0].reward)
        self.assertTrue(collector.tracker.closed)

    def test_online_pose_changes_are_not_used_during_control(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        writer = _FakeWriter(None)
        collector = RealRLCollector(
            RealRLDeploySettings(config, "zero"), object(), Path("/tmp/test-real-rl-run"),
            tracker_factory=_FakeTracker,
            writer_factory=lambda *_args, **_kwargs: writer,
        )
        observation = SimpleNamespace(metadata={})
        collector.observe_boundary(0, observation, self.result(0))
        collector.record_action(self.result(0), True, action_timestamp=1.05)
        collector.tracker.pose = TagPose(
            2, 1.1, None, np.asarray([0.499, 0.0, 0.021]), 0.2
        )
        collector.observe_boundary(1, observation, self.result(1))
        sample = writer.items[0]
        self.assertFalse(sample.reward_valid)
        self.assertFalse(sample.trainable)
        self.assertEqual(sample.trainable_reason, "offline_apriltag_pending")
        self.assertIsNone(sample.reward)
        self.assertIsNone(sample.object_relative_to_ee_t)
        self.assertIsNone(sample.object_relative_to_ee_next)
        self.assertTrue(sample.safety_intervened)
        self.assertGreater(sample.intervention_magnitude, 0.0)

    def test_timestamp_validation_is_deferred_offline(self) -> None:
        config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        writer = _FakeWriter(None)
        collector = RealRLCollector(
            RealRLDeploySettings(config, "zero"), object(), Path("/tmp/test-real-rl-run"),
            tracker_factory=_FakeTracker,
            writer_factory=lambda *_args, **_kwargs: writer,
        )
        observation = SimpleNamespace(metadata={})
        collector.observe_boundary(0, observation, self.result(0))
        collector.record_action(self.result(0), True, action_timestamp=1.2)
        collector.tracker.pose = TagPose(
            2, 1.1, None, np.asarray([0.499, 0.0, 0.021]), 0.2
        )
        collector.observe_boundary(1, observation, self.result(1))
        self.assertFalse(writer.items[0].trainable)
        self.assertEqual(writer.items[0].trainable_reason, "offline_apriltag_pending")


class OfflineLabelTests(unittest.TestCase):
    def test_offline_boundary_frames_relabel_replay(self) -> None:
        base_config = load_real_rl_config(REPO_ROOT / "configs/real_residual_sac_0814.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "episode"
            raw_dir = run_dir / "raw_rgb"
            raw_dir.mkdir(parents=True)
            replay_path = root / "replay.sqlite3"
            config = replace(base_config, replay=replace(base_config.replay, path=replay_path))
            intrinsics = {
                "fx": 615.0, "fy": 615.0, "cx": 320.0, "cy": 240.0,
                "distortion_coefficients": [0.0] * 5,
                "distortion_model": "none",
            }
            records = []
            for step in range(2):
                path = raw_dir / f"boundary_{step:04d}.png"
                cv2.imwrite(str(path), np.zeros((32, 32, 3), dtype=np.uint8))
                model_input = {
                    "raw_rgb_path": str(path.relative_to(run_dir)),
                    "camera_frame": {
                        "sequence": step + 101,
                        "capture_timestamp": -100.0,
                        "camera_timestamp_ms": None,
                        "camera_intrinsics": intrinsics,
                    },
                    "offline_apriltag_boundary": {
                        "raw_rgb_path": str(path.relative_to(run_dir)),
                        "raw_rgb_shape": [32, 32, 3],
                        "camera_frame": {
                            "sequence": step + 1,
                            "capture_timestamp": float(step + 1),
                            "camera_timestamp_ms": None,
                            "camera_intrinsics": intrinsics,
                            "source": "real_policy_boundary_latest_packet",
                        },
                    },
                }
                if step == 1:
                    final = raw_dir / "boundary_0002.png"
                    cv2.imwrite(str(final), np.zeros((32, 32, 3), dtype=np.uint8))
                    model_input.update({
                        "next_raw_rgb_path": str(final.relative_to(run_dir)),
                        "next_boundary_camera_frame": {
                            "sequence": 3,
                            "capture_timestamp": 3.0,
                            "camera_timestamp_ms": None,
                            "camera_intrinsics": intrinsics,
                        },
                    })
                records.append({"step_index": step, "model_input": model_input})
            (run_dir / "rollout.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            first = transition(0, valid=False)
            first.trainable_reason = "offline_apriltag_pending"
            first.action_timestamp = 1.5
            second = transition(1, valid=False)
            second.trainable_reason = "offline_apriltag_pending"
            second.action_timestamp = 2.5
            second.truncated = True
            second.done = True
            for item in (first, second):
                item.episode_id = run_dir.name
                item.run_dir = str(run_dir)
            with ReplayBuffer(replay_path) as replay:
                replay.append(first)
                replay.append(second)

            class FakeDetector:
                def __init__(self, _config, _intrinsics) -> None:
                    pass

                def detect(self, boundary, _image, metadata):
                    return OfflineTagPose(
                        boundary=boundary,
                        sequence=int(metadata["sequence"]),
                        capture_timestamp=float(metadata["capture_timestamp"]),
                        camera_timestamp_ms=None,
                        object_position_base_m=np.asarray(
                            [0.5 - 0.001 * boundary, 0.0, 0.02 + 0.001 * boundary]
                        ),
                        reprojection_rms_px=0.1,
                        corners_px=np.zeros((4, 2)),
                        search_mode="full_frame",
                    )

            with mock.patch(
                "franka_sim2real.real_rl.offline_apriltag.OfflineAprilTagDetector",
                FakeDetector,
            ):
                report = label_episode_from_run(run_dir, config)
            self.assertEqual(report["boundary_frames"], 3)
            self.assertEqual(report["trainable_transitions"], 2)
            self.assertEqual(report["changed_transitions"], 2)
            self.assertEqual(report["label_revision"], 1)
            with ReplayBuffer(replay_path, create=False) as replay:
                self.assertEqual(replay.trainable_count(), 2)
                data = replay.load_trainable()
                self.assertEqual(data["reward"].shape, (2,))
                self.assertTrue(data["truncated"][1])
                np.testing.assert_array_equal(data["label_revision"], [1, 1])
                self.assertEqual(
                    replay.trainable_change_counts(
                        after_id=2, after_label_revision=0
                    )["relabeled"],
                    2,
                )
            second_report = label_episode_from_run(
                run_dir, config, reuse_saved_detections=True
            )
            self.assertTrue(second_report["reused_saved_detections"])
            self.assertEqual(second_report["changed_transitions"], 0)
            self.assertEqual(second_report["label_revision"], 1)


class CLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = REPO_ROOT / "scripts/real_rl/run_residual_sac.py"
        spec = importlib.util.spec_from_file_location("real_rl_cli_for_test", path)
        assert spec is not None and spec.loader is not None
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)

    def test_collect_requires_one_explicit_action_source(self) -> None:
        parser = self.cli.build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["collect"])

    def test_real_motion_requires_explicit_authorization(self) -> None:
        args = self.cli.build_parser().parse_args(["collect", "--zero-residual"])
        with self.assertRaisesRegex(ValueError, "enable-real-rl-control"):
            self.cli.collect_command(args)

    def test_checkpoint_fallback_opt_in_requires_checkpoint_mode(self) -> None:
        args = self.cli.build_parser().parse_args([
            "collect", "--zero-residual", "--allow-checkpoint-fallback-collect"
        ])
        config = load_real_rl_config(args.config)
        with self.assertRaisesRegex(ValueError, "only valid with checkpoint"):
            self.cli._settings(args, config)

    def test_collect_uses_compact_artifacts_by_default_with_debug_opt_in(self) -> None:
        parser = self.cli.build_parser()
        compact = parser.parse_args(["collect", "--zero-residual"])
        debug = parser.parse_args([
            "collect", "--zero-residual", "--save-debug-artifacts",
            "--defer-offline-label",
        ])
        self.assertFalse(compact.save_debug_artifacts)
        self.assertFalse(compact.defer_offline_label)
        self.assertTrue(debug.save_debug_artifacts)
        self.assertTrue(debug.defer_offline_label)

    def test_label_can_reuse_existing_detections(self) -> None:
        args = self.cli.build_parser().parse_args([
            "label", "--run-dir", "/tmp/run", "--reuse-existing-detections"
        ])
        self.assertTrue(args.reuse_existing_detections)


if __name__ == "__main__":
    unittest.main()
