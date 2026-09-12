from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from franka_sim2real.e2e_bundle import load_bundle_config, validate_bundle_artifacts
from real_rlpd.direct_bc import (
    ACTION_LIMIT,
    FEATURE_DIM,
    DirectBCArtifact,
    DirectBCConfig,
    DirectBCData,
    DirectBCHead,
    FeatureNormalizer,
    load_direct_bc_data,
    select_episodes,
    sha256_file,
    split_episodes,
)


class DirectBCTests(unittest.TestCase):
    REPO_ROOT = Path(__file__).resolve().parents[1]

    def test_head_and_artifact_are_four_dimensional_and_bounded(self) -> None:
        head = DirectBCHead()
        normalizer = FeatureNormalizer(
            mean=np.zeros(FEATURE_DIM, dtype=np.float32),
            std=np.ones(FEATURE_DIM, dtype=np.float32),
            clip=10.0,
        )
        artifact = DirectBCArtifact(head, normalizer)
        output = artifact(torch.randn(8, FEATURE_DIM))
        self.assertEqual(tuple(output.shape), (8, 4))
        self.assertTrue(torch.all(torch.abs(output) <= ACTION_LIMIT))
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_episode_split_has_no_leakage_and_keeps_success_in_holdouts(self) -> None:
        episode_ids = tuple(
            name
            for name in ("s0", "s1", "s2", "f0", "f1", "f2")
            for _ in range(2)
        )
        success = np.asarray([
            name.startswith("s") for name in episode_ids
        ], dtype=bool)
        data = DirectBCData(
            features=np.zeros((len(episode_ids), FEATURE_DIM), dtype=np.float32),
            actions=np.zeros((len(episode_ids), 4), dtype=np.float32),
            episode_ids=episode_ids,
            success=success,
        )
        split = split_episodes(data, seed=12)
        self.assertFalse(set(split.train) & set(split.validation))
        self.assertFalse(set(split.train) & set(split.test))
        self.assertFalse(set(split.validation) & set(split.test))
        self.assertTrue(any(name.startswith("s") for name in split.validation))
        self.assertTrue(any(name.startswith("s") for name in split.test))
        self.assertEqual(len(select_episodes(data, split.validation)), 4)

    def test_replay_loader_checks_contract_and_uses_executed_action(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "base.pt"
            model.write_bytes(b"base")
            replay = root / "replay.sqlite3"
            connection = sqlite3.connect(replay)
            connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            contract = {
                "adapter_id": "gelsight_reference_progress_three_frame_v1",
                "policy_kind": "tacex_rma_gelsight_x040_dr_three_frame_student_torchscript",
                "feature_dim": FEATURE_DIM,
                "action_dim": 4,
                "model_sha256": sha256_file(model),
            }
            connection.execute(
                "INSERT INTO meta(key,value) VALUES('contract',?)",
                (json.dumps(contract),),
            )
            connection.execute(
                """CREATE TABLE transitions(
                id INTEGER PRIMARY KEY,episode_id TEXT,state BLOB,
                executed_action BLOB,success INTEGER,trainable INTEGER)"""
            )
            state = np.arange(FEATURE_DIM + 4, dtype="<f4")
            action = np.asarray([0.01, -0.02, 0.03, -0.04], dtype="<f4")
            connection.execute(
                "INSERT INTO transitions VALUES(1,'episode',?,?,0,1)",
                (state.tobytes(), action.tobytes()),
            )
            connection.commit()
            connection.close()

            data, loaded_contract = load_direct_bc_data(
                replay, base_model_path=model
            )
            self.assertEqual(loaded_contract, contract)
            self.assertEqual(data.features.shape, (1, FEATURE_DIM))
            np.testing.assert_array_equal(data.features[0], state[:FEATURE_DIM])
            np.testing.assert_allclose(data.actions[0], action)

    def test_config_rejects_a_larger_direct_action_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_limit"):
            DirectBCConfig(action_limit=0.2).validate()

    def test_exported_0912_direct_bc_passes_contract_and_is_bounded(self) -> None:
        config = load_bundle_config(
            self.REPO_ROOT / "configs/e2e_bundle_real_exported_0912_direct_bc.json"
        )
        report = validate_bundle_artifacts(config)
        self.assertEqual(report["deployment_variant"], "frozen_encoder_direct_bc")
        self.assertLessEqual(max(abs(value) for value in report["smoke_test_output"]), 0.1)

        config.streaming.commissioning_action_limit = 0.09
        with self.assertRaisesRegex(ValueError, "commissioning_action_limit=0.1"):
            validate_bundle_artifacts(config)


if __name__ == "__main__":
    unittest.main()
