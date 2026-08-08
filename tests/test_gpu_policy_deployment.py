from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from franka_sim2real import e2e_bundle
from franka_sim2real.e2e_bundle import (
    BundleTorchScriptPolicy,
    load_bundle_config,
    validate_bundle_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_CONFIG = REPO_ROOT / "configs/e2e_bundle_real_exported_0802_dr_simactuator_gpu.json"
ORACLE_GPU_CONFIG = (
    REPO_ROOT / "configs/e2e_bundle_real_exported_0802_dr_simactuator_oracle_gpu.json"
)


class GPU0802DeploymentTests(unittest.TestCase):
    def test_gpu_config_selects_portable_dr_artifact(self) -> None:
        config = load_bundle_config(GPU_CONFIG)

        self.assertEqual(config.model.device, "cuda:0")
        self.assertTrue(config.model.enforce_policy_contract)
        self.assertTrue(config.model.model_path.endswith("rma_student_dr_sim2real_gpu.pt"))
        self.assertTrue(config.model.metadata_path.endswith("rma_student_dr_sim2real_gpu.json"))
        self.assertEqual(config.streaming.control_law, "sim_actuator_velocity")
        self.assertEqual(config.streaming.commissioning_action_limit, 0.1)

    def test_portable_v6_artifact_passes_full_cpu_smoke_test(self) -> None:
        config = load_bundle_config(GPU_CONFIG)
        config.model.device = "cpu"

        report = validate_bundle_artifacts(config)

        self.assertEqual(report["output_signature"], {"mean_actions": [4]})
        self.assertEqual(len(report["smoke_test_output"]), 4)
        self.assertTrue(report["policy_contract_enforced"])

    def test_cuda_rejects_legacy_v5_artifact_before_contract_checks(self) -> None:
        config = load_bundle_config(GPU_CONFIG)
        bundle = MagicMock()
        bundle.metadata = {"version": 5}

        with self.assertRaisesRegex(ValueError, "CUDA inference requires portable metadata version 6"):
            e2e_bundle._validate_tacex_rma_student_contract(
                bundle,
                config,
                history_scale_vector=MagicMock(),
            )

    def test_oracle_config_uses_true_position_and_zero_contact(self) -> None:
        config = load_bundle_config(ORACLE_GPU_CONFIG)
        self.assertEqual(config.model.rma_position_source, "oracle")
        self.assertEqual(config.model.rma_contact_source, "zero")
        self.assertEqual(config.model.rma_oracle_cube_position_root, [0.5, 0.0, 0.026])

    def test_oracle_actor_bypasses_rgb_and_changes_with_cube_position(self) -> None:
        config = load_bundle_config(ORACLE_GPU_CONFIG)
        bundle = BundleTorchScriptPolicy(
            config.model.model_path,
            config.model.metadata_path,
            device="cpu",
            rma_position_source="oracle",
            rma_contact_source="zero",
            rma_oracle_cube_position_root=[0.45, -0.05, 0.026],
        )
        proprio = np.asarray(
            config.initial_state.joint_positions
            + [0.0] * 7
            + [config.initial_state.gripper_width_m],
            dtype=np.float32,
        )
        history = np.zeros(4, dtype=np.float32)
        black = np.zeros((224, 224, 3), dtype=np.uint8)
        noise = np.random.default_rng(7).integers(
            0, 256, size=(224, 224, 3), dtype=np.uint8
        )

        action_black = bundle.predict(history, proprio, black)
        action_noise = bundle.predict(history, proprio, noise)
        np.testing.assert_array_equal(action_black, action_noise)
        self.assertEqual(bundle.last_inference_info["contact_state"], [0.0, 0.0])
        self.assertTrue(bundle.last_inference_info["vision_bypassed"])

        bundle.rma_oracle_cube_position_root = (0.55, 0.05, 0.026)
        action_other_position = bundle.predict(history, proprio, black)
        self.assertFalse(np.allclose(action_black, action_other_position, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
