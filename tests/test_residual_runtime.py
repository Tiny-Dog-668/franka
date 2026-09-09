from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from franka_sim2real.e2e_bundle import BundleDeployConfig
from franka_sim2real.residual_runtime import (
    ResidualDeploySettings,
    ResidualPolicyRuntime,
)
from franka_sim2real.streaming import (
    _PolicyResult,
    _policy_record,
    _write_streaming_artifacts,
)
from franka_sim2real.types import RobotAction, RobotObservation


class _Head(torch.nn.Module):
    def forward(self, actor_features, base_action):
        return actor_features[:, :3] + base_action[:, :3] * 0.0


def _artifacts(root: Path) -> tuple[Path, Path, Path]:
    base = root / "base.pt"
    base.write_bytes(b"exact base checkpoint")
    model = root / "residual.ts"
    traced = torch.jit.trace(_Head(), (torch.zeros(1, 1043), torch.zeros(1, 4)))
    torch.jit.save(traced, str(model))
    metadata = root / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "kind": "franka_hil_residual_bc",
                "format_version": 1,
                "input_contract": "actor_feature_1043_plus_base_action_4",
                "output_contract": "prelimit_normalized_residual_xyz_3",
                "gripper_source": "base_policy",
                "base_model_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "model_config": {
                    "actor_feature_dim": 1043,
                    "base_action_dim": 4,
                    "output_dim": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return base, model, metadata


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


class ResidualRuntimeTests(unittest.TestCase):
    def test_apply_caps_xyz_and_preserves_base_gripper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, model, metadata = _artifacts(Path(temporary))
            runtime = ResidualPolicyRuntime(
                ResidualDeploySettings(model, metadata, scale=0.5, max_abs=0.1),
                base_model_path=base,
                base_kind="tacex_rma_gelsight_size_buckets_student_torchscript",
                device=torch.device("cpu"),
            )
            features = torch.zeros((1, 1043))
            features[0, :3] = torch.tensor([0.5, -0.5, 0.1])
            base_action = torch.tensor([[0.2, 0.3, -0.4, 0.75]])
            final, predicted, applied = runtime.apply(features, base_action)
            torch.testing.assert_close(predicted, torch.tensor([[0.5, -0.5, 0.1]]))
            torch.testing.assert_close(applied, torch.tensor([[0.1, -0.1, 0.05]]))
            torch.testing.assert_close(final, torch.tensor([[0.3, 0.2, -0.35, 0.75]]))

    def test_wrong_base_sha_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, model, metadata = _artifacts(Path(temporary))
            base.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "different base policy"):
                ResidualPolicyRuntime(
                    ResidualDeploySettings(model, metadata),
                    base_model_path=base,
                    base_kind="tacex_rma_gelsight_size_buckets_student_torchscript",
                    device=torch.device("cpu"),
                )

    def test_settings_reject_unsafe_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base, model, metadata = _artifacts(Path(temporary))
            del base
            for scale, cap in ((0.0, 0.1), (1.1, 0.1), (1.0, 0.0), (1.0, 1.1)):
                with self.subTest(scale=scale, cap=cap):
                    with self.assertRaises(ValueError):
                        ResidualDeploySettings(model, metadata, scale, cap).validate()

    def test_policy_record_exposes_residual_fields(self) -> None:
        result = _PolicyResult(
            raw_action=np.asarray([0.3, 0.2, -0.35, 0.75], dtype=np.float32),
            executed_action=np.asarray([0.03, 0.02, -0.035, 0.075], dtype=np.float32),
            robot_action=RobotAction(),
            action_history=np.zeros(4, dtype=np.float32),
            proprio=np.zeros(15, dtype=np.float32),
            contact_force_n=None,
            image=np.zeros((4, 5, 3), dtype=np.uint8),
            model_rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            tactile_images={},
            tactile_references={},
            inference_info={
                "residual_bc": {
                    "base_action": [0.2, 0.3, -0.4, 0.75],
                    "predicted_residual_xyz": [0.5, -0.5, 0.1],
                    "applied_residual_xyz": [0.1, -0.1, 0.05],
                    "scale": 0.5,
                    "max_abs": 0.1,
                    "model_sha256": "abc",
                }
            },
            elapsed_ns=100,
        )
        record = _policy_record(0, _observation(), result, True, False)
        self.assertEqual(record["base_action"], [0.2, 0.3, -0.4, 0.75])
        self.assertEqual(record["predicted_residual_xyz"], [0.5, -0.5, 0.1])
        self.assertEqual(record["applied_residual_xyz"], [0.1, -0.1, 0.05])
        self.assertEqual(record["residual_model_sha256"], "abc")

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            _write_streaming_artifacts(
                run_dir,
                BundleDeployConfig(),
                {"passed": True},
                [record],
                [result.image],
                [],
                {},
                save_step_data=True,
            )
            logged = json.loads((run_dir / "rollout.jsonl").read_text())
            self.assertEqual(logged["predicted_residual_xyz"], [0.5, -0.5, 0.1])
            with np.load(run_dir / "step_data/step_0000.npz") as payload:
                self.assertIn("base_action", payload.files)
                self.assertIn("predicted_residual_xyz", payload.files)
                self.assertIn("applied_residual_xyz", payload.files)
                np.testing.assert_allclose(
                    payload["applied_residual_xyz"], [0.1, -0.1, 0.05]
                )


if __name__ == "__main__":
    unittest.main()
