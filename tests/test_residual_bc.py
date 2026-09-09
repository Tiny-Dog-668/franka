from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from franka_sim2real.residual_bc import (
    GROUP_HOLD,
    GROUP_MOVING,
    GROUP_NORMAL,
    FeatureMatrix,
    ResidualMLP,
    ResidualMLPConfig,
    balanced_sample_weights,
    discover_run_dirs,
    extract_feature_matrix,
    regression_metrics,
    scan_accepted_samples,
    split_episodes,
)


def _write_step(
    path: Path,
    *,
    accepted: bool,
    intervention: bool,
    base_action: tuple[float, ...] = (0.2, -0.1, 0.3, 0.4),
    human_action: tuple[float, ...] | None = None,
) -> None:
    base = np.asarray(base_action, dtype=np.float32)
    target = (
        np.asarray(human_action[:3], dtype=np.float32) - base[:3]
        if intervention and human_action is not None
        else np.zeros(3, dtype=np.float32)
    )
    values = {
        "wrist_rgb": np.zeros((4, 5, 3), dtype=np.uint8),
        "proprio_obs": np.zeros(15, dtype=np.float32),
        "action_history": np.zeros(4, dtype=np.float32),
        "gsmini_left_rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "gsmini_right_rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "gsmini_left_reference_rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "gsmini_right_reference_rgb": np.zeros((3, 4, 3), dtype=np.uint8),
        "base_action": base,
        "residual_target_xyz": target,
        "intervention": np.asarray(intervention),
        "policy_action_accepted": np.asarray(accepted),
        "episode_id": np.asarray(path.parent.parent.name),
        "step_id": np.asarray(int(path.stem.rsplit("_", 1)[-1]), dtype=np.int64),
    }
    if human_action is not None:
        values["human_action"] = np.asarray(human_action, dtype=np.float32)
    np.savez_compressed(path, **values)


def _run(root: Path, name: str) -> Path:
    path = root / name
    (path / "step_data").mkdir(parents=True)
    return path


class ResidualDatasetTests(unittest.TestCase):
    def test_scan_filters_misses_and_classifies_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run(Path(temporary), "episode_a")
            _write_step(
                run / "step_data/step_0000.npz",
                accepted=True,
                intervention=False,
            )
            _write_step(
                run / "step_data/step_0001.npz",
                accepted=True,
                intervention=True,
                human_action=(0.0, 0.0, 0.0, 0.4),
            )
            _write_step(
                run / "step_data/step_0002.npz",
                accepted=True,
                intervention=True,
                human_action=(0.1, 0.0, 0.0, 0.4),
            )
            _write_step(
                run / "step_data/step_0003.npz",
                accepted=False,
                intervention=True,
                human_action=(0.1, 0.0, 0.0, 0.4),
            )
            samples = scan_accepted_samples((run,))
            self.assertEqual(len(samples), 3)
            self.assertEqual(
                [sample.group for sample in samples],
                [GROUP_NORMAL, GROUP_HOLD, GROUP_MOVING],
            )

    def test_scan_rejects_incorrect_residual_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _run(Path(temporary), "episode_a")
            path = run / "step_data/step_0000.npz"
            _write_step(
                path,
                accepted=True,
                intervention=True,
                human_action=(0.1, 0.0, 0.0, 0.4),
            )
            with np.load(path, allow_pickle=False) as payload:
                values = {key: payload[key] for key in payload.files}
            values["residual_target_xyz"] = np.ones(3, dtype=np.float32)
            np.savez_compressed(path, **values)
            with self.assertRaisesRegex(ValueError, "does not match human-base"):
                scan_accepted_samples((run,))

    def test_discovery_skips_empty_run_and_auto_split_is_episode_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ["episode_a", "episode_b", "episode_c", "episode_d"]
            for name in names:
                run = _run(root, name)
                _write_step(
                    run / "step_data/step_0000.npz",
                    accepted=True,
                    intervention=False,
                )
            _run(root, "episode_empty")
            runs = discover_run_dirs(root, "episode_*")
            self.assertEqual([path.name for path in runs], names)
            split = split_episodes(runs)
            self.assertEqual([path.name for path in split.train], names[:2])
            self.assertEqual([path.name for path in split.validation], [names[2]])
            self.assertEqual([path.name for path in split.test], [names[3]])

    def test_explicit_split_rejects_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = tuple(_run(root, name) for name in ("a", "b", "c"))
            with self.assertRaisesRegex(ValueError, "both validation and test"):
                split_episodes(runs, validation_runs=("b",), test_runs=("b",))


class ResidualModelTests(unittest.TestCase):
    def test_mlp_output_shape_and_bound(self) -> None:
        config = ResidualMLPConfig(
            actor_feature_dim=5,
            base_action_dim=4,
            hidden_dims=(8, 4),
            output_scale=1.2,
        )
        model = ResidualMLP(config)
        output = model(torch.randn(6, 5), torch.randn(6, 4))
        self.assertEqual(tuple(output.shape), (6, 3))
        self.assertLessEqual(float(torch.max(torch.abs(output)).detach()), 1.2)

    def test_balanced_weights_equalize_group_mass(self) -> None:
        groups = torch.tensor([GROUP_NORMAL] * 6 + [GROUP_HOLD] * 2 + [GROUP_MOVING])
        weights = balanced_sample_weights(groups)
        masses = [float(weights[groups == group].sum()) for group in range(3)]
        np.testing.assert_allclose(masses, [1.0, 1.0, 1.0])

    def test_metrics_are_reported_per_group(self) -> None:
        target = torch.zeros((3, 3))
        prediction = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        groups = torch.tensor([GROUP_NORMAL, GROUP_HOLD, GROUP_MOVING])
        metrics = regression_metrics(prediction, target, groups)
        self.assertEqual(metrics["overall"]["count"], 3)
        self.assertEqual(metrics["normal"]["count"], 1)
        self.assertEqual(metrics["intervention_hold"]["count"], 1)
        self.assertEqual(metrics["intervention_moving"]["count"], 1)
        self.assertAlmostEqual(metrics["normal_prediction_norm_mean"], 0.0)

    def test_feature_matrix_length(self) -> None:
        matrix = FeatureMatrix(
            features=torch.zeros((2, 5)),
            base_actions=torch.zeros((2, 4)),
            targets=torch.zeros((2, 3)),
            groups=torch.zeros(2, dtype=torch.long),
            episode_ids=("a", "a"),
            step_ids=(0, 1),
            max_base_action_error=0.0,
        )
        self.assertEqual(len(matrix), 2)

    def test_feature_extraction_uses_deployment_batch_size_one(self) -> None:
        class BatchOneExtractor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.batch_sizes: list[int] = []

            def forward(self, *inputs):
                batch_size = int(inputs[0].shape[0])
                self.batch_sizes.append(batch_size)
                return torch.zeros((batch_size, 5)), torch.zeros((batch_size, 4))

        with tempfile.TemporaryDirectory() as temporary:
            run = _run(Path(temporary), "episode_a")
            for step in range(3):
                _write_step(
                    run / f"step_data/step_{step:04d}.npz",
                    accepted=True,
                    intervention=False,
                    base_action=(0.0, 0.0, 0.0, 0.0),
                )
            samples = scan_accepted_samples((run,))
            extractor = BatchOneExtractor()
            matrix = extract_feature_matrix(
                samples,
                extractor,  # type: ignore[arg-type]
                device=torch.device("cpu"),
                batch_size=3,
            )
            self.assertEqual(extractor.batch_sizes, [1, 1, 1])
            self.assertEqual(tuple(matrix.features.shape), (3, 5))


if __name__ == "__main__":
    unittest.main()
