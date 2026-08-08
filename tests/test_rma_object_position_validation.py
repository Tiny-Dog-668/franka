from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "policy" / "validate_rma_object_position.py"
SPEC = importlib.util.spec_from_file_location("validate_rma_object_position", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


class RMAObjectPositionValidationTests(unittest.TestCase):
    def test_current_portable_model_exposes_finite_position_predictions(self) -> None:
        predictor = validation.RMAObjectPositionPredictor(
            validation.DEFAULT_MODEL,
            validation.DEFAULT_MODEL.with_suffix(".json"),
            "cpu",
        )
        images = np.stack(
            [
                np.zeros((predictor.height, predictor.width, 3), dtype=np.uint8),
                np.full((predictor.height, predictor.width, 3), 255, dtype=np.uint8),
            ]
        )

        positions, contacts = predictor.predict(images, batch_size=1)

        self.assertEqual(positions.shape, (2, 3))
        self.assertEqual(contacts.shape, (2, 2))
        self.assertTrue(np.isfinite(positions).all())
        self.assertTrue(np.isfinite(contacts).all())
        self.assertTrue(np.all((contacts >= 0.0) & (contacts <= 1.0)))
        lower = predictor.position_center - predictor.position_scale
        upper = predictor.position_center + predictor.position_scale
        self.assertTrue(np.all(positions >= lower - 1e-6))
        self.assertTrue(np.all(positions <= upper + 1e-6))

    def test_metrics_use_3d_rmse_for_verdict(self) -> None:
        samples = [
            validation.Sample(
                "a",
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.asarray([0.50, 0.00, 0.026], dtype=np.float32),
            ),
            validation.Sample(
                "b",
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.asarray([0.50, 0.00, 0.026], dtype=np.float32),
            ),
        ]
        positions = np.asarray(
            [[0.503, 0.004, 0.026], [0.497, -0.004, 0.026]], dtype=np.float32
        )
        contacts = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        rows, summary = validation._build_results(
            samples, positions, contacts, threshold_m=0.006
        )

        self.assertEqual(summary["verdict"], "PASS")
        self.assertAlmostEqual(summary["rmse_3d_mm"], 5.0, places=4)
        self.assertEqual(summary["within_threshold_fraction"], 1.0)
        self.assertTrue(all(row["within_threshold"] for row in rows))

    def test_missing_ground_truth_is_not_reported_as_pass(self) -> None:
        samples = [validation.Sample("a", np.zeros((2, 2, 3), dtype=np.uint8))]

        _, summary = validation._build_results(
            samples,
            np.asarray([[0.5, 0.0, 0.026]], dtype=np.float32),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            threshold_m=0.010,
        )

        self.assertEqual(summary["verdict"], "NO_GROUND_TRUTH")


if __name__ == "__main__":
    unittest.main()
