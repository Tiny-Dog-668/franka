from __future__ import annotations

import importlib.util
import sys
import unittest
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibration" / "sweep_apriltag_exposure.py"
SPEC = importlib.util.spec_from_file_location("sweep_apriltag_exposure", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


class SweepAprilTagExposureTests(unittest.TestCase):
    def test_result_converts_errors_and_jitter_to_millimetres(self) -> None:
        summary = {
            "frame_count": 100,
            "accepted_detection_count": 75,
            "rejected_reprojection_count": 2,
            "camera_color_controls": {"exposure": 100.0, "gain": 64.0},
            "policy_prediction": {
                "std_m": [0.001, 0.002, 0.002],
                "vs_calibrated_error_3d_rmse_m": 0.012,
                "vs_calibrated_error_3d_median_m": 0.010,
                "vs_calibrated_error_3d_p95_m": 0.020,
            },
            "calibrated": {"std_m": [0.003, 0.004, 0.0]},
        }

        result = sweep._result_from_summary(100.0, 1, 0, Path("trial"), summary)

        self.assertEqual(result["detection_rate"], 0.75)
        self.assertEqual(result["policy_vs_tag_rmse_mm"], 12.0)
        self.assertAlmostEqual(result["policy_jitter_norm_mm"], 3.0)
        self.assertAlmostEqual(result["tag_jitter_norm_mm"], 5.0)

    def test_best_result_rejects_low_detection_rate(self) -> None:
        results = [
            {
                "requested_exposure": 40.0,
                "detection_rate": 0.2,
                "policy_vs_tag_rmse_mm": 1.0,
            },
            {
                "requested_exposure": 100.0,
                "detection_rate": 0.9,
                "policy_vs_tag_rmse_mm": 5.0,
            },
        ]

        metric, best = sweep._best_result(results, 0.5, False)

        self.assertEqual(metric, "policy_vs_tag_rmse_mm")
        self.assertIsNotNone(best)
        self.assertEqual(best["requested_exposure"], 100.0)

    def test_trial_command_fixes_gain_and_uses_id_two(self) -> None:
        args = Namespace(
            camera_config="camera.json",
            calibration_report="calibration.json",
            policy_config="policy.json",
            policy_model="model.pt",
            policy_metadata="model.json",
            policy_device="cuda:0",
            family="tag36h11",
            marker_id=2,
            marker_length_m=0.038,
            base_height_offset_m=0.020,
            detection_scale=4.0,
            max_reprojection_px=2.0,
            frames_per_exposure=120,
            gain=64.0,
            tag_to_object=(0.0, 0.0, -0.025),
            ground_truth=None,
            show_window=False,
        )

        command = sweep._build_trial_command(args, 100.0, Path("trial"))

        self.assertEqual(command[command.index("--id") + 1], "2")
        self.assertEqual(command[command.index("--exposure") + 1], "100")
        self.assertEqual(command[command.index("--gain") + 1], "64")
        self.assertIn("--headless", command)


if __name__ == "__main__":
    unittest.main()
