from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibration" / "live_apriltag_cube_pose.py"
SPEC = importlib.util.spec_from_file_location("live_apriltag_cube_pose", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
live_pose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = live_pose
SPEC.loader.exec_module(live_pose)


class LiveAprilTagCubePoseTests(unittest.TestCase):
    class _FakeOptionRange:
        min = 0.0
        max = 1000.0

    class _FakeColorSensor:
        def __init__(self) -> None:
            self.values = {"auto": 1.0, "exposure": 120.0, "gain": 16.0}

        def supports(self, option: str) -> bool:
            return option in self.values

        def get_option_range(self, option: str):
            return LiveAprilTagCubePoseTests._FakeOptionRange()

        def set_option(self, option: str, value: float) -> None:
            self.values[option] = value

        def get_option(self, option: str) -> float:
            return self.values[option]

    class _FakeRS:
        class option:
            enable_auto_exposure = "auto"
            exposure = "exposure"
            gain = "gain"

    def test_square_pnp_recovers_synthetic_pose(self) -> None:
        camera_matrix = np.asarray(
            [[604.8974, 0.0, 320.9801], [0.0, 605.0858, 247.9132], [0.0, 0.0, 1.0]]
        )
        distortion = np.zeros(5)
        object_points = live_pose._tag_object_points(0.040)
        expected_rotation = np.diag([1.0, -1.0, -1.0])
        expected_rvec, _ = cv2.Rodrigues(expected_rotation)
        expected_translation = np.asarray([[0.02], [-0.01], [0.8]])
        corners, _ = cv2.projectPoints(
            object_points,
            expected_rvec,
            expected_translation,
            camera_matrix,
            distortion,
        )

        camera_t_tag, _, reprojection = live_pose.solve_tag_pose(
            corners, camera_matrix, distortion, 0.040
        )

        np.testing.assert_allclose(camera_t_tag[:3, :3], expected_rotation, atol=1e-8)
        np.testing.assert_allclose(
            camera_t_tag[:3, 3], expected_translation.reshape(3), atol=1e-8
        )
        self.assertLess(reprojection, 1e-8)

    def test_centered_cube_center_is_25_mm_behind_tag_face(self) -> None:
        base_t_camera = np.eye(4)
        camera_t_tag = np.eye(4)
        camera_t_tag[:3, 3] = [0.1, 0.2, 0.8]

        base_t_object = live_pose.object_pose_in_base(
            base_t_camera,
            camera_t_tag,
            np.asarray([0.0, 0.0, -0.025]),
        )

        np.testing.assert_allclose(base_t_object[:3, 3], [0.1, 0.2, 0.775], atol=1e-12)

    def test_default_calibrated_extrinsic_is_valid_rigid_transform(self) -> None:
        transform = live_pose._load_accepted_base_t_camera(live_pose.DEFAULT_CALIBRATION_REPORT)

        np.testing.assert_allclose(
            transform[:3, 3],
            [1.1660910884070719, 0.03590160819685967, 0.5142003358975679],
            atol=1e-12,
        )
        np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(transform[:3, :3])), 1.0, places=10)

    def test_manual_color_controls_disable_auto_exposure(self) -> None:
        sensor = self._FakeColorSensor()
        controls = live_pose._configure_color_controls(
            self._FakeRS,
            sensor,
            auto_exposure=False,
            exposure=80.0,
            gain=64.0,
        )

        self.assertEqual(
            controls,
            {"auto_exposure": False, "exposure": 80.0, "gain": 64.0},
        )

    def test_auto_exposure_rejects_manual_controls(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            live_pose._validate_color_control_args(True, 80.0, None)


if __name__ == "__main__":
    unittest.main()
