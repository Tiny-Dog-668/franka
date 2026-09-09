from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibration" / "live_apriltag_drawer_pose.py"
SPEC = importlib.util.spec_from_file_location("live_apriltag_drawer_pose", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
drawer_pose = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = drawer_pose
SPEC.loader.exec_module(drawer_pose)


class LiveAprilTagDrawerPoseTests(unittest.TestCase):
    def test_default_drawer_points_follow_tag_axes(self) -> None:
        points = drawer_pose.drawer_points_in_tag(
            marker_paper_size_m=0.060,
            drawer_width_m=0.300,
            drawer_length_m=0.250,
            drawer_height_m=0.130,
        )

        np.testing.assert_allclose(points["top_left"], [-0.030, 0.030, 0.0])
        np.testing.assert_allclose(points["top_center"], [0.095, -0.120, 0.0])
        np.testing.assert_allclose(points["bottom_edge_center"], [0.095, -0.270, 0.0])
        np.testing.assert_allclose(points["volume_center"], [0.095, -0.120, -0.065])
        np.testing.assert_allclose(points["bottom_surface_center"], [0.095, -0.120, -0.130])

    def test_drawer_rotation_rotates_planar_offsets(self) -> None:
        points = drawer_pose.drawer_points_in_tag(
            marker_paper_size_m=0.060,
            drawer_width_m=0.300,
            drawer_length_m=0.250,
            drawer_height_m=0.130,
            drawer_rotation_deg=90.0,
        )

        np.testing.assert_allclose(points["top_center"], [0.120, 0.095, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            points["bottom_surface_center"], [0.120, 0.095, -0.130], atol=1e-12
        )

    def test_transform_point_uses_full_rigid_transform(self) -> None:
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        transform[:3, 3] = [0.4, -0.2, 0.1]

        actual = drawer_pose.transform_point(transform, np.asarray([0.1, 0.2, -0.3]))

        np.testing.assert_allclose(actual, [0.2, -0.1, -0.2], atol=1e-12)

    def test_image_orientation_selects_right_and_down_directions(self) -> None:
        camera_t_tag = np.eye(4)
        # A front-facing printed tag has +X right, +Y up, and +Z toward camera.
        camera_t_tag[:3, :3] = np.diag([1.0, -1.0, -1.0])
        camera_t_tag[:3, 3] = [0.0, 0.0, 1.0]
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )

        rotation = drawer_pose.image_aligned_drawer_rotation_deg(
            camera_t_tag,
            camera_matrix,
            np.zeros(5),
        )

        self.assertEqual(rotation, 0.0)

    def test_image_orientation_compensates_for_rotated_printed_tag(self) -> None:
        camera_t_tag = np.eye(4)
        # Rotate the canonical tag axes 90 degrees in the image.
        camera_t_tag[:3, :3] = np.asarray(
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]
        )
        camera_t_tag[:3, 3] = [0.0, 0.0, 1.0]
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )

        rotation = drawer_pose.image_aligned_drawer_rotation_deg(
            camera_t_tag,
            camera_matrix,
            np.zeros(5),
        )

        self.assertEqual(rotation, 90.0)

    def test_image_orientation_uses_local_axes_for_oblique_table_tag(self) -> None:
        camera_t_tag = np.eye(4)
        tag_x_camera = np.asarray([-0.2, 0.8, 0.565685424949238])
        tag_y_camera = np.asarray([0.970142500145332, 0.242535625036333, 0.0])
        tag_z_camera = np.cross(tag_x_camera, tag_y_camera)
        camera_t_tag[:3, :3] = np.column_stack(
            [tag_x_camera, tag_y_camera, tag_z_camera]
        )
        camera_t_tag[:3, 3] = [-0.30, 0.25, 0.70]
        camera_matrix = np.asarray(
            [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
        )

        rotation = drawer_pose.image_aligned_drawer_rotation_deg(
            camera_t_tag,
            camera_matrix,
            np.zeros(5),
        )

        self.assertEqual(rotation, 90.0)

    def test_non_positive_dimension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "drawer_height_m"):
            drawer_pose.drawer_points_in_tag(
                marker_paper_size_m=0.060,
                drawer_width_m=0.300,
                drawer_length_m=0.250,
                drawer_height_m=0.0,
            )


if __name__ == "__main__":
    unittest.main()
