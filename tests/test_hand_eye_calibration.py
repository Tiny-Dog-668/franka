from __future__ import annotations

import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from franka_sim2real.calibration.hand_eye import (
    EyeToHandConfig,
    build_trajectory,
    checkerboard_corner_symmetries,
    checkerboard_object_points,
    estimate_target_to_camera,
    hub_transform,
    interpolate_transforms,
    invert_transform,
    load_eye_to_hand_config,
    make_transform,
    resolve_checkerboard_symmetry,
    rotation_to_quaternion_xyzw,
    rpy_degrees_to_rotation,
    solve_eye_to_hand,
    target_transform,
    transform_error,
    transform_from_pose,
    validate_target_transform,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "eye_to_hand_d435_215322076207.json"


def transform(rpy_deg, translation) -> np.ndarray:
    return make_transform(rpy_degrees_to_rotation(rpy_deg), translation)


def synthetic_samples(config):
    center = transform([170.0, 3.0, 8.0], [0.50, 0.0, 0.285])
    base_to_camera = transform([8.0, -12.0, 20.0], [0.82, -0.10, 0.72])
    gripper_to_target = transform([4.0, 3.0, -5.0], [0.02, -0.04, 0.18])

    def make_sample(pose):
        base_to_gripper = target_transform(center, pose)
        target_to_camera = (
            invert_transform(base_to_camera)
            @ base_to_gripper
            @ gripper_to_target
        )
        return {
            "sample_id": pose.sample_id,
            "phase": pose.phase,
            "base_to_gripper": base_to_gripper.tolist(),
            "target_to_camera": target_to_camera.tolist(),
            "reprojection_rms_px": 0.1,
        }

    calibration = [make_sample(pose) for pose in build_trajectory("calibration")]
    validation = [make_sample(pose) for pose in build_trajectory("validation")]
    return base_to_camera, calibration, validation


class ConfigAndTrajectoryTests(unittest.TestCase):
    def test_config_and_trajectory_contract(self) -> None:
        config = load_eye_to_hand_config(CONFIG_PATH)
        self.assertEqual(config.camera.serial, "215322076207")
        self.assertEqual(config.board.pattern_size, (11, 8))
        self.assertEqual(config.board.square_size_m, 0.015)
        calibration = build_trajectory("calibration")
        validation = build_trajectory("validation")
        self.assertEqual(len(calibration), 25)
        self.assertEqual(len(validation), 5)
        self.assertEqual(len({pose.sample_id for pose in [*calibration, *validation]}), 30)
        for pose in [*calibration, *validation]:
            dx, dy, dz = pose.translation_offset_base_m
            roll, pitch, yaw = pose.rpy_offset_tool_deg
            self.assertLessEqual(abs(dx), 0.05)
            self.assertLessEqual(abs(dy), 0.05)
            self.assertGreaterEqual(dz, 0.0)
            self.assertLessEqual(dz, 0.03)
            self.assertLessEqual(abs(roll), 25.0)
            self.assertLessEqual(abs(pitch), 25.0)
            self.assertLessEqual(abs(yaw), 15.0)

    def test_invalid_board_size_is_rejected(self) -> None:
        raw = load_eye_to_hand_config(CONFIG_PATH).to_dict()
        raw["board"]["square_size_m"] = 0.0
        with self.assertRaisesRegex(ValueError, "square_size_m"):
            EyeToHandConfig.from_dict(raw)

    def test_tool_envelope_and_workspace_are_checked(self) -> None:
        config = load_eye_to_hand_config(CONFIG_PATH)
        safe = transform([0, 0, 0], [0.5, 0.0, 0.285])
        unsafe = transform([0, 0, 0], [0.5, 0.0, 0.20])
        self.assertEqual(validate_target_transform(safe, config), [])
        failures = validate_target_transform(unsafe, config)
        self.assertTrue(any("tool envelope" in failure for failure in failures))

    def test_raised_hub_is_exactly_thirty_millimeters(self) -> None:
        center = transform([10, 20, 30], [0.5, 0.0, 0.28])
        raised = hub_transform(center, "raised")
        np.testing.assert_allclose(raised[:3, 3], [0.5, 0.0, 0.31])
        np.testing.assert_allclose(raised[:3, :3], center[:3, :3])


class TransformTests(unittest.TestCase):
    def test_transform_inverse_and_quaternion_round_trip(self) -> None:
        original = transform([34.0, -21.0, 78.0], [0.4, -0.2, 0.7])
        np.testing.assert_allclose(
            original @ invert_transform(original),
            np.eye(4),
            rtol=0.0,
            atol=1e-12,
        )
        quaternion = rotation_to_quaternion_xyzw(original[:3, :3])
        recovered = transform_from_pose(original[:3, 3], quaternion)
        np.testing.assert_allclose(recovered, original, rtol=0.0, atol=1e-12)

    def test_interpolation_respects_segment_limits(self) -> None:
        start = transform([0, 0, 0], [0.5, 0.0, 0.3])
        target = transform([0, 0, 12], [0.525, 0.0, 0.3])
        segments = interpolate_transforms(start, target, 0.01, 5.0)
        self.assertEqual(len(segments), 3)
        previous = start
        for segment in segments:
            translation_m, rotation_deg = transform_error(previous, segment)
            self.assertLessEqual(translation_m, 0.010000001)
            self.assertLessEqual(rotation_deg, 5.000001)
            previous = segment
        np.testing.assert_allclose(segments[-1], target)


class PnPTests(unittest.TestCase):
    def test_rectangular_board_has_four_shape_preserving_corner_orders(self) -> None:
        config = load_eye_to_hand_config(CONFIG_PATH)
        corners = np.arange(11 * 8 * 2, dtype=np.float64).reshape(-1, 2)
        symmetries = checkerboard_corner_symmetries(corners, config.board)
        self.assertEqual(
            [name for name, _ in symmetries],
            ["identity", "rotate_180", "flip_vertical", "flip_horizontal"],
        )
        self.assertTrue(all(candidate.shape == (88, 2) for _, candidate in symmetries))

    def test_planar_ippe_recovers_target_to_camera(self) -> None:
        config = load_eye_to_hand_config(CONFIG_PATH)
        camera_matrix = np.asarray(
            [[604.9, 0.0, 320.98], [0.0, 605.1, 247.91], [0.0, 0.0, 1.0]]
        )
        expected = transform([10.0, -15.0, 5.0], [-0.04, -0.05, 0.62])
        rvec, _ = cv2.Rodrigues(expected[:3, :3])
        corners, _ = cv2.projectPoints(
            checkerboard_object_points(config.board),
            rvec,
            expected[:3, 3],
            camera_matrix,
            np.zeros(5),
        )
        result = estimate_target_to_camera(
            corners.reshape(-1, 2),
            camera_matrix,
            np.zeros(5),
            config.board,
        )
        translation_m, rotation_deg = transform_error(
            expected, np.asarray(result["target_to_camera"])
        )
        self.assertLess(translation_m, 1e-7)
        self.assertLess(rotation_deg, 1e-5)
        self.assertLess(result["reprojection_rms_px"], 1e-5)


class EyeToHandSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_eye_to_hand_config(CONFIG_PATH)
        (
            self.expected_base_to_camera,
            self.calibration,
            self.validation,
        ) = synthetic_samples(self.config)

    def test_exact_eye_to_hand_direction_and_acceptance(self) -> None:
        result = solve_eye_to_hand(
            self.calibration, self.validation, self.config.quality
        )
        self.assertTrue(result["accepted"])
        self.assertGreaterEqual(result["checks"]["valid_method_count"], 3)
        actual = np.asarray(result["T_base_color"])
        translation_m, rotation_deg = transform_error(
            self.expected_base_to_camera, actual
        )
        self.assertLess(translation_m, 1e-8)
        self.assertLess(rotation_deg, 1e-6)
        sample = self.calibration[7]
        gripper_to_target = (
            invert_transform(np.asarray(sample["base_to_gripper"]))
            @ actual
            @ np.asarray(sample["target_to_camera"])
        )
        reference = np.asarray(
            result["diagnostic_selected_candidate"]["gripper_to_target_reference"]
        )
        translation_m, rotation_deg = transform_error(reference, gripper_to_target)
        self.assertLess(translation_m, 1e-8)
        self.assertLess(rotation_deg, 1e-6)

    def test_square_board_symmetry_is_resolved_by_robot_motion(self) -> None:
        symmetries = [
            transform([0.0, 0.0, yaw], [0.0, 0.0, 0.0])
            for yaw in (0.0, 90.0, 180.0, 270.0)
        ]
        ambiguous = []
        for sample_index, sample in enumerate(self.calibration):
            true_target_to_camera = np.asarray(
                sample["target_to_camera"], dtype=np.float64
            )
            order = np.roll(np.arange(len(symmetries)), sample_index % 4)
            candidates = []
            for candidate_index, symmetry_index in enumerate(order):
                candidates.append(
                    {
                        "target_to_camera": (
                            true_target_to_camera @ symmetries[symmetry_index]
                        ).tolist(),
                        "reprojection_rms_px": 0.1,
                        "symmetry": f"synthetic_{symmetry_index}",
                        "symmetry_index": candidate_index,
                    }
                )
            ambiguous.append(
                {
                    **sample,
                    "target_to_camera_candidates": candidates,
                }
            )

        resolved, diagnostics = resolve_checkerboard_symmetry(
            ambiguous, self.config.quality
        )
        self.assertEqual(len(resolved), 25)
        self.assertGreaterEqual(diagnostics["converged_seed_count"], 1)
        result = solve_eye_to_hand(resolved, self.validation, self.config.quality)
        actual = np.asarray(
            result["diagnostic_selected_candidate"]["base_to_camera"]
        )
        translation_m, rotation_deg = transform_error(
            self.expected_base_to_camera, actual
        )
        self.assertLess(translation_m, 1e-8)
        self.assertLess(rotation_deg, 1e-6)

    def test_validation_outlier_fails_closed(self) -> None:
        validation = copy.deepcopy(self.validation)
        validation[0]["target_to_camera"][0][3] += 0.02
        result = solve_eye_to_hand(
            self.calibration, validation, self.config.quality
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["validation_pass"])
        self.assertIsNone(result["T_base_color"])
        self.assertIsNotNone(result["diagnostic_selected_candidate"])

    def test_reprojection_failure_fails_closed(self) -> None:
        calibration = copy.deepcopy(self.calibration)
        calibration[0]["reprojection_rms_px"] = 1.2
        result = solve_eye_to_hand(
            calibration, self.validation, self.config.quality
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["reprojection_pass"])
        self.assertIn(
            calibration[0]["sample_id"], result["checks"]["reprojection_failures"]
        )
        self.assertIsNone(result["T_base_color"])

    def test_fewer_than_three_valid_methods_fails_closed(self) -> None:
        exact_return = (
            self.expected_base_to_camera[:3, :3],
            self.expected_base_to_camera[:3, 3].reshape(3, 1),
        )
        side_effect = [
            exact_return,
            exact_return,
            RuntimeError("synthetic solver failure"),
            RuntimeError("synthetic solver failure"),
            RuntimeError("synthetic solver failure"),
        ]
        with patch("cv2.calibrateHandEye", side_effect=side_effect):
            result = solve_eye_to_hand(
                self.calibration, self.validation, self.config.quality
            )
        self.assertFalse(result["accepted"])
        self.assertEqual(result["checks"]["valid_method_count"], 2)
        self.assertFalse(result["checks"]["method_consensus_pass"])
        self.assertIsNone(result["T_base_color"])

    def test_missing_validation_samples_fails_closed_but_keeps_candidate(self) -> None:
        result = solve_eye_to_hand(self.calibration, [], self.config.quality)
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["sample_count_pass"])
        self.assertIsNone(result["T_base_color"])
        self.assertIsNotNone(result["diagnostic_selected_candidate"])

    def test_available_mode_emits_provisional_transform(self) -> None:
        calibration = self.calibration[:12]
        result = solve_eye_to_hand(
            calibration,
            [],
            self.config.quality,
            use_available_samples=True,
        )
        self.assertTrue(result["accepted"])
        self.assertTrue(result["provisional"])
        self.assertFalse(result["deployment_ready"])
        self.assertEqual(result["fit_mode"], "available")
        self.assertTrue(result["checks"]["sample_count_pass"])
        self.assertFalse(result["checks"]["dataset_complete"])
        self.assertEqual(result["checks"]["calibration_sample_count"], 12)
        self.assertEqual(result["checks"]["validation_sample_count"], 0)
        self.assertFalse(result["checks"]["validation_available"])
        self.assertEqual(result["checks"]["required_calibration_sample_count"], 3)
        self.assertEqual(result["checks"]["required_validation_sample_count"], 0)
        actual = np.asarray(result["T_base_color"])
        translation_m, rotation_deg = transform_error(
            self.expected_base_to_camera, actual
        )
        self.assertLess(translation_m, 1e-8)
        self.assertLess(rotation_deg, 1e-6)

    def test_available_mode_still_requires_three_calibration_samples(self) -> None:
        result = solve_eye_to_hand(
            self.calibration[:2],
            [],
            self.config.quality,
            use_available_samples=True,
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["provisional"])
        self.assertFalse(result["deployment_ready"])
        self.assertFalse(result["checks"]["sample_count_pass"])
        self.assertIsNone(result["T_base_color"])


if __name__ == "__main__":
    unittest.main()
