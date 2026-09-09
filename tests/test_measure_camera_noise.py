from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "calibration" / "measure_camera_noise.py"
SPEC = importlib.util.spec_from_file_location("measure_camera_noise", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
noise = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = noise
SPEC.loader.exec_module(noise)


def _heteroscedastic_frames(
    shape: tuple[int, int, int],
    frame_count: int,
    shot_coefficient: float,
    read_variance: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """构造已知噪声模型 sigma^2 = a*mu + b 的合成帧堆栈。"""
    generator = np.random.default_rng(seed)
    mean = np.linspace(20.0, 220.0, int(np.prod(shape))).reshape(shape)
    sigma = np.sqrt(shot_coefficient * mean + read_variance)
    frames = np.empty((frame_count, *shape), dtype=np.uint8)
    for index in range(frame_count):
        sample = mean + generator.normal(0.0, 1.0, size=shape) * sigma
        frames[index] = np.clip(np.rint(sample), 0.0, 255.0).astype(np.uint8)
    return frames, mean


class PixelStatisticsTests(unittest.TestCase):
    def test_flicker_removal_separates_global_brightness_drift(self) -> None:
        generator = np.random.default_rng(1)
        base = generator.uniform(40.0, 200.0, size=(16, 16, 3))
        gains = 1.0 + 0.05 * np.sin(np.linspace(0.0, 6.0, 64))
        frames = np.stack(
            [np.clip(np.rint(base * gain), 0.0, 255.0).astype(np.uint8) for gain in gains]
        )

        _, raw_sigma = noise.pixel_statistics(frames)
        weights, _ = noise.flicker_weights(noise.frame_means(frames))
        _, corrected_sigma = noise.pixel_statistics(frames, weights=weights)

        # 整帧同步的亮度调制属于照明变化，扣除后只应剩下量化误差。
        self.assertGreater(float(np.median(raw_sigma)), 2.0)
        self.assertLess(float(np.median(corrected_sigma)), 0.6)

    def test_pixel_statistics_requires_two_frames(self) -> None:
        with self.assertRaises(ValueError):
            noise.pixel_statistics(np.zeros((1, 4, 4, 3), dtype=np.uint8))


class SigmaLutTests(unittest.TestCase):
    def test_underpopulated_bins_are_reported_as_null(self) -> None:
        mean = np.full((8, 8, 3), 100.0)
        sigma = np.full((8, 8, 3), 3.0)

        lut = noise.build_sigma_lut(mean, sigma, bins=8, minimum_bin_count=10)

        populated = [value for value in lut["sigma_dn"] if value is not None]
        self.assertEqual(len(populated), 1)
        self.assertAlmostEqual(populated[0], 3.0)
        self.assertEqual(sum(lut["pixel_count"]), 8 * 8 * 3)
        normalized = [value for value in lut["sigma_normalized"] if value is not None]
        self.assertAlmostEqual(normalized[0], 3.0 / 255.0)

    def test_affine_fit_returns_none_without_enough_bins(self) -> None:
        lut = {
            "bin_center_dn": [10.0, 20.0],
            "sigma_dn": [None, 2.0],
            "sigma_normalized": [None, 2.0 / 255.0],
            "pixel_count": [0, 500],
        }

        fit = noise.fit_affine_noise_model(lut)

        self.assertIsNone(fit["a"])
        self.assertEqual(fit["fitted_bin_count"], 1)


class AnalyzeFrameStackTests(unittest.TestCase):
    def test_affine_fit_recovers_known_noise_model(self) -> None:
        shot_coefficient = 0.5
        read_variance = 4.0
        frames, _ = _heteroscedastic_frames(
            (96, 96, 3), 300, shot_coefficient, read_variance
        )

        trial = noise.analyze_frame_stack(
            frames,
            bins=32,
            minimum_bin_count=200,
            dark_threshold_dn=32.0,
            motion_sigma_dn=12.0,
        )
        fit = trial["affine_fit"]

        self.assertAlmostEqual(fit["a"], shot_coefficient, delta=0.08 * shot_coefficient)
        self.assertAlmostEqual(fit["b"], read_variance, delta=0.2 * read_variance)

        expected_sigma = float(np.sqrt(shot_coefficient * 128.0 + read_variance))
        fitted_sigma = float(np.sqrt(fit["a"] * 128.0 + fit["b"]))
        self.assertAlmostEqual(fitted_sigma, expected_sigma, delta=0.03 * expected_sigma)

    def test_static_scene_reports_no_motion_and_normalized_summary(self) -> None:
        frames, _ = _heteroscedastic_frames((32, 32, 3), 64, 0.2, 1.0, seed=2)

        trial = noise.analyze_frame_stack(
            frames,
            bins=16,
            minimum_bin_count=50,
            dark_threshold_dn=32.0,
            motion_sigma_dn=12.0,
        )

        self.assertEqual(trial["static_scene_check"]["motion_pixel_fraction"], 0.0)
        self.assertAlmostEqual(
            trial["overall"]["median_normalized"],
            trial["overall"]["median_dn"] / 255.0,
        )
        self.assertEqual(set(trial["per_channel"]), {"red", "green", "blue"})

    def test_moving_patch_is_detected(self) -> None:
        generator = np.random.default_rng(3)
        frames = np.full((40, 32, 32, 3), 100, dtype=np.uint8)
        # 左上角 8x8 的方块在两个亮度之间跳变，模拟视野里有东西在动。
        for index in range(frames.shape[0]):
            frames[index, :8, :8] = 40 if index % 2 else 220
        frames = np.clip(
            frames.astype(np.int16) + generator.integers(-1, 2, size=frames.shape),
            0,
            255,
        ).astype(np.uint8)

        trial = noise.analyze_frame_stack(
            frames,
            bins=16,
            minimum_bin_count=10,
            dark_threshold_dn=32.0,
            motion_sigma_dn=12.0,
        )

        expected_fraction = (8 * 8) / (32 * 32)
        self.assertAlmostEqual(
            trial["static_scene_check"]["motion_pixel_fraction"],
            expected_fraction,
            delta=0.01,
        )

    def test_rejects_stack_without_three_channels(self) -> None:
        with self.assertRaises(ValueError):
            noise.analyze_frame_stack(
                np.zeros((4, 8, 8), dtype=np.uint8),
                bins=8,
                minimum_bin_count=1,
                dark_threshold_dn=32.0,
                motion_sigma_dn=12.0,
            )


class ResolveModelViewSizeTests(unittest.TestCase):
    def test_reads_input_signature_from_bundle_config_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "student.json"
            metadata_path.write_text(
                json.dumps({"input_signature": {"wrist_rgb": [224, 224, 3]}}),
                encoding="utf-8",
            )
            bundle_path = root / "bundle.json"
            bundle_path.write_text(
                json.dumps({"model": {"metadata_path": str(metadata_path)}}),
                encoding="utf-8",
            )

            width, height, resolved = noise.resolve_model_view_size(bundle_path, None)

            self.assertEqual((width, height), (224, 224))
            self.assertEqual(resolved, metadata_path.resolve())

    def test_rejects_metadata_without_wrist_rgb_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "student.json"
            metadata_path.write_text(json.dumps({"input_signature": {}}), encoding="utf-8")

            with self.assertRaises(ValueError):
                noise.resolve_model_view_size(metadata_path, metadata_path)


if __name__ == "__main__":
    unittest.main()
