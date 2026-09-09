from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "diagnostics" / "compare_sim_real_images.py"
SPEC = importlib.util.spec_from_file_location("compare_sim_real_images", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def _constant_images(count: int, rgb: tuple[int, int, int], size: int = 16) -> np.ndarray:
    images = np.empty((count, size, size, 3), dtype=np.uint8)
    images[..., 0] = rgb[0]
    images[..., 1] = rgb[1]
    images[..., 2] = rgb[2]
    return images


def _statistics(images: np.ndarray) -> dict:
    return compare.image_statistics(
        images, dark_threshold_dn=16.0, bright_threshold_dn=120.0
    )


class HistogramMetricTests(unittest.TestCase):
    def test_wasserstein_equals_pure_shift(self) -> None:
        left = np.zeros(256)
        right = np.zeros(256)
        left[50] = 1.0
        right[70] = 1.0

        self.assertAlmostEqual(compare.wasserstein_distance_dn(left, right), 20.0, places=6)
        self.assertAlmostEqual(compare.kolmogorov_smirnov(left, right), 1.0, places=6)

    def test_identical_histograms_have_zero_distance(self) -> None:
        histogram = np.bincount(np.array([10, 10, 200, 30]), minlength=256).astype(float)

        self.assertEqual(compare.wasserstein_distance_dn(histogram, histogram), 0.0)
        self.assertEqual(compare.kolmogorov_smirnov(histogram, histogram), 0.0)

    def test_percentile_from_histogram(self) -> None:
        histogram = np.zeros(256)
        histogram[10] = 50.0
        histogram[200] = 50.0

        self.assertEqual(compare.percentile_from_histogram(histogram, 25.0), 10.0)
        self.assertEqual(compare.percentile_from_histogram(histogram, 75.0), 200.0)


class ImageStatisticsTests(unittest.TestCase):
    def test_constant_image_reports_exact_channel_means(self) -> None:
        stats = _statistics(_constant_images(4, (10, 40, 20)))

        means = stats["channels"]["overall_mean_dn"]
        self.assertAlmostEqual(means["red"], 10.0)
        self.assertAlmostEqual(means["green"], 40.0)
        self.assertAlmostEqual(means["blue"], 20.0)
        expected_luminance = 0.2126 * 10 + 0.7152 * 40 + 0.0722 * 20
        self.assertAlmostEqual(stats["luminance"]["mean_dn"], expected_luminance, places=3)

    def test_dark_region_recovers_green_cast(self) -> None:
        # 近黑但偏绿的背景，模拟机器人状态指示灯造成的色偏。
        stats = _statistics(_constant_images(2, (3, 11, 2)))

        dark = stats["channels"]["dark_region"]
        self.assertAlmostEqual(dark["pixel_fraction"], 1.0)
        self.assertAlmostEqual(dark["ratios"]["green_over_red"], 11.0 / 3.0, places=5)
        self.assertAlmostEqual(dark["ratios"]["green_over_blue"], 11.0 / 2.0, places=5)
        self.assertEqual(stats["channels"]["bright_region"]["pixel_fraction"], 0.0)

    def test_bimodal_scene_splits_into_dark_and_bright(self) -> None:
        images = _constant_images(1, (0, 0, 0), size=10)
        images[:, :2] = 200  # 20% 的行是亮的

        stats = _statistics(images)

        self.assertAlmostEqual(stats["channels"]["dark_region"]["pixel_fraction"], 0.8)
        self.assertAlmostEqual(stats["channels"]["bright_region"]["pixel_fraction"], 0.2)
        self.assertAlmostEqual(stats["luminance"]["fraction_below"]["dn_16"], 0.8)

    def test_chunking_matches_single_pass(self) -> None:
        generator = np.random.default_rng(0)
        images = generator.integers(0, 256, size=(70, 8, 8, 3), dtype=np.uint8)

        stats = _statistics(images)

        expected = images.reshape(-1, 3).mean(axis=0)
        for index, label in enumerate(compare.CHANNEL_LABELS):
            self.assertAlmostEqual(
                stats["channels"]["overall_mean_dn"][label], float(expected[index]), places=6
            )
        np.testing.assert_allclose(
            stats["_mean_image"], images.mean(axis=0, dtype=np.float64), rtol=0, atol=1e-9
        )

    def test_rejects_non_rgb_stack(self) -> None:
        with self.assertRaises(ValueError):
            _statistics(np.zeros((2, 8, 8), dtype=np.uint8))


class CompareStatisticsTests(unittest.TestCase):
    def test_deltas_are_sim_minus_real(self) -> None:
        real = _statistics(_constant_images(2, (10, 10, 10)))
        sim = _statistics(_constant_images(2, (30, 30, 30)))

        result = compare.compare_statistics(real, sim)

        self.assertAlmostEqual(result["luminance_mean_delta_dn"], 20.0, places=3)
        self.assertAlmostEqual(result["channel_mean_delta_dn"]["red"], 20.0, places=3)
        self.assertAlmostEqual(result["luminance_wasserstein_dn"], 20.0, places=3)
        # 真机全暗、仿真全亮，暗区占比应当整体下降。
        self.assertAlmostEqual(result["dark_pixel_fraction_delta"], -1.0, places=6)


class LoadImageSourceTests(unittest.TestCase):
    def test_loads_directory_and_subsamples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(10):
                Image.fromarray(_constant_images(1, (index, index, index))[0]).save(
                    root / f"step_{index:04d}.png"
                )

            images, source = compare.load_image_source(root, "wrist_rgb", max_images=4)

            self.assertEqual(images.shape, (4, 16, 16, 3))
            self.assertEqual(source["kind"], "directory")
            self.assertEqual(source["available"], 10)
            self.assertEqual(source["used"], 4)

    def test_loads_npz_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0000.npz"
            np.savez_compressed(path, wrist_rgb=_constant_images(6, (5, 6, 7)))

            images, source = compare.load_image_source(path, "wrist_rgb", max_images=100)

            self.assertEqual(images.shape, (6, 16, 16, 3))
            self.assertEqual(source["kind"], "npz")
            self.assertEqual(source["available"], 6)

    def test_concatenates_npz_directory_across_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(3):
                np.savez_compressed(
                    root / f"episode_{index:04d}.npz",
                    wrist_rgb=_constant_images(4, (index, index, index)),
                )
            # 同目录下的非图像产物不应干扰载入。
            (root / "episodes.csv").write_text("a,b\n", encoding="utf-8")

            images, source = compare.load_image_source(root, "wrist_rgb", max_images=100)

            self.assertEqual(images.shape, (12, 16, 16, 3))
            self.assertEqual(source["kind"], "npz_directory")
            self.assertEqual(source["archive_count"], 3)
            self.assertEqual(source["available"], 12)

    def test_missing_npz_key_lists_available_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0000.npz"
            np.savez_compressed(path, proprio_obs=np.zeros((3, 15)))

            with self.assertRaises(KeyError) as context:
                compare.load_image_source(path, "wrist_rgb", max_images=10)

            self.assertIn("proprio_obs", str(context.exception))


if __name__ == "__main__":
    unittest.main()
