from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from franka_sim2real.gelsight_devices import (
    GelSightDeviceSpec,
    discover_gelsight_cameras,
    select_gelsight_pair,
)
from franka_sim2real.e2e_bundle import (
    BundleTactileCameraConfig,
    resolve_gelsight_devices,
)


def _add_video_device(
    root: Path,
    number: int,
    *,
    name: str,
    index: int,
    interface: str | None = None,
    serial: str | None = None,
) -> None:
    video = root / f"video{number}"
    device = video / "device"
    device.mkdir(parents=True)
    (video / "name").write_text(name, encoding="utf-8")
    (video / "index").write_text(str(index), encoding="utf-8")
    if interface is not None:
        (device / "interface").write_text(interface, encoding="utf-8")
    if serial is not None:
        (video / "serial").write_text(serial, encoding="utf-8")


class GelSightDeviceDiscoveryTests(unittest.TestCase):
    def test_discovers_only_primary_gelsight_image_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _add_video_device(root, 0, name="Laptop Webcam", index=0)
            _add_video_device(
                root,
                10,
                name="GelSight Mini R0B: GelSight Mini",
                index=0,
                interface="GelSight Mini LEFT-SERIAL",
                serial="LEFTSERIAL",
            )
            _add_video_device(
                root,
                11,
                name="GelSight Mini R0B: GelSight Mini",
                index=1,
            )
            _add_video_device(
                root,
                12,
                name="GelSight Mini R0B: GelSight Mini",
                index=0,
                interface="GelSight Mini RIGHT-SERIAL",
                serial="RIGHTSERIAL",
            )

            specs = discover_gelsight_cameras(root)

        self.assertEqual([spec.cam_id for spec in specs], [10, 12])
        self.assertEqual([spec.serial for spec in specs], ["LEFTSERIAL", "RIGHTSERIAL"])
        self.assertEqual(specs[0].label, "GelSight Mini LEFT-SERIAL")

    def test_pair_selection_uses_current_device_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _add_video_device(root, 12, name="GelSight Mini B", index=0)
            _add_video_device(root, 10, name="GelSight Mini A", index=0)

            left, right = select_gelsight_pair(root)

        self.assertEqual((left.cam_id, right.cam_id), (10, 12))

    def test_pair_selection_fails_closed_unless_exactly_two_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _add_video_device(root, 10, name="GelSight Mini A", index=0)

            with self.assertRaisesRegex(RuntimeError, "found 1"):
                select_gelsight_pair(root)

    def test_runtime_config_is_replaced_with_discovered_ids(self) -> None:
        config = BundleTactileCameraConfig(
            enabled=True,
            auto_discover=True,
            left_device=4,
            right_device=6,
        )
        specs = (
            GelSightDeviceSpec(10, "GelSight A", "SERIALA"),
            GelSightDeviceSpec(12, "GelSight B", "SERIALB"),
        )

        resolved = resolve_gelsight_devices(config, selector=lambda: specs)

        self.assertEqual(resolved, (10, 12))
        self.assertEqual((config.left_device, config.right_device), (10, 12))


if __name__ == "__main__":
    unittest.main()
