from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GelSightDeviceSpec:
    cam_id: int
    label: str
    serial: str | None = None
    device_path: str | None = None


def _video_number(video_path: Path) -> int:
    name = video_path.name
    if not name.startswith("video") or not name[5:].isdigit():
        raise ValueError(f"Invalid video device name: {name!r}")
    return int(name[5:])


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    return value or None


def discover_gelsight_cameras(
    sysfs_root: str | Path = "/sys/class/video4linux",
) -> list[GelSightDeviceSpec]:
    """Return primary V4L2 image streams for connected GelSight cameras."""

    root = Path(sysfs_root)
    if not root.exists():
        return []

    specs: list[GelSightDeviceSpec] = []
    for video_path in sorted(root.glob("video*"), key=_video_number):
        name = _read_optional(video_path / "name")
        index = _read_optional(video_path / "index")
        if name is None or "gelsight" not in name.lower() or index != "0":
            continue

        interface = _read_optional(video_path / "device" / "interface")
        serial = _read_optional(video_path / "device" / ".." / "serial")
        try:
            device_path = str((video_path / "device").resolve(strict=True))
        except OSError:
            device_path = None
        specs.append(
            GelSightDeviceSpec(
                cam_id=_video_number(video_path),
                label=interface or name,
                serial=serial,
                device_path=device_path,
            )
        )
    return specs


def select_gelsight_pair(
    sysfs_root: str | Path = "/sys/class/video4linux",
) -> tuple[GelSightDeviceSpec, GelSightDeviceSpec]:
    """Discover exactly two primary streams and assign their current device order."""

    specs = discover_gelsight_cameras(sysfs_root)
    if len(specs) != 2:
        discovered = ", ".join(
            f"/dev/video{spec.cam_id} ({spec.label})" for spec in specs
        ) or "none"
        raise RuntimeError(
            "Automatic GelSight selection requires exactly two primary image streams; "
            f"found {len(specs)}: {discovered}"
        )
    return specs[0], specs[1]


__all__ = (
    "GelSightDeviceSpec",
    "discover_gelsight_cameras",
    "select_gelsight_pair",
)
