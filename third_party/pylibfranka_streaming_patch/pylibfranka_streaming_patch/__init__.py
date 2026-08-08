"""Install the local async-state API onto pylibfranka 0.21.1."""

from __future__ import annotations

from typing import Any

# Load the official extension first so its auditwheel-bundled libfranka/Poco
# dependencies are present before the sidecar is resolved.
import pylibfranka as _pylibfranka

from . import _native


def install(module: Any | None = None) -> Any:
    if module is None:
        module = _pylibfranka

    version = str(getattr(module, "__version__", "unknown"))
    if version != "0.21.1":
        raise RuntimeError(
            "pylibfranka-streaming-patch requires pylibfranka 0.21.1, "
            f"got {version}"
        )
    def read_once(handler: Any) -> Any:
        return _native.read_once(handler)

    module.AsyncPositionControlHandler.read_once = read_once
    module.TargetStatus = _native.TargetStatus
    return module


__all__ = ["install"]
