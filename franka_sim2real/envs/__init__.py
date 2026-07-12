from __future__ import annotations

from .base import BaseFrankaEnv
from .franka_real import RealFrankaEnv
from .mock_sim import MockFrankaSimEnv


def make_env(config) -> BaseFrankaEnv:
    if config.backend.kind == "sim":
        return MockFrankaSimEnv(config)
    if config.backend.kind == "real":
        return RealFrankaEnv(config)
    raise ValueError(f"Unsupported backend kind: {config.backend.kind}")
