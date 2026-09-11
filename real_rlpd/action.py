from __future__ import annotations

import math
from typing import Any

import numpy as np


ACTION_DIM = 4


def residual_span(commissioning_limit: float) -> float:
    limit = float(commissioning_limit)
    if not math.isfinite(limit) or not 0.0 < limit <= 1.0:
        raise ValueError("commissioning_limit must be in (0,1]")
    return 2.0 * limit


def expert_residual_target(
    base_limited: Any,
    expert_limited: Any,
    commissioning_limit: float,
) -> np.ndarray:
    base = np.asarray(base_limited, dtype=np.float32).reshape(-1)
    expert = np.asarray(expert_limited, dtype=np.float32).reshape(-1)
    if base.shape != (ACTION_DIM,) or expert.shape != (ACTION_DIM,):
        raise ValueError("RLPD actions must contain four values")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(expert)):
        raise ValueError("RLPD actions must be finite")
    unit = (expert - base) / residual_span(commissioning_limit)
    if np.any(np.abs(unit) > 1.0 + 1e-5):
        raise RuntimeError("Expert action is outside the post-commissioning residual span")
    return np.clip(unit, -1.0, 1.0).astype(np.float32)


def combine_post_limit(
    base_limited: Any,
    unit_residual: Any,
    commissioning_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_limited, dtype=np.float32).reshape(-1)
    unit = np.asarray(unit_residual, dtype=np.float32).reshape(-1)
    if base.shape != (ACTION_DIM,) or unit.shape != (ACTION_DIM,):
        raise ValueError("RLPD base and residual actions must be 4D")
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(unit)):
        raise ValueError("RLPD base and residual actions must be finite")
    unit = np.clip(unit, -1.0, 1.0)
    residual = unit * residual_span(commissioning_limit)
    return (base + residual).astype(np.float32), residual.astype(np.float32)
