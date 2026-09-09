#!/usr/bin/env python3
"""Run the 0815 60k-step three-frame GelSight policy on the real Franka."""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0815_gelsight_60000.json"


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
