#!/usr/bin/env python3
"""Run the 0813 GelSight reference-delta Student on the real Franka."""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0813_gelsight.json"


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
