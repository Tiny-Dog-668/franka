#!/usr/bin/env python3
"""Run the CUDA-validated 0815 10k-step three-frame GelSight policy."""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0815_gelsight_10000.json"


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
