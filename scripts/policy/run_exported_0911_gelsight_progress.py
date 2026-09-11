#!/usr/bin/env python3
"""Validate or preview the RLPD-only 0911 GelSight Progress base policy."""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "e2e_bundle_real_exported_0911_gelsight_progress.json"
)


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
