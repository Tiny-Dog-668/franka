#!/usr/bin/env python3
"""运行 0809 Direct-Action RMA Student。"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0809_direct_action.json"


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
