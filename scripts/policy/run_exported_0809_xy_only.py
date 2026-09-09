#!/usr/bin/env python3
"""运行使用夹爪状态接触近似的 0809 XY RMA Student。"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0809_xy_only.json"


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
