#!/usr/bin/env python3
"""运行 TacEx 0811 改变方块尺寸的 X040-Wide policy。"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


if __name__ == "__main__":
    raise SystemExit(main(
        default_config=REPO_ROOT / "configs/e2e_bundle_real_exported_0811_change_cube_size.json"
    ))
