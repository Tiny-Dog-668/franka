#!/usr/bin/env python3
"""运行 TacEx 0812 三帧 RGB X040-Wide policy。"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


if __name__ == "__main__":
    raise SystemExit(main(
        default_config=REPO_ROOT / "configs/e2e_bundle_real_exported_0812_three_picture.json"
    ))
