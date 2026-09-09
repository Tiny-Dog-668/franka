#!/usr/bin/env python3
"""校验或预览 TacEx 0812 三帧 Appearance-DR policy。

该 checkpoint 的 TacEx metadata 标为 evaluation-only；部署器会拒绝真机运动。
"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main


if __name__ == "__main__":
    raise SystemExit(main(
        default_config=REPO_ROOT / "configs/e2e_bundle_real_exported_0812_three_frame_extent_dr.json"
    ))
