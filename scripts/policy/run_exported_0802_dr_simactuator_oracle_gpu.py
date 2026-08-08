#!/usr/bin/env python3
"""Run the 0802 DR actor with oracle cube XYZ and zero contact.

The oracle actor bypasses ResNet and defaults to CPU; pass --gpu only when a
healthy CUDA device is available.
"""
from __future__ import annotations

from run_exported_0711 import REPO_ROOT, main

DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "e2e_bundle_real_exported_0802_dr_simactuator_oracle_gpu.json"
)


if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
