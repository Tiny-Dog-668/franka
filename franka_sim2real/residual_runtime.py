"""Runtime contract and loader for an independently trained Residual BC head."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_RESIDUAL_MAX_ABS = 0.1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ResidualDeploySettings:
    model_path: Path
    metadata_path: Path
    scale: float = 1.0
    max_abs: float = DEFAULT_RESIDUAL_MAX_ABS

    def validate(self) -> None:
        if not self.model_path.is_file():
            raise ValueError(f"Residual TorchScript does not exist: {self.model_path}")
        if not self.metadata_path.is_file():
            raise ValueError(f"Residual metadata does not exist: {self.metadata_path}")
        if not math.isfinite(self.scale) or not 0.0 < self.scale <= 1.0:
            raise ValueError("Residual scale must be finite and in (0,1]")
        if not math.isfinite(self.max_abs) or not 0.0 < self.max_abs <= 1.0:
            raise ValueError("Residual max abs must be finite and in (0,1]")


class ResidualPolicyRuntime:
    """Validated TorchScript residual head bound to one exact base checkpoint."""

    def __init__(
        self,
        settings: ResidualDeploySettings,
        *,
        base_model_path: Path,
        base_kind: str,
        device: torch.device,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.metadata = json.loads(settings.metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("kind") != "franka_hil_residual_bc":
            raise ValueError("Residual metadata kind must be 'franka_hil_residual_bc'")
        if self.metadata.get("format_version") != 1:
            raise ValueError("Residual metadata format_version must be 1")
        if self.metadata.get("input_contract") != "actor_feature_1043_plus_base_action_4":
            raise ValueError("Residual input contract must be actor_feature_1043_plus_base_action_4")
        if self.metadata.get("output_contract") != "prelimit_normalized_residual_xyz_3":
            raise ValueError("Residual output contract must be prelimit_normalized_residual_xyz_3")
        if self.metadata.get("gripper_source") != "base_policy":
            raise ValueError("Residual metadata must keep gripper_source='base_policy'")
        config = self.metadata.get("model_config", {})
        expected_config = {
            "actor_feature_dim": 1043,
            "base_action_dim": 4,
            "output_dim": 3,
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(f"Residual model_config.{key} must be {expected}")
        if base_kind != "tacex_rma_gelsight_size_buckets_student_torchscript":
            raise ValueError(
                "Residual BC v1 requires the 0814 GelSight size-buckets Student, got "
                f"{base_kind!r}"
            )
        expected_base_sha = self.metadata.get("base_model_sha256")
        actual_base_sha = _sha256_file(base_model_path)
        if not isinstance(expected_base_sha, str) or expected_base_sha != actual_base_sha:
            raise ValueError(
                "Residual checkpoint was trained for a different base policy: "
                f"expected_sha256={expected_base_sha!r}, actual_sha256={actual_base_sha}"
            )
        self.base_model_sha256 = actual_base_sha
        self.model_sha256 = _sha256_file(settings.model_path)
        self.device = device
        self.model = torch.jit.load(str(settings.model_path), map_location=device).eval()
        with torch.inference_mode():
            output = self.model(
                torch.zeros((1, 1043), dtype=torch.float32, device=device),
                torch.zeros((1, 4), dtype=torch.float32, device=device),
            )
        if tuple(output.shape) != (1, 3) or not torch.isfinite(output).all():
            raise ValueError(
                f"Residual TorchScript must return one finite [1,3] tensor, got {tuple(output.shape)}"
            )

    def apply(
        self,
        actor_features: torch.Tensor,
        base_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(actor_features.shape) != (1, 1043):
            raise RuntimeError(
                f"Residual actor features must have shape [1,1043], got {tuple(actor_features.shape)}"
            )
        if tuple(base_action.shape) != (1, 4):
            raise RuntimeError(
                f"Residual base action must have shape [1,4], got {tuple(base_action.shape)}"
            )
        predicted = self.model(actor_features, base_action)
        if tuple(predicted.shape) != (1, 3) or not torch.isfinite(predicted).all():
            raise RuntimeError("Residual model produced an invalid XYZ correction")
        applied = torch.clamp(
            predicted * float(self.settings.scale),
            min=-float(self.settings.max_abs),
            max=float(self.settings.max_abs),
        )
        final = base_action.clone()
        final[:, :3] = final[:, :3] + applied
        if not torch.isfinite(final).all():
            raise RuntimeError("Combined base+residual action contains NaN or Inf")
        return final, predicted, applied

    def report(self) -> dict[str, Any]:
        return {
            "model_path": str(self.settings.model_path.resolve()),
            "metadata_path": str(self.settings.metadata_path.resolve()),
            "model_sha256": self.model_sha256,
            "base_model_sha256": self.base_model_sha256,
            "scale": float(self.settings.scale),
            "max_abs": float(self.settings.max_abs),
            "input_contract": self.metadata["input_contract"],
            "output_contract": self.metadata["output_contract"],
            "gripper_source": self.metadata["gripper_source"],
        }


def validate_residual_artifacts(
    settings: ResidualDeploySettings,
    *,
    base_model_path: Path,
    base_metadata_path: Path,
    device: str,
) -> dict[str, Any]:
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    runtime = ResidualPolicyRuntime(
        settings,
        base_model_path=base_model_path,
        base_kind=str(base_metadata.get("kind", "unknown")),
        device=torch.device(device),
    )
    return runtime.report()
