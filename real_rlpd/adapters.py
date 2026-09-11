from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


FEATURE_DIM = 1043
ACTION_DIM = 4
SUPPORTED_KINDS = {
    "tacex_rma_gelsight_size_buckets_student_torchscript": "gelsight_reference_single_frame_v1",
    "tacex_rma_gelsight_size_buckets_progress_student_torchscript": (
        "gelsight_reference_progress_single_frame_v1"
    ),
    "tacex_rma_gelsight_x040_dr_three_frame_student_torchscript": "gelsight_reference_three_frame_v1",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_sha256(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AdapterContract:
    adapter_id: str
    policy_kind: str
    feature_dim: int
    action_dim: int
    model_sha256: str
    metadata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


class GelSightPolicyAdapter:
    """把 0814/0823/0911 的不同导出接口统一为 action-head feature。"""

    def __init__(
        self,
        model: torch.jit.ScriptModule,
        metadata: dict[str, Any],
        model_path: str | Path,
    ) -> None:
        kind = str(metadata.get("kind", ""))
        if kind not in SUPPORTED_KINDS:
            raise ValueError(f"RLPD does not support base policy kind {kind!r}")
        visual_method = (
            "encode_visual"
            if kind
            in {
                "tacex_rma_gelsight_size_buckets_student_torchscript",
                "tacex_rma_gelsight_size_buckets_progress_student_torchscript",
            }
            else "encode_visual_features"
        )
        required = (visual_method, "tactile_encoder", "normalizer", "action_head")
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError("Base policy is missing RLPD feature APIs: " + ", ".join(missing))
        self.model = model
        self.metadata = metadata
        self.kind = kind
        self.visual_method = visual_method
        self.contract = AdapterContract(
            adapter_id=SUPPORTED_KINDS[kind],
            policy_kind=kind,
            feature_dim=FEATURE_DIM,
            action_dim=ACTION_DIM,
            model_sha256=sha256_file(model_path),
            metadata_sha256=metadata_sha256(metadata),
        )

    def extract(
        self,
        wrist_rgb: torch.Tensor,
        proprio_obs: torch.Tensor,
        action_history: torch.Tensor,
        tactile: dict[str, torch.Tensor],
        base_action: torch.Tensor,
    ) -> torch.Tensor:
        names = (
            "gsmini_left_rgb", "gsmini_right_rgb",
            "gsmini_left_reference_rgb", "gsmini_right_reference_rgb",
        )
        missing = [name for name in names if name not in tactile]
        if missing:
            raise RuntimeError("RLPD adapter is missing tactile inputs: " + ", ".join(missing))
        if self.visual_method == "encode_visual":
            _layer3, visual = self.model.encode_visual(wrist_rgb)
        else:
            visual = self.model.encode_visual_features(wrist_rgb)
        left, right, _contacts = self.model.tactile_encoder(*(tactile[name] for name in names))
        features = torch.cat(
            (
                visual,
                left,
                right,
                self.model.normalizer.normalize_proprio(proprio_obs),
                self.model.normalizer.normalize_history(action_history),
            ),
            dim=-1,
        )
        if tuple(features.shape) != (1, FEATURE_DIM):
            raise RuntimeError(f"RLPD expected features [1,{FEATURE_DIM}], got {tuple(features.shape)}")
        recomputed = torch.tanh(self.model.action_head(features))
        if tuple(base_action.shape) != (1, ACTION_DIM):
            raise RuntimeError("RLPD base action must be [1,4]")
        error = float(torch.max(torch.abs(recomputed - base_action)).item())
        if error > 5e-4:
            raise RuntimeError(f"RLPD feature path does not reproduce base action: {error:.6g}")
        return features
