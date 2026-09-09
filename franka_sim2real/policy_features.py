"""Shared frozen feature contract for the 0814 GelSight Student."""

from __future__ import annotations

import torch


ACTOR_FEATURE_DIM = 1043
BASE_ACTION_DIM = 4


def validate_actor_feature_contract(
    model: torch.jit.ScriptModule,
    expected_dim: int = ACTOR_FEATURE_DIM,
) -> None:
    required = ("encode_visual", "tactile_encoder", "normalizer", "action_head")
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise ValueError(
            "Base model does not expose the actor feature contract: "
            + ", ".join(missing)
        )
    if int(expected_dim) != ACTOR_FEATURE_DIM:
        raise ValueError(f"0814 actor feature dimension must be {ACTOR_FEATURE_DIM}")


def extract_actor_features(
    model: torch.jit.ScriptModule,
    wrist_rgb: torch.Tensor,
    proprio_obs: torch.Tensor,
    action_history: torch.Tensor,
    left_current: torch.Tensor,
    right_current: torch.Tensor,
    left_reference: torch.Tensor,
    right_reference: torch.Tensor,
    *,
    expected_dim: int = ACTOR_FEATURE_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact action-head features and recomputed base action."""

    validate_actor_feature_contract(model, expected_dim)
    _layer3_feature, visual_feature = model.encode_visual(wrist_rgb)
    left_feature, right_feature, _contact_logits = model.tactile_encoder(
        left_current,
        right_current,
        left_reference,
        right_reference,
    )
    features = torch.cat(
        (
            visual_feature,
            left_feature,
            right_feature,
            model.normalizer.normalize_proprio(proprio_obs),
            model.normalizer.normalize_history(action_history),
        ),
        dim=-1,
    )
    if features.ndim != 2 or int(features.shape[-1]) != int(expected_dim):
        raise RuntimeError(
            f"Expected actor features [N,{expected_dim}], got {tuple(features.shape)}"
        )
    recomputed_base_action = torch.tanh(model.action_head(features))
    return features, recomputed_base_action
