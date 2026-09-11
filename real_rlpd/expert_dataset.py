from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ACTION_LABELS = ("dx", "dy", "dz", "gripper_delta_width")


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        for value in values:
            handle.write(json.dumps(value, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **values)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class _Boundary:
    step: int
    observation: dict[str, Any]
    camera_metadata: dict[str, Any]
    action_history: np.ndarray
    proprio: np.ndarray
    wrist_rgb: np.ndarray
    tactile_images: dict[str, np.ndarray]


class ExpertEpisodeDataset:
    """策略无关的专家源数据；base feature/residual 只属于派生 Replay。"""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        action_scales: Any,
        initial_object_z: float,
        collection_provenance: dict[str, Any],
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.action_scales = _vector(action_scales, 4, "action_scales")
        self.initial_object_z = float(initial_object_z)
        if not np.isfinite(self.initial_object_z):
            raise ValueError("initial_object_z must be finite")
        self.collection_provenance = dict(collection_provenance)
        self._boundaries: list[_Boundary] = []
        self._actions: list[dict[str, Any]] = []
        self._pending_action: dict[str, Any] | None = None
        self._references: dict[str, np.ndarray] | None = None
        self._closed = False

    def observe_boundary(
        self,
        step: int,
        observation: Any,
        result: Any,
        *,
        truncate: bool = False,
    ) -> None:
        expected_step = len(self._boundaries)
        if int(step) != expected_step:
            raise RuntimeError(
                f"Expert dataset expected boundary {expected_step}, got {int(step)}"
            )
        tactile = {
            name: np.asarray(image, dtype=np.uint8).copy()
            for name, image in result.tactile_images.items()
        }
        required = {"gsmini_left_rgb", "gsmini_right_rgb"}
        if set(tactile) != required:
            raise RuntimeError(
                "Expert dataset requires synchronized left/right GelSight images"
            )
        references = {
            name: np.asarray(image, dtype=np.uint8).copy()
            for name, image in result.tactile_references.items()
        }
        required_references = {
            "gsmini_left_reference_rgb", "gsmini_right_reference_rgb"
        }
        if set(references) != required_references:
            raise RuntimeError("Expert dataset requires fixed left/right GelSight references")
        if self._references is None:
            self._references = references
        else:
            for name in required_references:
                if not np.array_equal(self._references[name], references[name]):
                    raise RuntimeError("GelSight reference changed inside one expert episode")

        wrist_rgb = np.asarray(result.image, dtype=np.uint8)
        if wrist_rgb.ndim != 3 or wrist_rgb.shape[-1] != 3:
            raise RuntimeError(
                f"Expert dataset wrist_rgb must be HxWx3, got {wrist_rgb.shape}"
            )
        self._boundaries.append(_Boundary(
            step=int(step),
            observation=dict(observation.to_dict()),
            camera_metadata=dict(result.camera_metadata),
            action_history=_vector(result.action_history, 4, "action_history").copy(),
            proprio=_vector(result.proprio, 15, "proprio").copy(),
            wrist_rgb=wrist_rgb.copy(),
            tactile_images=tactile,
        ))
        if self._pending_action is not None:
            self._pending_action["next_boundary"] = int(step)
            self._pending_action["truncated"] = bool(truncate)
            self._actions.append(self._pending_action)
            self._pending_action = None

    def record_action(
        self,
        result: Any,
        accepted: bool,
        *,
        action_timestamp: float | None,
    ) -> None:
        if not self._boundaries or self._pending_action is not None:
            raise RuntimeError("Expert dataset action/boundary sequence is inconsistent")
        info = result.inference_info.get("rlpd")
        if not isinstance(info, dict) or not bool(info.get("expert_takeover")):
            raise RuntimeError("Expert dataset did not receive an expert takeover action")
        requested = _vector(result.raw_action, 4, "expert_requested_action")
        limited = _vector(result.executed_action, 4, "expert_limited_action")
        self._pending_action = {
            "step": int(self._boundaries[-1].step),
            "boundary": int(self._boundaries[-1].step),
            "next_boundary": None,
            "accepted": bool(accepted),
            "action_timestamp": (
                float(action_timestamp)
                if accepted and action_timestamp is not None
                else None
            ),
            "requested_action_normalized": requested.tolist(),
            "requested_action_physical_delta_m": (requested * self.action_scales).tolist(),
            "expert_action_normalized": limited.tolist(),
            "expert_action_physical_delta_m": (limited * self.action_scales).tolist(),
            "executed_action_normalized": limited.tolist() if accepted else None,
            "executed_action_physical_delta_m": (
                (limited * self.action_scales).tolist() if accepted else None
            ),
            "pressed_keys": list(info.get("pressed_keys", [])),
            "sampled_monotonic_ns": info.get("sampled_monotonic_ns"),
            "focused": bool(info.get("focused", False)),
            "truncated": False,
        }

    def close(self) -> dict[str, Any]:
        if self._closed:
            return self.report()
        self._closed = True
        if self._pending_action is not None:
            self._pending_action["truncated"] = True
            self._actions.append(self._pending_action)
            self._pending_action = None
        if not self._boundaries:
            return self.report()
        assert self._references is not None

        observation_dir = self.run_dir / "expert_observations"
        boundary_records: list[dict[str, Any]] = []
        for boundary in self._boundaries:
            relative_path = Path("expert_observations") / f"boundary_{boundary.step:04d}.npz"
            _atomic_npz(
                self.run_dir / relative_path,
                {
                    "wrist_rgb": boundary.wrist_rgb,
                    "proprio_obs": boundary.proprio,
                    "action_history": boundary.action_history,
                    **boundary.tactile_images,
                },
            )
            boundary_records.append({
                "boundary": boundary.step,
                "observation_path": relative_path.as_posix(),
                "observation": boundary.observation,
                "camera_frame": boundary.camera_metadata,
                "wrist_rgb_shape": list(boundary.wrist_rgb.shape),
                "tactile_rgb_shapes": {
                    name: list(image.shape)
                    for name, image in boundary.tactile_images.items()
                },
            })
        reference_path = observation_dir / "gelsight_reference.npz"
        _atomic_npz(reference_path, self._references)
        _atomic_jsonl(self.run_dir / "expert_boundaries.jsonl", boundary_records)
        _atomic_jsonl(self.run_dir / "expert_actions.jsonl", self._actions)
        _atomic_json(self.run_dir / "expert_manifest.json", {
            "schema_version": SCHEMA_VERSION,
            "kind": "franka_policy_independent_expert_episode",
            "episode_id": self.run_dir.name,
            "initial_object_z_in_base_m": self.initial_object_z,
            "boundary_count": len(self._boundaries),
            "action_count": len(self._actions),
            "complete_transition_count": sum(
                action["next_boundary"] is not None for action in self._actions
            ),
            "action_contract": {
                "labels": list(ACTION_LABELS),
                "normalized_scales_m": self.action_scales.tolist(),
                "canonical_value": "expert_action_physical_delta_m",
                "frame": "robot_root_xyz_plus_gripper_delta_width",
            },
            "observation_contract": {
                "wrist_rgb": "current_processed_frame_not_policy_history",
                "proprio_obs": 15,
                "action_history": 4,
                "tactile_current": ["gsmini_left_rgb", "gsmini_right_rgb"],
                "tactile_reference_path": reference_path.relative_to(self.run_dir).as_posix(),
                "apriltag_raw_boundaries": "raw_rgb/boundary_XXXX.png",
            },
            "derived_fields_excluded": [
                "base_policy_feature", "base_policy_action", "residual_action"
            ],
            "collection_provenance": self.collection_provenance,
        })
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            "source_dataset": str(self.run_dir / "expert_manifest.json"),
            "boundaries": len(self._boundaries),
            "actions": len(self._actions),
            "complete_transitions": sum(
                action["next_boundary"] is not None for action in self._actions
            ),
        }
