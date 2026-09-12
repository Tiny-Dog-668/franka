from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from franka_sim2real.real_rl.offline_apriltag import (
    OfflineAprilTagDetector,
    _load_boundaries,
    _offline_success_boundary,
)
from franka_sim2real.real_rl.config import X040ObservableAbsoluteRewardConfig

from .config import RLPDConfig
from .replay import ACTION_DIM, PRIVILEGED_DIM, ReplayBuffer, ReplayRole, _pack, _unpack
from .reward import (
    APRILTAG_X040_OBSERVABLE_ABSOLUTE_REWARD,
    RewardInput,
    build_reward,
)


def _success_boundary(
    poses: dict[int, Any | None], initial_z: float, config: RLPDConfig
) -> int | None:
    if config.reward_kind != APRILTAG_X040_OBSERVABLE_ABSOLUTE_REWARD:
        return _offline_success_boundary(poses, initial_z, config)
    streak = 0
    threshold = float(config.reward.success_height_m) - 1e-6
    for boundary in sorted(poses):
        pose = poses[boundary]
        if (
            pose is not None
            and float(pose.object_position_base_m[2] - initial_z) >= threshold
        ):
            streak += 1
        else:
            streak = 0
        if streak >= config.reward.success_consecutive_detections:
            return boundary
    return None


def label_episode(
    run_dir: str | Path,
    config: RLPDConfig,
    role: ReplayRole,
) -> dict[str, Any]:
    resolved = Path(run_dir).expanduser().resolve()
    boundaries, intrinsics = _load_boundaries(resolved)
    detector = OfflineAprilTagDetector(config, intrinsics)  # compatible AprilTag config contract
    poses: dict[int, Any | None] = {}
    records: list[dict[str, Any]] = []
    for boundary in sorted(boundaries):
        item = boundaries[boundary]
        image = cv2.imread(str(item["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read raw boundary frame: {item['path']}")
        pose = detector.detect(boundary, image, item["metadata"])
        poses[boundary] = pose
        records.append(
            pose.to_dict() if pose is not None else {
                "boundary": boundary,
                "detected": False,
                "sequence": item["metadata"].get("sequence"),
                "capture_timestamp": item["metadata"].get("capture_timestamp"),
            }
        )
    (resolved / "rlpd_apriltag_labels.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )

    replay_path = config.offline_replay_path if role == "offline" else config.online_replay_path
    reasons: dict[str, int] = {}
    trainable_count = 0
    reward_function = build_reward(config.reward_kind, config.reward)
    with ReplayBuffer(replay_path, role, create=False) as replay:
        rows = replay.connection.execute(
            "SELECT * FROM transitions WHERE episode_id=? ORDER BY step_id", (resolved.name,)
        ).fetchall()
        if not rows:
            raise RuntimeError(f"RLPD replay has no episode {resolved.name}")
        initial_z = float(rows[0]["initial_object_z"])
        success_boundary = _success_boundary(poses, initial_z, config)
        x040_reward = (
            config.reward
            if isinstance(config.reward, X040ObservableAbsoluteRewardConfig)
            else None
        )
        previous_executed_action = np.zeros(ACTION_DIM, dtype=np.float32)
        drop_armed = False
        drop_already_penalized = False
        first_pose = poses.get(min(poses)) if poses else None
        if x040_reward is not None and first_pose is not None:
            first_height = float(first_pose.object_position_base_m[2] - initial_z)
            drop_armed = first_height >= x040_reward.drop_arm_height_m
        for row in rows:
            step = int(row["step_id"])
            pose_t, pose_next = poses.get(step), poses.get(step + 1)
            timestamp = row["action_timestamp"]
            reason = "ok"
            if success_boundary is not None and step >= success_boundary:
                reason = "after_success"
            elif not bool(row["accepted"]):
                reason = "action_not_accepted"
            elif pose_t is None:
                reason = "tag_missing_t"
            elif pose_next is None:
                reason = "tag_missing_t1"
            elif timestamp is None:
                reason = "action_timestamp_missing"
            elif pose_t.capture_timestamp > float(timestamp):
                reason = "tag_t_after_action"
            elif float(timestamp) >= pose_next.capture_timestamp:
                reason = "tag_t1_not_after_action"
            valid = reason == "ok"
            success = bool(valid and success_boundary == step + 1)
            terminated = success
            truncated = bool(row["truncated"] and not success)
            privileged = next_privileged = None
            reward = None
            executed_action = _unpack(
                row["executed_action"], ACTION_DIM, "executed_action"
            )
            dropped = False
            if x040_reward is not None and pose_next is not None:
                observed_next_height = float(
                    pose_next.object_position_base_m[2] - initial_z
                )
                drop_armed = (
                    drop_armed
                    or observed_next_height >= x040_reward.drop_arm_height_m
                )
                dropped = bool(
                    drop_armed
                    and observed_next_height < x040_reward.drop_trigger_height_m
                    and not drop_already_penalized
                )
                drop_already_penalized = drop_already_penalized or dropped
            if valid:
                ee = _unpack(row["ee_position"], 3, "ee_position")
                next_ee = _unpack(row["next_ee_position"], 3, "next_ee_position")
                relative = pose_t.object_position_base_m - ee
                next_relative = pose_next.object_position_base_m - next_ee
                height = float(pose_t.object_position_base_m[2] - initial_z)
                next_height = float(pose_next.object_position_base_m[2] - initial_z)
                privileged = np.concatenate((relative, [height])).astype(np.float32)
                next_privileged = np.concatenate((next_relative, [next_height])).astype(np.float32)
                action = _unpack(row["action"], ACTION_DIM, "action")
                reward = reward_function.compute(RewardInput(
                    object_relative_to_ee=relative,
                    next_object_relative_to_ee=next_relative,
                    object_height=height,
                    next_object_height=next_height,
                    action=action,
                    success=success,
                    executed_action=executed_action,
                    previous_executed_action=previous_executed_action,
                    next_tool_tcp_position=next_ee,
                    dropped=dropped,
                )).total
                trainable_count += 1
            reasons[reason] = reasons.get(reason, 0) + 1
            replay.connection.execute(
                """UPDATE transitions SET privileged=?,next_privileged=?,reward=?,
                terminated=?,truncated=?,success=?,trainable=?,trainable_reason=? WHERE id=?""",
                (
                    None if privileged is None else _pack(privileged, PRIVILEGED_DIM, "privileged"),
                    None if next_privileged is None else _pack(next_privileged, PRIVILEGED_DIM, "next_privileged"),
                    reward, int(terminated), int(truncated), int(success), int(valid), reason, int(row["id"]),
                ),
            )
            if bool(row["accepted"]):
                previous_executed_action = executed_action
        replay.connection.commit()
    return {
        "episode_id": resolved.name,
        "role": role,
        "transitions": len(rows),
        "trainable_transitions": trainable_count,
        "detected_boundaries": sum(pose is not None for pose in poses.values()),
        "boundary_count": len(poses),
        "success_boundary": success_boundary,
        "reasons": reasons,
    }
