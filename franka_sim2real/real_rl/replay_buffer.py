from __future__ import annotations

import json
import math
import queue
import sqlite3
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 2
STATE_DIM = 1047
PRIVILEGED_DIM = 4


def real_rl_replay_contract(
    *, base_model_sha256: str, max_residual_m: float, action_scale_m: float
) -> dict[str, Any]:
    return {
        "base_model_sha256": str(base_model_sha256),
        "state_dim": STATE_DIM,
        "privileged_dim": PRIVILEGED_DIM,
        "action_dim": 3,
        "max_residual_m": float(max_residual_m),
        "action_scale_m": float(action_scale_m),
        "replay_schema_version": SCHEMA_VERSION,
        "trainability_contract": "offline_raw_boundary_frames_and_tag_t_le_action_lt_tag_t1",
        "height_contract": "object_z_in_base_minus_initial_object_z_in_base",
        "safety_intervention_norm": "l2_pre_minus_post_normalized_action_4d",
    }


@dataclass
class ReplayTransition:
    episode_id: str
    step_id: int
    run_dir: str
    rollout_step: int
    state: np.ndarray
    next_state: np.ndarray
    base_action: np.ndarray
    next_base_action: np.ndarray
    sac_unit_action: np.ndarray
    residual_action_normalized: np.ndarray
    residual_action_m: np.ndarray
    executed_action: np.ndarray | None
    policy_action_accepted: bool
    action_timestamp: float | None
    pre_safety_action: np.ndarray
    post_safety_action: np.ndarray
    safety_intervened: bool
    intervention_magnitude: float
    reward: float | None
    reach_reward: float | None
    lift_reward: float | None
    success_reward: float | None
    residual_penalty: float | None
    reward_valid: bool
    trainable: bool
    trainable_reason: str
    object_position_t: np.ndarray | None
    object_position_next: np.ndarray | None
    ee_position_t: np.ndarray
    ee_position_next: np.ndarray
    object_relative_to_ee_t: np.ndarray | None
    object_relative_to_ee_next: np.ndarray | None
    initial_object_z_in_base: float
    object_height_t: float | None
    object_height_next: float | None
    distance_t: float | None
    distance_next: float | None
    tag_sequence_t: int | None
    tag_sequence_next: int | None
    tag_age_t: float | None
    tag_age_next: float | None
    reprojection_error_t: float | None
    reprojection_error_next: float | None
    tag_capture_timestamp_t: float | None
    tag_capture_timestamp_next: float | None
    done: bool
    terminated: bool
    truncated: bool
    success: bool
    terminal_reason: str | None
    base_model_sha256: str
    residual_checkpoint_sha256: str | None

    def validate(self) -> None:
        _array(self.state, STATE_DIM, "state")
        _array(self.next_state, STATE_DIM, "next_state")
        for name in ("base_action", "next_base_action"):
            _array(getattr(self, name), 4, name)
        for name in ("pre_safety_action", "post_safety_action"):
            _array(getattr(self, name), 4, name)
        for name in (
            "sac_unit_action", "residual_action_normalized", "residual_action_m",
            "ee_position_t", "ee_position_next",
        ):
            _array(getattr(self, name), 3, name)
        for name in (
            "executed_action", "object_position_t", "object_position_next",
            "object_relative_to_ee_t", "object_relative_to_ee_next",
        ):
            value = getattr(self, name)
            if value is not None:
                _array(value, 4 if name == "executed_action" else 3, name)
        if self.policy_action_accepted:
            if self.executed_action is None or self.action_timestamp is None:
                raise ValueError("Accepted transition requires executed_action and action_timestamp")
            if not math.isfinite(float(self.action_timestamp)):
                raise ValueError("action_timestamp must be finite")
        elif self.executed_action is not None or self.action_timestamp is not None:
            raise ValueError("Unaccepted transition cannot claim execution or an action timestamp")
        for name in ("tag_capture_timestamp_t", "tag_capture_timestamp_next"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when present")
        if not math.isfinite(float(self.initial_object_z_in_base)):
            raise ValueError("initial_object_z_in_base must be finite")
        if self.reward_valid:
            required = (
                self.reward, self.reach_reward, self.lift_reward, self.success_reward,
                self.residual_penalty, self.object_relative_to_ee_t,
                self.object_relative_to_ee_next,
            )
            if any(value is None for value in required):
                raise ValueError("Trainable transition is missing reward/privileged fields")
        if self.trainable != self.reward_valid:
            raise ValueError("Real-RL v2 requires trainable and reward_valid to agree")
        if self.trainable and self.trainable_reason != "ok":
            raise ValueError("Trainable transition must use trainable_reason='ok'")
        if not self.trainable and not self.trainable_reason:
            raise ValueError("Non-trainable transition requires trainable_reason")
        if self.trainable:
            if (
                self.action_timestamp is None
                or self.tag_capture_timestamp_t is None
                or self.tag_capture_timestamp_next is None
            ):
                raise ValueError("Trainable transition requires action and Tag capture timestamps")
            if not (
                self.tag_capture_timestamp_t
                <= self.action_timestamp
                < self.tag_capture_timestamp_next
            ):
                raise ValueError("Trainable transition violates tag_t <= action_time < tag_t1")
        if not math.isfinite(float(self.intervention_magnitude)) or self.intervention_magnitude < 0.0:
            raise ValueError("intervention_magnitude must be finite and non-negative")
        expected_intervention = float(np.linalg.norm(
            np.asarray(self.post_safety_action, dtype=np.float32)
            - np.asarray(self.pre_safety_action, dtype=np.float32)
        ))
        if not math.isclose(
            float(self.intervention_magnitude), expected_intervention,
            rel_tol=1e-5, abs_tol=1e-7,
        ):
            raise ValueError("intervention_magnitude does not match pre/post safety actions")
        if self.safety_intervened != (expected_intervention > 1e-6):
            raise ValueError("safety_intervened does not match pre/post safety actions")
        if self.terminated and self.truncated:
            raise ValueError("A transition cannot be both terminated and truncated")
        if self.done != (self.terminated or self.truncated):
            raise ValueError("done must equal terminated or truncated")
        if not self.base_model_sha256:
            raise ValueError("base_model_sha256 is required")


def _array(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite float32 values")
    return result


def _pack(value: np.ndarray | None, size: int, name: str) -> bytes | None:
    if value is None:
        return None
    return _array(value, size, name).astype("<f4", copy=False).tobytes()


def _unpack(value: bytes | None, size: int) -> np.ndarray | None:
    if value is None:
        return None
    result = np.frombuffer(value, dtype="<f4").copy()
    if result.shape != (size,):
        raise RuntimeError(f"Replay array blob has {result.shape}, expected {(size,)}")
    return result


_ARRAY_COLUMNS = {
    "state": STATE_DIM, "next_state": STATE_DIM,
    "base_action": 4, "next_base_action": 4,
    "sac_unit_action": 3, "residual_action_normalized": 3,
    "residual_action_m": 3, "executed_action": 4,
    "pre_safety_action": 4, "post_safety_action": 4,
    "object_position_t": 3, "object_position_next": 3,
    "ee_position_t": 3, "ee_position_next": 3,
    "object_relative_to_ee_t": 3, "object_relative_to_ee_next": 3,
}
_BOOL_COLUMNS = {
    "policy_action_accepted", "reward_valid", "trainable", "safety_intervened",
    "done", "terminated", "truncated", "success"
}


class ReplayBuffer:
    def __init__(self, path: str | Path, *, create: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(f"Replay Buffer does not exist: {self.path}")
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS replay_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = self.connection.execute(
            "SELECT value FROM replay_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO replay_meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(row[0]) != SCHEMA_VERSION:
            raise RuntimeError(
                f"Replay schema version {row[0]} is incompatible with {SCHEMA_VERSION}"
            )
        columns: list[str] = []
        for item in fields(ReplayTransition):
            name = item.name
            if name in _ARRAY_COLUMNS:
                kind = "BLOB"
            elif name in _BOOL_COLUMNS or name in {
                "step_id", "rollout_step", "tag_sequence_t", "tag_sequence_next"
            }:
                kind = "INTEGER"
            elif name in {
                "episode_id", "run_dir", "terminal_reason", "trainable_reason", "base_model_sha256",
                "residual_checkpoint_sha256",
            }:
                kind = "TEXT"
            else:
                kind = "REAL"
            columns.append(f"{name} {kind}")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS transitions "
            f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {', '.join(columns)})"
        )
        existing_columns = {
            str(item[1])
            for item in self.connection.execute("PRAGMA table_info(transitions)").fetchall()
        }
        if "label_revision" not in existing_columns:
            # Additive v2 migration: label revisions are training metadata, not
            # part of the serialized policy/replay tensor contract.
            self.connection.execute(
                "ALTER TABLE transitions ADD COLUMN "
                "label_revision INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trainable ON transitions(trainable,id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_trainable_label_revision "
            "ON transitions(trainable,label_revision,id)"
        )
        self.connection.commit()

    def append(self, transition: ReplayTransition) -> int:
        transition.validate()
        names = [item.name for item in fields(ReplayTransition)]
        values: list[Any] = []
        for name in names:
            value = getattr(transition, name)
            if name in _ARRAY_COLUMNS:
                value = _pack(value, _ARRAY_COLUMNS[name], name)
            elif name in _BOOL_COLUMNS:
                value = int(bool(value))
            values.append(value)
        placeholders = ",".join("?" for _ in names)
        cursor = self.connection.execute(
            f"INSERT INTO transitions({','.join(names)}) VALUES({placeholders})", values
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM transitions").fetchone()[0])

    def trainable_count(self, *, after_id: int = 0) -> int:
        return int(
            self.connection.execute(
                "SELECT COUNT(*) FROM transitions WHERE id>? AND trainable=1",
                (int(after_id),),
            ).fetchone()[0]
        )

    def trainable_change_counts(
        self, *, after_id: int = 0, after_label_revision: int = 0
    ) -> dict[str, int]:
        new_ids = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM transitions WHERE trainable=1 AND id>?",
                (int(after_id),),
            ).fetchone()[0]
        )
        relabeled = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM transitions "
                "WHERE id<=? AND label_revision>?",
                (int(after_id), int(after_label_revision)),
            ).fetchone()[0]
        )
        return {
            "new_id": new_ids,
            "relabeled": relabeled,
            "total": new_ids + relabeled,
        }

    def valid_count(self, *, after_id: int = 0) -> int:
        return self.trainable_count(after_id=after_id)

    def max_id(self) -> int:
        return int(self.connection.execute("SELECT COALESCE(MAX(id),0) FROM transitions").fetchone()[0])

    def max_label_revision(self) -> int:
        return int(
            self.connection.execute(
                "SELECT COALESCE(MAX(label_revision),0) FROM transitions"
            ).fetchone()[0]
        )

    def base_model_sha256_values(self) -> set[str]:
        return {
            str(row[0])
            for row in self.connection.execute(
                "SELECT DISTINCT base_model_sha256 FROM transitions"
            ).fetchall()
            if row[0]
        }

    def load_trainable(self, *, limit: int | None = None) -> dict[str, np.ndarray]:
        sql = (
            "SELECT * FROM transitions WHERE trainable=1 ORDER BY id"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)
        rows = self.connection.execute(sql, params).fetchall()
        if not rows:
            return {}
        data: dict[str, Any] = {
            "id": np.asarray([row["id"] for row in rows], dtype=np.int64),
            "label_revision": np.asarray(
                [row["label_revision"] for row in rows], dtype=np.int64
            ),
        }
        for name, size in _ARRAY_COLUMNS.items():
            values = [_unpack(row[name], size) for row in rows]
            if any(value is None for value in values):
                if name in {"object_relative_to_ee_t", "object_relative_to_ee_next"}:
                    raise RuntimeError("Trainable Replay transition is missing privileged state")
                continue
            data[name] = np.stack(values).astype(np.float32)
        for name in (
            "reward", "reach_reward", "lift_reward", "success_reward",
            "residual_penalty", "object_height_t", "object_height_next",
            "intervention_magnitude",
        ):
            data[name] = np.asarray([row[name] for row in rows], dtype=np.float32)
        for name in (
            "action_timestamp", "tag_capture_timestamp_t", "tag_capture_timestamp_next",
        ):
            data[name] = np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in (
            "terminated", "truncated", "done", "success", "trainable",
            "reward_valid", "policy_action_accepted", "safety_intervened",
        ):
            data[name] = np.asarray([bool(row[name]) for row in rows], dtype=np.bool_)
        data["trainable_reason"] = np.asarray(
            [str(row["trainable_reason"]) for row in rows], dtype=object
        )
        data["privileged"] = np.concatenate(
            (data["object_relative_to_ee_t"], data["object_height_t"][:, None]), axis=1
        ).astype(np.float32)
        data["next_privileged"] = np.concatenate(
            (data["object_relative_to_ee_next"], data["object_height_next"][:, None]), axis=1
        ).astype(np.float32)
        return data

    def load_valid(self, *, limit: int | None = None) -> dict[str, np.ndarray]:
        return self.load_trainable(limit=limit)

    def get_meta(self, key: str) -> Any | None:
        row = self.connection.execute(
            "SELECT value FROM replay_meta WHERE key=?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row[0])

    def set_meta(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT INTO replay_meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReplayBuffer":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class ReplayWriter:
    """Bounded background writer; failures are surfaced to the control thread."""

    def __init__(self, path: str | Path, queue_size: int = 256) -> None:
        self.path = Path(path)
        self._queue: queue.Queue[ReplayTransition | object] = queue.Queue(maxsize=queue_size)
        self._stop = object()
        self._error: BaseException | None = None
        self._written = 0
        self._thread = threading.Thread(target=self._loop, name="real-rl-replay-writer", daemon=True)
        self._thread.start()

    @property
    def written(self) -> int:
        return self._written

    def _loop(self) -> None:
        try:
            with ReplayBuffer(self.path) as replay:
                while True:
                    item = self._queue.get()
                    try:
                        if item is self._stop:
                            return
                        assert isinstance(item, ReplayTransition)
                        replay.append(item)
                        self._written += 1
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._error = exc

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Replay writer failed: {self._error}") from self._error
        if not self._thread.is_alive():
            raise RuntimeError("Replay writer stopped unexpectedly")

    def submit(self, transition: ReplayTransition) -> None:
        self.check()
        try:
            self._queue.put_nowait(transition)
        except queue.Full as exc:
            raise RuntimeError("Replay writer queue is full; stopping episode") from exc

    def close(self) -> None:
        if self._thread.is_alive():
            self._queue.put(self._stop)
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("Replay writer did not stop")
        if self._error is not None:
            raise RuntimeError(f"Replay writer failed: {self._error}") from self._error

    def __enter__(self) -> "ReplayWriter":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
