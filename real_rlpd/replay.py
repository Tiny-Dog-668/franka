from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .learner import ACTION_DIM, PRIVILEGED_DIM, STATE_DIM


SCHEMA_VERSION = 1
ReplayRole = Literal["offline", "online"]


def _reward_independent_contract(contract: dict[str, Any]) -> dict[str, Any]:
    result = dict(contract)
    result.pop("reward_kind", None)
    result.pop("reward", None)
    return result


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _pack(value: Any, size: int, name: str) -> bytes:
    return _vector(value, size, name).astype("<f4", copy=False).tobytes()


def _unpack(value: bytes, size: int, name: str) -> np.ndarray:
    result = np.frombuffer(value, dtype="<f4").copy()
    if result.shape != (size,):
        raise RuntimeError(f"Replay {name} has shape {result.shape}, expected {(size,)}")
    return result


@dataclass(frozen=True)
class Transition:
    episode_id: str
    step_id: int
    state: np.ndarray
    next_state: np.ndarray
    action: np.ndarray
    base_limited_action: np.ndarray
    executed_action: np.ndarray
    ee_position: np.ndarray
    next_ee_position: np.ndarray
    initial_object_z: float
    privileged: np.ndarray | None
    next_privileged: np.ndarray | None
    reward: float | None
    terminated: bool
    truncated: bool
    success: bool
    accepted: bool
    trainable: bool
    trainable_reason: str
    action_timestamp: float | None
    run_dir: str
    checkpoint_sha256: str | None = None

    def validate(self) -> None:
        for name, size in (
            ("state", STATE_DIM), ("next_state", STATE_DIM),
            ("action", ACTION_DIM), ("base_limited_action", ACTION_DIM),
            ("executed_action", ACTION_DIM), ("ee_position", 3),
            ("next_ee_position", 3),
        ):
            _vector(getattr(self, name), size, name)
        for name in ("privileged", "next_privileged"):
            value = getattr(self, name)
            if value is not None:
                _vector(value, PRIVILEGED_DIM, name)
        if self.trainable:
            if self.reward is None or self.privileged is None or self.next_privileged is None:
                raise ValueError("Trainable RLPD transition lacks reward or privileged state")
            if self.trainable_reason != "ok":
                raise ValueError("Trainable transition must use reason 'ok'")
        elif not self.trainable_reason:
            raise ValueError("Non-trainable transition requires a reason")
        if self.terminated and self.truncated:
            raise ValueError("Transition cannot be both terminated and truncated")


class ReplayBuffer:
    def __init__(self, path: str | Path, role: ReplayRole, *, create: bool = True) -> None:
        if role not in {"offline", "online"}:
            raise ValueError("Replay role must be offline or online")
        self.path = Path(path).expanduser().resolve()
        self.role = role
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        elif not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        self.connection.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        existing = self.get_meta("schema_version")
        if existing is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
            self.set_meta("role", self.role)
        elif int(existing) != SCHEMA_VERSION or self.get_meta("role") != self.role:
            raise RuntimeError("RLPD replay schema or role is incompatible")
        self.connection.execute(
            """CREATE TABLE IF NOT EXISTS transitions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id TEXT NOT NULL, step_id INTEGER NOT NULL,
            state BLOB NOT NULL, next_state BLOB NOT NULL, action BLOB NOT NULL,
            base_limited_action BLOB NOT NULL, executed_action BLOB NOT NULL,
            ee_position BLOB NOT NULL, next_ee_position BLOB NOT NULL,
            initial_object_z REAL NOT NULL,
            privileged BLOB, next_privileged BLOB, reward REAL,
            terminated INTEGER NOT NULL, truncated INTEGER NOT NULL, success INTEGER NOT NULL,
            accepted INTEGER NOT NULL, trainable INTEGER NOT NULL,
            trainable_reason TEXT NOT NULL, action_timestamp REAL,
            run_dir TEXT NOT NULL, checkpoint_sha256 TEXT,
            UNIQUE(episode_id,step_id))"""
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS idx_trainable ON transitions(trainable,id)")
        self.connection.commit()

    def set_meta(self, key: str, value: Any) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            (str(key), json.dumps(value, sort_keys=True)),
        )
        self.connection.commit()

    def get_meta(self, key: str) -> Any | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (str(key),)).fetchone()
        return None if row is None else json.loads(row[0])

    def assert_contract(self, contract: dict[str, Any]) -> None:
        existing = self.get_meta("contract")
        if existing is None:
            self.set_meta("contract", contract)
        elif existing != contract:
            raise ValueError(f"RLPD replay contract mismatch: {existing!r}")

    def _insert(self, item: Transition) -> int:
        item.validate()
        cursor = self.connection.execute(
            """INSERT INTO transitions(
            episode_id,step_id,state,next_state,action,base_limited_action,executed_action,
            ee_position,next_ee_position,initial_object_z,
            privileged,next_privileged,reward,terminated,truncated,success,accepted,trainable,
            trainable_reason,action_timestamp,run_dir,checkpoint_sha256)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                item.episode_id, item.step_id,
                _pack(item.state, STATE_DIM, "state"),
                _pack(item.next_state, STATE_DIM, "next_state"),
                _pack(item.action, ACTION_DIM, "action"),
                _pack(item.base_limited_action, ACTION_DIM, "base_limited_action"),
                _pack(item.executed_action, ACTION_DIM, "executed_action"),
                _pack(item.ee_position, 3, "ee_position"),
                _pack(item.next_ee_position, 3, "next_ee_position"),
                float(item.initial_object_z),
                None if item.privileged is None else _pack(item.privileged, PRIVILEGED_DIM, "privileged"),
                None if item.next_privileged is None else _pack(item.next_privileged, PRIVILEGED_DIM, "next_privileged"),
                item.reward, int(item.terminated), int(item.truncated), int(item.success),
                int(item.accepted), int(item.trainable), item.trainable_reason,
                item.action_timestamp, item.run_dir, item.checkpoint_sha256,
            ),
        )
        return int(cursor.lastrowid)

    def append(self, item: Transition) -> int:
        row_id = self._insert(item)
        self.connection.commit()
        return row_id

    def append_many(self, items: list[Transition]) -> list[int]:
        try:
            row_ids = [self._insert(item) for item in items]
            self.connection.commit()
            return row_ids
        except BaseException:
            self.connection.rollback()
            raise

    def count(self, *, trainable_only: bool = False) -> int:
        where = " WHERE trainable=1" if trainable_only else ""
        return int(self.connection.execute("SELECT COUNT(*) FROM transitions" + where).fetchone()[0])

    def max_id(self) -> int:
        return int(self.connection.execute("SELECT COALESCE(MAX(id),0) FROM transitions").fetchone()[0])

    def load(self, *, trainable_only: bool = True) -> dict[str, np.ndarray]:
        where = " WHERE trainable=1" if trainable_only else ""
        rows = self.connection.execute("SELECT * FROM transitions" + where + " ORDER BY id").fetchall()
        if not rows:
            return {}
        return {
            "id": np.asarray([row["id"] for row in rows], dtype=np.int64),
            "state": np.stack([_unpack(row["state"], STATE_DIM, "state") for row in rows]),
            "next_state": np.stack([_unpack(row["next_state"], STATE_DIM, "next_state") for row in rows]),
            "action": np.stack([_unpack(row["action"], ACTION_DIM, "action") for row in rows]),
            "privileged": np.stack([_unpack(row["privileged"], PRIVILEGED_DIM, "privileged") for row in rows]),
            "next_privileged": np.stack([_unpack(row["next_privileged"], PRIVILEGED_DIM, "next_privileged") for row in rows]),
            "reward": np.asarray([row["reward"] for row in rows], dtype=np.float32),
            "terminated": np.asarray([row["terminated"] for row in rows], dtype=np.float32),
            "success": np.asarray([row["success"] for row in rows], dtype=np.bool_),
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ReplayBuffer":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


def clone_for_reward_relabel(
    source_path: str | Path,
    target_path: str | Path,
    role: ReplayRole,
    target_contract: dict[str, Any],
) -> dict[str, Any]:
    """复制 policy 派生字段并清空标签，供新 reward 契约重新标注。"""

    source = Path(source_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if source == target:
        raise ValueError("Reward relabel source and target Replay paths must differ")
    if target.exists():
        raise FileExistsError(
            f"Reward relabel target already exists; refusing to overwrite: {target}"
        )
    transition_count = 0
    episodes: set[str] = set()

    def pending_transition(row: sqlite3.Row) -> Transition:
        return Transition(
            episode_id=str(row["episode_id"]),
            step_id=int(row["step_id"]),
            state=_unpack(row["state"], STATE_DIM, "state"),
            next_state=_unpack(row["next_state"], STATE_DIM, "next_state"),
            action=_unpack(row["action"], ACTION_DIM, "action"),
            base_limited_action=_unpack(
                row["base_limited_action"], ACTION_DIM, "base_limited_action"
            ),
            executed_action=_unpack(
                row["executed_action"], ACTION_DIM, "executed_action"
            ),
            ee_position=_unpack(row["ee_position"], 3, "ee_position"),
            next_ee_position=_unpack(
                row["next_ee_position"], 3, "next_ee_position"
            ),
            initial_object_z=float(row["initial_object_z"]),
            privileged=None,
            next_privileged=None,
            reward=None,
            terminated=False,
            truncated=bool(row["truncated"]),
            success=False,
            accepted=bool(row["accepted"]),
            trainable=False,
            trainable_reason=(
                "offline_apriltag_pending"
                if bool(row["accepted"])
                else "action_not_accepted"
            ),
            action_timestamp=(
                None
                if row["action_timestamp"] is None
                else float(row["action_timestamp"])
            ),
            run_dir=str(row["run_dir"]),
            checkpoint_sha256=row["checkpoint_sha256"],
        )

    try:
        with ReplayBuffer(source, role, create=False) as old:
            source_contract = old.get_meta("contract")
            if not isinstance(source_contract, dict):
                raise ValueError("Source Replay has no policy contract")
            if _reward_independent_contract(
                source_contract
            ) != _reward_independent_contract(target_contract):
                raise ValueError(
                    "Source Replay policy/action contract does not match the target config"
                )
            with ReplayBuffer(target, role) as new:
                new.assert_contract(target_contract)
                cursor = old.connection.execute(
                    "SELECT * FROM transitions ORDER BY id"
                )
                while rows := cursor.fetchmany(256):
                    transitions = [pending_transition(row) for row in rows]
                    new.append_many(transitions)
                    transition_count += len(transitions)
                    episodes.update(item.run_dir for item in transitions)
    except BaseException:
        for path in (
            target,
            target.with_name(target.name + "-wal"),
            target.with_name(target.name + "-shm"),
        ):
            path.unlink(missing_ok=True)
        raise
    return {
        "source": str(source),
        "target": str(target),
        "role": role,
        "transitions": transition_count,
        "episodes": sorted(episodes),
    }
