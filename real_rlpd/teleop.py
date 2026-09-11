from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

KEYS = frozenset({"w", "s", "a", "d", "j", "k", "u", "i"})


class ExpertStopRequested(KeyboardInterrupt):
    pass


@dataclass(frozen=True)
class ExpertSnapshot:
    xyz_direction: tuple[float, float, float]
    gripper_direction: float
    pressed_keys: tuple[str, ...]
    focused: bool
    sampled_monotonic_ns: int


class ExpertKeyState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._focused = False

    def set_focus(self, focused: bool) -> None:
        with self._lock:
            self._focused = bool(focused)
            if not focused:
                self._pressed.clear()

    def set_key(self, key: str, pressed: bool) -> None:
        if key not in KEYS:
            return
        with self._lock:
            if not self._focused:
                return
            (self._pressed.add if pressed else self._pressed.discard)(key)

    def clear(self) -> None:
        with self._lock:
            self._pressed.clear()

    def snapshot(self) -> ExpertSnapshot:
        with self._lock:
            keys = set(self._pressed)
            focused = self._focused
        xyz = np.asarray([
            float("w" in keys) - float("s" in keys),
            float("a" in keys) - float("d" in keys),
            float("k" in keys) - float("j" in keys),
        ])
        norm = float(np.linalg.norm(xyz))
        if norm > 0.0:
            xyz /= norm
        gripper = float("u" in keys) - float("i" in keys)
        return ExpertSnapshot(
            tuple(float(v) for v in xyz), gripper, tuple(sorted(keys)), focused, time.monotonic_ns()
        )


class PygameExpertKeyboard:
    """全人工 4D 遥操窗口；失焦、ESC 或关闭窗口都会停止会话。"""

    def __init__(self, xyz_speed_m_s: float, gripper_speed_m_s: float) -> None:
        self.xyz_speed_m_s = float(xyz_speed_m_s)
        self.gripper_speed_m_s = float(gripper_speed_m_s)
        self.state = ExpertKeyState()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._initialized = False
        self._ready = False
        self._stop_requested = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rlpd-expert-keyboard", daemon=True)
        self._thread.start()
        with self._condition:
            while not self._initialized and self._error is None:
                self._condition.wait(0.1)
            self.check_health()

    def wait_until_ready(self) -> None:
        with self._condition:
            while not self._ready and not self._stop_requested and self._error is None:
                self._condition.wait(0.1)
        self.check_health()

    def sample(self) -> ExpertSnapshot:
        self.check_health()
        return self.state.snapshot()

    def check_health(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"Expert keyboard failed: {self._error}") from self._error
        if self._stop_requested:
            raise ExpertStopRequested("Expert teleoperation stop requested")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def _run(self) -> None:
        try:
            import pygame
            pygame.init()
            pygame.display.set_caption("Franka RLPD Expert Teleoperation")
            surface = pygame.display.set_mode((760, 260))
            font = pygame.font.Font(None, 28)
            keymap = {
                pygame.K_w: "w", pygame.K_s: "s", pygame.K_a: "a", pygame.K_d: "d",
                pygame.K_j: "j", pygame.K_k: "k", pygame.K_u: "u", pygame.K_i: "i",
            }
            self.state.set_focus(bool(pygame.key.get_focused()))
            with self._condition:
                self._initialized = True
                self._condition.notify_all()
            clock = pygame.time.Clock()
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        self._stop_requested = True
                        self.state.clear()
                    elif event.type == pygame.WINDOWFOCUSGAINED:
                        self.state.set_focus(True)
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        self.state.set_focus(False)
                        self._stop_requested = True
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                        with self._condition:
                            self._ready = True
                            self._condition.notify_all()
                    elif event.type == pygame.KEYDOWN and event.key in keymap:
                        self.state.set_key(keymap[event.key], True)
                    elif event.type == pygame.KEYUP and event.key in keymap:
                        self.state.set_key(keymap[event.key], False)
                surface.fill((20, 24, 30))
                lines = (
                    "FOCUS + ENTER TO ARM" if not self._ready else "EXPERT CONTROL ARMED",
                    "W/S: +/-X   A/D: +/-Y   J/K: +/-Z   U/I: open/close gripper",
                    "No key: hold   Focus loss/ESC/close: stop   Base policy: shadow-only",
                )
                for index, line in enumerate(lines):
                    surface.blit(font.render(line, True, (220, 230, 240)), (24, 24 + 55 * index))
                pygame.display.flip()
                clock.tick(120)
            pygame.quit()
        except BaseException as exc:
            self._error = exc
            with self._condition:
                self._initialized = True
                self._condition.notify_all()


def expert_raw_action(
    snapshot: ExpertSnapshot,
    *,
    xyz_speed_m_s: float,
    gripper_speed_m_s: float,
    policy_frequency_hz: float,
    action_scales: list[float],
) -> np.ndarray:
    scales = np.asarray(action_scales, dtype=np.float32)
    physical = np.asarray([
        *(np.asarray(snapshot.xyz_direction) * float(xyz_speed_m_s) / float(policy_frequency_hz)),
        snapshot.gripper_direction * float(gripper_speed_m_s) / float(policy_frequency_hz),
    ], dtype=np.float32)
    return physical / scales


def expert_step_info(
    expert_requested: np.ndarray,
    expert_limited: np.ndarray,
    snapshot: ExpertSnapshot,
) -> dict[str, Any]:
    """Return policy-independent expert command metadata.

    Base actions and residual targets belong to a policy-specific derived
    Replay and must not define the reusable expert source episode.
    """

    return {
        "expert_requested_action": np.asarray(
            expert_requested, dtype=np.float32
        ).tolist(),
        "expert_limited_action": np.asarray(expert_limited, dtype=np.float32).tolist(),
        "pressed_keys": list(snapshot.pressed_keys),
        "sampled_monotonic_ns": snapshot.sampled_monotonic_ns,
        "focused": snapshot.focused,
    }
