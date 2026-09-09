from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_HIL_SPEED_M_S = 0.05
HIL_KEYS = frozenset({"space", "w", "s", "a", "d", "r", "f"})


class HILStopRequested(KeyboardInterrupt):
    """Signal that the operator requested a normal HIL session stop."""


@dataclass(frozen=True)
class HILSettings:
    enabled: bool = False
    speed_m_s: float = DEFAULT_HIL_SPEED_M_S

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("HIL enabled must be a boolean")
        if (
            isinstance(self.speed_m_s, bool)
            or not isinstance(self.speed_m_s, (int, float))
            or not math.isfinite(float(self.speed_m_s))
            or float(self.speed_m_s) <= 0.0
        ):
            raise ValueError("--hil-speed-m-s must be finite and positive")


@dataclass(frozen=True)
class HILInputSnapshot:
    intervention: bool
    direction_xyz: tuple[float, float, float]
    pressed_keys: tuple[str, ...]
    sampled_monotonic_ns: int
    focused: bool


@dataclass(frozen=True)
class HILStepData:
    intervention: bool
    base_action: np.ndarray
    human_action: np.ndarray | None
    residual_target_xyz: np.ndarray
    input_snapshot: HILInputSnapshot


class HILKeyState:
    """Thread-safe key state with no dependency on the window implementation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._focused = False

    def set_focus(self, focused: bool) -> None:
        with self._lock:
            self._focused = bool(focused)
            if not self._focused:
                self._pressed.clear()

    def set_key(self, key: str, pressed: bool) -> None:
        normalized = str(key).lower()
        if normalized not in HIL_KEYS:
            return
        with self._lock:
            if not self._focused:
                return
            if pressed:
                self._pressed.add(normalized)
            else:
                self._pressed.discard(normalized)

    def clear(self) -> None:
        with self._lock:
            self._pressed.clear()

    def snapshot(self, sampled_monotonic_ns: int | None = None) -> HILInputSnapshot:
        with self._lock:
            pressed = set(self._pressed)
            focused = self._focused
        direction = np.asarray(
            [
                float("w" in pressed) - float("s" in pressed),
                float("a" in pressed) - float("d" in pressed),
                float("r" in pressed) - float("f" in pressed),
            ],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if norm > 0.0:
            direction /= norm
        return HILInputSnapshot(
            intervention=bool(focused and "space" in pressed),
            direction_xyz=tuple(float(value) for value in direction),
            pressed_keys=tuple(sorted(pressed)),
            sampled_monotonic_ns=(
                time.monotonic_ns()
                if sampled_monotonic_ns is None
                else int(sampled_monotonic_ns)
            ),
            focused=focused,
        )


def human_normalized_xyz(
    snapshot: HILInputSnapshot,
    *,
    speed_m_s: float,
    policy_frequency_hz: float,
    action_scales: list[float] | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    """Convert the held base-frame direction to the policy's normalized XYZ contract."""

    values = (speed_m_s, policy_frequency_hz)
    if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
        raise ValueError("HIL speed and policy frequency must be finite and positive")
    scales = np.asarray(action_scales, dtype=np.float64).reshape(-1)
    if scales.shape[0] < 3 or not np.all(np.isfinite(scales[:3])) or np.any(scales[:3] <= 0.0):
        raise ValueError("HIL requires three finite positive Cartesian action scales")
    direction = np.asarray(snapshot.direction_xyz, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("HIL direction must contain three finite values")
    physical_delta = direction * (float(speed_m_s) / float(policy_frequency_hz))
    return np.asarray(physical_delta / scales[:3], dtype=np.float32)


class PygameHILKeyboard:
    """Focused Pygame window whose thread only publishes current key state."""

    def __init__(self, settings: HILSettings) -> None:
        settings.validate()
        if not settings.enabled:
            raise ValueError("PygameHILKeyboard requires enabled HIL settings")
        self.settings = settings
        self.state = HILKeyState()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._initialized = False
        self._ready = False
        self._stop_requested = False
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("HIL keyboard has already been started")
        self._thread = threading.Thread(
            target=self._run,
            name="hil-pygame-keyboard",
            daemon=True,
        )
        self._thread.start()
        with self._condition:
            while not self._initialized and self._error is None:
                self._condition.wait(timeout=0.1)
            self._raise_if_unhealthy_locked()

    def wait_until_ready(self) -> None:
        with self._condition:
            while not self._ready and not self._stop_requested and self._error is None:
                self._condition.wait(timeout=0.1)
            self._raise_if_unhealthy_locked()
            if self._stop_requested:
                raise HILStopRequested("HIL keyboard window was closed before arming")

    def sample(self) -> HILInputSnapshot:
        self.check_health()
        return self.state.snapshot()

    def check_health(self) -> None:
        with self._condition:
            self._raise_if_unhealthy_locked()
            if self._stop_requested:
                raise HILStopRequested("HIL stop requested by the operator")

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("HIL keyboard thread did not stop")

    def _raise_if_unhealthy_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"HIL keyboard failed: {self._error}") from self._error

    def _run(self) -> None:
        try:
            try:
                import pygame
            except ImportError as exc:
                raise RuntimeError(
                    "Pygame is required for --hil. Install it with "
                    "`.venv/bin/python -m pip install pygame`."
                ) from exc

            pygame.init()
            pygame.display.set_caption("Franka HIL Residual BC")
            surface = pygame.display.set_mode((720, 260))
            font = pygame.font.Font(None, 30)
            small_font = pygame.font.Font(None, 24)
            key_names: dict[int, str] = {
                pygame.K_SPACE: "space",
                pygame.K_w: "w",
                pygame.K_s: "s",
                pygame.K_a: "a",
                pygame.K_d: "d",
                pygame.K_r: "r",
                pygame.K_f: "f",
            }
            self.state.set_focus(bool(pygame.key.get_focused()))
            with self._condition:
                self._initialized = True
                self._condition.notify_all()

            clock = pygame.time.Clock()
            while not self._stop.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        with self._condition:
                            self._stop_requested = True
                            self._condition.notify_all()
                        self.state.clear()
                    elif event.type == pygame.WINDOWFOCUSGAINED:
                        self.state.set_focus(True)
                    elif event.type == pygame.WINDOWFOCUSLOST:
                        self.state.set_focus(False)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            with self._condition:
                                self._stop_requested = True
                                self._condition.notify_all()
                            self.state.clear()
                        elif event.key == pygame.K_RETURN and pygame.key.get_focused():
                            with self._condition:
                                self._ready = True
                                self._condition.notify_all()
                        elif event.key in key_names:
                            self.state.set_key(key_names[event.key], True)
                    elif event.type == pygame.KEYUP and event.key in key_names:
                        self.state.set_key(key_names[event.key], False)

                snapshot = self.state.snapshot()
                surface.fill((20, 24, 30))
                status = "ARMED" if self._ready else "FOCUS THIS WINDOW, THEN PRESS ENTER"
                status_color = (80, 220, 120) if self._ready else (255, 190, 80)
                surface.blit(font.render(status, True, status_color), (24, 22))
                surface.blit(
                    small_font.render(
                        "Hold SPACE to intervene | W/S: +/-X  A/D: +/-Y  R/F: +/-Z",
                        True,
                        (230, 230, 230),
                    ),
                    (24, 72),
                )
                surface.blit(
                    small_font.render(
                        "Release SPACE: base policy | ESC or close: safe stop",
                        True,
                        (200, 205, 215),
                    ),
                    (24, 108),
                )
                surface.blit(
                    small_font.render(
                        f"speed={self.settings.speed_m_s:.3f} m/s  "
                        f"intervention={snapshot.intervention}  "
                        f"keys={','.join(snapshot.pressed_keys) or 'none'}",
                        True,
                        (130, 190, 255),
                    ),
                    (24, 164),
                )
                pygame.display.flip()
                clock.tick(120)
                with self._condition:
                    if self._stop_requested:
                        break
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._initialized = True
                self._condition.notify_all()
        finally:
            self.state.clear()
            try:
                pygame_module: Any = locals().get("pygame")
                if pygame_module is not None:
                    pygame_module.quit()
            except Exception:
                pass


__all__ = (
    "DEFAULT_HIL_SPEED_M_S",
    "HILInputSnapshot",
    "HILKeyState",
    "HILSettings",
    "HILStepData",
    "HILStopRequested",
    "PygameHILKeyboard",
    "human_normalized_xyz",
)
