from __future__ import annotations

import math
import multiprocessing as mp
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


def _state_payload(state: Any) -> dict[str, Any]:
    return {
        "width": float(state.width),
        "max_width": float(state.max_width),
        "is_grasped": bool(state.is_grasped),
    }


def _send(connection: Any, message: tuple[str, Any]) -> bool:
    try:
        connection.send(message)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _gripper_process_main(
    robot_ip: str,
    speed: float,
    tolerance: float,
    force: float,
    target: Any,
    stop_event: Any,
    control_session_active: Any,
    connection: Any,
    gripper_factory: Callable[[str], Any] | None,
) -> None:
    """Own every franky Hand call in a process outside the policy GIL."""

    gripper: Any | None = None
    future: Any | None = None
    future_kind: str | None = None
    future_target = 0.0
    future_start_width = 0.0
    future_generation = 0
    observed_generation = 0
    operation_started = False
    state: Any | None = None
    try:
        if gripper_factory is None:
            from franky import Gripper

            gripper_factory = Gripper
        gripper = gripper_factory(robot_ip)
        state = gripper.state
        if not _send(connection, ("ready", _state_payload(state))):
            return

        while not stop_event.is_set():
            if future is not None:
                if not future.wait(0.0):
                    stop_event.wait(0.002)
                    continue

                success = bool(future.get())
                state = gripper.state
                if not _send(connection, ("state", _state_payload(state))):
                    return
                current_width = float(state.width)
                if not success and not (
                    future_kind == "grasp" and bool(state.is_grasped)
                ):
                    blocked_close = (
                        future_kind == "move"
                        and future_target < future_start_width - tolerance
                        and current_width > future_target + tolerance
                    )
                    if blocked_close:
                        # Contact while closing is expected. Keep all follow-up
                        # Hand calls in this process and switch to force control.
                        future = gripper.grasp_async(current_width, speed, force)
                        future_kind = "grasp"
                        future_target = current_width
                        future_start_width = current_width
                        operation_started = True
                        continue
                    raise RuntimeError(
                        "Asynchronous gripper command failed: "
                        f"kind={future_kind}, target_width={future_target}, "
                        f"actual_width={current_width:.6f}, "
                        f"is_grasped={bool(state.is_grasped)}"
                    )

                completed_generation = future_generation
                future = None
                future_kind = None
                _send(connection, ("idle", completed_generation))
                continue

            with target.get_lock():
                desired_width = float(target[0])
                generation = int(target[1])
            if generation <= observed_generation:
                stop_event.wait(0.005)
                continue
            observed_generation = generation

            assert state is not None
            current_width = float(state.width)
            if (
                bool(state.is_grasped)
                and desired_width <= current_width + tolerance
            ) or abs(desired_width - current_width) <= tolerance:
                _send(connection, ("idle", generation))
                continue

            future = gripper.move_async(desired_width, speed)
            future_kind = "move"
            future_target = desired_width
            future_start_width = current_width
            future_generation = generation
            operation_started = True
    except BaseException as exc:
        _send(
            connection,
            (
                "error",
                {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            ),
        )
    finally:
        if gripper is not None and (
            control_session_active.is_set()
            or operation_started
            or future is not None
        ):
            try:
                gripper.stop()
            except BaseException as exc:
                _send(
                    connection,
                    ("stop_error", {"message": str(exc)}),
                )
        try:
            connection.close()
        except OSError:
            pass


class ProcessGripperQueue:
    """Latest-target Franka Hand queue backed by a separate owner process.

    The policy process never invokes franky/libfranka Hand APIs. This is
    stronger than a Python thread: a native Hand binding that retains the GIL
    cannot delay the arm policy loop.
    """

    def __init__(
        self,
        robot_ip: str,
        speed: float,
        tolerance: float,
        force: float = 20.0,
        *,
        start_method: str = "spawn",
        startup_timeout_s: float = 10.0,
        gripper_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.speed = float(speed)
        self.tolerance = float(tolerance)
        self.force = float(force)
        self._context = mp.get_context(start_method)
        self._target = self._context.Array("d", [math.nan, 0.0], lock=True)
        self._stop_event = self._context.Event()
        self._control_session_active = self._context.Event()
        receiver, sender = self._context.Pipe(duplex=False)
        self._receiver = receiver
        self._lock = threading.Condition(threading.RLock())
        self._cached_state: Any | None = None
        self._desired_width = 0.0
        self._generation = 0
        self._completed_generation = 0
        self._error: str | None = None
        self._stop_error: str | None = None
        self._closing = False
        self._closed = False
        self._process = self._context.Process(
            target=_gripper_process_main,
            name="franka-gripper-process",
            args=(
                robot_ip,
                self.speed,
                self.tolerance,
                self.force,
                self._target,
                self._stop_event,
                self._control_session_active,
                sender,
                gripper_factory,
            ),
            daemon=True,
        )
        self._process.start()
        sender.close()

        deadline = time.monotonic() + float(startup_timeout_s)
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out starting gripper owner process")
                if self._receiver.poll(min(0.05, remaining)):
                    kind, payload = self._receiver.recv()
                    if kind == "ready":
                        self._cached_state = SimpleNamespace(**payload)
                        self._desired_width = float(self._cached_state.width)
                        break
                    if kind == "error":
                        raise RuntimeError(
                            "Gripper owner process failed during startup: "
                            + str(payload["message"])
                        )
                if not self._process.is_alive():
                    raise RuntimeError(
                        "Gripper owner process exited before reporting its state"
                    )
        except BaseException:
            self._stop_event.set()
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._receiver.close()
            raise

        self._listener = threading.Thread(
            target=self._listen,
            name="franka-gripper-state-listener",
            daemon=True,
        )
        self._listener.start()

    @property
    def cached_state(self) -> Any:
        with self._lock:
            assert self._cached_state is not None
            return self._cached_state

    @property
    def desired_width(self) -> float:
        with self._lock:
            return self._desired_width

    def _listen(self) -> None:
        while True:
            try:
                if not self._receiver.poll(0.05):
                    with self._lock:
                        if self._closing and not self._process.is_alive():
                            return
                    continue
                kind, payload = self._receiver.recv()
            except (EOFError, OSError):
                with self._lock:
                    if not self._closing and self._error is None:
                        self._error = "Gripper owner process closed its state channel"
                    self._lock.notify_all()
                return
            with self._lock:
                if kind == "state":
                    self._cached_state = SimpleNamespace(**payload)
                elif kind == "idle":
                    self._completed_generation = max(
                        self._completed_generation, int(payload)
                    )
                elif kind == "error":
                    detail = str(payload.get("message", "unknown error"))
                    trace = str(payload.get("traceback", "")).strip()
                    self._error = detail + (f"\n{trace}" if trace else "")
                elif kind == "stop_error":
                    self._stop_error = str(payload.get("message", "unknown error"))
                self._lock.notify_all()

    def _raise_error_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                "Gripper owner process failed: " + self._error
            )
        if not self._closing and not self._process.is_alive():
            raise RuntimeError(
                f"Gripper owner process exited unexpectedly "
                f"(exitcode={self._process.exitcode})"
            )

    def check(self) -> None:
        with self._lock:
            self._raise_error_locked()
            if self._closed:
                raise RuntimeError("Gripper owner process is closed")

    def mark_control_session_active(self) -> None:
        with self._lock:
            self._raise_error_locked()
            if self._closing or self._closed:
                raise RuntimeError("Cannot activate a closed gripper owner process")
            self._control_session_active.set()

    def command(self, width: float) -> None:
        # The 30 Hz policy path only updates two shared doubles. It never uses
        # a Pipe queue and therefore cannot build a stale command backlog.
        with self._lock:
            self._raise_error_locked()
            if self._closing or self._closed:
                raise RuntimeError("Cannot command a closed gripper owner process")
            assert self._cached_state is not None
            max_width = float(getattr(self._cached_state, "max_width", 0.08))
            if not math.isfinite(max_width) or max_width <= 0.0:
                max_width = 0.08
            desired_width = float(np.clip(width, 0.0, max_width))
            self._generation += 1
            generation = self._generation
            self._desired_width = desired_width
        with self._target.get_lock():
            self._target[0] = desired_width
            self._target[1] = float(generation)

    def poll(self) -> None:
        self.check()

    def wait_idle(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + float(timeout_s)
        with self._lock:
            generation = self._generation
            while self._completed_generation < generation:
                self._raise_error_locked()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("Timed out waiting for gripper owner process")
                self._lock.wait(timeout=min(0.02, remaining))
            self._raise_error_locked()

    def stop(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._closing:
                while not self._closed:
                    self._lock.wait(timeout=0.05)
                return
            self._closing = True
            self._stop_event.set()
            self._lock.notify_all()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
            process_error = "Gripper owner process did not stop within 3 seconds"
        else:
            process_error = None
        self._listener.join(timeout=1.0)
        try:
            self._receiver.close()
        except OSError:
            pass
        with self._lock:
            worker_error = self._error
            stop_error = self._stop_error
            self._closed = True
            self._lock.notify_all()
        if process_error is not None:
            raise RuntimeError(process_error)
        if stop_error is not None:
            raise RuntimeError("Failed to stop gripper: " + stop_error)
        if worker_error is not None:
            raise RuntimeError("Gripper owner process failed: " + worker_error)
