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


def _completed_state(
    state: Any, kind: str | None, target_width: float
) -> SimpleNamespace:
    """Build the state implied by a successfully completed Hand action.

    Reading ``Gripper.state`` can block for hundreds of milliseconds on the
    real Hand.  A successful move/grasp future already confirms the commanded
    width, so querying the state after every 1 mm policy step only serializes
    those otherwise short actions and creates a large target lag.
    """

    return SimpleNamespace(
        width=float(target_width),
        max_width=float(getattr(state, "max_width", 0.08)),
        is_grasped=kind == "grasp",
    )


def _stop_active_gripper(gripper: Any) -> None:
    """Preempt an active Hand action from the isolated owner process.

    franky 1.1.x ``stopAsync()`` first waits for its current async action in
    ``setCurrentFuture()``, so it cannot preempt an active ``moveAsync()``.
    Its documented interruption pattern is the synchronous ``stop()`` call.
    This call remains outside the policy process, so a slow Hand RPC cannot
    retain the policy GIL or stall the arm loop.
    """

    result = gripper.stop()
    if result is False:
        raise RuntimeError("Gripper stop failed")


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
                # Do not perform a blocking Hand state query after a successful
                # 1 mm step. The completed future is the acknowledgement, and
                # its fixed target is the new state. A physical state read is
                # only necessary on failure to distinguish contact from an
                # actual command error.
                state = (
                    _completed_state(state, future_kind, future_target)
                    if success
                    else gripper.state
                )
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
        # A completed finite-width action is already stationary. Calling
        # Gripper.stop() after every session caused successful collections to
        # be reported as failures when that RPC hung. Stop only an action that
        # is still in flight; the owner process keeps the call off the policy
        # loop and its parent enforces the process shutdown timeout.
        future_is_active = False
        if future is not None:
            try:
                future_is_active = not future.wait(0.0)
            except BaseException:
                # If the future cannot report its state, conservatively issue
                # a stop because motion completion is unknown.
                future_is_active = True
        if (
            gripper is not None
            and control_session_active.is_set()
            and future_is_active
        ):
            try:
                _stop_active_gripper(gripper)
            except BaseException as exc:
                _send(
                    connection,
                    ("stop_error", {"message": str(exc)}),
                )
        try:
            connection.close()
        except OSError:
            pass


def _gripper_servo_process_main(
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
    """Track stepwise widths with one continuous Hand motion per direction.

    Franka Hand ``move`` actions have substantial fixed completion overhead.
    Sending the first 1 mm target as a finite action therefore leaves the
    owner unable to act on later 30 Hz targets for hundreds of milliseconds.
    The policy-facing desired width remains stepwise; this executor converts a
    run of same-direction steps into one endpoint motion at the requested
    stepwise velocity and stops it as soon as the logical target holds or
    reverses.
    """

    gripper: Any | None = None
    motion_future: Any | None = None
    motion_kind: str | None = None
    motion_direction = 0
    motion_target = 0.0
    motion_generation = 0
    completed_generation = 0
    latest_generation = 0
    latest_width = 0.0
    latest_signed_speed = 0.0
    latest_published_ns = 0
    state: Any | None = None

    def read_target() -> tuple[float, int, float, int]:
        with target.get_lock():
            return (
                float(target[0]),
                int(target[1]),
                float(target[2]),
                int(target[3]),
            )

    def send_state() -> bool:
        assert state is not None
        return _send(connection, ("state", _state_payload(state)))

    def send_timing(event: str, observed_ns: int, **values: Any) -> None:
        _send(
            connection,
            (
                "timing",
                {"event": event, "monotonic_ns": observed_ns, **values},
            ),
        )

    try:
        if gripper_factory is None:
            from franky import Gripper

            gripper_factory = Gripper
        gripper = gripper_factory(robot_ip)
        state = gripper.state
        latest_width = float(state.width)
        if not _send(connection, ("ready", _state_payload(state))):
            return

        while not stop_event.is_set():
            requested_width, generation, signed_speed, published_ns = read_target()
            target_changed = generation > latest_generation
            if target_changed:
                latest_width = requested_width
                latest_generation = generation
                latest_signed_speed = signed_speed
                latest_published_ns = published_ns

            if motion_future is not None:
                requested_direction = (
                    1 if latest_signed_speed > 1e-12
                    else -1 if latest_signed_speed < -1e-12
                    else 0
                )
                if target_changed:
                    if requested_direction == motion_direction:
                        motion_generation = latest_generation
                    elif motion_kind == "move":
                        stop_requested_ns = time.monotonic_ns()
                        send_timing(
                            "stop_requested",
                            stop_requested_ns,
                            generation=latest_generation,
                            publish_to_request_ms=(
                                stop_requested_ns - latest_published_ns
                            ) / 1e6,
                        )
                        stop_call_started_ns = time.monotonic_ns()
                        stop_success = gripper.stop()
                        stop_call_completed_ns = time.monotonic_ns()
                        send_timing(
                            "stop_returned",
                            stop_call_completed_ns,
                            generation=latest_generation,
                            call_ms=(
                                stop_call_completed_ns - stop_call_started_ns
                            ) / 1e6,
                        )
                        if stop_success is False:
                            raise RuntimeError("Gripper tracking stop failed")

                        motion_future = None
                        motion_kind = None
                        motion_direction = 0
                        motion_target = 0.0
                        # A newer key/policy command may have arrived while the
                        # Hand acknowledged the stop. Continue from only that
                        # newest command; intermediate targets are obsolete.
                        (
                            latest_width,
                            latest_generation,
                            latest_signed_speed,
                            latest_published_ns,
                        ) = read_target()
                        if abs(latest_signed_speed) <= 1e-12:
                            # A physical read is needed only for a stationary
                            # hold. On reversal it would add about 100 ms while
                            # providing no value to the endpoint command.
                            state_started_ns = time.monotonic_ns()
                            state = gripper.state
                            state_completed_ns = time.monotonic_ns()
                            send_timing(
                                "stop_state_completed",
                                state_completed_ns,
                                generation=latest_generation,
                                state_read_ms=(
                                    state_completed_ns - state_started_ns
                                ) / 1e6,
                            )
                            if not send_state():
                                return

                            # The command can change while state() blocks.
                            (
                                latest_width,
                                latest_generation,
                                latest_signed_speed,
                                latest_published_ns,
                            ) = read_target()
                        if abs(latest_signed_speed) <= 1e-12:
                            # On hold, make the measured physical stop point the
                            # next logical accumulator origin.
                            with target.get_lock():
                                if (
                                    int(target[1]) == latest_generation
                                    and abs(float(target[2])) <= 1e-12
                                ):
                                    target[0] = float(state.width)
                                    latest_width = float(state.width)
                            _send(
                                connection,
                                (
                                    "rebase",
                                    {
                                        "generation": latest_generation,
                                        "width": latest_width,
                                    },
                                ),
                            )
                            completed_generation = latest_generation
                            _send(connection, ("idle", completed_generation))
                        continue

                if not motion_future.wait(0.0):
                    stop_event.wait(0.002)
                    continue

                success = bool(motion_future.get())
                if success:
                    state = _completed_state(state, motion_kind, motion_target)
                else:
                    state = gripper.state
                if not send_state():
                    return

                current_width = float(state.width)
                if not success and not (
                    motion_kind == "grasp" and bool(state.is_grasped)
                ):
                    blocked_close = (
                        motion_kind == "move"
                        and motion_direction < 0
                        and current_width > tolerance
                    )
                    if blocked_close:
                        motion_future = gripper.grasp_async(
                            current_width, speed, force
                        )
                        motion_kind = "grasp"
                        motion_target = current_width
                        motion_generation = latest_generation
                        continue
                    raise RuntimeError(
                        "Continuous gripper command failed: "
                        f"kind={motion_kind}, direction={motion_direction}, "
                        f"actual_width={current_width:.6f}, "
                        f"is_grasped={bool(state.is_grasped)}"
                    )

                finished_generation = motion_generation
                motion_future = None
                motion_kind = None
                motion_direction = 0
                motion_target = 0.0
                completed_generation = max(completed_generation, finished_generation)
                _send(connection, ("idle", completed_generation))
                continue

            if latest_generation <= completed_generation:
                stop_event.wait(0.005)
                continue

            requested_direction = (
                1 if latest_signed_speed > 1e-12
                else -1 if latest_signed_speed < -1e-12
                else 0
            )
            assert state is not None
            if requested_direction == 0 or (
                requested_direction < 0 and bool(state.is_grasped)
            ):
                completed_generation = latest_generation
                _send(connection, ("idle", completed_generation))
                continue

            endpoint = float(state.max_width) if requested_direction > 0 else 0.0
            motion_requested_ns = time.monotonic_ns()
            send_timing(
                "motion_requested",
                motion_requested_ns,
                generation=latest_generation,
                direction=requested_direction,
                target_width=endpoint,
                speed=abs(latest_signed_speed),
                publish_to_request_ms=(
                    motion_requested_ns - latest_published_ns
                ) / 1e6,
            )
            motion_call_started_ns = time.monotonic_ns()
            motion_future = gripper.move_async(
                endpoint, abs(latest_signed_speed)
            )
            motion_call_completed_ns = time.monotonic_ns()
            send_timing(
                "move_async_returned",
                motion_call_completed_ns,
                generation=latest_generation,
                call_ms=(motion_call_completed_ns - motion_call_started_ns) / 1e6,
            )
            motion_kind = "move"
            motion_direction = requested_direction
            motion_target = endpoint
            motion_generation = latest_generation
    except BaseException as exc:
        _send(
            connection,
            (
                "error",
                {"message": str(exc), "traceback": traceback.format_exc()},
            ),
        )
    finally:
        if gripper is not None and control_session_active.is_set():
            try:
                if motion_future is not None and not motion_future.wait(0.0):
                    _stop_active_gripper(gripper)
            except BaseException as exc:
                _send(connection, ("stop_error", {"message": str(exc)}))
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
        servo_frequency_hz: float | None = None,
        start_method: str = "spawn",
        startup_timeout_s: float = 10.0,
        gripper_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.speed = float(speed)
        self.tolerance = float(tolerance)
        self.force = float(force)
        self.servo_frequency_hz = (
            None if servo_frequency_hz is None else float(servo_frequency_hz)
        )
        if self.servo_frequency_hz is not None and self.servo_frequency_hz <= 0.0:
            raise ValueError("servo_frequency_hz must be positive")
        self._context = mp.get_context(start_method)
        # width, generation, signed logical width velocity, publish monotonic
        # timestamp. The last two values are zero in legacy finite-target mode.
        self._target = self._context.Array(
            "d", [math.nan, 0.0, 0.0, 0.0], lock=True
        )
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
        self._timing_events: list[dict[str, Any]] = []
        self._closing = False
        self._closed = False
        process_target = (
            _gripper_process_main
            if self.servo_frequency_hz is None
            else _gripper_servo_process_main
        )
        self._process = self._context.Process(
            target=process_target,
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

    def timing_report(self) -> dict[str, Any]:
        """Return bounded owner-process timing diagnostics for run artifacts."""

        with self._lock:
            events = [dict(event) for event in self._timing_events]

        def maximum(event_name: str, field: str) -> float | None:
            values = [
                float(event[field])
                for event in events
                if event.get("event") == event_name
                and event.get(field) is not None
            ]
            return max(values) if values else None

        return {
            "event_count": len(events),
            "motion_request_count": sum(
                event.get("event") == "motion_requested" for event in events
            ),
            "stop_request_count": sum(
                event.get("event") == "stop_requested" for event in events
            ),
            "maximum_publish_to_motion_request_ms": maximum(
                "motion_requested", "publish_to_request_ms"
            ),
            "maximum_move_async_call_ms": maximum(
                "move_async_returned", "call_ms"
            ),
            "maximum_publish_to_stop_request_ms": maximum(
                "stop_requested", "publish_to_request_ms"
            ),
            "maximum_stop_call_ms": maximum("stop_returned", "call_ms"),
            "maximum_stop_state_read_ms": maximum(
                "stop_state_completed", "state_read_ms"
            ),
            "events": events,
        }

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
                elif kind == "rebase":
                    width = float(payload["width"])
                    # Only a zero-velocity command may rebase the accumulator.
                    # Preserve a newer key/policy update that raced with the
                    # physical stop-state response.
                    with self._target.get_lock():
                        if abs(float(self._target[2])) <= 1e-12:
                            self._desired_width = width
                            self._target[0] = width
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
                elif kind == "timing":
                    self._timing_events.append(dict(payload))
                    if len(self._timing_events) > 512:
                        del self._timing_events[:-512]
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
            previous_width = self._desired_width
            self._generation += 1
            generation = self._generation
            self._desired_width = desired_width
            signed_speed = 0.0
            if self.servo_frequency_hz is not None:
                signed_speed = float(
                    np.clip(
                        (desired_width - previous_width) * self.servo_frequency_hz,
                        -self.speed,
                        self.speed,
                    )
                )
        with self._target.get_lock():
            self._target[0] = desired_width
            self._target[1] = float(generation)
            self._target[2] = signed_speed
            self._target[3] = float(time.monotonic_ns())

    def hold_servo_target(self) -> None:
        """Stop continuous tracking without changing the logical width."""

        if self.servo_frequency_hz is None:
            return
        with self._lock:
            self._raise_error_locked()
            if self._closing or self._closed:
                raise RuntimeError("Cannot hold a closed gripper owner process")
            self._generation += 1
            generation = self._generation
            desired_width = self._desired_width
        with self._target.get_lock():
            self._target[0] = desired_width
            self._target[1] = float(generation)
            self._target[2] = 0.0
            self._target[3] = float(time.monotonic_ns())

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
