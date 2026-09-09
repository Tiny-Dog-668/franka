#!/usr/bin/env python3
"""Keyboard teleoperation and time-aligned real-robot trajectory recording.

The arm is controlled in small, *blocking* Cartesian increments.  This makes
each keyboard press an explicit, reproducible action and prevents a held key
from generating an unbounded velocity command.  A D435 capture thread records
RGB continuously while the terminal loop records robot state while idle and
immediately before/after every command.

All host-side records use ``host_monotonic_ns`` as their common time base.
RealSense device timestamps are also retained, but are deliberately not used
to join streams because their clock origin is device-dependent.

Run from a real terminal (not an IDE output panel):

  source .venv/bin/activate
  export FRANKA_ROBOT_IP=172.16.0.2
  python scripts/robot/teleop_record_real_trajectory.py --ip "$FRANKA_ROBOT_IP"
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs"
SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Teleoperate Franka with the keyboard and record time-aligned D435 RGB, "
            "robot states, and command actions."
        )
    )
    parser.add_argument(
        "--ip",
        default=os.environ.get("FRANKA_ROBOT_IP"),
        help="Robot IP address. Falls back to FRANKA_ROBOT_IP.",
    )
    parser.add_argument(
        "--camera-serial", default="215322076207", help="D435 serial number; empty selects the first camera."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--step-m", type=float, default=0.005, help="Translation increment per key press, in m.")
    parser.add_argument("--yaw-step-deg", type=float, default=2.0, help="Yaw increment per key press, in degrees.")
    parser.add_argument("--speed", type=float, default=0.05, help="Franka relative dynamics factor in [0, 1].")
    parser.add_argument("--gripper-step-m", type=float, default=0.005)
    parser.add_argument("--gripper-speed", type=float, default=0.03)
    parser.add_argument("--realtime", choices=("ignore", "enforce"), default="ignore")
    parser.add_argument(
        "--workspace-min", nargs=3, type=float, default=[0.2, -0.3, 0.0], metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--workspace-max", nargs=3, type=float, default=[0.65, 0.3, 0.45], metavar=("X", "Y", "Z")
    )
    parser.add_argument("--state-rate-hz", type=float, default=20.0, help="Robot state sampling rate while no arm motion is active.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Parent directory for a timestamped run.")
    parser.add_argument("--run-name", default="real_teleop_trajectory")
    parser.add_argument("--no-gripper", action="store_true", help="Do not connect to or command the gripper.")
    parser.add_argument("--preview-only", action="store_true", help="Record camera, state, and keys but never send arm/gripper commands.")
    parser.add_argument("--yes", action="store_true", help="Skip the startup safety confirmation.")
    return parser


def _clock_record(session_start_monotonic_ns: int, *, timestamp_ns: int | None = None) -> dict[str, int | float]:
    monotonic_ns = time.monotonic_ns() if timestamp_ns is None else timestamp_ns
    return {
        "host_monotonic_ns": monotonic_ns,
        "elapsed_s": (monotonic_ns - session_start_monotonic_ns) / 1e9,
        "host_unix_ns": time.time_ns(),
    }


def _active_error_names(errors: Any) -> list[str]:
    return sorted(
        name
        for name in dir(errors)
        if not name.startswith("_") and isinstance(getattr(errors, name), bool) and getattr(errors, name)
    )


def _mode_name(mode: Any) -> str:
    return str(mode).split(".")[-1]


def _rpy_quaternion_xyzw(yaw_deg: float) -> list[float]:
    yaw_rad = math.radians(yaw_deg)
    return [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)]


def _clamp_delta(current: list[float], delta: list[float], lower: list[float], upper: list[float]) -> list[float]:
    """Clamp a requested Cartesian translation to the configured base-frame workspace."""
    return [min(max(current[i] + delta[i], lower[i]), upper[i]) - current[i] for i in range(3)]


def _write_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


@dataclass
class FramePacket:
    index: int
    image: np.ndarray
    record: dict[str, Any]


class RealSenseRecorder:
    """Capture RGB in one thread and save PNGs in another to avoid disk I/O stalls."""

    def __init__(self, args: argparse.Namespace, run_dir: Path, session_start_monotonic_ns: int) -> None:
        self.args = args
        self.rgb_dir = run_dir / "rgb"
        self.frames_path = run_dir / "frames.jsonl"
        self.session_start_monotonic_ns = session_start_monotonic_ns
        self._pipeline: Any = None
        self._profile: Any = None
        self._stop = threading.Event()
        self._packets: queue.Queue[FramePacket | None] = queue.Queue(maxsize=max(30, args.fps * 4))
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._exception: BaseException | None = None
        self._index = 0
        self.dropped_frames = 0
        self.camera_metadata: dict[str, Any] = {}

    def start(self) -> None:
        import pyrealsense2 as rs

        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        config = rs.config()
        if self.args.camera_serial:
            config.enable_device(self.args.camera_serial)
        config.enable_stream(rs.stream.color, self.args.width, self.args.height, rs.format.rgb8, self.args.fps)
        self._pipeline = rs.pipeline()
        self._profile = self._pipeline.start(config)
        stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = stream.get_intrinsics()
        self.camera_metadata = {
            "serial": self.args.camera_serial or None,
            "stream": {"width": self.args.width, "height": self.args.height, "fps": self.args.fps, "format": "RGB8"},
            "intrinsics": {
                "fx_px": float(intrinsics.fx), "fy_px": float(intrinsics.fy),
                "cx_px": float(intrinsics.ppx), "cy_px": float(intrinsics.ppy),
                "distortion_model": str(intrinsics.model).split(".")[-1],
                "distortion_coefficients": [float(value) for value in intrinsics.coeffs],
            },
        }
        for _ in range(self.args.warmup_frames):
            self._pipeline.wait_for_frames()
        self._writer_thread = threading.Thread(target=self._writer_loop, name="d435-png-writer", daemon=True)
        self._capture_thread = threading.Thread(target=self._capture_loop, name="d435-capture", daemon=True)
        self._writer_thread.start()
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(500)
                color = frames.get_color_frame()
                if not color:
                    continue
                received_ns = time.monotonic_ns()
                image = np.asanyarray(color.get_data(), dtype=np.uint8).copy()
                index = self._index
                self._index += 1
                record: dict[str, Any] = {
                    "frame_index": index,
                    **_clock_record(self.session_start_monotonic_ns, timestamp_ns=received_ns),
                    "realsense_timestamp_ms": float(color.get_timestamp()),
                    # Python RealSense bindings expose this as a property in
                    # current releases (older bindings used a getter).
                    "realsense_timestamp_domain": str(color.frame_timestamp_domain).split(".")[-1],
                    "realsense_frame_number": int(color.get_frame_number()),
                    "image_path": f"rgb/frame_{index:06d}.png",
                }
                try:
                    self._packets.put_nowait(FramePacket(index, image, record))
                except queue.Full:
                    self.dropped_frames += 1
        except BaseException as exc:  # surface a hardware error in the terminal thread
            if not self._stop.is_set():
                self._exception = exc

    def _writer_loop(self) -> None:
        with self.frames_path.open("w", encoding="utf-8") as handle:
            while True:
                packet = self._packets.get()
                if packet is None:
                    break
                # Keep draining after a disk failure so ``stop()`` can always
                # join this thread instead of deadlocking on a full queue.
                if self._exception is not None:
                    continue
                try:
                    Image.fromarray(packet.image).save(self.rgb_dir / f"frame_{packet.index:06d}.png")
                    _write_jsonl(handle, packet.record)
                except BaseException as exc:
                    self._exception = exc
                    self._stop.set()

    def check_health(self) -> None:
        if self._exception is not None:
            raise RuntimeError("D435 capture failed") from self._exception

    def stop(self) -> None:
        self._stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._pipeline is not None:
            self._pipeline.stop()
        self._packets.put(None)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=15.0)
        self.check_health()


class CbreakKeyboard:
    """Read individual terminal key presses without an extra keyboard dependency."""

    def __enter__(self) -> "CbreakKeyboard":
        if not sys.stdin.isatty():
            raise RuntimeError("Keyboard teleoperation requires a real TTY terminal.")
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def read(self, timeout_s: float) -> str | None:
        ready, _, _ = select.select([self._fd], [], [], timeout_s)
        if not ready:
            return None
        value = os.read(self._fd, 1)
        return value.decode("utf-8", errors="ignore") or None

    def __exit__(self, exc_type, exc, traceback) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def _sample_state(robot: Any, gripper: Any | None, *, state_index: int, phase: str, action_index: int | None, session_start_monotonic_ns: int) -> dict[str, Any]:
    read_start_ns = time.monotonic_ns()
    state = robot.state
    pose = robot.current_pose.end_effector_pose
    gripper_width = gripper_max_width = gripper_is_grasped = None
    if gripper is not None:
        gripper_state = gripper.state
        gripper_width = float(gripper_state.width)
        gripper_max_width = float(gripper_state.max_width)
        gripper_is_grasped = bool(gripper_state.is_grasped)
    read_end_ns = time.monotonic_ns()
    midpoint_ns = (read_start_ns + read_end_ns) // 2
    return {
        "state_index": state_index,
        "phase": phase,
        "action_index": action_index,
        **_clock_record(session_start_monotonic_ns, timestamp_ns=midpoint_ns),
        "state_read_start_monotonic_ns": read_start_ns,
        "state_read_end_monotonic_ns": read_end_ns,
        "robot_mode": _mode_name(state.robot_mode),
        "has_errors": bool(robot.has_errors),
        "is_in_control": bool(robot.is_in_control),
        "control_command_success_rate": float(state.control_command_success_rate),
        "q_rad": [float(value) for value in state.q.tolist()],
        "dq_rad_s": [float(value) for value in state.dq.tolist()],
        "tcp_translation_m": [float(value) for value in pose.translation.tolist()],
        "tcp_quaternion_xyzw": [float(value) for value in pose.quaternion.tolist()],
        "external_wrench_base_n_nm": [float(value) for value in state.O_F_ext_hat_K.tolist()],
        "gripper_width_m": gripper_width,
        "gripper_max_width_m": gripper_max_width,
        "gripper_is_grasped": gripper_is_grasped,
        "current_errors": _active_error_names(state.current_errors),
        "last_motion_errors": _active_error_names(state.last_motion_errors),
    }


def _nearest_frame(frames: list[dict[str, Any]], timestamps: list[int], timestamp_ns: int) -> dict[str, int | float] | None:
    if not frames:
        return None
    position = bisect.bisect_left(timestamps, timestamp_ns)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
    index = min(candidates, key=lambda candidate: abs(timestamps[candidate] - timestamp_ns))
    return {"frame_index": int(frames[index]["frame_index"]), "delta_ns": int(timestamps[index] - timestamp_ns)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_alignment_index(run_dir: Path) -> dict[str, int]:
    """Join every host-timestamped state/action event to its nearest saved RGB frame."""
    frames = _read_jsonl(run_dir / "frames.jsonl")
    timestamps = [int(frame["host_monotonic_ns"]) for frame in frames]
    states = _read_jsonl(run_dir / "states.jsonl")
    actions = _read_jsonl(run_dir / "actions.jsonl")
    with (run_dir / "alignment.jsonl").open("w", encoding="utf-8") as handle:
        for state in states:
            _write_jsonl(handle, {
                "event_type": "state",
                "state_index": state["state_index"],
                "host_monotonic_ns": state["host_monotonic_ns"],
                "nearest_camera": _nearest_frame(frames, timestamps, int(state["host_monotonic_ns"])),
            })
        for action in actions:
            _write_jsonl(handle, {
                "event_type": "action",
                "action_index": action["action_index"],
                "command_start_monotonic_ns": action["command_start_monotonic_ns"],
                "command_end_monotonic_ns": action["command_end_monotonic_ns"],
                "nearest_camera_at_start": _nearest_frame(frames, timestamps, int(action["command_start_monotonic_ns"])),
                "nearest_camera_at_end": _nearest_frame(frames, timestamps, int(action["command_end_monotonic_ns"])),
                "pre_state_index": action["pre_state_index"],
                "post_state_index": action["post_state_index"],
            })
    return {"frames": len(frames), "states": len(states), "actions": len(actions)}


def print_controls() -> None:
    print(
        "\nControls (one key = one blocking, logged increment):\n"
        "  w/s: +X/-X    a/d: +Y/-Y    r/f: +Z/-Z\n"
        "  j/l: +yaw/-yaw  o/c: open/close gripper by one increment\n"
        "  p: record state now    m: timestamped marker    h or ?: help    q: finish and save\n"
        "All translations are in the Franka base frame and are clamped to --workspace-min/max.\n",
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    if not args.ip:
        print("Please pass --ip <robot_ip> or set FRANKA_ROBOT_IP.", file=sys.stderr)
        return 2
    if args.step_m <= 0 or args.yaw_step_deg <= 0 or args.gripper_step_m <= 0:
        raise ValueError("--step-m, --yaw-step-deg, and --gripper-step-m must be positive.")
    if not 0.0 < args.speed <= 1.0 or args.gripper_speed <= 0 or args.state_rate_hz <= 0:
        raise ValueError("--speed must be in (0, 1]; --gripper-speed and --state-rate-hz must be positive.")
    if any(lower >= upper for lower, upper in zip(args.workspace_min, args.workspace_max)):
        raise ValueError("Each --workspace-min coordinate must be smaller than its --workspace-max counterpart.")

    timestamp = datetime.now(SHANGHAI_TIMEZONE)
    run_dir = Path(args.output_dir).expanduser().resolve() / f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{args.run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    session_start_monotonic_ns = time.monotonic_ns()
    metadata_path = run_dir / "metadata.json"
    recorder: RealSenseRecorder | None = None
    robot = gripper = None
    state_index = action_index = 0
    state_handle = action_handle = marker_handle = None

    try:
        # Import only after argument validation so --help and offline validation work without Franky installed.
        from franky import Affine, CartesianMotion, ControlException, Gripper, RealtimeConfig, ReferenceType, Robot

        realtime_config = RealtimeConfig.Ignore if args.realtime == "ignore" else RealtimeConfig.Enforce
        robot = Robot(args.ip, realtime_config=realtime_config)
        robot.relative_dynamics_factor = args.speed
        gripper = None if args.no_gripper else Gripper(args.ip)
        recorder = RealSenseRecorder(args, run_dir, session_start_monotonic_ns)
        recorder.start()
        print(f"Recording directory: {run_dir}")
        print(f"Connected to Franka at {args.ip}; D435 recording started.")
        if args.preview_only:
            print("PREVIEW ONLY: no arm or gripper command will be sent.")
        print(f"Workspace: min={args.workspace_min}, max={args.workspace_max}; increment={args.step_m:.4f} m")
        if not args.yes:
            input("Clear the workspace, keep the emergency stop available, then press Enter to arm teleoperation... ")
        print_controls()

        state_handle = (run_dir / "states.jsonl").open("w", encoding="utf-8")
        action_handle = (run_dir / "actions.jsonl").open("w", encoding="utf-8")
        marker_handle = (run_dir / "markers.jsonl").open("w", encoding="utf-8")

        def save_state(phase: str, related_action: int | None) -> dict[str, Any]:
            nonlocal state_index
            sample = _sample_state(
                robot, gripper, state_index=state_index, phase=phase, action_index=related_action,
                session_start_monotonic_ns=session_start_monotonic_ns,
            )
            _write_jsonl(state_handle, sample)
            state_index += 1
            return sample

        save_state("initial", None)
        next_state_time = time.monotonic() + 1.0 / args.state_rate_hz
        key_actions = {
            "w": ("arm", [args.step_m, 0.0, 0.0], 0.0), "s": ("arm", [-args.step_m, 0.0, 0.0], 0.0),
            "a": ("arm", [0.0, args.step_m, 0.0], 0.0), "d": ("arm", [0.0, -args.step_m, 0.0], 0.0),
            "r": ("arm", [0.0, 0.0, args.step_m], 0.0), "f": ("arm", [0.0, 0.0, -args.step_m], 0.0),
            "j": ("arm", [0.0, 0.0, 0.0], args.yaw_step_deg), "l": ("arm", [0.0, 0.0, 0.0], -args.yaw_step_deg),
            "o": ("gripper", None, 1.0), "c": ("gripper", None, -1.0),
        }
        with CbreakKeyboard() as keyboard:
            while True:
                recorder.check_health()
                key = keyboard.read(max(0.0, next_state_time - time.monotonic()))
                if key is None:
                    save_state("idle", None)
                    next_state_time += 1.0 / args.state_rate_hz
                    continue
                key = key.lower()
                key_time = _clock_record(session_start_monotonic_ns)
                if key == "q":
                    break
                if key in {"h", "?"}:
                    print_controls()
                    continue
                if key == "p":
                    state = save_state("manual", None)
                    print(f"State {state['state_index']}: TCP={np.round(state['tcp_translation_m'], 4).tolist()}")
                    continue
                if key == "m":
                    _write_jsonl(marker_handle, {"marker_index": int(time.monotonic_ns()), "key": "m", **key_time})
                    print("Marker recorded.")
                    continue
                if key not in key_actions:
                    continue

                kind, requested_delta, aux = key_actions[key]
                pre_state = save_state("pre_action", action_index)
                record: dict[str, Any] = {
                    "action_index": action_index,
                    "key": key,
                    "kind": kind,
                    "key_received_monotonic_ns": key_time["host_monotonic_ns"],
                    "pre_state_index": pre_state["state_index"],
                    "requested_translation_delta_m": requested_delta,
                    "requested_yaw_delta_deg": aux if kind == "arm" else 0.0,
                    "command_sent": False,
                    "success": None,
                }
                command_start_ns = time.monotonic_ns()
                record["command_start_monotonic_ns"] = command_start_ns
                try:
                    if kind == "arm":
                        actual_delta = _clamp_delta(
                            pre_state["tcp_translation_m"], requested_delta, args.workspace_min, args.workspace_max
                        )
                        record["sent_translation_delta_m"] = actual_delta
                        record["sent_yaw_delta_deg"] = aux
                        record["workspace_clamped"] = actual_delta != requested_delta
                        if not any(actual_delta) and aux == 0.0:
                            record["success"] = False
                            record["skip_reason"] = "workspace_limit"
                            print("Action rejected: TCP is already at the configured workspace limit.")
                        elif args.preview_only:
                            record["success"] = True
                            record["preview_only"] = True
                        else:
                            record["command_sent"] = True
                            motion = CartesianMotion(
                                Affine(actual_delta, _rpy_quaternion_xyzw(aux)), ReferenceType.Relative, args.speed
                            )
                            robot.move(motion)
                            record["success"] = True
                    else:
                        if gripper is None:
                            record["success"] = False
                            record["skip_reason"] = "gripper_disabled"
                        else:
                            current = float(pre_state["gripper_width_m"] or 0.0)
                            max_width = float(pre_state["gripper_max_width_m"] or 0.0)
                            record["gripper_width_before_m"] = current
                            if max_width <= 0.0:
                                record["success"] = False
                                record["skip_reason"] = "gripper_not_homed_or_invalid_max_width"
                                print("Gripper command rejected: max_width is 0. Run gripper homing separately with an empty gripper.")
                            else:
                                target = min(max(current + aux * args.gripper_step_m, 0.0), max_width)
                                record["gripper_target_width_m"] = target
                                if args.preview_only:
                                    record["success"] = True
                                    record["preview_only"] = True
                                elif abs(target - current) < 1e-8:
                                    record["success"] = True
                                    record["skip_reason"] = "already_at_gripper_limit"
                                else:
                                    record["command_sent"] = True
                                    record["success"] = bool(gripper.move(target, args.gripper_speed))
                except ControlException as exc:
                    record["success"] = False
                    record["exception"] = str(exc)
                    print(f"Robot motion failed: {exc}", file=sys.stderr)
                except Exception as exc:
                    record["success"] = False
                    record["exception"] = str(exc)
                    print(f"Command failed: {exc}", file=sys.stderr)
                finally:
                    record["command_end_monotonic_ns"] = time.monotonic_ns()
                    post_state = save_state("post_action", action_index)
                    record["post_state_index"] = post_state["state_index"]
                    record["duration_ms"] = (record["command_end_monotonic_ns"] - command_start_ns) / 1e6
                    _write_jsonl(action_handle, record)
                    action_index += 1
                    next_state_time = time.monotonic() + 1.0 / args.state_rate_hz
                    print(f"Action {record['action_index']}: success={record['success']}, TCP={np.round(post_state['tcp_translation_m'], 4).tolist()}")
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping recording and saving collected data...", file=sys.stderr)
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
    finally:
        for handle in (state_handle, action_handle, marker_handle):
            if handle is not None:
                handle.close()
        if recorder is not None:
            recorder.stop()
        alignment_counts = write_alignment_index(run_dir)
        metadata = {
            "format_version": 1,
            "created_at": timestamp.isoformat(),
            "timezone": "Asia/Shanghai",
            "session_start_monotonic_ns": session_start_monotonic_ns,
            "primary_timebase": "host_monotonic_ns",
            "time_alignment": {
                "method": "nearest saved RGB frame by host_monotonic_ns",
                "camera_device_timestamp_note": "realsense_timestamp_ms is retained for diagnostics but is not directly comparable to host_monotonic_ns",
                "index_file": "alignment.jsonl",
            },
            "robot": {"ip": args.ip, "realtime": args.realtime, "state_rate_hz_while_idle": args.state_rate_hz},
            "camera": recorder.camera_metadata if recorder is not None else None,
            "teleop": {
                "step_m": args.step_m, "yaw_step_deg": args.yaw_step_deg, "speed": args.speed,
                "gripper_step_m": args.gripper_step_m, "gripper_speed": args.gripper_speed,
                "workspace_min": args.workspace_min, "workspace_max": args.workspace_max,
                "preview_only": args.preview_only,
            },
            "files": {
                "rgb": "rgb/frame_XXXXXX.png", "frames": "frames.jsonl", "states": "states.jsonl",
                "actions": "actions.jsonl", "markers": "markers.jsonl", "alignment": "alignment.jsonl",
            },
            "counts": {**alignment_counts, "dropped_camera_frames_due_to_writer_backlog": recorder.dropped_frames if recorder is not None else 0},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved trajectory to: {run_dir}")
        print(f"Frames/states/actions: {alignment_counts['frames']}/{alignment_counts['states']}/{alignment_counts['actions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
