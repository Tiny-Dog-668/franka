#!/usr/bin/env python3
"""Measure AprilTag and policy-position errors across manual exposures.

This is a camera-only wrapper around ``live_apriltag_cube_pose.py``. It never
connects to or commands the robot. Each exposure is measured in a fresh
process and receives its own raw CSV, summary, and final-frame image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
LIVE_SCRIPT = REPO_ROOT / "scripts" / "calibration" / "live_apriltag_cube_pose.py"
DEFAULT_CAMERA_CONFIG = REPO_ROOT / "configs" / "eye_to_hand_d435_215322076207.json"
DEFAULT_CALIBRATION_REPORT = (
    REPO_ROOT
    / "runs"
    / "20260802_204111_eye_to_hand"
    / "reports"
    / "20260802_205349"
    / "calibration_report.json"
)
DEFAULT_POLICY_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0803_dr_heatmap.json"
DEFAULT_POLICY_MODEL = (
    REPO_ROOT / "checkpoint" / "0803_dr3" / "rma_student_e2e_student_0040000.pt"
)
DEFAULT_POLICY_METADATA = DEFAULT_POLICY_MODEL.with_suffix(".json")
DEFAULT_EXPOSURES = (40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0)

CSV_FIELDS = [
    "requested_exposure",
    "effective_exposure",
    "effective_gain",
    "repeat",
    "status",
    "return_code",
    "frame_count",
    "accepted_detection_count",
    "rejected_reprojection_count",
    "detection_rate",
    "policy_vs_tag_rmse_mm",
    "policy_vs_tag_median_mm",
    "policy_vs_tag_p95_mm",
    "policy_vs_ground_truth_rmse_mm",
    "tag_vs_ground_truth_rmse_mm",
    "policy_jitter_norm_mm",
    "tag_jitter_norm_mm",
    "trial_directory",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exposures",
        nargs="+",
        type=float,
        default=list(DEFAULT_EXPOSURES),
        help="Manual exposure values to test (default: 40 60 80 100 120 160 200)",
    )
    parser.add_argument(
        "--gain",
        type=float,
        default=64.0,
        help="Fixed manual gain for every trial (default: 64)",
    )
    parser.add_argument("--frames-per-exposure", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--minimum-detection-rate",
        type=float,
        default=0.5,
        help="Minimum accepted-tag fraction for inclusion in the best-exposure result",
    )
    parser.add_argument("--camera-config", default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--calibration-report", default=str(DEFAULT_CALIBRATION_REPORT))
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument("--policy-model", default=str(DEFAULT_POLICY_MODEL))
    parser.add_argument("--policy-metadata", default=str(DEFAULT_POLICY_METADATA))
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--family", default="tag36h11")
    parser.add_argument("--id", type=int, default=2, dest="marker_id")
    parser.add_argument("--marker-length-m", type=float, default=0.038)
    parser.add_argument(
        "--tag-to-object",
        nargs=3,
        type=float,
        default=(0.0, 0.0, -0.025),
        metavar=("X_M", "Y_M", "Z_M"),
    )
    parser.add_argument("--base-height-offset-m", type=float, default=0.020)
    parser.add_argument("--detection-scale", type=float, default=4.0)
    parser.add_argument("--max-reprojection-px", type=float, default=2.0)
    parser.add_argument(
        "--ground-truth",
        nargs=3,
        type=float,
        metavar=("X_M", "Y_M", "Z_M"),
        help="Optional independently measured cube-center XYZ in the Franka base frame",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Show each trial's live window; default is unattended/headless",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-dir",
        help="Defaults to runs/<timestamp>_apriltag_exposure_sweep",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.exposures or any(not math.isfinite(value) for value in args.exposures):
        raise ValueError("--exposures requires one or more finite values")
    if not math.isfinite(args.gain):
        raise ValueError("--gain must be finite")
    if args.frames_per_exposure <= 0:
        raise ValueError("--frames-per-exposure must be positive")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if not 0.0 <= args.minimum_detection_rate <= 1.0:
        raise ValueError("--minimum-detection-rate must be in [0, 1]")


def _value_slug(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace(".", "p")


def _append_triplet(command: list[str], flag: str, values: Any) -> None:
    command.append(flag)
    command.extend(format(float(value), ".12g") for value in values)


def _build_trial_command(
    args: argparse.Namespace,
    exposure: float,
    trial_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(LIVE_SCRIPT),
        "--camera-config",
        str(Path(args.camera_config).expanduser().resolve()),
        "--calibration-report",
        str(Path(args.calibration_report).expanduser().resolve()),
        "--policy-config",
        str(Path(args.policy_config).expanduser().resolve()),
        "--policy-model",
        str(Path(args.policy_model).expanduser().resolve()),
        "--policy-metadata",
        str(Path(args.policy_metadata).expanduser().resolve()),
        "--policy-device",
        args.policy_device,
        "--family",
        args.family,
        "--id",
        str(args.marker_id),
        "--marker-length-m",
        format(args.marker_length_m, ".12g"),
        "--base-height-offset-m",
        format(args.base_height_offset_m, ".12g"),
        "--detection-scale",
        format(args.detection_scale, ".12g"),
        "--max-reprojection-px",
        format(args.max_reprojection_px, ".12g"),
        "--max-frames",
        str(args.frames_per_exposure),
        "--exposure",
        format(exposure, ".12g"),
        "--gain",
        format(args.gain, ".12g"),
        "--output-dir",
        str(trial_dir),
    ]
    _append_triplet(command, "--tag-to-object", args.tag_to_object)
    if args.ground_truth is not None:
        _append_triplet(command, "--ground-truth", args.ground_truth)
    if not args.show_window:
        command.append("--headless")
    return command


def _millimetres(section: dict[str, Any], key: str) -> float | None:
    value = section.get(key)
    return None if value is None else 1000.0 * float(value)


def _jitter_norm_mm(section: dict[str, Any]) -> float | None:
    values = section.get("std_m")
    if not isinstance(values, list) or len(values) != 3:
        return None
    return 1000.0 * math.sqrt(sum(float(value) ** 2 for value in values))


def _result_from_summary(
    requested_exposure: float,
    repeat: int,
    return_code: int,
    trial_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    frame_count = int(summary.get("frame_count", 0))
    accepted = int(summary.get("accepted_detection_count", 0))
    policy = summary.get("policy_prediction", {})
    calibrated = summary.get("calibrated", {})
    controls = summary.get("camera_color_controls", {})
    return {
        "requested_exposure": requested_exposure,
        "effective_exposure": controls.get("exposure"),
        "effective_gain": controls.get("gain"),
        "repeat": repeat,
        "status": "ok" if accepted > 0 else "no_accepted_tag",
        "return_code": return_code,
        "frame_count": frame_count,
        "accepted_detection_count": accepted,
        "rejected_reprojection_count": int(summary.get("rejected_reprojection_count", 0)),
        "detection_rate": accepted / frame_count if frame_count else 0.0,
        "policy_vs_tag_rmse_mm": _millimetres(
            policy, "vs_calibrated_error_3d_rmse_m"
        ),
        "policy_vs_tag_median_mm": _millimetres(
            policy, "vs_calibrated_error_3d_median_m"
        ),
        "policy_vs_tag_p95_mm": _millimetres(
            policy, "vs_calibrated_error_3d_p95_m"
        ),
        "policy_vs_ground_truth_rmse_mm": _millimetres(policy, "error_3d_rmse_m"),
        "tag_vs_ground_truth_rmse_mm": _millimetres(calibrated, "error_3d_rmse_m"),
        "policy_jitter_norm_mm": _jitter_norm_mm(policy),
        "tag_jitter_norm_mm": _jitter_norm_mm(calibrated),
        "trial_directory": str(trial_dir),
    }


def _failed_result(
    exposure: float,
    repeat: int,
    return_code: int,
    trial_dir: Path,
) -> dict[str, Any]:
    result = {field: None for field in CSV_FIELDS}
    result.update(
        {
            "requested_exposure": exposure,
            "repeat": repeat,
            "status": "failed",
            "return_code": return_code,
            "frame_count": 0,
            "accepted_detection_count": 0,
            "rejected_reprojection_count": 0,
            "detection_rate": 0.0,
            "trial_directory": str(trial_dir),
        }
    )
    return result


def _best_result(
    results: list[dict[str, Any]],
    minimum_detection_rate: float,
    has_ground_truth: bool,
) -> tuple[str, dict[str, Any] | None]:
    metric = (
        "policy_vs_ground_truth_rmse_mm"
        if has_ground_truth
        else "policy_vs_tag_rmse_mm"
    )
    candidates = [
        result
        for result in results
        if result.get(metric) is not None
        and float(result.get("detection_rate") or 0.0) >= minimum_detection_rate
    ]
    if not candidates:
        return metric, None
    return metric, min(candidates, key=lambda result: float(result[metric]))


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = build_parser().parse_args()
    _validate_args(args)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else REPO_ROOT
        / "runs"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_apriltag_exposure_sweep"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    interrupted = False
    print("[sweep] camera only; no Franka connection or motion commands", flush=True)
    print(
        f"[sweep] exposures={args.exposures}, fixed_gain={args.gain}, "
        f"frames={args.frames_per_exposure}, repeats={args.repeats}",
        flush=True,
    )
    for repeat in range(1, args.repeats + 1):
        for exposure in args.exposures:
            trial_dir = output_dir / (
                f"exposure_{_value_slug(exposure)}_repeat_{repeat:02d}"
            )
            command = _build_trial_command(args, exposure, trial_dir)
            print(
                f"\n[sweep] starting exposure={exposure:g}, repeat={repeat}/{args.repeats}",
                flush=True,
            )
            try:
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                return_code = completed.returncode
            except KeyboardInterrupt:
                interrupted = True
                print("\n[sweep] interrupted", flush=True)
                break

            summary_path = trial_dir / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                result = _result_from_summary(
                    exposure, repeat, return_code, trial_dir, summary
                )
            else:
                result = _failed_result(exposure, repeat, return_code, trial_dir)
            results.append(result)
            error = result.get("policy_vs_tag_rmse_mm")
            error_text = "n/a" if error is None else f"{float(error):.2f} mm"
            print(
                f"[sweep] exposure={exposure:g}: status={result['status']}, "
                f"detection={100.0 * float(result['detection_rate']):.1f}%, "
                f"policy-vs-tag RMSE={error_text}",
                flush=True,
            )
            if args.fail_fast and result["status"] == "failed":
                interrupted = True
                break
        if interrupted:
            break

    csv_path = output_dir / "exposure_results.csv"
    _write_csv(csv_path, results)
    metric, best = _best_result(
        results,
        args.minimum_detection_rate,
        args.ground_truth is not None,
    )
    report = {
        "kind": "apriltag_exposure_sweep",
        "camera_only_no_robot_connection": True,
        "exposures": args.exposures,
        "fixed_gain": args.gain,
        "frames_per_exposure": args.frames_per_exposure,
        "repeats": args.repeats,
        "minimum_detection_rate": args.minimum_detection_rate,
        "ranking_metric": metric,
        "best_result": best,
        "interrupted": interrupted,
        "results": results,
    }
    report_path = output_dir / "exposure_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n[sweep] CSV: {csv_path}")
    print(f"[sweep] report: {report_path}")
    if best is None:
        print(
            "[sweep] no exposure met the detection-rate threshold with a usable error",
            flush=True,
        )
    else:
        print(
            f"[sweep] best requested exposure={best['requested_exposure']:g}, "
            f"{metric}={float(best[metric]):.2f} mm, "
            f"detection={100.0 * float(best['detection_rate']):.1f}%",
            flush=True,
        )
    if interrupted:
        return 130
    return 0 if results and any(row["status"] != "failed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
