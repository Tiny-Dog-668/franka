#!/usr/bin/env python3
"""Solve and validate a Franka/D435 eye-to-hand calibration dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from franka_sim2real.calibration.hand_eye import (
    EyeToHandConfig,
    detect_checkerboard,
    draw_checkerboard_overlay,
    estimate_target_to_camera_candidates,
    resolve_checkerboard_symmetry,
    resolve_validation_symmetry,
    solve_eye_to_hand,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline PnP, eye-to-hand solve, consensus, and holdout validation."
    )
    parser.add_argument("--dataset", required=True, type=Path, help="Collected dataset directory.")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="SAMPLE_ID",
        help="Explicit sample IDs to omit. Samples are never silently removed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New report directory. Defaults to DATASET/reports/<timestamp>.",
    )
    parser.add_argument(
        "--use-available",
        action="store_true",
        help=(
            "Fit all currently accepted samples instead of requiring the complete "
            "configured calibration and validation counts. At least three calibration "
            "samples are required; incomplete results are marked provisional."
        ),
    )
    return parser


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _reprocess_sample(
    dataset_dir: Path,
    output_dir: Path,
    sample: dict[str, Any],
    config: EyeToHandConfig,
    camera: dict[str, Any],
) -> dict[str, Any]:
    image_path = dataset_dir / sample["selected_rgb"]
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read selected RGB frame: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    corners, sharpness = detect_checkerboard(rgb, config.board)
    if corners is None:
        raise RuntimeError(
            f"Offline checkerboard detection failed for {sample['sample_id']}"
        )
    intrinsics = camera["intrinsics"]
    pnp_candidates = estimate_target_to_camera_candidates(
        corners,
        intrinsics["camera_matrix"],
        intrinsics["distortion_coefficients"],
        config.board,
    )
    pnp = min(
        pnp_candidates,
        key=lambda candidate: candidate["reprojection_rms_px"],
    )
    overlay = draw_checkerboard_overlay(rgb, corners, config.board)
    overlay_dir = output_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    overlay_path = overlay_dir / f"{sample['sample_id']}.png"
    if not cv2.imwrite(
        str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")
    return {
        "sample_id": sample["sample_id"],
        "phase": sample["phase"],
        "base_to_gripper": sample["base_to_gripper"],
        "target_to_camera": pnp["target_to_camera"],
        "target_to_camera_candidates": pnp_candidates,
        "reprojection_rms_px": pnp["reprojection_rms_px"],
        "sharpness_laplacian_variance": sharpness,
        "source_rgb": str(image_path),
        "overlay": str(overlay_path),
        "pnp": pnp,
    }


def _format_matrix(matrix: list[list[float]] | None) -> str:
    if matrix is None:
        return "Not accepted"
    return "\n".join(
        "  " + " ".join(f"{value: .9f}" for value in row) for row in matrix
    )


def _markdown_report(report: dict[str, Any]) -> str:
    solve = report["solve"]
    checks = solve["checks"]
    lines = [
        "# Franka–D435 Eye-to-Hand Calibration Report",
        "",
        f"- Fit mode: `{solve['fit_mode']}`",
        f"- Fit accepted: **{solve['accepted']}**",
        f"- Provisional: **{solve['provisional']}**",
        f"- Deployment ready: **{solve['deployment_ready']}**",
        f"- Camera serial: `{report['camera']['serial']}`",
        f"- Selected method: `{solve['selected_method']}`",
        f"- Calibration samples: {checks['calibration_sample_count']}",
        f"- Validation samples: {checks['validation_sample_count']}",
        f"- Valid methods: {checks['valid_method_count']}",
        f"- Consensus methods: {', '.join(checks['selected_consensus_methods']) or 'none'}",
        "",
        "## Checks",
        "",
        f"- Sample count: {checks['sample_count_pass']}",
        f"- Complete configured dataset: {checks['dataset_complete']}",
        f"- Reprojection: {checks['reprojection_pass']}",
        f"- Method consensus: {checks['method_consensus_pass']}",
        "- Independent validation: "
        + (
            str(checks["validation_pass"])
            if checks["validation_available"]
            else "not available"
        ),
        "",
        "## T_base_color",
        "",
        "```text",
        _format_matrix(solve["T_base_color"]),
        "```",
        "",
        "This transform maps D435 RGB color optical-frame points into the Franka base frame.",
        "Depth points must first be aligned or transformed into the RGB color optical frame.",
        "",
    ]
    if solve["provisional"]:
        lines.extend(
            [
                "> This transform was fitted from the currently available samples. It is useful for",
                "> diagnostics, but it is not deployment-ready until the configured independent",
                "> validation sample count is satisfied.",
                "",
            ]
        )
    lines.extend(
        [
        "## Validation",
        "",
        ]
    )
    if not solve["validation"]:
        lines.append("No independent validation samples were available.")
    else:
        lines.extend(
            [
                "| Sample | Translation (mm) | Rotation (deg) | Reprojection (px) | Pass |",
                "|---|---:|---:|---:|:---:|",
            ]
        )
        for item in solve["validation"]:
            lines.append(
                f"| `{item['sample_id']}` | "
                f"{item['translation_m'] * 1000:.3f} | "
                f"{item['rotation_deg']:.4f} | "
                f"{item['reprojection_rms_px']:.4f} | "
                f"{item['passed']} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = args.dataset.expanduser().resolve()
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = EyeToHandConfig.from_dict(manifest["config"])
    excluded = set(args.exclude)
    known_ids = set(manifest["samples"])
    unknown = sorted(excluded - known_ids)
    if unknown:
        raise ValueError(f"Unknown --exclude sample IDs: {', '.join(unknown)}")

    if args.output:
        output_dir = args.output.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = dataset_dir / "reports" / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    processed: list[dict[str, Any]] = []
    processing_failures = []
    for sample_id, sample in manifest["samples"].items():
        if sample_id in excluded or sample.get("status") != "accepted":
            continue
        try:
            processed.append(
                _reprocess_sample(
                    dataset_dir,
                    output_dir,
                    sample,
                    config,
                    manifest["camera"],
                )
            )
        except Exception as exc:
            processing_failures.append(
                {"sample_id": sample_id, "error": str(exc)}
            )

    raw_calibration_samples = [
        sample for sample in processed if sample["phase"] == "calibration"
    ]
    raw_validation_samples = [
        sample for sample in processed if sample["phase"] == "validation"
    ]
    symmetry_error = None
    symmetry_diagnostics: dict[str, Any] | None = None
    validation_symmetry_diagnostics: list[dict[str, Any]] = []
    try:
        calibration_samples, symmetry_diagnostics = resolve_checkerboard_symmetry(
            raw_calibration_samples,
            config.quality,
        )
        validation_samples, validation_symmetry_diagnostics = (
            resolve_validation_symmetry(
                raw_validation_samples,
                symmetry_diagnostics["diagnostic_base_to_camera"],
                symmetry_diagnostics["gripper_to_target_reference"],
                config.quality,
            )
        )
    except Exception as exc:
        symmetry_error = str(exc)
        calibration_samples = []
        validation_samples = []

    solve = solve_eye_to_hand(
        calibration_samples,
        validation_samples,
        config.quality,
        use_available_samples=args.use_available,
    )
    solve["checkerboard_symmetry"] = symmetry_diagnostics
    solve["validation_symmetry"] = validation_symmetry_diagnostics
    solve["checks"]["checkerboard_symmetry_pass"] = symmetry_error is None
    if symmetry_error is not None:
        solve["accepted"] = False
        solve["T_base_color"] = None
        solve["T_color_base"] = None
        solve["base_to_color_quaternion_xyzw"] = None
    if processing_failures:
        solve["accepted"] = False
        solve["T_base_color"] = None
        solve["T_color_base"] = None
        solve["base_to_color_quaternion_xyzw"] = None
        solve["checks"]["offline_processing_pass"] = False
    else:
        solve["checks"]["offline_processing_pass"] = True

    report = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_dir),
        "fit_mode": "available" if args.use_available else "complete",
        "excluded_samples": sorted(excluded),
        "processing_failures": processing_failures,
        "symmetry_error": symmetry_error,
        "board": manifest["config"]["board"],
        "camera": manifest["camera"],
        "solve": solve,
        "processed_samples": processed,
    }
    _write_json(output_dir / "calibration_report.json", report)
    (output_dir / "calibration_report.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    print(f"Report: {output_dir}")
    print(f"Fit mode: {solve['fit_mode']}")
    print(f"Accepted: {solve['accepted']}")
    print(f"Provisional: {solve['provisional']}")
    print(f"Deployment ready: {solve['deployment_ready']}")
    print(f"Selected method: {solve['selected_method']}")
    if solve["T_base_color"] is not None:
        print("T_base_color:")
        print(np.asarray(solve["T_base_color"]))
    else:
        print("No deployable T_base_color was emitted; inspect the report checks.")
    return 0 if solve["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
