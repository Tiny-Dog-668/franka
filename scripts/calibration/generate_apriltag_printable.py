#!/usr/bin/env python3
"""Generate an exact-size printable AprilTag for a 50 mm cube face."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MM_TO_PT = 72.0 / 25.4
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
APRILTAG_FAMILIES = {
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _marker_cells(family: str, marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_FAMILIES[family])
    if marker_id < 0 or marker_id >= dictionary.bytesList.shape[0]:
        raise ValueError(
            f"ID {marker_id} is outside {family}'s range 0..{dictionary.bytesList.shape[0] - 1}"
        )
    # markerSize payload bits plus one black border bit on every side.
    cell_count = int(dictionary.markerSize) + 2
    marker = cv2.aruco.generateImageMarker(
        dictionary, marker_id, cell_count, borderBits=1
    )
    if marker.shape != (cell_count, cell_count) or not np.isin(marker, (0, 255)).all():
        raise RuntimeError("OpenCV returned an unexpected marker image")
    return marker


def _svg_marker_rectangles(
    cells: np.ndarray, origin_x_mm: float, origin_y_mm: float, tag_size_mm: float
) -> str:
    cell_mm = tag_size_mm / cells.shape[0]
    rectangles = []
    for row, column in np.argwhere(cells == 0):
        rectangles.append(
            f'<rect x="{origin_x_mm + column * cell_mm:.8f}" '
            f'y="{origin_y_mm + row * cell_mm:.8f}" '
            f'width="{cell_mm:.8f}" height="{cell_mm:.8f}" fill="#000"/>'
        )
    return "\n  ".join(rectangles)


def _write_cutout_svg(
    path: Path, cells: np.ndarray, family: str, marker_id: int, tag_mm: float, cut_mm: float
) -> None:
    margin = 0.5 * (cut_mm - tag_mm)
    rectangles = _svg_marker_rectangles(cells, margin, margin, tag_mm)
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{cut_mm}mm" height="{cut_mm}mm"
     viewBox="0 0 {cut_mm} {cut_mm}" shape-rendering="crispEdges">
  <title>AprilTag {family} ID {marker_id}, black square {tag_mm} mm</title>
  <rect width="{cut_mm}" height="{cut_mm}" fill="#fff"/>
  {rectangles}
</svg>
'''
    path.write_text(content, encoding="utf-8")


def _write_a4_svg(
    path: Path, cells: np.ndarray, family: str, marker_id: int, tag_mm: float, cut_mm: float
) -> None:
    cut_x = 20.0
    cut_y = 25.0
    margin = 0.5 * (cut_mm - tag_mm)
    rectangles = _svg_marker_rectangles(
        cells, cut_x + margin, cut_y + margin, tag_mm
    )
    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm"
     viewBox="0 0 210 297" shape-rendering="geometricPrecision">
  <title>Printable AprilTag {family} ID {marker_id}</title>
  <rect width="210" height="297" fill="#fff"/>
  <rect x="{cut_x}" y="{cut_y}" width="{cut_mm}" height="{cut_mm}"
        fill="none" stroke="#888" stroke-width="0.2" stroke-dasharray="1,1"/>
  <g shape-rendering="crispEdges">{rectangles}</g>
  <g font-family="sans-serif" fill="#000">
    <text x="85" y="30" font-size="5">AprilTag {family} / ID {marker_id}</text>
    <text x="85" y="39" font-size="4">Black marker square: {tag_mm:.1f} mm</text>
    <text x="85" y="46" font-size="4">Cut square: {cut_mm:.1f} x {cut_mm:.1f} mm</text>
    <text x="85" y="53" font-size="4" font-weight="bold">Print at Actual size / 100%</text>
    <line x1="85" y1="70" x2="135" y2="70" stroke="#000" stroke-width="0.4"/>
    <line x1="85" y1="68" x2="85" y2="72" stroke="#000" stroke-width="0.4"/>
    <line x1="135" y1="68" x2="135" y2="72" stroke="#000" stroke-width="0.4"/>
    <text x="85" y="78" font-size="3.5">This line must measure exactly 50 mm</text>
  </g>
</svg>
'''
    path.write_text(content, encoding="utf-8")


def _write_png(path: Path, cells: np.ndarray, tag_mm: float, cut_mm: float) -> None:
    # 24 px/mm gives an exact 1200 px, 50 mm cutout and 960 px, 40 mm tag.
    pixels_per_mm = 24
    cut_pixels = round(cut_mm * pixels_per_mm)
    tag_pixels = round(tag_mm * pixels_per_mm)
    if tag_pixels % cells.shape[0] != 0:
        raise ValueError("Requested dimensions do not map to whole marker cells in the PNG")
    margin_pixels = (cut_pixels - tag_pixels) // 2
    cell_pixels = tag_pixels // cells.shape[0]
    canvas = np.full((cut_pixels, cut_pixels), 255, dtype=np.uint8)
    marker = np.repeat(np.repeat(cells, cell_pixels, axis=0), cell_pixels, axis=1)
    canvas[
        margin_pixels : margin_pixels + tag_pixels,
        margin_pixels : margin_pixels + tag_pixels,
    ] = marker
    dpi = pixels_per_mm * 25.4
    Image.fromarray(canvas, mode="L").save(path, dpi=(dpi, dpi), compress_level=9)


def _pdf_bytes(objects: list[bytes]) -> bytes:
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(output)


def _write_a4_pdf(
    path: Path, cells: np.ndarray, family: str, marker_id: int, tag_mm: float, cut_mm: float
) -> None:
    page_width = A4_WIDTH_MM * MM_TO_PT
    page_height = A4_HEIGHT_MM * MM_TO_PT
    cut_x = 20.0 * MM_TO_PT
    cut_y = page_height - (25.0 + cut_mm) * MM_TO_PT
    margin = 0.5 * (cut_mm - tag_mm) * MM_TO_PT
    tag_x = cut_x + margin
    tag_y = cut_y + margin
    cell_pt = tag_mm * MM_TO_PT / cells.shape[0]

    commands = ["1 1 1 rg", f"0 0 {page_width:.6f} {page_height:.6f} re f", "0 0 0 rg"]
    for row, column in np.argwhere(cells == 0):
        x = tag_x + column * cell_pt
        y = tag_y + (cells.shape[0] - 1 - row) * cell_pt
        commands.append(f"{x:.6f} {y:.6f} {cell_pt:.6f} {cell_pt:.6f} re f")
    commands.extend(
        [
            "0.55 G 0.4 w [2 2] 0 d",
            f"{cut_x:.6f} {cut_y:.6f} {cut_mm * MM_TO_PT:.6f} {cut_mm * MM_TO_PT:.6f} re S",
            "0 G [] 0 d",
            "BT /F1 14 Tf 240 756 Td " + f"(AprilTag {family} / ID {marker_id}) Tj ET",
            "BT /F1 11 Tf 240 730 Td " + f"(Black marker square: {tag_mm:.1f} mm) Tj ET",
            "BT /F1 11 Tf 240 710 Td " + f"(Cut square: {cut_mm:.1f} x {cut_mm:.1f} mm) Tj ET",
            "BT /F1 12 Tf 240 686 Td (Print at Actual size / 100%) Tj ET",
        ]
    )
    line_x = 85.0 * MM_TO_PT
    line_y = page_height - 70.0 * MM_TO_PT
    line_length = 50.0 * MM_TO_PT
    tick = 2.0 * MM_TO_PT
    commands.extend(
        [
            "0 G 1 w",
            f"{line_x:.6f} {line_y:.6f} m {line_x + line_length:.6f} {line_y:.6f} l S",
            f"{line_x:.6f} {line_y - tick:.6f} m {line_x:.6f} {line_y + tick:.6f} l S",
            f"{line_x + line_length:.6f} {line_y - tick:.6f} m "
            f"{line_x + line_length:.6f} {line_y + tick:.6f} l S",
            "BT /F1 10 Tf 240 620 Td (This line must measure exactly 50 mm) Tj ET",
        ]
    )
    stream = ("\n".join(commands) + "\n").encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.6f} {page_height:.6f}] "
            "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    path.write_bytes(_pdf_bytes(objects))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=sorted(APRILTAG_FAMILIES), default="tag36h11")
    parser.add_argument("--id", type=int, default=0, dest="marker_id")
    parser.add_argument("--tag-size-mm", type=float, default=40.0)
    parser.add_argument("--cut-size-mm", type=float, default=50.0)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[2] / "assets" / "apriltag_cube_50mm"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.tag_size_mm <= 0 or args.cut_size_mm <= args.tag_size_mm:
        raise ValueError("cut-size-mm must be larger than the positive tag-size-mm")
    cells = _marker_cells(args.family, args.marker_id)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"apriltag_{args.family}_id{args.marker_id}_{args.tag_size_mm:g}mm"
    cutout_svg = output_dir / f"{stem}_cutout.svg"
    a4_svg = output_dir / f"{stem}_a4.svg"
    png = output_dir / f"{stem}_cutout.png"
    pdf = output_dir / f"{stem}_a4.pdf"
    _write_cutout_svg(
        cutout_svg, cells, args.family, args.marker_id, args.tag_size_mm, args.cut_size_mm
    )
    _write_a4_svg(a4_svg, cells, args.family, args.marker_id, args.tag_size_mm, args.cut_size_mm)
    _write_png(png, cells, args.tag_size_mm, args.cut_size_mm)
    _write_a4_pdf(pdf, cells, args.family, args.marker_id, args.tag_size_mm, args.cut_size_mm)
    artifacts = [cutout_svg, a4_svg, png, pdf]
    metadata = {
        "family": args.family,
        "id": args.marker_id,
        "tag_size_mm": args.tag_size_mm,
        "tag_size_definition": "outer edge of black marker square; excludes white quiet zone",
        "cut_size_mm": args.cut_size_mm,
        "white_margin_each_side_mm": 0.5 * (args.cut_size_mm - args.tag_size_mm),
        "cube_face_mm": 50.0,
        "print_scale": "Actual size / 100%; disable fit-to-page",
        "pose_solver_marker_length_m": args.tag_size_mm / 1000.0,
        "files": {item.name: {"sha256": _sha256(item)} for item in artifacts},
    }
    metadata_path = output_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {args.family} ID {args.marker_id}: {output_dir}")
    for item in (*artifacts, metadata_path):
        print(f"  {item.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
