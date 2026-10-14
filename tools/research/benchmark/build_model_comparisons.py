#!/usr/bin/env python3
"""Build pixel-preserving DEIM vs PaddleX-L side-by-side comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    fallback = Path(__file__).resolve().parents[3] / "_local" / "baseline"
    root = Path(os.environ.get("BASELINE_ROOT", str(fallback))).expanduser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--deim-dir", type=Path, default=root / "output" / "DEIM")
    parser.add_argument(
        "--paddlex-dir", type=Path, default=root / "output" / "paddlex_L"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "contrary" / "DEIMvsPaddlex",
    )
    return parser.parse_args()


def load_detection_index(result_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = result_dir / "detections.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for item in document["images"]:
        key = (str(item["group"]), Path(item["source_path"]).name)
        if key in result:
            raise RuntimeError(f"Duplicate result key in {path}: {key}")
        result[key] = item
    return result


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[Path] = []
    if os.environ.get("WINDIR"):
        font_dir = Path(os.environ["WINDIR"]) / "Fonts"
        candidates.extend(
            [
                font_dir / ("segoeuib.ttf" if bold else "segoeui.ttf"),
                font_dir / ("arialbd.ttf" if bold else "arial.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    region: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = region
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def compose_pair(
    deim_path: Path,
    paddlex_path: Path,
    group: str,
    source_name: str,
) -> Image.Image:
    with Image.open(deim_path) as source:
        deim = source.convert("RGB").copy()
    with Image.open(paddlex_path) as source:
        paddlex = source.convert("RGB").copy()
    if deim.size != paddlex.size:
        raise RuntimeError(
            f"Visualization sizes differ for {group}/{source_name}: "
            f"DEIM={deim.size}, PaddleX={paddlex.size}"
        )

    width, height = deim.size
    header_height = max(48, min(72, round(height * 0.075)))
    separator_width = max(4, round(width * 0.006))
    canvas = Image.new(
        "RGB", (width * 2 + separator_width, height + header_height), (38, 38, 38)
    )
    canvas.paste(deim, (0, header_height))
    canvas.paste(paddlex, (width + separator_width, header_height))
    draw = ImageDraw.Draw(canvas)

    label_font = load_font(max(16, min(28, header_height // 2)), bold=True)
    detail_font = load_font(max(11, min(16, header_height // 3)), bold=False)
    centered_text(
        draw,
        (0, 0, width, header_height - 15),
        "DEIM",
        label_font,
        (255, 255, 255),
    )
    centered_text(
        draw,
        (width + separator_width, 0, width * 2 + separator_width, header_height - 15),
        "PaddleX-L",
        label_font,
        (255, 255, 255),
    )
    detail = f"{group} / {source_name}  |  same input, threshold > 0.5"
    centered_text(
        draw,
        (0, header_height - 22, width * 2 + separator_width, header_height),
        detail,
        detail_font,
        (205, 205, 205),
    )
    draw.rectangle(
        [width, 0, width + separator_width - 1, height + header_height - 1],
        fill=(118, 118, 118),
    )
    return canvas


def write_contact_sheet(paths: list[Path], output_path: Path) -> None:
    if not paths:
        return
    target_width = 1600
    margin = 18
    title_height = 54
    thumbnails: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB").copy()
        scale = min(1.0, target_width / image.width)
        if scale < 1.0:
            image = image.resize(
                (target_width, max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        thumbnails.append(image)
    sheet_width = max(image.width for image in thumbnails) + margin * 2
    sheet_height = title_height + sum(image.height + margin for image in thumbnails)
    sheet = Image.new("RGB", (sheet_width, sheet_height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = load_font(24, bold=True)
    centered_text(
        draw,
        (0, 0, sheet_width, title_height),
        output_path.stem.replace("_", " "),
        font,
        (255, 255, 255),
    )
    y = title_height
    for image in thumbnails:
        x = (sheet_width - image.width) // 2
        sheet.paste(image, (x, y))
        y += image.height + margin
    sheet.save(output_path, format="JPEG", quality=92, optimize=True)


def main() -> int:
    args = parse_args()
    deim_dir = args.deim_dir.resolve()
    paddlex_dir = args.paddlex_dir.resolve()
    output_dir = args.output_dir.resolve()
    deim_index = load_detection_index(deim_dir)
    paddlex_index = load_detection_index(paddlex_dir)
    deim_document = json.loads(
        (deim_dir / "detections.json").read_text(encoding="utf-8")
    )
    paddlex_document = json.loads(
        (paddlex_dir / "detections.json").read_text(encoding="utf-8")
    )
    deim_config_path = deim_dir / "benchmark_config.json"
    paddlex_config_path = paddlex_dir / "benchmark_config.json"
    if set(deim_index) != set(paddlex_index):
        missing_deim = sorted(set(paddlex_index) - set(deim_index))
        missing_paddlex = sorted(set(deim_index) - set(paddlex_index))
        raise RuntimeError(
            f"Result sets differ; missing DEIM={missing_deim}, missing PaddleX={missing_paddlex}"
        )

    manifest: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "layout": "DEIM left; PaddleX-L right; original pixels preserved",
        "threshold": "score > 0.5 for both models",
        "models": {
            "deim": deim_document.get("model", "DEIM"),
            "paddlex_by_group": paddlex_document.get("models_by_group", {}),
        },
        "source_benchmark_configs": {
            "deim": {
                "path": str(deim_config_path.resolve()),
                "sha256": sha256_file(deim_config_path),
            },
            "paddlex": {
                "path": str(paddlex_config_path.resolve()),
                "sha256": sha256_file(paddlex_config_path),
            },
        },
        "pairs": [],
    }
    group_outputs: dict[str, list[Path]] = {}
    for group, source_name in sorted(deim_index, key=lambda key: (key[0], key[1].casefold())):
        deim_item = deim_index[(group, source_name)]
        paddlex_item = paddlex_index[(group, source_name)]
        deim_path = Path(deim_item["visualization_path"])
        paddlex_path = Path(paddlex_item["visualization_path"])
        if not deim_path.is_file() or not paddlex_path.is_file():
            raise FileNotFoundError(
                f"Missing visualization for {group}/{source_name}: "
                f"{deim_path}, {paddlex_path}"
            )

        group_dir = output_dir / group
        group_dir.mkdir(parents=True, exist_ok=True)
        output_path = group_dir / f"{Path(source_name).stem}_DEIM_vs_PaddleX.png"
        comparison = compose_pair(deim_path, paddlex_path, group, source_name)
        comparison.save(output_path, format="PNG", optimize=True)
        group_outputs.setdefault(group, []).append(output_path)
        manifest["pairs"].append(
            {
                "group": group,
                "source_name": source_name,
                "deim_visualization": str(deim_path.resolve()),
                "paddlex_visualization": str(paddlex_path.resolve()),
                "comparison": str(output_path.resolve()),
                "deim_detection_count": int(deim_item["detection_count"]),
                "paddlex_detection_count": int(paddlex_item["detection_count"]),
                "paddlex_model": paddlex_document.get("models_by_group", {}).get(group),
            }
        )

    for group, paths in group_outputs.items():
        write_contact_sheet(paths, output_dir / f"{group}_contact_sheet.jpg")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Created {len(manifest['pairs'])} comparison pairs under {output_dir}")
    for group, paths in group_outputs.items():
        print(f"  {group}: {len(paths)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
