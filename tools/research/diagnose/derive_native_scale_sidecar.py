#!/usr/bin/env python3
"""Derive local area/SGQS/EFGL geometry diagnostics from query CSV exports.

The training repository has no binary label that combines area and short side.
This sidecar therefore reports the three native quantities independently:

* DEIM area-small: normalized ``w*h < 0.0025``;
* SGQS preferred level: log-nearest short side to ``[32, 64, 128]``;
* EFGL smallness: ``exp(-short_side/16)``.

Both direct model-output (raw cxcywh) and clipped 640-input geometry are kept.
No checkpoint forward pass and no existing diagnostic artifact are modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw


AREA_SMALL_THRESHOLD = 0.0025
SGQS_PRIORS_PX = (32.0, 64.0, 128.0)
EFGL_SHORT_SIDE_SCALE_PX = 16.0
EPSILON = 1e-6
PERCENTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write native area/SGQS/EFGL sidecars for query diagnostics."
    )
    parser.add_argument("directories", nargs="+", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def preferred_scale(short_side: float) -> str:
    if short_side <= 0.0 or not math.isfinite(short_side):
        return "invalid"
    level = min(
        range(len(SGQS_PRIORS_PX)),
        key=lambda index: (
            abs(
                math.log(
                    (short_side + EPSILON) / (SGQS_PRIORS_PX[index] + EPSILON)
                )
            ),
            index,
        ),
    )
    return f"P{level + 3}"


def percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, Any]:
    materialized = sorted(float(value) for value in values)
    if not materialized:
        return {"count": 0, "mean": None, "quantiles": {}}
    return {
        "count": len(materialized),
        "mean": sum(materialized) / len(materialized),
        "quantiles": {
            f"p{int(fraction * 100):02d}": percentile(materialized, fraction)
            for fraction in PERCENTILES
        },
    }


def ordered_levels(counter: Counter[str]) -> dict[str, int]:
    return {name: int(counter.get(name, 0)) for name in ("P3", "P4", "P5", "invalid")}


class GeometryAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.count = 0
        self.images: set[str] = set()
        self.source_levels: Counter[str] = Counter()
        self.raw_area_small = 0
        self.clipped_area_small = 0
        self.raw_levels: Counter[str] = Counter()
        self.clipped_levels: Counter[str] = Counter()
        self.raw_short_sides: list[float] = []
        self.clipped_short_sides: list[float] = []
        self.raw_efgl: list[float] = []
        self.clipped_efgl: list[float] = []
        self.area_flag_disagreement = 0
        self.sgqs_level_disagreement = 0

    def add(
        self,
        row: dict[str, str],
        *,
        normalized_width_field: str,
        normalized_height_field: str,
        input_size: int,
    ) -> None:
        normalized_width = max(0.0, float(row[normalized_width_field]))
        normalized_height = max(0.0, float(row[normalized_height_field]))
        raw_width = normalized_width * input_size
        raw_height = normalized_height * input_size
        raw_short = min(raw_width, raw_height)
        raw_area_small = normalized_width * normalized_height < AREA_SMALL_THRESHOLD

        clipped_width = max(0.0, float(row["input_clipped_width_px"]))
        clipped_height = max(0.0, float(row["input_clipped_height_px"]))
        clipped_short = min(clipped_width, clipped_height)
        clipped_area_small = (
            (clipped_width / input_size) * (clipped_height / input_size)
            < AREA_SMALL_THRESHOLD
        )

        raw_level = preferred_scale(raw_short)
        clipped_level = preferred_scale(clipped_short)
        raw_efgl = math.exp(-raw_short / EFGL_SHORT_SIDE_SCALE_PX)
        clipped_efgl = math.exp(-clipped_short / EFGL_SHORT_SIDE_SCALE_PX)

        self.count += 1
        self.images.add(row["image_key"])
        self.source_levels[row["feature_name"]] += 1
        self.raw_area_small += int(raw_area_small)
        self.clipped_area_small += int(clipped_area_small)
        self.raw_levels[raw_level] += 1
        self.clipped_levels[clipped_level] += 1
        self.raw_short_sides.append(raw_short)
        self.clipped_short_sides.append(clipped_short)
        self.raw_efgl.append(raw_efgl)
        self.clipped_efgl.append(clipped_efgl)
        self.area_flag_disagreement += int(raw_area_small != clipped_area_small)
        self.sgqs_level_disagreement += int(raw_level != clipped_level)

    def result(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.count,
            "images": len(self.images),
            "source_feature_level": ordered_levels(self.source_levels),
            "raw_model_output": {
                "area_small_count": self.raw_area_small,
                "area_small_fraction": self.raw_area_small / self.count if self.count else None,
                "sgqs_preferred_level": ordered_levels(self.raw_levels),
                "short_side_px": distribution(self.raw_short_sides),
                "efgl_smallness": distribution(self.raw_efgl),
            },
            "clipped_model_input": {
                "area_small_count": self.clipped_area_small,
                "area_small_fraction": (
                    self.clipped_area_small / self.count if self.count else None
                ),
                "sgqs_preferred_level": ordered_levels(self.clipped_levels),
                "short_side_px": distribution(self.clipped_short_sides),
                "efgl_smallness": distribution(self.clipped_efgl),
            },
            "raw_vs_clipped": {
                "area_small_flag_disagreement_count": self.area_flag_disagreement,
                "sgqs_preferred_level_disagreement_count": self.sgqs_level_disagreement,
            },
        }


def read_input_size(source_summary: dict[str, Any]) -> int:
    value = source_summary.get("model", {}).get("input_size")
    if value is None:
        value = source_summary.get("semantics", {}).get("size_proxy", {}).get(
            "input_width_px"
        )
    if not isinstance(value, int) or value <= 0:
        raise ValueError("Could not recover a positive integer input size")
    return value


def visualization_name(image_key: str, suffix: str) -> str:
    stem = str(Path(image_key).with_suffix(""))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_")
    return f"{safe}_{suffix}.png"


def geometry_encoding(row: dict[str, str], input_size: int) -> tuple[str, float, bool]:
    width = max(0.0, float(row["input_clipped_width_px"]))
    height = max(0.0, float(row["input_clipped_height_px"]))
    short_side = min(width, height)
    level = preferred_scale(short_side)
    smallness = math.exp(-short_side / EFGL_SHORT_SIDE_SCALE_PX)
    area_small = (width / input_size) * (height / input_size) < AREA_SMALL_THRESHOLD
    return level, smallness, area_small


def load_grouped_rows(
    path: Path, *, selected_only: bool = False
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if selected_only and not parse_bool(row["selected_top640"]):
                continue
            grouped.setdefault(row["image_key"], []).append(row)
    return grouped


def render_geometry_overlay(
    image_path: Path,
    output_path: Path,
    records: list[dict[str, str]],
    *,
    input_size: int,
    box_prefix: str,
    title: str,
) -> None:
    level_colors = {
        "P3": (255, 70, 70),
        "P4": (255, 205, 40),
        "P5": (30, 220, 255),
        "invalid": (190, 190, 190),
    }
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    base_width = max(1, int(round(min(image.size) / 640.0)))

    prepared = []
    for row in records:
        level, smallness, area_small = geometry_encoding(row, input_size)
        prepared.append((smallness, level, area_small, row))
    # Larger short-side smallness is drawn last so the most scale-sensitive
    # predictions remain visible in dense overlays.
    prepared.sort(key=lambda item: item[0])
    for smallness, level, area_small, row in prepared:
        box = tuple(float(row[f"{box_prefix}_{axis}"]) for axis in ("x1", "y1", "x2", "y2"))
        alpha = max(35, min(225, int(round(35 + 190 * smallness))))
        color = (*level_colors[level], alpha)
        width = base_width * (3 if area_small else 1)
        draw.rectangle(box, outline=color, width=width)

    legend = (
        f"{title} | color: SGQS P3=red P4=yellow P5=cyan | "
        "opacity: EFGL exp(-short/16) | thick: area<0.0025"
    )
    text_box = draw.textbbox((7, 6), legend)
    draw.rectangle(
        (0, 0, min(image.width, text_box[2] + 14), text_box[3] + 12),
        fill=(0, 0, 0, 205),
    )
    draw.text((7, 6), legend, fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(image, overlay).convert("RGB").save(output_path)


def render_native_visualizations(
    directory: Path,
    source_summary: dict[str, Any],
    encoder_csv: Path,
    decoder_csv: Path,
    input_size: int,
) -> dict[str, Any]:
    data_dir_value = source_summary.get("paths", {}).get("data_dir")
    if not data_dir_value:
        raise ValueError("Source summary does not contain paths.data_dir")
    data_dir = Path(data_dir_value)
    encoder_by_image = load_grouped_rows(encoder_csv, selected_only=True)
    decoder_by_image = load_grouped_rows(decoder_csv)
    if set(encoder_by_image) != set(decoder_by_image):
        raise ValueError("Encoder and decoder CSV image sets differ")

    output_dir = directory / "native_scale_visualizations"
    output_files: list[str] = []
    for image_key in sorted(decoder_by_image):
        image_path = data_dir / Path(image_key)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        encoder_output = output_dir / visualization_name(
            image_key, "native_selected_encoder_geometry"
        )
        decoder_output = output_dir / visualization_name(
            image_key, "native_decoder_geometry"
        )
        render_geometry_overlay(
            image_path,
            encoder_output,
            encoder_by_image[image_key],
            input_size=input_size,
            box_prefix="encoder",
            title="Selected encoder Top-640",
        )
        render_geometry_overlay(
            image_path,
            decoder_output,
            decoder_by_image[image_key],
            input_size=input_size,
            box_prefix="final",
            title="Raw decoder 640 queries",
        )
        output_files.extend(
            [str(encoder_output.relative_to(directory)), str(decoder_output.relative_to(directory))]
        )
    return {
        "directory": str(output_dir),
        "images": len(encoder_by_image),
        "png_count": len(output_files),
        "files": output_files,
        "encoding": {
            "color": "clipped-input SGQS preferred level: P3 red, P4 yellow, P5 cyan",
            "opacity": "clipped-input EFGL continuous smallness exp(-short_side/16)",
            "line_width": "thick when clipped normalized area < 0.0025",
        },
    }


def derive_directory(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    encoder_csv = directory / "encoder_candidates_before_topk.csv"
    decoder_csv = directory / "decoder_queries_without_post_topk.csv"
    source_json = directory / "query_diagnostic_summary.json"
    for path in (encoder_csv, decoder_csv, source_json):
        if not path.is_file():
            raise FileNotFoundError(path)

    with source_json.open("r", encoding="utf-8") as handle:
        source_summary = json.load(handle)
    input_size = read_input_size(source_summary)

    dense = GeometryAccumulator("all_dense_encoder_candidates")
    selected = GeometryAccumulator("selected_encoder_top640")
    with encoder_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dense.add(
                row,
                normalized_width_field="encoder_w",
                normalized_height_field="encoder_h",
                input_size=input_size,
            )
            if parse_bool(row["selected_top640"]):
                selected.add(
                    row,
                    normalized_width_field="encoder_w",
                    normalized_height_field="encoder_h",
                    input_size=input_size,
                )

    decoder = GeometryAccumulator("raw_decoder_queries")
    with decoder_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            decoder.add(
                row,
                normalized_width_field="final_w",
                normalized_height_field="final_h",
                input_size=input_size,
            )

    visualizations = render_native_visualizations(
        directory, source_summary, encoder_csv, decoder_csv, input_size
    )

    result = {
        "protocol_version": "deim_query_native_scale_sidecar_v1",
        "purpose": (
            "Report local area, SGQS scale, and EFGL short-side quantities "
            "separately; no native combined binary small label exists."
        ),
        "definitions": {
            "area_small": {
                "formula": "normalized_width * normalized_height < 0.0025",
                "comparison": "strict less-than",
                "threshold": AREA_SMALL_THRESHOLD,
            },
            "sgqs_preferred_level": {
                "formula": "argmin_l abs(log((short_side+1e-6)/(prior_l+1e-6)))",
                "priors_px": list(SGQS_PRIORS_PX),
                "tie_break": "lower feature level wins",
                "meaning": "preferred supervision scale, not a small label",
            },
            "efgl_smallness": {
                "formula": "exp(-short_side_px/16)",
                "short_side_scale_px": EFGL_SHORT_SIDE_SCALE_PX,
                "meaning": "continuous weight component, not a hard threshold",
            },
            "coordinate_views": {
                "raw_model_output": "width/height read directly from normalized model cxcywh",
                "clipped_model_input": (
                    "width/height after clipping to the fixed input canvas; analogous "
                    "to sanitized training-target geometry"
                ),
            },
        },
        "limits": [
            "These are predicted-box geometry analogues; the input directory has no GT labels.",
            "P3/P4/P5 source feature levels and SGQS preferred levels are not object-size labels.",
            "The source hybrid area-or-short-side label is a separate diagnostic convention, not a native training rule.",
        ],
        "source": {
            "directory": str(directory),
            "protocol_version": source_summary.get("protocol_version"),
            "input_size": input_size,
            "files": {
                encoder_csv.name: sha256_file(encoder_csv),
                decoder_csv.name: sha256_file(decoder_csv),
                source_json.name: sha256_file(source_json),
            },
        },
        "integrity": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "statistics": {
            "all_dense_encoder_candidates": dense.result(),
            "selected_encoder_top640": selected.result(),
            "raw_decoder_queries": decoder.result(),
        },
        "visualizations": visualizations,
    }

    output_json = directory / "native_scale_summary.json"
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    scopes = result["statistics"]
    lines = [
        "# Native area / SGQS / EFGL geometry sidecar",
        "",
        "This report keeps the repository's three native geometry quantities separate. "
        "There is **no** native binary `area OR short-side => small` rule.",
        "",
        "- Area-small: normalized `w*h < 0.0025`.",
        "- SGQS: short side is assigned log-nearest to `[32,64,128]` as P3/P4/P5; this is a preferred supervision level, not a small label.",
        "- EFGL: `exp(-short_side/16)` is continuous; 16 px is not a cutoff.",
        "- Raw geometry is the direct model cxcywh output. Clipped geometry is bounded to the input canvas and is the closer analogue to sanitized training targets.",
        "- All values describe predicted boxes only. There are no GT annotations here, so they are not recall, precision, TP/FP, or AP.",
        "- Native visualization encoding: color = SGQS preferred P3/P4/P5, opacity = continuous EFGL smallness, thick outline = area-small.",
        "",
        "| Scope | Rows | Raw area-small | Clipped area-small | Raw SGQS P3/P4/P5 | Clipped SGQS P3/P4/P5 | Raw EFGL mean | Clipped EFGL mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in (
        "all_dense_encoder_candidates",
        "selected_encoder_top640",
        "raw_decoder_queries",
    ):
        item = scopes[key]
        raw = item["raw_model_output"]
        clipped = item["clipped_model_input"]
        raw_levels = raw["sgqs_preferred_level"]
        clipped_levels = clipped["sgqs_preferred_level"]
        lines.append(
            "| {name} | {rows:,} | {raw_area:,} | {clip_area:,} | "
            "{rp3}/{rp4}/{rp5} | {cp3}/{cp4}/{cp5} | {raw_efgl:.8f} | "
            "{clip_efgl:.8f} |".format(
                name=item["name"],
                rows=item["rows"],
                raw_area=raw["area_small_count"],
                clip_area=clipped["area_small_count"],
                rp3=raw_levels["P3"],
                rp4=raw_levels["P4"],
                rp5=raw_levels["P5"],
                cp3=clipped_levels["P3"],
                cp4=clipped_levels["P4"],
                cp5=clipped_levels["P5"],
                raw_efgl=raw["efgl_smallness"]["mean"],
                clip_efgl=clipped["efgl_smallness"]["mean"],
            )
        )
    lines.extend(
        [
            "",
            "Full short-side and EFGL quantiles, raw-vs-clipped disagreement counts, "
            "source hashes, and exact definitions are in `native_scale_summary.json`.",
            "",
            f"`native_scale_visualizations/` contains {visualizations['png_count']} PNGs "
            f"({visualizations['images']} images × selected-encoder/decoder views).",
        ]
    )
    (directory / "NATIVE_SCALE_README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    args = parse_args()
    for directory in args.directories:
        result = derive_directory(directory)
        decoder = result["statistics"]["raw_decoder_queries"]
        clipped = decoder["clipped_model_input"]
        levels = clipped["sgqs_preferred_level"]
        print(
            f"{directory}: rows={decoder['rows']} "
            f"area_small={clipped['area_small_count']} "
            f"SGQS={levels['P3']}/{levels['P4']}/{levels['P5']} "
            f"EFGL_mean={clipped['efgl_smallness']['mean']:.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
