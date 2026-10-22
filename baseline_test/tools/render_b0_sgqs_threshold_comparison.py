#!/usr/bin/env python3
"""Render native-resolution B0/SGQS encoder and decoder comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps


LEVEL_COLORS = {
    "P3": (20, 215, 85, 235),
    "P4": (25, 125, 255, 235),
    "P5": (205, 45, 245, 235),
}
SIZE_COLORS = {
    "small": (255, 35, 35),
    "medium": (255, 165, 0),
    "large": (20, 195, 235),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--b0-raw-dir", type=Path, required=True)
    parser.add_argument("--sgqs-raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--topk", type=int, default=300)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["image_key"]].append(row)
    return grouped


def ordered_clipped_box(
    values: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in values)
    left, right = sorted((max(0.0, min(width - 1.0, x1)), max(0.0, min(width - 1.0, x2))))
    top, bottom = sorted((max(0.0, min(height - 1.0, y1)), max(0.0, min(height - 1.0, y2))))
    return left, top, right, bottom


def fit_font(text: str, max_width: int, preferred_size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    ]
    for size in range(preferred_size, 9, -1):
        for path in candidates:
            if not path.exists():
                continue
            font = ImageFont.truetype(str(path), size=size)
            if font.getlength(text) <= max_width:
                return font
    return ImageFont.load_default()


def draw_encoder(image: Image.Image, rows: list[dict[str, str]]) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    radius = max(3, min(6, round(min(canvas.size) / 170)))
    for row in rows:
        if row["selected_top640"].lower() != "true":
            continue
        box = ordered_clipped_box(
            (row["encoder_x1"], row["encoder_y1"], row["encoder_x2"], row["encoder_y2"]),
            canvas.width,
            canvas.height,
        )
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        color = LEVEL_COLORS[row["feature_name"]]
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=color,
            outline=(255, 255, 255, 230),
            width=1,
        )
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: tuple[int, int, int, int],
    width: int,
    dash: int,
    gap: int,
) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 0:
        return
    ux, uy = dx / length, dy / length
    offset = 0.0
    while offset < length:
        finish = min(length, offset + dash)
        draw.line(
            (
                start[0] + ux * offset,
                start[1] + uy * offset,
                start[0] + ux * finish,
                start[1] + uy * finish,
            ),
            fill=fill,
            width=width,
        )
        offset += dash + gap


def dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    fill: tuple[int, int, int, int],
    width: int,
) -> None:
    x1, y1, x2, y2 = box
    dash = max(6, width * 2)
    gap = max(3, width)
    for start, end in (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ):
        dashed_line(draw, start, end, fill, width, dash, gap)


def draw_decoder(
    image: Image.Image,
    rows: list[dict[str, str]],
    threshold: float,
    topk: int,
) -> tuple[Image.Image, dict[str, int]]:
    canvas = image.convert("RGBA")
    low_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    high_overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    low_draw = ImageDraw.Draw(low_overlay)
    high_draw = ImageDraw.Draw(high_overlay)
    thick_width = max(3, min(6, round(min(canvas.size) / 180)))

    ranked = sorted(rows, key=lambda row: float(row["final_score"]), reverse=True)
    top_queries = {row["query_index"] for row in ranked[:topk]}
    counts = {"all": len(rows), "score_ge_threshold": 0, "solid_topk": 0, "dashed_outside_topk": 0}

    for row in sorted(rows, key=lambda item: float(item["final_score"])):
        score = float(row["final_score"])
        color = SIZE_COLORS[row["size_proxy"]]
        box = ordered_clipped_box(
            (row["final_x1"], row["final_y1"], row["final_x2"], row["final_y2"]),
            canvas.width,
            canvas.height,
        )
        if score < threshold:
            low_draw.rectangle(box, outline=(*color, 80), width=1)
            continue
        counts["score_ge_threshold"] += 1
        if row["query_index"] in top_queries:
            counts["solid_topk"] += 1
            high_draw.rectangle(box, outline=(*color, 255), width=thick_width)
        else:
            counts["dashed_outside_topk"] += 1
            dashed_rectangle(high_draw, box, (*color, 255), thick_width)

    return (
        Image.alpha_composite(Image.alpha_composite(canvas, low_overlay), high_overlay).convert("RGB"),
        counts,
    )


def compose_row(
    label: str,
    original: Image.Image,
    encoder: Image.Image,
    decoder: Image.Image,
    threshold: float,
    title_height: int,
) -> Image.Image:
    width, height = original.size
    row = Image.new("RGB", (width * 3, height + title_height), (28, 28, 28))
    titles = (
        f"{label} | Original",
        f"{label} | Encoder (P3 green / P4 blue / P5 magenta)",
        f"{label} | Output (T={threshold:.2f}; thin:<T / solid:Top300 / dashed:>=T outside Top300)",
    )
    for index, (title, panel) in enumerate(zip(titles, (original, encoder, decoder))):
        x = index * width
        font = fit_font(title, width - 14)
        draw = ImageDraw.Draw(row)
        bbox = draw.textbbox((0, 0), title, font=font)
        text_height = bbox[3] - bbox[1]
        draw.text((x + 7, (title_height - text_height) / 2 - bbox[1]), title, fill=(240, 240, 240), font=font)
        row.paste(panel, (x, title_height))
    return row


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "B0 e34": {
            "encoder": load_rows(args.b0_raw_dir / "encoder_candidates_before_topk.csv"),
            "decoder": load_rows(args.b0_raw_dir / "decoder_queries_without_post_topk.csv"),
        },
        "SGQS e34": {
            "encoder": load_rows(args.sgqs_raw_dir / "encoder_candidates_before_topk.csv"),
            "decoder": load_rows(args.sgqs_raw_dir / "decoder_queries_without_post_topk.csv"),
        },
    }

    stats: list[dict[str, object]] = []
    image_paths = sorted(
        (path for path in args.data_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}),
        key=lambda path: path.relative_to(args.data_dir).as_posix().lower(),
    )
    for index, image_path in enumerate(image_paths, start=1):
        image_key = image_path.relative_to(args.data_dir).as_posix()
        original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        title_height = 48
        rendered_rows: list[Image.Image] = []
        for label, data in models.items():
            encoder = draw_encoder(original, data["encoder"][image_key])
            decoder, counts = draw_decoder(
                original, data["decoder"][image_key], args.threshold, args.topk
            )
            rendered_rows.append(
                compose_row(label, original, encoder, decoder, args.threshold, title_height)
            )
            stats.append(
                {
                    "image_key": image_key,
                    "model": label,
                    "threshold": args.threshold,
                    **counts,
                }
            )

        comparison = Image.new(
            "RGB",
            (rendered_rows[0].width, sum(row.height for row in rendered_rows)),
            (28, 28, 28),
        )
        comparison.paste(rendered_rows[0], (0, 0))
        comparison.paste(rendered_rows[1], (0, rendered_rows[0].height))
        relative_output = image_path.relative_to(args.data_dir).with_suffix("")
        output_path = args.output_dir / relative_output.parent / f"{relative_output.name}_threshold_{args.threshold:.2f}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(output_path, optimize=True)
        print(f"[{index:02d}/{len(image_paths):02d}] {image_key} -> {output_path}")

    csv_path = args.output_dir / "threshold_counts.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    (args.output_dir / "threshold_counts.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
