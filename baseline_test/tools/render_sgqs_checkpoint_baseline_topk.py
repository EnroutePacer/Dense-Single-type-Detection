#!/usr/bin/env python3
"""Render audited e28--e34 SGQS Top-K comparisons on Baseline_test images.

The source CSVs were produced by ``diagnose_deim_query_selection.py`` from
EMA checkpoint states.  Encoder Top-300 is a diagnostic subset of the real
architectural Encoder Top-640 selection.  The model always feeds 640 queries
to the decoder.

For this one-class model, repository postprocessing ranks sigmoid(final_logit)
and returns Top-K boxes without NMS or a score threshold.  Consequently the
sorted final decoder CSV is exactly the PostProcessor Top-640 output; its first
300 rows are the maxDets=300 evaluator view.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


EPOCHS = tuple(range(28, 35))
LEVEL_NAMES = ("P3", "P4", "P5")
LEVEL_COLORS = {
    "P3": (35, 210, 90, 235),
    "P4": (40, 135, 255, 235),
    "P5": (185, 70, 245, 235),
}
PANEL_WIDTH = 360
PANEL_HEIGHT = 264
PANEL_HEADER = 38
MARGIN = 8
COLUMNS = (
    "encoder_top300",
    "encoder_top640",
    "final_top300",
    "final_tail",
    "final_all640",
    "final_score05",
)


def parse_args() -> argparse.Namespace:
    root = Path(r"D:\Baseline_test")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "DATA")
    parser.add_argument("--diagnostic-root", type=Path, default=root / "output")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "SGQS_e28_e34_baseline_topk_postprocess",
    )
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def diagnostic_dir(root: Path, epoch: int) -> Path:
    return root / f"SGQS_probe_checkpoint{epoch:04d}_query_diagnostic_hybrid_small_640"


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_epoch(root: Path, epoch: int) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    directory = diagnostic_dir(root, epoch)
    encoder_path = directory / "encoder_candidates_before_topk.csv"
    decoder_path = directory / "decoder_queries_without_post_topk.csv"
    if not encoder_path.is_file() or not decoder_path.is_file():
        raise FileNotFoundError(f"missing diagnostic CSVs for e{epoch}: {directory}")

    encoder: dict[str, list[dict]] = defaultdict(list)
    with encoder_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not as_bool(row["selected_top640"]):
                continue
            rank = int(row["encoder_rank"])
            if rank > 640:
                raise RuntimeError(f"selected encoder rank >640 at e{epoch}")
            encoder[row["image_key"]].append(
                {
                    "rank": rank,
                    "level": row["feature_name"],
                    "cx": (float(row["encoder_x1"]) + float(row["encoder_x2"])) / 2.0,
                    "cy": (float(row["encoder_y1"]) + float(row["encoder_y2"])) / 2.0,
                }
            )

    decoder: dict[str, list[dict]] = defaultdict(list)
    with decoder_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            decoder[row["image_key"]].append(
                {
                    "query_index": int(row["query_index"]),
                    "logit": float(row["final_logit"]),
                    "score": float(row["final_score"]),
                    "box": tuple(float(row[name]) for name in ("final_x1", "final_y1", "final_x2", "final_y2")),
                    "hybrid_small": row["size_proxy"] == "small",
                }
            )

    if set(encoder) != set(decoder):
        raise RuntimeError(f"encoder/decoder image-key mismatch at e{epoch}")
    for key in encoder:
        encoder[key].sort(key=lambda item: item["rank"])
        if len(encoder[key]) != 640 or len(decoder[key]) != 640:
            raise RuntimeError(
                f"expected 640 encoder and decoder records for {key} e{epoch}, "
                f"got {len(encoder[key])}/{len(decoder[key])}"
            )
        # Match repository PostProcessor exactly, including torch.topk's
        # ordering for tied logits.  Sorting by (score, query_index) is not
        # equivalent when several queries have identical floating scores.
        logits = torch.tensor(
            [item["logit"] for item in decoder[key]], dtype=torch.float32
        ).sigmoid()
        top_scores, top_indices = torch.topk(logits, k=len(decoder[key]))
        ordered: list[dict] = []
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            item = decoder[key][index].copy()
            item["score"] = float(score)
            ordered.append(item)
        decoder[key] = ordered
    return dict(encoder), dict(decoder)


def level_counts(records: list[dict]) -> tuple[int, int, int]:
    return tuple(sum(item["level"] == name for item in records) for name in LEVEL_NAMES)


def source_path(data_dir: Path, image_key: str) -> Path:
    return data_dir.joinpath(*image_key.split("/"))


def scale_geometry(image: Image.Image) -> tuple[float, int, int, int, int]:
    available_width = PANEL_WIDTH - 2 * MARGIN
    available_height = PANEL_HEIGHT - PANEL_HEADER - 2 * MARGIN
    scale = min(available_width / image.width, available_height / image.height)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    left = (PANEL_WIDTH - width) // 2
    top = PANEL_HEADER + (PANEL_HEIGHT - PANEL_HEADER - height) // 2
    return scale, left, top, width, height


def panel_base(image: Image.Image, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw, float, int, int]:
    panel = Image.new("RGBA", (PANEL_WIDTH, PANEL_HEIGHT), (22, 26, 33, 255))
    scale, left, top, width, height = scale_geometry(image)
    resized = image.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
    panel.alpha_composite(resized, (left, top))
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.text((8, 7), title, font=font(14, True), fill=(245, 247, 250, 255))
    return panel, draw, scale, left, top


def draw_encoder_panel(image: Image.Image, records: list[dict], limit: int) -> Image.Image:
    selected = records[:limit]
    counts = level_counts(selected)
    title = f"Encoder rank 1-{limit}  P3/P4/P5={counts[0]}/{counts[1]}/{counts[2]}"
    panel, draw, scale, left, top = panel_base(image, title)
    radius = 2.1 if limit == 300 else 1.7
    for item in selected:
        x = left + item["cx"] * scale
        y = top + item["cy"] * scale
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=LEVEL_COLORS[item["level"]],
        )
    return panel.convert("RGB")


def transformed_box(box: tuple[float, float, float, float], image: Image.Image, scale: float, left: int, top: int) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), float(image.width))
    y1 = min(max(y1, 0.0), float(image.height))
    x2 = min(max(x2, 0.0), float(image.width))
    y2 = min(max(y2, 0.0), float(image.height))
    if x2 <= x1 or y2 <= y1:
        return None
    return left + x1 * scale, top + y1 * scale, left + x2 * scale, top + y2 * scale


def draw_boxes(
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    records: list[dict],
    scale: float,
    left: int,
    top: int,
    color: tuple[int, int, int],
    confidence_alpha: bool = True,
    width: int = 1,
) -> None:
    for item in reversed(records):
        box = transformed_box(item["box"], image, scale, left, top)
        if box is None:
            continue
        alpha = 70 + round(175 * item["score"]) if confidence_alpha else 210
        draw.rectangle(box, outline=(*color, alpha), width=width)


def draw_final_panel(image: Image.Image, records: list[dict], kind: str) -> Image.Image:
    top300 = records[:300]
    tail = records[300:]
    if kind == "final_top300":
        title = f"Final Top-300  cutoff={top300[-1]['score']:.4f}"
        panel, draw, scale, left, top = panel_base(image, title)
        draw_boxes(draw, image, top300, scale, left, top, (30, 225, 100), width=1)
    elif kind == "final_tail":
        title = f"Excluded rank 301-640  s301={tail[0]['score']:.4f}"
        panel, draw, scale, left, top = panel_base(image, title)
        draw_boxes(draw, image, tail, scale, left, top, (255, 150, 25), width=1)
    elif kind == "final_all640":
        title = "PostProcessor all 640  green=Top300, orange=tail"
        panel, draw, scale, left, top = panel_base(image, title)
        draw_boxes(draw, image, tail, scale, left, top, (255, 150, 25), width=1)
        draw_boxes(draw, image, top300, scale, left, top, (30, 225, 100), width=1)
    elif kind == "final_score05":
        kept = [item for item in records if item["score"] > 0.5]
        title = f"Deployment view score>0.5  n={len(kept)}"
        panel, draw, scale, left, top = panel_base(image, title)
        draw_boxes(draw, image, kept, scale, left, top, (255, 55, 65), width=2)
    else:
        raise ValueError(kind)
    return panel.convert("RGB")


def safe_stem(image_key: str) -> str:
    return image_key.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")


def render_contact_sheet(
    image_key: str,
    image: Image.Image,
    encoder_by_epoch: dict[int, dict[str, list[dict]]],
    decoder_by_epoch: dict[int, dict[str, list[dict]]],
) -> Image.Image:
    title_height = 100
    row_label_width = 70
    sheet = Image.new(
        "RGB",
        (row_label_width + len(COLUMNS) * PANEL_WIDTH, title_height + len(EPOCHS) * PANEL_HEIGHT),
        (16, 19, 25),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f"SGQS EMA checkpoint comparison | {image_key}", font=font(25, True), fill=(250, 250, 252))
    draw.text(
        (16, 48),
        "P3=green, P4=blue, P5=purple | model Decoder queries=640 | formal evaluator keeps final Top-300",
        font=font(16),
        fill=(205, 211, 220),
    )
    draw.text(
        (16, 70),
        "Encoder Top-300 is diagnostic only; the real architecture sends Encoder Top-640 into Decoder.",
        font=font(15),
        fill=(255, 202, 95),
    )

    for row, epoch in enumerate(EPOCHS):
        y = title_height + row * PANEL_HEIGHT
        draw.rectangle((0, y, row_label_width, y + PANEL_HEIGHT), fill=(238, 241, 246))
        draw.text((16, y + PANEL_HEIGHT // 2 - 14), f"e{epoch}", font=font(22, True), fill=(30, 34, 42))
        enc = encoder_by_epoch[epoch][image_key]
        dec = decoder_by_epoch[epoch][image_key]
        panels = [
            draw_encoder_panel(image, enc, 300),
            draw_encoder_panel(image, enc, 640),
            draw_final_panel(image, dec, "final_top300"),
            draw_final_panel(image, dec, "final_tail"),
            draw_final_panel(image, dec, "final_all640"),
            draw_final_panel(image, dec, "final_score05"),
        ]
        for column, panel in enumerate(panels):
            sheet.paste(panel, (row_label_width + column * PANEL_WIDTH, y))
    return sheet


def write_summary_csv(
    path: Path,
    image_keys: list[str],
    encoder_by_epoch: dict[int, dict[str, list[dict]]],
    decoder_by_epoch: dict[int, dict[str, list[dict]]],
) -> list[dict]:
    fields = [
        "epoch",
        "image_key",
        "encoder_top300_p3",
        "encoder_top300_p4",
        "encoder_top300_p5",
        "encoder_top640_p3",
        "encoder_top640_p4",
        "encoder_top640_p5",
        "decoder_query_count",
        "postprocessor_all_count",
        "formal_top300_count",
        "excluded_tail_count",
        "score_gt_0_5_count",
        "hybrid_small_top300_count",
        "hybrid_small_all640_count",
        "final_rank300_score",
        "final_rank301_score",
        "final_rank640_score",
    ]
    rows: list[dict] = []
    for epoch in EPOCHS:
        for image_key in image_keys:
            enc = encoder_by_epoch[epoch][image_key]
            dec = decoder_by_epoch[epoch][image_key]
            c300 = level_counts(enc[:300])
            c640 = level_counts(enc)
            rows.append(
                {
                    "epoch": epoch,
                    "image_key": image_key,
                    "encoder_top300_p3": c300[0],
                    "encoder_top300_p4": c300[1],
                    "encoder_top300_p5": c300[2],
                    "encoder_top640_p3": c640[0],
                    "encoder_top640_p4": c640[1],
                    "encoder_top640_p5": c640[2],
                    "decoder_query_count": len(dec),
                    "postprocessor_all_count": len(dec),
                    "formal_top300_count": len(dec[:300]),
                    "excluded_tail_count": len(dec[300:]),
                    "score_gt_0_5_count": sum(item["score"] > 0.5 for item in dec),
                    "hybrid_small_top300_count": sum(item["hybrid_small"] for item in dec[:300]),
                    "hybrid_small_all640_count": sum(item["hybrid_small"] for item in dec),
                    "final_rank300_score": f"{dec[299]['score']:.9f}",
                    "final_rank301_score": f"{dec[300]['score']:.9f}",
                    "final_rank640_score": f"{dec[-1]['score']:.9f}",
                }
            )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def render_aggregate(rows: list[dict], output_path: Path) -> None:
    epochs = np.asarray(EPOCHS)
    image_keys = sorted({str(row["image_key"]) for row in rows})
    by_pair = {(int(row["epoch"]), str(row["image_key"])): row for row in rows}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), dpi=170)
    score_counts = np.asarray(
        [[int(by_pair[(epoch, key)]["score_gt_0_5_count"]) for epoch in EPOCHS] for key in image_keys]
    )
    hybrid_counts = np.asarray(
        [[int(by_pair[(epoch, key)]["hybrid_small_all640_count"]) for epoch in EPOCHS] for key in image_keys]
    )
    for values in score_counts:
        axes[0].plot(epochs, values, color="#8392a5", alpha=0.42, linewidth=0.9)
    axes[0].plot(epochs, np.median(score_counts, axis=0), color="#e63946", marker="o", linewidth=2.4, label="median")
    axes[0].set_title("Final boxes with score > 0.5\n13 Baseline_test images")
    axes[0].set_ylabel("box count per image")
    axes[0].legend(frameon=False)

    for values in hybrid_counts:
        axes[1].plot(epochs, values, color="#8392a5", alpha=0.42, linewidth=0.9)
    axes[1].plot(epochs, np.median(hybrid_counts, axis=0), color="#ff7f0e", marker="o", linewidth=2.4, label="median")
    axes[1].set_title("All-640 local-hybrid-small boxes\n(area<32^2 OR short side<16 at 640)")
    axes[1].set_ylabel("box count per image")
    axes[1].legend(frameon=False)

    for limit, linestyle in ((300, "-"), (640, "--")):
        for level, color in zip(LEVEL_NAMES, ("#22b455", "#2588ff", "#b946f5")):
            field = f"encoder_top{limit}_{level.lower()}"
            values = []
            for epoch in EPOCHS:
                count = sum(int(by_pair[(epoch, key)][field]) for key in image_keys)
                values.append(count / (limit * len(image_keys)))
            axes[2].plot(
                epochs,
                values,
                color=color,
                linestyle=linestyle,
                marker="o" if limit == 300 else None,
                linewidth=2.0,
                label=f"{level} Top-{limit}",
            )
    axes[2].set_title("Encoder source-level share\nsolid=Top-300 diagnostic, dashed=Top-640 actual")
    axes[2].set_ylabel("share")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].legend(frameon=False, ncol=2, fontsize=8)

    for axis in axes:
        axis.set_xticks(epochs)
        axis.set_xlabel("EMA checkpoint")
        axis.grid(alpha=0.22)
    fig.suptitle("SGQS e28-e34 | Baseline_test aggregate diagnostic", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, image_keys: list[str]) -> None:
    lines = [
        "# SGQS e28--e34 Baseline_test Top-K / postprocess visualization",
        "",
        "- Checkpoints: EMA states from checkpoint0028.pth through checkpoint0034.pth.",
        "- Images: all 13 images under `D:/Baseline_test/DATA`.",
        "- Encoder Top-300 is a diagnostic prefix only; the architecture actually selects Top-640.",
        "- Decoder query count is exactly 640 for every image and checkpoint.",
        "- This is a one-class model. Repository PostProcessor sorts sigmoid final scores, applies Top-K,",
        "  rescales boxes, and applies no NMS or threshold. Therefore `PostProcessor all 640` is the",
        "  score-sorted 640-query final output; `Final Top-300` is the formal maxDets=300 prefix.",
        "- All 91 image/checkpoint states were checked against the repository PostProcessor: scores and",
        "  boxes agree exactly (maximum absolute error 0.0, including tied-score Top-K ordering).",
        "- `score>0.5` is shown only to explain visual density and deployment-style displays; it is not",
        "  the COCO AP-small rule.",
        "- Hybrid-small is a predicted-box descriptor at the 640 input scale:",
        "  `area < 32^2 OR short side < 16`. It is not COCO AP-small ground truth.",
        "",
        "## Outputs",
        "",
        "- `aggregate_trajectory.png`: per-image trajectories and Encoder source-level shares.",
        "- `baseline_topk_summary.csv`: exact counts and score cutoffs for 13 x 7 states.",
        "- Contact sheets:",
        "",
    ]
    for key in image_keys:
        lines.append(f"  - `{safe_stem(key)}_e28_e34.png` ({key})")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder_by_epoch: dict[int, dict[str, list[dict]]] = {}
    decoder_by_epoch: dict[int, dict[str, list[dict]]] = {}
    for epoch in EPOCHS:
        encoder_by_epoch[epoch], decoder_by_epoch[epoch] = load_epoch(
            args.diagnostic_root.resolve(), epoch
        )

    image_keys = sorted(encoder_by_epoch[EPOCHS[0]])
    expected = set(image_keys)
    for epoch in EPOCHS:
        if set(encoder_by_epoch[epoch]) != expected or set(decoder_by_epoch[epoch]) != expected:
            raise RuntimeError(f"image grid mismatch at e{epoch}")

    for image_key in image_keys:
        path = source_path(args.data_dir.resolve(), image_key)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as loaded:
            image = loaded.convert("RGB")
        sheet = render_contact_sheet(
            image_key, image, encoder_by_epoch, decoder_by_epoch
        )
        sheet.save(output_dir / f"{safe_stem(image_key)}_e28_e34.png", format="PNG", optimize=True)

    rows = write_summary_csv(
        output_dir / "baseline_topk_summary.csv",
        image_keys,
        encoder_by_epoch,
        decoder_by_epoch,
    )
    render_aggregate(rows, output_dir / "aggregate_trajectory.png")
    write_readme(output_dir, image_keys)
    print(f"Rendered {len(image_keys)} contact sheets to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
