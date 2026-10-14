#!/usr/bin/env python3
"""Inspect DEIM/D-FINE encoder candidates before Top-K and raw decoder queries.

This diagnostic intentionally does not call the repository PostProcessor, so no
postprocessor Top-K, score threshold, or NMS is applied.  It preserves the
model's architectural encoder Top-K=640 path, but instruments that boundary and
exports all dense encoder candidates before selection.  Feeding all 8,400
locations through the decoder would change the trained architecture and make
query self-attention prohibitively expensive; that is not done here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

BENCHMARK_DIR = Path(__file__).resolve().parents[1] / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import benchmark_deim_cpu as base


PROTOCOL_VERSION = "deim_query_selection_diagnostic_640_v3_hybrid_input_scale"

# The training repository does not define a binary area+short-side label.  This
# diagnostic therefore keeps the two measurements explicit and combines them
# with a documented OR rule.  All thresholds are evaluated after clipping in
# the fixed model-input coordinate system, never in the source-image space.
SMALL_AREA_SIDE_PX = 32.0
MEDIUM_AREA_SIDE_PX = 96.0
SMALL_SHORT_SIDE_PX = 16.0
SGQS_SCALE_PRIORS_PX = (32.0, 64.0, 128.0)
SGQS_SCALE_BOUNDARIES_PX = tuple(
    math.sqrt(left * right)
    for left, right in zip(SGQS_SCALE_PRIORS_PX, SGQS_SCALE_PRIORS_PX[1:])
)
SIZE_NAMES = ("small", "medium", "large")


def parse_args() -> argparse.Namespace:
    root = base.default_baseline_root()
    parser = argparse.ArgumentParser(
        description=(
            "Dump every dense encoder candidate before Top-K and all 640 raw "
            "decoder queries without repository postprocessing."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=root / "DATA")
    parser.add_argument(
        "--config",
        type=Path,
        default=root
        / "models"
        / "DEIM"
        / "configs"
        / "table_cells"
        / "dfine_hgnetv2_m_table_cell.yml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "DEIM" / "outputs" / "last.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "DEIM_query_diagnostic_no_post_topk_640",
    )
    parser.add_argument("--groups", nargs="+", default=["wired", "wireless"])
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity", default="2")
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.input_size < 320 or args.input_size % 32:
        raise ValueError("--input-size must be >=320 and divisible by 32")
    if args.threads != 1:
        raise ValueError("This diagnostic is fixed to --threads 1")
    if base.parse_affinity(args.affinity) != [2]:
        raise ValueError("This diagnostic is fixed to --affinity 2")


def cxcywh_to_xyxy_pixels(
    box: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, bw, bh = (float(value) for value in box)
    x1 = (cx - bw / 2.0) * width
    y1 = (cy - bh / 2.0) * height
    x2 = (cx + bw / 2.0) * width
    y2 = (cy + bh / 2.0) * height
    return x1, y1, x2, y2


def clip_box(
    box: Iterable[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def box_area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_dimensions(box: Iterable[float]) -> tuple[float, float, float, float]:
    """Return non-negative width, height, short side and area for an xyxy box."""
    x1, y1, x2, y2 = (float(value) for value in box)
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width, height, min(width, height), width * height


def area_size_proxy(area: float) -> str:
    if area < SMALL_AREA_SIDE_PX**2:
        return "small"
    if area < MEDIUM_AREA_SIDE_PX**2:
        return "medium"
    return "large"


def is_short_side_small(short_side: float) -> bool:
    return short_side < SMALL_SHORT_SIDE_PX


def size_proxy(area: float, short_side: float) -> str:
    """Hybrid proxy: area buckets, with short-side<16 additionally marked small."""
    area_bucket = area_size_proxy(area)
    if area_bucket == "small" or is_short_side_small(short_side):
        return "small"
    return area_bucket


def small_reason(area: float, short_side: float) -> str:
    by_area = area < SMALL_AREA_SIDE_PX**2
    by_short_side = is_short_side_small(short_side)
    if by_area and by_short_side:
        return "area+short_side"
    if by_area:
        return "area"
    if by_short_side:
        return "short_side"
    return ""


def sgqs_preferred_scale(short_side: float) -> str:
    """Reproduce SGQS' log-nearest prior choice for reporting only."""
    if short_side <= 0.0:
        return "invalid"
    level = min(
        range(len(SGQS_SCALE_PRIORS_PX)),
        key=lambda index: (
            abs(math.log((short_side + 1e-6) / (SGQS_SCALE_PRIORS_PX[index] + 1e-6))),
            index,
        ),
    )
    return f"P{level + 3}"


def box_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = box_area((ax1, ay1, ax2, ay2)) + box_area(
        (bx1, by1, bx2, by2)
    ) - inter
    return inter / union if union > 0.0 else 0.0


def level_position(
    dense_index: int, spatial_shapes: list[tuple[int, int]], strides: list[int]
) -> tuple[int, int, int, int]:
    offset = 0
    for level, ((height, width), stride) in enumerate(zip(spatial_shapes, strides)):
        count = height * width
        if dense_index < offset + count:
            local = dense_index - offset
            return level, stride, local // width, local % width
        offset += count
    raise IndexError(f"Dense index {dense_index} is outside {offset} positions")


def percentile(values: list[float], q: float, np: Any) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def count_by_size(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(record["size_proxy"]) for record in records)
    return {name: int(counts.get(name, 0)) for name in SIZE_NAMES}


def count_by_field(
    records: list[dict[str, Any]], field: str, names: Iterable[str]
) -> dict[str, int]:
    counts = Counter(str(record[field]) for record in records)
    return {name: int(counts.get(name, 0)) for name in names}


def count_by_level(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(f"P{int(record['level']) + 3}" for record in records)
    return {name: int(counts.get(name, 0)) for name in ("P3", "P4", "P5")}


def draw_all_queries(
    image: Any,
    records: list[dict[str, Any]],
    output_path: Path,
    Image: Any,
    ImageDraw: Any,
) -> None:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    colors = {
        "small": (255, 40, 40),
        "medium": (255, 190, 30),
        "large": (30, 190, 255),
    }
    for record in sorted(records, key=lambda item: float(item["final_score"])):
        score = float(record["final_score"])
        rgb = colors[str(record["size_proxy"])]
        alpha = int(35 + 190 * max(0.0, min(1.0, score)))
        draw.rectangle(record["final_xyxy"], outline=(*rgb, alpha), width=1)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path)


def draw_small_queries(
    image: Any,
    records: list[dict[str, Any]],
    output_path: Path,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
) -> None:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    small = [record for record in records if record["size_proxy"] == "small"]
    for record in sorted(small, key=lambda item: float(item["final_score"])):
        score = float(record["final_score"])
        alpha = int(45 + 210 * max(0.0, min(1.0, score)))
        draw.rectangle(record["final_xyxy"], outline=(255, 25, 25, alpha), width=2)
    font = ImageFont.load_default()
    for record in sorted(
        small, key=lambda item: float(item["final_score"]), reverse=True
    )[:25]:
        x1, y1, _, _ = record["final_xyxy"]
        label = (
            f"q{record['query_index']} r{record['encoder_rank']} "
            f"{float(record['final_score']):.2f}"
        )
        draw.text((max(0, x1), max(0, y1 - 10)), label, fill=(255, 0, 0, 255), font=font)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path)


def draw_encoder_selection(
    image: Any,
    selected: list[dict[str, Any]],
    output_path: Path,
    Image: Any,
    ImageDraw: Any,
) -> None:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    level_colors = {0: (30, 220, 80, 210), 1: (30, 130, 255, 210), 2: (185, 60, 255, 210)}
    for record in selected:
        x1, y1, x2, y2 = record["encoder_xyxy"]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        color = level_colors[int(record["level"])]
        radius = 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(output_path)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    process_config = base.configure_process(args)

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    import torchvision.transforms as transforms

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.mkldnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    data_dir = args.data_dir.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    repo_dir = config_path.parents[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        (output_dir / group).mkdir(parents=True, exist_ok=True)

    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_metadata = base.validate_torch_checkpoint(checkpoint_path)
    records = base.find_images(data_dir, args.groups)
    transform = transforms.Compose(
        [
            transforms.Resize(
                (args.input_size, args.input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )

    wrapper, model_load_ms, model_metadata = base.load_deim_model(
        repo_dir, config_path, checkpoint_path, torch
    )
    detector = wrapper.detector
    decoder = detector.decoder
    if int(decoder.num_queries) != 640:
        raise RuntimeError(f"Expected 640 queries, got {decoder.num_queries}")
    if int(decoder.num_classes) != 1:
        raise RuntimeError(f"Expected one class, got {decoder.num_classes}")

    strides = [int(value) for value in decoder.feat_strides]
    spatial_shapes = [(args.input_size // stride, args.input_size // stride) for stride in strides]
    expected_dense_count = sum(height * width for height, width in spatial_shapes)

    capture: dict[str, Any] = {}
    original_select_topk = decoder._select_topk

    def instrumented_select_topk(
        memory: Any,
        outputs_logits: Any,
        outputs_anchors_unact: Any,
        topk: int,
    ) -> Any:
        if int(topk) != 640:
            raise RuntimeError(f"Unexpected architectural Top-K: {topk}")
        dense_logits = outputs_logits.max(-1).values
        topk_logits, topk_indices = torch.topk(dense_logits, topk, dim=-1)
        dense_boxes = torch.sigmoid(
            decoder.enc_bbox_head(memory) + outputs_anchors_unact
        )
        capture.clear()
        capture.update(
            {
                "dense_logits": dense_logits.detach().cpu(),
                "dense_class_logits": outputs_logits.detach().cpu(),
                "dense_boxes": dense_boxes.detach().cpu(),
                "anchors_unact": outputs_anchors_unact.detach().cpu(),
                "topk_logits": topk_logits.detach().cpu(),
                "topk_indices": topk_indices.detach().cpu(),
            }
        )
        return original_select_topk(
            memory, outputs_logits, outputs_anchors_unact, topk
        )

    decoder._select_topk = instrumented_select_topk

    encoder_csv_path = output_dir / "encoder_candidates_before_topk.csv"
    decoder_csv_path = output_dir / "decoder_queries_without_post_topk.csv"
    encoder_fields = [
        "image_key",
        "group",
        "image_name",
        "image_width",
        "image_height",
        "dense_index",
        "encoder_rank",
        "selected_top640",
        "query_index_if_selected",
        "level",
        "feature_name",
        "stride",
        "row",
        "column",
        "encoder_logit",
        "encoder_score",
        "encoder_cx",
        "encoder_cy",
        "encoder_w",
        "encoder_h",
        "encoder_x1",
        "encoder_y1",
        "encoder_x2",
        "encoder_y2",
        "display_clipped_area_px2",
        "input_clipped_width_px",
        "input_clipped_height_px",
        "input_short_side_px",
        "input_clipped_area_px2",
        "area_size_proxy",
        "short_side_small",
        "small_reason",
        "size_proxy",
        "sgqs_preferred_scale",
    ]
    decoder_fields = [
        "image_key",
        "group",
        "image_name",
        "image_width",
        "image_height",
        "query_index",
        "source_dense_index",
        "encoder_rank",
        "level",
        "feature_name",
        "stride",
        "row",
        "column",
        "encoder_logit",
        "encoder_score",
        "final_logit",
        "final_score",
        "final_cx",
        "final_cy",
        "final_w",
        "final_h",
        "final_x1",
        "final_y1",
        "final_x2",
        "final_y2",
        "display_clipped_area_px2",
        "input_clipped_width_px",
        "input_clipped_height_px",
        "input_short_side_px",
        "input_clipped_area_px2",
        "area_size_proxy",
        "short_side_small",
        "small_reason",
        "size_proxy",
        "sgqs_preferred_scale",
        "encoder_to_final_iou",
        "center_shift_px",
        "width_ratio_final_over_encoder",
        "height_ratio_final_over_encoder",
    ]

    summaries: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    with encoder_csv_path.open("w", encoding="utf-8-sig", newline="") as enc_stream, decoder_csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as dec_stream:
        enc_writer = csv.DictWriter(enc_stream, fieldnames=encoder_fields)
        dec_writer = csv.DictWriter(dec_stream, fieldnames=decoder_fields)
        enc_writer.writeheader()
        dec_writer.writeheader()

        try:
            with torch.inference_mode():
                for record_index, record in enumerate(records, start=1):
                    image = Image.open(record["path"]).convert("RGB")
                    width, height = image.size
                    image_key = f"{record['group']}/{record['path'].name}"
                    tensor = transform(image).unsqueeze(0)

                    capture.clear()
                    start_forward = time.perf_counter()
                    raw_output = detector(tensor)
                    forward_ms = (time.perf_counter() - start_forward) * 1000.0
                    if not capture:
                        raise RuntimeError("Top-K instrumentation did not capture encoder tensors")

                    dense_logits = capture["dense_logits"][0].numpy()
                    dense_scores = 1.0 / (1.0 + np.exp(-dense_logits))
                    dense_boxes = capture["dense_boxes"][0].numpy()
                    topk_indices = capture["topk_indices"][0].numpy().astype(np.int64)
                    if dense_logits.shape[0] != expected_dense_count:
                        raise RuntimeError(
                            f"Expected {expected_dense_count} dense positions, got {dense_logits.shape[0]}"
                        )
                    if topk_indices.shape[0] != 640:
                        raise RuntimeError(f"Expected 640 selected indices, got {topk_indices.shape[0]}")

                    order = np.argsort(-dense_logits, kind="stable")
                    ranks = np.empty_like(order)
                    ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
                    selected_to_query = {
                        int(dense_index): int(query_index)
                        for query_index, dense_index in enumerate(topk_indices.tolist())
                    }

                    dense_records: list[dict[str, Any]] = []
                    for dense_index in range(expected_dense_count):
                        level, stride, row, column = level_position(
                            dense_index, spatial_shapes, strides
                        )
                        normalized_box = dense_boxes[dense_index].tolist()
                        pixel_box = cxcywh_to_xyxy_pixels(normalized_box, width, height)
                        clipped = clip_box(pixel_box, width, height)
                        area = box_area(clipped)
                        input_box = cxcywh_to_xyxy_pixels(
                            normalized_box, args.input_size, args.input_size
                        )
                        input_clipped = clip_box(
                            input_box, args.input_size, args.input_size
                        )
                        (
                            input_width,
                            input_height,
                            input_short_side,
                            input_area,
                        ) = box_dimensions(input_clipped)
                        area_bucket = area_size_proxy(input_area)
                        entry = {
                            "image_key": image_key,
                            "group": record["group"],
                            "image_name": record["path"].name,
                            "image_width": width,
                            "image_height": height,
                            "dense_index": dense_index,
                            "encoder_rank": int(ranks[dense_index]),
                            "selected_top640": dense_index in selected_to_query,
                            "query_index_if_selected": selected_to_query.get(dense_index, ""),
                            "level": level,
                            "feature_name": f"P{level + 3}",
                            "stride": stride,
                            "row": row,
                            "column": column,
                            "encoder_logit": float(dense_logits[dense_index]),
                            "encoder_score": float(dense_scores[dense_index]),
                            "encoder_cx": float(normalized_box[0]),
                            "encoder_cy": float(normalized_box[1]),
                            "encoder_w": float(normalized_box[2]),
                            "encoder_h": float(normalized_box[3]),
                            "encoder_x1": float(clipped[0]),
                            "encoder_y1": float(clipped[1]),
                            "encoder_x2": float(clipped[2]),
                            "encoder_y2": float(clipped[3]),
                            "display_clipped_area_px2": float(area),
                            "input_clipped_width_px": float(input_width),
                            "input_clipped_height_px": float(input_height),
                            "input_short_side_px": float(input_short_side),
                            "input_clipped_area_px2": float(input_area),
                            "area_size_proxy": area_bucket,
                            "short_side_small": is_short_side_small(input_short_side),
                            "small_reason": small_reason(input_area, input_short_side),
                            "size_proxy": size_proxy(input_area, input_short_side),
                            "sgqs_preferred_scale": sgqs_preferred_scale(input_short_side),
                            "encoder_xyxy": clipped,
                        }
                        dense_records.append(entry)
                        enc_writer.writerow({key: entry[key] for key in encoder_fields})

                    final_logits = raw_output["pred_logits"][0, :, 0].detach().cpu().numpy()
                    final_scores = 1.0 / (1.0 + np.exp(-final_logits))
                    final_boxes = raw_output["pred_boxes"][0].detach().cpu().numpy()
                    if final_boxes.shape != (640, 4):
                        raise RuntimeError(f"Unexpected final box shape: {final_boxes.shape}")

                    query_records: list[dict[str, Any]] = []
                    for query_index, dense_index_value in enumerate(topk_indices.tolist()):
                        dense_index = int(dense_index_value)
                        source = dense_records[dense_index]
                        normalized_box = final_boxes[query_index].tolist()
                        pixel_box = cxcywh_to_xyxy_pixels(normalized_box, width, height)
                        clipped = clip_box(pixel_box, width, height)
                        area = box_area(clipped)
                        input_box = cxcywh_to_xyxy_pixels(
                            normalized_box, args.input_size, args.input_size
                        )
                        input_clipped = clip_box(
                            input_box, args.input_size, args.input_size
                        )
                        (
                            input_width,
                            input_height,
                            input_short_side,
                            input_area,
                        ) = box_dimensions(input_clipped)
                        area_bucket = area_size_proxy(input_area)
                        enc_box = source["encoder_xyxy"]
                        enc_cx = (enc_box[0] + enc_box[2]) / 2.0
                        enc_cy = (enc_box[1] + enc_box[3]) / 2.0
                        final_cx = (clipped[0] + clipped[2]) / 2.0
                        final_cy = (clipped[1] + clipped[3]) / 2.0
                        enc_width = max(0.0, enc_box[2] - enc_box[0])
                        enc_height = max(0.0, enc_box[3] - enc_box[1])
                        final_width = max(0.0, clipped[2] - clipped[0])
                        final_height = max(0.0, clipped[3] - clipped[1])
                        entry = {
                            "image_key": image_key,
                            "group": record["group"],
                            "image_name": record["path"].name,
                            "image_width": width,
                            "image_height": height,
                            "query_index": query_index,
                            "source_dense_index": dense_index,
                            "encoder_rank": int(ranks[dense_index]),
                            "level": source["level"],
                            "feature_name": source["feature_name"],
                            "stride": source["stride"],
                            "row": source["row"],
                            "column": source["column"],
                            "encoder_logit": source["encoder_logit"],
                            "encoder_score": source["encoder_score"],
                            "final_logit": float(final_logits[query_index]),
                            "final_score": float(final_scores[query_index]),
                            "final_cx": float(normalized_box[0]),
                            "final_cy": float(normalized_box[1]),
                            "final_w": float(normalized_box[2]),
                            "final_h": float(normalized_box[3]),
                            "final_x1": float(clipped[0]),
                            "final_y1": float(clipped[1]),
                            "final_x2": float(clipped[2]),
                            "final_y2": float(clipped[3]),
                            "display_clipped_area_px2": float(area),
                            "input_clipped_width_px": float(input_width),
                            "input_clipped_height_px": float(input_height),
                            "input_short_side_px": float(input_short_side),
                            "input_clipped_area_px2": float(input_area),
                            "area_size_proxy": area_bucket,
                            "short_side_small": is_short_side_small(input_short_side),
                            "small_reason": small_reason(input_area, input_short_side),
                            "size_proxy": size_proxy(input_area, input_short_side),
                            "sgqs_preferred_scale": sgqs_preferred_scale(input_short_side),
                            "encoder_to_final_iou": float(box_iou(enc_box, clipped)),
                            "center_shift_px": float(
                                math.hypot(final_cx - enc_cx, final_cy - enc_cy)
                            ),
                            "width_ratio_final_over_encoder": (
                                float(final_width / enc_width) if enc_width > 0.0 else None
                            ),
                            "height_ratio_final_over_encoder": (
                                float(final_height / enc_height) if enc_height > 0.0 else None
                            ),
                            "final_xyxy": clipped,
                        }
                        query_records.append(entry)
                        dec_writer.writerow({key: entry[key] for key in decoder_fields})

                    selected_dense = [dense_records[int(index)] for index in topk_indices]
                    rejected_small = sorted(
                        (
                            item
                            for item in dense_records
                            if not bool(item["selected_top640"])
                            and item["size_proxy"] == "small"
                        ),
                        key=lambda item: float(item["encoder_score"]),
                        reverse=True,
                    )
                    small_queries = [
                        item for item in query_records if item["size_proxy"] == "small"
                    ]

                    group_dir = output_dir / str(record["group"])
                    stem = record["path"].stem
                    draw_all_queries(
                        image,
                        query_records,
                        group_dir / f"{stem}_all_640_decoder_queries.png",
                        Image,
                        ImageDraw,
                    )
                    draw_small_queries(
                        image,
                        query_records,
                        group_dir / f"{stem}_small_decoder_queries.png",
                        Image,
                        ImageDraw,
                        ImageFont,
                    )
                    draw_encoder_selection(
                        image,
                        selected_dense,
                        group_dir / f"{stem}_encoder_selection.png",
                        Image,
                        ImageDraw,
                    )

                    small_scores = [float(item["final_score"]) for item in small_queries]
                    selected_ious = [
                        float(item["encoder_to_final_iou"]) for item in query_records
                    ]
                    top_small = sorted(
                        small_queries,
                        key=lambda item: float(item["final_score"]),
                        reverse=True,
                    )[:50]
                    summaries.append(
                        {
                            "image_key": image_key,
                            "source_path": str(record["path"].resolve()),
                            "width": width,
                            "height": height,
                            "forward_ms_diagnostic": forward_ms,
                            "dense_encoder_candidates": expected_dense_count,
                            "architectural_topk": 640,
                            "topk_cutoff": {
                                "encoder_logit_rank_640": float(
                                    dense_logits[topk_indices[-1]]
                                ),
                                "encoder_score_rank_640": float(
                                    dense_scores[topk_indices[-1]]
                                ),
                            },
                            "all_dense_encoder_size_proxy": count_by_size(dense_records),
                            "all_dense_encoder_area_size_proxy": count_by_field(
                                dense_records, "area_size_proxy", SIZE_NAMES
                            ),
                            "all_dense_encoder_short_side_small_count": sum(
                                bool(item["short_side_small"]) for item in dense_records
                            ),
                            "selected_encoder_size_proxy": count_by_size(selected_dense),
                            "selected_encoder_area_size_proxy": count_by_field(
                                selected_dense, "area_size_proxy", SIZE_NAMES
                            ),
                            "selected_encoder_short_side_small_count": sum(
                                bool(item["short_side_small"]) for item in selected_dense
                            ),
                            "selected_encoder_sgqs_preferred_scale": count_by_field(
                                selected_dense,
                                "sgqs_preferred_scale",
                                ("P3", "P4", "P5", "invalid"),
                            ),
                            "selected_encoder_level_counts": count_by_level(selected_dense),
                            "rejected_encoder_small_proxy_count": len(rejected_small),
                            "highest_rejected_small": (
                                {
                                    "dense_index": rejected_small[0]["dense_index"],
                                    "encoder_rank": rejected_small[0]["encoder_rank"],
                                    "encoder_score": rejected_small[0]["encoder_score"],
                                    "level": rejected_small[0]["feature_name"],
                                    "bbox_xyxy": list(rejected_small[0]["encoder_xyxy"]),
                                }
                                if rejected_small
                                else None
                            ),
                            "raw_decoder_queries_without_post_topk": 640,
                            "decoder_size_proxy": count_by_size(query_records),
                            "decoder_area_size_proxy": count_by_field(
                                query_records, "area_size_proxy", SIZE_NAMES
                            ),
                            "decoder_short_side_small_count": sum(
                                bool(item["short_side_small"]) for item in query_records
                            ),
                            "decoder_sgqs_preferred_scale": count_by_field(
                                query_records,
                                "sgqs_preferred_scale",
                                ("P3", "P4", "P5", "invalid"),
                            ),
                            "decoder_small_reasons": count_by_field(
                                small_queries,
                                "small_reason",
                                (
                                    "area",
                                    "short_side",
                                    "area+short_side",
                                ),
                            ),
                            "decoder_level_counts": count_by_level(query_records),
                            "small_decoder_query_score_counts": {
                                "all": len(small_queries),
                                "score_gt_0.01": sum(score > 0.01 for score in small_scores),
                                "score_gt_0.10": sum(score > 0.10 for score in small_scores),
                                "score_gt_0.30": sum(score > 0.30 for score in small_scores),
                                "score_gt_0.50": sum(score > 0.50 for score in small_scores),
                            },
                            "small_decoder_query_scores": {
                                "max": max(small_scores) if small_scores else None,
                                "p95": percentile(small_scores, 95, np),
                                "median": percentile(small_scores, 50, np),
                            },
                            "encoder_to_final_iou": {
                                "median": percentile(selected_ious, 50, np),
                                "p10": percentile(selected_ious, 10, np),
                                "p90": percentile(selected_ious, 90, np),
                            },
                            "top_50_small_decoder_queries": [
                                {
                                    "query_index": item["query_index"],
                                    "source_dense_index": item["source_dense_index"],
                                    "encoder_rank": item["encoder_rank"],
                                    "feature_name": item["feature_name"],
                                    "encoder_score": item["encoder_score"],
                                    "final_score": item["final_score"],
                                    "bbox_xyxy": list(item["final_xyxy"]),
                                    "display_area_px2": item["display_clipped_area_px2"],
                                    "input_area_px2": item["input_clipped_area_px2"],
                                    "input_short_side_px": item["input_short_side_px"],
                                    "small_reason": item["small_reason"],
                                    "sgqs_preferred_scale": item["sgqs_preferred_scale"],
                                    "encoder_to_final_iou": item[
                                        "encoder_to_final_iou"
                                    ],
                                }
                                for item in top_small
                            ],
                        }
                    )
                    print(
                        f"[{record_index:02d}/{len(records):02d}] {image_key}: "
                        f"dense={expected_dense_count}, selected=640, "
                        f"final-small={len(small_queries)}, "
                        f"small>0.5={sum(score > 0.5 for score in small_scores)}, "
                        f"forward={forward_ms:.1f} ms",
                        flush=True,
                    )
        finally:
            decoder._select_topk = original_select_topk

    elapsed_seconds = time.perf_counter() - start_all

    aggregate_decoder_size = Counter()
    aggregate_decoder_area_size = Counter()
    aggregate_decoder_short_side_small = 0
    aggregate_decoder_small_reasons = Counter()
    aggregate_decoder_sgqs_preferred_scale = Counter()
    aggregate_selected_encoder_size = Counter()
    aggregate_selected_encoder_area_size = Counter()
    aggregate_selected_encoder_short_side_small = 0
    aggregate_selected_encoder_sgqs_preferred_scale = Counter()
    aggregate_selected_levels = Counter()
    aggregate_small_score_counts = Counter()
    for item in summaries:
        aggregate_decoder_size.update(item["decoder_size_proxy"])
        aggregate_decoder_area_size.update(item["decoder_area_size_proxy"])
        aggregate_decoder_short_side_small += int(
            item["decoder_short_side_small_count"]
        )
        aggregate_decoder_small_reasons.update(item["decoder_small_reasons"])
        aggregate_decoder_sgqs_preferred_scale.update(item["decoder_sgqs_preferred_scale"])
        aggregate_selected_encoder_size.update(item["selected_encoder_size_proxy"])
        aggregate_selected_encoder_area_size.update(item["selected_encoder_area_size_proxy"])
        aggregate_selected_encoder_short_side_small += int(
            item["selected_encoder_short_side_small_count"]
        )
        aggregate_selected_encoder_sgqs_preferred_scale.update(
            item["selected_encoder_sgqs_preferred_scale"]
        )
        aggregate_selected_levels.update(item["selected_encoder_level_counts"])
        aggregate_small_score_counts.update(item["small_decoder_query_score_counts"])

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "Inspect small predicted query behavior and encoder selection. No GT annotations "
            "were present under DATA, so this is not a Hungarian/GT match audit."
        ),
        "semantics": {
            "repository_postprocessor_called": False,
            "postprocessor_topk": False,
            "score_threshold": None,
            "nms": False,
            "architectural_encoder_topk_preserved": 640,
            "all_dense_encoder_candidates_exported": True,
            "small_definition": (
                "hybrid predicted-size proxy in clipped 640x640 input space: "
                "area < 32^2 OR short side < 16 px; not a GT label"
            ),
            "size_proxy": {
                "name": "hybrid_input_area_or_short_side16_v1",
                "coordinate_space": "clipped model input",
                "input_width_px": args.input_size,
                "input_height_px": args.input_size,
                "comparison": "strict less-than",
                "small": "area < 32^2 OR short_side < 16",
                "medium": "not small AND area < 96^2",
                "large": "not small AND area >= 96^2",
                "area_thresholds_px2": [
                    SMALL_AREA_SIDE_PX**2,
                    MEDIUM_AREA_SIDE_PX**2,
                ],
                "short_side_small_threshold_px": SMALL_SHORT_SIDE_PX,
                "status": (
                    "diagnostic convention requested for this audit; the training "
                    "repository has no pre-existing binary area+short-side rule"
                ),
            },
            "sgqs_scale_reporting": {
                "coordinate_space": "clipped model input",
                "priors_px": list(SGQS_SCALE_PRIORS_PX),
                "assignment": "minimum absolute log-distance; lower level wins ties",
                "status": "reporting analogue of SGQS scale assignment, not a size label",
            },
        },
        "paths": {
            "data_dir": str(data_dir),
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
            "script": str(Path(__file__).resolve()),
        },
        "integrity": {
            "script_sha256": base.sha256_file(Path(__file__).resolve()),
            "config_sha256": base.sha256_file(config_path),
            "checkpoint": checkpoint_metadata,
        },
        "model": {
            **model_metadata,
            "input_size": args.input_size,
            "feature_strides": strides,
            "spatial_shapes": spatial_shapes,
            "dense_candidate_count": expected_dense_count,
            "model_load_ms": model_load_ms,
        },
        "runtime": {
            "device": "CPU",
            "dtype": "float32",
            "batch_size": 1,
            "torch_threads": 1,
            "mkldnn": False,
            "elapsed_seconds": elapsed_seconds,
            "process": process_config,
        },
        "aggregate": {
            "images": len(summaries),
            "dense_encoder_rows": len(summaries) * expected_dense_count,
            "raw_decoder_query_rows": len(summaries) * 640,
            "decoder_size_proxy": dict(aggregate_decoder_size),
            "decoder_area_size_proxy": dict(aggregate_decoder_area_size),
            "decoder_short_side_small_count": aggregate_decoder_short_side_small,
            "decoder_small_reasons": dict(aggregate_decoder_small_reasons),
            "decoder_sgqs_preferred_scale": dict(
                aggregate_decoder_sgqs_preferred_scale
            ),
            "selected_encoder_size_proxy": dict(aggregate_selected_encoder_size),
            "selected_encoder_area_size_proxy": dict(
                aggregate_selected_encoder_area_size
            ),
            "selected_encoder_short_side_small_count": (
                aggregate_selected_encoder_short_side_small
            ),
            "selected_encoder_sgqs_preferred_scale": dict(
                aggregate_selected_encoder_sgqs_preferred_scale
            ),
            "selected_encoder_level_counts": dict(aggregate_selected_levels),
            "small_decoder_query_score_counts": dict(aggregate_small_score_counts),
        },
        "images": summaries,
    }
    write_json(output_dir / "query_diagnostic_summary.json", result)

    markdown: list[str] = [
        "# DEIM 640 query-selection diagnostic (hybrid input-scale proxy)",
        "",
        "## What was actually run",
        "",
        f"- Checkpoint: `{args.checkpoint.resolve()}` (`ema.module`).",
        "- Input: every image in `wired` and `wireless`, resized directly to 640×640.",
        "- Repository PostProcessor was **not called**: no output Top-K, no score threshold, no NMS.",
        "- The trained architectural encoder Top-K=640 was preserved; all 8,400 pre-Top-K encoder candidates were captured and exported.",
        "- Size classification is computed after clipping in the **640×640 model-input space**, not the source-image display space.",
        "- Hybrid `small` means input area `<32²` **or** input short side `<16 px`; non-small boxes retain the area-based `medium` (`<96²`) / `large` split.",
        "- The 16 px short-side scale comes from the local EFGL configuration; the separate SGQS preferred-scale field still reproduces the log-nearest `[32,64,128]` assignment exactly.",
        "- This hybrid OR rule is an explicit diagnostic convention requested for this audit. The training code has no pre-existing binary area+short-side label.",
        "- Area-only and short-side-only buckets, the exact short side, the trigger reason, and the SGQS log-nearest `[32,64,128]` preferred scale are exported separately.",
        "- Every size label is a prediction-shape proxy, not a GT label.",
        f"- This no-GT pilot reads images from `{args.data_dir.resolve()}` and cannot measure Hungarian GT matching, small-object recall, or false negatives.",
        "",
        "## Aggregate",
        "",
        f"- Images: {len(summaries)}",
        f"- Dense encoder candidates exported: {len(summaries) * expected_dense_count:,}",
        f"- Raw decoder queries exported: {len(summaries) * 640:,}",
        f"- Decoder hybrid predicted-size proxy: `{dict(aggregate_decoder_size)}`",
        f"- Decoder area-only size proxy: `{dict(aggregate_decoder_area_size)}`",
        f"- Decoder short-side `<16 px` count: {aggregate_decoder_short_side_small:,}",
        f"- Decoder hybrid-small trigger reasons: `{dict(aggregate_decoder_small_reasons)}`",
        f"- Decoder SGQS preferred scale: `{dict(aggregate_decoder_sgqs_preferred_scale)}`",
        f"- Selected encoder hybrid size proxy: `{dict(aggregate_selected_encoder_size)}`",
        f"- Selected dense origins by feature level: `{dict(aggregate_selected_levels)}`",
        f"- Small-query score counts: `{dict(aggregate_small_score_counts)}`",
        "",
        "## Per image",
        "",
        "| Image | Encoder cutoff score at rank 640 | Selected P3/P4/P5 | Decoder small/all | Small score>0.5 | Rejected encoder-small proxy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        level = item["selected_encoder_level_counts"]
        small_counts = item["small_decoder_query_score_counts"]
        markdown.append(
            "| {image} | {cutoff:.6f} | {p3}/{p4}/{p5} | {small}/640 | {gt50} | {rejected} |".format(
                image=item["image_key"],
                cutoff=item["topk_cutoff"]["encoder_score_rank_640"],
                p3=level["P3"],
                p4=level["P4"],
                p5=level["P5"],
                small=small_counts["all"],
                gt50=small_counts["score_gt_0.50"],
                rejected=item["rejected_encoder_small_proxy_count"],
            )
        )
    markdown.extend(
        [
            "",
            "## Files",
            "",
            "- `encoder_candidates_before_topk.csv`: every one of the 8,400 encoder positions per image, including rank, selection flag, feature level, score, display-space box, 640-input area/short side, separate size buckets, hybrid trigger and SGQS preferred scale.",
            "- `decoder_queries_without_post_topk.csv`: all 640 decoder queries per image with the same geometry fields, mapped back to their source dense position; no threshold or NMS.",
            "- `query_diagnostic_summary.json`: machine-readable per-image statistics and the top 50 predicted-small queries.",
            "- `*_all_640_decoder_queries.png`: all raw decoder boxes (red=hybrid-small, yellow=hybrid-medium, cyan=hybrid-large).",
            "- `*_small_decoder_queries.png`: hybrid predicted-small decoder boxes; the top 25 are labeled with query index, encoder rank and final score.",
            "- `*_encoder_selection.png`: selected query origins as points only (green=P3, blue=P4, purple=P5); no rejected-candidate boxes are drawn.",
            "",
            "## Correct interpretation",
            "",
            "The postprocessor Top-640 was already a no-op for a one-class, 640-query model. The useful diagnostic boundary is therefore the encoder's 8,400→640 selection. This run exposes that boundary without changing decoder behavior. The hybrid label only describes predicted box geometry. A true small/short-side GT audit requires annotations and GT matching; P3/P4/P5 source levels and SGQS preferred scales are not GT size labels.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(markdown), encoding="utf-8")

    print(f"Artifacts written to: {output_dir}", flush=True)
    print(f"Summary: {output_dir / 'query_diagnostic_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
