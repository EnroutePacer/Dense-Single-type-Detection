#!/usr/bin/env python3
"""Fixed-CPU DEIM tiled and multi-scale inference experiments.

This driver deliberately reuses the model loading, checkpoint validation,
visualization, and statistics helpers from ``benchmark_deim_cpu.py``.  It adds
three source-image inference variants:

* tiled: overlapping crops, each resized to 640x640;
* multiscale: the full image resized to 512x512, 640x640, and 800x800;
* tiled_multiscale: full-image multi-scale predictions plus every non-trivial
  tile at all three input sizes.

All predictions are mapped to source-image coordinates and merged with
deterministic class-wise cross-source NMS.  Latency is reported per source image, including
the sum of every model call required by that variant.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import benchmark_deim_cpu as base


MODE_OUTPUT_NAMES = {
    "tiled": "DEIM_tiled",
    "multiscale": "DEIM_multiscale",
    "tiled_multiscale": "DEIM_tiled_multiscale",
}
METRIC_FIELDS = (
    "preprocess_ms",
    "inference_ms",
    "output_materialize_postprocess_ms",
    "runtime_with_output_ms",
    "pipeline_no_io_ms",
)
PROTOCOL_VERSION = "deim_tta_cpu_b1_t1_aff2_fp32_mkldnn0_dynamic_v1"


@dataclass(frozen=True)
class ViewSpec:
    view_type: str
    input_size: int
    tile_index: int | None
    crop_xyxy: tuple[int, int, int, int]
    ownership_xyxy: tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    root = base.default_baseline_root()
    parser = argparse.ArgumentParser(
        description="Benchmark DEIM tiled and/or multi-scale CPU inference."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=tuple(MODE_OUTPUT_NAMES),
    )
    parser.add_argument("--data-dir", type=Path, default=root / "DATA")
    parser.add_argument(
        "--model-registry",
        type=Path,
        default=root / "models" / "which_weight_to_use" / "path.txt",
    )
    parser.add_argument("--model-name", default="DEIM")
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", default=["wired", "wireless"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--scales", type=int, nargs="+", default=[512, 640, 800])
    parser.add_argument("--tile-input-size", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap-x", type=float, default=0.40)
    parser.add_argument("--tile-overlap-y", type=float, default=0.20)
    parser.add_argument("--merge-iou", type=float, default=0.80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--affinity", default="2")
    parser.add_argument("--seed", type=int, default=20260929)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.threads != 1:
        raise ValueError("This comparison protocol is fixed to --threads 1")
    if base.parse_affinity(args.affinity) != [2]:
        raise ValueError("This comparison protocol is fixed to --affinity 2")
    if args.warmup < 1 or args.repeats < 2:
        raise ValueError("Use at least one warm-up and two measured repeats")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if not 0.0 < args.tile_overlap_x < 0.5:
        raise ValueError("--tile-overlap-x must be in (0, 0.5)")
    if not 0.0 < args.tile_overlap_y < 0.5:
        raise ValueError("--tile-overlap-y must be in (0, 0.5)")
    if not 0.0 < args.merge_iou < 1.0:
        raise ValueError("--merge-iou must be in (0, 1)")
    sizes = set(args.scales + [args.tile_input_size])
    if any(size < 320 or size % 32 for size in sizes):
        raise ValueError("Every model input size must be >=320 and divisible by 32")
    if args.tile_size < 128:
        raise ValueError("--tile-size must be at least 128 source pixels")


def axis_windows(
    length: int,
    tile_size: int,
    overlap_pixels: int,
) -> list[tuple[int, int, float, float]]:
    """Return crop and center-ownership intervals for one image axis.

    A dimension no more than one overlap band larger than the requested tile
    is kept whole.  This avoids almost-identical duplicate tiles such as
    640px and 646px-high crops.
    """
    if length <= tile_size + overlap_pixels:
        return [(0, length, 0.0, float(length))]

    stride = tile_size - overlap_pixels
    count = int(math.ceil((length - tile_size) / stride)) + 1
    max_start = length - tile_size
    starts = [int(round(index * max_start / (count - 1))) for index in range(count)]
    starts = sorted(set(starts))
    crops = [(start, min(length, start + tile_size)) for start in starts]
    centers = [(start + end) / 2.0 for start, end in crops]

    windows: list[tuple[int, int, float, float]] = []
    for index, (start, end) in enumerate(crops):
        own_start = 0.0 if index == 0 else (centers[index - 1] + centers[index]) / 2.0
        own_end = (
            float(length)
            if index == len(crops) - 1
            else (centers[index] + centers[index + 1]) / 2.0
        )
        windows.append((start, end, own_start, own_end))
    return windows


def tile_views(
    width: int,
    height: int,
    tile_size: int,
    overlap_fraction_x: float,
    overlap_fraction_y: float,
    input_sizes: list[int],
) -> list[ViewSpec]:
    overlap_pixels_x = int(round(tile_size * overlap_fraction_x))
    overlap_pixels_y = int(round(tile_size * overlap_fraction_y))
    x_windows = axis_windows(width, tile_size, overlap_pixels_x)
    y_windows = axis_windows(height, tile_size, overlap_pixels_y)
    views: list[ViewSpec] = []
    tile_index = 0
    for y0, y1, own_y0, own_y1 in y_windows:
        for x0, x1, own_x0, own_x1 in x_windows:
            for input_size in input_sizes:
                views.append(
                    ViewSpec(
                        view_type="tile",
                        input_size=input_size,
                        tile_index=tile_index,
                        crop_xyxy=(x0, y0, x1, y1),
                        ownership_xyxy=(own_x0, own_y0, own_x1, own_y1),
                    )
                )
            tile_index += 1
    return views


def full_views(width: int, height: int, input_sizes: list[int]) -> list[ViewSpec]:
    return [
        ViewSpec(
            view_type="full",
            input_size=input_size,
            tile_index=None,
            crop_xyxy=(0, 0, width, height),
            ownership_xyxy=(0.0, 0.0, float(width), float(height)),
        )
        for input_size in input_sizes
    ]


def build_views(record: dict[str, Any], args: argparse.Namespace) -> list[ViewSpec]:
    width = int(record["width"])
    height = int(record["height"])
    if args.mode == "multiscale":
        return full_views(width, height, list(args.scales))
    if args.mode == "tiled":
        return tile_views(
            width,
            height,
            args.tile_size,
            args.tile_overlap_x,
            args.tile_overlap_y,
            [args.tile_input_size],
        )

    full = full_views(width, height, list(args.scales))
    tiled = tile_views(
        width,
        height,
        args.tile_size,
        args.tile_overlap_x,
        args.tile_overlap_y,
        list(args.scales),
    )
    unique_tiles = {view.crop_xyxy for view in tiled}
    if unique_tiles == {(0, 0, width, height)}:
        return full
    return full + tiled


def make_resizers(scales: list[int], transforms: Any) -> dict[int, Any]:
    return {
        scale: transforms.Compose(
            [
                transforms.Resize(
                    (scale, scale),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
            ]
        )
        for scale in scales
    }


def map_detections(
    detections: list[dict[str, Any]],
    view: ViewSpec,
    image_width: int,
    image_height: int,
) -> list[dict[str, Any]]:
    x0, y0, _, _ = view.crop_xyxy
    own_x0, own_y0, own_x1, own_y1 = view.ownership_xyxy
    mapped: list[dict[str, Any]] = []
    for detection in detections:
        bx1, by1, bx2, by2 = detection["bbox_xyxy"]
        gx1 = max(0.0, min(float(image_width), bx1 + x0))
        gy1 = max(0.0, min(float(image_height), by1 + y0))
        gx2 = max(0.0, min(float(image_width), bx2 + x0))
        gy2 = max(0.0, min(float(image_height), by2 + y0))
        if gx2 <= gx1 or gy2 <= gy1:
            continue
        center_x = (gx1 + gx2) / 2.0
        center_y = (gy1 + gy2) / 2.0
        if view.view_type == "tile":
            owns_x = own_x0 <= center_x and (
                center_x < own_x1
                or (own_x1 == float(image_width) and center_x <= own_x1)
            )
            owns_y = own_y0 <= center_y and (
                center_y < own_y1
                or (own_y1 == float(image_height) and center_y <= own_y1)
            )
            if not (owns_x and owns_y):
                continue
        mapped.append(
            {
                "class_id": detection["class_id"],
                "class_name": detection["class_name"],
                "score": detection["score"],
                "bbox_xyxy": [gx1, gy1, gx2, gy2],
                "source_view": {
                    "type": view.view_type,
                    "input_size": view.input_size,
                    "tile_index": view.tile_index,
                    "crop_xyxy": list(view.crop_xyxy),
                },
            }
        )
    return mapped


def nms_detections(
    detections: list[dict[str, Any]],
    iou_threshold: float,
    np: Any,
) -> list[dict[str, Any]]:
    if not detections:
        return []
    kept: list[dict[str, Any]] = []
    class_ids = sorted({int(item["class_id"]) for item in detections})
    for class_id in class_ids:
        items = [item for item in detections if int(item["class_id"]) == class_id]
        scores = np.asarray([item["score"] for item in items], dtype=np.float64)
        stable_index = np.arange(len(items), dtype=np.int64)
        order = np.lexsort((stable_index, -scores))
        class_kept: list[dict[str, Any]] = []
        for current in order.tolist():
            candidate = items[int(current)]
            candidate_source = candidate["source_view"]
            candidate_source_id = (
                candidate_source["type"],
                candidate_source["input_size"],
                candidate_source["tile_index"],
                tuple(candidate_source["crop_xyxy"]),
            )
            should_suppress = False
            ax1, ay1, ax2, ay2 = candidate["bbox_xyxy"]
            area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
            for previous in class_kept:
                previous_source = previous["source_view"]
                previous_source_id = (
                    previous_source["type"],
                    previous_source["input_size"],
                    previous_source["tile_index"],
                    tuple(previous_source["crop_xyxy"]),
                )
                if previous_source_id == candidate_source_id:
                    continue
                bx1, by1, bx2, by2 = previous["bbox_xyxy"]
                intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                    0.0, min(ay2, by2) - max(ay1, by1)
                )
                area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
                union = area_a + area_b - intersection
                iou = intersection / union if union > 0.0 else 0.0
                if iou >= iou_threshold:
                    should_suppress = True
                    break
            if not should_suppress:
                class_kept.append(candidate)
        kept.extend(class_kept)
    kept.sort(
        key=lambda item: (
            -float(item["score"]),
            int(item["class_id"]),
            *(round(float(value), 6) for value in item["bbox_xyxy"]),
        )
    )
    return kept


def detections_signature(detections: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        detections,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_source_image(
    record: dict[str, Any],
    views: list[ViewSpec],
    wrapper: Any,
    resizers: dict[int, Any],
    threshold: float,
    merge_iou: float,
    torch: Any,
    np: Any,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    candidates: list[dict[str, Any]] = []
    preprocess_ns = 0
    inference_ns = 0
    materialize_ns = 0
    raw_query_count = 0
    start_total = time.perf_counter_ns()

    for view in views:
        start = time.perf_counter_ns()
        crop = record["image"].crop(view.crop_xyxy)
        crop_width, crop_height = crop.size
        image_tensor = resizers[view.input_size](crop).unsqueeze(0)
        original_size = torch.tensor([[crop_width, crop_height]], dtype=torch.int64)
        after_preprocess = time.perf_counter_ns()
        output = wrapper(image_tensor, original_size)
        after_inference = time.perf_counter_ns()
        view_detections, _ = base.materialize_detections(
            output,
            crop_width,
            crop_height,
            threshold,
            np,
        )
        candidates.extend(
            map_detections(
                view_detections,
                view,
                int(record["width"]),
                int(record["height"]),
            )
        )
        after_materialize = time.perf_counter_ns()
        preprocess_ns += after_preprocess - start
        inference_ns += after_inference - after_preprocess
        materialize_ns += after_materialize - after_inference
        raw_query_count += int(output[2].shape[1])

    merge_start = time.perf_counter_ns()
    merged = nms_detections(candidates, merge_iou, np)
    merge_end = time.perf_counter_ns()
    materialize_ns += merge_end - merge_start
    total_ns = merge_end - start_total
    metrics: dict[str, float | int] = {
        "model_call_count": len(views),
        "raw_query_count": raw_query_count,
        "candidate_count_before_merge": len(candidates),
        "detection_count_after_merge": len(merged),
        "suppressed_by_merge": len(candidates) - len(merged),
        "preprocess_ms": preprocess_ns / 1_000_000.0,
        "inference_ms": inference_ns / 1_000_000.0,
        "output_materialize_postprocess_ms": materialize_ns / 1_000_000.0,
        "runtime_with_output_ms": (inference_ns + materialize_ns) / 1_000_000.0,
        "pipeline_no_io_ms": total_ns / 1_000_000.0,
    }
    return merged, metrics


def summarize_rows(rows: list[dict[str, Any]], np: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "samples": len(rows),
        "latency": {},
    }
    for field in METRIC_FIELDS:
        result["latency"][field] = base.summarize_values(
            [float(row[field]) for row in rows], np
        )
    for field in (
        "model_call_count",
        "raw_query_count",
        "candidate_count_before_merge",
        "detection_count_after_merge",
        "suppressed_by_merge",
    ):
        values = [float(row[field]) for row in rows]
        result[field] = {
            "mean": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
    return result


def serializable_views(views: list[ViewSpec]) -> list[dict[str, Any]]:
    return [
        {
            "view_type": view.view_type,
            "input_shape_nchw": [1, 3, view.input_size, view.input_size],
            "tile_index": view.tile_index,
            "crop_xyxy": list(view.crop_xyxy),
            "ownership_xyxy": list(view.ownership_xyxy),
        }
        for view in views
    ]


def main() -> int:
    args = parse_args()
    validate_args(args)
    process_config = base.configure_process(args)

    import numpy as np
    import PIL
    from PIL import Image, ImageDraw, ImageFont
    import psutil
    import torch
    import torchvision
    import torchvision.transforms as transforms

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.backends.mkldnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_grad_enabled(False)

    root = base.default_baseline_root()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (root / "output" / MODE_OUTPUT_NAMES[args.mode]).resolve()
    )
    data_dir = args.data_dir.resolve()
    registry_path = args.model_registry.resolve()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"DEIM config does not exist: {config_path}")
    checkpoint_path = base.resolve_checkpoint(registry_path, args.model_name)
    checkpoint_metadata = base.validate_torch_checkpoint(checkpoint_path)
    repo_dir = config_path.parents[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    for group in args.groups:
        (output_dir / group).mkdir(parents=True, exist_ok=True)

    records = base.find_images(data_dir, args.groups)
    base.load_images(records, Image)
    image_counts = {
        group: sum(record["group"] == group for record in records)
        for group in args.groups
    }
    views_by_key = {record["key"]: build_views(record, args) for record in records}
    input_sizes = sorted(
        {view.input_size for views in views_by_key.values() for view in views}
    )
    resizers = make_resizers(input_sizes, transforms)

    print(
        f"Mode={args.mode}; images={len(records)}; input_sizes={input_sizes}; "
        f"calls/image={[len(views_by_key[r['key']]) for r in records]}",
        flush=True,
    )
    wrapper, model_load_ms, model_metadata = base.load_deim_model(
        repo_dir, config_path, checkpoint_path, torch
    )
    wrapper.detector.encoder.eval_spatial_size = None
    wrapper.detector.decoder.eval_spatial_size = None
    model_metadata["dynamic_eval_spatial_size"] = True
    model_metadata["dynamic_components"] = ["HybridEncoder", "DFINETransformer"]

    median_record = sorted(
        records, key=lambda item: item["width"] * item["height"]
    )[len(records) // 2]
    with torch.inference_mode():
        for input_size in input_sizes:
            warm_view = full_views(
                int(median_record["width"]),
                int(median_record["height"]),
                [input_size],
            )[0]
            for _ in range(args.warmup):
                run_source_image(
                    median_record,
                    [warm_view],
                    wrapper,
                    resizers,
                    args.threshold,
                    args.merge_iou,
                    torch,
                    np,
                )
    print(
        f"Warm-up complete: {args.warmup} runs per input size ({input_sizes})",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    detections_by_key: dict[str, list[dict[str, Any]]] = {}
    first_metrics_by_key: dict[str, dict[str, float | int]] = {}
    signatures_by_key: dict[str, str] = {}
    non_deterministic_keys: set[str] = set()
    sequence_index = 0
    rng = random.Random(args.seed)

    gc.collect()
    gc.disable()
    benchmark_start = datetime.now().astimezone()
    try:
        with torch.inference_mode():
            for repeat_index in range(1, args.repeats + 1):
                order = list(records)
                rng.shuffle(order)
                repeat_pipeline: list[float] = []
                for record in order:
                    sequence_index += 1
                    detections, metrics = run_source_image(
                        record,
                        views_by_key[record["key"]],
                        wrapper,
                        resizers,
                        args.threshold,
                        args.merge_iou,
                        torch,
                        np,
                    )
                    signature = detections_signature(detections)
                    if record["key"] not in signatures_by_key:
                        signatures_by_key[record["key"]] = signature
                        detections_by_key[record["key"]] = detections
                        first_metrics_by_key[record["key"]] = metrics
                    elif signatures_by_key[record["key"]] != signature:
                        non_deterministic_keys.add(record["key"])

                    row = {
                        "sequence_index": sequence_index,
                        "repeat": repeat_index,
                        "group": record["group"],
                        "image_key": record["key"],
                        "image_name": record["path"].name,
                        "width": record["width"],
                        "height": record["height"],
                        **metrics,
                    }
                    rows.append(row)
                    repeat_pipeline.append(float(metrics["pipeline_no_io_ms"]))
                print(
                    f"Measured repeat {repeat_index:02d}/{args.repeats}: "
                    f"pipeline mean={statistics.fmean(repeat_pipeline):.3f} ms/source image",
                    flush=True,
                )
    finally:
        gc.enable()
    benchmark_end = datetime.now().astimezone()

    if non_deterministic_keys:
        print(
            "WARNING: final detections changed across repeats for: "
            + ", ".join(sorted(non_deterministic_keys)),
            flush=True,
        )

    csv_path = output_dir / "latency_raw.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    grouped_rows = {
        group: [row for row in rows if row["group"] == group]
        for group in args.groups
    }
    summary: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "mode": args.mode,
        "primary_metric": "pipeline_no_io_ms per source image across all required views",
        "overall": summarize_rows(rows, np),
        "groups": {
            group: summarize_rows(group_rows, np)
            for group, group_rows in grouped_rows.items()
        },
        "per_image": {
            record["key"]: summarize_rows(
                [row for row in rows if row["image_key"] == record["key"]], np
            )
            for record in records
        },
        "detection_count_at_threshold_after_merge": {
            record["key"]: len(detections_by_key[record["key"]])
            for record in records
        },
        "output_deterministic_across_repeats": not non_deterministic_keys,
        "non_deterministic_images": sorted(non_deterministic_keys),
    }
    base.write_json(output_dir / "latency_summary.json", summary)

    detection_document = {
        "model": "DEIM / D-FINE-M table-cell detector",
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_state": model_metadata["state_source"],
        "threshold_rule_before_merge": f"score > {args.threshold}",
        "merge": {
            "method": "deterministic class-wise cross-source greedy NMS",
            "scope": "different tile/scale sources only",
            "iou_threshold": args.merge_iou,
        },
        "images": [],
    }
    for record in records:
        detections = detections_by_key[record["key"]]
        visual = base.draw_visualization(
            record["image"], detections, ImageDraw, ImageFont
        )
        visual_path = output_dir / record["group"] / f"{record['path'].stem}_vis.png"
        visual.save(visual_path, format="PNG")
        first_metrics = first_metrics_by_key[record["key"]]
        detection_document["images"].append(
            {
                "group": record["group"],
                "source_path": str(record["path"].resolve()),
                "visualization_path": str(visual_path.resolve()),
                "width": record["width"],
                "height": record["height"],
                "decode_ms_not_in_benchmark": record["decode_ms"],
                "model_call_count": first_metrics["model_call_count"],
                "candidate_count_before_merge": first_metrics[
                    "candidate_count_before_merge"
                ],
                "detection_count": len(detections),
                "suppressed_by_merge": first_metrics["suppressed_by_merge"],
                "views": serializable_views(views_by_key[record["key"]]),
                "detections": detections,
            }
        )
    base.write_json(output_dir / "detections.json", detection_document)

    config_document = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "benchmark_started_at": benchmark_start.isoformat(),
        "benchmark_finished_at": benchmark_end.isoformat(),
        "model_name": args.model_name,
        "architecture": "DEIM with HGNetv2-B2 / D-FINE transformer",
        "inference_variant": {
            "mode": args.mode,
            "full_image_scales": list(args.scales)
            if args.mode in {"multiscale", "tiled_multiscale"}
            else [],
            "tile_input_scales": (
                list(args.scales)
                if args.mode == "tiled_multiscale"
                else ([args.tile_input_size] if args.mode == "tiled" else [])
            ),
            "tile_size_source_pixels": args.tile_size,
            "tile_overlap_fraction_xy": [
                args.tile_overlap_x,
                args.tile_overlap_y,
            ],
            "tile_overlap_pixels_xy": [
                int(round(args.tile_size * args.tile_overlap_x)),
                int(round(args.tile_size * args.tile_overlap_y)),
            ],
            "tile_positioning": "evenly spaced edge-anchored windows per axis",
            "tile_snap_rule": "keep an axis whole when length <= tile_size + overlap_pixels",
            "tile_ownership": "keep a tile detection only when its center lies in that tile's midpoint-partitioned ownership region",
            "combined_mode": "full-image multi-scale plus non-trivial tiled multi-scale",
            "merge_method": "deterministic class-wise cross-source greedy NMS",
            "merge_iou_threshold": args.merge_iou,
            "score_threshold_before_merge": args.threshold,
        },
        "paths": {
            "data_dir": str(data_dir),
            "model_registry": str(registry_path),
            "checkpoint_from_registry": str(checkpoint_path),
            "config": str(config_path),
            "repo_dir": str(repo_dir),
            "output_dir": str(output_dir),
            "script": str(Path(__file__).resolve()),
            "script_sha256": base.sha256_file(Path(__file__).resolve()),
        },
        "registry": {
            "format_source": str((registry_path.parent / "README.md").resolve()),
            "model_entry": args.model_name,
            "registry_sha256": base.sha256_file(registry_path),
        },
        "checkpoint": checkpoint_metadata,
        "config_sha256": base.sha256_file(config_path),
        "model": model_metadata,
        "data": {
            "groups": list(args.groups),
            "image_counts": image_counts,
            "total_images": len(records),
            "input_sizes": input_sizes,
            "model_calls_per_source_image": {
                record["key"]: len(views_by_key[record["key"]])
                for record in records
            },
            "total_model_calls_per_repeat": sum(
                len(views_by_key[record["key"]]) for record in records
            ),
            "input_images": [
                {
                    "key": record["key"],
                    "width": record["width"],
                    "height": record["height"],
                    "size_bytes": record["path"].stat().st_size,
                    "sha256": base.sha256_file(record["path"]),
                    "views": serializable_views(views_by_key[record["key"]]),
                }
                for record in records
            ],
        },
        "preprocessing": [
            "Pillow decode and RGB conversion once per source image (excluded from benchmark)",
            "crop the recorded source-pixel tile, or use the full source image",
            "torchvision square Resize to the recorded input size; BILINEAR, antialias=true, keep_ratio=false",
            "torchvision ToTensor: CHW float32 in [0,1]",
            "add batch dimension",
        ],
        "postprocessing": {
            "per_view_implementation": "repository PostProcessor.deploy()",
            "labels": ["cell"],
            "score_threshold": args.threshold,
            "comparison": "strict greater-than before ownership filtering and NMS",
            "boxes_clipped_to_source_image": True,
            "extra_nms": True,
            "nms_scope": "different tile/scale sources only; same-source DETR outputs are preserved",
            "nms_iou_threshold": args.merge_iou,
        },
        "benchmark": {
            "warmup_runs_per_input_size": args.warmup,
            "warmup_input_sizes": input_sizes,
            "measured_repeats_per_source_image": args.repeats,
            "measured_source_image_samples": len(rows),
            "total_model_forwards": sum(int(row["model_call_count"]) for row in rows),
            "shuffle_seed": args.seed,
            "ordering": "deterministic seeded shuffle of source images per repeat; deterministic view order within an image",
            "timer": "time.perf_counter_ns",
            "primary_metric": "pipeline_no_io_ms per source image across all model calls",
            "inference_metric": "sum of wrapper forward times for every required view of one source image",
            "excluded_from_latency": [
                "checkpoint/model/config loading and deploy conversion",
                "source image disk read/decode",
                "visualization drawing",
                "output file encoding/writes",
            ],
            "model_load_ms_reported_separately": model_load_ms,
        },
        "runtime": {
            "device": "CPU",
            "batch_size_per_model_call": 1,
            "dtype": "float32",
            "execution": "PyTorch eager inference_mode with dynamic eval spatial size",
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "torch_mkldnn_enabled": torch.backends.mkldnn.enabled,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_available": torch.cuda.is_available(),
        },
        "process": process_config,
        "host": {
            "cpu": base.get_cpu_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cpus": psutil.cpu_count(logical=True),
            "power_plan_observed": base.get_power_plan(),
            "operating_system": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "psutil": psutil.__version__,
        },
        "comparability_note": (
            "CPU 2, one thread, FP32, batch 1 per model call, PyTorch eager, "
            "MKLDNN off, score threshold, warm-up policy, repeat count, and seed "
            "are fixed. Variant latency is per source image and includes every view; "
            "do not compare it to a single-forward latency without considering the "
            "recorded model_call_count."
        ),
    }
    base.write_json(output_dir / "benchmark_config.json", config_document)

    inference_stats = summary["overall"]["latency"]["inference_ms"]
    pipeline_stats = summary["overall"]["latency"]["pipeline_no_io_ms"]
    log_lines = [
        "DEIM fixed-CPU inference variant benchmark",
        f"Protocol: {PROTOCOL_VERSION}",
        f"Mode: {args.mode}",
        f"Registered checkpoint: {checkpoint_path}",
        f"Checkpoint state: {model_metadata['state_source']}",
        f"Images: {len(records)} ({image_counts})",
        f"Input sizes: {input_sizes}",
        f"Model calls/repeat: {config_document['data']['total_model_calls_per_repeat']}",
        f"Warm-up: {args.warmup}/input size; repeats/source image: {args.repeats}",
        f"Threshold: score > {args.threshold}; NMS IoU: {args.merge_iou}",
        f"CPU affinity: {process_config['affinity_applied']}; threads: {args.threads}",
        "Backend: PyTorch eager FP32; dynamic eval size; MKLDNN: OFF; batch size: 1/call",
        (
            "Summed inference/source image: "
            f"mean={inference_stats['mean_ms']:.3f} ms, "
            f"P50={inference_stats['median_p50_ms']:.3f} ms, "
            f"P95={inference_stats['p95_ms']:.3f} ms"
        ),
        (
            "Pipeline/source image: "
            f"mean={pipeline_stats['mean_ms']:.3f} ms, "
            f"P50={pipeline_stats['median_p50_ms']:.3f} ms, "
            f"P90={pipeline_stats['p90_ms']:.3f} ms, "
            f"P95={pipeline_stats['p95_ms']:.3f} ms, "
            f"std={pipeline_stats['std_ms']:.3f} ms"
        ),
    ]
    for group in args.groups:
        group_stats = summary["groups"][group]["latency"]["pipeline_no_io_ms"]
        log_lines.append(
            f"{group} pipeline/source image: mean={group_stats['mean_ms']:.3f} ms, "
            f"P50={group_stats['median_p50_ms']:.3f} ms, "
            f"P95={group_stats['p95_ms']:.3f} ms"
        )
    with (output_dir / "run.log").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(log_lines) + "\n")

    print("\n".join(log_lines), flush=True)
    print(f"Artifacts written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
