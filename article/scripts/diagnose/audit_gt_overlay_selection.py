#!/usr/bin/env python3
"""Approximate GT-to-encoder-selection audit from a rendered GT overlay.

This script is intentionally read-only with respect to its inputs.  It recovers
rectangular cell annotations from a blue GT visualization, registers them to
the corresponding source image, and compares them with the dense encoder CSV
written by ``diagnose_deim_query_selection.py``.

The recovered boxes are *not* a substitute for the original annotation JSON.
They are useful for hypothesis formation and debugging; publication metrics
must be regenerated from authoritative annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


PROTOCOL_VERSION = "gt_overlay_encoder_selection_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover table-cell boxes from a blue GT overlay and audit the "
            "encoder dense-to-Top-K selection boundary."
        )
    )
    parser.add_argument(
        "--gt-overlay",
        type=Path,
        required=True,
        help="GT visualization containing blue rectangular box outlines.",
    )
    parser.add_argument(
        "--source-image",
        type=Path,
        required=True,
        help="Unannotated source image corresponding to the GT overlay.",
    )
    parser.add_argument(
        "--encoder-csv",
        type=Path,
        required=True,
        help="encoder_candidates_before_topk.csv from the query diagnostic.",
    )
    parser.add_argument(
        "--image-key",
        required=True,
        help="Exact image_key value to select from the encoder CSV.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help="Square model input size used for stride-space statistics (default: 640).",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=(8, 16, 32),
        help="Feature strides to report (default: 8 16 32).",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=(0.3, 0.5, 0.75),
        help="IoU thresholds for dense-versus-Top-K recall (default: 0.3 0.5 0.75).",
    )
    parser.add_argument(
        "--body-group-period",
        type=int,
        default=5,
        help=(
            "Number of body rows between full-width source table separators; "
            "used for piecewise vertical registration (default: 5)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON report path. Without this option, no file is written.",
    )
    parser.add_argument("--blue-hue-min", type=int, default=95)
    parser.add_argument("--blue-hue-max", type=int, default=115)
    parser.add_argument("--blue-saturation-min", type=int, default=40)
    parser.add_argument("--blue-value-min", type=int, default=50)
    parser.add_argument("--rectangle-solidity-min", type=float, default=0.90)
    parser.add_argument("--rectangle-width-ratio-min", type=float, default=0.05)
    parser.add_argument("--source-dark-max", type=int, default=80)
    parser.add_argument("--source-green-min", type=int, default=70)
    parser.add_argument("--source-green-margin", type=int, default=30)
    parser.add_argument("--source-green-row-fraction", type=float, default=0.40)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for label in ("gt_overlay", "source_image", "encoder_csv"):
        path = getattr(args, label)
        if not path.is_file():
            raise FileNotFoundError(f"{label.replace('_', ' ')} not found: {path}")
    if args.input_size <= 0:
        raise ValueError("--input-size must be positive")
    if not args.strides or any(value <= 0 for value in args.strides):
        raise ValueError("--strides must contain positive integers")
    if args.body_group_period <= 0:
        raise ValueError("--body-group-period must be positive")
    if not args.iou_thresholds or any(not 0.0 <= value <= 1.0 for value in args.iou_thresholds):
        raise ValueError("--iou-thresholds must lie in [0, 1]")


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def consecutive_centers(values: Iterable[int]) -> list[float]:
    ordered = sorted({int(value) for value in values})
    if not ordered:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous))
        start = previous = value
    runs.append((start, previous))
    return [(start + end) / 2.0 for start, end in runs]


def cluster_records_by_value(
    records: Sequence[dict[str, Any]], key: str, tolerance: int = 2
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for record in sorted(records, key=lambda item: int(item[key])):
        value = int(record[key])
        if not groups:
            groups.append([record])
            continue
        center = float(np.median([int(item[key]) for item in groups[-1]]))
        if abs(value - center) <= tolerance:
            groups[-1].append(record)
        else:
            groups.append([record])
    return groups


def longest_true_run(flags: Sequence[bool]) -> tuple[int, int]:
    best_start = best_end = -1
    current_start = -1
    for index, flag in enumerate(list(flags) + [False]):
        if flag and current_start < 0:
            current_start = index
        elif not flag and current_start >= 0:
            if best_start < 0 or index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            current_start = -1
    if best_start < 0:
        raise RuntimeError("Could not identify a repeated full-column body grid")
    return best_start, best_end


def extract_overlay_boxes(
    overlay: np.ndarray, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    height, width = overlay.shape[:2]
    hsv = cv2.cvtColor(overlay, cv2.COLOR_BGR2HSV)
    blue_mask = (
        (hsv[:, :, 0] >= args.blue_hue_min)
        & (hsv[:, :, 0] <= args.blue_hue_max)
        & (hsv[:, :, 1] >= args.blue_saturation_min)
        & (hsv[:, :, 2] >= args.blue_value_min)
    )
    contours, _ = cv2.findContours(
        blue_mask.astype(np.uint8) * 255,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    records: list[dict[str, Any]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        rectangle_area = float(box_width * box_height)
        solidity = float(cv2.contourArea(contour)) / max(rectangle_area, 1.0)
        if box_width < width * args.rectangle_width_ratio_min:
            continue
        if box_height < height * 0.008:
            continue
        if box_width >= width * 0.95 or box_height >= height * 0.50:
            continue
        if solidity < args.rectangle_solidity_min:
            continue
        records.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(box_width),
                "height": int(box_height),
                "x2": int(x + box_width - 1),
                "y2": int(y + box_height - 1),
                "solidity": solidity,
            }
        )

    if not records:
        raise RuntimeError("No rectangular blue GT contours were recovered")
    records.sort(key=lambda item: (item["y"], item["x"]))

    row_groups = cluster_records_by_value(records, "y")
    maximum_columns = max(len(group) for group in row_groups)
    run_start, run_end = longest_true_run(
        [len(group) == maximum_columns for group in row_groups]
    )
    body_record_ids = {
        id(record)
        for group in row_groups[run_start:run_end]
        for record in group
    }
    for record in records:
        record["is_body"] = id(record) in body_record_ids

    body_rows = row_groups[run_start:run_end]
    body_row_count = len(body_rows)
    if body_row_count < 2:
        raise RuntimeError("Recovered body grid is too short to audit")

    outer_x1 = min(item["x"] for item in records)
    outer_y1 = min(item["y"] for item in records)
    outer_x2 = max(item["x2"] for item in records)
    outer_y2 = max(item["y2"] for item in records)

    row_scores = blue_mask[:, outer_x1 : outer_x2 + 1].sum(axis=1)
    column_scores = blue_mask[outer_y1 : outer_y2 + 1, :].sum(axis=0)
    horizontal_lines = consecutive_centers(
        np.flatnonzero(row_scores >= 0.65 * (outer_x2 - outer_x1 + 1))
    )
    vertical_lines = consecutive_centers(
        np.flatnonzero(column_scores >= 0.65 * (outer_y2 - outer_y1 + 1))
    )

    first_body_y = min(item["y"] for item in records if item["is_body"])
    last_body_y = max(item["y2"] for item in records if item["is_body"])
    body_line_start = min(horizontal_lines, key=lambda value: abs(value - first_body_y))
    body_line_end = min(horizontal_lines, key=lambda value: abs(value - last_body_y))
    body_horizontal_lines = [
        value
        for value in horizontal_lines
        if body_line_start <= value <= body_line_end
    ]
    if len(body_horizontal_lines) != body_row_count + 1:
        raise RuntimeError(
            "Body grid boundary mismatch: "
            f"{body_row_count} rows but {len(body_horizontal_lines)} horizontal boundaries"
        )

    metadata = {
        "image_width": width,
        "image_height": height,
        "maximum_columns": maximum_columns,
        "body_rows": body_row_count,
        "outer_bounds": [outer_x1, outer_y1, outer_x2, outer_y2],
        "vertical_lines": vertical_lines,
        "horizontal_lines": horizontal_lines,
        "body_horizontal_lines": body_horizontal_lines,
    }
    return records, blue_mask, metadata


def largest_dark_component(
    source: np.ndarray, dark_max: int
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    dark = gray < dark_max
    contours, _ = cv2.findContours(
        dark.astype(np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise RuntimeError("Could not find a dark connected table component in source image")
    contour = max(contours, key=cv2.contourArea)
    x, y, width, height = cv2.boundingRect(contour)
    if width * height < source.shape[0] * source.shape[1] * 0.10:
        raise RuntimeError("Largest dark source component is too small to be the table")
    return dark, (int(x), int(y), int(width), int(height))


def build_registration(
    overlay_records: Sequence[dict[str, Any]],
    overlay_metadata: dict[str, Any],
    source: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dark, (source_x, source_y, source_width, source_height) = largest_dark_component(
        source, args.source_dark_max
    )
    source_x2 = source_x + source_width - 1
    source_y2 = source_y + source_height - 1

    column_scores = dark[source_y : source_y2 + 1, :].sum(axis=0)
    source_vertical_lines = consecutive_centers(
        np.flatnonzero(column_scores >= 0.75 * float(column_scores.max()))
    )
    overlay_vertical_lines = list(overlay_metadata["vertical_lines"])
    if len(source_vertical_lines) != len(overlay_vertical_lines):
        raise RuntimeError(
            "Vertical grid registration failed: "
            f"overlay has {len(overlay_vertical_lines)} lines, "
            f"source has {len(source_vertical_lines)}"
        )

    blue, green, red = cv2.split(source)
    blue_i = blue.astype(np.int16)
    green_i = green.astype(np.int16)
    red_i = red.astype(np.int16)
    green_mask = (
        (green_i >= args.source_green_min)
        & (green_i >= blue_i + args.source_green_margin)
        & (green_i >= red_i + args.source_green_margin)
    )
    green_row_fraction = green_mask[:, source_x : source_x2 + 1].mean(axis=1)
    header_rows = np.flatnonzero(
        green_row_fraction >= args.source_green_row_fraction
    )
    if not len(header_rows):
        raise RuntimeError("Could not identify the green table header in source image")
    source_body_top = float(header_rows[-1] + 1)

    row_scores = dark[:, source_x : source_x2 + 1].sum(axis=1)
    separator_rows = np.flatnonzero(
        (np.arange(source.shape[0]) > source_body_top + 2)
        & (row_scores >= 0.80 * source_width)
    )
    source_body_separators = consecutive_centers(separator_rows)

    body_row_count = int(overlay_metadata["body_rows"])
    period = args.body_group_period
    overlay_body_lines = list(overlay_metadata["body_horizontal_lines"])
    overlay_anchor_indices = list(range(0, body_row_count, period))
    if overlay_anchor_indices[-1] != body_row_count:
        overlay_anchor_indices.append(body_row_count)
    expected_source_separator_count = len(overlay_anchor_indices) - 1
    if len(source_body_separators) != expected_source_separator_count:
        raise RuntimeError(
            "Horizontal registration failed: expected "
            f"{expected_source_separator_count} body separators for "
            f"{body_row_count} rows at period {period}, found "
            f"{len(source_body_separators)}"
        )

    overlay_body_anchors = np.array(
        [overlay_body_lines[index] for index in overlay_anchor_indices], dtype=float
    )
    source_body_anchors = np.array(
        [source_body_top, *source_body_separators], dtype=float
    )

    overlay_outer_top = float(overlay_metadata["horizontal_lines"][0])
    source_outer_top = float(source_y) + 0.5
    overlay_y_anchors = np.array(
        [overlay_outer_top, *overlay_body_anchors], dtype=float
    )
    source_y_anchors = np.array(
        [source_outer_top, *source_body_anchors], dtype=float
    )
    overlay_x_anchors = np.array(overlay_vertical_lines, dtype=float)
    source_x_anchors = np.array(source_vertical_lines, dtype=float)

    mapped_boxes: list[list[float]] = []
    body_mask: list[bool] = []
    for item in overlay_records:
        mapped_boxes.append(
            [
                float(np.interp(item["x"], overlay_x_anchors, source_x_anchors)),
                float(np.interp(item["y"], overlay_y_anchors, source_y_anchors)),
                float(np.interp(item["x2"], overlay_x_anchors, source_x_anchors)),
                float(np.interp(item["y2"], overlay_y_anchors, source_y_anchors)),
            ]
        )
        body_mask.append(bool(item["is_body"]))

    registration = {
        "method": "piecewise_table_grid_registration",
        "source_table_bounds": [source_x, source_y, source_x2, source_y2],
        "overlay_x_anchors": overlay_x_anchors.tolist(),
        "source_x_anchors": source_x_anchors.tolist(),
        "overlay_y_anchors": overlay_y_anchors.tolist(),
        "source_y_anchors": source_y_anchors.tolist(),
    }
    return np.asarray(mapped_boxes, dtype=float), np.asarray(body_mask), registration


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_encoder_rows(path: Path, image_key: str) -> dict[str, np.ndarray]:
    records: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        required = {
            "image_key",
            "image_width",
            "image_height",
            "encoder_rank",
            "selected_top640",
            "feature_name",
            "encoder_score",
            "encoder_x1",
            "encoder_y1",
            "encoder_x2",
            "encoder_y2",
        }
        missing = sorted(required.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        for record in reader:
            if record["image_key"] == image_key:
                records.append(record)
    if not records:
        raise ValueError(f"No rows for image_key={image_key!r} in {path}")

    widths = {int(record["image_width"]) for record in records}
    heights = {int(record["image_height"]) for record in records}
    if len(widths) != 1 or len(heights) != 1:
        raise ValueError("CSV rows disagree on image dimensions")

    return {
        "boxes": np.asarray(
            [
                [
                    float(record["encoder_x1"]),
                    float(record["encoder_y1"]),
                    float(record["encoder_x2"]),
                    float(record["encoder_y2"]),
                ]
                for record in records
            ],
            dtype=float,
        ),
        "ranks": np.asarray([int(record["encoder_rank"]) for record in records]),
        "selected": np.asarray(
            [parse_bool(record["selected_top640"]) for record in records], dtype=bool
        ),
        "levels": np.asarray([record["feature_name"] for record in records]),
        "scores": np.asarray(
            [float(record["encoder_score"]) for record in records], dtype=float
        ),
        "image_width": np.asarray([next(iter(widths))]),
        "image_height": np.asarray([next(iter(heights))]),
    }


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    quantiles = (0, 10, 25, 50, 75, 90, 100)
    results = np.percentile(values, quantiles)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "median", "p75", "p90", "max"), results
        )
    }


def pairwise_iou(gt_boxes: np.ndarray, candidate_boxes: np.ndarray) -> np.ndarray:
    gt = gt_boxes[:, None, :]
    candidate = candidate_boxes[None, :, :]
    intersection_width = np.maximum(
        0.0, np.minimum(gt[:, :, 2], candidate[:, :, 2]) - np.maximum(gt[:, :, 0], candidate[:, :, 0])
    )
    intersection_height = np.maximum(
        0.0, np.minimum(gt[:, :, 3], candidate[:, :, 3]) - np.maximum(gt[:, :, 1], candidate[:, :, 1])
    )
    intersection = intersection_width * intersection_height
    gt_area = (gt[:, :, 2] - gt[:, :, 0]) * (gt[:, :, 3] - gt[:, :, 1])
    candidate_area = (candidate[:, :, 2] - candidate[:, :, 0]) * (
        candidate[:, :, 3] - candidate[:, :, 1]
    )
    return intersection / np.maximum(gt_area + candidate_area - intersection, 1e-12)


def level_sort_key(level: str) -> tuple[int, str]:
    digits = "".join(character for character in level if character.isdigit())
    return (int(digits) if digits else math.inf, level)


def center_coverage(
    inside: np.ndarray,
    selected: np.ndarray,
    levels: np.ndarray,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    level_names: list[str | None] = [None, *sorted(set(levels.tolist()), key=level_sort_key)]
    for level in level_names:
        mask = selected if level is None else selected & (levels == level)
        counts_per_gt = inside[:, mask].sum(axis=1)
        covered = counts_per_gt > 0
        centers_inside = inside[:, mask].any(axis=0)
        key = "all" if level is None else level
        histogram = Counter(int(value) for value in counts_per_gt.tolist())
        report[key] = {
            "selected_candidates": int(mask.sum()),
            "selected_centers_inside_any_body_gt": int(centers_inside.sum()),
            "unique_body_gt_covered": int(covered.sum()),
            "unique_body_gt_coverage": float(covered.mean()),
            "body_gt_without_center": int((~covered).sum()),
            "mean_centers_per_covered_gt": (
                float(counts_per_gt.sum() / covered.sum()) if covered.any() else 0.0
            ),
            "max_centers_per_gt": int(counts_per_gt.max(initial=0)),
            "centers_per_gt_histogram": {
                str(key_value): int(count)
                for key_value, count in sorted(histogram.items())
            },
        }
    return report


def build_report(
    args: argparse.Namespace,
    overlay: np.ndarray,
    source: np.ndarray,
    overlay_records: Sequence[dict[str, Any]],
    overlay_metadata: dict[str, Any],
    gt_boxes: np.ndarray,
    body_mask: np.ndarray,
    registration: dict[str, Any],
    encoder: dict[str, np.ndarray],
) -> dict[str, Any]:
    source_height, source_width = source.shape[:2]
    csv_width = int(encoder["image_width"][0])
    csv_height = int(encoder["image_height"][0])
    if (source_width, source_height) != (csv_width, csv_height):
        raise ValueError(
            "Source and CSV dimensions differ: "
            f"source={source_width}x{source_height}, CSV={csv_width}x{csv_height}"
        )

    body_boxes = gt_boxes[body_mask]
    source_dimensions = body_boxes[:, 2:4] - body_boxes[:, 0:2]
    input_scale = np.asarray(
        [args.input_size / source_width, args.input_size / source_height], dtype=float
    )
    input_dimensions = source_dimensions * input_scale
    short_side = input_dimensions.min(axis=1)
    input_area = input_dimensions.prod(axis=1)

    strides: dict[str, Any] = {}
    for stride in args.strides:
        occupancy = input_dimensions / float(stride)
        strides[str(stride)] = {
            "width_cells": percentile_dict(occupancy[:, 0]),
            "height_cells": percentile_dict(occupancy[:, 1]),
            "height_lt_1_cell": int((occupancy[:, 1] < 1.0).sum()),
            "height_lt_1_cell_ratio": float((occupancy[:, 1] < 1.0).mean()),
            "height_lt_2_cells": int((occupancy[:, 1] < 2.0).sum()),
            "height_lt_2_cells_ratio": float((occupancy[:, 1] < 2.0).mean()),
        }

    candidate_boxes = encoder["boxes"]
    ranks = encoder["ranks"]
    selected = encoder["selected"]
    levels = encoder["levels"]
    scores = encoder["scores"]
    selected_count = int(selected.sum())
    if selected_count == 0:
        raise RuntimeError("Encoder CSV contains no selected candidates")

    centers = np.column_stack(
        (
            (candidate_boxes[:, 0] + candidate_boxes[:, 2]) / 2.0,
            (candidate_boxes[:, 1] + candidate_boxes[:, 3]) / 2.0,
        )
    )
    gt = body_boxes[:, None, :]
    inside = (
        (centers[None, :, 0] >= gt[:, :, 0])
        & (centers[None, :, 0] <= gt[:, :, 2])
        & (centers[None, :, 1] >= gt[:, :, 1])
        & (centers[None, :, 1] <= gt[:, :, 3])
    )

    iou = pairwise_iou(body_boxes, candidate_boxes)
    iou_recall: dict[str, Any] = {}
    level_iou_recall: dict[str, Any] = {}
    level_names = sorted(set(levels.tolist()), key=level_sort_key)
    for threshold in args.iou_thresholds:
        dense_covered = (iou >= threshold).any(axis=1)
        selected_covered = (iou[:, selected] >= threshold).any(axis=1)
        starved = dense_covered & ~selected_covered
        key = f"{threshold:g}"
        iou_recall[key] = {
            "dense_covered": int(dense_covered.sum()),
            "dense_recall": float(dense_covered.mean()),
            "topk_covered": int(selected_covered.sum()),
            "topk_recall": float(selected_covered.mean()),
            "selection_starved": int(starved.sum()),
            "selection_starved_ratio": float(starved.mean()),
        }
        level_iou_recall[key] = {}
        for level in level_names:
            level_dense_mask = levels == level
            level_selected_mask = selected & level_dense_mask
            dense_level = (iou[:, level_dense_mask] >= threshold).any(axis=1)
            selected_level = (iou[:, level_selected_mask] >= threshold).any(axis=1)
            level_iou_recall[key][level] = {
                "dense_covered": int(dense_level.sum()),
                "dense_recall": float(dense_level.mean()),
                "topk_covered": int(selected_level.sum()),
                "topk_recall": float(selected_level.mean()),
                "selection_starved": int((dense_level & ~selected_level).sum()),
            }

    best_dense_indices = np.argmax(iou, axis=1)
    best_dense_iou = iou[np.arange(len(body_boxes)), best_dense_indices]
    selected_indices = np.flatnonzero(selected)
    best_selected_iou = iou[:, selected].max(axis=1)
    best_dense_ranks = ranks[best_dense_indices]

    dense_by_level = Counter(levels.tolist())
    selected_by_level = Counter(levels[selected].tolist())
    report: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "caveat": (
            "Boxes were recovered from a rendered GT overlay and registered to the "
            "source table grid. This is an approximate diagnostic, not authoritative "
            "annotation-based evaluation."
        ),
        "inputs": {
            "gt_overlay": str(args.gt_overlay.resolve()),
            "source_image": str(args.source_image.resolve()),
            "encoder_csv": str(args.encoder_csv.resolve()),
            "image_key": args.image_key,
        },
        "overlay_recovery": {
            "overlay_width": int(overlay.shape[1]),
            "overlay_height": int(overlay.shape[0]),
            "total_boxes": len(overlay_records),
            "body_boxes": int(body_mask.sum()),
            "header_boxes": int((~body_mask).sum()),
            "body_rows": int(overlay_metadata["body_rows"]),
            "columns": int(overlay_metadata["maximum_columns"]),
        },
        "registration": registration,
        "geometry_at_model_input": {
            "input_size": args.input_size,
            "body_width_px": percentile_dict(input_dimensions[:, 0]),
            "body_height_px": percentile_dict(input_dimensions[:, 1]),
            "body_short_side_px": percentile_dict(short_side),
            "body_area_px2": percentile_dict(input_area),
            "coco_small_area_lt_1024": int((input_area < 32.0**2).sum()),
            "coco_small_area_lt_1024_ratio": float((input_area < 32.0**2).mean()),
            "stride_occupancy": strides,
        },
        "encoder_selection": {
            "dense_candidates": int(len(candidate_boxes)),
            "selected_topk": selected_count,
            "dense_by_level": {
                key: int(value) for key, value in sorted(dense_by_level.items(), key=lambda item: level_sort_key(item[0]))
            },
            "selected_by_level": {
                key: int(value) for key, value in sorted(selected_by_level.items(), key=lambda item: level_sort_key(item[0]))
            },
            "cutoff_score_min_selected": float(scores[selected].min()),
            "center_unique_coverage": center_coverage(inside, selected, levels),
            "iou_recall": iou_recall,
            "level_iou_recall": level_iou_recall,
            "best_match": {
                "dense_best_iou": percentile_dict(best_dense_iou),
                "topk_best_iou": percentile_dict(best_selected_iou),
                "dense_best_candidate_rank": percentile_dict(best_dense_ranks.astype(float)),
                "dense_best_rank_gt_topk": int((best_dense_ranks > selected_count).sum()),
                "dense_best_rank_gt_topk_ratio": float((best_dense_ranks > selected_count).mean()),
                "selected_candidate_indices": int(len(selected_indices)),
            },
        },
    }
    return report


def format_percentile_span(stats: dict[str, float]) -> str:
    return (
        f"{stats['min']:.2f}-{stats['max']:.2f} "
        f"(median {stats['median']:.2f})"
    )


def print_report(report: dict[str, Any]) -> None:
    recovery = report["overlay_recovery"]
    geometry = report["geometry_at_model_input"]
    selection = report["encoder_selection"]
    coverage = selection["center_unique_coverage"]
    best = selection["best_match"]

    print("GT overlay recovery")
    print(
        f"  boxes: {recovery['total_boxes']} total = "
        f"{recovery['body_boxes']} body + {recovery['header_boxes']} header "
        f"({recovery['body_rows']}x{recovery['columns']} body grid)"
    )
    print(
        f"  body @ {geometry['input_size']}: width "
        f"{format_percentile_span(geometry['body_width_px'])} px; short side "
        f"{format_percentile_span(geometry['body_short_side_px'])} px"
    )
    print(
        f"  COCO area<32^2: {geometry['coco_small_area_lt_1024']}/"
        f"{recovery['body_boxes']} "
        f"({geometry['coco_small_area_lt_1024_ratio']:.1%})"
    )
    for stride, stats in geometry["stride_occupancy"].items():
        print(
            f"  stride {stride}: median WxH="
            f"{stats['width_cells']['median']:.2f}x{stats['height_cells']['median']:.2f} cells; "
            f"H range {stats['height_cells']['min']:.2f}-{stats['height_cells']['max']:.2f}; "
            f"H<1 {stats['height_lt_1_cell_ratio']:.1%}, "
            f"H<2 {stats['height_lt_2_cells_ratio']:.1%}"
        )

    selected_levels = ", ".join(
        f"{level}={count}" for level, count in selection["selected_by_level"].items()
    )
    print("Encoder selection")
    print(
        f"  dense -> Top-K: {selection['dense_candidates']} -> "
        f"{selection['selected_topk']}; {selected_levels}; "
        f"cutoff={selection['cutoff_score_min_selected']:.9f}"
    )
    for level in ("all", *selection["selected_by_level"].keys()):
        stats = coverage[level]
        print(
            f"  center coverage {level}: {stats['unique_body_gt_covered']}/"
            f"{recovery['body_boxes']} ({stats['unique_body_gt_coverage']:.1%}); "
            f"{stats['selected_centers_inside_any_body_gt']} centers inside; "
            f"max duplicates/GT={stats['max_centers_per_gt']}"
        )

    print("Dense vs Top-K encoder-box recall against recovered body GT")
    for threshold, stats in selection["iou_recall"].items():
        print(
            f"  IoU>={threshold}: dense {stats['dense_covered']}/"
            f"{recovery['body_boxes']} ({stats['dense_recall']:.1%}); "
            f"Top-K {stats['topk_covered']}/{recovery['body_boxes']} "
            f"({stats['topk_recall']:.1%}); starved={stats['selection_starved']}"
        )
    print(
        f"  best IoU median: dense={best['dense_best_iou']['median']:.3f}, "
        f"Top-K={best['topk_best_iou']['median']:.3f}"
    )
    print(
        f"  dense best-candidate rank: median "
        f"{best['dense_best_candidate_rank']['median']:.0f}; "
        f">Top-K {best['dense_best_rank_gt_topk']}/{recovery['body_boxes']} "
        f"({best['dense_best_rank_gt_topk_ratio']:.1%})"
    )
    print("CAVEAT: " + report["caveat"])


def main() -> int:
    args = parse_args()
    validate_args(args)
    overlay = read_image(args.gt_overlay)
    source = read_image(args.source_image)
    overlay_records, _, overlay_metadata = extract_overlay_boxes(overlay, args)
    gt_boxes, body_mask, registration = build_registration(
        overlay_records, overlay_metadata, source, args
    )
    encoder = load_encoder_rows(args.encoder_csv, args.image_key)
    report = build_report(
        args,
        overlay,
        source,
        overlay_records,
        overlay_metadata,
        gt_boxes,
        body_mask,
        registration,
        encoder,
    )
    print_report(report)
    if args.output_json is not None:
        args.output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"JSON report: {args.output_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
