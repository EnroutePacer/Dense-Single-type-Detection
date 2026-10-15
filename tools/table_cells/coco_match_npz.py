"""Scalable, resumable COCO one-to-one match export for full validation.

The full table-cell validation set is too large for one JSON dictionary per
decoder query.  This tool consumes a compact NPZ written by the replay pass,
runs the repository's exact ``faster_coco_eval`` C++ matcher in image shards,
and writes:

* gzip-compressed, per-annotation CSV shards containing TP/FN status and the
  matched decoder query/score/rank/IoU;
* compact per-shard detection status arrays from which the exact global
  score ordering, PR curve, and TP/FP score separation can be reconstructed;
* per-shard aggregate counters for safe resume;
* a small merged CSV with GT TP/FN and detection TP/FP/ignored/truncated
  counts by small-definition, density, and size buckets.

The NPZ contract is intentionally simple and contains no Python objects:

``image_ids [I]``, ``boxes_xywh [I,Q,4]`` in original-image pixels,
``scores [I,Q]``, scalar ``category_id``, and optional ``input_size [2]``.
The decoder query index is the Q-axis index.  Both IoU thresholds and both
area ranges are evaluated together in C++, so no model inference is repeated.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.table_cells.coco_match_export import (
    _density_bucket,
    _size_fields,
)


SCOPES: tuple[tuple[str, tuple[float, float]], ...] = (
    ("all", (0.0, 1.0e10)),
    ("coco_small_eval", (0.0, 32.0**2)),
)
STATUS_ORDER = ("tp", "fp", "fn", "ignored", "truncated")
AggregateKey = tuple[str, str, float, int, str, str, str]


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _threshold_label(threshold: float) -> str:
    return f"iou{int(round(threshold * 100)):02d}"


def _normalise_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in result):
        raise ValueError("IoU thresholds must be finite values in (0, 1]")
    return result


def _normalise_budgets(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(values)))
    if not result or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in result):
        raise ValueError("maxDets budgets must be positive integers")
    return result


def _annotation_index(
    dataset: Mapping[str, Any], category_id: int
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], dict[int, int]]:
    images: dict[int, dict[str, Any]] = {}
    for image in dataset.get("images", []):
        image_id = image.get("id")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in images:
            raise ValueError(f"invalid or duplicate image id: {image_id!r}")
        if float(image.get("width", 0)) <= 0 or float(image.get("height", 0)) <= 0:
            raise ValueError(f"image {image_id} has invalid dimensions")
        images[image_id] = image

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    image_density = {image_id: 0 for image_id in images}
    annotation_ids: set[int] = set()
    for ann in dataset.get("annotations", []):
        annotation_id = ann.get("id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool) or annotation_id in annotation_ids:
            raise ValueError(f"invalid or duplicate annotation id: {annotation_id!r}")
        annotation_ids.add(annotation_id)
        if ann.get("image_id") not in images:
            raise ValueError(f"annotation {annotation_id} names an unknown image")
        if not ann.get("iscrowd", 0):
            image_density[ann["image_id"]] += 1
        if ann.get("category_id") != category_id:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"annotation {annotation_id} has no xywh bbox")
        bbox = [float(value) for value in bbox]
        area = float(ann.get("area", bbox[2] * bbox[3]))
        if (
            any(not math.isfinite(value) for value in bbox)
            or bbox[2] < 0.0
            or bbox[3] < 0.0
            or not math.isfinite(area)
            or area < 0.0
        ):
            raise ValueError(f"annotation {annotation_id} has invalid geometry")
        annotations_by_image[ann["image_id"]].append(ann)
    return images, annotations_by_image, image_density


def _load_npz_contract(
    path: Path,
) -> tuple[np.lib.npyio.NpzFile, np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int], str]:
    archive = np.load(path, allow_pickle=False)
    required = {"image_ids", "boxes_xywh", "scores", "category_id"}
    missing = required - set(archive.files)
    if missing:
        archive.close()
        raise ValueError(f"prediction NPZ is missing keys: {sorted(missing)}")

    image_ids = np.asarray(archive["image_ids"])
    boxes = np.asarray(archive["boxes_xywh"])
    scores = np.asarray(archive["scores"])
    category_value = np.asarray(archive["category_id"])
    if image_ids.ndim != 1 or not np.issubdtype(image_ids.dtype, np.integer):
        archive.close()
        raise ValueError("image_ids must be a one-dimensional integer array")
    if boxes.ndim != 3 or boxes.shape[0] != len(image_ids) or boxes.shape[2] != 4:
        archive.close()
        raise ValueError("boxes_xywh must have shape [images, queries, 4]")
    if scores.shape != boxes.shape[:2]:
        archive.close()
        raise ValueError("scores must have shape [images, queries]")
    if not np.issubdtype(boxes.dtype, np.number) or not np.issubdtype(scores.dtype, np.number):
        archive.close()
        raise ValueError("boxes_xywh and scores must be numeric arrays")
    if category_value.size != 1:
        archive.close()
        raise ValueError("category_id must be a scalar")
    category_id = int(category_value.reshape(-1)[0])
    if "query_indices" in archive.files:
        query_indices = np.asarray(archive["query_indices"])
        expected = np.arange(boxes.shape[1], dtype=query_indices.dtype)
        if query_indices.ndim != 1 or not np.array_equal(query_indices, expected):
            archive.close()
            raise ValueError("query_indices must be the one-dimensional sequence 0..Q-1")
    input_size = (640, 640)
    if "input_size" in archive.files:
        values = np.asarray(archive["input_size"]).reshape(-1)
        if len(values) != 2:
            archive.close()
            raise ValueError("input_size must contain width and height")
        input_size = (int(values[0]), int(values[1]))
    if any(value <= 0 for value in input_size):
        archive.close()
        raise ValueError("input_size values must be positive")
    state = path.stem
    if "state" in archive.files:
        state_value = np.asarray(archive["state"])
        if state_value.size != 1:
            archive.close()
            raise ValueError("state must be a scalar string")
        state = str(state_value.reshape(-1)[0])
    if len(set(int(value) for value in image_ids)) != len(image_ids):
        archive.close()
        raise ValueError("image_ids contains duplicates")
    return archive, image_ids.astype(np.int64, copy=False), boxes, scores, category_id, input_size, state


def _build_shard_evaluation(
    dataset: Mapping[str, Any],
    images: Mapping[int, Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    shard_image_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    category_id: int,
    thresholds: Sequence[float],
    max_det: int,
) -> list[tuple]:
    """Run the C++ COCO matcher and return its per-image serializable states."""

    image_set = {int(value) for value in shard_image_ids}
    shard_dataset = {
        "info": dataset.get("info", {}),
        "images": [images[image_id] for image_id in sorted(image_set)],
        "annotations": [
            ann
            for image_id in sorted(image_set)
            for ann in annotations_by_image.get(image_id, ())
        ],
        "categories": dataset["categories"],
    }
    # COCOeval orders images by sorted unique id.  Enforce that same order in
    # the array contract so the C++ ImageEvaluation states have an unambiguous
    # mapping back to NPZ rows.
    if list(map(int, shard_image_ids)) != sorted(map(int, shard_image_ids)):
        raise ValueError("image_ids must be globally sorted ascending")

    coco_gt = COCO()
    coco_gt.dataset = shard_dataset
    coco_gt.createIndex()
    query_count = boxes.shape[1]
    predictions: list[dict[str, Any]] = []
    append = predictions.append
    for local_image_index, image_id_value in enumerate(shard_image_ids):
        image_id = int(image_id_value)
        for query_index in range(query_count):
            append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(value) for value in boxes[local_image_index, query_index]],
                    "score": float(scores[local_image_index, query_index]),
                    "query_index": query_index,
                }
            )
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval_faster(
        coco_gt,
        coco_dt,
        iouType="bbox",
        print_function=lambda *args: None,
        separate_eval=True,
    )
    evaluator.params.imgIds = list(map(int, shard_image_ids))
    evaluator.params.catIds = [category_id]
    evaluator.params.iouThrs = np.asarray(thresholds, dtype=np.float64)
    evaluator.params.areaRng = [list(area_range) for _, area_range in SCOPES]
    evaluator.params.areaRngLbl = [name for name, _ in SCOPES]
    evaluator.params.maxDets = [max_det]
    evaluator.evaluate()
    states = [entry.__getstate__() for entry in evaluator._evalImgs_cpp]
    expected = len(SCOPES) * len(shard_image_ids)
    if len(states) != expected:
        raise RuntimeError(f"COCO evaluator returned {len(states)} image states, expected {expected}")
    return states


def _prediction_buckets(
    boxes: np.ndarray, image: Mapping[str, Any], input_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    area = boxes[:, 2] * boxes[:, 3]
    input_width = boxes[:, 2] * input_size[0] / float(image["width"])
    input_height = boxes[:, 3] * input_size[1] / float(image["height"])
    input_area = input_width * input_height
    input_short = np.minimum(input_width, input_height)

    def area_codes(values: np.ndarray) -> np.ndarray:
        return np.where(values < 32.0**2, 0, np.where(values < 96.0**2, 1, 2))

    short_codes = np.select(
        [input_short < 8.0, input_short < 16.0, input_short < 32.0, input_short < 64.0],
        [0, 1, 2, 3],
        default=4,
    )
    coco_small = area < 32.0**2
    hybrid_small = (input_area < 32.0**2) | (input_short < 16.0)
    # neither=0, COCO-only=1, hybrid-only=2, intersection=3
    small_partition = coco_small.astype(np.uint8) + 2 * hybrid_small.astype(np.uint8)
    return {
        "prediction_coco_area": area_codes(area),
        "prediction_input_area": area_codes(input_area),
        "prediction_input_short": short_codes,
        "prediction_small_partition": small_partition,
    }


def _add_count(
    counts: Counter[AggregateKey],
    unit: str,
    scope: str,
    threshold: float,
    budget: int,
    axis: str,
    group: str,
    status: str,
    count: int = 1,
) -> None:
    if count:
        counts[(unit, scope, threshold, budget, axis, group, status)] += int(count)


def _aggregate_detection_statuses(
    counts: Counter[AggregateKey],
    *,
    scope: str,
    threshold: float,
    budget: int,
    statuses: np.ndarray,
    buckets: Mapping[str, np.ndarray],
    density_bucket: str,
) -> None:
    status_names = np.asarray(STATUS_ORDER, dtype=object)
    for status_code, status in enumerate(status_names):
        mask = statuses == status_code
        _add_count(counts, "detection", scope, threshold, budget, "all", "all", str(status), int(mask.sum()))
        _add_count(
            counts,
            "detection",
            scope,
            threshold,
            budget,
            "density",
            density_bucket,
            str(status),
            int(mask.sum()),
        )
        for axis, codes in buckets.items():
            if axis == "prediction_input_short":
                labels = ("lt8", "8to16", "16to32", "32to64", "ge64")
            elif axis == "prediction_small_partition":
                labels = ("neither", "coco_only", "hybrid_only", "coco_and_hybrid")
            else:
                labels = ("small", "medium", "large")
            for code, label in enumerate(labels):
                _add_count(
                    counts,
                    "detection",
                    scope,
                    threshold,
                    budget,
                    axis,
                    label,
                    str(status),
                    int(np.count_nonzero(mask & (codes == code))),
                )


def _gt_axes(size: Mapping[str, Any], density_bucket: str) -> tuple[tuple[str, str], ...]:
    axes = [
        ("all", "all"),
        ("small_partition", str(size["small_partition"])),
        ("coco_area", str(size["coco_area_group"])),
        ("input_area", str(size["input_area_group"])),
        ("input_short", str(size["input_short_bucket"])),
        ("density", density_bucket),
    ]
    if size["coco_small"]:
        axes.append(("small_cohort", "coco_small"))
    if size["hybrid_small"]:
        axes.append(("small_cohort", "hybrid_small"))
    return tuple(axes)


def _fieldnames(
    scopes: Sequence[str], thresholds: Sequence[float], budgets: Sequence[int]
) -> list[str]:
    fields = [
        "state",
        "annotation_id",
        "image_id",
        "category_id",
        "iscrowd",
        "coco_area",
        "coco_area_group",
        "input_area_px2",
        "input_area_group",
        "input_short_px",
        "input_short_bucket",
        "coco_small",
        "hybrid_small",
        "small_partition",
        "image_gt_count",
        "density_bucket",
    ]
    for scope in scopes:
        for threshold in thresholds:
            prefix = f"{scope}_{_threshold_label(threshold)}"
            fields.extend(f"{prefix}_status_d{budget}" for budget in budgets)
            fields.extend(
                [
                    f"{prefix}_query_index",
                    f"{prefix}_rank",
                    f"{prefix}_score",
                    f"{prefix}_iou",
                ]
            )
    return fields


def _format_float(value: float) -> str:
    return format(float(value), ".9g")


def _write_shard(
    *,
    output_path: Path,
    detection_status_path: Path,
    summary_path: Path,
    dataset: Mapping[str, Any],
    images: Mapping[int, Mapping[str, Any]],
    annotations_by_image: Mapping[int, Sequence[Mapping[str, Any]]],
    image_density: Mapping[int, int],
    shard_image_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    category_id: int,
    input_size: tuple[int, int],
    state_name: str,
    thresholds: Sequence[float],
    budgets: Sequence[int],
    shard_index: int,
) -> dict[str, Any]:
    query_count = boxes.shape[1]
    max_det = budgets[-1]
    states = _build_shard_evaluation(
        dataset,
        images,
        annotations_by_image,
        shard_image_ids,
        boxes,
        scores,
        category_id,
        thresholds,
        max_det,
    )
    counts: Counter[AggregateKey] = Counter()
    fields = _fieldnames([name for name, _ in SCOPES], thresholds, budgets)
    annotation_rows = 0
    if query_count > np.iinfo(np.uint16).max:
        raise ValueError("query count exceeds compact uint16 rank schema")
    # Query-index order is retained here.  A stable descending sort of the
    # flattened image-major score array exactly reproduces COCO's global tie
    # order after per-image stable sorting.  The explicit budget axis avoids
    # making downstream analysis reinterpret rank cutoffs.
    detection_status = np.full(
        (
            len(budgets),
            len(SCOPES),
            len(thresholds),
            len(shard_image_ids),
            query_count,
        ),
        STATUS_ORDER.index("truncated"),
        dtype=np.uint8,
    )
    query_rank = np.empty((len(shard_image_ids), query_count), dtype=np.uint16)
    nonignored_gt_count = np.zeros(
        (len(SCOPES), len(shard_image_ids)), dtype=np.uint16
    )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                writer = csv.DictWriter(text_handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
                writer.writeheader()
                for local_image_index, image_id_value in enumerate(shard_image_ids):
                    image_id = int(image_id_value)
                    image = images[image_id]
                    image_annotations = list(annotations_by_image.get(image_id, ()))
                    density = int(image_density[image_id])
                    density_label = _density_bucket(density)
                    full_score_order = np.argsort(
                        -scores[local_image_index], kind="stable"
                    )
                    query_rank[local_image_index, full_score_order] = np.arange(
                        1, query_count + 1, dtype=np.uint16
                    )
                    score_order = full_score_order[:max_det]
                    sorted_detection_ids = local_image_index * query_count + score_order + 1
                    sorted_scores = scores[local_image_index, score_order]
                    bucket_arrays = _prediction_buckets(boxes[local_image_index], image, input_size)
                    scope_matches: dict[tuple[str, float], dict[int, tuple[int, int, float, float]]] = {}
                    scope_ignored_gt: dict[str, set[int]] = {}

                    for scope_index, (scope_name, area_range) in enumerate(SCOPES):
                        state_tuple = states[scope_index * len(shard_image_ids) + local_image_index]
                        dt_matches_flat, _gt_matches_flat, dt_scores, _gt_ignore, dt_ignore_flat, match_triples = state_tuple
                        detection_count = len(dt_scores)
                        if detection_count != len(score_order):
                            raise RuntimeError(
                                f"image {image_id}: evaluator returned {detection_count} detections, "
                                f"expected {len(score_order)}"
                            )
                        if not np.allclose(np.asarray(dt_scores), sorted_scores, rtol=0.0, atol=1e-7):
                            raise RuntimeError(f"image {image_id}: evaluator score order disagrees with stable rank")
                        match_iou = {
                            (int(detection_id), int(annotation_id)): float(iou)
                            for detection_id, annotation_id, iou in match_triples
                        }
                        ignored_gt = {
                            int(ann["id"])
                            for ann in image_annotations
                            if ann.get("iscrowd", 0)
                            or float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])) < area_range[0]
                            or float(ann.get("area", ann["bbox"][2] * ann["bbox"][3])) > area_range[1]
                        }
                        scope_ignored_gt[scope_name] = ignored_gt
                        positive_count = len(image_annotations) - len(ignored_gt)
                        if positive_count > np.iinfo(np.uint16).max:
                            raise ValueError(
                                f"image {image_id}: nonignored GT count exceeds uint16 schema"
                            )
                        nonignored_gt_count[scope_index, local_image_index] = positive_count

                        expected_flat = len(thresholds) * detection_count
                        if len(dt_matches_flat) != expected_flat or len(dt_ignore_flat) != expected_flat:
                            raise RuntimeError(f"image {image_id}: malformed C++ match state")
                        for threshold_index, threshold in enumerate(thresholds):
                            start = threshold_index * detection_count
                            stop = start + detection_count
                            matched_gt_ids = np.asarray(dt_matches_flat[start:stop], dtype=np.int64)
                            ignored_detections = np.asarray(dt_ignore_flat[start:stop], dtype=bool)
                            gt_matches: dict[int, tuple[int, int, float, float]] = {}
                            for rank_index, annotation_id in enumerate(matched_gt_ids):
                                if annotation_id == 0 or ignored_detections[rank_index]:
                                    continue
                                detection_id = int(sorted_detection_ids[rank_index])
                                query_index = int(score_order[rank_index])
                                gt_matches[int(annotation_id)] = (
                                    query_index,
                                    rank_index + 1,
                                    float(scores[local_image_index, query_index]),
                                    match_iou[(detection_id, int(annotation_id))],
                                )
                            scope_matches[(scope_name, threshold)] = gt_matches

                            for budget_index, budget in enumerate(budgets):
                                considered = min(budget, detection_count)
                                # Status codes follow STATUS_ORDER: TP=0, FP=1,
                                # FN=2 (not used for detections), ignored=3,
                                # truncated=4.
                                statuses = np.full(query_count, 4, dtype=np.uint8)
                                considered_queries = score_order[:considered]
                                considered_matches = matched_gt_ids[:considered]
                                considered_ignore = ignored_detections[:considered]
                                statuses[considered_queries] = 1
                                statuses[considered_queries[considered_ignore]] = 3
                                true_positive = (considered_matches > 0) & ~considered_ignore
                                statuses[considered_queries[true_positive]] = 0
                                detection_status[
                                    budget_index,
                                    scope_index,
                                    threshold_index,
                                    local_image_index,
                                ] = statuses
                                _aggregate_detection_statuses(
                                    counts,
                                    scope=scope_name,
                                    threshold=threshold,
                                    budget=budget,
                                    statuses=statuses,
                                    buckets=bucket_arrays,
                                    density_bucket=density_label,
                                )

                    for ann in image_annotations:
                        annotation_id = int(ann["id"])
                        area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
                        size = _size_fields(ann["bbox"], image, coco_area=area, input_size=input_size)
                        row: dict[str, Any] = {
                            "state": state_name,
                            "annotation_id": annotation_id,
                            "image_id": image_id,
                            "category_id": category_id,
                            "iscrowd": int(bool(ann.get("iscrowd", 0))),
                            "coco_area": _format_float(area),
                            "coco_area_group": size["coco_area_group"],
                            "input_area_px2": _format_float(size["input_area_px2"]),
                            "input_area_group": size["input_area_group"],
                            "input_short_px": _format_float(size["input_short_px"]),
                            "input_short_bucket": size["input_short_bucket"],
                            "coco_small": int(size["coco_small"]),
                            "hybrid_small": int(size["hybrid_small"]),
                            "small_partition": size["small_partition"],
                            "image_gt_count": density,
                            "density_bucket": density_label,
                        }
                        for scope_name, _area_range in SCOPES:
                            gt_ignored = annotation_id in scope_ignored_gt[scope_name]
                            for threshold in thresholds:
                                prefix = f"{scope_name}_{_threshold_label(threshold)}"
                                match = scope_matches[(scope_name, threshold)].get(annotation_id)
                                for budget in budgets:
                                    if gt_ignored:
                                        status = "ignored"
                                    elif match is not None and match[1] <= budget:
                                        status = "tp"
                                    else:
                                        status = "fn"
                                    row[f"{prefix}_status_d{budget}"] = status
                                    # Detailed GT axes are reported for the
                                    # all-area match.  The official COCO-small
                                    # scope gets an overall row and remains in
                                    # the per-annotation CSV.
                                    axes = _gt_axes(size, density_label) if scope_name == "all" else (("all", "all"),)
                                    for axis, group in axes:
                                        _add_count(
                                            counts,
                                            "ground_truth",
                                            scope_name,
                                            threshold,
                                            budget,
                                            axis,
                                            group,
                                            status,
                                        )
                                if match is None or gt_ignored:
                                    values: tuple[Any, Any, Any, Any] = ("", "", "", "")
                                else:
                                    values = (
                                        match[0],
                                        match[1],
                                        _format_float(match[2]),
                                        _format_float(match[3]),
                                    )
                                row[f"{prefix}_query_index"] = values[0]
                                row[f"{prefix}_rank"] = values[1]
                                row[f"{prefix}_score"] = values[2]
                                row[f"{prefix}_iou"] = values[3]
                        writer.writerow(row)
                        annotation_rows += 1
    temporary.replace(output_path)

    _atomic_npz(
        detection_status_path,
        schema=np.asarray("coco_detection_status_shard_v1"),
        state=np.asarray(state_name),
        shard_index=np.asarray(shard_index, dtype=np.int32),
        image_ids=np.asarray(shard_image_ids, dtype=np.int64),
        scores=np.asarray(scores),
        query_rank=query_rank,
        status=detection_status,
        nonignored_gt_count=nonignored_gt_count,
        max_dets=np.asarray(budgets, dtype=np.int32),
        scope_names=np.asarray([name for name, _ in SCOPES]),
        iou_thresholds=np.asarray(thresholds, dtype=np.float64),
        status_names=np.asarray(STATUS_ORDER),
        axis_order=np.asarray("max_dets,scopes,iou_thresholds,images,queries"),
        query_axis=np.asarray("zero-based decoder query index"),
        score_tie_order=np.asarray(
            "stable descending score; ties preserve image-id then query-index order"
        ),
        budget_note=np.asarray(
            "status is explicit per maxDets; detections outside a budget are truncated"
        ),
    )

    summary = {
        "schema": "coco_one_to_one_match_shard_v2",
        "shard_index": shard_index,
        "first_image_id": int(shard_image_ids[0]),
        "last_image_id": int(shard_image_ids[-1]),
        "images": len(shard_image_ids),
        "annotations": annotation_rows,
        "predictions": len(shard_image_ids) * query_count,
        "gt_matches_file": output_path.name,
        "gt_matches_sha256": sha256(output_path),
        "detection_status_file": detection_status_path.name,
        "detection_status_sha256": sha256(detection_status_path),
        "counts": [
            {
                "unit": key[0],
                "scope": key[1],
                "iou_threshold": key[2],
                "max_dets": key[3],
                "axis": key[4],
                "group": key[5],
                "status": key[6],
                "count": count,
            }
            for key, count in sorted(counts.items())
        ],
    }
    _atomic_json(summary_path, summary)
    return summary


def _counter_from_summary(summary: Mapping[str, Any]) -> Counter[AggregateKey]:
    result: Counter[AggregateKey] = Counter()
    for row in summary["counts"]:
        key: AggregateKey = (
            row["unit"],
            row["scope"],
            float(row["iou_threshold"]),
            int(row["max_dets"]),
            row["axis"],
            row["group"],
            row["status"],
        )
        result[key] += int(row["count"])
    return result


def _pivot_summary(counts: Counter[AggregateKey], state: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float, int, str, str], dict[str, int]] = defaultdict(dict)
    for key, count in counts.items():
        groups[key[:-1]][key[-1]] = int(count)
    rows: list[dict[str, Any]] = []
    for key, status_counts in sorted(groups.items()):
        unit, scope, threshold, budget, axis, group = key
        tp = status_counts.get("tp", 0)
        fp = status_counts.get("fp", 0) if unit == "detection" else None
        fn = status_counts.get("fn", 0) if unit == "ground_truth" else None
        precision = tp / (tp + fp) if fp is not None and tp + fp else None
        recall = tp / (tp + fn) if fn is not None and tp + fn else None
        rows.append(
            {
                "state": state,
                "unit": unit,
                "scope": scope,
                "iou_threshold": threshold,
                "max_dets": budget,
                "axis": axis,
                "group": group,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ignored": status_counts.get("ignored", 0),
                "truncated": status_counts.get("truncated", 0),
                "precision": precision,
                "recall": recall,
                "note": (
                    "GT rows intentionally omit FP; unmatched detections have no GT cohort."
                    if unit == "ground_truth"
                    else "Detections beyond maxDets are truncated, not FP."
                ),
            }
        )

    # Join the two unambiguous overall views into conventional TP/FP/FN rows.
    for scope, _ in SCOPES:
        thresholds = sorted({key[2] for key in groups if key[1] == scope})
        budgets = sorted({key[3] for key in groups if key[1] == scope})
        for threshold in thresholds:
            for budget in budgets:
                gt = groups.get(("ground_truth", scope, threshold, budget, "all", "all"), {})
                dt = groups.get(("detection", scope, threshold, budget, "all", "all"), {})
                tp, fp, fn = gt.get("tp", 0), dt.get("fp", 0), gt.get("fn", 0)
                precision = tp / (tp + fp) if tp + fp else None
                recall = tp / (tp + fn) if tp + fn else None
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if precision is not None and recall is not None and precision + recall
                    else 0.0
                )
                rows.append(
                    {
                        "state": state,
                        "unit": "overall",
                        "scope": scope,
                        "iou_threshold": threshold,
                        "max_dets": budget,
                        "axis": "all",
                        "group": "all",
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "ignored": dt.get("ignored", 0),
                        "truncated": dt.get("truncated", 0),
                        "precision": precision,
                        "recall": recall,
                        "note": f"one-to-one F1={f1:.9g}; not independent best-IoU coverage",
                    }
                )
    return rows


def _write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty merged summary")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reconstruct_global_pr(
    output_dir: Path,
    *,
    scope: str,
    iou_threshold: float,
    max_det: int,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Exactly reconstruct one global COCO-style PR stream from status shards.

    Returned ``event_scores`` and ``event_is_tp`` contain only evaluated TP/FP
    detections in stable global score order.  Ignored and budget-truncated
    detections are counted but excluded from the precision denominator, as in
    COCO.  The 101-point sampled precision and score arrays follow
    ``COCOeval.accumulate`` for this repository's single-category protocol.
    """

    manifest_path = output_dir / "detection_status_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "coco_detection_status_manifest_v1":
        raise ValueError(f"unknown detection-status manifest schema: {manifest_path}")
    scopes = tuple(manifest["scope_names"])
    budgets = tuple(int(value) for value in manifest["max_dets"])
    thresholds = tuple(float(value) for value in manifest["iou_thresholds"])
    if scope not in scopes:
        raise ValueError(f"unknown scope {scope!r}; choose from {scopes}")
    if max_det not in budgets:
        raise ValueError(f"unknown maxDets {max_det}; choose from {budgets}")
    threshold_matches = [
        index
        for index, value in enumerate(thresholds)
        if math.isclose(value, float(iou_threshold), rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(threshold_matches) != 1:
        raise ValueError(
            f"unknown or ambiguous IoU threshold {iou_threshold}; choose from {thresholds}"
        )
    budget_index = budgets.index(max_det)
    scope_index = scopes.index(scope)
    threshold_index = threshold_matches[0]
    tp_code = STATUS_ORDER.index("tp")
    fp_code = STATUS_ORDER.index("fp")
    ignored_code = STATUS_ORDER.index("ignored")
    truncated_code = STATUS_ORDER.index("truncated")
    event_score_parts: list[np.ndarray] = []
    event_tp_parts: list[np.ndarray] = []
    num_positive = 0
    status_counts = np.zeros(len(STATUS_ORDER), dtype=np.int64)

    for shard in manifest["shards"]:
        shard_path = output_dir / shard["file"]
        if verify_hashes and sha256(shard_path) != shard["sha256"]:
            raise ValueError(f"detection-status shard hash mismatch: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as archive:
            if str(archive["schema"]) != "coco_detection_status_shard_v1":
                raise ValueError(f"unknown detection-status shard schema: {shard_path}")
            status = np.asarray(
                archive["status"][
                    budget_index, scope_index, threshold_index
                ],
                dtype=np.uint8,
            )
            scores = np.asarray(archive["scores"])
            if status.shape != scores.shape:
                raise ValueError(f"status/score shape mismatch: {shard_path}")
            counts = np.bincount(status.reshape(-1), minlength=len(STATUS_ORDER))
            status_counts += counts[: len(STATUS_ORDER)]
            evaluated = (status == tp_code) | (status == fp_code)
            event_score_parts.append(scores[evaluated])
            event_tp_parts.append(status[evaluated] == tp_code)
            num_positive += int(
                np.asarray(archive["nonignored_gt_count"])[scope_index].sum(
                    dtype=np.int64
                )
            )

    event_scores = (
        np.concatenate(event_score_parts)
        if event_score_parts
        else np.empty(0, dtype=np.float32)
    )
    event_is_tp = (
        np.concatenate(event_tp_parts)
        if event_tp_parts
        else np.empty(0, dtype=bool)
    )
    order = np.argsort(-event_scores, kind="stable")
    event_scores = event_scores[order]
    event_is_tp = event_is_tp[order]
    cumulative_tp = np.cumsum(event_is_tp, dtype=np.int64)
    cumulative_fp = np.cumsum(~event_is_tp, dtype=np.int64)
    if num_positive:
        recall = cumulative_tp.astype(np.float64) / num_positive
        precision = cumulative_tp.astype(np.float64) / (
            cumulative_tp + cumulative_fp + np.spacing(1)
        )
        precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        sampled_precision = np.zeros_like(recall_thresholds)
        sampled_score = np.zeros_like(recall_thresholds)
        indexes = np.searchsorted(recall, recall_thresholds, side="left")
        valid = indexes < len(precision_envelope)
        sampled_precision[valid] = precision_envelope[indexes[valid]]
        sampled_score[valid] = event_scores[indexes[valid]]
        ap = float(sampled_precision.mean())
        ar = float(recall[-1]) if len(recall) else 0.0
    else:
        recall = np.empty(0, dtype=np.float64)
        precision = np.empty(0, dtype=np.float64)
        recall_thresholds = np.linspace(0.0, 1.0, 101)
        sampled_precision = np.full_like(recall_thresholds, -1.0)
        sampled_score = np.zeros_like(recall_thresholds)
        ap = -1.0
        ar = -1.0
    return {
        "scope": scope,
        "iou_threshold": float(thresholds[threshold_index]),
        "max_dets": max_det,
        "num_positive": num_positive,
        "num_tp": int(status_counts[tp_code]),
        "num_fp": int(status_counts[fp_code]),
        "num_ignored": int(status_counts[ignored_code]),
        "num_truncated": int(status_counts[truncated_code]),
        "event_scores": event_scores,
        "event_is_tp": event_is_tp,
        "cumulative_tp": cumulative_tp,
        "cumulative_fp": cumulative_fp,
        "recall": recall,
        "precision": precision,
        "recall_thresholds": recall_thresholds,
        "sampled_precision": sampled_precision,
        "sampled_score": sampled_score,
        "ap": ap,
        "ar": ar,
    }


def export_npz_matches(
    annotation_path: Path,
    prediction_path: Path,
    output_dir: Path,
    *,
    max_dets: Iterable[int] = (300, 640),
    iou_thresholds: Iterable[float] = tuple(
        round(0.50 + 0.05 * index, 2) for index in range(10)
    ),
    images_per_shard: int = 256,
    max_images: int | None = None,
) -> Path:
    """Export one prediction state; reruns resume at completed shard files."""

    budgets = _normalise_budgets(max_dets)
    thresholds = _normalise_thresholds(iou_thresholds)
    if images_per_shard <= 0:
        raise ValueError("images_per_shard must be positive")
    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    archive, image_ids, boxes, scores, category_id, input_size, state_name = _load_npz_contract(prediction_path)
    try:
        categories = {item.get("id") for item in dataset.get("categories", [])}
        if category_id not in categories:
            raise ValueError(f"NPZ category_id {category_id} is absent from annotations")
        images, annotations_by_image, image_density = _annotation_index(dataset, category_id)
        if list(map(int, image_ids)) != sorted(map(int, image_ids)):
            raise ValueError("image_ids must be globally sorted ascending")
        if boxes.shape[1] < budgets[-1]:
            raise ValueError(f"NPZ has {boxes.shape[1]} queries, fewer than maxDets={budgets[-1]}")
        selected_images = len(image_ids) if max_images is None else min(max_images, len(image_ids))
        if selected_images <= 0:
            raise ValueError("max_images selects no images")
        annotation_image_ids = set(images)
        prediction_image_ids = set(map(int, image_ids))
        if max_images is None:
            if prediction_image_ids != annotation_image_ids:
                missing = len(annotation_image_ids - prediction_image_ids)
                extra = len(prediction_image_ids - annotation_image_ids)
                raise ValueError(
                    f"prediction/annotation image mismatch: missing={missing}, extra={extra}"
                )
        else:
            # Smoke replays commonly contain only a sorted subset of a full
            # annotation file.  Require every prediction image to be known,
            # but do not require the archive to cover unused annotation images.
            extra = prediction_image_ids - annotation_image_ids
            if extra:
                raise ValueError(
                    "prediction image IDs are absent from annotations: "
                    f"{sorted(extra)[:10]}"
                )

        output_dir.mkdir(parents=True, exist_ok=True)
        protocol = {
            "schema": "coco_one_to_one_npz_export_v2",
            "annotation": str(annotation_path.resolve()),
            "annotation_sha256": sha256(annotation_path),
            "prediction": str(prediction_path.resolve()),
            "prediction_sha256": sha256(prediction_path),
            "state": state_name,
            "category_id": category_id,
            "input_size": list(input_size),
            "images": selected_images,
            "annotation_images": len(annotation_image_ids),
            "prediction_images": len(image_ids),
            "max_images_requested": max_images,
            "first_selected_image_id": int(image_ids[0]),
            "last_selected_image_id": int(image_ids[selected_images - 1]),
            "queries_per_image": int(boxes.shape[1]),
            "boxes_dtype": str(boxes.dtype),
            "scores_dtype": str(scores.dtype),
            "max_dets": list(budgets),
            "iou_thresholds": list(thresholds),
            "scopes": {name: list(area_range) for name, area_range in SCOPES},
            "images_per_shard": images_per_shard,
            "matching": (
                "faster_coco_eval C++ bbox matcher; stable descending score; one-to-one non-crowd; "
                "crowd/area matches ignored; maxDets truncation is not FP"
            ),
            "small_boundary_note": (
                "descriptive coco_small cohorts use area < 32^2, matching the local analysis code; "
                "the coco_small_eval scope uses the evaluator's official [0, 32^2] areaRange, "
                "whose upper endpoint is inclusive"
            ),
            "hybrid_small": (
                "after direct input resize: bbox area < 32^2 OR bbox short side < 16; "
                "used as an all-area GT cohort, not a COCO areaRange"
            ),
            "coverage_warning": "formal COCO one-to-one matches; not independent per-GT best-IoU coverage",
            "detection_status": (
                "compressed query-index status arrays retain TP/FP/ignored/truncated and scores "
                "for exact stable global PR reconstruction"
            ),
        }
        protocol_path = output_dir / "protocol.json"
        if protocol_path.exists():
            existing = json.loads(protocol_path.read_text(encoding="utf-8"))
            if existing != protocol:
                raise ValueError(f"existing output protocol differs: {protocol_path}")
        else:
            _atomic_json(protocol_path, protocol)

        shard_summaries: list[dict[str, Any]] = []
        for shard_index, start in enumerate(range(0, selected_images, images_per_shard)):
            stop = min(start + images_per_shard, selected_images)
            csv_path = output_dir / f"gt_matches.part{shard_index:04d}.csv.gz"
            status_path = output_dir / f"detection_status.part{shard_index:04d}.npz"
            summary_path = output_dir / f"summary.part{shard_index:04d}.json"
            if csv_path.exists() and status_path.exists() and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    summary.get("schema") != "coco_one_to_one_match_shard_v2"
                    or summary.get("shard_index") != shard_index
                    or summary.get("images") != stop - start
                    or summary.get("first_image_id") != int(image_ids[start])
                    or summary.get("last_image_id") != int(image_ids[stop - 1])
                    or summary.get("gt_matches_file") != csv_path.name
                    or summary.get("detection_status_file") != status_path.name
                ):
                    raise ValueError(f"invalid resume summary: {summary_path}")
                if summary.get("gt_matches_sha256") != sha256(csv_path):
                    raise ValueError(f"resumed GT match shard hash mismatch: {csv_path}")
                if summary.get("detection_status_sha256") != sha256(status_path):
                    raise ValueError(
                        f"resumed detection-status shard hash mismatch: {status_path}"
                    )
                shard_summaries.append(summary)
                print(f"resume shard {shard_index:04d}: {start}:{stop}")
                continue
            shard_boxes = np.asarray(boxes[start:stop])
            shard_scores = np.asarray(scores[start:stop])
            if (
                not np.isfinite(shard_boxes).all()
                or not np.isfinite(shard_scores).all()
                or (shard_boxes[..., 2:] < 0).any()
            ):
                raise ValueError(f"non-finite score or invalid bbox in shard {shard_index}")
            summary = _write_shard(
                output_path=csv_path,
                detection_status_path=status_path,
                summary_path=summary_path,
                dataset=dataset,
                images=images,
                annotations_by_image=annotations_by_image,
                image_density=image_density,
                shard_image_ids=image_ids[start:stop],
                boxes=shard_boxes,
                scores=shard_scores,
                category_id=category_id,
                input_size=input_size,
                state_name=state_name,
                thresholds=thresholds,
                budgets=budgets,
                shard_index=shard_index,
            )
            shard_summaries.append(summary)
            print(f"completed shard {shard_index:04d}: {start}:{stop}")

        merged: Counter[AggregateKey] = Counter()
        for summary in shard_summaries:
            merged.update(_counter_from_summary(summary))
        summary_rows = _pivot_summary(merged, state_name)
        summary_csv = output_dir / "match_aggregates.csv"
        _write_summary_csv(summary_csv, summary_rows)
        status_manifest = {
            "schema": "coco_detection_status_manifest_v1",
            "state": state_name,
            "scope_names": [name for name, _ in SCOPES],
            "iou_thresholds": list(thresholds),
            "max_dets": list(budgets),
            "status_codes": {
                name: index for index, name in enumerate(STATUS_ORDER)
            },
            "axis_order": "max_dets,scopes,iou_thresholds,images,queries",
            "score_axis": "images,queries in zero-based decoder query-index order",
            "global_order": (
                "flatten shards in listed order, then image-major/query-index order; "
                "stable sort descending score; exclude ignored and truncated"
            ),
            "positive_count": (
                "sum nonignored_gt_count over images for the requested scope"
            ),
            "shards": [
                {
                    "index": int(summary["shard_index"]),
                    "file": summary["detection_status_file"],
                    "sha256": summary["detection_status_sha256"],
                    "first_image_id": int(summary["first_image_id"]),
                    "last_image_id": int(summary["last_image_id"]),
                    "images": int(summary["images"]),
                }
                for summary in shard_summaries
            ],
        }
        status_manifest_path = output_dir / "detection_status_manifest.json"
        _atomic_json(status_manifest_path, status_manifest)
        gt_shard_manifest = {
            "schema": "coco_gt_match_shard_manifest_v1",
            "state": state_name,
            "shards": [
                {
                    "index": int(summary["shard_index"]),
                    "summary_file": f"summary.part{int(summary['shard_index']):04d}.json",
                    "summary_sha256": sha256(
                        output_dir
                        / f"summary.part{int(summary['shard_index']):04d}.json"
                    ),
                    "gt_matches_file": summary["gt_matches_file"],
                    "gt_matches_sha256": summary["gt_matches_sha256"],
                    "detection_status_file": summary["detection_status_file"],
                    "detection_status_sha256": summary["detection_status_sha256"],
                    "first_image_id": int(summary["first_image_id"]),
                    "last_image_id": int(summary["last_image_id"]),
                    "images": int(summary["images"]),
                    "annotations": int(summary["annotations"]),
                }
                for summary in shard_summaries
            ],
        }
        gt_shard_manifest_path = output_dir / "gt_shard_manifest.json"
        _atomic_json(gt_shard_manifest_path, gt_shard_manifest)
        complete = {
            "schema": "coco_one_to_one_npz_complete_v2",
            "protocol_sha256": sha256(protocol_path),
            "shards": len(shard_summaries),
            "images": sum(int(item["images"]) for item in shard_summaries),
            "annotations": sum(int(item["annotations"]) for item in shard_summaries),
            "predictions": sum(int(item["predictions"]) for item in shard_summaries),
            "aggregate_csv": summary_csv.name,
            "aggregate_sha256": sha256(summary_csv),
            "detection_status_manifest": status_manifest_path.name,
            "detection_status_manifest_sha256": sha256(status_manifest_path),
            "gt_shard_manifest": gt_shard_manifest_path.name,
            "gt_shard_manifest_sha256": sha256(gt_shard_manifest_path),
        }
        complete_path = output_dir / "complete.json"
        _atomic_json(complete_path, complete)
        return complete_path
    finally:
        archive.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True, help="compact predictions.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dets", type=int, nargs="+", default=[300, 640])
    parser.add_argument(
        "--iou-thresholds",
        type=float,
        nargs="+",
        default=[round(0.50 + 0.05 * index, 2) for index in range(10)],
    )
    parser.add_argument("--images-per-shard", type=int, default=256)
    parser.add_argument("--max-images", type=int, help="smoke-test prefix; becomes part of protocol")
    args = parser.parse_args(argv)
    complete = export_npz_matches(
        args.annotation,
        args.predictions,
        args.output,
        max_dets=args.max_dets,
        iou_thresholds=args.iou_thresholds,
        images_per_shard=args.images_per_shard,
        max_images=args.max_images,
    )
    print(complete.resolve())


if __name__ == "__main__":
    main()
