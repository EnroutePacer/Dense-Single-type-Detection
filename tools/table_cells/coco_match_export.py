"""Export auditable COCO one-to-one matches from saved predictions.

This module is deliberately separate from the per-GT best-IoU *coverage*
replay.  It runs the same score-ordered, one-to-one greedy matcher used by
``faster_coco_eval`` and records which decoder query became a TP, FP, or FN.

Predictions use the ordinary COCO result schema (``image_id``,
``category_id``, ``bbox`` in original-image xywh pixels, and ``score``).  A
``query_index`` field is optional but strongly recommended: COCO preserves
extra result fields, so the exported match can then be traced back to the
decoder query without another model forward pass.

This in-memory JSON interface is the reference/small-subset implementation.
For full validation (millions of detections), use ``coco_match_npz.py``; it
uses the same evaluator semantics with compact arrays, gzip shards, and safe
resume.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


COCO_SMALL_AREA = 32.0**2
COCO_MEDIUM_AREA = 96.0**2
LOCAL_SHORT_SIDE = 16.0


@dataclass
class MatchResult:
    """One prediction state evaluated at one or more budgets/thresholds."""

    state: str
    max_dets: tuple[int, ...]
    iou_thresholds: tuple[float, ...]
    ground_truth_rows: list[dict[str, Any]]
    detection_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    metadata: dict[str, Any]


def _normalise_positive_ints(value: int | Iterable[int], name: str) -> tuple[int, ...]:
    values = (value,) if isinstance(value, int) else tuple(value)
    if not values or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return tuple(sorted(set(values)))


def _normalise_thresholds(value: float | Iterable[float]) -> tuple[float, ...]:
    values = (value,) if isinstance(value, (int, float)) else tuple(value)
    values = tuple(float(item) for item in values)
    if not values or any(not math.isfinite(item) or item <= 0.0 or item > 1.0 for item in values):
        raise ValueError("iou_thresholds must be finite values in (0, 1]")
    return tuple(sorted(set(values)))


def _coco_area_group(area: float) -> str:
    if area < COCO_SMALL_AREA:
        return "small"
    if area < COCO_MEDIUM_AREA:
        return "medium"
    return "large"


def _short_side_bucket(short_side: float) -> str:
    if short_side < 8.0:
        return "lt8"
    if short_side < 16.0:
        return "8to16"
    if short_side < 32.0:
        return "16to32"
    if short_side < 64.0:
        return "32to64"
    return "ge64"


def _density_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count <= 32:
        return "1to32"
    if count <= 128:
        return "33to128"
    if count <= 300:
        return "129to300"
    return "ge301"


def _small_partition(coco_small: bool, hybrid_small: bool) -> str:
    if coco_small and hybrid_small:
        return "coco_and_hybrid"
    if coco_small:
        return "coco_only"
    if hybrid_small:
        return "hybrid_only"
    return "neither"


def _size_fields(
    bbox: Sequence[float],
    image: Mapping[str, Any],
    *,
    coco_area: float,
    input_size: tuple[int, int],
) -> dict[str, Any]:
    input_width, input_height = input_size
    width = float(bbox[2]) * input_width / float(image["width"])
    height = float(bbox[3]) * input_height / float(image["height"])
    input_area = width * height
    input_short = min(width, height)
    coco_small = coco_area < COCO_SMALL_AREA
    hybrid_small = input_area < COCO_SMALL_AREA or input_short < LOCAL_SHORT_SIDE
    return {
        "coco_area": coco_area,
        "coco_area_group": _coco_area_group(coco_area),
        "input_width_px": width,
        "input_height_px": height,
        "input_area_px2": input_area,
        "input_area_group": _coco_area_group(input_area),
        "input_short_px": input_short,
        "input_short_bucket": _short_side_bucket(input_short),
        "coco_small": coco_small,
        "hybrid_small": hybrid_small,
        "small_partition": _small_partition(coco_small, hybrid_small),
    }


def _dataset_dict(coco_dataset: Mapping[str, Any] | COCO) -> dict[str, Any]:
    if isinstance(coco_dataset, COCO):
        dataset = coco_dataset.dataset
    elif isinstance(coco_dataset, Mapping):
        dataset = coco_dataset
    else:
        raise TypeError("coco_dataset must be a COCO object or a dataset mapping")
    result = copy.deepcopy(dict(dataset))
    for key in ("images", "annotations", "categories"):
        if key not in result or not isinstance(result[key], list):
            raise ValueError(f"COCO dataset is missing list field {key!r}")
    return result


def _validate_dataset(dataset: Mapping[str, Any]) -> tuple[dict[int, dict], dict[int, dict]]:
    images: dict[int, dict] = {}
    for image in dataset["images"]:
        image_id = image.get("id")
        if not isinstance(image_id, int) or isinstance(image_id, bool) or image_id in images:
            raise ValueError(f"invalid or duplicate image id: {image_id!r}")
        if float(image.get("width", 0)) <= 0 or float(image.get("height", 0)) <= 0:
            raise ValueError(f"image {image_id} has invalid dimensions")
        images[image_id] = image

    annotations: dict[int, dict] = {}
    categories = {item.get("id") for item in dataset["categories"]}
    for ann in dataset["annotations"]:
        annotation_id = ann.get("id")
        if not isinstance(annotation_id, int) or isinstance(annotation_id, bool) or annotation_id in annotations:
            raise ValueError(f"invalid or duplicate annotation id: {annotation_id!r}")
        if ann.get("image_id") not in images:
            raise ValueError(f"annotation {annotation_id} names an unknown image")
        if ann.get("category_id") not in categories:
            raise ValueError(f"annotation {annotation_id} names an unknown category")
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"annotation {annotation_id} has no xywh bbox")
        values = [float(item) for item in bbox]
        if any(not math.isfinite(item) for item in values) or values[2] < 0 or values[3] < 0:
            raise ValueError(f"annotation {annotation_id} has an invalid bbox")
        annotations[annotation_id] = ann
    return images, annotations


def _validate_predictions(
    predictions: Sequence[Mapping[str, Any]],
    images: Mapping[int, Mapping[str, Any]],
    categories: set[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    query_ids: set[tuple[int, Any]] = set()
    for index, original in enumerate(predictions):
        prediction = copy.deepcopy(dict(original))
        image_id = prediction.get("image_id")
        category_id = prediction.get("category_id")
        if image_id not in images:
            raise ValueError(f"prediction {index} names unknown image {image_id!r}")
        if category_id not in categories:
            raise ValueError(f"prediction {index} names unknown category {category_id!r}")
        bbox = prediction.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"prediction {index} has no xywh bbox")
        values = [float(item) for item in bbox]
        score = float(prediction.get("score", float("nan")))
        if any(not math.isfinite(item) for item in values) or values[2] < 0 or values[3] < 0:
            raise ValueError(f"prediction {index} has an invalid bbox")
        if not math.isfinite(score):
            raise ValueError(f"prediction {index} has a non-finite score")
        prediction["bbox"] = values
        prediction["score"] = score
        prediction["_source_prediction_index"] = index
        if "query_index" in prediction:
            key = (image_id, prediction["query_index"])
            if key in query_ids:
                raise ValueError(f"duplicate query_index {key[1]!r} for image {image_id}")
            query_ids.add(key)
        result.append(prediction)
    return result


def _make_coco(dataset: Mapping[str, Any]) -> COCO:
    coco = COCO()
    # ``_dataset_dict`` already made a private deep copy.  Reusing it here is
    # important for full validation sets whose annotation JSON can exceed
    # 100 MB; COCO evaluation only adds derived ``ignore`` fields to that
    # private copy.
    coco.dataset = dict(dataset)
    coco.createIndex()
    return coco


def _run_thresholds(
    dataset: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    thresholds: Sequence[float],
    max_det: int,
) -> tuple[dict[float, dict[int, tuple[int, float]]], dict[float, set[int]]]:
    """Return per-threshold COCO matches and ignored detection ids.

    Evaluation uses one all-area range.  Non-crowd matches are the formal
    one-to-one TP relation.  Matches to a crowd are retained only to mark the
    detection ignored, exactly as COCO does; crowd GTs are never TP/FN rows.

    The indexed GT/DT objects are intentionally reused across thresholds.
    This avoids repeatedly copying a full annotation set and millions of
    decoder predictions while leaving the evaluator's matching semantics
    unchanged.
    """

    coco_gt = _make_coco(dataset)
    # ``predictions`` is already a private validated copy, so allowing
    # loadRes to append result ids/areas avoids another full-size deepcopy.
    coco_dt = coco_gt.loadRes(predictions)
    all_matches: dict[float, dict[int, tuple[int, float]]] = {}
    all_ignored: dict[float, set[int]] = {}
    for threshold in thresholds:
        evaluator = COCOeval_faster(
            coco_gt,
            coco_dt,
            iouType="bbox",
            print_function=lambda *args: None,
            separate_eval=True,
            extra_calc=True,
        )
        evaluator.params.iouThrs = np.asarray([threshold], dtype=np.float64)
        evaluator.params.areaRng = [[0.0, 1.0e10]]
        evaluator.params.areaRngLbl = ["all"]
        evaluator.params.maxDets = [max_det]
        evaluator.evaluate()
        evaluator.accumulate()

        matches: dict[int, tuple[int, float]] = {}
        ignored: set[int] = set()
        for pair, iou in evaluator.eval.get("matched", {}).items():
            detection_id_text, annotation_id_text = pair.split("_", 1)
            detection_id = int(detection_id_text)
            annotation_id = int(annotation_id_text)
            matches[detection_id] = (annotation_id, float(iou))
            if coco_gt.anns[annotation_id].get("iscrowd", 0):
                ignored.add(detection_id)
        all_matches[threshold] = matches
        all_ignored[threshold] = ignored
    return all_matches, all_ignored


def _cohorts(row: Mapping[str, Any]) -> tuple[str, ...]:
    groups = ["all", row["small_partition"]]
    if row["coco_small"]:
        groups.append("coco_small")
    if row["hybrid_small"]:
        groups.append("hybrid_small")
    return tuple(groups)


def _summary_rows(
    state: str,
    gt_rows: Sequence[dict[str, Any]],
    detection_rows: Sequence[dict[str, Any]],
    max_dets: Sequence[int],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gt_axes = {
        "small_cohort": lambda row: _cohorts(row),
        "density": lambda row: (row["density_bucket"],),
        "coco_area": lambda row: (row["coco_area_group"],),
        "input_area": lambda row: (row["input_area_group"],),
        "input_short": lambda row: (row["input_short_bucket"],),
    }
    detection_axes = {
        "all": lambda row: ("all",),
        "prediction_coco_area": lambda row: (row["coco_area_group"],),
        "prediction_input_area": lambda row: (row["input_area_group"],),
        "prediction_input_short": lambda row: (row["input_short_bucket"],),
        "density": lambda row: (row["density_bucket"],),
    }
    for threshold in thresholds:
        threshold_key = f"iou{threshold:.2f}"
        for budget in max_dets:
            budget_key = f"maxDets{budget}"
            for axis, selector in gt_axes.items():
                groups: dict[str, list[dict[str, Any]]] = {}
                for row in gt_rows:
                    if row["iscrowd"]:
                        continue
                    for group in selector(row):
                        groups.setdefault(group, []).append(row)
                for group, members in sorted(groups.items()):
                    statuses = [row["matches"][threshold_key][budget_key]["status"] for row in members]
                    tp = statuses.count("tp")
                    fn = statuses.count("fn")
                    rows.append(
                        {
                            "state": state,
                            "iou_threshold": threshold,
                            "max_dets": budget,
                            "unit": "ground_truth",
                            "axis": axis,
                            "group": group,
                            "objects": len(members),
                            "tp": tp,
                            "fp": None,
                            "fn": fn,
                            "ignored": 0,
                            "truncated": None,
                            "precision": None,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "f1": None,
                            "note": "GT cohorts report TP/FN only; an unmatched FP has no GT cohort.",
                        }
                    )
            for axis, selector in detection_axes.items():
                groups = {}
                for row in detection_rows:
                    for group in selector(row):
                        groups.setdefault(group, []).append(row)
                for group, members in sorted(groups.items()):
                    statuses = [row["matches"][threshold_key][budget_key]["status"] for row in members]
                    tp = statuses.count("tp")
                    fp = statuses.count("fp")
                    ignored = statuses.count("ignored")
                    truncated = statuses.count("truncated")
                    rows.append(
                        {
                            "state": state,
                            "iou_threshold": threshold,
                            "max_dets": budget,
                            "unit": "detection",
                            "axis": axis,
                            "group": group,
                            "objects": len(members),
                            "tp": tp,
                            "fp": fp,
                            "fn": None,
                            "ignored": ignored,
                            "truncated": truncated,
                            "precision": tp / (tp + fp) if tp + fp else None,
                            "recall": None,
                            "f1": None,
                            "note": "truncated detections are outside maxDets and are not FP.",
                        }
                    )

            noncrowd_gt = [row for row in gt_rows if not row["iscrowd"]]
            gt_statuses = [row["matches"][threshold_key][budget_key]["status"] for row in noncrowd_gt]
            detection_statuses = [
                row["matches"][threshold_key][budget_key]["status"] for row in detection_rows
            ]
            tp = gt_statuses.count("tp")
            fn = gt_statuses.count("fn")
            fp = detection_statuses.count("fp")
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
            rows.append(
                {
                    "state": state,
                    "iou_threshold": threshold,
                    "max_dets": budget,
                    "unit": "overall",
                    "axis": "all",
                    "group": "all",
                    "objects": len(noncrowd_gt),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "ignored": detection_statuses.count("ignored"),
                    "truncated": detection_statuses.count("truncated"),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "note": "COCO one-to-one match counts, not independent best-IoU coverage.",
                }
            )
    return rows


def match_coco_predictions(
    coco_dataset: Mapping[str, Any] | COCO,
    predictions: Sequence[Mapping[str, Any]],
    *,
    max_dets: int | Iterable[int] = (300, 640),
    iou_thresholds: float | Iterable[float] = (0.5, 0.75),
    input_size: tuple[int, int] = (640, 640),
    state: str = "unspecified",
) -> MatchResult:
    """Match saved predictions and return per-GT/per-detection audit rows.

    Matching is performed once at the largest requested ``max_dets``.  Lower
    budgets are exact prefixes of COCO's stable score ordering, so their
    matches are obtained by filtering on rank; a lower-scored detection cannot
    alter a higher-scored detection's greedy match.
    """

    budgets = _normalise_positive_ints(max_dets, "max_dets")
    thresholds = _normalise_thresholds(iou_thresholds)
    if (
        len(input_size) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in input_size)
    ):
        raise ValueError("input_size must be (positive integer width, positive integer height)")

    dataset = _dataset_dict(coco_dataset)
    images, annotations = _validate_dataset(dataset)
    categories = {item["id"] for item in dataset["categories"]}
    clean_predictions = _validate_predictions(predictions, images, categories)

    image_gt_count = {image_id: 0 for image_id in images}
    image_category_gt_count: dict[tuple[int, int], int] = {}
    for ann in annotations.values():
        if ann.get("iscrowd", 0):
            continue
        image_gt_count[ann["image_id"]] += 1
        key = (ann["image_id"], ann["category_id"])
        image_category_gt_count[key] = image_category_gt_count.get(key, 0) + 1

    # COCO uses stable mergesort by descending score independently for each
    # image/category pair.  Keep the original input order as the tie breaker.
    grouped: dict[tuple[int, int], list[int]] = {}
    for index, prediction in enumerate(clean_predictions):
        grouped.setdefault((prediction["image_id"], prediction["category_id"]), []).append(index)
    ranks: dict[int, int] = {}
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: -clean_predictions[index]["score"])
        ranks.update({prediction_index: rank for rank, prediction_index in enumerate(ordered, 1)})

    largest_budget = budgets[-1]
    threshold_matches, threshold_ignored = _run_thresholds(
        dataset, clean_predictions, thresholds, largest_budget
    )

    gt_rows: list[dict[str, Any]] = []
    gt_row_by_id: dict[int, dict[str, Any]] = {}
    for annotation_id in sorted(annotations):
        ann = annotations[annotation_id]
        image = images[ann["image_id"]]
        area = float(ann.get("area", float(ann["bbox"][2]) * float(ann["bbox"][3])))
        row = {
            "state": state,
            "annotation_id": annotation_id,
            "image_id": ann["image_id"],
            "category_id": ann["category_id"],
            "bbox": [float(item) for item in ann["bbox"]],
            "iscrowd": bool(ann.get("iscrowd", 0)),
            "image_gt_count": image_gt_count[ann["image_id"]],
            "image_category_gt_count": image_category_gt_count.get(
                (ann["image_id"], ann["category_id"]), 0
            ),
            "density_bucket": _density_bucket(image_gt_count[ann["image_id"]]),
            **_size_fields(ann["bbox"], image, coco_area=area, input_size=input_size),
            "matches": {},
        }
        gt_rows.append(row)
        gt_row_by_id[annotation_id] = row

    detection_rows: list[dict[str, Any]] = []
    for index, prediction in enumerate(clean_predictions):
        image = images[prediction["image_id"]]
        area = float(prediction["bbox"][2]) * float(prediction["bbox"][3])
        row = {
            "state": state,
            "prediction_index": index,
            "detection_id": index + 1,
            "query_index": prediction.get("query_index"),
            "image_id": prediction["image_id"],
            "category_id": prediction["category_id"],
            "bbox": prediction["bbox"],
            "score": prediction["score"],
            "rank": ranks[index],
            "image_gt_count": image_gt_count[prediction["image_id"]],
            "image_category_gt_count": image_category_gt_count.get(
                (prediction["image_id"], prediction["category_id"]), 0
            ),
            "density_bucket": _density_bucket(image_gt_count[prediction["image_id"]]),
            **_size_fields(prediction["bbox"], image, coco_area=area, input_size=input_size),
            "matches": {},
        }
        detection_rows.append(row)

    for threshold in thresholds:
        threshold_key = f"iou{threshold:.2f}"
        matches = threshold_matches[threshold]
        ignored_detection_ids = threshold_ignored[threshold]
        gt_to_detection = {
            annotation_id: (detection_id, iou)
            for detection_id, (annotation_id, iou) in matches.items()
            if detection_id not in ignored_detection_ids
        }
        for row in gt_rows:
            row["matches"][threshold_key] = {}
        for row in detection_rows:
            row["matches"][threshold_key] = {}

        for budget in budgets:
            budget_key = f"maxDets{budget}"
            matched_gt_ids: set[int] = set()
            for row in detection_rows:
                detection_id = row["detection_id"]
                if row["rank"] > budget:
                    result = {
                        "status": "truncated",
                        "annotation_id": None,
                        "iou": None,
                    }
                elif detection_id in matches:
                    annotation_id, iou = matches[detection_id]
                    gt_row = gt_row_by_id[annotation_id]
                    if detection_id in ignored_detection_ids or gt_row["iscrowd"]:
                        status = "ignored"
                    else:
                        status = "tp"
                        matched_gt_ids.add(annotation_id)
                    result = {
                        "status": status,
                        "annotation_id": annotation_id,
                        "iou": iou,
                        "matched_gt_coco_small": gt_row["coco_small"],
                        "matched_gt_hybrid_small": gt_row["hybrid_small"],
                        "matched_gt_small_partition": gt_row["small_partition"],
                    }
                else:
                    result = {"status": "fp", "annotation_id": None, "iou": None}
                row["matches"][threshold_key][budget_key] = result

            for row in gt_rows:
                if row["iscrowd"]:
                    result = {
                        "status": "ignored",
                        "prediction_index": None,
                        "detection_id": None,
                        "query_index": None,
                        "score": None,
                        "rank": None,
                        "iou": None,
                    }
                elif row["annotation_id"] not in matched_gt_ids:
                    result = {
                        "status": "fn",
                        "prediction_index": None,
                        "detection_id": None,
                        "query_index": None,
                        "score": None,
                        "rank": None,
                        "iou": None,
                    }
                else:
                    detection_id, iou = gt_to_detection[row["annotation_id"]]
                    detection = detection_rows[detection_id - 1]
                    result = {
                        "status": "tp",
                        "prediction_index": detection["prediction_index"],
                        "detection_id": detection_id,
                        "query_index": detection["query_index"],
                        "score": detection["score"],
                        "rank": detection["rank"],
                        "iou": iou,
                    }
                row["matches"][threshold_key][budget_key] = result

    summary_rows = _summary_rows(state, gt_rows, detection_rows, budgets, thresholds)
    metadata = {
        "schema": "coco_one_to_one_matches_v1",
        "state": state,
        "images": len(images),
        "annotations": len(annotations),
        "noncrowd_annotations": sum(not row["iscrowd"] for row in gt_rows),
        "predictions": len(detection_rows),
        "max_dets": list(budgets),
        "iou_thresholds": list(thresholds),
        "input_size": list(input_size),
        "matching": (
            "faster_coco_eval score-ordered one-to-one matching in the all-area range; "
            "crowd matches are ignored and detections outside maxDets are truncated, not FP"
        ),
        "coco_small": "annotation area < 32^2 in original-image pixels",
        "hybrid_small": (
            "after direct resize to input_size: bbox area < 32^2 OR bbox short side < 16"
        ),
        "coverage_warning": (
            "These are formal one-to-one match rows. They must not be merged with or named "
            "independent per-GT best-IoU coverage."
        ),
        "fp_bucket_warning": (
            "GT cohort summaries intentionally omit FP: an unmatched detection has no GT cohort. "
            "Use detection geometry or overall rows for FP/precision."
        ),
    }
    return MatchResult(
        state=state,
        max_dets=budgets,
        iou_thresholds=thresholds,
        ground_truth_rows=gt_rows,
        detection_rows=detection_rows,
        summary_rows=summary_rows,
        metadata=metadata,
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load a COCO result JSON list or one JSON object per JSONL line."""

    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(row)
        return rows
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"{path} must contain a JSON list of prediction objects")
    return data


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_match_result(result: MatchResult, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = result.state.replace("/", "_").replace("\\", "_")
    paths = {
        "ground_truth": output_dir / f"{prefix}.coco_matches.gt.jsonl",
        "detections": output_dir / f"{prefix}.coco_matches.detections.jsonl",
        "summary": output_dir / f"{prefix}.coco_matches.summary.csv",
        "metadata": output_dir / f"{prefix}.coco_matches.metadata.json",
    }
    _write_jsonl(paths["ground_truth"], result.ground_truth_rows)
    _write_jsonl(paths["detections"], result.detection_rows)
    _write_csv(paths["summary"], result.summary_rows)
    paths["metadata"].write_text(
        json.dumps(result.metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def _prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be STATE=PATH, for example ema=predictions.jsonl")
    state, path = value.split("=", 1)
    if not state or not path:
        raise argparse.ArgumentTypeError("prediction must be STATE=PATH")
    return state, Path(path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True, help="COCO ground-truth JSON")
    parser.add_argument(
        "--prediction",
        type=_prediction_spec,
        action="append",
        required=True,
        metavar="STATE=PATH",
        help="saved COCO result JSON/JSONL; repeat for raw and EMA",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dets", type=int, nargs="+", default=[300, 640])
    parser.add_argument("--iou-thresholds", type=float, nargs="+", default=[0.5, 0.75])
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"))
    args = parser.parse_args(argv)

    dataset = json.loads(args.annotation.read_text(encoding="utf-8"))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    states: dict[str, dict[str, Any]] = {}
    for state, path in args.prediction:
        if state in states:
            raise ValueError(f"duplicate prediction state: {state}")
        result = match_coco_predictions(
            dataset,
            load_predictions(path),
            max_dets=args.max_dets,
            iou_thresholds=args.iou_thresholds,
            input_size=tuple(args.input_size),
            state=state,
        )
        states[state] = {
            "prediction_file": str(path.resolve()),
            "files": write_match_result(result, args.output),
            "metadata": result.metadata,
        }
    manifest = {
        "schema": "coco_one_to_one_match_export_v1",
        "annotation": str(args.annotation.resolve()),
        "states": states,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
