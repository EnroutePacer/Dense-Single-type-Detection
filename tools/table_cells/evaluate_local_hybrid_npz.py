"""Evaluate compact detector predictions with the project's local-small rule.

This is a deliberately separate, CPU-friendly evaluator for the single-GPU
micro probe.  It consumes the ``sgqs_compact_predictions_v1`` NPZ emitted by
``full_val_replay.py`` and a COCO annotation subset, then reports exact
one-to-one COCO-style AP/AR for the local hybrid-small population:

``scaled_area < 32**2 OR scaled_short_side < 16``

where the scale is the direct 640x640 resize used by the table-cell code.  The
formal matcher in :mod:`coco_match_npz` is intentionally left unchanged; this
tool is an additive diagnostic scope.  In addition to AP/AR/AP75 at budgets
300 and 640, it writes per-image recall@0.75 rows so two states can be paired
without treating GT boxes or queries as independent observations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

# When invoked as ``python tools/table_cells/evaluate_local_hybrid_npz.py``
# (the documented remote command), Python puts ``tools/table_cells`` rather
# than the repository root on ``sys.path``.  Add the root explicitly so the
# sibling package import below works both as a script and with ``python -m``.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.table_cells.local_small_eval import (
    dense_image_ids,
    evaluate_predictions,
    is_local_small,
)
from engine.data.dataset.coco_backend import COCO


SCHEMA = "sgqs_local_hybrid_npz_eval_v1"
PREDICTION_SCHEMA = "sgqs_compact_predictions_v1"
INPUT_SIZE = (640, 640)
RECALL_IOU = 0.75
DIAGNOSTIC_IOUS = (0.50, 0.75, 0.90)


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _prediction_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "prediction must be STATE=PATH, for example ema=val.ema.predictions.npz"
        )
    state, path = value.split("=", 1)
    state = state.strip()
    path = path.strip()
    if not state or not path:
        raise argparse.ArgumentTypeError("prediction must be STATE=PATH")
    return state, Path(path)


def _load_prediction(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    required = {"image_ids", "boxes_xywh", "scores"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"prediction archive is missing {sorted(missing)}: {path}")
    schema = arrays.get("schema")
    if schema is not None and _text(schema.reshape(-1)[0]) != PREDICTION_SCHEMA:
        raise ValueError(f"unknown compact prediction schema in {path}: {schema!r}")
    input_size = arrays.get("input_size")
    if input_size is not None and tuple(int(value) for value in input_size.reshape(-1)) != INPUT_SIZE:
        raise ValueError(f"local hybrid scope requires input_size={INPUT_SIZE}: {path}")
    box_format = arrays.get("box_format")
    if box_format is not None and _text(box_format.reshape(-1)[0]) != "xywh_abs_original_pixels":
        raise ValueError(f"unsupported compact box format in {path}: {box_format!r}")
    image_ids = arrays["image_ids"].astype(np.int64, copy=False)
    boxes = arrays["boxes_xywh"].astype(np.float64, copy=False)
    scores = arrays["scores"].astype(np.float64, copy=False)
    if image_ids.ndim != 1 or boxes.ndim != 3 or boxes.shape[-1] != 4:
        raise ValueError(f"invalid prediction shapes in {path}")
    if scores.shape != boxes.shape[:2] or len(image_ids) != boxes.shape[0]:
        raise ValueError(f"image/query dimensions disagree in {path}")
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(f"duplicate image ids in {path}")
    if not np.array_equal(image_ids, np.sort(image_ids)):
        raise ValueError(
            f"prediction image ids must be sorted ascending for image-aligned rows: {path}"
        )
    if boxes.shape[1] < 1:
        raise ValueError(f"prediction archive has no queries: {path}")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError(f"prediction archive contains non-finite values: {path}")
    if (boxes[..., 2:] < 0).any():
        raise ValueError(f"prediction archive contains negative box extents: {path}")
    category = arrays.get("category_id")
    if category is None:
        category_id = None
    else:
        flat = category.reshape(-1)
        if len(flat) != 1:
            raise ValueError(f"category_id must be scalar in {path}")
        category_id = int(flat[0])
    return {
        "image_ids": image_ids,
        "boxes_xywh": boxes,
        "scores": scores,
        "category_id": category_id,
    }


def _image_iou(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """IoU between one xywh box and an array of xywh boxes."""

    if len(gt_boxes) == 0:
        return np.empty(0, dtype=np.float64)
    box_x1, box_y1 = box[0], box[1]
    box_x2, box_y2 = box_x1 + box[2], box_y1 + box[3]
    gt_x1, gt_y1 = gt_boxes[:, 0], gt_boxes[:, 1]
    gt_x2, gt_y2 = gt_x1 + gt_boxes[:, 2], gt_y1 + gt_boxes[:, 3]
    inter_x1 = np.maximum(box_x1, gt_x1)
    inter_y1 = np.maximum(box_y1, gt_y1)
    inter_x2 = np.minimum(box_x2, gt_x2)
    inter_y2 = np.minimum(box_y2, gt_y2)
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area = max(float(box[2]), 0.0) * max(float(box[3]), 0.0)
    gt_area = np.maximum(gt_boxes[:, 2], 0.0) * np.maximum(gt_boxes[:, 3], 0.0)
    union = area + gt_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _local_gt(coco: COCO, image_id: int) -> np.ndarray:
    image = coco.imgs[int(image_id)]
    rows = [
        ann
        for ann in coco.imgToAnns.get(int(image_id), [])
        if not ann.get("iscrowd", 0) and is_local_small(ann["bbox"], image)
    ]
    rows.sort(key=lambda ann: int(ann.get("id", 0)))
    if not rows:
        return np.empty((0, 4), dtype=np.float64)
    boxes = np.asarray([ann["bbox"] for ann in rows], dtype=np.float64)
    if (boxes[:, 2:] <= 0).any():
        raise ValueError(f"non-positive local GT bbox in image {image_id}")
    return boxes


def _match_counts(
    boxes: np.ndarray,
    scores: np.ndarray,
    gt_boxes: np.ndarray,
    budget: int,
    threshold: float,
) -> tuple[int, int]:
    """Return greedy score-ordered one-to-one TP/FP counts for one image."""

    order = np.argsort(-scores, kind="stable")[: int(budget)]
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for query_index in order:
        ious = _image_iou(boxes[int(query_index)], gt_boxes)
        if len(ious) == 0:
            continue
        ious[matched] = -1.0
        gt_index = int(np.argmax(ious))
        if ious[gt_index] >= float(threshold):
            matched[gt_index] = True
            tp += 1
    return int(tp), int(len(order) - tp)


def per_image_recall75(
    coco: COCO,
    image_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    """Return image-cluster rows with recall/precision at several IoUs.

    The legacy ``recall75_*`` fields remain the primary paired signal.  The
    additional 0.50/0.90 and precision fields are cheap CPU-side guards: a
    candidate cannot hide a large false-positive or localization regression
    behind a recall gain at one threshold.
    """

    rows: list[dict[str, Any]] = []
    for image_index, image_id_value in enumerate(image_ids):
        image_id = int(image_id_value)
        gt_boxes = _local_gt(coco, image_id)
        row: dict[str, Any] = {
            "image_id": image_id,
            "local_gt_count": int(len(gt_boxes)),
        }
        for budget in budgets:
            for threshold in DIAGNOSTIC_IOUS:
                tp, fp = _match_counts(
                    boxes[image_index], scores[image_index], gt_boxes,
                    int(budget), threshold,
                )
                suffix = f"{int(round(threshold * 100))}"
                row[f"tp{suffix}_{int(budget)}"] = int(tp)
                row[f"fp{suffix}_{int(budget)}"] = int(fp)
                row[f"recall{suffix}_{int(budget)}"] = (
                    float(tp / len(gt_boxes)) if len(gt_boxes) else None
                )
                row[f"precision{suffix}_{int(budget)}"] = (
                    float(tp / (tp + fp)) if (tp + fp) else None
                )
            row[f"predictions_{int(budget)}"] = int(min(int(budget), scores.shape[1]))
        rows.append(row)
    return rows


def _prediction_matrix(
    image_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    category_id: int,
) -> np.ndarray:
    count, queries = boxes.shape[:2]
    result = np.empty((count * queries, 7), dtype=np.float64)
    result[:, 0] = np.repeat(image_ids, queries)
    result[:, 1:5] = boxes.reshape(-1, 4)
    result[:, 5] = scores.reshape(-1)
    result[:, 6] = category_id
    return result


def evaluate_state(
    annotation: Path,
    prediction: Path,
    *,
    budgets: Sequence[int] = (300, 640),
    dense_min_gt: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one compact state and return summary plus per-image rows."""

    coco = COCO(str(annotation))
    if len(coco.getCatIds()) != 1:
        raise ValueError("local hybrid evaluator requires exactly one category")
    payload = _load_prediction(prediction)
    image_ids = payload["image_ids"]
    expected_ids = np.asarray(sorted(coco.getImgIds()), dtype=np.int64)
    if not np.array_equal(np.sort(image_ids), expected_ids):
        raise ValueError(
            f"prediction image IDs do not exactly match annotation: "
            f"{prediction} ({len(image_ids)} vs {len(expected_ids)})"
        )
    category_id = payload["category_id"]
    if category_id is None:
        category_id = int(coco.getCatIds()[0])
    if category_id not in set(int(value) for value in coco.getCatIds()):
        raise ValueError(f"prediction category {category_id} is absent from annotation")
    budgets = tuple(sorted({int(value) for value in budgets}))
    if not budgets or min(budgets) < 1:
        raise ValueError("budgets must be positive")
    rows = per_image_recall75(
        coco, image_ids, payload["boxes_xywh"], payload["scores"], budgets
    )
    result = evaluate_predictions(
        coco,
        _prediction_matrix(
            image_ids, payload["boxes_xywh"], payload["scores"], category_id
        ),
        budgets,
    )
    local_metrics = {
        str(budget): result["budgets"][str(budget)]["local_hybrid_small"]
        for budget in budgets
    }
    summary = {
        "schema": SCHEMA,
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation),
        "prediction": str(prediction.resolve()),
        "prediction_sha256": sha256(prediction),
        "input_size": list(INPUT_SIZE),
        "recall_iou": RECALL_IOU,
        "diagnostic_ious": list(DIAGNOSTIC_IOUS),
        "images": int(len(coco.imgs)),
        "local_small_gt": int(sum(row["local_gt_count"] for row in rows)),
        "budgets": local_metrics,
        "full_metrics": result,
        "per_image_rows": len(rows),
        "definition": "direct 640x640 resize: area < 32^2 OR short side < 16",
    }
    if dense_min_gt is not None:
        if int(dense_min_gt) < 1:
            raise ValueError("dense_min_gt must be positive")
        dense_ids = dense_image_ids(coco, int(dense_min_gt))
        summary[f"dense_ge{int(dense_min_gt)}"] = {
            "selection": "annotation_non_crowd_gt_count",
            "minimum_gt": int(dense_min_gt),
            "image_ids": [int(value) for value in dense_ids],
            "metrics": evaluate_predictions(
                coco,
                _prediction_matrix(
                    image_ids,
                    payload["boxes_xywh"],
                    payload["scores"],
                    category_id,
                ),
                budgets,
                image_ids=dense_ids,
            ),
        }
    return summary, rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def evaluate(
    annotation: Path,
    predictions: Sequence[tuple[str, Path]],
    output: Path,
    *,
    budgets: Sequence[int] = (300, 640),
    dense_min_gt: int | None = None,
) -> dict[str, Any]:
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not predictions:
        raise ValueError("at least one --prediction STATE=PATH is required")
    names = [name for name, _ in predictions]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate prediction states: {names}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    state_summaries: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for state, path in predictions:
        summary, rows = evaluate_state(
            annotation,
            path,
            budgets=budgets,
            dense_min_gt=dense_min_gt,
        )
        state_summaries[state] = summary
        all_rows.extend({"state": state, **row} for row in rows)
    per_image_path = output / "per_image_recall75.csv"
    _write_csv(per_image_path, all_rows)
    result = {
        "schema": SCHEMA,
        "annotation": str(annotation.resolve()),
        "annotation_sha256": sha256(annotation),
        # Repeat the immutable evaluator contract at the top level so the
        # paired gate can verify a sidecar without having to trust one
        # arbitrarily selected state.  Each state keeps the same fields too
        # for per-prediction auditability.
        "input_size": list(INPUT_SIZE),
        "definition": "direct 640x640 resize: area < 32^2 OR short side < 16",
        "budgets": [int(value) for value in sorted({int(value) for value in budgets})],
        "dense_min_gt": None if dense_min_gt is None else int(dense_min_gt),
        "states": state_summaries,
        "per_image_recall75": str(per_image_path.name),
        "per_image_recall75_sha256": sha256(per_image_path),
        "pairing_unit": "image_id",
    }
    (output / "summary.json").write_text(
        json.dumps(_jsonable(result), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument(
        "--prediction", type=_prediction_spec, action="append", required=True,
        metavar="STATE=PATH", help="compact predictions NPZ; repeat for states",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[300, 640])
    parser.add_argument(
        "--dense-min-gt",
        type=int,
        help="also evaluate the image cohort with at least this many non-crowd GT boxes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate(
        args.annotation,
        args.prediction,
        args.output,
        budgets=args.budgets,
        dense_min_gt=args.dense_min_gt,
    )
    print(json.dumps({"schema": result["schema"], "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
