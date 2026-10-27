"""Evaluate B0/SGQS e34 with all 640 decoder queries and score cutoffs.

This intentionally does not use the legacy COCO maxDets=300 cap.  For every
image, all decoder queries that pass the selected score threshold are sent to
COCOeval, whose evaluation budget is fixed to 640.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from faster_coco_eval import COCO, COCOeval_faster

from tools.table_cells.full_val_replay import summary_at_budget


FIELDS = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR640",
    "ARs",
    "ARm",
    "ARl",
)


def evaluate_threshold(
    coco,
    image_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    category_id: int,
    threshold: float,
    *,
    batch_size: int,
) -> tuple[list[float], np.ndarray]:
    evaluator = COCOeval_faster(
        coco,
        iouType="bbox",
        print_function=lambda *args: None,
        separate_eval=True,
    )
    evaluator.params.maxDets = [1, 10, 640]
    chunks = []
    counts = np.zeros(len(image_ids), dtype=np.int16)

    for start in range(0, len(image_ids), batch_size):
        stop = min(start + batch_size, len(image_ids))
        detections = []
        for index in range(start, stop):
            selected = scores[index] >= threshold
            count = int(selected.sum())
            counts[index] = count
            if not count:
                continue
            item = np.empty((count, 7), dtype=np.float64)
            item[:, 0] = image_ids[index]
            item[:, 1:5] = boxes[index, selected]
            item[:, 5] = scores[index, selected]
            item[:, 6] = category_id
            detections.append(item)

        batch_detections = (
            np.concatenate(detections, axis=0)
            if detections
            else np.empty((0, 7), dtype=np.float64)
        )
        evaluator.cocoDt = coco.loadRes(batch_detections)
        batch_ids = image_ids[start:stop].tolist()
        evaluator.params.imgIds = batch_ids
        evaluator.evaluate()
        chunks.append(
            np.asarray(evaluator._evalImgs_cpp, dtype=object).reshape(
                len(evaluator.params.catIds),
                len(evaluator.params.areaRng),
                len(batch_ids),
            )
        )

    evaluator.params.imgIds = image_ids.tolist()
    evaluator._paramsEval = copy.deepcopy(evaluator.params)
    evaluator._evalImgs_cpp = np.concatenate(chunks, axis=2).ravel().tolist()
    evaluator.accumulate()
    return summary_at_budget(evaluator, 640), counts


def load_predictions(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
        boxes = np.asarray(archive["boxes_xywh"], dtype=np.float32)
        scores = np.asarray(archive["scores"], dtype=np.float32)
        category_id = int(archive["category_id"])
    expected = (len(image_ids), 640)
    if scores.shape != expected or boxes.shape != (*expected, 4):
        raise ValueError(f"expected 640-query predictions, got {boxes.shape=} {scores.shape=}")
    return image_ids, boxes, scores, category_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--b0-predictions", type=Path, required=True)
    parser.add_argument("--sgqs-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=(0.3, 0.4, 0.5, 0.6)
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        parser.error("thresholds must be in [0,1]")

    coco = COCO(str(args.annotation))
    rows = []
    for model, path in (
        ("B0", args.b0_predictions),
        ("SGQS", args.sgqs_predictions),
    ):
        image_ids, boxes, scores, category_id = load_predictions(path)
        if set(image_ids.tolist()) != set(coco.getImgIds()):
            raise ValueError(f"{model} prediction image IDs do not match annotation")
        for threshold in args.thresholds:
            started = time.monotonic()
            metrics, counts = evaluate_threshold(
                coco,
                image_ids,
                boxes,
                scores,
                category_id,
                threshold,
                batch_size=args.batch_size,
            )
            row = {
                "model": model,
                "epoch": 34,
                "state": "ema",
                "score_threshold": threshold,
                "max_dets": 640,
                "images": len(image_ids),
                "detections": int(counts.sum()),
                "detections_mean": float(counts.mean()),
                "detections_median": float(np.median(counts)),
                "detections_p95": float(np.quantile(counts, 0.95)),
                "detections_max": int(counts.max()),
                **dict(zip(FIELDS, metrics)),
                "seconds": time.monotonic() - started,
            }
            rows.append(row)
            print(
                f"{model} threshold={threshold:g} AP={row['AP']:.9f} "
                f"APs={row['APs']:.9f}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
