"""Analyze fixed-val128 SGQS replay outputs using EMA weights only.

This is an offline audit helper.  It does not import or modify training code.
The per-GT quantities are independent best-IoU coverage, not COCO one-to-one
matching.  See tools/table_cells/replay_probe.py for how the rows were emitted.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ.get("SGQS_REPLAY_ROOT", Path.cwd())).resolve()
ANNOTATION_FILE = Path(
    os.environ.get(
        "SGQS_REPLAY_ANNOTATION",
        ROOT / "probe_fixed128_valonly" / "val.json",
    )
).resolve()
OUT = Path(
    os.environ.get("SGQS_REPLAY_OUTPUT", ROOT / "val128_ema_analysis")
).resolve()
EPOCHS = tuple(range(28, 35))
THRESHOLDS = (0.5, 0.75)
STAGES = (
    ("dense_all", "encoder_dense_best_iou_all"),
    ("dense_top640", "encoder_dense_best_iou_top640"),
    ("decoder_all640", "decoder_best_iou_all"),
    ("decoder_top300", "decoder_best_iou_top300"),
)
PAIRS = ((28, 29), (28, 30), (30, 34))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path):
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            annotation_id = row["annotation_id"]
            if annotation_id in rows:
                raise ValueError(f"duplicate annotation_id {annotation_id} in {path}:{line_number}")
            rows[annotation_id] = row
    return rows


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summaries(values: list[float]):
    if not values:
        return {"median": None, "p10": None, "p90": None}
    return {
        "median": statistics.median(values),
        "p10": quantile(values, 0.1),
        "p90": quantile(values, 0.9),
    }


annotation = load_json(ANNOTATION_FILE)
images = {image["id"]: image for image in annotation["images"]}
annotations = {
    ann["id"]: ann
    for ann in annotation["annotations"]
    if not ann.get("iscrowd", 0)
}

meta = {}
for annotation_id, ann in annotations.items():
    image = images[ann["image_id"]]
    _, _, width, height = ann["bbox"]
    input_width = width / image["width"] * 640.0
    input_height = height / image["height"] * 640.0
    input_area = input_width * input_height
    input_short = min(input_width, input_height)
    coco_small = ann["area"] < 32**2
    hybrid_small = input_area < 32**2 or input_short < 16.0
    meta[annotation_id] = {
        "image_id": ann["image_id"],
        "coco_area": ann["area"],
        "input_area": input_area,
        "input_short": input_short,
        "coco_small": coco_small,
        "hybrid_small": hybrid_small,
    }

epoch_rows = {}
epoch_summaries = {}
reference_ids = set(annotations)
subset_hashes = set()
for epoch in EPOCHS:
    replay_dir = ROOT / f"replay_val128_e00{epoch}"
    summary = load_json(replay_dir / "summary.json")
    if summary["last_epoch"] != epoch:
        raise ValueError(f"last_epoch mismatch for e{epoch}: {summary['last_epoch']}")
    subset_hashes.add(summary["subset_sha256"])
    rows = load_jsonl(replay_dir / "val.ema.gt.jsonl")
    if set(rows) != reference_ids:
        missing = len(reference_ids - set(rows))
        extra = len(set(rows) - reference_ids)
        raise ValueError(f"annotation-id mismatch e{epoch}: missing={missing}, extra={extra}")
    for annotation_id, row in rows.items():
        expected = meta[annotation_id]
        if row["image_id"] != expected["image_id"]:
            raise ValueError(f"image mismatch e{epoch}, annotation {annotation_id}")
        if not math.isclose(row["coco_area"], expected["coco_area"], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"COCO area mismatch e{epoch}, annotation {annotation_id}")
        if not math.isclose(row["input_short_px"], expected["input_short"], rel_tol=0, abs_tol=1e-3):
            raise ValueError(f"input short-side mismatch e{epoch}, annotation {annotation_id}")
    epoch_rows[epoch] = rows
    epoch_summaries[epoch] = summary["results"]["val/ema"]

if len(subset_hashes) != 1:
    raise ValueError(f"multiple subset hashes found: {sorted(subset_hashes)}")

groups = {
    "coco_small": {annotation_id for annotation_id, item in meta.items() if item["coco_small"]},
    "hybrid_small": {annotation_id for annotation_id, item in meta.items() if item["hybrid_small"]},
}
exclusive_groups = {
    "coco_and_hybrid": groups["coco_small"] & groups["hybrid_small"],
    "coco_only": groups["coco_small"] - groups["hybrid_small"],
    "hybrid_only": groups["hybrid_small"] - groups["coco_small"],
}

# COCO stats array: AP, AP50, AP75, APs, APm, APl, AR1, AR10, ARmax,
# ARs, ARm, ARl.  Only APs and ARs are requested here.
coco_curve = []
for epoch in EPOCHS:
    result = epoch_summaries[epoch]
    at300 = result["coco_maxdets300"]
    at640 = result["coco_maxdets640"]
    coco_curve.append(
        {
            "epoch": epoch,
            "APs_at300": at300[3],
            "ARs_at300": at300[9],
            "APs_at640": at640[3],
            "ARs_at640": at640[9],
        }
    )

coverage = []
for epoch in EPOCHS:
    rows = epoch_rows[epoch]
    for group_name, ids in groups.items():
        for threshold in THRESHOLDS:
            for stage_name, field in STAGES:
                covered = sum(rows[annotation_id][field] >= threshold for annotation_id in ids)
                coverage.append(
                    {
                        "epoch": epoch,
                        "group": group_name,
                        "n_gt": len(ids),
                        "iou_threshold": threshold,
                        "stage": stage_name,
                        "covered": covered,
                        "coverage": covered / len(ids),
                    }
                )


def endpoint_state(row: dict, threshold: float):
    return tuple(row[field] >= threshold for _, field in STAGES)


def stranded_location(state: tuple[bool, ...]):
    """Deepest stage that still covers a GT when final top-300 misses it."""
    dense_all, dense_top, decoder_all, decoder_top = state
    if decoder_top:
        return "covered_top300"
    if decoder_all:
        return "decoder_present_rank_gt300"
    if dense_top:
        return "dense_selected_decoder_miss"
    if dense_all:
        return "dense_candidate_not_selected"
    return "no_iou_qualified_dense_candidate"


stage_transitions = []
final_transitions = []
stranding = []
transition_annotation_rows = []
for start, end in PAIRS:
    for group_name, ids in groups.items():
        for threshold in THRESHOLDS:
            start_rows = epoch_rows[start]
            end_rows = epoch_rows[end]
            for stage_name, field in STAGES:
                counts = Counter()
                for annotation_id in ids:
                    before = start_rows[annotation_id][field] >= threshold
                    after = end_rows[annotation_id][field] >= threshold
                    counts[(before, after)] += 1
                stage_transitions.append(
                    {
                        "start_epoch": start,
                        "end_epoch": end,
                        "group": group_name,
                        "n_gt": len(ids),
                        "iou_threshold": threshold,
                        "stage": stage_name,
                        "stable_miss_00": counts[(False, False)],
                        "gain_01": counts[(False, True)],
                        "loss_10": counts[(True, False)],
                        "stable_cover_11": counts[(True, True)],
                        "net_gain": counts[(False, True)] - counts[(True, False)],
                    }
                )

            final_field = "decoder_best_iou_top300"
            gained = []
            lost = []
            stable_covered = []
            stable_missed = []
            for annotation_id in ids:
                before = start_rows[annotation_id][final_field] >= threshold
                after = end_rows[annotation_id][final_field] >= threshold
                if not before and after:
                    gained.append(annotation_id)
                elif before and not after:
                    lost.append(annotation_id)
                elif before and after:
                    stable_covered.append(annotation_id)
                else:
                    stable_missed.append(annotation_id)

            final_transitions.append(
                {
                    "start_epoch": start,
                    "end_epoch": end,
                    "group": group_name,
                    "n_gt": len(ids),
                    "iou_threshold": threshold,
                    "gained_top300": len(gained),
                    "lost_top300": len(lost),
                    "net_gain": len(gained) - len(lost),
                    "stable_covered": len(stable_covered),
                    "stable_missed": len(stable_missed),
                }
            )

            for direction, selected, location_epoch in (
                ("gain_source", gained, start),
                ("gain_destination", gained, end),
                ("loss_source", lost, start),
                ("loss_destination", lost, end),
            ):
                location_counts = Counter(
                    stranded_location(endpoint_state(epoch_rows[location_epoch][annotation_id], threshold))
                    for annotation_id in selected
                )
                for location in (
                    "covered_top300",
                    "decoder_present_rank_gt300",
                    "dense_selected_decoder_miss",
                    "dense_candidate_not_selected",
                    "no_iou_qualified_dense_candidate",
                ):
                    stranding.append(
                        {
                            "start_epoch": start,
                            "end_epoch": end,
                            "group": group_name,
                            "iou_threshold": threshold,
                            "direction": direction,
                            "n_transition_gt": len(selected),
                            "location_epoch": location_epoch,
                            "location": location,
                            "count": location_counts[location],
                            "fraction": location_counts[location] / len(selected) if selected else 0.0,
                        }
                    )

            for transition, selected in (("gain", gained), ("loss", lost)):
                for annotation_id in selected:
                    before = start_rows[annotation_id]
                    after = end_rows[annotation_id]
                    row = {
                        "start_epoch": start,
                        "end_epoch": end,
                        "group": group_name,
                        "iou_threshold": threshold,
                        "transition": transition,
                        "annotation_id": annotation_id,
                        "image_id": meta[annotation_id]["image_id"],
                        "coco_area": meta[annotation_id]["coco_area"],
                        "input_area": meta[annotation_id]["input_area"],
                        "input_short": meta[annotation_id]["input_short"],
                        "decoder_rank_before": before[f"decoder_first_rank_iou{threshold}"],
                        "decoder_rank_after": after[f"decoder_first_rank_iou{threshold}"],
                    }
                    for stage_name, field in STAGES:
                        row[f"{stage_name}_before"] = before[field]
                        row[f"{stage_name}_after"] = after[field]
                    transition_annotation_rows.append(row)

# Summarize decoder ranks for final gain/loss sets.  Rank is one-based and 641
# means no query reaches the requested IoU threshold.
rank_shifts = []
for start, end in PAIRS:
    for group_name in groups:
        for threshold in THRESHOLDS:
            selected_rows = [
                row
                for row in transition_annotation_rows
                if row["start_epoch"] == start
                and row["end_epoch"] == end
                and row["group"] == group_name
                and row["iou_threshold"] == threshold
            ]
            for transition in ("gain", "loss"):
                matches = [row for row in selected_rows if row["transition"] == transition]
                before_values = [row["decoder_rank_before"] for row in matches]
                after_values = [row["decoder_rank_after"] for row in matches]
                before_summary = summaries(before_values)
                after_summary = summaries(after_values)
                rank_shifts.append(
                    {
                        "start_epoch": start,
                        "end_epoch": end,
                        "group": group_name,
                        "iou_threshold": threshold,
                        "transition": transition,
                        "count": len(matches),
                        "rank_before_median": before_summary["median"],
                        "rank_before_p10": before_summary["p10"],
                        "rank_before_p90": before_summary["p90"],
                        "rank_after_median": after_summary["median"],
                        "rank_after_p10": after_summary["p10"],
                        "rank_after_p90": after_summary["p90"],
                    }
                )

# Mutually exclusive buckets make it possible to see which definition is
# receiving Top-300 capacity.  Keep this separate from the two requested
# headline groups so their denominators remain unambiguous.
exclusive_top300 = []
for epoch in EPOCHS:
    for group_name, ids in exclusive_groups.items():
        for threshold in THRESHOLDS:
            covered = sum(
                epoch_rows[epoch][annotation_id]["decoder_best_iou_top300"] >= threshold
                for annotation_id in ids
            )
            exclusive_top300.append(
                {
                    "epoch": epoch,
                    "group": group_name,
                    "n_gt": len(ids),
                    "iou_threshold": threshold,
                    "covered": covered,
                    "coverage": covered / len(ids),
                }
            )

exclusive_transitions = []
for start, end in PAIRS:
    for group_name, ids in exclusive_groups.items():
        for threshold in THRESHOLDS:
            counts = Counter()
            for annotation_id in ids:
                before = epoch_rows[start][annotation_id]["decoder_best_iou_top300"] >= threshold
                after = epoch_rows[end][annotation_id]["decoder_best_iou_top300"] >= threshold
                counts[(before, after)] += 1
            exclusive_transitions.append(
                {
                    "start_epoch": start,
                    "end_epoch": end,
                    "group": group_name,
                    "n_gt": len(ids),
                    "iou_threshold": threshold,
                    "gain_01": counts[(False, True)],
                    "loss_10": counts[(True, False)],
                    "net_gain": counts[(False, True)] - counts[(True, False)],
                }
            )

image_gt_counts = Counter(item["image_id"] for item in meta.values())
image_transition_counter = Counter(
    (
        row["start_epoch"],
        row["end_epoch"],
        row["group"],
        row["iou_threshold"],
        row["transition"],
        row["image_id"],
    )
    for row in transition_annotation_rows
)
image_transitions = [
    {
        "start_epoch": key[0],
        "end_epoch": key[1],
        "group": key[2],
        "iou_threshold": key[3],
        "transition": key[4],
        "image_id": key[5],
        "transition_gt": count,
        "image_total_gt": image_gt_counts[key[5]],
    }
    for key, count in sorted(image_transition_counter.items())
]

OUT.mkdir(parents=True, exist_ok=True)
write_csv(OUT / "coco_small_curve.csv", coco_curve)
write_csv(OUT / "stage_coverage.csv", coverage)
write_csv(OUT / "stage_transitions.csv", stage_transitions)
write_csv(OUT / "final_top300_transitions.csv", final_transitions)
write_csv(OUT / "transition_stranding.csv", stranding)
write_csv(OUT / "transition_rank_shifts.csv", rank_shifts)
write_csv(OUT / "transition_annotations.csv", transition_annotation_rows)
write_csv(OUT / "exclusive_top300_coverage.csv", exclusive_top300)
write_csv(OUT / "exclusive_top300_transitions.csv", exclusive_transitions)
write_csv(OUT / "transition_image_counts.csv", image_transitions)

audit = {
    "epochs": list(EPOCHS),
    "ema_only": True,
    "subset_sha256": next(iter(subset_hashes)),
    "images": len(images),
    "noncrowd_gt": len(annotations),
    "group_counts": {name: len(ids) for name, ids in groups.items()},
    "group_overlap": {
        "intersection": len(groups["coco_small"] & groups["hybrid_small"]),
        "coco_only": len(groups["coco_small"] - groups["hybrid_small"]),
        "hybrid_only": len(groups["hybrid_small"] - groups["coco_small"]),
        "neither": len(reference_ids - groups["coco_small"] - groups["hybrid_small"]),
    },
    "images_with_coco_small": len({meta[annotation_id]["image_id"] for annotation_id in groups["coco_small"]}),
    "definitions": {
        "coco_small": "annotation area in original-image pixels < 32^2",
        "hybrid_small": "after direct 640x640 resize: bbox area < 32^2 OR bbox short side < 16",
        "coverage": "each GT independently asks whether any candidate/query reaches the IoU threshold; not one-to-one COCO matching",
    },
}
(OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

print(json.dumps(audit, indent=2))
print(f"wrote analysis tables to {OUT}")
