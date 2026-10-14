"""Analyze the fixed-128 SGQS replay without conflating it with COCO matching."""

from __future__ import annotations

from collections import Counter, OrderedDict
import json
import os
from pathlib import Path
import statistics


ROOT = Path(os.environ.get("SGQS_REPLAY_ROOT", Path.cwd())).resolve()
ANNOTATION_FILE = Path(
    os.environ.get(
        "SGQS_REPLAY_ANNOTATION",
        ROOT / "probe_fixed128_valonly" / "val.json",
    )
).resolve()
OUT = Path(os.environ.get("SGQS_REPLAY_OUTPUT", ROOT)).resolve()
EPOCHS = tuple(range(28, 35))
STAGES = OrderedDict(
    [
        ("dense_all", "encoder_dense_best_iou_all"),
        ("dense_top640", "encoder_dense_best_iou_top640"),
        ("decoder_all", "decoder_best_iou_all"),
        ("decoder_top300", "decoder_best_iou_top300"),
    ]
)


def load_rows(epoch: int) -> dict[tuple[int, int], dict]:
    path = ROOT / f"replay_val128_e00{epoch}" / "val.ema.gt.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["image_id"], row["annotation_id"])
            if key in rows:
                raise ValueError(f"duplicate row: {key}")
            rows[key] = row
    return rows


def load_metrics(epoch: int) -> dict[str, float]:
    path = ROOT / f"replay_val128_e00{epoch}" / "summary.json"
    result = json.loads(path.read_text(encoding="utf-8"))["results"]["val/ema"]
    s300 = result["coco_maxdets300"]
    s640 = result["coco_maxdets640"]
    return {
        "aps300": s300[3],
        "ars300": s300[9],
        "aps640": s640[3],
        "ars640": s640[9],
    }


def input_geometry() -> tuple[
    dict[tuple[int, int], dict[str, float | bool | str]], dict[int, str]
]:
    data = json.loads(ANNOTATION_FILE.read_text(encoding="utf-8"))
    images = {image["id"]: image for image in data["images"]}
    noncrowd_per_image = Counter(
        ann["image_id"] for ann in data["annotations"] if not ann.get("iscrowd", 0)
    )
    density = {
        image_id: "le100" if count <= 100 else "101_300" if count <= 300 else "gt300"
        for image_id, count in noncrowd_per_image.items()
    }
    geometry = {}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image = images[ann["image_id"]]
        input_w = ann["bbox"][2] / image["width"] * 640.0
        input_h = ann["bbox"][3] / image["height"] * 640.0
        area_small = input_w * input_h < 32.0**2
        short_small = min(input_w, input_h) < 16.0
        if area_small and short_small:
            reason = "both"
        elif area_small:
            reason = "area_only"
        elif short_small:
            reason = "short_only"
        else:
            reason = "neither"
        geometry[(ann["image_id"], ann["id"])] = {
            "input_area": input_w * input_h,
            "input_short": min(input_w, input_h),
            "area_small": area_small,
            "short_small": short_small,
            "hybrid_small": area_small or short_small,
            "hybrid_reason": reason,
        }
    return geometry, density


def final_failure(row: dict, threshold: float) -> str:
    if row["decoder_best_iou_all"] >= threshold:
        return "final_rank" if row["decoder_best_iou_top300"] < threshold else "covered"
    if row["encoder_dense_best_iou_top640"] >= threshold:
        return "decoder_geometry"
    if row["encoder_dense_best_iou_all"] >= threshold:
        return "encoder_top640"
    return "dense_geometry"


def pct(n: int, d: int) -> str:
    return f"{100.0*n/d:.2f}%" if d else "n/a"


def main() -> None:
    rows = {epoch: load_rows(epoch) for epoch in EPOCHS}
    keys = set(rows[EPOCHS[0]])
    for epoch in EPOCHS[1:]:
        if set(rows[epoch]) != keys:
            raise ValueError(f"annotation keys differ at epoch {epoch}")

    geometry, density = input_geometry()
    if not keys <= geometry.keys():
        raise ValueError("replay contains annotations absent from frozen JSON")
    for key in keys:
        if abs(rows[28][key]["input_short_px"] - geometry[key]["input_short"]) > 1e-3:
            raise ValueError(f"input geometry mismatch: {key}")

    cohorts = {
        "coco_small": {key for key in keys if rows[28][key]["coco_area_group"] == "small"},
        "hybrid_small": {key for key in keys if geometry[key]["hybrid_small"]},
        "hybrid_area_only": {key for key in keys if geometry[key]["hybrid_reason"] == "area_only"},
        "hybrid_short_only": {key for key in keys if geometry[key]["hybrid_reason"] == "short_only"},
        "hybrid_both": {key for key in keys if geometry[key]["hybrid_reason"] == "both"},
    }

    result: dict = {
        "note": "Per-GT independent best-IoU coverage; not one-to-one COCO TP/FN matching.",
        "metrics": {str(epoch): load_metrics(epoch) for epoch in EPOCHS},
        "image_density_counts": dict(Counter(density.values())),
        "cohort_counts": {name: len(members) for name, members in cohorts.items()},
        "cohort_density_counts": {
            name: dict(Counter(density[key[0]] for key in members))
            for name, members in cohorts.items()
        },
        "coverage": {},
        "rank_summary": {},
        "transitions": {},
    }

    for cohort_name, members in cohorts.items():
        result["coverage"][cohort_name] = {}
        result["rank_summary"][cohort_name] = {}
        result["transitions"][cohort_name] = {}
        for threshold in (0.5, 0.75):
            threshold_name = str(threshold)
            result["coverage"][cohort_name][threshold_name] = {}
            result["rank_summary"][cohort_name][threshold_name] = {}
            for epoch in EPOCHS:
                result["coverage"][cohort_name][threshold_name][str(epoch)] = {
                    stage: sum(rows[epoch][key][field] >= threshold for key in members)
                    for stage, field in STAGES.items()
                }
                rank_field = f"decoder_first_rank_iou{threshold}"
                ranks = sorted(rows[epoch][key][rank_field] for key in members)
                result["rank_summary"][cohort_name][threshold_name][str(epoch)] = {
                    "median": statistics.median(ranks),
                    "q1": ranks[(len(ranks) - 1) // 4],
                    "q3": ranks[(3 * (len(ranks) - 1)) // 4],
                    "le100": sum(rank <= 100 for rank in ranks),
                    "le300": sum(rank <= 300 for rank in ranks),
                    "le640": sum(rank <= 640 for rank in ranks),
                }
            transition_results = {}
            for before, after in ((28, 29), (28, 30), (30, 34)):
                before_cov = {
                    key for key in members if rows[before][key]["decoder_best_iou_top300"] >= threshold
                }
                after_cov = {
                    key for key in members if rows[after][key]["decoder_best_iou_top300"] >= threshold
                }
                gained = after_cov - before_cov
                lost = before_cov - after_cov
                transition_results[f"{before}_to_{after}"] = {
                    "stable_covered": len(before_cov & after_cov),
                    "stable_uncovered": len(members - (before_cov | after_cov)),
                    "gained": len(gained),
                    "lost": len(lost),
                    "net": len(after_cov) - len(before_cov),
                    "gained_prior_failure": dict(
                        Counter(final_failure(rows[before][key], threshold) for key in gained)
                    ),
                    "lost_later_failure": dict(
                        Counter(final_failure(rows[after][key], threshold) for key in lost)
                    ),
                    "gained_density": dict(Counter(density[key[0]] for key in gained)),
                    "lost_density": dict(Counter(density[key[0]] for key in lost)),
                    "gained_images": dict(Counter(str(key[0]) for key in gained).most_common()),
                    "lost_images": dict(Counter(str(key[0]) for key in lost).most_common()),
                    "gained_rank_before_median": statistics.median(
                        rows[before][key][f"decoder_first_rank_iou{threshold}"] for key in gained
                    ) if gained else None,
                    "gained_rank_after_median": statistics.median(
                        rows[after][key][f"decoder_first_rank_iou{threshold}"] for key in gained
                    ) if gained else None,
                    "lost_rank_before_median": statistics.median(
                        rows[before][key][f"decoder_first_rank_iou{threshold}"] for key in lost
                    ) if lost else None,
                    "lost_rank_after_median": statistics.median(
                        rows[after][key][f"decoder_first_rank_iou{threshold}"] for key in lost
                    ) if lost else None,
                }
            result["transitions"][cohort_name][threshold_name] = transition_results

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "val128_replay_analysis.json"
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Fixed-128 SGQS replay audit",
        "",
        "> Per-GT independent best-IoU coverage only; it is not one-to-one COCO TP/FN matching.",
        "",
        "## EMA COCO subset metrics",
        "",
        "| epoch | APs@300 | ARs@300 | APs@640 | ARs@640 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for epoch in EPOCHS:
        metric = result["metrics"][str(epoch)]
        lines.append(
            f"| {epoch} | {metric['aps300']:.6f} | {metric['ars300']:.6f} | "
            f"{metric['aps640']:.6f} | {metric['ars640']:.6f} |"
        )
    lines.extend(["", "## Cohort sizes", ""])
    for name, count in result["cohort_counts"].items():
        lines.append(f"- {name}: {count:,}")

    for cohort_name in ("coco_small", "hybrid_small"):
        size = len(cohorts[cohort_name])
        lines.extend(["", f"## {cohort_name} coverage", ""])
        for threshold in (0.5, 0.75):
            threshold_name = str(threshold)
            lines.extend(
                [
                    f"### IoU >= {threshold}",
                    "",
                    "| epoch | dense all | dense Top-640 | decoder all | decoder Top-300 |",
                    "|---:|---:|---:|---:|---:|",
                ]
            )
            for epoch in EPOCHS:
                coverage = result["coverage"][cohort_name][threshold_name][str(epoch)]
                cells = [f"{coverage[stage]} ({pct(coverage[stage], size)})" for stage in STAGES]
                lines.append(f"| {epoch} | " + " | ".join(cells) + " |")
            lines.extend(["", "Transitions at decoder Top-300:", ""])
            for label, trans in result["transitions"][cohort_name][threshold_name].items():
                lines.append(
                    f"- {label}: gained={trans['gained']}, lost={trans['lost']}, net={trans['net']}; "
                    f"gain source={trans['gained_prior_failure']}; loss destination={trans['lost_later_failure']}; "
                    f"gain rank median {trans['gained_rank_before_median']}->{trans['gained_rank_after_median']}; "
                    f"loss rank median {trans['lost_rank_before_median']}->{trans['lost_rank_after_median']}; "
                    f"gain density={trans['gained_density']}; loss density={trans['lost_density']}"
                )

    out_md = OUT / "val128_replay_analysis.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
