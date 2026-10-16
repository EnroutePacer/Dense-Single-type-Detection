"""Build the SGQS training-loss hypothesis audit from registered evidence.

The script does not run inference or training.  It joins the completed SGQS/B0
epoch logs, their sampled eight-rank probe summaries, the registered full-val
query-probe tensor, and the Baseline_test contact sheets.  Outputs are
deterministic CSV tables and publication figures under EXP-DIAG-007.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
FULL_VAL_ROOT = (
    REPOSITORY_ROOT
    / "article"
    / "experiments"
    / "EXP-DIAG-006_sgqs-full-val-checkpoint-replay"
)
RUNS = {
    "sgqs": REPOSITORY_ROOT / "outputs" / "probes" / "sgqs_e28_r7_seed0",
    "b0": REPOSITORY_ROOT / "outputs" / "probes" / "b0_e28_r7_seed0",
}
COCO_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar_1",
    "ar_10",
    "ar_maxdet",
    "ar_small",
    "ar_medium",
    "ar_large",
)
SGQS_KEYS = (
    "loss_sgqs_dense",
    "loss_sgqs_bbox",
    "loss_sgqs_giou",
    "loss_sgqs_survival",
)
SELECTED_PROBE_METRICS = {
    "encoder_selected_best_iou_all",
    "encoder_selected_best_iou_top300",
    "decoder_best_iou_all",
    "decoder_best_iou_top300",
    "decoder_first_rank_iou0.75",
    "go_missing",
    "go_query_count",
    "owner_assigned",
    "owner_kept",
    "owner_selected_raw",
    "owner_iou_raw",
    "owner_rank_raw",
    "owner_p3_kept",
    "owner_p4_kept",
    "owner_p5_kept",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_epoch_logs() -> dict[str, dict[int, dict]]:
    result: dict[str, dict[int, dict]] = {}
    for arm, run in RUNS.items():
        rows = [json.loads(line) for line in (run / "log.txt").read_text(encoding="utf-8").splitlines()]
        epochs = {int(row["epoch"]): row for row in rows}
        if sorted(epochs) != list(range(28, 35)):
            raise ValueError(f"{arm} does not contain exactly epochs 28--34")
        result[arm] = epochs
    return result


def _sum_loss_keys(row: dict, keys: tuple[str, ...]) -> float:
    return sum(float(row.get(f"train_{key}", 0.0)) for key in keys)


def build_epoch_trajectory(logs: dict[str, dict[int, dict]]) -> list[dict]:
    rows: list[dict] = []
    for arm in ("sgqs", "b0"):
        for epoch in range(28, 35):
            source = logs[arm][epoch]
            sgqs_total = _sum_loss_keys(source, SGQS_KEYS)
            loss_total = float(source["train_loss"])
            record = {
                "arm": arm,
                "epoch": epoch,
                "train_loss_total": loss_total,
                "train_loss_sgqs_total": sgqs_total,
                "train_loss_base_remainder": loss_total - sgqs_total,
                "sgqs_fraction_of_total": sgqs_total / loss_total if loss_total else 0.0,
                "train_loss_base_decoder_main": _sum_loss_keys(
                    source, ("loss_vfl", "loss_bbox", "loss_giou", "loss_fgl")
                ),
                "train_loss_base_encoder": _sum_loss_keys(
                    source, ("loss_vfl_enc_0", "loss_bbox_enc_0", "loss_giou_enc_0")
                ),
                "train_loss_base_pre": _sum_loss_keys(
                    source, ("loss_vfl_pre", "loss_bbox_pre", "loss_giou_pre")
                ),
                "train_loss_ddf_all": sum(
                    float(value)
                    for key, value in source.items()
                    if key.startswith("train_loss_ddf")
                ),
            }
            for key in SGQS_KEYS:
                record[f"train_{key}"] = float(source.get(f"train_{key}", 0.0))
            for name, value in zip(COCO_NAMES, source["test_coco_eval_bbox"], strict=True):
                record[name] = float(value)
            for name in (
                "sgqs_owner_coverage",
                "sgqs_owner_count",
                "sgqs_owner_iou",
                "sgqs_pre_topk_survival_rate",
                "sgqs_owner_cutoff_gap",
                "sgqs_survival_ramp",
                "sgqs_conflicting_owner_fraction",
                "sgqs_owner_p3_count",
                "sgqs_owner_p4_count",
                "sgqs_owner_p5_count",
            ):
                record[name] = source.get(f"train_{name}", "")
            p3 = record["sgqs_owner_p3_count"]
            p5 = record["sgqs_owner_p5_count"]
            record["implicit_p5_to_p3_per_positive_normalizer_ratio"] = (
                float(p3) / float(p5) if p3 != "" and p5 not in ("", 0.0) else ""
            )
            rows.append(record)
    return rows


def read_probe_loss_steps(path: Path, maximum_step: int = 500) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) != 28 or int(row["step"]) > maximum_step:
                continue
            prefix = "losses_weighted_local/"
            if row["metric"].startswith(prefix):
                by_step[int(row["step"])][row["metric"][len(prefix) :]] = float(row["mean"])
    return dict(by_step)


def build_early_steps() -> list[dict]:
    source = {
        arm: read_probe_loss_steps(run / "probe_steps.csv") for arm, run in RUNS.items()
    }
    rows: list[dict] = []
    for step in sorted(set(source["sgqs"]).intersection(source["b0"])):
        sgqs = source["sgqs"][step]
        b0 = source["b0"][step]
        sgqs_branch = sum(float(sgqs.get(key, 0.0)) for key in SGQS_KEYS)
        sgqs_total = sum(sgqs.values())
        b0_total = sum(b0.values())
        rows.append(
            {
                "epoch": 28,
                "step": step,
                "sgqs_total_weighted_loss": sgqs_total,
                "b0_total_weighted_loss": b0_total,
                "sgqs_branch_total": sgqs_branch,
                "sgqs_dense": sgqs.get("loss_sgqs_dense", 0.0),
                "sgqs_bbox": sgqs.get("loss_sgqs_bbox", 0.0),
                "sgqs_giou": sgqs.get("loss_sgqs_giou", 0.0),
                "sgqs_survival": sgqs.get("loss_sgqs_survival", 0.0),
                "sgqs_base_remainder": sgqs_total - sgqs_branch,
                "total_delta_sgqs_minus_b0": sgqs_total - b0_total,
                "base_remainder_delta_sgqs_minus_b0": sgqs_total - sgqs_branch - b0_total,
            }
        )
    if not rows or rows[0]["step"] != 0:
        raise ValueError("paired epoch-28 step zero is missing")
    return rows


def build_sampled_probe_summary() -> list[dict]:
    aggregate: dict[tuple[str, int, str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0, "sampled_steps": 0.0}
    )
    for arm, run in RUNS.items():
        with (run / "probe_steps.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                parts = row["metric"].split("/")
                if len(parts) != 4 or parts[0] != "groups" or parts[2] != "metrics":
                    continue
                group, metric = parts[1], parts[3]
                if metric not in SELECTED_PROBE_METRICS:
                    continue
                key = (arm, int(row["epoch"]), group, metric)
                target = aggregate[key]
                target["count"] += float(row["count"])
                target["sum"] += float(row["sum"])
                target["sum_sq"] += float(row["sum_sq"])
                target["sampled_steps"] += 1.0
    rows = []
    for (arm, epoch, group, metric), values in sorted(aggregate.items()):
        count = values["count"]
        rows.append(
            {
                "arm": arm,
                "epoch": epoch,
                "input_area_group": group,
                "metric": metric,
                "sampled_steps": int(values["sampled_steps"]),
                "sampled_count": int(count),
                "sum": values["sum"],
                "mean": values["sum"] / count if count else "",
                "sampling_scope": (
                    "sampled training batches; e28 first 200 steps are oversampled; not an unbiased epoch mean"
                ),
            }
        )
    return rows


def build_owner_level_group_summary(sampled_rows: list[dict]) -> list[dict]:
    source: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for row in sampled_rows:
        if row["arm"] != "sgqs" or row["metric"] not in {
            "owner_p3_kept",
            "owner_p4_kept",
            "owner_p5_kept",
        }:
            continue
        source[(int(row["epoch"]), row["input_area_group"])][row["metric"]] = float(row["sum"])
    rows: list[dict] = []
    for epoch in range(28, 35):
        p5_total = sum(source[(epoch, group)].get("owner_p5_kept", 0.0) for group in (
            "small_input",
            "medium_input",
            "large_input",
        ))
        for group in ("small_input", "medium_input", "large_input"):
            values = source[(epoch, group)]
            p3 = values.get("owner_p3_kept", 0.0)
            p4 = values.get("owner_p4_kept", 0.0)
            p5 = values.get("owner_p5_kept", 0.0)
            rows.append(
                {
                    "epoch": epoch,
                    "input_area_group": group,
                    "sampled_p3_kept": int(p3),
                    "sampled_p4_kept": int(p4),
                    "sampled_p5_kept": int(p5),
                    "p5_group_share_within_epoch": p5 / p5_total if p5_total else "",
                    "sampling_scope": (
                        "sampled augmented training batches; e28 first 200 steps are oversampled; "
                        "area group is not COCO evaluation size"
                    ),
                }
            )
    totals = {
        group: sum(
            source[(epoch, group)].get("owner_p5_kept", 0.0)
            for epoch in range(28, 35)
        )
        for group in ("small_input", "medium_input", "large_input")
    }
    grand_total = sum(totals.values())
    for group in ("small_input", "medium_input", "large_input"):
        rows.append(
            {
                "epoch": "all_sampled",
                "input_area_group": group,
                "sampled_p3_kept": "",
                "sampled_p4_kept": "",
                "sampled_p5_kept": int(totals[group]),
                "p5_group_share_within_epoch": totals[group] / grand_total,
                "sampling_scope": (
                    "pooled sampled augmented training batches; e28 first 200 steps are oversampled; "
                    "descriptive association only"
                ),
            }
        )
    return rows


def build_full_val_mechanism_summaries() -> tuple[list[dict], list[dict]]:
    metrics_path = FULL_VAL_ROOT / "data" / "metrics_14_states.csv"
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        metrics = [row for row in csv.DictReader(handle) if row["state"] == "ema"]
    budget_rows = [
        {
            "epoch": int(row["epoch"]),
            "ap_small_d300": float(row["ap_small_d300"]),
            "ap_small_d640": float(row["ap_small_d640"]),
            "ar_small_d300": float(row["ar_small_d300"]),
            "ar_small_d640": float(row["ar_small_d640"]),
            "scope": "local full validation; formal evaluator-small; EMA",
        }
        for row in sorted(metrics, key=lambda value: int(value["epoch"]))
    ]

    transition_path = FULL_VAL_ROOT / "data" / "transition_attribution.csv"
    with transition_path.open(encoding="utf-8", newline="") as handle:
        transitions = list(csv.DictReader(handle))
    causes = (
        "candidate_absence",
        "encoder_selection",
        "decoder_geometry",
        "rank_gt300",
        "one_to_one_competition",
    )
    transition_rows: list[dict] = []
    for start, end, direction in ((28, 29, "gain"), (28, 30, "gain"), (30, 34, "loss")):
        selected = [
            row
            for row in transitions
            if row["state"] == "ema"
            and row["match_scope"] == "coco_small_eval"
            and row["cohort"] == "coco_small_eval"
            and int(row["epoch_start"]) == start
            and int(row["epoch_end"]) == end
            and row["transition"] == direction
            and int(row["max_dets"]) == 300
        ]
        if len(selected) != 10 * len(causes):
            raise ValueError(f"unexpected transition slice size for e{start}->e{end} {direction}")
        totals = {
            cause: sum(int(row["count"]) for row in selected if row["cause"] == cause)
            for cause in causes
        }
        total = sum(totals.values())
        transition_rows.append(
            {
                "transition": f"e{start}_to_e{end}_{direction}",
                **totals,
                "ten_iou_transition_events": total,
                "rank_gt300_fraction": totals["rank_gt300"] / total,
                "unit": "GT x IoU transition events; IoU=.50:.05:.95; not unique GT or additive AP",
            }
        )
    return budget_rows, transition_rows


def box_iou_xywh(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_xyxy = np.concatenate((left[:, :2], left[:, :2] + left[:, 2:]), axis=1)
    right_xyxy = np.concatenate((right[:, :2], right[:, :2] + right[:, 2:]), axis=1)
    upper_left = np.maximum(left_xyxy[:, None, :2], right_xyxy[None, :, :2])
    lower_right = np.minimum(left_xyxy[:, None, 2:], right_xyxy[None, :, 2:])
    width_height = np.maximum(lower_right - upper_left, 0.0)
    intersection = width_height[..., 0] * width_height[..., 1]
    left_area = np.prod(np.maximum(left_xyxy[:, 2:] - left_xyxy[:, :2], 0.0), axis=1)
    right_area = np.prod(np.maximum(right_xyxy[:, 2:] - right_xyxy[:, :2], 0.0), axis=1)
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / np.maximum(union, 1e-12)


def build_query_forensics(npz_path: Path) -> tuple[list[dict], list[dict]]:
    payload = np.load(npz_path, allow_pickle=False)
    if str(payload["schema"]) != "sgqs_checkpoint_query_probe_data_v1":
        raise ValueError("unexpected query-probe schema")
    rows: list[dict] = []
    for image_index, image_id in enumerate(payload["image_ids"]):
        for epoch_index, epoch in enumerate(payload["epochs"]):
            scores = payload["decoder_scores"][image_index, epoch_index]
            order = np.argsort(-scores, kind="stable")
            sorted_scores = scores[order]
            boxes = payload["decoder_boxes_xywh"][image_index, epoch_index, order]
            tail_overlap = box_iou_xywh(boxes[300:], boxes[:300])
            levels = payload["encoder_levels"][image_index, epoch_index]
            hybrid = payload["decoder_hybrid_small"][image_index, epoch_index]
            rows.append(
                {
                    "image_id": int(image_id),
                    "epoch": int(epoch),
                    "encoder_p3_count": int(np.count_nonzero(levels == 0)),
                    "encoder_p4_count": int(np.count_nonzero(levels == 1)),
                    "encoder_p5_count": int(np.count_nonzero(levels == 2)),
                    "decoder_query_count": int(len(scores)),
                    "decoder_hybrid_small_count": int(np.count_nonzero(hybrid)),
                    "approx_unique_integer_rounded_xywh": int(
                        np.unique(np.rint(boxes).astype(np.int64), axis=0).shape[0]
                    ),
                    "tail_301_640_overlap_top300_iou_gt_0_9": int(
                        np.count_nonzero(np.any(tail_overlap > 0.9, axis=1))
                    ),
                    "score_gt_0_5_count": int(np.count_nonzero(sorted_scores > 0.5)),
                    "rank300_score": float(sorted_scores[299]),
                    "rank301_score": float(sorted_scores[300]),
                    "rank500_score": float(sorted_scores[499]),
                    "rank640_score": float(sorted_scores[639]),
                    "semantics": (
                        "decoder always has 640 queries; approximate unique rounds xywh to integer pixels; "
                        "overlap is descriptive and not NMS/evaluator output"
                    ),
                }
            )
    change_rows: list[dict] = []
    by_key = {(row["image_id"], row["epoch"]): row for row in rows}
    image_ids = sorted({row["image_id"] for row in rows})
    for end in (29, 30, 34):
        unique_decrease = 0
        overlap_increase = 0
        score_half_decrease = 0
        for image_id in image_ids:
            start_row = by_key[(image_id, 28)]
            end_row = by_key[(image_id, end)]
            unique_decrease += (
                end_row["approx_unique_integer_rounded_xywh"]
                < start_row["approx_unique_integer_rounded_xywh"]
            )
            overlap_increase += (
                end_row["tail_301_640_overlap_top300_iou_gt_0_9"]
                > start_row["tail_301_640_overlap_top300_iou_gt_0_9"]
            )
            score_half_decrease += end_row["score_gt_0_5_count"] < start_row["score_gt_0_5_count"]
        change_rows.append(
            {
                "transition": f"e28_to_e{end}",
                "selected_probe_images": len(image_ids),
                "images_with_approx_unique_decrease": unique_decrease,
                "images_with_tail_overlap_increase": overlap_increase,
                "images_with_score_gt_0_5_count_decrease": score_half_decrease,
                "selection_warning": (
                    "29 transition-selected probes; qualitative mechanism set, not a prevalence sample"
                ),
            }
        )
    return rows, change_rows


def plot_epoch_trajectory(rows: list[dict], output: Path) -> None:
    by_arm = {
        arm: {int(row["epoch"]): row for row in rows if row["arm"] == arm}
        for arm in ("sgqs", "b0")
    }
    epochs = np.arange(28, 35)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    axis = axes[0, 0]
    for arm, color in (("sgqs", "#d1495b"), ("b0", "#277da1")):
        axis.plot(epochs, [by_arm[arm][e]["ap_small"] for e in epochs], "o-", lw=2, label=arm.upper(), color=color)
    axis.axvline(30, color="#666666", ls="--", lw=1)
    axis.set(title="EMA evaluator AP-small", xlabel="epoch", ylabel="AP-small")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[0, 1]
    for metric, label, color in (
        ("ap", "AP", "#4d908e"),
        ("ap_small", "AP-small", "#d1495b"),
        ("ap_large", "AP-large", "#577590"),
    ):
        delta = [(by_arm["sgqs"][e][metric] - by_arm["b0"][e][metric]) * 100 for e in epochs]
        axis.plot(epochs, delta, "o-", lw=2, label=label, color=color)
    axis.axhline(0, color="black", lw=1)
    axis.set(title="SGQS minus B0 (percentage points)", xlabel="epoch", ylabel="delta")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[1, 0]
    sgqs_rows = by_arm["sgqs"]
    for metric, label, color in (
        ("train_loss_sgqs_total", "SGQS branch", "#f94144"),
        ("train_loss_sgqs_dense", "dense", "#f8961e"),
        ("train_loss_base_encoder", "base Encoder", "#277da1"),
    ):
        axis.plot(epochs, [sgqs_rows[e][metric] for e in epochs], "o-", lw=2, label=label, color=color)
    axis.set(title="Whole-epoch weighted loss means", xlabel="epoch", ylabel="loss")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)

    axis = axes[1, 1]
    for metric, label, color in (
        ("sgqs_owner_p3_count", "P3 owners/image", "#2fbf71"),
        ("sgqs_owner_p4_count", "P4 owners/image", "#4d7cff"),
        ("sgqs_owner_p5_count", "P5 owners/image", "#ffb000"),
    ):
        axis.plot(epochs, [sgqs_rows[e][metric] for e in epochs], "o-", lw=2, label=label, color=color)
    axis.set_yscale("log")
    axis.set(title="Kept owners and level imbalance", xlabel="epoch", ylabel="owners/image (log)")
    ratio_axis = axis.twinx()
    ratio_axis.plot(
        epochs,
        [sgqs_rows[e]["implicit_p5_to_p3_per_positive_normalizer_ratio"] for e in epochs],
        "s--",
        color="#7b2cbf",
        label="P5:P3 per-positive normalizer",
    )
    ratio_axis.set_ylabel("P3 count / P5 count")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = ratio_axis.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
    axis.grid(alpha=0.25)

    fig.suptitle("SGQS loss-hypothesis audit: paired seed-0 continuation", fontsize=15, weight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_early_steps(rows: list[dict], output: Path) -> None:
    steps = np.array([row["step"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True, constrained_layout=True)
    axes[0].plot(steps, [row["sgqs_total_weighted_loss"] for row in rows], "o-", label="SGQS total", color="#d1495b")
    axes[0].plot(steps, [row["b0_total_weighted_loss"] for row in rows], "o-", label="B0 total", color="#277da1")
    axes[0].plot(steps, [row["sgqs_base_remainder"] for row in rows], "o--", label="SGQS base remainder", color="#4d908e")
    axes[0].set(ylabel="weighted loss", title="Resume shock: paired sampled training steps")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.25)
    first = rows[0]
    axes[0].annotate(
        f"step 0: +{first['total_delta_sgqs_minus_b0']:.3f}",
        xy=(0, first["sgqs_total_weighted_loss"]),
        xytext=(55, first["sgqs_total_weighted_loss"] + 0.7),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )

    for key, label, color in (
        ("sgqs_dense", "dense", "#f8961e"),
        ("sgqs_bbox", "owner L1", "#43aa8b"),
        ("sgqs_giou", "owner GIoU", "#577590"),
        ("sgqs_survival", "survival", "#7b2cbf"),
    ):
        axes[1].plot(steps, [row[key] for row in rows], "o-", label=label, color=color)
    axes[1].set(xlabel="epoch-28 optimizer step", ylabel="weighted loss", title="Added SGQS components")
    axes[1].legend(frameon=False, ncol=4)
    axes[1].grid(alpha=0.25)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_image5367(rows: list[dict], output: Path) -> None:
    selected = sorted((row for row in rows if row["image_id"] == 5367), key=lambda row: row["epoch"])
    epochs = np.array([row["epoch"] for row in selected])
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2), constrained_layout=True)
    axes[0, 0].stackplot(
        epochs,
        [row["encoder_p3_count"] for row in selected],
        [row["encoder_p4_count"] for row in selected],
        [row["encoder_p5_count"] for row in selected],
        labels=("P3", "P4", "P5"),
        colors=("#2fbf71", "#4d7cff", "#ffb000"),
        alpha=0.85,
    )
    axes[0, 0].set(title="Encoder Top-640 source levels", ylabel="query count")
    axes[0, 0].legend(frameon=False, ncol=3)

    for key, label, color in (
        ("decoder_hybrid_small_count", "hybrid-small", "#f94144"),
        ("approx_unique_integer_rounded_xywh", "approx. unique", "#577590"),
        ("score_gt_0_5_count", "score > 0.5", "#43aa8b"),
    ):
        axes[0, 1].plot(epochs, [row[key] for row in selected], "o-", lw=2, label=label, color=color)
    axes[0, 1].set(title="640-query descriptive counts", ylabel="count")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(epochs, [row["rank300_score"] for row in selected], "o-", lw=2, label="rank 300", color="#277da1")
    axes[1, 0].plot(epochs, [row["rank500_score"] for row in selected], "o-", lw=2, label="rank 500", color="#d1495b")
    axes[1, 0].set(title="Score boundary", xlabel="epoch", ylabel="score", ylim=(-0.02, 1.03))
    axes[1, 0].legend(frameon=False)

    axes[1, 1].bar(
        epochs,
        [row["tail_301_640_overlap_top300_iou_gt_0_9"] for row in selected],
        color="#f8961e",
    )
    axes[1, 1].set(title="Tail queries overlapping Top-300 (IoU > .9)", xlabel="epoch", ylabel="tail-query count")
    for axis in axes.flat:
        axis.set_xticks(epochs)
        axis.grid(alpha=0.22)
    fig.suptitle("Image 5367: apparent missing boxes are an e29 tail-score/duplication event", fontsize=14, weight="bold")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def copy_registered_figures_and_baseline(baseline_dir: Path) -> list[Path]:
    required = [
        baseline_dir / "README.md",
        baseline_dir / "baseline_topk_summary.csv",
        baseline_dir / "aggregate_trajectory.png",
        *sorted(baseline_dir.glob("*_e28_e34.png")),
    ]
    if len(required) != 16 or any(not path.is_file() for path in required):
        raise ValueError("Baseline_test bundle must contain README, CSV, aggregate plot, and 13 contact sheets")
    destination = EXPERIMENT_ROOT / "figures" / "baseline_test"
    destination.mkdir(parents=True, exist_ok=True)
    for source in required:
        target = destination / source.name
        if source.suffix.lower() in {".csv", ".md"}:
            target.write_text(source.read_text(encoding="utf-8-sig"), encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(source, target)
    (EXPERIMENT_ROOT / "data" / "baseline_topk_summary.csv").write_text(
        (baseline_dir / "baseline_topk_summary.csv").read_text(encoding="utf-8-sig"),
        encoding="utf-8",
        newline="\n",
    )

    registered = [
        FULL_VAL_ROOT / "figures" / "query_probes" / "query_probe_image_5367_ema_e28_e34.png",
        FULL_VAL_ROOT / "figures" / "representative_ema_e30_e34_loss_iou75.png",
    ]
    instance_destination = EXPERIMENT_ROOT / "figures" / "image5367"
    instance_destination.mkdir(parents=True, exist_ok=True)
    for source in registered:
        shutil.copyfile(source, instance_destination / source.name)
    return required + registered


def build_input_manifest(baseline_and_registered: list[Path]) -> list[dict]:
    inputs = [
        RUNS["sgqs"] / "log.txt",
        RUNS["sgqs"] / "probe_steps.csv",
        RUNS["b0"] / "log.txt",
        RUNS["b0"] / "probe_steps.csv",
        REPOSITORY_ROOT / "engine" / "sgqs" / "criterion.py",
        REPOSITORY_ROOT / "configs" / "sgqs" / "sgqs_hgnetv2_m_table_cell_fixed640.yml",
        REPOSITORY_ROOT / "configs" / "sgqs_e" / "b1_sgqs.yml",
        FULL_VAL_ROOT / "figures" / "query_probes" / "query_probe_data.npz",
        FULL_VAL_ROOT / "data" / "metrics_14_states.csv",
        FULL_VAL_ROOT / "data" / "transition_attribution.csv",
        FULL_VAL_ROOT / "data" / "stage_coverage.csv",
        FULL_VAL_ROOT / "data" / "match_gt_buckets.csv",
        *baseline_and_registered,
    ]
    seen: set[Path] = set()
    rows: list[dict] = []
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            logical_name = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
            if "outputs" in path.parts:
                role = "canonical run evidence"
            elif "EXP-DIAG-006" in str(path):
                role = "registered full-val evidence"
            else:
                role = "registered method implementation/config"
        except ValueError:
            logical_name = f"Baseline_test/{path.name}"
            role = "Baseline_test qualitative evidence"
        rows.append(
            {
                "logical_name": logical_name,
                "role": role,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def build_verdict_rows() -> list[dict]:
    return [
        {
            "hypothesis": "H1_aggressive_encoder_loss_causes_near_overfit_and_overall_erosion",
            "verdict": "partially_supported_but_overfit_claim_not_established",
            "supported": "step-0 SGQS adds 5.0711 weighted loss, including 4.7029 dense; no dense warm-up; rapid first-two-epoch ranking/source changes",
            "contradicted_or_limited": "final overall AP is 0.000915 above B0; APs@640 improves late; no paired train-set AP trajectory; loss magnitude is not gradient norm",
            "required_causal_test": "dense/owner warm-up or temporary SGQS-off branch with identical data order and optimizer state; report shared-parameter gradients and train-vs-val AP",
        },
        {
            "hypothesis": "H2_missing_end_to_end_follow_through_then_base_loss_recovers",
            "verdict": "proxy_mismatch_supported_specific_loss_rebound_story_rejected",
            "supported": "SGQS directly ends at pre-TopK/Encoder while full-val failure is final Decoder Top-300 ranking; owner coverage/survival improve as AP-small falls",
            "contradicted_or_limited": "base Encoder/main losses keep decreasing through e34; e31 has no demonstrated base-loss rebound; validation AP-large cannot trigger backpropagation",
            "required_causal_test": "add a single downstream owner/query identity or final Top-300 ranking objective, or stop SGQS after e30, while keeping every other setting fixed",
        },
        {
            "hypothesis": "H3_equal_P3_P4_P5_dense_normalization_overweights_rare_P5",
            "verdict": "mechanism_exists_causal_role_unproven",
            "supported": "dense positives and negatives are normalized per level then averaged equally; the ratio of epoch-mean P3/P5 kept-owner counts rises from about 70x to 135x; 98.9% of sampled kept P5 owners are in the augmented-training large-input group",
            "contradicted_or_limited": "no per-level dense loss or shared-parameter gradient was logged; sampled batches are not an unbiased epoch census and P5 is not identical to COCO large",
            "required_causal_test": "compare current equal-level mean with global owner-count normalization; log per-level positive/negative loss and clipped shared-parameter gradient norms",
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path(r"D:\Baseline_test\output\SGQS_e28_e34_baseline_topk_postprocess"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = EXPERIMENT_ROOT / "data"
    figures_dir = EXPERIMENT_ROOT / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logs = read_epoch_logs()
    epoch_rows = build_epoch_trajectory(logs)
    write_csv(data_dir / "hypothesis_epoch_trajectory.csv", list(epoch_rows[0]), epoch_rows)

    early_rows = build_early_steps()
    write_csv(data_dir / "early_resume_steps.csv", list(early_rows[0]), early_rows)

    sampled_rows = build_sampled_probe_summary()
    write_csv(data_dir / "probe_sampled_epoch_summary.csv", list(sampled_rows[0]), sampled_rows)
    owner_rows = build_owner_level_group_summary(sampled_rows)
    write_csv(data_dir / "owner_level_group_summary.csv", list(owner_rows[0]), owner_rows)

    budget_rows, transition_rows = build_full_val_mechanism_summaries()
    write_csv(data_dir / "full_val_small_budget_trajectory.csv", list(budget_rows[0]), budget_rows)
    write_csv(data_dir / "full_val_transition_summary.csv", list(transition_rows[0]), transition_rows)

    query_npz = FULL_VAL_ROOT / "figures" / "query_probes" / "query_probe_data.npz"
    query_rows, change_rows = build_query_forensics(query_npz)
    write_csv(data_dir / "query_probe_image_forensics.csv", list(query_rows[0]), query_rows)
    image5367 = [row for row in query_rows if row["image_id"] == 5367]
    write_csv(data_dir / "image5367_forensics.csv", list(image5367[0]), image5367)
    write_csv(data_dir / "query_probe_change_summary.csv", list(change_rows[0]), change_rows)

    verdict_rows = build_verdict_rows()
    write_csv(data_dir / "hypothesis_verdicts.csv", list(verdict_rows[0]), verdict_rows)

    copied_inputs = copy_registered_figures_and_baseline(args.baseline_dir)
    manifest_rows = build_input_manifest(copied_inputs)
    write_csv(data_dir / "input_manifest.csv", list(manifest_rows[0]), manifest_rows)

    plot_epoch_trajectory(epoch_rows, figures_dir / "hypothesis_trajectory.png")
    plot_early_steps(early_rows, figures_dir / "early_resume_shock.png")
    plot_image5367(query_rows, figures_dir / "image5367_forensics.png")

    print(
        json.dumps(
            {
                "epoch_rows": len(epoch_rows),
                "early_step_rows": len(early_rows),
                "sampled_probe_rows": len(sampled_rows),
                "query_probe_rows": len(query_rows),
                "baseline_contact_sheets": 13,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
