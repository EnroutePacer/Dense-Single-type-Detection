#!/usr/bin/env python3
"""Render deterministic static figures from EXP-DIAG-006 analysis tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = {
    "raw": "#6B7280",
    "ema": "#6F57D9",
    "encoder_dense_all": "#56B4E9",
    "encoder_top640": "#0072B2",
    "decoder_all640": "#009E73",
    "decoder_top300": "#D55E00",
    "candidate_absence": "#9CA3AF",
    "encoder_selection": "#56B4E9",
    "decoder_geometry": "#009E73",
    "rank_gt300": "#D55E00",
    "one_to_one_competition": "#CC79A7",
}

CAUSES = (
    "candidate_absence",
    "encoder_selection",
    "decoder_geometry",
    "rank_gt300",
    "one_to_one_competition",
)

FORMAL_EPOCHS = tuple(range(28, 35))
FORMAL_STATES = ("raw", "ema")
FORMAL_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
FORMAL_BUDGETS = (300, 640)
FORMAL_IMAGES = 12_841
FORMAL_NONCROWD_GT = 1_458_060
FORMAL_ANALYSIS_OUTPUTS = frozenset(
    {
        "metrics_14_states.csv",
        "coco_small_per_iou.csv",
        "coco_small_per_iou_budget_gap.csv",
        "raw_ema_comparison.csv",
        "stage_coverage.csv",
        "match_aggregates.csv",
        "match_gt_buckets.csv",
        "transition_summary.csv",
        "transition_attribution.csv",
        "representative_annotations.csv",
        "representative_images.csv",
        "global_pr_curve.csv",
        "tp_fp_score_separation.csv",
        "tp_fp_score_histogram.csv",
        "transition_details.csv.gz",
        "analysis_summary.md",
        "audit.json",
        "artifact_manifest.csv",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: object, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"formal analysis audit lacks {label}")
    return value


def _require_sha(value: object, label: str) -> str:
    normalized = str(value).lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"formal analysis audit lacks valid {label} SHA-256")
    return normalized


def _require_descriptor(value: object, label: str) -> Mapping:
    descriptor = _require_mapping(value, label)
    _require_sha(descriptor.get("sha256"), label)
    try:
        byte_count = int(descriptor.get("bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"formal analysis audit lacks valid {label} byte count") from exc
    if byte_count < 0:
        raise ValueError(f"formal analysis audit lacks valid {label} byte count")
    return descriptor


def _validate_formal_identity(audit: Mapping) -> None:
    identity = _require_mapping(audit.get("formal_identity"), "formal identity gate")
    _require_sha(identity.get("config_sha256"), "config")
    _require_sha(identity.get("annotation_sha256"), "annotation")
    inference = _require_mapping(identity.get("inference_identity"), "inference backend gate")
    if not isinstance(inference.get("device"), str) or not inference["device"]:
        raise ValueError("formal analysis audit lacks inference device identity")
    if not isinstance(inference.get("torch_version"), str) or not inference["torch_version"]:
        raise ValueError("formal analysis audit lacks inference torch identity")
    if "torch_cuda_version" not in inference:
        raise ValueError("formal analysis audit lacks inference CUDA identity")
    try:
        if int(inference.get("batch_size", 0)) < 1:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("formal analysis audit lacks inference batch-size identity") from exc

    inputs = _require_mapping(audit.get("input_identity"), "input identity gate")
    replay_inputs = _require_mapping(inputs.get("replays"), "replay hash gate")
    expected_replays = {f"e{epoch}" for epoch in FORMAL_EPOCHS}
    if set(replay_inputs) != expected_replays:
        raise ValueError("formal analysis audit lacks the complete replay hash gate")
    for replay_key in sorted(expected_replays):
        replay = _require_mapping(replay_inputs[replay_key], f"{replay_key} replay identity")
        _require_sha(replay.get("summary_sha256"), f"{replay_key} summary")
        _require_sha(replay.get("checkpoint_sha256"), f"{replay_key} checkpoint")
        _require_descriptor(replay.get("metadata"), f"{replay_key} metadata")
        state_inputs = _require_mapping(replay.get("states"), f"{replay_key} state hashes")
        if set(state_inputs) != set(FORMAL_STATES):
            raise ValueError(f"formal analysis audit lacks raw/EMA hashes for {replay_key}")
        for state in FORMAL_STATES:
            artifacts = _require_mapping(state_inputs[state], f"{replay_key}/{state} artifacts")
            for artifact_name in ("predictions", "gt_stages"):
                _require_descriptor(
                    artifacts.get(artifact_name),
                    f"{replay_key}/{state}/{artifact_name}",
                )

    matcher_inputs = _require_mapping(inputs.get("matchers"), "matcher hash gate")
    expected_states = {
        f"e{epoch}/{state}" for epoch in FORMAL_EPOCHS for state in FORMAL_STATES
    }
    if set(matcher_inputs) != expected_states:
        raise ValueError("formal analysis audit lacks the complete 14-state matcher hash gate")
    matcher_roots = (
        "complete.json",
        "protocol.json",
        "match_aggregates.csv",
        "detection_status_manifest.json",
        "gt_shard_manifest.json",
    )
    for state_key in sorted(expected_states):
        roots = _require_mapping(matcher_inputs[state_key], f"{state_key} matcher roots")
        for root_name in matcher_roots:
            _require_descriptor(roots.get(root_name), f"{state_key}/{root_name}")

    matrix = _require_descriptor(inputs.get("matcher_matrix"), "matcher matrix gate")
    replay_runtime = _require_mapping(
        matrix.get("replay_runtime_backend"), "replay runtime backend gate"
    )
    matcher_runtime = _require_mapping(matrix.get("matcher_runtime"), "matcher runtime backend gate")
    if replay_runtime.get("requested_device") != inference["device"]:
        raise ValueError("formal analysis replay backend device disagrees with inference identity")
    for key in ("torch_version", "torch_cuda_version"):
        if key not in replay_runtime or replay_runtime.get(key) != inference.get(key):
            raise ValueError(f"formal analysis replay backend {key} identity mismatch")
    for key in ("python_version", "numpy_version"):
        if not isinstance(matcher_runtime.get(key), str) or not matcher_runtime[key]:
            raise ValueError(f"formal analysis matcher backend lacks {key}")


def validate_formal_analysis(data_dir: Path) -> dict:
    """Fail closed unless ``data_dir`` is a complete, hashed formal analysis."""

    audit_path = data_dir / "audit.json"
    manifest_path = data_dir / "artifact_manifest.csv"
    if not audit_path.is_file():
        raise FileNotFoundError(audit_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("analysis audit must be a JSON object")
    if audit.get("schema") != "sgqs_full_val_analysis_v2":
        raise ValueError("unsupported or missing formal analysis audit schema")
    if audit.get("status") != "complete" or audit.get("formal_contract") is not True:
        raise ValueError("plotting refuses testing_noncomplete or non-formal analysis")
    if audit.get("split") != "val":
        raise ValueError("formal analysis audit is not the validation split")
    if tuple(map(int, audit.get("epochs", ()))) != FORMAL_EPOCHS:
        raise ValueError("formal analysis audit lacks the e28-e34 population")
    if tuple(map(str, audit.get("states", ()))) != FORMAL_STATES:
        raise ValueError("formal analysis audit lacks raw+EMA states")
    if tuple(map(float, audit.get("thresholds", ()))) != FORMAL_THRESHOLDS:
        raise ValueError("formal analysis audit lacks the ten-IoU grid")
    if tuple(map(int, audit.get("budgets", ()))) != FORMAL_BUDGETS:
        raise ValueError("formal analysis audit lacks maxDets=300/640")
    if int(audit.get("transition_budget", -1)) != 300:
        raise ValueError("formal analysis audit has the wrong transition budget")
    if (
        int(audit.get("images", -1)) != FORMAL_IMAGES
        or int(audit.get("noncrowd_annotations", -1)) != FORMAL_NONCROWD_GT
    ):
        raise ValueError("formal analysis population gate is missing or mismatched")
    if audit.get("global_pr_status") != "complete":
        raise ValueError("formal analysis global PR gate is not complete")
    expected_state_keys = {
        f"e{epoch}/{state}" for epoch in FORMAL_EPOCHS for state in FORMAL_STATES
    }
    pr_backends = _require_mapping(audit.get("global_pr_backends"), "global PR backend gate")
    if set(pr_backends) != expected_state_keys or any(
        backend not in {"persisted_detection_status", "evaluator_fallback"}
        for backend in pr_backends.values()
    ):
        raise ValueError("formal analysis global PR backend gate is incomplete")
    _validate_formal_identity(audit)

    declared_outputs = audit.get("outputs")
    if not isinstance(declared_outputs, list) or set(map(str, declared_outputs)) != FORMAL_ANALYSIS_OUTPUTS:
        raise ValueError("formal analysis audit output inventory is incomplete")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["file", "bytes", "sha256"]:
            raise ValueError("analysis artifact manifest has an unexpected schema")
        rows = list(reader)
    manifest: dict[str, dict[str, str]] = {}
    for row in rows:
        name = str(row.get("file", ""))
        if name in manifest:
            raise ValueError(f"duplicate analysis artifact manifest entry: {name}")
        manifest[name] = row
    expected_manifest_files = FORMAL_ANALYSIS_OUTPUTS - {"artifact_manifest.csv"}
    if set(manifest) != expected_manifest_files:
        raise ValueError("analysis artifact manifest is incomplete or contains unexpected files")
    for name in sorted(expected_manifest_files):
        path = data_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        row = manifest[name]
        try:
            expected_bytes = int(row["bytes"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid byte count in analysis manifest: {name}") from exc
        expected_sha = _require_sha(row.get("sha256"), f"manifest {name}")
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"analysis artifact byte-size mismatch: {name}")
        if sha256(path) != expected_sha:
            raise ValueError(f"analysis artifact SHA-256 mismatch: {name}")
    return audit


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#9CA3AF",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_metrics(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "metrics_14_states.csv")
    epochs = sorted({int(row["epoch"]) for row in rows})
    by_key = {(int(row["epoch"]), row["state"]): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    specs = (
        ("ap_small_d300", "COCO AP-small at maxDets=300"),
        ("ar_small_d300", "COCO AR-small at maxDets=300"),
    )
    for ax, (field, title) in zip(axes, specs):
        for state in ("raw", "ema"):
            values = [float(by_key[(epoch, state)][field]) for epoch in epochs]
            ax.plot(epochs, values, marker="o", linewidth=2.0, color=COLORS[state], label=state.upper())
        ax.set_title(title)
        ax.set_xlabel("checkpoint epoch")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xticks(epochs)
        ax.margins(x=0.04)
    axes[0].set_ylabel("metric")
    axes[1].legend(loc="best")
    fig.suptitle("Full-validation raw and EMA checkpoint trajectories", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "metrics_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stage_coverage(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "stage_coverage.csv")
    selected = [
        row
        for row in rows
        if row["state"] == "ema"
        and row["axis"] == "cohort"
        and row["group"] in {"coco_small", "local_hybrid_small"}
        and float(row["iou_threshold"]) in {0.5, 0.75}
    ]
    epochs = sorted({int(row["epoch"]) for row in selected})
    index = {
        (row["group"], float(row["iou_threshold"]), row["stage"], int(row["epoch"])): float(row["coverage"])
        for row in selected
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True)
    for row_index, cohort in enumerate(("coco_small", "local_hybrid_small")):
        for column_index, threshold in enumerate((0.5, 0.75)):
            ax = axes[row_index, column_index]
            for stage in COLORS:
                if stage not in {"encoder_dense_all", "encoder_top640", "decoder_all640", "decoder_top300"}:
                    continue
                values = [index[(cohort, threshold, stage, epoch)] for epoch in epochs]
                ax.plot(epochs, values, marker="o", linewidth=1.8, color=COLORS[stage], label=stage)
            ax.set_title(f"{cohort}, IoU≥{threshold:g}")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xticks(epochs)
            ax.set_ylabel("independent GT coverage")
            if row_index == 1:
                ax.set_xlabel("checkpoint epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "Full-validation EMA stage coverage (independent per GT, not COCO recall)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = output_dir / "stage_coverage.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attribution(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "transition_attribution.csv")
    selected = [
        row
        for row in rows
        if row["state"] == "ema"
        and row["match_scope"] == "coco_small_eval"
        and row["cohort"] == "coco_small_eval"
        and int(row["max_dets"]) == 300
    ]
    thresholds = sorted({float(row["iou_threshold"]) for row in selected})
    pairs = sorted({(int(row["epoch_start"]), int(row["epoch_end"])) for row in selected})
    labels = [f"e{start}→e{end}\ngain" for start, end in pairs] + [f"e{start}→e{end}\nloss" for start, end in pairs]
    keys = [(pair, "gain") for pair in pairs] + [(pair, "loss") for pair in pairs]
    totals = {}
    for row in selected:
        key = ((int(row["epoch_start"]), int(row["epoch_end"])), row["transition"], row["cause"])
        totals[key] = totals.get(key, 0.0) + int(row["count"])
    index = {key: value / len(thresholds) for key, value in totals.items()}
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    bottom = [0] * len(keys)
    for cause in CAUSES:
        values = [index.get((pair, direction, cause), 0) for pair, direction in keys]
        ax.bar(labels, values, bottom=bottom, color=COLORS[cause], label=cause.replace("_", " "))
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_ylabel("mean formal GT transitions per IoU slice")
    ax.set_title("EMA COCO-small transition endpoints averaged over IoU=.50:.05:.95")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = output_dir / "transition_attribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_iou(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "coco_small_per_iou.csv")
    selected = [row for row in rows if row["state"] == "ema" and int(row["max_dets"]) == 300]
    epochs = sorted({int(row["epoch"]) for row in selected})
    thresholds = sorted({float(row["iou_threshold"]) for row in selected})
    index = {
        (int(row["epoch"]), float(row["iou_threshold"])): row
        for row in selected
    }
    gap_rows = [row for row in read_csv(data_dir / "coco_small_per_iou_budget_gap.csv") if row["state"] == "ema"]
    gap_index = {
        (int(row["epoch"]), float(row["iou_threshold"])): row
        for row in gap_rows
    }
    colors = plt.cm.viridis([position / max(len(thresholds) - 1, 1) for position in range(len(thresholds))])
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True)
    for threshold, color in zip(thresholds, colors):
        axes[0, 0].plot(
            epochs,
            [float(index[(epoch, threshold)]["ap_small"]) for epoch in epochs],
            marker="o",
            color=color,
            label=f"IoU {threshold:.2f}",
        )
        axes[0, 1].plot(
            epochs,
            [float(index[(epoch, threshold)]["ar_small"]) for epoch in epochs],
            marker="o",
            color=color,
        )
        axes[1, 0].plot(
            epochs,
            [float(gap_index[(epoch, threshold)]["ap_d640_minus_d300"]) for epoch in epochs],
            marker="o",
            color=color,
        )
        axes[1, 1].plot(
            epochs,
            [float(gap_index[(epoch, threshold)]["ar_d640_minus_d300"]) for epoch in epochs],
            marker="o",
            color=color,
        )
    titles = (
        "AP-small @300",
        "AR-small @300",
        "AP-small @640 minus @300",
        "AR-small @640 minus @300",
    )
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xticks(epochs)
        ax.set_xlabel("checkpoint epoch")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0, 0].set_ylabel("metric")
    axes[1, 0].set_ylabel("budget gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.10, 0.5), ncol=1)
    fig.suptitle("EMA COCO-small per-IoU trajectories and maxDets gap", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    path = output_dir / "per_iou_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_pr(data_dir: Path, output_dir: Path) -> Path:
    pr_rows = read_csv(data_dir / "global_pr_curve.csv")
    separation = read_csv(data_dir / "tp_fp_score_separation.csv")
    available_epochs = sorted({int(row["epoch"]) for row in pr_rows})
    preferred = [epoch for epoch in (28, 29, 30, 34) if epoch in available_epochs]
    epochs = preferred or available_epochs
    selected_pr = [
        row
        for row in pr_rows
        if row["state"] == "ema"
        and row["scope"] == "coco_small_eval"
        and float(row["iou_threshold"]) == 0.75
        and int(row["max_dets"]) == 300
        and int(row["epoch"]) in epochs
    ]
    colors = plt.cm.viridis([position / max(len(epochs) - 1, 1) for position in range(len(epochs))])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for epoch, color in zip(epochs, colors):
        rows = sorted(
            (row for row in selected_pr if int(row["epoch"]) == epoch),
            key=lambda row: float(row["recall_threshold"]),
        )
        axes[0].plot(
            [float(row["recall_threshold"]) for row in rows],
            [float(row["interpolated_precision"]) for row in rows],
            color=color,
            linewidth=1.8,
            label=f"e{epoch}",
        )
    axes[0].set_title("EMA global PR, IoU=0.75, maxDets=300")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("interpolated precision")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend()

    selected_scores = [
        row
        for row in separation
        if row["scope"] == "coco_small_eval"
        and float(row["iou_threshold"]) == 0.75
        and int(row["max_dets"]) == 300
    ]
    score_index = {(int(row["epoch"]), row["state"]): row for row in selected_scores}
    all_epochs = sorted({key[0] for key in score_index})
    for state in ("raw", "ema"):
        axes[1].plot(
            all_epochs,
            [float(score_index[(epoch, state)]["tp_minus_fp_median"]) for epoch in all_epochs],
            marker="o",
            linewidth=2,
            color=COLORS[state],
            label=state.upper(),
        )
    axes[1].axhline(0, color="#9CA3AF", linewidth=0.8)
    axes[1].set_title("TP median score minus FP median score")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].set_ylabel("score separation")
    axes[1].set_xticks(all_epochs)
    axes[1].legend()
    fig.suptitle("Global score ordering diagnostics (formal COCO-small matching)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "global_pr_score_separation.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_buckets(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "match_gt_buckets.csv")
    axes_order = {
        "domain": ("wired", "wireless", "unknown"),
        "density": ("1to32", "33to128", "129to300", "ge301"),
        "input_short": ("lt8", "8to16", "16to32", "32to64", "ge64"),
    }
    selected = [
        row
        for row in rows
        if row["state"] == "ema"
        and row["scope"] == "all"
        and float(row["iou_threshold"]) == 0.75
        and int(row["max_dets"]) == 300
        and row["axis"] in axes_order
        and row["recall"] != ""
    ]
    epochs = sorted({int(row["epoch"]) for row in selected})
    index = {(row["axis"], row["group"], int(row["epoch"])): float(row["recall"]) for row in selected}
    fig, panels = plt.subplots(1, 3, figsize=(15.0, 4.8))
    epoch_colors = plt.cm.viridis([index / max(len(epochs) - 1, 1) for index in range(len(epochs))])
    for ax, (axis_name, preferred_groups) in zip(panels, axes_order.items()):
        groups = [group for group in preferred_groups if any((axis_name, group, epoch) in index for epoch in epochs)]
        x = list(range(len(groups)))
        for epoch, color in zip(epochs, epoch_colors):
            values = [index.get((axis_name, group, epoch), float("nan")) for group in groups]
            ax.plot(x, values, marker="o", linewidth=1.6, color=color, label=f"e{epoch}")
        ax.set_xticks(x, groups, rotation=25, ha="right")
        ax.set_title(axis_name.replace("_", " "))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("formal GT recall")
    panels[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("EMA all-area IoU≥0.75 GT recall by domain, density and input short side", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = output_dir / "bucket_recall.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    data_dir = args.data_dir.resolve()
    output_dir = args.output.resolve()
    validate_formal_analysis(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    for path in (
        plot_metrics(data_dir, output_dir),
        plot_per_iou(data_dir, output_dir),
        plot_stage_coverage(data_dir, output_dir),
        plot_attribution(data_dir, output_dir),
        plot_buckets(data_dir, output_dir),
        plot_global_pr(data_dir, output_dir),
    ):
        print(path)


if __name__ == "__main__":
    main()
