#!/usr/bin/env python3
"""Render deterministic static figures from EXP-DIAG-006 analysis tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
FORMAL_PAIRS = (
    (28, 29),
    (29, 30),
    (30, 31),
    (31, 32),
    (32, 33),
    (33, 34),
    (28, 30),
    (30, 34),
)
FORMAL_IMAGES = 12_841
FORMAL_NONCROWD_GT = 1_458_060
FORMAL_MATCHER_SCRIPT_SHA256 = (
    "9dafff504264bb501276f13ed403e45ae9eb98a181885b8255efdb7c9fc9028e"
)
FORMAL_MATCH_EXPORT_SHA256 = (
    "8151bf2514e00c171384b0571d8fc645d32c53aae7f3a96d95c293563f931237"
)
GLOBAL_PR_PLOT_EPOCHS = (28, 29, 30, 34)
GLOBAL_PR_RECALL_GRID = tuple(round(index / 100.0, 2) for index in range(101))
STAGE_COVERAGE_COHORTS = (
    (
        "coco_small_eval",
        "Evaluator-small\n(original annotation area <= 32^2)",
    ),
    (
        "local_hybrid_small",
        "Local hybrid-small\n(640-resized area < 32^2 OR short side < 16 px)",
    ),
)
STAGE_COVERAGE_STAGES = (
    "encoder_dense_all",
    "encoder_top640",
    "decoder_all640",
    "decoder_top300",
)
BUCKET_RECALL_SCOPE = "coco_small_eval"
BUCKET_RECALL_IOU = 0.75
BUCKET_RECALL_MAX_DETS = 300
BUCKET_RECALL_GROUPS = {
    "domain": ("wired", "wireless"),
    "density": ("1to32", "33to128", "129to300", "ge301"),
    "input_short": ("lt8", "8to16", "16to32", "32to64", "ge64"),
}
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
    if _require_sha(
        matrix.get("matcher_script_sha256"), "matcher script"
    ) != FORMAL_MATCHER_SCRIPT_SHA256:
        raise ValueError("formal analysis matcher script identity mismatch")
    if _require_sha(
        matrix.get("match_export_sha256"), "match exporter"
    ) != FORMAL_MATCH_EXPORT_SHA256:
        raise ValueError("formal analysis match exporter identity mismatch")
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
    try:
        pairs = tuple(tuple(map(int, pair)) for pair in audit.get("pairs", ()))
    except (TypeError, ValueError) as exc:
        raise ValueError("formal analysis audit has invalid transition pairs") from exc
    if pairs != FORMAL_PAIRS:
        raise ValueError("formal analysis audit lacks the fixed transition-pair matrix")
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


def _finite_float(row: Mapping[str, str], field: str, context: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{context} has non-finite {field}: {value}")
    return value


def _formal_iou(row: Mapping[str, str], context: str) -> float:
    """Return the canonical two-decimal IoU key without hiding off-grid input."""

    value = _finite_float(row, "iou_threshold", context)
    normalized = round(value, 2)
    if not math.isclose(value, normalized, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context} has off-grid iou_threshold: {value}")
    return normalized


def _integer(row: Mapping[str, str], field: str, context: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid {field}") from exc


def _require_exact_keys(
    actual: set[tuple],
    expected: set[tuple],
    label: str,
) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    raise ValueError(
        f"{label} matrix mismatch: expected {len(expected)} cells, got {len(actual)}; "
        f"missing={missing[:3]}; unexpected={unexpected[:3]}"
    )


def _bounded_float(
    row: Mapping[str, str],
    field: str,
    context: str,
    lower: float,
    upper: float,
) -> float:
    value = _finite_float(row, field, context)
    if not lower <= value <= upper:
        raise ValueError(
            f"{context} has out-of-range {field}: {value} not in [{lower}, {upper}]"
        )
    return value


def select_metric_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Validate the exact 7-epoch x raw/EMA metric matrix used by the plot."""

    expected = {(epoch, state) for epoch in FORMAL_EPOCHS for state in FORMAL_STATES}
    selected: dict[tuple[int, str], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        context = f"metric row {row_number}"
        epoch = _integer(row, "epoch", context)
        state = str(row.get("state", ""))
        key = (epoch, state)
        if key in selected:
            raise ValueError(f"duplicate metric cell: {key}")
        for field in ("ap_small_d300", "ar_small_d300"):
            _bounded_float(row, field, context, 0.0, 1.0)
        selected[key] = row
    _require_exact_keys(set(selected), expected, "metric")
    return [selected[key] for key in sorted(expected)]


def select_per_iou_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Validate the exact EMA/maxDets=300 per-IoU slice rendered below."""

    expected = {(epoch, threshold) for epoch in FORMAL_EPOCHS for threshold in FORMAL_THRESHOLDS}
    selected: dict[tuple[int, float], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("state") != "ema":
            continue
        context = f"per-IoU row {row_number}"
        max_dets = _integer(row, "max_dets", context)
        if max_dets != 300:
            continue
        epoch = _integer(row, "epoch", context)
        threshold = _formal_iou(row, context)
        key = (epoch, threshold)
        if key in selected:
            raise ValueError(f"duplicate EMA/maxDets=300 per-IoU cell: {key}")
        for field in ("ap_small", "ar_small"):
            _bounded_float(row, field, context, 0.0, 1.0)
        selected[key] = row
    _require_exact_keys(set(selected), expected, "EMA/maxDets=300 per-IoU")
    return [selected[key] for key in sorted(expected)]


def select_per_iou_gap_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Validate the exact EMA Top-640-minus-Top-300 per-IoU slice."""

    expected = {(epoch, threshold) for epoch in FORMAL_EPOCHS for threshold in FORMAL_THRESHOLDS}
    selected: dict[tuple[int, float], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("state") != "ema":
            continue
        context = f"per-IoU budget-gap row {row_number}"
        epoch = _integer(row, "epoch", context)
        threshold = _formal_iou(row, context)
        key = (epoch, threshold)
        if key in selected:
            raise ValueError(f"duplicate EMA per-IoU budget-gap cell: {key}")
        for field in ("ap_d640_minus_d300", "ar_d640_minus_d300"):
            _bounded_float(row, field, context, -1.0, 1.0)
        selected[key] = row
    _require_exact_keys(set(selected), expected, "EMA per-IoU budget gap")
    return [selected[key] for key in sorted(expected)]


def select_attribution_rows(
    rows: Sequence[Mapping[str, str]],
    pairs: Sequence[tuple[int, int]] = FORMAL_PAIRS,
) -> list[Mapping[str, str]]:
    """Validate the complete formal EMA evaluator-small attribution slice."""

    pair_tuple = tuple(tuple(map(int, pair)) for pair in pairs)
    expected = {
        (start, end, threshold, transition, cause)
        for start, end in pair_tuple
        for threshold in FORMAL_THRESHOLDS
        for transition in ("gain", "loss")
        for cause in CAUSES
    }
    selected: dict[tuple[int, int, float, str, str], Mapping[str, str]] = {}
    group_totals: dict[tuple[int, int, float, str], set[int]] = {}
    for row_number, row in enumerate(rows, start=2):
        if (
            row.get("state") != "ema"
            or row.get("match_scope") != "coco_small_eval"
            or row.get("cohort") != "coco_small_eval"
        ):
            continue
        context = f"transition attribution row {row_number}"
        max_dets = _integer(row, "max_dets", context)
        if max_dets != 300:
            continue
        start = _integer(row, "epoch_start", context)
        end = _integer(row, "epoch_end", context)
        threshold = _formal_iou(row, context)
        transition = str(row.get("transition", ""))
        cause = str(row.get("cause", ""))
        key = (start, end, threshold, transition, cause)
        if key in selected:
            raise ValueError(f"duplicate transition attribution cell: {key}")
        count = _integer(row, "count", context)
        transition_total = _integer(row, "transition_total", context)
        if count < 0 or transition_total < 0 or count > transition_total:
            raise ValueError(
                f"{context} has invalid count/transition_total: {count}/{transition_total}"
            )
        fraction_text = str(row.get("fraction", "")).strip()
        if transition_total:
            fraction = _finite_float(row, "fraction", context)
            if not math.isclose(
                fraction, count / transition_total, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{context} fraction disagrees with count/transition_total")
        elif fraction_text not in {"", "0", "0.0"}:
            raise ValueError(f"{context} has a fraction for an empty transition stratum")
        selected[key] = row
        group = (start, end, threshold, transition)
        group_totals.setdefault(group, set()).add(transition_total)

    _require_exact_keys(set(selected), expected, "formal transition attribution")
    for group, totals in group_totals.items():
        if len(totals) != 1:
            raise ValueError(f"transition attribution total is inconsistent for {group}: {totals}")
        total = next(iter(totals))
        observed = sum(
            _integer(selected[(*group, cause)], "count", f"transition attribution {group}")
            for cause in CAUSES
        )
        if observed != total:
            raise ValueError(
                f"transition attribution causes do not sum to total for {group}: {observed} != {total}"
            )
    return [selected[key] for key in sorted(expected)]


def select_global_pr_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Validate the exact four EMA PR curves displayed in the formal figure."""

    expected = {
        (epoch, recall)
        for epoch in GLOBAL_PR_PLOT_EPOCHS
        for recall in GLOBAL_PR_RECALL_GRID
    }
    selected: dict[tuple[int, float], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("state") != "ema" or row.get("scope") != "coco_small_eval":
            continue
        context = f"global PR row {row_number}"
        threshold = _formal_iou(row, context)
        max_dets = _integer(row, "max_dets", context)
        epoch = _integer(row, "epoch", context)
        if threshold != 0.75 or max_dets != 300 or epoch not in GLOBAL_PR_PLOT_EPOCHS:
            continue
        recall = _finite_float(row, "recall_threshold", context)
        normalized_recall = round(recall, 2)
        if not math.isclose(recall, normalized_recall, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{context} has off-grid recall_threshold: {recall}")
        key = (epoch, normalized_recall)
        if key in selected:
            raise ValueError(f"duplicate global PR cell: {key}")
        _bounded_float(row, "interpolated_precision", context, 0.0, 1.0)
        selected[key] = row
    _require_exact_keys(set(selected), expected, "formal global PR")
    return [selected[key] for key in sorted(expected)]


def select_score_separation_rows(
    rows: Sequence[Mapping[str, str]],
) -> list[Mapping[str, str]]:
    """Validate the exact 7-epoch x raw/EMA score-separation slice."""

    expected = {(epoch, state) for epoch in FORMAL_EPOCHS for state in FORMAL_STATES}
    selected: dict[tuple[int, str], Mapping[str, str]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("scope") != "coco_small_eval":
            continue
        context = f"score-separation row {row_number}"
        threshold = _formal_iou(row, context)
        max_dets = _integer(row, "max_dets", context)
        if threshold != 0.75 or max_dets != 300:
            continue
        epoch = _integer(row, "epoch", context)
        state = str(row.get("state", ""))
        key = (epoch, state)
        if key in selected:
            raise ValueError(f"duplicate score-separation cell: {key}")
        _finite_float(row, "tp_minus_fp_median", context)
        selected[key] = row
    _require_exact_keys(set(selected), expected, "formal score separation")
    return [selected[key] for key in sorted(expected)]


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
    rows = select_metric_rows(read_csv(data_dir / "metrics_14_states.csv"))
    epochs = list(FORMAL_EPOCHS)
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


def select_stage_coverage_rows(rows: Sequence[Mapping[str, str]]) -> list[Mapping[str, str]]:
    """Select the two declared small definitions for independent coverage.

    ``coco_small`` is intentionally excluded: it is a descriptive ``area <
    32^2`` cohort, whereas ``coco_small_eval`` is the inclusive evaluator
    population that defines AP-small/AR-small.
    """
    groups = tuple(group for group, _label in STAGE_COVERAGE_COHORTS)
    thresholds = (0.5, 0.75)
    expected = {
        (group, threshold, stage, epoch)
        for group in groups
        for threshold in thresholds
        for stage in STAGE_COVERAGE_STAGES
        for epoch in FORMAL_EPOCHS
    }
    selected: dict[tuple[str, float, str, int], Mapping[str, str]] = {}
    population: dict[str, set[int]] = {group: set() for group in groups}
    for row_number, row in enumerate(rows, start=2):
        if (
            row.get("state") != "ema"
            or row.get("axis") != "cohort"
            or row.get("group") not in groups
        ):
            continue
        context = f"stage coverage row {row_number}"
        threshold = _formal_iou(row, context)
        if threshold not in thresholds:
            continue
        stage = row.get("stage")
        if stage not in STAGE_COVERAGE_STAGES:
            raise ValueError(f"{context} has unexpected stage: {stage!r}")
        epoch = _integer(row, "epoch", context)
        group = str(row["group"])
        key = (group, threshold, stage, epoch)
        if key in selected:
            raise ValueError(f"duplicate stage coverage cell: {key}")

        n_gt = _integer(row, "n_gt", context)
        covered = _integer(row, "covered", context)
        coverage = _finite_float(row, "coverage", context)
        if n_gt <= 0 or not 0 <= covered <= n_gt:
            raise ValueError(
                f"{context} has invalid counts: covered={covered}, n_gt={n_gt}"
            )
        expected_coverage = covered / n_gt
        if not 0.0 <= coverage <= 1.0 or not math.isclose(
            coverage,
            expected_coverage,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{context} coverage disagrees with covered/n_gt: "
                f"{coverage} != {covered}/{n_gt}"
            )
        selected[key] = row
        population[group].add(n_gt)

    _require_exact_keys(set(selected), expected, "stage coverage")
    for group, values in population.items():
        if len(values) != 1:
            raise ValueError(f"stage coverage has inconsistent population for {group}: {values}")
    return [selected[key] for key in sorted(expected)]


def plot_stage_coverage(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "stage_coverage.csv")
    selected = select_stage_coverage_rows(rows)
    epochs = list(FORMAL_EPOCHS)
    index = {
        (row["group"], float(row["iou_threshold"]), row["stage"], int(row["epoch"])): float(row["coverage"])
        for row in selected
    }
    population = {
        cohort: {int(row["n_gt"]) for row in selected if row["group"] == cohort}
        for cohort, _label in STAGE_COVERAGE_COHORTS
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.3), sharex=True, sharey=True)
    for row_index, (cohort, cohort_label) in enumerate(STAGE_COVERAGE_COHORTS):
        for column_index, threshold in enumerate((0.5, 0.75)):
            ax = axes[row_index, column_index]
            for stage in STAGE_COVERAGE_STAGES:
                values = [index[(cohort, threshold, stage, epoch)] for epoch in epochs]
                ax.plot(epochs, values, marker="o", linewidth=1.8, color=COLORS[stage], label=stage)
            n_gt = next(iter(population[cohort]))
            ax.set_title(f"{cohort_label}; n={n_gt:,}\nIoU≥{threshold:g}")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_xticks(epochs)
            ax.set_ylabel("independent GT coverage")
            if row_index == 1:
                ax.set_xlabel("checkpoint epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(
        "EMA stage coverage: evaluator-small versus local hybrid-small\n"
        "Independent best-IoU coverage per GT; not one-to-one COCO recall",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    path = output_dir / "stage_coverage.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attribution(data_dir: Path, output_dir: Path) -> Path:
    selected = select_attribution_rows(
        read_csv(data_dir / "transition_attribution.csv"), FORMAL_PAIRS
    )
    thresholds = list(FORMAL_THRESHOLDS)
    pairs = list(FORMAL_PAIRS)
    labels = [f"e{start}→e{end}\ngain" for start, end in pairs] + [f"e{start}→e{end}\nloss" for start, end in pairs]
    keys = [(pair, "gain") for pair in pairs] + [(pair, "loss") for pair in pairs]
    totals = {}
    for row in selected:
        key = ((int(row["epoch_start"]), int(row["epoch_end"])), row["transition"], row["cause"])
        totals[key] = totals.get(key, 0.0) + int(row["count"])
    index = {key: value / len(thresholds) for key, value in totals.items()}
    fig, ax = plt.subplots(figsize=(17.5, 6.2))
    bottom = [0] * len(keys)
    for cause in CAUSES:
        values = [index.get((pair, direction, cause), 0) for pair, direction in keys]
        ax.bar(labels, values, bottom=bottom, color=COLORS[cause], label=cause.replace("_", " "))
        bottom = [left + right for left, right in zip(bottom, values)]
    ax.set_ylabel("mean formal GT transitions per IoU slice")
    ax.set_title("EMA evaluator-small transition endpoints averaged over IoU=.50:.05:.95")
    ax.tick_params(axis="x", labelrotation=32)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    path = output_dir / "transition_attribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_iou(data_dir: Path, output_dir: Path) -> Path:
    selected = select_per_iou_rows(read_csv(data_dir / "coco_small_per_iou.csv"))
    epochs = list(FORMAL_EPOCHS)
    thresholds = list(FORMAL_THRESHOLDS)
    index = {
        (int(row["epoch"]), round(float(row["iou_threshold"]), 2)): row
        for row in selected
    }
    gap_rows = select_per_iou_gap_rows(
        read_csv(data_dir / "coco_small_per_iou_budget_gap.csv")
    )
    gap_index = {
        (int(row["epoch"]), round(float(row["iou_threshold"]), 2)): row
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
    fig.suptitle(
        "EMA evaluator-small per-IoU trajectories and maxDets gap",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.90, 0.95))
    path = output_dir / "per_iou_trajectory.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_pr(data_dir: Path, output_dir: Path) -> Path:
    selected_pr = select_global_pr_rows(read_csv(data_dir / "global_pr_curve.csv"))
    selected_scores = select_score_separation_rows(
        read_csv(data_dir / "tp_fp_score_separation.csv")
    )
    epochs = list(GLOBAL_PR_PLOT_EPOCHS)
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

    score_index = {(int(row["epoch"]), row["state"]): row for row in selected_scores}
    all_epochs = list(FORMAL_EPOCHS)
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


def select_bucket_recall_rows(
    rows: Sequence[Mapping[str, str]],
    axes: Sequence[str],
) -> list[Mapping[str, str]]:
    """Select formal evaluator-small TP/FN recall rows for bucket plots."""
    requested_axes = tuple(axes)
    if len(set(requested_axes)) != len(requested_axes):
        raise ValueError(f"duplicate bucket axes requested: {requested_axes}")
    unknown_axes = set(requested_axes) - set(BUCKET_RECALL_GROUPS)
    if unknown_axes:
        raise ValueError(f"unsupported bucket axes requested: {sorted(unknown_axes)}")
    expected = {
        (axis, group, epoch)
        for axis in requested_axes
        for group in BUCKET_RECALL_GROUPS[axis]
        for epoch in FORMAL_EPOCHS
    }
    selected: dict[tuple[str, str, int], Mapping[str, str]] = {}
    populations: dict[tuple[str, str], set[tuple[int, int, int]]] = {}
    for row_number, row in enumerate(rows, start=2):
        axis = row.get("axis")
        if (
            row.get("state") != "ema"
            or row.get("scope") != BUCKET_RECALL_SCOPE
            or axis not in requested_axes
        ):
            continue
        context = f"bucket recall row {row_number}"
        threshold = _formal_iou(row, context)
        max_dets = _integer(row, "max_dets", context)
        if threshold != BUCKET_RECALL_IOU or max_dets != BUCKET_RECALL_MAX_DETS:
            continue
        group = row.get("group")
        if group not in BUCKET_RECALL_GROUPS[str(axis)]:
            # The analysis emits density=0 with an empty evaluator population.
            # It is deliberately absent from the plotted formal matrix.
            if row.get("recall") not in {None, ""}:
                raise ValueError(f"{context} has unexpected non-empty group: {axis}/{group}")
            continue
        epoch = _integer(row, "epoch", context)
        key = (str(axis), str(group), epoch)
        if key in selected:
            raise ValueError(f"duplicate bucket recall cell: {key}")

        recall = _finite_float(row, "recall", context)
        tp = _integer(row, "tp", context)
        fn = _integer(row, "fn", context)
        evaluable = _integer(row, "evaluable", context)
        objects = _integer(row, "objects", context)
        ignored = _integer(row, "ignored", context)
        if (
            tp < 0
            or fn < 0
            or ignored < 0
            or evaluable <= 0
            or tp + fn != evaluable
            or objects != evaluable + ignored
        ):
            raise ValueError(
                f"{context} has inconsistent population counts: objects={objects}, "
                f"tp={tp}, fn={fn}, evaluable={evaluable}, ignored={ignored}"
            )
        expected_recall = tp / evaluable
        if not 0.0 <= recall <= 1.0 or not math.isclose(
            recall,
            expected_recall,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"{context} recall disagrees with tp/evaluable: "
                f"{recall} != {tp}/{evaluable}"
            )
        selected[key] = row
        populations.setdefault((str(axis), str(group)), set()).add(
            (objects, evaluable, ignored)
        )

    _require_exact_keys(set(selected), expected, "bucket recall")
    for bucket, values in populations.items():
        if len(values) != 1:
            raise ValueError(
                f"bucket recall population changes across epochs for {bucket}: {sorted(values)}"
            )
    return [selected[key] for key in sorted(expected)]


def plot_buckets(data_dir: Path, output_dir: Path) -> Path:
    rows = read_csv(data_dir / "match_gt_buckets.csv")
    axes_order = BUCKET_RECALL_GROUPS
    selected = select_bucket_recall_rows(rows, tuple(axes_order))
    epochs = list(FORMAL_EPOCHS)
    index = {(row["axis"], row["group"], int(row["epoch"])): float(row["recall"]) for row in selected}
    fig, panels = plt.subplots(1, 3, figsize=(15.0, 4.8), sharey=True)
    epoch_colors = plt.cm.viridis([index / max(len(epochs) - 1, 1) for index in range(len(epochs))])
    for ax, (axis_name, preferred_groups) in zip(panels, axes_order.items()):
        groups = list(preferred_groups)
        x = list(range(len(groups)))
        for epoch, color in zip(epochs, epoch_colors):
            values = [index[(axis_name, group, epoch)] for group in groups]
            ax.plot(x, values, marker="o", linewidth=1.6, color=color, label=f"e{epoch}")
        ax.set_xticks(x, groups, rotation=25, ha="right")
        ax.set_title(axis_name.replace("_", " "))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    panels[0].set_ylabel("evaluator-small GT recall")
    panels[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        "EMA evaluator-small one-to-one GT recall by domain, density and input short side\n"
        "Original annotation area <= 32^2; IoU≥0.75; maxDets=300",
        fontsize=14,
        fontweight="bold",
    )
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
