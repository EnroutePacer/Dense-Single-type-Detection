"""Lightweight fail-closed tests for the EXP-DIAG-006 plotting gate."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("plot_full_val.py")
SPEC = importlib.util.spec_from_file_location("exp_diag_006_plot_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def descriptor(label: str) -> dict:
    return {
        "logical_name": label,
        "bytes": 1,
        "sha256": "a" * 64,
    }


def formal_audit() -> dict:
    replay_inputs = {}
    matcher_inputs = {}
    global_pr_backends = {}
    for epoch in MODULE.FORMAL_EPOCHS:
        replay_inputs[f"e{epoch}"] = {
            "summary_logical_name": f"e{epoch}/summary.json",
            "summary_sha256": "b" * 64,
            "metadata": descriptor(f"e{epoch}/metadata.npz"),
            "checkpoint_sha256": "c" * 64,
            "states": {
                state: {
                    "predictions": descriptor(f"e{epoch}/{state}/predictions.npz"),
                    "gt_stages": descriptor(f"e{epoch}/{state}/gt_stages.npz"),
                }
                for state in MODULE.FORMAL_STATES
            },
        }
        for state in MODULE.FORMAL_STATES:
            key = f"e{epoch}/{state}"
            matcher_inputs[key] = {
                name: descriptor(f"{key}/{name}")
                for name in (
                    "complete.json",
                    "protocol.json",
                    "match_aggregates.csv",
                    "detection_status_manifest.json",
                    "gt_shard_manifest.json",
                )
            }
            global_pr_backends[key] = "persisted_detection_status"
    matcher_matrix = descriptor("matcher_matrix/matrix.complete.json")
    matcher_matrix.update(
        matcher_script_sha256=MODULE.FORMAL_MATCHER_SCRIPT_SHA256,
        match_export_sha256=MODULE.FORMAL_MATCH_EXPORT_SHA256,
        replay_runtime_backend={
            "requested_device": "cpu",
            "torch_version": "test-torch",
            "torch_cuda_version": None,
        },
        matcher_runtime={
            "python_version": "3.test",
            "numpy_version": "test-numpy",
        },
    )
    return {
        "schema": "sgqs_full_val_analysis_v2",
        "status": "complete",
        "formal_contract": True,
        "split": "val",
        "epochs": list(MODULE.FORMAL_EPOCHS),
        "states": list(MODULE.FORMAL_STATES),
        "thresholds": list(MODULE.FORMAL_THRESHOLDS),
        "budgets": list(MODULE.FORMAL_BUDGETS),
        "pairs": [list(pair) for pair in MODULE.FORMAL_PAIRS],
        "transition_budget": 300,
        "images": MODULE.FORMAL_IMAGES,
        "noncrowd_annotations": MODULE.FORMAL_NONCROWD_GT,
        "global_pr_status": "complete",
        "global_pr_backends": global_pr_backends,
        "formal_identity": {
            "config_sha256": "d" * 64,
            "annotation_sha256": "e" * 64,
            "inference_identity": {
                "device": "cpu",
                "batch_size": 1,
                "torch_version": "test-torch",
                "torch_cuda_version": None,
            },
        },
        "input_identity": {
            "replays": replay_inputs,
            "matchers": matcher_inputs,
            "matcher_matrix": matcher_matrix,
        },
        "outputs": sorted(MODULE.FORMAL_ANALYSIS_OUTPUTS),
    }


def rewrite_manifest(root: Path) -> None:
    with (root / "artifact_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "bytes", "sha256"), lineterminator="\n")
        writer.writeheader()
        for name in sorted(MODULE.FORMAL_ANALYSIS_OUTPUTS - {"artifact_manifest.csv"}):
            path = root / name
            writer.writerow(
                {
                    "file": name,
                    "bytes": path.stat().st_size,
                    "sha256": MODULE.sha256(path),
                }
            )


def write_fixture(root: Path, audit: dict | None = None) -> dict:
    value = formal_audit() if audit is None else audit
    for name in MODULE.FORMAL_ANALYSIS_OUTPUTS - {"audit.json", "artifact_manifest.csv"}:
        (root / name).write_text(f"synthetic {name}\n", encoding="utf-8")
    (root / "audit.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rewrite_manifest(root)
    return value


def stage_coverage_rows() -> list[dict[str, str]]:
    rows = []
    for epoch in MODULE.FORMAL_EPOCHS:
        for group, _label in MODULE.STAGE_COVERAGE_COHORTS:
            n_gt = 20 if group == "coco_small_eval" else 40
            for threshold in (0.5, 0.75):
                for stage in MODULE.STAGE_COVERAGE_STAGES:
                    covered = n_gt // 2
                    rows.append(
                        {
                            "epoch": str(epoch),
                            "state": "ema",
                            "axis": "cohort",
                            "group": group,
                            "iou_threshold": str(threshold),
                            "stage": stage,
                            "n_gt": str(n_gt),
                            "covered": str(covered),
                            "coverage": str(covered / n_gt),
                        }
                    )
    return rows


def bucket_recall_rows(axes: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for epoch in MODULE.FORMAL_EPOCHS:
        for axis in axes:
            for group in MODULE.BUCKET_RECALL_GROUPS[axis]:
                evaluable = 20
                tp = epoch - 20
                fn = evaluable - tp
                rows.append(
                    {
                        "epoch": str(epoch),
                        "state": "ema",
                        "scope": MODULE.BUCKET_RECALL_SCOPE,
                        "iou_threshold": str(MODULE.BUCKET_RECALL_IOU),
                        "max_dets": str(MODULE.BUCKET_RECALL_MAX_DETS),
                        "axis": axis,
                        "group": group,
                        "evaluable": str(evaluable),
                        "objects": "40",
                        "tp": str(tp),
                        "fn": str(fn),
                        "ignored": "20",
                        "recall": str(tp / evaluable),
                    }
                )
    return rows


def metric_rows() -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "state": state,
            "ap_small_d300": "0.7",
            "ar_small_d300": "0.8",
        }
        for epoch in MODULE.FORMAL_EPOCHS
        for state in MODULE.FORMAL_STATES
    ]


def per_iou_rows() -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "state": "ema",
            "max_dets": "300",
            "iou_threshold": str(threshold),
            "ap_small": "0.7",
            "ar_small": "0.8",
        }
        for epoch in MODULE.FORMAL_EPOCHS
        for threshold in MODULE.FORMAL_THRESHOLDS
    ]


def per_iou_gap_rows() -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "state": "ema",
            "iou_threshold": str(threshold),
            "ap_d640_minus_d300": "0.01",
            "ar_d640_minus_d300": "0.02",
        }
        for epoch in MODULE.FORMAL_EPOCHS
        for threshold in MODULE.FORMAL_THRESHOLDS
    ]


def attribution_rows() -> list[dict[str, str]]:
    rows = []
    for start, end in MODULE.FORMAL_PAIRS:
        for threshold in MODULE.FORMAL_THRESHOLDS:
            for transition in ("gain", "loss"):
                for cause in MODULE.CAUSES:
                    count = 5 if cause == MODULE.CAUSES[0] else 0
                    rows.append(
                        {
                            "epoch_start": str(start),
                            "epoch_end": str(end),
                            "state": "ema",
                            "match_scope": "coco_small_eval",
                            "cohort": "coco_small_eval",
                            "iou_threshold": str(threshold),
                            "max_dets": "300",
                            "transition": transition,
                            "cause": cause,
                            "count": str(count),
                            "transition_total": "5",
                            "fraction": str(count / 5),
                        }
                    )
    return rows


def global_pr_rows() -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "state": "ema",
            "scope": "coco_small_eval",
            "iou_threshold": "0.75",
            "max_dets": "300",
            "recall_threshold": str(recall),
            "interpolated_precision": str(1.0 - recall / 2.0),
        }
        for epoch in MODULE.GLOBAL_PR_PLOT_EPOCHS
        for recall in MODULE.GLOBAL_PR_RECALL_GRID
    ]


def score_separation_rows() -> list[dict[str, str]]:
    return [
        {
            "epoch": str(epoch),
            "state": state,
            "scope": "coco_small_eval",
            "iou_threshold": "0.75",
            "max_dets": "300",
            "tp_minus_fp_median": "0.1",
        }
        for epoch in MODULE.FORMAL_EPOCHS
        for state in MODULE.FORMAL_STATES
    ]


class PlotFormalGateTest(unittest.TestCase):
    def test_accepts_complete_formal_hashed_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit = write_fixture(root)
            self.assertEqual(MODULE.validate_formal_analysis(root), audit)

    def test_rejects_nonformal_population_and_backend_gaps(self):
        cases = {
            "testing audit": lambda audit: audit.update(status="testing_noncomplete"),
            "population": lambda audit: audit.update(images=MODULE.FORMAL_IMAGES - 1),
            "backend": lambda audit: audit["input_identity"]["matcher_matrix"].pop("matcher_runtime"),
            "pairs": lambda audit: audit.update(pairs=[[28, 29]]),
            "pair order": lambda audit: audit.update(
                pairs=list(reversed(audit["pairs"]))
            ),
            "matcher script": lambda audit: audit["input_identity"]["matcher_matrix"].update(
                matcher_script_sha256="f" * 64
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                audit = formal_audit()
                mutate(audit)
                write_fixture(root, audit)
                with self.assertRaises(ValueError):
                    MODULE.validate_formal_analysis(root)

    def test_rejects_artifact_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_fixture(root)
            with (root / "metrics_14_states.csv").open("a", encoding="utf-8") as handle:
                handle.write("tampered\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                MODULE.validate_formal_analysis(root)


class PlotSmallScopeSelectionTest(unittest.TestCase):
    def test_formal_plot_slices_accept_complete_exact_matrices(self):
        cases = (
            (MODULE.select_metric_rows, metric_rows(), 14),
            (MODULE.select_per_iou_rows, per_iou_rows(), 70),
            (MODULE.select_per_iou_gap_rows, per_iou_gap_rows(), 70),
            (MODULE.select_attribution_rows, attribution_rows(), 800),
            (MODULE.select_global_pr_rows, global_pr_rows(), 404),
            (MODULE.select_score_separation_rows, score_separation_rows(), 14),
        )
        for selector, rows, expected in cases:
            with self.subTest(selector=selector.__name__):
                self.assertEqual(len(selector(rows)), expected)

    def test_formal_iou_keys_accept_binary_float_serialization_artifact(self):
        cases = (
            (MODULE.select_per_iou_rows, per_iou_rows()),
            (MODULE.select_per_iou_gap_rows, per_iou_gap_rows()),
            (MODULE.select_attribution_rows, attribution_rows()),
        )
        for selector, rows in cases:
            for row in rows:
                if row["iou_threshold"] == "0.9":
                    row["iou_threshold"] = "0.8999999999999999"
            with self.subTest(selector=selector.__name__):
                selector(rows)

        row = per_iou_rows()[0]
        row["iou_threshold"] = "0.8998"
        with self.assertRaisesRegex(ValueError, "off-grid iou_threshold"):
            MODULE._formal_iou(row, "synthetic")

    def test_formal_plot_slices_reject_missing_duplicate_and_nonfinite_cells(self):
        cases = (
            (MODULE.select_metric_rows, metric_rows, "ap_small_d300"),
            (MODULE.select_per_iou_rows, per_iou_rows, "ap_small"),
            (
                MODULE.select_per_iou_gap_rows,
                per_iou_gap_rows,
                "ap_d640_minus_d300",
            ),
            (MODULE.select_global_pr_rows, global_pr_rows, "interpolated_precision"),
            (
                MODULE.select_score_separation_rows,
                score_separation_rows,
                "tp_minus_fp_median",
            ),
        )
        for selector, factory, finite_field in cases:
            with self.subTest(selector=selector.__name__, mutation="missing"):
                rows = factory()
                rows.pop()
                with self.assertRaisesRegex(ValueError, "matrix mismatch"):
                    selector(rows)
            with self.subTest(selector=selector.__name__, mutation="duplicate"):
                rows = factory()
                rows.append(dict(rows[0]))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    selector(rows)
            with self.subTest(selector=selector.__name__, mutation="nan"):
                rows = factory()
                rows[0][finite_field] = "nan"
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    selector(rows)

    def test_attribution_rejects_missing_duplicate_nan_and_count_mismatch(self):
        rows = attribution_rows()
        for mutation, message in (
            (lambda values: values.pop(), "matrix mismatch"),
            (lambda values: values.append(dict(values[0])), "duplicate"),
            (lambda values: values[0].update(fraction="nan"), "non-finite"),
            (
                lambda values: values[0].update(count="4", fraction="0.8"),
                "do not sum",
            ),
        ):
            with self.subTest(message=message):
                candidate = [dict(row) for row in rows]
                mutation(candidate)
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.select_attribution_rows(candidate)

    def test_stage_coverage_compares_evaluator_and_hybrid_definitions(self):
        rows = stage_coverage_rows()
        rows.extend(
            [
                {**rows[0], "group": "coco_small"},
                {**rows[0], "group": "all"},
            ]
        )
        selected = MODULE.select_stage_coverage_rows(rows)
        self.assertEqual(len(selected), 112)
        self.assertEqual(
            {row["group"] for row in selected},
            {"coco_small_eval", "local_hybrid_small"},
        )

    def test_bucket_recall_uses_formal_evaluator_small_scope(self):
        axes = tuple(MODULE.BUCKET_RECALL_GROUPS)
        rows = bucket_recall_rows(axes)
        rows.append({**rows[0], "scope": "all"})
        selected = MODULE.select_bucket_recall_rows(rows, axes)
        self.assertEqual(len(selected), 77)
        self.assertEqual({row["scope"] for row in selected}, {"coco_small_eval"})

    def test_stage_coverage_rejects_missing_and_duplicate_cells(self):
        cases = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.append(dict(rows[0])),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                rows = stage_coverage_rows()
                mutate(rows)
                with self.assertRaisesRegex(ValueError, "stage coverage|duplicate"):
                    MODULE.select_stage_coverage_rows(rows)

    def test_stage_coverage_rejects_wrong_epoch_and_inconsistent_ratio(self):
        cases = {
            "epoch": ("epoch", "35", "matrix mismatch"),
            "ratio": ("coverage", "0.4", "covered/n_gt"),
            "nan": ("coverage", "nan", "non-finite"),
        }
        for label, (field, value, message) in cases.items():
            with self.subTest(label=label):
                rows = stage_coverage_rows()
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.select_stage_coverage_rows(rows)

    def test_bucket_recall_rejects_missing_and_duplicate_cells(self):
        cases = {
            "missing": lambda rows: rows.pop(),
            "duplicate": lambda rows: rows.append(dict(rows[0])),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                rows = bucket_recall_rows(("domain",))
                mutate(rows)
                with self.assertRaisesRegex(ValueError, "bucket recall|duplicate"):
                    MODULE.select_bucket_recall_rows(rows, ("domain",))

    def test_bucket_recall_rejects_wrong_epoch_inconsistent_ratio_and_nan(self):
        cases = {
            "epoch": ("epoch", "35", "matrix mismatch"),
            "ratio": ("recall", "0.1", "tp/evaluable"),
            "nan": ("recall", "nan", "non-finite"),
        }
        for label, (field, value, message) in cases.items():
            with self.subTest(label=label):
                rows = bucket_recall_rows(("domain",))
                rows[0][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    MODULE.select_bucket_recall_rows(rows, ("domain",))

    def test_bucket_recall_rejects_count_identity_and_population_drift(self):
        cases = {
            "identity": lambda row: row.update(objects="39"),
            "population drift": lambda row: row.update(
                objects="41", evaluable="21", ignored="20", fn="7", recall=str(14 / 21)
            ),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                rows = bucket_recall_rows(("domain",))
                mutate(rows[-1])
                with self.assertRaisesRegex(ValueError, "population"):
                    MODULE.select_bucket_recall_rows(rows, ("domain",))


if __name__ == "__main__":
    unittest.main()
