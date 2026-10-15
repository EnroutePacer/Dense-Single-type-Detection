"""Synthetic and legacy-regression tests for the full-val analysis pipeline."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("analyze_full_val.py")
SPEC = importlib.util.spec_from_file_location("exp_diag_006_analyze", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
PLOT_SCRIPT = Path(__file__).with_name("plot_full_val.py")
PLOT_SPEC = importlib.util.spec_from_file_location("exp_diag_006_plot", PLOT_SCRIPT)
PLOT_MODULE = importlib.util.module_from_spec(PLOT_SPEC)
assert PLOT_SPEC.loader is not None
sys.modules[PLOT_SPEC.name] = PLOT_MODULE
PLOT_SPEC.loader.exec_module(PLOT_MODULE)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_gzip_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)


def cohort_masks(coco: np.ndarray, hybrid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(coco), dtype=bool),
        "coco_small": coco,
        "local_hybrid_small": hybrid,
        "coco_and_hybrid": coco & hybrid,
        "coco_only": coco & ~hybrid,
        "hybrid_only": hybrid & ~coco,
        "neither": ~coco & ~hybrid,
    }


def create_fixture(root: Path) -> tuple[list[tuple[int, Path]], list[tuple[tuple[int, str], Path]]]:
    image_ids = np.asarray([10, 20], dtype=np.int64)
    annotation_ids = np.asarray([101, 102, 103, 104, 105], dtype=np.int64)
    coco_area = np.asarray([100.0, 500.0, 2000.0, 10000.0, 1024.0])
    coco = coco_area < 32**2
    hybrid = np.asarray([True, False, True, False, True])
    membership = np.asarray([0, 1, 2, 3, 2], dtype=np.uint8)
    metadata_arrays = {
        "schema": np.asarray("sgqs_full_val_metadata_v1"),
        "image_ids": image_ids,
        "image_file_name": np.asarray(["val/wired/a.png", "val/wireless/b.png"]),
        "image_gt_start": np.asarray([0, 3, 5], dtype=np.int64),
        "annotation_ids": annotation_ids,
        "gt_image_index": np.asarray([0, 0, 0, 1, 1], dtype=np.int32),
        "coco_area": coco_area,
        "coco_area_group": np.asarray([0, 0, 1, 2, 1], dtype=np.uint8),
        "input_area_px2": np.asarray([100, 1500, 500, 10000, 100], dtype=np.float32),
        "input_short_px": np.asarray([10, 20, 10, 80, 7], dtype=np.float32),
        "coco_small": coco,
        "local_hybrid_small": hybrid,
        "small_definition_membership": membership,
    }
    stage_before = {
        "encoder_dense_all": np.asarray([0.60, 0.90, 0.90, 0.90, 0.90], dtype=np.float32),
        "encoder_top640": np.asarray([0.60, 0.60, 0.90, 0.90, 0.90], dtype=np.float32),
        "decoder_all640": np.asarray([0.60, 0.60, 0.60, 0.90, 0.90], dtype=np.float32),
        "decoder_top300": np.asarray([0.60, 0.60, 0.60, 0.60, 0.90], dtype=np.float32),
    }
    stage_after = {stage: np.full(5, 0.90, dtype=np.float32) for stage in MODULE.STAGES}

    replay_specs = []
    match_specs = []
    for epoch, stage_values in ((28, stage_before), (29, stage_after)):
        replay = root / f"replay_e{epoch}"
        replay.mkdir()
        metadata_path = replay / "val.metadata.npz"
        np.savez_compressed(metadata_path, **metadata_arrays)
        summary = {
            "last_epoch": epoch,
            "checkpoint_sha256": f"synthetic-e{epoch}",
            "splits": {
                "val": {
                    "images": 2,
                    "noncrowd_annotations": 5,
                    "metadata": "val.metadata.npz",
                    "metadata_sha256": MODULE.sha256(metadata_path),
                }
            },
            "results": {},
        }
        for state in ("raw", "ema"):
            stage_file = f"val.{state}.gt_stages.npz"
            arrays = {
                "schema": np.asarray("sgqs_compact_gt_stages_v1"),
                "stage_names": np.asarray(MODULE.STAGES),
            }
            for stage, best in stage_values.items():
                arrays[f"{stage}__best_iou"] = best
                for threshold in (0.5, 0.75):
                    if epoch == 28:
                        default_rank = np.where(best >= threshold, 100, 641).astype(np.uint16)
                        if stage == "decoder_all640" and best[3] >= threshold:
                            default_rank[3] = 350
                        if stage == "decoder_top300" and best[3] < threshold:
                            default_rank[3] = 301
                    else:
                        default_rank = np.full(5, 100, dtype=np.uint16)
                    suffix = f"{threshold:g}"
                    arrays[f"{stage}__first_rank_iou{suffix}"] = default_rank
                    arrays[f"{stage}__first_score_iou{suffix}"] = np.where(
                        best >= threshold, 0.8, np.nan
                    ).astype(np.float32)
            np.savez_compressed(replay / stage_file, **arrays)
            stage_path = replay / stage_file
            coverage = {}
            for cohort, mask in cohort_masks(coco, hybrid).items():
                coverage[cohort] = {}
                for threshold in (0.5, 0.75):
                    coverage[cohort][f"iou{threshold:g}"] = {
                        stage: int(np.count_nonzero(mask & (best >= threshold)))
                        for stage, best in stage_values.items()
                    }
            metrics300 = [0.5 + epoch / 10000] * 12
            metrics640 = [0.6 + epoch / 10000] * 12
            summary["results"][f"val/{state}"] = {
                "images": 2,
                "noncrowd_annotations": 5,
                "coco_maxdets300": metrics300,
                "coco_maxdets640": metrics640,
                "coco_small_per_iou_maxdets300": [
                    {"iou": threshold, "ap": metrics300[3], "ar": metrics300[9]}
                    for threshold in MODULE.DEFAULT_THRESHOLDS
                ],
                "coco_small_per_iou_maxdets640": [
                    {"iou": threshold, "ap": metrics640[3], "ar": metrics640[9]}
                    for threshold in MODULE.DEFAULT_THRESHOLDS
                ],
                "independent_gt_coverage": coverage,
                "artifacts": {
                    "gt_stages": {
                        "file": stage_file,
                        "bytes": stage_path.stat().st_size,
                        "sha256": MODULE.sha256(stage_path),
                    }
                },
            }
            match_dir = root / f"matches_e{epoch}_{state}"
            match_dir.mkdir()
            rows = []
            for index, annotation_id in enumerate(annotation_ids):
                before = epoch == 28
                all_status_300 = "fn" if before else "tp"
                all_status_640 = "tp" if before and index == 3 else all_status_300
                is_coco_eval = bool(coco_area[index] <= 32**2)
                small_status_300 = all_status_300 if is_coco_eval else "ignored"
                small_status_640 = all_status_640 if is_coco_eval else "ignored"
                all_rank = 350 if before and index == 3 else ("" if before else 10)
                all_query = 3 if before and index == 3 else ("" if before else index)
                all_score = 0.8 if all_rank != "" else ""
                all_iou = 0.9 if all_rank != "" else ""
                if is_coco_eval:
                    small_rank, small_query, small_score, small_iou = all_rank, all_query, all_score, all_iou
                else:
                    small_rank = small_query = small_score = small_iou = ""
                match_row = {
                    "state": f"checkpoint{epoch:04d}:{state}",
                    "annotation_id": int(annotation_id),
                }
                for threshold_label in ("iou50", "iou75"):
                    match_row.update(
                        {
                            f"all_{threshold_label}_status_d300": all_status_300,
                            f"all_{threshold_label}_status_d640": all_status_640,
                            f"all_{threshold_label}_query_index": all_query,
                            f"all_{threshold_label}_rank": all_rank,
                            f"all_{threshold_label}_score": all_score,
                            f"all_{threshold_label}_iou": all_iou,
                            f"coco_small_eval_{threshold_label}_status_d300": small_status_300,
                            f"coco_small_eval_{threshold_label}_status_d640": small_status_640,
                            f"coco_small_eval_{threshold_label}_query_index": small_query,
                            f"coco_small_eval_{threshold_label}_rank": small_rank,
                            f"coco_small_eval_{threshold_label}_score": small_score,
                            f"coco_small_eval_{threshold_label}_iou": small_iou,
                        }
                    )
                rows.append(match_row)
            write_gzip_csv(match_dir / "gt_matches.part0000.csv.gz", rows)
            write_json(
                match_dir / "protocol.json",
                {"schema": "synthetic", "state": f"checkpoint{epoch:04d}:{state}"},
            )
            write_json(match_dir / "complete.json", {"shards": 1})
            with (match_dir / "match_aggregates.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["state", "unit", "scope", "iou_threshold", "max_dets", "axis", "group", "tp", "fp", "fn", "ignored", "truncated", "precision", "recall", "note"],
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "state": state,
                        "unit": "overall",
                        "scope": "all",
                        "iou_threshold": 0.75,
                        "max_dets": 300,
                        "axis": "all",
                        "group": "all",
                        "tp": 0 if before else 5,
                        "fp": 0,
                        "fn": 5 if before else 0,
                        "ignored": 0,
                        "truncated": 0,
                        "precision": "",
                        "recall": 0 if before else 1,
                        "note": "synthetic fixture",
                    }
                )
            match_specs.append(((epoch, state), match_dir))
        write_json(replay / "summary.json", summary)
        replay_specs.append((epoch, replay))
    return replay_specs, match_specs


class FullValAnalysisTest(unittest.TestCase):
    def test_default_pairs_include_complete_adjacent_trajectory(self):
        adjacent = set(zip(MODULE.FORMAL_EPOCHS[:-1], MODULE.FORMAL_EPOCHS[1:]))
        self.assertLessEqual(adjacent, set(MODULE.DEFAULT_PAIRS))
        self.assertLessEqual({(28, 30), (30, 34)}, set(MODULE.DEFAULT_PAIRS))

    def test_representative_bucket_is_bounded_and_keeps_top_k(self):
        bucket = {}
        candidates = (
            (1, (-0.1, 20, 1), "image-1"),
            (2, (-0.9, 50, 2), "image-2"),
            (3, (-0.8, 5, 3), "image-3"),
            (4, (-0.8, 3, 4), "image-4"),
            (4, (-0.7, 1, 5), "ignored-worse-same-image"),
            (3, (-1.0, 99, 6), "improved-image-3"),
        )
        for image_id, sort_key, marker in candidates:
            MODULE.retain_representative_candidate(
                bucket,
                image_id,
                sort_key,
                {"marker": marker},
                2,
            )
            self.assertLessEqual(len(bucket), 2)

        ordered = sorted(bucket.items(), key=lambda item: item[1][0])
        self.assertEqual([image_id for image_id, _candidate in ordered], [3, 2])
        self.assertEqual(ordered[0][1][1]["marker"], "improved-image-3")

    def test_synthetic_end_to_end_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            replays, matches = create_fixture(root)
            outputs = []
            for suffix in ("a", "b"):
                output = root / f"analysis_{suffix}"
                args = MODULE.parse_args(
                    [
                        *(item for epoch, path in replays for item in ("--replay", f"{epoch}={path}")),
                        *(item for (epoch, state), path in matches for item in ("--match", f"{epoch}:{state}={path}")),
                        "--output",
                        str(output),
                        "--pair",
                        "28:29",
                        "--thresholds",
                        "0.5",
                        "0.75",
                        "--testing",
                        "--skip-global-pr",
                    ]
                )
                audit = MODULE.run_analysis(args)
                # Five all-area gains plus three evaluator-small gains (the
                # third lies exactly at area==32^2) for each state.
                self.assertEqual(audit["transition_detail_rows"], 32)
                self.assertEqual(audit["status"], "testing_noncomplete")
                self.assertTrue((output / "transition_details.csv.gz").is_file())
                outputs.append(output)

            with (outputs[0] / "metrics_14_states.csv").open(encoding="utf-8", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)
            with (outputs[0] / "transition_attribution.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            selected = {
                row["cause"]: int(row["count"])
                for row in rows
                if row["state"] == "ema"
                and row["match_scope"] == "all"
                and row["cohort"] == "all"
                and row["transition"] == "gain"
                and float(row["iou_threshold"]) == 0.75
            }
            self.assertEqual(selected, {cause: 1 for cause in MODULE.CAUSES})
            with (outputs[0] / "transition_summary.csv").open(encoding="utf-8", newline="") as handle:
                transition_rows = list(csv.DictReader(handle))
            evaluator_row = next(
                row
                for row in transition_rows
                if row["state"] == "ema"
                and row["match_scope"] == "coco_small_eval"
                and row["cohort"] == "coco_small_eval"
                and float(row["iou_threshold"]) == 0.75
            )
            strict_row = next(
                row
                for row in transition_rows
                if row["state"] == "ema"
                and row["match_scope"] == "all"
                and row["cohort"] == "coco_small"
                and float(row["iou_threshold"]) == 0.75
            )
            self.assertEqual(int(evaluator_row["evaluable"]), 3)
            self.assertEqual(int(strict_row["evaluable"]), 2)
            with (outputs[0] / "representative_annotations.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                representative_reader = csv.DictReader(handle)
                self.assertEqual(
                    tuple(representative_reader.fieldnames or ()),
                    MODULE.REPRESENTATIVE_FIELDS,
                )
                representative_rows = list(representative_reader)
            rank_example = next(
                row
                for row in representative_rows
                if row["state"] == "ema"
                and row["match_scope"] == "all"
                and row["cause"] == "rank_gt300"
                and float(row["iou_threshold"]) == 0.75
            )
            self.assertEqual(
                (rank_example["status_before"], rank_example["status_after"]),
                ("fn", "tp"),
            )
            self.assertEqual(
                (
                    int(rank_example["formal_query_before"]),
                    int(rank_example["formal_query_after"]),
                    int(rank_example["formal_rank_before"]),
                    int(rank_example["formal_rank_after"]),
                ),
                (3, 3, 350, 10),
            )
            self.assertEqual(
                (
                    int(rank_example["decoder_all_rank_before"]),
                    int(rank_example["decoder_top300_rank_before"]),
                    int(rank_example["decoder_all_rank_after"]),
                    int(rank_example["decoder_top300_rank_after"]),
                ),
                (350, 301, 100, 100),
            )
            self.assertEqual(
                (outputs[0] / "representative_annotations.csv").read_bytes(),
                (outputs[1] / "representative_annotations.csv").read_bytes(),
            )
            strata = {}
            for row in representative_rows:
                stratum = tuple(
                    row[field]
                    for field in (
                        "state",
                        "epoch_start",
                        "epoch_end",
                        "match_scope",
                        "iou_threshold",
                        "transition",
                        "cause",
                        "small_partition",
                        "domain",
                    )
                )
                strata.setdefault(stratum, []).append(row)
            self.assertTrue(strata)
            self.assertTrue(all(len(rows) <= 1 for rows in strata.values()))
            for rows in strata.values():
                self.assertEqual(
                    [int(row["selection_rank"]) for row in rows],
                    list(range(1, len(rows) + 1)),
                )
            for name in ("analysis_summary.md", "audit.json"):
                self.assertNotIn(b"\r", (outputs[0] / name).read_bytes())
                self.assertEqual(
                    (outputs[0] / name).read_bytes(),
                    (outputs[1] / name).read_bytes(),
                )
            self.assertNotIn(str(root), (outputs[0] / "audit.json").read_text(encoding="utf-8"))

            # The end-to-end fixture skips the expensive evaluator pass, so
            # seed only the two global-PR tables needed to exercise rendering.
            pr_rows = []
            for epoch in (28, 29):
                for recall in np.linspace(0.0, 1.0, 101):
                    pr_rows.append(
                        {
                            "epoch": epoch,
                            "state": "ema",
                            "scope": "coco_small_eval",
                            "iou_threshold": 0.75,
                            "max_dets": 300,
                            "recall_threshold": recall,
                            "interpolated_precision": max(0.0, 1.0 - recall * 0.2),
                            "minimum_score_at_recall": max(0.0, 1.0 - recall),
                        }
                    )
            score_rows = [
                {
                    "epoch": epoch,
                    "state": state,
                    "scope": "coco_small_eval",
                    "iou_threshold": 0.75,
                    "max_dets": 300,
                    "tp_minus_fp_median": 0.1 + epoch / 10000,
                }
                for epoch in (28, 29)
                for state in ("raw", "ema")
            ]
            MODULE.write_csv(outputs[0] / "global_pr_curve.csv", pr_rows)
            MODULE.write_csv(outputs[0] / "tp_fp_score_separation.csv", score_rows)
            figure_dir = root / "figures"
            figure_dir.mkdir()
            PLOT_MODULE.configure_style()
            figures = (
                PLOT_MODULE.plot_metrics(outputs[0], figure_dir),
                PLOT_MODULE.plot_per_iou(outputs[0], figure_dir),
                PLOT_MODULE.plot_stage_coverage(outputs[0], figure_dir),
                PLOT_MODULE.plot_attribution(outputs[0], figure_dir),
                PLOT_MODULE.plot_buckets(outputs[0], figure_dir),
                PLOT_MODULE.plot_global_pr(outputs[0], figure_dir),
            )
            self.assertTrue(all(path.stat().st_size > 1000 for path in figures))

    def test_global_pr_reconstruction_matches_replay_evaluator(self):
        from faster_coco_eval import COCO
        from tools.table_cells.coco_match_npz import export_npz_matches, reconstruct_global_pr
        from tools.table_cells.full_val_replay import compact_coco_metrics

        dataset = {
            "images": [
                {"id": 1, "width": 100, "height": 100, "file_name": "val/wired/one.png"},
                {"id": 2, "width": 100, "height": 100, "file_name": "val/wireless/two.png"},
            ],
            "categories": [{"id": 1, "name": "cell"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 20, 10, 10], "area": 100, "iscrowd": 0},
            ],
        }
        boxes = np.full((2, 640, 4), [80, 80, 1, 1], dtype=np.float32)
        boxes[0, 0] = [10, 10, 10, 10]
        boxes[1, 1] = [20, 20, 10, 10]
        scores = np.tile(np.linspace(0.8, 0.001, 640, dtype=np.float32), (2, 1))
        # Deliberate TP/FP ties within and across images exercise the proof
        # that one global stable sort followed by filtering is equivalent to
        # filtering each status slice before its own stable sort.
        scores[0, 0] = 0.90  # TP, first in image-major/query order
        scores[0, 2] = 0.90  # FP tied with both TPs
        scores[1, 1] = 0.90  # TP in the second image
        scores[1, 2] = 0.90  # another tied FP
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotation_path = root / "val.json"
            prediction_path = root / "predictions.npz"
            write_json(annotation_path, dataset)
            np.savez_compressed(
                prediction_path,
                image_ids=np.asarray([1, 2], dtype=np.int64),
                boxes_xywh=boxes,
                scores=scores,
                category_id=np.asarray(1, dtype=np.int64),
                input_size=np.asarray([640, 640], dtype=np.int32),
                state=np.asarray("checkpoint0028:ema"),
                query_indices=np.arange(640, dtype=np.uint16),
            )
            reference = compact_coco_metrics(COCO(dataset), prediction_path, batch_size=1)
            expected = {}
            for budget in (300, 640):
                for row in reference["small_per_iou"][budget]:
                    expected[(28, "ema", budget, round(float(row["iou"]), 2))] = {
                        "ap_small": row["ap"],
                        "ar_small": row["ar"],
                    }
            match_dir = root / "match"
            export_npz_matches(
                annotation_path,
                prediction_path,
                match_dir,
                max_dets=(300, 640),
                iou_thresholds=MODULE.DEFAULT_THRESHOLDS,
                images_per_shard=1,
            )
            record = MODULE.ReplayState(
                epoch=28,
                state="ema",
                directory=root,
                summary_path=root / "summary.json",
                metadata_path=root / "metadata.npz",
                prediction_path=prediction_path,
                stage_path=root / "stages.npz",
                summary={},
                result={"images": 2},
            )
            pr, separation, histogram, reconciliations = MODULE.compute_global_score_pr(
                record,
                match_dir,
                MODULE.DEFAULT_THRESHOLDS,
                (300, 640),
                expected,
                {},
                root,
            )
            self.assertEqual(reconciliations, 20)
            self.assertEqual(len(pr), 2 * 10 * 2 * 101)
            self.assertEqual(len(separation), 2 * 10 * 2)
            self.assertEqual(len(histogram), 2 * 10 * 2 * 100)
            small_rows = [row for row in separation if row["scope"] == "coco_small_eval"]
            self.assertTrue(all(abs(float(row["ap_reconciliation_delta"])) < 1e-8 for row in small_rows))

            # Every persisted-status slice must agree with the matcher's
            # independent reconstruction helper, not just the canonical
            # production cross-check slice.
            indexed = {
                (row["scope"], float(row["iou_threshold"]), int(row["max_dets"])): row
                for row in separation
            }
            for scope in ("all", "coco_small_eval"):
                for threshold in MODULE.DEFAULT_THRESHOLDS:
                    for budget in (300, 640):
                        helper = reconstruct_global_pr(
                            match_dir,
                            scope=scope,
                            iou_threshold=threshold,
                            max_det=budget,
                        )
                        observed = indexed[(scope, threshold, budget)]
                        self.assertAlmostEqual(helper["ap"], observed["ap_from_global_pr"], places=12)
                        self.assertAlmostEqual(helper["ar"], observed["ar_from_global_pr"], places=12)
                        self.assertEqual(helper["num_tp"], observed["tp_count"])
                        self.assertEqual(helper["num_fp"], observed["fp_count"])

            # Keep the legacy evaluator replay as a compatibility fallback
            # and require it to match the persisted-status route.
            _legacy_pr, legacy_separation, _legacy_histogram, legacy_reconciliations = (
                MODULE.compute_global_score_pr_from_evaluator(
                    record,
                    match_dir,
                    MODULE.DEFAULT_THRESHOLDS,
                    (300, 640),
                    expected,
                    {},
                    root,
                )
            )
            self.assertEqual(legacy_reconciliations, 20)
            legacy_indexed = {
                (row["scope"], float(row["iou_threshold"]), int(row["max_dets"])): row
                for row in legacy_separation
            }
            for key, observed in indexed.items():
                self.assertAlmostEqual(
                    observed["ap_from_global_pr"],
                    legacy_indexed[key]["ap_from_global_pr"],
                    places=12,
                )
                self.assertAlmostEqual(
                    observed["ar_from_global_pr"],
                    legacy_indexed[key]["ar_from_global_pr"],
                    places=12,
                )

            MODULE.match_shards(match_dir, formal=True)
            with (match_dir / "gt_matches.part0000.csv.gz").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(
                ValueError, "(rooted GT shard identity|GT shard SHA-256) mismatch"
            ):
                MODULE.match_shards(match_dir, formal=True)

    def test_formal_fail_closed_and_replay_hash_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            replays, matches = create_fixture(root)
            common = [
                *(item for epoch, path in replays for item in ("--replay", f"{epoch}={path}")),
                *(item for (epoch, state), path in matches for item in ("--match", f"{epoch}:{state}={path}")),
                "--output",
                str(root / "formal_rejected"),
                "--pair",
                "28:29",
                "--thresholds",
                "0.5",
                "0.75",
            ]
            with self.assertRaisesRegex(ValueError, "formal analysis requires epochs"):
                MODULE.run_analysis(MODULE.parse_args(common))

            stage_path = replays[0][1] / "val.raw.gt_stages.npz"
            with stage_path.open("ab") as handle:
                handle.write(b"corruption")
            testing_args = MODULE.parse_args(
                [*common, "--output", str(root / "hash_rejected"), "--testing", "--skip-global-pr"]
            )
            with self.assertRaisesRegex(ValueError, "stage artifact (byte-size|SHA-256) mismatch"):
                MODULE.run_analysis(testing_args)

    def test_match_matrix_binds_predictions_completions_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix_root = root / "matches"
            matrix_root.mkdir()
            match_dirs = {}
            records = {}
            prediction_inputs = {}
            matrix_states = {}
            checkpoint_contract = {}
            for epoch in MODULE.FORMAL_EPOCHS:
                replay_dir = root / "replays" / f"checkpoint{epoch:04d}"
                replay_dir.mkdir(parents=True)
                summary_path = replay_dir / "summary.json"
                checkpoint_hash = f"{epoch:064x}"[-64:]
                write_json(summary_path, {"checkpoint_sha256": checkpoint_hash})
                checkpoint_contract[f"checkpoint{epoch:04d}.pth"] = {
                    "file_sha256": checkpoint_hash
                }
                for state in MODULE.DEFAULT_STATES:
                    key = f"e{epoch}/{state}"
                    prediction_hash = MODULE.hashlib.sha256(key.encode()).hexdigest()
                    marker = replay_dir / f"val.{state}.complete.json"
                    write_json(marker, {"state": state})
                    record = SimpleNamespace(
                        summary={"checkpoint_sha256": checkpoint_hash},
                        result={"artifacts": {"predictions": {"sha256": prediction_hash}}},
                        summary_path=summary_path,
                        directory=replay_dir,
                    )
                    records[(epoch, state)] = record
                    match_dir = matrix_root / f"checkpoint{epoch:04d}" / state
                    match_dir.mkdir(parents=True)
                    complete_path = match_dir / "complete.json"
                    write_json(
                        complete_path,
                        {"gt_shard_manifest_sha256": MODULE.hashlib.sha256((key + "gt").encode()).hexdigest()},
                    )
                    match_dirs[(epoch, state)] = match_dir
                    prediction_inputs[key] = {
                        "prediction_sha256": prediction_hash,
                        "replay_summary_sha256": MODULE.sha256(summary_path),
                        "replay_state_marker_sha256": MODULE.sha256(marker),
                    }
                    matrix_states[key] = {
                        "complete_sha256": MODULE.sha256(complete_path),
                        "gt_shard_manifest_sha256": MODULE.read_json(complete_path)[
                            "gt_shard_manifest_sha256"
                        ],
                    }
            formal_identity = {
                "config_sha256": "c" * 64,
                "annotation_sha256": "a" * 64,
                "inference_identity": {
                    "device": "cpu",
                    "torch_version": "test",
                    "torch_cuda_version": None,
                },
            }
            base_matrix = {
                "schema": "sgqs_coco_match_matrix_orchestration_v1",
                "status": "complete",
                "epochs": list(MODULE.FORMAL_EPOCHS),
                "max_dets": [300, 640],
                "iou_thresholds": list(MODULE.DEFAULT_THRESHOLDS),
                "annotation_sha256": formal_identity["annotation_sha256"],
                "config_sha256": formal_identity["config_sha256"],
                "replay_runtime_backend": {
                    "requested_device": "cpu",
                    "torch_version": "test",
                    "torch_cuda_version": None,
                },
                "matcher_runtime": {"numpy_version": np.__version__},
                "checkpoint_contract": checkpoint_contract,
                "prediction_inputs": prediction_inputs,
                "states": matrix_states,
            }
            matrix_path = matrix_root / "matrix.complete.json"
            write_json(matrix_path, base_matrix)
            MODULE.validate_match_matrix_audit(
                match_dirs, records, formal_identity, formal=True
            )

            corruptions = (
                ("prediction_inputs", "e28/raw", "prediction_sha256"),
                ("states", "e28/raw", "complete_sha256"),
                ("checkpoint_contract", "checkpoint0028.pth", "file_sha256"),
            )
            for section, key, field in corruptions:
                changed = json.loads(json.dumps(base_matrix))
                changed[section][key][field] = "f" * 64
                write_json(matrix_path, changed)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    MODULE.validate_match_matrix_audit(
                        match_dirs, records, formal_identity, formal=True
                    )

    def test_legacy_val128_rule_reproduces_documented_attribution(self):
        repository = SCRIPT.parents[4]
        root = repository / "_local" / "server_sgqs_probe_20261014"
        before_path = root / "replay_val128_e0028" / "val.ema.gt.jsonl"
        after_path = root / "replay_val128_e0029" / "val.ema.gt.jsonl"
        if not before_path.is_file() or not after_path.is_file():
            self.skipTest("optional private val128 replay is absent")

        def load(path: Path) -> dict[int, dict]:
            with path.open(encoding="utf-8") as handle:
                return {int(row["annotation_id"]): row for row in map(json.loads, handle)}

        before, after = load(before_path), load(after_path)
        selected = [annotation_id for annotation_id, row in before.items() if float(row["coco_area"]) < 32**2]
        fields = (
            "encoder_dense_best_iou_all",
            "encoder_dense_best_iou_top640",
            "decoder_best_iou_all",
            "decoder_best_iou_top300",
        )

        def cause(row: dict) -> str:
            code = 0
            for stage_code, field in enumerate(fields):
                if float(row[field]) >= 0.75:
                    code = stage_code + 1
            return MODULE.CAUSES[code]

        gains = [annotation_id for annotation_id in selected if before[annotation_id][fields[-1]] < 0.75 <= after[annotation_id][fields[-1]]]
        losses = [annotation_id for annotation_id in selected if after[annotation_id][fields[-1]] < 0.75 <= before[annotation_id][fields[-1]]]
        self.assertEqual(Counter(cause(before[item]) for item in gains), Counter({"rank_gt300": 49}))
        self.assertEqual(
            Counter(cause(after[item]) for item in losses),
            Counter({"encoder_selection": 15, "rank_gt300": 2}),
        )


if __name__ == "__main__":
    unittest.main()
