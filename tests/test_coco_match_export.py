import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from faster_coco_eval import COCO

from tools.table_cells.coco_match_export import (
    load_predictions,
    match_coco_predictions,
    write_match_result,
)
from tools.table_cells.coco_match_npz import export_npz_matches, reconstruct_global_pr
from tools.table_cells.full_val_replay import IOU_THRESHOLDS, compact_coco_metrics


def dataset(images=None, annotations=None):
    return {
        "images": images or [{"id": 1, "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "cell"}],
        "annotations": annotations or [],
    }


def prediction(bbox, score, query_index, image_id=1):
    return {
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
        "score": score,
        "query_index": query_index,
    }


class CocoMatchExportTest(unittest.TestCase):
    def test_budget_rank_and_query_are_exported(self):
        annotations = [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
        ]
        predictions = [
            prediction([80, 80, 5, 5], 0.99, 20),
            prediction([10, 10, 10, 10], 0.90, 21),
            prediction([50, 50, 10, 10], 0.80, 22),
        ]
        result = match_coco_predictions(
            dataset(annotations=annotations),
            predictions,
            max_dets=(2, 3),
            iou_thresholds=(0.5, 0.75),
            state="ema",
        )

        gt = {row["annotation_id"]: row for row in result.ground_truth_rows}
        self.assertEqual(gt[10]["matches"]["iou0.75"]["maxDets2"]["query_index"], 21)
        self.assertEqual(gt[10]["matches"]["iou0.75"]["maxDets2"]["rank"], 2)
        self.assertEqual(gt[11]["matches"]["iou0.75"]["maxDets2"]["status"], "fn")
        self.assertEqual(gt[11]["matches"]["iou0.75"]["maxDets3"]["query_index"], 22)
        self.assertEqual(
            result.detection_rows[2]["matches"]["iou0.75"]["maxDets2"]["status"],
            "truncated",
        )

        overall = {
            (row["iou_threshold"], row["max_dets"]): row
            for row in result.summary_rows
            if row["unit"] == "overall"
        }
        self.assertEqual((overall[(0.75, 2)]["tp"], overall[(0.75, 2)]["fp"], overall[(0.75, 2)]["fn"]), (1, 1, 1))
        self.assertEqual((overall[(0.75, 3)]["tp"], overall[(0.75, 3)]["fp"], overall[(0.75, 3)]["fn"]), (2, 1, 0))

    def test_duplicate_gt_coverage_is_not_duplicate_true_positive(self):
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
        ]
        result = match_coco_predictions(
            dataset(annotations=annotations),
            [prediction([10, 10, 10, 10], 0.9, 7)],
            max_dets=1,
            iou_thresholds=0.5,
        )
        statuses = [row["matches"]["iou0.50"]["maxDets1"]["status"] for row in result.ground_truth_rows]
        self.assertEqual(statuses.count("tp"), 1)
        self.assertEqual(statuses.count("fn"), 1)
        self.assertIn("one-to-one", result.metadata["coverage_warning"])

    def test_local_small_partition_uses_resized_area_or_short_side(self):
        images = [
            {"id": 1, "width": 100, "height": 100},
            {"id": 2, "width": 1000, "height": 1000},
        ]
        annotations = [
            # COCO-small in original pixels, but large after 640x640 resize.
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 30, 30], "area": 900, "iscrowd": 0},
            # Not COCO-small, but short/area-small after 640x640 resize.
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 200, 10], "area": 2000, "iscrowd": 0},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [300, 300, 10, 10], "area": 100, "iscrowd": 0},
        ]
        result = match_coco_predictions(
            dataset(images=images, annotations=annotations),
            [],
            max_dets=300,
            iou_thresholds=0.75,
        )
        rows = {row["annotation_id"]: row for row in result.ground_truth_rows}
        self.assertEqual(rows[1]["small_partition"], "coco_only")
        self.assertEqual(rows[2]["small_partition"], "hybrid_only")
        self.assertEqual(rows[3]["small_partition"], "coco_and_hybrid")
        self.assertAlmostEqual(rows[2]["input_short_px"], 6.4)

        cohort_rows = {
            row["group"]: row
            for row in result.summary_rows
            if row["unit"] == "ground_truth"
            and row["axis"] == "small_cohort"
            and row["iou_threshold"] == 0.75
            and row["max_dets"] == 300
        }
        self.assertEqual(cohort_rows["coco_small"]["objects"], 2)
        self.assertEqual(cohort_rows["hybrid_small"]["objects"], 2)
        self.assertIsNone(cohort_rows["coco_small"]["fp"])

    def test_crowd_match_is_ignored_and_ties_keep_input_order(self):
        annotations = [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "area": 400, "iscrowd": 1},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
        ]
        result = match_coco_predictions(
            dataset(annotations=annotations),
            [
                prediction([0, 0, 10, 10], 0.5, 100),
                prediction([50, 50, 10, 10], 0.5, 101),
            ],
            max_dets=2,
            iou_thresholds=0.5,
        )
        self.assertEqual([row["rank"] for row in result.detection_rows], [1, 2])
        self.assertEqual(result.detection_rows[0]["matches"]["iou0.50"]["maxDets2"]["status"], "ignored")
        self.assertEqual(result.detection_rows[1]["matches"]["iou0.50"]["maxDets2"]["status"], "tp")

    def test_jsonl_loader_and_writers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prediction_path = root / "predictions.jsonl"
            prediction_path.write_text(
                json.dumps(prediction([10, 10, 10, 10], 0.9, 4)) + "\n",
                encoding="utf-8",
            )
            result = match_coco_predictions(
                dataset(
                    annotations=[
                        {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0}
                    ]
                ),
                load_predictions(prediction_path),
                max_dets=1,
                iou_thresholds=0.5,
                state="raw",
            )
            paths = write_match_result(result, root / "out")
            self.assertEqual(set(paths), {"ground_truth", "detections", "summary", "metadata"})
            metadata = json.loads(Path(paths["metadata"]).read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema"], "coco_one_to_one_matches_v1")

    def test_compact_npz_export_is_sharded_resumable_and_area_aware(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    dataset(
                        images=[
                            {"id": 1, "width": 100, "height": 100},
                            {"id": 2, "width": 100, "height": 100},
                        ],
                        annotations=[
                            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},
                            {"id": 11, "image_id": 1, "category_id": 1, "bbox": [50, 50, 40, 40], "area": 1600, "iscrowd": 0},
                            {"id": 12, "image_id": 1, "category_id": 1, "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 1},
                        ],
                    )
                ),
                encoding="utf-8",
            )
            prediction_path = root / "predictions.npz"
            np.savez_compressed(
                prediction_path,
                image_ids=np.asarray([1, 2], dtype=np.int64),
                boxes_xywh=np.asarray(
                    [
                        [[20, 20, 5, 5], [10, 10, 10, 10], [50, 50, 40, 40]],
                        [[0, 0, 5, 5], [10, 10, 5, 5], [20, 20, 5, 5]],
                    ],
                    dtype=np.float32,
                ),
                scores=np.asarray([[0.99, 0.9, 0.8], [0.7, 0.6, 0.5]], dtype=np.float32),
                category_id=np.asarray(1, dtype=np.int64),
                input_size=np.asarray([640, 640], dtype=np.int64),
                state=np.asarray("ema"),
            )
            output = root / "matches"
            complete = export_npz_matches(
                annotation_path,
                prediction_path,
                output,
                max_dets=(2, 3),
                iou_thresholds=(0.5, 0.75),
                images_per_shard=1,
            )
            self.assertTrue(complete.is_file())
            self.assertEqual(len(list(output.glob("gt_matches.part*.csv.gz"))), 2)
            self.assertEqual(len(list(output.glob("detection_status.part*.npz"))), 2)
            complete_payload = json.loads(complete.read_text(encoding="utf-8"))
            gt_manifest_path = output / complete_payload["gt_shard_manifest"]
            self.assertEqual(
                complete_payload["gt_shard_manifest_sha256"],
                hashlib.sha256(gt_manifest_path.read_bytes()).hexdigest(),
            )
            gt_manifest = json.loads(gt_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(gt_manifest["schema"], "coco_gt_match_shard_manifest_v1")
            self.assertEqual(len(gt_manifest["shards"]), 2)
            self.assertEqual(
                gt_manifest["shards"][0]["gt_matches_file"],
                "gt_matches.part0000.csv.gz",
            )
            self.assertEqual(
                gt_manifest["shards"][0]["summary_file"],
                "summary.part0000.json",
            )

            with gzip.open(output / "gt_matches.part0000.csv.gz", "rt", encoding="utf-8", newline="") as handle:
                rows = {int(row["annotation_id"]): row for row in csv.DictReader(handle)}
            self.assertEqual(rows[10]["all_iou75_status_d2"], "tp")
            self.assertEqual(rows[10]["all_iou75_query_index"], "1")
            self.assertEqual(rows[11]["all_iou75_status_d2"], "fn")
            self.assertEqual(rows[11]["all_iou75_status_d3"], "tp")
            self.assertEqual(rows[11]["all_iou75_rank"], "3")
            self.assertEqual(rows[11]["coco_small_eval_iou75_status_d3"], "ignored")
            self.assertEqual(rows[12]["all_iou75_status_d3"], "ignored")

            with (output / "match_aggregates.csv").open("r", encoding="utf-8", newline="") as handle:
                aggregate = list(csv.DictReader(handle))
            overall = next(
                row
                for row in aggregate
                if row["unit"] == "overall"
                and row["scope"] == "all"
                and row["iou_threshold"] == "0.75"
                and row["max_dets"] == "2"
            )
            self.assertEqual((overall["tp"], overall["fp"], overall["fn"]), ("1", "2", "1"))
            self.assertEqual(overall["ignored"], "1")

            with np.load(
                output / "detection_status.part0000.npz", allow_pickle=False
            ) as status:
                self.assertEqual(
                    status["status"].shape,
                    (2, 2, 2, 1, 3),
                )
                self.assertEqual(status["query_rank"][0].tolist(), [1, 2, 3])
                # all / IoU=.75 / image 1 in query-index order
                self.assertEqual(status["status"][0, 0, 1, 0].tolist(), [3, 0, 4])
                self.assertEqual(status["status"][1, 0, 1, 0].tolist(), [3, 0, 0])
                # Official small scope includes only annotation 10; annotation
                # 11 and its matching detection are ignored.
                self.assertEqual(status["nonignored_gt_count"][:, 0].tolist(), [2, 1])

            curve = reconstruct_global_pr(
                output, scope="all", iou_threshold=0.75, max_det=2
            )
            self.assertEqual(curve["num_positive"], 2)
            self.assertEqual(
                (curve["num_tp"], curve["num_fp"], curve["num_ignored"], curve["num_truncated"]),
                (1, 2, 1, 2),
            )
            np.testing.assert_allclose(curve["event_scores"], [0.9, 0.7, 0.6])
            self.assertEqual(curve["event_is_tp"].tolist(), [True, False, False])
            self.assertAlmostEqual(curve["ap"], 51 / 101, places=12)
            self.assertAlmostEqual(curve["ar"], 0.5, places=12)

            # A second invocation must validate the identical protocol and
            # consume the completed shard summaries instead of recomputing.
            complete_again = export_npz_matches(
                annotation_path,
                prediction_path,
                output,
                max_dets=(2, 3),
                iou_thresholds=(0.5, 0.75),
                images_per_shard=1,
            )
            self.assertEqual(complete_again, complete)

    def test_max_images_accepts_prediction_subset_of_full_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    dataset(
                        images=[
                            {"id": 1, "width": 100, "height": 100},
                            {"id": 2, "width": 100, "height": 100},
                        ],
                        annotations=[
                            {
                                "id": 10,
                                "image_id": 1,
                                "category_id": 1,
                                "bbox": [10, 10, 10, 10],
                                "area": 100,
                                "iscrowd": 0,
                            },
                            {
                                "id": 11,
                                "image_id": 2,
                                "category_id": 1,
                                "bbox": [20, 20, 10, 10],
                                "area": 100,
                                "iscrowd": 0,
                            },
                        ],
                    )
                ),
                encoding="utf-8",
            )
            prediction_path = root / "predictions.npz"
            np.savez_compressed(
                prediction_path,
                image_ids=np.asarray([1], dtype=np.int64),
                boxes_xywh=np.asarray([[[10, 10, 10, 10]]], dtype=np.float32),
                scores=np.asarray([[0.9]], dtype=np.float32),
                category_id=np.asarray(1, dtype=np.int64),
            )
            with self.assertRaisesRegex(ValueError, "prediction/annotation image mismatch"):
                export_npz_matches(
                    annotation_path,
                    prediction_path,
                    root / "without_subset_mode",
                    max_dets=(1,),
                    iou_thresholds=(0.5,),
                    images_per_shard=1,
                )
            complete = export_npz_matches(
                annotation_path,
                prediction_path,
                root / "subset_mode",
                max_dets=(1,),
                iou_thresholds=(0.5,),
                images_per_shard=1,
                max_images=1,
            )
            payload = json.loads(complete.read_text(encoding="utf-8"))
            self.assertEqual(payload["images"], 1)
            protocol = json.loads(
                (root / "subset_mode" / "protocol.json").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["annotation_images"], 2)
            self.assertEqual(protocol["prediction_images"], 1)

    def test_reconstructed_small_pr_matches_coco_for_all_thresholds_and_budgets(self):
        """The compact status shards must reproduce official global COCO PR."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = dataset(
                images=[
                    {"id": 1, "width": 100, "height": 100},
                    {"id": 2, "width": 100, "height": 100},
                ],
                annotations=[
                    {
                        "id": 10,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    },
                    # A medium object exercises COCO's area-range ignore path.
                    {
                        "id": 11,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [50, 50, 40, 40],
                        "area": 1600,
                        "iscrowd": 0,
                    },
                    {
                        "id": 12,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [20, 20, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    },
                    {
                        "id": 13,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [70, 70, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    },
                ],
            )
            annotation_path = root / "annotations.json"
            annotation_path.write_text(json.dumps(payload), encoding="utf-8")

            query_count = 640
            boxes = np.zeros((2, query_count, 4), dtype=np.float32)
            boxes[:] = [0, 80, 2, 2]
            # Image 1: one small TP plus one detection that must be ignored in
            # the small scope because it matches a medium ground truth.
            boxes[0, 0] = [10, 10, 10, 10]
            boxes[0, 1] = [50, 50, 40, 40]
            # Image 2: the first candidate has IoU 0.70; the exact duplicate
            # behind it changes TP/FP identity above IoU 0.70.  A second GT is
            # deliberately below rank 300 to distinguish the two budgets.
            boxes[1, 0] = [20, 20, 20, 14]
            boxes[1, 1] = [20, 20, 20, 20]
            boxes[1, 350] = [70, 70, 10, 10]
            scores = np.repeat(
                np.linspace(0.999, 0.001, query_count, dtype=np.float32)[None],
                2,
                axis=0,
            )
            prediction_path = root / "predictions.npz"
            np.savez_compressed(
                prediction_path,
                image_ids=np.asarray([1, 2], dtype=np.int64),
                boxes_xywh=boxes,
                scores=scores,
                category_id=np.asarray(1, dtype=np.int64),
                input_size=np.asarray([640, 640], dtype=np.int64),
                state=np.asarray("ema"),
            )

            output = root / "matches"
            export_npz_matches(
                annotation_path,
                prediction_path,
                output,
                max_dets=(300, 640),
                iou_thresholds=IOU_THRESHOLDS,
                images_per_shard=1,
            )
            official = compact_coco_metrics(
                COCO(payload), prediction_path, batch_size=1
            )["small_per_iou"]
            for budget in (300, 640):
                for expected in official[budget]:
                    actual = reconstruct_global_pr(
                        output,
                        scope="coco_small_eval",
                        iou_threshold=expected["iou"],
                        max_det=budget,
                    )
                    self.assertAlmostEqual(actual["ap"], expected["ap"], places=12)
                    self.assertAlmostEqual(actual["ar"], expected["ar"], places=12)


if __name__ == "__main__":
    unittest.main()
