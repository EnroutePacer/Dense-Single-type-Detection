"""Compact full-validation replay schema, math, and resume tests."""

import gc
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from faster_coco_eval import COCO
from PIL import Image

from tests.test_supervision_probe import fixture
from tools.table_cells.full_val_replay import (
    CaptureCandidates,
    IOU_THRESHOLDS,
    build_layout,
    compact_coco_metrics,
    flush_progress,
    open_staging,
    prediction_specs,
    ranked_gt_diagnostics,
    replay_state,
    stage_specs,
    _verify_embedded_probe_identity,
)
from tools.table_cells.replay_probe import coco_metrics


class FullValReplayTest(unittest.TestCase):
    def test_embedded_micro_probe_identity_is_verified_and_compacted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "config.yml"
            config.write_text("criterion: DEIMCriterion\n", encoding="utf-8")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            manifest_hash = "a" * 64
            checkpoint = {
                "micro_probe_identity": {
                    "schema": "sgqs_single_gpu_micro_probe_v1",
                    "arm": "r5",
                    "config_sha256": config_hash,
                    "resolved_config_sha256": "b" * 64,
                    "manifest_sha256": manifest_hash,
                    "target_strata_coverage_status": "pass",
                }
            }
            result = _verify_embedded_probe_identity(
                checkpoint,
                config=config,
                source_identity={"manifest_sha256": manifest_hash},
            )
            self.assertTrue(result["verified_against_replay_inputs"])
            self.assertEqual(result["verified_config_sha256"], config_hash)
            self.assertEqual(result["verified_manifest_sha256"], manifest_hash)

    def test_embedded_micro_probe_identity_rejects_wrong_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / "config.yml"
            config.write_text("criterion: DEIMCriterion\n", encoding="utf-8")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "manifest hash"):
                _verify_embedded_probe_identity(
                    {
                        "micro_probe_identity": {
                            "config_sha256": config_hash,
                            "manifest_sha256": "a" * 64,
                        }
                    },
                    config=config,
                    source_identity={"manifest_sha256": "b" * 64},
                )

    def test_capture_preserves_forward_and_records_real_membership(self):
        model, _, _ = fixture()
        model.eval()
        features = [torch.randn(1, 32, side, side) for side in (4, 2, 1)]
        with torch.no_grad():
            expected = model(features)
            with CaptureCandidates(model) as capture:
                actual = model(features)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        self.assertEqual(capture.data["selected_indices"].shape, (1, 4))
        self.assertEqual(capture.data["selected_boxes"].shape, (1, 4, 4))
        selected = capture.data["boxes"].gather(
            1, capture.data["selected_indices"].unsqueeze(-1).expand(-1, -1, 4)
        )
        # The actual selected branch runs its own bbox-head GEMM, so only
        # numerical closeness to head(all)[selected] is expected.
        torch.testing.assert_close(selected, capture.data["selected_boxes"], rtol=1e-6, atol=1e-7)

    def test_ranked_diagnostics_records_best_and_first_qualified(self):
        gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        boxes = torch.tensor(
            [
                [0.1, 0.1, 0.2, 0.2],
                [0.5, 0.5, 0.2, 0.2],
                [0.5, 0.5, 0.1, 0.2],
            ]
        )
        scores = torch.tensor([0.9, 0.2, 0.8])
        result = ranked_gt_diagnostics(
            boxes, scores, gt, candidate_indices=torch.tensor([10, 11, 12])
        )
        self.assertAlmostEqual(float(result["best_iou"][0]), 1.0, places=6)
        self.assertEqual(int(result["best_iou_rank"][0]), 3)
        self.assertAlmostEqual(float(result["best_iou_score"][0]), 0.2, places=6)
        self.assertEqual(int(result["best_iou_candidate_index"][0]), 11)
        self.assertEqual(int(result["first_rank_iou0.5"][0]), 2)
        self.assertAlmostEqual(float(result["first_score_iou0.5"][0]), 0.8, places=6)
        self.assertEqual(int(result["first_candidate_index_iou0.5"][0]), 12)
        self.assertEqual(int(result["first_rank_iou0.75"][0]), 3)
        self.assertEqual(int(result["first_rank_iou0.95"][0]), 3)
        self.assertAlmostEqual(float(result["first_score_iou0.95"][0]), 0.2, places=6)

        missed = ranked_gt_diagnostics(boxes[:1], scores[:1], gt)
        self.assertEqual(int(missed["first_rank_iou0.5"][0]), 2)
        self.assertTrue(torch.isnan(missed["first_score_iou0.5"][0]))
        self.assertEqual(int(missed["first_candidate_index_iou0.5"][0]), -1)

        # Encoder logits can both round to probability 1.0. Membership/rank
        # must still follow the raw selector logits while reported scores are
        # probabilities.
        saturated = ranked_gt_diagnostics(
            torch.stack((boxes[1], boxes[1])),
            torch.tensor([1.0, 1.0]),
            gt,
            candidate_indices=torch.tensor([20, 21]),
            ranking_scores=torch.tensor([20.0, 30.0]),
        )
        self.assertEqual(int(saturated["first_candidate_index_iou0.75"][0]), 21)

    def test_layout_keeps_coco_and_local_hybrid_small_separate(self):
        dataset = {
            "images": [{"id": 1, "width": 2000, "height": 2000, "file_name": "x.png"}],
            "categories": [{"id": 1, "name": "cell"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 100], "area": 2000, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 31, 31], "area": 961, "iscrowd": 0},
                {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 4, "image_id": 1, "category_id": 1, "bbox": [0, 0, 200, 200], "area": 40000, "iscrowd": 0},
            ],
        }
        layout = build_layout(COCO(dataset))
        # On a 2000x2000 source, id=1 is hybrid-only through short side;
        # ids 2 and 3 are both; id=4 is neither.
        self.assertEqual(layout["coco_small"].tolist(), [False, True, True, False])
        self.assertEqual(layout["local_hybrid_small"].tolist(), [True, True, True, False])
        self.assertEqual(layout["small_definition_membership"].tolist(), [2, 0, 0, 3])

    def test_compact_streaming_coco_matches_legacy_evaluator(self):
        dataset = {
            "images": [
                {"id": 1, "width": 100, "height": 100},
                {"id": 2, "width": 100, "height": 100},
            ],
            "categories": [{"id": 1, "name": "cell"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 0},
            ],
        }
        coco = COCO(dataset)
        boxes = np.asarray(
            [
                [[40, 40, 10, 10], [0, 0, 2, 2], [60, 60, 2, 2], [10, 10, 2, 2]],
                [[0, 0, 2, 2], [20, 20, 20, 20], [60, 60, 2, 2], [10, 10, 2, 2]],
            ],
            dtype=np.float32,
        )
        scores = np.asarray([[0.9, 0.8, 0.7, 0.6], [0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
        predictions = [
            {
                "image_id": image_id,
                "category_id": 1,
                "bbox": box.tolist(),
                "score": float(score),
            }
            for image_index, image_id in enumerate((1, 2))
            for box, score in zip(boxes[image_index], scores[image_index])
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "predictions.npz"
            np.savez_compressed(
                path,
                image_ids=np.asarray([1, 2], dtype=np.int64),
                boxes_xywh=boxes,
                scores=scores,
                category_id=np.asarray(1, dtype=np.int64),
            )
            compact = compact_coco_metrics(coco, path, batch_size=1)
        for budget in (300, 640):
            expected = coco_metrics(coco, predictions, budget)
            np.testing.assert_allclose(compact[budget], expected, rtol=0, atol=1e-12)
            rows = compact["small_per_iou"][budget]
            np.testing.assert_allclose(
                [row["iou"] for row in rows], IOU_THRESHOLDS, rtol=0, atol=1e-12
            )
            self.assertAlmostEqual(
                float(np.mean([row["ap"] for row in rows])), compact[budget][3], places=12
            )
            self.assertAlmostEqual(
                float(np.mean([row["ar"] for row in rows])), compact[budget][9], places=12
            )

    def test_memmap_progress_reopens_from_durable_image(self):
        identity = {"checkpoint": "abc", "state": "ema"}
        specs = prediction_specs(2, 4)
        specs.update(stage_specs(3, 2))
        self.assertEqual(
            specs["image_independent_coverage"][1],
            (2, 7, len(IOU_THRESHOLDS), 4),
        )
        self.assertIn("decoder_top300__first_rank_iou0.95", specs)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / ".val.ema.partial"
            arrays, start = open_staging(directory, specs, identity, resume=False)
            self.assertEqual(start, 0)
            arrays["scores"][0, 0] = 0.75
            flush_progress(directory, arrays, identity, 1)
            arrays.clear()
            gc.collect()
            arrays, start = open_staging(directory, specs, identity, resume=True)
            self.assertEqual(start, 1)
            self.assertAlmostEqual(float(arrays["scores"][0, 0]), 0.75, places=6)
            progress = json.loads((directory / "progress.json").read_text())
            self.assertEqual(progress["completed_images"], 1)
            arrays.clear()
            gc.collect()

    def test_end_to_end_compact_artifacts_and_complete_marker(self):
        class FakeDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.query_select_method = "default"
                self.num_queries = 4
                self.num_classes = 1
                self.enc_bbox_head = torch.nn.Identity()

            def _select_topk(self, memory, logits, anchors, k):
                _, indexes = torch.topk(logits[..., 0], k, dim=-1)
                return (
                    memory.gather(1, indexes.unsqueeze(-1).expand(-1, -1, 4)),
                    None,
                    anchors.gather(1, indexes.unsqueeze(-1).expand(-1, -1, 4)),
                )

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = FakeDecoder()
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                batch = len(images)
                dense_boxes = torch.tensor(
                    [
                        [0.45, 0.45, 0.10, 0.10],
                        [0.10, 0.10, 0.05, 0.05],
                        [0.80, 0.80, 0.05, 0.05],
                        [0.20, 0.80, 0.05, 0.05],
                        [0.80, 0.20, 0.05, 0.05],
                    ]
                ).repeat(batch, 1, 1)
                anchors = torch.logit(dense_boxes)
                memory = torch.zeros_like(anchors)
                logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]).reshape(1, 5, 1).repeat(batch, 1, 1)
                selected_memory, _, selected_anchors = self.decoder._select_topk(
                    memory, logits, anchors, self.decoder.num_queries
                )
                return {
                    "pred_boxes": (selected_memory + selected_anchors).sigmoid(),
                    "pred_logits": logits[:, : self.decoder.num_queries],
                }

        dataset = {
            "images": [{"id": 1, "width": 100, "height": 100, "file_name": "one.png"}],
            "categories": [{"id": 1, "name": "cell"}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [40, 40, 10, 10], "area": 100, "iscrowd": 0}
            ],
        }
        coco = COCO(dataset)
        layout = build_layout(coco)
        model = FakeModel().eval()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.new("RGB", (100, 100), "white").save(root / "one.png")
            output = root / "out"
            output.mkdir()
            identity = {
                "run_identity": {"checkpoint": str(root / "checkpoint0028.pth")},
                "state": "ema",
            }
            result = replay_state(
                model,
                coco,
                layout,
                root,
                output,
                "val.ema",
                identity,
                device="cpu",
                batch_size=1,
                durable_images=1,
                progress_every=0,
                eval_batch_size=1,
                resume=False,
                keep_staging=False,
                legacy_reference=None,
                legacy_atol=1e-6,
            )
            self.assertEqual(model.calls, 1)
            self.assertTrue((output / "val.ema.complete.json").is_file())
            self.assertFalse((output / ".val.ema.partial").exists())
            self.assertEqual(result["images"], 1)
            with np.load(output / "val.ema.predictions.npz", allow_pickle=False) as predictions:
                np.testing.assert_allclose(
                    predictions["boxes_xywh"][0, 0], [40, 40, 10, 10], atol=1e-5
                )
                self.assertEqual(str(predictions["box_format"]), "xywh_abs_original_pixels")
            with np.load(output / "val.ema.gt_stages.npz", allow_pickle=False) as stages:
                self.assertAlmostEqual(
                    float(stages["decoder_top300__best_iou"][0]), 1.0, places=5
                )
                self.assertEqual(len(stages["iou_thresholds"]), 10)
                self.assertEqual(
                    int(stages["decoder_top300__first_rank_iou0.95"][0]), 1
                )
            self.assertEqual(len(result["coco_small_per_iou_maxdets300"]), 10)

            replay_state(
                model,
                coco,
                layout,
                root,
                output,
                "val.ema",
                identity,
                device="cpu",
                batch_size=1,
                durable_images=1,
                progress_every=0,
                eval_batch_size=1,
                resume=True,
                keep_staging=False,
                legacy_reference=None,
                legacy_atol=1e-6,
            )
            self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
