"""Low-resource synthetic tests for representative transition rendering."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).with_name("render_representative_transitions.py")
SPEC = importlib.util.spec_from_file_location("exp_diag_006_representatives", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

GATE_TEST_SCRIPT = Path(__file__).with_name("test_plot_full_val_gate.py")
GATE_SPEC = importlib.util.spec_from_file_location(
    "exp_diag_006_representative_gate_fixture", GATE_TEST_SCRIPT
)
GATE = importlib.util.module_from_spec(GATE_SPEC)
assert GATE_SPEC.loader is not None
sys.modules[GATE_SPEC.name] = GATE
GATE_SPEC.loader.exec_module(GATE)


def descriptor(path: Path, logical_name: str) -> dict:
    return {
        "logical_name": logical_name,
        "bytes": path.stat().st_size,
        "sha256": MODULE.formal_plot.sha256(path),
    }


def representative_row(
    *,
    start: int,
    end: int,
    transition: str,
    threshold: float,
    domain: str,
    image_id: int,
    annotation_id: int,
    file_name: str,
) -> dict[str, object]:
    gain = transition == "gain"
    before_tp = not gain
    after_tp = gain

    def value(tp: bool, present, missing=""):
        return present if tp else missing

    row: dict[str, object] = {
        "selection_rank": 1,
        "state": "ema",
        "epoch_start": start,
        "epoch_end": end,
        "match_scope": "coco_small_eval",
        "iou_threshold": threshold,
        "transition": transition,
        "cause": "candidate_absence",
        "small_partition": "coco_and_hybrid",
        "domain": domain,
        "annotation_id": annotation_id,
        "image_id": image_id,
        "file_name": file_name,
        "image_gt_count": 1,
        "evidence_strength": threshold,
        "rank_distance_from_300": "",
        "selection_rule": "synthetic strongest evidence",
        "status_before": "tp" if before_tp else "fn",
        "status_after": "tp" if after_tp else "fn",
        "formal_query_before": value(before_tp, 0),
        "formal_query_after": value(after_tp, 0),
        "formal_rank_before": value(before_tp, 1),
        "formal_rank_after": value(after_tp, 1),
        "formal_score_before": value(before_tp, 0.99),
        "formal_score_after": value(after_tp, 0.99),
        "formal_iou_before": value(before_tp, 1.0),
        "formal_iou_after": value(after_tp, 1.0),
    }
    for prefix in ("dense", "top640", "decoder_all", "decoder_top300"):
        row[f"{prefix}_iou_before"] = 1.0 if before_tp else 0.0
        row[f"{prefix}_iou_after"] = 1.0 if after_tp else 0.0
    row.update(
        {
            "dense_rank_before": 1 if before_tp else 8401,
            "dense_rank_after": 1 if after_tp else 8401,
            "top640_rank_before": 1 if before_tp else 641,
            "top640_rank_after": 1 if after_tp else 641,
            "decoder_all_rank_before": 1 if before_tp else 641,
            "decoder_all_rank_after": 1 if after_tp else 641,
            "decoder_top300_rank_before": 1 if before_tp else 301,
            "decoder_top300_rank_after": 1 if after_tp else 301,
        }
    )
    return row


def write_representatives(path: Path) -> None:
    rows = []
    images = (
        ("wired", 10, 101, "val/wired/a.png"),
        ("wireless", 20, 102, "val/wireless/b.png"),
    )
    for start, end, transition in MODULE.SHEET_SPECS:
        for threshold in MODULE.DISPLAY_THRESHOLDS:
            for domain, image_id, annotation_id, file_name in images:
                rows.append(
                    representative_row(
                        start=start,
                        end=end,
                        transition=transition,
                        threshold=threshold,
                        domain=domain,
                        image_id=image_id,
                        annotation_id=annotation_id,
                        file_name=file_name,
                    )
                )
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_prediction(path: Path, epoch: int, tp: bool) -> None:
    boxes = np.full((2, MODULE.QUERY_COUNT, 4), [0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    if tp:
        boxes[0, 0] = [50.0, 50.0, 20.0, 20.0]
        boxes[1, 0] = [100.0, 60.0, 10.0, 30.0]
    scores = np.tile(
        np.linspace(0.99, 0.01, MODULE.QUERY_COUNT, dtype=np.float32), (2, 1)
    )
    np.savez_compressed(
        path,
        schema=np.asarray("sgqs_compact_predictions_v1"),
        image_ids=np.asarray([10, 20], dtype=np.int64),
        boxes_xywh=boxes,
        scores=scores,
        category_id=np.asarray(1, dtype=np.int64),
        input_size=np.asarray([640, 640], dtype=np.int32),
        state=np.asarray(f"checkpoint{epoch:04d}:ema"),
        box_format=np.asarray("xywh_abs_original_pixels"),
        query_indices=np.arange(MODULE.QUERY_COUNT, dtype=np.uint16),
    )


def create_fixture(root: Path) -> tuple[Path, Path, Path, dict[int, Path]]:
    data_dir = root / "data"
    replay_root = root / "replays"
    images_root = root / "images"
    data_dir.mkdir()
    replay_root.mkdir()
    (images_root / "val" / "wired").mkdir(parents=True)
    (images_root / "val" / "wireless").mkdir(parents=True)
    Image.new("RGB", (200, 160), (238, 240, 244)).save(images_root / "val" / "wired" / "a.png")
    Image.new("RGB", (220, 180), (244, 240, 235)).save(images_root / "val" / "wireless" / "b.png")

    checkpoint28 = replay_root / "checkpoint0028"
    checkpoint28.mkdir()
    metadata_path = checkpoint28 / "val.metadata.npz"
    np.savez_compressed(
        metadata_path,
        schema=np.asarray("sgqs_full_val_metadata_v1"),
        input_size=np.asarray([640, 640], dtype=np.int32),
        image_ids=np.asarray([10, 20], dtype=np.int64),
        image_width=np.asarray([200, 220], dtype=np.int32),
        image_height=np.asarray([160, 180], dtype=np.int32),
        image_file_name=np.asarray(["val/wired/a.png", "val/wireless/b.png"]),
        image_gt_start=np.asarray([0, 1, 2], dtype=np.int64),
        annotation_ids=np.asarray([101, 102], dtype=np.int64),
        gt_image_index=np.asarray([0, 1], dtype=np.int32),
        gt_bbox_xywh_original=np.asarray(
            [[50.0, 50.0, 20.0, 20.0], [100.0, 60.0, 10.0, 30.0]], dtype=np.float32
        ),
    )

    prediction_paths: dict[int, Path] = {}
    for epoch, tp in ((28, False), (29, True), (30, True), (34, False)):
        checkpoint = replay_root / f"checkpoint{epoch:04d}"
        checkpoint.mkdir(exist_ok=True)
        prediction = checkpoint / "val.ema.predictions.npz"
        make_prediction(prediction, epoch, tp)
        prediction_paths[epoch] = prediction

    audit = GATE.formal_audit()
    audit["images"] = 2
    audit["noncrowd_annotations"] = 2
    audit["pairs"] = [[28, 29], [30, 34]]
    audit["input_identity"]["replays"]["e28"]["metadata"] = descriptor(
        metadata_path, "e28/val.metadata.npz"
    )
    for epoch, prediction in prediction_paths.items():
        audit["input_identity"]["replays"][f"e{epoch}"]["states"]["ema"][
            "predictions"
        ] = descriptor(prediction, f"e{epoch}/ema/val.ema.predictions.npz")
    GATE.write_fixture(data_dir, audit)
    write_representatives(data_dir / "representative_annotations.csv")
    GATE.rewrite_manifest(data_dir)
    return data_dir, replay_root, images_root, prediction_paths


class RepresentativeRendererTest(unittest.TestCase):
    def test_candidate_choice_reconciles_all_five_causes(self):
        gt_box = np.asarray([50.0, 50.0, 20.0, 20.0], dtype=np.float32)

        def prediction():
            boxes = np.full(
                (MODULE.QUERY_COUNT, 4),
                [0.0, 0.0, 5.0, 5.0],
                dtype=np.float32,
            )
            scores = np.linspace(
                0.99, 0.01, MODULE.QUERY_COUNT, dtype=np.float32
            )
            return boxes, scores

        def row(
            cause,
            *,
            decoder_rank,
            decoder_iou,
            top300_rank,
            top300_iou,
            formal_query="",
            formal_rank="",
            formal_score="",
            formal_iou="",
        ):
            return {
                "cause": cause,
                "iou_threshold": 0.5,
                "status_before": "fn",
                "formal_query_before": formal_query,
                "formal_rank_before": formal_rank,
                "formal_score_before": formal_score,
                "formal_iou_before": formal_iou,
                "decoder_all_rank_before": decoder_rank,
                "decoder_all_iou_before": decoder_iou,
                "decoder_top300_rank_before": top300_rank,
                "decoder_top300_iou_before": top300_iou,
            }

        for cause in ("candidate_absence", "encoder_selection", "decoder_geometry"):
            boxes, scores = prediction()
            scores[17] = 1.0
            candidate = MODULE.choose_display_candidate(
                row(
                    cause,
                    decoder_rank=MODULE.QUERY_COUNT + 1,
                    decoder_iou=0.0,
                    top300_rank=MODULE.MAX_DETS + 1,
                    top300_iou=0.0,
                ),
                "before",
                boxes,
                scores,
                gt_box,
            )
            self.assertEqual(
                (candidate.query, candidate.rank, candidate.kind),
                (17, 1, "best-IoU query; below threshold"),
            )

        boxes, scores = prediction()
        boxes[349] = gt_box
        rank_gt300 = MODULE.choose_display_candidate(
            row(
                "rank_gt300",
                decoder_rank=350,
                decoder_iou=1.0,
                top300_rank=MODULE.MAX_DETS + 1,
                top300_iou=0.0,
                formal_query=349,
                formal_rank=350,
                formal_score=float(scores[349]),
                formal_iou=1.0,
            ),
            "before",
            boxes,
            scores,
            gt_box,
        )
        self.assertEqual(
            (rank_gt300.query, rank_gt300.rank, rank_gt300.kind),
            (349, 350, "formal maxDets=640 match"),
        )

        boxes, scores = prediction()
        boxes[9] = gt_box
        competition = MODULE.choose_display_candidate(
            row(
                "one_to_one_competition",
                decoder_rank=10,
                decoder_iou=1.0,
                top300_rank=10,
                top300_iou=1.0,
            ),
            "before",
            boxes,
            scores,
            gt_box,
        )
        self.assertEqual(
            (competition.query, competition.rank, competition.kind),
            (9, 10, "first saved IoU-qualified query"),
        )

    def test_deterministic_six_sheet_render_and_hash_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_dir, replay_root, images_root, prediction_paths = create_fixture(root)
            output_a = root / "figures_a"
            output_b = root / "figures_b"
            with (
                mock.patch.object(MODULE.formal_plot, "FORMAL_IMAGES", 2),
                mock.patch.object(MODULE.formal_plot, "FORMAL_NONCROWD_GT", 2),
            ):
                outputs_a = MODULE.run(
                    SimpleNamespace(
                        data_dir=data_dir,
                        replay_root=replay_root,
                        images_root=images_root,
                        output=output_a,
                    )
                )
                outputs_b = MODULE.run(
                    SimpleNamespace(
                        data_dir=data_dir,
                        replay_root=replay_root,
                        images_root=images_root,
                        output=output_b,
                    )
                )

            self.assertEqual(len(outputs_a), 7)
            self.assertEqual([path.name for path in outputs_a], [path.name for path in outputs_b])
            for first, second in zip(outputs_a, outputs_b):
                self.assertEqual(first.read_bytes(), second.read_bytes())
            for figure in outputs_a[:-1]:
                with Image.open(figure) as image:
                    self.assertEqual(image.size, (1220, 3088))

            with outputs_a[-1].open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 60)
            selected = [row for row in rows if row["panel_status"] == "selected"]
            absent = [row for row in rows if row["panel_status"] == "absent"]
            self.assertEqual((len(selected), len(absent)), (12, 48))
            self.assertTrue(all(row["cause"] == "candidate_absence" for row in selected))
            self.assertTrue(all(row["display_query_before"] == "0" for row in selected))
            self.assertNotIn(str(root), outputs_a[-1].read_text(encoding="utf-8"))

            with prediction_paths[28].open("ab") as handle:
                handle.write(b"corruption")
            with (
                mock.patch.object(MODULE.formal_plot, "FORMAL_IMAGES", 2),
                mock.patch.object(MODULE.formal_plot, "FORMAL_NONCROWD_GT", 2),
                self.assertRaisesRegex(ValueError, "prediction.*byte-size mismatch"),
            ):
                MODULE.run(
                    SimpleNamespace(
                        data_dir=data_dir,
                        replay_root=replay_root,
                        images_root=images_root,
                        output=root / "tampered",
                    )
                )


if __name__ == "__main__":
    unittest.main()
