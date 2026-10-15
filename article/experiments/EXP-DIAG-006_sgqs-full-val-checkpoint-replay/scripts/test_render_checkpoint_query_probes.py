"""Low-resource tests for deterministic selective Encoder/Decoder query probes."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).with_name("render_checkpoint_query_probes.py")
SPEC = importlib.util.spec_from_file_location("exp_diag_006_checkpoint_probes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def representative_row(
    *,
    track: str,
    start: int,
    end: int,
    transition: str,
    evidence: float,
    annotation_id: int,
    image_id: int,
    partition: str | None = None,
) -> dict[str, str]:
    if track == "evaluator_small":
        scope = "coco_small_eval"
        partition = partition or "coco_and_hybrid"
    else:
        scope = "all"
        partition = "hybrid_only"
    flags = {
        "coco_and_hybrid": ("1", "1"),
        "coco_only": ("1", "0"),
        "hybrid_only": ("0", "1"),
        "neither": ("0", "0"),
    }[partition]
    gain = transition == "gain"
    return {
        "selection_rank": "1",
        "state": "ema",
        "epoch_start": str(start),
        "epoch_end": str(end),
        "match_scope": scope,
        "iou_threshold": "0.75",
        "transition": transition,
        "cause": "rank_gt300",
        "small_partition": partition,
        "domain": "wired",
        "annotation_id": str(annotation_id),
        "image_id": str(image_id),
        "file_name": f"wired/image_{image_id}.png",
        "image_gt_count": "1",
        "evidence_strength": str(evidence),
        "rank_distance_from_300": "2",
        "status_before": "fn" if gain else "tp",
        "status_after": "tp" if gain else "fn",
        "coco_small": flags[0],
        "local_hybrid_small": flags[1],
    }


def make_prediction(path: Path, *, query_note: str | None = None) -> None:
    boxes = np.full((2, MODULE.QUERY_COUNT, 4), 20.0, dtype=np.float32)
    boxes[0, 0, 2] = -1.0  # finite negative extent is preserved, not rejected
    scores = np.linspace(0.0, 1.0, MODULE.QUERY_COUNT, dtype=np.float32)
    scores = np.stack([scores, scores], axis=0)
    np.savez_compressed(
        path,
        schema=np.asarray("sgqs_compact_predictions_v1"),
        image_ids=np.asarray([10, 20], dtype=np.int64),
        boxes_xywh=boxes,
        scores=scores,
        category_id=np.asarray(0, dtype=np.int64),
        input_size=np.asarray([640, 640], dtype=np.int32),
        state=np.asarray("checkpoint0028:ema"),
        box_format=np.asarray("xywh_abs_original_pixels"),
        query_indices=np.arange(MODULE.QUERY_COUNT, dtype=np.uint16),
        query_index_note=np.asarray(
            query_note or "column index; repeated identically for every image"
        ),
    )


def make_probe_fixture(
    image_ids: tuple[int, ...] = (10,),
) -> tuple[
    dict[int, MODULE.transition_render.ImageMetadata],
    dict[int, dict[int, MODULE.ProbeObservation]],
]:
    metadata = {}
    observations_by_image = {}
    for image_id in image_ids:
        metadata[image_id] = MODULE.transition_render.ImageMetadata(
            image_id=image_id,
            file_name=f"wired/image_{image_id}.png",
            width=160,
            height=100,
            annotation_ids=np.asarray([image_id], dtype=np.int64),
            boxes_xywh=np.asarray([[20, 20, 30, 15]], dtype=np.float32),
        )
        encoder_boxes = np.zeros((MODULE.QUERY_COUNT, 4), dtype=np.float32)
        encoder_boxes[:, 0] = np.linspace(0.05, 0.95, MODULE.QUERY_COUNT)
        encoder_boxes[:, 1] = 0.5
        dense_indices = np.arange(MODULE.QUERY_COUNT, dtype=np.int64)
        levels, counts = MODULE.dense_source_levels(dense_indices, (8, 16, 32))
        decoder_boxes = np.full(
            (MODULE.QUERY_COUNT, 4), [15, 15, 20, 10], dtype=np.float32
        )
        decoder_scores = np.linspace(0.0, 1.0, MODULE.QUERY_COUNT, dtype=np.float32)
        classes, hybrid = MODULE.decoder_size_classes(decoder_boxes, 160, 100)
        recon = MODULE.Reconciliation(0.0, 0.0, True, True, True, True)
        observations_by_image[image_id] = {
            epoch: MODULE.ProbeObservation(
                epoch=epoch,
                image_id=image_id,
                encoder_boxes_cxcywh=encoder_boxes,
                encoder_dense_indices=dense_indices,
                encoder_levels=levels,
                level_counts=counts,
                decoder_boxes_xywh=decoder_boxes,
                decoder_scores=decoder_scores,
                decoder_classes=classes,
                decoder_hybrid_small=hybrid,
                reconciliation=recon,
                prediction_logical_name=f"e{epoch}/ema/predictions.npz",
                prediction_sha256="0" * 64,
            )
            for epoch in MODULE.EPOCHS
        }
    return metadata, observations_by_image


def make_manifest_rows(
    observations_by_image: dict[int, dict[int, MODULE.ProbeObservation]],
    *,
    probe_data_path: Path | None = None,
) -> list[dict[str, object]]:
    runtime = {"requested_device": "cpu", "torch_num_threads": 1}
    runtime_json = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
    if probe_data_path is None:
        data_file, data_bytes, data_sha = "query_probe_data.npz", 1, "a" * 64
    else:
        data_file = probe_data_path.name
        data_bytes = probe_data_path.stat().st_size
        data_sha = MODULE.formal_plot.sha256(probe_data_path)
    rows = []
    for image_id in sorted(observations_by_image):
        for epoch in MODULE.EPOCHS:
            observation = observations_by_image[image_id][epoch]
            row: dict[str, object] = {
                field: "value" for field in MODULE.MANIFEST_FIELDS
            }
            for field in MODULE.MANIFEST_FIELDS:
                if field.endswith("_sha256"):
                    row[field] = "a" * 64
            for field in MODULE.MANIFEST_LOGICAL_PATH_FIELDS:
                row[field] = f"logical/{field}.bin"
            for field in (
                "figure_bytes",
                "source_image_bytes",
                "checkpoint_bytes",
                "probe_data_bytes",
                "replay_matrix_bytes",
            ):
                row[field] = 1
            row.update(
                state=MODULE.STATE,
                epoch=epoch,
                image_id=image_id,
                checkpoint_file=f"checkpoint{epoch:04d}.pth",
                checkpoint_state="ema.module",
                probe_data_file=data_file,
                probe_data_bytes=data_bytes,
                probe_data_sha256=data_sha,
                probe_data_schema=MODULE.PROBE_DATA_SCHEMA,
                encoder_selected_count=MODULE.QUERY_COUNT,
                encoder_p3_count=observation.level_counts[0],
                encoder_p4_count=observation.level_counts[1],
                encoder_p5_count=observation.level_counts[2],
                encoder_selected_indices_sha256=MODULE._array_sha256(
                    observation.encoder_dense_indices
                ),
                encoder_selected_boxes_sha256=MODULE._array_sha256(
                    observation.encoder_boxes_cxcywh
                ),
                decoder_query_count=MODULE.QUERY_COUNT,
                decoder_hybrid_small_count=int(
                    np.count_nonzero(observation.decoder_hybrid_small)
                ),
                decoder_hybrid_mask_sha256=MODULE._array_sha256(
                    observation.decoder_hybrid_small
                ),
                decoder_boxes_exact=1,
                decoder_scores_exact=1,
                decoder_rank_order_exact=1,
                decoder_hybrid_mask_exact=1,
                decoder_box_max_abs_error="0",
                decoder_score_max_abs_error="0",
                runtime_backend_json=runtime_json,
                runtime_backend_sha256=MODULE._json_sha256(runtime),
            )
            rows.append(row)
    return rows


class SelectionTests(unittest.TestCase):
    def test_selection_is_bounded_deterministic_and_covers_both_tracks(self):
        rows = [
            representative_row(
                track="evaluator_small",
                start=28,
                end=29,
                transition="gain",
                evidence=0.8,
                annotation_id=1,
                image_id=10,
            ),
            representative_row(
                track="evaluator_small",
                start=28,
                end=29,
                transition="gain",
                evidence=0.9,
                annotation_id=2,
                image_id=20,
                partition="coco_only",
            ),
            representative_row(
                track="evaluator_small",
                start=30,
                end=34,
                transition="loss",
                evidence=0.7,
                annotation_id=3,
                image_id=30,
            ),
            representative_row(
                track="hybrid_only",
                start=28,
                end=29,
                transition="gain",
                evidence=0.6,
                annotation_id=4,
                image_id=40,
            ),
            representative_row(
                track="hybrid_only",
                start=30,
                end=34,
                transition="loss",
                evidence=0.5,
                annotation_id=5,
                image_id=50,
            ),
        ]
        expected = MODULE.select_probe_rows(rows)
        shuffled = rows.copy()
        random.Random(20261015).shuffle(shuffled)
        self.assertEqual(MODULE.select_probe_rows(shuffled), expected)
        self.assertEqual(len(expected), 4)
        self.assertEqual(
            {(row["track"], row["epoch_start"], row["epoch_end"]) for row in expected},
            {
                ("evaluator_small", 28, 29),
                ("evaluator_small", 30, 34),
                ("hybrid_only", 28, 29),
                ("hybrid_only", 30, 34),
            },
        )
        selected_formal_gain = next(
            row
            for row in expected
            if row["track"] == "evaluator_small" and row["epoch_start"] == 28
        )
        self.assertEqual(selected_formal_gain["annotation_id"], 2)

    def test_partition_flags_and_status_are_fail_closed(self):
        rows = []
        for track in ("evaluator_small", "hybrid_only"):
            rows.extend(
                [
                    representative_row(
                        track=track,
                        start=28,
                        end=29,
                        transition="gain",
                        evidence=0.8,
                        annotation_id=len(rows) + 1,
                        image_id=len(rows) + 10,
                    ),
                    representative_row(
                        track=track,
                        start=30,
                        end=34,
                        transition="loss",
                        evidence=0.7,
                        annotation_id=len(rows) + 2,
                        image_id=len(rows) + 20,
                    ),
                ]
            )
        broken = [dict(row) for row in rows]
        broken[0]["local_hybrid_small"] = "0"
        with self.assertRaisesRegex(ValueError, "small_partition"):
            MODULE.select_probe_rows(broken)
        broken = [dict(row) for row in rows]
        broken[0]["status_before"] = "tp"
        with self.assertRaisesRegex(ValueError, "status mismatch"):
            MODULE.select_probe_rows(broken)

    def test_exact_representative_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "exact.csv"
            with exact.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    MODULE.formal_analysis.REPRESENTATIVE_FIELDS
                )
            self.assertEqual(MODULE.read_formal_representatives(exact), [])
            extra = root / "extra.csv"
            with extra.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    (*MODULE.formal_analysis.REPRESENTATIVE_FIELDS, "unexpected")
                )
            with self.assertRaisesRegex(ValueError, "non-canonical schema"):
                MODULE.read_formal_representatives(extra)


class GeometryTests(unittest.TestCase):
    def test_dense_indices_are_true_mutually_exclusive_source_levels(self):
        indices = np.concatenate(
            [np.arange(637, dtype=np.int64), np.asarray([6399, 6400, 8000])]
        )
        levels, counts = MODULE.dense_source_levels(indices, (8, 16, 32))
        self.assertEqual(counts, (638, 1, 1))
        self.assertEqual(levels[-3:].tolist(), [0, 1, 2])
        duplicate = indices.copy()
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            MODULE.dense_source_levels(duplicate, (8, 16, 32))

    def test_hybrid_small_uses_area_or_short_side_and_preserves_negative_extent(self):
        boxes = np.full((MODULE.QUERY_COUNT, 4), [0, 0, 100, 100], dtype=np.float32)
        boxes[0] = [0, 0, 31, 31]
        boxes[1] = [0, 0, 100, 15]
        boxes[2] = [0, 0, 40, 40]
        boxes[3] = [0, 0, -1, 100]
        classes, hybrid = MODULE.decoder_size_classes(boxes, 640, 640)
        self.assertEqual(classes[:4].tolist(), ["small", "small", "medium", "small"])
        self.assertEqual(hybrid[:4].tolist(), [True, True, False, True])

    def test_decoder_reconciliation_is_per_query_and_bit_exact(self):
        boxes = np.full((MODULE.QUERY_COUNT, 4), 1.0, dtype=np.float32)
        scores = np.linspace(0.0, 1.0, MODULE.QUERY_COUNT, dtype=np.float32)
        result = MODULE.reconcile_decoder(boxes, scores, boxes.copy(), scores.copy(), 640, 640)
        self.assertTrue(result.boxes_exact)
        changed = boxes.copy()
        changed[5, 0] = np.nextafter(changed[5, 0], np.float32(2.0))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            MODULE.reconcile_decoder(changed, scores, boxes, scores, 640, 640)


class ArtifactTests(unittest.TestCase):
    def test_prediction_schema_accepts_finite_negative_extent_and_rejects_metadata_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.npz"
            make_prediction(valid)
            MODULE.validate_prediction_archive(valid, epoch=28, formal_images=2)
            invalid = root / "invalid.npz"
            make_prediction(invalid, query_note="wrong")
            with self.assertRaisesRegex(ValueError, "query_index_note"):
                MODULE.validate_prediction_archive(invalid, epoch=28, formal_images=2)

    def test_probe_tensor_archive_is_complete_portable_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, observations = make_probe_fixture((10, 20))
            first = root / "first.npz"
            second = root / "second.npz"
            MODULE.write_query_probe_data(first, observations, metadata)
            MODULE.write_query_probe_data(second, observations, metadata)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(str(root).encode(), first.read_bytes())
            shapes = MODULE.validate_query_probe_data_archive(
                first, expected_image_ids=(10, 20)
            )
            self.assertEqual(
                shapes["encoder_selected_indices"],
                (2, len(MODULE.EPOCHS), MODULE.QUERY_COUNT),
            )
            self.assertEqual(
                shapes["encoder_boxes_cxcywh"],
                (2, len(MODULE.EPOCHS), MODULE.QUERY_COUNT, 4),
            )
            with np.load(first, allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), MODULE.PROBE_DATA_FIELDS)
                self.assertEqual(archive["encoder_selected_indices"].dtype, np.int64)
                self.assertEqual(archive["encoder_levels"].dtype, np.uint8)
                self.assertEqual(archive["decoder_hybrid_small"].dtype, np.bool_)
                self.assertEqual(
                    MODULE._array_sha256(archive["encoder_selected_indices"][0, 0]),
                    MODULE._array_sha256(
                        observations[10][MODULE.EPOCHS[0]].encoder_dense_indices
                    ),
                )

    def test_manifest_gate_requires_complete_provenance_and_portable_paths(self):
        _metadata, observations = make_probe_fixture()
        rows = make_manifest_rows(observations)
        MODULE.validate_probe_manifest_rows(rows)

        absolute_rows = [dict(row) for row in rows]
        absolute_rows[0]["probe_data_file"] = r"D:\private\query_probe_data.npz"
        with self.assertRaisesRegex(ValueError, "not portable"):
            MODULE.validate_probe_manifest_rows(absolute_rows)
        missing_rows = [dict(row) for row in rows]
        missing_rows[0]["engine_python_tree_sha256"] = ""
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            MODULE.validate_probe_manifest_rows(missing_rows)
        incomplete_rows = rows[:-1]
        with self.assertRaisesRegex(ValueError, "unique image"):
            MODULE.validate_probe_manifest_rows(incomplete_rows)

    def test_manifest_slices_reconcile_to_probe_tensor_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata, observations = make_probe_fixture((10, 20))
            data_path = root / "query_probe_data.npz"
            MODULE.write_query_probe_data(data_path, observations, metadata)
            rows = make_manifest_rows(observations, probe_data_path=data_path)
            MODULE.validate_probe_manifest_rows(rows, probe_data_path=data_path)
            drifted = [dict(row) for row in rows]
            drifted[0]["encoder_selected_indices_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "disagrees with NPZ"):
                MODULE.validate_probe_manifest_rows(drifted, probe_data_path=data_path)

    def test_replay_source_engine_and_runtime_identity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path = root / "matrix.complete.json"
            torch = MODULE.full_val_replay.torch
            runtime = MODULE._current_runtime_backend("cpu", torch)
            replay_sha = MODULE.formal_plot.sha256(
                Path(MODULE.full_val_replay.__file__).resolve()
            )
            supervision_sha = MODULE.formal_plot.sha256(
                Path(MODULE.full_val_replay.supervision_probe.__file__).resolve()
            )
            engine = MODULE.replay_matrix.engine_tree_identity()
            config_sha = "1" * 64
            annotation_sha = "2" * 64
            checkpoints = {
                f"checkpoint{epoch:04d}.pth": {
                    "file_sha256": format(epoch, "064x"),
                    "verification": "recomputed_equal",
                }
                for epoch in MODULE.EPOCHS
            }
            matrix = {
                "schema": MODULE.REPLAY_MATRIX_SCHEMA,
                "status": "complete",
                "epochs": list(MODULE.EPOCHS),
                "states": ["raw", "ema"],
                "config_sha256": config_sha,
                "annotation_sha256": annotation_sha,
                "device": "cpu",
                "batch_size": 1,
                "replay_script_sha256": replay_sha,
                "supervision_probe_sha256": supervision_sha,
                "engine_python_tree": engine,
                "checkpoint_contract": checkpoints,
                "runtime_backend": runtime,
            }

            def write_matrix_and_audit(payload):
                matrix_path.write_text(
                    json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
                )
                return {
                    "formal_identity": {
                        "config_sha256": config_sha,
                        "annotation_sha256": annotation_sha,
                        "inference_identity": {
                            "device": "cpu",
                            "batch_size": 1,
                            "torch_version": str(torch.__version__),
                            "torch_cuda_version": torch.version.cuda,
                        },
                    },
                    "input_identity": {
                        "matcher_matrix": {
                            "replay_matrix_complete_sha256": MODULE.formal_plot.sha256(
                                matrix_path
                            )
                        },
                        "replays": {
                            f"e{epoch}": {
                                "checkpoint_sha256": checkpoints[
                                    f"checkpoint{epoch:04d}.pth"
                                ]["file_sha256"]
                            }
                            for epoch in MODULE.EPOCHS
                        },
                    },
                }

            audit = write_matrix_and_audit(matrix)
            identity = MODULE._validate_replay_execution_identity(
                audit, root, "cpu", torch
            )
            self.assertEqual(identity["matrix_sha256"], MODULE.formal_plot.sha256(matrix_path))
            self.assertEqual(identity["runtime_backend_sha256"], MODULE._json_sha256(runtime))

            drifted = dict(matrix)
            drifted["supervision_probe_sha256"] = "0" * 64
            audit = write_matrix_and_audit(drifted)
            with self.assertRaisesRegex(ValueError, "supervision_probe"):
                MODULE._validate_replay_execution_identity(audit, root, "cpu", torch)

    def test_trajectory_render_is_deterministic_without_model_forward(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            Image.new("RGB", (160, 100), (245, 245, 245)).save(images / "sample.png")
            metadata = MODULE.transition_render.ImageMetadata(
                image_id=10,
                file_name="sample.png",
                width=160,
                height=100,
                annotation_ids=np.asarray([1], dtype=np.int64),
                boxes_xywh=np.asarray([[20, 20, 30, 15]], dtype=np.float32),
            )
            encoder_boxes = np.zeros((MODULE.QUERY_COUNT, 4), dtype=np.float32)
            encoder_boxes[:, 0] = np.linspace(0.05, 0.95, MODULE.QUERY_COUNT)
            encoder_boxes[:, 1] = 0.5
            dense_indices = np.arange(MODULE.QUERY_COUNT, dtype=np.int64)
            levels, counts = MODULE.dense_source_levels(dense_indices, (8, 16, 32))
            decoder_boxes = np.full(
                (MODULE.QUERY_COUNT, 4), [15, 15, 20, 10], dtype=np.float32
            )
            decoder_scores = np.linspace(0.0, 1.0, MODULE.QUERY_COUNT, dtype=np.float32)
            classes, hybrid = MODULE.decoder_size_classes(decoder_boxes, 160, 100)
            recon = MODULE.Reconciliation(0.0, 0.0, True, True, True, True)
            observations = {
                epoch: MODULE.ProbeObservation(
                    epoch=epoch,
                    image_id=10,
                    encoder_boxes_cxcywh=encoder_boxes,
                    encoder_dense_indices=dense_indices,
                    encoder_levels=levels,
                    level_counts=counts,
                    decoder_boxes_xywh=decoder_boxes,
                    decoder_scores=decoder_scores,
                    decoder_classes=classes,
                    decoder_hybrid_small=hybrid,
                    reconciliation=recon,
                    prediction_logical_name=f"e{epoch}/ema/predictions.npz",
                    prediction_sha256="0" * 64,
                )
                for epoch in MODULE.EPOCHS
            }
            selected = [
                {
                    "track": "evaluator_small",
                    "epoch_start": 28,
                    "epoch_end": 29,
                    "transition": "gain",
                    "cause": "rank_gt300",
                    "domain": "wired",
                    "annotation_id": 1,
                }
            ]
            first = root / "first.png"
            second = root / "second.png"
            MODULE.render_image_trajectory(first, metadata, selected, observations, images)
            MODULE.render_image_trajectory(second, metadata, selected, observations, images)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with Image.open(first) as rendered:
                self.assertEqual(rendered.size, (1348, 2022))


if __name__ == "__main__":
    unittest.main()
