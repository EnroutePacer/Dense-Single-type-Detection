"""Plan, replay gate, and completion tests for the matcher matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.table_cells.run_coco_match_matrix import (
    IMAGES_PER_SHARD,
    IOU_THRESHOLDS,
    MAX_DETS,
    QUERIES_PER_IMAGE,
    base_audit,
    build_specs,
    matcher_command,
    orchestrate,
    replay_wait_reasons,
    sha256,
    validate_match_complete,
    verify_existing_matrix_identity,
)


def arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python=root / "python.exe",
        matcher_script=root / "coco_match_npz.py",
        match_export_script=root / "coco_match_export.py",
        annotation=root / "instance_val.json",
        images_root=root / "images",
        config=root / "resolved.yml",
        checkpoint_identity=root / "checkpoint_identity.csv",
        weights_root=root / "weights",
        replay_root=root / "replays",
        output_root=root / "matches",
        expected_annotation_sha256="a" * 64,
        expected_matcher_sha256="b" * 64,
        expected_match_export_sha256="c" * 64,
        expected_checkpoint_identity_sha256="d" * 64,
        replay_device="cpu",
        replay_batch_size=1,
        cpu_threads=20,
        minimum_available_gib=10.0,
        minimum_available_commit_gib=6.0,
        minimum_free_disk_gib=40.0,
        active_grace_seconds=900.0,
        resume_recent=False,
    )


class CocoMatchMatrixTest(unittest.TestCase):
    def test_plan_is_epoch_then_raw_ema_with_fixed_protocol(self):
        root = Path("C:/local")
        args = arguments(root)
        specs = build_specs(args.replay_root, args.output_root)
        self.assertEqual(
            [(spec.epoch, spec.state) for spec in specs],
            [(epoch, state) for epoch in range(28, 35) for state in ("raw", "ema")],
        )
        for spec in specs:
            command = matcher_command(args, spec)
            joined = " ".join(command)
            self.assertIn("--max-dets 300 640", joined)
            self.assertIn("--images-per-shard 256", joined)
            threshold_start = command.index("--iou-thresholds") + 1
            threshold_stop = command.index("--images-per-shard")
            self.assertEqual(len(command[threshold_start:threshold_stop]), 10)
            self.assertNotIn("full_val_replay.py", joined)

    def test_missing_replay_matrix_is_reported_before_matcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            waiting = replay_wait_reasons(args)
            self.assertEqual(set(waiting), set(range(28, 35)))
            self.assertIn("summary.json", waiting[28])
            self.assertIn("val.raw.complete.json", waiting[28])
            self.assertIn("val.ema.complete.json", waiting[28])

    def test_complete_match_state_verifies_every_shard_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            args.annotation.write_bytes(b"annotation")
            args.expected_annotation_sha256 = sha256(args.annotation)
            spec = build_specs(args.replay_root, args.output_root)[0]
            spec.prediction.parent.mkdir(parents=True)
            spec.prediction.write_bytes(b"predictions")
            prediction_descriptor = {
                "file": spec.prediction.name,
                "bytes": spec.prediction.stat().st_size,
                "sha256": sha256(spec.prediction),
            }
            spec.output.mkdir(parents=True)

            protocol = {
                "schema": "coco_one_to_one_npz_export_v2",
                "annotation": str(args.annotation.resolve()),
                "annotation_sha256": args.expected_annotation_sha256,
                "prediction": str(spec.prediction.resolve()),
                "prediction_sha256": prediction_descriptor["sha256"],
                "state": spec.state_id,
                "images": 2,
                "annotation_images": 2,
                "prediction_images": 2,
                "max_images_requested": None,
                "queries_per_image": QUERIES_PER_IMAGE,
                "max_dets": list(MAX_DETS),
                "iou_thresholds": list(IOU_THRESHOLDS),
                "images_per_shard": 1,
            }
            protocol_path = spec.output / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            shard_manifest = []
            gt_shard_manifest = []
            annotation_total = 0
            for index, annotations in enumerate((2, 1)):
                gt_path = spec.output / f"gt_matches.part{index:04d}.csv.gz"
                status_path = spec.output / f"detection_status.part{index:04d}.npz"
                gt_path.write_bytes(f"gt-{index}".encode())
                status_path.write_bytes(f"status-{index}".encode())
                summary = {
                    "schema": "coco_one_to_one_match_shard_v2",
                    "shard_index": index,
                    "first_image_id": index + 1,
                    "last_image_id": index + 1,
                    "images": 1,
                    "annotations": annotations,
                    "predictions": QUERIES_PER_IMAGE,
                    "gt_matches_file": gt_path.name,
                    "gt_matches_sha256": sha256(gt_path),
                    "detection_status_file": status_path.name,
                    "detection_status_sha256": sha256(status_path),
                    "counts": [],
                }
                summary_path = spec.output / f"summary.part{index:04d}.json"
                summary_path.write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                shard_manifest.append(
                    {
                        "index": index,
                        "file": status_path.name,
                        "sha256": sha256(status_path),
                        "first_image_id": index + 1,
                        "last_image_id": index + 1,
                        "images": 1,
                    }
                )
                gt_shard_manifest.append(
                    {
                        "index": index,
                        "summary_file": summary_path.name,
                        "summary_sha256": sha256(summary_path),
                        "gt_matches_file": gt_path.name,
                        "gt_matches_sha256": sha256(gt_path),
                        "detection_status_file": status_path.name,
                        "detection_status_sha256": sha256(status_path),
                        "first_image_id": index + 1,
                        "last_image_id": index + 1,
                        "images": 1,
                        "annotations": annotations,
                    }
                )
                annotation_total += annotations
            manifest = {
                "schema": "coco_detection_status_manifest_v1",
                "state": spec.state_id,
                "scope_names": ["all", "coco_small_eval"],
                "iou_thresholds": list(IOU_THRESHOLDS),
                "max_dets": list(MAX_DETS),
                "shards": shard_manifest,
            }
            manifest_path = spec.output / "detection_status_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            gt_manifest_path = spec.output / "gt_shard_manifest.json"
            gt_manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "coco_gt_match_shard_manifest_v1",
                        "state": spec.state_id,
                        "shards": gt_shard_manifest,
                    }
                ),
                encoding="utf-8",
            )
            aggregate_path = spec.output / "match_aggregates.csv"
            aggregate_path.write_text("state,unit\n", encoding="utf-8")
            complete = {
                "schema": "coco_one_to_one_npz_complete_v2",
                "protocol_sha256": sha256(protocol_path),
                "shards": 2,
                "images": 2,
                "annotations": annotation_total,
                "predictions": 2 * QUERIES_PER_IMAGE,
                "aggregate_csv": aggregate_path.name,
                "aggregate_sha256": sha256(aggregate_path),
                "detection_status_manifest": manifest_path.name,
                "detection_status_manifest_sha256": sha256(manifest_path),
                "gt_shard_manifest": gt_manifest_path.name,
                "gt_shard_manifest_sha256": sha256(gt_manifest_path),
            }
            (spec.output / "complete.json").write_text(json.dumps(complete), encoding="utf-8")

            with (
                mock.patch("tools.table_cells.run_coco_match_matrix.EXPECTED_IMAGES", 2),
                mock.patch("tools.table_cells.run_coco_match_matrix.EXPECTED_ANNOTATIONS", 3),
                mock.patch("tools.table_cells.run_coco_match_matrix.IMAGES_PER_SHARD", 1),
            ):
                self.assertEqual(
                    validate_match_complete(args, spec, prediction_descriptor), complete
                )
                first_summary = spec.output / "summary.part0000.json"
                original_summary = first_summary.read_bytes()
                first_summary.write_bytes(original_summary + b" ")
                with self.assertRaisesRegex(Exception, "GT-shard hash chain mismatch"):
                    validate_match_complete(args, spec, prediction_descriptor)
                first_summary.write_bytes(original_summary)
                (spec.output / "detection_status.part0001.npz").write_bytes(b"corrupt")
                with self.assertRaisesRegex(Exception, "SHA-256 mismatch"):
                    validate_match_complete(args, spec, prediction_descriptor)

    def test_audit_binds_replay_identity_and_every_prediction_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            args.config.write_bytes(b"config")
            spec = build_specs(args.replay_root, args.output_root)[0]
            spec.replay_output.mkdir(parents=True)
            spec.prediction.write_bytes(b"predictions")
            (spec.replay_output / "summary.json").write_bytes(b"summary")
            (spec.replay_output / "val.raw.complete.json").write_bytes(b"marker")
            descriptor = {
                "file": spec.prediction.name,
                "bytes": spec.prediction.stat().st_size,
                "sha256": sha256(spec.prediction),
            }
            replay_summaries = {
                spec.epoch: {
                    "results": {"val/raw": {"artifacts": {"predictions": descriptor}}}
                }
            }
            replay_audit = {
                "matrix_complete_sha256": "e" * 64,
                "checkpoint_contract": {"checkpoint0028.pth": {"verification": "recomputed_equal"}},
                "runtime_backend": {"backend": "cpu", "torch_version": "test"},
                "engine_python_tree": {"sha256": "f" * 64},
            }
            audit = base_audit(
                args, [spec], replay_summaries, replay_audit, {"numpy_version": "test"}
            )
            evidence = audit["prediction_inputs"]["e28/raw"]
            self.assertEqual(evidence["prediction_sha256"], descriptor["sha256"])
            self.assertEqual(audit["config_sha256"], sha256(args.config))
            self.assertEqual(audit["replay_runtime_backend"], replay_audit["runtime_backend"])
            self.assertEqual(audit["checkpoint_contract"], replay_audit["checkpoint_contract"])

    def test_resume_rejects_changed_runtime_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix_status.json"
            path.write_text(json.dumps({"matcher_runtime": {"numpy_version": "old"}}))
            with self.assertRaisesRegex(Exception, "different matcher_runtime"):
                verify_existing_matrix_identity(
                    path, {"matcher_runtime": {"numpy_version": "new"}}
                )

    def test_skipped_complete_state_keeps_prediction_and_replay_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            args.config.write_bytes(b"config")
            spec = build_specs(args.replay_root, args.output_root)[0]
            spec.replay_output.mkdir(parents=True)
            spec.prediction.write_bytes(b"predictions")
            (spec.replay_output / "summary.json").write_bytes(b"summary")
            (spec.replay_output / "val.raw.complete.json").write_bytes(b"marker")
            descriptor = {
                "file": spec.prediction.name,
                "bytes": spec.prediction.stat().st_size,
                "sha256": sha256(spec.prediction),
            }
            replay_summaries = {
                spec.epoch: {
                    "results": {"val/raw": {"artifacts": {"predictions": descriptor}}}
                }
            }
            replay_audit = {
                "matrix_complete_sha256": "e" * 64,
                "checkpoint_contract": {"checkpoint0028.pth": {"verification": "recomputed_equal"}},
                "runtime_backend": {"backend": "cpu", "torch_version": "test"},
                "engine_python_tree": {"sha256": "f" * 64},
            }
            with mock.patch(
                "tools.table_cells.run_coco_match_matrix.validate_match_complete",
                return_value={
                    "schema": "complete",
                    "gt_shard_manifest_sha256": "a" * 64,
                },
            ):
                spec.output.mkdir(parents=True)
                (spec.output / "complete.json").write_bytes(b"complete")
                orchestrate(
                    args,
                    [spec],
                    replay_summaries,
                    replay_audit,
                    {"numpy_version": "test"},
                )
            status = json.loads((args.output_root / "matrix_status.json").read_text())
            state = status["states"]["e28/raw"]
            self.assertEqual(state["status"], "skipped_verified_complete")
            self.assertEqual(state["gt_shard_manifest_sha256"], "a" * 64)
            self.assertEqual(state["prediction_sha256"], descriptor["sha256"])
            self.assertEqual(state["replay_summary_sha256"], sha256(spec.replay_output / "summary.json"))
            self.assertEqual(
                state["replay_state_marker_sha256"],
                sha256(spec.replay_output / "val.raw.complete.json"),
            )


if __name__ == "__main__":
    unittest.main()
