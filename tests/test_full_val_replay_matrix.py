"""Static plan and completion-contract tests for the full-val orchestrator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from tools.table_cells.run_full_val_replay_matrix import (
    EPOCHS,
    EXPECTED_ANNOTATIONS,
    EXPECTED_IMAGES,
    MatrixError,
    STATES,
    base_audit,
    build_specs,
    compare_checkpoint_contracts,
    orchestrate,
    recent_incomplete,
    replay_command,
    resource_gate,
    sha256,
    validate_complete,
    validate_partial_progress,
    verify_checkpoint_contract,
)


def arguments(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        python=root / "python.exe",
        replay_script=root / "full_val_replay.py",
        supervision_probe=root / "supervision_probe.py",
        checkpoint_identity=root / "checkpoint_identity.csv",
        checkpoint_hash_script=root / "checkpoint_state_hash.py",
        config=root / "resolved.yml",
        annotation=root / "instance_val.json",
        images_root=root / "images",
        expected_annotation_sha256="a" * 64,
        expected_replay_sha256="b" * 64,
        expected_supervision_probe_sha256="c" * 64,
        expected_checkpoint_identity_sha256="d" * 64,
        expected_checkpoint_hash_script_sha256="e" * 64,
        expected_engine_tree_sha256="f" * 64,
        split_name="val",
        device="cpu",
        batch_size=1,
        eval_batch_size=16,
        durable_images=64,
        progress_every=256,
        active_grace_seconds=900.0,
        minimum_available_gib=12.0,
        minimum_available_commit_gib=6.0,
        minimum_free_disk_gib=20.0,
        cpu_threads=4,
        resume_recent=False,
        output_root=root / "outputs",
    )


class FullValReplayMatrixTest(unittest.TestCase):
    @staticmethod
    def checkpoint_row(epoch: int) -> dict:
        return {
            "file": f"checkpoint{epoch:04d}.pth",
            "bytes": 100 + epoch,
            "last_epoch": epoch,
            "file_sha256": "1" * 64,
            "raw_state_sha256": "2" * 64,
            "ema_state_sha256": "3" * 64,
        }

    def test_plan_is_exactly_e28_to_e34_raw_and_ema(self):
        root = Path("C:/local")
        specs = build_specs(root / "weights", root / "outputs")
        self.assertEqual([spec.epoch for spec in specs], list(EPOCHS))
        self.assertEqual(
            [spec.checkpoint.name for spec in specs],
            [f"checkpoint{epoch:04d}.pth" for epoch in EPOCHS],
        )
        combined = " ".join(str(item) for spec in specs for item in replay_command(arguments(root), spec))
        self.assertIn("--states raw ema", combined)
        self.assertIn("--resume", combined)
        self.assertNotIn("best_stg1.pth", combined)
        self.assertNotIn("last.pth", combined)

    def test_completed_output_requires_both_markers_and_verified_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            spec = build_specs(root / "weights", root / "outputs")[0]
            spec.output.mkdir(parents=True)
            spec.checkpoint.parent.mkdir(parents=True)
            spec.checkpoint.write_bytes(b"checkpoint")
            args.config.write_bytes(b"config")
            metadata = spec.output / "val.metadata.npz"
            metadata.write_bytes(b"metadata")
            run_identity = {
                "checkpoint": str(spec.checkpoint.resolve()),
                "checkpoint_sha256": sha256(spec.checkpoint),
                "config": str(args.config.resolve()),
                "config_sha256": sha256(args.config),
                "source": {
                    "mode": "direct_coco",
                    "annotation": str(args.annotation.resolve()),
                    "annotation_sha256": args.expected_annotation_sha256,
                    "images_root": str(args.images_root.resolve()),
                },
                "max_images": None,
                "image_ids": None,
                "states": list(STATES),
                "inference_identity": {"device": "cpu", "batch_size": 1},
            }
            results = {}
            for state in STATES:
                artifacts = {}
                for kind in ("predictions", "gt_stages"):
                    path = spec.output / f"val.{state}.{kind}.npz"
                    path.write_bytes(f"{state}-{kind}".encode())
                    artifacts[kind] = {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                result = {
                    "images": EXPECTED_IMAGES,
                    "noncrowd_annotations": EXPECTED_ANNOTATIONS,
                    "predictions": EXPECTED_IMAGES * 640,
                    "coco_maxdets300": [0.0] * 12,
                    "coco_maxdets640": [0.0] * 12,
                    "coco_small_per_iou_maxdets300": [{}] * 10,
                    "coco_small_per_iou_maxdets640": [{}] * 10,
                    "artifacts": artifacts,
                }
                results[f"val/{state}"] = result
                marker = {
                    "identity": {
                        "run_identity": run_identity,
                        "split": "val",
                        "state": state,
                        "images": EXPECTED_IMAGES,
                        "annotations": EXPECTED_ANNOTATIONS,
                    },
                    "result": result,
                }
                (spec.output / f"val.{state}.complete.json").write_text(
                    json.dumps(marker), encoding="utf-8"
                )
            summary = {
                **run_identity,
                "last_epoch": spec.epoch,
                "splits": {
                    "val": {
                        "images": EXPECTED_IMAGES,
                        "noncrowd_annotations": EXPECTED_ANNOTATIONS,
                        "metadata": metadata.name,
                        "metadata_sha256": sha256(metadata),
                    }
                },
                "results": results,
            }
            (spec.output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            self.assertEqual(validate_complete(args, spec), summary)

            (spec.output / "val.ema.predictions.npz").write_bytes(b"corrupt")
            with self.assertRaisesRegex(MatrixError, "artifact size mismatch"):
                validate_complete(args, spec)

    def test_recent_incomplete_output_is_detected_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            spec = build_specs(root / "weights", root / "outputs")[0]
            spec.output.mkdir(parents=True)
            progress = spec.output / ".val.raw.partial" / "progress.json"
            progress.parent.mkdir()
            progress.write_text("{}", encoding="utf-8")
            recent, age = recent_incomplete(args, spec)
            self.assertTrue(recent)
            self.assertIsNotNone(age)
            old = time.time() - args.active_grace_seconds - 10
            progress.touch()
            # The directory itself also carries a fresh timestamp, so age the
            # entire synthetic output tree before checking stale recovery.
            for path in (progress, progress.parent, spec.output):
                path.touch()
                import os

                os.utime(path, (old, old))
            recent, _ = recent_incomplete(args, spec)
            self.assertFalse(recent)

    def test_partial_progress_must_stay_inside_full_dataset_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            spec = build_specs(root / "weights", root / "outputs")[0]
            partial = spec.output / ".val.raw.partial"
            partial.mkdir(parents=True)
            run_identity = {
                "checkpoint": str(spec.checkpoint.resolve()),
                "config": str(args.config.resolve()),
                "source": {
                    "mode": "direct_coco",
                    "annotation": str(args.annotation.resolve()),
                    "annotation_sha256": args.expected_annotation_sha256,
                    "images_root": str(args.images_root.resolve()),
                },
                "max_images": None,
                "image_ids": None,
                "states": list(STATES),
                "inference_identity": {"device": "cpu", "batch_size": 1},
            }
            progress = {
                "identity": {
                    "run_identity": run_identity,
                    "split": "val",
                    "state": "raw",
                    "images": EXPECTED_IMAGES,
                    "annotations": EXPECTED_ANNOTATIONS,
                },
                "completed_images": EXPECTED_IMAGES + 1,
            }
            (partial / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            with self.assertRaisesRegex(MatrixError, "outside"):
                validate_partial_progress(args, spec)

    def test_checkpoint_contract_rejects_raw_or_ema_state_mismatch(self):
        expected = {f"checkpoint{epoch:04d}.pth": self.checkpoint_row(epoch) for epoch in EPOCHS}
        for field in ("raw_state_sha256", "ema_state_sha256"):
            observed = {name: dict(row) for name, row in expected.items()}
            observed["checkpoint0028.pth"][field] = "9" * 64
            with self.subTest(field=field):
                with self.assertRaisesRegex(MatrixError, field):
                    compare_checkpoint_contracts(expected, observed)

    def test_checkpoint_hash_subprocess_is_consumed_without_loading_weights_in_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            rows = [self.checkpoint_row(epoch) for epoch in EPOCHS]
            header = list(rows[0])
            args.checkpoint_identity.write_text(
                ",".join(header) + "\n" + "\n".join(
                    ",".join(str(row[key]) for key in header) for row in rows
                ),
                encoding="utf-8",
            )
            specs = build_specs(root / "weights", root / "outputs")
            stdout = args.checkpoint_identity.read_text(encoding="utf-8")
            completed = mock.Mock(returncode=0, stdout=stdout, stderr="")
            with mock.patch(
                "tools.table_cells.run_full_val_replay_matrix.subprocess.run",
                return_value=completed,
            ) as run:
                verified = verify_checkpoint_contract(args, specs)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(
                all(row["verification"] == "recomputed_equal" for row in verified.values())
            )

    def test_backend_identity_is_audited_and_resume_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            args.config.write_bytes(b"config")
            contract = {f"checkpoint{epoch:04d}.pth": self.checkpoint_row(epoch) for epoch in EPOCHS}
            backend = {"backend": "cpu", "torch_version": "2.test", "torch_num_threads": 4}
            engine = {"sha256": "f" * 64, "files": 66, "entries": {}}
            specs = build_specs(root / "weights", args.output_root)
            audit = base_audit(args, specs, contract, backend, engine)
            self.assertEqual(audit["runtime_backend"], backend)
            args.output_root.mkdir(parents=True)
            existing = dict(audit)
            existing["runtime_backend"] = {**backend, "torch_num_threads": 8}
            (args.output_root / "matrix_status.json").write_text(
                json.dumps(existing), encoding="utf-8"
            )
            with self.assertRaisesRegex(MatrixError, "different runtime_backend"):
                orchestrate(args, [], contract, backend, engine)

    def test_skipped_complete_checkpoint_keeps_container_and_state_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = arguments(root)
            args.config.write_bytes(b"config")
            spec = build_specs(root / "weights", args.output_root)[0]
            row = self.checkpoint_row(spec.epoch)
            contract = {spec.checkpoint.name: row}
            backend = {"backend": "cpu", "torch_version": "2.test", "torch_num_threads": 4}
            engine = {"sha256": "f" * 64, "files": 66, "entries": {}}
            with (
                mock.patch(
                    "tools.table_cells.run_full_val_replay_matrix.validate_complete",
                    return_value={"checkpoint_sha256": row["file_sha256"]},
                ),
                mock.patch("tools.table_cells.run_full_val_replay_matrix.append_log"),
            ):
                orchestrate(args, [spec], contract, backend, engine)
            status = json.loads((args.output_root / "matrix_status.json").read_text())
            checkpoint = status["checkpoints"][str(spec.epoch)]
            self.assertEqual(checkpoint["status"], "skipped_verified_complete")
            self.assertEqual(checkpoint["checkpoint_sha256"], row["file_sha256"])
            self.assertEqual(checkpoint["raw_state_sha256"], row["raw_state_sha256"])
            self.assertEqual(checkpoint["ema_state_sha256"], row["ema_state_sha256"])

    def test_resource_gate_rejects_low_windows_commit_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            gib = 1024**3
            status = {
                "available_physical": 16 * gib,
                "available_commit": 2 * gib,
                "commit_limit": 38 * gib,
            }
            disk = mock.Mock(free=100 * gib)
            with (
                mock.patch(
                    "tools.table_cells.run_full_val_replay_matrix.memory_status_bytes",
                    return_value=status,
                ),
                mock.patch(
                    "tools.table_cells.run_full_val_replay_matrix.shutil.disk_usage",
                    return_value=disk,
                ),
                mock.patch("tools.table_cells.run_full_val_replay_matrix.os.name", "nt"),
            ):
                with self.assertRaisesRegex(MatrixError, "commit/pagefile headroom"):
                    resource_gate(args)

    def test_resource_gate_audits_safe_commit_headroom(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = arguments(Path(temporary))
            gib = 1024**3
            status = {
                "available_physical": 16 * gib,
                "available_commit": 8 * gib,
                "commit_limit": 38 * gib,
            }
            disk = mock.Mock(free=100 * gib)
            with (
                mock.patch(
                    "tools.table_cells.run_full_val_replay_matrix.memory_status_bytes",
                    return_value=status,
                ),
                mock.patch(
                    "tools.table_cells.run_full_val_replay_matrix.shutil.disk_usage",
                    return_value=disk,
                ),
                mock.patch("tools.table_cells.run_full_val_replay_matrix.os.name", "nt"),
            ):
                observed = resource_gate(args)
            self.assertEqual(observed["available_commit_gib"], 8.0)
            self.assertEqual(observed["commit_limit_gib"], 38.0)


if __name__ == "__main__":
    unittest.main()
