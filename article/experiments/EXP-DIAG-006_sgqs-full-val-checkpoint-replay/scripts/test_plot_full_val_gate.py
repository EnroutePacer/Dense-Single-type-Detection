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


if __name__ == "__main__":
    unittest.main()
