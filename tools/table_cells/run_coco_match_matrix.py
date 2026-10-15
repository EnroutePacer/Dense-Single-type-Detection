"""Serially export formal COCO matches for the 14 local SGQS replay states.

This is the second stage of the full-validation pipeline.  It never loads a
model: it consumes only fully verified ``predictions.npz`` files from the
e28--e34 raw/EMA replay matrix.  All seven replay summaries and all fourteen
replay completion markers must exist before the first matcher child starts.

The matching protocol is fixed to maxDets 300/640, IoU .50:.05:.95, and 256
images per shard.  Re-running resumes verified shards; complete states are
skipped only after their marker, protocol, aggregate, manifest, GT shards, and
detection-status shards pass size/count/SHA-256 postconditions.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.table_cells.run_full_val_replay_matrix import (
    CHECKPOINT_IDENTITY_SHA256,
    ENGINE_PYTHON_TREE_SHA256,
    EPOCHS,
    EXPECTED_ANNOTATIONS,
    EXPECTED_IMAGES,
    LOCAL_ANNOTATION_SHA256,
    REPLAY_SCRIPT_SHA256,
    STATES,
    SUPERVISION_PROBE_SHA256,
    MatrixError,
    append_log,
    atomic_json,
    build_specs as build_replay_specs,
    command_text,
    expected_checkpoint_contract,
    exclusive_matrix_lock,
    normalized_path,
    require_file,
    resource_gate,
    run_child,
    sha256,
    utc_now,
    validate_complete as validate_replay_complete,
)


MAX_DETS = (300, 640)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
IMAGES_PER_SHARD = 256
QUERIES_PER_IMAGE = 640
MATCHER_SCRIPT_SHA256 = "4423bbb6fad325fd1f8f78d24b8cdff9ef921f03a0e9790e036c437b97d8945b"
MATCH_EXPORT_SHA256 = "8151bf2514e00c171384b0571d8fc645d32c53aae7f3a96d95c293563f931237"
REPLAY_MATRIX_SCHEMA = "sgqs_full_val_matrix_orchestration_v1"
MATCH_MATRIX_SCHEMA = "sgqs_coco_match_matrix_orchestration_v1"


@dataclass(frozen=True)
class MatchSpec:
    epoch: int
    state: str
    replay_output: Path
    prediction: Path
    output: Path
    log: Path

    @property
    def state_id(self) -> str:
        return f"checkpoint{self.epoch:04d}:{self.state}"


def build_specs(replay_root: Path, output_root: Path) -> list[MatchSpec]:
    specs = []
    for epoch in EPOCHS:
        replay_output = replay_root / f"checkpoint{epoch:04d}"
        for state in STATES:
            specs.append(
                MatchSpec(
                    epoch=epoch,
                    state=state,
                    replay_output=replay_output,
                    prediction=replay_output / f"val.{state}.predictions.npz",
                    output=output_root / f"checkpoint{epoch:04d}" / state,
                    log=output_root / "logs" / f"checkpoint{epoch:04d}.{state}.log",
                )
            )
    return specs


def matcher_command(args: argparse.Namespace, spec: MatchSpec) -> list[str]:
    return [
        os.fspath(args.python),
        "-u",
        os.fspath(args.matcher_script),
        "--annotation",
        os.fspath(args.annotation),
        "--predictions",
        os.fspath(spec.prediction),
        "--output",
        os.fspath(spec.output),
        "--max-dets",
        *(str(value) for value in MAX_DETS),
        "--iou-thresholds",
        *(f"{value:.2f}" for value in IOU_THRESHOLDS),
        "--images-per-shard",
        str(IMAGES_PER_SHARD),
    ]


def replay_validation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.config,
        annotation=args.annotation,
        images_root=args.images_root,
        expected_annotation_sha256=args.expected_annotation_sha256,
        split_name="val",
        device=args.replay_device,
        batch_size=args.replay_batch_size,
    )


def replay_wait_reasons(args: argparse.Namespace) -> dict[int, list[str]]:
    """Perform a cheap all-epoch marker gate before hashing any large artifact."""
    waiting: dict[int, list[str]] = {}
    for spec in build_replay_specs(args.weights_root, args.replay_root):
        missing = []
        for name in ("summary.json", "val.raw.complete.json", "val.ema.complete.json"):
            if not (spec.output / name).is_file():
                missing.append(name)
        if missing:
            waiting[spec.epoch] = missing
    return waiting


def load_replay_matrix_audit(args: argparse.Namespace) -> dict:
    """Validate the completed replay orchestration identity before matching."""
    path = args.replay_root / "matrix.complete.json"
    require_file(path, "completed replay matrix audit")
    audit = _load_json(path, "completed replay matrix audit")
    if audit.get("schema") != REPLAY_MATRIX_SCHEMA or audit.get("status") != "complete":
        raise MatrixError(f"replay matrix audit is not complete: {path}")
    if audit.get("epochs") != list(EPOCHS) or audit.get("states") != list(STATES):
        raise MatrixError(f"replay matrix audit has a different 7x2 plan: {path}")
    if normalized_path(audit.get("annotation", "")) != normalized_path(args.annotation):
        raise MatrixError(f"replay matrix annotation path mismatch: {path}")
    if audit.get("annotation_sha256", "").lower() != args.expected_annotation_sha256:
        raise MatrixError(f"replay matrix annotation SHA-256 mismatch: {path}")
    if normalized_path(audit.get("config", "")) != normalized_path(args.config):
        raise MatrixError(f"replay matrix config path mismatch: {path}")
    if audit.get("config_sha256", "").lower() != sha256(args.config):
        raise MatrixError(f"replay matrix config SHA-256 mismatch: {path}")
    if audit.get("replay_script_sha256", "").lower() != REPLAY_SCRIPT_SHA256:
        raise MatrixError(f"replay matrix used an unexpected replay implementation: {path}")
    if audit.get("supervision_probe_sha256", "").lower() != SUPERVISION_PROBE_SHA256:
        raise MatrixError(f"replay matrix used an unexpected supervision probe: {path}")
    if audit.get("checkpoint_identity_sha256", "").lower() != args.expected_checkpoint_identity_sha256:
        raise MatrixError(f"replay matrix checkpoint identity SHA-256 mismatch: {path}")
    expected_contract = expected_checkpoint_contract(args.checkpoint_identity)
    observed_contract = audit.get("checkpoint_contract")
    if not isinstance(observed_contract, dict):
        raise MatrixError(f"replay matrix checkpoint contract is missing: {path}")
    for name, expected in expected_contract.items():
        observed = observed_contract.get(name)
        if not isinstance(observed, dict):
            raise MatrixError(f"replay matrix checkpoint contract is missing {name}: {path}")
        if any(observed.get(key) != value for key, value in expected.items()):
            raise MatrixError(f"replay matrix checkpoint contract mismatch for {name}: {path}")
        if observed.get("verification") != "recomputed_equal":
            raise MatrixError(f"replay matrix checkpoint was not independently verified: {name}")
    backend = audit.get("runtime_backend")
    if not isinstance(backend, dict) or backend.get("requested_device") != args.replay_device:
        raise MatrixError(f"replay matrix backend identity mismatch: {path}")
    if audit.get("device") != args.replay_device:
        raise MatrixError(f"replay matrix device mismatch: {path}")
    if int(audit.get("batch_size", -1)) != args.replay_batch_size:
        raise MatrixError(f"replay matrix batch-size mismatch: {path}")
    engine = audit.get("engine_python_tree")
    if not isinstance(engine, dict) or engine.get("sha256") != ENGINE_PYTHON_TREE_SHA256:
        raise MatrixError(f"replay matrix engine source identity mismatch: {path}")
    audit = dict(audit)
    audit["matrix_complete_sha256"] = sha256(path)
    return audit


def validate_replay_matrix(args: argparse.Namespace) -> tuple[dict[int, dict], dict]:
    waiting = replay_wait_reasons(args)
    if waiting:
        details = "; ".join(
            f"e{epoch}: {','.join(names)}" for epoch, names in sorted(waiting.items())
        )
        raise MatrixError(
            "replay matrix is not complete; wait for replay rather than starting matcher: "
            + details
        )
    replay_audit = load_replay_matrix_audit(args)
    replay_args = replay_validation_args(args)
    replay_args.runtime_backend = replay_audit["runtime_backend"]
    summaries: dict[int, dict] = {}
    for spec in build_replay_specs(args.weights_root, args.replay_root):
        summary = validate_replay_complete(replay_args, spec)
        if summary is None:
            raise MatrixError(f"e{spec.epoch} replay is incomplete")
        checkpoint_identity = replay_audit["checkpoint_contract"][spec.checkpoint.name]
        if (
            summary.get("checkpoint_sha256", "").lower()
            != checkpoint_identity["file_sha256"]
            or int(summary.get("last_epoch", -1)) != spec.epoch
        ):
            raise MatrixError(
                f"e{spec.epoch} replay summary does not match the frozen checkpoint contract"
            )
        summaries[spec.epoch] = summary
    return summaries, replay_audit


def replay_prediction_descriptor(
    replay_summaries: Mapping[int, dict], spec: MatchSpec
) -> dict:
    summary = replay_summaries[spec.epoch]
    result = summary.get("results", {}).get(f"val/{spec.state}", {})
    descriptor = result.get("artifacts", {}).get("predictions")
    if not isinstance(descriptor, dict):
        raise MatrixError(f"e{spec.epoch}/{spec.state}: replay prediction descriptor missing")
    expected = spec.replay_output / descriptor.get("file", "")
    if normalized_path(expected) != normalized_path(spec.prediction):
        raise MatrixError(f"e{spec.epoch}/{spec.state}: unexpected replay prediction path")
    return descriptor


def preflight(args: argparse.Namespace, specs: Sequence[MatchSpec]) -> None:
    require_file(args.python, "Python interpreter")
    require_file(args.matcher_script, "compact COCO matcher")
    require_file(args.match_export_script, "COCO matcher implementation")
    require_file(args.annotation, "merged validation annotation")
    require_file(args.config, "resolved replay config")
    require_file(args.checkpoint_identity, "frozen checkpoint identity table")
    if not args.images_root.is_dir():
        raise MatrixError(f"missing merged image view: {args.images_root}")
    for epoch in EPOCHS:
        require_file(
            args.weights_root / f"checkpoint{epoch:04d}.pth",
            f"epoch-{epoch} replay checkpoint",
        )
    checks = (
        ("annotation", args.annotation, args.expected_annotation_sha256),
        ("matcher", args.matcher_script, args.expected_matcher_sha256),
        ("match exporter", args.match_export_script, args.expected_match_export_sha256),
        (
            "checkpoint identity",
            args.checkpoint_identity,
            args.expected_checkpoint_identity_sha256,
        ),
    )
    for label, path, expected in checks:
        observed = sha256(path)
        if observed.lower() != expected.lower():
            raise MatrixError(f"{label} SHA-256 mismatch: expected {expected}, got {observed}")
    if len(specs) != len(EPOCHS) * len(STATES):
        raise MatrixError("matcher plan is not the complete 7x2 matrix")


def matcher_incomplete_age(spec: MatchSpec) -> float | None:
    if not spec.output.exists():
        return None
    newest = spec.output.stat().st_mtime
    for path in spec.output.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except FileNotFoundError:
            continue
    return max(0.0, time.time() - newest)


def _load_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"invalid {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"invalid {label} object: {path}")
    return value


def _hash_matches(path: Path, expected: str, label: str) -> None:
    require_file(path, label)
    if len(expected) != 64 or sha256(path) != expected.lower():
        raise MatrixError(f"{label} SHA-256 mismatch: {path}")


def probe_matcher_runtime(args: argparse.Namespace) -> dict:
    """Probe the exact lightweight runtime used by matcher child processes."""
    probe = r'''
import importlib.metadata, json, platform
import faster_coco_eval
import numpy

def distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "numpy_version": str(numpy.__version__),
    "faster_coco_eval_version": distribution_version("faster-coco-eval"),
    "faster_coco_eval_module_version": getattr(faster_coco_eval, "__version__", None),
}, sort_keys=True))
'''
    completed = subprocess.run(
        [os.fspath(args.python), "-c", probe],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode:
        raise MatrixError(
            f"matcher runtime probe failed with exit code {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    try:
        identity = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise MatrixError("matcher runtime probe returned invalid JSON") from exc
    if not identity.get("numpy_version"):
        raise MatrixError("matcher runtime probe did not identify numpy")
    return identity


def validate_match_complete(
    args: argparse.Namespace,
    spec: MatchSpec,
    prediction_descriptor: Mapping[str, object],
) -> dict | None:
    """Validate a state completely, returning ``None`` only when resumable."""
    complete_path = spec.output / "complete.json"
    protocol_path = spec.output / "protocol.json"
    aggregate_path = spec.output / "match_aggregates.csv"
    manifest_path = spec.output / "detection_status_manifest.json"
    gt_manifest_path = spec.output / "gt_shard_manifest.json"
    if not all(
        path.is_file()
        for path in (
            complete_path,
            protocol_path,
            aggregate_path,
            manifest_path,
            gt_manifest_path,
        )
    ):
        return None

    complete = _load_json(complete_path, "matcher completion marker")
    if complete.get("schema") != "coco_one_to_one_npz_complete_v2":
        raise MatrixError(f"unknown matcher completion schema: {complete_path}")
    expected_shards = math.ceil(EXPECTED_IMAGES / IMAGES_PER_SHARD)
    expected_predictions = EXPECTED_IMAGES * QUERIES_PER_IMAGE
    if (
        int(complete.get("shards", -1)) != expected_shards
        or int(complete.get("images", -1)) != EXPECTED_IMAGES
        or int(complete.get("annotations", -1)) != EXPECTED_ANNOTATIONS
        or int(complete.get("predictions", -1)) != expected_predictions
    ):
        raise MatrixError(f"matcher completion population mismatch: {complete_path}")
    if complete.get("aggregate_csv") != aggregate_path.name:
        raise MatrixError(f"matcher aggregate filename mismatch: {complete_path}")
    if complete.get("detection_status_manifest") != manifest_path.name:
        raise MatrixError(f"matcher status-manifest filename mismatch: {complete_path}")
    if complete.get("gt_shard_manifest") != gt_manifest_path.name:
        raise MatrixError(f"matcher GT-shard manifest filename mismatch: {complete_path}")
    _hash_matches(protocol_path, complete.get("protocol_sha256", ""), "matcher protocol")
    _hash_matches(aggregate_path, complete.get("aggregate_sha256", ""), "matcher aggregate")
    _hash_matches(
        manifest_path,
        complete.get("detection_status_manifest_sha256", ""),
        "detection-status manifest",
    )
    _hash_matches(
        gt_manifest_path,
        complete.get("gt_shard_manifest_sha256", ""),
        "GT-shard manifest",
    )

    protocol = _load_json(protocol_path, "matcher protocol")
    if protocol.get("schema") != "coco_one_to_one_npz_export_v2":
        raise MatrixError(f"unknown matcher protocol schema: {protocol_path}")
    if normalized_path(protocol.get("annotation", "")) != normalized_path(args.annotation):
        raise MatrixError(f"matcher annotation path mismatch: {protocol_path}")
    if protocol.get("annotation_sha256", "").lower() != args.expected_annotation_sha256:
        raise MatrixError(f"matcher annotation SHA-256 mismatch: {protocol_path}")
    if normalized_path(protocol.get("prediction", "")) != normalized_path(spec.prediction):
        raise MatrixError(f"matcher prediction path mismatch: {protocol_path}")
    prediction_hash = str(prediction_descriptor.get("sha256", "")).lower()
    if protocol.get("prediction_sha256", "").lower() != prediction_hash:
        raise MatrixError(f"matcher prediction SHA-256 mismatch: {protocol_path}")
    if (
        protocol.get("state") != spec.state_id
        or int(protocol.get("images", -1)) != EXPECTED_IMAGES
        or int(protocol.get("annotation_images", -1)) != EXPECTED_IMAGES
        or int(protocol.get("prediction_images", -1)) != EXPECTED_IMAGES
        or protocol.get("max_images_requested") is not None
        or int(protocol.get("queries_per_image", -1)) != QUERIES_PER_IMAGE
        or protocol.get("max_dets") != list(MAX_DETS)
        or protocol.get("iou_thresholds") != list(IOU_THRESHOLDS)
        or int(protocol.get("images_per_shard", -1)) != IMAGES_PER_SHARD
    ):
        raise MatrixError(f"matcher protocol parameters mismatch: {protocol_path}")

    manifest = _load_json(manifest_path, "detection-status manifest")
    if (
        manifest.get("schema") != "coco_detection_status_manifest_v1"
        or manifest.get("state") != spec.state_id
        or manifest.get("scope_names") != ["all", "coco_small_eval"]
        or manifest.get("max_dets") != list(MAX_DETS)
        or manifest.get("iou_thresholds") != list(IOU_THRESHOLDS)
    ):
        raise MatrixError(f"detection-status protocol mismatch: {manifest_path}")
    manifest_shards = manifest.get("shards")
    if not isinstance(manifest_shards, list) or len(manifest_shards) != expected_shards:
        raise MatrixError(f"detection-status shard count mismatch: {manifest_path}")
    gt_manifest = _load_json(gt_manifest_path, "GT-shard manifest")
    gt_manifest_shards = gt_manifest.get("shards")
    if (
        gt_manifest.get("schema") != "coco_gt_match_shard_manifest_v1"
        or gt_manifest.get("state") != spec.state_id
        or not isinstance(gt_manifest_shards, list)
        or len(gt_manifest_shards) != expected_shards
    ):
        raise MatrixError(f"GT-shard manifest mismatch: {gt_manifest_path}")

    annotation_total = 0
    prediction_total = 0
    image_total = 0
    expected_summary_names = set()
    expected_gt_names = set()
    expected_status_names = set()
    for index, manifest_shard in enumerate(manifest_shards):
        gt_manifest_shard = gt_manifest_shards[index]
        shard_images = min(IMAGES_PER_SHARD, EXPECTED_IMAGES - index * IMAGES_PER_SHARD)
        summary_path = spec.output / f"summary.part{index:04d}.json"
        gt_path = spec.output / f"gt_matches.part{index:04d}.csv.gz"
        status_path = spec.output / f"detection_status.part{index:04d}.npz"
        expected_summary_names.add(summary_path.name)
        expected_gt_names.add(gt_path.name)
        expected_status_names.add(status_path.name)
        if not all(path.is_file() for path in (summary_path, gt_path, status_path)):
            return None
        shard = _load_json(summary_path, "matcher shard summary")
        if (
            shard.get("schema") != "coco_one_to_one_match_shard_v2"
            or int(shard.get("shard_index", -1)) != index
            or int(shard.get("images", -1)) != shard_images
            or shard.get("gt_matches_file") != gt_path.name
            or shard.get("detection_status_file") != status_path.name
        ):
            raise MatrixError(f"matcher shard summary mismatch: {summary_path}")
        _hash_matches(gt_path, shard.get("gt_matches_sha256", ""), "GT match shard")
        _hash_matches(
            status_path,
            shard.get("detection_status_sha256", ""),
            "detection-status shard",
        )
        if (
            int(gt_manifest_shard.get("index", -1)) != index
            or gt_manifest_shard.get("summary_file") != summary_path.name
            or gt_manifest_shard.get("summary_sha256", "") != sha256(summary_path)
            or gt_manifest_shard.get("gt_matches_file") != gt_path.name
            or gt_manifest_shard.get("gt_matches_sha256", "")
            != shard.get("gt_matches_sha256")
            or gt_manifest_shard.get("detection_status_file") != status_path.name
            or gt_manifest_shard.get("detection_status_sha256", "")
            != shard.get("detection_status_sha256")
            or int(gt_manifest_shard.get("first_image_id", -1))
            != int(shard.get("first_image_id", -2))
            or int(gt_manifest_shard.get("last_image_id", -1))
            != int(shard.get("last_image_id", -2))
            or int(gt_manifest_shard.get("images", -1)) != shard_images
            or int(gt_manifest_shard.get("annotations", -1))
            != int(shard.get("annotations", -2))
        ):
            raise MatrixError(f"GT-shard hash chain mismatch: {summary_path}")
        if (
            int(manifest_shard.get("index", -1)) != index
            or manifest_shard.get("file") != status_path.name
            or manifest_shard.get("sha256", "") != shard.get("detection_status_sha256")
            or int(manifest_shard.get("images", -1)) != shard_images
            or int(manifest_shard.get("first_image_id", -1))
            != int(shard.get("first_image_id", -2))
            or int(manifest_shard.get("last_image_id", -1))
            != int(shard.get("last_image_id", -2))
        ):
            raise MatrixError(f"matcher shard manifest mismatch: {summary_path}")
        image_total += int(shard["images"])
        annotation_total += int(shard.get("annotations", -1))
        prediction_total += int(shard.get("predictions", -1))

    actual_summary_names = {path.name for path in spec.output.glob("summary.part*.json")}
    actual_gt_names = {path.name for path in spec.output.glob("gt_matches.part*.csv.gz")}
    actual_status_names = {
        path.name for path in spec.output.glob("detection_status.part*.npz")
    }
    if (
        actual_summary_names != expected_summary_names
        or actual_gt_names != expected_gt_names
        or actual_status_names != expected_status_names
    ):
        raise MatrixError(f"unexpected or missing matcher shard files: {spec.output}")
    if (
        image_total != EXPECTED_IMAGES
        or annotation_total != EXPECTED_ANNOTATIONS
        or prediction_total != expected_predictions
    ):
        raise MatrixError(f"matcher shard totals disagree with completion marker: {spec.output}")
    return complete


def replay_input_evidence(
    replay_summaries: Mapping[int, dict], spec: MatchSpec
) -> dict[str, object]:
    descriptor = replay_prediction_descriptor(replay_summaries, spec)
    return {
        "prediction": os.fspath(spec.prediction),
        "prediction_bytes": int(descriptor["bytes"]),
        "prediction_sha256": str(descriptor["sha256"]).lower(),
        "replay_summary_sha256": sha256(spec.replay_output / "summary.json"),
        "replay_state_marker_sha256": sha256(
            spec.replay_output / f"val.{spec.state}.complete.json"
        ),
    }


def base_audit(
    args: argparse.Namespace,
    specs: Sequence[MatchSpec],
    replay_summaries: Mapping[int, dict],
    replay_audit: Mapping[str, object],
    matcher_runtime: Mapping[str, object],
) -> dict:
    prediction_inputs = {
        f"e{spec.epoch}/{spec.state}": replay_input_evidence(replay_summaries, spec)
        for spec in specs
    }
    return {
        "schema": MATCH_MATRIX_SCHEMA,
        "epochs": list(EPOCHS),
        "states": list(STATES),
        "serial_jobs": 1,
        "max_dets": list(MAX_DETS),
        "iou_thresholds": list(IOU_THRESHOLDS),
        "images_per_shard": IMAGES_PER_SHARD,
        "annotation": os.fspath(args.annotation),
        "annotation_sha256": args.expected_annotation_sha256,
        "config": os.fspath(args.config),
        "config_sha256": sha256(args.config),
        "matcher_script_sha256": args.expected_matcher_sha256,
        "match_export_sha256": args.expected_match_export_sha256,
        "checkpoint_identity_file": os.fspath(args.checkpoint_identity),
        "checkpoint_identity_sha256": args.expected_checkpoint_identity_sha256,
        "checkpoint_contract": replay_audit["checkpoint_contract"],
        "replay_matrix_complete_sha256": replay_audit["matrix_complete_sha256"],
        "replay_runtime_backend": replay_audit["runtime_backend"],
        "replay_engine_python_tree": replay_audit["engine_python_tree"],
        "matcher_runtime": dict(matcher_runtime),
        "prediction_inputs": prediction_inputs,
        "minimum_available_gib": args.minimum_available_gib,
        "minimum_available_commit_gib": args.minimum_available_commit_gib,
        "minimum_free_disk_gib": args.minimum_free_disk_gib,
        "replay_root": os.fspath(args.replay_root),
        "output_root": os.fspath(args.output_root),
        "runs": [
            {
                **asdict(spec),
                "replay_output": os.fspath(spec.replay_output),
                "prediction": os.fspath(spec.prediction),
                "output": os.fspath(spec.output),
                "log": os.fspath(spec.log),
            }
            for spec in specs
        ],
    }


MATCH_IDENTITY_KEYS = (
    "epochs",
    "states",
    "max_dets",
    "iou_thresholds",
    "images_per_shard",
    "annotation_sha256",
    "config_sha256",
    "matcher_script_sha256",
    "match_export_sha256",
    "checkpoint_identity_sha256",
    "checkpoint_contract",
    "replay_matrix_complete_sha256",
    "replay_runtime_backend",
    "replay_engine_python_tree",
    "matcher_runtime",
    "prediction_inputs",
    "minimum_available_gib",
    "minimum_available_commit_gib",
    "minimum_free_disk_gib",
    "runs",
)


def verify_existing_matrix_identity(path: Path, audit: Mapping[str, object]) -> None:
    if not path.is_file():
        return
    existing = _load_json(path, "existing matcher matrix status")
    for key in MATCH_IDENTITY_KEYS:
        if existing.get(key) != audit.get(key):
            raise MatrixError(f"existing matcher matrix status has a different {key}")


def dry_run(
    args: argparse.Namespace,
    specs: Sequence[MatchSpec],
    waiting: Mapping[int, Sequence[str]],
    replay_summaries: Mapping[int, dict] | None,
) -> None:
    for spec in specs:
        if waiting:
            names = ",".join(waiting.get(spec.epoch, ()))
            status = f"WAIT replay e{spec.epoch}{(': ' + names) if names else ''}"
        else:
            assert replay_summaries is not None
            descriptor = replay_prediction_descriptor(replay_summaries, spec)
            complete = validate_match_complete(args, spec, descriptor)
            if complete is not None:
                status = "SKIP verified complete"
            else:
                age = matcher_incomplete_age(spec)
                if age is not None and age < args.active_grace_seconds:
                    status = f"REFUSE recent incomplete ({age:.1f}s)"
                else:
                    status = "RESUME" if spec.output.exists() else "START"
        print(f"e{spec.epoch}/{spec.state}: {status}")
        print(f"  {command_text(matcher_command(args, spec))}")
    if waiting:
        print("matcher matrix blocked: replay matrix is incomplete; no matcher was started")


def orchestrate(
    args: argparse.Namespace,
    specs: Sequence[MatchSpec],
    replay_summaries: Mapping[int, dict],
    replay_audit: Mapping[str, object],
    matcher_runtime: Mapping[str, object],
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_path = args.output_root / "matrix_status.json"
    complete_path = args.output_root / "matrix.complete.json"
    audit = base_audit(args, specs, replay_summaries, replay_audit, matcher_runtime)
    audit.update(status="running", started_at=utc_now(), states={})
    env = os.environ.copy()
    env.update(
        PYTHONUNBUFFERED="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS=str(args.cpu_threads),
        MKL_NUM_THREADS=str(args.cpu_threads),
    )

    with exclusive_matrix_lock(args.output_root / ".matcher_matrix.lock"):
        verify_existing_matrix_identity(status_path, audit)
        atomic_json(status_path, audit)
        for spec in specs:
            key = f"e{spec.epoch}/{spec.state}"
            state = {"epoch": spec.epoch, "state": spec.state, "updated_at": utc_now()}
            audit["states"][key] = state
            descriptor = replay_prediction_descriptor(replay_summaries, spec)
            state.update(replay_input_evidence(replay_summaries, spec))
            complete = validate_match_complete(args, spec, descriptor)
            if complete is not None:
                state.update(
                    status="skipped_verified_complete",
                    complete_sha256=sha256(spec.output / "complete.json"),
                    gt_shard_manifest_sha256=complete["gt_shard_manifest_sha256"],
                )
                append_log(spec.log, f"[{utc_now()}] SKIP verified complete {key}")
                atomic_json(status_path, audit)
                continue

            age = matcher_incomplete_age(spec)
            if age is not None and age < args.active_grace_seconds and not args.resume_recent:
                state.update(status="refused_recent_incomplete", newest_age_seconds=age)
                audit.update(status="failed", failed_state=key, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(
                    f"{key} matcher output changed {age:.1f}s ago and may have a live writer; "
                    "wait, or use --resume-recent only after confirming no matcher is writing it"
                )

            resources = resource_gate(args)
            command = matcher_command(args, spec)
            state.update(
                status="running",
                command=command_text(command),
                started_at=utc_now(),
                resource_gate=resources,
            )
            atomic_json(status_path, audit)
            return_code = run_child(command, spec.log, env)
            if return_code:
                state.update(status="failed", return_code=return_code, finished_at=utc_now())
                audit.update(status="failed", failed_state=key, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(f"{key} matcher failed with exit code {return_code}")
            complete = validate_match_complete(args, spec, descriptor)
            if complete is None:
                state.update(status="failed_postcondition", finished_at=utc_now())
                audit.update(status="failed", failed_state=key, finished_at=utc_now())
                atomic_json(status_path, audit)
                raise MatrixError(f"{key} matcher exited without complete verified artifacts")
            state.update(
                status="verified_complete",
                finished_at=utc_now(),
                complete_sha256=sha256(spec.output / "complete.json"),
                gt_shard_manifest_sha256=complete["gt_shard_manifest_sha256"],
            )
            atomic_json(status_path, audit)

        audit.update(status="complete", finished_at=utc_now())
        atomic_json(status_path, audit)
        atomic_json(complete_path, audit)
        print(complete_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--matcher-script", type=Path, default=ROOT / "tools/table_cells/coco_match_npz.py"
    )
    parser.add_argument(
        "--match-export-script",
        type=Path,
        default=ROOT / "tools/table_cells/coco_match_export.py",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=ROOT / "_local/fullval_dataset_view/annotations/instance_val.json",
    )
    parser.add_argument(
        "--images-root", type=Path, default=ROOT / "_local/fullval_dataset_view/images"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "_local/server_sgqs_probe_20261014/resolved.yml"
    )
    parser.add_argument(
        "--checkpoint-identity",
        type=Path,
        default=(
            ROOT
            / "article/experiments/EXP-DIAG-005_sgqs-top300-ranking-replay/data/checkpoint_identity.csv"
        ),
    )
    parser.add_argument(
        "--weights-root", type=Path, default=ROOT / "_local/server_sgqs_probe_20261014"
    )
    parser.add_argument("--replay-root", type=Path, default=ROOT / "_local/fullval_cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "_local/fullval_matches")
    parser.add_argument("--expected-annotation-sha256", default=LOCAL_ANNOTATION_SHA256)
    parser.add_argument("--expected-matcher-sha256", default=MATCHER_SCRIPT_SHA256)
    parser.add_argument("--expected-match-export-sha256", default=MATCH_EXPORT_SHA256)
    parser.add_argument(
        "--expected-checkpoint-identity-sha256", default=CHECKPOINT_IDENTITY_SHA256
    )
    parser.add_argument("--replay-device", default="cpu")
    parser.add_argument("--replay-batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=20)
    parser.add_argument("--minimum-available-gib", type=float, default=10.0)
    parser.add_argument(
        "--minimum-available-commit-gib",
        type=float,
        default=6.0,
        help="minimum Windows system commit headroom before each matcher child",
    )
    parser.add_argument("--minimum-free-disk-gib", type=float, default=40.0)
    parser.add_argument("--active-grace-seconds", type=float, default=900.0)
    parser.add_argument(
        "--resume-recent",
        action="store_true",
        help="resume recent output only after confirming that no matcher process is writing it",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("replay_batch_size", "cpu_threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.minimum_available_gib <= 0
        or args.minimum_available_commit_gib <= 0
        or args.minimum_free_disk_gib <= 0
    ):
        parser.error("resource thresholds must be positive")
    if args.active_grace_seconds < 0:
        parser.error("--active-grace-seconds cannot be negative")
    for name in (
        "expected_annotation_sha256",
        "expected_matcher_sha256",
        "expected_match_export_sha256",
        "expected_checkpoint_identity_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
            parser.error(f"--{name.replace('_', '-')} must contain 64 hex digits")
    for name in (
        "python",
        "matcher_script",
        "match_export_script",
        "annotation",
        "images_root",
        "config",
        "checkpoint_identity",
        "weights_root",
        "replay_root",
        "output_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = build_specs(args.replay_root, args.output_root)
    try:
        preflight(args, specs)
        waiting = replay_wait_reasons(args)
        if waiting:
            if args.dry_run:
                dry_run(args, specs, waiting, None)
                return 0
            details = "; ".join(
                f"e{epoch}: {','.join(names)}" for epoch, names in sorted(waiting.items())
            )
            raise MatrixError(
                "replay matrix incomplete; matcher did not start. Wait for: " + details
            )
        replay_summaries, replay_audit = validate_replay_matrix(args)
        if args.dry_run:
            dry_run(args, specs, {}, replay_summaries)
        else:
            matcher_runtime = probe_matcher_runtime(args)
            orchestrate(args, specs, replay_summaries, replay_audit, matcher_runtime)
    except MatrixError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
