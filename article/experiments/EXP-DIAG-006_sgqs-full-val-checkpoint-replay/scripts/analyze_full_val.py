#!/usr/bin/env python3
"""Analyze paired full-validation SGQS checkpoint replays.

Inputs are the compact artifacts emitted by ``full_val_replay.py`` and the
formal one-to-one shards emitted by ``coco_match_npz.py``.  The script keeps
independent per-GT coverage and COCO one-to-one matching in separate tables.

No checkpoint inference is performed here.  Every reported transition is a
paired comparison of the same annotation id at two checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import gc
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


DEFAULT_STATES = ("raw", "ema")
FORMAL_EPOCHS = tuple(range(28, 35))
FORMAL_IMAGES = 12_841
FORMAL_NONCROWD_GT = 1_458_060
DEFAULT_PAIRS = (
    *((start, start + 1) for start in FORMAL_EPOCHS[:-1]),
    (28, 30),
    (30, 34),
)
DEFAULT_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
STAGES = (
    "encoder_dense_all",
    "encoder_top640",
    "decoder_all640",
    "decoder_top300",
)
CAUSES = (
    "candidate_absence",
    "encoder_selection",
    "decoder_geometry",
    "rank_gt300",
    "one_to_one_competition",
)
METRIC_NAMES = (
    "ap",
    "ap50",
    "ap75",
    "ap_small",
    "ap_medium",
    "ap_large",
    "ar1",
    "ar10",
    "ar_max",
    "ar_small",
    "ar_medium",
    "ar_large",
)
STATUS_CODE = {"tp": 1, "fn": 2, "ignored": 3}
STATUS_NAME = {value: key for key, value in STATUS_CODE.items()}
MISSING_STATUS = np.uint8(255)


def sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    if not rows and fieldnames is None:
        raise ValueError(f"cannot infer columns for empty table: {path}")
    columns = list(fieldnames or rows[0].keys())
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return format(value, ".9g")
    return str(value)


def parse_epoch_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected EPOCH=PATH")
    epoch_text, path_text = value.split("=", 1)
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid epoch: {epoch_text!r}") from error
    if epoch < 0 or not path_text:
        raise argparse.ArgumentTypeError("expected non-negative EPOCH=PATH")
    return epoch, Path(path_text)


def parse_match_path(value: str) -> tuple[tuple[int, str], Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("expected EPOCH:STATE=PATH")
    left, path_text = value.split("=", 1)
    epoch_text, state = left.split(":", 1)
    try:
        epoch = int(epoch_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid epoch: {epoch_text!r}") from error
    if epoch < 0 or not state or not path_text:
        raise argparse.ArgumentTypeError("expected non-negative EPOCH:STATE=PATH")
    return (epoch, state), Path(path_text)


def parse_pair(value: str) -> tuple[int, int]:
    separator = ":" if ":" in value else "->" if "->" in value else None
    if separator is None:
        raise argparse.ArgumentTypeError("expected START:END")
    left, right = value.split(separator, 1)
    try:
        start, end = int(left), int(right)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid pair: {value!r}") from error
    if start == end:
        raise argparse.ArgumentTypeError("transition endpoints must differ")
    return start, end


def unique_mapping(items: Iterable[tuple], name: str) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate {name}: {key}")
        result[key] = value.resolve()
    return result


def threshold_label(threshold: float) -> str:
    return f"iou{int(round(threshold * 100)):02d}"


def stage_threshold_suffix(threshold: float) -> str:
    return f"{threshold:g}"


def domain_from_file_name(value: str) -> str:
    """Derive the wired/wireless domain from an explicit path component."""
    parts = [part.casefold() for part in PurePosixPath(value.replace("\\", "/")).parts]
    if "wireless" in parts:
        return "wireless"
    if "wired" in parts:
        return "wired"
    for part in parts:
        if part.startswith("wireless_"):
            return "wireless"
        if part.startswith("wired_"):
            return "wired"
    return "unknown"


def density_codes(counts: np.ndarray) -> np.ndarray:
    return np.select(
        [counts == 0, counts <= 32, counts <= 128, counts <= 300],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.uint8)


def short_side_codes(values: np.ndarray) -> np.ndarray:
    return np.select(
        [values < 8.0, values < 16.0, values < 32.0, values < 64.0],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.uint8)


@dataclass(frozen=True)
class Metadata:
    path: Path
    image_ids: np.ndarray
    image_file_name: np.ndarray
    annotation_ids: np.ndarray
    gt_image_index: np.ndarray
    coco_area: np.ndarray
    coco_small: np.ndarray
    coco_small_eval: np.ndarray
    hybrid_small: np.ndarray
    membership: np.ndarray
    coco_area_group: np.ndarray
    input_area_group: np.ndarray
    input_short_px: np.ndarray
    input_short_bucket: np.ndarray
    image_gt_count: np.ndarray
    gt_density_bucket: np.ndarray
    image_domain: np.ndarray
    gt_domain: np.ndarray
    id_to_row: dict[int, int]

    def group_masks(self) -> Iterator[tuple[str, str, np.ndarray]]:
        count = len(self.annotation_ids)
        all_mask = np.ones(count, dtype=bool)
        yield "cohort", "all", all_mask
        yield "cohort", "coco_small", self.coco_small
        yield "cohort", "coco_small_eval", self.coco_small_eval
        yield "cohort", "local_hybrid_small", self.hybrid_small
        yield "cohort", "coco_and_hybrid", self.membership == 0
        yield "cohort", "coco_only", self.membership == 1
        yield "cohort", "hybrid_only", self.membership == 2
        yield "cohort", "neither", self.membership == 3
        for label in sorted(set(map(str, self.image_domain))):
            yield "domain", label, self.gt_domain == label
        for code, label in enumerate(("0", "1to32", "33to128", "129to300", "ge301")):
            yield "density", label, self.gt_density_bucket == code
        for code, label in enumerate(("lt8", "8to16", "16to32", "32to64", "ge64")):
            yield "input_short", label, self.input_short_bucket == code
        for code, label in enumerate(("small", "medium", "large")):
            yield "coco_area", label, self.coco_area_group == code
            yield "input_area", label, self.input_area_group == code
        for code, label in enumerate(("coco_and_hybrid", "coco_only", "hybrid_only", "neither")):
            yield "small_partition", label, self.membership == code


def load_metadata(path: Path) -> Metadata:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "image_ids",
            "image_file_name",
            "image_gt_start",
            "annotation_ids",
            "gt_image_index",
            "coco_area",
            "coco_area_group",
            "input_area_px2",
            "input_short_px",
            "coco_small",
            "local_hybrid_small",
            "small_definition_membership",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"metadata missing keys {sorted(missing)}: {path}")
        image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
        image_file_name = np.asarray(archive["image_file_name"], dtype=np.str_)
        starts = np.asarray(archive["image_gt_start"], dtype=np.int64)
        annotation_ids = np.asarray(archive["annotation_ids"], dtype=np.int64)
        gt_image_index = np.asarray(archive["gt_image_index"], dtype=np.int64)
        coco_area = np.asarray(archive["coco_area"], dtype=np.float64)
        coco_area_group = np.asarray(archive["coco_area_group"], dtype=np.uint8)
        input_area = np.asarray(archive["input_area_px2"], dtype=np.float32)
        input_area_group = np.where(input_area < 32**2, 0, np.where(input_area < 96**2, 1, 2)).astype(np.uint8)
        input_short = np.asarray(archive["input_short_px"], dtype=np.float32)
        coco_small = np.asarray(archive["coco_small"], dtype=bool)
        coco_small_eval = coco_area <= 32**2
        hybrid_small = np.asarray(archive["local_hybrid_small"], dtype=bool)
        membership = np.asarray(archive["small_definition_membership"], dtype=np.uint8)

    if len(starts) != len(image_ids) + 1 or starts[0] != 0 or starts[-1] != len(annotation_ids):
        raise ValueError(f"invalid image_gt_start: {path}")
    if len(set(map(int, annotation_ids))) != len(annotation_ids):
        raise ValueError(f"duplicate annotation ids: {path}")
    if len(image_file_name) != len(image_ids) or len(gt_image_index) != len(annotation_ids):
        raise ValueError(f"metadata array length mismatch: {path}")
    image_gt_count = np.diff(starts).astype(np.int64)
    image_density_bucket = density_codes(image_gt_count)
    image_domain = np.asarray([domain_from_file_name(str(value)) for value in image_file_name])
    return Metadata(
        path=path,
        image_ids=image_ids,
        image_file_name=image_file_name,
        annotation_ids=annotation_ids,
        gt_image_index=gt_image_index,
        coco_area=coco_area,
        coco_small=coco_small,
        coco_small_eval=coco_small_eval,
        hybrid_small=hybrid_small,
        membership=membership,
        coco_area_group=coco_area_group,
        input_area_group=input_area_group,
        input_short_px=input_short,
        input_short_bucket=short_side_codes(input_short),
        image_gt_count=image_gt_count,
        gt_density_bucket=image_density_bucket[gt_image_index],
        image_domain=image_domain,
        gt_domain=image_domain[gt_image_index],
        id_to_row={int(value): index for index, value in enumerate(annotation_ids)},
    )


def assert_metadata_compatible(reference: Metadata, candidate_path: Path) -> None:
    with np.load(candidate_path, allow_pickle=False) as archive:
        checks = {
            "image_ids": reference.image_ids,
            "annotation_ids": reference.annotation_ids,
            "gt_image_index": reference.gt_image_index,
            "coco_area": reference.coco_area,
            "coco_small": reference.coco_small,
            "local_hybrid_small": reference.hybrid_small,
            "small_definition_membership": reference.membership,
        }
        for key, expected in checks.items():
            if key not in archive.files or not np.array_equal(archive[key], expected):
                raise ValueError(f"metadata mismatch for {key}: {candidate_path}")


def validate_file_identity(
    path: Path,
    descriptor: Mapping,
    *,
    label: str,
    required: bool,
    require_bytes: bool = True,
) -> None:
    """Verify a summary-declared immutable artifact before reading it."""

    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = descriptor.get("bytes")
    expected_sha = descriptor.get("sha256")
    if required and (not expected_sha or (require_bytes and expected_bytes is None)):
        raise ValueError(f"formal {label} identity is incomplete: {path}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"{label} byte-size mismatch: {path}")
    if expected_sha and sha256(path) != str(expected_sha):
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


@dataclass(frozen=True)
class ReplayState:
    epoch: int
    state: str
    directory: Path
    summary_path: Path
    metadata_path: Path
    prediction_path: Path
    stage_path: Path
    summary: dict
    result: dict


def load_replays(
    replay_dirs: Mapping[int, Path], split: str, states: Sequence[str], *, formal: bool
) -> tuple[Metadata, dict[tuple[int, str], ReplayState], list[dict], list[dict]]:
    records: dict[tuple[int, str], ReplayState] = {}
    metrics: list[dict] = []
    per_iou_metrics: list[dict] = []
    reference_metadata: Metadata | None = None
    for epoch, directory in sorted(replay_dirs.items()):
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = read_json(summary_path)
        if summary.get("last_epoch") is not None and int(summary["last_epoch"]) != epoch:
            raise ValueError(f"epoch {epoch} disagrees with {summary_path}: {summary.get('last_epoch')}")
        split_info = summary.get("splits", {}).get(split)
        if not isinstance(split_info, dict):
            raise ValueError(f"split {split!r} absent from {summary_path}")
        metadata_path = directory / split_info.get("metadata", f"{split}.metadata.npz")
        metadata_descriptor = {
            "sha256": split_info.get("metadata_sha256"),
            "bytes": split_info.get("metadata_bytes"),
        }
        validate_file_identity(
            metadata_path,
            metadata_descriptor,
            label="metadata",
            required=formal,
            require_bytes=False,
        )
        if reference_metadata is None:
            reference_metadata = load_metadata(metadata_path)
        else:
            assert_metadata_compatible(reference_metadata, metadata_path)
        for state in states:
            key = f"{split}/{state}"
            result = summary.get("results", {}).get(key)
            if not isinstance(result, dict):
                raise ValueError(f"result {key!r} absent from {summary_path}")
            artifact = result.get("artifacts", {}).get("gt_stages", {})
            stage_path = directory / artifact.get("file", f"{split}.{state}.gt_stages.npz")
            prediction_artifact = result.get("artifacts", {}).get("predictions", {})
            prediction_path = directory / prediction_artifact.get(
                "file", f"{split}.{state}.predictions.npz"
            )
            validate_file_identity(
                stage_path,
                artifact,
                label=f"e{epoch}/{state} stage artifact",
                required=formal,
            )
            if formal or prediction_path.is_file() or prediction_artifact:
                validate_file_identity(
                    prediction_path,
                    prediction_artifact,
                    label=f"e{epoch}/{state} prediction artifact",
                    required=formal,
                )
            if int(result.get("noncrowd_annotations", -1)) != len(reference_metadata.annotation_ids):
                raise ValueError(f"GT count mismatch for e{epoch}/{state}")
            if int(result.get("images", -1)) != len(reference_metadata.image_ids):
                raise ValueError(f"image count mismatch for e{epoch}/{state}")
            record = ReplayState(
                epoch=epoch,
                state=state,
                directory=directory,
                summary_path=summary_path,
                metadata_path=metadata_path,
                prediction_path=prediction_path,
                stage_path=stage_path,
                summary=summary,
                result=result,
            )
            records[(epoch, state)] = record
            row = {
                "epoch": epoch,
                "state": state,
                "last_epoch": summary.get("last_epoch"),
                "checkpoint_sha256": summary.get("checkpoint_sha256", ""),
                "images": result["images"],
                "noncrowd_annotations": result["noncrowd_annotations"],
            }
            for budget in (300, 640):
                values = result.get(f"coco_maxdets{budget}")
                if not isinstance(values, list) or len(values) != len(METRIC_NAMES):
                    raise ValueError(f"invalid COCO metric vector for e{epoch}/{state}/d{budget}")
                for name, value in zip(METRIC_NAMES, values):
                    row[f"{name}_d{budget}"] = float(value)
                per_iou = result.get(f"coco_small_per_iou_maxdets{budget}")
                if not isinstance(per_iou, list) or len(per_iou) != len(DEFAULT_THRESHOLDS):
                    raise ValueError(
                        f"missing 10-slice COCO-small metrics for e{epoch}/{state}/d{budget}"
                    )
                observed_thresholds = []
                for item in per_iou:
                    if not isinstance(item, dict) or not {"iou", "ap", "ar"}.issubset(item):
                        raise ValueError(f"invalid per-IoU metric row for e{epoch}/{state}/d{budget}")
                    observed_thresholds.append(round(float(item["iou"]), 2))
                    per_iou_metrics.append(
                        {
                            "epoch": epoch,
                            "state": state,
                            "max_dets": budget,
                            "iou_threshold": float(item["iou"]),
                            "ap_small": float(item["ap"]),
                            "ar_small": float(item["ar"]),
                        }
                    )
                if tuple(observed_thresholds) != DEFAULT_THRESHOLDS:
                    raise ValueError(
                        f"unexpected per-IoU grid for e{epoch}/{state}/d{budget}: "
                        f"{observed_thresholds}"
                    )
            metrics.append(row)
    if reference_metadata is None:
        raise RuntimeError("no replay metadata loaded")
    return reference_metadata, records, metrics, per_iou_metrics


def validate_formal_contract(
    args: argparse.Namespace,
    replay_dirs: Mapping[int, Path],
    meta: Metadata,
    records: Mapping[tuple[int, str], ReplayState],
) -> dict:
    """Fail closed unless the complete paper-analysis matrix is present."""

    if tuple(sorted(replay_dirs)) != FORMAL_EPOCHS:
        raise ValueError(f"formal analysis requires epochs {FORMAL_EPOCHS}")
    if tuple(args.states) != DEFAULT_STATES:
        raise ValueError(f"formal analysis requires states {DEFAULT_STATES} in this order")
    if tuple(sorted(set(map(float, args.thresholds)))) != DEFAULT_THRESHOLDS:
        raise ValueError("formal analysis requires IoU=.50:.05:.95")
    if tuple(sorted(set(map(int, args.budgets)))) != (300, 640):
        raise ValueError("formal analysis requires maxDets=300/640")
    if args.split != "val" or tuple(args.scopes) != ("all", "coco_small_eval"):
        raise ValueError("formal analysis requires val scopes all,coco_small_eval")
    if args.transition_budget != 300:
        raise ValueError("formal analysis requires transition maxDets=300")
    if len(meta.image_ids) != FORMAL_IMAGES or len(meta.annotation_ids) != FORMAL_NONCROWD_GT:
        raise ValueError(
            f"formal population mismatch: images={len(meta.image_ids)}, "
            f"noncrowd_gt={len(meta.annotation_ids)}"
        )

    config_hashes: set[str] = set()
    annotation_hashes: set[str] = set()
    backends: set[str] = set()
    for epoch in FORMAL_EPOCHS:
        summary = records[(epoch, "raw")].summary
        if summary.get("max_images") is not None or summary.get("image_ids") is not None:
            raise ValueError(f"formal replay e{epoch} is only a subset")
        if tuple(summary.get("states", ())) != DEFAULT_STATES:
            raise ValueError(f"formal replay e{epoch} lacks raw+EMA")
        config_hash = str(summary.get("config_sha256", ""))
        annotation_hash = str(
            summary.get("splits", {}).get(args.split, {}).get("annotation_sha256", "")
        )
        backend = summary.get("inference_identity")
        if not config_hash or not annotation_hash or not isinstance(backend, dict):
            raise ValueError(f"formal replay identity incomplete: e{epoch}")
        config_hashes.add(config_hash)
        annotation_hashes.add(annotation_hash)
        backends.add(json.dumps(backend, sort_keys=True, separators=(",", ":")))
    if len(config_hashes) != 1:
        raise ValueError("formal replays use different configs")
    if len(annotation_hashes) != 1:
        raise ValueError("formal replays use different annotations")
    if len(backends) != 1:
        raise ValueError("formal replays use different inference backends")
    return {
        "config_sha256": next(iter(config_hashes)),
        "annotation_sha256": next(iter(annotation_hashes)),
        "inference_identity": json.loads(next(iter(backends))),
    }


def validate_match_protocol(
    record: ReplayState,
    match_dir: Path,
    *,
    split: str,
    formal: bool,
) -> dict:
    protocol = read_json(match_dir / "protocol.json")
    prediction_hash = record.result.get("artifacts", {}).get("predictions", {}).get("sha256")
    annotation_hash = record.summary.get("splits", {}).get(split, {}).get("annotation_sha256")
    if protocol.get("prediction_sha256") not in (None, prediction_hash):
        raise ValueError(f"matcher prediction identity mismatch: {match_dir}")
    if protocol.get("annotation_sha256") not in (None, annotation_hash):
        raise ValueError(f"matcher annotation identity mismatch: {match_dir}")
    if formal:
        if protocol.get("prediction_sha256") != prediction_hash:
            raise ValueError(f"formal matcher lacks replay prediction hash: {match_dir}")
        if protocol.get("annotation_sha256") != annotation_hash:
            raise ValueError(f"formal matcher lacks replay annotation hash: {match_dir}")
        if tuple(map(int, protocol.get("max_dets", ()))) != (300, 640):
            raise ValueError(f"formal matcher maxDets mismatch: {match_dir}")
        if tuple(map(float, protocol.get("iou_thresholds", ()))) != DEFAULT_THRESHOLDS:
            raise ValueError(f"formal matcher IoU grid mismatch: {match_dir}")
        if int(protocol.get("images", -1)) != FORMAL_IMAGES:
            raise ValueError(f"formal matcher population mismatch: {match_dir}")
        if int(protocol.get("queries_per_image", -1)) != 640:
            raise ValueError(f"formal matcher query count mismatch: {match_dir}")
    return protocol


def validate_match_matrix_audit(
    match_dirs: Mapping[tuple[int, str], Path],
    records: Mapping[tuple[int, str], ReplayState],
    formal_identity: Mapping | None,
    *,
    formal: bool,
) -> dict | None:
    """Bind all 14 matcher directories to one recorded runtime matrix."""

    roots = {directory.parent.parent for directory in match_dirs.values()}
    if len(roots) != 1:
        if formal:
            raise ValueError("formal matcher directories do not share one matrix root")
        return None
    root = next(iter(roots))
    path = root / "matrix.complete.json"
    if not path.is_file():
        if formal:
            raise FileNotFoundError(path)
        return None
    matrix = read_json(path)
    if matrix.get("schema") != "sgqs_coco_match_matrix_orchestration_v1":
        raise ValueError(f"unknown matcher matrix schema: {path}")
    if matrix.get("status") != "complete":
        raise ValueError(f"matcher matrix is not complete: {path}")
    if formal:
        if formal_identity is None:
            raise RuntimeError("formal replay identity was not established")
        if tuple(map(int, matrix.get("epochs", ()))) != FORMAL_EPOCHS:
            raise ValueError("matcher matrix epoch identity mismatch")
        if tuple(map(int, matrix.get("max_dets", ()))) != (300, 640):
            raise ValueError("matcher matrix maxDets identity mismatch")
        if tuple(map(float, matrix.get("iou_thresholds", ()))) != DEFAULT_THRESHOLDS:
            raise ValueError("matcher matrix IoU identity mismatch")
        if matrix.get("annotation_sha256") != formal_identity["annotation_sha256"]:
            raise ValueError("matcher matrix annotation identity mismatch")
        if matrix.get("config_sha256") != formal_identity["config_sha256"]:
            raise ValueError("matcher matrix config identity mismatch")
        replay_runtime = matrix.get("replay_runtime_backend")
        matcher_runtime = matrix.get("matcher_runtime")
        if not isinstance(replay_runtime, dict) or not isinstance(matcher_runtime, dict):
            raise ValueError("matcher matrix runtime identity is incomplete")
        summary_backend = formal_identity["inference_identity"]
        for summary_key, runtime_key in (
            ("torch_version", "torch_version"),
            ("torch_cuda_version", "torch_cuda_version"),
            ("device", "requested_device"),
        ):
            if summary_backend.get(summary_key) != replay_runtime.get(runtime_key):
                raise ValueError(f"matcher/replay runtime mismatch: {summary_key}")
        prediction_inputs = matrix.get("prediction_inputs")
        matrix_states = matrix.get("states")
        checkpoint_contract = matrix.get("checkpoint_contract")
        if not all(isinstance(value, dict) for value in (prediction_inputs, matrix_states, checkpoint_contract)):
            raise ValueError("matcher matrix per-state identity is incomplete")
        expected_state_keys = {
            f"e{epoch}/{state}" for epoch in FORMAL_EPOCHS for state in DEFAULT_STATES
        }
        if set(prediction_inputs) != expected_state_keys or set(matrix_states) != expected_state_keys:
            raise ValueError("matcher matrix state identity mismatch")
        for epoch in FORMAL_EPOCHS:
            checkpoint_name = f"checkpoint{epoch:04d}.pth"
            checkpoint_row = checkpoint_contract.get(checkpoint_name, {})
            replay_checkpoint_hash = str(records[(epoch, "raw")].summary.get("checkpoint_sha256", ""))
            if checkpoint_row.get("file_sha256") != replay_checkpoint_hash:
                raise ValueError(f"matcher matrix checkpoint identity mismatch: e{epoch}")
            for state in DEFAULT_STATES:
                key = f"e{epoch}/{state}"
                record = records[(epoch, state)]
                prediction_hash = record.result["artifacts"]["predictions"]["sha256"]
                evidence = prediction_inputs.get(key, {})
                if evidence.get("prediction_sha256") != prediction_hash:
                    raise ValueError(f"matcher matrix prediction identity mismatch: {key}")
                if evidence.get("replay_summary_sha256") != sha256(record.summary_path):
                    raise ValueError(f"matcher matrix replay-summary identity mismatch: {key}")
                state_marker = record.directory / f"val.{state}.complete.json"
                if (
                    not state_marker.is_file()
                    or evidence.get("replay_state_marker_sha256") != sha256(state_marker)
                ):
                    raise ValueError(f"matcher matrix replay-state identity mismatch: {key}")
                complete_path = match_dirs[(epoch, state)] / "complete.json"
                state_evidence = matrix_states.get(key, {})
                if state_evidence.get("complete_sha256") != sha256(complete_path):
                    raise ValueError(f"matcher matrix completion identity mismatch: {key}")
                complete_payload = read_json(complete_path)
                if state_evidence.get("gt_shard_manifest_sha256") != complete_payload.get(
                    "gt_shard_manifest_sha256"
                ):
                    raise ValueError(f"matcher matrix GT-root identity mismatch: {key}")
    return {
        "logical_name": "matcher_matrix/matrix.complete.json",
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "replay_matrix_complete_sha256": matrix.get("replay_matrix_complete_sha256"),
        "replay_runtime_backend": matrix.get("replay_runtime_backend"),
        "matcher_runtime": matrix.get("matcher_runtime"),
    }


def per_iou_gap_rows(per_iou_metrics: Sequence[Mapping]) -> list[dict]:
    index = {
        (
            int(row["epoch"]),
            str(row["state"]),
            round(float(row["iou_threshold"]), 2),
            int(row["max_dets"]),
        ): row
        for row in per_iou_metrics
    }
    rows = []
    for epoch, state, threshold in sorted({key[:3] for key in index}):
        at300 = index[(epoch, state, threshold, 300)]
        at640 = index[(epoch, state, threshold, 640)]
        rows.append(
            {
                "epoch": epoch,
                "state": state,
                "iou_threshold": threshold,
                "ap_small_d300": at300["ap_small"],
                "ap_small_d640": at640["ap_small"],
                "ap_d640_minus_d300": float(at640["ap_small"]) - float(at300["ap_small"]),
                "ar_small_d300": at300["ar_small"],
                "ar_small_d640": at640["ar_small"],
                "ar_d640_minus_d300": float(at640["ar_small"]) - float(at300["ar_small"]),
            }
        )
    return rows


def validate_stage_archive(path: Path, gt_count: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if "stage_names" in archive.files and tuple(map(str, archive["stage_names"])) != STAGES:
            raise ValueError(f"unexpected stage order: {path}")
        for stage in STAGES:
            key = f"{stage}__best_iou"
            if key not in archive.files or archive[key].shape != (gt_count,):
                raise ValueError(f"invalid or missing {key}: {path}")


def stage_coverage_rows(
    meta: Metadata,
    records: Mapping[tuple[int, str], ReplayState],
    thresholds: Sequence[float],
) -> tuple[list[dict], int]:
    rows: list[dict] = []
    reconciliations = 0
    masks = list(meta.group_masks())
    for (epoch, state), record in sorted(records.items()):
        validate_stage_archive(record.stage_path, len(meta.annotation_ids))
        with np.load(record.stage_path, allow_pickle=False) as archive:
            best = {stage: np.asarray(archive[f"{stage}__best_iou"], dtype=np.float32) for stage in STAGES}
            for axis, group, mask in masks:
                n_gt = int(mask.sum())
                for threshold in thresholds:
                    for stage in STAGES:
                        covered = int(np.count_nonzero(mask & (best[stage] >= threshold)))
                        rows.append(
                            {
                                "epoch": epoch,
                                "state": state,
                                "axis": axis,
                                "group": group,
                                "n_gt": n_gt,
                                "iou_threshold": threshold,
                                "stage": stage,
                                "covered": covered,
                                "coverage": covered / n_gt if n_gt else None,
                            }
                        )
                        if axis == "cohort":
                            expected = (
                                record.result.get("independent_gt_coverage", {})
                                .get(group, {})
                                .get(f"iou{threshold:g}", {})
                                .get(stage)
                            )
                            # The first compact replay schema materialized
                            # summary counters only for IoU=.50/.75.  best_iou
                            # still supports all ten COCO AP thresholds.
                            if expected is None:
                                continue
                            if int(expected) != covered:
                                raise ValueError(
                                    f"summary/stage coverage mismatch e{epoch}/{state}/"
                                    f"{group}/iou{threshold:g}/{stage}: {expected} != {covered}"
                                )
                            reconciliations += 1
    return rows, reconciliations


def raw_ema_rows(metrics: Sequence[Mapping], states: Sequence[str]) -> list[dict]:
    if not {"raw", "ema"}.issubset(states):
        return []
    by_key = {(int(row["epoch"]), str(row["state"])): row for row in metrics}
    rows = []
    for epoch in sorted({key[0] for key in by_key}):
        raw, ema = by_key[(epoch, "raw")], by_key[(epoch, "ema")]
        for budget in (300, 640):
            for metric in METRIC_NAMES:
                raw_value = float(raw[f"{metric}_d{budget}"])
                ema_value = float(ema[f"{metric}_d{budget}"])
                rows.append(
                    {
                        "epoch": epoch,
                        "max_dets": budget,
                        "metric": metric,
                        "raw": raw_value,
                        "ema": ema_value,
                        "ema_minus_raw": ema_value - raw_value,
                    }
                )
    return rows


@dataclass
class MatchData:
    statuses: dict[tuple[str, float, int], np.ndarray]
    details: dict[tuple[str, float, str], np.ndarray]
    source_states: tuple[str, ...]


def match_shards(match_dir: Path, *, formal: bool) -> list[Path]:
    complete_path = match_dir / "complete.json"
    protocol_path = match_dir / "protocol.json"
    aggregate_path = match_dir / "match_aggregates.csv"
    for path in (complete_path, protocol_path, aggregate_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    complete = read_json(complete_path)
    protocol = read_json(protocol_path)
    if formal and complete.get("schema") != "coco_one_to_one_npz_complete_v2":
        raise ValueError(f"formal matcher complete marker is not v2: {complete_path}")
    expected_protocol_hash = complete.get("protocol_sha256")
    expected_aggregate_hash = complete.get("aggregate_sha256")
    if formal and (not expected_protocol_hash or not expected_aggregate_hash):
        raise ValueError(f"formal matcher complete marker lacks hash chain: {complete_path}")
    if expected_protocol_hash and sha256(protocol_path) != expected_protocol_hash:
        raise ValueError(f"matcher protocol SHA-256 mismatch: {protocol_path}")
    if expected_aggregate_hash and sha256(aggregate_path) != expected_aggregate_hash:
        raise ValueError(f"matcher aggregate SHA-256 mismatch: {aggregate_path}")
    gt_manifest_name = complete.get("gt_shard_manifest")
    gt_manifest_hash = complete.get("gt_shard_manifest_sha256")
    if formal and (not gt_manifest_name or not gt_manifest_hash):
        raise ValueError(f"formal matcher GT shard manifest is missing: {complete_path}")
    gt_manifest = None
    if gt_manifest_name:
        gt_manifest_path = match_dir / str(gt_manifest_name)
        if not gt_manifest_path.is_file() or sha256(gt_manifest_path) != gt_manifest_hash:
            raise ValueError(f"matcher GT shard manifest SHA-256 mismatch: {gt_manifest_path}")
        gt_manifest = read_json(gt_manifest_path)
        if gt_manifest.get("schema") != "coco_gt_match_shard_manifest_v1":
            raise ValueError(f"unknown matcher GT shard manifest schema: {gt_manifest_path}")
        if gt_manifest.get("state") != protocol.get("state"):
            raise ValueError(f"matcher GT shard manifest state mismatch: {gt_manifest_path}")
    shards = sorted(match_dir.glob("gt_matches.part*.csv.gz"))
    if int(complete.get("shards", -1)) != len(shards):
        raise ValueError(f"match shard count mismatch: {match_dir}")
    if not shards:
        raise ValueError(f"no match shards: {match_dir}")
    summaries = sorted(match_dir.glob("summary.part*.json"))
    if formal and len(summaries) != len(shards):
        raise ValueError(f"formal matcher shard summaries are incomplete: {match_dir}")
    if summaries:
        if len(summaries) != len(shards):
            raise ValueError(f"matcher shard/summary count mismatch: {match_dir}")
        for index, (shard, summary_path) in enumerate(zip(shards, summaries)):
            summary = read_json(summary_path)
            rooted = None
            if gt_manifest is not None:
                manifest_shards = gt_manifest.get("shards", ())
                if len(manifest_shards) != len(shards):
                    raise ValueError(f"matcher GT shard manifest count mismatch: {match_dir}")
                rooted = manifest_shards[index]
                if int(rooted.get("index", -1)) != index:
                    raise ValueError(f"matcher GT shard manifest index mismatch: {match_dir}")
                if (
                    rooted.get("summary_file") != summary_path.name
                    or rooted.get("summary_sha256") != sha256(summary_path)
                    or rooted.get("gt_matches_file") != shard.name
                    or rooted.get("gt_matches_sha256") != sha256(shard)
                ):
                    raise ValueError(f"matcher rooted GT shard identity mismatch: {summary_path}")
            if int(summary.get("shard_index", -1)) != index:
                raise ValueError(f"matcher summary index mismatch: {summary_path}")
            if summary.get("gt_matches_file") != shard.name:
                raise ValueError(f"matcher summary names different GT shard: {summary_path}")
            expected_gt_hash = summary.get("gt_matches_sha256")
            if not expected_gt_hash or sha256(shard) != expected_gt_hash:
                raise ValueError(f"matcher GT shard SHA-256 mismatch: {shard}")
            if formal and summary.get("schema") != "coco_one_to_one_match_shard_v2":
                raise ValueError(f"formal matcher shard is not v2: {summary_path}")
            status_name = summary.get("detection_status_file")
            status_hash = summary.get("detection_status_sha256")
            if formal and (not status_name or not status_hash):
                raise ValueError(f"formal matcher status identity missing: {summary_path}")
            if status_name:
                status_path = match_dir / str(status_name)
                if not status_path.is_file() or sha256(status_path) != status_hash:
                    raise ValueError(f"matcher detection-status SHA-256 mismatch: {status_path}")
                if rooted is not None and (
                    rooted.get("detection_status_file") != status_path.name
                    or rooted.get("detection_status_sha256") != status_hash
                ):
                    raise ValueError(f"matcher rooted status shard identity mismatch: {summary_path}")
    manifest_name = complete.get("detection_status_manifest")
    manifest_hash = complete.get("detection_status_manifest_sha256")
    if formal and (not manifest_name or not manifest_hash):
        raise ValueError(f"formal matcher detection manifest is missing: {complete_path}")
    if manifest_name:
        manifest_path = match_dir / str(manifest_name)
        if not manifest_path.is_file() or sha256(manifest_path) != manifest_hash:
            raise ValueError(f"matcher detection manifest SHA-256 mismatch: {manifest_path}")
    return shards


def load_match_data(
    match_dir: Path,
    meta: Metadata,
    scopes: Sequence[str],
    thresholds: Sequence[float],
    budgets: Sequence[int],
    *,
    formal: bool,
) -> MatchData:
    shards = match_shards(match_dir, formal=formal)
    gt_count = len(meta.annotation_ids)
    statuses = {
        (scope, threshold, budget): np.full(gt_count, MISSING_STATUS, dtype=np.uint8)
        for scope in scopes
        for threshold in thresholds
        for budget in budgets
    }
    details: dict[tuple[str, float, str], np.ndarray] = {}
    for scope in scopes:
        for threshold in thresholds:
            for field in ("query_index", "rank"):
                details[(scope, threshold, field)] = np.full(gt_count, -1, dtype=np.int32)
            for field in ("score", "iou"):
                details[(scope, threshold, field)] = np.full(gt_count, np.nan, dtype=np.float32)
    seen = np.zeros(gt_count, dtype=bool)
    source_states: set[str] = set()
    required_columns = {"annotation_id", "state"}
    for scope in scopes:
        for threshold in thresholds:
            prefix = f"{scope}_{threshold_label(threshold)}"
            required_columns.update(f"{prefix}_status_d{budget}" for budget in budgets)
            required_columns.update(f"{prefix}_{field}" for field in ("query_index", "rank", "score", "iou"))

    for shard in shards:
        with gzip.open(shard, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"match shard missing columns {sorted(missing)}: {shard}")
            for line_number, row in enumerate(reader, 2):
                annotation_id = int(row["annotation_id"])
                index = meta.id_to_row.get(annotation_id)
                if index is None:
                    # The match export retains crowd annotations; replay metadata does not.
                    continue
                if seen[index]:
                    raise ValueError(f"duplicate annotation {annotation_id}: {shard}:{line_number}")
                seen[index] = True
                source_states.add(row["state"])
                for scope in scopes:
                    for threshold in thresholds:
                        prefix = f"{scope}_{threshold_label(threshold)}"
                        for budget in budgets:
                            status = row[f"{prefix}_status_d{budget}"]
                            if status not in STATUS_CODE:
                                raise ValueError(f"unexpected GT status {status!r}: {shard}:{line_number}")
                            statuses[(scope, threshold, budget)][index] = STATUS_CODE[status]
                        for field in ("query_index", "rank"):
                            text = row[f"{prefix}_{field}"]
                            details[(scope, threshold, field)][index] = int(text) if text else -1
                        for field in ("score", "iou"):
                            text = row[f"{prefix}_{field}"]
                            details[(scope, threshold, field)][index] = float(text) if text else np.nan
    if not seen.all():
        missing_ids = meta.annotation_ids[~seen]
        raise ValueError(f"{len(missing_ids)} non-crowd annotations absent from match shards: {missing_ids[:5]}")
    for key, values in statuses.items():
        if np.any(values == MISSING_STATUS):
            raise ValueError(f"unfilled match statuses for {key}: {match_dir}")
    return MatchData(statuses=statuses, details=details, source_states=tuple(sorted(source_states)))


def merge_match_aggregates(
    match_dirs: Mapping[tuple[int, str], Path]
) -> list[dict]:
    rows = []
    for (epoch, state), directory in sorted(match_dirs.items()):
        path = directory / "match_aggregates.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({"epoch": epoch, "analysis_state": state, **row})
    return rows


def _quantile(values: np.ndarray, probability: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values.astype(np.float64, copy=False), probability))


def _score_fields(prefix: str, values: np.ndarray) -> dict[str, float | int | None]:
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_score_mean": float(values.mean()) if values.size else None,
        f"{prefix}_score_p10": _quantile(values, 0.10),
        f"{prefix}_score_median": _quantile(values, 0.50),
        f"{prefix}_score_p90": _quantile(values, 0.90),
    }


def compute_global_score_pr_from_detection_status(
    record: ReplayState,
    match_dir: Path,
    thresholds: Sequence[float],
    budgets: Sequence[int],
    expected_per_iou: Mapping[tuple[int, str, int, float], Mapping],
    scratch_root: Path,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Reconstruct global PR from the matcher's persisted detection states.

    The matcher already ran the repository COCO backend per image.  Reusing
    its query-index status tensors avoids a second evaluator pass over all 14
    checkpoint states.  A disk-backed staging tensor keeps resident memory
    bounded while retaining the exact image-major/query-index tie order.
    """

    from tools.table_cells.coco_match_npz import (  # imported lazily for table-only use
        STATUS_ORDER as MATCH_STATUS_ORDER,
    )

    manifest_path = match_dir / "detection_status_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema") != "coco_detection_status_manifest_v1":
        raise ValueError(f"unknown detection-status manifest schema: {manifest_path}")
    complete = read_json(match_dir / "complete.json")
    if complete.get("detection_status_manifest") not in (None, manifest_path.name):
        raise ValueError(f"complete marker names a different detection manifest: {match_dir}")
    expected_manifest_hash = complete.get("detection_status_manifest_sha256")
    if expected_manifest_hash is not None and sha256(manifest_path) != expected_manifest_hash:
        raise ValueError(f"detection-status manifest hash mismatch: {manifest_path}")
    manifest_scopes = tuple(map(str, manifest.get("scope_names", ())))
    manifest_thresholds = tuple(map(float, manifest.get("iou_thresholds", ())))
    manifest_budgets = tuple(map(int, manifest.get("max_dets", ())))
    status_codes = {str(key): int(value) for key, value in manifest.get("status_codes", {}).items()}
    expected_codes = {name: index for index, name in enumerate(MATCH_STATUS_ORDER)}
    if status_codes != expected_codes:
        raise ValueError(f"unexpected detection status codes: {manifest_path}")
    scope_indexes = {}
    for scope in ("all", "coco_small_eval"):
        if scope in manifest_scopes:
            scope_indexes[scope] = manifest_scopes.index(scope)
    if not scope_indexes:
        raise ValueError(f"no supported detection scopes in {manifest_path}")
    threshold_indexes = {}
    for threshold in thresholds:
        matches = [
            index
            for index, value in enumerate(manifest_thresholds)
            if math.isclose(value, float(threshold), rel_tol=0.0, abs_tol=1e-12)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"IoU threshold {threshold:g} is missing or ambiguous in {manifest_path}"
            )
        threshold_indexes[float(threshold)] = matches[0]
    budget_indexes = {}
    for budget in budgets:
        if int(budget) not in manifest_budgets:
            raise ValueError(f"maxDets={budget} is absent from {manifest_path}")
        budget_indexes[int(budget)] = manifest_budgets.index(int(budget))

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError(f"no detection-status shards in {manifest_path}")
    total_images = sum(int(item.get("images", -1)) for item in shards)
    if total_images <= 0:
        raise ValueError(f"invalid detection-status image count in {manifest_path}")
    if total_images != int(record.result.get("images", -1)):
        raise ValueError(
            f"detection-status image count disagrees with replay e{record.epoch}/{record.state}"
        )

    tp_code = expected_codes["tp"]
    fp_code = expected_codes["fp"]
    ignored_code = expected_codes["ignored"]
    truncated_code = expected_codes["truncated"]
    source_state = str(manifest.get("state", record.state))
    score_bins = np.linspace(0.0, 1.0, 101)
    recall_grid = np.linspace(0.0, 1.0, 101)

    with tempfile.TemporaryDirectory(prefix="global_pr_status_", dir=scratch_root) as temporary:
        temporary_path = Path(temporary)
        score_matrix = None
        statuses = None
        positive_counts = np.zeros(len(manifest_scopes), dtype=np.int64)
        write_offset = 0
        previous_image_id: int | None = None
        query_count: int | None = None
        for expected_shard_index, shard in enumerate(shards):
            if int(shard.get("index", -1)) != expected_shard_index:
                raise ValueError(f"non-contiguous detection-status shard indexes: {manifest_path}")
            shard_path = match_dir / str(shard.get("file", ""))
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if sha256(shard_path) != str(shard.get("sha256", "")):
                raise ValueError(f"detection-status shard hash mismatch: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as archive:
                if str(archive["schema"]) != "coco_detection_status_shard_v1":
                    raise ValueError(f"unknown detection-status shard schema: {shard_path}")
                shard_state = str(archive["state"])
                if shard_state != source_state:
                    raise ValueError(f"detection-status state mismatch: {shard_path}")
                image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
                shard_scores = np.asarray(archive["scores"], dtype=np.float32)
                shard_status = np.asarray(archive["status"], dtype=np.uint8)
                shard_positive = np.asarray(archive["nonignored_gt_count"], dtype=np.int64)
                shard_ranks = np.asarray(archive["query_rank"])
                if tuple(map(int, archive["max_dets"])) != manifest_budgets:
                    raise ValueError(f"detection-status budget axis mismatch: {shard_path}")
                if tuple(map(str, archive["scope_names"])) != manifest_scopes:
                    raise ValueError(f"detection-status scope axis mismatch: {shard_path}")
                if not np.array_equal(
                    np.asarray(archive["iou_thresholds"], dtype=np.float64),
                    np.asarray(manifest_thresholds, dtype=np.float64),
                ):
                    raise ValueError(f"detection-status IoU axis mismatch: {shard_path}")
                if tuple(map(str, archive["status_names"])) != tuple(MATCH_STATUS_ORDER):
                    raise ValueError(f"detection-status names mismatch: {shard_path}")
                expected_images = int(shard.get("images", -1))
                if image_ids.ndim != 1 or len(image_ids) != expected_images:
                    raise ValueError(f"detection-status image shape mismatch: {shard_path}")
                if len(image_ids) and (
                    np.any(image_ids[1:] <= image_ids[:-1])
                    or (previous_image_id is not None and int(image_ids[0]) <= previous_image_id)
                ):
                    raise ValueError(f"detection-status image order is not globally increasing: {shard_path}")
                if len(image_ids):
                    previous_image_id = int(image_ids[-1])
                if shard_scores.ndim != 2 or shard_scores.shape[0] != expected_images:
                    raise ValueError(f"detection-status score shape mismatch: {shard_path}")
                if not np.isfinite(shard_scores).all():
                    raise ValueError(f"non-finite detection score: {shard_path}")
                if query_count is None:
                    query_count = int(shard_scores.shape[1])
                    score_matrix = np.lib.format.open_memmap(
                        temporary_path / "scores.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(total_images, query_count),
                    )
                    statuses = np.lib.format.open_memmap(
                        temporary_path / "statuses.npy",
                        mode="w+",
                        dtype=np.uint8,
                        shape=(
                            len(manifest_budgets),
                            len(manifest_scopes),
                            len(manifest_thresholds),
                            total_images,
                            query_count,
                        ),
                    )
                if query_count is None or score_matrix is None or statuses is None:
                    raise RuntimeError(f"failed to initialize detection-status staging: {shard_path}")
                expected_status_shape = (
                    len(manifest_budgets),
                    len(manifest_scopes),
                    len(manifest_thresholds),
                    expected_images,
                    query_count,
                )
                if shard_scores.shape[1] != query_count or shard_status.shape != expected_status_shape:
                    raise ValueError(f"detection-status tensor shape mismatch: {shard_path}")
                if shard_positive.shape != (len(manifest_scopes), expected_images):
                    raise ValueError(f"detection-status positive-count shape mismatch: {shard_path}")
                if shard_ranks.shape != shard_scores.shape:
                    raise ValueError(f"detection-status query-rank shape mismatch: {shard_path}")
                if shard_status.size and int(shard_status.max()) >= len(MATCH_STATUS_ORDER):
                    raise ValueError(f"unknown detection status value: {shard_path}")
                stop = write_offset + expected_images
                score_matrix[write_offset:stop] = shard_scores
                statuses[:, :, :, write_offset:stop] = shard_status
                positive_counts += shard_positive.sum(axis=1, dtype=np.int64)
                write_offset = stop
        if write_offset != total_images or score_matrix is None or statuses is None:
            raise RuntimeError(f"incomplete detection-status staging: {manifest_path}")
        score_matrix.flush()
        statuses.flush()

        pr_rows: list[dict] = []
        score_rows: list[dict] = []
        histogram_rows: list[dict] = []
        reconciliations = 0
        # Sort the complete image-major/query-index score stream exactly once
        # per checkpoint state.  For a stable sort, filtering a sorted stream
        # is equivalent to stably sorting the filtered subsequence: unequal
        # scores keep the same total ordering, while equal scores keep their
        # original image-major/query-index order in both constructions.
        # Therefore each status slice can select TP/FP along this shared order
        # without performing another O(N log N) sort.
        scores_flat_all = np.asarray(score_matrix).reshape(-1)
        global_score_order = np.argsort(-scores_flat_all, kind="stable")
        ordered_scores_all = scores_flat_all[global_score_order]
        for scope, scope_index in scope_indexes.items():
            denominator = int(positive_counts[scope_index])
            if denominator <= 0:
                raise ValueError(f"scope {scope} has no non-ignored GT: {match_dir}")
            for threshold in thresholds:
                threshold_index = threshold_indexes[float(threshold)]
                for budget in budgets:
                    budget_index = budget_indexes[int(budget)]
                    status_flat = np.asarray(
                        statuses[budget_index, scope_index, threshold_index]
                    ).reshape(-1)
                    ordered_status = status_flat[global_score_order]
                    evaluated_ordered = (
                        (ordered_status == tp_code) | (ordered_status == fp_code)
                    )
                    ordered_scores = ordered_scores_all[evaluated_ordered]
                    ordered_tp = ordered_status[evaluated_ordered] == tp_code
                    cumulative_tp = np.cumsum(ordered_tp, dtype=np.int64)
                    cumulative_fp = np.cumsum(~ordered_tp, dtype=np.int64)
                    recall = cumulative_tp.astype(np.float64) / denominator
                    precision = cumulative_tp.astype(np.float64) / (
                        cumulative_tp + cumulative_fp + np.spacing(1)
                    )
                    envelope = np.maximum.accumulate(precision[::-1])[::-1]
                    positions = np.searchsorted(recall, recall_grid, side="left")
                    available = positions < len(envelope)
                    sampled_precision = np.zeros_like(recall_grid)
                    sampled_score = np.full_like(recall_grid, np.nan)
                    sampled_precision[available] = envelope[positions[available]]
                    sampled_score[available] = ordered_scores[positions[available]]
                    ap = float(sampled_precision.mean())
                    ar = float(recall[-1]) if recall.size else 0.0
                    tp_scores = ordered_scores[ordered_tp]
                    fp_scores = ordered_scores[~ordered_tp]
                    expected_row = expected_per_iou.get(
                        (record.epoch, record.state, int(budget), round(float(threshold), 2))
                    )
                    summary_ap = summary_ar = ap_delta = ar_delta = None
                    if scope == "coco_small_eval":
                        if expected_row is None:
                            raise ValueError(
                                f"missing summary per-IoU small metric e{record.epoch}/"
                                f"{record.state}/d{budget}/IoU{threshold:g}"
                            )
                        summary_ap = float(expected_row["ap_small"])
                        summary_ar = float(expected_row["ar_small"])
                        ap_delta = ap - summary_ap
                        ar_delta = ar - summary_ar
                        if abs(ap_delta) > 1e-8 or abs(ar_delta) > 1e-8:
                            raise AssertionError(
                                f"detection-status PR disagrees with evaluator summary e{record.epoch}/"
                                f"{record.state}/d{budget}/IoU{threshold:g}: "
                                f"AP delta={ap_delta}, AR delta={ar_delta}"
                            )
                        reconciliations += 1
                    score_rows.append(
                        {
                            "epoch": record.epoch,
                            "state": record.state,
                            "source_state": source_state,
                            "scope": scope,
                            "iou_threshold": threshold,
                            "max_dets": budget,
                            "nonignored_gt": denominator,
                            "ignored_detections": int(np.count_nonzero(status_flat == ignored_code)),
                            "truncated_detections": int(np.count_nonzero(status_flat == truncated_code)),
                            **_score_fields("tp", tp_scores),
                            **_score_fields("fp", fp_scores),
                            "tp_minus_fp_median": (
                                _quantile(tp_scores, 0.5) - _quantile(fp_scores, 0.5)
                                if tp_scores.size and fp_scores.size
                                else None
                            ),
                            "fp_fraction_above_tp_median": (
                                float(np.mean(fp_scores >= _quantile(tp_scores, 0.5)))
                                if tp_scores.size and fp_scores.size
                                else None
                            ),
                            "ap_from_global_pr": ap,
                            "ar_from_global_pr": ar,
                            "summary_ap": summary_ap,
                            "summary_ar": summary_ar,
                            "ap_reconciliation_delta": ap_delta,
                            "ar_reconciliation_delta": ar_delta,
                            "note": (
                                "global stable score sort from persisted COCO one-to-one detection states"
                            ),
                        }
                    )
                    for recall_threshold, precision_value, score_value in zip(
                        recall_grid, sampled_precision, sampled_score
                    ):
                        pr_rows.append(
                            {
                                "epoch": record.epoch,
                                "state": record.state,
                                "scope": scope,
                                "iou_threshold": threshold,
                                "max_dets": budget,
                                "recall_threshold": recall_threshold,
                                "interpolated_precision": precision_value,
                                "minimum_score_at_recall": (
                                    float(score_value) if np.isfinite(score_value) else None
                                ),
                            }
                        )
                    tp_hist, _ = np.histogram(tp_scores, bins=score_bins)
                    fp_hist, _ = np.histogram(fp_scores, bins=score_bins)
                    for bin_index, (tp_count, fp_count) in enumerate(zip(tp_hist, fp_hist)):
                        histogram_rows.append(
                            {
                                "epoch": record.epoch,
                                "state": record.state,
                                "scope": scope,
                                "iou_threshold": threshold,
                                "max_dets": budget,
                                "score_bin_left": score_bins[bin_index],
                                "score_bin_right": score_bins[bin_index + 1],
                                "tp": int(tp_count),
                                "fp": int(fp_count),
                            }
                        )

        del (
            scores_flat_all,
            global_score_order,
            ordered_scores_all,
            status_flat,
            ordered_status,
            evaluated_ordered,
            score_matrix,
            statuses,
        )
        gc.collect()
        return pr_rows, score_rows, histogram_rows, reconciliations


def compute_global_score_pr_from_evaluator(
    record: ReplayState,
    match_dir: Path,
    thresholds: Sequence[float],
    budgets: Sequence[int],
    expected_per_iou: Mapping[tuple[int, str, int, float], Mapping],
    dataset_cache: dict[Path, dict],
    scratch_root: Path,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Reconstruct exact global score-ranked PR from evaluator match states.

    COCO first matches detections independently inside each image, then sorts
    all non-ignored detections globally by score.  The second step is why
    per-image Top-K TP/FN totals alone cannot explain AP.
    """

    from tools.table_cells.coco_match_npz import (  # imported lazily for table-only use
        SCOPES as MATCH_SCOPES,
        _annotation_index,
        _build_shard_evaluation,
        _load_npz_contract,
    )

    protocol = read_json(match_dir / "protocol.json")
    annotation_path = Path(protocol["annotation"]).resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    prediction_path = record.prediction_path
    if not prediction_path.is_file():
        protocol_prediction = Path(protocol["prediction"]).resolve()
        if not protocol_prediction.is_file():
            raise FileNotFoundError(prediction_path)
        prediction_path = protocol_prediction
    dataset = dataset_cache.get(annotation_path)
    if dataset is None:
        dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
        dataset_cache[annotation_path] = dataset

    archive, image_ids, boxes, scores, category_id, _input_size, source_state = _load_npz_contract(
        prediction_path
    )
    try:
        images, annotations_by_image, _image_density = _annotation_index(dataset, category_id)
        if any(int(image_id) not in images for image_id in image_ids):
            raise ValueError(f"prediction image id absent from annotation: {prediction_path}")
        if boxes.shape[1] < max(budgets):
            raise ValueError(f"prediction query count is below requested maxDets: {prediction_path}")
        match_scope_names = tuple(name for name, _ in MATCH_SCOPES)
        image_count = len(image_ids)
        max_budget = max(budgets)
        images_per_shard = int(protocol.get("images_per_shard", 128))
        if images_per_shard < 1:
            raise ValueError(f"invalid images_per_shard in {match_dir / 'protocol.json'}")

        # The memmaps cap resident memory while preserving the exact stable
        # image-major ordering used by COCOeval.accumulate().
        with tempfile.TemporaryDirectory(prefix="global_pr_", dir=scratch_root) as temporary:
            temporary_path = Path(temporary)
            score_matrix = np.lib.format.open_memmap(
                temporary_path / "scores.npy",
                mode="w+",
                dtype=np.float32,
                shape=(image_count, max_budget),
            )
            statuses = np.lib.format.open_memmap(
                temporary_path / "statuses.npy",
                mode="w+",
                dtype=np.uint8,
                shape=(len(MATCH_SCOPES), len(thresholds), image_count, max_budget),
            )
            statuses[:] = MISSING_STATUS
            for start in range(0, image_count, images_per_shard):
                stop = min(start + images_per_shard, image_count)
                shard_ids = image_ids[start:stop]
                shard_boxes = np.asarray(boxes[start:stop])
                shard_scores = np.asarray(scores[start:stop])
                states = _build_shard_evaluation(
                    dataset,
                    images,
                    annotations_by_image,
                    shard_ids,
                    shard_boxes,
                    shard_scores,
                    category_id,
                    thresholds,
                    max_budget,
                )
                for scope_index in range(len(MATCH_SCOPES)):
                    for local_image_index in range(stop - start):
                        state_tuple = states[scope_index * (stop - start) + local_image_index]
                        dt_matches_flat, _gt_matches_flat, dt_scores, _gt_ignore, dt_ignore_flat, _triples = state_tuple
                        detection_count = len(dt_scores)
                        if detection_count != max_budget:
                            raise RuntimeError(
                                f"evaluator returned {detection_count} detections, expected {max_budget}"
                            )
                        if scope_index == 0:
                            score_matrix[start + local_image_index] = np.asarray(dt_scores, dtype=np.float32)
                        matches_flat = np.asarray(dt_matches_flat, dtype=np.int64)
                        ignore_flat = np.asarray(dt_ignore_flat, dtype=bool)
                        expected = len(thresholds) * detection_count
                        if matches_flat.size != expected or ignore_flat.size != expected:
                            raise RuntimeError("malformed per-detection evaluator state")
                        matches_matrix = matches_flat.reshape(len(thresholds), detection_count)
                        ignore_matrix = ignore_flat.reshape(len(thresholds), detection_count)
                        encoded = np.where(
                            ignore_matrix,
                            0,
                            np.where(matches_matrix > 0, 1, 2),
                        ).astype(np.uint8)
                        statuses[scope_index, :, start + local_image_index, :] = encoded
            score_matrix.flush()
            statuses.flush()
            if np.any(statuses == MISSING_STATUS):
                raise RuntimeError(f"global PR status staging is incomplete: e{record.epoch}/{record.state}")

            selected_image_ids = set(map(int, image_ids))
            gt_denominators = {}
            for scope_name, area_range in MATCH_SCOPES:
                count = 0
                for image_id in selected_image_ids:
                    for ann in annotations_by_image.get(image_id, ()):
                        area = float(ann.get("area", ann["bbox"][2] * ann["bbox"][3]))
                        if (
                            not ann.get("iscrowd", 0)
                            and area_range[0] <= area <= area_range[1]
                        ):
                            count += 1
                if count <= 0:
                    raise ValueError(f"scope {scope_name} has no non-ignored GT")
                gt_denominators[scope_name] = count

            pr_rows: list[dict] = []
            score_rows: list[dict] = []
            histogram_rows: list[dict] = []
            reconciliations = 0
            recall_grid = np.linspace(0.0, 1.0, 101)
            score_bins = np.linspace(0.0, 1.0, 101)
            for scope_index, (scope_name, _area_range) in enumerate(MATCH_SCOPES):
                for threshold_index, threshold in enumerate(thresholds):
                    for budget in budgets:
                        scores_flat = np.asarray(score_matrix[:, :budget]).reshape(-1)
                        status_flat = np.asarray(
                            statuses[scope_index, threshold_index, :, :budget]
                        ).reshape(-1)
                        valid = status_flat != 0
                        valid_scores = scores_flat[valid]
                        valid_status = status_flat[valid]
                        order = np.argsort(-valid_scores, kind="mergesort")
                        ordered_scores = valid_scores[order]
                        ordered_tp = valid_status[order] == 1
                        ordered_fp = valid_status[order] == 2
                        cumulative_tp = np.cumsum(ordered_tp, dtype=np.int64)
                        cumulative_fp = np.cumsum(ordered_fp, dtype=np.int64)
                        recall = cumulative_tp / gt_denominators[scope_name]
                        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
                        envelope = np.maximum.accumulate(precision[::-1])[::-1]
                        positions = np.searchsorted(recall, recall_grid, side="left")
                        sampled_precision = np.zeros_like(recall_grid)
                        sampled_score = np.full_like(recall_grid, np.nan)
                        available = positions < len(envelope)
                        sampled_precision[available] = envelope[positions[available]]
                        sampled_score[available] = ordered_scores[positions[available]]
                        ap = float(sampled_precision.mean())
                        ar = float(recall[-1]) if recall.size else 0.0
                        tp_scores = valid_scores[valid_status == 1]
                        fp_scores = valid_scores[valid_status == 2]
                        expected_row = expected_per_iou.get(
                            (record.epoch, record.state, budget, round(float(threshold), 2))
                        )
                        summary_ap = summary_ar = ap_delta = ar_delta = None
                        if scope_name == "coco_small_eval":
                            if expected_row is None:
                                raise ValueError(
                                    f"missing summary per-IoU small metric e{record.epoch}/"
                                    f"{record.state}/d{budget}/IoU{threshold:g}"
                                )
                            summary_ap = float(expected_row["ap_small"])
                            summary_ar = float(expected_row["ar_small"])
                            ap_delta = ap - summary_ap
                            ar_delta = ar - summary_ar
                            if abs(ap_delta) > 1e-8 or abs(ar_delta) > 1e-8:
                                raise AssertionError(
                                    f"global PR disagrees with evaluator summary e{record.epoch}/"
                                    f"{record.state}/d{budget}/IoU{threshold:g}: "
                                    f"AP delta={ap_delta}, AR delta={ar_delta}"
                                )
                            reconciliations += 1
                        score_rows.append(
                            {
                                "epoch": record.epoch,
                                "state": record.state,
                                "source_state": source_state,
                                "scope": scope_name,
                                "iou_threshold": threshold,
                                "max_dets": budget,
                                "nonignored_gt": gt_denominators[scope_name],
                                "ignored_detections": int(np.count_nonzero(status_flat == 0)),
                                "truncated_detections": image_count * (max_budget - budget),
                                **_score_fields("tp", tp_scores),
                                **_score_fields("fp", fp_scores),
                                "tp_minus_fp_median": (
                                    _quantile(tp_scores, 0.5) - _quantile(fp_scores, 0.5)
                                    if tp_scores.size and fp_scores.size
                                    else None
                                ),
                                "fp_fraction_above_tp_median": (
                                    float(np.mean(fp_scores >= _quantile(tp_scores, 0.5)))
                                    if tp_scores.size and fp_scores.size
                                    else None
                                ),
                                "ap_from_global_pr": ap,
                                "ar_from_global_pr": ar,
                                "summary_ap": summary_ap,
                                "summary_ar": summary_ar,
                                "ap_reconciliation_delta": ap_delta,
                                "ar_reconciliation_delta": ar_delta,
                                "note": (
                                    "global score sort after per-image COCO matching; ignored detections excluded"
                                ),
                            }
                        )
                        for recall_threshold, precision_value, score_value in zip(
                            recall_grid, sampled_precision, sampled_score
                        ):
                            pr_rows.append(
                                {
                                    "epoch": record.epoch,
                                    "state": record.state,
                                    "scope": scope_name,
                                    "iou_threshold": threshold,
                                    "max_dets": budget,
                                    "recall_threshold": recall_threshold,
                                    "interpolated_precision": precision_value,
                                    "minimum_score_at_recall": (
                                        float(score_value) if np.isfinite(score_value) else None
                                    ),
                                }
                            )
                        tp_hist, _ = np.histogram(tp_scores, bins=score_bins)
                        fp_hist, _ = np.histogram(fp_scores, bins=score_bins)
                        for bin_index, (tp_count, fp_count) in enumerate(zip(tp_hist, fp_hist)):
                            histogram_rows.append(
                                {
                                    "epoch": record.epoch,
                                    "state": record.state,
                                    "scope": scope_name,
                                    "iou_threshold": threshold,
                                    "max_dets": budget,
                                    "score_bin_left": score_bins[bin_index],
                                    "score_bin_right": score_bins[bin_index + 1],
                                    "tp": int(tp_count),
                                    "fp": int(fp_count),
                                }
                            )
            # Windows keeps an mmap file locked while any ndarray view still
            # references it.  Drop loop-local views before TemporaryDirectory
            # removes the staging files.
            del (
                scores_flat,
                status_flat,
                valid_scores,
                valid_status,
                ordered_scores,
                ordered_tp,
                ordered_fp,
            )
            del score_matrix, statuses
            gc.collect()
            return pr_rows, score_rows, histogram_rows, reconciliations
    finally:
        archive.close()


def compute_global_score_pr(
    record: ReplayState,
    match_dir: Path,
    thresholds: Sequence[float],
    budgets: Sequence[int],
    expected_per_iou: Mapping[tuple[int, str, int, float], Mapping],
    dataset_cache: dict[Path, dict],
    scratch_root: Path,
) -> tuple[list[dict], list[dict], list[dict], int]:
    """Prefer persisted matcher states; retain evaluator replay for v1 artifacts."""

    if (match_dir / "detection_status_manifest.json").is_file():
        return compute_global_score_pr_from_detection_status(
            record,
            match_dir,
            thresholds,
            budgets,
            expected_per_iou,
            scratch_root,
        )
    return compute_global_score_pr_from_evaluator(
        record,
        match_dir,
        thresholds,
        budgets,
        expected_per_iou,
        dataset_cache,
        scratch_root,
    )


def match_bucket_rows(
    epoch: int,
    state: str,
    data: MatchData,
    meta: Metadata,
    scopes: Sequence[str],
    thresholds: Sequence[float],
    budgets: Sequence[int],
) -> list[dict]:
    rows = []
    for scope in scopes:
        for threshold in thresholds:
            for budget in budgets:
                values = data.statuses[(scope, threshold, budget)]
                for axis, group, mask in meta.group_masks():
                    tp = int(np.count_nonzero(mask & (values == STATUS_CODE["tp"])))
                    fn = int(np.count_nonzero(mask & (values == STATUS_CODE["fn"])))
                    ignored = int(np.count_nonzero(mask & (values == STATUS_CODE["ignored"])))
                    rows.append(
                        {
                            "epoch": epoch,
                            "state": state,
                            "scope": scope,
                            "iou_threshold": threshold,
                            "max_dets": budget,
                            "axis": axis,
                            "group": group,
                            "objects": int(mask.sum()),
                            "evaluable": tp + fn,
                            "tp": tp,
                            "fn": fn,
                            "ignored": ignored,
                            "recall": tp / (tp + fn) if tp + fn else None,
                            "note": "formal one-to-one GT TP/FN; FP is intentionally not assigned to GT buckets",
                        }
                    )
    return rows


def load_stage_endpoint(path: Path, threshold: float, gt_count: int) -> dict[str, dict[str, np.ndarray]]:
    suffix = stage_threshold_suffix(threshold)
    result = {"best_iou": {}, "rank": {}, "score": {}}
    with np.load(path, allow_pickle=False) as archive:
        for stage in STAGES:
            best_key = f"{stage}__best_iou"
            rank_key = f"{stage}__first_rank_iou{suffix}"
            score_key = f"{stage}__first_score_iou{suffix}"
            if best_key not in archive.files or archive[best_key].shape != (gt_count,):
                raise ValueError(f"invalid or missing {best_key}: {path}")
            result["best_iou"][stage] = np.asarray(archive[best_key], dtype=np.float32)
            if rank_key in archive.files:
                if archive[rank_key].shape != (gt_count,):
                    raise ValueError(f"invalid {rank_key}: {path}")
                result["rank"][stage] = np.asarray(archive[rank_key], dtype=np.int32)
            else:
                result["rank"][stage] = np.full(gt_count, -1, dtype=np.int32)
            if score_key in archive.files:
                if archive[score_key].shape != (gt_count,):
                    raise ValueError(f"invalid {score_key}: {path}")
                result["score"][stage] = np.asarray(archive[score_key], dtype=np.float32)
            else:
                result["score"][stage] = np.full(gt_count, np.nan, dtype=np.float32)
    return result


def classify_missing(endpoint: Mapping[str, Mapping[str, np.ndarray]], threshold: float) -> np.ndarray:
    """Assign the deepest surviving stage for each formally unmatched GT."""
    count = len(endpoint["best_iou"][STAGES[0]])
    result = np.zeros(count, dtype=np.uint8)
    result[endpoint["best_iou"]["encoder_dense_all"] >= threshold] = 1
    result[endpoint["best_iou"]["encoder_top640"] >= threshold] = 2
    result[endpoint["best_iou"]["decoder_all640"] >= threshold] = 3
    result[endpoint["best_iou"]["decoder_top300"] >= threshold] = 4
    return result


DETAIL_FIELDS = (
    "epoch_start",
    "epoch_end",
    "state",
    "match_scope",
    "iou_threshold",
    "transition",
    "cause",
    "annotation_id",
    "image_id",
    "file_name",
    "domain",
    "image_gt_count",
    "density_bucket",
    "coco_small",
    "local_hybrid_small",
    "small_partition",
    "coco_area_group",
    "input_area_group",
    "input_short_bucket",
    "input_short_px",
    "status_before",
    "status_after",
    "formal_query_before",
    "formal_query_after",
    "formal_rank_before",
    "formal_rank_after",
    "formal_score_before",
    "formal_score_after",
    "formal_iou_before",
    "formal_iou_after",
    "dense_iou_before",
    "dense_iou_after",
    "top640_iou_before",
    "top640_iou_after",
    "decoder_all_iou_before",
    "decoder_all_iou_after",
    "decoder_top300_iou_before",
    "decoder_top300_iou_after",
    "dense_rank_before",
    "dense_rank_after",
    "top640_rank_before",
    "top640_rank_after",
    "decoder_all_rank_before",
    "decoder_all_rank_after",
    "decoder_top300_rank_before",
    "decoder_top300_rank_after",
    "evidence_strength",
    "rank_distance_from_300",
)


REPRESENTATIVE_DETAIL_FIELDS = (
    "coco_small",
    "local_hybrid_small",
    "coco_area_group",
    "input_area_group",
    "input_short_bucket",
    "input_short_px",
    "status_before",
    "status_after",
    "formal_query_before",
    "formal_query_after",
    "formal_rank_before",
    "formal_rank_after",
    "formal_score_before",
    "formal_score_after",
    "formal_iou_before",
    "formal_iou_after",
    "dense_iou_before",
    "dense_iou_after",
    "top640_iou_before",
    "top640_iou_after",
    "decoder_all_iou_before",
    "decoder_all_iou_after",
    "decoder_top300_iou_before",
    "decoder_top300_iou_after",
    "dense_rank_before",
    "dense_rank_after",
    "top640_rank_before",
    "top640_rank_after",
    "decoder_all_rank_before",
    "decoder_all_rank_after",
    "decoder_top300_rank_before",
    "decoder_top300_rank_after",
)


REPRESENTATIVE_FIELDS = (
    "selection_rank",
    "state",
    "epoch_start",
    "epoch_end",
    "match_scope",
    "iou_threshold",
    "transition",
    "cause",
    "small_partition",
    "domain",
    "annotation_id",
    "image_id",
    "file_name",
    "image_gt_count",
    "evidence_strength",
    "rank_distance_from_300",
    "selection_rule",
    *REPRESENTATIVE_DETAIL_FIELDS,
)


PARTITION_LABELS = ("coco_and_hybrid", "coco_only", "hybrid_only", "neither")
DENSITY_LABELS = ("0", "1to32", "33to128", "129to300", "ge301")
SHORT_LABELS = ("lt8", "8to16", "16to32", "32to64", "ge64")
AREA_LABELS = ("small", "medium", "large")


def transition_views(meta: Metadata, scope: str) -> list[tuple[str, np.ndarray]]:
    views = [("all", np.ones(len(meta.annotation_ids), dtype=bool))]
    if scope == "all":
        views.extend(
            [
                ("coco_small", meta.coco_small),
                ("local_hybrid_small", meta.hybrid_small),
                ("coco_and_hybrid", meta.membership == 0),
                ("coco_only", meta.membership == 1),
                ("hybrid_only", meta.membership == 2),
                ("neither", meta.membership == 3),
            ]
        )
    elif scope == "coco_small_eval":
        # COCOeval uses the inclusive [0, 32^2] areaRange.  This formal
        # evaluator scope must not be intersected with the descriptive local
        # ``area < 32^2`` cohort, which would drop the boundary annotations.
        views = [("coco_small_eval", meta.coco_small_eval)]
    else:
        raise ValueError(f"unsupported formal match scope: {scope}")
    return views


def build_detail_row(
    index: int,
    start: int,
    end: int,
    state: str,
    scope: str,
    threshold: float,
    direction: str,
    cause: str,
    before_match: MatchData,
    after_match: MatchData,
    before_stage: Mapping,
    after_stage: Mapping,
    meta: Metadata,
    budget: int,
) -> dict:
    image_index = int(meta.gt_image_index[index])
    missing_stage = before_stage if direction == "gain" else after_stage
    cause_stage = {
        "candidate_absence": "encoder_dense_all",
        "encoder_selection": "encoder_dense_all",
        "decoder_geometry": "encoder_top640",
        "rank_gt300": "decoder_all640",
        "one_to_one_competition": "decoder_top300",
    }[cause]
    if cause == "candidate_absence":
        evidence = max(0.0, threshold - float(missing_stage["best_iou"][cause_stage][index]))
    else:
        evidence = max(0.0, float(missing_stage["best_iou"][cause_stage][index]) - threshold)
    missing_rank = int(missing_stage["rank"]["decoder_all640"][index])
    status_before = int(before_match.statuses[(scope, threshold, budget)][index])
    status_after = int(after_match.statuses[(scope, threshold, budget)][index])
    row = {
        "epoch_start": start,
        "epoch_end": end,
        "state": state,
        "match_scope": scope,
        "iou_threshold": threshold,
        "transition": direction,
        "cause": cause,
        "annotation_id": int(meta.annotation_ids[index]),
        "image_id": int(meta.image_ids[image_index]),
        "file_name": str(meta.image_file_name[image_index]),
        "domain": str(meta.image_domain[image_index]),
        "image_gt_count": int(meta.image_gt_count[image_index]),
        "density_bucket": DENSITY_LABELS[int(meta.gt_density_bucket[index])],
        "coco_small": int(meta.coco_small[index]),
        "local_hybrid_small": int(meta.hybrid_small[index]),
        "small_partition": PARTITION_LABELS[int(meta.membership[index])],
        "coco_area_group": AREA_LABELS[int(meta.coco_area_group[index])],
        "input_area_group": AREA_LABELS[int(meta.input_area_group[index])],
        "input_short_bucket": SHORT_LABELS[int(meta.input_short_bucket[index])],
        "input_short_px": float(meta.input_short_px[index]),
        "status_before": STATUS_NAME[status_before],
        "status_after": STATUS_NAME[status_after],
        "evidence_strength": evidence,
        "rank_distance_from_300": abs(missing_rank - 300) if missing_rank > 0 else None,
    }
    for name, source in (("before", before_match), ("after", after_match)):
        for field in ("query_index", "rank", "score", "iou"):
            value = source.details[(scope, threshold, field)][index]
            row[f"formal_{'query' if field == 'query_index' else field}_{name}"] = (
                None if (field in ("query_index", "rank") and int(value) < 0) or (field in ("score", "iou") and np.isnan(value)) else value.item()
            )
    aliases = {
        "encoder_dense_all": "dense",
        "encoder_top640": "top640",
        "decoder_all640": "decoder_all",
        "decoder_top300": "decoder_top300",
    }
    for stage, alias in aliases.items():
        row[f"{alias}_iou_before"] = float(before_stage["best_iou"][stage][index])
        row[f"{alias}_iou_after"] = float(after_stage["best_iou"][stage][index])
        row[f"{alias}_rank_before"] = int(before_stage["rank"][stage][index])
        row[f"{alias}_rank_after"] = int(after_stage["rank"][stage][index])
    return row


def retain_representative_candidate(
    bucket: dict[int, tuple[tuple, dict]],
    image_id: int,
    sort_key: tuple,
    row: dict,
    limit: int,
) -> None:
    """Keep only the deterministic best candidate per image and top-K images."""
    previous = bucket.get(image_id)
    if previous is None or sort_key < previous[0]:
        bucket[image_id] = (sort_key, row)
    if len(bucket) > limit:
        worst_image = max(
            bucket,
            key=lambda candidate_image: (bucket[candidate_image][0], candidate_image),
        )
        del bucket[worst_image]


def analyze_transitions(
    output_path: Path,
    meta: Metadata,
    records: Mapping[tuple[int, str], ReplayState],
    matches: Mapping[tuple[int, str], MatchData],
    pairs: Sequence[tuple[int, int]],
    states: Sequence[str],
    scopes: Sequence[str],
    thresholds: Sequence[float],
    budget: int,
    representatives_per_stratum: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], int]:
    summaries: list[dict] = []
    attributions: list[dict] = []
    representative_pool: dict[tuple, dict[int, tuple[tuple, dict]]] = defaultdict(dict)
    detail_count = 0
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with (
        temporary.open("wb") as raw_handle,
        gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_handle, compresslevel=6, mtime=0
        ) as compressed_handle,
        io.TextIOWrapper(compressed_handle, encoding="utf-8", newline="") as detail_handle,
    ):
        detail_writer = csv.DictWriter(detail_handle, fieldnames=DETAIL_FIELDS, extrasaction="raise", lineterminator="\n")
        detail_writer.writeheader()
        for state in states:
            for start, end in pairs:
                before_match = matches[(start, state)]
                after_match = matches[(end, state)]
                for threshold in thresholds:
                    before_stage = load_stage_endpoint(records[(start, state)].stage_path, threshold, len(meta.annotation_ids))
                    after_stage = load_stage_endpoint(records[(end, state)].stage_path, threshold, len(meta.annotation_ids))
                    before_causes = classify_missing(before_stage, threshold)
                    after_causes = classify_missing(after_stage, threshold)
                    for scope in scopes:
                        before_status = before_match.statuses[(scope, threshold, budget)]
                        after_status = after_match.statuses[(scope, threshold, budget)]
                        evaluable = np.isin(before_status, (1, 2)) & np.isin(after_status, (1, 2))
                        gain = evaluable & (before_status == 2) & (after_status == 1)
                        loss = evaluable & (before_status == 1) & (after_status == 2)
                        stable_fn = evaluable & (before_status == 2) & (after_status == 2)
                        stable_tp = evaluable & (before_status == 1) & (after_status == 1)
                        for cohort, cohort_mask in transition_views(meta, scope):
                            mask = evaluable & cohort_mask
                            gain_mask = gain & cohort_mask
                            loss_mask = loss & cohort_mask
                            summaries.append(
                                {
                                    "epoch_start": start,
                                    "epoch_end": end,
                                    "state": state,
                                    "match_scope": scope,
                                    "cohort": cohort,
                                    "iou_threshold": threshold,
                                    "max_dets": budget,
                                    "evaluable": int(mask.sum()),
                                    "stable_fn": int((stable_fn & cohort_mask).sum()),
                                    "gain": int(gain_mask.sum()),
                                    "loss": int(loss_mask.sum()),
                                    "stable_tp": int((stable_tp & cohort_mask).sum()),
                                    "net_tp": int(gain_mask.sum()) - int(loss_mask.sum()),
                                }
                            )
                            for direction, selected, cause_values in (
                                ("gain", gain_mask, before_causes),
                                ("loss", loss_mask, after_causes),
                            ):
                                total = int(selected.sum())
                                for cause_code, cause in enumerate(CAUSES):
                                    count = int(np.count_nonzero(selected & (cause_values == cause_code)))
                                    attributions.append(
                                        {
                                            "epoch_start": start,
                                            "epoch_end": end,
                                            "state": state,
                                            "match_scope": scope,
                                            "cohort": cohort,
                                            "iou_threshold": threshold,
                                            "max_dets": budget,
                                            "transition": direction,
                                            "cause": cause,
                                            "count": count,
                                            "transition_total": total,
                                            "fraction": count / total if total else None,
                                        }
                                    )

                        for direction, selected, cause_values in (
                            ("gain", gain, before_causes),
                            ("loss", loss, after_causes),
                        ):
                            for index in np.flatnonzero(selected):
                                cause = CAUSES[int(cause_values[index])]
                                row = build_detail_row(
                                    int(index), start, end, state, scope, threshold, direction, cause,
                                    before_match, after_match, before_stage, after_stage, meta, budget,
                                )
                                detail_writer.writerow({key: format_value(row.get(key)) for key in DETAIL_FIELDS})
                                detail_count += 1
                                stratum = (
                                    state, start, end, scope, threshold, direction, cause,
                                    row["small_partition"], row["domain"],
                                )
                                image_id = int(row["image_id"])
                                sort_key = (
                                    -float(row["evidence_strength"]),
                                    int(row["rank_distance_from_300"])
                                    if row["rank_distance_from_300"] is not None
                                    else 10**9,
                                    int(row["annotation_id"]),
                                )
                                bucket = representative_pool[stratum]
                                retain_representative_candidate(
                                    bucket,
                                    image_id,
                                    sort_key,
                                    row,
                                    representatives_per_stratum,
                                )
    temporary.replace(output_path)

    representative_rows = []
    for stratum, by_image in sorted(representative_pool.items()):
        selected = sorted(by_image.values(), key=lambda item: item[0])[:representatives_per_stratum]
        for selection_rank, (_key, row) in enumerate(selected, 1):
            representative = {
                "selection_rank": selection_rank,
                "state": stratum[0],
                "epoch_start": stratum[1],
                "epoch_end": stratum[2],
                "match_scope": stratum[3],
                "iou_threshold": stratum[4],
                "transition": stratum[5],
                "cause": stratum[6],
                "small_partition": stratum[7],
                "domain": stratum[8],
                "annotation_id": row["annotation_id"],
                "image_id": row["image_id"],
                "file_name": row["file_name"],
                "image_gt_count": row["image_gt_count"],
                "evidence_strength": row["evidence_strength"],
                "rank_distance_from_300": row["rank_distance_from_300"],
                "selection_rule": (
                    "best evidence per image, then evidence desc, distance to rank 300 asc, "
                    "annotation_id asc"
                ),
            }
            representative.update(
                {field: row[field] for field in REPRESENTATIVE_DETAIL_FIELDS}
            )
            if tuple(representative) != REPRESENTATIVE_FIELDS:
                raise RuntimeError("representative annotation schema changed unexpectedly")
            representative_rows.append(representative)
    image_groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in representative_rows:
        image_groups[(int(row["image_id"]), str(row["file_name"]))].append(row)
    representative_images = []
    for (image_id, file_name), rows in sorted(image_groups.items()):
        representative_images.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "domain": rows[0]["domain"],
                "image_gt_count": rows[0]["image_gt_count"],
                "selected_strata": len(rows),
                "annotation_ids": ";".join(map(str, sorted({int(row["annotation_id"]) for row in rows}))),
                "causes": ";".join(sorted({str(row["cause"]) for row in rows})),
                "transitions": ";".join(sorted({f"e{row['epoch_start']}-e{row['epoch_end']}:{row['transition']}" for row in rows})),
            }
        )
    return summaries, attributions, representative_rows, representative_images, detail_count


def write_generated_report(
    path: Path,
    metrics: Sequence[Mapping],
    per_iou_gaps: Sequence[Mapping],
    transition_summaries: Sequence[Mapping],
    attributions: Sequence[Mapping],
    score_separation: Sequence[Mapping],
    *,
    formal: bool,
) -> None:
    def report_float(value, spec: str = ".6f") -> str:
        if value is None:
            return "NA"
        number = float(value)
        return format(number, spec) if math.isfinite(number) else "NA"

    lines = [
        "# Full-val checkpoint replay: generated factual summary",
        "",
        (
            "This file is generated from the complete formal replay/matching matrix. Independent"
            if formal
            else "TESTING ONLY: this file comes from an incomplete synthetic/subset matrix. Independent"
        ),
        "coverage is not called TP/FN. Formal TP/FN counts come only from COCO one-to-one shards.",
        "",
        "## COCO metrics",
        "",
        "| epoch | state | AP@300 | AP-small@300 | AR-small@300 | AR-small@640 |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    for row in sorted(metrics, key=lambda item: (int(item["epoch"]), str(item["state"]))):
        lines.append(
            f"| {row['epoch']} | {row['state']} | {float(row['ap_d300']):.6f} | "
            f"{float(row['ap_small_d300']):.6f} | {float(row['ar_small_d300']):.6f} | "
            f"{float(row['ar_small_d640']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-IoU COCO-small maxDets gap",
            "",
            "Each row is a canonical evaluator slice. AP-small is their mean, not the sum of",
            "the TP/FN transition counts below.",
            "",
            "| epoch | state | IoU | AP@300 | AP@640 | AP gap | AR@300 | AR@640 | AR gap |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        per_iou_gaps,
        key=lambda item: (int(item["epoch"]), str(item["state"]), float(item["iou_threshold"])),
    ):
        lines.append(
            f"| {row['epoch']} | {row['state']} | {float(row['iou_threshold']):.2f} | "
            f"{float(row['ap_small_d300']):.6f} | {float(row['ap_small_d640']):.6f} | "
            f"{float(row['ap_d640_minus_d300']):+.6f} | {float(row['ar_small_d300']):.6f} | "
            f"{float(row['ar_small_d640']):.6f} | {float(row['ar_d640_minus_d300']):+.6f} |"
        )

    if score_separation:
        lines.extend(
            [
                "",
                "## Global score ordering slice",
                "",
                "This table uses formal small-area matching at IoU=0.75 and maxDets=300, then",
                "globally sorts non-ignored detections by score exactly as COCO accumulation does.",
                "",
                "| epoch | state | TP | FP | TP median score | FP median score | gap | AP slice |",
                "|---:|:---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected_scores = [
            row
            for row in score_separation
            if row["scope"] == "coco_small_eval"
            and float(row["iou_threshold"]) == 0.75
            and int(row["max_dets"]) == 300
        ]
        for row in sorted(selected_scores, key=lambda item: (int(item["epoch"]), str(item["state"]))):
            lines.append(
                f"| {row['epoch']} | {row['state']} | {row['tp_count']} | {row['fp_count']} | "
                f"{report_float(row['tp_score_median'])} | {report_float(row['fp_score_median'])} | "
                f"{report_float(row['tp_minus_fp_median'], '+.6f')} | "
                f"{report_float(row['ap_from_global_pr'])} |"
            )

    lines.extend(["", "## Paired one-to-one transition attribution", ""])
    selected = [
        row for row in transition_summaries
        if row["cohort"] in {"coco_small", "coco_small_eval", "local_hybrid_small"}
        and float(row["iou_threshold"]) == 0.75
    ]
    attr_index = defaultdict(dict)
    for row in attributions:
        key = (
            row["epoch_start"], row["epoch_end"], row["state"], row["match_scope"],
            row["cohort"], row["iou_threshold"], row["transition"],
        )
        attr_index[key][row["cause"]] = int(row["count"])
    lines.extend(
        [
            "| pair | state | scope | cohort | gain | loss | candidate | encoder | decoder | rank | competition |",
            "|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        base = (
            row["epoch_start"], row["epoch_end"], row["state"], row["match_scope"],
            row["cohort"], row["iou_threshold"],
        )
        combined = Counter()
        for direction in ("gain", "loss"):
            combined.update(attr_index.get(base + (direction,), {}))
        lines.append(
            f"| e{row['epoch_start']}→e{row['epoch_end']} | {row['state']} | {row['match_scope']} | "
            f"{row['cohort']} | {row['gain']} | {row['loss']} | "
            f"{combined['candidate_absence']} | {combined['encoder_selection']} | "
            f"{combined['decoder_geometry']} | {combined['rank_gt300']} | "
            f"{combined['one_to_one_competition']} |"
        )
    lines.extend(
        [
            "",
            "The table reports the direct missing endpoint, not the training-level root cause and",
            "not an additive decomposition of AP. A",
            "`rank_gt300` row means an IoU-qualified final Decoder query still exists among all 640",
            "queries but lies outside the Top-300 budget. `one_to_one_competition` means independent",
            "Top-300 coverage exists but formal greedy one-to-one matching leaves the GT unmatched.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def output_manifest(output: Path, names: Sequence[str]) -> list[dict]:
    rows = []
    for name in names:
        path = output / name
        rows.append({"file": name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return rows


def run_analysis(args: argparse.Namespace) -> dict:
    formal = not args.testing
    replay_dirs = unique_mapping(args.replay, "replay epoch")
    match_dirs = unique_mapping(args.match, "match epoch/state")
    states = tuple(args.states)
    if len(set(states)) != len(states):
        raise ValueError("states must be unique")
    thresholds = tuple(sorted(set(float(value) for value in args.thresholds)))
    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    if args.transition_budget not in budgets:
        raise ValueError("transition budget must be included in --budgets")
    pairs = tuple(args.pair or DEFAULT_PAIRS)
    epochs = set(replay_dirs)
    for start, end in pairs:
        if start not in epochs or end not in epochs:
            raise ValueError(f"transition pair e{start}->e{end} is absent from replay inputs")
    expected_match_keys = {(epoch, state) for epoch in epochs for state in states}
    if set(match_dirs) != expected_match_keys:
        missing = sorted(expected_match_keys - set(match_dirs))
        extra = sorted(set(match_dirs) - expected_match_keys)
        raise ValueError(f"match mapping must exactly cover replay states; missing={missing}, extra={extra}")

    if formal and args.skip_global_pr:
        raise ValueError("formal analysis cannot skip global PR")
    if formal:
        if tuple(sorted(replay_dirs)) != FORMAL_EPOCHS:
            raise ValueError(f"formal analysis requires epochs {FORMAL_EPOCHS}")
        if states != DEFAULT_STATES:
            raise ValueError(f"formal analysis requires states {DEFAULT_STATES}")
        if thresholds != DEFAULT_THRESHOLDS or budgets != (300, 640):
            raise ValueError("formal analysis requires ten IoUs and maxDets=300/640")
    args.output.mkdir(parents=True, exist_ok=True)
    meta, records, metrics, per_iou_metrics = load_replays(
        replay_dirs, args.split, states, formal=formal
    )
    formal_identity = validate_formal_contract(args, replay_dirs, meta, records) if formal else None
    matcher_matrix_identity = validate_match_matrix_audit(
        match_dirs, records, formal_identity, formal=formal
    )
    per_iou_gaps = per_iou_gap_rows(per_iou_metrics)
    stage_rows, reconciliations = stage_coverage_rows(meta, records, thresholds)
    raw_ema = raw_ema_rows(metrics, states)
    merged_aggregates = merge_match_aggregates(match_dirs)

    match_data: dict[tuple[int, str], MatchData] = {}
    match_buckets: list[dict] = []
    for key, directory in sorted(match_dirs.items()):
        epoch, state = key
        protocol = validate_match_protocol(
            records[key], directory, split=args.split, formal=formal
        )
        data = load_match_data(
            directory, meta, args.scopes, thresholds, budgets, formal=formal
        )
        protocol_state = str(protocol.get("state", ""))
        expected_protocol_state = f"checkpoint{epoch:04d}:{state}"
        if protocol_state != expected_protocol_state:
            raise ValueError(
                f"matcher state is not bound to requested e{epoch}/{state}: "
                f"{protocol_state!r} != {expected_protocol_state!r}"
            )
        if len(data.source_states) != 1 or data.source_states[0] != protocol_state:
            raise ValueError(
                f"formal GT match state disagrees with protocol e{epoch}/{state}: "
                f"rows={data.source_states}, protocol={protocol_state!r}"
            )
        status_manifest_path = directory / "detection_status_manifest.json"
        if status_manifest_path.is_file():
            status_state = str(read_json(status_manifest_path).get("state", ""))
            if status_state != protocol_state:
                raise ValueError(
                    f"detection-status state disagrees with protocol e{epoch}/{state}: "
                    f"status={status_state!r}, protocol={protocol_state!r}"
                )
        match_data[key] = data
        match_buckets.extend(match_bucket_rows(epoch, state, data, meta, args.scopes, thresholds, budgets))

    global_pr_rows: list[dict] = []
    score_separation_rows: list[dict] = []
    score_histogram_rows: list[dict] = []
    global_pr_reconciliations = 0
    global_pr_backends: dict[str, str] = {}
    if not args.skip_global_pr:
        expected_per_iou = {
            (
                int(row["epoch"]),
                str(row["state"]),
                int(row["max_dets"]),
                round(float(row["iou_threshold"]), 2),
            ): row
            for row in per_iou_metrics
        }
        dataset_cache: dict[Path, dict] = {}
        for key, record in sorted(records.items()):
            backend = (
                "persisted_detection_status"
                if (match_dirs[key] / "detection_status_manifest.json").is_file()
                else "evaluator_fallback"
            )
            global_pr_backends[f"e{record.epoch}/{record.state}"] = backend
            print(
                f"global PR reconstruction e{record.epoch}/{record.state} via {backend}",
                flush=True,
            )
            pr, separation, histogram, reconciled = compute_global_score_pr(
                record,
                match_dirs[key],
                thresholds,
                budgets,
                expected_per_iou,
                dataset_cache,
                args.output,
            )
            global_pr_rows.extend(pr)
            score_separation_rows.extend(separation)
            score_histogram_rows.extend(histogram)
            global_pr_reconciliations += reconciled
        if formal:
            expected_reconciliations = (
                len(FORMAL_EPOCHS) * len(DEFAULT_STATES) * len(DEFAULT_THRESHOLDS) * 2
            )
            if global_pr_reconciliations != expected_reconciliations:
                raise ValueError(
                    "formal global-PR reconciliation matrix is incomplete: "
                    f"{global_pr_reconciliations} != {expected_reconciliations}"
                )

    details_path = args.output / "transition_details.csv.gz"
    transition_summary, attribution, representatives, representative_images, detail_count = analyze_transitions(
        details_path,
        meta,
        records,
        match_data,
        pairs,
        states,
        args.scopes,
        thresholds,
        args.transition_budget,
        args.representatives_per_stratum,
    )

    tables = {
        "metrics_14_states.csv": metrics,
        "coco_small_per_iou.csv": per_iou_metrics,
        "coco_small_per_iou_budget_gap.csv": per_iou_gaps,
        "raw_ema_comparison.csv": raw_ema,
        "stage_coverage.csv": stage_rows,
        "match_aggregates.csv": merged_aggregates,
        "match_gt_buckets.csv": match_buckets,
        "transition_summary.csv": transition_summary,
        "transition_attribution.csv": attribution,
        "representative_annotations.csv": representatives,
        "representative_images.csv": representative_images,
    }
    if not args.skip_global_pr:
        tables.update(
            {
                "global_pr_curve.csv": global_pr_rows,
                "tp_fp_score_separation.csv": score_separation_rows,
                "tp_fp_score_histogram.csv": score_histogram_rows,
            }
        )
    for name, rows in tables.items():
        if rows:
            write_csv(
                args.output / name,
                rows,
                REPRESENTATIVE_FIELDS if name == "representative_annotations.csv" else None,
            )
        elif name == "raw_ema_comparison.csv":
            write_csv(
                args.output / name,
                [],
                ("epoch", "max_dets", "metric", "raw", "ema", "ema_minus_raw"),
            )
        elif name == "representative_annotations.csv":
            write_csv(
                args.output / name,
                [],
                REPRESENTATIVE_FIELDS,
            )
        elif name == "representative_images.csv":
            write_csv(
                args.output / name,
                [],
                (
                    "image_id", "file_name", "domain", "image_gt_count", "selected_strata",
                    "annotation_ids", "causes", "transitions",
                ),
            )
        else:
            raise ValueError(f"unexpected empty analysis table: {name}")
    generated_report = args.output / "analysis_summary.md"
    write_generated_report(
        generated_report,
        metrics,
        per_iou_gaps,
        transition_summary,
        attribution,
        score_separation_rows,
        formal=formal,
    )
    output_names = [*tables, "transition_details.csv.gz", "analysis_summary.md"]
    replay_input_identity = {}
    for epoch, directory in sorted(replay_dirs.items()):
        first_record = records[(epoch, states[0])]
        split_info = first_record.summary["splits"][args.split]
        state_artifacts = {}
        for state in states:
            state_record = records[(epoch, state)]
            state_artifacts[state] = {
                name: {
                    "logical_name": f"e{epoch}/{state}/{descriptor['file']}",
                    "bytes": int(descriptor["bytes"]),
                    "sha256": str(descriptor["sha256"]),
                }
                for name, descriptor in state_record.result.get("artifacts", {}).items()
            }
        replay_input_identity[f"e{epoch}"] = {
            "summary_logical_name": f"e{epoch}/summary.json",
            "summary_sha256": sha256(directory / "summary.json"),
            "metadata": {
                "logical_name": f"e{epoch}/{split_info['metadata']}",
                "bytes": first_record.metadata_path.stat().st_size,
                "sha256": str(split_info["metadata_sha256"]),
            },
            "checkpoint_sha256": str(first_record.summary.get("checkpoint_sha256", "")),
            "states": state_artifacts,
        }
    matcher_input_identity = {}
    for (epoch, state), directory in sorted(match_dirs.items()):
        roots = {}
        for name in (
            "complete.json",
            "protocol.json",
            "match_aggregates.csv",
            "detection_status_manifest.json",
            "gt_shard_manifest.json",
        ):
            path = directory / name
            if path.is_file():
                roots[name] = {
                    "logical_name": f"e{epoch}/{state}/{name}",
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        matcher_input_identity[f"e{epoch}/{state}"] = roots
    audit = {
        "schema": "sgqs_full_val_analysis_v2",
        "status": "complete" if formal else "testing_noncomplete",
        "formal_contract": formal,
        "split": args.split,
        "epochs": sorted(epochs),
        "states": list(states),
        "pairs": [list(pair) for pair in pairs],
        "thresholds": list(thresholds),
        "budgets": list(budgets),
        "transition_budget": args.transition_budget,
        "images": len(meta.image_ids),
        "noncrowd_annotations": len(meta.annotation_ids),
        "stage_summary_reconciliations": reconciliations,
        "global_pr_summary_reconciliations": global_pr_reconciliations,
        "global_pr_status": "skipped_by_explicit_flag" if args.skip_global_pr else "complete",
        "global_pr_backends": global_pr_backends,
        "transition_detail_rows": detail_count,
        "definitions": {
            "coco_small": "original annotation area < 32^2",
            "coco_small_eval": "original annotation area <= 32^2; formal evaluator scope",
            "local_hybrid_small": "direct 640x640 resize: area < 32^2 OR short side < 16",
            "stage_coverage": "independent per-GT best-IoU coverage; not TP/FN/recall",
            "formal_matching": "COCO score-ordered one-to-one matching from match shards",
            "ap_scope": (
                "default analysis spans IoU=.50:.05:.95, matching the ten IoU slices averaged by "
                "COCO AP; TP/FN transition counts are still not an additive AP decomposition because "
                "AP integrates precision over score-ranked recall and detection FP"
            ),
            "match_detail_budget": (
                "query/rank/score/iou detail columns describe the largest exported maxDets match; "
                "lower-budget status is obtained from the budget-specific status column and rank cutoff"
            ),
            "attribution": (
                "deepest surviving stage at the formal FN endpoint: no dense candidate, not selected "
                "into Encoder Top-640, lost in Decoder geometry, final query rank>300, or Top-300 "
                "independent coverage but one-to-one competition"
            ),
            "domain": "wired/wireless only when present as an explicit file-name path component; otherwise unknown",
        },
        "input_identity": {
            "replays": replay_input_identity,
            "matchers": matcher_input_identity,
            "matcher_matrix": matcher_matrix_identity,
        },
        "formal_identity": formal_identity,
        "outputs": [*output_names, "audit.json", "artifact_manifest.csv"],
    }
    audit_path = args.output / "audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_rows = output_manifest(args.output, [*output_names, "audit.json"])
    write_csv(args.output / "artifact_manifest.csv", manifest_rows)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=parse_epoch_path, action="append", required=True, metavar="EPOCH=DIR")
    parser.add_argument("--match", type=parse_match_path, action="append", required=True, metavar="EPOCH:STATE=DIR")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument(
        "--pair",
        type=parse_pair,
        action="append",
        help="repeat; default all adjacent e28:e29 through e33:e34 plus e28:e30 and e30:e34",
    )
    parser.add_argument("--thresholds", type=float, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[300, 640])
    parser.add_argument("--transition-budget", type=int, default=300)
    parser.add_argument("--scopes", nargs="+", default=["all", "coco_small_eval"])
    parser.add_argument("--representatives-per-stratum", type=int, default=3)
    parser.add_argument(
        "--testing",
        action="store_true",
        help="allow an incomplete synthetic/subset matrix; audit is marked testing_noncomplete",
    )
    parser.add_argument(
        "--skip-global-pr",
        action="store_true",
        help="testing only: skip evaluator reconstruction of global score-ranked PR and TP/FP scores",
    )
    args = parser.parse_args(argv)
    if args.representatives_per_stratum < 1:
        parser.error("--representatives-per-stratum must be positive")
    if any(value <= 0 or value > 1 for value in args.thresholds):
        parser.error("--thresholds must be in (0,1]")
    if any(value <= 0 for value in args.budgets):
        parser.error("--budgets must be positive")
    if args.skip_global_pr and not args.testing:
        parser.error("--skip-global-pr is permitted only with --testing")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    audit = run_analysis(args)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
