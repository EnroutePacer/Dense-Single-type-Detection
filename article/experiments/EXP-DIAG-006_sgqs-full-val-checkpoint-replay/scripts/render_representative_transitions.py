#!/usr/bin/env python3
"""Render audited full-validation transition examples without model inference.

The renderer consumes only the formal EXP-DIAG-006 analysis, replay metadata,
saved final-decoder predictions, and source images.  Encoder candidate geometry
was not persisted by the replay, so encoder-stage causes are shown as numeric
attribution rather than as invented encoder boxes.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Mapping, Sequence
import zipfile

from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import plot_full_val as formal_plot  # noqa: E402


STATE = "ema"
MATCH_SCOPE = "coco_small_eval"
MAX_DETS = 300
DISPLAY_THRESHOLDS = (0.50, 0.75, 0.95)
DOMAINS = ("wired", "wireless")
SHEET_SPECS = (
    (28, 29, "gain"),
    (30, 34, "loss"),
)
QUERY_COUNT = 640

REQUIRED_REPRESENTATIVE_FIELDS = frozenset(
    {
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
    }
)

MANIFEST_FIELDS = (
    "figure",
    "figure_bytes",
    "figure_sha256",
    "state",
    "epoch_start",
    "epoch_end",
    "transition",
    "match_scope",
    "max_dets",
    "iou_threshold",
    "cause",
    "domain",
    "panel_status",
    "selection_rank",
    "annotation_id",
    "image_id",
    "file_name",
    "source_image_sha256",
    "small_partition",
    "image_gt_count",
    "evidence_strength",
    "status_before",
    "status_after",
    "display_query_before",
    "display_rank_before",
    "display_score_before",
    "display_iou_before",
    "display_kind_before",
    "display_query_after",
    "display_rank_after",
    "display_score_after",
    "display_iou_after",
    "display_kind_after",
    "analysis_audit_sha256",
    "representatives_sha256",
    "metadata_logical_name",
    "metadata_sha256",
    "prediction_before_logical_name",
    "prediction_before_sha256",
    "prediction_after_logical_name",
    "prediction_after_sha256",
    "renderer_sha256",
    "numpy_version",
    "pillow_version",
    "font_family",
    "selection_rule",
    "geometry_note",
)


@dataclass(frozen=True)
class ImageMetadata:
    image_id: int
    file_name: str
    width: int
    height: int
    annotation_ids: np.ndarray
    boxes_xywh: np.ndarray


@dataclass(frozen=True)
class PredictionRows:
    logical_name: str
    sha256: str
    boxes: Mapping[int, np.ndarray]
    scores: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class DisplayCandidate:
    query: int
    rank: int
    score: float
    iou: float
    kind: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render six audited EMA transition contact sheets without inference."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _optional_int(value: object) -> int | None:
    text = str(value).strip()
    return None if text == "" else int(text)


def _optional_float(value: object) -> float | None:
    text = str(value).strip()
    return None if text == "" else float(text)


def read_representatives(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = REQUIRED_REPRESENTATIVE_FIELDS - fields
        if missing:
            raise ValueError(
                "representative annotation table lacks rendering fields: "
                + ", ".join(sorted(missing))
            )
        rows = list(reader)
    seen: set[tuple] = set()
    for row in rows:
        key = (
            row["state"],
            int(row["epoch_start"]),
            int(row["epoch_end"]),
            row["match_scope"],
            round(float(row["iou_threshold"]), 2),
            row["transition"],
            row["cause"],
            row["small_partition"],
            row["domain"],
            int(row["selection_rank"]),
        )
        if key in seen:
            raise ValueError(f"duplicate representative stratum/rank: {key}")
        seen.add(key)
    return rows


def _selection_sort_key(row: Mapping[str, str]) -> tuple:
    rank_distance = _optional_float(row["rank_distance_from_300"])
    return (
        -float(row["evidence_strength"]),
        math.inf if rank_distance is None else rank_distance,
        int(row["annotation_id"]),
        int(row["image_id"]),
        row["small_partition"],
        int(row["selection_rank"]),
    )


def select_panels(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, int, str, float, str, str], Mapping[str, str] | None]:
    selected: dict[tuple[int, int, str, float, str, str], Mapping[str, str] | None] = {}
    for start, end, transition in SHEET_SPECS:
        expected_status = ("fn", "tp") if transition == "gain" else ("tp", "fn")
        for threshold in DISPLAY_THRESHOLDS:
            for cause in formal_plot.CAUSES:
                for domain in DOMAINS:
                    candidates = [
                        row
                        for row in rows
                        if row["state"] == STATE
                        and int(row["epoch_start"]) == start
                        and int(row["epoch_end"]) == end
                        and row["match_scope"] == MATCH_SCOPE
                        and round(float(row["iou_threshold"]), 2) == threshold
                        and row["transition"] == transition
                        and row["cause"] == cause
                        and row["domain"] == domain
                    ]
                    key = (start, end, transition, threshold, cause, domain)
                    if not candidates:
                        selected[key] = None
                        continue
                    row = min(candidates, key=_selection_sort_key)
                    if (row["status_before"], row["status_after"]) != expected_status:
                        raise ValueError(
                            f"representative transition status mismatch for {key}: "
                            f"{row['status_before']}->{row['status_after']}"
                        )
                    selected[key] = row
    return selected


def formal_attribution_counts(
    rows: Sequence[Mapping[str, str]],
) -> dict[tuple[int, int, str, float, str], int]:
    """Return audited cross-domain cause counts for renderer reconciliation."""

    selected = formal_plot.select_attribution_rows(rows, formal_plot.FORMAL_PAIRS)
    return {
        (
            int(row["epoch_start"]),
            int(row["epoch_end"]),
            str(row["transition"]),
            float(row["iou_threshold"]),
            str(row["cause"]),
        ): int(row["count"])
        for row in selected
    }


def reconcile_panel_selections(
    selections: Mapping[tuple[int, int, str, float, str, str], Mapping[str, str] | None],
    attribution_counts: Mapping[tuple[int, int, str, float, str], int],
) -> None:
    """Prevent a missing representative row from being reported as a zero stratum.

    Attribution is aggregated across domains.  Therefore it can prove that both
    domain panels are empty when the aggregate count is zero, and it can require
    at least one representative when the count is positive.  It cannot prove
    that an individually missing wired/wireless panel has zero transitions; the
    renderer labels that case as unresolved instead of ``absent``.
    """

    for start, end, transition in SHEET_SPECS:
        for threshold in DISPLAY_THRESHOLDS:
            for cause in formal_plot.CAUSES:
                aggregate_key = (start, end, transition, threshold, cause)
                if aggregate_key not in attribution_counts:
                    raise ValueError(
                        f"formal attribution lacks representative aggregate: {aggregate_key}"
                    )
                count = int(attribution_counts[aggregate_key])
                if count < 0:
                    raise ValueError(f"negative formal attribution count: {aggregate_key}")
                present_domains = [
                    domain
                    for domain in DOMAINS
                    if selections[(start, end, transition, threshold, cause, domain)]
                    is not None
                ]
                if count == 0 and present_domains:
                    raise ValueError(
                        "representative rows exist for a zero-count formal stratum: "
                        f"{aggregate_key}, domains={present_domains}"
                    )
                if count > 0 and not present_domains:
                    raise ValueError(
                        "positive formal stratum lacks every domain representative: "
                        f"{aggregate_key}, count={count}"
                    )
                if len(present_domains) > count:
                    raise ValueError(
                        "more represented domains than formal transitions: "
                        f"{aggregate_key}, domains={present_domains}, count={count}"
                    )


def _descriptor(audit: Mapping, epoch: int, artifact: str, state: str | None = None) -> Mapping:
    replay = audit["input_identity"]["replays"][f"e{epoch}"]
    return replay["metadata"] if state is None else replay["states"][state][artifact]


def require_shared_metadata_identity(audit: Mapping, epochs: Sequence[int]) -> Mapping:
    """Require every rendered endpoint to use the exact same validation metadata."""

    endpoint_epochs = tuple(sorted(set(map(int, epochs))))
    if not endpoint_epochs:
        raise ValueError("no endpoint epochs were supplied for metadata validation")
    reference_epoch = endpoint_epochs[0]
    reference = _descriptor(audit, reference_epoch, "metadata")
    reference_identity = (int(reference["bytes"]), str(reference["sha256"]).lower())
    for epoch in endpoint_epochs[1:]:
        descriptor = _descriptor(audit, epoch, "metadata")
        identity = (int(descriptor["bytes"]), str(descriptor["sha256"]).lower())
        if identity != reference_identity:
            raise ValueError(
                "rendered endpoint metadata identity mismatch: "
                f"e{epoch} {identity} != e{reference_epoch} {reference_identity}"
            )
    return reference


def _artifact_path(
    replay_root: Path,
    descriptor: Mapping,
    epoch: int,
    *,
    state: str | None = None,
) -> tuple[Path, str]:
    logical = str(descriptor["logical_name"]).replace("\\", "/")
    parts = PurePosixPath(logical).parts
    expected_prefix = (f"e{epoch}",) if state is None else (f"e{epoch}", state)
    if len(parts) != len(expected_prefix) + 1 or tuple(parts[:-1]) != expected_prefix:
        raise ValueError(f"unexpected replay logical name: {logical}")
    filename = parts[-1]
    if filename in {"", ".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError(f"unsafe replay artifact name: {logical}")
    checkpoint = (replay_root / f"checkpoint{epoch:04d}").resolve()
    path = (checkpoint / filename).resolve()
    try:
        path.relative_to(checkpoint)
    except ValueError as exc:
        raise ValueError(f"replay artifact escapes checkpoint directory: {logical}") from exc
    return path, logical


def _validate_descriptor_file(path: Path, descriptor: Mapping, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(descriptor["bytes"])
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"{label} byte-size mismatch")
    expected_sha = str(descriptor["sha256"]).lower()
    observed_sha = formal_plot.sha256(path)
    if observed_sha != expected_sha:
        raise ValueError(f"{label} SHA-256 mismatch")
    return observed_sha


def load_selected_metadata(
    path: Path,
    audit: Mapping,
    selected_rows: Sequence[Mapping[str, str]],
) -> dict[int, ImageMetadata]:
    wanted_images = {int(row["image_id"]) for row in selected_rows}
    wanted_annotations = {int(row["annotation_id"]) for row in selected_rows}
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "image_ids",
            "image_width",
            "image_height",
            "image_file_name",
            "image_gt_start",
            "annotation_ids",
            "gt_image_index",
            "gt_bbox_xywh_original",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"replay metadata lacks fields: {sorted(missing)}")
        if str(archive["schema"].item()) != "sgqs_full_val_metadata_v1":
            raise ValueError("unsupported replay metadata schema")
        image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
        widths = np.asarray(archive["image_width"], dtype=np.int64)
        heights = np.asarray(archive["image_height"], dtype=np.int64)
        file_names = np.asarray(archive["image_file_name"], dtype=np.str_)
        starts = np.asarray(archive["image_gt_start"], dtype=np.int64)
        annotation_ids = np.asarray(archive["annotation_ids"], dtype=np.int64)
        gt_image_index = np.asarray(archive["gt_image_index"], dtype=np.int64)
        boxes = np.asarray(archive["gt_bbox_xywh_original"], dtype=np.float32)

    if len(image_ids) != int(audit["images"]) or len(annotation_ids) != int(
        audit["noncrowd_annotations"]
    ):
        raise ValueError("replay metadata population disagrees with formal analysis")
    if not (
        len(widths) == len(heights) == len(file_names) == len(image_ids)
        and len(starts) == len(image_ids) + 1
        and starts[0] == 0
        and starts[-1] == len(annotation_ids)
        and len(gt_image_index) == len(annotation_ids)
        and boxes.shape == (len(annotation_ids), 4)
    ):
        raise ValueError("replay metadata arrays are misaligned")
    image_lookup = {int(value): index for index, value in enumerate(image_ids) if int(value) in wanted_images}
    if set(image_lookup) != wanted_images:
        raise ValueError("representative image id is absent from replay metadata")

    result: dict[int, ImageMetadata] = {}
    annotation_to_image: dict[int, int] = {}
    for image_id, image_index in image_lookup.items():
        start, end = int(starts[image_index]), int(starts[image_index + 1])
        ids = annotation_ids[start:end].copy()
        image_boxes = boxes[start:end].copy()
        if not np.all(gt_image_index[start:end] == image_index):
            raise ValueError(f"GT/image alignment mismatch for image {image_id}")
        for annotation_id in ids:
            integer_id = int(annotation_id)
            if integer_id in wanted_annotations:
                if integer_id in annotation_to_image:
                    raise ValueError(f"duplicate representative annotation id: {integer_id}")
                annotation_to_image[integer_id] = image_id
        result[image_id] = ImageMetadata(
            image_id=image_id,
            file_name=str(file_names[image_index]),
            width=int(widths[image_index]),
            height=int(heights[image_index]),
            annotation_ids=ids,
            boxes_xywh=image_boxes,
        )
    if set(annotation_to_image) != wanted_annotations:
        raise ValueError("representative annotation id is absent from selected image metadata")
    for row in selected_rows:
        image_id = int(row["image_id"])
        annotation_id = int(row["annotation_id"])
        if annotation_to_image[annotation_id] != image_id:
            raise ValueError("representative annotation/image id mismatch")
        if result[image_id].file_name != row["file_name"]:
            raise ValueError("representative file name disagrees with replay metadata")
    return result


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise ValueError("truncated NPY member in prediction archive")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _discard_exact(stream, count: int) -> None:
    remaining = count
    while remaining:
        chunk = stream.read(min(remaining, 4 * 1024 * 1024))
        if not chunk:
            raise ValueError("truncated NPY member in prediction archive")
        remaining -= len(chunk)


def _stream_npy_rows(
    path: Path,
    member: str,
    row_indices: Sequence[int],
    expected_shape: tuple[int, ...],
) -> dict[int, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        member_name = f"{member}.npy"
        if member_name not in archive.namelist():
            raise ValueError(f"prediction archive lacks {member}")
        with archive.open(member_name) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError(f"unsupported NPY version for {member}: {version}")
            if tuple(shape) != expected_shape:
                raise ValueError(
                    f"prediction {member} shape mismatch: {tuple(shape)} != {expected_shape}"
                )
            dtype = np.dtype(dtype)
            if fortran_order or dtype.hasobject:
                raise ValueError(f"prediction {member} must be C-order numeric data")
            row_shape = tuple(shape[1:])
            row_values = int(np.prod(row_shape, dtype=np.int64))
            row_bytes = row_values * dtype.itemsize
            result: dict[int, np.ndarray] = {}
            current = 0
            for index in sorted(set(map(int, row_indices))):
                if index < current or index >= shape[0]:
                    raise ValueError(f"invalid selected image row {index} for {member}")
                _discard_exact(stream, (index - current) * row_bytes)
                raw = _read_exact(stream, row_bytes)
                result[index] = np.frombuffer(raw, dtype=dtype).reshape(row_shape).copy()
                current = index + 1
            return result


def load_prediction_rows(
    path: Path,
    descriptor: Mapping,
    logical_name: str,
    epoch: int,
    audit: Mapping,
    wanted_image_ids: set[int],
) -> PredictionRows:
    observed_sha = _validate_descriptor_file(path, descriptor, f"e{epoch}/{STATE} predictions")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "schema",
            "image_ids",
            "boxes_xywh",
            "scores",
            "state",
            "box_format",
            "query_indices",
        }
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"prediction archive lacks fields: {sorted(missing)}")
        if str(archive["schema"].item()) != "sgqs_compact_predictions_v1":
            raise ValueError("unsupported compact prediction schema")
        if str(archive["state"].item()) != f"checkpoint{epoch:04d}:{STATE}":
            raise ValueError(f"prediction state mismatch for e{epoch}/{STATE}")
        if str(archive["box_format"].item()) != "xywh_abs_original_pixels":
            raise ValueError("representative renderer requires original-pixel xywh boxes")
        image_ids = np.asarray(archive["image_ids"], dtype=np.int64)
        queries = np.asarray(archive["query_indices"], dtype=np.int64)
    if len(image_ids) != int(audit["images"]):
        raise ValueError("prediction image population disagrees with formal analysis")
    if len(queries) != QUERY_COUNT or not np.array_equal(queries, np.arange(QUERY_COUNT)):
        raise ValueError("prediction query axis is not the fixed 0..639 sequence")
    if len(set(map(int, image_ids))) != len(image_ids):
        raise ValueError("prediction image ids are not unique")
    image_lookup = {
        int(value): index for index, value in enumerate(image_ids) if int(value) in wanted_image_ids
    }
    if set(image_lookup) != wanted_image_ids:
        raise ValueError("selected image id is absent from compact predictions")
    row_indices = sorted(image_lookup.values())
    boxes_by_row = _stream_npy_rows(
        path, "boxes_xywh", row_indices, (len(image_ids), QUERY_COUNT, 4)
    )
    scores_by_row = _stream_npy_rows(
        path, "scores", row_indices, (len(image_ids), QUERY_COUNT)
    )
    return PredictionRows(
        logical_name=logical_name,
        sha256=observed_sha,
        boxes={image_id: boxes_by_row[row] for image_id, row in image_lookup.items()},
        scores={image_id: scores_by_row[row] for image_id, row in image_lookup.items()},
    )


def _xywh_to_xyxy(box: np.ndarray) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def _ious_xywh(boxes: np.ndarray, target: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    left = np.maximum(boxes[:, 0], target[0])
    top = np.maximum(boxes[:, 1], target[1])
    right = np.minimum(boxes[:, 0] + boxes[:, 2], target[0] + target[2])
    bottom = np.minimum(boxes[:, 1] + boxes[:, 3], target[1] + target[3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    target_area = max(0.0, float(target[2])) * max(0.0, float(target[3]))
    union = box_area + target_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def _assert_close(observed: float, expected: object, label: str, atol: float = 5e-5) -> None:
    parsed = _optional_float(expected)
    if parsed is None or not math.isclose(observed, parsed, rel_tol=0.0, abs_tol=atol):
        raise ValueError(f"{label} disagrees with saved transition evidence")


def choose_display_candidate(
    row: Mapping[str, str],
    side: str,
    boxes: np.ndarray,
    scores: np.ndarray,
    gt_box: np.ndarray,
) -> DisplayCandidate:
    if boxes.shape != (QUERY_COUNT, 4) or scores.shape != (QUERY_COUNT,):
        raise ValueError("selected prediction row has an unexpected shape")
    threshold = float(row["iou_threshold"])
    ious = _ious_xywh(boxes, gt_box)
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(QUERY_COUNT, dtype=np.int64)
    ranks[order] = np.arange(1, QUERY_COUNT + 1)
    eligible = order[ious[order] >= threshold]
    # ``ranked_gt_diagnostics`` resolves equal-IoU candidates after stable
    # score sorting.  Preserve that tie rule here so an all-zero-IoU row does
    # not silently display query 0 when a different query is rank 1.
    best_query = int(order[np.argmax(ious[order])])
    saved_decoder_rank = int(row[f"decoder_all_rank_{side}"])

    formal_query = _optional_int(row[f"formal_query_{side}"])
    if formal_query is not None:
        if not 0 <= formal_query < QUERY_COUNT:
            raise ValueError(f"formal query index is out of range: {formal_query}")
        query = formal_query
        kind = "formal TP" if row[f"status_{side}"] == "tp" else "formal maxDets=640 match"
        formal_rank = _optional_int(row[f"formal_rank_{side}"])
        if formal_rank != int(ranks[query]):
            raise ValueError("formal query rank disagrees with compact predictions")
        _assert_close(float(scores[query]), row[f"formal_score_{side}"], "formal score", 1e-6)
        _assert_close(float(ious[query]), row[f"formal_iou_{side}"], "formal IoU")
    elif saved_decoder_rank <= QUERY_COUNT:
        query = int(order[saved_decoder_rank - 1])
        if float(ious[query]) < threshold - 5e-5:
            raise ValueError("saved decoder-all rank points to a non-qualifying query")
        kind = "first saved IoU-qualified query"
    else:
        query = best_query
        kind = "best-IoU query; below threshold"

    decoder_rank = int(ranks[int(eligible[0])]) if len(eligible) else QUERY_COUNT + 1
    top300 = order[:MAX_DETS]
    top300_ious = ious[top300]
    top300_eligible = top300[top300_ious >= threshold]
    top300_rank = int(ranks[int(top300_eligible[0])]) if len(top300_eligible) else MAX_DETS + 1
    _assert_close(float(np.max(ious)), row[f"decoder_all_iou_{side}"], "decoder-all best IoU")
    _assert_close(
        float(np.max(top300_ious)),
        row[f"decoder_top300_iou_{side}"],
        "decoder-top300 best IoU",
    )
    if saved_decoder_rank != decoder_rank and not np.any(
        np.abs(ious - threshold) <= 5e-5
    ):
        raise ValueError("decoder-all threshold rank disagrees with compact predictions")
    saved_top300_rank = int(row[f"decoder_top300_rank_{side}"])
    if saved_top300_rank != top300_rank and not np.any(
        np.abs(top300_ious - threshold) <= 5e-5
    ):
        raise ValueError("decoder-top300 threshold rank disagrees with compact predictions")
    return DisplayCandidate(
        query=query,
        rank=int(ranks[query]),
        score=float(scores[query]),
        iou=float(ious[query]),
        kind=kind,
    )


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size=size)


def _safe_image_path(images_root: Path, file_name: str) -> Path:
    relative = PurePosixPath(file_name.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"unsafe representative image path: {file_name}")
    if ":" in relative.parts[0]:
        raise ValueError(f"unsafe representative image path: {file_name}")
    # The formal dataset view intentionally contains directory junctions into
    # the immutable source dataset.  Resolve ``..`` lexically, but do not
    # dereference those controlled junctions and mistake their targets for a
    # path traversal outside the view.
    root = Path(os.path.abspath(images_root))
    path = Path(os.path.abspath(root.joinpath(*relative.parts)))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"representative image escapes images root: {file_name}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _target_box(metadata: ImageMetadata, annotation_id: int) -> np.ndarray:
    matches = np.flatnonzero(metadata.annotation_ids == annotation_id)
    if len(matches) != 1:
        raise ValueError(f"annotation {annotation_id} is not unique within image metadata")
    return metadata.boxes_xywh[int(matches[0])]


def _crop_box(gt_box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = map(float, gt_box)
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    crop_width = min(float(width), max(160.0, box_width * 8.0, box_height * 5.0))
    crop_height = min(float(height), max(120.0, box_height * 8.0, box_width * 3.0))
    target_aspect = 480.0 / 258.0
    if crop_width / crop_height < target_aspect:
        crop_width = min(float(width), crop_height * target_aspect)
    else:
        crop_height = min(float(height), crop_width / target_aspect)
    left = max(0.0, min(float(width) - crop_width, center_x - crop_width / 2.0))
    top = max(0.0, min(float(height) - crop_height, center_y - crop_height / 2.0))
    return (
        int(math.floor(left)),
        int(math.floor(top)),
        int(math.ceil(left + crop_width)),
        int(math.ceil(top + crop_height)),
    )


def _fit(image: Image.Image, size: tuple[int, int], background=(24, 28, 35)) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, background)
    canvas.paste(copy, ((size[0] - copy.width) // 2, (size[1] - copy.height) // 2))
    return canvas


def render_endpoint(
    source: Image.Image,
    metadata: ImageMetadata,
    gt_box: np.ndarray,
    prediction_box: np.ndarray,
    candidate: DisplayCandidate,
    status: str,
    epoch: int,
    crop: tuple[int, int, int, int],
) -> Image.Image:
    left, top, _right, _bottom = crop
    canvas = source.crop(crop).convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line = max(1, round(min(canvas.size) / 640))

    def translated(box: np.ndarray) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = _xywh_to_xyxy(box)
        return x1 - left, y1 - top, x2 - left, y2 - top

    for box in metadata.boxes_xywh:
        x1, y1, x2, y2 = translated(box)
        if x2 >= 0 and y2 >= 0 and x1 <= canvas.width and y1 <= canvas.height:
            draw.rectangle((x1, y1, x2, y2), outline=(205, 213, 225, 110), width=line)
    draw.rectangle(translated(gt_box), outline=(255, 45, 190, 255), width=line * 4)
    prediction_color = (30, 220, 120, 255) if status == "tp" else (255, 160, 35, 255)
    draw.rectangle(
        translated(prediction_box), outline=prediction_color, width=line * 3
    )
    composited = Image.alpha_composite(canvas, overlay).convert("RGB")
    panel = Image.new("RGB", (480, 300), (24, 28, 35))
    panel.paste(_fit(composited, (480, 258)), (0, 42))
    header = ImageDraw.Draw(panel)
    status_color = (65, 220, 140) if status == "tp" else (255, 175, 65)
    header.text((10, 5), f"e{epoch}  D300 {status.upper()}", font=_font(15, True), fill=status_color)
    header.text(
        (10, 23),
        f"q{candidate.query} r{candidate.rank} s={candidate.score:.3f} IoU={candidate.iou:.3f}",
        font=_font(12),
        fill=(235, 238, 244),
    )
    return panel


def _absent_panel(epoch: int) -> Image.Image:
    panel = Image.new("RGB", (480, 300), (43, 48, 57))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (20, 120),
        f"e{epoch}: ZERO TRANSITIONS",
        font=_font(18, True),
        fill=(180, 187, 199),
    )
    return panel


def _unresolved_domain_panel(epoch: int, aggregate_count: int) -> Image.Image:
    panel = Image.new("RGB", (480, 300), (43, 48, 57))
    draw = ImageDraw.Draw(panel)
    draw.text(
        (20, 105),
        f"e{epoch}: NO DOMAIN REPRESENTATIVE",
        font=_font(17, True),
        fill=(235, 190, 95),
    )
    draw.text(
        (20, 137),
        f"cross-domain cause count = {aggregate_count}",
        font=_font(14),
        fill=(200, 205, 215),
    )
    return panel


def render_sheet(
    output_path: Path,
    start: int,
    end: int,
    transition: str,
    threshold: float,
    selections: Mapping[tuple, Mapping[str, str] | None],
    metadata: Mapping[int, ImageMetadata],
    predictions: Mapping[int, PredictionRows],
    images_root: Path,
    image_hash_cache: dict[Path, str],
    attribution_counts: Mapping[tuple[int, int, str, float, str], int],
) -> list[dict]:
    label_width, cell_width, cell_height = 260, 480, 300
    header_height = 88
    rows = len(formal_plot.CAUSES) * len(DOMAINS)
    sheet = Image.new(
        "RGB",
        (label_width + cell_width * 2, header_height + rows * cell_height),
        (250, 251, 253),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (18, 10),
        f"EMA e{start}->e{end} formal {transition} | evaluator-small | IoU={threshold:.2f} | maxDets=300",
        font=_font(22, True),
        fill=(25, 31, 42),
    )
    draw.text(
        (18, 43),
        "Boxes are final Decoder only; cause labels come from saved stage IoU/rank tables.",
        font=_font(14),
        fill=(75, 85, 99),
    )
    draw.text((label_width + 190, 66), f"before e{start}", font=_font(14, True), fill=(45, 55, 70))
    draw.text(
        (label_width + cell_width + 190, 66),
        f"after e{end}",
        font=_font(14, True),
        fill=(45, 55, 70),
    )

    manifest_rows: list[dict] = []
    row_index = 0
    for cause in formal_plot.CAUSES:
        for domain in DOMAINS:
            key = (start, end, transition, threshold, cause, domain)
            representative = selections[key]
            y = header_height + row_index * cell_height
            draw.rectangle((0, y, sheet.width, y + cell_height - 1), outline=(210, 214, 221), width=1)
            draw.text((14, y + 16), cause, font=_font(16, True), fill=(35, 42, 54))
            draw.text((14, y + 43), domain, font=_font(15), fill=(70, 82, 99))
            base_manifest = {
                "figure": output_path.name,
                "state": STATE,
                "epoch_start": start,
                "epoch_end": end,
                "transition": transition,
                "match_scope": MATCH_SCOPE,
                "max_dets": MAX_DETS,
                "iou_threshold": f"{threshold:.2f}",
                "cause": cause,
                "domain": domain,
            }
            if representative is None:
                aggregate_key = (start, end, transition, threshold, cause)
                aggregate_count = int(attribution_counts[aggregate_key])
                if aggregate_count == 0:
                    panel_status = "absent"
                    label = "aggregate count = 0"
                    label_color = (150, 68, 68)
                    before_empty = _absent_panel(start)
                    after_empty = _absent_panel(end)
                    geometry_note = "formal cross-domain cause count is zero"
                else:
                    panel_status = "domain_not_resolved"
                    label = f"domain count unavailable; aggregate={aggregate_count}"
                    label_color = (155, 105, 20)
                    before_empty = _unresolved_domain_panel(start, aggregate_count)
                    after_empty = _unresolved_domain_panel(end, aggregate_count)
                    geometry_note = (
                        "formal attribution is positive across domains, but the compact "
                        "attribution table cannot prove this domain has zero transitions"
                    )
                draw.text((14, y + 72), label, font=_font(12, True), fill=label_color)
                sheet.paste(before_empty, (label_width, y))
                sheet.paste(after_empty, (label_width + cell_width, y))
                manifest_rows.append(
                    {
                        **base_manifest,
                        "panel_status": panel_status,
                        "geometry_note": geometry_note,
                    }
                )
                row_index += 1
                continue

            image_id = int(representative["image_id"])
            annotation_id = int(representative["annotation_id"])
            image_metadata = metadata[image_id]
            image_path = _safe_image_path(images_root, image_metadata.file_name)
            if image_path not in image_hash_cache:
                image_hash_cache[image_path] = formal_plot.sha256(image_path)
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
            if source.size != (image_metadata.width, image_metadata.height):
                raise ValueError(f"image dimensions disagree with replay metadata: {image_metadata.file_name}")
            gt_box = _target_box(image_metadata, annotation_id)
            crop = _crop_box(gt_box, image_metadata.width, image_metadata.height)
            before = choose_display_candidate(
                representative,
                "before",
                predictions[start].boxes[image_id],
                predictions[start].scores[image_id],
                gt_box,
            )
            after = choose_display_candidate(
                representative,
                "after",
                predictions[end].boxes[image_id],
                predictions[end].scores[image_id],
                gt_box,
            )
            before_panel = render_endpoint(
                source,
                image_metadata,
                gt_box,
                predictions[start].boxes[image_id][before.query],
                before,
                representative["status_before"],
                start,
                crop,
            )
            after_panel = render_endpoint(
                source,
                image_metadata,
                gt_box,
                predictions[end].boxes[image_id][after.query],
                after,
                representative["status_after"],
                end,
                crop,
            )
            sheet.paste(before_panel, (label_width, y))
            sheet.paste(after_panel, (label_width + cell_width, y))
            draw.text(
                (14, y + 72),
                f"ann {annotation_id} | img {image_id}",
                font=_font(12),
                fill=(55, 65, 80),
            )
            draw.text(
                (14, y + 94),
                f"{representative['small_partition']} | GT/image {representative['image_gt_count']}",
                font=_font(11),
                fill=(75, 85, 99),
            )
            draw.text(
                (14, y + 116),
                f"evidence={float(representative['evidence_strength']):.3f}",
                font=_font(11),
                fill=(75, 85, 99),
            )
            manifest_rows.append(
                {
                    **base_manifest,
                    "panel_status": "selected",
                    "selection_rank": representative["selection_rank"],
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "file_name": image_metadata.file_name,
                    "source_image_sha256": image_hash_cache[image_path],
                    "small_partition": representative["small_partition"],
                    "image_gt_count": representative["image_gt_count"],
                    "evidence_strength": representative["evidence_strength"],
                    "status_before": representative["status_before"],
                    "status_after": representative["status_after"],
                    "display_query_before": before.query,
                    "display_rank_before": before.rank,
                    "display_score_before": f"{before.score:.9g}",
                    "display_iou_before": f"{before.iou:.9g}",
                    "display_kind_before": before.kind,
                    "display_query_after": after.query,
                    "display_rank_after": after.rank,
                    "display_score_after": f"{after.score:.9g}",
                    "display_iou_after": f"{after.iou:.9g}",
                    "display_kind_after": after.kind,
                    "selection_rule": (
                        f"{representative['selection_rule']}; renderer keeps strongest evidence "
                        "across small partitions for the fixed cause/domain"
                    ),
                    "geometry_note": (
                        "magenta=selected GT; green=formal TP query; orange=FN-side decoder "
                        "evidence; encoder geometry was not persisted"
                    ),
                }
            )
            row_index += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    sheet.save(temporary, format="PNG", compress_level=9, optimize=False)
    os.replace(temporary, output_path)
    figure_bytes = output_path.stat().st_size
    figure_sha = formal_plot.sha256(output_path)
    for row in manifest_rows:
        row["figure_bytes"] = figure_bytes
        row["figure_sha256"] = figure_sha
    return manifest_rows


def write_manifest(path: Path, rows: Sequence[Mapping]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> list[Path]:
    data_dir = args.data_dir.resolve()
    replay_root = args.replay_root.resolve()
    images_root = args.images_root.resolve()
    output = args.output.resolve()
    audit = formal_plot.validate_formal_analysis(data_dir)
    required_pairs = {(start, end) for start, end, _transition in SHEET_SPECS}
    audited_pairs = {tuple(map(int, pair)) for pair in audit.get("pairs", ())}
    if not required_pairs <= audited_pairs:
        raise ValueError("formal analysis audit lacks required e28->e29/e30->e34 pairs")

    representatives_path = data_dir / "representative_annotations.csv"
    representatives = read_representatives(representatives_path)
    selections = select_panels(representatives)
    attribution_counts = formal_attribution_counts(
        formal_plot.read_csv(data_dir / "transition_attribution.csv")
    )
    reconcile_panel_selections(selections, attribution_counts)
    selected_rows = [row for row in selections.values() if row is not None]

    endpoint_epochs = sorted({epoch for spec in SHEET_SPECS for epoch in spec[:2]})
    metadata_descriptor = require_shared_metadata_identity(audit, endpoint_epochs)
    metadata_path, metadata_logical = _artifact_path(
        replay_root, metadata_descriptor, 28
    )
    metadata_sha = _validate_descriptor_file(metadata_path, metadata_descriptor, "e28 metadata")
    metadata = load_selected_metadata(metadata_path, audit, selected_rows)

    epoch_image_ids: dict[int, set[int]] = {epoch: set() for spec in SHEET_SPECS for epoch in spec[:2]}
    for (start, end, _transition, _threshold, _cause, _domain), row in selections.items():
        if row is not None:
            image_id = int(row["image_id"])
            epoch_image_ids[start].add(image_id)
            epoch_image_ids[end].add(image_id)

    predictions: dict[int, PredictionRows] = {}
    for epoch in sorted(epoch_image_ids):
        descriptor = _descriptor(audit, epoch, "predictions", STATE)
        path, logical = _artifact_path(replay_root, descriptor, epoch, state=STATE)
        predictions[epoch] = load_prediction_rows(
            path,
            descriptor,
            logical,
            epoch,
            audit,
            epoch_image_ids[epoch],
        )

    output.mkdir(parents=True, exist_ok=True)
    audit_sha = formal_plot.sha256(data_dir / "audit.json")
    representatives_sha = formal_plot.sha256(representatives_path)
    renderer_sha = formal_plot.sha256(Path(__file__).resolve())
    image_hash_cache: dict[Path, str] = {}
    manifest_rows: list[dict] = []
    figures: list[Path] = []
    for start, end, transition in SHEET_SPECS:
        for threshold in DISPLAY_THRESHOLDS:
            filename = (
                f"representative_ema_e{start}_e{end}_{transition}_"
                f"iou{int(round(threshold * 100)):02d}.png"
            )
            figure = output / filename
            rows = render_sheet(
                figure,
                start,
                end,
                transition,
                threshold,
                selections,
                metadata,
                predictions,
                images_root,
                image_hash_cache,
                attribution_counts,
            )
            for row in rows:
                row.update(
                    {
                        "analysis_audit_sha256": audit_sha,
                        "representatives_sha256": representatives_sha,
                        "metadata_logical_name": metadata_logical,
                        "metadata_sha256": metadata_sha,
                        "prediction_before_logical_name": predictions[start].logical_name,
                        "prediction_before_sha256": predictions[start].sha256,
                        "prediction_after_logical_name": predictions[end].logical_name,
                        "prediction_after_sha256": predictions[end].sha256,
                        "renderer_sha256": renderer_sha,
                        "numpy_version": np.__version__,
                        "pillow_version": PILLOW_VERSION,
                        "font_family": "DejaVu Sans",
                    }
                )
            manifest_rows.extend(rows)
            figures.append(figure)
    manifest = output / "representative_transition_figure_manifest.csv"
    write_manifest(manifest, manifest_rows)
    return [*figures, manifest]


def main(argv: Sequence[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
