#!/usr/bin/env python3
"""Render deterministic Encoder/Decoder probes for formal EXP-DIAG-006 examples.

The full-validation replay deliberately persisted compact per-GT stage summaries
and final Decoder predictions, not all Encoder Top-640 boxes.  This script fills
that qualitative gap without replaying the full validation set: it selects a
small, deterministic set of images from the *formal* representative table,
reruns only those images through every EMA checkpoint, and captures the real
Encoder selector with :class:`tools.table_cells.full_val_replay.CaptureCandidates`.

The selective forward is accepted only when its final Decoder boxes and scores
are bit-identical to the canonical compact replay.  The resulting figures are
illustrations of audited full-validation transitions; their sample frequency is
not an estimate of AP, recall, or mechanism prevalence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import subprocess
import sys
from typing import Mapping, Sequence
import zipfile

import numpy as np
from PIL import Image, ImageDraw, __version__ as PILLOW_VERSION


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
for import_root in (SCRIPT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import analyze_full_val as formal_analysis  # noqa: E402
import plot_full_val as formal_plot  # noqa: E402
import render_representative_transitions as transition_render  # noqa: E402
from tools.table_cells import full_val_replay  # noqa: E402
from tools.table_cells import run_full_val_replay_matrix as replay_matrix  # noqa: E402


EPOCHS = tuple(range(28, 35))
STATE = "ema"
IOU_THRESHOLD = 0.75
QUERY_COUNT = 640
TARGET_TRANSITIONS = ((28, 29, "gain"), (30, 34, "loss"))
TRACKS = (
    ("evaluator_small", "coco_small_eval", None),
    ("hybrid_only", "all", "hybrid_only"),
)
DOMAINS = ("wired", "wireless")
SOURCE_LEVEL_NAMES = ("P3", "P4", "P5")
SOURCE_LEVEL_COLORS = (
    (35, 210, 90, 225),
    (40, 135, 255, 225),
    (185, 70, 245, 225),
)
DECODER_CLASS_COLORS = {
    "small": (255, 45, 50),
    "medium": (255, 190, 30),
    "large": (35, 190, 245),
}
MAX_UNIQUE_IMAGES = 40
PROBE_DATA_SCHEMA = "sgqs_checkpoint_query_probe_data_v1"
REPLAY_MATRIX_SCHEMA = "sgqs_full_val_matrix_orchestration_v1"

PROBE_DATA_FIELDS = (
    "schema",
    "image_ids",
    "image_sizes_wh",
    "epochs",
    "state",
    "encoder_selected_indices",
    "encoder_levels",
    "encoder_boxes_cxcywh",
    "decoder_boxes_xywh",
    "decoder_scores",
    "decoder_hybrid_small",
    "strides",
    "spatial_shapes",
    "input_size",
)

SELECTION_FIELDS = (
    "probe_rank",
    "track",
    "state",
    "epoch_start",
    "epoch_end",
    "transition",
    "match_scope",
    "iou_threshold",
    "cause",
    "small_partition",
    "domain",
    "annotation_id",
    "image_id",
    "file_name",
    "image_gt_count",
    "evidence_strength",
    "rank_distance_from_300",
    "source_selection_rank",
    "selection_rule",
)

MANIFEST_FIELDS = (
    "figure",
    "figure_bytes",
    "figure_sha256",
    "state",
    "epoch",
    "image_id",
    "file_name",
    "source_image_bytes",
    "source_image_sha256",
    "selected_tracks",
    "selected_strata",
    "selected_annotation_ids",
    "checkpoint_file",
    "checkpoint_bytes",
    "checkpoint_sha256",
    "checkpoint_state",
    "config_sha256",
    "annotation_sha256",
    "metadata_logical_name",
    "metadata_sha256",
    "prediction_logical_name",
    "prediction_sha256",
    "probe_data_file",
    "probe_data_bytes",
    "probe_data_sha256",
    "probe_data_schema",
    "encoder_selected_count",
    "encoder_feat_strides",
    "encoder_spatial_shapes",
    "encoder_p3_count",
    "encoder_p4_count",
    "encoder_p5_count",
    "encoder_selected_indices_sha256",
    "encoder_selected_boxes_sha256",
    "decoder_query_count",
    "decoder_hybrid_small_count",
    "decoder_hybrid_mask_sha256",
    "decoder_boxes_exact",
    "decoder_scores_exact",
    "decoder_rank_order_exact",
    "decoder_hybrid_mask_exact",
    "decoder_box_max_abs_error",
    "decoder_score_max_abs_error",
    "analysis_audit_sha256",
    "analysis_artifact_manifest_sha256",
    "representatives_sha256",
    "selection_sha256",
    "probe_script_sha256",
    "full_val_replay_sha256",
    "analyze_full_val_sha256",
    "plot_full_val_sha256",
    "transition_renderer_sha256",
    "replay_matrix_file",
    "replay_matrix_bytes",
    "replay_matrix_sha256",
    "supervision_probe_sha256",
    "engine_python_tree_sha256",
    "runtime_backend_sha256",
    "runtime_backend_json",
    "torch_version",
    "torchvision_version",
    "torch_cuda_version",
    "numpy_version",
    "pillow_version",
    "runtime_device",
    "semantics",
)

MANIFEST_LOGICAL_PATH_FIELDS = (
    "figure",
    "file_name",
    "checkpoint_file",
    "metadata_logical_name",
    "prediction_logical_name",
    "probe_data_file",
    "replay_matrix_file",
)


@dataclass(frozen=True)
class Reconciliation:
    box_max_abs_error: float
    score_max_abs_error: float
    boxes_exact: bool
    scores_exact: bool
    rank_order_exact: bool
    hybrid_mask_exact: bool


@dataclass(frozen=True)
class ProbeObservation:
    epoch: int
    image_id: int
    encoder_boxes_cxcywh: np.ndarray
    encoder_dense_indices: np.ndarray
    encoder_levels: np.ndarray
    level_counts: tuple[int, int, int]
    decoder_boxes_xywh: np.ndarray
    decoder_scores: np.ndarray
    decoder_classes: np.ndarray
    decoder_hybrid_small: np.ndarray
    reconciliation: Reconciliation
    prediction_logical_name: str
    prediction_sha256: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-data", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _optional_float(value: object) -> float | None:
    text = str(value).strip()
    return None if not text else float(text)


def _selection_sort_key(row: Mapping[str, str]) -> tuple:
    distance = _optional_float(row["rank_distance_from_300"])
    return (
        -float(row["evidence_strength"]),
        math.inf if distance is None else distance,
        int(row["annotation_id"]),
        int(row["image_id"]),
        str(row["small_partition"]),
        int(row["selection_rank"]),
    )


def read_formal_representatives(path: Path) -> list[dict[str, str]]:
    """Read the exact analyzer-owned schema, rejecting extra or reordered fields."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(formal_analysis.REPRESENTATIVE_FIELDS):
            raise ValueError("representative annotation table has a non-canonical schema")
        rows = list(reader)
    # Reuse the existing duplicate-stratum and rendering-field validation too.
    transition_render.read_representatives(path)
    return rows


def select_probe_rows(
    rows: Sequence[Mapping[str, str]], *, max_per_stratum: int = 1
) -> list[dict[str, object]]:
    """Select bounded, deterministic formal and hybrid-only probe rows."""

    if max_per_stratum != 1:
        raise ValueError("query probe protocol fixes max_per_stratum=1")
    required = {
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
        "status_before",
        "status_after",
        "coco_small",
        "local_hybrid_small",
    }
    selected: list[dict[str, object]] = []
    seen_source_rows: set[tuple] = set()
    coverage: set[tuple[str, int, int, str]] = set()

    for track, scope, required_partition in TRACKS:
        for start, end, transition in TARGET_TRANSITIONS:
            expected_status = ("fn", "tp") if transition == "gain" else ("tp", "fn")
            for cause in formal_plot.CAUSES:
                for domain in DOMAINS:
                    candidates = []
                    for row in rows:
                        missing = required - set(row)
                        if missing:
                            raise ValueError(
                                "representative row lacks query-probe fields: "
                                + ", ".join(sorted(missing))
                            )
                        if (
                            row["state"] != STATE
                            or int(row["epoch_start"]) != start
                            or int(row["epoch_end"]) != end
                            or row["transition"] != transition
                            or row["match_scope"] != scope
                            or not math.isclose(
                                float(row["iou_threshold"]),
                                IOU_THRESHOLD,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                            or row["cause"] != cause
                            or row["domain"] != domain
                            or (
                                required_partition is not None
                                and row["small_partition"] != required_partition
                            )
                        ):
                            continue
                        try:
                            partition_flags = (
                                int(row["coco_small"]),
                                int(row["local_hybrid_small"]),
                            )
                        except (TypeError, ValueError) as exc:
                            raise ValueError("representative small flags must be binary") from exc
                        expected_partition = {
                            (1, 1): "coco_and_hybrid",
                            (1, 0): "coco_only",
                            (0, 1): "hybrid_only",
                            (0, 0): "neither",
                        }.get(partition_flags)
                        if expected_partition != row["small_partition"]:
                            raise ValueError(
                                "representative small_partition disagrees with stored flags"
                            )
                        if (row["status_before"], row["status_after"]) != expected_status:
                            raise ValueError(
                                f"representative status mismatch for {track}/e{start}->e{end}/"
                                f"{cause}/{domain}"
                            )
                        source_key = (
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
                        if source_key in seen_source_rows:
                            raise ValueError(f"duplicate representative source row: {source_key}")
                        seen_source_rows.add(source_key)
                        candidates.append(row)

                    for probe_rank, row in enumerate(
                        sorted(candidates, key=_selection_sort_key)[:max_per_stratum], 1
                    ):
                        evidence = float(row["evidence_strength"])
                        if not math.isfinite(evidence):
                            raise ValueError("representative evidence_strength must be finite")
                        coverage.add((track, start, end, transition))
                        selected.append(
                            {
                                "probe_rank": probe_rank,
                                "track": track,
                                "state": STATE,
                                "epoch_start": start,
                                "epoch_end": end,
                                "transition": transition,
                                "match_scope": row["match_scope"],
                                "iou_threshold": format(IOU_THRESHOLD, ".2f"),
                                "cause": row["cause"],
                                "small_partition": row["small_partition"],
                                "domain": row["domain"],
                                "annotation_id": int(row["annotation_id"]),
                                "image_id": int(row["image_id"]),
                                "file_name": row["file_name"],
                                "image_gt_count": int(row["image_gt_count"]),
                                "evidence_strength": format(evidence, ".9g"),
                                "rank_distance_from_300": row["rank_distance_from_300"],
                                "source_selection_rank": int(row["selection_rank"]),
                                "selection_rule": (
                                    "formal representative table; IoU=.75; best evidence desc, "
                                    "distance to rank 300 asc, annotation_id/image_id/partition/"
                                    "source selection_rank asc; one row per present cause/domain"
                                ),
                            }
                        )

    expected_coverage = {
        (track, start, end, transition)
        for track, _scope, _partition in TRACKS
        for start, end, transition in TARGET_TRANSITIONS
    }
    missing_coverage = expected_coverage - coverage
    if missing_coverage:
        raise ValueError(
            "formal representative table lacks required query-probe track/transition: "
            + ", ".join(map(str, sorted(missing_coverage)))
        )
    if not selected:
        raise ValueError("formal representative table yielded no query-probe rows")
    unique_images = {int(row["image_id"]) for row in selected}
    if len(unique_images) > MAX_UNIQUE_IMAGES:
        raise ValueError(
            f"query probe selection exceeds {MAX_UNIQUE_IMAGES} unique images: "
            f"{len(unique_images)}"
        )
    return selected


def dense_source_levels(
    dense_indices: np.ndarray,
    strides: Sequence[int],
    *,
    input_size: tuple[int, int] = full_val_replay.INPUT_SIZE,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Map true dense candidate indices to their P3/P4/P5 source levels."""

    indices = np.asarray(dense_indices, dtype=np.int64)
    if indices.shape != (QUERY_COUNT,):
        raise ValueError("Encoder selected index vector must contain exactly 640 entries")
    if len(strides) != len(SOURCE_LEVEL_NAMES) or any(int(value) <= 0 for value in strides):
        raise ValueError("query probe requires three positive P3/P4/P5 feature strides")
    height, width = map(int, input_size)
    counts = np.asarray(
        [(height // int(stride)) * (width // int(stride)) for stride in strides],
        dtype=np.int64,
    )
    boundaries = np.cumsum(counts)
    if np.any(indices < 0) or np.any(indices >= boundaries[-1]):
        raise ValueError("selected dense index lies outside the P3/P4/P5 feature grid")
    if len(set(map(int, indices))) != QUERY_COUNT:
        raise ValueError("Encoder Top-640 selected dense indices must be unique")
    levels = np.searchsorted(boundaries, indices, side="right").astype(np.uint8)
    level_counts = tuple(int(np.count_nonzero(levels == level)) for level in range(3))
    if sum(level_counts) != QUERY_COUNT:
        raise RuntimeError("P3/P4/P5 counts do not partition Encoder Top-640")
    return levels, level_counts


def decoder_size_classes(
    boxes_xywh: np.ndarray, image_width: int, image_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Classify final Decoder boxes in the fixed 640x640 input coordinate system."""

    boxes = np.asarray(boxes_xywh, dtype=np.float64)
    if boxes.shape != (QUERY_COUNT, 4) or not np.isfinite(boxes).all():
        raise ValueError("Decoder boxes must be finite [640,4] original-pixel xywh")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    scale_x = full_val_replay.INPUT_SIZE[1] / float(image_width)
    scale_y = full_val_replay.INPUT_SIZE[0] / float(image_height)
    x1 = np.clip(boxes[:, 0] * scale_x, 0.0, full_val_replay.INPUT_SIZE[1])
    y1 = np.clip(boxes[:, 1] * scale_y, 0.0, full_val_replay.INPUT_SIZE[0])
    x2 = np.clip((boxes[:, 0] + boxes[:, 2]) * scale_x, 0.0, full_val_replay.INPUT_SIZE[1])
    y2 = np.clip((boxes[:, 1] + boxes[:, 3]) * scale_y, 0.0, full_val_replay.INPUT_SIZE[0])
    widths = np.maximum(0.0, x2 - x1)
    heights = np.maximum(0.0, y2 - y1)
    areas = widths * heights
    short_sides = np.minimum(widths, heights)
    hybrid_small = (areas < 32.0**2) | (short_sides < 16.0)
    classes = np.full(QUERY_COUNT, "large", dtype="<U6")
    classes[areas < 96.0**2] = "medium"
    classes[hybrid_small] = "small"
    return classes, hybrid_small


def decoder_size_classes_normalized(
    boxes_cxcywh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify raw normalized Decoder boxes exactly in the 640-input frame."""

    boxes = np.asarray(boxes_cxcywh, dtype=np.float64)
    if boxes.shape != (QUERY_COUNT, 4) or not np.isfinite(boxes).all():
        raise ValueError("normalized Decoder boxes must be finite [640,4]")
    width = float(full_val_replay.INPUT_SIZE[1])
    height = float(full_val_replay.INPUT_SIZE[0])
    x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2.0) * width, 0.0, width)
    y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2.0) * height, 0.0, height)
    x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2.0) * width, 0.0, width)
    y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2.0) * height, 0.0, height)
    widths = np.maximum(0.0, x2 - x1)
    heights = np.maximum(0.0, y2 - y1)
    areas = widths * heights
    short_sides = np.minimum(widths, heights)
    hybrid_small = (areas < 32.0**2) | (short_sides < 16.0)
    classes = np.full(QUERY_COUNT, "large", dtype="<U6")
    classes[areas < 96.0**2] = "medium"
    classes[hybrid_small] = "small"
    return classes, hybrid_small


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(tuple(contiguous.shape)).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _query_probe_arrays(
    observations_by_image: Mapping[int, Mapping[int, ProbeObservation]],
    metadata: Mapping[int, transition_render.ImageMetadata],
) -> dict[str, np.ndarray]:
    """Materialize the captured tensors in one portable, deterministic layout."""

    image_ids = tuple(sorted(map(int, observations_by_image)))
    if not image_ids:
        raise ValueError("query probe data requires at least one selected image")
    if any(image_id not in metadata for image_id in image_ids):
        raise ValueError("query probe metadata does not cover every selected image")
    for image_id in image_ids:
        observations = observations_by_image[image_id]
        if tuple(sorted(map(int, observations))) != EPOCHS:
            raise ValueError(f"query probe image {image_id} does not contain e28-e34")
        for epoch in EPOCHS:
            observation = observations[epoch]
            if observation.image_id != image_id or observation.epoch != epoch:
                raise ValueError("query probe observation key disagrees with its identity")

    def stack(name: str, dtype: np.dtype) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack(
                [
                    np.stack(
                        [np.asarray(getattr(observations_by_image[image_id][epoch], name)) for epoch in EPOCHS],
                        axis=0,
                    )
                    for image_id in image_ids
                ],
                axis=0,
            ),
            dtype=dtype,
        )

    strides = np.asarray((8, 16, 32), dtype=np.int16)
    height, width = map(int, full_val_replay.INPUT_SIZE)
    spatial_shapes = np.asarray(
        [(height // int(stride), width // int(stride)) for stride in strides],
        dtype=np.int16,
    )
    arrays = {
        "schema": np.asarray(PROBE_DATA_SCHEMA),
        "image_ids": np.asarray(image_ids, dtype=np.int64),
        "image_sizes_wh": np.asarray(
            [(metadata[image_id].width, metadata[image_id].height) for image_id in image_ids],
            dtype=np.int32,
        ),
        "epochs": np.asarray(EPOCHS, dtype=np.int16),
        "state": np.asarray(STATE),
        "encoder_selected_indices": stack("encoder_dense_indices", np.dtype(np.int64)),
        "encoder_levels": stack("encoder_levels", np.dtype(np.uint8)),
        "encoder_boxes_cxcywh": stack("encoder_boxes_cxcywh", np.dtype(np.float32)),
        "decoder_boxes_xywh": stack("decoder_boxes_xywh", np.dtype(np.float32)),
        "decoder_scores": stack("decoder_scores", np.dtype(np.float32)),
        "decoder_hybrid_small": stack("decoder_hybrid_small", np.dtype(np.bool_)),
        "strides": strides,
        "spatial_shapes": spatial_shapes,
        "input_size": np.asarray(full_val_replay.INPUT_SIZE, dtype=np.int32),
    }
    return arrays


def validate_query_probe_data_archive(
    path: Path, *, expected_image_ids: Sequence[int] | None = None
) -> dict[str, tuple[int, ...]]:
    """Validate the complete persisted probe tensor contract without pickle."""

    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != PROBE_DATA_FIELDS:
            raise ValueError("query probe data archive has an unexpected field order or set")
        arrays = {name: archive[name] for name in PROBE_DATA_FIELDS}
        schema = arrays["schema"]
        state = arrays["state"]
        if schema.shape != () or str(schema.item()) != PROBE_DATA_SCHEMA:
            raise ValueError("query probe data schema mismatch")
        if state.shape != () or str(state.item()) != STATE:
            raise ValueError("query probe data state mismatch")

        image_ids = arrays["image_ids"]
        image_sizes = arrays["image_sizes_wh"]
        epochs = arrays["epochs"]
        if image_ids.dtype != np.dtype(np.int64) or image_ids.ndim != 1 or not len(image_ids):
            raise ValueError("query probe image_ids must be a non-empty int64 vector")
        image_count = int(len(image_ids))
        if len(set(map(int, image_ids))) != image_count or np.any(np.diff(image_ids) <= 0):
            raise ValueError("query probe image_ids must be unique and strictly sorted")
        if expected_image_ids is not None and not np.array_equal(
            image_ids, np.asarray(sorted(map(int, expected_image_ids)), dtype=np.int64)
        ):
            raise ValueError("query probe image_ids disagree with the selected image population")
        if (
            image_sizes.dtype != np.dtype(np.int32)
            or image_sizes.shape != (image_count, 2)
            or np.any(image_sizes <= 0)
        ):
            raise ValueError("query probe image sizes must be positive int32 [image,2]")
        if (
            epochs.dtype != np.dtype(np.int16)
            or epochs.shape != (len(EPOCHS),)
            or not np.array_equal(epochs, np.asarray(EPOCHS, dtype=np.int16))
        ):
            raise ValueError("query probe epoch axis must be int16 e28-e34")

        expected_shapes_and_dtypes = {
            "encoder_selected_indices": (
                (image_count, len(EPOCHS), QUERY_COUNT),
                np.dtype(np.int64),
            ),
            "encoder_levels": (
                (image_count, len(EPOCHS), QUERY_COUNT),
                np.dtype(np.uint8),
            ),
            "encoder_boxes_cxcywh": (
                (image_count, len(EPOCHS), QUERY_COUNT, 4),
                np.dtype(np.float32),
            ),
            "decoder_boxes_xywh": (
                (image_count, len(EPOCHS), QUERY_COUNT, 4),
                np.dtype(np.float32),
            ),
            "decoder_scores": (
                (image_count, len(EPOCHS), QUERY_COUNT),
                np.dtype(np.float32),
            ),
            "decoder_hybrid_small": (
                (image_count, len(EPOCHS), QUERY_COUNT),
                np.dtype(np.bool_),
            ),
        }
        shapes = {}
        for name, (expected_shape, expected_dtype) in expected_shapes_and_dtypes.items():
            array = arrays[name]
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(
                    f"query probe {name} schema mismatch: shape={array.shape}, dtype={array.dtype}"
                )
            shapes[name] = tuple(map(int, array.shape))

        strides = arrays["strides"]
        spatial_shapes = arrays["spatial_shapes"]
        input_size = arrays["input_size"]
        expected_strides = np.asarray((8, 16, 32), dtype=np.int16)
        expected_spatial = np.asarray(((80, 80), (40, 40), (20, 20)), dtype=np.int16)
        if strides.dtype != np.dtype(np.int16) or not np.array_equal(strides, expected_strides):
            raise ValueError("query probe strides must be int16 [8,16,32]")
        if (
            spatial_shapes.dtype != np.dtype(np.int16)
            or not np.array_equal(spatial_shapes, expected_spatial)
        ):
            raise ValueError("query probe spatial_shapes must be int16 P3/P4/P5 shapes")
        if (
            input_size.dtype != np.dtype(np.int32)
            or not np.array_equal(
                input_size, np.asarray(full_val_replay.INPUT_SIZE, dtype=np.int32)
            )
        ):
            raise ValueError("query probe input_size must be int32 [640,640]")

        indices = arrays["encoder_selected_indices"]
        if np.any(indices < 0) or np.any(indices >= int(np.prod(expected_spatial, axis=1).sum())):
            raise ValueError("persisted Encoder index leaves the P3/P4/P5 dense grid")
        if np.any(np.diff(np.sort(indices, axis=-1), axis=-1) == 0):
            raise ValueError("persisted Encoder Top-640 indices are not unique")
        boundaries = np.cumsum(np.prod(expected_spatial, axis=1, dtype=np.int64))
        expected_levels = np.searchsorted(boundaries, indices, side="right").astype(np.uint8)
        if not np.array_equal(arrays["encoder_levels"], expected_levels):
            raise ValueError("persisted Encoder level does not match its dense index")
        for name in ("encoder_boxes_cxcywh", "decoder_boxes_xywh", "decoder_scores"):
            if not np.isfinite(arrays[name]).all():
                raise ValueError(f"query probe {name} contains non-finite values")
        scores = arrays["decoder_scores"]
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("query probe Decoder scores leave [0,1]")
        for image_index, (image_width, image_height) in enumerate(image_sizes):
            for epoch_index in range(len(EPOCHS)):
                _, expected_hybrid = decoder_size_classes(
                    arrays["decoder_boxes_xywh"][image_index, epoch_index],
                    int(image_width),
                    int(image_height),
                )
                if not np.array_equal(
                    arrays["decoder_hybrid_small"][image_index, epoch_index],
                    expected_hybrid,
                ):
                    raise ValueError("persisted Decoder hybrid-small mask is not reproducible")
    return shapes


def write_query_probe_data(
    path: Path,
    observations_by_image: Mapping[int, Mapping[int, ProbeObservation]],
    metadata: Mapping[int, transition_render.ImageMetadata],
) -> None:
    """Atomically write a byte-deterministic, compressed probe tensor archive."""

    arrays = _query_probe_arrays(observations_by_image, metadata)
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"query probe data output already exists: {path}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    validate_query_probe_data_archive(
        temporary, expected_image_ids=tuple(sorted(observations_by_image))
    )
    os.replace(temporary, path)


def _npy_member_header(path: Path, member: str) -> tuple[tuple[int, ...], bool, np.dtype]:
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
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def validate_prediction_archive(
    path: Path, *, epoch: int, formal_images: int
) -> None:
    """Validate the complete compact-prediction schema before selective reads."""

    expected_fields = {
        "schema",
        "image_ids",
        "boxes_xywh",
        "scores",
        "category_id",
        "input_size",
        "state",
        "box_format",
        "query_indices",
        "query_index_note",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError("compact prediction archive has an unexpected field set")
        scalar_expectations = {
            "schema": "sgqs_compact_predictions_v1",
            "state": f"checkpoint{epoch:04d}:{STATE}",
            "box_format": "xywh_abs_original_pixels",
            "query_index_note": "column index; repeated identically for every image",
        }
        for field, expected in scalar_expectations.items():
            value = archive[field]
            if value.shape != () or str(value.item()) != expected:
                raise ValueError(f"compact prediction {field} mismatch for e{epoch}")
        category = archive["category_id"]
        if category.shape != () or category.dtype != np.dtype(np.int64) or int(category) != 0:
            raise ValueError("compact prediction category_id must be int64 scalar 0")
        input_size = archive["input_size"]
        if (
            input_size.dtype != np.dtype(np.int32)
            or input_size.shape != (2,)
            or not np.array_equal(input_size, np.asarray(full_val_replay.INPUT_SIZE, dtype=np.int32))
        ):
            raise ValueError("compact prediction input_size must be int32 [640,640]")
        image_ids = archive["image_ids"]
        query_indices = archive["query_indices"]
        if image_ids.dtype != np.dtype(np.int64) or image_ids.shape != (formal_images,):
            raise ValueError("compact prediction image_ids schema mismatch")
        if len(set(map(int, image_ids))) != formal_images:
            raise ValueError("compact prediction image_ids must be unique")
        if (
            query_indices.dtype != np.dtype(np.uint16)
            or query_indices.shape != (QUERY_COUNT,)
            or not np.array_equal(query_indices, np.arange(QUERY_COUNT, dtype=np.uint16))
        ):
            raise ValueError("compact prediction query_indices must be uint16 0..639")

    for member, expected_shape, expected_dtype in (
        ("boxes_xywh", (formal_images, QUERY_COUNT, 4), np.dtype(np.float32)),
        ("scores", (formal_images, QUERY_COUNT), np.dtype(np.float32)),
    ):
        shape, fortran_order, dtype = _npy_member_header(path, member)
        if shape != expected_shape or fortran_order or dtype != expected_dtype:
            raise ValueError(
                f"compact prediction {member} header mismatch: "
                f"shape={shape}, fortran={fortran_order}, dtype={dtype}"
            )


def reconcile_decoder(
    rerun_boxes: np.ndarray,
    rerun_scores: np.ndarray,
    canonical_boxes: np.ndarray,
    canonical_scores: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    rerun_hybrid_mask: np.ndarray | None = None,
) -> Reconciliation:
    """Require bit-identical final Decoder output and derived ordering/mask."""

    rerun_boxes = np.asarray(rerun_boxes)
    rerun_scores = np.asarray(rerun_scores)
    canonical_boxes = np.asarray(canonical_boxes)
    canonical_scores = np.asarray(canonical_scores)
    if (
        rerun_boxes.shape != (QUERY_COUNT, 4)
        or canonical_boxes.shape != (QUERY_COUNT, 4)
        or rerun_scores.shape != (QUERY_COUNT,)
        or canonical_scores.shape != (QUERY_COUNT,)
    ):
        raise ValueError("Decoder reconciliation requires [640,4] boxes and [640] scores")
    if not all(
        np.isfinite(value).all()
        for value in (rerun_boxes, rerun_scores, canonical_boxes, canonical_scores)
    ):
        raise ValueError("Decoder reconciliation refuses non-finite values")

    box_error = float(np.max(np.abs(rerun_boxes.astype(np.float64) - canonical_boxes)))
    score_error = float(np.max(np.abs(rerun_scores.astype(np.float64) - canonical_scores)))
    boxes_exact = np.array_equal(rerun_boxes, canonical_boxes)
    scores_exact = np.array_equal(rerun_scores, canonical_scores)
    rerun_order = np.argsort(-rerun_scores, kind="stable")
    canonical_order = np.argsort(-canonical_scores, kind="stable")
    rank_order_exact = np.array_equal(rerun_order, canonical_order)
    if rerun_hybrid_mask is None:
        _, rerun_hybrid = decoder_size_classes(rerun_boxes, image_width, image_height)
    else:
        rerun_hybrid = np.asarray(rerun_hybrid_mask, dtype=np.bool_)
        if rerun_hybrid.shape != (QUERY_COUNT,):
            raise ValueError("rerun hybrid mask must contain exactly 640 entries")
    _, canonical_hybrid = decoder_size_classes(canonical_boxes, image_width, image_height)
    hybrid_mask_exact = np.array_equal(rerun_hybrid, canonical_hybrid)
    result = Reconciliation(
        box_max_abs_error=box_error,
        score_max_abs_error=score_error,
        boxes_exact=boxes_exact,
        scores_exact=scores_exact,
        rank_order_exact=rank_order_exact,
        hybrid_mask_exact=hybrid_mask_exact,
    )
    if not all((boxes_exact, scores_exact, rank_order_exact, hybrid_mask_exact)):
        raise ValueError(
            "selective forward disagrees with canonical Decoder replay: "
            f"boxes_exact={boxes_exact}, scores_exact={scores_exact}, "
            f"rank_exact={rank_order_exact}, hybrid_exact={hybrid_mask_exact}, "
            f"box_max_abs={box_error:.9g}, score_max_abs={score_error:.9g}"
        )
    return result


def _decoder_xywh_original(pred_cxcywh, width: int, height: int):
    converted = pred_cxcywh.float().clone()
    converted[:, 0] = (pred_cxcywh[:, 0] - pred_cxcywh[:, 2] / 2) * width
    converted[:, 1] = (pred_cxcywh[:, 1] - pred_cxcywh[:, 3] / 2) * height
    converted[:, 2] = pred_cxcywh[:, 2] * width
    converted[:, 3] = pred_cxcywh[:, 3] * height
    return converted


def capture_selected_images(
    model,
    device: str,
    epoch: int,
    metadata: Mapping[int, transition_render.ImageMetadata],
    images_root: Path,
    canonical: transition_render.PredictionRows,
) -> dict[int, ProbeObservation]:
    """Run batch=1 selective inference and capture true Encoder Top-640 tensors."""

    torch = full_val_replay.torch
    import torchvision

    transform = full_val_replay.v2.Compose(
        [
            full_val_replay.v2.Resize(list(full_val_replay.INPUT_SIZE)),
            full_val_replay.v2.ToImage(),
            full_val_replay.v2.ToDtype(torch.float32, scale=True),
        ]
    )
    strides = tuple(int(value) for value in model.decoder.feat_strides)
    if strides != (8, 16, 32):
        raise ValueError(f"query probe requires P3/P4/P5 strides (8,16,32), got {strides}")
    expected_dense = sum(
        (full_val_replay.INPUT_SIZE[0] // stride)
        * (full_val_replay.INPUT_SIZE[1] // stride)
        for stride in strides
    )
    observations: dict[int, ProbeObservation] = {}
    with full_val_replay.CaptureCandidates(model.decoder) as capture, torch.inference_mode():
        for image_id in sorted(metadata):
            item = metadata[image_id]
            image_path = transition_render._safe_image_path(images_root, item.file_name)
            with Image.open(image_path) as opened:
                source = opened.convert("RGB")
            if source.size != (item.width, item.height):
                raise ValueError(f"source image dimensions disagree for image {image_id}")
            output = model(transform(source).unsqueeze(0).to(device))
            captured = capture.data
            if captured is None or int(captured.get("k", -1)) != QUERY_COUNT:
                raise RuntimeError("Encoder Top-640 capture did not run as expected")
            if (
                captured["boxes"].shape != (1, expected_dense, 4)
                or captured["selection_scores"].shape != (1, expected_dense)
                or captured["selected_boxes"].shape != (1, QUERY_COUNT, 4)
                or captured["selected_scores"].shape != (1, QUERY_COUNT)
                or captured["selected_indices"].shape != (1, QUERY_COUNT)
            ):
                raise ValueError("captured Encoder tensors disagree with the 640-input P3/P4/P5 grid")
            if not full_val_replay.torch.isfinite(captured["selected_scores"]).all():
                raise ValueError("selected Encoder logits must be finite")
            pred = output["pred_boxes"]
            logits = output["pred_logits"]
            if pred.shape != (1, QUERY_COUNT, 4) or logits.shape != (1, QUERY_COUNT, 1):
                raise ValueError("selective forward changed the fixed 640-query output shape")
            rerun_boxes = (
                _decoder_xywh_original(pred[0], item.width, item.height)
                .detach()
                .cpu()
                .numpy()
            )
            rerun_classes, rerun_hybrid = decoder_size_classes_normalized(
                pred[0].float().detach().cpu().numpy()
            )
            rerun_scores = logits[0, :, 0].sigmoid().float().detach().cpu().numpy()
            canonical_boxes = np.asarray(canonical.boxes[image_id])
            canonical_scores = np.asarray(canonical.scores[image_id])
            reconciliation = reconcile_decoder(
                rerun_boxes,
                rerun_scores,
                canonical_boxes,
                canonical_scores,
                item.width,
                item.height,
                rerun_hybrid_mask=rerun_hybrid,
            )

            selected_indices = (
                captured["selected_indices"][0].detach().cpu().numpy().astype(np.int64, copy=False)
            )
            selected_boxes = (
                captured["selected_boxes"][0].float().detach().cpu().numpy()
            )
            if selected_boxes.shape != (QUERY_COUNT, 4) or not np.isfinite(selected_boxes).all():
                raise ValueError("captured Encoder selected boxes are not finite [640,4]")
            levels, counts = dense_source_levels(selected_indices, strides)
            canonical_classes, canonical_hybrid = decoder_size_classes(
                canonical_boxes, item.width, item.height
            )
            if not np.array_equal(rerun_classes, canonical_classes):
                raise ValueError("direct 640-input Decoder size classes disagree with canonical replay")
            observations[image_id] = ProbeObservation(
                epoch=epoch,
                image_id=image_id,
                encoder_boxes_cxcywh=selected_boxes.copy(),
                encoder_dense_indices=selected_indices.copy(),
                encoder_levels=levels,
                level_counts=counts,
                decoder_boxes_xywh=canonical_boxes.copy(),
                decoder_scores=canonical_scores.copy(),
                decoder_classes=rerun_classes,
                decoder_hybrid_small=canonical_hybrid,
                reconciliation=reconciliation,
                prediction_logical_name=canonical.logical_name,
                prediction_sha256=canonical.sha256,
            )
    return observations


def _selection_rows_by_image(
    selections: Sequence[Mapping[str, object]],
) -> dict[int, list[Mapping[str, object]]]:
    result: dict[int, list[Mapping[str, object]]] = {}
    for row in selections:
        result.setdefault(int(row["image_id"]), []).append(row)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                str(row["track"]),
                int(row["epoch_start"]),
                str(row["cause"]),
                str(row["domain"]),
                int(row["annotation_id"]),
            )
        )
    return result


def _target_boxes(
    item: transition_render.ImageMetadata,
    selection_rows: Sequence[Mapping[str, object]],
) -> list[np.ndarray]:
    lookup = {
        int(annotation_id): box
        for annotation_id, box in zip(item.annotation_ids, item.boxes_xywh)
    }
    result = []
    for annotation_id in sorted({int(row["annotation_id"]) for row in selection_rows}):
        if annotation_id not in lookup:
            raise ValueError(f"selected annotation {annotation_id} is absent from image metadata")
        result.append(np.asarray(lookup[annotation_id], dtype=np.float64))
    return result


def _scaled_canvas(source: Image.Image) -> tuple[Image.Image, float, int, int]:
    content_width, content_height = 420, 238
    scale = min(content_width / source.width, content_height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    ).convert("RGBA")
    canvas = Image.new("RGBA", (content_width, content_height), (23, 27, 34, 255))
    offset_x = (content_width - resized.width) // 2
    offset_y = (content_height - resized.height) // 2
    canvas.alpha_composite(resized, (offset_x, offset_y))
    return canvas, scale, offset_x, offset_y


def _draw_target_boxes(
    draw: ImageDraw.ImageDraw,
    boxes: Sequence[np.ndarray],
    scale: float,
    offset_x: int,
    offset_y: int,
) -> None:
    for box in boxes:
        x, y, width, height = map(float, box)
        draw.rectangle(
            (
                offset_x + x * scale,
                offset_y + y * scale,
                offset_x + (x + width) * scale,
                offset_y + (y + height) * scale,
            ),
            outline=(255, 40, 190, 255),
            width=3,
        )


def render_probe_panel(
    source: Image.Image,
    item: transition_render.ImageMetadata,
    observation: ProbeObservation,
    selected_gt: Sequence[np.ndarray],
    kind: str,
) -> Image.Image:
    panel = Image.new("RGB", (420, 274), (23, 27, 34))
    canvas, scale, offset_x, offset_y = _scaled_canvas(source)
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if kind == "encoder":
        for box, level in zip(
            observation.encoder_boxes_cxcywh, observation.encoder_levels
        ):
            cx = offset_x + float(box[0]) * item.width * scale
            cy = offset_y + float(box[1]) * item.height * scale
            color = SOURCE_LEVEL_COLORS[int(level)]
            draw.ellipse((cx - 1.5, cy - 1.5, cx + 1.5, cy + 1.5), fill=color)
        title = (
            f"Encoder Top-640 source  P3/P4/P5="
            f"{observation.level_counts[0]}/{observation.level_counts[1]}/"
            f"{observation.level_counts[2]}"
        )
    else:
        indexes = (
            np.flatnonzero(observation.decoder_hybrid_small)
            if kind == "hybrid"
            else np.arange(QUERY_COUNT)
        )
        order = indexes[np.argsort(observation.decoder_scores[indexes], kind="stable")]
        for query in order:
            x, y, width, height = map(float, observation.decoder_boxes_xywh[query])
            if width <= 0.0 or height <= 0.0:
                continue
            color = DECODER_CLASS_COLORS[str(observation.decoder_classes[query])]
            score = float(observation.decoder_scores[query])
            alpha = int(35 + 190 * max(0.0, min(1.0, score)))
            draw.rectangle(
                (
                    offset_x + x * scale,
                    offset_y + y * scale,
                    offset_x + (x + width) * scale,
                    offset_y + (y + height) * scale,
                ),
                outline=(*color, alpha),
                width=2 if kind == "hybrid" else 1,
            )
        title = (
            f"Decoder local-hybrid-small  n={len(indexes)}"
            if kind == "hybrid"
            else "Decoder all 640 (local size colors)"
        )

    _draw_target_boxes(draw, selected_gt, scale, offset_x, offset_y)
    canvas = Image.alpha_composite(canvas, overlay).convert("RGB")
    panel.paste(canvas, (0, 36))
    header = ImageDraw.Draw(panel)
    header.text(
        (8, 9),
        title,
        font=transition_render._font(12, True),
        fill=(235, 239, 246),
    )
    return panel


def render_image_trajectory(
    output_path: Path,
    item: transition_render.ImageMetadata,
    selection_rows: Sequence[Mapping[str, object]],
    observations: Mapping[int, ProbeObservation],
    images_root: Path,
) -> None:
    if tuple(sorted(observations)) != EPOCHS:
        raise ValueError("trajectory requires all EMA checkpoints e28-e34")
    source_path = transition_render._safe_image_path(images_root, item.file_name)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    if source.size != (item.width, item.height):
        raise ValueError("trajectory source dimensions disagree with metadata")
    gt_boxes = _target_boxes(item, selection_rows)

    label_width, panel_width, panel_height = 88, 420, 274
    header_height = 104
    sheet = Image.new(
        "RGB",
        (label_width + 3 * panel_width, header_height + len(EPOCHS) * panel_height),
        (248, 249, 252),
    )
    draw = ImageDraw.Draw(sheet)
    tracks = ", ".join(sorted({str(row["track"]) for row in selection_rows}))
    strata = "; ".join(
        sorted(
            {
                f"e{row['epoch_start']}->{row['epoch_end']} {row['transition']} "
                f"{row['cause']}/{row['domain']}"
                for row in selection_rows
            }
        )
    )
    draw.text(
        (16, 10),
        f"EMA checkpoint query probe | image {item.image_id} | tracks: {tracks}",
        font=transition_render._font(19, True),
        fill=(25, 31, 42),
    )
    draw.text(
        (16, 39),
        f"Selected formal strata ({len(selection_rows)}): {strata[:180]}",
        font=transition_render._font(11),
        fill=(65, 75, 90),
    )
    draw.text(
        (16, 61),
        "Magenta=selected GT; Encoder colors are actual source levels, not SGQS scale preference.",
        font=transition_render._font(11, True),
        fill=(85, 52, 91),
    )
    draw.text(
        (16, 81),
        "Decoder red=local hybrid-small; yellow=medium; cyan=large. Qualitative illustration only.",
        font=transition_render._font(11),
        fill=(65, 75, 90),
    )

    for row_index, epoch in enumerate(EPOCHS):
        y = header_height + row_index * panel_height
        draw.rectangle((0, y, sheet.width - 1, y + panel_height - 1), outline=(210, 215, 223))
        draw.text(
            (19, y + panel_height // 2 - 10),
            f"e{epoch}",
            font=transition_render._font(16, True),
            fill=(36, 46, 61),
        )
        observation = observations[epoch]
        for column, kind in enumerate(("encoder", "decoder", "hybrid")):
            panel = render_probe_panel(source, item, observation, gt_boxes, kind)
            sheet.paste(panel, (label_width + column * panel_width, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    sheet.save(temporary, format="PNG", compress_level=9, optimize=False)
    os.replace(temporary, output_path)


def _write_csv(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fields), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _stable_file_identity(path: Path) -> tuple[int, str]:
    before = path.stat()
    digest = formal_plot.sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {path.name}")
    return int(after.st_size), digest


def _json_sha256(value: Mapping) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _current_runtime_backend(device: str, torch) -> dict[str, object]:
    """Mirror the replay-matrix backend probe in the actual forward process."""

    result: dict[str, object] = {
        "requested_device": device,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
    }
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("canonical CUDA device is unavailable")
        requested = torch.device(device)
        index = torch.cuda.current_device() if requested.index is None else requested.index
        properties = torch.cuda.get_device_properties(index)
        result.update(
            backend="cuda",
            device_index=int(index),
            device_name=str(properties.name),
            compute_capability=list(torch.cuda.get_device_capability(index)),
            total_memory_bytes=int(properties.total_memory),
        )
        try:
            driver = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).splitlines()[index].strip()
        except Exception:
            driver = None
        result["cuda_driver_version"] = driver
    else:
        result.update(
            backend="cpu",
            device_index=None,
            device_name=platform.processor() or platform.machine(),
            compute_capability=None,
            total_memory_bytes=None,
            cuda_driver_version=None,
        )
    return result


def _validate_replay_execution_identity(
    audit: Mapping, replay_root: Path, device: str, torch
) -> dict[str, object]:
    """Bind the selective hook and runtime to the replay matrix used by matching."""

    input_identity = audit.get("input_identity")
    if not isinstance(input_identity, Mapping):
        raise ValueError("formal audit lacks input identity")
    matcher_matrix = input_identity.get("matcher_matrix")
    if not isinstance(matcher_matrix, Mapping):
        raise ValueError("formal audit lacks the matcher-to-replay matrix identity")
    expected_matrix_sha = _require_sha256(
        matcher_matrix.get("replay_matrix_complete_sha256"),
        "formal replay matrix",
    )
    matrix_path = replay_root / "matrix.complete.json"
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    initial_stat = matrix_path.stat()
    matrix_bytes, matrix_sha = _stable_file_identity(matrix_path)
    if matrix_sha != expected_matrix_sha:
        raise ValueError("replay matrix SHA-256 differs from the matrix bound by formal matching")
    try:
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("replay matrix complete marker is not valid JSON") from exc
    final_stat = matrix_path.stat()
    if (initial_stat.st_size, initial_stat.st_mtime_ns) != (
        final_stat.st_size,
        final_stat.st_mtime_ns,
    ):
        raise ValueError("replay matrix changed while validating the selective probe")
    if not isinstance(matrix, dict):
        raise ValueError("replay matrix complete marker must be a JSON object")
    if matrix.get("schema") != REPLAY_MATRIX_SCHEMA or matrix.get("status") != "complete":
        raise ValueError("replay matrix is not a completed formal orchestration")
    if matrix.get("epochs") != list(EPOCHS) or matrix.get("states") != ["raw", "ema"]:
        raise ValueError("replay matrix does not describe the formal e28-e34 raw/EMA plan")

    formal_identity = audit.get("formal_identity")
    if not isinstance(formal_identity, Mapping):
        raise ValueError("formal audit lacks inference identity")
    if matrix.get("config_sha256") != formal_identity.get("config_sha256"):
        raise ValueError("replay matrix config identity differs from formal analysis")
    if matrix.get("annotation_sha256") != formal_identity.get("annotation_sha256"):
        raise ValueError("replay matrix annotation identity differs from formal analysis")
    if matrix.get("device") != device or int(matrix.get("batch_size", -1)) != 1:
        raise ValueError("selective probe requires the formal device with batch_size=1")

    replay_script_path = Path(full_val_replay.__file__).resolve()
    supervision_path = Path(full_val_replay.supervision_probe.__file__).resolve()
    _replay_bytes, replay_script_sha = _stable_file_identity(replay_script_path)
    _supervision_bytes, supervision_sha = _stable_file_identity(supervision_path)
    if replay_script_sha != _require_sha256(
        matrix.get("replay_script_sha256"), "replay script"
    ):
        raise ValueError("current full_val_replay.py differs from the formal replay implementation")
    if supervision_sha != _require_sha256(
        matrix.get("supervision_probe_sha256"), "supervision probe"
    ):
        raise ValueError("current supervision_probe.py differs from the formal replay implementation")

    canonical_engine = matrix.get("engine_python_tree")
    current_engine = replay_matrix.engine_tree_identity()
    if not isinstance(canonical_engine, dict) or canonical_engine != current_engine:
        raise ValueError("current engine Python tree differs from the formal replay implementation")

    checkpoint_contract = matrix.get("checkpoint_contract")
    replay_inputs = input_identity.get("replays")
    if not isinstance(checkpoint_contract, dict) or not isinstance(replay_inputs, Mapping):
        raise ValueError("replay matrix checkpoint identity is incomplete")
    expected_checkpoints = {f"checkpoint{epoch:04d}.pth" for epoch in EPOCHS}
    if set(checkpoint_contract) != expected_checkpoints:
        raise ValueError("replay matrix checkpoint population differs from e28-e34")
    for epoch in EPOCHS:
        name = f"checkpoint{epoch:04d}.pth"
        contract = checkpoint_contract[name]
        replay_input = replay_inputs.get(f"e{epoch}")
        if not isinstance(contract, Mapping) or not isinstance(replay_input, Mapping):
            raise ValueError(f"replay matrix checkpoint identity is incomplete for e{epoch}")
        if (
            contract.get("file_sha256") != replay_input.get("checkpoint_sha256")
            or contract.get("verification") != "recomputed_equal"
        ):
            raise ValueError(f"replay matrix checkpoint contract differs from formal e{epoch}")

    canonical_runtime = matrix.get("runtime_backend")
    current_runtime = _current_runtime_backend(device, torch)
    if not isinstance(canonical_runtime, dict) or canonical_runtime != current_runtime:
        raise ValueError(
            "current runtime backend differs from formal replay "
            "(device/Python/Torch/CUDA/threads/OMP/MKL must match exactly)"
        )
    inference = formal_identity.get("inference_identity")
    if not isinstance(inference, Mapping):
        raise ValueError("formal audit inference identity is incomplete")
    for inference_key, runtime_key in (
        ("device", "requested_device"),
        ("torch_version", "torch_version"),
        ("torch_cuda_version", "torch_cuda_version"),
    ):
        if inference.get(inference_key) != current_runtime.get(runtime_key):
            raise ValueError(f"formal inference identity differs for {inference_key}")

    runtime_json = json.dumps(
        current_runtime,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return {
        "matrix_file": matrix_path.name,
        "matrix_bytes": matrix_bytes,
        "matrix_sha256": matrix_sha,
        "replay_script_sha256": replay_script_sha,
        "supervision_probe_sha256": supervision_sha,
        "engine_python_tree_sha256": str(current_engine["sha256"]),
        "runtime_backend": current_runtime,
        "runtime_backend_json": runtime_json,
        "runtime_backend_sha256": _json_sha256(current_runtime),
    }


def _is_absolute_or_escaping(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or normalized.startswith("/")
        or ".." in normalized.split("/")
    )


def validate_probe_manifest_rows(
    rows: Sequence[Mapping[str, object]], *, probe_data_path: Path | None = None
) -> None:
    """Refuse incomplete provenance and host-specific paths before publication."""

    if not rows:
        raise ValueError("query probe manifest cannot be empty")
    shared_fields = (
        "probe_data_file",
        "probe_data_bytes",
        "probe_data_sha256",
        "probe_data_schema",
        "analysis_audit_sha256",
        "analysis_artifact_manifest_sha256",
        "representatives_sha256",
        "selection_sha256",
        "probe_script_sha256",
        "full_val_replay_sha256",
        "analyze_full_val_sha256",
        "plot_full_val_sha256",
        "transition_renderer_sha256",
        "replay_matrix_file",
        "replay_matrix_bytes",
        "replay_matrix_sha256",
        "supervision_probe_sha256",
        "engine_python_tree_sha256",
        "runtime_backend_sha256",
        "runtime_backend_json",
    )
    shared_values = {field: set() for field in shared_fields}
    row_keys: list[tuple[int, int]] = []
    for row in rows:
        if set(row) != set(MANIFEST_FIELDS):
            raise ValueError("query probe manifest row does not match the canonical schema")
        for field in MANIFEST_FIELDS:
            if field.endswith("_sha256"):
                _require_sha256(row[field], f"manifest {field}")
        for field in shared_fields:
            value = str(row[field]).strip()
            if not value:
                raise ValueError(f"query probe manifest provenance is empty: {field}")
            shared_values[field].add(value)
        for field in MANIFEST_LOGICAL_PATH_FIELDS:
            value = str(row[field]).strip()
            if not value or _is_absolute_or_escaping(value):
                raise ValueError(f"query probe manifest path is not portable: {field}")
        if str(row["probe_data_schema"]) != PROBE_DATA_SCHEMA:
            raise ValueError("query probe manifest data schema mismatch")
        runtime = json.loads(str(row["runtime_backend_json"]))
        if not isinstance(runtime, dict) or _json_sha256(runtime) != row["runtime_backend_sha256"]:
            raise ValueError("query probe manifest runtime identity hash mismatch")
        for field in ("figure_bytes", "source_image_bytes", "checkpoint_bytes", "probe_data_bytes", "replay_matrix_bytes"):
            if int(row[field]) <= 0:
                raise ValueError(f"query probe manifest has invalid byte count: {field}")
        image_id = int(row["image_id"])
        epoch = int(row["epoch"])
        if str(row["state"]) != STATE or epoch not in EPOCHS:
            raise ValueError("query probe manifest must contain EMA e28-e34 only")
        if str(row["checkpoint_file"]) != f"checkpoint{epoch:04d}.pth":
            raise ValueError("query probe manifest checkpoint filename disagrees with epoch")
        if str(row["checkpoint_state"]) != "ema.module":
            raise ValueError("query probe manifest checkpoint state is not ema.module")
        level_counts = tuple(
            int(row[field])
            for field in ("encoder_p3_count", "encoder_p4_count", "encoder_p5_count")
        )
        if (
            int(row["encoder_selected_count"]) != QUERY_COUNT
            or any(count < 0 for count in level_counts)
            or sum(level_counts) != QUERY_COUNT
            or int(row["decoder_query_count"]) != QUERY_COUNT
        ):
            raise ValueError("query probe manifest P3/P4/P5 or Decoder counts do not partition 640")
        if any(
            int(row[field]) != 1
            for field in (
                "decoder_boxes_exact",
                "decoder_scores_exact",
                "decoder_rank_order_exact",
                "decoder_hybrid_mask_exact",
            )
        ) or any(
            float(row[field]) != 0.0
            for field in ("decoder_box_max_abs_error", "decoder_score_max_abs_error")
        ):
            raise ValueError("query probe manifest contains a non-exact Decoder reconciliation")
        row_keys.append((image_id, epoch))
    if any(len(values) != 1 for values in shared_values.values()):
        raise ValueError("query probe manifest shared provenance changes between rows")
    image_ids = tuple(sorted({image_id for image_id, _epoch in row_keys}))
    expected_keys = {(image_id, epoch) for image_id in image_ids for epoch in EPOCHS}
    if len(row_keys) != len(expected_keys) or set(row_keys) != expected_keys:
        raise ValueError("query probe manifest lacks a unique image × e28-e34 grid")

    if probe_data_path is None:
        return
    data_identity = _stable_file_identity(probe_data_path)
    if probe_data_path.name != str(rows[0]["probe_data_file"]):
        raise ValueError("query probe manifest names a different tensor archive")
    if data_identity != (
        int(rows[0]["probe_data_bytes"]),
        str(rows[0]["probe_data_sha256"]),
    ):
        raise ValueError("query probe tensor archive bytes/SHA disagree with manifest")
    validate_query_probe_data_archive(probe_data_path, expected_image_ids=image_ids)
    with np.load(probe_data_path, allow_pickle=False) as archive:
        archive_image_ids = tuple(map(int, archive["image_ids"]))
        archive_epochs = tuple(map(int, archive["epochs"]))
        image_indexes = {image_id: index for index, image_id in enumerate(archive_image_ids)}
        epoch_indexes = {epoch: index for index, epoch in enumerate(archive_epochs)}
        encoder_indices = archive["encoder_selected_indices"]
        encoder_levels = archive["encoder_levels"]
        encoder_boxes = archive["encoder_boxes_cxcywh"]
        decoder_masks = archive["decoder_hybrid_small"]
        for row in rows:
            image_index = image_indexes[int(row["image_id"])]
            epoch_index = epoch_indexes[int(row["epoch"])]
            levels = encoder_levels[image_index, epoch_index]
            observed_level_counts = tuple(
                int(np.count_nonzero(levels == level)) for level in range(3)
            )
            expected_level_counts = tuple(
                int(row[field])
                for field in ("encoder_p3_count", "encoder_p4_count", "encoder_p5_count")
            )
            if observed_level_counts != expected_level_counts:
                raise ValueError("query probe manifest P3/P4/P5 counts disagree with NPZ")
            mask = decoder_masks[image_index, epoch_index]
            if int(np.count_nonzero(mask)) != int(row["decoder_hybrid_small_count"]):
                raise ValueError("query probe manifest hybrid-small count disagrees with NPZ")
            for field, array in (
                ("encoder_selected_indices_sha256", encoder_indices[image_index, epoch_index]),
                ("encoder_selected_boxes_sha256", encoder_boxes[image_index, epoch_index]),
                ("decoder_hybrid_mask_sha256", mask),
            ):
                if _array_sha256(array) != str(row[field]):
                    raise ValueError(f"query probe manifest {field} disagrees with NPZ")
    if _stable_file_identity(probe_data_path) != data_identity:
        raise ValueError("query probe tensor archive changed during manifest reconciliation")


def _validate_runtime(audit: Mapping, torch) -> str:
    inference = audit["formal_identity"]["inference_identity"]
    if int(inference["batch_size"]) != 1:
        raise ValueError(
            "bit-identical selective probe requires canonical replay batch_size=1"
        )
    if str(inference["torch_version"]) != str(torch.__version__):
        raise ValueError("current Torch version differs from canonical replay")
    if inference.get("torch_cuda_version") != torch.version.cuda:
        raise ValueError("current Torch CUDA identity differs from canonical replay")
    device = str(inference["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("canonical CUDA device is unavailable")
    return device


def _validate_input_hashes(
    audit: Mapping,
    config: Path,
    annotation: Path,
    checkpoint_root: Path,
) -> dict[int, tuple[Path, int, int, str]]:
    formal_identity = audit["formal_identity"]
    _config_bytes, config_sha = _stable_file_identity(config)
    _annotation_bytes, annotation_sha = _stable_file_identity(annotation)
    if config_sha != str(formal_identity["config_sha256"]):
        raise ValueError("config SHA-256 differs from formal replay")
    if annotation_sha != str(formal_identity["annotation_sha256"]):
        raise ValueError("annotation SHA-256 differs from formal replay")
    root = checkpoint_root.resolve()
    result = {}
    for epoch in EPOCHS:
        path = (root / f"checkpoint{epoch:04d}.pth").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("checkpoint path escapes checkpoint root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        byte_count, observed = _stable_file_identity(path)
        expected = str(
            audit["input_identity"]["replays"][f"e{epoch}"]["checkpoint_sha256"]
        ).lower()
        if observed != expected:
            raise ValueError(f"checkpoint SHA-256 mismatch for e{epoch}")
        result[epoch] = (path, byte_count, int(path.stat().st_mtime_ns), observed)
    return result


def _build_model(config_path: Path, device: str):
    config = full_val_replay.YAMLConfig(str(config_path))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    if config.yaml_cfg.get("eval_spatial_size") != list(full_val_replay.INPUT_SIZE):
        raise ValueError("query probe requires eval_spatial_size=[640,640]")
    model = config.model.to(device).eval()
    if model.decoder.num_queries != QUERY_COUNT or model.decoder.num_classes != 1:
        raise ValueError("query probe requires the single-class 640-query model")
    if tuple(map(int, model.decoder.feat_strides)) != (8, 16, 32):
        raise ValueError("query probe requires the fixed P3/P4/P5 strides (8,16,32)")
    if str(model.decoder.query_select_method) != "default":
        raise ValueError("query probe requires the trained default query selector")
    return model


def run(args: argparse.Namespace) -> list[Path]:
    import torchvision

    analysis_data = args.analysis_data.resolve()
    replay_root = args.replay_root.resolve()
    annotation = args.annotation.resolve()
    images_root = args.images_root.resolve()
    config_path = args.config.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"query probe output already exists: {output}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"stale query probe staging directory exists: {staging}")

    source_paths = {
        "probe": Path(__file__).resolve(),
        "full_val_replay": Path(full_val_replay.__file__).resolve(),
        "analyze_full_val": Path(formal_analysis.__file__).resolve(),
        "plot_full_val": Path(formal_plot.__file__).resolve(),
        "transition_renderer": Path(transition_render.__file__).resolve(),
    }
    code_source_identities = {
        name: _stable_file_identity(path) for name, path in source_paths.items()
    }
    representatives_path = analysis_data / "representative_annotations.csv"
    formal_input_paths = {
        "analysis_audit": analysis_data / "audit.json",
        "analysis_artifact_manifest": analysis_data / "artifact_manifest.csv",
        "representatives": representatives_path,
    }
    formal_input_identities = {
        name: _stable_file_identity(path) for name, path in formal_input_paths.items()
    }

    audit = formal_plot.validate_formal_analysis(analysis_data)
    representatives = read_formal_representatives(representatives_path)
    selections = select_probe_rows(representatives)
    rows_by_image = _selection_rows_by_image(selections)
    wanted_image_ids = set(rows_by_image)

    metadata_descriptor = transition_render.require_shared_metadata_identity(audit, EPOCHS)
    metadata_path, metadata_logical = transition_render._artifact_path(
        replay_root, metadata_descriptor, EPOCHS[0]
    )
    metadata_sha = transition_render._validate_descriptor_file(
        metadata_path, metadata_descriptor, "shared replay metadata"
    )
    metadata = transition_render.load_selected_metadata(metadata_path, audit, selections)
    for row in selections:
        item = metadata[int(row["image_id"])]
        if int(row["image_gt_count"]) != len(item.annotation_ids):
            raise ValueError("representative image_gt_count disagrees with replay metadata")
    image_source_identities = {
        image_id: _stable_file_identity(
            transition_render._safe_image_path(images_root, item.file_name)
        )
        for image_id, item in metadata.items()
    }

    canonical_predictions: dict[int, transition_render.PredictionRows] = {}
    for epoch in EPOCHS:
        descriptor = transition_render._descriptor(audit, epoch, "predictions", STATE)
        path, logical = transition_render._artifact_path(
            replay_root, descriptor, epoch, state=STATE
        )
        # load_prediction_rows checks whole-file bytes/SHA before its selective
        # row stream.  The stricter full schema check then closes metadata/dtype
        # fields that the general representative renderer does not consume.
        canonical_predictions[epoch] = transition_render.load_prediction_rows(
            path, descriptor, logical, epoch, audit, wanted_image_ids
        )
        validate_prediction_archive(path, epoch=epoch, formal_images=int(audit["images"]))
        for image_id in wanted_image_ids:
            boxes = canonical_predictions[epoch].boxes[image_id]
            scores = canonical_predictions[epoch].scores[image_id]
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise ValueError(f"canonical selected row is non-finite: e{epoch}/image {image_id}")
            if np.any(scores < 0.0) or np.any(scores > 1.0):
                raise ValueError(f"canonical selected scores leave [0,1]: e{epoch}/image {image_id}")

    torch = full_val_replay.torch
    device = _validate_runtime(audit, torch)
    execution_identity = _validate_replay_execution_identity(
        audit, replay_root, device, torch
    )
    checkpoint_files = _validate_input_hashes(
        audit, config_path, annotation, checkpoint_root
    )
    model = _build_model(config_path, device)

    observations_by_image: dict[int, dict[int, ProbeObservation]] = {
        image_id: {} for image_id in sorted(wanted_image_ids)
    }
    for epoch in EPOCHS:
        checkpoint_path, _checkpoint_bytes, checkpoint_mtime, _checkpoint_sha = checkpoint_files[epoch]
        checkpoint_stat = checkpoint_path.stat()
        if (
            checkpoint_stat.st_size != _checkpoint_bytes
            or checkpoint_stat.st_mtime_ns != checkpoint_mtime
        ):
            raise ValueError(f"checkpoint changed before loading e{epoch}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_stat_after = checkpoint_path.stat()
        if (
            checkpoint_stat_after.st_size != _checkpoint_bytes
            or checkpoint_stat_after.st_mtime_ns != checkpoint_mtime
        ):
            raise ValueError(f"checkpoint changed while loading e{epoch}")
        if not isinstance(checkpoint, dict):
            raise TypeError(f"checkpoint e{epoch} is not a mapping")
        if int(checkpoint.get("last_epoch", -1)) != epoch:
            raise ValueError(f"checkpoint last_epoch mismatch for e{epoch}")
        weights = full_val_replay.checkpoint_states(checkpoint, [STATE])
        model.load_state_dict(weights[STATE], strict=True)
        del weights, checkpoint
        captured = capture_selected_images(
            model,
            device,
            epoch,
            metadata,
            images_root,
            canonical_predictions[epoch],
        )
        for image_id, observation in captured.items():
            observations_by_image[image_id][epoch] = observation
        gc.collect()

    if _validate_replay_execution_identity(audit, replay_root, device, torch) != execution_identity:
        raise ValueError("formal replay execution identity changed during selective forward")
    for name, path in formal_input_paths.items():
        if _stable_file_identity(path) != formal_input_identities[name]:
            raise ValueError(f"{name} changed during selective forward")

    staging.mkdir(parents=True, exist_ok=False)
    selection_path = staging / "query_probe_selection.csv"
    _write_csv(selection_path, selections, SELECTION_FIELDS)
    selection_sha = formal_plot.sha256(selection_path)
    probe_data_path = staging / "query_probe_data.npz"
    write_query_probe_data(probe_data_path, observations_by_image, metadata)
    probe_data_bytes, probe_data_sha = _stable_file_identity(probe_data_path)
    audit_sha = formal_input_identities["analysis_audit"][1]
    analysis_artifact_manifest_sha = formal_input_identities[
        "analysis_artifact_manifest"
    ][1]
    representatives_sha = formal_input_identities["representatives"][1]
    for name, path in source_paths.items():
        if _stable_file_identity(path) != code_source_identities[name]:
            raise ValueError(f"{name} source changed during selective forward")
    script_sha = code_source_identities["probe"][1]
    replay_script_sha = code_source_identities["full_val_replay"][1]
    if replay_script_sha != execution_identity["replay_script_sha256"]:
        raise ValueError("full_val_replay.py changed during selective forward")
    analysis_script_sha = code_source_identities["analyze_full_val"][1]
    plot_script_sha = code_source_identities["plot_full_val"][1]
    transition_script_sha = code_source_identities["transition_renderer"][1]
    _config_bytes, config_sha = _stable_file_identity(config_path)
    _annotation_bytes, annotation_sha = _stable_file_identity(annotation)
    if config_sha != str(audit["formal_identity"]["config_sha256"]):
        raise ValueError("config changed during selective forward")
    if annotation_sha != str(audit["formal_identity"]["annotation_sha256"]):
        raise ValueError("annotation changed during selective forward")

    manifest_rows: list[dict[str, object]] = []
    figures: list[Path] = []
    for image_id in sorted(observations_by_image):
        item = metadata[image_id]
        filename = f"query_probe_image_{image_id}_ema_e28_e34.png"
        figure = staging / filename
        render_image_trajectory(
            figure,
            item,
            rows_by_image[image_id],
            observations_by_image[image_id],
            images_root,
        )
        source_path = transition_render._safe_image_path(images_root, item.file_name)
        source_bytes, source_sha = _stable_file_identity(source_path)
        if (source_bytes, source_sha) != image_source_identities[image_id]:
            raise ValueError(f"source image changed during selective forward: {item.file_name}")
        selected_tracks = ";".join(
            sorted({str(row["track"]) for row in rows_by_image[image_id]})
        )
        selected_strata = ";".join(
            sorted(
                {
                    f"{row['track']}:e{row['epoch_start']}-e{row['epoch_end']}:"
                    f"{row['transition']}:"
                    f"{row['cause']}:{row['domain']}"
                    for row in rows_by_image[image_id]
                }
            )
        )
        selected_annotations = ";".join(
            map(
                str,
                sorted({int(row["annotation_id"]) for row in rows_by_image[image_id]}),
            )
        )
        figure_sha = formal_plot.sha256(figure)
        for epoch in EPOCHS:
            observation = observations_by_image[image_id][epoch]
            checkpoint_path, checkpoint_bytes, _checkpoint_mtime, checkpoint_sha = checkpoint_files[epoch]
            recon = observation.reconciliation
            manifest_rows.append(
                {
                    "figure": filename,
                    "figure_bytes": figure.stat().st_size,
                    "figure_sha256": figure_sha,
                    "state": STATE,
                    "epoch": epoch,
                    "image_id": image_id,
                    "file_name": item.file_name,
                    "source_image_bytes": source_bytes,
                    "source_image_sha256": source_sha,
                    "selected_tracks": selected_tracks,
                    "selected_strata": selected_strata,
                    "selected_annotation_ids": selected_annotations,
                    "checkpoint_file": checkpoint_path.name,
                    "checkpoint_bytes": checkpoint_bytes,
                    "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_state": "ema.module",
                    "config_sha256": config_sha,
                    "annotation_sha256": annotation_sha,
                    "metadata_logical_name": metadata_logical,
                    "metadata_sha256": metadata_sha,
                    "prediction_logical_name": observation.prediction_logical_name,
                    "prediction_sha256": observation.prediction_sha256,
                    "probe_data_file": probe_data_path.name,
                    "probe_data_bytes": probe_data_bytes,
                    "probe_data_sha256": probe_data_sha,
                    "probe_data_schema": PROBE_DATA_SCHEMA,
                    "encoder_selected_count": QUERY_COUNT,
                    "encoder_feat_strides": "8;16;32",
                    "encoder_spatial_shapes": "80x80;40x40;20x20",
                    "encoder_p3_count": observation.level_counts[0],
                    "encoder_p4_count": observation.level_counts[1],
                    "encoder_p5_count": observation.level_counts[2],
                    "encoder_selected_indices_sha256": _array_sha256(
                        observation.encoder_dense_indices
                    ),
                    "encoder_selected_boxes_sha256": _array_sha256(
                        observation.encoder_boxes_cxcywh
                    ),
                    "decoder_query_count": QUERY_COUNT,
                    "decoder_hybrid_small_count": int(
                        np.count_nonzero(observation.decoder_hybrid_small)
                    ),
                    "decoder_hybrid_mask_sha256": _array_sha256(
                        observation.decoder_hybrid_small
                    ),
                    "decoder_boxes_exact": int(recon.boxes_exact),
                    "decoder_scores_exact": int(recon.scores_exact),
                    "decoder_rank_order_exact": int(recon.rank_order_exact),
                    "decoder_hybrid_mask_exact": int(recon.hybrid_mask_exact),
                    "decoder_box_max_abs_error": format(
                        recon.box_max_abs_error, ".9g"
                    ),
                    "decoder_score_max_abs_error": format(
                        recon.score_max_abs_error, ".9g"
                    ),
                    "analysis_audit_sha256": audit_sha,
                    "analysis_artifact_manifest_sha256": analysis_artifact_manifest_sha,
                    "representatives_sha256": representatives_sha,
                    "selection_sha256": selection_sha,
                    "probe_script_sha256": script_sha,
                    "full_val_replay_sha256": replay_script_sha,
                    "analyze_full_val_sha256": analysis_script_sha,
                    "plot_full_val_sha256": plot_script_sha,
                    "transition_renderer_sha256": transition_script_sha,
                    "replay_matrix_file": execution_identity["matrix_file"],
                    "replay_matrix_bytes": execution_identity["matrix_bytes"],
                    "replay_matrix_sha256": execution_identity["matrix_sha256"],
                    "supervision_probe_sha256": execution_identity[
                        "supervision_probe_sha256"
                    ],
                    "engine_python_tree_sha256": execution_identity[
                        "engine_python_tree_sha256"
                    ],
                    "runtime_backend_sha256": execution_identity[
                        "runtime_backend_sha256"
                    ],
                    "runtime_backend_json": execution_identity[
                        "runtime_backend_json"
                    ],
                    "torch_version": str(torch.__version__),
                    "torchvision_version": str(torchvision.__version__),
                    "torch_cuda_version": "" if torch.version.cuda is None else torch.version.cuda,
                    "numpy_version": np.__version__,
                    "pillow_version": PILLOW_VERSION,
                    "runtime_device": device,
                    "semantics": (
                        "Encoder colors=true selected dense source P3/P4/P5; Decoder "
                        "hybrid-small=640-resized area<32^2 OR short-side<16; selected "
                        "images are qualitative illustrations, not prevalence estimates"
                    ),
                }
            )
        figures.append(figure)

    manifest_path = staging / "query_probe_manifest.csv"
    for name, path in source_paths.items():
        if _stable_file_identity(path) != code_source_identities[name]:
            raise ValueError(f"{name} source changed before query-probe publication")
    for name, path in formal_input_paths.items():
        if _stable_file_identity(path) != formal_input_identities[name]:
            raise ValueError(f"{name} changed before query-probe publication")
    validate_probe_manifest_rows(manifest_rows, probe_data_path=probe_data_path)
    _write_csv(manifest_path, manifest_rows, MANIFEST_FIELDS)
    os.replace(staging, output)
    return [
        *(output / figure.name for figure in figures),
        output / selection_path.name,
        output / probe_data_path.name,
        output / manifest_path.name,
    ]


def main(argv: Sequence[str] | None = None) -> None:
    outputs = run(parse_args(argv))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
