"""Resumable, compact full-validation replay for SGQS checkpoints.

Unlike :mod:`replay_probe`, this entry point is designed for millions of GT
rows and many checkpoint states.  Inference is batched, incomplete states are
kept as restartable ``.npy`` memmaps, and completed artifacts are compressed
NPZ files rather than tens of millions of JSON objects.

The four diagnostic stages are independent per-GT coverage views, not COCO
TP/FN matching.  Standard COCO AP/AR is evaluated separately and the compact
prediction NPZ is also the input to ``coco_match_npz.py``.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from engine.core import YAMLConfig
from engine import sgqs_v2  # noqa: F401 - register SGQSV2Supervisor/criterion
from engine.data.dataset.coco_backend import (
    COCO,
    EVAL_IMAGE_ATTRIBUTE,
    make_evaluator,
)
from tools.table_cells import supervision_probe  # noqa: F401 - register config wrappers
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.supervision_probe import overlap


INPUT_SIZE = (640, 640)
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
STAGE_NAMES = (
    "encoder_dense_all",
    "encoder_top640",
    "decoder_all640",
    "decoder_top300",
)
COHORT_NAMES = (
    "all",
    "coco_small",
    "local_hybrid_small",
    "coco_and_hybrid",
    "coco_only",
    "hybrid_only",
    "neither",
)
THRESHOLD_SUFFIXES = tuple(f"{threshold:g}" for threshold in IOU_THRESHOLDS)
FLOAT_FIELDS = (
    "best_iou",
    "best_iou_score",
    *(f"first_score_iou{suffix}" for suffix in THRESHOLD_SUFFIXES),
)
RANK_FIELDS = (
    "best_iou_rank",
    *(f"first_rank_iou{suffix}" for suffix in THRESHOLD_SUFFIXES),
)
INDEX_FIELDS = (
    "best_iou_candidate_index",
    *(f"first_candidate_index_iou{suffix}" for suffix in THRESHOLD_SUFFIXES),
)


def _canonical_value_hash(value, digest: "hashlib._Hash") -> None:
    """Hash checkpoint values independently of pickle byte layout.

    A micro-probe checkpoint embeds its provenance inside the checkpoint.  A
    hash of the whole pickle would therefore be circular (adding the hash
    changes the pickle).  This typed traversal hashes the state payload while
    excluding that provenance field, and is shared with replay-side
    verification below.
    """

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _canonical_value_hash(str(key), digest)
            _canonical_value_hash(value[key], digest)
        digest.update(b"/dict\0")
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"seq\0")
        for item in value:
            _canonical_value_hash(item, digest)
        digest.update(b"/seq\0")
        return
    if value is None or isinstance(value, (bool, int, float, str)):
        digest.update(b"scalar\0")
        digest.update(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8"))
        digest.update(b"\0")
        return
    if isinstance(value, np.generic):
        _canonical_value_hash(value.item(), digest)
        return
    raise TypeError(f"unsupported checkpoint value in provenance hash: {type(value)!r}")


def checkpoint_payload_sha256(checkpoint: dict) -> str:
    """Hash a checkpoint after removing its self-referential identity field."""

    payload = dict(checkpoint)
    payload.pop("micro_probe_identity", None)
    digest = hashlib.sha256()
    _canonical_value_hash(payload, digest)
    return digest.hexdigest()


def resolved_config_sha256(config: YAMLConfig) -> str:
    """Hash the include-expanded source config before replay overrides."""

    payload = json.dumps(
        config.yaml_cfg,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_scores(decoder, logits: torch.Tensor) -> torch.Tensor:
    """Return the scalar score used by the real encoder selector."""
    if decoder.query_select_method == "default":
        return logits.max(-1).values
    if decoder.query_select_method == "one2many":
        if logits.shape[-1] != 1:
            raise ValueError("full replay requires single-class one2many logits")
        return logits[..., 0]
    if decoder.query_select_method == "agnostic":
        return logits.squeeze(-1)
    raise ValueError(f"unknown query_select_method: {decoder.query_select_method!r}")


class CaptureCandidates:
    """Capture dense and actually selected encoder candidates without mutation."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.original = decoder._select_topk
        self.data = None

    def __enter__(self):
        def capture(memory, logits, anchors, k):
            result = self.original(memory, logits, anchors, k)
            scores = selection_scores(self.decoder, logits)
            _, selected_indices = torch.topk(scores, k, dim=-1)
            gathered_anchors = anchors.gather(
                1, selected_indices.unsqueeze(-1).expand(-1, -1, anchors.shape[-1])
            )
            if not torch.equal(gathered_anchors, result[2]):
                raise RuntimeError("recomputed Top-K differs from the selector output")
            dense_boxes = (self.decoder.enc_bbox_head(memory) + anchors).sigmoid().detach()
            # Keep this separate from gathering dense_boxes: the real decoder
            # initialization evaluates the bbox head on result[0].  A second
            # GEMM can differ by a few ulps from head(all)[selected] and the
            # replay must follow the actual selected-box computation.
            selected_boxes = (
                self.decoder.enc_bbox_head(result[0]) + result[2]
            ).sigmoid().detach()
            self.data = {
                "boxes": dense_boxes,
                "selection_scores": scores.detach(),
                "selected_boxes": selected_boxes,
                "selected_scores": scores.gather(1, selected_indices).detach(),
                "selected_indices": selected_indices.detach(),
                "k": int(k),
            }
            return result

        self.decoder._select_topk = capture
        return self

    def __exit__(self, *args):
        self.decoder._select_topk = self.original


def ranked_gt_diagnostics(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    gt: torch.Tensor,
    *,
    candidate_indices: torch.Tensor | None = None,
    presorted: bool = False,
    ranking_scores: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | int]:
    """Return best geometry plus first score-qualified hit for every GT.

    Missing IoU-qualified candidates use ``N+1`` for rank, NaN for score, and
    ``-1`` for candidate index.  Scores must be probabilities; ordering remains
    identical to the corresponding logits in this single-class model.
    """
    scores = scores.float().reshape(-1)
    ranking_scores = scores if ranking_scores is None else ranking_scores.float().reshape(-1)
    if boxes.ndim != 2 or boxes.shape[-1] != 4 or len(boxes) != len(scores):
        raise ValueError("boxes must be [N,4] and scores must contain N values")
    if len(ranking_scores) != len(scores):
        raise ValueError("ranking_scores must contain N values")
    if gt.ndim != 2 or gt.shape[-1] != 4:
        raise ValueError("gt must be [M,4]")
    if candidate_indices is None:
        candidate_indices = torch.arange(len(boxes), device=boxes.device)
    candidate_indices = candidate_indices.to(device=boxes.device, dtype=torch.long).reshape(-1)
    if len(candidate_indices) != len(boxes):
        raise ValueError("candidate_indices must contain one value per box")

    order = (
        torch.arange(len(scores), device=scores.device)
        if presorted
        else ranking_scores.argsort(descending=True, stable=True)
    )
    ranked_scores = scores[order]
    ranked_indices = candidate_indices[order]
    ranked_ious = overlap(boxes, gt)[order]
    result: dict[str, torch.Tensor | int] = {"candidate_count": len(boxes)}
    if len(boxes):
        best_iou, best_zero = ranked_ious.max(0)
        result.update(
            best_iou=best_iou,
            best_iou_rank=best_zero + 1,
            best_iou_score=ranked_scores[best_zero],
            best_iou_candidate_index=ranked_indices[best_zero],
        )
    else:
        count = len(gt)
        result.update(
            best_iou=gt.new_zeros(count),
            best_iou_rank=torch.ones(count, device=gt.device, dtype=torch.long),
            best_iou_score=torch.full((count,), torch.nan, device=gt.device),
            best_iou_candidate_index=torch.full(
                (count,), -1, device=gt.device, dtype=torch.long
            ),
        )

    for threshold in IOU_THRESHOLDS:
        suffix = f"{threshold:g}"
        eligible = ranked_ious >= threshold
        found = eligible.any(0)
        if len(boxes):
            first_zero = eligible.to(torch.int8).argmax(0)
            first_score = ranked_scores[first_zero]
            first_index = ranked_indices[first_zero]
            first_rank = torch.where(found, first_zero + 1, len(boxes) + 1)
            first_score = torch.where(found, first_score, torch.full_like(first_score, torch.nan))
            first_index = torch.where(found, first_index, torch.full_like(first_index, -1))
        else:
            count = len(gt)
            first_rank = torch.ones(count, device=gt.device, dtype=torch.long)
            first_score = torch.full((count,), torch.nan, device=gt.device)
            first_index = torch.full((count,), -1, device=gt.device, dtype=torch.long)
        result[f"first_rank_iou{suffix}"] = first_rank
        result[f"first_score_iou{suffix}"] = first_score
        result[f"first_candidate_index_iou{suffix}"] = first_index
    return result


def restrict_coco(coco, *, max_images: int | None = None, image_ids=None):
    """Create an evaluator-correct deterministic image subset for smoke tests."""
    if max_images is not None and image_ids is not None:
        raise ValueError("max_images and image_ids are mutually exclusive")
    available = set(coco.getImgIds())
    if image_ids is not None:
        selected = set(int(value) for value in image_ids)
        missing = selected - available
        if missing:
            raise ValueError(f"unknown image IDs: {sorted(missing)[:10]}")
    elif max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        selected = set(sorted(available)[:max_images])
    else:
        return coco
    dataset = {
        key: value
        for key, value in coco.dataset.items()
        if key not in {"images", "annotations"}
    }
    dataset["images"] = [item for item in coco.dataset["images"] if item["id"] in selected]
    dataset["annotations"] = [
        item for item in coco.dataset.get("annotations", []) if item["image_id"] in selected
    ]
    return COCO(dataset)


def build_layout(coco) -> dict:
    """Freeze deterministic image/GT row order and both small definitions."""
    if len(coco.getCatIds()) != 1:
        raise ValueError("full replay is single-class only")
    image_ids = np.asarray(sorted(coco.getImgIds()), dtype=np.int64)
    infos = [coco.imgs[int(image_id)] for image_id in image_ids]
    annotations = [
        sorted(
            (item for item in coco.imgToAnns.get(int(image_id), []) if not item.get("iscrowd", 0)),
            key=lambda item: item["id"],
        )
        for image_id in image_ids
    ]
    starts = np.zeros(len(image_ids) + 1, dtype=np.int64)
    for index, items in enumerate(annotations):
        starts[index + 1] = starts[index] + len(items)
    gt_count = int(starts[-1])
    annotation_ids = np.empty(gt_count, dtype=np.int64)
    gt_image_index = np.empty(gt_count, dtype=np.int32)
    bboxes = np.empty((gt_count, 4), dtype=np.float32)
    coco_area = np.empty(gt_count, dtype=np.float64)
    input_width = np.empty(gt_count, dtype=np.float32)
    input_height = np.empty(gt_count, dtype=np.float32)

    for image_index, (info, items) in enumerate(zip(infos, annotations)):
        start, end = int(starts[image_index]), int(starts[image_index + 1])
        if start == end:
            continue
        annotation_ids[start:end] = [item["id"] for item in items]
        gt_image_index[start:end] = image_index
        bboxes[start:end] = [item["bbox"] for item in items]
        coco_area[start:end] = [
            item.get("area", item["bbox"][2] * item["bbox"][3]) for item in items
        ]
        input_width[start:end] = bboxes[start:end, 2] / info["width"] * INPUT_SIZE[1]
        input_height[start:end] = bboxes[start:end, 3] / info["height"] * INPUT_SIZE[0]

    if gt_count and ((bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0)).any():
        bad = int(np.flatnonzero((bboxes[:, 2] <= 0) | (bboxes[:, 3] <= 0))[0])
        raise ValueError(f"non-positive GT bbox for annotation {annotation_ids[bad]}")
    input_area = input_width * input_height
    input_short = np.minimum(input_width, input_height)
    coco_small = coco_area < 32**2
    hybrid_small = (input_area < 32**2) | (input_short < 16.0)
    area_group = np.where(coco_area < 32**2, 0, np.where(coco_area < 96**2, 1, 2)).astype(
        np.uint8
    )
    membership = np.where(
        coco_small & hybrid_small,
        0,
        np.where(coco_small, 1, np.where(hybrid_small, 2, 3)),
    ).astype(np.uint8)
    return {
        "image_ids": image_ids,
        "image_width": np.asarray([item["width"] for item in infos], dtype=np.int32),
        "image_height": np.asarray([item["height"] for item in infos], dtype=np.int32),
        "image_file_name": np.asarray([item["file_name"] for item in infos], dtype=np.str_),
        "image_gt_start": starts,
        "annotation_ids": annotation_ids,
        "gt_image_index": gt_image_index,
        "gt_bbox_xywh_original": bboxes,
        "coco_area": coco_area,
        "coco_area_group": area_group,
        "input_width_px": input_width,
        "input_height_px": input_height,
        "input_area_px2": input_area,
        "input_short_px": input_short,
        "coco_small": coco_small,
        "local_hybrid_small": hybrid_small,
        "small_definition_membership": membership,
        "_infos": infos,
        "_annotations": annotations,
    }


def cohort_masks(coco_small: torch.Tensor, hybrid_small: torch.Tensor) -> torch.Tensor:
    """Return [cohort, GT] masks in ``COHORT_NAMES`` order."""
    return torch.stack(
        (
            torch.ones_like(coco_small, dtype=torch.bool),
            coco_small,
            hybrid_small,
            coco_small & hybrid_small,
            coco_small & ~hybrid_small,
            hybrid_small & ~coco_small,
            ~coco_small & ~hybrid_small,
        )
    )


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def write_metadata(path: Path, layout: dict, *, resume: bool) -> None:
    stored = {key: value for key, value in layout.items() if not key.startswith("_")}
    stored.update(
        schema=np.asarray("sgqs_full_val_metadata_v1"),
        input_size=np.asarray(INPUT_SIZE, dtype=np.int32),
        cohort_names=np.asarray(COHORT_NAMES),
        area_group_names=np.asarray(("small", "medium", "large")),
        membership_names=np.asarray(("coco_and_hybrid", "coco_only", "hybrid_only", "neither")),
        coco_small_definition=np.asarray("original annotation area < 32^2"),
        local_hybrid_small_definition=np.asarray(
            "direct 640x640 resize: bbox area < 32^2 OR bbox short side < 16"
        ),
    )
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        with np.load(path, allow_pickle=False) as existing:
            if not np.array_equal(existing["image_ids"], layout["image_ids"]):
                raise ValueError(f"metadata image IDs changed: {path}")
            if not np.array_equal(existing["annotation_ids"], layout["annotation_ids"]):
                raise ValueError(f"metadata annotation IDs changed: {path}")
        return
    atomic_npz(path, **stored)


def stage_specs(gt_count: int, image_count: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for stage in STAGE_NAMES:
        for field in FLOAT_FIELDS:
            specs[f"{stage}__{field}"] = (np.dtype("float32"), (gt_count,))
        for field in RANK_FIELDS:
            specs[f"{stage}__{field}"] = (np.dtype("uint16"), (gt_count,))
        for field in INDEX_FIELDS:
            specs[f"{stage}__{field}"] = (np.dtype("int32"), (gt_count,))
    specs["image_candidate_counts"] = (np.dtype("uint16"), (image_count, len(STAGE_NAMES)))
    specs["image_independent_coverage"] = (
        np.dtype("uint32"),
        (image_count, len(COHORT_NAMES), len(IOU_THRESHOLDS), len(STAGE_NAMES)),
    )
    return specs


def prediction_specs(image_count: int, queries: int) -> dict[str, tuple[np.dtype, tuple[int, ...]]]:
    return {
        "boxes_xywh": (np.dtype("float32"), (image_count, queries, 4)),
        "scores": (np.dtype("float32"), (image_count, queries)),
    }


def open_staging(
    directory: Path,
    specs: dict[str, tuple[np.dtype, tuple[int, ...]]],
    identity: dict,
    *,
    resume: bool,
) -> tuple[dict[str, np.memmap], int]:
    progress_path = directory / "progress.json"
    if directory.exists():
        if not resume:
            raise FileExistsError(f"incomplete state exists; pass --resume: {directory}")
        if not progress_path.exists():
            raise ValueError(f"incomplete staging directory has no progress marker: {directory}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != identity:
            raise ValueError(f"resume identity mismatch: {directory}")
        arrays = {key: np.load(directory / f"{key}.npy", mmap_mode="r+") for key in specs}
        for key, (dtype, shape) in specs.items():
            if arrays[key].dtype != dtype or arrays[key].shape != shape:
                raise ValueError(f"staging array mismatch: {key}")
        return arrays, int(progress["completed_images"])

    directory.mkdir(parents=False, exist_ok=False)
    arrays = {
        key: np.lib.format.open_memmap(
            directory / f"{key}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for key, (dtype, shape) in specs.items()
    }
    for array in arrays.values():
        array.flush()
    atomic_json(progress_path, {"identity": identity, "completed_images": 0})
    return arrays, 0


def flush_progress(
    directory: Path,
    arrays: dict[str, np.memmap],
    identity: dict,
    completed_images: int,
) -> None:
    for array in arrays.values():
        array.flush()
    atomic_json(
        directory / "progress.json",
        {"identity": identity, "completed_images": int(completed_images)},
    )


def write_stage_slice(
    arrays: dict[str, np.memmap], stages: dict[str, dict], row_slice: slice
) -> None:
    """Use three GPU-to-CPU copies per image rather than one per field."""
    for fields, dtype in (
        (FLOAT_FIELDS, np.float32),
        (RANK_FIELDS, np.uint16),
        (INDEX_FIELDS, np.int32),
    ):
        values = torch.stack(
            [stages[stage][field] for stage in STAGE_NAMES for field in fields]
        ).detach().cpu().numpy()
        values = values.astype(dtype, copy=False)
        cursor = 0
        for stage in STAGE_NAMES:
            for field in fields:
                arrays[f"{stage}__{field}"][row_slice] = values[cursor]
                cursor += 1


def image_coverage(stages: dict[str, dict], masks: torch.Tensor) -> torch.Tensor:
    best = torch.stack([stages[stage]["best_iou"] for stage in STAGE_NAMES])
    outputs = []
    for threshold in IOU_THRESHOLDS:
        outputs.append(((best[None, :, :] >= threshold) & masks[:, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)  # cohort, threshold, stage


def batched_indices(start: int, stop: int, batch_size: int):
    for batch_start in range(start, stop, batch_size):
        yield range(batch_start, min(batch_start + batch_size, stop))


@torch.inference_mode()
def infer_to_staging(
    model,
    coco,
    layout: dict,
    images_root: Path,
    device,
    arrays: dict[str, np.memmap],
    partial_dir: Path,
    identity: dict,
    *,
    start_image: int,
    batch_size: int,
    durable_images: int,
    progress_every: int,
) -> None:
    transform = v2.Compose(
        [v2.Resize(list(INPUT_SIZE)), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    )
    image_count = len(layout["image_ids"])
    query_count = model.decoder.num_queries
    category = coco.getCatIds()[0]
    if category != int(category):
        raise ValueError("category ID must be integer")
    started = time.monotonic()
    durable_at = start_image

    with CaptureCandidates(model.decoder) as capture:
        for batch_range in batched_indices(start_image, image_count, batch_size):
            batch_indexes = list(batch_range)
            tensors = []
            for image_index in batch_indexes:
                info = layout["_infos"][image_index]
                image_path = Path(images_root) / info["file_name"]
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    if image.size != (info["width"], info["height"]):
                        raise ValueError(f"image/annotation dimensions disagree: {image_path}")
                    tensors.append(transform(image))
            output = model(torch.stack(tensors).to(device))
            if capture.data is None:
                raise RuntimeError("encoder capture did not run")

            for batch_index, image_index in enumerate(batch_indexes):
                width = int(layout["image_width"][image_index])
                height = int(layout["image_height"][image_index])
                pred = output["pred_boxes"][batch_index].float()
                scores = output["pred_logits"][batch_index, :, 0].sigmoid().float()
                if len(pred) != query_count:
                    raise ValueError("decoder query count changed during replay")
                if not torch.isfinite(pred).all() or not torch.isfinite(scores).all():
                    raise ValueError(f"non-finite decoder output at image {layout['image_ids'][image_index]}")
                pred_original = pred.clone()
                pred_original[:, 0] = (pred[:, 0] - pred[:, 2] / 2) * width
                pred_original[:, 1] = (pred[:, 1] - pred[:, 3] / 2) * height
                pred_original[:, 2] = pred[:, 2] * width
                pred_original[:, 3] = pred[:, 3] * height
                arrays["boxes_xywh"][image_index] = pred_original.cpu().numpy()
                arrays["scores"][image_index] = scores.cpu().numpy()

                gt_start = int(layout["image_gt_start"][image_index])
                gt_end = int(layout["image_gt_start"][image_index + 1])
                row_slice = slice(gt_start, gt_end)
                gt_original = torch.as_tensor(
                    layout["gt_bbox_xywh_original"][row_slice], device=device
                )
                gt = gt_original.clone()
                gt[:, 0] = (gt_original[:, 0] + gt_original[:, 2] / 2) / width
                gt[:, 1] = (gt_original[:, 1] + gt_original[:, 3] / 2) / height
                gt[:, 2] = gt_original[:, 2] / width
                gt[:, 3] = gt_original[:, 3] / height

                captured = capture.data
                all_dense_scores = captured["selection_scores"][batch_index]
                valid = torch.isfinite(all_dense_scores)
                dense_indices = torch.arange(len(all_dense_scores), device=device)[valid]
                query_indices = torch.arange(query_count, device=device)
                decoder_order = scores.argsort(descending=True, stable=True)
                top300 = decoder_order[: min(300, query_count)]
                stages = {
                    "encoder_dense_all": ranked_gt_diagnostics(
                        captured["boxes"][batch_index, valid],
                        all_dense_scores[valid].sigmoid(),
                        gt,
                        candidate_indices=dense_indices,
                        ranking_scores=all_dense_scores[valid],
                    ),
                    "encoder_top640": ranked_gt_diagnostics(
                        captured["selected_boxes"][batch_index],
                        captured["selected_scores"][batch_index].sigmoid(),
                        gt,
                        candidate_indices=captured["selected_indices"][batch_index],
                        presorted=True,
                    ),
                    "decoder_all640": ranked_gt_diagnostics(
                        pred, scores, gt, candidate_indices=query_indices
                    ),
                    "decoder_top300": ranked_gt_diagnostics(
                        pred[top300],
                        scores[top300],
                        gt,
                        candidate_indices=query_indices[top300],
                        presorted=True,
                    ),
                }
                write_stage_slice(arrays, stages, row_slice)
                candidate_counts = [stages[stage]["candidate_count"] for stage in STAGE_NAMES]
                if max(candidate_counts, default=0) >= np.iinfo(np.uint16).max:
                    raise ValueError("candidate count exceeds compact uint16 schema")
                arrays["image_candidate_counts"][image_index] = candidate_counts
                masks = cohort_masks(
                    torch.as_tensor(layout["coco_small"][row_slice], device=device),
                    torch.as_tensor(layout["local_hybrid_small"][row_slice], device=device),
                )
                arrays["image_independent_coverage"][image_index] = (
                    image_coverage(stages, masks).cpu().numpy().astype(np.uint32, copy=False)
                )

            completed = batch_indexes[-1] + 1
            if completed - durable_at >= durable_images or completed == image_count:
                flush_progress(partial_dir, arrays, identity, completed)
                durable_at = completed
            if progress_every and (
                completed % progress_every < len(batch_indexes) or completed == image_count
            ):
                elapsed = time.monotonic() - started
                done_now = completed - start_image
                print(
                    f"inference {completed}/{image_count} images; "
                    f"session {done_now / max(elapsed, 1e-9):.2f} image/s",
                    flush=True,
                )


def compress_staging(
    arrays: dict[str, np.memmap],
    layout: dict,
    category_id: int,
    prediction_path: Path,
    stage_path: Path,
    state_id: str,
) -> None:
    if not prediction_path.exists():
        atomic_npz(
            prediction_path,
            schema=np.asarray("sgqs_compact_predictions_v1"),
            image_ids=layout["image_ids"],
            boxes_xywh=arrays["boxes_xywh"],
            scores=arrays["scores"],
            category_id=np.asarray(category_id, dtype=np.int64),
            input_size=np.asarray(INPUT_SIZE, dtype=np.int32),
            state=np.asarray(state_id),
            box_format=np.asarray("xywh_abs_original_pixels"),
            query_indices=np.arange(arrays["scores"].shape[1], dtype=np.uint16),
            query_index_note=np.asarray("column index; repeated identically for every image"),
        )
    if not stage_path.exists():
        stage_arrays = {
            key: value
            for key, value in arrays.items()
            if key not in {"boxes_xywh", "scores"}
        }
        atomic_npz(
            stage_path,
            schema=np.asarray("sgqs_compact_gt_stages_v1"),
            stage_names=np.asarray(STAGE_NAMES),
            cohort_names=np.asarray(COHORT_NAMES),
            iou_thresholds=np.asarray(IOU_THRESHOLDS, dtype=np.float32),
            gt_row_alignment=np.asarray("split metadata annotation_ids row order"),
            score_definition=np.asarray("sigmoid(single-class logit)"),
            coverage_note=np.asarray(
                "independent per-GT best-IoU coverage; not one-to-one COCO TP/FN"
            ),
            **stage_arrays,
        )


def summary_at_budget(evaluator, budget: int) -> list[float]:
    """Return the legacy 12-element COCO stats vector for one maxDets value."""
    precision = np.asarray(evaluator.eval["precision"])
    recall = np.asarray(evaluator.eval["recall"])
    params = evaluator.params

    def summarize(ap: int, iouThr=None, areaRng="all", maxDets=100) -> float:
        area_index = params.areaRngLbl.index(areaRng)
        budget_index = params.maxDets.index(maxDets)
        if ap:
            values = precision[:, :, :, area_index, budget_index]
        else:
            values = recall[:, :, area_index, budget_index]
        if iouThr is not None:
            threshold = np.flatnonzero(np.isclose(params.iouThrs, iouThr))
            if not len(threshold):
                return -1.0
            values = values[threshold]
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else -1.0

    return [
        float(summarize(1, maxDets=budget)),
        float(summarize(1, iouThr=0.5, maxDets=budget)),
        float(summarize(1, iouThr=0.75, maxDets=budget)),
        float(summarize(1, areaRng="small", maxDets=budget)),
        float(summarize(1, areaRng="medium", maxDets=budget)),
        float(summarize(1, areaRng="large", maxDets=budget)),
        float(summarize(0, maxDets=1)),
        float(summarize(0, maxDets=10)),
        float(summarize(0, maxDets=budget)),
        float(summarize(0, areaRng="small", maxDets=budget)),
        float(summarize(0, areaRng="medium", maxDets=budget)),
        float(summarize(0, areaRng="large", maxDets=budget)),
    ]


def small_per_iou_at_budget(evaluator, budget: int) -> list[dict[str, float]]:
    """Return COCO-small AP/AR at each official IoU threshold.

    AP is the mean over the 101 globally score-ranked precision samples.  It
    therefore retains the score calibration/FP ordering effect that a per-GT
    coverage count cannot expose.  AR is the corresponding per-IoU recall.
    """
    precision = np.asarray(evaluator.eval["precision"])
    recall = np.asarray(evaluator.eval["recall"])
    area_index = evaluator.params.areaRngLbl.index("small")
    max_dets_index = evaluator.params.maxDets.index(budget)
    rows = []
    for threshold_index, threshold in enumerate(evaluator.params.iouThrs):
        precision_slice = precision[
            threshold_index, :, :, area_index, max_dets_index
        ]
        recall_slice = recall[threshold_index, :, area_index, max_dets_index]
        valid_precision = precision_slice[precision_slice > -1]
        valid_recall = recall_slice[recall_slice > -1]
        rows.append(
            {
                "iou": float(threshold),
                "ap": (
                    float(valid_precision.mean()) if valid_precision.size else -1.0
                ),
                "ar": float(valid_recall.mean()) if valid_recall.size else -1.0,
            }
        )
    return rows


def compact_coco_metrics(
    coco, prediction_path: Path, *, batch_size: int = 32
) -> dict[int | str, object]:
    """Evaluate compact predictions without materializing 8.2M Python dicts."""
    with np.load(prediction_path, allow_pickle=False) as archive:
        image_ids = archive["image_ids"]
        boxes = archive["boxes_xywh"]
        scores = archive["scores"]
        category_id = int(archive["category_id"])
        if boxes.shape[:2] != scores.shape or boxes.shape[-1] != 4:
            raise ValueError("invalid compact prediction shapes")
        if not np.array_equal(image_ids, np.asarray(sorted(coco.getImgIds()))):
            raise ValueError("prediction image IDs do not match evaluator image IDs")

        evaluator = make_evaluator(
            coco,
            iouType="bbox",
            print_function=lambda *args: None,
            separate_eval=True,
        )
        evaluator.params.maxDets = [1, 10, 300, 640]
        chunks = []
        for batch_range in batched_indices(0, len(image_ids), batch_size):
            indexes = np.asarray(list(batch_range), dtype=np.int64)
            batch_ids = image_ids[indexes]
            batch_boxes = boxes[indexes]
            batch_scores = scores[indexes]
            detections = np.empty((batch_boxes.shape[0] * batch_boxes.shape[1], 7), dtype=np.float64)
            detections[:, 0] = np.repeat(batch_ids, batch_boxes.shape[1])
            detections[:, 1:5] = batch_boxes.reshape(-1, 4)
            detections[:, 5] = batch_scores.reshape(-1)
            detections[:, 6] = category_id
            evaluator.cocoDt = coco.loadRes(detections)
            evaluator.params.imgIds = batch_ids.tolist()
            evaluator.evaluate()
            batch_results = getattr(evaluator, EVAL_IMAGE_ATTRIBUTE)
            chunks.append(
                np.asarray(batch_results, dtype=object).reshape(
                    len(evaluator.params.catIds),
                    len(evaluator.params.areaRng),
                    len(batch_ids),
                )
            )
            if len(chunks) == 1 or len(chunks) % 25 == 0 or batch_range.stop == len(image_ids):
                print(
                    f"COCO matching images={batch_range.stop}/{len(image_ids)} "
                    f"backend={type(evaluator).__name__}",
                    flush=True,
                )

        evaluator.params.imgIds = image_ids.tolist()
        evaluator._paramsEval = copy.deepcopy(evaluator.params)
        setattr(
            evaluator,
            EVAL_IMAGE_ATTRIBUTE,
            np.concatenate(chunks, axis=2).ravel().tolist(),
        )
        evaluator.accumulate()
        result: dict[int | str, object] = {
            budget: summary_at_budget(evaluator, budget) for budget in (300, 640)
        }
        result["small_per_iou"] = {
            budget: small_per_iou_at_budget(evaluator, budget)
            for budget in (300, 640)
        }
        return result


def summarize_stage_artifact(layout: dict, stage_path: Path) -> dict:
    with np.load(stage_path, allow_pickle=False) as archive:
        coverage = archive["image_independent_coverage"].sum(axis=0, dtype=np.uint64)
    coco_small = layout["coco_small"]
    hybrid_small = layout["local_hybrid_small"]
    masks = (
        np.ones_like(coco_small, dtype=bool),
        coco_small,
        hybrid_small,
        coco_small & hybrid_small,
        coco_small & ~hybrid_small,
        hybrid_small & ~coco_small,
        ~coco_small & ~hybrid_small,
    )
    nested = {}
    for cohort_index, cohort in enumerate(COHORT_NAMES):
        nested[cohort] = {}
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            nested[cohort][f"iou{threshold:g}"] = {
                stage: int(coverage[cohort_index, threshold_index, stage_index])
                for stage_index, stage in enumerate(STAGE_NAMES)
            }
    return {
        "gt_counts": {
            name: int((layout["coco_area_group"] == index).sum())
            for index, name in enumerate(("small", "medium", "large"))
        },
        "cohort_counts": {
            cohort: int(mask.sum()) for cohort, mask in zip(COHORT_NAMES, masks)
        },
        "independent_gt_coverage": nested,
    }


def compare_legacy(
    reference_dir: Path,
    split: str,
    state_name: str,
    layout: dict,
    stage_path: Path,
    metrics: dict[int, list[float]],
    *,
    atol: float,
) -> dict:
    """Fail a val128 smoke run if compact results differ from legacy output."""
    row_path = reference_dir / f"{split}.{state_name}.gt.jsonl"
    summary_path = reference_dir / "summary.json"
    if not row_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"legacy reference is incomplete: {reference_dir}")
    id_to_row = {int(value): index for index, value in enumerate(layout["annotation_ids"])}
    legacy_rows = []
    with row_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            legacy_rows.append((id_to_row[int(row["annotation_id"])], row))
    if len(legacy_rows) != len(id_to_row):
        raise ValueError("legacy/compact annotation counts differ")
    field_map = {
        "encoder_dense_best_iou_all": "encoder_dense_all__best_iou",
        "encoder_dense_best_iou_top640": "encoder_top640__best_iou",
        "decoder_best_iou_all": "decoder_all640__best_iou",
        "decoder_best_iou_top300": "decoder_top300__best_iou",
        "encoder_dense_first_rank_iou0.5": "encoder_dense_all__first_rank_iou0.5",
        "encoder_dense_first_rank_iou0.75": "encoder_dense_all__first_rank_iou0.75",
        "decoder_first_rank_iou0.5": "decoder_all640__first_rank_iou0.5",
        "decoder_first_rank_iou0.75": "decoder_all640__first_rank_iou0.75",
    }
    maxima = {}
    with np.load(stage_path, allow_pickle=False) as archive:
        for legacy_field, compact_field in field_map.items():
            expected = np.asarray([row[legacy_field] for _, row in legacy_rows])
            indexes = np.asarray([index for index, _ in legacy_rows])
            actual = archive[compact_field][indexes]
            if "rank" in legacy_field:
                if not np.array_equal(actual, expected):
                    raise AssertionError(f"legacy rank mismatch: {legacy_field}")
                maxima[legacy_field] = 0.0
            else:
                maximum = float(np.max(np.abs(actual.astype(np.float64) - expected)))
                maxima[legacy_field] = maximum
                if maximum > atol:
                    raise AssertionError(
                        f"legacy value mismatch {legacy_field}: max_abs={maximum} > {atol}"
                    )
    legacy_result = json.loads(summary_path.read_text(encoding="utf-8"))["results"][
        f"{split}/{state_name}"
    ]
    metric_maxima = {}
    for budget in (300, 640):
        expected = np.asarray(legacy_result[f"coco_maxdets{budget}"], dtype=np.float64)
        actual = np.asarray(metrics[budget], dtype=np.float64)
        maximum = float(np.max(np.abs(expected - actual)))
        metric_maxima[str(budget)] = maximum
        if maximum > atol:
            raise AssertionError(f"legacy COCO@{budget} mismatch: {maximum} > {atol}")
    return {
        "reference": str(reference_dir.resolve()),
        "atol": atol,
        "max_abs_by_field": maxima,
        "metric_max_abs": metric_maxima,
    }


def replay_state(
    model,
    coco,
    layout: dict,
    images_root: Path,
    output: Path,
    prefix: str,
    identity: dict,
    *,
    device,
    batch_size: int,
    durable_images: int,
    progress_every: int,
    eval_batch_size: int,
    resume: bool,
    keep_staging: bool,
    legacy_reference: Path | None,
    legacy_atol: float,
) -> dict:
    prediction_path = output / f"{prefix}.predictions.npz"
    stage_path = output / f"{prefix}.gt_stages.npz"
    complete_path = output / f"{prefix}.complete.json"
    partial_dir = output / f".{prefix}.partial"
    if complete_path.exists():
        if not resume:
            raise FileExistsError(complete_path)
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if marker["identity"] != identity:
            raise ValueError(f"completed state identity mismatch: {complete_path}")
        return marker["result"]
    if (prediction_path.exists() or stage_path.exists()) and not resume:
        raise FileExistsError(f"state artifacts already exist for {prefix}")

    if not (prediction_path.exists() and stage_path.exists()):
        specs = prediction_specs(len(layout["image_ids"]), model.decoder.num_queries)
        specs.update(stage_specs(len(layout["annotation_ids"]), len(layout["image_ids"])))
        arrays, start_image = open_staging(partial_dir, specs, identity, resume=resume)
        if start_image < len(layout["image_ids"]):
            infer_to_staging(
                model,
                coco,
                layout,
                images_root,
                device,
                arrays,
                partial_dir,
                identity,
                start_image=start_image,
                batch_size=batch_size,
                durable_images=durable_images,
                progress_every=progress_every,
            )
        compress_staging(
            arrays,
            layout,
            coco.getCatIds()[0],
            prediction_path,
            stage_path,
            f"{Path(identity['run_identity']['checkpoint']).stem}:{identity['state']}",
        )
        arrays.clear()
        gc.collect()
        if not keep_staging:
            shutil.rmtree(partial_dir)
    elif partial_dir.exists() and not keep_staging:
        # Both immutable NPZ artifacts survived; a crash between compression
        # and cleanup must not leave hundreds of megabytes of stale memmaps.
        progress_path = partial_dir / "progress.json"
        if not progress_path.exists() or json.loads(
            progress_path.read_text(encoding="utf-8")
        ).get("identity") != identity:
            raise ValueError(f"refusing to remove unverified staging directory: {partial_dir}")
        shutil.rmtree(partial_dir)

    metrics = compact_coco_metrics(coco, prediction_path, batch_size=eval_batch_size)
    result = {
        "schema": "sgqs_full_val_replay_result_v1",
        "images": len(layout["image_ids"]),
        "noncrowd_annotations": len(layout["annotation_ids"]),
        "predictions": int(len(layout["image_ids"]) * model.decoder.num_queries),
        "coco_maxdets300": metrics[300],
        "coco_maxdets640": metrics[640],
        "coco_small_per_iou_maxdets300": metrics["small_per_iou"][300],
        "coco_small_per_iou_maxdets640": metrics["small_per_iou"][640],
        **summarize_stage_artifact(layout, stage_path),
        "artifacts": {
            "predictions": {
                "file": prediction_path.name,
                "bytes": prediction_path.stat().st_size,
                "sha256": sha256(prediction_path),
            },
            "gt_stages": {
                "file": stage_path.name,
                "bytes": stage_path.stat().st_size,
                "sha256": sha256(stage_path),
            },
        },
        "coverage_note": (
            "per-GT best-IoU fields are independent coverage, not one-to-one COCO TP/FN"
        ),
    }
    if legacy_reference is not None:
        split, state_name = prefix.rsplit(".", 1)
        result["legacy_consistency"] = compare_legacy(
            legacy_reference,
            split,
            state_name,
            layout,
            stage_path,
            metrics,
            atol=legacy_atol,
        )
    atomic_json(complete_path, {"identity": identity, "result": result})
    return result


def resolve_sources(args) -> tuple[list[dict], dict]:
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("schema") != "sgqs_fixed_probe_v1":
            raise ValueError("unknown fixed-subset manifest schema")
        # A frozen manifest may have been created on another machine.  An
        # explicit root is portable because every selected image is still
        # verified against the manifest SHA-256 below.
        root = args.images_root or Path(manifest["images_root"])
        names = [args.manifest_split] if args.manifest_split else list(manifest["splits"])
        sources = []
        for name in names:
            meta = manifest["splits"][name]
            annotation = args.manifest.parent / meta["annotation"]
            if sha256(annotation) != meta["annotation_sha256"]:
                raise ValueError(f"fixed annotation changed: {annotation}")
            for filename, expected in meta["image_sha256"].items():
                if sha256(root / filename) != expected:
                    raise ValueError(f"fixed image changed: {filename}")
            sources.append(
                {
                    "name": name,
                    "annotation": annotation,
                    "annotation_sha256": meta["annotation_sha256"],
                    "images_root": root,
                }
            )
        return sources, {
            "mode": "fixed_manifest",
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
        }

    if args.images_root is None:
        raise ValueError("--images-root is required with --annotation")
    annotation_hash = sha256(args.annotation)
    if args.expected_annotation_sha256 and (
        annotation_hash.lower() != args.expected_annotation_sha256.lower()
    ):
        raise ValueError(
            f"annotation SHA-256 mismatch: expected {args.expected_annotation_sha256}, "
            f"got {annotation_hash}"
        )
    return [
        {
            "name": args.split_name,
            "annotation": args.annotation,
            "annotation_sha256": annotation_hash,
            "images_root": args.images_root,
        }
    ], {
        "mode": "direct_coco",
        "annotation": str(args.annotation.resolve()),
        "annotation_sha256": annotation_hash,
        "images_root": str(args.images_root.resolve()),
    }


def checkpoint_states(checkpoint: dict, names: list[str]) -> dict:
    result = {}
    for name in names:
        if name == "raw":
            if "model" not in checkpoint:
                raise KeyError("checkpoint has no raw model state")
            result[name] = checkpoint["model"]
        else:
            ema = checkpoint.get("ema")
            if not isinstance(ema, dict) or "module" not in ema:
                raise KeyError("checkpoint has no EMA module state")
            result[name] = ema["module"]
    return result


def _verify_embedded_probe_identity(
    checkpoint: dict,
    *,
    config: Path,
    source_identity: dict,
) -> dict | None:
    """Verify and compactly expose identity embedded by a micro probe.

    ``single_gpu_micro_probe`` stores its final identity in ``probe.pth``.
    Replaying that artifact with a different config or fixed manifest is an
    easy way to invalidate a paired comparison while still obtaining plausible
    predictions.  Ordinary training checkpoints do not carry this field and
    remain fully compatible; when it is present, the two immutable hashes are
    checked before inference starts.
    """

    embedded = checkpoint.get("micro_probe_identity")
    if embedded is None:
        return None
    if not isinstance(embedded, dict):
        raise ValueError("checkpoint micro_probe_identity must be a JSON object")

    expected_config = embedded.get("config_sha256")
    actual_config = sha256(config)
    if expected_config is not None and str(expected_config).lower() != actual_config.lower():
        raise ValueError(
            "micro-probe checkpoint config hash differs from replay config: "
            f"expected {expected_config}, got {actual_config}"
        )

    expected_payload = embedded.get("probe_payload_sha256")
    if expected_payload is not None:
        actual_payload = checkpoint_payload_sha256(checkpoint)
        if str(expected_payload).lower() != actual_payload.lower():
            raise ValueError(
                "micro-probe checkpoint payload hash differs from embedded identity: "
                f"expected {expected_payload}, got {actual_payload}"
            )

    expected_base_config = embedded.get("base_resolved_config_sha256")
    if expected_base_config is not None:
        # The caller supplies the source YAML path; construct a clean config
        # before replay-specific overrides so this verifies the same
        # include-expanded source configuration used by the micro probe.
        source_config = YAMLConfig(str(config))
        actual_base_config = resolved_config_sha256(source_config)
        if str(expected_base_config).lower() != actual_base_config.lower():
            raise ValueError(
                "micro-probe checkpoint base resolved-config hash differs from replay config: "
                f"expected {expected_base_config}, got {actual_base_config}"
            )

    expected_manifest = embedded.get("manifest_sha256")
    actual_manifest = source_identity.get("manifest_sha256")
    if expected_manifest is not None and actual_manifest is not None:
        if str(expected_manifest).lower() != str(actual_manifest).lower():
            raise ValueError(
                "micro-probe checkpoint manifest hash differs from replay manifest: "
                f"expected {expected_manifest}, got {actual_manifest}"
            )

    # Keep the summary compact and JSON-only; the full checkpoint remains the
    # authoritative artifact, while these fields make a report self-auditing.
    keys = (
        "schema", "arm", "config", "config_sha256", "base_resolved_config_sha256",
        "resolved_config_sha256", "probe_payload_sha256",
        "checkpoint", "checkpoint_sha256", "manifest", "manifest_sha256",
        "seed", "steps", "batch_size", "accumulation", "device",
        "target_strata_coverage_status", "probe_checkpoint_sha256",
    )
    result = {key: embedded[key] for key in keys if key in embedded}
    result["verified_config_sha256"] = actual_config
    if expected_base_config is not None:
        result["verified_base_resolved_config_sha256"] = actual_base_config
    if expected_payload is not None:
        result["verified_probe_payload_sha256"] = actual_payload
    if actual_manifest is not None:
        result["verified_manifest_sha256"] = str(actual_manifest)
    result["verified_against_replay_inputs"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotation", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--expected-annotation-sha256")
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--manifest-split", choices=("train", "val"))
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument("--max-images", type=int)
    subset.add_argument("--image-ids", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", nargs="+", choices=("raw", "ema"), default=("raw", "ema"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--durable-images", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument("--legacy-reference", type=Path)
    parser.add_argument("--legacy-atol", type=float, default=1e-6)
    args = parser.parse_args()
    for name in ("batch_size", "eval_batch_size", "durable_images"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")

    sources, source_identity = resolve_sources(args)
    checkpoint_hash = sha256(args.checkpoint)
    run_identity = {
        "schema": "sgqs_full_val_replay_run_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "source": source_identity,
        "max_images": args.max_images,
        "image_ids": args.image_ids,
        "states": list(args.states),
        "diagnostic_iou_thresholds": list(IOU_THRESHOLDS),
        "inference_identity": {
            "device": str(args.device),
            "batch_size": args.batch_size,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
        },
    }
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"output exists; pass --resume only for this run: {args.output}")
        saved_identity = json.loads(
            (args.output / "run_identity.json").read_text(encoding="utf-8")
        )
        if saved_identity != run_identity:
            raise ValueError("output run identity differs from requested replay")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output / "run_identity.json", run_identity)

    config = YAMLConfig(str(args.config))
    config.yaml_cfg["HGNetv2"]["pretrained"] = False
    if config.yaml_cfg.get("eval_spatial_size") != list(INPUT_SIZE):
        raise ValueError("full replay requires eval_spatial_size=[640,640]")
    model = config.model.to(args.device).eval()
    if model.decoder.num_queries != 640 or model.decoder.num_classes != 1:
        raise ValueError("full replay requires a single-class 640-query model")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    embedded_probe_identity = _verify_embedded_probe_identity(
        checkpoint,
        config=args.config,
        source_identity=source_identity,
    )
    weights = checkpoint_states(checkpoint, list(args.states))
    summary = {
        **run_identity,
        "last_epoch": checkpoint.get("last_epoch"),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "results": {},
    }
    if embedded_probe_identity is not None:
        summary["checkpoint_micro_probe_identity"] = embedded_probe_identity

    for source_item in sources:
        coco = COCO(str(source_item["annotation"]))
        coco = restrict_coco(coco, max_images=args.max_images, image_ids=args.image_ids)
        layout = build_layout(coco)
        metadata_path = args.output / f"{source_item['name']}.metadata.npz"
        write_metadata(metadata_path, layout, resume=args.resume)
        summary.setdefault("splits", {})[source_item["name"]] = {
            "annotation": str(source_item["annotation"].resolve()),
            "annotation_sha256": source_item["annotation_sha256"],
            "images": len(layout["image_ids"]),
            "noncrowd_annotations": len(layout["annotation_ids"]),
            "metadata": metadata_path.name,
            "metadata_sha256": sha256(metadata_path),
        }
        for state_name in args.states:
            model.load_state_dict(weights[state_name], strict=True)
            prefix = f"{source_item['name']}.{state_name}"
            state_identity = {
                "run_identity": run_identity,
                "split": source_item["name"],
                "annotation_sha256": source_item["annotation_sha256"],
                "state": state_name,
                "images": len(layout["image_ids"]),
                "annotations": len(layout["annotation_ids"]),
            }
            summary["results"][f"{source_item['name']}/{state_name}"] = replay_state(
                model,
                coco,
                layout,
                source_item["images_root"],
                args.output,
                prefix,
                state_identity,
                device=args.device,
                batch_size=args.batch_size,
                durable_images=args.durable_images,
                progress_every=args.progress_every,
                eval_batch_size=args.eval_batch_size,
                resume=args.resume,
                keep_staging=args.keep_staging,
                legacy_reference=args.legacy_reference,
                legacy_atol=args.legacy_atol,
            )
            atomic_json(args.output / "summary.partial.json", summary)

    atomic_json(args.output / "summary.json", summary)
    partial_summary = args.output / "summary.partial.json"
    if partial_summary.exists():
        partial_summary.unlink()
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
