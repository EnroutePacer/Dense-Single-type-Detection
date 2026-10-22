from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EPOCHS = range(28, 35)
QUERY_COUNT = 640
TOPK = 300


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_xywh_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax1, ay1, aw, ah = a.T
    bx1, by1, bw, bh = b.T
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def safe_rate(mask: np.ndarray, denominator: np.ndarray | None = None) -> float:
    selected = mask if denominator is None else mask[denominator]
    return float(selected.mean()) if len(selected) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = ROOT / "_local" / "fullval_cpu"
    metadata_path = base / "checkpoint0028" / "val.metadata.npz"
    with np.load(metadata_path, allow_pickle=False) as metadata:
        image_index = np.asarray(metadata["gt_image_index"], dtype=np.int32)
        gt_boxes = np.asarray(metadata["gt_bbox_xywh_original"], dtype=np.float32)
        coco_small = np.asarray(metadata["coco_small"], dtype=bool)

    coverage_rows: list[dict] = []
    rank_rows: list[dict] = []

    for epoch in EPOCHS:
        folder = base / f"checkpoint{epoch:04d}"
        with np.load(folder / "val.ema.predictions.npz", allow_pickle=False) as pred:
            scores = np.asarray(pred["scores"], dtype=np.float32)
            boxes = np.asarray(pred["boxes_xywh"], dtype=np.float32)

        order = np.argsort(-scores, axis=1, kind="stable")
        decoder_rank = np.empty_like(order, dtype=np.int32)
        rows = np.arange(len(scores), dtype=np.int32)[:, None]
        decoder_rank[rows, order] = np.arange(QUERY_COUNT, dtype=np.int32)

        encoder_rank = np.arange(QUERY_COUNT, dtype=np.float64)
        rank_delta = decoder_rank.astype(np.float64) - encoder_rank[None, :]
        rank_rho = 1.0 - (
            6.0 * np.square(rank_delta).sum(axis=1)
            / (QUERY_COUNT * (QUERY_COUNT**2 - 1.0))
        )
        decoder_top1_source = order[:, 0]
        rank_rows.append(
            {
                "epoch": epoch,
                "images": len(scores),
                "all_query_pairs": scores.size,
                "rank_spearman_mean_per_image": float(rank_rho.mean()),
                "encoder_top300_retention": float((order[:, :TOPK] < TOPK).mean()),
                "decoder_top1_source_encoder_rank_mean": float((decoder_top1_source + 1).mean()),
                "decoder_top1_source_encoder_rank_median": float(np.median(decoder_top1_source + 1)),
                "decoder_top1_from_encoder_top1": float((decoder_top1_source < 1).mean()),
                "decoder_top1_from_encoder_top5": float((decoder_top1_source < 5).mean()),
                "decoder_top1_from_encoder_top10": float((decoder_top1_source < 10).mean()),
                "decoder_top1_from_encoder_top32": float((decoder_top1_source < 32).mean()),
                "decoder_top1_from_encoder_top64": float((decoder_top1_source < 64).mean()),
                "decoder_top1_from_encoder_top128": float((decoder_top1_source < 128).mean()),
                "decoder_top1_from_encoder_top300": float((decoder_top1_source < TOPK).mean()),
            }
        )

        with np.load(folder / "val.ema.gt_stages.npz", allow_pickle=False) as stage:
            encoder_first_rank = np.asarray(
                stage["encoder_top640__first_rank_iou0.5"], dtype=np.int32
            )
            decoder_all_first = np.asarray(
                stage["decoder_all640__first_candidate_index_iou0.5"], dtype=np.int32
            )
            decoder_top_first = np.asarray(
                stage["decoder_top300__first_candidate_index_iou0.5"], dtype=np.int32
            )

        owner_query = encoder_first_rank - 1
        encoder_has_owner = (owner_query >= 0) & (owner_query < QUERY_COUNT)
        safe_query = np.clip(owner_query, 0, QUERY_COUNT - 1)
        owner_decoder_rank = decoder_rank[image_index, safe_query]
        owner_decoder_score = scores[image_index, safe_query]
        owner_decoder_box = boxes[image_index, safe_query]
        owner_decoder_iou = paired_xywh_iou(owner_decoder_box, gt_boxes)

        for cohort, cohort_mask in (
            ("all_gt", np.ones(len(gt_boxes), dtype=bool)),
            ("coco_small", coco_small),
        ):
            denom = cohort_mask & encoder_has_owner
            owner_geometry_survives = owner_decoder_iou >= 0.5
            owner_encoder_top300 = owner_query < TOPK
            owner_in_top300 = owner_decoder_rank < TOPK
            direct_delivered = owner_geometry_survives & owner_in_top300
            any_all_hit = decoder_all_first >= 0
            any_top_hit = decoder_top_first >= 0
            exact_all_winner = decoder_all_first == owner_query
            exact_top_winner = decoder_top_first == owner_query
            other_top_winner_while_owner_delivered = (
                direct_delivered & any_top_hit & ~exact_top_winner
            )
            coverage_rescue_top300 = ~direct_delivered & any_top_hit
            top300_failure = ~direct_delivered & ~any_top_hit

            coverage_rows.append(
                {
                    "epoch": epoch,
                    "cohort": cohort,
                    "gt_count": int(cohort_mask.sum()),
                    "encoder_owner_count": int(denom.sum()),
                    "encoder_owner_coverage": safe_rate(encoder_has_owner, cohort_mask),
                    "owner_final_geometry_iou50": safe_rate(owner_geometry_survives, denom),
                    "owner_encoder_initial_top300": safe_rate(owner_encoder_top300, denom),
                    "owner_final_top300": safe_rate(owner_in_top300, denom),
                    "owner_encoder_top300_retained": safe_rate(
                        owner_in_top300, denom & owner_encoder_top300
                    ),
                    "owner_encoder_tail_promoted_top300": safe_rate(
                        owner_in_top300, denom & ~owner_encoder_top300
                    ),
                    "direct_owner_delivered_iou50_top300": safe_rate(direct_delivered, denom),
                    "top300_exact_local_winner_same_query": safe_rate(exact_top_winner, denom),
                    "top300_other_winner_owner_also_delivered": safe_rate(
                        other_top_winner_while_owner_delivered, denom
                    ),
                    "top300_coverage_rescue_owner_not_delivered": safe_rate(
                        coverage_rescue_top300, denom
                    ),
                    "top300_failure": safe_rate(top300_failure, denom),
                    "all640_any_hit": safe_rate(any_all_hit, denom),
                    "all640_exact_local_winner_same_query": safe_rate(exact_all_winner, denom),
                    "owner_decoder_score_mean": float(owner_decoder_score[denom].mean()),
                    "owner_encoder_rank_mean_1based": float(
                        (owner_query[denom] + 1).mean()
                    ),
                    "owner_encoder_rank_median_1based": float(
                        np.median(owner_query[denom] + 1)
                    ),
                    "owner_decoder_rank_mean_1based": float(
                        (owner_decoder_rank[denom] + 1).mean()
                    ),
                    "owner_decoder_rank_median_1based": float(
                        np.median(owner_decoder_rank[denom] + 1)
                    ),
                }
            )

        del scores, boxes, order, decoder_rank
        print(f"finished e{epoch}", flush=True)

    write_csv(OUT / "gt_local_fire_coverage.csv", coverage_rows)
    write_csv(OUT / "global_rank_alignment.csv", rank_rows)


if __name__ == "__main__":
    main()
