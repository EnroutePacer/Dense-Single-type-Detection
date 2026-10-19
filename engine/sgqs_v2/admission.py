"""Training-only, budget-feasible final Decoder boundary exchange probe."""
import torch
import torch.nn.functional as F

from ..deim.box_ops import box_cxcywh_to_xyxy
from ..misc.box_ops import elementwise_box_iou


@torch.no_grad()
def exchange_pairs(scores, quality, small, budget=300, minimum_iou=.5, quality_gap=.05):
    """Pair disjoint excluded small positives with lower-quality admitted queries.

    Each query participates at most once. Never require all GT to fit the budget.
    Stable ties use original query order. Quality and membership are detached.
    """
    order = scores.detach().argsort(descending=True, stable=True)
    inside, outside = order[:budget], order[budget:]
    outside = outside[small[outside] & (quality[outside] >= minimum_iou)]
    outside = outside[quality[outside].argsort(descending=True, stable=True)]
    inside = inside.flip(0)  # quality ties: prefer the query closest to the cutoff
    inside = inside[quality[inside].argsort(stable=True)]
    count = min(len(outside), len(inside))
    positive, negative = outside[:count], inside[:count]
    keep = quality[positive] - quality[negative] >= quality_gap
    return positive[keep], negative[keep]


def admission_loss(outputs, targets, indices, input_size, denominator,
                   budget=300, margin=.2, temperature=.1):
    """Use final-layer Hungarian matches, never the GO union or DN matches."""
    logits, boxes = outputs['pred_logits'].float(), outputs['pred_boxes'].detach().float()
    if logits.shape[-1] != 1:
        raise ValueError('Admission probe is single-class only')
    height, width = input_size
    loss = logits.sum() * 0.
    count = 0
    for batch, (src, dst) in enumerate(indices):
        src, dst = src.to(logits.device), dst.to(logits.device)
        quality = logits.new_zeros(logits.shape[1])
        small = torch.zeros_like(quality, dtype=torch.bool)
        gt = targets[batch]['boxes'][dst].detach().float()
        if len(src):
            quality[src], _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(boxes[batch, src]), box_cxcywh_to_xyxy(gt))
            w, h = gt[:, 2] * width, gt[:, 3] * height
            small[src] = (w * h < 32**2) | (torch.minimum(w, h) < 16)
        pos, neg = exchange_pairs(logits[batch, :, 0], quality, small, budget)
        count += len(pos)
        loss = loss + (temperature * F.softplus(
            (margin + logits[batch, neg, 0] - logits[batch, pos, 0]) / temperature)).sum()
    return loss / denominator(count, logits.device), count
