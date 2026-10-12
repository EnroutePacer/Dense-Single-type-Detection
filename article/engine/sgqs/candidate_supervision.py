"""Pre-Top-K candidate supervision for dense single-type detection.

The module in this file has no learnable parameters.  It is injected into the
existing D-FINE decoder and is executed only while training.  Ground-truth
boxes are used to choose one unique encoder owner position per target after all
data augmentation has been applied.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..core import register


def _spatial_shape_list(spatial_shapes) -> List[Tuple[int, int]]:
    if isinstance(spatial_shapes, torch.Tensor):
        spatial_shapes = spatial_shapes.detach().cpu().tolist()
    return [(int(shape[0]), int(shape[1])) for shape in spatial_shapes]


@torch.no_grad()
def assign_scale_aware_owners(
    targets,
    spatial_shapes,
    feat_strides: Sequence[int],
    valid_mask: torch.Tensor,
    scale_priors: Sequence[float],
    initial_radius: int = 1,
    max_radius: int = 4,
    ignore_radius: int = 1,
) -> Dict[str, object]:
    """Assign a unique, valid encoder point to each augmented GT box.

    The assignment is deliberately discrete and does not participate in
    autograd.  GTs are processed from the smallest pixel short side to the
    largest.  Collisions are resolved first in a local neighborhood and then by
    trying the remaining feature levels in scale-distance order.
    """

    shapes = _spatial_shape_list(spatial_shapes)
    num_levels = len(shapes)
    if num_levels == 0:
        raise ValueError('spatial_shapes must contain at least one level')
    if len(feat_strides) != num_levels:
        raise ValueError('feat_strides and spatial_shapes must have equal length')
    if len(scale_priors) != num_levels:
        raise ValueError('scale_priors and spatial_shapes must have equal length')
    if initial_radius < 0 or max_radius < initial_radius:
        raise ValueError('owner radii must satisfy 0 <= initial_radius <= max_radius')
    if ignore_radius < 0:
        raise ValueError('ignore_radius must be non-negative')

    batch_size = len(targets)
    device = valid_mask.device
    if valid_mask.ndim == 3:
        valid_mask = valid_mask.squeeze(-1)
    if valid_mask.ndim != 2:
        raise ValueError('valid_mask must have shape [B, N] or [B, N, 1]')
    if valid_mask.shape[0] == 1 and batch_size > 1:
        valid_mask = valid_mask.expand(batch_size, -1)
    if valid_mask.shape[0] != batch_size:
        raise ValueError('valid_mask batch dimension does not match targets')
    valid_mask = valid_mask.to(dtype=torch.bool)

    level_sizes = [h * w for h, w in shapes]
    offsets = []
    running = 0
    for size in level_sizes:
        offsets.append(running)
        running += size
    if valid_mask.shape[1] != running:
        raise ValueError('valid_mask length does not match spatial_shapes')

    input_h = shapes[0][0] * int(feat_strides[0])
    input_w = shapes[0][1] * int(feat_strides[0])
    valid_cpu = valid_mask.detach().cpu()
    ignore_cpu = torch.zeros_like(valid_cpu, dtype=torch.bool)

    owner_indices: List[torch.Tensor] = []
    target_indices: List[torch.Tensor] = []
    owner_levels: List[torch.Tensor] = []
    assignment_misses: List[int] = []

    for batch_idx, target in enumerate(targets):
        boxes = target['boxes'].detach().to(dtype=torch.float32, device='cpu')
        if boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError('target boxes must have shape [G, 4] in cxcywh format')

        gt_info = []
        for gt_idx, box in enumerate(boxes.tolist()):
            cx, cy, width, height = box
            is_valid = all(math.isfinite(v) for v in box) and width > 0 and height > 0
            short_side = min(width * input_w, height * input_h) if is_valid else math.inf
            if is_valid:
                level_order = sorted(
                    range(num_levels),
                    key=lambda level: (
                        abs(math.log((short_side + 1e-6) / (float(scale_priors[level]) + 1e-6))),
                        int(feat_strides[level]),
                        level,
                    ),
                )
                preferred_level = level_order[0]
                preferred_h, preferred_w = shapes[preferred_level]
                x_cont = min(max(cx, 0.0), 1.0) * preferred_w - 0.5
                y_cont = min(max(cy, 0.0), 1.0) * preferred_h - 0.5
                preferred_col = math.floor(x_cont + 0.5)
                preferred_row = math.floor(y_cont + 0.5)
                if 0 <= preferred_col < preferred_w and 0 <= preferred_row < preferred_h:
                    preferred_idx = offsets[preferred_level] + preferred_row * preferred_w + preferred_col
                else:
                    preferred_idx = None
            else:
                level_order = []
                preferred_level = preferred_row = preferred_col = None
                preferred_idx = None
            gt_info.append((
                short_side, gt_idx, is_valid, cx, cy, level_order,
                preferred_idx, preferred_level, preferred_row, preferred_col,
            ))

        preferred_counts = Counter(
            info[6] for info in gt_info
            if info[6] is not None and bool(valid_cpu[batch_idx, info[6]])
        )

        used = set()
        assigned = []
        missed = 0
        for (
            short_side, gt_idx, gt_valid, cx, cy, level_order,
            preferred_idx, preferred_level, preferred_row, preferred_col,
        ) in sorted(gt_info, key=lambda item: (item[0], item[1])):
            if not gt_valid:
                missed += 1
                continue

            chosen = None
            if (
                preferred_idx is not None
                and preferred_counts[preferred_idx] == 1
                and preferred_idx not in used
                and bool(valid_cpu[batch_idx, preferred_idx])
            ):
                chosen = (preferred_idx, preferred_level, preferred_row, preferred_col)

            for level in level_order:
                if chosen is not None:
                    break
                height_l, width_l = shapes[level]
                x_cont = min(max(cx, 0.0), 1.0) * width_l - 0.5
                y_cont = min(max(cy, 0.0), 1.0) * height_l - 0.5
                center_col = math.floor(x_cont + 0.5)
                center_row = math.floor(y_cont + 0.5)

                for radius in range(initial_radius, max_radius + 1):
                    candidates = []
                    row_lo = max(0, center_row - radius)
                    row_hi = min(height_l - 1, center_row + radius)
                    col_lo = max(0, center_col - radius)
                    col_hi = min(width_l - 1, center_col + radius)
                    for row in range(row_lo, row_hi + 1):
                        for col in range(col_lo, col_hi + 1):
                            global_idx = offsets[level] + row * width_l + col
                            if global_idx in used or not bool(valid_cpu[batch_idx, global_idx]):
                                continue
                            distance = (col - x_cont) ** 2 + (row - y_cont) ** 2
                            candidates.append((distance, global_idx, row, col))
                    if candidates:
                        _, global_idx, row, col = min(candidates, key=lambda item: (item[0], item[1]))
                        chosen = (global_idx, level, row, col)
                        break
                if chosen is not None:
                    break

            if chosen is None:
                missed += 1
                continue

            global_idx, level, row, col = chosen
            used.add(global_idx)
            assigned.append((gt_idx, global_idx, level, row, col))

        for _, _, level, row, col in assigned:
            height_l, width_l = shapes[level]
            for ring_row in range(max(0, row - ignore_radius), min(height_l - 1, row + ignore_radius) + 1):
                for ring_col in range(max(0, col - ignore_radius), min(width_l - 1, col + ignore_radius) + 1):
                    ring_idx = offsets[level] + ring_row * width_l + ring_col
                    if bool(valid_cpu[batch_idx, ring_idx]):
                        ignore_cpu[batch_idx, ring_idx] = True

        assigned_owner_indices = [item[1] for item in assigned]
        if assigned_owner_indices:
            ignore_cpu[batch_idx, assigned_owner_indices] = False

        owner_indices.append(torch.tensor(assigned_owner_indices, dtype=torch.long, device=device))
        target_indices.append(torch.tensor([item[0] for item in assigned], dtype=torch.long, device=device))
        owner_levels.append(torch.tensor([item[2] for item in assigned], dtype=torch.long, device=device))
        assignment_misses.append(missed)

    return {
        'owner_indices': owner_indices,
        'target_indices': target_indices,
        'owner_levels': owner_levels,
        'ignore_mask': ignore_cpu.to(device=device),
        'valid_mask': valid_mask,
        'assignment_misses': torch.tensor(assignment_misses, dtype=torch.long, device=device),
        'spatial_shapes': torch.tensor(shapes, dtype=torch.long, device=device),
        'input_size': (input_h, input_w),
    }


@register()
class SGQSQuerySupervisor(nn.Module):
    """Build the training-only payload consumed by :class:`SGQSCriterion`."""

    def __init__(
        self,
        scale_priors=(32.0, 64.0, 128.0),
        initial_radius=1,
        max_radius=4,
        ignore_radius=1,
    ):
        super().__init__()
        self.scale_priors = tuple(float(value) for value in scale_priors)
        self.initial_radius = int(initial_radius)
        self.max_radius = int(max_radius)
        self.ignore_radius = int(ignore_radius)

    def forward(
        self,
        output_memory: torch.Tensor,
        dense_logits: torch.Tensor,
        selection_logits: torch.Tensor,
        anchors_unact: torch.Tensor,
        valid_mask: torch.Tensor,
        spatial_shapes,
        targets,
        bbox_head: nn.Module,
        feat_strides: Sequence[int],
        num_queries: int,
        query_select_method: str,
    ) -> Dict[str, object]:
        if targets is None:
            raise ValueError('SGQS training requires augmented targets')

        assignment = assign_scale_aware_owners(
            targets=targets,
            spatial_shapes=spatial_shapes,
            feat_strides=feat_strides,
            valid_mask=valid_mask,
            scale_priors=self.scale_priors,
            initial_radius=self.initial_radius,
            max_radius=self.max_radius,
            ignore_radius=self.ignore_radius,
        )

        if anchors_unact.shape[0] == 1 and output_memory.shape[0] > 1:
            anchors_unact = anchors_unact.expand(output_memory.shape[0], -1, -1)

        owner_boxes = []
        for batch_idx, indices in enumerate(assignment['owner_indices']):
            owner_memory = output_memory[batch_idx].index_select(0, indices)
            owner_anchors = anchors_unact[batch_idx].index_select(0, indices)
            owner_boxes.append(torch.sigmoid(bbox_head(owner_memory) + owner_anchors))

        if query_select_method == 'default':
            selection_pool = selection_logits.max(dim=-1).values
        elif query_select_method == 'agnostic':
            selection_pool = selection_logits.squeeze(-1)
        elif query_select_method == 'one2many':
            selection_pool = selection_logits.flatten(1)
        else:
            raise ValueError(f'Unsupported query_select_method: {query_select_method}')

        effective_k = min(int(num_queries), int(selection_pool.shape[1]))
        if effective_k <= 0:
            raise ValueError('num_queries and the selection pool must be non-empty')
        topk_result = torch.topk(selection_pool, effective_k, dim=1)
        if not torch.isfinite(topk_result.values).all():
            raise ValueError('SGQS query budget exceeds the finite valid-anchor selection pool')
        topk_cutoff = topk_result.values[:, -1].detach()
        selected_indices = topk_result.indices
        if query_select_method == 'one2many':
            selected_indices = selected_indices // dense_logits.shape[-1]

        return {
            'dense_logits': dense_logits,
            'owner_boxes': owner_boxes,
            'owner_indices': assignment['owner_indices'],
            'target_indices': assignment['target_indices'],
            'owner_levels': assignment['owner_levels'],
            'ignore_mask': assignment['ignore_mask'],
            'valid_mask': assignment['valid_mask'],
            'assignment_misses': assignment['assignment_misses'],
            'spatial_shapes': assignment['spatial_shapes'],
            'topk_cutoff': topk_cutoff,
            'selection_scores': selection_pool.detach(),
            'query_select_method': query_select_method,
            'selected_indices': selected_indices.detach(),
            'effective_k': effective_k,
            'selection_pool_size': int(selection_pool.shape[1]),
        }
