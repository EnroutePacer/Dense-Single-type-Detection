"""SGQS-E: detached, bounded edge weights on D-FINE's existing FGL.

Only training losses use this module. All geometry uses augmented input pixels;
the original distribution targets, matching, DN sampling and DDF are untouched.
"""

from dataclasses import dataclass
import math

import torch

from ..deim.box_ops import box_cxcywh_to_xyxy


@dataclass(frozen=True)
class EFGLConfig:
    enabled: bool = False
    strength: float = 0.5
    neighbor_strength: float = 0.5
    short_side_scale: float = 16.0
    pixel_floor: float = 2.0
    neighbors: int = 4
    neighbor_radius: float = 4.0
    compensation_fraction: float = 0.3
    compensation_start_epoch: float = 0.0
    compensation_duration_epochs: float | None = None
    normal_branch: bool = True
    dn_branch: bool = True

    def __post_init__(self):
        for key in ('enabled', 'normal_branch', 'dn_branch'):
            if not isinstance(getattr(self, key), bool):
                raise ValueError(f'EFGL {key} must be boolean')
        for key in ('strength', 'neighbor_strength'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'EFGL {key} must be in [0, 1]')
        for key in ('short_side_scale', 'pixel_floor', 'neighbor_radius'):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'EFGL {key} must be finite and positive')
        if not 0 < self.compensation_fraction <= 1:
            raise ValueError('EFGL compensation_fraction must be in (0, 1]')
        if (not math.isfinite(self.compensation_start_epoch)
                or self.compensation_start_epoch < 0):
            raise ValueError('EFGL compensation_start_epoch must be finite and non-negative')
        if (self.compensation_duration_epochs is not None
                and (not math.isfinite(self.compensation_duration_epochs)
                     or self.compensation_duration_epochs <= 0)):
            raise ValueError('EFGL compensation_duration_epochs must be finite and positive')
        if isinstance(self.neighbors, bool) or not isinstance(self.neighbors, int) or self.neighbors < 0:
            raise ValueError('EFGL neighbors must be a non-negative integer')


@torch.no_grad()
def build_edge_context(targets, input_size, config, epoch, step, epoch_step, total_epochs):
    if input_size is None or len(input_size) != 2:
        raise ValueError('EFGL needs input_size=(H,W) from actual augmented samples')
    height, width = input_size
    if height <= 0 or width <= 0 or total_epochs <= 0 or epoch_step <= 0:
        raise ValueError('EFGL image size, total epochs and epoch_step must be positive')
    progress = float(epoch) + float(step) / float(epoch_step)
    if not math.isfinite(progress) or progress < 0:
        raise ValueError('EFGL requires finite, non-negative training progress')
    duration = config.compensation_duration_epochs
    if duration is None:
        duration = total_epochs * config.compensation_fraction
    if progress < config.compensation_start_epoch:
        ramp = 0.0
    else:
        ramp = max(0.0, 1.0 - (progress - config.compensation_start_epoch) / duration)
    context = dict(config=config, ramp=ramp, graphs=[])
    if ramp == 0 or config.strength == 0:
        return context  # Exact original FGL; no neighbor construction after annealing.
    for target in targets:
        boxes = target['boxes'].detach().float()
        labels = target['labels']
        if boxes.shape != (len(labels), 4):
            raise ValueError('EFGL boxes must have shape [number of labels, 4]')
        if not torch.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
            raise ValueError('EFGL requires finite boxes with positive width and height')
        pixels = boxes.new_tensor([width, height, width, height])
        corners = box_cxcywh_to_xyxy(boxes) * pixels
        sides = boxes[:, 2:] * pixels[:2]
        dimensions = sides.repeat(1, 2).clamp_min(config.pixel_floor)
        count = len(boxes)
        k = min(config.neighbors, max(0, count - 1)) if config.neighbor_strength else 0
        neighbor_rows, valid_rows = [], []
        # O(G^2) distances, but bounded temporary memory instead of [G,G,4].
        for start in range(0, count if k else 0, 256):
            delta = ((corners[start:start+256, None] - corners[None]).abs()
                     / dimensions[start:start+256, None]).sum(-1)
            valid = ((labels[start:start+256, None] == labels[None])
                     & (delta > 1e-6) & (delta <= config.neighbor_radius))
            distances, neighbors = delta.masked_fill(~valid, float('inf')).topk(k, largest=False)
            neighbor_rows.append(neighbors)
            valid_rows.append(torch.isfinite(distances))
        neighbors = (torch.cat(neighbor_rows) if neighbor_rows else
                     torch.empty((count, k), dtype=torch.long, device=boxes.device))
        valid = (torch.cat(valid_rows) if valid_rows else
                 torch.empty((count, k), dtype=torch.bool, device=boxes.device))
        neighbor_boxes = corners[neighbors]
        edge_distance = (neighbor_boxes - corners[:, None]).abs() / dimensions[:, None]
        separation = edge_distance.sum(-1)
        # Invalid/duplicate pairs are masked, not biased by adding epsilon to valid distances.
        safe_separation = torch.where(valid, separation, torch.ones_like(separation))
        context['graphs'].append(dict(
            corners=corners, dimensions=dimensions, pixels=pixels,
            smallness=torch.exp(-sides.amin(-1) / config.short_side_scale),
            neighbor_boxes=neighbor_boxes, valid=valid,
            separation=safe_separation, edge_share=edge_distance / safe_separation[..., None]))
    return context


@torch.no_grad()
def edge_weights(context, predicted_boxes, indices, ious, is_dn=False):
    """Use exactly the supplied base-criterion owners, including GO union / DN."""
    config = context['config']
    original = ious.detach().reshape(-1, 1).expand(-1, 4)
    enabled = config.dn_branch if is_dn else config.normal_branch
    if not enabled or context['ramp'] == 0 or config.strength == 0:
        return original
    bonuses = []
    for graph, boxes, (queries, owners) in zip(context['graphs'], predicted_boxes, indices):
        corners = box_cxcywh_to_xyxy(boxes[queries].detach().float()) * graph['pixels']
        relative = (corners - graph['corners'][owners]).abs() / graph['dimensions'][owners]
        need = relative / (1.0 + relative)
        if graph['neighbor_boxes'].shape[1]:
            own_distance = relative.sum(-1)
            other_distance = ((corners[:, None] - graph['neighbor_boxes'][owners]).abs()
                              / graph['dimensions'][owners, None]).sum(-1)
            rho = (other_distance - own_distance[:, None]) / graph['separation'][owners]
            # Fixed kappa=1/2. One compensation schedule is enough; no second annealing.
            ambiguity = (1.0 - 2.0 * rho).clamp(0, 1) * graph['valid'][owners]
            edge_ambiguity = (ambiguity[..., None] * graph['edge_share'][owners]).amax(1)
            need = (need + config.neighbor_strength * edge_ambiguity).clamp(max=1)
        bonuses.append(graph['smallness'][owners, None] * need)
    bonus = torch.cat(bonuses)
    original = original.float()
    return original + config.strength * context['ramp'] * (1.0 - original) * bonus
