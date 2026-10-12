"""Losses for Scale-aware GT Query Survival (SGQS)."""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import register
from ..deim.box_ops import box_cxcywh_to_xyxy
from ..misc.box_ops import elementwise_box_iou, elementwise_generalized_box_iou
from ..misc.dist_utils import get_world_size, is_dist_available_and_initialized


@register()
class SGQSCriterion(nn.Module):
    """Wrap the unmodified base criterion with one pre-Top-K SGQS branch."""

    __inject__ = ['base_criterion']

    def __init__(
        self,
        base_criterion: nn.Module,
        weight_dict,
        q_min=0.25,
        dense_alpha=0.75,
        dense_gamma=2.0,
        negative_weight=1.0,
        survival_margin=0.2,
        survival_temperature=0.1,
        survival_quality_power=1.0,
        survival_start_epoch=0.0,
        survival_ramp_epochs=1.0,
        survival_threshold='topk',
        protect_encoder_supervision=False,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.weight_dict = dict(weight_dict)
        self.q_min = float(q_min)
        self.dense_alpha = float(dense_alpha)
        self.dense_gamma = float(dense_gamma)
        self.negative_weight = float(negative_weight)
        self.survival_margin = float(survival_margin)
        self.survival_temperature = float(survival_temperature)
        self.survival_quality_power = float(survival_quality_power)
        self.survival_start_epoch = float(survival_start_epoch)
        self.survival_ramp_epochs = float(survival_ramp_epochs)
        self.survival_threshold = survival_threshold
        self.protect_encoder_supervision = bool(protect_encoder_supervision)
        self._logging_metrics: Dict[str, torch.Tensor] = {}

        required = {
            'loss_sgqs_dense',
            'loss_sgqs_bbox',
            'loss_sgqs_giou',
            'loss_sgqs_survival',
        }
        missing = required.difference(self.weight_dict)
        if missing:
            raise ValueError(f'SGQS weight_dict is missing: {sorted(missing)}')
        if not 0.0 < self.q_min <= 1.0:
            raise ValueError('q_min must be in (0, 1]')
        if self.dense_alpha < 0.0 or self.dense_gamma < 0.0:
            raise ValueError('dense focal parameters must be non-negative')
        if self.negative_weight < 0.0:
            raise ValueError('negative_weight must be non-negative')
        if self.survival_temperature <= 0.0:
            raise ValueError('survival_temperature must be positive')
        if self.survival_quality_power < 0.0:
            raise ValueError('survival_quality_power must be non-negative')
        if self.survival_ramp_epochs < 0.0:
            raise ValueError('survival_ramp_epochs must be non-negative')
        if self.survival_threshold not in ('topk', 'non_owner_budget'):
            raise ValueError('survival_threshold must be topk or non_owner_budget')

    @staticmethod
    def _distributed_average_denominator(value, device, minimum=1e-12):
        denominator = torch.as_tensor(value, dtype=torch.float32, device=device).detach().clone()
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(denominator)
        denominator = denominator / get_world_size()
        return denominator.clamp_min(float(minimum))

    @staticmethod
    @torch.no_grad()
    def _distributed_sum(values: torch.Tensor) -> torch.Tensor:
        values = values.detach().to(dtype=torch.float32).clone()
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(values)
        return values

    def _survival_ramp(self, metadata) -> float:
        if 'epoch' not in metadata:
            return 1.0
        epoch = float(metadata['epoch'])
        step = float(metadata.get('step', 0))
        epoch_step = max(float(metadata.get('epoch_step', 1)), 1.0)
        progress = epoch + step / epoch_step
        if self.survival_ramp_epochs == 0.0:
            return float(progress >= self.survival_start_epoch)
        value = (progress - self.survival_start_epoch) / self.survival_ramp_epochs
        return max(0.0, min(1.0, value))

    def _quality_target(self, quality: torch.Tensor) -> torch.Tensor:
        return self.q_min + (1.0 - self.q_min) * quality

    def _owner_quality_and_regression(self, payload, targets):
        dense_logits = payload['dense_logits'].float()
        zero = dense_logits.sum() * 0.0
        l1_sum = zero
        giou_sum = zero
        owner_count = 0
        qualities: List[torch.Tensor] = []

        for batch_idx, owner_boxes_raw in enumerate(payload['owner_boxes']):
            owner_boxes = owner_boxes_raw.float()
            target_indices = payload['target_indices'][batch_idx]
            if target_indices.numel() == 0:
                qualities.append(owner_boxes.new_zeros((0,)))
                continue
            target_boxes = targets[batch_idx]['boxes'].index_select(0, target_indices).float()
            pred_xyxy = box_cxcywh_to_xyxy(owner_boxes)
            target_xyxy = box_cxcywh_to_xyxy(target_boxes)
            iou, _ = elementwise_box_iou(pred_xyxy, target_xyxy)
            quality = iou.clamp(min=0.0, max=1.0).detach()
            qualities.append(quality)
            l1_sum = l1_sum + F.l1_loss(owner_boxes, target_boxes, reduction='none').sum(dim=-1).sum()
            giou_sum = giou_sum + (1.0 - elementwise_generalized_box_iou(pred_xyxy, target_xyxy)).sum()
            owner_count += int(target_indices.numel())

        denominator = self._distributed_average_denominator(owner_count, dense_logits.device)
        return qualities, l1_sum / denominator, giou_sum / denominator

    def _dense_loss(self, payload, targets, qualities):
        dense_logits = payload['dense_logits'].float()
        valid_mask = payload['valid_mask'].to(dtype=torch.bool, device=dense_logits.device)
        ignore_mask = payload['ignore_mask'].to(dtype=torch.bool, device=dense_logits.device)
        spatial_shapes = payload['spatial_shapes'].detach().cpu().tolist()
        num_levels = len(spatial_shapes)
        num_classes = dense_logits.shape[-1]

        positive_sums = [dense_logits.sum() * 0.0 for _ in range(num_levels)]
        negative_sums = [dense_logits.sum() * 0.0 for _ in range(num_levels)]
        positive_counts = [0 for _ in range(num_levels)]
        negative_counts = [0 for _ in range(num_levels)]

        offset = 0
        level_ranges = []
        for height, width in spatial_shapes:
            size = int(height) * int(width)
            level_ranges.append((offset, offset + size))
            offset += size

        for batch_idx in range(dense_logits.shape[0]):
            owner_indices = payload['owner_indices'][batch_idx]
            target_indices = payload['target_indices'][batch_idx]
            owner_levels = payload['owner_levels'][batch_idx]
            labels = targets[batch_idx]['labels'].index_select(0, target_indices).long()
            if labels.numel() and ((labels < 0).any() or (labels >= num_classes).any()):
                raise ValueError('SGQS owner label is outside dense_logits class range')

            for level, (start, end) in enumerate(level_ranges):
                logits_l = dense_logits[batch_idx, start:end]
                negative_mask = (
                    valid_mask[batch_idx, start:end].unsqueeze(-1)
                    & ~ignore_mask[batch_idx, start:end].unsqueeze(-1)
                ).expand(-1, num_classes).clone()

                level_owner_mask = owner_levels == level
                if level_owner_mask.any():
                    owner_global = owner_indices[level_owner_mask]
                    owner_local = owner_global - start
                    owner_labels = labels[level_owner_mask]
                    owner_quality = qualities[batch_idx][level_owner_mask]
                    quality_target = self._quality_target(owner_quality)
                    owner_logits = logits_l[owner_local, owner_labels]
                    positive_loss = quality_target * F.binary_cross_entropy_with_logits(
                        owner_logits,
                        quality_target,
                        reduction='none',
                    )
                    positive_sums[level] = positive_sums[level] + positive_loss.sum()
                    positive_counts[level] += int(owner_logits.numel())
                    negative_mask[owner_local, owner_labels] = False

                negative_logits = logits_l[negative_mask]
                if negative_logits.numel():
                    negative_loss = (
                        self.dense_alpha
                        * negative_logits.sigmoid().pow(self.dense_gamma)
                        * F.binary_cross_entropy_with_logits(
                            negative_logits,
                            torch.zeros_like(negative_logits),
                            reduction='none',
                        )
                    )
                    negative_sums[level] = negative_sums[level] + negative_loss.sum()
                    negative_counts[level] += int(negative_logits.numel())

        loss = dense_logits.sum() * 0.0
        for level in range(num_levels):
            positive_denominator = self._distributed_average_denominator(
                positive_counts[level], dense_logits.device)
            negative_denominator = self._distributed_average_denominator(
                negative_counts[level], dense_logits.device)
            loss = loss + positive_sums[level] / positive_denominator
            loss = loss + self.negative_weight * negative_sums[level] / negative_denominator
        return loss / max(num_levels, 1)

    def _survival_loss(self, payload, targets, qualities, metadata):
        dense_logits = payload['dense_logits'].float()
        cutoff = payload['topk_cutoff'].float().detach()
        effective_k = int(payload['effective_k'])
        pool_size = int(payload['selection_pool_size'])
        weighted_sum = dense_logits.sum() * 0.0
        weight_sum = dense_logits.new_zeros(())

        for batch_idx in range(dense_logits.shape[0]):
            gt_count = int(targets[batch_idx]['labels'].numel())
            if gt_count == 0 or gt_count > effective_k or effective_k >= pool_size:
                continue
            owner_indices = payload['owner_indices'][batch_idx]
            target_indices = payload['target_indices'][batch_idx]
            if owner_indices.numel() == 0:
                continue
            labels = targets[batch_idx]['labels'].index_select(0, target_indices).long()
            owner_logits = dense_logits[batch_idx, owner_indices, labels]
            quality_weights = self._quality_target(
                qualities[batch_idx]).pow(self.survival_quality_power).detach()
            threshold = cutoff[batch_idx]
            if self.survival_threshold == 'non_owner_budget':
                threshold = self._competition_threshold(payload, batch_idx)
                if threshold is None:
                    continue
            margin_gap = threshold + self.survival_margin - owner_logits
            survival = self.survival_temperature * F.softplus(
                margin_gap / self.survival_temperature)
            weighted_sum = weighted_sum + (quality_weights * survival).sum()
            weight_sum = weight_sum + quality_weights.sum()

        denominator = self._distributed_average_denominator(
            weight_sum, dense_logits.device, minimum=1e-12)
        return weighted_sum / denominator * self._survival_ramp(metadata)

    @staticmethod
    @torch.no_grad()
    def _competition_threshold(payload, batch_idx):
        """All A owners fit in K if they beat non-owner rank K-A+1.

        Excluding owners removes the self-chasing margin at A=K. Scores are
        exactly those of the masked selector; the real Top-K cutoff is logged
        separately. A is the active owner count after the supervision guard.
        """
        if payload['dense_logits'].shape[-1] != 1:
            raise ValueError('non_owner_budget is defined for single-type detection')
        scores = payload['selection_scores'][batch_idx].float().detach()
        owners = payload['owner_indices'][batch_idx]
        valid = torch.isfinite(scores)
        k = int(payload['effective_k'])
        if k > int(valid.sum()):
            raise ValueError('query budget exceeds valid selector scores')
        if not len(owners) or k >= int(valid.sum()):
            return None
        valid = valid.clone()
        valid[owners] = False
        rank = k - len(owners) + 1
        if rank <= 0:
            return None
        competitors = scores[valid]
        if rank > len(competitors):
            return None
        return competitors.topk(rank).values[-1]

    def _guard_encoder_supervision(self, payload):
        """Avoid adding opposite GT/background supervision on selected memory.

        Non-owner encoder positives are not SGQS negatives. A selected owner
        is omitted from SGQS if the original encoder class/box supervision
        disagrees with its target (including background class supervision).
        Original matcher outputs and every base loss remain unchanged.
        """
        if payload['dense_logits'].shape[-1] != 1:
            raise ValueError('encoder supervision protection requires a single class')
        if not hasattr(self.base_criterion, 'get_encoder_supervision'):
            raise ValueError('SGQS-E requires the base criterion encoder assignments')
        records = self.base_criterion.get_encoder_supervision()
        if len(records) != 1:
            raise ValueError('SGQS-E expects exactly one encoder candidate output')
        result = dict(payload)
        result['ignore_mask'] = payload['ignore_mask'].clone()
        for key in ('owner_indices', 'target_indices', 'owner_levels', 'owner_boxes'):
            result[key] = []
        conflicts = payload['dense_logits'].new_zeros(())
        for batch, owners in enumerate(payload['owner_indices']):
            targets = payload['target_indices'][batch]
            selected = payload['selected_indices'][batch]
            size = payload['dense_logits'].shape[1]
            class_target = torch.full((size,), -2, dtype=torch.long, device=owners.device)
            class_target[selected] = -1  # selected but unmatched = base background
            cls_queries, cls_targets = records[0]['classification'][batch]
            # SciPy Hungarian matching returns CPU indices, even for CUDA models.
            class_target[selected[cls_queries]] = cls_targets.to(device=owners.device)
            box_target = torch.full_like(class_target, -2)
            box_queries, box_targets = records[0]['boxes'][batch]
            box_target[selected[box_queries]] = box_targets.to(device=owners.device)
            conflict = ((class_target[owners] != -2) & (class_target[owners] != targets))
            conflict |= ((box_target[owners] >= 0) & (box_target[owners] != targets))
            keep = ~conflict
            # Preserve base positives and suppress contradictory SGQS positives,
            # while leaving safe background negatives supervised.
            result['ignore_mask'][batch, selected[cls_queries]] = True
            result['ignore_mask'][batch, selected[box_queries]] = True
            result['ignore_mask'][batch, owners[conflict]] = True
            for key in ('owner_indices', 'target_indices', 'owner_levels', 'owner_boxes'):
                result[key].append(payload[key][batch][keep])
            conflicts += conflict.sum()
        result['supervision_conflicts'] = conflicts
        return result

    @torch.no_grad()
    def _update_logging_metrics(self, payload, targets, qualities, metadata):
        device = payload['dense_logits'].device
        gt_counts = [int(target['labels'].numel()) for target in targets]
        owner_count = sum(int(indices.numel()) for indices in payload['owner_indices'])
        assignment_misses = int(payload['assignment_misses'].sum().item())
        effective_k = int(payload['effective_k'])
        nonempty_images = sum(count > 0 for count in gt_counts)
        budget_overflows = sum(count > effective_k for count in gt_counts)
        level_count = int(payload['spatial_shapes'].shape[0])
        owner_level_counts = [0 for _ in range(level_count)]
        quality_sum = payload['dense_logits'].new_zeros(())
        eligible_owner_count = 0
        survived_owner_count = 0
        cutoff_gap_sum = payload['dense_logits'].new_zeros(())

        for batch_idx, owner_indices in enumerate(payload['owner_indices']):
            for level in range(level_count):
                owner_level_counts[level] += int(
                    (payload['owner_levels'][batch_idx] == level).sum().item())
            quality_sum = quality_sum + qualities[batch_idx].sum()

            gt_count = gt_counts[batch_idx]
            if (
                owner_indices.numel() == 0
                or gt_count == 0
                or gt_count > effective_k
                or effective_k >= int(payload['selection_pool_size'])
            ):
                continue
            target_indices = payload['target_indices'][batch_idx]
            labels = targets[batch_idx]['labels'].index_select(0, target_indices).long()
            owner_logits = payload['dense_logits'][batch_idx, owner_indices, labels].detach()
            selected = payload['selected_indices'][batch_idx]
            survived = (owner_indices[:, None] == selected[None, :]).any(dim=1)
            eligible_owner_count += int(owner_indices.numel())
            survived_owner_count += int(survived.sum().item())
            cutoff_gap_sum = cutoff_gap_sum + (
                owner_logits - payload['topk_cutoff'][batch_idx]).sum()

        totals = self._distributed_sum(torch.tensor(
            [
                sum(gt_counts),
                owner_count,
                assignment_misses,
                len(targets),
                nonempty_images,
                budget_overflows,
            ],
            dtype=torch.float32,
            device=device,
        ))
        total_gt, owners, misses, images, nonempty, overflows = totals
        mechanism_totals = self._distributed_sum(torch.stack([
            quality_sum.to(dtype=torch.float32),
            torch.tensor(float(eligible_owner_count), device=device),
            torch.tensor(float(survived_owner_count), device=device),
            cutoff_gap_sum.to(dtype=torch.float32),
            *[
                torch.tensor(float(count), device=device)
                for count in owner_level_counts
            ],
        ]))
        quality_total, eligible_owners, survived_owners, cutoff_gap_total = mechanism_totals[:4]
        self._logging_metrics = {
            'sgqs_assignment_miss_rate': misses / total_gt.clamp_min(1.0),
            'sgqs_owner_coverage': owners / total_gt.clamp_min(1.0),
            'sgqs_owner_count': owners / images.clamp_min(1.0),
            'sgqs_budget_overflow_rate': overflows / nonempty.clamp_min(1.0),
            'sgqs_owner_iou': quality_total / owners.clamp_min(1.0),
            'sgqs_pre_topk_survival_rate': (
                survived_owners / eligible_owners.clamp_min(1.0)),
            'sgqs_owner_cutoff_gap': (
                cutoff_gap_total / eligible_owners.clamp_min(1.0)),
            'sgqs_survival_ramp': torch.tensor(
                self._survival_ramp(metadata), dtype=torch.float32, device=device),
        }
        if 'supervision_conflicts' in payload:
            conflicts = self._distributed_sum(payload['supervision_conflicts'])
            self._logging_metrics['sgqs_conflicting_owner_fraction'] = (
                conflicts / (owners + conflicts).clamp_min(1.0))
        for level, count in enumerate(mechanism_totals[4:]):
            self._logging_metrics[f'sgqs_owner_p{level + 3}_count'] = (
                count / images.clamp_min(1.0))

    def get_logging_metrics(self) -> Dict[str, torch.Tensor]:
        """Return synchronized, detached diagnostics that are not losses."""
        metrics = (self.base_criterion.get_logging_metrics()
                   if hasattr(self.base_criterion, 'get_logging_metrics') else {})
        return {**metrics, **self._logging_metrics}

    def loss_sgqs(self, payload, targets, **metadata) -> Dict[str, torch.Tensor]:
        qualities, loss_bbox, loss_giou = self._owner_quality_and_regression(payload, targets)
        losses = {
            'loss_sgqs_dense': self._dense_loss(payload, targets, qualities),
            'loss_sgqs_bbox': loss_bbox,
            'loss_sgqs_giou': loss_giou,
            'loss_sgqs_survival': self._survival_loss(payload, targets, qualities, metadata),
        }
        for name, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f'Non-finite SGQS loss: {name}')
        self._update_logging_metrics(payload, targets, qualities, metadata)
        return losses

    def forward(self, outputs, targets, **metadata):
        self._logging_metrics = {}
        losses = self.base_criterion(outputs, targets, **metadata)
        payload = outputs.get('sgqs_aux_outputs')
        if payload is None:
            return losses

        if self.protect_encoder_supervision:
            payload = self._guard_encoder_supervision(payload)

        sgqs_losses = self.loss_sgqs(payload, targets, **metadata)
        losses.update({name: value * self.weight_dict[name] for name, value in sgqs_losses.items()})
        return losses
