"""Geometry support and detached, GT-constrained Encoder budget allocation."""
import math

import torch
import torch.nn.functional as F

from ..core import register
from ..misc.dist_utils import get_world_size
from ..sgqs.criterion import SGQSCriterion
from .allocation import STAT_NAMES, allocation_loss, build_allocation


@register()
class SGQSV2Criterion(SGQSCriterion):
    __inject__ = ['base_criterion']

    def __init__(self, base_criterion, weight_dict, q_min=.25,
                 dense_alpha=.75, dense_gamma=2., negative_weight=1.,
                 start_epoch=28., ramp_epochs=1., allocation_weight=.25,
                 iou_threshold=.5, ambiguity_gap=.05, teacher_score=.5,
                 background_score=.1, margin=.2, iou_chunk_size=512,
                 redundancy_ios_threshold=.8, overflow_tolerance=10,
                 overflow_score_threshold=.8):
        required = {'loss_sgqs_dense', 'loss_sgqs_bbox', 'loss_sgqs_giou'}
        if set(weight_dict) != required:
            raise ValueError('v2 requires exactly dense/bbox/giou weights; no owner survival')
        scalars = [*weight_dict.values(), q_min, dense_alpha, dense_gamma,
                   negative_weight, start_epoch, ramp_epochs, allocation_weight,
                   iou_threshold, ambiguity_gap, teacher_score, background_score, margin,
                   redundancy_ios_threshold, overflow_tolerance, overflow_score_threshold]
        if not all(math.isfinite(float(v)) and float(v) >= 0 for v in scalars):
            raise ValueError('SGQS v2 parameters must be finite and non-negative')
        if not 0 < iou_threshold <= 1 or not 0 < ambiguity_gap <= 1:
            raise ValueError('IoU and ambiguity thresholds must be in (0, 1]')
        if not 0 <= background_score < teacher_score <= 1:
            raise ValueError('Require 0 <= background_score < teacher_score <= 1')
        if not 0 < redundancy_ios_threshold <= 1 or not 0 < overflow_score_threshold < 1:
            raise ValueError('Require IoS in (0, 1] and overflow score in (0, 1)')
        if isinstance(overflow_tolerance, bool) or int(overflow_tolerance) != overflow_tolerance:
            raise ValueError('overflow_tolerance must be a non-negative integer')
        if isinstance(iou_chunk_size, bool) or int(iou_chunk_size) != iou_chunk_size or iou_chunk_size < 1:
            raise ValueError('iou_chunk_size must be a positive integer')
        super().__init__(base_criterion, {**weight_dict, 'loss_sgqs_survival': 0.},
                         q_min=q_min, dense_alpha=dense_alpha, dense_gamma=dense_gamma,
                         negative_weight=negative_weight, protect_encoder_supervision=True)
        self.weight_dict = dict(weight_dict)
        self.start_epoch, self.ramp_epochs = float(start_epoch), float(ramp_epochs)
        self.allocation_weight, self.margin = float(allocation_weight), float(margin)
        self.overflow_score_threshold = float(overflow_score_threshold)
        self.allocation_options = dict(iou_threshold=float(iou_threshold),
            ambiguity_gap=float(ambiguity_gap), teacher_score=float(teacher_score),
            background_score=float(background_score), margin=self.margin,
            iou_chunk_size=int(iou_chunk_size),
            redundancy_ios_threshold=float(redundancy_ios_threshold),
            overflow_tolerance=int(overflow_tolerance),
            overflow_score_threshold=self.overflow_score_threshold)

    def _ramp(self, metadata):
        if not {'epoch', 'step', 'epoch_step'} <= metadata.keys():
            raise ValueError('SGQS v2 requires absolute epoch/step/epoch_step metadata')
        epoch, step, length = (float(metadata[k]) for k in ('epoch', 'step', 'epoch_step'))
        if not all(math.isfinite(x) for x in (epoch, step, length)) or length <= 0 or not 0 <= step < length:
            raise ValueError('Invalid absolute epoch/step metadata')
        progress = epoch + step / length
        if self.ramp_epochs == 0:
            return float(progress >= self.start_epoch)
        return max(0., min(1., (progress - self.start_epoch) / self.ramp_epochs))

    def _dense_loss(self, payload, targets, qualities):
        """IoU targets, a global owner mean and the original per-level negatives.

        Rescue receivers leave both auxiliary classification sets this step;
        their original geometry supervision stays in owner_boxes unchanged.
        """
        logits = payload['dense_logits'].float()
        if logits.shape[-1] != 1:
            raise ValueError('SGQS v2 dense supervision requires one class')
        valid, ignore = payload['valid_mask'].bool(), payload['ignore_mask'].bool()
        skip = payload.get('classification_skip', torch.zeros_like(valid))
        shapes = payload['spatial_shapes'].detach().cpu().tolist()
        positive, count = logits.sum() * 0., 0
        owner_mask = torch.zeros_like(valid)
        for batch, owners in enumerate(payload['owner_indices']):
            owner_mask[batch, owners] = True
            keep = ~skip[batch, owners]
            iou = qualities[batch][keep].detach()
            z = logits[batch, owners[keep], 0]
            positive = positive + (self._quality_target(iou) * F.binary_cross_entropy_with_logits(
                z, iou, reduction='none')).sum()
            count += len(z)
        result = positive / self._distributed_average_denominator(count, logits.device)
        negatives = valid & ~ignore & ~owner_mask & ~skip
        offset = 0
        for height, width in shapes:
            end = offset + int(height) * int(width)
            z = logits[:, offset:end, 0][negatives[:, offset:end]]
            numerator = (self.dense_alpha * z.sigmoid().pow(self.dense_gamma)
                         * F.binary_cross_entropy_with_logits(z, torch.zeros_like(z), reduction='none')).sum()
            denominator = self._distributed_average_denominator(z.numel(), logits.device)
            result = result + self.negative_weight * numerator / denominator / len(shapes)
            offset = end
        if offset != logits.shape[1]:
            raise ValueError('Spatial shapes do not cover the dense candidate axis')
        return result

    def loss_sgqs(self, payload, targets, **metadata):
        qualities, l1, giou = self._owner_quality_and_regression(payload, targets)
        self._owner_iou_sum = sum((x.detach().sum() for x in qualities), l1.detach() * 0.)
        return {'loss_sgqs_dense': self._dense_loss(payload, targets, qualities),
                'loss_sgqs_bbox': l1, 'loss_sgqs_giou': giou}

    def forward(self, outputs, targets, **metadata):
        self._logging_metrics = {}
        payload = outputs.get('sgqs_aux_outputs')
        if payload is None:
            if self.training:
                raise ValueError('Training SGQS v2 requires SGQSV2Supervisor payload')
            return self.base_criterion(outputs, targets, **metadata)
        ramp = self._ramp(metadata)
        captured = []

        def capture(_module, args, result):
            if not captured:
                if args[0]['pred_logits'] is not outputs['pred_logits']:
                    raise ValueError('Expected final Decoder matching before auxiliary matches')
                captured.append(result['indices'])

        handle = self.base_criterion.matcher.register_forward_hook(capture)
        try:
            losses = self.base_criterion(outputs, targets, **metadata)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError('The base criterion did not produce final matching')
        def owner_counts(data):
            total, small = 0, 0
            for target, ids in zip(targets, data['target_indices']):
                boxes = target['boxes'][ids].detach()
                # Declared local definition, normalized geometry mapped to 640.
                w, h = boxes[:, 2] * 640, boxes[:, 3] * 640
                total += len(ids)
                small += int(((w * h < 32**2) | (torch.minimum(w, h) < 16)).sum())
            return total, small
        before = owner_counts(payload)
        payload = self._guard_encoder_supervision(payload)
        after = owner_counts(payload)
        self._owner_counts = torch.tensor([*before, after[1]],
            device=payload['dense_logits'].device, dtype=torch.float32)
        records = self.base_criterion.get_encoder_supervision()
        pairs, overflow, stats = build_allocation(payload, outputs, targets, captured[0], records[0],
                                       enabled=self.allocation_weight > 0, **self.allocation_options)
        payload = dict(payload)
        payload['classification_skip'] = torch.zeros_like(payload['valid_mask'], dtype=torch.bool)
        for batch, (_, receivers, rescue) in enumerate(pairs):
            payload['classification_skip'][batch, receivers[rescue]] = True
        auxiliary = self.loss_sgqs(payload, targets, **metadata)
        allocate, _ = allocation_loss(payload['dense_logits'], pairs, self.margin,
                                     self._distributed_average_denominator, overflow,
                                     self.overflow_score_threshold)
        for key, value in auxiliary.items():
            losses[key] = value * self.weight_dict[key] * ramp
        losses['loss_sgqs_allocation'] = allocate * self.allocation_weight * ramp
        # All ranks must take the same branch before metric collectives.
        from .runtime import require_finite
        require_finite(torch.stack([*auxiliary.values(), allocate]),
                       payload['dense_logits'].device, 'SGQS v2 auxiliary loss')
        self._record_metrics(payload, stats, auxiliary, allocate, ramp)
        return losses

    @torch.no_grad()
    def _record_metrics(self, payload, stats, auxiliary, allocate, ramp):
        device = payload['dense_logits'].device
        owner_count = sum(len(x) for x in payload['owner_indices'])
        local_extra = torch.tensor([owner_count, float(payload['classification_skip'].sum())], device=device)
        vector = torch.stack([stats[name] for name in STAT_NAMES] + [
            local_extra[0], local_extra[1], payload['supervision_conflicts'], self._owner_iou_sum,
            *[x.detach() for x in auxiliary.values()], allocate.detach(), *self._owner_counts])
        totals = self._distributed_sum(vector)
        values = dict(zip(STAT_NAMES, totals[:len(STAT_NAMES)]))
        owners, masked, conflicts, owner_iou = totals[len(STAT_NAMES):len(STAT_NAMES)+4]
        raw_losses = totals[len(STAT_NAMES)+4:len(STAT_NAMES)+8]
        before, small_before, small_after = totals[-3:]
        images = values['images'].clamp_min(1.)
        pair_count = values['rescue_pairs'] + values['prefix_pairs']
        # Global sums give interpretable denominators, not per-rank mean-of-means.
        metrics = {f'v2_{name}': value for name, value in values.items()}
        metrics.update(v2_ramp=torch.tensor(ramp, device=device),
            v2_pair_count=pair_count, v2_pairs_per_image=pair_count / images,
            v2_allocation_count=pair_count + values['overflow_penalized'],
            v2_overflow_score=values['overflow_score_sum'] / values['overflow_penalized'].clamp_min(1.),
            v2_donor_utilization=pair_count / values['donor_available'].clamp_min(1.),
            v2_selected_gt_coverage=values['selected_covered_gt'] / values['gt'].clamp_min(1.),
            v2_pair_logit_gap=values['pair_gap_sum'] / pair_count.clamp_min(1.),
            v2_donor_score=values['donor_score_sum'] / pair_count.clamp_min(1.),
            v2_rescue_score=values['rescue_score_sum'] / values['rescue_pairs'].clamp_min(1.),
            v2_prefix_score=values['prefix_score_sum'] / values['prefix_pairs'].clamp_min(1.),
            v2_keeper_score=values['keeper_score_sum'] / values['reliable_representatives'].clamp_min(1.),
            v2_owners=owners, v2_owner_iou_sum=owner_iou,
            v2_owners_before_guard=before, v2_small_owners_before_guard=small_before,
            v2_small_owners=small_after,
            v2_owner_iou=owner_iou / owners.clamp_min(1.),
            v2_owner_conflicts=conflicts, v2_rescue_classification_masked=masked)
        for name, value in zip(('dense', 'bbox', 'giou', 'allocation'), raw_losses):
            metrics[f'v2_raw_{name}'] = value / get_world_size()
        self._logging_metrics = metrics
