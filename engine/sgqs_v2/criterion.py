"""Dense-only interventions, inheriting owner/guard/survival unchanged from v1."""
import torch
import torch.nn.functional as F

from ..core import register
from ..sgqs.criterion import SGQSCriterion


@register()
class SGQSV2Criterion(SGQSCriterion):
    __inject__ = ['base_criterion']

    def __init__(self, base_criterion, weight_dict, q_min=0.25,
                 dense_alpha=0.75, dense_gamma=2.0, negative_weight=1.0,
                 survival_margin=0.2, survival_temperature=0.1,
                 survival_quality_power=1.0, survival_start_epoch=0.0,
                 survival_ramp_epochs=1.0, survival_threshold='topk',
                 protect_encoder_supervision=False, dense_positive_target='quality',
                 dense_positive_normalization='per_level'):
        super().__init__(
            base_criterion, weight_dict, q_min, dense_alpha, dense_gamma,
            negative_weight, survival_margin, survival_temperature,
            survival_quality_power, survival_start_epoch, survival_ramp_epochs,
            survival_threshold, protect_encoder_supervision)
        if dense_positive_target not in ('quality', 'iou'):
            raise ValueError('dense_positive_target must be quality or iou')
        if dense_positive_normalization not in ('per_level', 'global'):
            raise ValueError('dense_positive_normalization must be per_level or global')
        if dense_positive_target == 'iou' and dense_positive_normalization == 'global':
            raise ValueError('This single-factor screen must not combine both interventions')
        self.dense_positive_target = dense_positive_target
        self.dense_positive_normalization = dense_positive_normalization

    def _dense_loss(self, payload, targets, qualities):
        # Retain v1 arithmetic and reduction order in the unchanged configuration.
        if (self.dense_positive_target == 'quality'
                and self.dense_positive_normalization == 'per_level'):
            return super()._dense_loss(payload, targets, qualities)
        logits = payload['dense_logits'].float()
        valid = payload['valid_mask'].to(dtype=torch.bool, device=logits.device)
        ignore = payload['ignore_mask'].to(dtype=torch.bool, device=logits.device)
        shapes = payload['spatial_shapes'].detach().cpu().tolist()
        levels, classes = len(shapes), logits.shape[-1]
        positive = [logits.sum() * 0.0 for _ in shapes]
        negative = [logits.sum() * 0.0 for _ in shapes]
        pos_count, neg_count = [0] * levels, [0] * levels
        ranges, offset = [], 0
        for height, width in shapes:
            end = offset + int(height) * int(width)
            ranges.append((offset, end))
            offset = end
        for batch in range(logits.shape[0]):
            owners = payload['owner_indices'][batch]
            owner_levels = payload['owner_levels'][batch]
            labels = targets[batch]['labels'][payload['target_indices'][batch]].long()
            if labels.numel() and ((labels < 0).any() or (labels >= classes).any()):
                raise ValueError('SGQS owner label is outside dense_logits class range')
            for level, (start, end) in enumerate(ranges):
                local = logits[batch, start:end]
                neg_mask = (valid[batch, start:end, None]
                            & ~ignore[batch, start:end, None]).expand(-1, classes).clone()
                chosen = owner_levels == level
                if chosen.any():
                    indices, owner_labels = owners[chosen] - start, labels[chosen]
                    iou = qualities[batch][chosen].detach()
                    q = self._quality_target(iou)
                    target = iou if self.dense_positive_target == 'iou' else q
                    pos_logits = local[indices, owner_labels]
                    positive[level] = positive[level] + (q * F.binary_cross_entropy_with_logits(
                        pos_logits, target, reduction='none')).sum()
                    pos_count[level] += pos_logits.numel()
                    neg_mask[indices, owner_labels] = False
                neg_logits = local[neg_mask]
                if neg_logits.numel():
                    negative[level] = negative[level] + (
                        self.dense_alpha * neg_logits.sigmoid().pow(self.dense_gamma)
                        * F.binary_cross_entropy_with_logits(
                            neg_logits, torch.zeros_like(neg_logits), reduction='none')).sum()
                    neg_count[level] += neg_logits.numel()
        loss = logits.sum() * 0.0
        if self.dense_positive_normalization == 'global':
            # One global owner mean, NOT another division by the number of levels.
            denominator = self._distributed_average_denominator(sum(pos_count), logits.device)
            loss = sum(positive) / denominator
            for level in range(levels):
                denominator = self._distributed_average_denominator(neg_count[level], logits.device)
                loss = loss + self.negative_weight * negative[level] / denominator / max(levels, 1)
        else:
            for level in range(levels):
                pos_den = self._distributed_average_denominator(pos_count[level], logits.device)
                neg_den = self._distributed_average_denominator(neg_count[level], logits.device)
                loss = loss + positive[level] / pos_den
                loss = loss + self.negative_weight * negative[level] / neg_den
            loss = loss / max(levels, 1)
        return loss
