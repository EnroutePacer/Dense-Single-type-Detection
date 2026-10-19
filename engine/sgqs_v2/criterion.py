"""Independent SGQS interventions; preserve v1 matching and inference code."""
import torch
import torch.nn.functional as F

from ..core import register
from ..sgqs.criterion import SGQSCriterion
from ..misc.dist_utils import get_world_size
from .admission import admission_loss


@register()
class SGQSV2Criterion(SGQSCriterion):
    __inject__ = ['base_criterion']

    def __init__(self, base_criterion, weight_dict, q_min=0.25,
                 dense_alpha=0.75, dense_gamma=2.0, negative_weight=1.0,
                 survival_margin=0.2, survival_temperature=0.1,
                 survival_quality_power=1.0, survival_start_epoch=0.0,
                 survival_ramp_epochs=1.0, survival_threshold='topk',
                 protect_encoder_supervision=False, dense_positive_target='quality',
                 dense_positive_normalization='per_level', dense_owner_ramp_epochs=0.,
                 admission_weight=0.):
        super().__init__(
            base_criterion, weight_dict, q_min, dense_alpha, dense_gamma,
            negative_weight, survival_margin, survival_temperature,
            survival_quality_power, survival_start_epoch, survival_ramp_epochs,
            survival_threshold, protect_encoder_supervision)
        if dense_positive_target not in ('quality', 'iou'):
            raise ValueError('dense_positive_target must be quality or iou')
        if dense_positive_normalization not in ('per_level', 'global'):
            raise ValueError('dense_positive_normalization must be per_level or global')
        if dense_owner_ramp_epochs < 0 or admission_weight < 0:
            raise ValueError('Ramp duration and admission weight must be non-negative')
        if sum([dense_positive_target == 'iou', dense_positive_normalization == 'global',
                dense_owner_ramp_epochs > 0, admission_weight > 0]) > 1:
            raise ValueError('This single-factor screen must not combine both interventions')
        self.dense_positive_target = dense_positive_target
        self.dense_positive_normalization = dense_positive_normalization
        self.dense_owner_ramp_epochs = float(dense_owner_ramp_epochs)
        self.admission_weight = float(admission_weight)

    def loss_sgqs(self, payload, targets, **metadata):
        losses = super().loss_sgqs(payload, targets, **metadata)
        if self.dense_owner_ramp_epochs:
            if not {'epoch', 'step', 'epoch_step'} <= metadata.keys():
                raise ValueError('Dense/owner ramp requires absolute epoch and step metadata')
            progress = metadata['epoch'] + metadata['step'] / max(metadata['epoch_step'], 1)
            ramp = max(0., min(1., (progress - self.survival_start_epoch) / self.dense_owner_ramp_epochs))
            for name in ('loss_sgqs_dense', 'loss_sgqs_bbox', 'loss_sgqs_giou'):
                self._logging_metrics[f'sgqs_before_ramp_{name.removeprefix("loss_sgqs_")}'] = (
                    self._distributed_sum((losses[name] * self.weight_dict[name]).detach()) / get_world_size())
                losses[name] = losses[name] * ramp
            self._logging_metrics['sgqs_dense_owner_ramp'] = payload['dense_logits'].new_tensor(ramp)
        return losses

    def forward(self, outputs, targets, **metadata):
        if not self.admission_weight:
            return super().forward(outputs, targets, **metadata)
        if 'input_size' not in metadata:
            raise ValueError('Admission probe requires augmented input_size')
        # Observe the already-existing FIRST matcher call (final decoder layer).
        # No extra matching, no edits to frozen DEIM, and hook removed even on error.
        captured = []
        def capture(_module, _args, result):
            if not captured:
                captured.append(result['indices'])
        handle = self.base_criterion.matcher.register_forward_hook(capture)
        try:
            losses = super().forward(outputs, targets, **metadata)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError('Final-layer matching was not observed')
        loss, count = admission_loss(outputs, targets, captured[0], metadata['input_size'],
                                     self._distributed_average_denominator)
        losses['loss_sgqs_admission'] = self.admission_weight * loss
        self._logging_metrics['sgqs_admission_pairs_per_rank'] = (
            self._distributed_average_denominator(count, outputs['pred_logits'].device, minimum=0.))
        return losses

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
