"""Opt-in observation of the existing criterion; never adds a training loss.

Loaded by train_with_probes.py, not by the frozen runtime's package imports.
JSONL is rank-local: reduce sums/counts offline, never average rank means.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from engine.core import register
from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
from engine.misc.box_ops import elementwise_box_iou


def moments(values):
    values = values.detach().double().reshape(-1)
    return {'count': values.numel(), 'sum': values.sum().item(),
            'sum_sq': values.square().sum().item()}


def area_group(area):
    return torch.where(area < 32**2, 0, torch.where(area < 96**2, 1, 2))


def overlap(pred, gt):
    return box_iou(box_cxcywh_to_xyxy(pred.float()), box_cxcywh_to_xyxy(gt.float()))[0]


def ranking_rows(boxes, logits, gt, budgets=(300, 640)):
    """Independent GT coverage, NOT COCO recall (queries may cover >1 GT)."""
    ious = overlap(boxes, gt)
    order = logits.float().reshape(-1).argsort(descending=True, stable=True)
    result = {'best_iou_all': ious.max(0).values if len(boxes) else gt.new_zeros(len(gt))}
    for budget in budgets:
        keep = order[:min(budget, len(order))]
        result[f'best_iou_top{budget}'] = ious[keep].max(0).values if len(keep) else gt.new_zeros(len(gt))
    for threshold in (.5, .75):
        eligible = ious[order] >= threshold
        rank = torch.arange(1, len(order)+1, device=gt.device)[:, None].expand_as(eligible)
        result[f'first_rank_iou{threshold}'] = torch.where(eligible, rank, len(order)+1).min(0).values
    return result


def ddf_parts(output, indices, targets, num_pos, num_neg, reg_max, dn_meta=None):
    """Reconstruct the actual DDF scalar from detached tensors, including padding."""
    student = output['pred_corners'].detach().reshape(*output['pred_boxes'].shape[:2], 4, reg_max+1)
    teacher = output['teacher_corners'].detach().reshape_as(student)
    p_teacher = F.softmax(teacher / 5, -1)
    kl = 25 * F.kl_div(F.log_softmax(student / 5, -1), p_teacher, reduction='none').sum(-1)
    entropy = -(p_teacher.float() * p_teacher.float().clamp_min(1e-30).log()).sum(-1)
    weights = output['teacher_logits'].detach().sigmoid().amax(-1)
    positive = torch.zeros_like(weights, dtype=torch.bool)
    scale = torch.full_like(weights, -1, dtype=torch.long)
    # Tag positive GT and true DN negatives by augmented input area later.
    for b, (src, dst) in enumerate(indices):
        src, dst = src.to(weights.device), dst.to(weights.device)
        positive[b, src] = True
        matched, _ = elementwise_box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][b, src].detach()),
                                       box_cxcywh_to_xyxy(targets[b]['boxes'][dst].float()))
        weights[b, src] = matched.to(weights.dtype)
        scale[b, src] = targets[b]['probe_area_group'][dst]
    genuine_negative = ~positive
    padding = torch.zeros_like(positive)
    if dn_meta is not None:
        genuine_negative = torch.zeros_like(positive)
        width = weights.shape[1] // (2 * int(dn_meta['dn_num_group']))
        for b, target in enumerate(targets):
            for group in range(int(dn_meta['dn_num_group'])):
                ids = torch.arange(len(target['boxes']), device=weights.device)
                slots = group * 2 * width + width + ids
                genuine_negative[b, slots] = True
                scale[b, slots] = target['probe_area_group']
        padding = ~(positive | genuine_negative)
    weighted = weights[..., None] * kl
    rho_pos, rho_neg = float(num_pos), float(num_neg)
    denom = rho_pos + rho_neg
    def mean(mask):
        return weighted[mask].mean().item() if mask.any() else 0.
    reconstructed = (rho_pos * mean(positive) + rho_neg * mean(~positive)) / denom if denom else 0.
    groups = {}
    for label, mask in [('positive', positive), ('negative', genuine_negative), ('padding', padding)]:
        for bucket, selector in [('all', torch.ones_like(mask)), ('small_input', scale==0),
                                 ('medium_input', scale==1), ('large_input', scale==2)]:
            if label == 'padding' and bucket != 'all':
                continue
            use = mask & selector
            groups[f'{label}/{bucket}'] = {'kl_T2': moments(kl[use]),
                'weight': moments(weights[use]), 'weighted_kl': moments(weighted[use]),
                'teacher_entropy': moments(entropy[use])}
    return {'rho_pos': rho_pos, 'rho_neg': rho_neg, 'unweighted_loss': reconstructed,
            'groups': groups}


@register()
class SupervisionProbeCriterion(nn.Module):
    __inject__ = ['base_criterion']
    __share__ = ['output_dir']

    def __init__(self, base_criterion, output_dir=None, interval=100, early_epoch=28,
                 early_steps=200, early_interval=10, max_gt_rows=64, enabled=True,
                 corner_gradients=True):
        super().__init__()
        self.base_criterion = base_criterion
        self.output_dir = output_dir
        self.interval, self.early_epoch = int(interval), int(early_epoch)
        self.early_steps, self.early_interval = int(early_steps), int(early_interval)
        self.max_gt_rows, self.enabled = int(max_gt_rows), bool(enabled)
        self.corner_gradients = bool(corner_gradients)
        if self.enabled and not self.output_dir:
            raise ValueError('enabled probes require an output_dir')
        if self.interval < 1 or self.early_interval < 1 or self.max_gt_rows < 0:
            raise ValueError('probe intervals must be positive and max_gt_rows non-negative')

    def get_logging_metrics(self):
        return self.base_criterion.get_logging_metrics()

    def due(self, epoch, step, length):
        return (step % self.interval == 0 or step == length-1 or
                (epoch == self.early_epoch and step < self.early_steps and step % self.early_interval == 0))

    def forward(self, outputs, targets, **metadata):
        losses = self.base_criterion(outputs, targets, **metadata)
        if self.enabled and self.training and self.due(metadata['epoch'], metadata['step'], metadata['epoch_step']):
            with torch.no_grad():
                record = self.observe(outputs, targets, losses, metadata)
            if self.corner_gradients and torch.is_grad_enabled():
                # Local distribution-logit conflict only; not shared-parameter or Adam attribution.
                dn = outputs.get('dn_outputs', [])
                if dn and 'loss_ddf_dn_0' in losses:
                    tensor = dn[0]['pred_corners']
                    pair = [torch.autograd.grad(losses[k], tensor, retain_graph=True, allow_unused=True)[0]
                            for k in ('loss_fgl_dn_0', 'loss_ddf_dn_0')]
                    if all(g is not None for g in pair):
                        a, b = [g.detach().float() for g in pair]
                        record['dn0_corner_gradient'] = {'space': 'returned_dn0_distribution_logits',
                            'autograd_dtype': str(tensor.dtype),
                            'fgl_norm_sq': a.square().sum().item(), 'ddf_norm_sq': b.square().sum().item(),
                            'dot': (a*b).sum().item()}
            path = Path(self.output_dir) / 'probes'
            path.mkdir(parents=True, exist_ok=True)
            with (path / f'train.rank{record["rank"]}.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
        return losses

    def observe(self, outputs, targets, losses, metadata):
        wrapped = self.base_criterion
        sgqs = wrapped if hasattr(wrapped, '_guard_encoder_supervision') else None
        if sgqs is not None and not sgqs.protect_encoder_supervision:
            raise ValueError('SGQS probe protocol expects the reviewed encoder guard enabled')
        base = wrapped.base_criterion if sgqs is not None else wrapped
        records = base.get_encoder_supervision()
        if len(records) != 1 or not base.use_uni_set or base.num_classes != 1:
            raise ValueError('probe v1 requires one encoder output, single-class and GO enabled')
        go = records[0]['boxes']
        h, w = metadata['input_size']
        tagged = [dict(t, probe_area_group=area_group(t['boxes'][:, 2]*w*t['boxes'][:, 3]*h)) for t in targets]
        result = {'schema': 'sgqs_supervision_probe_v1', **metadata,
                  'rank': int(os.environ.get('RANK', 0)), 'world_size': int(os.environ.get('WORLD_SIZE', 1)),
                  'losses_weighted_local': {k:v.detach().item() for k,v in losses.items()},
                  'groups': {}, 'gt_rows': [], 'ddf': {}}
        payload = outputs.get('sgqs_aux_outputs')
        for b, target in enumerate(targets):
            gt = target['boxes'].detach().float(); n = len(gt)
            src, dst = [x.to(gt.device) for x in go[b]]
            multiplicity = torch.bincount(dst, minlength=n).float()
            group = tagged[b]['probe_area_group']
            measures = {'go_query_count': multiplicity, 'go_missing': (multiplicity==0).float()}
            if n:
                delta = (gt[:,None,:2]-gt[None,:,:2]).abs()
                sizes = gt[:,2:].clamp_min(gt.new_tensor([2/w,2/h]))
                near = (delta/sizes[:,None]).sum(-1) <= 2
                near.fill_diagonal_(False)
                measures['neighbors_center_l1_le2'] = near.sum(1).float()
            for stage, output in [('encoder_selected', outputs['enc_aux_outputs'][0]), ('decoder', outputs)]:
                if n:
                    measures.update({stage+'_'+k:v for k,v in ranking_rows(output['pred_boxes'][b],
                        output['pred_logits'][b], gt).items()})
            # Actual matched-query quality and regression error, with GO multiplicity retained.
            for stage, output in [('encoder', outputs['enc_aux_outputs'][0]), ('decoder', outputs),
                                  ('decoder0', outputs['aux_outputs'][0])]:
                iou, _ = elementwise_box_iou(box_cxcywh_to_xyxy(output['pred_boxes'][b, src].detach().float()),
                                             box_cxcywh_to_xyxy(gt[dst]))
                sums = gt.new_zeros(n).scatter_add_(0, dst, iou)
                measures[stage+'_go_iou_sum'] = sums
            if payload is not None:
                owners, gids = payload['owner_indices'][b], payload['target_indices'][b]
                selected = payload['selected_indices'][b]
                class_to = torch.full((payload['dense_logits'].shape[1],), -2, device=gt.device, dtype=torch.long)
                class_to[selected] = -1
                q, g = records[0]['classification'][b]
                qi, gi = q.to(gt.device), g.to(gt.device)
                enc = outputs['enc_aux_outputs'][0]['pred_boxes'][b, qi].detach().float()
                matched_iou, _ = elementwise_box_iou(box_cxcywh_to_xyxy(enc), box_cxcywh_to_xyxy(gt[gi]))
                measures['encoder_hungarian_iou'] = gt.new_zeros(n).scatter_(0,gi,matched_iou)
                class_to[selected[q.to(gt.device)]] = g.to(gt.device)
                box_to = torch.full_like(class_to, -2)
                box_to[selected[src]] = dst
                background = class_to[owners] == -1
                foreign = (class_to[owners] >= 0) & (class_to[owners] != gids)
                go_foreign = (box_to[owners] >= 0) & (box_to[owners] != gids)
                conflict = background | foreign | go_foreign
                owner_iou, _ = elementwise_box_iou(box_cxcywh_to_xyxy(payload['owner_boxes'][b].detach().float()),
                                                  box_cxcywh_to_xyxy(gt[gids]))
                survived = (owners[:, None] == selected[None, :]).any(1)
                rank_order = payload['selection_scores'][b].argsort(descending=True, stable=True)
                ranks = torch.empty_like(rank_order); ranks[rank_order] = torch.arange(1,len(rank_order)+1,device=gt.device)
                owner_values = {'owner_assigned': torch.ones_like(owner_iou), 'owner_kept': (~conflict).float(),
                    'owner_selected_raw': survived.float(), 'owner_iou_raw': owner_iou,
                    'owner_rank_raw': ranks[owners].float(), 'conflict_background': background.float(),
                    'conflict_encoder_foreign': foreign.float(),
                    'conflict_go_only': (go_foreign & ~background & ~foreign).float(),
                    'conflict_go_any': go_foreign.float()}
                for key, values in owner_values.items():
                    measures[key] = gt.new_zeros(n).scatter_(0,gids,values)
                for level in range(3):
                    measures[f'owner_p{level+3}_kept'] = gt.new_zeros(n).scatter_(0,gids,
                        ((payload['owner_levels'][b]==level)&~conflict).float())
            for bucket in range(3):
                mask = group == bucket
                key = ('small_input','medium_input','large_input')[bucket]
                dest = result['groups'].setdefault(key, {'gt_count':0,'metrics':{}})
                dest['gt_count'] += int(mask.sum())
                for name, values in measures.items():
                    stats = moments(values[mask]); existing = dest['metrics'].setdefault(name, {'count':0,'sum':0.,'sum_sq':0.})
                    for field in stats: existing[field] += stats[field]
            for g in range(n):
                if len(result['gt_rows']) >= self.max_gt_rows: break
                row = {'image_id':int(target.get('image_id',torch.tensor([b])).reshape(-1)[0]),
                       'target_index_in_augmented_batch':g, 'input_area_group':int(group[g]),
                       'input_short_px':float(torch.minimum(gt[g,2]*w,gt[g,3]*h)),
                       'gt_count_in_image':n}
                row.update({key:float(value[g]) for key,value in measures.items()})
                result['gt_rows'].append(row)
        for prefix, branch_outputs, indices, dn_meta in [
            ('ordinary', outputs.get('aux_outputs', []), go, None),
            ('dn', outputs.get('dn_outputs', []),
             base.get_cdn_matched_indices(outputs['dn_meta'], targets) if 'dn_meta' in outputs else [], outputs.get('dn_meta'))]:
            for layer, output in enumerate(branch_outputs):
                loss_key = f'loss_ddf_{"dn" if prefix=="dn" else "aux"}_{layer}'
                if loss_key not in losses: continue
                parts = ddf_parts(output, indices, tagged, base.num_pos, base.num_neg, base.reg_max, dn_meta)
                parts['weighted_reconstruction'] = parts['unweighted_loss'] * base.weight_dict['loss_ddf']
                parts['observed_weighted_loss'] = float(losses[loss_key].detach())
                result['ddf'][f'{prefix}_{layer}'] = parts
        return result
