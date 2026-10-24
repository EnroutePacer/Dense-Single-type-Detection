"""GT-constrained candidate allocation; every routing decision is detached."""
import torch

from ..deim.box_ops import box_cxcywh_to_xyxy


STAT_NAMES = (
    'images', 'gt', 'selected_queries', 'selected_covered_gt', 'missing_gt',
    'rescue_available_gt', 'no_geometry_gt', 'reliable_representatives',
    'stage_disagreement', 'donor_geometric', 'donor_cross_gt_overlap',
    'donor_final_protected', 'donor_encoder_protected', 'donor_owner_protected',
    'donor_go_conflict', 'donor_go_same_gt', 'donor_available',
    'rescue_pairs', 'prefix_pairs', 'pair_gap_sum',
    'donor_score_sum', 'rescue_score_sum', 'prefix_score_sum', 'keeper_score_sum',
)


@torch.no_grad()
def geometry_assignment(boxes, gt, valid, iou_threshold=.5, ambiguity_gap=.05,
                        chunk_size=512):
    """Return unambiguous nearest GT and coverage, without a dense NxG cache."""
    boxes, gt = boxes.detach().float(), gt.detach().float()
    n, g = len(boxes), len(gt)
    owner = torch.full((n,), -1, dtype=torch.long, device=boxes.device)
    best, second = boxes.new_zeros(n), boxes.new_zeros(n)
    covered = torch.zeros(g, dtype=torch.bool, device=boxes.device)
    if not n or not g:
        return owner, best, second, covered
    xy, truth = box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt)
    gt_area = (truth[:, 2:] - truth[:, :2]).clamp_min(0).prod(-1)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        local = xy[start:end]
        wh = (torch.minimum(local[:, None, 2:], truth[None, :, 2:])
              - torch.maximum(local[:, None, :2], truth[None, :, :2])).clamp_min(0)
        intersection = wh.prod(-1)
        area = (local[:, 2:] - local[:, :2]).clamp_min(0).prod(-1)
        iou = intersection / (area[:, None] + gt_area - intersection).clamp_min(1e-12)
        iou = torch.where(valid[start:end, None], iou, torch.zeros_like(iou))
        values, indices = iou.topk(min(2, g), dim=1)
        best[start:end] = values[:, 0]
        if g > 1:
            second[start:end] = values[:, 1]
        clear = (valid[start:end] & (values[:, 0] >= iou_threshold)
                 & (values[:, 0] - second[start:end] >= ambiguity_gap))
        owner[start:end] = torch.where(clear, indices[:, 0], -1)
        covered |= (iou >= iou_threshold).any(0)
    return owner, best, second, covered


def _target_map(indices, count, device):
    result = torch.full((count,), -1, dtype=torch.long, device=device)
    src, dst = indices
    result[src.to(device)] = dst.to(device)
    return result


@torch.no_grad()
def build_allocation(payload, outputs, targets, final_indices, encoder_record,
                     *, iou_threshold=.5, ambiguity_gap=.05, teacher_score=.5,
                     background_score=.1, margin=.2, iou_chunk_size=512,
                     enabled=True):
    """Return per-image (donor source, receiver source, rescue flag) and sums.

    The main Hungarian result is reused. GO is a box assignment, not a score
    positive: same-GT GO donors are allowed; conflicting-GT donors are rejected.
    """
    logits = payload['dense_logits'].detach().float()
    boxes = payload['dense_boxes'].detach().float()
    selected = payload['selected_indices']
    valid = payload['valid_mask'].bool()
    dec_boxes = outputs['pred_boxes'].detach().float()
    dec_scores = outputs['pred_logits'].detach().float().sigmoid()[..., 0]
    if logits.shape[-1] != 1 or outputs['pred_logits'].shape[-1] != 1:
        raise ValueError('SGQS v2 allocation is single-class only')
    if boxes.shape != logits.shape[:2] + (4,) or selected.shape != dec_scores.shape:
        raise ValueError('Dense source/query shapes do not match final Decoder queries')
    encoder = outputs.get('enc_aux_outputs', [])
    if len(encoder) != 1:
        raise ValueError('SGQS v2 expects one original Encoder output')
    source_logits = logits.gather(1, selected[..., None])
    if not torch.equal(source_logits, encoder[0]['pred_logits'].detach().float()):
        raise ValueError('Encoder selected_indices do not reproduce the original selected scores')
    # Route selected candidates using their actual native (possibly AMP) boxes.
    # Do not introduce a precision-dependent approximate-equality gate between
    # a new FP32 head evaluation and the original AMP head. Scores above verify
    # the selector mapping; the actual selected boxes remain authoritative.
    boxes = boxes.clone()
    boxes.scatter_(1, selected[..., None].expand(-1, -1, 4), encoder[0]['pred_boxes'].detach().float())
    stats = {name: logits.new_zeros(()) for name in STAT_NAMES}
    pairs = []
    for batch, target in enumerate(targets):
        gt = target['boxes'].detach().float().to(logits.device)
        if target['labels'].numel() and not target['labels'].eq(0).all():
            raise ValueError('SGQS v2 targets must use the single class label 0')
        ids, scores = selected[batch], logits[batch, :, 0]
        n, k, g = len(scores), len(ids), len(gt)
        if len(ids.unique()) != k or not valid[batch, ids].all():
            raise ValueError('Selected sources must be unique valid anchors')
        stats['images'] += 1
        stats['gt'] += g
        stats['selected_queries'] += k
        enc_owner, _, enc_second, dense_covered = geometry_assignment(
            boxes[batch], gt, valid[batch], iou_threshold, ambiguity_gap, iou_chunk_size)
        _, _, _, selected_covered = geometry_assignment(
            boxes[batch, ids], gt, torch.ones(k, dtype=torch.bool, device=ids.device),
            iou_threshold, ambiguity_gap, iou_chunk_size)
        dec_owner, _, dec_second, _ = geometry_assignment(
            dec_boxes[batch], gt, torch.ones(k, dtype=torch.bool, device=ids.device),
            iou_threshold, ambiguity_gap, iou_chunk_size)
        final_gt = _target_map(final_indices[batch], k, ids.device)
        cls_gt = _target_map(encoder_record['classification'][batch], k, ids.device)
        go_gt = _target_map(encoder_record['boxes'][batch], k, ids.device)
        owner_gt = torch.full((n,), -1, dtype=torch.long, device=ids.device)
        owner_gt[payload['owner_indices'][batch]] = payload['target_indices'][batch]
        clear_enc = enc_owner[ids]
        reliable = ((final_gt >= 0) & (clear_enc == final_gt)
                    & (dec_owner == final_gt) & (dec_scores[batch] >= teacher_score))
        represented = torch.zeros(g, dtype=torch.bool, device=ids.device)
        represented[final_gt[reliable]] = True
        stats['selected_covered_gt'] += selected_covered.sum()
        stats['missing_gt'] += (~selected_covered).sum()
        stats['no_geometry_gt'] += (~dense_covered).sum()
        stats['reliable_representatives'] += reliable.sum()
        stats['keeper_score_sum'] += dec_scores[batch, reliable].sum()
        stats['stage_disagreement'] += ((clear_enc >= 0) & (dec_owner >= 0)
                                        & (clear_enc != dec_owner)).sum()
        donor = (clear_enc >= 0) & (clear_enc == dec_owner) & (dec_scores[batch] < background_score)
        if g:
            donor &= represented[clear_enc.clamp_min(0)]
        stats['donor_geometric'] += donor.sum()
        gates = (
            ('donor_cross_gt_overlap', (enc_second[ids] >= iou_threshold) | (dec_second >= iou_threshold)),
            ('donor_final_protected', final_gt >= 0),
            ('donor_encoder_protected', cls_gt >= 0),
            ('donor_owner_protected', owner_gt[ids] >= 0),
            ('donor_go_conflict', (go_gt >= 0) & (go_gt != clear_enc)),
        )
        for name, blocked in gates:
            stats[name] += (donor & blocked).sum()
            donor &= ~blocked
        stats['donor_go_same_gt'] += (donor & (go_gt >= 0)).sum()
        stats['donor_available'] += donor.sum()
        inside = torch.zeros(n, dtype=torch.bool, device=ids.device)
        inside[ids] = True
        missing = torch.zeros(n, dtype=torch.bool, device=ids.device)
        if g:
            missing = (enc_owner >= 0) & ~selected_covered[enc_owner.clamp_min(0)]
        rescue_candidates = torch.nonzero(valid[batch] & ~inside & missing, as_tuple=True)[0]
        rescue_candidates = rescue_candidates[scores[rescue_candidates].argsort(descending=True, stable=True)]
        # A few bulk transfers avoid thousands of CUDA scalar synchronizations
        # in the deterministic, non-differentiable routing loop.
        cpu_scores, cpu_owner = scores.tolist(), enc_owner.tolist()
        cpu_ids, cpu_final = ids.tolist(), final_gt.tolist()
        receivers, receiver_gt, rescue_flags, seen = [], [], [], set()
        for j in rescue_candidates.tolist():
            gj = cpu_owner[j]
            if gj not in seen:
                receivers.append(j); receiver_gt.append(gj); rescue_flags.append(True)
                seen.add(gj)
        stats['rescue_available_gt'] += len(receivers)
        prefix = reliable & (cls_gt == final_gt)
        prefix &= (go_gt < 0) | (go_gt == final_gt)
        prefix &= (owner_gt[ids] < 0) | (owner_gt[ids] == final_gt)
        prefix_indices = torch.nonzero(prefix, as_tuple=True)[0]
        prefix_indices = prefix_indices[scores[ids[prefix_indices]].argsort(descending=True, stable=True)]
        for query in prefix_indices.tolist():
            gj = cpu_final[query]
            if gj not in seen:
                receivers.append(cpu_ids[query]); receiver_gt.append(gj); rescue_flags.append(False)
                seen.add(gj)
        donors = torch.nonzero(donor, as_tuple=True)[0]
        donors = donors[scores[ids[donors]].argsort(stable=True)].tolist()
        used = set()
        chosen_i, chosen_j, chosen_rescue = [], [], []
        if enabled:
            for j, gj, rescue in zip(receivers, receiver_gt, rescue_flags):
                for query in donors:
                    i = cpu_ids[query]
                    if query in used or cpu_owner[i] == gj or margin + cpu_scores[i] <= cpu_scores[j]:
                        continue
                    chosen_i.append(i); chosen_j.append(j); chosen_rescue.append(rescue)
                    used.add(query)
                    break
        i = torch.tensor(chosen_i, dtype=torch.long, device=ids.device)
        j = torch.tensor(chosen_j, dtype=torch.long, device=ids.device)
        rescue = torch.tensor(chosen_rescue, dtype=torch.bool, device=ids.device)
        pairs.append((i, j, rescue))
        stats['rescue_pairs'] += rescue.sum()
        stats['prefix_pairs'] += (~rescue).sum()
        stats['pair_gap_sum'] += (scores[i] - scores[j]).sum()
        stats['donor_score_sum'] += scores[i].sigmoid().sum()
        stats['rescue_score_sum'] += scores[j[rescue]].sigmoid().sum()
        stats['prefix_score_sum'] += scores[j[~rescue]].sigmoid().sum()
    return pairs, stats


def allocation_loss(logits, pairs, margin, denominator):
    """Only rescue receivers get positive ranking gradients."""
    values = logits.float()[..., 0]
    total, count = values.sum() * 0., 0
    for batch, (donor, receiver, rescue) in enumerate(pairs):
        reference = torch.where(rescue, values[batch, receiver], values[batch, receiver].detach())
        total = total + (margin + values[batch, donor] - reference).clamp_min(0).sum()
        count += len(donor)
    return total / denominator(count, logits.device), count
