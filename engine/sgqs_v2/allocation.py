"""GT-constrained transfers and receiver-free, overconfident overflow caps."""
import math

import torch

from ..deim.box_ops import box_cxcywh_to_xyxy
from ..misc.box_ops import elementwise_box_iou


STAT_NAMES = (
    'images', 'gt', 'selected_queries', 'selected_covered_gt', 'missing_gt',
    'selected_no_center_gt', 'selected_ambiguous_only_gt', 'selected_low_ios_gt',
    'dense_no_unique_center_gt', 'legacy_iou_covered_gt', 'iou_only_covered_gt',
    'source_only_covered_gt', 'source_qualified', 'source_center_ambiguous',
    'selected_source_qualified', 'selected_source_area_ratio_sum',
    'selected_source_cross_ios_sum', 'selected_source_cross_ios_ge_threshold',
    'source_representatives', 'donor_source_protected',
    'small_gt', 'small_selected_covered_gt', 'small_missing_no_center_gt',
    'small_missing_ambiguous_gt', 'small_missing_low_ios_gt',
    'rescue_available_gt', 'no_geometry_gt', 'reliable_representatives',
    'stage_disagreement', 'donor_geometric', 'donor_cross_gt_overlap',
    'donor_final_protected', 'donor_encoder_protected', 'donor_owner_protected',
    'donor_go_conflict', 'donor_go_same_gt', 'donor_available',
    'donor_center_ambiguous', 'donor_ios_only', 'no_pair_images',
    'overflow_gt', 'overflow_excess', 'overflow_reserved', 'overflow_overconfident',
    'overflow_penalized', 'overflow_score_sum',
    'rescue_pairs', 'prefix_pairs', 'pair_gap_sum',
    'donor_score_sum', 'rescue_score_sum', 'prefix_score_sum', 'keeper_score_sum',
    'rescue_area_ratio_sum', 'rescue_cross_ios_sum',
)


@torch.no_grad()
def geometry_assignment(boxes, gt, valid, iou_threshold=.5, ambiguity_gap=.05,
                        chunk_size=512, metric='iou'):
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
        if metric == 'iou':
            denominator = area[:, None] + gt_area - intersection
        elif metric == 'ios':
            denominator = torch.minimum(area[:, None], gt_area)
        else:
            raise ValueError('Unknown overlap metric')
        iou = intersection / denominator.clamp_min(1e-12)
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


@torch.no_grad()
def center_assignment(centers, gt, valid, chunk_size=512):
    """Unique GT containing a native anchor center; overlapping GTs abstain."""
    centers, gt = centers.detach().float(), gt.detach().float()
    owner = torch.full((len(centers),), -1, dtype=torch.long, device=centers.device)
    ambiguous = torch.zeros_like(valid, dtype=torch.bool)
    if not len(gt):
        return owner, ambiguous
    truth = box_cxcywh_to_xyxy(gt)
    for start in range(0, len(centers), chunk_size):
        end = start + chunk_size
        point = centers[start:end, None]
        inside = ((point >= truth[None, :, :2]) & (point <= truth[None, :, 2:])).all(-1)
        inside &= valid[start:end, None]
        count = inside.sum(-1)
        owner[start:end] = torch.where(count == 1, inside.long().argmax(-1), -1)
        ambiguous[start:end] = count > 1
    return owner, ambiguous


@torch.no_grad()
def source_coverage(centers, boxes, gt, valid, selected, ios_threshold=.8,
                    chunk_size=512):
    """Native-center ownership AND Encoder IoS, on the same source candidate.

    IoS is intersection / min(area(E), area(G)); boxes are not clipped.
    Center ambiguity abstains. Other-GT IoS and area ratio are diagnostics,
    not extra acceptance gates. In particular no best-IoS assignment is used.
    """
    centers, boxes, gt = (x.detach().float() for x in (centers, boxes, gt))
    valid = valid.bool()
    n, g = len(boxes), len(gt)
    if centers.shape != (n, 2) or boxes.shape != (n, 4) or gt.shape != (g, 4):
        raise ValueError('Source coverage expects native Nx2 centers and cxcywh boxes')
    if valid.shape != (n,) or not 0 < ios_threshold <= 1 or chunk_size < 1:
        raise ValueError('Invalid source coverage mask, IoS threshold or chunk size')
    center_gt, ambiguous = center_assignment(centers, gt, valid, chunk_size)
    ios, ratio, cross_ios = (boxes.new_zeros(n) for _ in range(3))
    raw_selected = torch.zeros(g, dtype=torch.bool, device=boxes.device)
    in_selected = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    in_selected[selected] = True
    if g:
        xy, truth = box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt)
        gt_area = (truth[:, 2:] - truth[:, :2]).clamp_min(0).prod(-1)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            points = centers[start:end, None]
            inside = ((points >= truth[None, :, :2]) & (points <= truth[None, :, 2:])).all(-1)
            raw_selected |= (inside & (valid[start:end] & in_selected[start:end])[:, None]).any(0)
            local = xy[start:end]
            area = (local[:, 2:] - local[:, :2]).clamp_min(0).prod(-1)
            wh = (torch.minimum(local[:, None, 2:], truth[None, :, 2:])
                  - torch.maximum(local[:, None, :2], truth[None, :, :2])).clamp_min(0)
            overlap = wh.prod(-1) / torch.minimum(area[:, None], gt_area).clamp_min(1e-12)
            own = center_gt[start:end]
            rows = torch.arange(end - start, device=boxes.device)
            assigned = own >= 0
            own_ios = overlap[rows, own.clamp_min(0)]
            ios[start:end] = torch.where(assigned, own_ios, 0.)
            ratio[start:end] = torch.where(assigned, area / gt_area[own.clamp_min(0)].clamp_min(1e-12), 0.)
            overlap[rows, own.clamp_min(0)] = 0.
            cross_ios[start:end] = torch.where(assigned, overlap.amax(-1), 0.)
    qualified = torch.where((center_gt >= 0) & (ios >= ios_threshold), center_gt, -1)

    def covered(owners, subset):
        result = torch.zeros(g, dtype=torch.bool, device=boxes.device)
        indices = owners[subset]
        result[indices[indices >= 0]] = True
        return result

    return dict(center_gt=center_gt, ambiguous=ambiguous, qualified_gt=qualified,
                ios=ios, area_ratio=ratio, cross_gt_ios=cross_ios,
                selected_covered=covered(qualified, selected),
                dense_covered=covered(qualified, valid),
                selected_center_covered=covered(center_gt, selected),
                dense_center_covered=covered(center_gt, valid),
                selected_raw_center_covered=raw_selected)


def _target_map(indices, count, device):
    result = torch.full((count,), -1, dtype=torch.long, device=device)
    src, dst = indices
    result[src.to(device)] = dst.to(device)
    return result


@torch.no_grad()
def build_allocation(payload, outputs, targets, final_indices, encoder_record,
                     *, iou_threshold=.5, ambiguity_gap=.05, teacher_score=.5,
                     background_score=.1, margin=.2, iou_chunk_size=512,
                     redundancy_ios_threshold=.8, overflow_tolerance=10,
                     overflow_score_threshold=.8,
                     coverage_ios_threshold=.8, small_reference_size=(640, 640),
                     enabled=True):
    """Return transfer pairs, receiver-free overflow sources, and metric sums.

    The main Hungarian result is reused. GO is a box assignment, not a score
    positive: same-GT GO donors are allowed; conflicting-GT donors are rejected.
    """
    logits = payload['dense_logits'].detach().float()
    boxes = payload['dense_boxes'].detach().float()
    centers = payload['anchor_centers'].detach().float()
    selected = payload['selected_indices']
    valid = payload['valid_mask'].bool()
    dec_boxes = outputs['pred_boxes'].detach().float()
    dec_scores = outputs['pred_logits'].detach().float().sigmoid()[..., 0]
    if logits.shape[-1] != 1 or outputs['pred_logits'].shape[-1] != 1:
        raise ValueError('SGQS v2 allocation is single-class only')
    if boxes.shape != logits.shape[:2] + (4,) or selected.shape != dec_scores.shape:
        raise ValueError('Dense source/query shapes do not match final Decoder queries')
    if centers.shape != logits.shape[:2] + (2,):
        raise ValueError('Native anchor centers must match the dense source axis')
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
    pairs, overflow = [], []
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
        enc_owner, enc_best, enc_second, _ = geometry_assignment(
            boxes[batch], gt, valid[batch], iou_threshold, ambiguity_gap, iou_chunk_size)
        _, _, _, legacy_covered = geometry_assignment(
            boxes[batch, ids], gt, torch.ones(k, dtype=torch.bool, device=ids.device),
            iou_threshold, ambiguity_gap, iou_chunk_size)
        dec_owner, dec_best, dec_second, _ = geometry_assignment(
            dec_boxes[batch], gt, torch.ones(k, dtype=torch.bool, device=ids.device),
            iou_threshold, ambiguity_gap, iou_chunk_size)
        source = source_coverage(centers[batch], boxes[batch], gt, valid[batch], ids,
                                 coverage_ios_threshold, iou_chunk_size)
        selected_covered = source['selected_covered']
        source_gt = source['qualified_gt']
        center_gt, ambiguous_center = source['center_gt'][ids], source['ambiguous'][ids]
        ios_owner, _, ios_second, _ = geometry_assignment(
            dec_boxes[batch], gt, valid[batch, ids], redundancy_ios_threshold,
            ambiguity_gap, iou_chunk_size, metric='ios')
        final_gt = _target_map(final_indices[batch], k, ids.device)
        cls_gt = _target_map(encoder_record['classification'][batch], k, ids.device)
        go_gt = _target_map(encoder_record['boxes'][batch], k, ids.device)
        owner_gt = torch.full((n,), -1, dtype=torch.long, device=ids.device)
        owner_gt[payload['owner_indices'][batch]] = payload['target_indices'][batch]
        clear_enc = enc_owner[ids]
        reliable = ((final_gt >= 0) & (clear_enc == final_gt)
                    & (center_gt == final_gt) & (dec_owner == final_gt)
                    & (dec_scores[batch] >= teacher_score))
        represented = torch.zeros(g, dtype=torch.bool, device=ids.device)
        represented[final_gt[reliable]] = True
        # Fix a source representative BEFORE transfer/overflow selection. A
        # reliable Decoder keeper with IoU >= .5 need not itself have IoS >= .8.
        # Prefer that keeper when qualified, otherwise preserve the best-score
        # qualified selected source. Never demote every valid source together.
        source_protected = torch.zeros(k, dtype=torch.bool, device=ids.device)
        qualified_queries = torch.nonzero(source_gt[ids] >= 0, as_tuple=True)[0]
        ordered = qualified_queries[scores[ids[qualified_queries]].argsort(descending=True, stable=True)]
        preferred = ordered[reliable[ordered]]
        remaining = ordered[~reliable[ordered]]
        source_seen, protected_queries = set(), []
        source_gt_cpu = source_gt[ids].tolist()
        for query in torch.cat([preferred, remaining]).tolist():
            gj = source_gt_cpu[query]
            if gj not in source_seen:
                protected_queries.append(query)
                source_seen.add(gj)
        source_protected[torch.tensor(protected_queries, dtype=torch.long, device=ids.device)] = True
        stats['source_representatives'] += source_protected.sum()
        stats['selected_covered_gt'] += selected_covered.sum()
        stats['missing_gt'] += (~selected_covered).sum()
        no_center = ~source['selected_raw_center_covered']
        ambiguous_only = source['selected_raw_center_covered'] & ~source['selected_center_covered']
        low_ios = source['selected_center_covered'] & ~selected_covered
        stats['selected_no_center_gt'] += no_center.sum()
        stats['selected_ambiguous_only_gt'] += ambiguous_only.sum()
        stats['selected_low_ios_gt'] += low_ios.sum()
        stats['dense_no_unique_center_gt'] += (~source['dense_center_covered']).sum()
        stats['no_geometry_gt'] += (~source['dense_covered']).sum()
        stats['legacy_iou_covered_gt'] += legacy_covered.sum()
        stats['iou_only_covered_gt'] += (legacy_covered & ~selected_covered).sum()
        stats['source_only_covered_gt'] += (selected_covered & ~legacy_covered).sum()
        stats['source_qualified'] += (source_gt >= 0).sum()
        stats['source_center_ambiguous'] += source['ambiguous'].sum()
        qualified_ids = ids[source_gt[ids] >= 0]
        stats['selected_source_qualified'] += len(qualified_ids)
        stats['selected_source_area_ratio_sum'] += source['area_ratio'][qualified_ids].sum()
        stats['selected_source_cross_ios_sum'] += source['cross_gt_ios'][qualified_ids].sum()
        stats['selected_source_cross_ios_ge_threshold'] += (source['cross_gt_ios'][qualified_ids] >= coverage_ios_threshold).sum()
        height, width = small_reference_size
        gw, gh = gt[:, 2] * width, gt[:, 3] * height
        small = (gw * gh < 32**2) | (torch.minimum(gw, gh) < 16)
        stats['small_gt'] += small.sum()
        stats['small_selected_covered_gt'] += (small & selected_covered).sum()
        stats['small_missing_no_center_gt'] += (small & no_center).sum()
        stats['small_missing_ambiguous_gt'] += (small & ambiguous_only).sum()
        stats['small_missing_low_ios_gt'] += (small & low_ios).sum()
        stats['reliable_representatives'] += reliable.sum()
        stats['keeper_score_sum'] += dec_scores[batch, reliable].sum()
        stats['stage_disagreement'] += ((center_gt >= 0) & (ios_owner >= 0)
                                        & (center_gt != ios_owner)).sum()
        stats['donor_center_ambiguous'] += ambiguous_center.sum()
        donor = (center_gt >= 0) & (center_gt == ios_owner) & (dec_scores[batch] < background_score)
        if g:
            donor &= represented[center_gt.clamp_min(0)]
        stats['donor_geometric'] += donor.sum()
        # Preserve previous IoU-based cross-instance protection as well as IoS
        # ambiguity: a containing box must not consume several GTs' support.
        foreign = ((enc_second[ids] >= iou_threshold) | (dec_second >= iou_threshold)
                   | ((enc_best[ids] >= iou_threshold) & (clear_enc != center_gt))
                   | ((dec_best >= iou_threshold) & (dec_owner != center_gt))
                   | (ios_second >= redundancy_ios_threshold))
        gates = (
            ('donor_cross_gt_overlap', foreign),
            ('donor_final_protected', final_gt >= 0),
            ('donor_encoder_protected', cls_gt >= 0),
            ('donor_owner_protected', owner_gt[ids] >= 0),
            ('donor_source_protected', source_protected),
            ('donor_go_conflict', (go_gt >= 0) & (go_gt != center_gt)),
        )
        for name, blocked in gates:
            stats[name] += (donor & blocked).sum()
            donor &= ~blocked
        stats['donor_go_same_gt'] += (donor & (go_gt >= 0)).sum()
        stats['donor_available'] += donor.sum()
        stats['donor_ios_only'] += (donor & ((clear_enc != center_gt) | (dec_owner != center_gt))).sum()
        inside = torch.zeros(n, dtype=torch.bool, device=ids.device)
        inside[ids] = True
        missing = torch.zeros(n, dtype=torch.bool, device=ids.device)
        if g:
            missing = (source_gt >= 0) & ~selected_covered[source_gt.clamp_min(0)]
        rescue_candidates = torch.nonzero(valid[batch] & ~inside & missing, as_tuple=True)[0]
        rescue_candidates = rescue_candidates[scores[rescue_candidates].argsort(descending=True, stable=True)]
        # A few bulk transfers avoid thousands of CUDA scalar synchronizations
        # in the deterministic, non-differentiable routing loop.
        cpu_scores, cpu_owner = scores.tolist(), source_gt.tolist()
        cpu_ids, cpu_final = ids.tolist(), final_gt.tolist()
        cpu_center = center_gt.tolist()
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
                    if query in used or cpu_center[query] == gj or margin + cpu_scores[i] <= cpu_scores[j]:
                        continue
                    chosen_i.append(i); chosen_j.append(j); chosen_rescue.append(rescue)
                    used.add(query)
                    break
        # Per-image exclusivity: any executable transfer disables fallback for
        # the entire image, including unpaired donors. No receiver is fabricated.
        capped = []
        stats['no_pair_images'] += not bool(chosen_i)
        if enabled and not chosen_i and g:
            cap_logit = scores.new_tensor(math.log(
                overflow_score_threshold / (1. - overflow_score_threshold))).item()
            target_iou, _ = elementwise_box_iou(
                box_cxcywh_to_xyxy(dec_boxes[batch]),
                box_cxcywh_to_xyxy(gt[center_gt.clamp_min(0)]))
            cpu_quality, cpu_dec_scores = target_iou.tolist(), dec_scores[batch].tolist()
            safe_queries = set(donors)
            keeper_queries = set(torch.nonzero(reliable, as_tuple=True)[0].tolist())
            for gj in torch.nonzero(represented, as_tuple=True)[0].tolist():
                # Count all selected, uniquely contained source centers except
                # the reliable representative, not only already-safe donors.
                extras = [q for q in range(k) if cpu_center[q] == gj and q not in keeper_queries]
                excess = max(len(extras) - overflow_tolerance, 0)
                if not excess:
                    continue
                stats['overflow_gt'] += 1
                stats['overflow_excess'] += excess
                # Reserve useful/protected backups BEFORE checking confidence.
                # Ranking the reserve by Encoder scores would rotate punishment
                # through all backups as those very scores are pushed down.
                extras.sort(key=lambda q: (q in safe_queries, -cpu_quality[q],
                                          -cpu_dec_scores[q], cpu_ids[q]))
                stats['overflow_reserved'] += min(len(extras), overflow_tolerance)
                eligible = [q for q in extras[overflow_tolerance:]
                            if q in safe_queries and cpu_scores[cpu_ids[q]] > cap_logit]
                stats['overflow_overconfident'] += len(eligible)
                eligible.sort(key=lambda q: (-cpu_scores[cpu_ids[q]], cpu_ids[q]))
                capped.extend(cpu_ids[q] for q in eligible[:excess])
        capped = torch.tensor(capped, dtype=torch.long, device=ids.device)
        overflow.append(capped)
        stats['overflow_penalized'] += len(capped)
        stats['overflow_score_sum'] += scores[capped].sigmoid().sum()
        i = torch.tensor(chosen_i, dtype=torch.long, device=ids.device)
        j = torch.tensor(chosen_j, dtype=torch.long, device=ids.device)
        rescue = torch.tensor(chosen_rescue, dtype=torch.bool, device=ids.device)
        pairs.append((i, j, rescue))
        stats['rescue_pairs'] += rescue.sum()
        stats['prefix_pairs'] += (~rescue).sum()
        stats['pair_gap_sum'] += (scores[i] - scores[j]).sum()
        stats['donor_score_sum'] += scores[i].sigmoid().sum()
        stats['rescue_score_sum'] += scores[j[rescue]].sigmoid().sum()
        stats['rescue_area_ratio_sum'] += source['area_ratio'][j[rescue]].sum()
        stats['rescue_cross_ios_sum'] += source['cross_gt_ios'][j[rescue]].sum()
        stats['prefix_score_sum'] += scores[j[~rescue]].sigmoid().sum()
    return pairs, overflow, stats


def allocation_loss(logits, pairs, margin, denominator, overflow=None, overflow_score_threshold=.8):
    """One global mean over transfers and receiver-free caps; one collective."""
    values = logits.float()[..., 0]
    total, count = values.sum() * 0., 0
    for batch, (donor, receiver, rescue) in enumerate(pairs):
        reference = torch.where(rescue, values[batch, receiver], values[batch, receiver].detach())
        total = total + (margin + values[batch, donor] - reference).clamp_min(0).sum()
        count += len(donor)
    if overflow is not None:
        cap_logit = math.log(overflow_score_threshold / (1. - overflow_score_threshold))
        for batch, sources in enumerate(overflow):
            total = total + (values[batch, sources] - cap_logit).clamp_min(0).sum()
            count += len(sources)
    return total / denominator(count, logits.device), count
