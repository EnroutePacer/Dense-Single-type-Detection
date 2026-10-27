"""Strict COCO-style AP for the declared 640-input hybrid-small population.

Use COCO's area-range ignore machinery with binary membership encoded as area.
Real bbox coordinates and IoUs are never changed. Apply maxDets before grouping.
Unmatched out-of-group detections are ignored, exactly as for COCO area slices.
"""
import numpy as np
from faster_coco_eval import COCO, COCOeval_faster


def is_local_small(box, image, input_size=(640, 640)):
    w, h = box[2] * input_size[1] / image['width'], box[3] * input_size[0] / image['height']
    return w * h < 32**2 or min(w, h) < 16


def _mean_valid(values):
    values = values[values > -1]
    return float(values.mean()) if values.size else None


def extract_metrics(evaluator, area_index, budget_index):
    p = evaluator.eval['precision'][:, :, :, area_index, budget_index]
    r = evaluator.eval['recall'][:, :, area_index, budget_index]
    return {'AP': _mean_valid(p), 'AP50': _mean_valid(p[0]),
            'AP75': _mean_valid(p[5]), 'AP95': _mean_valid(p[9]),
            'AR': _mean_valid(r), 'AR75': _mean_valid(r[5]), 'AR95': _mean_valid(r[9])}


def dense_image_ids(coco, minimum_gt=301):
    """Predeclared scene cohort, selected only from non-crowd annotations."""
    if minimum_gt < 1:
        raise ValueError('minimum_gt must be positive')
    return sorted(image_id for image_id in coco.imgs
                  if sum(not a.get('iscrowd', 0) for a in coco.imgToAnns[image_id]) >= minimum_gt)


def evaluate_predictions(coco, predictions, budgets=(300, 640), image_ids=None, input_size=(640, 640)):
    """Accept every prediction, including duplicates and background detections.

    Returns AP@[.50:.05:.95], 101-point interpolation, one-to-one non-crowd
    matching. An empty GT population is undefined (None), not zero AP.
    """
    if len(coco.getCatIds()) != 1:
        raise ValueError('This protocol requires exactly one category')
    budgets = tuple(sorted(set(budgets)))
    if not budgets or min(budgets) < 1:
        raise ValueError('Budgets must be positive')
    population = sorted(coco.imgs) if image_ids is None else list(image_ids)
    if len(population) != len(set(population)) or set(population) - set(coco.imgs):
        raise ValueError('Image cohort contains duplicate or unknown IDs')
    selected = set(population)
    annotations = [a for image_id in population for a in coco.imgToAnns[image_id]]
    result = {'protocol': ('hybrid640_coco_slice_v1' if tuple(input_size) == (640, 640)
                           else 'hybrid_input_coco_slice_v1'), 'input_size': list(input_size),
              'images': len(population),
              'noncrowd_gt': sum(not a.get('iscrowd', 0) for a in annotations),
              'local_small_gt': sum(is_local_small(a['bbox'], coco.imgs[a['image_id']], input_size)
                                    and not a.get('iscrowd', 0) for a in annotations),
              'budgets': {}}
    if not population:
        names = ('coco_all', 'coco_small', 'coco_medium', 'coco_large', 'local_hybrid_small')
        keys = ('AP', 'AP50', 'AP75', 'AP95', 'AR', 'AR75', 'AR95')
        result['budgets'] = {str(b): {name: dict.fromkeys(keys) for name in names} for b in budgets}
        return result
    if image_ids is not None:
        predictions = (predictions[np.isin(predictions[:, 0], population)]
                       if isinstance(predictions, np.ndarray)
                       else [p for p in predictions if p['image_id'] in selected])
    if len(predictions):
        dt = coco.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {'images': list(coco.imgs.values()),
                      'categories': list(coco.cats.values()), 'annotations': []}
        dt.createIndex()
    ev = COCOeval_faster(coco, dt, iouType='bbox', print_function=lambda *a: None)
    ev.params.imgIds = population
    ev.params.maxDets = list(budgets)
    ev.params.iouThrs = np.linspace(.5, .95, 10)
    ev.params.recThrs = np.linspace(0., 1., 101)
    ev.evaluate(); ev.accumulate()
    for index, budget in enumerate(budgets):
        result['budgets'][str(budget)] = {
            name: extract_metrics(ev, area, index) for area, name in enumerate(
                ('coco_all', 'coco_small', 'coco_medium', 'coco_large'))}
    # loadRes computes real areas. Change only the temporary evaluator areas,
    # after loadRes, and restore every original annotation even on exceptions.
    gt_areas = [a['area'] for a in annotations]
    try:
        for members in (annotations, dt.anns.values()):
            for ann in members:
                ann['area'] = 1. if is_local_small(ann['bbox'], coco.imgs[ann['image_id']], input_size) else 2.
        local = COCOeval_faster(coco, dt, iouType='bbox', print_function=lambda *a: None)
        local.params.imgIds = population
        local.params.maxDets = list(budgets)
        local.params.iouThrs = np.linspace(.5, .95, 10)
        local.params.recThrs = np.linspace(0., 1., 101)
        local.params.areaRng = [[.5, 1.5]]
        local.params.areaRngLbl = ['local_hybrid_small']
        local.evaluate(); local.accumulate()
        for index, budget in enumerate(budgets):
            result['budgets'][str(budget)]['local_hybrid_small'] = extract_metrics(local, 0, index)
    finally:
        for ann, area in zip(annotations, gt_areas):
            ann['area'] = area
    return result


def fixed_box_exchange(boxes, scores, gt_boxes, input_size=(640, 640), budget=300):
    """GT-assisted score-swap diagnostic, NOT a deployable metric or upper bound.

    Geometry-only Hungarian assignment differs from the training matcher's full
    cost. Keep all boxes and the score multiset fixed; exchange disjoint scores.
    """
    import torch
    from scipy.optimize import linear_sum_assignment
    from engine.deim.box_ops import box_cxcywh_to_xyxy, box_iou
    from engine.sgqs_v2.admission import exchange_pairs
    quality = torch.zeros_like(scores)
    small = torch.zeros_like(scores, dtype=torch.bool)
    if len(gt_boxes):
        iou = box_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(gt_boxes))[0]
        src, dst = linear_sum_assignment(-iou.cpu().numpy())
        src, dst = torch.as_tensor(src, device=scores.device), torch.as_tensor(dst, device=scores.device)
        quality[src] = iou[src, dst]
        w, h = gt_boxes[dst, 2] * input_size[1], gt_boxes[dst, 3] * input_size[0]
        small[src] = (w * h < 32**2) | (torch.minimum(w, h) < 16)
    pos, neg = exchange_pairs(scores, quality, small, budget)
    swapped = scores.clone()
    swapped[pos], swapped[neg] = scores[neg], scores[pos]
    return swapped, len(pos)
