"""NumPy xywh IoU for the bundled bbox evaluator; masks are not approximated."""
import numpy as np


def iou(detections, truth, iscrowd):
    detections = np.asarray(detections, dtype=np.float64).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    crowd = np.asarray(iscrowd, dtype=bool)
    if crowd.shape != (len(truth),):
        raise ValueError('iscrowd must contain one flag per ground-truth box')
    dwh = detections[:, 2:].clip(min=0)
    gwh = truth[:, 2:].clip(min=0)
    left = np.maximum(detections[:, None, :2], truth[None, :, :2])
    right = np.minimum((detections[:, :2] + dwh)[:, None], (truth[:, :2] + gwh)[None])
    intersection = (right - left).clip(min=0).prod(-1)
    dt_area, gt_area = dwh.prod(-1)[:, None], gwh.prod(-1)[None]
    denominator = np.where(crowd[None], dt_area, dt_area + gt_area - intersection)
    return np.divide(intersection, denominator, out=np.zeros_like(intersection), where=denominator > 0)


def _unsupported(*args, **kwargs):
    raise NotImplementedError('The bundled COCO fallback supports bbox detection only; '
                              'polygon/RLE mask operations need an existing COCO mask backend.')


encode = decode = area = toBbox = frPyObjects = merge = _unsupported
