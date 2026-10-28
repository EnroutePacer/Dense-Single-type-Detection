"""COCO APIs without installing packages or replacing global pycocotools modules.

Prefer installed accelerated/standard APIs; fall back to the bundled bbox
reference when neither compiled package is available. Never install packages.
"""
import contextlib
import copy
import importlib.metadata
import io
import warnings

import numpy as np

try:
    from faster_coco_eval import COCO, COCOeval_faster as _Evaluator
    import faster_coco_eval.core.mask as mask_util
    BACKEND = 'faster_coco_eval'
    EVAL_IMAGE_ATTRIBUTE = '_evalImgs_cpp'
except (ImportError, OSError) as accelerated_error:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval as _StandardEvaluator
        from pycocotools import mask as mask_util
        BACKEND = 'pycocotools'
    except (ImportError, OSError) as standard_error:
        from ._coco_reference.coco import COCO
        from ._coco_reference.cocoeval import COCOeval as _StandardEvaluator
        from ._coco_reference import mask as mask_util
        BACKEND = 'bundled_coco_bbox'
    EVAL_IMAGE_ATTRIBUTE = 'evalImgs'
    warnings.warn(f'faster_coco_eval is unavailable; using {BACKEND} '
                  'for COCO matching and evaluation (slower; no environment changes).',
                  RuntimeWarning, stacklevel=2)

    class _Evaluator(_StandardEvaluator):
        def __init__(self, cocoGt=None, cocoDt=None, iouType='segm',
                     print_function=print, separate_eval=True):
            if BACKEND == 'bundled_coco_bbox' and iouType != 'bbox':
                raise ValueError('The bundled COCO backend supports bbox evaluation only')
            super().__init__(cocoGt, cocoDt, iouType)
            self.print_function = print_function

        def _run(self, function):
            # pycocotools prints directly; keep the caller's existing logging API.
            with contextlib.redirect_stdout(io.StringIO()) as output:
                result = function()
            for line in output.getvalue().splitlines():
                self.print_function(line)
            return result

        def evaluate(self):
            return self._run(super().evaluate)

        def accumulate(self):
            return self._run(super().accumulate)

        def summarize(self):
            """Standard summaries at the configured budget, never implicit 100.

            Upstream COCO's bbox stats[0] hardcodes maxDets=100, whereas this
            repository intentionally uses 300/640. Matching tensors are intact.
            """
            if not self.eval:
                raise RuntimeError('Run accumulate() before summarize()')
            p = self.params

            def metric(ap, area='all', budget=None, threshold=None):
                budget = p.maxDets[-1] if budget is None else budget
                data = self.eval['precision' if ap else 'recall']
                if threshold is not None:
                    data = data[np.isclose(p.iouThrs, threshold)]
                data = np.take(data, [i for i, name in enumerate(p.areaRngLbl) if name == area], axis=-2)
                data = np.take(data, [i for i, value in enumerate(p.maxDets) if value == budget], axis=-1)
                valid = data[data > -1]
                value = float(valid.mean()) if valid.size else -1.
                iou = f'{p.iouThrs[0]:.2f}:{p.iouThrs[-1]:.2f}' if threshold is None else f'{threshold:.2f}'
                self.print_function(f" Average {'Precision' if ap else 'Recall'} "
                    f"({'AP' if ap else 'AR'}) @[ IoU={iou} | area={area} | maxDets={budget} ] = {value:.3f}")
                return value

            if p.iouType in ('bbox', 'segm'):
                values = [metric(True), metric(True, threshold=.5), metric(True, threshold=.75)]
                values += [metric(True, area=a) for a in p.areaRngLbl[1:]]
                values += [metric(False, budget=b) for b in p.maxDets]
                values += [metric(False, area=a) for a in p.areaRngLbl[1:]]
            elif p.iouType == 'keypoints':
                values = [metric(True), metric(True, threshold=.5), metric(True, threshold=.75),
                          metric(True, area='medium'), metric(True, area='large'), metric(False),
                          metric(False, threshold=.5), metric(False, threshold=.75),
                          metric(False, area='medium'), metric(False, area='large')]
            else:
                raise ValueError(f'Unsupported COCO IoU type: {p.iouType}')
            self.stats = np.asarray(values)
            self.all_stats = self.stats.copy()


def backend_info():
    if BACKEND == 'bundled_coco_bbox':
        return {'name': BACKEND, 'version': 'pycocotools-2.0.11+numpy-bbox-v1'}
    try:
        version = importlib.metadata.version(BACKEND.replace('_', '-'))
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown'
    return {'name': BACKEND, 'version': version}


def make_evaluator(coco_gt, coco_dt=None, **kwargs):
    return _Evaluator(coco_gt, coco_dt, **kwargs)


def load_results(coco_gt, predictions):
    """Handle empty predictions and older pycocotools without mutating GT info."""
    if len(predictions):
        if 'info' not in coco_gt.dataset:
            coco_gt = copy.copy(coco_gt)
            coco_gt.dataset = dict(coco_gt.dataset, info={})
        return coco_gt.loadRes(predictions)
    result = COCO()
    result.dataset = {'images': list(coco_gt.imgs.values()),
                      'categories': list(coco_gt.cats.values()), 'annotations': []}
    result.createIndex()
    return result
