"""Exercise real standard/accelerated COCO paths without changing packages."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from engine.data.dataset.coco_backend import COCO, backend_info
from engine.data.dataset.coco_dataset import CocoDetection
from engine.data.dataset.coco_eval import CocoEvaluator, merge
from tools.table_cells.local_small_eval import evaluate_predictions


def standard_available():
    try:
        return importlib.util.find_spec('pycocotools') is not None
    except ImportError:
        return False


def exercise_backend():
    """Numeric signature includes crowd, ties, duplicates, empty and large K."""
    # Imports the real training lifecycle, not just a standalone COCO class.
    from engine.sgqs_v2.trainer import SGQSV2Trainer  # noqa: F401
    coco = COCO()
    boxes = [[2 * (i % 50), 2 * (i // 50), 1, 1] for i in range(301)]
    coco.dataset = {'images': [{'id': 1, 'width': 640, 'height': 640}],
        'categories': [{'id': 1, 'name': 'cell'}],
        'annotations': [dict(id=i+1, image_id=1, category_id=1,
                            bbox=b, area=1, iscrowd=0) for i, b in enumerate(boxes)]}
    coco.createIndex()
    xyxy = torch.tensor(boxes, dtype=torch.float32)
    xyxy[:, 2:] += xyxy[:, :2]
    stats = []
    for budget in (300, 640):
        evaluator = CocoEvaluator(coco, ['bbox'], max_dets=budget)
        # Repeated image represents distributed sampler padding.
        prediction = {1: dict(boxes=xyxy, scores=torch.linspace(.99, .5, 301),
                              labels=torch.ones(301, dtype=torch.long))}
        evaluator.update({})
        evaluator.update(prediction)
        evaluator.update(prediction)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        stats.append(evaluator.coco_eval['bbox'].stats.tolist())
        evaluator.cleanup()
        assert evaluator.coco_eval['bbox'].params.maxDets == [1, 10, budget]
        evaluator.update({1: dict(boxes=torch.empty(0, 4), scores=torch.empty(0),
                                  labels=torch.empty(0, dtype=torch.long))})
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()
        assert evaluator.coco_eval['bbox'].stats[0] == 0

    coco.dataset['annotations'] = [
        dict(id=1, image_id=1, category_id=1, bbox=[10,10,10,10], area=100, iscrowd=0),
        dict(id=2, image_id=1, category_id=1, bbox=[40,10,10,10], area=100, iscrowd=0),
        dict(id=3, image_id=1, category_id=1, bbox=[100,10,10,10], area=100, iscrowd=1),
        dict(id=4, image_id=1, category_id=1, bbox=[10,100,200,10], area=2000, iscrowd=0)]
    coco.createIndex()
    rows = np.array([[1,10,10,10,10,.9,1], [1,10,10,10,10,.9,1],
                     [1,100,10,10,10,.9,1], [1,100,10,10,10,.9,1],
                     [1,40,10,10,10,.8,1], [1,10,100,200,10,.7,1]])
    result = evaluate_predictions(coco, rows, budgets=(1, 3, 640))
    assert coco.anns[4]['area'] == 2000
    empty = evaluate_predictions(coco, [], budgets=(640,))
    assert empty['budgets']['640']['local_hybrid_small']['AP'] == 0

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        Image.new('RGB', (640, 640)).save(root / 'sample.png')
        coco.dataset['images'][0]['file_name'] = 'sample.png'
        annotation = root / 'ann.json'
        annotation.write_text(json.dumps(coco.dataset), encoding='utf-8')
        data = CocoDetection(str(root), str(annotation), transforms=None, return_masks=False)
        image, target = data[0]
        assert image.size == (640, 640) and len(target['boxes']) == 3
    return {'backend': backend_info(), 'summary': stats, 'budgets': result['budgets']}


class CocoBackendTest(unittest.TestCase):
    def test_bundled_backend_when_both_compiled_packages_are_missing(self):
        script = r'''
import importlib.abc, sys, json, io, contextlib, unittest
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'tests'))
class MissingCOCO(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('faster_coco_eval', 'pycocotools'):
            raise ModuleNotFoundError('simulated fixed server', name=fullname)
sys.meta_path.insert(0, MissingCOCO())
with contextlib.redirect_stdout(io.StringIO()):
    from engine.data.dataset.coco_backend import BACKEND, make_evaluator
    assert BACKEND == 'bundled_coco_bbox'
    from tests.test_coco_backend import exercise_backend
    result = exercise_backend()
    from tests.test_sgqs_v2_training import FrameworkTest
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrameworkTest)
    report = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    assert report.wasSuccessful(), (report.failures, report.errors)
    from tests.test_sgqs_generic_launch import GenericLaunchTest
    from unittest import mock
    import torch
    entry = GenericLaunchTest('test_real_general_entry_scratch_no_amp_no_ema_native_schedule_and_recovery')
    with mock.patch.object(torch.amp, 'GradScaler', None):
        report = unittest.TextTestRunner(stream=io.StringIO()).run(unittest.TestSuite([entry]))
    assert report.wasSuccessful(), (report.failures, report.errors)
    try:
        make_evaluator(None, iouType='segm')
    except ValueError:
        pass
    else:
        raise AssertionError('bbox-only fallback must not pretend to support masks')
assert 'faster_coco_eval' not in sys.modules and 'pycocotools' not in sys.modules
print(json.dumps(result))
'''
        process = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                 env=os.environ.copy(), text=True, capture_output=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_backend_signature(json.loads(process.stdout))

    def test_native_dataset_and_metric_paths(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = exercise_backend()
        self.assertAlmostEqual(result['summary'][1][0], 1.)
        self.assertLess(result['summary'][0][0], 1.)

    def test_distributed_merge_deduplicates_image_axis_and_empty_rank(self):
        # Shape is category x area x image; the second copy of ID=1 differs.
        local = np.array([[['3a', '1a', '1duplicate'], ['3b', '1b', '1duplicate']]], dtype=object)
        remote = np.array([[['2a'], ['2b']]], dtype=object)
        with mock.patch('engine.data.dataset.coco_eval.dist_utils.all_gather',
                        side_effect=[[[3,1,1], [], [2]], [[local], [], [remote]]]):
            ids, values = merge([3,1,1], [local], shape=(1,2))
        self.assertEqual(ids, [1,2,3])
        self.assertEqual(values, ['1a','2a','3a','1b','2b','3b'])
        with mock.patch('engine.data.dataset.coco_eval.dist_utils.all_gather',
                        side_effect=[[[], []], [[], []]]):
            self.assertEqual(merge([], [], shape=(1,4)), ([], []))

    @unittest.skipUnless(standard_available(),
                         'Real pycocotools is needed to exercise the optional fallback')
    def test_real_pycocotools_when_accelerated_package_is_unavailable(self):
        script = r'''
import importlib.abc, sys, json, io, contextlib
class MissingAccelerated(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'faster_coco_eval' or fullname.startswith('faster_coco_eval.'):
            raise ModuleNotFoundError("simulated server: no faster_coco_eval", name=fullname)
sys.meta_path.insert(0, MissingAccelerated())
with contextlib.redirect_stdout(io.StringIO()):
    from tests.test_coco_backend import exercise_backend
    result = exercise_backend()
from pycocotools.coco import COCO
assert COCO.__module__ == 'pycocotools.coco'
assert result['backend']['name'] == 'pycocotools'
print(json.dumps(result))
'''
        process = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                                 env=os.environ.copy(), text=True, capture_output=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assert_backend_signature(json.loads(process.stdout))

    def assert_backend_signature(self, standard):
        with contextlib.redirect_stdout(io.StringIO()):
            native = exercise_backend()
        np.testing.assert_allclose(standard['summary'], native['summary'], rtol=0, atol=1e-12)
        for budget, groups in standard['budgets'].items():
            for group, metrics in groups.items():
                for name, value in metrics.items():
                    expected = native['budgets'][budget][group][name]
                    if value is None:
                        self.assertIsNone(expected)
                    else:
                        self.assertAlmostEqual(value, expected, places=12)


if __name__ == '__main__':
    unittest.main()
