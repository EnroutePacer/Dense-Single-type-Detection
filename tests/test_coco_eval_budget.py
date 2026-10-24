import unittest
import torch

from faster_coco_eval import COCO
from engine.data.dataset.coco_eval import CocoEvaluator


class EvaluatorBudgetTest(unittest.TestCase):
    def test_budget_survives_cleanup_and_default_is_legacy(self):
        coco = COCO()
        coco.dataset = {'images': [{'id': 1, 'width': 640, 'height': 640}],
                        'categories': [{'id': 1, 'name': 'cell'}], 'annotations': []}
        coco.createIndex()
        for kwargs, expected in (({}, 300), ({'max_dets': 640}, 640)):
            evaluator = CocoEvaluator(coco, ['bbox'], **kwargs)
            self.assertEqual(evaluator.coco_eval['bbox'].params.maxDets, [1, 10, expected])
            evaluator.cleanup()
            self.assertEqual(evaluator.coco_eval['bbox'].params.maxDets, [1, 10, expected])

    def test_invalid_budget_cannot_silently_change_evaluation(self):
        for value in (True, 0, 9, 640.0, '640'):
            with self.assertRaises(ValueError):
                CocoEvaluator(COCO(), ['bbox'], max_dets=value)

    def test_summary_uses_actual_640_budget_not_a_default_300(self):
        coco = COCO()
        boxes = [[2*(i % 50), 2*(i // 50), 1, 1] for i in range(301)]
        coco.dataset = {'images': [{'id': 1, 'width': 640, 'height': 640}],
            'categories': [{'id': 1, 'name': 'cell'}],
            'annotations': [dict(id=i+1,image_id=1,category_id=1,bbox=b,area=1,iscrowd=0)
                            for i,b in enumerate(boxes)]}
        coco.createIndex()
        xyxy = torch.tensor(boxes,dtype=torch.float32)
        xyxy[:,2:] += xyxy[:,:2]
        for budget in (300,640):
            evaluator=CocoEvaluator(coco,['bbox'],max_dets=budget)
            evaluator.update({1:dict(boxes=xyxy,scores=torch.linspace(.99,.5,301),
                                     labels=torch.ones(301,dtype=torch.long))})
            evaluator.synchronize_between_processes()
            evaluator.accumulate()
            evaluator.summarize()
            ap=evaluator.coco_eval['bbox'].stats[0]
            if budget==640: self.assertAlmostEqual(ap,1.)
            else: self.assertLess(ap,1.)


if __name__ == '__main__':
    unittest.main()
