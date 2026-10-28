from pathlib import Path
import unittest

import numpy as np
import torch
from engine.data.dataset.coco_backend import COCO

from tools.table_cells.local_small_eval import evaluate_predictions, is_local_small, fixed_box_exchange, dense_image_ids
from tools.table_cells.evaluate_sgqs_v2_matrix import jobs


def dataset(boxes, crowds=None):
    coco = COCO()
    coco.dataset = {'images': [{'id': 1, 'width': 640, 'height': 640}],
                    'categories': [{'id': 1, 'name': 'cell'}],
                    'annotations': [dict(id=i+1, image_id=1, category_id=1, bbox=box,
                                         area=box[2]*box[3], iscrowd=(crowds or [0]*len(boxes))[i])
                                    for i, box in enumerate(boxes)]}
    coco.createIndex()
    return coco


def detection(box, score):
    return dict(image_id=1, category_id=1, bbox=box, score=score)


class LocalSmallAPTest(unittest.TestCase):
    def test_strict_boundary_and_anisotropic_resize(self):
        image = dict(width=640, height=640)
        for box, expected in [([0,0,32,32], False), ([0,0,16,64], False),
                              ([0,0,15,200], True), ([0,0,20,40], True)]:
            self.assertEqual(is_local_small(box, image), expected)
        self.assertFalse(is_local_small([0,0,32,64], dict(width=1280,height=640)))
        self.assertTrue(is_local_small([0,0,31,64], dict(width=1280,height=640)))

    def test_skinny_large_area_has_local_ap_but_not_coco_small_ap(self):
        box = [10, 10, 200, 10]
        coco = dataset([box])
        result = evaluate_predictions(coco, [detection(box, .9)], (300,))['budgets']['300']
        self.assertAlmostEqual(result['local_hybrid_small']['AP'], 1.)
        self.assertIsNone(result['coco_small']['AP'])
        self.assertEqual(coco.anns[1]['area'], 2000)

    def test_one_prediction_cannot_recall_two_gt(self):
        box = [10,10,10,10]
        result = evaluate_predictions(dataset([box, box]), [detection(box,.9)], (300,))
        metrics = result['budgets']['300']['local_hybrid_small']
        self.assertAlmostEqual(metrics['AR'], .5)
        self.assertLess(metrics['AP'], .51)

    def test_duplicates_lower_precision_and_agree_with_native_when_all_small(self):
        a, b = [10,10,10,10], [40,40,10,10]
        result = evaluate_predictions(dataset([a,b]),
            [detection(a,.9), detection(a,.8), detection(b,.7)], (300,))['budgets']['300']
        self.assertLess(result['local_hybrid_small']['AP'], 1.)
        self.assertEqual(result['local_hybrid_small'], result['coco_all'])

    def test_budget_is_applied_before_group_filtering(self):
        small, large = [10,10,10,10], [100,100,100,100]
        result = evaluate_predictions(dataset([small,large]),
            [detection(large,.9),detection(small,.8)], (1,2))['budgets']
        self.assertEqual(result['1']['local_hybrid_small']['AP'], 0.)
        self.assertAlmostEqual(result['2']['local_hybrid_small']['AP'], 1.)

    def test_unmatched_group_policy_and_crowd(self):
        small, background = [10,10,10,10], [100,100,100,100]
        result = evaluate_predictions(dataset([small]),
            [detection(background,.9),detection(small,.8)], (300,))['budgets']['300']
        self.assertAlmostEqual(result['local_hybrid_small']['AP'], 1.)
        self.assertLess(result['coco_all']['AP'], 1.)
        false_small = [200,200,10,10]
        result = evaluate_predictions(dataset([small]),
            [detection(false_small,.9),detection(small,.8)], (300,))
        self.assertLess(result['budgets']['300']['local_hybrid_small']['AP'], 1.)
        result = evaluate_predictions(dataset([small,false_small],[0,1]),
            [detection(false_small,.9),detection(false_small,.85),detection(small,.8)], (300,))
        self.assertAlmostEqual(result['budgets']['300']['local_hybrid_small']['AP'], 1.)

    def test_empty_predictions_zero_and_empty_gt_undefined(self):
        result = evaluate_predictions(dataset([[10,10,10,10]]), [], (300,))
        self.assertEqual(result['budgets']['300']['local_hybrid_small']['AP'], 0.)
        result = evaluate_predictions(dataset([]), [], (300,))
        self.assertIsNone(result['budgets']['300']['local_hybrid_small']['AP'])

    def test_numpy_prediction_path(self):
        rows = np.array([[1, 10, 10, 10, 10, .9, 1]])
        result = evaluate_predictions(dataset([[10,10,10,10]]), rows)
        self.assertAlmostEqual(result['budgets']['640']['local_hybrid_small']['AP'], 1.)


class CohortAndMatrixTest(unittest.TestCase):
    def test_scene_cohort_uses_gt_counts_not_predictions_and_excludes_crowd(self):
        coco = dataset([[10,10,10,10]] * 301)
        coco.dataset['images'].extend([dict(id=2,width=640,height=640), dict(id=3,width=640,height=640)])
        coco.dataset['annotations'].extend([
            dict(id=302+i,image_id=2,category_id=1,bbox=[20,20,10,10],area=100,iscrowd=0)
            for i in range(300)])
        coco.dataset['annotations'].append(dict(id=602,image_id=2,category_id=1,
            bbox=[30,30,10,10],area=100,iscrowd=1))
        coco.createIndex()
        self.assertEqual(dense_image_ids(coco), [1])
        self.assertEqual(dense_image_ids(coco,300), [1,2])
        # Predictions from excluded images must not improve or lower cohort AP.
        predictions = [dict(image_id=2,category_id=1,bbox=[20,20,10,10],score=.99),
                       detection([10,10,10,10],.5)]
        scoped = evaluate_predictions(coco,predictions,(640,),image_ids=[1])
        self.assertEqual(scoped['images'],1)
        self.assertEqual(scoped['noncrowd_gt'],301)
        self.assertEqual(scoped['local_small_gt'],301)
        self.assertAlmostEqual(scoped['budgets']['640']['local_hybrid_small']['AR'],1/301)
        self.assertEqual(coco.anns[1]['area'],100)
        self.assertEqual(coco.anns[302]['area'],100)
        empty = evaluate_predictions(coco,predictions,(640,),image_ids=[])
        self.assertEqual(empty['images'],0)
        self.assertIsNone(empty['budgets']['640']['coco_all']['AP'])
        self.assertIsNone(empty['budgets']['640']['local_hybrid_small']['AP'])
        with self.assertRaises(ValueError):
            evaluate_predictions(coco,predictions,image_ids=[1,1])
        with self.assertRaises(ValueError):
            evaluate_predictions(coco,predictions,image_ids=[99])

    def test_numpy_predictions_respect_scene_scope(self):
        coco=dataset([[10,10,10,10]])
        coco.dataset['images'].append(dict(id=2,width=640,height=640))
        coco.createIndex()
        rows=np.array([[2,20,20,10,10,.99,1], [1,10,10,10,10,.5,1]])
        scoped=evaluate_predictions(coco,rows,(640,),image_ids=[1])
        self.assertAlmostEqual(scoped['budgets']['640']['local_hybrid_small']['AP'],1.)
        self.assertLess(evaluate_predictions(coco,rows,(640,))['budgets']['640']['local_hybrid_small']['AP'],1.)

    def test_legacy_fixed_box_swap_remains_explicitly_separate(self):
        boxes=torch.tensor([[.8,.8,.1,.1],[.2,.2,.01,.1]])
        scores=torch.tensor([.9,.3]); before=boxes.clone()
        swapped,count=fixed_box_exchange(boxes,scores,boxes[1:],budget=1)
        self.assertEqual(count,1)
        torch.testing.assert_close(boxes,before)
        torch.testing.assert_close(scores.sort().values,swapped.sort().values)
        self.assertGreater(swapped[1],swapped[0])

    def test_matrix_expands_real_query_budgets(self):
        values=jobs(Path('.'),['b0','support','v2'],list(range(28,35)),['raw','ema'],[640,256])
        self.assertEqual(len(values),84)
        self.assertEqual({v[3] for v in values},{640,256})
        self.assertEqual(len(jobs(Path('.'),['v2'],[34],['ema'])),1)
        for args in ((['v2','v2'],[28],['raw'],[640]),
                     (['v2'],[28],['raw'],[640,640]),
                     (['v2'],[28],['raw'],[0]),
                     (['v2'],[],['raw'],[640])):
            with self.assertRaises(ValueError):
                jobs(Path('.'),*args)


if __name__=='__main__':
    unittest.main()
