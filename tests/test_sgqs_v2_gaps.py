import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from faster_coco_eval import COCO

from engine.sgqs_v2.admission import admission_loss, exchange_pairs
from engine.sgqs_v2 import SGQSV2Criterion
from test_sgqs_v2 import fixture, criterion
from tools.table_cells.local_small_eval import evaluate_predictions, is_local_small, fixed_box_exchange
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


class MissingInterventionTest(unittest.TestCase):
    def test_ramp_only_scales_dense_and_owner(self):
        payload, targets, _ = fixture()
        for epoch, step, ramp in [(28,0,0.),(28,50,.5),(29,0,1.)]:
            meta = dict(epoch=epoch,step=step,epoch_step=100)
            old = criterion(survival_start_epoch=28.).loss_sgqs(payload,targets,**meta)
            new = criterion(dense_owner_ramp_epochs=1., survival_start_epoch=28.).loss_sgqs(payload,targets,**meta)
            for key in old:
                factor = 1. if key == 'loss_sgqs_survival' else ramp
                torch.testing.assert_close(new[key], old[key]*factor)
        with self.assertRaisesRegex(ValueError, 'metadata'):
            criterion(dense_owner_ramp_epochs=1.).loss_sgqs(payload,targets)

    def test_no_mixed_factors(self):
        with self.assertRaisesRegex(ValueError, 'single-factor'):
            criterion(dense_owner_ramp_epochs=1., admission_weight=.25)

    def test_boundary_exchange_disjoint_and_quality_improves(self):
        scores = torch.tensor([.9,.8,.7,.6,.5])
        quality = torch.tensor([1.,.2,.8,.7,1.])
        small = torch.tensor([True]*5)
        pos,neg = exchange_pairs(scores,quality,small,budget=2)
        self.assertEqual(pos.tolist(),[4])
        self.assertEqual(neg.tolist(),[1])
        self.assertEqual(len(set(torch.cat([pos,neg]).tolist())),2*len(pos))
        self.assertEqual(len(exchange_pairs(scores,quality,small,budget=5)[0]),0)

    def test_admission_gradient_reaches_only_paired_scores(self):
        scores = torch.tensor([[[1.],[0.]]],requires_grad=True)
        boxes = torch.tensor([[[.8,.8,.1,.1],[.2,.2,.01,.1]]],requires_grad=True)
        targets=[{'boxes': boxes.detach()[0,1:]}]
        loss,count = admission_loss(dict(pred_logits=scores,pred_boxes=boxes),targets,
            [(torch.tensor([1]),torch.tensor([0]))],(640,640),
            lambda count,device:max(count,1),budget=1)
        loss.backward()
        self.assertEqual(count,1)
        self.assertGreater(scores.grad[0,0,0],0)
        self.assertLess(scores.grad[0,1,0],0)
        self.assertIsNone(boxes.grad)

    def test_first_matcher_call_used_and_hook_removed(self):
        class Matcher(nn.Module):
            def __init__(self):
                super().__init__(); self.calls=0
            def forward(self, outputs, targets):
                self.calls+=1
                return {'indices':[(torch.tensor([300 if self.calls==1 else 0]),torch.tensor([0]))]}
        class Base(nn.Module):
            def __init__(self):
                super().__init__(); self.matcher=Matcher()
            def forward(self, outputs, targets, **metadata):
                self.matcher(outputs,targets); self.matcher(outputs,targets)
                return {}
        loss_fn=criterion(admission_weight=.25); loss_fn.base_criterion=Base()
        scores=torch.linspace(1,0,301).reshape(1,301,1).requires_grad_()
        boxes=torch.tensor([.2,.2,.01,.1]).expand(1,301,4)
        losses=loss_fn(dict(pred_logits=scores,pred_boxes=boxes),
            [dict(boxes=boxes[0,:1])],input_size=(640,640))
        self.assertEqual(loss_fn.base_criterion.matcher.calls,2)
        self.assertFalse(loss_fn.base_criterion.matcher._forward_hooks)
        losses['loss_sgqs_admission'].backward()
        self.assertLess(scores.grad[0,300,0],0)

    def test_fixed_box_swap_preserves_scores_and_geometry(self):
        boxes=torch.tensor([[.8,.8,.1,.1],[.2,.2,.01,.1]])
        scores=torch.tensor([.9,.3]); before=boxes.clone()
        swapped,count=fixed_box_exchange(boxes,scores,boxes[1:],budget=1)
        self.assertEqual(count,1)
        torch.testing.assert_close(boxes,before)
        torch.testing.assert_close(scores.sort().values,swapped.sort().values)
        self.assertGreater(swapped[1],swapped[0])

    def test_complete_matrix_cardinality_and_no_duplicates(self):
        self.assertEqual(len(jobs(Path('.'),['r0','r1','r2','r3'],list(range(28,35)),['raw','ema'])),56)
        self.assertEqual(len(jobs(Path('.'),['r0','r1','r2','r3','r4','r5'],list(range(28,35)),['raw','ema'])),84)
        with self.assertRaises(ValueError):
            jobs(Path('.'),['r0','r0'],[28],['raw'])

    def test_actual_detector_dn_and_probe_backward_for_new_arms(self):
        import tempfile
        from test_supervision_probe import fixture as detector_fixture
        from tools.table_cells.supervision_probe import SupervisionProbeCriterion
        for options in (dict(dense_owner_ramp_epochs=1.),dict(admission_weight=.25)):
            model, original, targets = detector_fixture()
            revised = SGQSV2Criterion(original.base_criterion, original.weight_dict,
                survival_start_epoch=28., survival_threshold='non_owner_budget',
                protect_encoder_supervision=True, **options)
            with tempfile.TemporaryDirectory() as directory:
                wrapped = SupervisionProbeCriterion(revised, output_dir=directory)
                outputs = model([torch.randn(2,32,size,size) for size in (4,2,1)], targets)
                losses = wrapped(outputs,targets,epoch=28,step=0,epoch_step=10,input_size=(32,32))
                self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                sum(losses.values()).backward()
                self.assertFalse(revised.base_criterion.matcher._forward_hooks)


if __name__=='__main__':
    unittest.main()
