"""Probe mathematics and noninterference, including actual DN and backward."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from engine.core import YAMLConfig
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.matcher import HungarianMatcher
from engine.sgqs import SGQSCriterion, SGQSQuerySupervisor
from tools.table_cells.supervision_probe import SupervisionProbeCriterion, ranking_rows
from tools.table_cells.replay_probe import CaptureCandidates, coco_metrics

torch.set_num_threads(1)

def fixture(sgqs=True):
    model = DFINETransformer(num_classes=1, hidden_dim=32, num_queries=4,
        feat_channels=[32]*3, feat_strides=[8,16,32], num_levels=3,
        num_points=[1]*3, nhead=4, num_layers=2, dim_feedforward=64,
        dropout=0., num_denoising=8, eval_spatial_size=[32,32], eval_idx=-1,
        aux_loss=True, reg_max=4,
        sgqs_supervisor=SGQSQuerySupervisor(scale_priors=[8,16,32]) if sgqs else None)
    # Fresh zero-initialized residual heads produce identical layers and omit DDF.
    # Exercise the resumed-training case with non-identical distributions.
    with torch.no_grad():
        for head in model.dec_bbox_head:
            head.layers[-1].weight.normal_(0, .02)
    criterion = DEIMCriterion(HungarianMatcher({'cost_class':2.,'cost_bbox':5.,'cost_giou':2.}),
        {'loss_vfl':1.,'loss_bbox':5.,'loss_giou':2.,'loss_fgl':.15,'loss_ddf':1.5},
        ['vfl','boxes','local'],num_classes=1,reg_max=4)
    if sgqs:
        criterion=SGQSCriterion(criterion,{'loss_sgqs_dense':1.,'loss_sgqs_bbox':2.5,
            'loss_sgqs_giou':1.,'loss_sgqs_survival':.25},
            survival_threshold='non_owner_budget',protect_encoder_supervision=True)
    gt=[{'boxes':torch.tensor([[.4,.4,.2,.1],[.6,.6,.2,.1]]), 'labels':torch.zeros(2,dtype=torch.long),
         'image_id':torch.tensor([1])},
        {'boxes':torch.tensor([[.4,.4,.2,.1]]),'labels':torch.zeros(1,dtype=torch.long),'image_id':torch.tensor([2])}]
    return model,criterion,gt


class ProbeTest(unittest.TestCase):
    def test_candidate_capture_leaves_eval_unchanged(self):
        model,_,_=fixture();model.eval()
        features=[torch.randn(1,32,n,n) for n in (4,2,1)]
        with torch.no_grad():
            expected=model(features)
            rng=torch.get_rng_state()
            with CaptureCandidates(model) as capture: actual=model(features)
            torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
            for k in expected:torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
            self.assertEqual(capture.data['boxes'].shape[1],21)
            self.assertEqual(capture.data['selected_boxes'].shape[1],4)

    def test_coco_budget_replay_uses_real_one_to_one_evaluator(self):
        from faster_coco_eval import COCO
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gt.json'
            path.write_text(json.dumps({'images':[{'id':1,'width':100,'height':100}],
                'categories':[{'id':1,'name':'cell'}],
                'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[40,40,10,10],'area':100,'iscrowd':0}]}))
            coco=COCO(str(path))
            predictions=[{'image_id':1,'category_id':1,'bbox':[0,0,1,1],'score':.9} for _ in range(300)]
            predictions.append({'image_id':1,'category_id':1,'bbox':[40,40,10,10],'score':.8})
            capped=coco_metrics(coco,predictions,300);all_queries=coco_metrics(coco,predictions,640)
            self.assertEqual(capped[9],0.)
            self.assertAlmostEqual(all_queries[9],1.,places=6)

    def test_actual_training_losses_gradients_rng_and_ddf_reconstruction(self):
        for use_sgqs in (False,True):
            with self.subTest(sgqs=use_sgqs), tempfile.TemporaryDirectory() as temp:
                torch.manual_seed(10)
                left,base,gt=fixture(use_sgqs);right=copy.deepcopy(left)
                probe=SupervisionProbeCriterion(copy.deepcopy(base),temp)
                features=[torch.randn(2,32,n,n) for n in (4,2,1)]
                kwargs=dict(epoch=28,step=0,epoch_step=5631,input_size=(32,32),global_step=28*5631)
                torch.manual_seed(22)
                expected=base(left(features,gt),gt,**kwargs)
                sum(expected.values()).backward()
                expected_rng=torch.get_rng_state()
                torch.manual_seed(22)
                actual=probe(right(features,gt),gt,**kwargs)
                sum(actual.values()).backward()
                torch.testing.assert_close(torch.get_rng_state(),expected_rng,rtol=0,atol=0)
                self.assertEqual(set(actual),set(expected))
                for k in actual:torch.testing.assert_close(actual[k],expected[k],rtol=0,atol=0)
                for p,q in zip(left.parameters(),right.parameters()):
                    self.assertEqual(p.grad is None,q.grad is None)
                    if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                record=json.loads(next((Path(temp)/'probes').glob('*.jsonl')).read_text())
                self.assertEqual(sum(g['gt_count'] for g in record['groups'].values()),3)
                self.assertIn('dn0_corner_gradient',record)
                for values in record['ddf'].values():
                    self.assertAlmostEqual(values['weighted_reconstruction'],values['observed_weighted_loss'],places=5)
                dn=record['ddf']['dn_0']['groups']
                self.assertGreater(dn['padding/all']['kl_T2']['count'],0)
                self.assertGreater(dn['negative/all']['kl_T2']['count'],0)

    def test_ranking_separates_geometry_from_score_budget(self):
        gt=torch.tensor([[.5,.5,.1,.1]])
        boxes=torch.tensor([[.1,.1,.1,.1],[.5,.5,.1,.1]])
        result=ranking_rows(boxes,torch.tensor([[3.],[1.]]),gt,budgets=(1,2))
        self.assertEqual(result['best_iou_top1'].item(),0.)
        self.assertAlmostEqual(result['best_iou_all'].item(),1.,places=5)
        self.assertEqual(result['first_rank_iou0.5'].item(),2)

    def test_go_can_omit_a_gt_despite_per_layer_coverage(self):
        perms=[[0,1,2],[0,2,1],[0,2,1],[1,0,2],[2,0,1],[2,0,1]]
        indices=[[(torch.arange(3),torch.tensor(p))] for p in perms]
        result=DEIMCriterion._get_go_indices(None,indices[0],indices[1:])
        self.assertEqual(torch.bincount(result[0][1],minlength=3).tolist(),[2,1,0])

    def test_configuration_and_sampling(self):
        for name,base in [('sgqs','SGQSCriterion'),('b0','DEIMCriterion')]:
            config=YAMLConfig(f'configs/probes/{name}_e28_r7.yml')
            self.assertEqual(config.yaml_cfg['epoches'],35)
            self.assertEqual(config.yaml_cfg['checkpoint_freq'],1)
            self.assertEqual(config.yaml_cfg['SupervisionProbeCriterion']['base_criterion'],base)
            probe=config.criterion
            self.assertTrue(probe.due(28,10,5631));self.assertFalse(probe.due(29,10,5631))
            self.assertTrue(probe.due(30,5630,5631));self.assertFalse(probe.due(30,1,5631))

    def test_empty_batch_is_observed_without_nan(self):
        with tempfile.TemporaryDirectory() as temp:
            model,base,gt=fixture()
            for target in gt:
                target['boxes']=torch.empty(0,4);target['labels']=torch.empty(0,dtype=torch.long)
            probe=SupervisionProbeCriterion(base,temp)
            losses=probe(model([torch.randn(2,32,n,n) for n in (4,2,1)],gt),gt,
                         epoch=28,step=0,epoch_step=5,input_size=(32,32))
            sum(losses.values()).backward()
            record=json.loads(next((Path(temp)/'probes').glob('*.jsonl')).read_text())
            self.assertEqual(record['gt_rows'],[])


if __name__=='__main__': unittest.main()
