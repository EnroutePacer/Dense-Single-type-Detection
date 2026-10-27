"""Candidate identity, conflicting supervision, gradients and native DN training."""
import copy
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from engine.sgqs_v2 import SGQSV2Criterion, SGQSV2Supervisor
from engine.sgqs_v2.allocation import allocation_loss, build_allocation, geometry_assignment
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.matcher import HungarianMatcher

torch.set_num_threads(1)
WEIGHTS = {'loss_sgqs_dense': 1., 'loss_sgqs_bbox': 2.5, 'loss_sgqs_giou': 1.}
META = dict(epoch=29, step=0, epoch_step=100, input_size=(640, 640))


def match(src, dst):
    return torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)


def fixture(rescue=True):
    gt = torch.tensor([[.2,.2,.1,.1], [.6,.6,.1,.1], [.8,.2,.1,.1]])
    background = torch.tensor([.2,.8,.05,.05])
    boxes = torch.stack([gt[0], gt[0], gt[1], background,
                         gt[2] if rescue else background, background])
    logits = torch.tensor([4.,3.,1.,0.,-.5,-1.]).reshape(1,6,1).requires_grad_()
    selected = torch.tensor([[0,1,2,3]])
    payload = dict(dense_logits=logits,dense_boxes=boxes[None],
        anchor_centers=boxes[None, :, :2].clone(),
        valid_mask=torch.ones(1,6,dtype=torch.bool),ignore_mask=torch.zeros(1,6,dtype=torch.bool),
        spatial_shapes=torch.tensor([[1,6]]),selected_indices=selected,
        owner_indices=[torch.tensor([0,4])],target_indices=[torch.tensor([0,2])],
        owner_levels=[torch.tensor([0,0])],owner_boxes=[(boxes[[0,4]]+.005).requires_grad_()],
        assignment_misses=torch.zeros(1,dtype=torch.long),effective_k=4,selection_pool_size=6,
        topk_cutoff=torch.tensor([0.]),selection_scores=logits.detach()[...,0])
    outputs=dict(pred_logits=torch.tensor([3.,-4.,3.,-4.]).reshape(1,4,1).requires_grad_(),
        pred_boxes=boxes[None,:4].clone().requires_grad_(),sgqs_aux_outputs=payload,
        enc_aux_outputs=[dict(pred_logits=logits[:,:4],pred_boxes=boxes[None,:4].clone())])
    final=[match([0,2,3],[0,1,2])]
    record={'classification':[match([0,2,3],[0,1,2])],
            'boxes':[match([0,1,2,3],[0,0,1,2])]}
    targets=[dict(boxes=gt,labels=torch.zeros(3,dtype=torch.long))]
    return payload,outputs,targets,final,record


class FixedMatcher(nn.Module):
    def __init__(self, indices):
        super().__init__(); self.indices=indices; self.calls=0
    def forward(self, outputs, targets):
        self.calls += 1
        return {'indices':self.indices}


class Base(nn.Module):
    def __init__(self, indices, record):
        super().__init__(); self.matcher=FixedMatcher(indices); self.record=record
    def forward(self, outputs, targets, **meta):
        self.matcher(outputs,targets)
        return {'loss_base':outputs['pred_logits'].sum()*0.}
    def get_encoder_supervision(self):
        return [self.record]


def criterion(data=None, **options):
    data = fixture() if data is None else data
    return SGQSV2Criterion(Base(data[3],data[4]),WEIGHTS,**options)


class AllocationTest(unittest.TestCase):
    def test_rescue_replaces_duplicate_and_detaches_teacher(self):
        data=fixture(); payload,out,gt,indices,record=data
        pairs,overflow,stats=build_allocation(*data)
        i,j,rescue=pairs[0]
        self.assertEqual((i.tolist(),j.tolist(),rescue.tolist()),([1],[4],[True]))
        self.assertEqual(stats['selected_covered_gt'].item(),2)
        self.assertEqual(stats['donor_go_same_gt'].item(),1)
        loss,count=allocation_loss(payload['dense_logits'],pairs,.2,lambda n,d:max(n,1))
        grads=torch.autograd.grad(loss,[payload['dense_logits'],out['pred_logits'],out['pred_boxes']],allow_unused=True)
        self.assertEqual(count,1)
        torch.testing.assert_close(grads[0].flatten(),torch.tensor([0.,1.,0.,0.,-1.,0.]))
        self.assertIsNone(grads[1]); self.assertIsNone(grads[2])

    def test_prefix_only_lowers_redundant_query(self):
        data=fixture(False)
        pairs,overflow,stats=build_allocation(*data)
        self.assertEqual(tuple(x.tolist() for x in pairs[0]),([1],[2],[False]))
        self.assertEqual(stats['rescue_pairs'].item(),0)
        loss,_=allocation_loss(data[0]['dense_logits'],pairs,.2,lambda n,d:max(n,1))
        grad=torch.autograd.grad(loss,data[0]['dense_logits'])[0].flatten()
        torch.testing.assert_close(grad,torch.tensor([0.,1.,0.,0.,0.,0.]))

    def test_prefix_margin_is_enforced_after_rank_order_is_correct(self):
        data=fixture(False)
        with torch.no_grad(): data[0]['dense_logits'][0,2]=3.1
        pairs,_,_=build_allocation(*data,margin=.2)
        loss,_=allocation_loss(data[0]['dense_logits'],pairs,.2,lambda n,d:max(n,1))
        self.assertAlmostEqual(loss.item(),.1,places=5)
        with torch.no_grad(): data[0]['dense_logits'][0,2]=3.3
        self.assertEqual(len(build_allocation(*data,margin=.2)[0][0][0]),0)

    def test_cross_instance_drift_is_not_redundancy(self):
        data=fixture(); p,o,t,_,_=data
        p['anchor_centers'][0,1]=t[0]['boxes'][1,:2]
        p['dense_boxes'][0,1]=t[0]['boxes'][1]
        o['enc_aux_outputs'][0]['pred_boxes'][0,1]=t[0]['boxes'][1]
        pairs,overflow,stats=build_allocation(*data)
        self.assertEqual(len(pairs[0][0]),0)
        self.assertEqual(stats['stage_disagreement'].item(),1)

    def test_other_gt_overlap_is_protected_even_if_best_gt_is_clear(self):
        data=fixture();p,o,t,_,_=data
        t[0]['boxes'][1]=torch.tensor([.23,.2,.1,.1])
        p['dense_boxes'][0,2]=t[0]['boxes'][1]
        o['enc_aux_outputs'][0]['pred_boxes'][0,2]=t[0]['boxes'][1]
        with torch.no_grad(): o['pred_boxes'][0,2]=t[0]['boxes'][1]
        pairs,overflow,stats=build_allocation(*data)
        self.assertEqual(len(pairs[0][0]),0)
        self.assertGreater(stats['donor_center_ambiguous'].item(),0)

    def test_positive_and_cross_gt_go_protections(self):
        for protection in ('final','classification','owner','go'):
            data=fixture();p,o,t,f,r=data
            if protection=='final': f[0]=match([0,1,2],[0,2,1])
            elif protection=='classification': r['classification'][0]=match([0,1,2],[0,2,1])
            elif protection=='go': r['boxes'][0]=match([0,1,2,3],[0,1,1,2])
            else:
                p['owner_indices'][0]=torch.tensor([0,1,4])
                p['target_indices'][0]=torch.tensor([0,0,2])
            pairs,_,_=build_allocation(*data)
            self.assertEqual(len(pairs[0][0]),0,protection)

    def test_prefix_receiver_requires_compatible_encoder_positive(self):
        data=fixture(False)
        data[4]['classification'][0]=match([0,3],[0,2])
        self.assertEqual(len(build_allocation(*data)[0][0][0]),0)

    def test_same_gt_cannot_pair_and_each_donor_is_used_once(self):
        data=fixture()
        pairs,_,_=build_allocation(*data)
        self.assertEqual(len(pairs[0][0].unique()),len(pairs[0][0]))
        # Rescue takes the only donor, so B's prefix cannot reuse it.
        self.assertEqual(pairs[0][2].tolist(),[True])
        data=fixture(False)
        with torch.no_grad(): data[0]['dense_logits'][0,2]=5.
        self.assertEqual(len(build_allocation(*data)[0][0][0]),0)

    def test_empty_gt_and_no_pairs_are_connected_finite_zero(self):
        data=fixture();p,o,t,f,r=data
        t[0]['boxes']=torch.empty(0,4);t[0]['labels']=torch.empty(0,dtype=torch.long)
        p['owner_indices']=[torch.empty(0,dtype=torch.long)];p['target_indices']=[torch.empty(0,dtype=torch.long)]
        f[0]=match([],[]);r['classification'][0]=match([],[]);r['boxes'][0]=match([],[])
        pairs,_,_=build_allocation(*data)
        loss,_=allocation_loss(p['dense_logits'],pairs,.2,lambda n,d:max(n,1e-12))
        self.assertEqual(loss.item(),0.)
        self.assertTrue(torch.equal(torch.autograd.grad(loss,p['dense_logits'])[0],torch.zeros_like(p['dense_logits'])))

    def test_invalid_anchor_and_source_identity_mismatch(self):
        data=fixture();data[0]['valid_mask'][0,4]=False
        self.assertEqual(build_allocation(*data)[0][0][1].tolist(),[2])
        data=fixture();data[0]['selected_indices']=torch.tensor([[1,0,2,3]])
        with self.assertRaisesRegex(ValueError,'scores'):
            build_allocation(*data)

    def test_chunked_geometry_ties_ambiguity_and_coverage(self):
        gt=torch.tensor([[.2,.2,.1,.1],[.2,.2,.1,.1]])
        boxes=torch.cat([gt,torch.tensor([[.8,.8,.1,.1]])])
        valid=torch.tensor([True,False,True])
        a=geometry_assignment(boxes,gt,valid,chunk_size=1)
        b=geometry_assignment(boxes,gt,valid,chunk_size=8)
        for first,second in zip(a,b): torch.testing.assert_close(first,second)
        self.assertEqual(a[0].tolist(),[-1,-1,-1])
        self.assertEqual(a[3].tolist(),[True,True])

    def test_sparse_ddp_gradient_matches_pooled_not_rank_mean(self):
        # One global pair on two ranks. DDP averages gradients, hence denominator=.5.
        c=criterion()
        with mock.patch('engine.sgqs.criterion.is_dist_available_and_initialized',return_value=True), \
             mock.patch('engine.sgqs.criterion.get_world_size',return_value=2), \
             mock.patch('torch.distributed.all_reduce',side_effect=lambda x:x.fill_(1)):
            self.assertEqual(c._distributed_average_denominator(1,'cpu').item(),.5)
        p=torch.tensor(.7,requires_grad=True)
        pair=(torch.tensor([0]),torch.tensor([1]),torch.tensor([True]))
        z=torch.stack([p,-p]).reshape(1,2,1)
        pooled,_=allocation_loss(z,[pair],.2,lambda n,d:1.)
        rank0,_=allocation_loss(z,[pair],.2,lambda n,d:.5)
        empty=(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.bool))
        rank1,_=allocation_loss(z,[empty],.2,lambda n,d:.5)
        want=torch.autograd.grad(pooled,p,retain_graph=True)[0]
        actual=torch.autograd.grad((rank0+rank1)/2,p)[0]
        torch.testing.assert_close(want,actual)


class CriterionTest(unittest.TestCase):
    def test_rescue_leaves_aux_classification_but_preserves_geometry(self):
        data=fixture();p,o,t,_,_=data
        c=criterion(data);losses=c(o,t,**META)
        self.assertEqual(c.base_criterion.matcher.calls,1)
        self.assertFalse(c.base_criterion.matcher._forward_hooks)
        self.assertNotIn('loss_sgqs_survival',losses)
        dense_grad=torch.autograd.grad(losses['loss_sgqs_dense'],p['dense_logits'],retain_graph=True)[0]
        self.assertEqual(dense_grad[0,4,0].item(),0.)
        geometry=torch.autograd.grad(losses['loss_sgqs_bbox'],p['owner_boxes'][0],retain_graph=True)[0]
        self.assertGreater(geometry[1].abs().sum().item(),0.)
        total_grad=torch.autograd.grad(sum(losses.values()),p['dense_logits'])[0]
        self.assertLess(total_grad[0,4,0].item(),0.)
        self.assertEqual(c.get_logging_metrics()['v2_rescue_classification_masked'].item(),1.)

    def test_support_keeps_owner_classification_without_allocation_mask(self):
        data=fixture();p,o,t,_,_=data
        c=criterion(data,allocation_weight=0.)
        losses=c(o,t,**META)
        self.assertEqual(losses['loss_sgqs_allocation'].item(),0.)
        self.assertEqual(c.get_logging_metrics()['v2_rescue_classification_masked'].item(),0.)
        grad=torch.autograd.grad(losses['loss_sgqs_dense'],p['dense_logits'])[0]
        self.assertNotEqual(grad[0,4,0].item(),0.)

    def test_iou_is_bce_target_and_positive_mean_is_global(self):
        data=fixture();p,_,t,_,_=data
        p['spatial_shapes']=torch.tensor([[1,4],[1,1],[1,1]])
        p['owner_levels']=[torch.tensor([0,1])]
        qualities=[torch.tensor([0.,.8])]
        c=criterion(data)
        loss=c._dense_loss(p,t,qualities)
        grad=torch.autograd.grad(loss,p['dense_logits'])[0]
        owners=p['owner_indices'][0]
        expected=(.25+.75*qualities[0])*(p['dense_logits'][0,owners,0].sigmoid()-qualities[0])/2
        torch.testing.assert_close(grad[0,owners,0],expected)
        self.assertGreater(grad[0,0,0].item(),0.)

    def test_one_absolute_ramp_for_all_auxiliary_terms(self):
        reference_data=fixture(); reference=criterion(reference_data)(reference_data[1],reference_data[2],**META)
        for step,ramp in [(0,0.),(50,.5)]:
            data=fixture();c=criterion(data)
            losses=c(data[1],data[2],epoch=28,step=step,epoch_step=100)
            for key in reference:
                torch.testing.assert_close(losses[key],reference[key]*ramp)
        with self.assertRaisesRegex(ValueError,'metadata'):
            criterion()._ramp(dict(epoch=28))
        self.assertEqual(criterion()._ramp(dict(epoch=30,step=0,epoch_step=1)),1.)

    def test_no_pair_is_not_a_missing_collective(self):
        counts=[]
        for has_rescue in (True,False):
            data=fixture(has_rescue)
            if not has_rescue:
                data[4]['classification'][0]=match([0,1,2,3],[0,0,1,2])
            c=criterion(data)
            with mock.patch.object(c,'_distributed_average_denominator',wraps=c._distributed_average_denominator) as den, \
                 mock.patch.object(c,'_distributed_sum',wraps=c._distributed_sum) as red:
                c(data[1],data[2],**META)
                counts.append((den.call_count,red.call_count))
        self.assertEqual(counts[0],counts[1])

    def test_legacy_parameters_and_invalid_values_rejected(self):
        with self.assertRaises(TypeError): criterion(dense_positive_target='iou')
        for options in ({'ramp_epochs':-1},{'allocation_weight':float('nan')},
                        {'teacher_score':.05},{'iou_chunk_size':0},{'ambiguity_gap':0}):
            with self.assertRaises(ValueError): criterion(**options)


class NativeIntegrationTest(unittest.TestCase):
    def model(self,supervisor):
        return DFINETransformer(num_classes=1,hidden_dim=32,num_queries=4,
            feat_channels=[32]*3,feat_strides=[8,16,32],num_levels=3,num_points=[1]*3,
            nhead=4,num_layers=2,dim_feedforward=64,dropout=0.,num_denoising=8,
            eval_spatial_size=[32,32],eval_idx=-1,aux_loss=True,reg_max=4,
            sgqs_supervisor=supervisor)

    def test_native_dn_backward_and_eval_exact_noninterference(self):
        torch.manual_seed(37)
        supervisor=SGQSV2Supervisor(scale_priors=[8,16,32],bbox_chunk_size=7)
        model=self.model(supervisor)
        with torch.no_grad():
            for head in model.dec_bbox_head: head.layers[-1].weight.normal_(0,.02)
        base=DEIMCriterion(HungarianMatcher({'cost_class':2.,'cost_bbox':5.,'cost_giou':2.}),
            {'loss_vfl':1.,'loss_bbox':5.,'loss_giou':2.,'loss_fgl':.15,'loss_ddf':1.5},
            ['vfl','boxes','local'],num_classes=1,reg_max=4)
        c=SGQSV2Criterion(base,WEIGHTS)
        features=[torch.randn(2,32,n,n,requires_grad=True) for n in (4,2,1)]
        gt=[dict(boxes=torch.tensor([[.4,.4,.2,.1],[.6,.6,.2,.1]]),labels=torch.zeros(2,dtype=torch.long)),
            dict(boxes=torch.tensor([[.4,.4,.2,.1]]),labels=torch.zeros(1,dtype=torch.long))]
        model.train();outputs=model(features,gt)
        self.assertIn('dn_outputs',outputs)
        self.assertFalse(outputs['sgqs_aux_outputs']['dense_boxes'].requires_grad)
        centers = outputs['sgqs_aux_outputs']['anchor_centers']
        self.assertFalse(centers.requires_grad)
        torch.testing.assert_close(centers[0, 0], torch.tensor([.125, .125]))
        torch.testing.assert_close(centers[0, 1], torch.tensor([.375, .125]))
        losses=c(outputs,gt,**META)
        self.assertTrue(any('ddf_dn' in name for name in losses))
        sum(losses.values()).backward()
        self.assertTrue(all(torch.isfinite(f.grad).all() for f in features))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        plain=self.model(None);plain.load_state_dict(model.state_dict(),strict=True)
        plain.eval();model.eval()
        with torch.no_grad():
            expected=plain([f.detach() for f in features]);actual=model([f.detach() for f in features])
        self.assertNotIn('sgqs_aux_outputs',actual)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(expected[key],actual[key],rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
