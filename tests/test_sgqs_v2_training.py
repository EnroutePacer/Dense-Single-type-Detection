"""Durability, same-forward archives and a real tiny-model interrupted run."""
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from faster_coco_eval import COCO

from engine.sgqs_v2.runtime import (atomic_write, atomic_json, capture_rng, restore_rng,
    StepRecorder, latest_checkpoint, validate_recovery, digest, require_finite)
from engine.sgqs_v2.evaluation import EncoderCapture, read_predictions, verified_result
from engine.sgqs_v2.trainer import SGQSV2Trainer
from engine.solver._solver import BaseSolver
from engine.sgqs_v2 import SGQSV2Criterion, SGQSV2Supervisor
from engine.deim import DFINETransformer, DEIMCriterion, HungarianMatcher
from engine.deim.postprocessor import PostProcessor
from engine.optim import ModelEMA


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.project = torch.nn.Conv2d(3, 32, 1)
        self.decoder = DFINETransformer(num_classes=1, hidden_dim=32, num_queries=4,
            feat_channels=[32]*3, feat_strides=[8,16,32], num_levels=3, num_points=[1]*3,
            nhead=4, num_layers=2, dim_feedforward=64, dropout=0., num_denoising=8,
            eval_spatial_size=[32,32], eval_idx=-1, aux_loss=True, reg_max=4,
            sgqs_supervisor=SGQSV2Supervisor(scale_priors=[8,16,32]))

    def forward(self, images, targets=None):
        x = self.project(images)
        return self.decoder([torch.nn.functional.adaptive_avg_pool2d(x, n) for n in (4,2,1)], targets)


class Loader:
    collate_fn = SimpleNamespace(stop_epoch=105)
    sampler = None
    def __init__(self, train):
        self.train = train
    def __len__(self):
        return 1
    def set_epoch(self, epoch):
        self.epoch = epoch
    def __iter__(self):
        # Training consumes RNG; evaluation must not change its future sequence.
        images = torch.rand(2,3,32,32) if self.train else torch.ones(2,3,32,32)
        yield images, [dict(boxes=torch.tensor([[.4,.4,.2,.1]]),
            labels=torch.zeros(1,dtype=torch.long), orig_size=torch.tensor([32,32]),
            image_id=torch.tensor([i])) for i in (1,2)]


class TinyTrainer(SGQSV2Trainer):
    def train(self):
        torch.manual_seed(39); random.seed(39); np.random.seed(39)
        self.model = TinyModel()
        base = DEIMCriterion(HungarianMatcher({'cost_class':2.,'cost_bbox':5.,'cost_giou':2.}),
            {'loss_vfl':1.,'loss_bbox':5.,'loss_giou':2.,'loss_fgl':.15,'loss_ddf':1.5},
            ['vfl','boxes','local'],num_classes=1,reg_max=4)
        self.criterion = SGQSV2Criterion(base, {'loss_sgqs_dense':1., 'loss_sgqs_bbox':2.5, 'loss_sgqs_giou':1.})
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=.001, momentum=.9)
        for group in self.optimizer.param_groups:
            group['initial_lr'] = .001
        self.ema = ModelEMA(self.model)
        self.scaler = torch.amp.GradScaler('cpu', enabled=False)
        self.lr_scheduler = None
        self.device = torch.device('cpu')
        self.postprocessor = PostProcessor(num_classes=1, num_top_queries=4)
        self.train_dataloader, self.val_dataloader = Loader(True), Loader(False)
        self.last_epoch = 27
        self.output_dir = Path(self.cfg.output_dir)
        coco = COCO()
        coco.dataset = {'images':[{'id':i,'width':32,'height':32} for i in (1,2)],
            'categories':[{'id':0,'name':'cell'}], 'annotations':[
                {'id':i,'image_id':i,'category_id':0,'bbox':[9.6,11.2,6.4,3.2],
                 'area':20.48,'iscrowd':0} for i in (1,2)]}
        coco.createIndex()
        self.evaluator = SimpleNamespace(coco_gt=coco)
        if self.recovery:
            self.load_resume_state(self.cfg.resume)


def config(root):
    return SimpleNamespace(output_dir=str(root), resume='', lrsheduler='flatcosine',
        lr_gamma=1., epoches=30, warmup_iter=1, flat_epoch=4000000, no_aug_epoch=0,
        yaml_cfg={'lrsheduler':'flatcosine', 'training_policy': {
            'checkpoint_every':1,'eval_every':1,'eval_initial':True,'eval_budgets':[300,640],
            'dense_min_gt':301,'small_reference_size':[640,640]}}, clip_max_norm=.1, print_freq=10)


class FrameworkTest(unittest.TestCase):
    def test_full_validation_shards_have_no_padding_or_dropped_images(self):
        indices = []
        for rank in (0,1):
            solver = SGQSV2Trainer(config('.'), 'test')
            solver.val_dataloader = torch.utils.data.DataLoader(list(range(5)),batch_size=2)
            with mock.patch.object(BaseSolver,'train'), \
                    mock.patch('engine.misc.dist_utils.is_dist_available_and_initialized',return_value=True), \
                    mock.patch('engine.misc.dist_utils.get_rank',return_value=rank), \
                    mock.patch('engine.misc.dist_utils.get_world_size',return_value=2):
                solver.train()
            indices.extend(int(x) for batch in solver.val_dataloader for x in batch)
        self.assertEqual(sorted(indices),list(range(5)))

    def test_atomic_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint'
            path.write_bytes(b'good')
            def broken(stream):
                stream.write(b'partial'); raise OSError('disk full')
            with self.assertRaises(OSError): atomic_write(path,broken)
            self.assertEqual(path.read_bytes(),b'good')
            self.assertEqual(list(Path(tmp).glob('*.part')),[])
            with self.assertRaises(FileExistsError): atomic_write(path,lambda f:f.write(b'bad'),False)

    def test_rng_roundtrip_and_finite_guard(self):
        rng=capture_rng()
        expected=(random.random(),np.random.rand(),torch.rand(2))
        restore_rng(rng)
        actual=(random.random(),np.random.rand(),torch.rand(2))
        self.assertEqual(expected[:2],actual[:2]); torch.testing.assert_close(expected[2],actual[2])
        with self.assertRaises(FloatingPointError): require_finite(torch.tensor(float('nan')),'cpu','test')

    def test_population_rate_is_not_average_of_batch_rates(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=StepRecorder(tmp,28)
            for covered,total in ((1,1),(0,9)):
                writer.record({'amp_skipped':False,'losses':{'loss':2.},'metrics':{
                    'v2_selected_covered_gt':covered,'v2_gt':total,
                    'v2_selected_gt_coverage':covered/total}})
            summary=writer.summary(); writer.close()
            self.assertEqual(summary['population_rates']['selected_gt_coverage'],.1)
            self.assertEqual(summary['step_means']['loss'],2.)
            self.assertTrue(summary['warnings'])
            writer=StepRecorder(tmp,29)
            writer.record({'amp_skipped':False,'allocation_enabled':False,
                           'losses':{},'metrics':{'v2_gt':10}})
            self.assertEqual(writer.summary()['warnings'],[])
            writer.close()

    def test_capture_preserves_real_forward_and_removes_hooks(self):
        model=TinyModel().eval(); images=torch.ones(2,3,32,32)
        with torch.no_grad():
            expected=model(images)
            with EncoderCapture(model) as capture:
                actual=model(images); data=capture.selected()
        torch.testing.assert_close(expected['pred_boxes'],actual['pred_boxes'],rtol=0,atol=0)
        self.assertEqual(data['dense_scores'].shape,(2,21))
        self.assertEqual(data['source_indices'].shape,(2,4))
        self.assertFalse(model.decoder.enc_score_head._forward_hooks)
        self.assertFalse(model.decoder.decoder._forward_pre_hooks)

    def test_validation_dedup_coverage_and_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path=root/'part.npz'
            predictions=np.zeros((3,4,7)); predictions[:,:,0]=np.array([1,2,1])[:,None]
            np.savez(path,image_ids=[1,2,1],predictions=predictions)
            shards=[{'path':path.name,'sha256':digest(path)}]
            self.assertEqual(read_predictions(root,shards,[1,2]).shape,(8,7))
            with self.assertRaises(ValueError): read_predictions(root,shards,[1,2,3])
            atomic_json(root/'complete.json',{'checkpoint_sha256':'abc','shards':shards})
            self.assertIsNotNone(verified_result(root,'abc'))
            with self.assertRaises(ValueError): verified_result(root,'other')
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): verified_result(root,'abc')

    def test_real_training_recovery_keeps_state_and_completes_raw_ema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); regular=root/'regular'; interrupted=root/'interrupted'
            for run in (regular,interrupted): (run/'attempts/0000').mkdir(parents=True)
            TinyTrainer(config(regular),'run').fit()
            solver=TinyTrainer(config(interrupted),'run')
            original=solver.validate_boundary
            def fail_after_checkpoint(checkpoint):
                if solver.last_epoch==28: raise RuntimeError('simulated interruption before validation')
                original(checkpoint)
            with mock.patch.object(solver,'validate_boundary',side_effect=fail_after_checkpoint):
                with self.assertRaisesRegex(RuntimeError,'simulated'): solver.fit()
            checkpoint=latest_checkpoint(interrupted,'run',1)
            self.assertEqual(checkpoint['epoch'],28)
            (interrupted/'attempts/0001').mkdir()
            cfg=config(interrupted); cfg.resume=checkpoint['path']
            TinyTrainer(cfg,'run',attempt=1,recovery=True).fit()
            expected=torch.load(regular/'checkpoint0029.pth',weights_only=False)
            actual=torch.load(interrupted/'checkpoint0029.pth',weights_only=False)
            for key in expected['model']:
                torch.testing.assert_close(expected['model'][key],actual['model'][key],rtol=0,atol=0)
            for key in expected['ema']['module']:
                torch.testing.assert_close(expected['ema']['module'][key],actual['ema']['module'][key],rtol=0,atol=0)
            self.assertEqual(expected['ema']['updates'],actual['ema']['updates'])
            self.assertEqual(len(list(interrupted.glob('validation/epoch*/raw/complete.json'))),3)
            self.assertEqual(len(list(interrupted.glob('validation/epoch*/ema/complete.json'))),3)
            self.assertTrue((interrupted/'completed.json').exists())
            with self.assertRaises(ValueError): validate_recovery(actual,'different',1)
            with self.assertRaises(ValueError): validate_recovery(actual,'run',8)
            (interrupted/'checkpoint0029.pth').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'): latest_checkpoint(interrupted,'run',1)


if __name__=='__main__':
    unittest.main()
