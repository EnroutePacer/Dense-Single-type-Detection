"""Actual-K execution, full matrix identities, and real one-to-one evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from PIL import Image
import torch
from torch import nn
from faster_coco_eval import COCO

from engine.deim.dfine_decoder import DFINETransformer
from tools.table_cells.evaluate_sgqs_v2_matrix import main, predict, set_query_budget


class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=SimpleNamespace(num_queries=640,learn_query_content=False)
        self.executed=[]
    def forward(self, inputs):
        k=self.decoder.num_queries
        self.executed.append(k)
        boxes=torch.tensor([.8,.8,.1,.1]).expand(1,k,4).clone()
        boxes[0,-1]=torch.tensor([1.5/32,1.5/32,1/32,1/32])
        return {'pred_boxes':boxes,'pred_logits':torch.linspace(4.,-4.,k).reshape(1,k,1)}


class NativeQueryBudgetTest(unittest.TestCase):
    def test_real_dfine_selector_and_decoder_see_reduced_k_with_same_weights(self):
        torch.manual_seed(42)
        decoder=DFINETransformer(num_classes=1,hidden_dim=32,num_queries=6,
            feat_channels=[32]*3,feat_strides=[8,16,32],num_levels=3,num_points=[1]*3,
            nhead=4,num_layers=2,dim_feedforward=64,dropout=0.,num_denoising=0,
            eval_spatial_size=[32,32],eval_idx=-1,aux_loss=True,reg_max=4)
        class Wrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__(); self.decoder=decoder
            def forward(self, features):
                return self.decoder(features)
        model=Wrapper(decoder).eval()
        weights=copy.deepcopy(model.state_dict())
        features=[torch.randn(1,32,n,n) for n in (4,2,1)]
        bbox_counts=[]; attention_counts=[]
        bbox_hook=decoder.enc_bbox_head.register_forward_pre_hook(
            lambda _module,args:bbox_counts.append(args[0].shape[1]))
        attention_hook=decoder.decoder.register_forward_pre_hook(
            lambda _module,args:attention_counts.append(args[0].shape[1]))
        try:
            with torch.no_grad():
                large=model(features)
                set_query_budget(model,3)
                model.load_state_dict(weights,strict=True)
                small=model(features)
                set_query_budget(model,6)
                model.load_state_dict(weights,strict=True)
                restored=model(features)
        finally:
            bbox_hook.remove(); attention_hook.remove()
        self.assertEqual(bbox_counts,[6,3,6])
        self.assertEqual(attention_counts,[6,3,6])
        self.assertEqual(small['pred_boxes'].shape,(1,3,4))
        for key in large:
            torch.testing.assert_close(large[key],restored[key],rtol=0,atol=0)
        decoder.learn_query_content=True
        with self.assertRaisesRegex(ValueError,'learn_query_content'):
            set_query_budget(model,3)

    def test_timing_excludes_warmup_and_output_count_is_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            Image.new('RGB',(32,32),'white').save(root/'one.png')
            coco=COCO()
            coco.dataset={'images':[dict(id=1,width=32,height=32,file_name='one.png')],
                          'categories':[dict(id=1,name='cell')],'annotations':[]}
            coco.createIndex()
            model=Detector().eval()
            rows,swapped,pairs,timing=predict(model,coco,root,'cpu',expected_queries=256,warmup_iterations=2)
            self.assertEqual(model.executed,[256,256,256])
            self.assertEqual(rows.shape,(256,7))
            self.assertEqual(timing['images'],1)
            self.assertEqual(timing['query_budget'],256)
            self.assertEqual(timing['warmup_iterations'],2)
            self.assertGreater(timing['mean_ms'],0)
            self.assertIsNone(swapped)
            self.assertEqual(pairs,0)
            class WrongCount(Detector):
                def forward(self,inputs):
                    result=super().forward(inputs)
                    result['pred_boxes']=result['pred_boxes'][:,:1]
                    return result
            with self.assertRaisesRegex(ValueError,'exactly 256'):
                predict(WrongCount().eval(),coco,root,'cpu',expected_queries=256,warmup_iterations=0)


class FullMatrixTest(unittest.TestCase):
    def test_full_matrix_real_metrics_query_cache_resume_and_image_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            images=root/'images'; images.mkdir()
            Image.new('RGB',(32,32),'white').save(images/'one.png')
            annotation=root/'val.json'
            annotation.write_text(json.dumps({
                'images':[dict(id=1,width=32,height=32,file_name='one.png')],
                'categories':[dict(id=1,name='cell')],
                'annotations':[dict(id=1,image_id=1,category_id=1,bbox=[1,1,1,1],area=1,iscrowd=0)]}))
            run=root/'runs/v2_e28_r7_seed0'; run.mkdir(parents=True)
            config=run/'resolved.yml'; config.write_text('test: true\n')
            digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (run/'run_manifest.json').write_text(json.dumps(dict(
                schema='sgqs_v2_coverage_allocation_v1',arm='v2',
                inputs={'val_annotation':{'sha256':digest(annotation)}},
                resolved_config_sha256=digest(config))))
            for epoch in (28,29):
                torch.save(dict(last_epoch=epoch,model={},ema={'module':{}}),run/f'checkpoint{epoch:04d}.pth')
            output=root/'results'
            argv=['matrix','--run-root',str(root/'runs'),'--annotation',str(annotation),
                  '--images',str(images),'--output',str(output),'--arms','v2',
                  '--epochs','28','29','--device','cpu','--query-budgets','640','256',
                  '--warmup-iterations','1']
            detector=Detector()
            cfg=SimpleNamespace(yaml_cfg={'HGNetv2':{},'eval_spatial_size':[640,640]},model=detector)
            runtime=mock.patch('tools.table_cells.evaluate_sgqs_v2_matrix.load_runtime_files',
                               return_value=({}, {'sha256':'fixture'}))
            with runtime:
                with mock.patch.object(sys,'argv',argv),mock.patch(
                        'tools.table_cells.evaluate_sgqs_v2_matrix.YAMLConfig',return_value=cfg):
                    main()
                completed=json.loads((output/'completed.json').read_text())
                self.assertEqual(completed['inference_runs'],8)
                self.assertEqual(completed['checkpoint_states'],4)
                self.assertEqual(completed['budget_rows'],16)
                self.assertEqual(detector.executed,[640,640,256,256]*4)
                full=json.loads((output/'v2.e28.raw.k640.json').read_text())
                small=json.loads((output/'v2.e28.raw.k256.json').read_text())
                self.assertEqual(full['identity']['query_budget'],640)
                self.assertEqual(small['identity']['query_budget'],256)
                self.assertNotEqual(full['predictions_sha256'],small['predictions_sha256'])
                self.assertEqual(full['metrics']['budgets']['300']['local_hybrid_small']['AP'],0.)
                self.assertAlmostEqual(full['metrics']['budgets']['640']['local_hybrid_small']['AP'],1.)
                self.assertAlmostEqual(small['metrics']['budgets']['300']['local_hybrid_small']['AP'],1.)
                self.assertIsNone(full['dense_scene_metrics']['budgets']['640']['coco_all']['AP'])
                with mock.patch.object(sys,'argv',argv+['--skip-completed']),mock.patch(
                        'tools.table_cells.evaluate_sgqs_v2_matrix.predict',side_effect=AssertionError('must reuse')):
                    main()
                with mock.patch.object(sys,'argv',argv+['--skip-completed','--query-budgets','640']):
                    with self.assertRaisesRegex(ValueError,'changed'):
                        main()
                Image.new('RGB',(32,32),'black').save(images/'one.png')
                with mock.patch.object(sys,'argv',argv+['--skip-completed']):
                    with self.assertRaisesRegex(ValueError,'changed|cannot be reused'):
                        main()
                (run/'checkpoint0029.pth').unlink()
                with mock.patch.object(sys,'argv',argv+['--skip-completed']):
                    with self.assertRaises(FileNotFoundError):
                        main()


if __name__=='__main__':
    unittest.main()