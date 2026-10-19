"""Exercise full matrix bookkeeping and real AP evaluation on a tiny CPU fixture."""
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

from tools.table_cells.evaluate_sgqs_v2_matrix import main


class Detector(nn.Module):
    def forward(self, inputs):
        boxes = torch.tensor([.8,.8,.1,.1]).expand(1,640,4).clone()
        boxes[0,-1] = torch.tensor([1.5/32,1.5/32,1/32,1/32])
        return {'pred_boxes': boxes,
                'pred_logits': torch.linspace(4.,-4.,640).reshape(1,640,1)}


class FullMatrixTest(unittest.TestCase):
    def test_full_requested_matrix_real_metrics_resume_and_image_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root/'images'; images.mkdir()
            Image.new('RGB',(32,32),'white').save(images/'one.png')
            annotation = root/'val.json'
            annotation.write_text(json.dumps({
                'images':[dict(id=1,width=32,height=32,file_name='one.png')],
                'categories':[dict(id=1,name='cell')],
                'annotations':[dict(id=1,image_id=1,category_id=1,bbox=[1,1,1,1],area=1,iscrowd=0)]}))
            run = root/'runs/r0_e28_r7_seed0'; run.mkdir(parents=True)
            config = run/'resolved.yml'; config.write_text('test: true\n')
            digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
            (run/'run_manifest.json').write_text(json.dumps(dict(
                inputs={'val_annotation': {'sha256': digest(annotation)}},
                resolved_config_sha256=digest(config))))
            for epoch in (28,29):
                torch.save(dict(last_epoch=epoch,model={},ema={'module':{}}),run/f'checkpoint{epoch:04d}.pth')
            output=root/'results'
            argv=['matrix','--run-root',str(root/'runs'),'--annotation',str(annotation),
                  '--images',str(images),'--output',str(output),'--arms','r0',
                  '--epochs','28','29','--device','cpu']
            cfg=SimpleNamespace(yaml_cfg={'HGNetv2':{},'eval_spatial_size':[640,640]},model=Detector())
            with mock.patch.object(sys,'argv',argv), mock.patch(
                    'tools.table_cells.evaluate_sgqs_v2_matrix.YAMLConfig',return_value=cfg):
                main()
            completed=json.loads((output/'completed.json').read_text())
            self.assertEqual(completed['states'],4)
            self.assertEqual(completed['budget_rows'],8)
            metrics=json.loads((output/'r0.e28.raw.json').read_text())['metrics']['budgets']
            self.assertEqual(metrics['300']['local_hybrid_small']['AP'],0.)
            self.assertAlmostEqual(metrics['640']['local_hybrid_small']['AP'],1.)
            with mock.patch.object(sys,'argv',argv+['--skip-completed']), mock.patch(
                    'tools.table_cells.evaluate_sgqs_v2_matrix.predict',side_effect=AssertionError('must reuse')):
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
