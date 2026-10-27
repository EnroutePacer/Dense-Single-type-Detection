"""Runtime choices must come from the command, never an experiment identifier."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from engine.sgqs_v2.configuration import parse_args, resolved_config, prepare_run, verify_recovery, ROOT
from tools.table_cells import train_sgqs_v2 as launch
from engine.core import YAMLConfig, register
from engine.core.workspace import GLOBAL_CONFIG
from engine.sgqs_v2.trainer import SGQSV2Trainer
from test_sgqs_v2_training import TinyModel


def command(root, **changes):
    values = dict(config=ROOT/'configs/sgqs_v2/train.yml', output_dir=root/'run', mode='scratch',
        device='cpu', epochs=3, batch_size=6, val_batch_size=4, workers=0, seed=19,
        num_queries=12, checkpoint_every=2, eval_every=2, train_images=root/'images',
        val_images=root/'images', train_annotations=root/'train.json', val_annotations=root/'val.json',
        sgqs_start_epoch=0, sgqs_ramp_epochs=2)
    values.update(changes)
    argv = [v for key,value in values.items() for v in ('--'+key.replace('_','-'),str(value))]
    return argv+['--no-amp','--no-ema','--eval-initial','--input-size','32','32','--eval-budgets','5','12']


class GenericLaunchTest(unittest.TestCase):
    def test_real_general_entry_scratch_no_amp_no_ema_native_schedule_and_recovery(self):
        from PIL import Image
        if 'TinyModel' not in GLOBAL_CONFIG:
            register()(TinyModel)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir()
            for i in (1,2): Image.new('RGB',(32,32),(i*60,100,80)).save(root/'images'/f'{i}.png')
            dataset={'images':[{'id':i,'width':32,'height':32,'file_name':f'{i}.png'} for i in (1,2)],
                'categories':[{'id':0,'name':'cell'}], 'annotations':[
                    {'id':i,'image_id':i,'category_id':0,'bbox':[9.6,11.2,6.4,3.2],
                     'area':20.48,'iscrowd':0} for i in (1,2)]}
            for split in ('train','val'): (root/f'{split}.json').write_text(json.dumps(dataset),encoding='utf-8')
            cfg=YAMLConfig(str(ROOT/'configs/sgqs_v2/train.yml')).yaml_cfg
            cfg.pop('__include__',None)
            cfg.update(model='TinyModel', optimizer={'type':'SGD','lr':.001,'momentum':.9},
                lrsheduler=None, lr_scheduler={'type':'MultiStepLR','milestones':[1,2],'gamma':.5})
            cfg.pop('lr_warmup_scheduler',None)
            cfg['DEIMCriterion']['reg_max']=4
            path=root/'tiny.yml'; path.write_text(yaml.safe_dump(cfg),encoding='utf-8')
            argv=command(root,config=path,num_queries=4,batch_size=2,val_batch_size=2)+[
                '--eval-budgets','2','4','--no-eval-initial']
            original=SGQSV2Trainer.validate_boundary
            def interruption(solver, checkpoint):
                if solver.last_epoch==1: raise RuntimeError('injected generic-run interruption')
                return original(solver,checkpoint)
            with mock.patch.dict('os.environ',{'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}), \
                    mock.patch('sys.stdout',new_callable=io.StringIO):
                with mock.patch.object(SGQSV2Trainer,'validate_boundary',interruption):
                    with self.assertRaisesRegex(RuntimeError,'injected generic'):
                        launch.main(argv)
                self.assertTrue((root/'run/checkpoint_init.pth').exists())
                self.assertFalse((root/'run/checkpoint0000.pth').exists())
                self.assertTrue((root/'run/checkpoint0001.pth').exists())
                launch.main(['--mode','recover','--output-dir',str(root/'run'),'--device','cpu','--attempt','1'])
            state=torch.load(root/'run/checkpoint0002.pth',weights_only=False)
            self.assertNotIn('ema',state)
            self.assertEqual(state['scaler'],{})
            self.assertEqual(state['last_epoch'],2)
            self.assertAlmostEqual(state['optimizer']['param_groups'][0]['lr'],.00025)
            self.assertEqual(state['sgqs_v2_training']['initial_epoch'],-1)
            self.assertFalse((root/'run/validation/initial').exists())
            self.assertFalse(list((root/'run/validation').glob('*/ema')))
            metrics=json.loads((root/'run/validation/epoch0002/raw/complete.json').read_text())
            self.assertEqual(set(metrics['full']['budgets']),{'2','4'})
            self.assertEqual(metrics['full']['input_size'],[32,32])
            self.assertNotIn('dense_ge301',metrics)
            self.assertTrue((root/'run/completed.json').exists())

    def setup_inputs(self, root):
        (root/'images').mkdir()
        for name in ('train','val'):
            (root/f'{name}.json').write_text('{"images":[]}',encoding='utf-8')

    def test_runtime_values_override_experiment_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.setup_inputs(root)
            args=parse_args(command(root,config=ROOT/'configs/sgqs_v2/v2_e28_r7.yml')+['-u','sync_bn=False'])
            cfg,inputs=resolved_config(args,2)
            self.assertEqual(cfg['epoches'],3)
            self.assertEqual(cfg['seed'],19)
            self.assertEqual(cfg['train_dataloader']['total_batch_size'],6)
            self.assertFalse(cfg['use_amp']); self.assertFalse(cfg['use_ema'])
            self.assertEqual(cfg['SGQSV2Criterion']['start_epoch'],0)
            self.assertEqual(cfg['DFINETransformer']['num_queries'],12)
            self.assertEqual(cfg['training_policy']['eval_budgets'],[5,12])
            self.assertIsNone(cfg['training_policy']['dense_min_gt'])
            self.assertEqual(cfg['eval_spatial_size'],[32,32])
            self.assertIsNone(cfg['resume']); self.assertIsNone(cfg['tuning'])
            self.assertEqual(cfg['val_dataloader']['dataset']['transforms']['ops'][0]['size'],[32,32])
            with self.assertRaisesRegex(ValueError,'divisible'):
                resolved_config(args,4)

    def test_explicit_overrides_and_arbitrary_resume_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.setup_inputs(root)
            checkpoint=root/'weights.pth'
            torch.save({'model':{},'optimizer':{},'last_epoch':4},checkpoint)
            args=parse_args(command(root,mode='resume',checkpoint=checkpoint,epochs=9,sgqs_start_epoch=5)+
                            ['-u','optimizer.lr=0.0007','lrsheduler=null'])
            cfg,_=resolved_config(args,1)
            self.assertEqual(cfg['resume'],str(checkpoint.resolve()))
            self.assertEqual(cfg['optimizer']['lr'],.0007)
            self.assertIsNone(cfg['lrsheduler'])
            self.assertEqual(cfg['SGQSV2Criterion']['start_epoch'],5)
            self.assertEqual(cfg['epoches'],9)
            args.epochs=5
            with self.assertRaisesRegex(ValueError,'endpoint'): resolved_config(args,1)

    def test_prepare_and_recovery_freeze_actual_cli_not_a_fixed_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.setup_inputs(root)
            args=parse_args(command(root))
            runtime={'engine/example.py':{'sha256':'abc','bytes':3}}
            with mock.patch('engine.sgqs_v2.configuration.runtime_identity',return_value=runtime):
                manifest=prepare_run(args,1)
                self.assertEqual(manifest['cli']['seed'],19)
                with self.assertRaises(FileExistsError): prepare_run(args,1)
                recover=parse_args(['--mode','recover','--output-dir',str(root/'run'),
                                    '--device','cpu','--attempt','2'])
                self.assertEqual(verify_recovery(recover,1),manifest)
                with self.assertRaisesRegex(ValueError,'world size'): verify_recovery(recover,2)
                (root/'val.json').write_text('changed',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'Input changed'): verify_recovery(recover,1)

    def test_missing_choices_and_implicit_recovery_overrides_are_rejected(self):
        with mock.patch('sys.stderr',new_callable=io.StringIO):
            with self.assertRaises(SystemExit): parse_args(['--mode','scratch','--output-dir','x','--device','cpu'])
            for extra in ([],['--attempt','0'],['--attempt','1','--epochs','100']):
                with self.assertRaises(SystemExit):
                    parse_args(['--mode','recover','--output-dir','x','--device','cpu',*extra])

    def test_single_cpu_needs_no_cuda_or_distributed_initialization(self):
        with mock.patch.dict('os.environ',{'WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}), \
                mock.patch.object(launch.dist,'init_process_group') as init, \
                mock.patch.object(launch.torch.cuda,'set_device') as cuda:
            self.assertEqual(launch.setup_processes('cpu'),(1,0,0))
            init.assert_not_called(); cuda.assert_not_called()

    def test_two_ranks_are_supported_without_an_eight_gpu_contract(self):
        with mock.patch.dict('os.environ',{'WORLD_SIZE':'2','RANK':'1','LOCAL_RANK':'1'}), \
                mock.patch.object(launch.dist,'init_process_group') as init:
            self.assertEqual(launch.setup_processes('cpu'),(2,1,1))
            self.assertEqual(init.call_args.kwargs['backend'],'gloo')

    def test_no_experiment_dependency_in_generic_entry(self):
        for name in ('engine/sgqs_v2/configuration.py','tools/table_cells/train_sgqs_v2.py',
                     'tools/table_cells/run_sgqs_v2.sh'):
            text=(ROOT/name).read_text(encoding='utf-8')
            self.assertNotIn('EXP-TRAIN-008',text)
            self.assertNotIn('frozen_inputs',text)
            self.assertNotIn('prepare_sgqs_v2_run',text)


if __name__=='__main__':
    unittest.main()
