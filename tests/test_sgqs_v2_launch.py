"""Exercise the coverage-allocation configuration and prepared-run contract."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import yaml

from tools.table_cells import prepare_sgqs_v2_run as prepare
from tools.table_cells import train_sgqs_v2 as launch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint():
    return dict(last_epoch=27, model={}, ema={'module': {}}, optimizer={}, scaler={})


class ConfigContractTest(unittest.TestCase):
    def test_active_runtime_manifest_matches_deployment_files(self):
        files, _ = prepare.load_runtime_files(prepare.ROOT, prepare.RUNTIME_FILES)
        for name in ('engine/sgqs_v2/allocation.py', 'engine/sgqs_v2/candidate_supervision.py',
                     'engine/sgqs_v2/criterion.py', 'engine/data/dataset/coco_eval.py',
                     'configs/sgqs_v2/common.yml', 'tools/table_cells/train_sgqs_v2.py'):
            self.assertIn(name, files)

    def test_active_arms_and_shared_training_recipe(self):
        self.assertEqual(set(prepare.ARMS), {'b0', 'support', 'v2', 'sgqs'})
        self.assertEqual(prepare.PRIMARY_ARMS, ('b0', 'support', 'v2'))
        baseline = prepare.arm_config('b0')
        self.assertEqual(baseline['criterion'], 'DEIMCriterion')
        self.assertIsNone(baseline['DFINETransformer']['sgqs_supervisor'])
        for arm in prepare.ARMS:
            config = prepare.arm_config(arm)
            self.assertEqual(config['epoches'], 35)
            self.assertEqual(config['checkpoint_freq'], 1)
            self.assertEqual(config['seed'], 0)
            self.assertTrue(config['use_amp'])
            self.assertTrue(config['use_ema'])
            self.assertEqual(config['DFINETransformer']['num_queries'], 640)
            self.assertFalse(config['DFINETransformer']['learn_query_content'])
            self.assertEqual(config['PostProcessor']['num_top_queries'], 640)
            self.assertEqual(config['evaluator']['max_dets'], 640)
            self.assertFalse(config['efgl']['enabled'])
            self.assertNotEqual(config['criterion'], 'SupervisionProbeCriterion')
            self.assertEqual(config['optimizer'], baseline['optimizer'])
            self.assertEqual(config['DEIMCriterion'], baseline['DEIMCriterion'])
            for split in ('train', 'val'):
                self.assertEqual(config[f'{split}_dataloader'], baseline[f'{split}_dataloader'])

    def test_support_changes_only_allocation_weight(self):
        support = copy.deepcopy(prepare.arm_config('support'))
        full = copy.deepcopy(prepare.arm_config('v2'))
        for config in (support, full):
            config.pop('__include__', None)
            config.pop('output_dir')
        self.assertEqual(support['SGQSV2Criterion'].pop('allocation_weight'), 0.)
        self.assertEqual(full['SGQSV2Criterion'].pop('allocation_weight'), .25)
        self.assertEqual(support, full)
        method = full['SGQSV2Criterion']
        self.assertEqual(method['start_epoch'], 28.)
        self.assertEqual(method['ramp_epochs'], 1.)
        self.assertEqual(method['redundancy_ios_threshold'], .8)
        self.assertEqual(method['overflow_tolerance'], 10)
        self.assertEqual(method['overflow_score_threshold'], .8)
        self.assertEqual(set(method['weight_dict']),
                         {'loss_sgqs_dense', 'loss_sgqs_bbox', 'loss_sgqs_giou'})
        self.assertNotIn('admission_weight', method)
        self.assertNotIn('survival_margin', method)

    def test_original_sgqs_remains_an_explicit_historical_reference(self):
        config = prepare.arm_config('sgqs')
        self.assertEqual(config['criterion'], 'SGQSCriterion')
        original = config['SGQSCriterion']
        self.assertEqual(original['survival_start_epoch'], 28.)
        self.assertEqual(original['survival_threshold'], 'non_owner_budget')
        self.assertTrue(original['protect_encoder_supervision'])
        self.assertNotIn('dense_owner_ramp_epochs', original)

    def test_registry_constructs_the_actual_v2_and_control(self):
        from engine.core import YAMLConfig
        from engine.sgqs_v2 import SGQSV2Criterion
        for arm in ('support', 'v2'):
            cfg = YAMLConfig(str(prepare.ROOT / 'configs/sgqs_v2' / f'{prepare.ARMS[arm]}.yml'))
            self.assertIsInstance(cfg.criterion, SGQSV2Criterion)


class PreparedRunTest(unittest.TestCase):
    def test_checkpoint_requires_real_resume_and_ema_payload(self):
        prepare.validate_checkpoint(checkpoint())
        for missing in ('model', 'ema', 'optimizer', 'scaler'):
            state = checkpoint()
            state.pop(missing)
            with self.assertRaises(ValueError):
                prepare.validate_checkpoint(state)
        for state in ({**checkpoint(), 'last_epoch': 28}, {**checkpoint(), 'ema': {}}):
            with self.assertRaises(ValueError):
                prepare.validate_checkpoint(state)

    def test_prepare_uses_frozen_inputs_and_writes_new_schema_without_overwrite(self):
        config = prepare.arm_config('v2')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen_dir = root / 'configs/e35_r7'
            frozen_dir.mkdir(parents=True)
            data = root / 'data'
            (data / 'annotations').mkdir(parents=True)
            (data / 'images').mkdir()
            paths = {'resume_checkpoint': root / 'checkpoint0027.pth',
                     'train_annotation': data / 'annotations/instance_train.json',
                     'val_annotation': data / 'annotations/instance_val.json'}
            torch.save(checkpoint(), paths['resume_checkpoint'])
            for key in ('train_annotation', 'val_annotation'):
                paths[key].write_text('{"images": []}', encoding='utf-8')
            frozen = {key: {'bytes': path.stat().st_size, 'sha256': digest(path)}
                      for key, path in paths.items()}
            (frozen_dir / 'frozen_inputs.yml').write_text(yaml.safe_dump(frozen), encoding='utf-8')
            run = root / 'v2_e28_r7_seed0'
            argv = ['--arm', 'v2', '--resume', str(paths['resume_checkpoint']),
                    '--data-root', str(data), '--output', str(run)]
            with mock.patch.object(prepare, 'ROOT', root), \
                    mock.patch.object(prepare, 'source_identity', return_value={
                        'commit': 'a' * 40, 'mode': 'git_worktree', 'git_status': ''}), \
                    mock.patch.object(prepare, 'load_runtime_files', return_value=({}, {'path': 'manifest'})), \
                    mock.patch.object(prepare, 'probe_topology', return_value={
                        'world_size': 8, 'batch_per_gpu': 8, 'global_batch': 64}), \
                    mock.patch.object(prepare, 'arm_config', return_value=copy.deepcopy(config)):
                prepare.main(argv)
                with self.assertRaises(FileExistsError):
                    prepare.main(argv)
                paths['val_annotation'].write_text('changed', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'byte count|frozen continuation'):
                    prepare.main(argv[:-1] + [str(root / 'retry')])
            manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
            resolved = yaml.safe_load((run / 'resolved.yml').read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema'], prepare.SCHEMA)
            self.assertEqual(manifest['experiment_id'], 'EXP-TRAIN-008')
            self.assertEqual(manifest['epochs'], list(range(28, 35)))
            self.assertEqual(manifest['evaluation_max_dets'], 640)
            self.assertEqual(manifest['resolved_config_sha256'], digest(run / 'resolved.yml'))
            self.assertEqual(resolved['resume'], str(paths['resume_checkpoint'].resolve()))
            self.assertFalse(resolved['HGNetv2']['pretrained'])
            self.assertEqual(resolved['train_dataloader']['total_batch_size'], 64)

    def test_verify_rejects_old_schema_tampering_and_old_evaluation_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / 'resolved.yml'
            config = prepare.arm_config('v2')
            runtime = ({'one.py': {'sha256': 'test', 'bytes': 1}}, {'path': 'runtime.sha256'})
            manifest = {'schema': prepare.SCHEMA, 'experiment_id': 'EXP-TRAIN-008',
                        'arm': 'v2', 'runtime_files': runtime[0], 'runtime_manifest': runtime[1]}
            def write():
                config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
                manifest['resolved_config_sha256'] = digest(config_path)
                (root / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            write()
            with mock.patch.object(launch, 'load_runtime_files', return_value=runtime):
                launch.verify_run(['-c', str(config_path)])
                config_path.write_text('seed: 3', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'config changed'):
                    launch.verify_run(['-c', str(config_path)])
                manifest['schema'] = 'sgqs_v2_single_factor_v1'
                write()
                with self.assertRaisesRegex(ValueError, 'coverage-allocation'):
                    launch.verify_run(['-c', str(config_path)])
                manifest['schema'] = prepare.SCHEMA
                config['evaluator']['max_dets'] = 300
                write()
                with self.assertRaisesRegex(ValueError, 'budget must both remain 640'):
                    launch.verify_run(['-c', str(config_path)])
                config['evaluator']['max_dets'] = 640
                write()
            with mock.patch.object(launch, 'load_runtime_files', return_value=({}, runtime[1])):
                with self.assertRaisesRegex(ValueError, 'Runtime changed'):
                    launch.verify_run(['-c', str(config_path)])

    def test_entrypoint_supplies_seed_and_amp_without_probe_wrapper(self):
        args = type('Args', (), {'config': Path('run/resolved.yml')})()
        with mock.patch.object(launch, 'verify_run', return_value=args), \
                mock.patch.object(launch, 'require_probe_topology') as topology, \
                mock.patch.object(launch, 'reserve_probe_rank') as reserve, \
                mock.patch.object(launch.runpy, 'run_path') as run, \
                mock.patch.object(sys, 'argv', ['launcher']):
            launch.main(['-c', str(args.config)])
            self.assertEqual(sys.argv[1:], ['-c', str(args.config), '--seed', '0', '--use-amp'])
            topology.assert_called_once_with()
            reserve.assert_called_once_with(['-c', str(args.config)])
            run.assert_called_once_with(str(prepare.ROOT / 'train.py'), run_name='__main__')

    def test_unregistered_cli_override_is_rejected(self):
        with mock.patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            launch.verify_run(['-c', 'resolved.yml', '-u', 'seed=3'])


if __name__ == '__main__':
    unittest.main()
