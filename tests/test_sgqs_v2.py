import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

from engine.sgqs import SGQSCriterion
from engine.sgqs_v2 import SGQSV2Criterion
from tools.table_cells import supervision_probe  # noqa: F401
from tools.table_cells.prepare_sgqs_v2_run import (
    ROOT, RUNTIME_FILES, arm_config, validate_checkpoint)
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.train_sgqs_v2 import verify_run
from engine.core import YAMLConfig


WEIGHTS = dict(loss_sgqs_dense=1., loss_sgqs_bbox=2.5,
               loss_sgqs_giou=1., loss_sgqs_survival=.25)


def criterion(cls=SGQSV2Criterion, **kwargs):
    return cls(nn.Identity(), WEIGHTS, **kwargs)


def fixture():
    logits = torch.linspace(-1.5, 1.5, 8).reshape(1, 8, 1).requires_grad_()
    boxes = torch.tensor([[.2, .2, .1, .1], [.3, .3, .1, .1],
                          [.4, .4, .1, .1], [.6, .6, .3, .3]])
    payload = dict(dense_logits=logits, valid_mask=torch.ones(1, 8, dtype=torch.bool),
                   ignore_mask=torch.zeros(1, 8, dtype=torch.bool),
                   spatial_shapes=torch.tensor([[1, 4], [1, 2], [1, 2]]),
                   owner_indices=[torch.tensor([0, 1, 2, 6])],
                   target_indices=[torch.arange(4)], owner_levels=[torch.tensor([0, 0, 0, 2])],
                   owner_boxes=[(boxes + .01).requires_grad_()],
                   effective_k=5, selection_pool_size=8,
                   selected_indices=torch.tensor([[3, 4, 5, 6, 7]]),
                   topk_cutoff=torch.tensor([0.]), assignment_misses=torch.tensor([0]))
    return payload, [dict(labels=torch.zeros(4, dtype=torch.long), boxes=boxes)], [torch.tensor([0., .2, .6, .8])]


class V2LossTest(unittest.TestCase):
    def test_unchanged_mode_exact_v1_loss_and_gradient(self):
        payload, targets, quality = fixture()
        a = criterion(SGQSCriterion)._dense_loss(payload, targets, quality)
        b = criterion()._dense_loss(payload, targets, quality)
        self.assertTrue(torch.equal(a, b))
        ga = torch.autograd.grad(a, payload['dense_logits'])[0]
        gb = torch.autograd.grad(b, payload['dense_logits'])[0]
        self.assertTrue(torch.equal(ga, gb))

    def test_r2_changes_only_bce_target(self):
        payload, targets, quality = fixture()
        logits, owners = payload['dense_logits'], payload['owner_indices'][0]
        old = criterion(SGQSCriterion)._dense_loss(payload, targets, quality)
        new = criterion(dense_positive_target='iou')._dense_loss(payload, targets, quality)
        ga = torch.autograd.grad(old, logits)[0]
        gb = torch.autograd.grad(new, logits)[0]
        q = .25 + .75 * quality[0]
        den = torch.tensor([9., 9., 9., 3.])
        expected = q * (logits[0, owners, 0].sigmoid() - quality[0]) / den
        torch.testing.assert_close(gb[0, owners, 0], expected)
        torch.testing.assert_close(gb[0, [3, 4, 5, 7]], ga[0, [3, 4, 5, 7]], rtol=0, atol=0)
        # Quality weighting is not accidentally changed to q_min=0.
        self.assertGreater(gb[0, 0, 0].item(), 0.)

    def test_r3_global_positive_mean_keeps_negative_gradients(self):
        payload, targets, quality = fixture()
        logits, owners = payload['dense_logits'], payload['owner_indices'][0]
        old = criterion(SGQSCriterion)._dense_loss(payload, targets, quality)
        new = criterion(dense_positive_normalization='global')._dense_loss(payload, targets, quality)
        ga = torch.autograd.grad(old, logits)[0]
        gb = torch.autograd.grad(new, logits)[0]
        q = .25 + .75 * quality[0]
        torch.testing.assert_close(gb[0, owners, 0], q * (logits[0, owners, 0].sigmoid() - q) / 4)
        torch.testing.assert_close(gb[0, [3, 4, 5, 7]], ga[0, [3, 4, 5, 7]])
        # P3 has three owners, P4 zero, P5 one: do not divide global mean by 3.
        pos = q * F.binary_cross_entropy_with_logits(logits[0, owners, 0], q, reduction='none')
        old_positive = (pos[:3].mean() + pos[3]) / 3
        torch.testing.assert_close(new, old - old_positive + pos.mean())

    def test_regression_and_survival_unchanged_and_backward_finite(self):
        payload, targets, _ = fixture()
        baseline = criterion(SGQSCriterion).loss_sgqs(payload, targets, epoch=29)
        for options in [dict(dense_positive_target='iou'), dict(dense_positive_normalization='global')]:
            losses = criterion(**options).loss_sgqs(payload, targets, epoch=29)
            for name in ('loss_sgqs_bbox', 'loss_sgqs_giou', 'loss_sgqs_survival'):
                torch.testing.assert_close(losses[name], baseline[name], rtol=0, atol=0)
            gradients = torch.autograd.grad(sum(losses.values()),
                [payload['dense_logits'], payload['owner_boxes'][0]], retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_empty_owners_are_finite_for_both_interventions(self):
        payload, targets, _ = fixture()
        for field in ('owner_indices', 'target_indices', 'owner_levels'):
            payload[field] = [torch.empty(0, dtype=torch.long)]
        for options in [dict(dense_positive_target='iou'), dict(dense_positive_normalization='global')]:
            loss = criterion(**options)._dense_loss(payload, targets, [torch.empty(0)])
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(torch.autograd.grad(loss, payload['dense_logits'])[0]).all())

    def test_distributed_denominator_uses_global_count_over_world_size(self):
        with mock.patch('engine.sgqs.criterion.is_dist_available_and_initialized', return_value=True), \
             mock.patch('engine.sgqs.criterion.get_world_size', return_value=2), \
             mock.patch('torch.distributed.all_reduce', side_effect=lambda value: value.add_(6)):
            self.assertEqual(criterion()._distributed_average_denominator(2, 'cpu').item(), 4.)

    def test_combining_factors_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'single-factor'):
            criterion(dense_positive_target='iou', dense_positive_normalization='global')

    def test_global_normalization_ddp_gradient_matches_pooled_batch(self):
        # Uneven owner counts (3 vs 1) and an empty positive level on both ranks.
        # Simulate the actual DDP count reductions and final gradient averaging.
        parameter = torch.tensor(.3, requires_grad=True)
        full, targets, quality = fixture()
        full['dense_logits'] = parameter.expand(2, 8, 1)
        full['valid_mask'] = full['valid_mask'].expand(2, -1)
        full['ignore_mask'] = full['ignore_mask'].expand(2, -1)
        full['owner_indices'] = [torch.tensor([0, 1, 2]), torch.tensor([6])]
        full['owner_levels'] = [torch.tensor([0, 0, 0]), torch.tensor([2])]
        full['target_indices'] = [torch.tensor([0, 1, 2]), torch.tensor([3])]
        qualities = [quality[0][:3], quality[0][3:]]
        pooled = criterion(dense_positive_normalization='global')._dense_loss(
            full, targets * 2, qualities)
        reference = torch.autograd.grad(pooled, parameter)[0]
        gradients = []
        for rank in range(2):
            local = dict(full)
            for key in ('dense_logits', 'valid_mask', 'ignore_mask',
                        'owner_indices', 'owner_levels', 'target_indices'):
                local[key] = full[key][rank:rank + 1]
            loss_fn = criterion(dense_positive_normalization='global')
            with mock.patch.object(loss_fn, '_distributed_average_denominator',
                                   side_effect=[2., 2.5, 2., 1.5]):
                loss = loss_fn._dense_loss(local, targets, qualities[rank:rank + 1])
            gradients.append(torch.autograd.grad(loss, parameter)[0])
        torch.testing.assert_close(torch.stack(gradients).mean(), reference)


class V2DeploymentTest(unittest.TestCase):
    def test_config_differences_are_single_factor(self):
        original = arm_config('r0')
        for arm, key, value in [('r2', 'dense_positive_target', 'iou'),
                                ('r3', 'dense_positive_normalization', 'global')]:
            actual, expected = arm_config(arm), copy.deepcopy(original)
            for config in (actual, expected):
                config.pop('output_dir', None)
                config.pop('__include__', None)
            expected['SupervisionProbeCriterion']['base_criterion'] = 'SGQSV2Criterion'
            expected['SGQSV2Criterion'] = dict(expected['SGQSCriterion'], **{key: value})
            self.assertEqual(actual, expected)
        baseline = arm_config('r1')
        self.assertIsNone(baseline['DFINETransformer']['sgqs_supervisor'])
        self.assertEqual(baseline['SupervisionProbeCriterion']['base_criterion'], 'DEIMCriterion')
        for arm in ('r0', 'r1', 'r2', 'r3'):
            config = arm_config(arm)
            self.assertEqual(config['epoches'], 35)
            self.assertEqual(config['checkpoint_freq'], 1)
            self.assertEqual(config['DFINETransformer']['num_queries'], 640)
            self.assertFalse(config['efgl']['enabled'])

    def test_registry_constructs_each_actual_criterion(self):
        from tools.table_cells.prepare_sgqs_v2_run import ARMS
        for arm, name in ARMS.items():
            config = YAMLConfig(str(ROOT / 'configs/sgqs_v2' / f'{name}.yml'))
            wrapped = config.criterion.base_criterion
            if arm in ('r2', 'r3'):
                self.assertIsInstance(wrapped, SGQSV2Criterion)
                self.assertEqual(wrapped.survival_start_epoch, 28.)
                self.assertTrue(wrapped.protect_encoder_supervision)
                self.assertEqual(wrapped.survival_threshold, 'non_owner_budget')
            elif arm == 'r0':
                self.assertIs(type(wrapped), SGQSCriterion)

    def test_preflight_requires_full_resume_state(self):
        state = dict(last_epoch=27, model={}, ema={}, optimizer={}, scaler={})
        validate_checkpoint(state)
        for key in ('ema', 'optimizer', 'scaler'):
            invalid = state.copy()
            del invalid[key]
            with self.assertRaises(ValueError):
                validate_checkpoint(invalid)

    def test_runtime_manifest_and_launch_tampering_check(self):
        files, identity = load_runtime_files(ROOT, RUNTIME_FILES)
        self.assertIn('engine/sgqs_v2/criterion.py', files)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'resolved.yml'
            config.write_text('seed: 0\n', encoding='utf-8')
            manifest = dict(schema='sgqs_v2_single_factor_v1', runtime_manifest=identity,
                            runtime_files=files, resolved_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())
            (config.parent / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            verify_run(['-c', str(config)])
            config.write_text('seed: 1\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'config changed'):
                verify_run(['-c', str(config)])


if __name__ == '__main__':
    unittest.main()
