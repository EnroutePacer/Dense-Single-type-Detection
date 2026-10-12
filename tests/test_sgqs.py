import unittest
from pathlib import Path

import torch
import torch.nn as nn

import engine  # noqa: F401 - imports registerable modules
from engine.core.yaml_config import YAMLConfig
from engine.core.workspace import create
from engine.deim.dfine_decoder import DFINETransformer
from engine.sgqs import SGQSCriterion, SGQSQuerySupervisor, assign_scale_aware_owners


class _EmptyCriterion(nn.Module):
    def forward(self, outputs, targets, **metadata):
        return {}


class _FixedScoreHead(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer('logits', logits)

    def forward(self, memory):
        return self.logits.expand(memory.shape[0], -1, -1)


class SGQSAssignmentTest(unittest.TestCase):
    def setUp(self):
        self.shapes = [[80, 80], [40, 40], [20, 20]]
        self.strides = [8, 16, 32]
        self.valid = torch.ones(1, 8400, 1, dtype=torch.bool)

    def test_short_side_scale_assignment_and_offsets(self):
        targets = [{
            'boxes': torch.tensor([
                [0.20, 0.20, 0.02, 0.08],
                [0.50, 0.50, 0.08, 0.20],
                [0.80, 0.80, 0.18, 0.40],
            ]),
            'labels': torch.zeros(3, dtype=torch.long),
        }]
        result = assign_scale_aware_owners(
            targets, self.shapes, self.strides, self.valid, [32, 64, 128])

        self.assertEqual(result['target_indices'][0].tolist(), [0, 1, 2])
        self.assertEqual(result['owner_levels'][0].tolist(), [0, 1, 2])
        indices = result['owner_indices'][0].tolist()
        self.assertLess(indices[0], 6400)
        self.assertGreaterEqual(indices[1], 6400)
        self.assertLess(indices[1], 8000)
        self.assertGreaterEqual(indices[2], 8000)

    def test_collisions_are_unique_and_invalid_point_is_skipped(self):
        targets = [{
            'boxes': torch.tensor([
                [0.50, 0.50, 0.02, 0.08],
                [0.50, 0.50, 0.02, 0.08],
            ]),
            'labels': torch.zeros(2, dtype=torch.long),
        }]
        preferred = 39 * 80 + 39
        self.valid[0, preferred] = False
        result = assign_scale_aware_owners(
            targets, self.shapes, self.strides, self.valid, [32, 64, 128])
        indices = result['owner_indices'][0].tolist()
        self.assertEqual(len(indices), 2)
        self.assertEqual(len(set(indices)), 2)
        self.assertNotIn(preferred, indices)
        self.assertFalse(result['ignore_mask'][0, indices].any())

    def test_empty_targets_are_supported(self):
        targets = [{
            'boxes': torch.empty(0, 4),
            'labels': torch.empty(0, dtype=torch.long),
        }]
        result = assign_scale_aware_owners(
            targets, self.shapes, self.strides, self.valid, [32, 64, 128])
        self.assertEqual(result['owner_indices'][0].numel(), 0)
        self.assertEqual(result['assignment_misses'][0].item(), 0)


class SGQSLossTest(unittest.TestCase):
    def _build_payload(self, num_queries=4, two_targets=False):
        torch.manual_seed(7)
        shapes = [[4, 4], [2, 2], [1, 1]]
        memory = torch.randn(1, 21, 8, requires_grad=True)
        logits = torch.randn(1, 21, 1, requires_grad=True)
        anchors = torch.zeros(1, 21, 4)
        valid = torch.ones(1, 21, 1, dtype=torch.bool)
        boxes = [[0.5, 0.5, 0.25, 0.25]]
        if two_targets:
            boxes.append([0.2, 0.2, 0.20, 0.20])
        targets = [{
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.zeros(len(boxes), dtype=torch.long),
        }]
        bbox_head = nn.Linear(8, 4)
        supervisor = SGQSQuerySupervisor(
            scale_priors=[8, 16, 32], initial_radius=0, max_radius=2, ignore_radius=1)
        payload = supervisor(
            output_memory=memory,
            dense_logits=logits,
            selection_logits=logits.masked_fill(~valid, -torch.inf),
            anchors_unact=anchors,
            valid_mask=valid,
            spatial_shapes=shapes,
            targets=targets,
            bbox_head=bbox_head,
            feat_strides=[8, 16, 32],
            num_queries=num_queries,
            query_select_method='default',
        )
        return payload, targets, memory, logits, bbox_head

    @staticmethod
    def _criterion():
        return SGQSCriterion(
            base_criterion=_EmptyCriterion(),
            weight_dict={
                'loss_sgqs_dense': 1.0,
                'loss_sgqs_bbox': 2.5,
                'loss_sgqs_giou': 1.0,
                'loss_sgqs_survival': 0.25,
            },
        )

    def test_losses_are_finite_and_reach_score_box_and_memory(self):
        payload, targets, memory, logits, bbox_head = self._build_payload()
        losses = self._criterion().loss_sgqs(payload, targets)
        self.assertEqual(set(losses), {
            'loss_sgqs_dense', 'loss_sgqs_bbox',
            'loss_sgqs_giou', 'loss_sgqs_survival',
        })
        self.assertTrue(all(torch.isfinite(value).all() for value in losses.values()))
        sum(losses.values()).backward()
        owner_idx = payload['owner_indices'][0]
        self.assertGreater(logits.grad[0, owner_idx].abs().sum().item(), 0.0)
        self.assertGreater(memory.grad.abs().sum().item(), 0.0)
        self.assertGreater(bbox_head.weight.grad.abs().sum().item(), 0.0)

    def test_budget_overflow_disables_only_survival(self):
        payload, targets, _, _, _ = self._build_payload(num_queries=1, two_targets=True)
        losses = self._criterion().loss_sgqs(payload, targets)
        self.assertEqual(losses['loss_sgqs_survival'].item(), 0.0)
        self.assertGreater(losses['loss_sgqs_dense'].item(), 0.0)

    def test_diagnostics_are_logged_but_never_added_to_total_loss(self):
        payload, targets, _, _, _ = self._build_payload(num_queries=1, two_targets=True)
        criterion = self._criterion()
        losses = criterion({'sgqs_aux_outputs': payload}, targets, epoch=0, step=0, epoch_step=10)
        metrics = criterion.get_logging_metrics()

        self.assertEqual(set(losses), {
            'loss_sgqs_dense', 'loss_sgqs_bbox',
            'loss_sgqs_giou', 'loss_sgqs_survival',
        })
        self.assertTrue(set(losses).isdisjoint(metrics))
        self.assertEqual(metrics['sgqs_assignment_miss_rate'].item(), 0.0)
        self.assertEqual(metrics['sgqs_owner_coverage'].item(), 1.0)
        self.assertEqual(metrics['sgqs_owner_count'].item(), 2.0)
        self.assertEqual(metrics['sgqs_budget_overflow_rate'].item(), 1.0)
        self.assertEqual(metrics['sgqs_survival_ramp'].item(), 0.0)

    def test_zero_iou_owner_keeps_a_survival_gradient(self):
        dense_logits = torch.tensor([[[-2.0], [0.0], [1.0]]], requires_grad=True)
        targets = [{
            'boxes': torch.tensor([[0.9, 0.9, 0.1, 0.1]]),
            'labels': torch.zeros(1, dtype=torch.long),
        }]
        payload = {
            'dense_logits': dense_logits,
            'owner_boxes': [torch.tensor([[0.1, 0.1, 0.1, 0.1]])],
            'owner_indices': [torch.tensor([0])],
            'target_indices': [torch.tensor([0])],
            'owner_levels': [torch.tensor([0])],
            'ignore_mask': torch.zeros(1, 3, dtype=torch.bool),
            'valid_mask': torch.ones(1, 3, dtype=torch.bool),
            'assignment_misses': torch.zeros(1, dtype=torch.long),
            'spatial_shapes': torch.tensor([[1, 3]]),
            'topk_cutoff': torch.tensor([0.0]),
            'selected_indices': torch.tensor([[1, 2]]),
            'effective_k': 2,
            'selection_pool_size': 3,
        }

        survival = self._criterion().loss_sgqs(payload, targets)['loss_sgqs_survival']
        survival.backward()

        self.assertGreater(survival.item(), 0.0)
        self.assertLess(dense_logits.grad[0, 0, 0].item(), 0.0)


class SGQSIntegrationTest(unittest.TestCase):
    def _decoder(self, supervisor=None):
        return DFINETransformer(
            num_classes=1,
            hidden_dim=32,
            num_queries=4,
            feat_channels=[32, 32, 32],
            feat_strides=[8, 16, 32],
            num_levels=3,
            num_points=[1, 1, 1],
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            num_denoising=0,
            eval_spatial_size=[32, 32],
            eval_idx=-1,
            aux_loss=True,
            reg_max=4,
            sgqs_supervisor=supervisor,
        )

    def test_eval_outputs_and_state_dict_match_baseline(self):
        torch.manual_seed(11)
        baseline = self._decoder()
        sgqs = self._decoder(SGQSQuerySupervisor(scale_priors=[8, 16, 32]))
        sgqs.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(set(baseline.state_dict()), set(sgqs.state_dict()))
        baseline.eval()
        sgqs.eval()
        features = [
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
            torch.randn(1, 32, 1, 1),
        ]
        with torch.no_grad():
            expected = baseline(features)
            actual = sgqs(features)
        self.assertNotIn('sgqs_aux_outputs', actual)
        torch.testing.assert_close(actual['pred_logits'], expected['pred_logits'], rtol=0, atol=0)
        torch.testing.assert_close(actual['pred_boxes'], expected['pred_boxes'], rtol=0, atol=0)

    def test_training_emits_dense_payload_before_four_query_decoder(self):
        decoder = self._decoder(SGQSQuerySupervisor(scale_priors=[8, 16, 32]))
        decoder.train()
        features = [
            torch.randn(1, 32, 4, 4),
            torch.randn(1, 32, 2, 2),
            torch.randn(1, 32, 1, 1),
        ]
        targets = [{
            'boxes': torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
            'labels': torch.zeros(1, dtype=torch.long),
        }]
        outputs = decoder(features, targets)
        self.assertEqual(outputs['pred_logits'].shape, (1, 4, 1))
        self.assertEqual(outputs['sgqs_aux_outputs']['dense_logits'].shape, (1, 21, 1))
        self.assertEqual(outputs['sgqs_aux_outputs']['effective_k'], 4)

    def test_invalid_anchor_with_high_logit_cannot_enter_global_topk(self):
        decoder = self._decoder()
        decoder.train()
        spatial_shapes = [[80, 80], [40, 40], [20, 20]]
        anchors, valid_mask = decoder._generate_anchors(spatial_shapes)
        candidate_count = anchors.shape[1]
        logits = torch.arange(candidate_count, dtype=torch.float32).view(1, -1, 1)
        logits = logits / candidate_count
        logits = logits.masked_fill(~valid_mask, 1000.0)
        decoder.enc_score_head = _FixedScoreHead(logits)
        memory = torch.zeros(1, candidate_count, decoder.hidden_dim)

        _, _, _, topk_logits, _ = decoder._get_decoder_input(memory, spatial_shapes)

        self.assertFalse(valid_mask.all())
        self.assertTrue(torch.isfinite(topk_logits[0]).all())
        self.assertLess(topk_logits[0].max().item(), 100.0)

    def test_yaml_builds_registered_sgqs_criterion(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / 'configs' / 'sgqs' / 'sgqs_hgnetv2_m_table_cell_fixed640.yml'
        )
        config = YAMLConfig(str(config_path))
        self.assertEqual(config.yaml_cfg['criterion'], 'SGQSCriterion')
        self.assertIsInstance(config.criterion, SGQSCriterion)
        decoder = create('DFINETransformer', config.global_cfg)
        self.assertIsInstance(decoder.sgqs_supervisor, SGQSQuerySupervisor)


if __name__ == '__main__':
    unittest.main()
