"""Mathematical regressions and complete CPU training checks for SGQS-E."""

from pathlib import Path
from contextlib import redirect_stdout
import hashlib
import io
import unittest
import warnings

import torch
import torch.nn as nn
import yaml

import engine  # noqa: F401
from engine.core.yaml_config import YAMLConfig
from engine.core.workspace import create
from engine.deim.deim_criterion import DEIMCriterion
from engine.deim.dfine_decoder import DFINETransformer
from engine.deim.matcher import HungarianMatcher
from engine.sgqs import SGQSCriterion, SGQSQuerySupervisor
from engine.sgqs.edge_localization import EFGLConfig, build_edge_context, edge_weights
from engine.deim.deim import DEIM
from engine.solver.det_engine import train_one_epoch


torch.set_num_threads(1)


def targets(empty=False):
    boxes = torch.empty(0, 4) if empty else torch.tensor([
        [.5, .40, .6, .05], [.5, .45, .6, .05]])
    return [{'boxes': boxes, 'labels': torch.zeros(len(boxes), dtype=torch.long)}]


def context(gt=None, epoch=0, **options):
    return build_edge_context(gt or targets(), (100, 200),
                              EFGLConfig(enabled=True, **options), epoch, 0, 10, 120)


def base_criterion(efgl=False, label_loss='vfl'):
    return DEIMCriterion(
        HungarianMatcher(weight_dict={'cost_class': 2., 'cost_bbox': 5., 'cost_giou': 2.}),
        {f'loss_{label_loss}': 1., 'loss_bbox': 5., 'loss_giou': 2., 'loss_fgl': .15, 'loss_ddf': 1.5},
        [label_loss, 'boxes', 'local'], num_classes=1, reg_max=4,
        efgl={'enabled': efgl}, epoches=10)


class _EmptyCriterion(nn.Module):
    def forward(self, *args, **kwargs):
        return {}


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 32, 1)

    def forward(self, images):
        features = self.projection(images)
        return [torch.nn.functional.adaptive_avg_pool2d(features, n) for n in (4, 2, 1)]


class _TraceDecoder(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, p3, p4, p5):
        result = self.decoder([p3, p4, p5])
        return result['pred_logits'], result['pred_boxes']


def sgqs(base=None, **options):
    return SGQSCriterion(
        base if base is not None else _EmptyCriterion(),
        {'loss_sgqs_dense': 1., 'loss_sgqs_bbox': 2.5,
         'loss_sgqs_giou': 1., 'loss_sgqs_survival': .25},
        **options)


class EdgeGeometryTest(unittest.TestCase):
    def test_neighbor_information_focuses_on_distinguishing_edges(self):
        pred = targets()[0]['boxes'][:1].clone()
        pred[:, 1] += .025  # midpoint between two vertically adjacent instances
        pred = pred[None].requires_grad_()
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        iou = torch.tensor([.3], requires_grad=True)
        with_neighbor = edge_weights(context(), pred, indices, iou)
        without_neighbor = edge_weights(context(neighbor_strength=0), pred, indices, iou)
        torch.testing.assert_close(with_neighbor[:, [0, 2]], without_neighbor[:, [0, 2]])
        self.assertTrue((with_neighbor[:, [1, 3]] > without_neighbor[:, [1, 3]]).all())
        self.assertFalse(with_neighbor.requires_grad)
        self.assertTrue((with_neighbor >= iou.detach()[:, None]).all())
        self.assertTrue((with_neighbor <= 1).all())

    def test_correct_boxes_and_shared_boundaries_have_no_extra_weight(self):
        gt = targets()
        indices = [(torch.arange(2), torch.arange(2))]
        iou = torch.tensor([.2, .7])
        actual = edge_weights(context(gt), gt[0]['boxes'][None], indices, iou)
        torch.testing.assert_close(actual, iou[:, None].expand(-1, 4), rtol=0, atol=0)

    def test_duplicate_other_class_empty_and_chunked_neighborhoods(self):
        gt = targets()
        gt[0]['boxes'][1] = gt[0]['boxes'][0]
        self.assertFalse(context(gt)['graphs'][0]['valid'].any())
        gt = targets()
        gt[0]['labels'][1] = 1
        self.assertFalse(context(gt)['graphs'][0]['valid'].any())
        gt = targets(empty=True)
        ctx = context(gt)
        empty_idx = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
        out = edge_weights(ctx, torch.empty(1, 0, 4), empty_idx, torch.empty(0))
        self.assertEqual(out.shape, (0, 4))
        boxes = torch.tensor([[(i % 16 + .5) / 16, (i // 16 + .5) / 20, .04, .03]
                              for i in range(320)])
        ctx = context([{'boxes': boxes, 'labels': torch.zeros(320, dtype=torch.long)}])
        self.assertEqual(ctx['graphs'][0]['neighbor_boxes'].shape, (320, 4, 4))

    def test_annealing_and_branch_switches_recover_original_weights(self):
        pred = targets()[0]['boxes'][None] + .01
        indices = [(torch.arange(2), torch.arange(2))]
        iou = torch.tensor([.2, .4])
        for ctx, is_dn in [(context(epoch=36), False), (context(strength=0), False),
                           (context(normal_branch=False), False), (context(dn_branch=False), True)]:
            actual = edge_weights(ctx, pred, indices, iou, is_dn)
            torch.testing.assert_close(actual, iou[:, None].expand(-1, 4), rtol=0, atol=0)
        self.assertEqual(context(epoch=36)['graphs'], [])

    def test_resume_curriculum_uses_an_explicit_absolute_start(self):
        config = EFGLConfig(
            enabled=True,
            compensation_start_epoch=28.0,
            compensation_duration_epochs=2.0,
        )
        before = build_edge_context(targets(), (100, 200), config, 27, 0, 10, 35)
        start = build_edge_context(targets(), (100, 200), config, 28, 0, 10, 35)
        middle = build_edge_context(targets(), (100, 200), config, 29, 0, 10, 35)
        end = build_edge_context(targets(), (100, 200), config, 30, 0, 10, 35)
        self.assertEqual(before['ramp'], 0.0)
        self.assertEqual(before['graphs'], [])
        self.assertEqual(start['ramp'], 1.0)
        self.assertAlmostEqual(middle['ramp'], 0.5)
        self.assertEqual(end['ramp'], 0.0)
        self.assertEqual(end['graphs'], [])

    def test_pixel_size_comes_from_input_not_original_image(self):
        gt = targets()
        first = context(gt)
        gt[0]['orig_size'] = torch.tensor([9000, 10000])
        second = context(gt)
        torch.testing.assert_close(first['graphs'][0]['smallness'], second['graphs'][0]['smallness'])
        torch.testing.assert_close(second['graphs'][0]['dimensions'][0], torch.tensor([120., 5., 120., 5.]))

    def test_zero_iou_still_gets_direct_fgl_gradient(self):
        criterion = base_criterion(True)
        gt = targets()
        pred = torch.tensor([[[.8, .8, .1, .1]]])
        corners = torch.zeros(1, 1, 20, requires_grad=True)
        outputs = dict(pred_boxes=pred, pred_corners=corners,
                       ref_points=pred.detach(), reg_scale=torch.tensor([4.]), up=torch.tensor([.5]))
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        before = criterion.loss_local(outputs, gt, indices, 1)['loss_fgl']
        self.assertEqual(before.item(), 0.)
        criterion._clear_cache()
        criterion._edge_context = context(gt)
        after = criterion.loss_local(outputs, gt, indices, 1)['loss_fgl']
        after.backward()
        self.assertGreater(after.item(), 0.)
        self.assertGreater(corners.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(corners.grad).all())

    def test_invalid_configuration_fails_instead_of_silent_clamping(self):
        for options in ({'strength': 1.1}, {'pixel_floor': 0}, {'neighbors': -1},
                        {'neighbor_strength': float('nan')}, {'compensation_fraction': 0},
                        {'compensation_start_epoch': -1},
                        {'compensation_duration_epochs': 0}):
            with self.assertRaises(ValueError):
                EFGLConfig(**options)
        with self.assertRaises(ValueError):
            build_edge_context(targets(), None, EFGLConfig(), 0, 0, 1, 120)


class SurvivalAndSupervisionTest(unittest.TestCase):
    def _payload(self, shift=0.):
        logits = torch.tensor([[[shift], [shift], [-1.]]], requires_grad=True)
        return dict(dense_logits=logits, selection_scores=logits.detach().squeeze(-1),
                    topk_cutoff=torch.tensor([shift]), effective_k=2, selection_pool_size=3,
                    owner_indices=[torch.tensor([0, 1])], target_indices=[torch.tensor([0, 1])])

    def test_non_owner_threshold_eliminates_self_chasing_when_owners_equal_k(self):
        gt = targets()
        qualities = [torch.ones(2)]
        legacy, fixed = sgqs(), sgqs(survival_threshold='non_owner_budget')
        old, new = [], []
        for shift in (0., 2.):
            payload = self._payload(shift)
            old.append(legacy._survival_loss(payload, gt, qualities, {}).item())
            loss = fixed._survival_loss(payload, gt, qualities, {})
            new.append(loss.item())
            loss.backward()
            self.assertLess(payload['dense_logits'].grad[0, 0, 0], 0)
            self.assertEqual(payload['dense_logits'].grad[0, 2, 0], 0)
        self.assertAlmostEqual(old[0], old[1], places=6)
        self.assertLess(new[1], new[0] * 1e-6)

    def test_competitor_rank_guarantees_all_active_owners_fit(self):
        for count in (1, 2, 4):
            scores = torch.tensor([-8.] * count + [3., 2., 1., 0., -1., -2.])
            payload = dict(dense_logits=scores[None, :, None], selection_scores=scores[None],
                           owner_indices=[torch.arange(count)], effective_k=4)
            threshold = SGQSCriterion._competition_threshold(payload, 0)
            scores[:count] = threshold + .2
            selected = scores.topk(4).indices
            self.assertTrue(torch.isin(torch.arange(count), selected).all())

    def test_invalid_scores_do_not_set_threshold_and_all_valid_selected_has_no_loss(self):
        payload = self._payload()
        payload['selection_scores'][0, 2] = -torch.inf
        self.assertIsNone(SGQSCriterion._competition_threshold(payload, 0))
        payload['effective_k'] = 3
        with self.assertRaises(ValueError):
            SGQSCriterion._competition_threshold(payload, 0)

    def test_guard_protects_base_positives_and_preserves_owner_box_gradients(self):
        base = _EmptyCriterion()
        base.get_encoder_supervision = lambda: [{
            'classification': [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
            'boxes': [(torch.tensor([0, 1]), torch.tensor([0, 1]))]}]
        criterion = sgqs(base, protect_encoder_supervision=True)
        owner_boxes = torch.randn(3, 4, requires_grad=True)
        payload = dict(dense_logits=torch.zeros(1, 6, 1, requires_grad=True),
                       owner_indices=[torch.tensor([0, 1, 4])],
                       target_indices=[torch.tensor([1, 0, 0])], owner_levels=[torch.tensor([0, 0, 0])],
                       owner_boxes=[owner_boxes], ignore_mask=torch.zeros(1, 6, dtype=torch.bool),
                       selected_indices=torch.tensor([[0, 2, 3]]))
        guarded = criterion._guard_encoder_supervision(payload)
        # Owner 0 disagrees with base GT 0; owner 1/4 remain unselected and trainable.
        self.assertEqual(guarded['owner_indices'][0].tolist(), [1, 4])
        self.assertTrue(guarded['ignore_mask'][0, 2])  # a non-owner base positive
        self.assertTrue(guarded['ignore_mask'][0, 0])
        guarded['owner_boxes'][0].sum().backward()
        self.assertEqual(owner_boxes.grad[0].sum().item(), 0)
        self.assertEqual(owner_boxes.grad[1:].sum().item(), 8)
        self.assertFalse(payload['ignore_mask'].any())  # no mutation of base payload


class SGQSEIntegrationTest(unittest.TestCase):
    @staticmethod
    def decoder(supervisor=True):
        return DFINETransformer(
            num_classes=1, hidden_dim=32, num_queries=4,
            feat_channels=[32, 32, 32], feat_strides=[8, 16, 32], num_levels=3,
            num_points=[1, 1, 1], nhead=4, num_layers=2, dim_feedforward=64,
            dropout=0., num_denoising=8, eval_spatial_size=[32, 32], eval_idx=-1,
            aux_loss=True, reg_max=4,
            sgqs_supervisor=SGQSQuerySupervisor(scale_priors=[8, 16, 32]) if supervisor else None)

    def test_two_step_training_with_dn_empty_images_and_vfl_or_mal(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device, label_loss in ((d, loss) for d in devices for loss in ('vfl', 'mal')):
            torch.manual_seed(21)
            model = self.decoder().to(device).train()
            criterion = sgqs(base_criterion(True, label_loss),
                             survival_threshold='non_owner_budget', protect_encoder_supervision=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            gt = [{key: value.to(device) for key, value in t.items()}
                  for t in targets() + targets(empty=True)]
            features = [torch.randn(2, 32, n, n, device=device) for n in (4, 2, 1)]
            for step in range(2):
                outputs = model(features, gt)
                losses = criterion(outputs, gt, epoch=1, step=step, epoch_step=2, input_size=(32, 32))
                self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
                optimizer.zero_grad()
                sum(losses.values()).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(model.dec_bbox_head[0].layers[-1].weight.grad.abs().sum().item(), 0)
                metrics = criterion.get_logging_metrics()
                self.assertIn('efgl_weight_bonus_mean', metrics)
                self.assertEqual(metrics['efgl_weight_bonus_mean'].device.type, device)
                self.assertTrue(set(metrics).isdisjoint(losses))
                optimizer.step()
            # A fully empty batch must have finite losses and no stale geometry.
            empty = [{key: value.to(device) for key, value in t.items()}
                     for t in targets(empty=True) + targets(empty=True)]
            outputs = model(features, empty)
            losses = criterion(outputs, empty, epoch=1, input_size=(32, 32))
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            self.assertEqual(losses['loss_fgl'].item(), 0.)
            self.assertEqual(criterion.get_logging_metrics()['efgl_weight_bonus_mean'].device.type, device)

    def test_denoising_and_eval_outputs_are_unchanged(self):
        torch.manual_seed(18)
        baseline, model = self.decoder(False), self.decoder(True)
        model.load_state_dict(baseline.state_dict(), strict=True)
        self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
        features = [torch.randn(2, 32, n, n) for n in (4, 2, 1)]
        gt = targets() + targets(empty=True)
        torch.manual_seed(22)
        before = baseline.train()(features, gt)
        torch.manual_seed(22)
        after = model.train()(features, gt)
        for key in ('pred_logits', 'pred_boxes'):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
            for left, right in zip(before['dn_outputs'], after['dn_outputs']):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
        with torch.no_grad():
            before, after = baseline.eval()(features), model.eval()(features)
        self.assertEqual(set(after), {'pred_logits', 'pred_boxes'})
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    def test_native_training_loop_supplies_actual_size_and_progress(self):
        model = DEIM(_TinyBackbone(), nn.Identity(), self.decoder())
        criterion = sgqs(base_criterion(True), survival_threshold='non_owner_budget',
                         protect_encoder_supervision=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loader = [(torch.randn(2, 3, 32, 32), targets() + targets(empty=True)) for _ in range(2)]
        with redirect_stdout(io.StringIO()):
            result = train_one_epoch(False, None, model, criterion, loader, optimizer,
                                     torch.device('cpu'), epoch=1, print_freq=100)
        self.assertTrue(torch.isfinite(torch.tensor(result['loss'])))
        # Final batch is at epoch 1.5; compensation ends at epoch 3.
        self.assertAlmostEqual(criterion.base_criterion._edge_context['ramp'], .5)

    def test_traced_eval_graph_and_outputs_match(self):
        baseline, model = self.decoder(False).eval(), self.decoder(True).eval()
        model.load_state_dict(baseline.state_dict(), strict=True)
        features = tuple(torch.randn(1, 32, n, n) for n in (4, 2, 1))
        with warnings.catch_warnings(), torch.no_grad():
            warnings.simplefilter('ignore', torch.jit.TracerWarning)
            left = torch.jit.trace(_TraceDecoder(baseline), features)
            right = torch.jit.trace(_TraceDecoder(model), features)
        self.assertEqual([node.kind() for node in left.inlined_graph.nodes()],
                         [node.kind() for node in right.inlined_graph.nodes()])
        for expected, actual in zip(left(*features), right(*features)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_end_of_curriculum_restores_all_original_losses(self):
        torch.manual_seed(8)
        model = self.decoder(False).train()
        features = [torch.randn(2, 32, n, n) for n in (4, 2, 1)]
        gt = targets() + targets(empty=True)
        outputs = model(features, gt)
        original = base_criterion()(outputs, gt)
        enhanced = base_criterion(True)(outputs, gt, epoch=3, input_size=(32, 32))
        self.assertEqual(set(original), set(enhanced))
        for key in original:
            torch.testing.assert_close(original[key], enhanced[key], rtol=0, atol=0)

    def test_four_yaml_arms_resolve_the_intended_switches(self):
        root = Path(__file__).resolve().parents[1]
        names = ('b0_baseline', 'b1_sgqs', 'b2_efgl', 'b3_sgqs_e')
        for i, name in enumerate(names):
            config = YAMLConfig(str(root / 'configs' / 'sgqs_e' / f'{name}.yml'))
            criterion = config.criterion
            self.assertEqual(isinstance(criterion, SGQSCriterion), i in (1, 3))
            base = criterion.base_criterion if isinstance(criterion, SGQSCriterion) else criterion
            self.assertEqual(base.efgl.enabled, i in (2, 3))
            decoder = create('DFINETransformer', config.global_cfg)
            self.assertEqual(decoder.num_queries, 640)
            self.assertEqual(decoder.sgqs_supervisor is not None, i in (1, 3))

    def test_e35_r7_matrix_resolves_fixed_budget_and_resume_curricula(self):
        root = Path(__file__).resolve().parents[1]
        matrix = root / 'configs/e35_r7'
        fresh = YAMLConfig(str(matrix / 'deim_core_e35.yml'))
        self.assertEqual(fresh.yaml_cfg['epoches'], 35)
        self.assertEqual(fresh.yaml_cfg['eval_spatial_size'], [640, 640])
        self.assertEqual(fresh.yaml_cfg['DFINETransformer']['num_queries'], 640)
        self.assertEqual(fresh.yaml_cfg['PostProcessor']['num_top_queries'], 640)
        self.assertEqual(fresh.yaml_cfg['DEIMCriterion']['losses'], ['mal', 'boxes', 'local'])
        self.assertEqual(
            fresh.yaml_cfg['train_dataloader']['dataset']['transforms']['policy']['epoch'],
            [1, 17, 31],
        )
        collate = fresh.yaml_cfg['train_dataloader']['collate_fn']
        self.assertIsNone(collate['scales'])
        self.assertEqual(collate['base_size'], 640)
        self.assertEqual(collate['stop_epoch'], 35)

        names = (
            'resume_b0_baseline',
            'resume_b1_sgqs',
            'resume_b2_efgl',
            'resume_b3_sgqs_e',
        )
        for i, name in enumerate(names):
            config = YAMLConfig(str(matrix / f'{name}.yml'))
            self.assertEqual(config.yaml_cfg['epoches'], 35)
            self.assertEqual(config.yaml_cfg['eval_spatial_size'], [640, 640])
            self.assertEqual(config.yaml_cfg['DFINETransformer']['num_queries'], 640)
            self.assertEqual(config.yaml_cfg['PostProcessor']['num_top_queries'], 640)
            self.assertEqual(config.yaml_cfg['DEIMCriterion']['losses'], ['vfl', 'boxes', 'local'])
            criterion = config.criterion
            is_sgqs = isinstance(criterion, SGQSCriterion)
            self.assertEqual(is_sgqs, i in (1, 3))
            base = criterion.base_criterion if is_sgqs else criterion
            self.assertEqual(base.efgl.enabled, i in (2, 3))
            if is_sgqs:
                self.assertEqual(criterion.survival_start_epoch, 28.0)
                self.assertEqual(criterion.survival_ramp_epochs, 1.0)
            if base.efgl.enabled:
                self.assertEqual(base.efgl.compensation_start_epoch, 28.0)
                self.assertEqual(base.efgl.compensation_duration_epochs, 2.1)

        small_reg = YAMLConfig(str(matrix / 'resume_c1_small_reg.yml')).yaml_cfg
        self.assertEqual(small_reg['DEIMCriterion']['losses'], ['vfl', 'boxes', 'local'])
        self.assertEqual(small_reg['DEIMCriterion']['small_object_box_weight'], 1.30)
        self.assertEqual(small_reg['DEIMCriterion']['small_object_local_weight'], 1.15)

        mal_only = YAMLConfig(str(matrix / 'resume_c2_mal_only.yml')).yaml_cfg
        self.assertEqual(mal_only['DEIMCriterion']['losses'], ['mal', 'boxes', 'local'])
        self.assertEqual(
            [op['type'] for op in mal_only['train_dataloader']['dataset']['transforms']['ops']],
            ['RandomPhotometricDistort', 'RandomHorizontalFlip', 'Resize',
             'SanitizeBoundingBoxes', 'ConvertPILImage', 'ConvertBoxes'],
        )

        mosaic_only = YAMLConfig(str(matrix / 'resume_c3_mosaic_only.yml')).yaml_cfg
        mosaic_transforms = mosaic_only['train_dataloader']['dataset']['transforms']
        self.assertEqual(mosaic_transforms['policy']['epoch'], [28, 35, 35])
        self.assertEqual(mosaic_transforms['policy']['ops'], ['Mosaic'])
        self.assertEqual(mosaic_only['train_dataloader']['collate_fn']['mixup_prob'], 0.0)

        deim_retrofit = YAMLConfig(str(matrix / 'resume_c4_deim_core.yml')).yaml_cfg
        self.assertEqual(deim_retrofit['DEIMCriterion']['losses'], ['mal', 'boxes', 'local'])
        self.assertEqual(
            deim_retrofit['train_dataloader']['dataset']['transforms']['policy']['epoch'],
            [28, 35, 35],
        )
        self.assertEqual(deim_retrofit['train_dataloader']['collate_fn']['mixup_epochs'],
                         [28, 35])

    def test_e35_r7_method_provenance_tracks_documented_compatible_adapters(self):
        root = Path(__file__).resolve().parents[1]
        provenance = yaml.safe_load(
            (root / 'configs/e35_r7/method_provenance.yml').read_text(encoding='utf-8'))
        self.assertEqual(
            provenance['reference']['commit'],
            '6fd2c301bd8c09ff3d49fa077c34eaf75ef35b9a',
        )
        delta = provenance['allowed_runtime_delta']
        self.assertEqual(delta['path'], 'engine/sgqs/edge_localization.py')
        for relative, reference_hash in provenance['reference_files'].items():
            data = (root / relative).read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            actual_hash = hashlib.sha256(data).hexdigest()
            expected_hash = (delta['runtime_sha256'] if relative == delta['path'] else
                             provenance.get('compatible_runtime_deltas', {}).get(
                                 relative, {}).get('runtime_sha256', reference_hash))
            self.assertEqual(actual_hash, expected_hash, relative)

    def test_loading_joint_config_does_not_contaminate_a_later_legacy_config(self):
        root = Path(__file__).resolve().parents[1]
        YAMLConfig(str(root / 'configs/sgqs_e/b3_sgqs_e.yml'))
        clean = YAMLConfig(str(root / 'configs/table_cells/dfine_hgnetv2_m_table_cell_fixed640.yml'))
        self.assertNotIn('efgl', clean.yaml_cfg)
        self.assertNotIn('sgqs_supervisor', clean.yaml_cfg['DFINETransformer'])
        self.assertEqual(clean.yaml_cfg['criterion'], 'DEIMCriterion')


if __name__ == '__main__':
    unittest.main()
