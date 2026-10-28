"""Source coverage uses one native center and its own Encoder geometry.

Synthetic regressions only: the supplied 100-image audit is not reproduced here.
"""
import unittest

import torch

from engine.sgqs_v2.allocation import (
    allocation_loss, build_allocation, geometry_assignment, source_coverage,
)
from tests.test_sgqs_v2 import META, criterion, fixture, match


def source(centers, boxes, gt, selected=None, valid=None, **options):
    centers = torch.as_tensor(centers, dtype=torch.float32).reshape(-1, 2)
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    gt = torch.as_tensor(gt, dtype=torch.float32).reshape(-1, 4)
    valid = (torch.ones(len(boxes), dtype=torch.bool) if valid is None
             else torch.as_tensor(valid, dtype=torch.bool))
    selected = torch.as_tensor(list(range(len(boxes))) if selected is None else selected,
                               dtype=torch.long)
    return source_coverage(centers, boxes, gt, valid, selected, **options)


class SourceCoverageTest(unittest.TestCase):
    def test_perfect_regression_from_foreign_native_center_is_not_coverage(self):
        gt = [[.5, .5, .2, .2]]
        result = source([[.1, .1]], gt, gt)
        self.assertEqual(result['center_gt'].tolist(), [-1])
        self.assertEqual(result['qualified_gt'].tolist(), [-1])
        self.assertEqual(result['selected_covered'].tolist(), [False])
        self.assertEqual(result['selected_raw_center_covered'].tolist(), [False])

    def test_center_and_ios_must_belong_to_the_same_candidate(self):
        gt = [[.5, .5, .2, .2]]
        result = source([[.5, .5], [.1, .1]],
                        [[.1, .1, .1, .1], gt[0]], gt)
        self.assertEqual(result['center_gt'].tolist(), [0, -1])
        self.assertEqual(result['selected_center_covered'].tolist(), [True])
        self.assertEqual(result['qualified_gt'].tolist(), [-1, -1])
        self.assertEqual(result['dense_covered'].tolist(), [False])
        self.assertEqual(result['selected_covered'].tolist(), [False])

    def test_contained_fragment_and_containing_box_need_no_iou_or_area_gate(self):
        gt = torch.tensor([[.5, .5, .4, .4]])
        boxes = torch.tensor([[.5, .5, .04, .04], [.5, .5, .9, .9]])
        valid = torch.ones(2, dtype=torch.bool)
        self.assertEqual(geometry_assignment(boxes, gt, valid)[0].tolist(), [-1, -1])
        result = source(boxes[:, :2], boxes, gt)
        self.assertEqual(result['qualified_gt'].tolist(), [0, 0])
        torch.testing.assert_close(result['ios'], torch.ones(2))
        torch.testing.assert_close(result['area_ratio'], torch.tensor([.01, 5.0625]))
        self.assertEqual(result['selected_covered'].tolist(), [True])

    def test_ios_threshold_is_inclusive_and_configurable(self):
        # Binary-exact coordinates give intersection/area = 16/20 = .8.
        gt = [[.5, .5, .625, .5]]
        boxes = [[.625, .5, .625, .5], [.6251, .5, .625, .5]]
        result = source([[.5, .5], [.5, .5]], boxes, gt)
        self.assertEqual(result['qualified_gt'].tolist(), [0, -1])
        self.assertAlmostEqual(result['ios'][0].item(), .8, places=6)
        self.assertLess(result['ios'][1].item(), .8)
        stricter = source([[.5, .5], [.5, .5]], boxes, gt, ios_threshold=.81)
        self.assertEqual(stricter['qualified_gt'].tolist(), [-1, -1])

    def test_shared_boundary_is_ambiguous_instead_of_two_covered_gt(self):
        gt = [[.25, .5, .5, .5], [.75, .5, .5, .5]]
        result = source([[.5, .5]], [[.5, .5, 1., .5]], gt)
        self.assertEqual(result['ambiguous'].tolist(), [True])
        self.assertEqual(result['center_gt'].tolist(), [-1])
        self.assertEqual(result['selected_raw_center_covered'].tolist(), [True, True])
        self.assertEqual(result['selected_center_covered'].tolist(), [False, False])
        self.assertEqual(result['selected_covered'].tolist(), [False, False])

    def test_overlap_uses_center_assigned_gt_not_highest_ios_gt(self):
        gt = [[.35, .5, .3, .4], [.65, .5, .3, .4]]
        # The Encoder box covers GT 1 fully and GT 0 by 5/6, while its
        # native center belongs only to GT 0. Cross-GT IoS is diagnostic.
        result = source([[.3, .5]], [[.55, .5, .6, .4]], gt)
        self.assertEqual(result['center_gt'].tolist(), [0])
        self.assertEqual(result['qualified_gt'].tolist(), [0])
        self.assertAlmostEqual(result['ios'][0].item(), 5 / 6, places=5)
        self.assertAlmostEqual(result['cross_gt_ios'][0].item(), 1., places=5)
        self.assertEqual(result['selected_covered'].tolist(), [True, False])

    def test_invalid_anchors_cannot_contribute_centers_or_geometry(self):
        gt = [[.3, .3, .2, .2], [.7, .7, .2, .2]]
        result = source([[.3, .3], [.7, .7]], gt, gt, valid=[False, True])
        self.assertEqual(result['center_gt'].tolist(), [-1, 1])
        self.assertEqual(result['qualified_gt'].tolist(), [-1, 1])
        self.assertEqual(result['selected_covered'].tolist(), [False, True])
        self.assertEqual(result['dense_center_covered'].tolist(), [False, True])

    def test_empty_gt_candidates_and_selected_set_have_well_defined_shapes(self):
        empty_gt = source([[.5, .5]], [[.5, .5, .1, .1]], [])
        self.assertEqual(empty_gt['qualified_gt'].tolist(), [-1])
        self.assertEqual(empty_gt['selected_covered'].numel(), 0)
        empty_sources = source([], [], [[.5, .5, .2, .2]])
        self.assertEqual(empty_sources['qualified_gt'].numel(), 0)
        self.assertEqual(empty_sources['dense_covered'].tolist(), [False])
        no_selection = source([[.5, .5]], [[.5, .5, .2, .2]],
                              [[.5, .5, .2, .2]], selected=[])
        self.assertEqual(no_selection['dense_covered'].tolist(), [True])
        self.assertEqual(no_selection['selected_covered'].tolist(), [False])

    def test_chunk_size_changes_no_assignment_and_routing_is_detached(self):
        gt = torch.tensor([[.3, .3, .2, .2], [.7, .7, .2, .2]], requires_grad=True)
        boxes = torch.tensor([[.3, .3, .03, .03], [.7, .7, .8, .8],
                              [.1, .8, .1, .1], [.7, .7, .2, .2]], requires_grad=True)
        centers = torch.tensor([[.3, .3], [.7, .7], [.3, .3], [.7, .7]], requires_grad=True)
        valid = torch.tensor([True, True, True, False])
        selected = torch.tensor([2, 0])
        first = source_coverage(centers, boxes, gt, valid, selected, chunk_size=1)
        second = source_coverage(centers, boxes, gt, valid, selected, chunk_size=20)
        self.assertEqual(set(first), set(second))
        for key, value in first.items():
            torch.testing.assert_close(value, second[key], msg=key)
            self.assertFalse(value.requires_grad, key)


class SourceReceiverTest(unittest.TestCase):
    def test_missing_reasons_partition_absence_ambiguity_and_low_ios(self):
        for kind, expected in (('absent', (1, 0, 0)), ('ambiguous', (0, 2, 0)),
                               ('low_ios', (0, 0, 1))):
            data = fixture()
            p, _, targets, _, _ = data
            if kind == 'low_ios':
                p['anchor_centers'][0, 3] = targets[0]['boxes'][2, :2]
            elif kind == 'ambiguous':
                targets[0]['boxes'] = torch.cat([targets[0]['boxes'],
                                                torch.tensor([[.82, .2, .1, .1]])])
                targets[0]['labels'] = torch.zeros(4, dtype=torch.long)
                p['anchor_centers'][0, 3] = torch.tensor([.81, .2])
            _, _, stats = build_allocation(*data)
            actual = tuple(int(stats[key]) for key in ('selected_no_center_gt',
                'selected_ambiguous_only_gt', 'selected_low_ios_gt'))
            self.assertEqual(actual, expected, kind)
            self.assertEqual(sum(actual), int(stats['missing_gt']), kind)
            self.assertEqual(int(stats['gt']), int(stats['selected_covered_gt']) + sum(actual), kind)

    def test_legacy_iou_coverage_does_not_hide_native_source_starvation(self):
        data = fixture()
        data[0]['anchor_centers'][0, 2] = torch.tensor([.1, .9])
        _, _, stats = build_allocation(*data)
        self.assertEqual(stats['legacy_iou_covered_gt'].item(), 2)
        self.assertEqual(stats['selected_covered_gt'].item(), 1)
        self.assertEqual(stats['iou_only_covered_gt'].item(), 1)
        self.assertEqual(stats['missing_gt'].item(), 2)

    def test_highest_score_qualified_outside_source_wins_per_gt(self):
        data = fixture()
        p, _, targets, _, _ = data
        p['anchor_centers'][0, 5] = targets[0]['boxes'][2, :2]
        p['dense_boxes'][0, 5] = targets[0]['boxes'][2]
        with torch.no_grad():
            p['dense_logits'][0, 5] = -.25
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(tuple(x.tolist() for x in pairs[0]), ([1], [5], [True]))
        self.assertEqual(stats['rescue_available_gt'].item(), 1)

    def test_foreign_native_center_rejects_otherwise_perfect_receiver(self):
        data = fixture()
        data[0]['anchor_centers'][0, 4] = torch.tensor([.1, .9])
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(stats['rescue_available_gt'].item(), 0)
        self.assertEqual(pairs[0][2].tolist(), [False])
        self.assertNotIn(4, pairs[0][1].tolist())

    def test_encoder_geometry_not_decoder_geometry_defines_missing(self):
        data = fixture()
        p, outputs, targets, _, _ = data
        # Query 3's final box is perfect and its source center is inside GT 2,
        # but its selected Encoder box is still background.
        p['anchor_centers'][0, 3] = targets[0]['boxes'][2, :2]
        with torch.no_grad():
            outputs['pred_boxes'][0, 3] = targets[0]['boxes'][2]
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(stats['selected_covered_gt'].item(), 2)
        self.assertEqual(pairs[0][1].tolist(), [4])
        self.assertEqual(pairs[0][2].tolist(), [True])

    def test_selected_native_encoder_box_is_authoritative_over_dense_redecode(self):
        data = fixture()
        p, outputs, targets, _, _ = data
        p['anchor_centers'][0, 3] = targets[0]['boxes'][2, :2]
        # Extra FP32 dense inspection says background; actual selected head is
        # a contained fragment. Its low IoU must not create a false rescue.
        outputs['enc_aux_outputs'][0]['pred_boxes'][0, 3] = targets[0]['boxes'][2]
        outputs['enc_aux_outputs'][0]['pred_boxes'][0, 3, 2:] = .02
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(stats['selected_covered_gt'].item(), 3)
        self.assertEqual(stats['rescue_available_gt'].item(), 0)
        self.assertNotIn(True, pairs[0][2].tolist())

    def test_no_qualified_geometry_does_not_fabricate_a_rescue(self):
        data = fixture()
        data[0]['dense_boxes'][0, 4] = torch.tensor([.1, .9, .05, .05])
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(stats['rescue_available_gt'].item(), 0)
        self.assertNotIn(True, pairs[0][2].tolist())
        self.assertGreaterEqual(stats['no_geometry_gt'].item(), 1)

    def test_low_iou_high_ios_receiver_has_logit_gradient_and_keeps_regression(self):
        data = fixture()
        p, outputs, targets, _, _ = data
        p['dense_boxes'][0, 4, 2:] = .02
        p['dense_boxes'] = p['dense_boxes'].clone().requires_grad_()
        p['anchor_centers'] = p['anchor_centers'].clone().requires_grad_()
        pairs, _, _ = build_allocation(*data)
        self.assertEqual(pairs[0][1].tolist(), [4])
        ranking, _ = allocation_loss(p['dense_logits'], pairs, .2, lambda n, d: max(n, 1))
        gradients = torch.autograd.grad(ranking, [p['dense_logits'], p['dense_boxes'],
            p['anchor_centers'], outputs['pred_boxes'], outputs['pred_logits']], allow_unused=True)
        torch.testing.assert_close(gradients[0].flatten(), torch.tensor([0., 1., 0., 0., -1., 0.]))
        self.assertTrue(all(g is None for g in gradients[1:]))
        losses = criterion(data)(outputs, targets, **META)
        dense_gradient = torch.autograd.grad(losses['loss_sgqs_dense'], p['dense_logits'],
                                             retain_graph=True)[0]
        self.assertEqual(dense_gradient[0, 4, 0].item(), 0.)
        regression = torch.autograd.grad(losses['loss_sgqs_bbox'], p['owner_boxes'][0])[0]
        self.assertGreater(regression[1].abs().sum().item(), 0.)

    def test_only_source_qualified_candidate_is_protected_when_keeper_ios_is_low(self):
        data = fixture()
        p, outputs, _, _, _ = data
        # Final keeper still passes IoU .5, but Encoder IoS is only .75.
        # Query 1 is the sole R-qualified source of GT 0, despite final score<.1.
        outputs['enc_aux_outputs'][0]['pred_boxes'][0, 0, 0] += .025
        pairs, caps, stats = build_allocation(*data)
        self.assertEqual(stats['donor_source_protected'].item(), 1)
        self.assertEqual(stats['donor_available'].item(), 0)
        self.assertEqual(len(pairs[0][0]) + len(caps[0]), 0)

    def test_two_qualified_sources_keep_highest_and_can_transfer_the_other(self):
        data = fixture()
        p, outputs, targets, final, record = data
        outputs['enc_aux_outputs'][0]['pred_boxes'][0, 0, 0] += .025
        p['anchor_centers'][0, 3] = targets[0]['boxes'][0, :2]
        p['dense_boxes'][0, 3] = targets[0]['boxes'][0]
        outputs['enc_aux_outputs'][0]['pred_boxes'][0, 3] = targets[0]['boxes'][0]
        with torch.no_grad():
            outputs['pred_boxes'][0, 3] = targets[0]['boxes'][0]
        final[0] = match([0, 2], [0, 1])
        record['classification'][0] = match([0, 2], [0, 1])
        record['boxes'][0] = match([0, 1, 2, 3], [0, 0, 1, 0])
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(stats['donor_source_protected'].item(), 1)
        self.assertEqual(pairs[0][0].tolist(), [3])
        self.assertEqual(pairs[0][1].tolist(), [4])
        self.assertNotIn(1, pairs[0][0].tolist())


if __name__ == '__main__':
    unittest.main()
