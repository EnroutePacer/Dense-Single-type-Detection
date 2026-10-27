"""Containment fragments, source identity, tolerance and receiver-free caps."""
import math
import unittest

import torch

from engine.sgqs_v2.allocation import (
    allocation_loss, build_allocation, center_assignment, geometry_assignment,
)
from tests.test_sgqs_v2 import META, criterion, fixture, match


def crowded(extra=13, score=.95):
    k = extra + 1
    truth = torch.tensor([[.5, .5, .6, .12]])
    boxes = truth.expand(k + 1, -1).clone()
    boxes[1:k, 2:] = .02
    boxes[k] = torch.tensor([.1, .1, .02, .02])
    z = torch.full((1, k + 1, 1), math.log(score / (1 - score)))
    z[0, 0] = 6.; z[0, k] = -3.; z.requires_grad_()
    centers = boxes[None, :, :2].clone()
    centers[0, 1:k, 0] = torch.linspace(.24, .76, extra)
    p = dict(dense_logits=z, dense_boxes=boxes[None], anchor_centers=centers,
             selected_indices=torch.arange(k)[None], valid_mask=torch.ones(1, k + 1, dtype=torch.bool),
             ignore_mask=torch.zeros(1, k + 1, dtype=torch.bool), spatial_shapes=torch.tensor([[1, k + 1]]),
             owner_indices=[torch.tensor([0])], target_indices=[torch.tensor([0])],
             owner_levels=[torch.tensor([0])], owner_boxes=[truth.clone().requires_grad_()])
    decoded = boxes[:k].clone()
    decoded[1:, 2:] = torch.tensor([.02, .08])
    c = torch.full((1, k, 1), -4.); c[0, 0] = 4.; c.requires_grad_()
    outputs = dict(pred_logits=c, pred_boxes=decoded[None].requires_grad_(), sgqs_aux_outputs=p,
                   enc_aux_outputs=[dict(pred_logits=z[:, :k], pred_boxes=boxes[None, :k])])
    record = dict(classification=[match([0], [0])], boxes=[match(list(range(k)), [0] * k)])
    return p, outputs, [dict(boxes=truth, labels=torch.zeros(1, dtype=torch.long))], [match([0], [0])], record


class RedundancyTest(unittest.TestCase):
    def test_fragment_and_containing_box_get_transfer_despite_low_iou(self):
        for size in ((.02, .02), (.25, .25)):
            data = fixture()
            with torch.no_grad():
                data[1]['pred_boxes'][0, 1, 2:] = torch.tensor(size)
                data[1]['enc_aux_outputs'][0]['pred_boxes'][0, 1, 2:] = .02
            pairs, caps, stats = build_allocation(*data)
            self.assertEqual(pairs[0][0].tolist(), [1])
            self.assertEqual(pairs[0][1].tolist(), [4])
            self.assertEqual(len(caps[0]), 0)
            self.assertEqual(stats['donor_ios_only'].item(), 1)

    def test_native_center_outside_is_rejected_even_with_perfect_regressed_boxes(self):
        data = fixture()
        data[0]['anchor_centers'][0, 1] = torch.tensor([.4, .4])
        pairs, caps, stats = build_allocation(*data)
        self.assertEqual(stats['donor_available'].item(), 0)
        self.assertEqual(len(pairs[0][0]) + len(caps[0]), 0)

    def test_containing_multiple_gt_is_not_safe_redundancy(self):
        data = fixture()
        with torch.no_grad(): data[1]['pred_boxes'][0, 1] = torch.tensor([.5, .5, 1., 1.])
        self.assertEqual(build_allocation(*data)[2]['donor_available'].item(), 0)

    def test_regressed_encoder_covering_other_gt_stays_protected(self):
        data = fixture()
        data[1]['enc_aux_outputs'][0]['pred_boxes'][0, 1] = data[2][0]['boxes'][1]
        pairs, _, stats = build_allocation(*data)
        self.assertEqual(len(pairs[0][0]), 0)
        self.assertEqual(stats['donor_cross_gt_overlap'].item(), 1)

    def test_ios_boundary_and_center_ambiguity(self):
        gt = torch.tensor([[.5, .5, .5, .5]])
        boxes = torch.tensor([[.5, .5, .1, .1], [.5, .5, .8, .8], [.1, .1, .1, .1]])
        valid = torch.ones(3, dtype=torch.bool)
        iou = geometry_assignment(boxes, gt, valid)[0]
        ios = geometry_assignment(boxes, gt, valid, .8, metric='ios')[0]
        self.assertEqual(iou.tolist(), [-1, -1, -1])
        self.assertEqual(ios.tolist(), [0, 0, -1])
        owners, ambiguous = center_assignment(boxes[:, :2], gt.repeat(2, 1), valid, 1)
        self.assertEqual(owners.tolist(), [-1, -1, -1])
        self.assertEqual(ambiguous.tolist(), [True, True, False])

    def test_ten_extras_are_tolerated_but_thirteen_allow_three_caps(self):
        for extra, expected in ((10, 0), (11, 1), (13, 3)):
            data = crowded(extra)
            pairs, caps, stats = build_allocation(*data)
            self.assertEqual(len(pairs[0][0]), 0)
            self.assertEqual(len(caps[0]), expected)
            self.assertEqual(stats['overflow_penalized'].item(), expected)
            self.assertNotIn(0, caps[0].tolist())

    def test_overflow_requires_high_confidence_and_stops_at_cap(self):
        for score in (.7, .8):
            data = crowded(score=score)
            pairs, caps, _ = build_allocation(*data)
            self.assertEqual(len(caps[0]), 0)
        z = torch.tensor([0., math.log(.8 / .2), 3.]).reshape(1, 3, 1).requires_grad_()
        empty = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool))]
        loss, count = allocation_loss(z, empty, .2, lambda n, d: max(n, 1), [torch.tensor([0, 2])])
        self.assertEqual(count, 2)
        torch.testing.assert_close(torch.autograd.grad(loss, z)[0].flatten(), torch.tensor([0., 0., .5]))

    def test_overflow_counts_all_extras_but_respects_protected_roles(self):
        data = crowded()
        # Eleven extras have explicit positive duties; only two can be capped.
        data[3][0] = match(list(range(12)), [0] * 12)
        pairs, caps, stats = build_allocation(*data)
        self.assertEqual(stats['overflow_excess'].item(), 3)
        self.assertEqual(caps[0].tolist(), [12, 13])

    def test_highest_confidence_excess_is_chosen_deterministically(self):
        data = crowded()
        with torch.no_grad(): data[0]['dense_logits'][0, 13] = 5.
        self.assertEqual(build_allocation(*data)[1][0].tolist(), [13, 11, 12])

    def test_encoder_score_changes_do_not_rotate_through_reserved_backups(self):
        data = crowded()
        first = build_allocation(*data)[1][0]
        self.assertEqual(first.tolist(), [11, 12, 13])
        with torch.no_grad(): data[0]['dense_logits'][0, first] = math.log(.8 / .2)
        self.assertEqual(len(build_allocation(*data)[1][0]), 0)

    def test_reserve_prefers_better_decoder_geometry_not_encoder_confidence(self):
        data = crowded()
        with torch.no_grad():
            data[1]['pred_boxes'][0, 13] = data[2][0]['boxes'][0]
            data[0]['dense_logits'][0, 13] = 5.
        selected = build_allocation(*data)[1][0].tolist()
        self.assertNotIn(13, selected)
        self.assertEqual(len(selected), 3)

    def test_any_executable_receiver_disables_whole_image_fallback(self):
        data = crowded()
        p, o, t, f, r = data
        outside = len(p['dense_logits'][0]) - 1
        other = torch.tensor([.1, .1, .02, .02])
        t[0]['boxes'] = torch.cat([t[0]['boxes'], other[None]])
        t[0]['labels'] = torch.zeros(2, dtype=torch.long)
        pairs, caps, stats = build_allocation(*data)
        self.assertEqual(pairs[0][1].tolist(), [outside])
        self.assertEqual(len(caps[0]), 0)
        self.assertEqual(stats['overflow_excess'].item(), 0)

    def test_transfer_needs_no_count_or_absolute_confidence_threshold(self):
        data = fixture()
        with torch.no_grad(): data[0]['dense_logits'][0, 1] = .5
        pairs, caps, _ = build_allocation(*data)
        self.assertEqual(pairs[0][0].tolist(), [1])
        self.assertEqual(len(caps[0]), 0)

    def test_criterion_caps_only_encoder_and_support_disables_both_branches(self):
        data = crowded()
        c = criterion(data)
        loss = c(data[1], data[2], **META)['loss_sgqs_allocation']
        gradients = torch.autograd.grad(loss, [data[0]['dense_logits'], data[1]['pred_logits'],
                                             data[1]['pred_boxes']], allow_unused=True)
        self.assertEqual(int((gradients[0] > 0).sum()), 3)
        self.assertEqual(gradients[0][0, 0, 0].item(), 0.)
        self.assertIsNone(gradients[1]); self.assertIsNone(gradients[2])
        self.assertEqual(c.get_logging_metrics()['v2_allocation_count'].item(), 3)
        support = criterion(data, allocation_weight=0.)
        self.assertEqual(support(data[1], data[2], **META)['loss_sgqs_allocation'].item(), 0.)
        self.assertEqual(support.get_logging_metrics()['v2_overflow_penalized'].item(), 0.)

    def test_invalid_overflow_configuration_rejected(self):
        for options in ({'overflow_tolerance': -1}, {'overflow_tolerance': .5},
                        {'overflow_tolerance': True}, {'overflow_score_threshold': 1.},
                        {'overflow_score_threshold': 0.}, {'redundancy_ios_threshold': 1.1}):
            with self.assertRaises(ValueError): criterion(**options)


if __name__ == '__main__':
    unittest.main()
