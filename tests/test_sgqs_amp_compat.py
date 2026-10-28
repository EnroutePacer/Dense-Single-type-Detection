"""Exercise the pre-2.3 CUDA AMP entry without changing installed PyTorch."""
import unittest
from unittest import mock

import torch

from engine.sgqs_v2.trainer import make_grad_scaler


class ScalerCompatibilityTest(unittest.TestCase):
    def test_legacy_cuda_keeps_enabled_and_checkpoint_parameters(self):
        with mock.patch.object(torch.amp, 'GradScaler', None), \
             mock.patch('torch.cuda.amp.GradScaler') as legacy:
            result = make_grad_scaler('cuda', True, init_scale=1024., growth_interval=16)
        legacy.assert_called_once_with(enabled=True, init_scale=1024., growth_interval=16)
        self.assertIs(result, legacy.return_value)

    def test_legacy_cpu_disabled_works_and_enabled_is_not_silently_ignored(self):
        with mock.patch.object(torch.amp, 'GradScaler', None), \
             mock.patch('torch.cuda.amp.GradScaler') as legacy:
            make_grad_scaler('cpu', False)
            legacy.assert_called_once_with(enabled=False)
            with self.assertRaisesRegex(ValueError, 'CPU'):
                make_grad_scaler('cpu', True)

    def test_modern_interface_keeps_device(self):
        with mock.patch.object(torch.amp, 'GradScaler') as modern:
            result = make_grad_scaler('cpu', False, growth_factor=3.)
        modern.assert_called_once_with('cpu', enabled=False, growth_factor=3.)
        self.assertIs(result, modern.return_value)


if __name__ == '__main__':
    unittest.main()
