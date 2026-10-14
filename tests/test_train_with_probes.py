import unittest
from unittest import mock

from tools.table_cells.train_with_probes import main, require_probe_topology


class ProbeEntrypointTopologyTest(unittest.TestCase):
    def test_exact_eight_gpu_launch_is_accepted(self):
        values = require_probe_topology(
            {'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8'}, visible_device_count=8)
        self.assertEqual(values, {'WORLD_SIZE': 8, 'LOCAL_WORLD_SIZE': 8})

    def test_every_topology_mismatch_is_rejected(self):
        cases = [
            ({'LOCAL_WORLD_SIZE': '8'}, 8),
            ({'WORLD_SIZE': '8'}, 8),
            ({'WORLD_SIZE': 'bad', 'LOCAL_WORLD_SIZE': '8'}, 8),
            ({'WORLD_SIZE': '1', 'LOCAL_WORLD_SIZE': '8'}, 8),
            ({'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '1'}, 8),
            ({'WORLD_SIZE': '8', 'LOCAL_WORLD_SIZE': '8'}, 1),
        ]
        for environ, visible_devices in cases:
            with self.subTest(environ=environ, visible_devices=visible_devices):
                with self.assertRaises(RuntimeError):
                    require_probe_topology(environ, visible_devices)

    @mock.patch('tools.table_cells.train_with_probes.runpy.run_path')
    @mock.patch('tools.table_cells.train_with_probes.require_probe_topology')
    def test_guard_precedes_training_construction(self, guard, run_path):
        guard.side_effect = RuntimeError('topology mismatch')
        with self.assertRaisesRegex(RuntimeError, 'topology mismatch'):
            main()
        run_path.assert_not_called()


if __name__ == '__main__':
    unittest.main()
