import unittest
import json
import tempfile
from pathlib import Path
from unittest import mock

from tools.table_cells.train_with_probes import main, require_probe_topology, reserve_probe_rank


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

    def test_rank_reservation_is_atomic_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / 'resolved.yml').write_text('task: detection\n', encoding='utf-8')
            (run / 'run_manifest.json').write_text('{}\n', encoding='utf-8')
            marker = reserve_probe_rank(
                ['-c', str(run / 'resolved.yml')], {'RANK': '3'})
            self.assertEqual(json.loads(marker.read_text(encoding='utf-8'))['rank'], 3)
            with self.assertRaisesRegex(RuntimeError, 'already been launched'):
                reserve_probe_rank(['-c', str(run / 'resolved.yml')], {'RANK': '3'})

    def test_rank_reservation_requires_prepared_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'resolved.yml'
            config.write_text('task: detection\n', encoding='utf-8')
            with self.assertRaisesRegex(RuntimeError, 'run_manifest'):
                reserve_probe_rank(['-c', str(config)], {'RANK': '0'})

    @mock.patch('tools.table_cells.train_with_probes.runpy.run_path')
    @mock.patch('tools.table_cells.train_with_probes.reserve_probe_rank')
    @mock.patch('tools.table_cells.train_with_probes.require_probe_topology')
    def test_guard_precedes_training_construction(self, guard, reserve, run_path):
        guard.side_effect = RuntimeError('topology mismatch')
        with self.assertRaisesRegex(RuntimeError, 'topology mismatch'):
            main()
        reserve.assert_not_called()
        run_path.assert_not_called()


if __name__ == '__main__':
    unittest.main()
