import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.table_cells.prepare_probe_run import load_runtime_files, probe_topology, source_identity


class ProbeRunDeploymentTest(unittest.TestCase):
    def test_runtime_manifest_verifies_size_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'code.py').write_text('value = 1\n', encoding='utf-8')
            import hashlib
            digest = hashlib.sha256((root / 'code.py').read_bytes()).hexdigest()
            byte_count = (root / 'code.py').stat().st_size
            (root / 'probe.txt').write_text(
                f'{digest}\t{byte_count}\tcode.py\n', encoding='utf-8')
            files, manifest = load_runtime_files(root, root / 'probe.txt')
        self.assertEqual(files['code.py'], {'bytes': byte_count, 'sha256': digest})
        self.assertEqual(manifest['path'], 'probe.txt')

    def test_runtime_manifest_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'
            root.mkdir()
            (Path(tmp) / 'outside.py').write_text('', encoding='utf-8')
            (root / 'probe.txt').write_text(f'{"a" * 64}\t0\t../outside.py\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'project-relative'):
                load_runtime_files(root, root / 'probe.txt')

    def test_runtime_manifest_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'code.py').write_text('', encoding='utf-8')
            empty = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
            (root / 'probe.txt').write_text(
                f'{empty}\t0\tcode.py\n{empty}\t0\tcode.py\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Duplicate runtime path'):
                load_runtime_files(root, root / 'probe.txt')

    @mock.patch('tools.table_cells.prepare_probe_run.subprocess.check_output')
    def test_no_git_requires_explicit_revision(self, check_output):
        check_output.side_effect = subprocess.CalledProcessError(128, 'git')
        with self.assertRaisesRegex(RuntimeError, 'source-revision'):
            source_identity(Path('.'))

    @mock.patch('tools.table_cells.prepare_probe_run.subprocess.check_output')
    def test_no_git_accepts_full_attested_revision(self, check_output):
        check_output.side_effect = FileNotFoundError()
        revision = 'a' * 40
        identity = source_identity(Path('.'), revision)
        self.assertEqual(identity['commit'], revision)
        self.assertEqual(identity['mode'], 'declared_no_git_runtime_hash_verified')
        self.assertIsNone(identity['git_status'])

    @mock.patch('tools.table_cells.prepare_probe_run.subprocess.check_output')
    def test_git_revision_mismatch_fails(self, check_output):
        check_output.side_effect = ['b' * 40 + '\n', '']
        with self.assertRaisesRegex(ValueError, 'differs'):
            source_identity(Path('.'), 'a' * 40)

    @mock.patch('tools.table_cells.prepare_probe_run.torch.cuda.get_device_name')
    @mock.patch('tools.table_cells.prepare_probe_run.torch.cuda.device_count', return_value=8)
    @mock.patch('tools.table_cells.prepare_probe_run.torch.cuda.is_available', return_value=True)
    def test_exact_eight_gpu_topology_is_recorded(self, _available, _count, get_name):
        get_name.side_effect = lambda index: f'gpu-{index}'
        topology = probe_topology()
        self.assertEqual(topology['world_size'], 8)
        self.assertEqual(topology['global_batch'], 64)
        self.assertEqual(len(topology['visible_device_names']), 8)

    @mock.patch('tools.table_cells.prepare_probe_run.torch.cuda.device_count', return_value=1)
    @mock.patch('tools.table_cells.prepare_probe_run.torch.cuda.is_available', return_value=True)
    def test_single_gpu_is_rejected(self, _available, _count):
        with self.assertRaisesRegex(RuntimeError, 'requires exactly 8 visible GPUs'):
            probe_topology()


if __name__ == '__main__':
    unittest.main()
