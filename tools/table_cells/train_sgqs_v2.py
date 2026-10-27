"""Verified v2 lifecycle: durable epochs, full raw/EMA validation and recovery."""
import argparse
import json
import os
from datetime import timedelta
import sys
from pathlib import Path

import yaml
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import sgqs_v2  # noqa: F401 - explicit registration, no v1 import changes
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.prepare_sgqs_v2_run import ARMS, SCHEMA
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.sgqs_v2.runtime import atomic_json, digest, main_call, latest_checkpoint
from engine.sgqs_v2.trainer import SGQSV2Trainer


def verify_run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--attempt', type=int, default=0)
    # Method/data/AMP overrides would invalidate the prepared comparison.
    args = parser.parse_args(argv)
    if args.attempt < 0 or args.recover != (args.attempt > 0):
        raise ValueError('Fresh runs require attempt 0; recovery requires a new positive attempt number')
    if args.config.name != 'resolved.yml' or args.config.is_symlink():
        raise ValueError('Require a regular prepared resolved.yml')
    manifest = json.loads((args.config.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema'] != SCHEMA or manifest.get('experiment_id') != 'EXP-TRAIN-008'
            or manifest.get('arm') not in ARMS):
        raise ValueError('Require a prepared SGQS v2 coverage-allocation run')
    if sha256(args.config) != manifest['resolved_config_sha256']:
        raise ValueError('Prepared config changed')
    files, identity = load_runtime_files(ROOT, manifest['runtime_manifest']['path'])
    if files != manifest['runtime_files'] or identity != manifest['runtime_manifest']:
        raise ValueError('Runtime changed after preparation')
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    expected = {'seed': 0, 'use_amp': True, 'use_ema': True, 'epoches': 35,
                'checkpoint_freq': 1, 'quick_probe': False, 'max_train_steps': None}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Prepared run must retain seed 0, AMP/EMA and all seven continuation epochs')
    if (config['DFINETransformer']['num_queries'] != 640
            or config['PostProcessor']['num_top_queries'] != 640
            or config['evaluator']['max_dets'] != 640):
        raise ValueError('Training queries and evaluation budget must both remain 640')
    if any(config[f'{split}_dataloader']['total_batch_size'] != 64 for split in ('train', 'val')):
        raise ValueError('Prepared run must retain global batch size 64')
    return args


def require_topology():
    values = tuple(int(os.environ.get(k, -1)) for k in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE'))
    if values != (8, 8) or torch.cuda.device_count() != 8:
        raise RuntimeError('EXP-TRAIN-008 requires one node, eight ranks and eight visible CUDA devices')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if rank != local or not 0 <= local < 8:
        raise RuntimeError('Invalid single-node rank topology')
    return local


def run_training(args):
    local_rank = require_topology()
    torch.cuda.set_device(local_rank)
    # No silent single-process fallback. Timeout accommodates rank-0 full AP.
    dist.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    lock = None
    solver = None
    reserved = False
    try:
        dist_utils.setup_print(dist_utils.is_main_process())
        dist_utils.setup_seed(0)
        run = args.config.resolve().parent
        run_id = digest(run / 'run_manifest.json')
        manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
        def reserve():
            nonlocal lock
            import fcntl
            lock = (run / '.training.lock').open('a+')
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (run / 'completed.json').exists():
                raise ValueError('Run is already complete')
            directory = run / 'attempts' / f'{args.attempt:04d}'
            directory.mkdir(parents=True, exist_ok=False)
            for name, item in manifest['inputs'].items():
                # Frozen epoch-27 file need not remain after a durable local boundary exists.
                if args.recover and name == 'resume_checkpoint':
                    continue
                if digest(item['path']) != item['sha256']:
                    raise ValueError(f'Input changed: {name}')
            atomic_json(directory / 'start.json', {'run_id': run_id, 'recover': args.recover,
                'attempt': args.attempt, 'world_size': 8, 'torch': torch.__version__,
                'python': sys.version, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'device': torch.cuda.get_device_name(local_rank)})
        main_call(reserve)
        reserved = True
        recovery = main_call(lambda: latest_checkpoint(run, run_id, 8)) if args.recover else None
        overrides = dict(seed=0, use_amp=True)
        if recovery:
            overrides['resume'] = recovery['path']
        cfg = YAMLConfig(str(args.config), **overrides)
        solver = SGQSV2Trainer(cfg, run_id, args.attempt, args.recover)
        solver.fit()
    except Exception as exc:
        directory = args.config.resolve().parent / 'attempts' / f'{args.attempt:04d}'
        if reserved and directory.exists():
            try:
                atomic_json(directory / f'fatal.rank{local_rank:02d}.json', {
                    'phase': getattr(solver, 'phase', {'stage': 'initialization'}),
                    'error': f'{type(exc).__name__}: {exc}'})
            except OSError:
                pass  # Keep the original error if the output disk itself failed.
        raise
    finally:
        if lock is not None:
            lock.close()
        # Avoid the legacy atexit barrier after an exception.
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    run_training(verify_run(argv))


if __name__ == '__main__':
    main()
