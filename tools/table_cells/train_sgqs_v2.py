"""General single-type training entry; all runtime choices come from the CLI."""
import os
from datetime import timedelta
import sys
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import sgqs_v2  # noqa: F401 - register the optional training method
from engine.core import YAMLConfig
from engine.misc import dist_utils
from engine.sgqs_v2.configuration import parse_args, prepare_run, verify_recovery, runtime_identity
from engine.sgqs_v2.runtime import atomic_json, digest, main_call, latest_checkpoint
from engine.sgqs_v2.trainer import SGQSV2Trainer


def setup_processes(device):
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world < 1 or not 0 <= rank < world or local < 0:
        raise ValueError('Invalid torchrun topology')
    if device == 'cuda':
        if not torch.cuda.is_available() or local >= torch.cuda.device_count():
            raise RuntimeError('Requested local CUDA device is unavailable')
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group(backend='nccl' if device == 'cuda' else 'gloo',
                                timeout=timedelta(hours=2))
    return world, rank, local


def run_training(args):
    world, rank, local = setup_processes(args.device)
    lock, solver, reserved = None, None, False
    run = args.output_dir.resolve()
    try:
        # Keep normal print untouched in a plain Python process (useful for embedding/tests).
        if world > 1:
            dist_utils.setup_print(rank == 0)
        recovering = args.mode == 'recover'
        manifest = main_call(lambda: verify_recovery(args, world) if recovering else prepare_run(args, world))
        identical = torch.tensor(int(runtime_identity() == manifest['runtime_files']), device=args.device)
        if dist.is_initialized():
            dist.all_reduce(identical, op=dist.ReduceOp.MIN)
        if not identical.item():
            raise RuntimeError('Runtime source differs between ranks or changed during launch')
        run_id = digest(run / 'run_manifest.json')
        def reserve():
            nonlocal lock
            lock = (run / '.training.lock').open('a+b')
            if os.name == 'nt':
                import msvcrt
                if not lock.seek(0, os.SEEK_END):
                    lock.write(b'0'); lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (run / 'completed.json').exists():
                raise ValueError('Run is already complete; start a new resume run to extend training')
            directory = run / 'attempts' / f'{args.attempt:04d}'
            directory.mkdir(parents=True, exist_ok=False)
            atomic_json(directory / 'start.json', {'run_id': run_id, 'recover': recovering,
                'attempt': args.attempt, 'world_size': world, 'torch': torch.__version__,
                'python': sys.version, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                'device': torch.cuda.get_device_name(local) if args.device == 'cuda' else 'cpu'})
        main_call(reserve)
        reserved = True
        recovery = main_call(lambda: latest_checkpoint(run, run_id, world)) if recovering else None
        overrides = {'device': f'cuda:{local}' if args.device == 'cuda' else 'cpu'}
        if recovery:
            overrides.update(resume=recovery['path'], tuning=None)
        cfg = YAMLConfig(str(run / 'resolved.yml'), **overrides)
        dist_utils.setup_seed(cfg.seed)
        if rank == 0:
            print(f'Resolved training configuration: {run / "resolved.yml"}', flush=True)
            print(f'mode={args.mode}, world={world}, seed={cfg.seed}, epochs={cfg.epoches}, '
                  f'global_batch={cfg.yaml_cfg["train_dataloader"]["total_batch_size"]}, '
                  f'AMP={cfg.use_amp}, EMA={cfg.use_ema}, policy={cfg.yaml_cfg["training_policy"]}', flush=True)
        solver = SGQSV2Trainer(cfg, run_id, args.attempt, recovering)
        solver.fit()
    except Exception as exc:
        directory = run / 'attempts' / f'{args.attempt:04d}'
        if reserved and directory.exists():
            try:
                atomic_json(directory / f'fatal.rank{rank:02d}.json', {
                    'phase': getattr(solver, 'phase', {'stage': 'initialization'}),
                    'error': f'{type(exc).__name__}: {exc}'})
            except OSError:
                pass
        raise
    finally:
        if lock is not None:
            lock.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    run_training(parse_args(argv))


if __name__ == '__main__':
    main()
