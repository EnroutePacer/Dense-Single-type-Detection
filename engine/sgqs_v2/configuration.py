"""General run configuration; experiment recipes are not imported here."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import torch
import yaml

from ..core import YAMLConfig, yaml_utils
from .runtime import atomic_json, atomic_write, digest

ROOT = Path(__file__).resolve().parents[2]
RUN_SCHEMA = 'sgqs_training_run_v1'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='SGQS training with explicit runtime choices')
    p.add_argument('-c', '--config', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mode', choices=('scratch', 'resume', 'tuning', 'recover'), required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--weights-state', choices=('raw', 'ema'))
    p.add_argument('--attempt', type=int)
    for name in ('epochs', 'batch-size', 'val-batch-size', 'workers', 'seed',
                 'num-queries', 'checkpoint-every', 'eval-every', 'dense-min-gt'):
        p.add_argument('--' + name, type=int)
    for name in ('amp', 'ema', 'eval-initial'):
        p.add_argument('--' + name, action=argparse.BooleanOptionalAction, default=None)
    for name in ('train-images', 'train-annotations', 'val-images', 'val-annotations'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--input-size', type=int, nargs=2, metavar=('HEIGHT', 'WIDTH'))
    p.add_argument('--eval-budgets', type=int, nargs='+')
    p.add_argument('--sgqs-start-epoch', type=float)
    p.add_argument('--sgqs-ramp-epochs', type=float)
    p.add_argument('-u', '--update', nargs='+', default=[])
    a = p.parse_args(argv)
    fresh = ['config', 'epochs', 'batch_size', 'val_batch_size', 'workers', 'seed',
             'num_queries', 'checkpoint_every', 'eval_every', 'amp', 'ema', 'eval_initial',
             'train_images', 'train_annotations', 'val_images', 'val_annotations',
             'input_size', 'eval_budgets']
    if a.mode == 'recover':
        if a.attempt is None or a.attempt < 1:
            p.error('recover requires a new positive --attempt')
        extra = fresh + ['checkpoint', 'weights_state', 'dense_min_gt', 'sgqs_start_epoch', 'sgqs_ramp_epochs']
        if any(getattr(a, k) is not None for k in extra) or a.update:
            p.error('recover reuses the saved configuration; use resume into a new output directory to change it')
    else:
        missing = [k.replace('_', '-') for k in fresh if getattr(a, k) is None]
        if missing:
            p.error('Explicit arguments required: ' + ', '.join('--' + k for k in missing))
        if a.attempt not in (None, 0):
            p.error('new runs use attempt 0')
        a.attempt = 0
        if (a.mode in ('resume', 'tuning')) != (a.checkpoint is not None):
            p.error('resume/tuning require --checkpoint; scratch does not accept one')
        if (a.mode == 'tuning') != (a.weights_state is not None):
            p.error('tuning requires explicit --weights-state raw|ema; other modes do not accept it')
        positive = ['epochs', 'batch_size', 'val_batch_size', 'num_queries', 'checkpoint_every']
        if any(getattr(a, k) <= 0 for k in positive) or min(*a.input_size, *a.eval_budgets) <= 0:
            p.error('epochs, batches, queries, checkpoint interval, sizes and budgets must be positive')
        if min(a.workers, a.seed, a.eval_every) < 0 or (a.dense_min_gt is not None and a.dense_min_gt < 1):
            p.error('workers/seed/eval-every must be nonnegative; dense-min-gt must be positive')
        if any(v is not None and (not math.isfinite(v) or v < 0)
               for v in (a.sgqs_start_epoch, a.sgqs_ramp_epochs)):
            p.error('SGQS ramp arguments must be finite and nonnegative')
    return a


def runtime_identity():
    paths = list((ROOT / 'engine').rglob('*.py'))
    paths += [ROOT / 'tools/table_cells' / name for name in
              ('train_sgqs_v2.py', 'run_sgqs_v2.sh', 'recover_sgqs_v2.sh', 'local_small_eval.py')]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(paths)}


def resolved_config(args, world_size):
    cfg = YAMLConfig(str(args.config), **yaml_utils.parse_cli(args.update)).yaml_cfg
    cfg.pop('__include__', None)
    for key in ('lr_scheduler', 'lr_warmup_scheduler'):
        if cfg.get(key) is None:
            cfg.pop(key, None)
    if args.batch_size % world_size or args.val_batch_size % world_size:
        raise ValueError('Global train/validation batch sizes must be divisible by the launched world size')
    if args.device == 'cpu' and world_size > 1 and cfg.get('sync_bn', False):
        raise ValueError('CPU DDP does not support SyncBatchNorm; explicitly set -u sync_bn=False')
    if cfg.get('num_classes', 1) != 1:
        raise ValueError('SGQS v2 and its candidate archive currently require single-type detection')
    cfg.update(output_dir=str(args.output_dir.resolve()), device=args.device, seed=args.seed,
        epoches=args.epochs, use_amp=args.amp, use_ema=args.ema, last_epoch=-1,
        checkpoint_freq=args.checkpoint_every, resume=None, tuning=None, tuning_state=args.weights_state,
        quick_probe=False, max_train_steps=None, eval_spatial_size=args.input_size)
    cfg['DFINETransformer']['num_queries'] = args.num_queries
    cfg['PostProcessor']['num_top_queries'] = args.num_queries
    cfg.setdefault('evaluator', {'type': 'CocoEvaluator', 'iou_types': ['bbox']})['max_dets'] = max(10, *args.eval_budgets)
    if cfg['DFINETransformer'].get('learn_query_content', False):
        raise ValueError('Candidate archive requires Encoder-derived query contents')
    if cfg.get('criterion') == 'SGQSV2Criterion':
        if args.sgqs_start_epoch is None or args.sgqs_ramp_epochs is None:
            raise ValueError('SGQS v2 requires explicit --sgqs-start-epoch and --sgqs-ramp-epochs')
        cfg['SGQSV2Criterion'].update(start_epoch=args.sgqs_start_epoch, ramp_epochs=args.sgqs_ramp_epochs)
    elif args.sgqs_start_epoch is not None or args.sgqs_ramp_epochs is not None:
        raise ValueError('SGQS v2 ramp arguments require SGQSV2Criterion')
    inputs = {}
    for split in ('train', 'val'):
        images, annotation = getattr(args, split + '_images').resolve(), getattr(args, split + '_annotations').resolve()
        if not images.is_dir() or not annotation.is_file():
            raise FileNotFoundError(f'Missing {split} image directory or annotation')
        loader = cfg[split + '_dataloader']
        loader.pop('batch_size', None)
        loader.update(total_batch_size=args.batch_size if split == 'train' else args.val_batch_size,
                      num_workers=args.workers)
        loader['dataset'].update(img_folder=str(images), ann_file=str(annotation))
        ops = loader['dataset']['transforms']['ops']
        resize = [op for op in ops if op.get('type') == 'Resize']
        if len(resize) != 1:
            raise ValueError('Explicit input-size requires exactly one final Resize in each data pipeline')
        resize[0]['size'] = list(args.input_size)
        collate = loader.get('collate_fn', {})
        if isinstance(collate, dict) and collate.get('type') == 'BatchImageCollateFunction':
            if args.input_size[0] != args.input_size[1] and (collate.get('scales') or collate.get('base_size_repeat')):
                raise ValueError('Rectangular input requires fixed-size collate; configure multiscale explicitly')
            collate['base_size'] = args.input_size[0]
        inputs[split + '_annotation'] = {'path': str(annotation), 'sha256': digest(annotation)}
    if args.checkpoint is not None:
        path = args.checkpoint.resolve()
        state = torch.load(path, map_location='cpu', weights_only=False)
        if args.mode == 'tuning' and args.weights_state == 'ema' and 'module' not in state.get('ema', {}):
            raise ValueError('Requested EMA initialization is absent from the checkpoint')
        if args.mode == 'resume':
            required = {'model', 'optimizer', 'last_epoch'} | ({'ema'} if args.ema else set()) | ({'scaler'} if args.amp else set())
            if not required <= state.keys():
                raise ValueError(f'Full resume requires {sorted(required)}; use tuning for weight-only initialization')
            if args.ema and 'module' not in state['ema']:
                raise ValueError('Incomplete EMA checkpoint')
            if not -1 <= state['last_epoch'] < args.epochs - 1:
                raise ValueError('--epochs is the absolute training endpoint and must exceed the loaded epoch plus one')
        cfg['resume' if args.mode == 'resume' else 'tuning'] = str(path)
        if 'HGNetv2' in cfg:
            cfg['HGNetv2']['pretrained'] = False
        inputs['initial_checkpoint'] = {'path': str(path), 'sha256': digest(path)}
    cfg['training_policy'] = {'checkpoint_every': args.checkpoint_every, 'eval_every': args.eval_every,
        'eval_initial': args.eval_initial, 'eval_budgets': sorted(set(args.eval_budgets)),
        'dense_min_gt': args.dense_min_gt, 'small_reference_size': list(args.input_size)}
    return cfg, inputs


def prepare_run(args, world_size):
    run = args.output_dir.resolve()
    # Schedulers and shell pipelines commonly create the destination before the
    # Python process starts (for example, ``mkdir -p "$RUN_DIR"`` followed by
    # ``tee "$RUN_DIR/console.log"``). Treat that as a fresh run only when the
    # directory is empty or contains the console stream alone. Any scientific
    # artifact still makes the directory non-fresh, so a previous run can never
    # be overwritten accidentally.
    if run.exists():
        if not run.is_dir():
            raise FileExistsError('New run output path exists and is not a directory')
        unexpected = sorted(p.name for p in run.iterdir() if p.name != 'console.log')
        if unexpected:
            preview = ', '.join(unexpected[:5])
            if len(unexpected) > 5:
                preview += ', ...'
            raise FileExistsError(
                'New runs need a fresh output directory (empty or console.log only); '
                f'found: {preview}. Use --mode recover for interruption recovery')
    cfg, inputs = resolved_config(args, world_size)
    data = yaml.safe_dump(cfg, sort_keys=False).encode('utf-8')
    try:
        source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
        revision = source.stdout.strip() if source.returncode == 0 else None
    except FileNotFoundError:
        revision = None
    manifest = {'schema': RUN_SCHEMA, 'mode': args.mode, 'world_size': world_size,
        'device_type': args.device, 'inputs': inputs, 'runtime_files': runtime_identity(),
        'source_revision': revision,
        'cli': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    run.mkdir(parents=True, exist_ok=True)
    atomic_write(run / 'resolved.yml', lambda f: f.write(data), replace=False)
    manifest['resolved_config_sha256'] = digest(run / 'resolved.yml')
    atomic_json(run / 'run_manifest.json', manifest, replace=False)
    return manifest


def verify_recovery(args, world_size):
    run = args.output_dir.resolve()
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != RUN_SCHEMA:
        raise ValueError('Legacy run: use its original runner or resume into a new generic run')
    if manifest['world_size'] != world_size or manifest['device_type'] != args.device:
        raise ValueError('Exact run recovery requires its original world size and device type; use resume for a new run')
    if digest(run / 'resolved.yml') != manifest['resolved_config_sha256']:
        raise ValueError('Resolved configuration changed')
    if runtime_identity() != manifest['runtime_files']:
        raise ValueError('Runtime code changed; restore it or use resume into a new run')
    for name, item in manifest['inputs'].items():
        if name != 'initial_checkpoint' and digest(item['path']) != item['sha256']:
            raise ValueError(f'Input changed: {name}')
    return manifest
