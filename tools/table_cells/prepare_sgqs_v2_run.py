"""Prepare EXP-TRAIN-007 with the same frozen inputs and 8 x 8 topology as v1."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.core import YAMLConfig
from tools.table_cells.prepare_probe_run import (
    GLOBAL_BATCH, load_runtime_files, probe_topology, source_identity)
from tools.table_cells.prepare_probe_subset import sha256

ARMS = {
    'r0': 'r0_original_e28_r7',
    'r1': 'r1_baseline_e28_r7',
    'r2': 'r2_iou_target_e28_r7',
    'r3': 'r3_global_positive_e28_r7',
}
RUNTIME_FILES = Path('configs/sgqs_v2/runtime_files.sha256')


def arm_config(arm):
    return YAMLConfig(str(ROOT / 'configs/sgqs_v2' / f'{ARMS[arm]}.yml')).yaml_cfg


def validate_checkpoint(checkpoint):
    required = {'model', 'ema', 'optimizer', 'scaler'}
    missing = required - checkpoint.keys()
    if checkpoint.get('last_epoch') != 27 or missing:
        raise ValueError(f'Require epoch-27 resume with model/EMA/optimizer/scaler; missing {sorted(missing)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--source-revision')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output must not already exist: {args.output}')
    if args.workers < 0:
        raise ValueError('workers must be non-negative')
    identity = source_identity(ROOT, args.source_revision)
    runtime_files, runtime_manifest = load_runtime_files(ROOT, RUNTIME_FILES)
    topology = probe_topology()
    frozen = yaml.safe_load((ROOT / 'configs/e35_r7/frozen_inputs.yml').read_text(encoding='utf-8'))
    paths = {'resume_checkpoint': args.resume,
             'train_annotation': args.data_root / 'annotations/instance_train.json',
             'val_annotation': args.data_root / 'annotations/instance_val.json'}
    identities = {}
    for name, path in paths.items():
        if path.stat().st_size != frozen[name]['bytes']:
            raise ValueError(f'{name} byte count differs from the frozen continuation input')
        digest = sha256(path)
        if digest != frozen[name]['sha256']:
            raise ValueError(f'{name} differs from the frozen continuation input')
        identities[name] = {'path': str(path.resolve()), 'sha256': digest,
                            'bytes': path.stat().st_size}
    checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
    validate_checkpoint(checkpoint)
    del checkpoint
    config = arm_config(args.arm)
    config.pop('__include__', None)
    config.update(output_dir=str(args.output.resolve()), seed=0, use_amp=True,
                  resume=str(args.resume.resolve()), quick_probe=False, max_train_steps=None)
    config['HGNetv2']['pretrained'] = False
    for split in ('train', 'val'):
        loader = config[f'{split}_dataloader']
        loader.update(total_batch_size=GLOBAL_BATCH, num_workers=args.workers)
        loader['dataset'].update(img_folder=str((args.data_root / 'images').resolve()),
                                 ann_file=str(paths[f'{split}_annotation'].resolve()))
    resolved = yaml.safe_dump(config, sort_keys=False)
    metadata = {
        'schema': 'sgqs_v2_single_factor_v1', 'experiment_id': 'EXP-TRAIN-007',
        'arm': args.arm, 'inputs': identities, **topology,
        'source_revision': {'value': identity['commit'], 'verification': identity['mode']},
        'git_status': identity['git_status'], 'runtime_manifest': runtime_manifest,
        'runtime_files': runtime_files,
        'resolved_config_sha256': hashlib.sha256(resolved.encode('utf-8')).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'resolved.yml').write_text(resolved, encoding='utf-8', newline='\n')
    (args.output / 'run_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(args.output / 'resolved.yml')


if __name__ == '__main__':
    main()
