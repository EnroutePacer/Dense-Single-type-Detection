"""Verify a coverage-allocation run, register its training loss, use the base trainer."""
import argparse
import json
import runpy
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import sgqs_v2  # noqa: F401 - explicit registration, no v1 import changes
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.prepare_sgqs_v2_run import ARMS, SCHEMA
from tools.table_cells.train_with_probes import require_probe_topology, reserve_probe_rank


def verify_run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    # Method/data/AMP overrides would invalidate the prepared comparison.
    args = parser.parse_args(argv)
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


def main(argv=None):
    args = verify_run(argv)
    try:
        require_probe_topology()
    except RuntimeError as exc:
        raise RuntimeError(str(exc).replace('EXP-TRAIN-006', 'EXP-TRAIN-008')) from None
    reserve_probe_rank(['-c', str(args.config)])
    # train.py initializes RNG from the CLI before loading YAML and defaults AMP
    # to False; supply both explicitly even though they are in resolved.yml.
    sys.argv = [sys.argv[0], '-c', str(args.config), '--seed', '0', '--use-amp']
    # No SupervisionProbeCriterion or repeated DDF/GO gradient tracing is loaded.
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
