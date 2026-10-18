"""Register the opt-in v2 loss, verify the prepared run, then use the v1 trainer."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine import sgqs_v2  # noqa: F401 - explicit registration, no v1 import changes
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.train_with_probes import main as train


def verify_run(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=Path, required=True)
    # Do not permit CLI overrides to invalidate the prepared single-factor config.
    args = parser.parse_args(argv)
    manifest = json.loads((args.config.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'sgqs_v2_single_factor_v1':
        raise ValueError('Require a prepared SGQS v2 run')
    if sha256(args.config) != manifest['resolved_config_sha256']:
        raise ValueError('Prepared config changed')
    files, identity = load_runtime_files(ROOT, manifest['runtime_manifest']['path'])
    if files != manifest['runtime_files'] or identity != manifest['runtime_manifest']:
        raise ValueError('Runtime changed after preparation')
    return args


if __name__ == '__main__':
    args = verify_run()
    # train.py initializes RNG from the CLI before loading YAML and defaults AMP
    # to False; supply both explicitly even though they are in resolved.yml.
    sys.argv = [sys.argv[0], '-c', str(args.config), '--seed', '0', '--use-amp']
    train()
