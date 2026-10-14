"""Fail-closed entrypoint for the fixed eight-GPU supervision probe."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPECTED_WORLD_SIZE = 8


def require_probe_topology(environ=None, visible_device_count=None):
    """Reject launches that do not match the registered 8 x 8 probe design."""
    environ = os.environ if environ is None else environ
    values = {}
    for name in ('WORLD_SIZE', 'LOCAL_WORLD_SIZE'):
        raw = environ.get(name)
        try:
            values[name] = int(raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                f'{name} must be set to {EXPECTED_WORLD_SIZE} by torchrun; got {raw!r}') from None
    if visible_device_count is None:
        visible_device_count = torch.cuda.device_count()
    if (values['WORLD_SIZE'], values['LOCAL_WORLD_SIZE'], visible_device_count) != (
            EXPECTED_WORLD_SIZE, EXPECTED_WORLD_SIZE, EXPECTED_WORLD_SIZE):
        raise RuntimeError(
            'EXP-TRAIN-006 requires WORLD_SIZE=8, LOCAL_WORLD_SIZE=8, and exactly '
            f'8 visible CUDA devices; got WORLD_SIZE={values["WORLD_SIZE"]}, '
            f'LOCAL_WORLD_SIZE={values["LOCAL_WORLD_SIZE"]}, '
            f'visible CUDA devices={visible_device_count}')
    return values


def reserve_probe_rank(argv=None, environ=None):
    """Atomically prevent any rank from reusing a prepared experiment run."""
    environ = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-c', '--config', required=True)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    config = Path(args.config).resolve()
    run_dir = config.parent
    manifest = run_dir / 'run_manifest.json'
    if config.name != 'resolved.yml' or config.is_symlink() or not config.is_file():
        raise RuntimeError(f'Probe training requires a regular prepared resolved.yml: {config}')
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f'Probe training requires the prepared run_manifest.json: {manifest}')
    try:
        rank = int(environ['RANK'])
    except (KeyError, ValueError):
        raise RuntimeError(f'RANK must be an integer set by torchrun; got {environ.get("RANK")!r}') from None
    if rank < 0 or rank >= EXPECTED_WORLD_SIZE:
        raise RuntimeError(f'RANK must be in [0, {EXPECTED_WORLD_SIZE - 1}]; got {rank}')
    marker = run_dir / f'.probe_started.rank{rank}'
    payload = json.dumps({'rank': rank, 'world_size': EXPECTED_WORLD_SIZE}, sort_keys=True) + '\n'
    try:
        with marker.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    except FileExistsError:
        raise RuntimeError(
            f'Probe rank {rank} has already been launched for this run: {marker}. '
            'Refusing to overwrite or append to an existing experiment.') from None
    return marker


def main(argv=None):
    require_probe_topology()
    reserve_probe_rank(argv)
    from tools.table_cells import supervision_probe  # noqa: F401,E402
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
