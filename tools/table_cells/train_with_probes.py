"""Fail-closed entrypoint for the fixed eight-GPU supervision probe."""
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


def main():
    require_probe_topology()
    from tools.table_cells import supervision_probe  # noqa: F401,E402
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
