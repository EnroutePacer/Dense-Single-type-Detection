"""Explicit entrypoint: register observational wrapper, then run unchanged train.py."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.table_cells import supervision_probe  # noqa: F401,E402

if __name__ == '__main__':
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
