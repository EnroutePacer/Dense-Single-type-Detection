"""Resolve a DEIM YAML include chain plus CLI updates without starting training."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig, yaml_utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-u', '--update', nargs='*', default=[])
    args = parser.parse_args()

    updates = yaml_utils.parse_cli(args.update)
    config = YAMLConfig(args.config, **updates)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.yaml_cfg, allow_unicode=True, sort_keys=False),
        encoding='utf-8',
        newline='\n',
    )


if __name__ == '__main__':
    main()
