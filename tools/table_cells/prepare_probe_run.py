"""Materialize and identify the fixed 28+7 probe run before torchrun."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from tools.table_cells.prepare_probe_subset import sha256


DEFAULT_RUNTIME_FILES = Path('configs/probes/runtime_files.sha256')
WORLD_SIZE = 8
BATCH_PER_GPU = 8
GLOBAL_BATCH = WORLD_SIZE * BATCH_PER_GPU


def _inside(root, path):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_runtime_files(root, manifest):
    """Verify an explicit path/size/SHA-256 deployment manifest."""
    root = Path(root).resolve()
    manifest = Path(manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    unresolved_manifest = manifest
    manifest = manifest.resolve()
    if not _inside(root, manifest):
        raise ValueError(f'Runtime manifest is outside project root: {manifest}')
    if unresolved_manifest.is_symlink() or not manifest.is_file():
        raise FileNotFoundError(f'Runtime manifest must be a regular file: {manifest}')
    files = {}
    for line_number, raw in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        entry = raw.strip()
        if not entry or entry.startswith('#'):
            continue
        fields = entry.split('\t')
        if len(fields) != 3:
            raise ValueError(f'Runtime manifest line {line_number} must be sha256<TAB>bytes<TAB>path')
        expected_sha, expected_bytes, name = fields
        expected_sha = expected_sha.lower()
        if not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
            raise ValueError(f'Invalid SHA-256 on runtime manifest line {line_number}')
        try:
            expected_bytes = int(expected_bytes)
        except ValueError as exc:
            raise ValueError(f'Invalid byte count on runtime manifest line {line_number}') from exc
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Runtime path must be project-relative without ..: {name}')
        unresolved = root / relative
        full = unresolved.resolve()
        if not _inside(root, full):
            raise ValueError(f'Runtime path escapes project root: {name}')
        if unresolved.is_symlink() or not full.is_file():
            raise FileNotFoundError(f'Runtime path must be a regular non-symlink file: {name}')
        normalized = full.relative_to(root).as_posix()
        if normalized in files:
            raise ValueError(f'Duplicate runtime path: {normalized}')
        actual_bytes = full.stat().st_size
        actual_sha = sha256(full)
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise ValueError(
                f'Runtime identity mismatch: {normalized}; expected '
                f'{expected_bytes}/{expected_sha}, got {actual_bytes}/{actual_sha}')
        files[normalized] = {'bytes': actual_bytes, 'sha256': actual_sha}
    if not files:
        raise ValueError('Runtime manifest contains no files')
    return dict(sorted(files.items())), {
        'path': manifest.relative_to(root).as_posix(),
        'bytes': manifest.stat().st_size,
        'sha256': sha256(manifest),
    }


def source_identity(root, expected_revision=None):
    """Use Git when present; otherwise require an explicit deployment attestation."""
    expected = expected_revision.lower() if expected_revision else None
    if expected and not re.fullmatch(r'[0-9a-f]{40}', expected):
        raise ValueError('--source-revision must be a full 40-character Git commit')
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip().lower()
        status = subprocess.check_output(
            ['git', 'status', '--short'], cwd=root, text=True,
            stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        if not expected:
            raise RuntimeError(
                'Project has no usable .git metadata; pass the exact deployed '
                '--source-revision and retain the runtime-file hashes')
        return {'commit': expected, 'git_status': None,
                'mode': 'declared_no_git_runtime_hash_verified'}
    if expected and commit != expected:
        raise ValueError(f'Git HEAD {commit} differs from --source-revision {expected}')
    return {'commit': commit, 'git_status': status, 'mode': 'git_worktree'}


def probe_topology():
    """Fail closed unless this pre-registered 8 x 8 comparison is possible."""
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != WORLD_SIZE:
        raise RuntimeError(
            f'EXP-TRAIN-006 requires exactly {WORLD_SIZE} visible GPUs; found {count}. '
            'Do not silently convert this paired experiment to a single-GPU run.')
    return {
        'world_size': WORLD_SIZE,
        'batch_per_gpu': BATCH_PER_GPU,
        'global_batch': GLOBAL_BATCH,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'visible_device_count': count,
        'visible_device_names': [torch.cuda.get_device_name(i) for i in range(count)],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['b0','sgqs'],required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--source-revision')
    p.add_argument('--runtime-files-list',type=Path,default=DEFAULT_RUNTIME_FILES)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output must not already exist: {args.output}')
    identity=source_identity(ROOT,args.source_revision)
    runtime_files,runtime_manifest=load_runtime_files(ROOT,args.runtime_files_list)
    topology=probe_topology()
    frozen=yaml.safe_load((ROOT/'configs/e35_r7/frozen_inputs.yml').read_text())
    paths={'resume_checkpoint':args.resume,'train_annotation':args.data_root/'annotations/instance_train.json',
           'val_annotation':args.data_root/'annotations/instance_val.json'}
    identities={}
    for name,path in paths.items():
        if path.stat().st_size != frozen[name]['bytes']:
            raise ValueError(f'{name} byte count differs from the paired continuation input')
        digest=sha256(path)
        if digest!=frozen[name]['sha256']:raise ValueError(f'{name} differs from the paired continuation input')
        identities[name]={'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
    if checkpoint.get('last_epoch')!=27 or 'model' not in checkpoint or 'ema' not in checkpoint:
        raise ValueError('Require epoch-27 checkpoint with online model and EMA')
    config=YAMLConfig(str(ROOT/f'configs/probes/{args.arm}_e28_r7.yml')).yaml_cfg
    config.pop('__include__',None)
    config.update(output_dir=str(args.output.resolve()),seed=0,use_amp=True,resume=str(args.resume.resolve()),
                  quick_probe=False,max_train_steps=None)
    config['HGNetv2']['pretrained']=False
    for split in ('train','val'):
        loader=config[f'{split}_dataloader'];loader['total_batch_size']=GLOBAL_BATCH;loader['num_workers']=args.workers
        loader['dataset']['img_folder']=str((args.data_root/'images').resolve())
        loader['dataset']['ann_file']=str(paths[f'{split}_annotation'].resolve())
    resolved_text=yaml.safe_dump(config,sort_keys=False)
    resolved_sha=hashlib.sha256(resolved_text.encode('utf-8')).hexdigest()
    metadata={'schema':'sgqs_probe_run_v2','arm':args.arm,'inputs':identities,
              **topology,
              'source_revision':{'value':identity['commit'],'verification':identity['mode']},
              'git_status':identity['git_status'],'runtime_manifest':runtime_manifest,
              'runtime_files':runtime_files,
              'code_sha256':{name:item['sha256'] for name,item in runtime_files.items()},
              'resolved_config_sha256':resolved_sha}
    args.output.mkdir(parents=True,exist_ok=False)
    resolved=args.output/'resolved.yml';resolved.write_text(resolved_text,encoding='utf-8')
    (args.output/'run_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(resolved)


if __name__=='__main__':main()
