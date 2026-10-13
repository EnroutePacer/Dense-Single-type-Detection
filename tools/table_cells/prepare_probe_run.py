"""Materialize and identify the fixed 28+7 probe run before torchrun."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from engine.core import YAMLConfig
from tools.table_cells.prepare_probe_subset import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=['b0','sgqs'],required=True)
    p.add_argument('--resume',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    frozen=yaml.safe_load((ROOT/'configs/e35_r7/frozen_inputs.yml').read_text())
    paths={'resume_checkpoint':args.resume,'train_annotation':args.data_root/'annotations/instance_train.json',
           'val_annotation':args.data_root/'annotations/instance_val.json'}
    identities={}
    for name,path in paths.items():
        digest=sha256(path)
        if digest!=frozen[name]['sha256']:raise ValueError(f'{name} differs from the paired continuation input')
        identities[name]={'path':str(path.resolve()),'sha256':digest,'bytes':path.stat().st_size}
    checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
    if checkpoint.get('last_epoch')!=27 or 'model' not in checkpoint or 'ema' not in checkpoint:
        raise ValueError('Require epoch-27 checkpoint with online model and EMA')
    args.output.mkdir(parents=True,exist_ok=False)
    config=YAMLConfig(str(ROOT/f'configs/probes/{args.arm}_e28_r7.yml')).yaml_cfg
    config.pop('__include__',None)
    config.update(output_dir=str(args.output.resolve()),seed=0,use_amp=True,resume=str(args.resume.resolve()),
                  quick_probe=False,max_train_steps=None)
    config['HGNetv2']['pretrained']=False
    for split in ('train','val'):
        loader=config[f'{split}_dataloader'];loader['total_batch_size']=64;loader['num_workers']=args.workers
        loader['dataset']['img_folder']=str((args.data_root/'images').resolve())
        loader['dataset']['ann_file']=str(paths[f'{split}_annotation'].resolve())
    resolved=args.output/'resolved.yml';resolved.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    files=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    # Include new probe source even before a local commit; logs disclose dirty trees.
    files=sorted(set(files)|{p.relative_to(ROOT).as_posix() for p in (ROOT/'tools/table_cells').glob('*probe*.py')})
    code={name:sha256(ROOT/name) for name in files if name.endswith(('.py','.yml','.yaml'))
          and name.startswith(('engine/','tools/table_cells/','configs/'))}
    metadata={'schema':'sgqs_probe_run_v1','arm':args.arm,'inputs':identities,'world_size':8,'batch_per_gpu':8,
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
              'code_sha256':code,'resolved_config_sha256':sha256(resolved)}
    (args.output/'run_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(resolved)


if __name__=='__main__':main()
