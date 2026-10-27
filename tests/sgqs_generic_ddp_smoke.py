"""Exercise the real generic entry and CPU DDP setup on uneven full validation."""
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from PIL import Image
import torch
import torch.distributed as dist
import yaml
from engine.core import YAMLConfig, register
from engine.core.workspace import GLOBAL_CONFIG
from test_sgqs_v2_training import TinyModel
from tools.table_cells import train_sgqs_v2 as launch


def worker(rank, folder, store):
    torch.set_num_threads(1)
    root = Path(folder)
    register()(TinyModel)
    dist.init_process_group('gloo', init_method=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    argv = ['--config', str(root/'tiny.yml'), '--mode','scratch','--device','cpu',
        '--output-dir',str(root/'run'),'--train-images',str(root/'images'),
        '--val-images',str(root/'images'),'--train-annotations',str(root/'train.json'),
        '--val-annotations',str(root/'val.json'),'--epochs','1','--batch-size','4',
        '--val-batch-size','2','--workers','0','--seed','23','--no-amp','--ema',
        '--num-queries','4','--input-size','32','32','--eval-budgets','2','4',
        '--checkpoint-every','1','--eval-every','1','--no-eval-initial',
        '--sgqs-start-epoch','0','--sgqs-ramp-epochs','1']
    try:
        with mock.patch.object(launch,'setup_processes',return_value=(2,rank,rank)):
            launch.main(argv)
        if rank == 0:
            for kind in ('raw','ema'):
                result=json.loads((root/f'run/validation/epoch0000/{kind}/complete.json').read_text())
                assert result['full']['images']==3
                assert sum(s['images'] for s in result['shards'])==3  # no padding
            state=torch.load(root/'run/checkpoint0000.pth',weights_only=False)
            assert len(state['sgqs_v2_training']['rng_by_rank'])==2
            print('Generic entry: two CPU ranks, epoch0, EMA, native scheduler, 3-image unpadded full validation passed.',flush=True)
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder); (root/'images').mkdir()
        images=[]; annotations=[]
        for i in range(1,5):
            Image.new('RGB',(32,32),(i*40,100,80)).save(root/'images'/f'{i}.png')
            images.append({'id':i,'width':32,'height':32,'file_name':f'{i}.png'})
            annotations.append({'id':i,'image_id':i,'category_id':0,'bbox':[9.6,11.2,6.4,3.2],
                                'area':20.48,'iscrowd':0})
        for name,n in (('train',4),('val',3)):
            (root/f'{name}.json').write_text(json.dumps({'images':images[:n],'annotations':annotations[:n],
                'categories':[{'id':0,'name':'cell'}]}),encoding='utf-8')
        cfg=YAMLConfig(str(Path(__file__).resolve().parents[1]/'configs/sgqs_v2/train.yml')).yaml_cfg
        cfg.pop('__include__',None)
        cfg.update(model='TinyModel',sync_bn=False,optimizer={'type':'SGD','lr':.001,'momentum':.9},
            lrsheduler=None,lr_scheduler={'type':'MultiStepLR','milestones':[1],'gamma':.5})
        cfg.pop('lr_warmup_scheduler',None); cfg['DEIMCriterion']['reg_max']=4
        (root/'tiny.yml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
        torch.multiprocessing.spawn(worker,args=(folder,(root/'store').as_uri()),nprocs=2)
