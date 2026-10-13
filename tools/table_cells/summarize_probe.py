"""Pool rank-local sufficient statistics per sampled step; reject missing/duplicate ranks."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def leaves(value,prefix=''):
    if isinstance(value,dict):
        if set(value)=={'count','sum','sum_sq'}:
            yield prefix,value
        else:
            for key,item in value.items():yield from leaves(item,f'{prefix}/{key}')
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        yield prefix,{'count':1,'sum':value,'sum_sq':value*value}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    expected=json.loads((args.run/'run_manifest.json').read_text())['world_size']
    pool=defaultdict(dict);seen=defaultdict(set)
    for path in sorted((args.run/'probes').glob('train.rank*.jsonl')):
        with path.open(encoding='utf-8') as f:
            for line in f:
                record=json.loads(line);key=(record['epoch'],record['step']);rank=record['rank']
                if rank in seen[key]:raise ValueError(f'Duplicate epoch/step/rank: {key}/{rank}')
                if record['world_size']!=expected:raise ValueError('World size changed')
                seen[key].add(rank)
                for family in ['groups','ddf','losses_weighted_local','dn0_corner_gradient']:
                    for name,stats in leaves(record.get(family,{}),family):
                        dest=pool[key].setdefault(name,{'count':0,'sum':0.,'sum_sq':0.})
                        for field in dest:dest[field]+=stats[field]
    if not seen:raise ValueError('No probe records')
    if any(ranks!=set(range(expected)) for ranks in seen.values()):raise ValueError('Missing rank records; refusing biased averages')
    with args.output.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['epoch','step','metric','count','sum','sum_sq','mean']);writer.writeheader()
        for (epoch,step),metrics in sorted(pool.items()):
            for metric,values in sorted(metrics.items()):
                writer.writerow(dict(epoch=epoch,step=step,metric=metric,**values,
                                     mean=values['sum']/values['count'] if values['count'] else ''))


if __name__=='__main__':main()
