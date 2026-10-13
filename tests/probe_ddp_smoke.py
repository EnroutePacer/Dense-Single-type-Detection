"""Run with torchrun --nproc_per_node=2; isolated CPU/Gloo parity smoke."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_supervision_probe import fixture
from tools.table_cells.supervision_probe import SupervisionProbeCriterion


def worker(local_rank, output, store=None):
    if store:
        os.environ['RANK']=str(local_rank);os.environ['WORLD_SIZE']='2'
        dist.init_process_group('gloo',init_method=store,rank=local_rank,world_size=2)
    else:dist.init_process_group('gloo')
    rank=dist.get_rank();torch.set_num_threads(1)
    torch.manual_seed(10);a,ca,gt=fixture()
    torch.manual_seed(10);b,cb,_=fixture()
    a=DistributedDataParallel(a);b=DistributedDataParallel(b)
    probe=SupervisionProbeCriterion(cb,output,interval=1)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4)
    features=[torch.randn(2,32,n,n) for n in (4,2,1)]
    for step in range(2):
        kwargs=dict(epoch=28,step=step,epoch_step=2,input_size=(32,32))
        oa.zero_grad();ob.zero_grad()
        torch.manual_seed(22+rank+step);la=ca(a(features,gt),gt,**kwargs)
        sum(la.values()).backward();rng=torch.get_rng_state()
        torch.manual_seed(22+rank+step);lb=probe(b(features,gt),gt,**kwargs)
        sum(lb.values()).backward()
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        for key in la:torch.testing.assert_close(la[key],lb[key],rtol=0,atol=0)
        for x,y in zip(a.parameters(),b.parameters()):
            if x.grad is not None:torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
        oa.step();ob.step()
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
    if rank==0:print('Two-rank two-step loss/gradient/AdamW/RNG parity passed.',flush=True)
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--local-spawn',action='store_true');args=p.parse_args()
    if args.local_spawn:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            torch.multiprocessing.spawn(worker,args=(args.output,(Path(tmp)/'store').as_uri()),nprocs=2)
    else:worker(None,args.output)


if __name__=='__main__':main()
