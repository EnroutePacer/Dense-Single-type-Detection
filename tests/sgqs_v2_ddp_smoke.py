"""Two-process CPU/Gloo check: sparse transfers, overflow, mixed and empty."""
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from engine.sgqs.criterion import SGQSCriterion
from engine.sgqs_v2.allocation import allocation_loss


class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(.7))

    def forward(self):
        return torch.stack([self.value, -self.value]).reshape(1, 2, 1)


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(ScoreModel())
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        for case, expected in (('pair', 2.), ('overflow', 1.), ('mixed', 1.5), ('empty', 0.)):
            with torch.no_grad(): model.module.value.fill_(.7)
            opt.zero_grad()
            donor = torch.tensor([0] if rank == 0 and case in ('pair', 'mixed') else [], dtype=torch.long)
            receiver = torch.ones_like(donor)
            pairs = [(donor, receiver, torch.ones_like(donor, dtype=torch.bool))]
            overflow = [torch.tensor([0] if rank == 1 and case in ('overflow', 'mixed') else [], dtype=torch.long)]
            loss, _ = allocation_loss(model(), pairs, .2,
                                       SGQSCriterion._distributed_average_denominator,
                                       overflow=overflow, overflow_score_threshold=.6)
            loss.backward()
            torch.testing.assert_close(model.module.value.grad, torch.tensor(expected), rtol=0, atol=0)
            opt.step()
            torch.testing.assert_close(model.module.value, torch.tensor(.7 - .1 * expected), rtol=0, atol=1e-7)
        if rank == 0:
            print('Two-rank Gloo: sparse transfer, overflow, mixed, empty gradients/SGD passed.', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if not dist.is_gloo_available():
        raise RuntimeError('Gloo is required for this check')
    with tempfile.TemporaryDirectory() as folder:
        torch.multiprocessing.spawn(worker, args=((Path(folder)/'store').as_uri(),), nprocs=2)
