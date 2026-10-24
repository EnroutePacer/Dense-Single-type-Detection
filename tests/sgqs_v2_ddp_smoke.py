"""Two-process CPU/Gloo check: one rank has a pair, the other has none."""
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
        for empty_everywhere in (False, True):
            opt.zero_grad()
            donor = torch.tensor([0] if rank == 0 and not empty_everywhere else [], dtype=torch.long)
            receiver = torch.ones_like(donor)
            pairs = [(donor, receiver, torch.ones_like(donor, dtype=torch.bool))]
            loss, _ = allocation_loss(model(), pairs, .2,
                                       SGQSCriterion._distributed_average_denominator)
            loss.backward()
            expected = 0. if empty_everywhere else 2.
            torch.testing.assert_close(model.module.value.grad, torch.tensor(expected), rtol=0, atol=0)
            opt.step()
        torch.testing.assert_close(model.module.value, torch.tensor(.5), rtol=0, atol=1e-7)
        if rank == 0:
            print('Two-rank Gloo: sparse pair and globally empty pair gradients/SGD passed.', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if not dist.is_gloo_available():
        raise RuntimeError('Gloo is required for this check')
    with tempfile.TemporaryDirectory() as folder:
        torch.multiprocessing.spawn(worker, args=((Path(folder)/'store').as_uri(),), nprocs=2)
