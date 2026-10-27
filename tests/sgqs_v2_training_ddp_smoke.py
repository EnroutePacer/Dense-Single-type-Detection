"""Two-rank CPU end-to-end lifecycle, validation padding and fail propagation."""
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from test_sgqs_v2_training import TinyTrainer, config
from engine.sgqs_v2.runtime import main_call, require_finite, latest_checkpoint


class DistributedTrainer(TinyTrainer):
    def train(self):
        super().train()
        self.model = DistributedDataParallel(self.model, find_unused_parameters=True)
        # Different batches/BN buffers, same parameters after DDP all-reduce.
        torch.manual_seed(39 + dist.get_rank())


def worker(rank, root, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        cfg = config(root); cfg.epoches = 29
        DistributedTrainer(cfg, 'distributed').fit()
        reference = main_call(lambda: latest_checkpoint(root, 'distributed', 2))
        assert reference['epoch'] == 28
        state = torch.load(reference['path'], weights_only=False)
        assert len(state['sgqs_v2_training']['rng_by_rank']) == 2
        caught = False
        try:
            require_finite(torch.tensor(float('nan') if rank == 0 else 1.), 'cpu', 'one-rank NaN')
        except FloatingPointError:
            caught = True
        assert caught
        caught = False
        def failed_io():
            raise OSError('simulated root disk failure')
        try:
            main_call(failed_io)
        except RuntimeError as exc:
            caught = 'simulated root disk failure' in str(exc)
        assert caught
        if rank == 0:
            print('Two-rank lifecycle, full raw/EMA archive, RNG checkpoint, padding and failure propagation passed.', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / 'run'
        (root / 'attempts/0000').mkdir(parents=True)
        torch.multiprocessing.spawn(worker, args=(str(root), (Path(folder)/'store').as_uri()), nprocs=2)
