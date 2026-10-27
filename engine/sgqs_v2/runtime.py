"""Durable artifacts and epoch-boundary recovery for the dedicated v2 runner."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from ..misc import dist_utils
from .allocation import STAT_NAMES

FRAMEWORK_SCHEMA = 'sgqs_v2_training_v2'


def checkpoint_name(epoch):
    return 'checkpoint_init.pth' if epoch == -1 else f'checkpoint{epoch:04d}.pth'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_write(path, writer, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value, replace=True):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    atomic_write(path, lambda stream: stream.write(data), replace)


def main_call(function):
    """Propagate rank-0 I/O/evaluator failures before other ranks continue."""
    reply = [None]
    if dist_utils.is_main_process():
        try:
            reply[0] = {'result': function()}
        except Exception as exc:
            reply[0] = {'error': f'{type(exc).__name__}: {exc}'}
    if dist_utils.is_dist_available_and_initialized():
        dist.broadcast_object_list(reply, src=0)
    if 'error' in reply[0]:
        raise RuntimeError(reply[0]['error'])
    return reply[0]['result']


def capture_rng(device=None):
    use_cuda = torch.cuda.is_available() and (device is None or torch.device(device).type == 'cuda')
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if use_cuda else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def checkpoint_payload(solver, run_id, steps_per_epoch):
    local_rng = capture_rng(solver.device)
    rng = [local_rng]
    if dist_utils.is_dist_available_and_initialized():
        rng = [None] * dist_utils.get_world_size()
        dist.all_gather_object(rng, local_rng)
    # Only rank zero constructs the large model/optimizer state dictionary.
    if not dist_utils.is_main_process():
        return None
    state = solver.state_dict()
    state['sgqs_v2_training'] = {
        'schema': FRAMEWORK_SCHEMA, 'phase': 'epoch_boundary', 'run_id': run_id,
        'world_size': len(rng), 'rng_by_rank': rng,
        'steps_per_epoch': steps_per_epoch,
        'next_global_step': (solver.last_epoch + 1) * steps_per_epoch,
        'train_summary': getattr(solver, 'train_summary', None),
        'scheduler': solver.cfg.yaml_cfg.get('lrsheduler'),
        'ema_enabled': solver.ema is not None,
        'initial_epoch': getattr(solver, 'initial_epoch', solver.last_epoch),
        'validation_due': getattr(solver, 'validation_due', True),
        'attempt': solver.attempt,
        'attempt_start_epoch': getattr(solver, 'attempt_start_epoch', solver.last_epoch + 1),
    }
    return state


def save_checkpoint(solver, run_id, steps_per_epoch):
    state = checkpoint_payload(solver, run_id, steps_per_epoch)
    path = solver.output_dir / checkpoint_name(solver.last_epoch)
    def write():
        atomic_write(path, lambda stream: torch.save(state, stream), replace=False)
        reference = {'path': path.name, 'epoch': solver.last_epoch,
                     'sha256': digest(path), 'run_id': run_id, 'schema': FRAMEWORK_SCHEMA}
        atomic_json(path.with_suffix('.json'), reference)
        atomic_json(solver.output_dir / 'latest.json', reference)
        return reference
    return main_call(write)


def validate_recovery(state, run_id, world_size):
    meta = state.get('sgqs_v2_training', {})
    required = {'model', 'optimizer', 'scaler', 'last_epoch'}
    if not required <= state.keys() or meta.get('schema') != FRAMEWORK_SCHEMA:
        raise ValueError('Not a complete v2 framework checkpoint')
    if meta.get('phase') != 'epoch_boundary' or meta.get('run_id') != run_id:
        raise ValueError('Checkpoint phase/run identity mismatch')
    if meta.get('world_size') != world_size or len(meta.get('rng_by_rank', [])) != world_size:
        raise ValueError('Recovery requires the same rank topology')
    if meta.get('next_global_step') != (state['last_epoch'] + 1) * meta.get('steps_per_epoch', -1):
        raise ValueError('Checkpoint epoch/global-step inconsistency')
    if meta.get('ema_enabled') and (not isinstance(state.get('ema'), dict) or 'module' not in state['ema']):
        raise ValueError('Checkpoint EMA is incomplete')
    return meta


def latest_checkpoint(run, run_id, world_size):
    # Also finds a complete, atomically renamed file if power failed before its
    # sidecar/latest pointer was committed. Never fall back silently from corruption.
    import re
    paths = [p for p in Path(run).glob('checkpoint*.pth')
             if p.name == 'checkpoint_init.pth' or re.fullmatch(r'checkpoint\d+\.pth', p.name)]
    paths.sort(key=lambda p: -1 if p.name == 'checkpoint_init.pth' else int(p.stem[10:]))
    if not paths:
        raise FileNotFoundError('No epoch-boundary checkpoint available for recovery')
    path = paths[-1]
    sha = digest(path)
    if path.with_suffix('.json').exists():
        reference = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if reference['sha256'] != sha:
            raise ValueError('Recovery checkpoint checksum mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_recovery(state, run_id, world_size)
    if path.name != checkpoint_name(state['last_epoch']):
        raise ValueError('Checkpoint filename/epoch mismatch')
    return {'path': str(path.resolve()), 'sha256': sha, 'epoch': state['last_epoch']}


COUNT_KEYS = {f'v2_{key}' for key in STAT_NAMES} | {
    'v2_owners', 'v2_owner_conflicts', 'v2_owner_iou_sum', 'v2_rescue_classification_masked',
    'v2_pair_count', 'v2_allocation_count', 'v2_owners_before_guard',
    'v2_small_owners_before_guard', 'v2_small_owners'}


class StepRecorder:
    """One global scalar record per optimizer attempt; no mean-of-step-ratios."""
    def __init__(self, directory, epoch):
        self.directory, self.epoch = Path(directory), epoch
        self.steps, self.sums, self.counts = 0, {}, {}
        self.skipped = 0
        self.allocation_enabled = False
        self.stream = (self.directory / f'epoch{epoch:04d}.steps.jsonl').open('x', encoding='utf-8')

    def record(self, row):
        self.stream.write(json.dumps(row, allow_nan=False) + '\n')
        self.stream.flush()
        self.steps += 1
        if self.steps % 50 == 0:
            os.fsync(self.stream.fileno())
        self.skipped += int(row['amp_skipped'])
        self.allocation_enabled |= bool(row.get('allocation_enabled', True))
        for key, value in {**row['losses'], **row['metrics']}.items():
            destination = self.counts if key in COUNT_KEYS else self.sums
            destination[key] = destination.get(key, 0.) + value

    def summary(self):
        def ratio(numerator, denominator):
            d = self.counts.get(denominator, 0.)
            return self.counts.get(numerator, 0.) / d if d else None
        rates = {
            'selected_gt_coverage': ratio('v2_selected_covered_gt', 'v2_gt'),
            'owner_iou': ratio('v2_owner_iou_sum', 'v2_owners'),
            'rescue_score': ratio('v2_rescue_score_sum', 'v2_rescue_pairs'),
            'prefix_score': ratio('v2_prefix_score_sum', 'v2_prefix_pairs'),
            'keeper_score': ratio('v2_keeper_score_sum', 'v2_reliable_representatives'),
            'overflow_score': ratio('v2_overflow_score_sum', 'v2_overflow_penalized'),
            'donor_score': ratio('v2_donor_score_sum', 'v2_pair_count'),
            'pair_logit_gap': ratio('v2_pair_gap_sum', 'v2_pair_count'),
            'owner_retention': ratio('v2_owners', 'v2_owners_before_guard'),
            'small_owner_retention': ratio('v2_small_owners', 'v2_small_owners_before_guard'),
        }
        warnings = []
        interventions = sum(self.counts.get('v2_' + k, 0.) for k in ('rescue_pairs', 'prefix_pairs', 'overflow_penalized'))
        if self.allocation_enabled and 'v2_gt' in self.counts and not interventions:
            warnings.append('No allocation interventions this epoch; zero loss is not evidence of success.')
        if self.skipped:
            warnings.append(f'{self.skipped} AMP optimizer updates skipped.')
        return {'epoch': self.epoch, 'steps': self.steps, 'amp_skipped': self.skipped,
                'global_counts': self.counts, 'population_rates': rates,
                'step_means': {k: v / max(self.steps, 1) for k, v in self.sums.items()},
                'warnings': warnings}

    def close(self):
        self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close()


def require_finite(value, device, label):
    flag = torch.tensor(int(bool(torch.isfinite(value).all())), device=device)
    if dist_utils.is_dist_available_and_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise FloatingPointError(f'Non-finite {label} on at least one rank')
