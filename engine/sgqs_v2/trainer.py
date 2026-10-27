"""Dedicated v2 lifecycle, shared by all registered comparison arms."""
import json
import math
import time
import os
from pathlib import Path

import torch

from ..solver._solver import BaseSolver
from ..misc import dist_utils
from ..optim.lr_scheduler import FlatCosineLRScheduler
from .runtime import (StepRecorder, atomic_json, digest, main_call, save_checkpoint,
                      validate_recovery, restore_rng, require_finite)
from .evaluation import validate


class SGQSV2Trainer(BaseSolver):
    def __init__(self, cfg, run_id, attempt=0, recovery=False):
        super().__init__(cfg)
        self.run_id, self.attempt, self.recovery = run_id, attempt, recovery
        self.recovery_meta = None

    def _setup(self):
        cfg = self.cfg
        self.device = torch.device(cfg.device)
        self.model = cfg.model
        if cfg.tuning:
            self.load_tuning_state(cfg.tuning)
        self.model = self.model.to(self.device)
        if dist_utils.is_dist_available_and_initialized():
            if cfg.sync_bn and self.device.type == 'cuda':
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            ids = [int(os.environ.get('LOCAL_RANK', 0))] if self.device.type == 'cuda' else None
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=ids, find_unused_parameters=cfg.find_unused_parameters)
        self.criterion = self.to(cfg.criterion, self.device)
        self.postprocessor = self.to(cfg.postprocessor, self.device)
        self.ema = self.to(cfg.ema, self.device)
        options = {k: v for k, v in cfg.yaml_cfg.get('scaler', {}).items()
                   if k not in ('type', 'enabled', 'device')}
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=cfg.use_amp, **options)
        self.last_epoch = cfg.last_epoch
        self.output_dir = Path(cfg.output_dir)
        self.writer = None  # Structured per-step artifacts are the durable interface.

    def load_tuning_state(self, path):
        state = torch.load(path, map_location='cpu', weights_only=False)
        weights = (state['ema']['module'] if self.cfg.yaml_cfg['tuning_state'] == 'ema'
                   else state.get('model', state))
        matched, info = self._matched_state(self.model.state_dict(), weights)
        if not matched:
            raise ValueError('No compatible parameters in the requested initialization state')
        self.model.load_state_dict(matched, strict=False)
        print(f'Weight initialization: {info}', flush=True)

    def train(self):
        super().train()
        # Evaluation forwards use the unwrapped model, so unequal rank lengths
        # are safe. Avoid padded duplicates and score-tie ordering differences.
        loader = self.val_dataloader
        if dist_utils.is_dist_available_and_initialized():
            subset = torch.utils.data.Subset(loader.dataset,
                range(dist_utils.get_rank(), len(loader.dataset), dist_utils.get_world_size()))
            self.val_dataloader = torch.utils.data.DataLoader(subset,
                batch_size=loader.batch_size, shuffle=False, drop_last=False,
                num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                pin_memory=loader.pin_memory, worker_init_fn=loader.worker_init_fn)

    def state_dict(self):
        state = {'last_epoch': self.last_epoch}
        for name in ('model', 'criterion', 'optimizer', 'ema', 'scaler',
                     'lr_scheduler', 'lr_warmup_scheduler'):
            value = getattr(self, name, None)
            if value is not None and hasattr(value, 'state_dict'):
                state[name] = dist_utils.de_parallel(value).state_dict()
        if self.ema is not None:
            state['ema']['before_start'] = self.ema.before_start
        return state

    def load_resume_state(self, path):
        # Only locally prepared, hash-verified checkpoints are accepted by entry.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if self.recovery:
            self.recovery_meta = validate_recovery(state, self.run_id, dist_utils.get_world_size())
        self.load_state_dict(state)
        if self.ema is not None:
            self.ema.before_start = state.get('ema', {}).get('before_start', 0)

    def fit(self):
        self.phase = {'stage': 'setup'}
        self.train()
        cfg = self.cfg
        steps = len(self.train_dataloader)
        if not steps:
            raise ValueError('Training loader is empty; check dataset and batch size')
        self.policy = cfg.yaml_cfg['training_policy']
        self.self_lr_scheduler = cfg.lrsheduler is not None
        if self.self_lr_scheduler:
            if cfg.lrsheduler != 'flatcosine':
                raise ValueError('Unknown custom scheduler; use flatcosine or a configured native scheduler')
            self.lr_scheduler = FlatCosineLRScheduler(self.optimizer, cfg.lr_gamma, steps,
                cfg.epoches, cfg.warmup_iter, cfg.flat_epoch, cfg.no_aug_epoch)
        self.attempt_dir = self.output_dir / 'attempts' / f'{self.attempt:04d}'
        self.attempt_start_epoch = self.last_epoch + 1
        if self.recovery_meta:
            if steps != self.recovery_meta['steps_per_epoch']:
                raise ValueError('Training-loader length changed on recovery')
            self.train_summary = self.recovery_meta['train_summary']
            self.initial_epoch = self.recovery_meta['initial_epoch']
            self.validation_due = self.recovery_meta['validation_due']
            restore_rng(self.recovery_meta['rng_by_rank'][dist_utils.get_rank()])
            checkpoint = main_call(lambda: {'path': Path(cfg.resume).name,
                'sha256': digest(cfg.resume), 'epoch': self.last_epoch})
        else:
            if not -1 <= self.last_epoch < cfg.epoches - 1:
                raise ValueError('Total epochs must extend beyond the loaded checkpoint')
            self.initial_epoch = self.last_epoch
            self.validation_due = self.policy['eval_initial']
            self.train_summary = None
            checkpoint = save_checkpoint(self, self.run_id, steps)
        if self.validation_due:
            self.validate_boundary(checkpoint)
        for epoch in range(self.last_epoch + 1, cfg.epoches):
            self.train_dataloader.set_epoch(epoch)
            if hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)
            self.train_summary = self.train_epoch(epoch)
            if not self.self_lr_scheduler and self.lr_scheduler is not None:
                warmup = getattr(self, 'lr_warmup_scheduler', None)
                if warmup is None or warmup.finished():
                    self.lr_scheduler.step()
            self.last_epoch = epoch
            self.phase = {'stage': 'checkpoint', 'epoch': epoch}
            final = epoch == cfg.epoches - 1
            self.validation_due = bool(self.policy['eval_every'] and
                ((epoch + 1) % self.policy['eval_every'] == 0 or final))
            if self.validation_due or final or (epoch + 1) % self.policy['checkpoint_every'] == 0:
                checkpoint = save_checkpoint(self, self.run_id, steps)
            if self.validation_due:
                self.validate_boundary(checkpoint)
        main_call(lambda: atomic_json(self.output_dir / 'completed.json', {
            'run_id': self.run_id, 'last_epoch': self.last_epoch,
            'checkpoint': checkpoint, 'attempt': self.attempt}))

    def validate_boundary(self, checkpoint):
        for kind in (('raw', 'ema') if self.ema is not None else ('raw',)):
            self.phase = {'stage': 'validation', 'epoch': self.last_epoch, 'state': kind}
            result = validate(self, checkpoint, kind)
            def best():
                path = self.output_dir / f'best_{kind}.json'
                old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                for cohort, metrics in result.items():
                    if not isinstance(metrics, dict) or 'budgets' not in metrics:
                        continue
                    for budget, groups in metrics['budgets'].items():
                        for group in ('coco_all', 'local_hybrid_small'):
                            key = f'{cohort}/{group}/AP@{budget}'
                            score = groups[group]['AP']
                            if score is not None and (key not in old or score > old[key]['score']):
                                old[key] = {'score': score, 'epoch': self.last_epoch, 'checkpoint': checkpoint}
                atomic_json(path, old)
            main_call(best)

    def train_epoch(self, epoch):
        epoch_started = time.monotonic()
        self.model.train(); self.criterion.train()
        rank = dist_utils.get_rank()
        recorder = StepRecorder(self.attempt_dir, epoch) if rank == 0 else None
        samples_path = self.attempt_dir / f'epoch{epoch:04d}.rank{rank:02d}.samples.jsonl'
        sample_stream = samples_path.open('x', encoding='utf-8')
        i, image_ids = -1, []
        try:
            for i, (samples, targets) in enumerate(self.train_dataloader):
                self.phase = {'stage': 'training', 'epoch': epoch, 'step': i}
                started = time.monotonic()
                image_ids = [int(t['image_id'].item()) for t in targets]
                sample_stream.write(json.dumps({'step': i, 'image_ids': image_ids}) + '\n')
                sample_stream.flush()
                samples = samples.to(self.device)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                global_step = epoch * len(self.train_dataloader) + i
                meta = dict(epoch=epoch, step=i, global_step=global_step,
                            epoch_step=len(self.train_dataloader), input_size=tuple(samples.shape[-2:]),
                            small_reference_size=self.policy['small_reference_size'])
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    outputs = self.model(samples, targets=targets)
                require_finite(outputs['pred_boxes'], self.device, 'Decoder boxes')
                require_finite(outputs['pred_logits'], self.device, 'Decoder logits')
                with torch.autocast(device_type=self.device.type, enabled=False):
                    losses = self.criterion(outputs, targets, **meta)
                require_finite(torch.stack(list(losses.values())), self.device, 'training losses')
                loss = sum(losses.values())
                scale_before = self.scaler.get_scale()
                lr_before = [group['lr'] for group in self.optimizer.param_groups]
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                    self.cfg.clip_max_norm if self.cfg.clip_max_norm > 0 else math.inf)
                if not self.scaler.is_enabled():
                    require_finite(norm, self.device, 'unscaled gradients')
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skipped = self.scaler.get_scale() < scale_before
                # DDP must not let one rank advance EMA when another skipped.
                flags = torch.tensor([int(skipped), int(not skipped)], device=self.device)
                if dist_utils.is_dist_available_and_initialized():
                    torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
                if flags.tolist() == [1, 1]:
                    raise RuntimeError('AMP update diverged between ranks; recover last boundary')
                if not skipped and self.ema is not None:
                    self.ema.update(self.model)
                # Preserve baseline scheduler order and absolute attempted-step clock.
                if self.self_lr_scheduler:
                    self.lr_scheduler.step(global_step, self.optimizer)
                elif getattr(self, 'lr_warmup_scheduler', None) is not None:
                    self.lr_warmup_scheduler.step()
                reduced = dist_utils.reduce_dict({k: v.detach() for k, v in losses.items()})
                metrics = (self.criterion.get_logging_metrics()
                           if hasattr(self.criterion, 'get_logging_metrics') else {})
                row = {'epoch': epoch, 'step': i, 'global_step': global_step,
                    'allocation_enabled': getattr(self.criterion, 'allocation_weight', 0.) > 0,
                    'losses': {k: float(v) for k, v in reduced.items()},
                    'metrics': {k: float(v) for k, v in metrics.items()},
                    'lr_before': lr_before, 'lr_after': [g['lr'] for g in self.optimizer.param_groups],
                    'amp_scale_before': scale_before, 'amp_scale_after': self.scaler.get_scale(),
                    'amp_skipped': skipped, 'gradient_norm_before_clip': float(norm) if torch.isfinite(norm) else None,
                    'seconds': time.monotonic() - started}
                main_call(lambda: recorder.record(row))
                if rank == 0 and i % self.cfg.print_freq == 0:
                    print(f'Epoch {epoch} step {i}/{len(self.train_dataloader)} loss={sum(row["losses"].values()):.5f} skipped={skipped}', flush=True)
            def finish():
                summary = recorder.summary()
                summary['epoch_wall_seconds'] = time.monotonic() - epoch_started
                summary['optimizer_updates'] = summary['steps'] - summary['amp_skipped']
                atomic_json(self.attempt_dir / f'epoch{epoch:04d}.summary.json', summary, replace=False)
                return summary
            return main_call(finish)
        except Exception as exc:
            atomic_json(self.attempt_dir / f'failure.rank{rank:02d}.json', {
                'epoch': epoch, 'step': i, 'image_ids': image_ids,
                'error': f'{type(exc).__name__}: {exc}'})
            raise
        finally:
            sample_stream.close()
            if recorder:
                recorder.close()
