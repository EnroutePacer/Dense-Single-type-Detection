"""Same-forward validation archives; no extra head evaluation or inference edits."""
from pathlib import Path
import json

import numpy as np
import torch

from ..misc import dist_utils
from .runtime import atomic_write, atomic_json, digest, main_call, capture_rng, restore_rng


class EncoderCapture:
    def __init__(self, model):
        model = dist_utils.de_parallel(model)
        self.decoder = model if hasattr(model, 'enc_score_head') else model.decoder
        self.handles = []

    def __enter__(self):
        d = self.decoder
        def logits(_m, _a, output):
            self.logits = output.detach()
        def delta(_m, _a, output):
            self.delta = output.detach()
        def reference(_m, args):
            self.reference = args[1].detach()
        self.handles = [d.enc_score_head.register_forward_hook(logits),
                        d.enc_bbox_head.register_forward_hook(delta),
                        d.decoder.register_forward_pre_hook(reference)]
        return self

    def selected(self):
        d = self.decoder
        if self.logits.shape[-1] != 1:
            raise ValueError('Validation archive requires single-class queries')
        masked = self.logits.masked_fill(~d.valid_mask, -torch.inf).squeeze(-1)
        ids = masked.topk(d.num_queries, dim=-1).indices
        anchors = d.anchors.expand(len(ids), -1, -1).gather(1, ids[..., None].expand(-1, -1, 4))
        if not torch.allclose(self.delta + anchors, self.reference, atol=1e-5, rtol=1e-5):
            raise ValueError('Encoder archive/query identity mismatch')
        return {'source_indices': ids.cpu().numpy().astype(np.int32),
                'dense_scores': masked.sigmoid().float().cpu().numpy(),
                'encoder_boxes': self.reference.sigmoid().float().cpu().numpy(),
                'encoder_scores': masked.gather(1, ids).sigmoid().float().cpu().numpy()}

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


def read_predictions(root, shards, expected_ids):
    """Deduplicate only DistributedSampler padding, requiring identical output."""
    by_id = {}
    for item in shards:
        path = Path(root) / item['path']
        if digest(path) != item['sha256']:
            raise ValueError(f'Archive checksum mismatch: {path}')
        with np.load(path, allow_pickle=False) as chunk:
            for image_id, rows in zip(chunk['image_ids'], chunk['predictions']):
                image_id = int(image_id)
                if image_id in by_id:
                    if not np.allclose(by_id[image_id], rows, rtol=1e-5, atol=1e-4):
                        raise ValueError('Sampler padding yielded inconsistent predictions')
                else:
                    by_id[image_id] = rows.copy()
    if set(by_id) != set(expected_ids):
        raise ValueError('Validation archive does not cover the complete validation set')
    return np.concatenate([by_id[i] for i in sorted(by_id)], axis=0)


def verified_result(root, checkpoint_sha):
    path = Path(root) / 'complete.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['checkpoint_sha256'] != checkpoint_sha:
        raise ValueError('Validation/checkpoint identity mismatch')
    for item in result['shards']:
        if digest(Path(root) / item['path']) != item['sha256']:
            raise ValueError('Completed validation archive has been changed')
    return result


@torch.no_grad()
def validate(solver, checkpoint, kind):
    from tools.table_cells.local_small_eval import evaluate_predictions, dense_image_ids
    root = solver.output_dir / 'validation' / f'epoch{solver.last_epoch:04d}' / kind
    cached = main_call(lambda: verified_result(root, checkpoint['sha256']))
    if cached is not None:
        return cached
    rng = capture_rng()
    saved_buffers = []
    try:
        model = dist_utils.de_parallel(solver.model) if kind == 'raw' else solver.ema.module
        model.eval()
        # Rank-local BN/EMA buffers may differ after the last training batch.
        # Evaluate the rank-0 checkpoint on every shard, then restore local state.
        if dist_utils.is_dist_available_and_initialized():
            for buffer in model.buffers():
                saved_buffers.append((buffer, buffer.clone()))
                torch.distributed.broadcast(buffer, src=0)
        directory = root / f'attempt{solver.attempt:04d}'
        directory.mkdir(parents=True, exist_ok=True)
        rank = dist_utils.get_rank()
        shards, pending, count = [], [], 0

        def flush():
            if not pending:
                return
            arrays = {key: np.concatenate([row[key] for row in pending]) for key in pending[0]}
            if any(not np.isfinite(v).all() for v in arrays.values()):
                raise FloatingPointError('Non-finite validation archive')
            path = directory / f'rank{rank:02d}.chunk{len(shards):05d}.npz'
            atomic_write(path, lambda stream: np.savez_compressed(stream, **arrays), replace=False)
            shards.append({'path': str(path.relative_to(root)), 'sha256': digest(path),
                           'images': len(arrays['image_ids'])})
            pending.clear()

        with EncoderCapture(model) as capture:
            for step, (samples, targets) in enumerate(solver.val_dataloader):
                outputs = model(samples.to(solver.device))
                sizes = torch.stack([t['orig_size'] for t in targets]).to(solver.device)
                results = solver.postprocessor(outputs, sizes)
                ids = np.array([int(t['image_id'].item()) for t in targets], dtype=np.int64)
                predictions = []
                for image_id, result in zip(ids, results):
                    boxes = result['boxes'].detach().float().cpu().numpy().copy()
                    boxes[:, 2:] -= boxes[:, :2]
                    predictions.append(np.column_stack([np.full(len(boxes), image_id), boxes,
                        result['scores'].detach().cpu().numpy(), result['labels'].detach().cpu().numpy()]))
                pending.append({**capture.selected(), 'image_ids': ids,
                    'orig_sizes': sizes.cpu().numpy(), 'predictions': np.stack(predictions),
                    'decoder_boxes': outputs['pred_boxes'].float().cpu().numpy(),
                    'decoder_scores': outputs['pred_logits'].sigmoid().squeeze(-1).float().cpu().numpy()})
                count += len(ids)
                if sum(len(p['image_ids']) for p in pending) >= 128:
                    flush()
                if step % 25 == 0 and rank == 0:
                    print(f'Validation epoch={solver.last_epoch} {kind} batch={step}/{len(solver.val_dataloader)}', flush=True)
        flush()
        manifests = [shards]
        if dist_utils.is_dist_available_and_initialized():
            manifests = [None] * dist_utils.get_world_size()
            torch.distributed.all_gather_object(manifests, shards)

        def finish():
            shards_all = [item for entries in manifests for item in entries]
            coco = solver.evaluator.coco_gt
            predictions = read_predictions(root, shards_all, coco.imgs)
            result = {'epoch': solver.last_epoch, 'state': kind, 'run_id': solver.run_id,
                      'checkpoint_sha256': checkpoint['sha256'], 'shards': shards_all,
                      'full': evaluate_predictions(coco, predictions),
                      'dense_ge301': evaluate_predictions(coco, predictions, image_ids=dense_image_ids(coco))}
            atomic_json(root / 'complete.json', result, replace=False)
            return result
        return main_call(finish)
    finally:
        for buffer, previous in saved_buffers:
            buffer.copy_(previous)
        restore_rng(rng)
