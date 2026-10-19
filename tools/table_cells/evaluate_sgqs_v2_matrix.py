"""Complete validation matrix: every arm x epoch x raw/EMA x both budgets."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
import faster_coco_eval
from faster_coco_eval import COCO
from engine.core import YAMLConfig
from engine import sgqs_v2  # noqa: F401
from tools.table_cells import supervision_probe  # noqa: F401
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.prepare_sgqs_v2_run import ARMS
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.prepare_sgqs_v2_run import RUNTIME_FILES
from tools.table_cells.local_small_eval import evaluate_predictions, fixed_box_exchange


def jobs(run_root, arms, epochs, states):
    if any(len(values) != len(set(values)) for values in (arms, epochs, states)):
        raise ValueError('Duplicate matrix coordinates')
    return [(arm, epoch, state, run_root / f'{arm}_e28_r7_seed0')
            for arm in arms for epoch in epochs for state in states]


def image_fingerprint(coco, root):
    digest = hashlib.sha256()
    for index, image_id in enumerate(sorted(coco.imgs)):
        name = coco.imgs[image_id]['file_name']
        digest.update(json.dumps([image_id, name, sha256(root/name)], ensure_ascii=False).encode())
        if index % 1000 == 0:
            print(f'Image identity check: {index + 1}/{len(coco.imgs)}', flush=True)
    return digest.hexdigest()


@torch.no_grad()
def predict(model, coco, images_root, device, swap=False):
    transform = v2.Compose([v2.Resize([640, 640]), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    category = coco.getCatIds()[0]
    # [image_id, x, y, w, h, score, category_id], no prefilter or score threshold.
    rows, swapped_rows, pairs = [], [], 0
    for image_id in sorted(coco.imgs):
        info = coco.imgs[image_id]
        with Image.open(images_root / info['file_name']) as image:
            if image.size != (info['width'], info['height']):
                raise ValueError(f'Image/annotation size mismatch: {image_id}')
            output = model(transform(image.convert('RGB')).unsqueeze(0).to(device))
        boxes = output['pred_boxes'][0].float()
        if output['pred_logits'].shape[-1] != 1:
            raise ValueError('Require single-class predictions')
        scores = output['pred_logits'][0, :, 0].float().sigmoid()
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise FloatingPointError(f'Non-finite predictions: image {image_id}')
        if len(boxes) != 640:
            raise ValueError('Expected every one of the 640 decoder queries')
        data = np.zeros((len(boxes), 7), dtype=np.float64)
        data[:, 0], data[:, 6] = image_id, category
        normalized = boxes.cpu().numpy()
        data[:, 1:3] = normalized[:, :2] - normalized[:, 2:] / 2
        data[:, 3:5] = normalized[:, 2:]
        data[:, 1:5] *= [info['width'], info['height'], info['width'], info['height']]
        data[:, 5] = scores.cpu().numpy()
        rows.append(data)
        if swap:
            anns = [a for a in coco.imgToAnns[image_id] if not a.get('iscrowd', 0)]
            gt = boxes.new_tensor([[(a['bbox'][0] + a['bbox'][2]/2)/info['width'],
                                    (a['bbox'][1] + a['bbox'][3]/2)/info['height'],
                                    a['bbox'][2]/info['width'], a['bbox'][3]/info['height']]
                                   for a in anns]).reshape(-1, 4)
            new_scores, count = fixed_box_exchange(boxes, scores, gt)
            changed = data.copy(); changed[:, 5] = new_scores.cpu().numpy()
            swapped_rows.append(changed); pairs += count
    return np.concatenate(rows), np.concatenate(swapped_rows) if swap else None, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--epochs', nargs='+', type=int, default=list(range(28, 35)))
    p.add_argument('--states', nargs='+', choices=['raw', 'ema'], default=['raw', 'ema'])
    p.add_argument('--device', default='cuda')
    p.add_argument('--fixed-box-exchange', action='store_true')
    p.add_argument('--skip-completed', action='store_true')
    args = p.parse_args()
    _, runtime_manifest = load_runtime_files(ROOT, RUNTIME_FILES)
    matrix = jobs(args.run_root, args.arms, args.epochs, args.states)
    annotation_sha = sha256(args.annotation)
    identities = {}
    # Check the whole requested matrix before any expensive evaluation starts.
    for arm, epoch, state, run in matrix:
        for path in (run/'resolved.yml', run/'run_manifest.json', run/f'checkpoint{epoch:04d}.pth'):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        if manifest['inputs']['val_annotation']['sha256'] != annotation_sha:
            raise ValueError(f'{arm}: validation annotation identity differs from training')
        if sha256(run/'resolved.yml') != manifest['resolved_config_sha256']:
            raise ValueError(f'{arm}: prepared config changed')
        checkpoint = run/f'checkpoint{epoch:04d}.pth'
        identities.setdefault(str(checkpoint), None)
    for checkpoint in identities:
        identities[checkpoint] = sha256(Path(checkpoint))
    coco = COCO(str(args.annotation))
    if len(coco.cats) != 1 or not coco.imgs:
        raise ValueError('Require a nonempty, single-class full validation dataset')
    image_sha = image_fingerprint(coco, args.images)
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_files = ['tools/table_cells/local_small_eval.py',
                      'tools/table_cells/evaluate_sgqs_v2_matrix.py', 'engine/sgqs_v2/admission.py']
    protocol_hash = hashlib.sha256(''.join(sha256(ROOT/f) for f in protocol_files).encode()).hexdigest()
    environment = {'torch': torch.__version__, 'coco_eval': faster_coco_eval.__version__,
                   'numpy': np.__version__, 'pillow': importlib.metadata.version('pillow'),
                   'torchvision': importlib.metadata.version('torchvision'),
                   'platform': platform.platform(), 'device': args.device,
                   'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name() if args.device.startswith('cuda') else None}
    request = {'arms': args.arms, 'epochs': args.epochs, 'states': args.states,
               'annotation_sha256': annotation_sha, 'image_population_sha256': image_sha,
               'checkpoints': identities, 'runtime_manifest': runtime_manifest,
               'protocol_sha256': protocol_hash, 'environment': environment,
               'fixed_box_exchange': args.fixed_box_exchange}
    request_file = args.output/'request.json'
    if request_file.exists():
        if json.loads(request_file.read_text(encoding='utf-8')) != request:
            raise ValueError('Matrix request or inputs changed; use a separate output directory')
    else:
        with request_file.open('x', encoding='utf-8') as stream:
            json.dump(request, stream, indent=2)
    records = []
    model, current_arm, weights, current_checkpoint = None, None, None, None
    for arm, epoch, state, run in matrix:
        checkpoint = run/f'checkpoint{epoch:04d}.pth'
        identity = {'arm': arm, 'epoch': epoch, 'state': state, 'annotation_sha256': annotation_sha,
                    'image_population_sha256': image_sha, 'runtime_manifest': runtime_manifest,
                    'images_root': str(args.images.resolve()), 'images': len(coco.imgs),
                    'checkpoint_sha256': identities[str(checkpoint)],
                    'config_sha256': sha256(run/'resolved.yml'), 'protocol_sha256': protocol_hash,
                    'fixed_box_exchange': args.fixed_box_exchange, 'environment': environment}
        destination = args.output/f'{arm}.e{epoch}.{state}.json'
        if destination.exists():
            record = json.loads(destination.read_text(encoding='utf-8'))
            if not args.skip_completed or record['identity'] != identity:
                raise ValueError(f'Existing result cannot be reused: {destination}')
        else:
            if current_arm != arm:
                cfg = YAMLConfig(str(run/'resolved.yml'))
                cfg.yaml_cfg['HGNetv2']['pretrained'] = False
                if cfg.yaml_cfg['eval_spatial_size'] != [640, 640]:
                    raise ValueError('Require fixed 640 input')
                model = cfg.model.to(args.device).eval(); current_arm = arm
            if current_checkpoint != checkpoint:
                weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if weights.get('last_epoch') != epoch:
                    raise ValueError(f'Checkpoint epoch disagrees with filename: {checkpoint}')
                current_checkpoint = checkpoint
            model.load_state_dict(weights['model'] if state == 'raw' else weights['ema']['module'], strict=True)
            predictions, swapped, pairs = predict(model, coco, args.images, args.device, args.fixed_box_exchange)
            prediction_hash = hashlib.sha256(predictions.tobytes()).hexdigest()
            record = {'identity': identity, 'predictions_sha256': prediction_hash,
                      'metrics': evaluate_predictions(coco, predictions)}
            if swapped is not None:
                record['fixed_box_exchange'] = {'pairs': pairs, 'metrics': evaluate_predictions(coco, swapped)}
            temporary = destination.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(destination)
            del predictions, swapped
        for budget, groups in record['metrics']['budgets'].items():
            row = {'arm': arm, 'epoch': epoch, 'state': state, 'budget': int(budget)}
            for group, metrics in groups.items():
                row.update({f'{group}_{key}': value for key, value in metrics.items()})
            records.append(row)
        print(destination, flush=True)
    with (args.output/'matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (args.output/'completed.json').write_text(json.dumps({
        'states': len(matrix), 'budget_rows': len(records), 'arms': args.arms,
        'epochs': args.epochs, 'inference_states': args.states, 'annotation_sha256': annotation_sha,
        'image_population_sha256': image_sha,
        'environment': environment, 'scope': 'full_annotation_population'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
