"""Full validation of SGQS v2, including actual pre-Decoder query budgets.

Output maxDets stays 300/640. Changing input K always runs the model again;
this tool never presents a slice of a 640-query forward as a cheaper forward.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from engine.data.dataset.coco_backend import COCO, backend_info
from engine.core import YAMLConfig
from engine import sgqs_v2  # noqa: F401
from tools.table_cells.prepare_probe_subset import sha256
from tools.table_cells.prepare_sgqs_v2_run import ARMS, RUNTIME_FILES
from tools.table_cells.prepare_probe_run import load_runtime_files
from tools.table_cells.local_small_eval import evaluate_predictions, fixed_box_exchange, dense_image_ids


OUTPUT_BUDGETS = (300, 640)
DENSE_MINIMUM_GT = 301


def jobs(run_root, arms, epochs, states, query_budgets=(640,)):
    coordinates = (arms, epochs, states, query_budgets)
    if any(not values or len(values) != len(set(values)) for values in coordinates):
        raise ValueError('Empty or duplicate matrix coordinates')
    if any(k < 1 or k > 640 for k in query_budgets):
        raise ValueError('Query budgets must be in [1, 640]')
    return [(arm, epoch, state, k, run_root / f'{arm}_e28_r7_seed0')
            for arm in arms for epoch in epochs for state in states for k in query_budgets]


def set_query_budget(model, budget):
    """Change native Encoder Top-K before bbox decoding and Decoder attention."""
    decoder = getattr(model, 'decoder', None)
    if decoder is None or not hasattr(decoder, 'num_queries'):
        raise ValueError('Require a model with the native DFINE decoder query selector')
    if decoder.learn_query_content:
        raise ValueError('Changing K requires learn_query_content=False')
    if budget < 1 or budget > 640:
        raise ValueError('Query budget must be in [1, 640]')
    decoder.num_queries = int(budget)


def image_fingerprint(coco, root):
    digest = hashlib.sha256()
    for index, image_id in enumerate(sorted(coco.imgs)):
        name = coco.imgs[image_id]['file_name']
        digest.update(json.dumps([image_id, name, sha256(root/name)], ensure_ascii=False).encode())
        if index % 1000 == 0:
            print(f'Image identity check: {index + 1}/{len(coco.imgs)}', flush=True)
    return digest.hexdigest()


def _synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def predict(model, coco, images_root, device, swap=False, expected_queries=640, warmup_iterations=3):
    if warmup_iterations < 0:
        raise ValueError('warmup_iterations must be non-negative')
    if model.training:
        raise ValueError('Evaluation requires model.eval()')
    set_query_budget(model, expected_queries)
    transform = v2.Compose([v2.Resize([640, 640]), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    category = coco.getCatIds()[0]
    # [image_id, x, y, w, h, score, category_id], no prefilter or score threshold.
    rows, swapped_rows, pairs, durations = [], [], 0, []
    for position, image_id in enumerate(sorted(coco.imgs)):
        info = coco.imgs[image_id]
        with Image.open(images_root / info['file_name']) as image:
            if image.size != (info['width'], info['height']):
                raise ValueError(f'Image/annotation size mismatch: {image_id}')
            inputs = transform(image.convert('RGB')).unsqueeze(0).to(device)
        if position == 0:
            for _ in range(warmup_iterations):
                model(inputs)
        _synchronize(device)
        started = time.perf_counter()
        output = model(inputs)
        _synchronize(device)
        durations.append(time.perf_counter() - started)
        boxes = output['pred_boxes'][0].float()
        if output['pred_logits'].shape[-1] != 1:
            raise ValueError('Require single-class predictions')
        scores = output['pred_logits'][0, :, 0].float().sigmoid()
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise FloatingPointError(f'Non-finite predictions: image {image_id}')
        if len(boxes) != expected_queries or len(scores) != expected_queries:
            raise ValueError(f'Expected exactly {expected_queries} native Decoder outputs')
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
    elapsed = np.asarray(durations, dtype=np.float64)
    timing = {'scope': 'synchronized_model_forward_only_not_deployment_latency',
              'excludes': ['image_io', 'preprocessing', 'host_to_device', 'postprocessing', 'warmup'],
              'warmup_iterations': warmup_iterations, 'images': len(durations),
              'batch_size': 1, 'query_budget': expected_queries, 'input_size': [640, 640],
              'seconds': float(elapsed.sum()),
              'mean_ms': float(elapsed.mean()*1000) if len(elapsed) else None,
              'median_ms': float(np.median(elapsed)*1000) if len(elapsed) else None,
              'p95_ms': float(np.quantile(elapsed, .95)*1000) if len(elapsed) else None}
    return (np.concatenate(rows) if rows else np.empty((0, 7)),
            np.concatenate(swapped_rows) if swapped_rows else None, pairs, timing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--annotation', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['b0', 'support', 'v2'])
    p.add_argument('--epochs', nargs='+', type=int, default=list(range(28, 35)))
    p.add_argument('--states', nargs='+', choices=['raw', 'ema'], default=['raw', 'ema'])
    p.add_argument('--query-budgets', nargs='+', type=int, default=[640])
    p.add_argument('--warmup-iterations', type=int, default=3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--fixed-box-exchange', action='store_true', help='Historical GT-assisted output-score diagnostic only')
    p.add_argument('--skip-completed', action='store_true')
    args = p.parse_args()
    if args.warmup_iterations < 0:
        p.error('--warmup-iterations must be non-negative')
    _, runtime_manifest = load_runtime_files(ROOT, RUNTIME_FILES)
    matrix = jobs(args.run_root, args.arms, args.epochs, args.states, args.query_budgets)
    annotation_sha = sha256(args.annotation)
    identities = {}
    # Check every requested checkpoint before expensive image checks or evaluation.
    for arm, epoch, state, k, run in matrix:
        for path in (run/'resolved.yml', run/'run_manifest.json', run/f'checkpoint{epoch:04d}.pth'):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
        if manifest['schema'] != 'sgqs_v2_coverage_allocation_v1' or manifest['arm'] != arm:
            raise ValueError(f'{arm}: require the matching prepared coverage-allocation run')
        if manifest['inputs']['val_annotation']['sha256'] != annotation_sha:
            raise ValueError(f'{arm}: validation annotation identity differs from training')
        if sha256(run/'resolved.yml') != manifest['resolved_config_sha256']:
            raise ValueError(f'{arm}: prepared config changed')
        identities.setdefault(str(run/f'checkpoint{epoch:04d}.pth'), None)
    for checkpoint in identities:
        identities[checkpoint] = sha256(Path(checkpoint))
    coco = COCO(str(args.annotation))
    if len(coco.cats) != 1 or not coco.imgs:
        raise ValueError('Require a nonempty, single-class full validation dataset')
    image_sha = image_fingerprint(coco, args.images)
    dense_ids = dense_image_ids(coco, DENSE_MINIMUM_GT)
    dense_cohort = {'selection': 'annotation_non_crowd_gt_count', 'minimum_gt': DENSE_MINIMUM_GT,
                    'images': len(dense_ids), 'image_ids_sha256': hashlib.sha256(
                        json.dumps(dense_ids).encode()).hexdigest()}
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_files = ['tools/table_cells/local_small_eval.py',
                      'tools/table_cells/evaluate_sgqs_v2_matrix.py', 'engine/sgqs_v2/admission.py',
                      'engine/data/dataset/coco_backend.py']
    protocol_hash = hashlib.sha256(''.join(sha256(ROOT/f) for f in protocol_files).encode()).hexdigest()
    environment = {'torch': torch.__version__, 'coco_eval': backend_info(),
                   'numpy': np.__version__, 'pillow': importlib.metadata.version('pillow'),
                   'torchvision': importlib.metadata.version('torchvision'),
                   'platform': platform.platform(), 'device': args.device,
                   'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
                   'torch_threads': torch.get_num_threads(),
                   'torch_interop_threads': torch.get_num_interop_threads(),
                   'dtype': 'float32', 'autocast': False,
                   'cudnn_benchmark': torch.backends.cudnn.benchmark,
                   'cudnn_deterministic': torch.backends.cudnn.deterministic}
    request = {'protocol': 'sgqs_v2_coverage_matrix_v1',
               'arms': args.arms, 'epochs': args.epochs, 'states': args.states,
               'query_budgets': args.query_budgets, 'output_budgets': list(OUTPUT_BUDGETS),
               'warmup_iterations': args.warmup_iterations, 'dense_cohort': dense_cohort,
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
    for arm, epoch, state, k, run in matrix:
        checkpoint = run/f'checkpoint{epoch:04d}.pth'
        identity = {'arm': arm, 'epoch': epoch, 'state': state, 'query_budget': k,
                    'output_budgets': list(OUTPUT_BUDGETS), 'warmup_iterations': args.warmup_iterations,
                    'annotation_sha256': annotation_sha, 'dense_cohort': dense_cohort,
                    'image_population_sha256': image_sha, 'runtime_manifest': runtime_manifest,
                    'images_root': str(args.images.resolve()), 'images': len(coco.imgs),
                    'checkpoint_sha256': identities[str(checkpoint)],
                    'config_sha256': sha256(run/'resolved.yml'), 'protocol_sha256': protocol_hash,
                    'fixed_box_exchange': args.fixed_box_exchange, 'environment': environment}
        destination = args.output/f'{arm}.e{epoch}.{state}.k{k}.json'
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
            predictions, swapped, pairs, timing = predict(
                model, coco, args.images, args.device, args.fixed_box_exchange,
                expected_queries=k, warmup_iterations=args.warmup_iterations)
            record = {'identity': identity, 'predictions_sha256': hashlib.sha256(predictions.tobytes()).hexdigest(),
                      'timing': timing,
                      'metrics': evaluate_predictions(coco, predictions, OUTPUT_BUDGETS),
                      'dense_scene_metrics': evaluate_predictions(coco, predictions, OUTPUT_BUDGETS, dense_ids)}
            if swapped is not None:
                record['fixed_box_exchange'] = {'pairs': pairs, 'metrics': evaluate_predictions(coco, swapped, OUTPUT_BUDGETS)}
            temporary = destination.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
            temporary.replace(destination)
            del predictions, swapped
        for budget, groups in record['metrics']['budgets'].items():
            row = {'arm': arm, 'epoch': epoch, 'state': state, 'query_budget': k,
                   'output_budget': int(budget), 'forward_mean_ms': record['timing']['mean_ms'],
                   'forward_p95_ms': record['timing']['p95_ms']}
            for group, metrics in groups.items():
                row.update({f'{group}_{key}': value for key, value in metrics.items()})
            for group, metrics in record['dense_scene_metrics']['budgets'][budget].items():
                row.update({f'dense_scene_{group}_{key}': value for key, value in metrics.items()})
            records.append(row)
        print(destination, flush=True)
    with (args.output/'matrix.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (args.output/'completed.json').write_text(json.dumps({
        'inference_runs': len(matrix), 'checkpoint_states': len(matrix)//len(args.query_budgets),
        'budget_rows': len(records), 'arms': args.arms, 'query_budgets': args.query_budgets,
        'output_budgets': list(OUTPUT_BUDGETS), 'dense_cohort': dense_cohort,
        'epochs': args.epochs, 'inference_states': args.states, 'annotation_sha256': annotation_sha,
        'image_population_sha256': image_sha,
        'environment': environment, 'scope': 'full_annotation_population'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
