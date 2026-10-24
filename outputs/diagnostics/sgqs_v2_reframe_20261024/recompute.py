"""Recompute descriptive, GT-free statistics from committed 13-image exports.
No model is run. Rank retention is a static identity proxy, not pruned-model recall.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'baseline_test/output/data_threshold_0.30_compare'
OUT = Path(__file__).with_name('verified_candidate_audit.json')

def box_matrix(frame):
    a = frame[['final_cx', 'final_cy', 'final_w', 'final_h']].to_numpy(dtype=float)
    return np.concatenate([a[:, :2] - a[:, 2:] / 2, a[:, :2] + a[:, 2:] / 2], axis=1)

def pair_iou(a, b):
    wh = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]))
    inter = wh.prod(axis=2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    ab = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)

result = {
    'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'scope': '13 fixed images without GT; epoch34 EMA; full640 existing outputs; no pruning rerun',
    'box_space': 'normalized final cxcywh converted to unclipped xyxy',
    'source_distance': 'feature grid centers in 640x640 input pixels, not regressed encoder box centers',
    'rank_retention_definition': 'fraction of original final_score>=0.5 queries whose encoder_rank<=K',
}
for method in ['b0', 'sgqs']:
    folder = BASE / f'{method}_e34_raw_complete'
    dec_path = folder / 'decoder_queries_without_post_topk.csv'
    enc_path = folder / 'encoder_candidates_before_topk.csv'
    dec = pd.read_csv(dec_path)
    enc = pd.read_csv(enc_path)
    info = json.loads((folder / 'query_diagnostic_summary.json').read_text(encoding='utf-8-sig'))
    joined = dec.merge(enc[['image_key', 'dense_index', 'encoder_score']], left_on=['image_key','source_dense_index'], right_on=['image_key','dense_index'], suffixes=('', '_source'), validate='one_to_one')
    assert len(joined) == len(dec) == 8320
    assert np.allclose(joined.encoder_score, joined.encoder_score_source, rtol=0, atol=1e-12)
    assert dec.groupby('image_key').size().eq(640).all()
    high = dec.final_score.ge(0.5)
    m = {
        'state_source': info['model']['state_source'],
        'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [dec_path, enc_path]},
        'images': dec.image_key.nunique(),
        'queries': len(dec),
        'encoder_score_mean': float(dec.encoder_score.mean()),
        'encoder_score_median': float(dec.encoder_score.median()),
        'encoder_score_p90': float(dec.encoder_score.quantile(.9)),
        'decoder_score_mean': float(dec.final_score.mean()),
        'encoder_cutoff_mean': float(dec.loc[dec.encoder_rank.eq(640), 'encoder_score'].mean()),
        'high_final_queries': int(high.sum()),
        'rank_retention': {str(k): float((high & dec.encoder_rank.le(k)).sum()/high.sum()) for k in [64,128,192,256,384,512,640]},
        'encoder_threshold': {},
        'suspect_duplicates': {'encoder_ge': .3, 'decoder_lt': .1, 'keeper_decoder_ge': .5, 'final_iou_ge': .7, 'suspect_count': 0, 'overlapping_suspect_count': 0},
        'overlap': [],
    }
    for cut in [.1,.2,.3,.4,.5]:
        kept = dec.encoder_score.ge(cut)
        m['encoder_threshold'][str(cut)] = {'queries_per_image': float(kept.sum()/m['images']), 'high_final_retained': float((kept & high).sum()/high.sum())}
    for _, f in dec.groupby('image_key'):
        suspect = f.encoder_score.ge(.3) & f.final_score.lt(.1)
        keeper = f.final_score.ge(.5)
        m['suspect_duplicates']['suspect_count'] += int(suspect.sum())
        if suspect.any() and keeper.any():
            m['suspect_duplicates']['overlapping_suspect_count'] += int((pair_iou(box_matrix(f[suspect]), box_matrix(f[keeper])).max(axis=1) >= .7).sum())
    for cut in [None, .3, .5]:
        pairs = involved = population = denominator = same = 0
        distances = []
        for _, f in dec.groupby('image_key'):
            if cut is not None:
                f = f[f.final_score.ge(cut)]
            n = len(f)
            population += n
            denominator += n*(n-1)//2
            a,b = np.where(np.triu(pair_iou(box_matrix(f), box_matrix(f)) >= .7, k=1))
            pairs += len(a)
            involved += len(np.unique(np.concatenate([a,b])))
            centers = (f[['column','row']].to_numpy() + .5)*f.stride.to_numpy()[:,None]
            distances.extend(np.linalg.norm(centers[a]-centers[b],axis=1).tolist())
            levels = f.level.to_numpy()
            same += int((levels[a]==levels[b]).sum())
        m['overlap'].append({'score_ge': cut, 'iou_ge': .7, 'queries':population, 'pairs':pairs, 'pair_fraction':pairs/max(denominator,1), 'participating_queries':involved, 'participating_query_fraction':involved/max(population,1), 'mean_source_grid_distance_px':float(np.mean(distances)) if distances else None, 'same_level_fraction':same/max(pairs,1)})
    result[method] = m
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
for method in ['b0','sgqs']:
    m = result[method]
    print(method, json.dumps({k:m[k] for k in ['state_source','encoder_score_mean','encoder_score_median','rank_retention','suspect_duplicates','overlap']}, ensure_ascii=False))
print(OUT)
