"""Reproduce the B0/B1/B2 log comparison without changing training code."""
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'article/experiments/EXP-TRAIN-005_epoch27-sgqs-e-resume-screen/diagnostics/20261013_b1_b2'
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = {
    'B0': ROOT / 'outputs/resume_control_e28_to_e34_gb64/console.log',
    'B1': ROOT / 'outputs/e35_r7_sgqse/resume_b1_sgqs_seed0/console.log',
    'B2': ROOT / 'outputs/e35_r7_sgqse/resume_b2_efgl_seed0/console.log',
}
PAIR = re.compile(r'([a-zA-Z_][\w]*):\s+([-+\deE.]+)(?:\s+\(([-+\deE.]+)\))?')


def parse(path):
    raw = path.read_bytes()
    lines = raw.decode('utf-8-sig').splitlines()
    epoch, rows, steps, evals, config = 27, {}, [], [], None
    for number, line in enumerate(lines, 1):
        if line.startswith('cfg:'):
            config = ast.literal_eval(line.split('cfg:', 1)[1].strip())
        match = re.search(r'Epoch: \[(\d+)\]', line)
        if match:
            epoch = int(match[1])
        if line.startswith('Averaged stats:') and 'loss:' in line:
            rows[epoch] = dict(line=number, **{k: float(a or v) for k, v, a in PAIR.findall(line)})
        match = re.search(r'Epoch: \[(\d+)\]\s+\[\s*(\d+)/(\d+)\]', line)
        if match:
            steps.append(dict(epoch=int(match[1]), step=int(match[2]), length=int(match[3]),
                              line=number, median={k: float(v) for k, v, a in PAIR.findall(line)},
                              running={k: float(a or v) for k, v, a in PAIR.findall(line)}))
        if line.strip().startswith('Average Precision') and 'IoU=0.50:0.95' in line and 'area=   all' in line:
            evals.append(dict(epoch=epoch, line=number, metrics=[]))
        if line.strip().startswith(('Average Precision', 'Average Recall')):
            evals[-1]['metrics'].append(float(line.split('=')[-1]))
        if line.startswith('best_stat:'):
            evals[-1]['best_stat'] = ast.literal_eval(line.split('best_stat:', 1)[1])
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                lines=len(lines), config=config, averages=rows, steps=steps, evals=evals,
                completion=lines[-1])


runs = {name: parse(path) for name, path in INPUTS.items()}
(OUT / 'parsed.json').write_text(json.dumps(runs, ensure_ascii=False, indent=2), encoding='utf-8')
baseline = {int(row['epoch']): row for row in csv.DictReader(
    (ROOT / 'outputs/resume_control_e28_to_e34_gb64/metrics.csv').open(encoding='utf-8-sig'))}
rows = []
for name, run in runs.items():
    assert len(run['steps']) == 406 and len(run['evals']) == 8
    assert list(run['averages']) == list(range(28, 35))
    for ev in run['evals']:
        epoch, best = ev['epoch'], ev['best_stat']
        # A best_stat from a previous epoch is NOT the current epoch's exact AP.
        exact = best['coco_eval_bbox'] if best['epoch'] == epoch else None
        small, recall = ev['metrics'][3], ev['metrics'][9]
        if name == 'B0' and epoch in baseline:
            small, recall = float(baseline[epoch]['ap_small']), float(baseline[epoch]['ar_small'])
        rows.append(dict(arm=name, epoch=epoch, map_exact=exact, map_console=ev['metrics'][0],
                         ap_small=small, ar_small=recall, ap_small_precision=('B0 CSV (12 decimals)' if name == 'B0' and epoch in baseline else 'rounded to 0.001'),
                         source_line=ev['line']))
with (OUT / 'epoch_metrics.csv').open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
colors = {'B0': '#687587', 'B1': '#1769aa', 'B2': '#cc6b1b'}
labels = {'B0': 'B0 control', 'B1': 'B1 SGQS', 'B2': 'B2 EFGL'}
for name in runs:
    data = [row for row in rows if row['arm'] == name and row['epoch'] >= 28]
    x = [row['epoch'] for row in data]
    y = [row['map_exact'] if row['map_exact'] is not None else row['map_console'] for row in data]
    axes[0, 0].plot(x, y, 'o-', color=colors[name], label=labels[name], markersize=4)
    axes[0, 1].plot(x, [row['ap_small'] for row in data], 'o-', color=colors[name], label=labels[name], markersize=4)
axes[0, 0].errorbar([29], [.864], yerr=.0005, fmt='none', color=colors['B1'], capsize=4)
axes[0, 0].set(title='Validation mAP: B1 has a small overall gain', ylabel='mAP @ IoU 0.50:0.95')
axes[0, 0].text(.03, .04, 'B1 epoch 29: console-rounded value + rounding interval', transform=axes[0, 0].transAxes, fontsize=8)
axes[0, 1].set(title='COCO AP-small: early B1 gain is not retained', ylabel='AP-small (original-image area bins)')
axes[0, 1].set_ylim(.675, .708)
axes[0, 1].text(.03, .04, 'B1/B2 AP-small is rounded to 0.001 in the supplied logs', transform=axes[0, 1].transAxes, fontsize=8)
for ax in axes[0]:
    ax.legend(frameon=False)

epoch_means = runs['B1']['averages']
for key, label, color in [
    ('sgqs_owner_coverage', 'Retained owners / all GT', '#1769aa'),
    ('sgqs_conflicting_owner_fraction', 'Dropped owners / assigned owners', '#bd493a'),
    ('sgqs_pre_topk_survival_rate', 'Selected / retained eligible owners', '#358052')]:
    axes[1, 0].plot(list(epoch_means), [100 * row[key] for row in epoch_means.values()], 'o-', label=label, color=color, markersize=4)
axes[1, 0].set(title='SGQS: the reported survival denominator is filtered', ylabel='Epoch mean of per-batch ratios (%)', ylim=(0, 105))
axes[1, 0].legend(frameon=False, fontsize=8, loc='lower right')

means = runs['B2']['averages']
axes[1, 1].bar(list(means), [row['efgl_weight_bonus_mean'] for row in means.values()], color=colors['B2'], width=.55, label='Logged mean weight bonus')
axes[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
axes[1, 1].set(title='EFGL: small mean correction and 2.1-epoch duration', ylabel='Main normal branch mean (rounded to 0.0001)')
second = axes[1, 1].twinx()
px = [28 + i / 100 for i in range(701)]
second.plot(px, [max(0, 1 - (t - 28) / 2.1) for t in px], '--', color='#687587', label='Configured compensation ramp')
second.set(ylabel='c(t)', ylim=(0, 1.08))
axes[1, 1].axvline(30.1, color='#999999', linestyle=':', linewidth=1)
axes[1, 1].legend(frameon=False, fontsize=8, loc='upper right')
second.legend(frameon=False, fontsize=8, loc='center right')
for ax in axes.flat:
    ax.set_xlabel('Absolute epoch')
    ax.grid(axis='y', alpha=.15)
    ax.set_xticks(range(28, 35))
fig.suptitle('Seven-epoch continuation: observed outcomes and supervision strength', fontsize=14)
fig.savefig(OUT / 'diagnosis.png', dpi=160)
print('Parsed 3 logs; wrote epoch_metrics.csv, parsed.json and diagnosis.png')
