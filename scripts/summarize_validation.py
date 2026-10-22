"""Summarize the 18 validation runs; no training or test evaluation."""
import argparse
import csv
import json
import re
import statistics as st
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=Path, default=Path('results'))
parser.add_argument('--output-dir', type=Path, default=Path('results/validation_summary'))
args = parser.parse_args()
metrics = ['macro_f1', 'hate_precision', 'hate_recall', 'false_positive_rate', 'average_precision']
models = ['bert', 'hatebert']
strategies = ['unweighted', 'weighted', 'oversampling']
seeds = [42, 123, 999]
runs = {}
for path in sorted(args.results_dir.glob('*/threshold_summary.json')):
    match = re.fullmatch(r'(bert|hatebert)_(unweighted|weighted|oversampling)_seed(42|123|999)_\d{8}_\d{6}', path.parent.name)
    if not match:
        continue
    model, strategy, seed = match.groups()
    key = (model, strategy, int(seed))
    if key in runs:
        raise ValueError(f'Duplicate run for {key}; resolve before summarizing.')
    data = json.loads(path.read_text())
    if data['split'] != 'validation':
        raise ValueError(f'Not validation: {path}')
    for mode in ['default', 'tuned']:
        d = data[mode]
        if d['tn'] + d['fp'] != 573 or d['tp'] + d['fn'] != 427:
            raise ValueError(f'Unexpected validation class totals: {path}')
    runs[key] = (path.parent.name, data)
expected = {(m,s,n) for m in models for s in strategies for n in seeds}
if set(runs) != expected:
    raise ValueError(f'Missing runs: {expected - set(runs)}')

def values(data, mode):
    return {k: data['average_precision'] if k == 'average_precision' else data[mode][k] for k in metrics}

def write_csv(name, rows):
    with (args.output_dir / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

per_run, aggregate, paired, delta_summary = [], [], [], []
for model in models:
    for strategy in strategies:
        for mode in ['default', 'tuned']:
            for seed in seeds:
                run, d = runs[model,strategy,seed]
                per_run.append(dict(model=model,strategy=strategy,seed=seed,mode=mode,run=run,selected_epoch=d['selected_epoch'],threshold=d[mode]['threshold'],**values(d,mode)))
            row = dict(model=model,strategy=strategy,mode=mode,n=3)
            for metric in metrics:
                v = [100 * values(runs[model,strategy,n][1], mode)[metric] for n in seeds]
                row[metric+'_mean_pct'] = st.mean(v)
                row[metric+'_sd_pp'] = st.stdev(v)
            aggregate.append(row)
        # Both operating points compared with the same tuned unweighted reference.
        for mode in ['default', 'tuned']:
            for seed in seeds:
                current = values(runs[model,strategy,seed][1],mode)
                reference = values(runs[model,'unweighted',seed][1],'tuned')
                paired.append(dict(model=model,strategy=strategy,mode=mode,seed=seed,**{k+'_delta_pp':100*(current[k]-reference[k]) for k in metrics}))
            row = dict(model=model,strategy=strategy,mode=mode,n=3)
            subset = paired[-3:]
            for k in metrics:
                v = [r[k+'_delta_pp'] for r in subset]
                row[k+'_delta_mean_pp'] = st.mean(v)
                row[k+'_delta_sd_pp'] = st.stdev(v)
            delta_summary.append(row)
args.output_dir.mkdir(parents=True,exist_ok=True)
write_csv('per_run.csv',per_run)
write_csv('aggregate.csv',aggregate)
write_csv('paired_deltas.csv',paired)
write_csv('delta_summary.csv',delta_summary)
lines = ['# Preliminary validation results', '', 'All values below are percentages, reported as mean ± sample standard deviation across seeds 42, 123, and 999 (n=3). Average precision is also multiplied by 100. These are validation results: checkpoints and tuned thresholds were selected on this same split, so these are not unbiased test estimates. AP is threshold-independent and is not trapezoidal PR-AUC.', '']
for mode in ['default','tuned']:
    lines += [f'## {mode.capitalize()} thresholds', '', '| Model | Strategy | Macro-F1 | Hate precision | Hate recall | FPR | AP |', '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregate:
        if r['mode'] == mode:
            cells = [f"{r[k+'_mean_pct']:.2f} ± {r[k+'_sd_pp']:.2f}" for k in metrics]
            lines.append('| '+ ' | '.join([r['model'],r['strategy']]+cells)+' |')
    lines.append('')
lines += ['## Tuned strategies minus tuned unweighted baseline', '', 'Differences are paired by model and seed; mean ± sample standard deviation, in percentage points. Positive FPR differences mean more false positives. Three seeds describe variability but do not establish statistical significance.', '', '| Model | Strategy | Δ macro-F1 | Δ hate recall | Δ FPR | Δ AP |','|---|---|---:|---:|---:|---:|']
for r in delta_summary:
    if r['mode'] == 'tuned' and r['strategy'] != 'unweighted':
        cells = [f"{r[k+'_delta_mean_pp']:+.2f} ± {r[k+'_delta_sd_pp']:.2f}" for k in ['macro_f1','hate_recall','false_positive_rate','average_precision']]
        lines.append('| '+' | '.join([r['model'],r['strategy']]+cells)+' |')
(args.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
print(f'\nVerified 18 unique runs. Saved five summary files to {args.output_dir}')
