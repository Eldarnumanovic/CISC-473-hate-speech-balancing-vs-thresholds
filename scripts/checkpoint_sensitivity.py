"""Validation-only checkpoint sensitivity; never reads test data or changes runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
def average_precision_score(y, scores):
    # Integrate precision over recall increments at distinct descending scores.
    order = np.argsort(-scores, kind="stable")
    ranked_y, ranked_scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked_scores)), len(y)-1]
    tp = np.cumsum(ranked_y)[ends]
    precision = tp / (ends + 1)
    return np.sum(np.diff(np.r_[0, tp]) * precision) / y.sum()


METRICS = ['macro_f1', 'hate_precision', 'hate_recall', 'false_positive_rate', 'average_precision']

def evaluate(y, s, t):
    pred = s >= t
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    return dict(threshold=float(t), macro_f1=(2*tp/(2*tp+fp+fn)+2*tn/(2*tn+fp+fn))/2,
                hate_precision=tp/(tp+fp) if tp+fp else 0., hate_recall=tp/(tp+fn),
                false_positive_rate=fp/(fp+tn), tn=tn,fp=fp,fn=fn,tp=tp)

def tune(y,s):
    rows=[evaluate(y,s,t) for t in np.linspace(0,1,1001)]
    best=max(r['macro_f1'] for r in rows)
    return min((r for r in rows if abs(r['macro_f1']-best)<=1e-12),key=lambda r:(abs(r['threshold']-.5),r['threshold']))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--output-dir',type=Path,default=Path('results/checkpoint_sensitivity'))
    a=p.parse_args()
    records=[]; manifest=[]; seen=set(); shared=None
    for path in sorted(a.results_dir.glob('*/threshold_summary.json')):
        config_path=path.parent/'config.json'
        config=json.loads(config_path.read_text())
        if config.get('smoke_test'): continue
        model={'google-bert/bert-base-uncased':'bert','GroNLP/hateBERT':'hatebert'}[config['model']]
        strategy=config['strategy']; seed=config['seed']; key=(model,strategy,seed)
        if key in seen: raise ValueError(f'Duplicate configuration: {key}')
        seen.add(key)
        original=json.loads(path.read_text()); assert original['split']=='validation'
        print(f'Checking {path.parent.name}', flush=True)
        epochs=[]
        inputs=[path, config_path, path.parent/'metrics.json',path.parent/'best_epoch.json']
        history=json.loads(inputs[2].read_text())
        assert len(history)==3
        for h in history:
            e=h['epoch']; f=path.parent/f'validation_epoch{e}.csv'; inputs.append(f)
            d=pd.read_csv(f); y=d.label.to_numpy(dtype=int); s=d.hate_score.to_numpy()
            assert len(y)==1000 and int(y.sum())==427 and set(y)=={0,1}
            assert np.array_equal(d.validation_index.to_numpy(),np.arange(1000))
            if shared is None: shared=y.copy()
            assert np.array_equal(y,shared)
            assert np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
            ap=float(average_precision_score(y,s)); default=evaluate(y,s,.5)
            assert abs(ap-h['average_precision'])<1e-10
            assert abs(default['macro_f1']-h['macro_f1'])<1e-10
            epochs.append(dict(epoch=e,y=y,s=s,ap=ap,default=default))
        by_f1=max(epochs,key=lambda r:(r['default']['macro_f1'],-r['epoch']))
        by_ap=max(epochs,key=lambda r:(r['ap'],-r['epoch']))
        assert by_f1['epoch']==original['selected_epoch']==json.loads(inputs[3].read_text())['epoch']
        for rule,e in [('f1_at_05',by_f1),('average_precision',by_ap)]:
            tuned=tune(e['y'],e['s'])
            if rule=='f1_at_05':
                for k in ['threshold']+METRICS[:-1]: assert abs(tuned[k]-original['tuned'][k])<1e-10,(path,k)
            for mode,m in [('default',e['default']),('tuned',tuned)]:
                records.append(dict(model=model,strategy=strategy,seed=seed,rule=rule,mode=mode,run=path.parent.name,epoch=e['epoch'],average_precision=e['ap'],**m))
        manifest.extend(dict(path=str(f.relative_to(a.results_dir)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in inputs)
    expected={(m,s,n) for m in ['bert','hatebert'] for s in ['unweighted','weighted','oversampling'] for n in [42,123,999]}
    assert seen==expected, f'Missing or extra configurations: {seen ^ expected}'
    df=pd.DataFrame(records)
    agg=df.groupby(['model','strategy','rule','mode'])[METRICS].agg(['mean','std'])*100
    agg.columns=['_'.join(c)+'_pp' for c in agg.columns]
    paired=[]
    for row in records:
        if row['mode']!='tuned': continue
        base=df[(df.model==row['model'])&(df.strategy=='unweighted')&(df.seed==row['seed'])&(df.rule==row['rule'])&(df['mode']=='tuned')].iloc[0]
        paired.append({**{k:row[k] for k in ['model','strategy','seed','rule']},**{k:100*(row[k]-base[k]) for k in METRICS}})
    pairs=pd.DataFrame(paired)
    delta=pairs.groupby(['model','strategy','rule'])[METRICS].agg(['mean','std'])
    delta.columns=['_'.join(c)+'_pp' for c in delta.columns]
    selection=df[df['mode']=='tuned'].pivot(index=['model','strategy','seed'],columns='rule',values='epoch').reset_index()
    selection['changed']=selection.f1_at_05!=selection.average_precision
    lines=['# Validation checkpoint-selection sensitivity','',
    'Original rule: maximum validation macro-F1 at threshold 0.5. Alternative: maximum validation average precision. Exact epoch ties favor the earlier epoch. Both rules tune thresholds on the selected epoch using the original grid and tie rules. Values are mean ± sample SD across three seeds, expressed in percent. This is a validation sensitivity analysis, not a test evaluation or a change to the frozen primary checkpoints.','',
    f"AP selection changes the epoch in {selection.changed.sum()} of 18 runs.",'',
    '| Model | Strategy | Selection | Tuned macro-F1 | AP | Δ tuned F1 vs matched unweighted |','|---|---|---|---:|---:|---:|']
    for m in ['bert','hatebert']:
        for s in ['unweighted','weighted','oversampling']:
            for rule in ['f1_at_05','average_precision']:
                r=agg.loc[m,s,rule,'tuned']; z=delta.loc[m,s,rule]
                lines.append(f"| {m} | {s} | {rule} | {r.macro_f1_mean_pp:.2f} ± {r.macro_f1_std_pp:.2f} | {r.average_precision_mean_pp:.2f} ± {r.average_precision_std_pp:.2f} | {z.macro_f1_mean_pp:+.2f} ± {z.macro_f1_std_pp:.2f} |")
    lines += ['', 'Increasing validation AP under AP selection is expected by construction. Neither selection rule provides unbiased estimates on the validation set used to select it. Saved predictions permit this comparison but cannot recreate unsaved epoch weights for test or perturbation evaluation.']
    a.output_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(a.output_dir/'per_run.csv',index=False)
    agg.to_csv(a.output_dir/'aggregate.csv')
    pairs.to_csv(a.output_dir/'paired_deltas.csv',index=False)
    delta.to_csv(a.output_dir/'delta_summary.csv')
    selection.to_csv(a.output_dir/'epoch_selection.csv',index=False)
    (a.output_dir/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (a.output_dir/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines)); print('\nVerified original selections, logged metrics, thresholds, and aligned validation labels for all 18 runs.')

if __name__=='__main__': main()
