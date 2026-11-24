"""Plot validation PR curves for both checkpoint-selection rules; no test access."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

STRATEGIES = [('unweighted', 'Unweighted', '#0072B2'), ('weighted', 'Weighted', '#D55E00'), ('oversampling', 'Oversampling', '#009E73')]
MODELS = [('bert', 'BERT'), ('hatebert', 'HateBERT')]
SEEDS = [42, 123, 999]

def curve(y, scores):
    order = np.argsort(-scores, kind='stable')
    yy, ss = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(y)-1]
    tp = np.cumsum(yy)[ends]
    precision = tp/(ends+1)
    recall = tp/y.sum()
    ap = float(np.sum(np.diff(np.r_[0, recall])*precision))
    # Ascending recall, including the no-positive-predictions endpoint.
    return np.r_[0, recall], np.r_[1, precision], ap

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--sensitivity-dir',type=Path,default=Path('results/checkpoint_sensitivity'))
    p.add_argument('--output-dir',type=Path,default=Path('results/pr_comparisons'))
    p.add_argument('--formats', nargs='+', choices=['png','pdf'], default=['png','pdf'])
    a=p.parse_args()
    source=a.sensitivity_dir/'per_run.csv'
    with source.open() as f: rows=list(csv.DictReader(f))
    lookup={}
    for r in rows:
        key=(r['model'],r['strategy'],int(r['seed']),r['rule'],r['mode'])
        if key in lookup: raise ValueError(f'Duplicate row: {key}')
        lookup[key]=r
    expected={(m,s,n,rule,mode) for m,_ in MODELS for s,_,_ in STRATEGIES for n in SEEDS for rule in ['f1_at_05','average_precision'] for mode in ['default','tuned']}
    if set(lookup)!=expected: raise ValueError('Expected exactly 72 sensitivity rows.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    manifest={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    shared=None
    for rule,title in [('f1_at_05','Checkpoints selected by validation macro-F1 at threshold 0.5'),('average_precision','Checkpoints selected by validation average precision')]:
        fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
        for i,(model,label) in enumerate(MODELS):
            for j,seed in enumerate(SEEDS):
                ax=axes[i,j]
                for strategy,name,color in STRATEGIES:
                    default=lookup[model,strategy,seed,rule,'default']
                    tuned=lookup[model,strategy,seed,rule,'tuned']
                    assert default['epoch']==tuned['epoch'] and default['run']==tuned['run']
                    path=a.results_dir/default['run']/f"validation_epoch{default['epoch']}.csv"
                    with path.open() as f: data=list(csv.DictReader(f))
                    ids=[int(r['validation_index']) for r in data]
                    y=np.array([int(r['label']) for r in data]); scores=np.array([float(r['hate_score']) for r in data])
                    assert ids==list(range(1000)) and set(y)=={0,1} and y.sum()==427
                    assert np.isfinite(scores).all() and ((scores>=0)&(scores<=1)).all()
                    if shared is None: shared=y.copy()
                    assert np.array_equal(shared,y)
                    recall,precision,ap=curve(y,scores)
                    assert abs(ap-float(default['average_precision']))<1e-10
                    ax.step(recall,precision,where='pre',color=color,lw=1.5,label=f"{name}: AP={ap:.3f}, epoch {default['epoch']}")
                    for record,marker in [(default,'o'),(tuned,'s')]:
                        pred=scores>=float(record['threshold'])
                        tp=int(((y==1)&pred).sum()); fp=int(((y==0)&pred).sum())
                        pr=tp/(tp+fp) if tp+fp else 0; re=tp/y.sum()
                        assert abs(pr-float(record['hate_precision']))<1e-10
                        assert abs(re-float(record['hate_recall']))<1e-10
                        ax.scatter(re,pr,s=62,marker=marker,facecolors='none',edgecolors=color,linewidths=1.6,zorder=5)
                    manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                ax.axhline(.427,color='#777777',ls='--',lw=1)
                ax.set(title=f'{label} · seed {seed}',xlim=(0,1),ylim=(0,1.02))
                ax.grid(alpha=.18)
                ax.legend(loc='lower left',fontsize=8,framealpha=.95)
                if i==1: ax.set_xlabel('Hate-class recall')
                if j==0: ax.set_ylabel('Hate-class precision')
        fig.suptitle('Validation precision–recall comparison\n'+title,fontsize=15,y=.985)
        handles=[Line2D([],[],marker='o',color='black',markerfacecolor='none',ls='None',label='Default threshold: 0.5'),Line2D([],[],marker='s',color='black',markerfacecolor='none',ls='None',label='Validation-tuned threshold'),Line2D([],[],color='#777777',ls='--',label='Hate-class prevalence: 42.7%')]
        fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.005))
        fig.tight_layout(rect=(0,.055,1,.92))
        for suffix in a.formats: fig.savefig(a.output_dir/f'validation_pr_{rule}.{suffix}',dpi=200)
        plt.close(fig)
    (a.output_dir/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    (a.output_dir/'captions.txt').write_text('Validation precision–recall curves on the same 1,000 TweetEval-hate examples. Rows show models; columns show seeds. Colors identify training strategies. Circles mark threshold 0.5; squares mark validation-tuned thresholds. The dashed line is hate-class prevalence (42.7%). Legends report average precision (AP) and selected epoch. The f1_at_05 figure uses checkpoints selected by validation macro-F1 at 0.5; the average_precision figure uses checkpoints selected by validation AP. Curves show individual seeds, not averaged curves. All selection and tuning use validation data; these figures are not held-out test estimates.\n')
    print(f'Verified all 36 curves and 72 operating points. Saved two figures in the requested formats to {a.output_dir}.')

if __name__=='__main__': main()
