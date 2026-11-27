"""Evaluate jointly reviewed VALIDATION pairs with existing F1-selected checkpoints.
No training, threshold tuning, test loading, or independent-review requirement.
"""
import argparse
import csv
import gc
import hashlib
import json
from datetime import datetime
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(y, pred):
    tp=sum(a==1 and b==1 for a,b in zip(y,pred));tn=sum(a==0 and b==0 for a,b in zip(y,pred))
    fp=sum(a==0 and b==1 for a,b in zip(y,pred));fn=sum(a==1 and b==0 for a,b in zip(y,pred))
    div=lambda a,b: a/b if b else 0.0
    return dict(macro_f1=(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))/2,
                hate_precision=div(tp,tp+fp),hate_recall=div(tp,tp+fn),false_positive_rate=div(fp,fp+tn),tn=tn,fp=fp,fn=fn,tp=tp)


def transitions(y, original, modified):
    rows=[]
    for name,label in [('all',None),('non_hate',0),('hate',1)]:
        ids=[i for i,a in enumerate(y) if label is None or a==label]
        cc=sum(original[i]==y[i] and modified[i]==y[i] for i in ids)
        ci=sum(original[i]==y[i] and modified[i]!=y[i] for i in ids)
        ic=sum(original[i]!=y[i] and modified[i]==y[i] for i in ids)
        ii=sum(original[i]!=y[i] and modified[i]!=y[i] for i in ids)
        n=len(ids);assert cc+ci+ic+ii==n
        flips=sum(original[i]!=modified[i] for i in ids);assert flips==ci+ic
        rows.append(dict(group=name,n=n,flips=flips,flip_rate=flips/n if n else None,
                         correct_to_correct=cc,correct_to_incorrect=ci,incorrect_to_correct=ic,incorrect_to_incorrect=ii,
                         originally_correct=cc+ci,originally_incorrect=ic+ii,
                         break_rate=ci/(cc+ci) if cc+ci else None,
                         repair_rate=ic/(ic+ii) if ic+ii else None))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pilot-dir',type=Path,default=Path('results/robustness_pilot'))
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--checkpoints-dir',type=Path,default=Path('checkpoints'))
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='Evaluate the first model only; excludes results from full-run aggregates.')
    a=p.parse_args()
    edits=read_csv(a.pilot_dir/'edits.csv'); decisions=read_csv(a.pilot_dir/'decisions.csv')
    ed={r['example_id']:r for r in edits};de={r['example_id']:r for r in decisions}
    assert len(ed)==len(edits)==len(de)==len(decisions)==20 and set(ed)==set(de),'Expected the same 20 unique candidate IDs.'
    manifest=json.loads((a.pilot_dir/'manifest.json').read_text())
    assert manifest['split']=='validation' and set(manifest['selected_ids'])==set(ed)
    accepted=[]
    for key,e in ed.items():
        assert key==f"validation_{int(e['validation_index']):04d}"
        d=de[key]['final_decision'].strip().lower();assert d in ['accept','reject'],f'Incomplete decision: {key}'
        if d=='reject':
            assert de[key]['exclusion_reason'].strip(),f'Missing exclusion reason: {key}'
            continue
        old=e['original_word'];new=e['replacement_word'];s=e['original_text'];t=e['modified_text']
        assert old and new and old!=new and len(old.split())==len(new.split())==1
        assert any(t==s[:i]+new+s[i+len(old):] for i in range(len(s)) if s.startswith(old,i)),f'Not one exact substitution: {key}'
        accepted.append(e)
    y=[int(e['label']) for e in accepted];assert set(y)=={0,1}
    runs=[];seen=set()
    for f in sorted(a.results_dir.glob('*/threshold_summary.json')):
        config=json.loads((f.parent/'config.json').read_text())
        if config.get('smoke_test'): continue
        model={'google-bert/bert-base-uncased':'bert','GroNLP/hateBERT':'hatebert'}[config['model']]
        key=(model,config['strategy'],config['seed']);assert key not in seen;seen.add(key)
        summary=json.loads(f.read_text());best=json.loads((f.parent/'best_epoch.json').read_text())
        assert summary['split']=='validation' and summary['selected_epoch']==best['epoch']
        checkpoint=a.checkpoints_dir/f.parent.name/'best'
        assert (checkpoint/'model.safetensors').is_file() and (checkpoint/'config.json').is_file(),f'Missing checkpoint: {checkpoint}'
        predictions={int(r['validation_index']):r for r in read_csv(f.parent/f"validation_epoch{best['epoch']}.csv")}
        for e in accepted: assert int(predictions[int(e['validation_index'])]['label'])==int(e['label'])
        runs.append((f.parent,config,model,summary,checkpoint,predictions))
    expected={(m,s,n) for m in ['bert','hatebert'] for s in ['unweighted','weighted','oversampling'] for n in [42,123,999]}
    assert seen==expected,'Missing or extra experiment configurations.'
    print(f'Validated {len(accepted)} accepted pairs: {y.count(0)} non-hate, {y.count(1)} hate; all 18 checkpoints present.',flush=True)
    if a.check_only: return
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable; this protocol uses the original MPS device.')
    device=torch.device('mps')
    out=a.output_dir or a.results_dir/f"{'smoke_pilot' if a.smoke else 'robustness_pilot_eval'}_{datetime.now():%Y%m%d_%H%M%S}"
    if out.exists(): raise ValueError('Output directory already exists; refusing to overwrite.')
    out.mkdir(parents=True)
    provenance={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [a.pilot_dir/'edits.csv',a.pilot_dir/'decisions.csv',a.pilot_dir/'manifest.json']}
    (out/'evaluation_config.json').write_text(json.dumps(dict(split='validation',smoke=a.smoke,checkpoint_rule='validation macro-F1 at 0.5',review_method='joint in-person consensus; no independent reviewer agreement measured',accepted_ids=[e['example_id'] for e in accepted],input_hashes=provenance,thresholds='existing default and validation-tuned; no retuning',original_score_tolerance=1e-4),indent=2))
    scores_rows=[];metric_rows=[];transition_rows=[]
    for folder,config,name,summary,checkpoint,previous in (runs[:1] if a.smoke else runs):
        print(f'Loading {folder.name}',flush=True)
        tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True,attn_implementation='eager').to(device)
        model.eval()
        def score(texts):
            result=[]
            with torch.inference_mode():
                for text in texts:
                    batch=tokenizer(text,padding=True,truncation=True,max_length=config['max_length'],return_tensors='pt')
                    result.append(float(model(**{k:v.to(device) for k,v in batch.items()}).logits.softmax(-1)[0,1].cpu()))
            return result
        original=score([e['original_text'] for e in accepted]);modified=score([e['modified_text'] for e in accepted])
        gap=max(abs(s-float(previous[int(e['validation_index'])]['hate_score'])) for e,s in zip(accepted,original))
        print(f'Original-score maximum difference from saved predictions: {gap:.8f}',flush=True)
        if gap>1e-4: raise RuntimeError('Original predictions do not reproduce within tolerance. Investigate before interpreting robustness results.')
        common=dict(model=name,strategy=config['strategy'],seed=config['seed'],run=folder.name,epoch=summary['selected_epoch'])
        for e,s,t in zip(accepted,original,modified):scores_rows.append(dict(**common,example_id=e['example_id'],label=int(e['label']),original_score=s,modified_score=t))
        for mode in ['default','tuned']:
            threshold=summary[mode]['threshold'];op=[int(s>=threshold) for s in original];mp=[int(s>=threshold) for s in modified]
            baseline=[int(float(previous[int(e['validation_index'])]['hate_score'])>=threshold) for e in accepted]
            if op!=baseline: raise RuntimeError('Original labels changed at a fixed threshold; investigate numerical reproducibility.')
            before=metrics(y,op);after=metrics(y,mp)
            metric_rows.append(dict(**common,mode=mode,threshold=threshold,n=len(y),**{f'original_{k}':v for k,v in before.items()},**{f'modified_{k}':v for k,v in after.items()},**{f'delta_{k}_pp':100*(after[k]-before[k]) for k in ['macro_f1','hate_precision','hate_recall','false_positive_rate']}))
            for row in transitions(y,op,mp):transition_rows.append(dict(**common,mode=mode,threshold=threshold,**row))
        write_csv(out/'scores.csv',scores_rows);write_csv(out/'metrics.csv',metric_rows);write_csv(out/'transitions.csv',transition_rows)
        del model,tokenizer;gc.collect();torch.mps.empty_cache()
        print(f'Completed {folder.name}',flush=True)
    (out/'COMPLETE.json').write_text(json.dumps(dict(completed_runs=1 if a.smoke else 18,accepted_pairs=len(y),smoke=a.smoke),indent=2))
    print(f'Finished. Saved paired scores, metrics, and class-specific transitions in {out}',flush=True)

if __name__=='__main__': main()
