"""Select validation candidates without model predictions; never loads test data."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

def write_csv(path,fields,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output-dir',type=Path,default=Path('results/robustness_pilot'))
    p.add_argument('--seed',type=int,default=473)
    p.add_argument('--validation-arrow',type=Path,help='Optional local cached validation Arrow file')
    a=p.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        raise SystemExit('Output directory is not empty; refusing to overwrite annotations.')
    if a.validation_arrow:
        if a.validation_arrow.name!='tweet_eval-validation.arrow':
            raise ValueError('Only the cached TweetEval validation file is accepted.')
        import pyarrow as pa
        with pa.memory_map(str(a.validation_arrow),'r') as f:
            data=pa.ipc.open_stream(f).read_all().to_pylist()
        source='Local TweetEval-hate validation Arrow cache'
    else:
        from datasets import load_dataset
        data=list(load_dataset('cardiffnlp/tweet_eval','hate',split='validation'))
        source='cardiffnlp/tweet_eval, hate, validation'
    assert len(data)==1000 and sum(r['label'] for r in data)==427
    rng=random.Random(a.seed)
    candidates=[]; selected=[]
    for label in [0,1]:
        ids=[i for i,r in enumerate(data) if r['label']==label];rng.shuffle(ids)
        for rank,i in enumerate(ids,1):
            row=dict(example_id=f'validation_{i:04d}',validation_index=i,label=label,class_rank=rank,original_text=data[i]['text'])
            candidates.append(row)
            if rank<=10: selected.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    base=['example_id','validation_index','label','class_rank','original_text']
    write_csv(a.output_dir/'candidate_order.csv',base,candidates)
    edit_fields=base+['modified_text','original_word','replacement_word','edit_notes']
    write_csv(a.output_dir/'edits.csv',edit_fields,[dict(r,modified_text='',original_word='',replacement_word='',edit_notes='') for r in selected])
    review_fields=['example_id','meaning_preserved','label_preserved','target_preserved','natural_wording','one_substitution','decision','notes']
    for reviewer in ['eldar','lucas']:
        write_csv(a.output_dir/f'review_{reviewer}.csv',review_fields,[{k:r['example_id'] if k=='example_id' else '' for k in review_fields} for r in selected])
    final_fields=['example_id','final_decision','exclusion_reason','adjudication_notes']
    write_csv(a.output_dir/'decisions.csv',final_fields,[{k:r['example_id'] if k=='example_id' else '' for k in final_fields} for r in selected])
    payload=json.dumps([{'text':r['text'],'label':r['label']} for r in data],ensure_ascii=False,separators=(',',':'))
    (a.output_dir/'manifest.json').write_text(json.dumps(dict(dataset='cardiffnlp/tweet_eval',configuration='hate',split='validation',source=source,selection_seed=a.seed,candidates_per_class=10,selected_ids=[r['example_id'] for r in selected],validation_sha256=hashlib.sha256(payload.encode()).hexdigest(),status='unreviewed pilot candidates',selection_uses_predictions=False),indent=2))
    (a.output_dir/'INSTRUCTIONS.txt').write_text('''VALIDATION PILOT — DRAFT PROTOCOL
These are 20 candidate pairs, not 20 verified perturbations. No test data or model predictions were used in selection.

1. Keep original text, labels and IDs unchanged. In edits.csv, replace one word occurrence with a synonym. Record the original word, replacement, and complete modified text. Preserve all other wording, punctuation and capitalization as far as possible. Avoid changing identity targets, negation, quoted status, slurs, or intensity. If a natural equivalent is unavailable, leave the edit blank and record why.
2. Freeze the proposed edits before review. Eldar and Lucas review the same edits independently in their own review CSV, without seeing model predictions or the other reviewer's decisions. Enter yes/no/uncertain in each criterion and accept/reject/uncertain for decision. A proposer reviewing their own edit is not a blinded reviewer; record who proposed each edit in edit_notes.
3. Preserve meaning, target, naturalness, and the supplied hate/non-hate label. Flag ambiguous original labels instead of silently relabeling them. Reject uncertain or invalid edits.
4. After independent review, discuss disagreements and record the outcome in decisions.csv. final_decision is accept or reject. Record reasons for all rejected pairs. Accept only after both reviewers agree all criteria are satisfied.
5. Pilot failures are informative: do not replace them silently to reach 20 accepted pairs. candidate_order.csv records the full seeded order for additional validation candidates if needed. Preserve exclusions and document any subsequent pilot round separately.
6. Do not score edited text until reviews are complete. Use the same accepted pairs for every configuration, with existing checkpoints and thresholds fixed. Report acceptance counts by class, class-specific prediction flips, correct-to-incorrect and incorrect-to-correct transitions, and paired metric changes.
7. This pilot may inform protocol revisions. Freeze and commit the final protocol before constructing or evaluating test perturbations. Keep these validation pairs separate from the eventual 200 test pairs.

The 200-pair final target is conditional on enough valid pairs. Report exclusions and actual sample size rather than weakening validity rules to meet a quota. A synonym-editable subset may be unrepresentative of the full dataset.
''')
    print(f'Created 20 unreviewed validation candidates (10 per class) in {a.output_dir}.')

if __name__=='__main__': main()
