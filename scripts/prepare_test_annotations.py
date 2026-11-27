"""Prepare ordered test annotation candidates, without model inference."""
import argparse,csv,hashlib,json,random
from pathlib import Path
BASE=['example_id','test_index','label','class_rank','original_text']
EDIT=BASE+['modified_text','original_word','replacement_word','edit_notes']
DEC=['example_id','final_decision','exclusion_reason','adjudication_notes']
def write(path,fields,rows):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--output-dir',type=Path,default=Path('results/robustness_test_annotations'))
 p.add_argument('--protocol-dir',type=Path,default=Path('protocols/final_evaluation'))
 p.add_argument('--test-arrow',type=Path,help='Optional local TweetEval test Arrow cache')
 a=p.parse_args()
 if a.output_dir.exists() and any(a.output_dir.iterdir()):raise SystemExit('Refusing to overwrite annotation files.')
 protocol=a.protocol_dir/'evaluation_protocol.txt';frozen=a.protocol_dir/'frozen_configurations.json'
 assert protocol.is_file() and frozen.is_file(),'Copy and commit the final protocol first.'
 assert len(json.loads(frozen.read_text())['runs'])==18
 if a.test_arrow:
  assert a.test_arrow.name=='tweet_eval-test.arrow'
  import pyarrow as pa
  with pa.memory_map(str(a.test_arrow),'r') as f:data=pa.ipc.open_stream(f).read_all().to_pylist()
 else:
  from datasets import load_dataset
  data=list(load_dataset('cardiffnlp/tweet_eval','hate',split='test'))
 assert len(data)==2970 and sum(r['label'] for r in data)==1252
 rng=random.Random(473);candidates=[]
 for label in [0,1]:
  indices=[i for i,r in enumerate(data) if r['label']==label];rng.shuffle(indices)
  for rank,i in enumerate(indices,1):candidates.append(dict(example_id=f'test_{i:04d}',test_index=i,label=label,class_rank=rank,original_text=data[i]['text']))
 selected=[r for r in candidates if r['class_rank']<=100]
 a.output_dir.mkdir(parents=True)
 write(a.output_dir/'candidate_order.csv',BASE,candidates)
 write(a.output_dir/'edits.csv',EDIT,[{**r,**{k:'' for k in EDIT if k not in BASE}} for r in selected])
 write(a.output_dir/'decisions.csv',DEC,[{k:r['example_id'] if k=='example_id' else '' for k in DEC} for r in selected])
 payload=json.dumps([{'text':r['text'],'label':r['label']} for r in data],ensure_ascii=False,separators=(',',':'))
 manifest=dict(dataset='cardiffnlp/tweet_eval',configuration='hate',split='test',selection_seed=473,target_accepted_per_class=100,initial_candidates_per_class=100,selected_ids=[r['example_id'] for r in selected],test_sha256=hashlib.sha256(payload.encode()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),frozen_configurations_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),selection_uses_predictions=False,status='unreviewed candidates, not finalized evaluation pairs')
 (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2))
 (a.output_dir/'INSTRUCTIONS.txt').write_text('''TEST ANNOTATION — NO MODEL PREDICTIONS
Read the committed final evaluation protocol first. edits.csv contains the first 100 candidates in each class, not 200 accepted pairs.

Keep IDs, indices, class ranks, labels and original text unchanged. Work through each class in rank order. Propose one single-word synonym replacement and preserve all other characters. Record the proposer and AI assistance in edit_notes. Do not change hashtags, identity targets, negation, slurs, quoted text, tone or intensity. If no suitable edit is available, leave modification fields blank.

Jointly review each proposal without model predictions. In decisions.csv enter accept or reject; document exclusions and record joint consensus in adjudication_notes. Do not invent independent reviewer files. Reject unresolved uncertainty and ambiguous original labels. Retain rejected proposals and their reasons.

After reviewing the initial candidates, count accepted pairs separately by label. If a class has fewer than 100 accepted, continue from its next class_rank in candidate_order.csv. Append each additional candidate to edits.csv and decisions.csv, keeping all earlier attempts. Do not skip candidates based on convenience or predicted model behavior. Ask for the replenishment tool after this batch; do not manually select appealing reserve examples. Stop each class at its 100th accepted pair. Review all earlier ranks before finalizing. If a class is exhausted, report the shortfall; do not weaken the criteria.

Save as UTF-8 CSV; disable Numbers table-name export if possible. Before evaluation, validate IDs, exact changes, acceptance counts and sequential selection; record hashes of the finalized pairs and commit. No model scoring occurs during annotation. These test pairs are separate from the validation pilot.
''')
 print(f'Prepared 200 unreviewed test candidates (100 per class) and the complete ordered reserve in {a.output_dir}. No models loaded or predictions generated.')
if __name__=='__main__':main()
