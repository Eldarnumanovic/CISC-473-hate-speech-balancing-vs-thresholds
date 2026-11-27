"""Record existing primary selections without reading test data or changing models."""
import argparse,csv,hashlib,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--results-dir',type=Path,default=Path('results'))
 p.add_argument('--checkpoints-dir',type=Path,default=Path('checkpoints'))
 p.add_argument('--output-dir',type=Path,default=Path('protocols/final_evaluation'))
 a=p.parse_args();rows=[];seen=set();inputs={}
 for f in sorted(a.results_dir.glob('*/threshold_summary.json')):
  cp=f.parent/'config.json';cfg=json.loads(cp.read_text())
  if cfg.get('smoke_test'):continue
  model={'google-bert/bert-base-uncased':'bert','GroNLP/hateBERT':'hatebert'}[cfg['model']]
  key=(model,cfg['strategy'],cfg['seed']);assert key not in seen;seen.add(key)
  s=json.loads(f.read_text());bp=f.parent/'best_epoch.json';best=json.loads(bp.read_text())
  assert s['split']=='validation' and s['selected_epoch']==best['epoch']
  ck=a.checkpoints_dir/f.parent.name/'best';assert (ck/'model.safetensors').is_file()
  for path in [f,cp,bp]:inputs[f'{f.parent.name}/{path.name}']=hashlib.sha256(path.read_bytes()).hexdigest()
  rows.append(dict(model=model,strategy=cfg['strategy'],seed=cfg['seed'],run=f.parent.name,selected_epoch=best['epoch'],default_threshold=s['default']['threshold'],tuned_threshold=s['tuned']['threshold'],max_length=cfg['max_length'],checkpoint=str(Path('checkpoints')/f.parent.name/'best')))
 expected={(m,s,n) for m in ['bert','hatebert'] for s in ['unweighted','weighted','oversampling'] for n in [42,123,999]}
 assert seen==expected
 a.output_dir.mkdir(parents=True,exist_ok=True)
 target=a.output_dir/'frozen_configurations.json'
 if target.exists():raise SystemExit('Manifest exists; refusing to overwrite the recorded configuration.')
 target.write_text(json.dumps(dict(checkpoint_rule='validation macro-F1 at 0.5',threshold_rule='existing validation-tuned threshold; no retuning',source_hashes=inputs,runs=rows,note='Paths identify existing checkpoints; this manifest does not hash model weights or assert whether test data were previously inspected.'),indent=2))
 with (a.output_dir/'frozen_configurations.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('Recorded 18 existing checkpoint selections and thresholds. No test data loaded. Commit the manifest and protocol before test annotation.')
if __name__=='__main__':main()
