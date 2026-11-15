"""Bounded orchestration. All fits precede outer evaluation; no prior test scoring."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,json,hashlib,argparse,subprocess,warnings,datetime,pickle
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix,brier_score_loss
from paths import data_path, output_root
OUT=None
from prepared_data import feature_names,TrainPreprocessor
from modified_hdc import fit_nn,fit_tree,tree_predict,predict,role_seed,write
PROTOCOL=json.loads(Path(__file__).with_name('protocol.json').read_text())

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def prepare():
 assert not (OUT/'frozen_manifest.json').exists(),'Already frozen; do not overwrite'
 OUT.mkdir(parents=True,exist_ok=True)
 raw=pd.read_csv(data_path(),low_memory=False)
 assert raw.VisitLink.notna().all() and set(raw.SMM.unique())=={0,1}
 patients=raw.groupby('VisitLink',sort=True).SMM.max()
 train_ids,test_ids=train_test_split(patients.index.to_numpy(),test_size=.2,random_state=53,stratify=patients.to_numpy())
 dev_idx=np.flatnonzero(raw.VisitLink.isin(train_ids));prior_test=set(test_ids)
 d=raw.iloc[dev_idx];g=d.groupby('VisitLink',sort=True).SMM.max();assert not set(g.index)&prior_test
 folds=StratifiedKFold(3,shuffle=True,random_state=53);cat,num=feature_names();allmembers=[];summary=[]
 for fold,(dev,outer) in enumerate(folds.split(g.index,g.to_numpy()),1):
  dest=OUT/f'fold_{fold}';dest.mkdir(exist_ok=False);ids=g.index.to_numpy()[dev]
  fit_ids,hold_ids=train_test_split(ids,test_size=.3,random_state=5300+fold,stratify=g.loc[ids])
  sel_ids,prune_ids=train_test_split(hold_ids,test_size=.5,random_state=5400+fold,stratify=g.loc[hold_ids])
  parts={'fit':fit_ids,'selection':sel_ids,'pruning':prune_ids,'outer':g.index.to_numpy()[outer]}
  indices={k:dev_idx[d.VisitLink.isin(v)] for k,v in parts.items()}
  assert len(np.unique(np.concatenate(list(indices.values()))))==len(d)
  prep=TrainPreprocessor(cat,num).fit(raw.iloc[indices['fit']]);frames={};stats={}
  for name,idx in indices.items():
   frame=prep.transform(raw.iloc[idx]);frame['SMM']=raw.iloc[idx].SMM.to_numpy();frame.attrs['patient_ids']=raw.iloc[idx].VisitLink.to_numpy();frame.attrs['source_rows']=idx;frames[name]=frame
   stats[name]={'n':len(idx),'patients':len(parts[name]),'smm':int(frame.SMM.sum())}
   allmembers.append(pd.DataFrame({'fold':fold,'partition':name,'source_row_0based':idx}))
  for i,a in enumerate(parts):
   for b in list(parts)[i+1:]:assert not set(parts[a])&set(parts[b])
  joblib.dump(prep,dest/'preprocessor.joblib');joblib.dump({k:v for k,v in frames.items() if k!='outer'},dest/'development.joblib');joblib.dump(frames['outer'],dest/'outer.joblib')
  write(dest/'partitions.json',stats);summary.append({'fold':fold,**stats})
 pd.concat(allmembers).to_csv(OUT/'partition_membership.csv',index=False)
 write(OUT/'partition_summary.json',summary)
 frozen={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')};frozen['protocol.json']=sha(Path(__file__).with_name('protocol.json'));frozen['partition_membership.csv']=sha(OUT/'partition_membership.csv')
 write(OUT/'frozen_manifest.json',{'frozen_at':datetime.datetime.now().astimezone().isoformat(),'sha256':frozen,'development_records':len(d),'development_patients':len(g),'prior_test_records_excluded':len(raw)-len(d),'data_sha256':sha(data_path())})
 print(json.dumps(summary,indent=2),flush=True)

def fit_fold(fold):
 dest=OUT/f'fold_{fold}';d=joblib.load(dest/'development.joblib');tr=d['fit'];sel=d['selection'];pr=d['pruning'];prep=joblib.load(dest/'preprocessor.joblib');stats={};started=time.time()
 for name in ['LR','RF']:
  p=dest/name;p.mkdir(exist_ok=False);t=time.time();m=LogisticRegression(**PROTOCOL[name]) if name=='LR' else RandomForestClassifier(**PROTOCOL[name])
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');m.fit(tr.drop(columns='SMM'),tr.SMM)
  if any(issubclass(w.category,ConvergenceWarning) for w in ws):raise RuntimeError('LR convergence failure')
  joblib.dump(m,p/'model.joblib');stats[name]={'seconds':time.time()-t,'warnings':[str(w.message) for w in ws]};write(p/'status.json',stats[name]);print(f'FOLD {fold} {name} complete',flush=True)
 t=time.time();m=fit_nn(tr,sel,prep.input_dim,dest/'NN',role_seed(fold,'root'));joblib.dump(m,dest/'NN/model.joblib');stats['NN']={'seconds':time.time()-t}
 t=time.time();tree=fit_tree(tr,sel,pr,prep.input_dim,m,dest/'HDC',fold)
 with (dest/'HDC/tree.pkl').open('wb') as f:pickle.dump(tree,f)
 def structure(n):return {'node':n.node_id,'children':[structure(c) for c in n.children]}
 write(dest/'HDC/retained_tree.json',structure(tree));stats['HDC']={'additional_seconds':time.time()-t,'root_reused_from_NN':True,'total_with_root_seconds':time.time()-t+stats['NN']['seconds']};stats['fold_seconds']=time.time()-started
 write(dest/'fit_complete.json',stats);print(f'FOLD {fold} ALL FITS COMPLETE',stats,flush=True)

def metrics(y,s,pred,w=None):
 tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1],sample_weight=w).ravel()
 return {'auc':float(roc_auc_score(y,s,sample_weight=w)),'ap':float(average_precision_score(y,s,sample_weight=w)),'brier':float(brier_score_loss(y,s,sample_weight=w)),'tp':float(tp),'fp':float(fp),'tn':float(tn),'fn':float(fn),'sensitivity':float(tp/(tp+fn)),'ppv':float(tp/(tp+fp)) if tp+fp else None,'positive_rate':float((tp+fp)/(tp+fp+tn+fn)),'f1':float(2*tp/(2*tp+fp+fn)),'accuracy':float((tp+tn)/(tp+fp+tn+fn))}

def evaluate():
 assert not (OUT/'fold_results.csv').exists(), 'Evaluation already exists; use a new output root'
 assert all((OUT/f'fold_{i}/fit_complete.json').exists() for i in [1,2,3])
 rows=[];boots=[]
 for fold in [1,2,3]:
  dest=OUT/f'fold_{fold}';dev=joblib.load(dest/'development.joblib');outer=joblib.load(dest/'outer.joblib');data={**dev,'outer':outer};scoremap={};thresholds={}
  for name in ['LR','RF','NN','HDC']:
   p=dest/name
   if name=='HDC':
    with (p/'tree.pkl').open('rb') as f:m=pickle.load(f)
   else:m=joblib.load(p/'model.joblib')
   scores={}
   for part,frame in data.items():
    x=frame.drop(columns='SMM');s=tree_predict(m,x) if name=='HDC' else predict(m,x) if name=='NN' else m.predict_proba(x)[:,1]
    assert len(s)==len(frame) and np.isfinite(s).all() and ((s>=0)&(s<=1)).all();scores[part]=s
    pd.DataFrame({'source_row_0based':frame.attrs['source_rows'],'smm':frame.SMM.to_numpy(),'score':s}).to_csv(p/f'{part}_scores.csv',index=False,float_format='%.17g')
   scoremap[name]=scores['outer'];y=outer.SMM.to_numpy();s=scores['outer'];v=scores['pruning']
   result={'outer_default':metrics(y,s,s>=.5),'development':{part:metrics(data[part].SMM,scores[part],scores[part]>=.5) for part in dev},'operating_points':[]}
   for rate in [.01,.05]:
    t=float(np.quantile(v,1-rate));z=metrics(y,s,s>=t);z.update({'target_pruning_rate':rate,'threshold':t,'achieved_pruning_rate':float((v>=t).mean()),'outer_threshold_ties':int((s==t).sum())});result['operating_points'].append(z)
    if rate==.01:thresholds[name]=t
   write(p/'evaluation.json',result)
   a=result['outer_default'];b=result['operating_points'][0];c=result['operating_points'][1]
   rows.append({'fold':fold,'model':name,**{k:a[k] for k in ['auc','ap','brier']},'sensitivity_1pct':b['sensitivity'],'ppv_1pct':b['ppv'],'positive_rate_1pct':b['positive_rate'],'sensitivity_5pct':c['sensitivity'],'ppv_5pct':c['ppv'],'positive_rate_5pct':c['positive_rate']})
  codes,ids=pd.factorize(outer.attrs['patient_ids']);rng=np.random.default_rng(53000+fold)
  for rep in range(200):
   weights=np.bincount(rng.integers(len(ids),size=len(ids)),minlength=len(ids))[codes]
   for name,s in scoremap.items():
    boots.append({'fold':fold,'replicate':rep,'model':name,'auc':roc_auc_score(y,s,sample_weight=weights),'ap':average_precision_score(y,s,sample_weight=weights),'sensitivity_1pct':float(np.sum(weights*y*(s>=thresholds[name]))/np.sum(weights*y))})
  print('Outer evaluation complete fold',fold,flush=True)
 pd.DataFrame(rows).to_csv(OUT/'fold_results.csv',index=False);pd.DataFrame(boots).to_csv(OUT/'paired_patient_bootstrap.csv',index=False)
 summary=pd.DataFrame(rows).groupby('model').mean(numeric_only=True).drop(columns='fold');summary.to_csv(OUT/'mean_results.csv')
 b=pd.DataFrame(boots).groupby(['replicate','model'])[['auc','ap','sensitivity_1pct']].mean();unc={}
 for name in ['LR','RF','NN']:
  delta=b.xs('HDC',level='model')-b.xs(name,level='model');unc['HDC_minus_'+name]={c:np.percentile(delta[c],[2.5,97.5]).tolist() for c in delta}
 write(OUT/'conditional_difference_intervals.json',unc);print(summary.to_string(),flush=True)

def main():
 global OUT
 parser=argparse.ArgumentParser(description='Final three-fold sensitivity comparison; each action is explicit.')
 action=parser.add_mutually_exclusive_group(required=True)
 action.add_argument('--prepare',action='store_true',help='Create frozen patient folds and fitting-only transforms; no model fitting')
 action.add_argument('--fit-fold',type=int,choices=[1,2,3],help='Fit one fold; saves checkpoints and logs; does not evaluate outer records')
 action.add_argument('--evaluate',action='store_true',help='Evaluate only after all three folds have completed fitting')
 a=parser.parse_args()
 OUT=output_root()/'comparison'
 if a.prepare: prepare()
 elif a.fit_fold: fit_fold(a.fit_fold)
 else: evaluate()
if __name__=='__main__': main()
