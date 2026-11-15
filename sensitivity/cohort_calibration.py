"""Source-cohort Table S11 and Figure S2/calibration from saved scores. Diagnostic logistic fits do not retrain prediction models."""
if __name__ != "__main__":
    raise RuntimeError("Run this analysis script explicitly; importing it does not execute analyses.")
from pathlib import Path
import sys,json,hashlib,ast,time
import numpy as np,pandas as pd
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paths import data_path, output_root, new_output, feature_names
R=output_root(); O=new_output('cohort_calibration')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()
def save(n,obj): (O/n).write_text(json.dumps(obj,indent=2),encoding='utf-8')
cat,num=feature_names()
cols=cat+num+['SMM','VisitLink','DSHOSPID']
d=pd.read_csv(data_path(),usecols=cols,dtype=str,keep_default_na=False,low_memory=False)
groups={'Source cohort':np.arange(len(d))}
summary=[];cats=[];cont=[];missing=[]
na_tokens={'','NA','N/A','NaN','nan','NULL','null','None','<NA>','n/a','-NaN','-nan','#N/A','#N/A N/A','#NA','1.#IND','-1.#IND','1.#QNAN','-1.#QNAN'}
for group,idx in groups.items():
 z=d.iloc[idx]; y=pd.to_numeric(z.SMM)
 summary.append({'group':group,'records':len(z),'patients':z.VisitLink.nunique(),'hospitals':z.DSHOSPID.nunique(),'smm':int(y.sum()),'smm_percent':100*y.mean()})
 for c in cat:
  counts=z[c].value_counts(dropna=False)
  for level,n in counts.sort_index().items():cats.append({'group':group,'variable':c,'raw_category':level,'n':int(n),'percent':100*n/len(z)})
  for level,n in counts.items():
   if level in na_tokens or any(t in level.lower() for t in ['missing','unknown']) or level.startswith('.'):
    missing.append({'group':group,'variable':c,'raw_category':level,'n':int(n),'percent':100*n/len(z),'interpretation':'literal None means no prior encounters, not clinical missingness' if c=='Encounter_Cat_past30days' and level=='None' else 'raw missing/unknown/special token; substantive meaning not inferred'})
 for c in num:
  v=pd.to_numeric(z[c],errors='coerce');v=v[np.isfinite(v)]
  cont.append({'group':group,'variable':c,'valid_n':len(v),'nonfinite_n':len(z)-len(v),'median':v.median(),'q1':v.quantile(.25),'q3':v.quantile(.75),'mean':v.mean(),'sd':v.std(),'min':v.min(),'max':v.max()})
pd.DataFrame(summary).to_csv(O/'cohort_counts.csv',index=False)
pd.DataFrame(cats).to_csv(O/'categorical_all.csv',index=False)
pd.DataFrame(cont).to_csv(O/'continuous_all.csv',index=False)
pd.DataFrame(missing).to_csv(O/'missing_and_special_categories.csv',index=False)
selected=['YEAR','AGE_CAT','RACE_CAT','PAY_CAT','PRIMLANG_CAT','MARITAL_CAT','MEDINCSTQ_CAT','UR_CAT','Encounter_Cat_past30days','OBESITY_delivery','MULTIPLE_GESTATION_delivery','HYPERTENSIVE_delivery','HEMORRHAGE_delivery','GESTATIONAL_delivery','comorbidities_delivery']
a=pd.DataFrame(cats);a=a[a.variable.isin(selected)].copy();a['n_percent']=a.apply(lambda r:f"{r['n']:,} ({r['percent']:.2f}%)",axis=1)
a.pivot(index=['variable','raw_category'],columns='group',values='n_percent').fillna('0 (0.00%)').to_csv(O/'cohort_table_categorical.csv')
b=pd.DataFrame(cont);b['median_iqr']=b.apply(lambda r:f"{r['median']:.3g} [{r['q1']:.3g}, {r['q3']:.3g}]",axis=1);b.pivot(index='variable',columns='group',values='median_iqr').to_csv(O/'cohort_table_continuous.csv')
print(pd.DataFrame(summary).to_string(index=False),flush=True)

def logistic_diagnostic(y,p,eps):
 x=logit(np.clip(p,eps,1-eps));n=len(y)
 def fun(v):
  eta=v[0]+v[1]*x
  return np.mean(np.logaddexp(0,eta)-y*eta),np.array([np.mean(expit(eta)-y),np.mean((expit(eta)-y)*x)])
 fit=minimize(fun,[0,1],jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':500})
 citl=brentq(lambda intercept:np.mean(expit(intercept+x))-y.mean(),-100,100,xtol=1e-12)
 assert np.max(np.abs(fun(fit.x)[1]))<1e-7,(fit.message,fit.x)
 return {'clip_epsilon':eps,'calibration_in_the_large':citl,'joint_intercept':float(fit.x[0]),'slope':float(fit.x[1]),'gradient_max':float(np.max(np.abs(fun(fit.x)[1]))),'optimizer_success':bool(fit.success),'optimizer_message':str(fit.message),'clipped_count':int(((p<eps)|(p>1-eps)).sum())}
metrics=[];diags=[];bins=[];sources=[]
outer_members=pd.read_csv(R/'comparison/partition_membership.csv'); seen=[]
fig,axes=plt.subplots(3,3,figsize=(12,10))
for fold in [1,2,3]:
 prior=None
 for j,model in enumerate(['LR','RF','NN','HDC']):
  path=R/f'comparison/fold_{fold}/{model}/outer_scores.csv'
  z=pd.read_csv(path,float_precision='round_trip'); sources.append({'path':str(path.relative_to(R)),'sha256':sha(path)})
  expected=outer_members[(outer_members.fold==fold)&(outer_members.partition=='outer')].source_row_0based.to_numpy()
  assert np.array_equal(z.source_row_0based,expected)
  assert np.array_equal(z.smm,pd.to_numeric(d.iloc[expected].SMM).to_numpy())
  y=z.smm.to_numpy();p=z.score.to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
  if model=='LR':seen.extend(expected.tolist())
  if model=='NN':prior=p.copy()
  if model=='HDC':assert np.array_equal(p,prior)
  prev=y.mean();brier=np.mean((p-y)**2);null=prev*(1-prev)
  met={'fold':fold,'model':model,'n':len(y),'events':int(y.sum()),'prevalence':prev,'mean_prediction':p.mean(),'observed_expected_ratio':y.sum()/p.sum(),'brier':brier,'prevalence_only_brier':null,'brier_skill_vs_fold_prevalence':1-brier/null,'zero_scores':int((p==0).sum()),'one_scores':int((p==1).sum())}
  for eps in [1e-6]:diags.append({'fold':fold,'model':model,**logistic_diagnostic(y,p,eps)})
  met.update({k:v for k,v in diags[-1].items() if k not in ['fold','model']});metrics.append(met)
  cuts=np.unique(np.quantile(p,np.linspace(0,1,11)));labels=np.searchsorted(cuts[1:-1],p,side='left');local=[]
  for label in np.unique(labels):
   mask=labels==label
   row={'fold':fold,'model':model,'bin':int(label)+1,'n':int(mask.sum()),'events':int(y[mask].sum()),'mean_prediction':p[mask].mean(),'observed_rate':y[mask].mean(),'min_prediction':p[mask].min(),'max_prediction':p[mask].max()};bins.append(row);local.append(row)
  assert sum(q['n'] for q in local)==len(y) and sum(q['events'] for q in local)==int(y.sum())
  if model!='HDC':
   ax=axes[fold-1,j];b=pd.DataFrame(local);lim=max(b.mean_prediction.max(),b.observed_rate.max())*1.08
   ax.plot([0,lim],[0,lim],color='gray',linestyle='--',linewidth=1);ax.plot(b.mean_prediction,b.observed_rate,'o-',color=['#0072B2','#009E73','#D55E00'][j]);ax.set_xlim(0,lim);ax.set_ylim(0,lim);ax.set_title(f"Fold {fold} | {'NN / HDC' if model=='NN' else model}");ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed SMM proportion');ax.grid(alpha=.15)
assert len(seen)==len(set(seen))==208931 and set(seen)==set(outer_members.loc[outer_members.partition.eq('outer'),'source_row_0based'])
fig.suptitle('Calibration of unchanged outer-fold scores\nQuantile groups; tied scores kept together; panel scales differ',fontsize=13);fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(O/'calibration_curves.png',dpi=300);fig.savefig(O/'calibration_curves.pdf');plt.close(fig)
pd.DataFrame(metrics).to_csv(O/'calibration_metrics.csv',index=False);pd.DataFrame(diags).to_csv(O/'calibration_diagnostics.csv',index=False);pd.DataFrame(bins).to_csv(O/'calibration_bins.csv',index=False);save('score_sources.json',sources)
print(pd.DataFrame(metrics)[['fold','model','prevalence','mean_prediction','observed_expected_ratio','brier','brier_skill_vs_fold_prevalence','calibration_in_the_large','joint_intercept','slope','clipped_count']].to_string(index=False),flush=True)
