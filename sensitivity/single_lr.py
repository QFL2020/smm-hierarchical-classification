"""New, single-seed patient-disjoint LR sensitivity analysis; not a historical rerun.
No outcome-informed exclusions. No tuning, resampling, weighting, or test selection.
Patient split stratified by any SMM among that patient's supplied records.
Same 76 predictors and LR C/penalty/solver as supplied study implementation.
"""
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import sys,json,time,warnings,platform
from pathlib import Path
from paths import data_path, new_output, feature_names
OUT=None
import numpy as np
import pandas as pd
import sklearn,category_encoders as ce,joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
import hashlib
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def features():
    cat,num=feature_names()
    return {'categorical_vars':cat,'continuous_vars':num}

def write(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str))
def metrics(y,p,t=.5,weights=None):
    z=p>=t
    tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1],sample_weight=weights).ravel()
    n=tn+fp+fn+tp
    return {'n':float(n),'tn':float(tn),'fp':float(fp),'fn':float(fn),'tp':float(tp),'threshold':float(t),'sensitivity':float(tp/(tp+fn)),'specificity':float(tn/(tn+fp)),'ppv':float(tp/(tp+fp)) if tp+fp else None,'accuracy':float((tp+tn)/n),'f1':float(2*tp/(2*tp+fp+fn)),'screen_positive_rate':float((tp+fp)/n),'roc_auc':float(roc_auc_score(y,p,sample_weight=weights)),'average_precision':float(average_precision_score(y,p,sample_weight=weights)),'brier_score':float(brier_score_loss(y,p,sample_weight=weights))}

if __name__=='__main__':
    OUT=new_output('single_lr')
    start=time.time(); seed=53
    fs=features();cat=fs['categorical_vars'];num=fs['continuous_vars']
    d=pd.read_csv(data_path(),low_memory=False)
    assert d.VisitLink.notna().all() and set(d.SMM.unique())=={0,1}
    patients=d.groupby('VisitLink',sort=True).SMM.max()
    trids,teids=train_test_split(patients.index.to_numpy(),test_size=.2,random_state=seed,stratify=patients.to_numpy())
    tr=np.flatnonzero(d.VisitLink.isin(trids));te=np.flatnonzero(d.VisitLink.isin(teids))
    assert not set(d.iloc[tr].VisitLink)&set(d.iloc[te].VisitLink)
    assert len(tr)+len(te)==len(d)
    pd.DataFrame({'source_row_0based':np.arange(len(d)),'partition':np.where(d.VisitLink.isin(trids),'train','test')}).to_csv(OUT/'partition_membership.csv',index=False)
    protocol={'analysis':'new single-seed patient-disjoint sensitivity analysis','seed':seed,'patient_split':'80:20 train_test_split on distinct VisitLink, stratify=patient any-SMM indicator','excluded_records':0,'features':fs,'categorical_encoding':'category_encoders BinaryEncoder 2.6.3; fit on train only; default parser and missing/unknown value handling','continuous_encoding':'StandardScaler fit on train only','lr':{'penalty':'l2','solver':'saga','C':.7,'max_iter':100000,'tol':.0001,'random_state':seed,'class_weight':None},'sampling':'none','test_used_for_fit_or_selection':False,'operating_points':'0.5 and training-score quantiles 0.995,0.99,0.95, with actual test workload reported','bootstrap':'200 patient-cluster percentile replicates; fixed fitted model; not training uncertainty','versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,'category_encoders':ce.__version__},'original_requirement_versions':{'numpy':'1.23.2','pandas':'2.1.4','sklearn':'1.4.2'},'data_sha256':digest(data_path())}
    write('protocol.json',protocol)
    print('Split complete',len(tr),len(te),'SMM',int(d.iloc[tr].SMM.sum()),int(d.iloc[te].SMM.sum()),flush=True)
    encoder=ce.BinaryEncoder(cols=cat,return_df=True)
    scaler=StandardScaler()
    xt=np.concatenate([encoder.fit_transform(d.iloc[tr][cat]).to_numpy(dtype=float),scaler.fit_transform(d.iloc[tr][num])],axis=1)
    xv=np.concatenate([encoder.transform(d.iloc[te][cat]).to_numpy(dtype=float),scaler.transform(d.iloc[te][num])],axis=1)
    assert np.isfinite(xt).all() and np.isfinite(xv).all()
    yt=d.iloc[tr].SMM.to_numpy();yv=d.iloc[te].SMM.to_numpy()
    print('Fit starting',xt.shape,'elapsed',time.time()-start,flush=True)
    model=LogisticRegression(**protocol['lr'])
    fitstart=time.time()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always');model.fit(xt,yt)
    fitsecs=time.time()-fitstart
    print('Fit complete',fitsecs,'iterations',model.n_iter_,flush=True)
    warnings_list=[str(w.message) for w in ws]
    if any(issubclass(w.category,ConvergenceWarning) for w in ws):
        write('nonconvergence.json',{'warnings':warnings_list,'elapsed':time.time()-start});sys.exit(2)
    ptr=model.predict_proba(xt)[:,1];p=model.predict_proba(xv)[:,1]
    joblib.dump({'model':model,'categorical_encoder':encoder,'continuous_scaler':scaler,'categorical_columns':cat,'continuous_columns':num},OUT/'lr_pipeline.joblib')
    pd.DataFrame({'source_row_0based':te,'smm':yv,'score':p,'prediction_at_0_5':(p>=.5).astype(int)}).to_csv(OUT/'test_predictions.csv',index=False)
    pd.DataFrame({'source_row_0based':tr,'smm':yt,'score':ptr}).to_csv(OUT/'train_predictions.csv',index=False)
    result={'train_n':len(tr),'test_n':len(te),'train_smm':int(yt.sum()),'test_smm':int(yv.sum()),'train_patients':len(trids),'test_patients':len(teids),'shared_patients':0,'shared_hospitals':len(set(d.iloc[tr].DSHOSPID)&set(d.iloc[te].DSHOSPID)),'fit_seconds':fitsecs,'iterations':model.n_iter_.tolist(),'warnings':warnings_list,'encoded_features':xt.shape[1],'test_prevalence':float(yv.mean()),'mean_predicted_risk':float(p.mean()),'default_threshold':metrics(yv,p),'training_quantile_thresholds':[]}
    for rate in [.005,.01,.05]:
        cutoff=float(np.quantile(ptr,1-rate));m=metrics(yv,p,cutoff);m.update({'nominal_training_positive_rate':rate,'actual_training_positive_rate':float((ptr>=cutoff).mean())});result['training_quantile_thresholds'].append(m)
    write('sensitivity_results.json',result)
    rng=np.random.default_rng(5301);codes,uniques=pd.factorize(d.iloc[te].VisitLink);ng=len(uniques);boot=[]
    for i in range(200):
        w=np.bincount(rng.integers(ng,size=ng),minlength=ng)[codes]
        boot.append(metrics(yv,p,weights=w))
    cis={}
    for k in ['roc_auc','average_precision','sensitivity','specificity','ppv','accuracy','f1','screen_positive_rate','brier_score']:
        v=[b[k] for b in boot if b[k] is not None];cis[k]=np.percentile(v,[2.5,97.5]).tolist()
    pd.DataFrame(boot).to_csv(OUT/'patient_bootstrap_metrics.csv',index=False)
    result['patient_bootstrap_95_percentile_ci']=cis;result['elapsed_seconds']=time.time()-start
    write('sensitivity_results.json',result)
    print(json.dumps(result,indent=2),flush=True)
