"""Figure S1 from saved continuous scores; no predictive fitting."""
if __name__ != "__main__":
    raise RuntimeError("Run this analysis script explicitly; importing it does not execute analyses.")
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,average_precision_score
from paths import output_root, new_output
P=output_root()/'comparison';OUT=new_output('precision_recall');W=OUT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(10.5,3.7),sharex=True,sharey=True);manifest=[];points=[]
for fold,ax in enumerate(axes,1):
 for name,color,label in [('LR','#0072B2','Logistic regression'),('RF','#009E73','Random forest'),('NN','#D55E00','NN / modified HDC')]:
  path=P/f'fold_{fold}/{name}/outer_scores.csv';d=pd.read_csv(path,float_precision='round_trip');manifest.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
  if name=='NN':
   h=pd.read_csv(P/f'fold_{fold}/HDC/outer_scores.csv',float_precision='round_trip');assert np.array_equal(d.score,h.score) and np.array_equal(d.source_row_0based,h.source_row_0based)
  precision,recall,threshold=precision_recall_curve(d.smm,d.score);ap=average_precision_score(d.smm,d.score)
  ax.step(recall,precision,where='post',color=color,lw=1.25,label=f'{label}  AP {ap:.3f}')
  points.append(pd.DataFrame({'fold':fold,'model':name,'recall':recall,'precision':precision}))
 prevalence=d.smm.mean();ax.axhline(prevalence,color='#666666',ls=':',lw=1,label=f'Prevalence {prevalence:.3%}')
 ax.set_title(f'Fold {fold}   n = {len(d):,}',fontsize=10);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_xlabel('Recall (sensitivity)');ax.grid(alpha=.15)
 ax.legend(loc='upper right',fontsize=7.3,frameon=False)
axes[0].set_ylabel('Precision (PPV)');fig.tight_layout(pad=1.1)
fig.savefig(OUT/'Figure_S1_precision_recall.png',dpi=400);fig.savefig(OUT/'Figure_S1_precision_recall.pdf');plt.close(fig)
pd.concat(points).to_csv(W/'figure1_curve_points.csv',index=False)
(W/'figure1_verification.json').write_text(json.dumps({'sources':manifest,'HDC_NN_scores_identical_all_folds':True,'curves':'fold-specific continuous outer scores, no pooling; precision_recall_curve; AP is average_precision_score','models_loaded_or_trained':False},indent=2))
print('Figure saved')
