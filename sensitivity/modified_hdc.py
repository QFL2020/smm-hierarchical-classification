"""Prespecified compact, AP-selected, bottom-up-pruned HDC; no outer data API."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,time,json,hashlib,copy
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.model_selection import train_test_split
from neural_runtime import get_cluster_model, BalancedSampler, FeatureDataset, setup_seed
torch.set_num_threads(2)

def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2,default=str))
def role_seed(fold,role):return int.from_bytes(hashlib.sha256(f'{fold}:{role}:53'.encode()).digest()[:4],'little')%2147483647
def ap(y,s):return float(average_precision_score(y,s))
def auc(y,s):return float(roc_auc_score(y,s))
def accept(parent_ap,parent_auc,child_ap,child_auc):return child_ap>=parent_ap+.002 and child_auc>=parent_auc-.01
def enough(frame,events):return int(frame.SMM.sum())>=events and int((frame.SMM==0).sum())>=1
def features(d):return d.drop(columns='SMM')
def subset(d,mask):
 a=d.iloc[np.flatnonzero(mask)].copy();a.attrs={k:np.asarray(v)[mask] for k,v in d.attrs.items()};return a
def predict(model,x):
 if 'SMM' in x:raise ValueError('Labels excluded from inference')
 if not len(x):return np.array([],dtype=float)
 model.model.eval()
 return np.concatenate([model.predict_proba(x.iloc[i:i+8192])[:,1] for i in range(0,len(x),8192)])
# Checkpoint selection uses only the selection subset; the outer fold is unavailable.
def fit_nn(tr,sel,dims,dest,seed,epochs=30,patience=5):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
 if set(tr.attrs['patient_ids'])&set(sel.attrs['patient_ids']):raise ValueError('Patient overlap')
 assert enough(tr,1) and enough(sel,1)
 setup_seed(seed);start=time.time()
 m=get_cluster_model('MLP_CleanedSMM',[64,32],epochs,dims,os.environ.get('SMM_DEVICE', 'cuda:0'),lr=-2);m._init_model(len(tr))
 x=features(tr).to_numpy();y=tr.SMM.to_numpy();vs=features(sel);vy=sel.SMM.to_numpy()
 ds=FeatureDataset(x,y.reshape(-1,1),device=os.environ.get('SMM_DEVICE', 'cuda:0'),dtype=torch.float64)
 loader=torch.utils.data.DataLoader(ds,sampler=BalancedSampler(ds,4096),batch_size=4096,shuffle=False)
 best=-np.inf;best_state=None;stale=0;history=[]
 for epoch in range(epochs):
  m.model.train();losses=[]
  for bx,by in loader:
   m.optimizer.zero_grad();loss=m.criterion(m.model(bx),by);loss.backward();m.optimizer.step();losses.append(float(loss.item()))
  m.scheduler.step();s=predict(m,vs);value=ap(vy,s)
  history.append({'epoch':epoch+1,'balanced_batch_training_bce':float(np.mean(losses)),'selection_ap':value,'selection_auc':auc(vy,s)})
  if value>best:
   best=value;best_state=copy.deepcopy(m.model.state_dict());m.best_epoch=epoch+1;stale=0
  else:stale+=1
  if stale>=patience:break
 m.model.load_state_dict(best_state);m.model.eval();m.fit_epochs=epoch+1
 # Keep inference state; optimizer and dataset tensors are not serialized.
 torch.save({'state_dict':m.model.state_dict(),'dims':dims,'architecture':[64,32],'seed':seed,'best_epoch':m.best_epoch},dest/'checkpoint.pt')
 pd.DataFrame(history).to_csv(dest/'epochs.csv',index=False)
 write(dest/'status.json',{'seed':seed,'train_n':len(tr),'train_smm':int(tr.SMM.sum()),'selection_n':len(sel),'selection_smm':int(sel.SMM.sum()),'epochs':m.fit_epochs,'best_epoch':m.best_epoch,'best_selection_ap':best,'seconds':time.time()-start})
 del m.optimizer,m.scheduler,m.criterion
 print('FIT',str(dest.name),'epochs',m.fit_epochs,'seconds',round(time.time()-start,2),flush=True)
 return m

class Node:
 def __init__(self,node_id,model):self.node_id=node_id;self.model=model;self.router=None;self.children=[]

def tree_predict(node,x):
 if 'SMM' in x:raise ValueError('Labels excluded from inference')
 if not node.children:return predict(node.model,x)
 route=predict(node.router,x)>=.5;s=np.empty(len(x))
 for side in [0,1]:
  mask=route==side
  if mask.any():s[mask]=tree_predict(node.children[side],x.iloc[np.flatnonzero(mask)])
 return s

# Grow supported branches, then prune bottom-up on the separate pruning subset.
def fit_tree(tr,sel,prune,dims,root_model,dest,fold):
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=False);records=[]
 for a,b in [(tr,sel),(tr,prune),(sel,prune)]:assert not set(a.attrs['patient_ids'])&set(b.attrs['patient_ids'])
 def grow(a,b,c,parent,node_id,depth):
  node=Node(node_id,parent);record={'node':node_id,'depth':depth,'training_n':len(a),'training_smm':int(a.SMM.sum()),'selection_n':len(b),'selection_smm':int(b.SMM.sum()),'pruning_n':len(c),'pruning_smm':int(c.SMM.sum()),'candidates':[]};records.append(record)
  if depth>=2:record['decision']='depth_cap';return node
  if not(enough(a,200) and enough(b,40) and enough(c,40)):
   record['decision']='insufficient_total_support';return node
  rng=np.random.default_rng(role_seed(fold,f'node_{node_id}_routing_initial'))
  targets=np.zeros(len(a),dtype=int);targets[rng.choice(len(a),len(a)//2,replace=False)]=1
  best=None;best_ap=-np.inf
  for iteration in range(2):
   folder=dest/f'node_{node_id}_iter_{iteration}';rr={'iteration':iteration};record['candidates'].append(rr)
   if min(targets.sum(),len(targets)-targets.sum())<10:rr['decision']='degenerate_routing_target';break
   # Hold out entire fitting patients for router epoch selection.
   temp=a.copy();temp['SMM']=targets
   gids=np.unique(a.attrs['patient_ids']);rt,rv=train_test_split(gids,test_size=.1,random_state=role_seed(fold,f'{node_id}_{iteration}_router_split'))
   mask=np.isin(a.attrs['patient_ids'],rt);ra=subset(temp,mask);rb=subset(temp,~mask)
   if not(enough(ra,1) and enough(rb,1)):rr['decision']='router_holdout_single_class';break
   router=fit_nn(ra,rb,dims,folder/'router',role_seed(fold,f'{node_id}_{iteration}_router'))
   routes=[predict(router,features(z))>=.5 for z in [a,b,c]]
   splits=[[subset(z,r==side) for z,r in zip([a,b,c],routes)] for side in [0,1]]
   rr['support']=[{'train_smm':int(z[0].SMM.sum()),'selection_smm':int(z[1].SMM.sum()),'pruning_smm':int(z[2].SMM.sum())} for z in splits]
   if not all(enough(z[0],100) and enough(z[1],20) and enough(z[2],20) for z in splits):
    rr['decision']='child_support_rejected';break
   branches=[fit_nn(z[0],z[1],dims,folder/f'branch_{side}',role_seed(fold,f'{node_id}_{iteration}_branch_{side}')) for side,z in enumerate(splits)]
   ss=np.empty(len(b))
   for side in [0,1]:ss[routes[1]==side]=predict(branches[side],features(splits[side][1]))
   value=ap(b.SMM,ss);rr.update({'decision':'supported_candidate','selection_ap':value,'selection_auc':auc(b.SMM,ss)})
   if value>best_ap:best=(router,branches,splits,iteration);best_ap=value
   predictions=[predict(m,features(a)) for m in branches]
   targets=(np.abs(predictions[1]-a.SMM.to_numpy())<np.abs(predictions[0]-a.SMM.to_numpy())).astype(int)
  if best is None:record['decision']='no_supported_candidate';return node
  router,branches,splits,iteration=best;node.router=router
  record['selected_iteration']=iteration
  node.children=[grow(*z,branches[side],2*node_id+1+side,depth+1) for side,z in enumerate(splits)]
  ps=predict(parent,features(c));cs=tree_predict(node,features(c));pa=ap(c.SMM,ps);ca=ap(c.SMM,cs);pu=auc(c.SMM,ps);cu=auc(c.SMM,cs)
  record.update({'parent_pruning_ap':pa,'subtree_pruning_ap':ca,'parent_pruning_auc':pu,'subtree_pruning_auc':cu})
  pd.DataFrame({'source_row_0based':c.attrs['source_rows'],'smm':c.SMM.to_numpy(),'parent_score':ps,'candidate_subtree_score':cs}).to_csv(dest/f'node_{node_id}_pruning_scores.csv',index=False,float_format='%.17g')
  if accept(pa,pu,ca,cu):record['decision']='accept_subtree'
  else:record['decision']='retain_parent';node.router=None;node.children=[]
  return node
 root=grow(tr,sel,prune,root_model,0,0);write(dest/'decisions.json',records)
 return root
