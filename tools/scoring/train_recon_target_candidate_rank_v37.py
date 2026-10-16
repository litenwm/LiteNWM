#!/usr/bin/env python3
"""Eight-bank whole-bank-OOF candidate safety ranker and fixed gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np,torch
REPO_ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO_ROOT));sys.path.insert(0,str(REPO_ROOT/"src"))
from scoring.dense_meta_v1 import wilson_lower
from tools.scoring.screen_recon_target_state_gate_v24 import predict
from tools.scoring.train_recon_target_state_gate_hashfold_v29 import train_fast
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256,write_json,write_npz

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--bank-g",type=Path,required=True);p.add_argument("--bank-h",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--epochs",type=int,default=180);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-2);p.add_argument("--hidden-dim",type=int,default=48);p.add_argument("--dropout",type=float,default=.4);p.add_argument("--ensemble-members",type=int,default=3);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--seed",type=int,default=20261008);return p.parse_args()

def load(name,feature,score,source):
 x,valid=compact_base_feature(feature,score,np.empty((100,0),np.float32))
 with np.load(source,allow_pickle=False) as p:record=p["record_key"].astype(str);route=p["route_group"].astype(str);targets=p["targets"].astype(np.float32);absolute=p["absolute_metrics"].astype(np.float32)
 delta=targets[:,1:,:4];dual=((delta[...,0]<-1e-7)&(delta[...,3]<-1e-7)).astype(np.float32);harm=((delta[...,0]>1e-7)|(delta[...,3]>1e-7)).astype(np.float32);return {"name":name,"record":record,"route":np.asarray([f"{name}::{r}" for r in route]),"x":x,"valid":valid,"delta":delta,"absolute":absolute,"dual":dual,"harm":harm}

def met(selected,b):
 promote=selected!=0;n=int(promote.sum());row=np.arange(100);dlt=np.zeros((100,4),np.float32);dlt[promote]=b["delta"][row[promote],selected[promote]-1];d=int(((dlt[promote,0]<-1e-7)&(dlt[promote,3]<-1e-7)).sum());h=int(((dlt[promote,0]>1e-7)|(dlt[promote,3]>1e-7)).sum());return {"coverage":n/100,"promotions":n,"dual":d/n if n else None,"wilson":wilson_lower(d,n),"harm":h/n if n else None}

def main():
 a=parse();result=a.output_dir/"result.json"
 if result.exists():raise FileExistsError(result)
 banks=[]
 for l in "abcdef":
  r=a.root/f"bank_{l}";banks.append(load(f"bank_{l}",next((r/"features").glob(f"recon_target_bank_{l}_v*_features.npz")),r/"universal_scores.npz",r/"source_labeled.npz"))
 banks.append(load("bank_g",a.bank_g/"features/recon_target_confirmation_v29_features.npz",a.bank_g/"universal_scores.npz",a.bank_g/"source_labeled.npz"));banks.append(load("bank_h",a.bank_h/"features/recon_target_cem_confirmation_v35_features.npz",a.bank_h/"universal_scores.npz",a.bank_h/"source_labeled.npz"));names=[b["name"] for b in banks];cohort_state=np.concatenate([np.full(100,n,dtype="<U16") for n in names]);group_state=np.concatenate([b["route"] for b in banks]);valid=np.concatenate([b["valid"] for b in banks]);cohort=np.repeat(cohort_state,16);group=np.repeat(group_state,16);dual=np.concatenate([b["dual"] for b in banks]).reshape(-1);harm=np.concatenate([b["harm"] for b in banks]).reshape(-1);a.output_dir.mkdir(parents=True,exist_ok=False);preds={};checkpoints={}
 for vn,(variant,sl) in enumerate((("candidate_universal_support",slice(474,486)),("candidate_motion_universal_support",slice(434,486)))):
  x=np.concatenate([b["x"][...,sl] for b in banks]).reshape(-1,sl.stop-sl.start);pd=np.zeros(len(x),np.float32);ph=np.zeros(len(x),np.float32);pds=np.zeros(len(x),np.float32);phs=np.zeros(len(x),np.float32);checkpoints[variant]={}
  for fi,n in enumerate(names):
   ti=np.flatnonzero((cohort!=n)&valid.reshape(-1));hi=np.flatnonzero(cohort==n);md=[];mh=[];checkpoints[variant][n]=[]
   for m in range(a.ensemble_members):
    seed=a.seed+vn*1000000+fi*100000+m*1009;model,mean,scale=train_fast(x,dual,harm,group,ti,a,seed);d,h=predict(model,mean,scale,x[hi],a.device);md.append(d);mh.append(h);path=a.output_dir/f"{variant}_exclude_{n}_member_{m}.pt";torch.save({"schema":"recon_target_candidate_rank_v37_member","variant":variant,"excluded_bank":n,"input_dim":x.shape[1],"hidden_dim":a.hidden_dim,"dropout":a.dropout,"feature_mean":mean,"feature_scale":scale,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"seed":seed},path);checkpoints[variant][n].append({"path":str(path),"sha256":file_sha256(path),"seed":seed});del model
   ds=np.stack(md);hs=np.stack(mh);pd[hi]=ds.mean(0);ph[hi]=hs.mean(0);pds[hi]=ds.std(0);phs[hi]=hs.std(0)
  preds[variant]={"dual":pd.reshape(800,16),"harm":ph.reshape(800,16),"dual_std":pds.reshape(800,16),"harm_std":phs.reshape(800,16)};pp=a.output_dir/f"{variant}_oof.npz";write_npz(pp,record_key=np.concatenate([b["record"] for b in banks]),cohort=cohort_state,**preds[variant]);preds[variant]["path"]=str(pp);preds[variant]["sha256"]=file_sha256(pp)
 feasible=[];topdiag=[]
 for blend in (.5,.6,.7,.8,.9,1.):
  d=blend*preds["candidate_universal_support"]["dual"]+(1-blend)*preds["candidate_motion_universal_support"]["dual"];h=blend*preds["candidate_universal_support"]["harm"]+(1-blend)*preds["candidate_motion_universal_support"]["harm"]
  for hw in (.25,.5,1.,2.,4.):
   for z in (0.,.5,1.):
    ds=blend*preds["candidate_universal_support"]["dual_std"]+(1-blend)*preds["candidate_motion_universal_support"]["dual_std"];hs=blend*preds["candidate_universal_support"]["harm_std"]+(1-blend)*preds["candidate_motion_universal_support"]["harm_std"];score=d-z*ds-hw*(h+z*hs);score=np.where(valid,score,-np.inf);best=score.argmax(1);conf=score[np.arange(800),best];thresholds=np.unique(conf)
    for t in thresholds:
     per={};ok=True
     for i,b in enumerate(banks):
      c=conf[i*100:(i+1)*100];sel=np.where(c>t,best[i*100:(i+1)*100]+1,0);m=met(sel,b);ok=ok and m["coverage"]>=.4 and m["dual"] is not None and m["dual"]>=.9 and m["wilson"] is not None and m["wilson"]>=.8 and m["harm"] is not None and m["harm"]<=.1;per[b["name"]]=m
     if ok:feasible.append({"blend":blend,"harm_weight":hw,"z":z,"threshold":float(t),"banks":per,"coverage_sum":sum(v["coverage"] for v in per.values())})
    td={}
    for i,b in enumerate(banks):
     c=conf[i*100:(i+1)*100];order=np.argsort(-c,kind="stable")[:40];sel=np.zeros(100,np.int64);sel[order]=best[i*100:(i+1)*100][order]+1;td[b["name"]]=met(sel,b)
    topdiag.append({"blend":blend,"harm_weight":hw,"z":z,"banks":td})
 best_policy=max(feasible,key=lambda x:x["coverage_sum"]) if feasible else None;best_top=max(topdiag,key=lambda x:min((v["dual"] or 0) for v in x["banks"].values()));report={"schema":"recon_target_candidate_rank_v37","status":"eight_bank_candidate_oof_complete","cem":False,"checkpoints":checkpoints,"predictions":{k:{"path":v["path"],"sha256":v["sha256"]} for k,v in preds.items()},"feasible_count":len(feasible),"best_policy":best_policy,"best_top40":best_top,"formal_claim_allowed":False};write_json(result,report);print(json.dumps({"feasible_count":len(feasible),"best":best_policy,"best_top40":best_top},indent=2,sort_keys=True))
if __name__=="__main__":main()
