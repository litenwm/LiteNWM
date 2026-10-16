#!/usr/bin/env python3
"""Retrain/calibrate an eight-bank whole-bank-excluded fixed-pool base gate."""

from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
import torch

REPO_ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO_ROOT));sys.path.insert(0,str(REPO_ROOT/"src"))
from scoring.dense_meta_v1 import wilson_lower
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_state_gate_v24 import predict
from tools.scoring.train_recon_target_state_gate_hashfold_v29 import train_fast
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256,write_json,write_npz

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--bank-g",type=Path,required=True);p.add_argument("--bank-h",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--epochs",type=int,default=180);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=1e-2);p.add_argument("--hidden-dim",type=int,default=48);p.add_argument("--dropout",type=float,default=.4);p.add_argument("--ensemble-members",type=int,default=3);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--seed",type=int,default=20261007);return p.parse_args()

def load(name,root,feature,score,source):
 compact,valid=compact_base_feature(feature,score,np.empty((100,0),np.float32))
 with np.load(source,allow_pickle=False) as p:record=p["record_key"].astype(str);route=p["route_group"].astype(str);targets=p["targets"].astype(np.float32);absolute=p["absolute_metrics"].astype(np.float32)
 with np.load(score,allow_pickle=False) as p:sr=p["record_key"].astype(str);s={"std":p["std"][:,1:].astype(np.float32),"dual":p["dual_probability"][:,1:].astype(np.float32),"harm":p["harm_probability"][:,1:].astype(np.float32)}
 if not np.array_equal(record,sr):raise RuntimeError(f"identity mismatch {name}")
 challenger=frozen_challenger(s,valid);delta=targets[:,1:,:4];row=np.arange(100);chosen=compact[row,challenger];cd=delta[row,challenger];dual=((cd[:,0]<-1e-7)&(cd[:,3]<-1e-7)).astype(np.float32);harm=((cd[:,0]>1e-7)|(cd[:,3]>1e-7)).astype(np.float32)
 return {"name":name,"record":record,"route":np.asarray([f"{name}::{x}" for x in route]),"feature":{"state_universal_support":chosen[:,474:486],"state_motion_universal_support":chosen[:,434:486]},"challenger":challenger+1,"dual":dual,"harm":harm,"delta":delta,"absolute":absolute}

def metrics(selected,bank):
 promote=selected!=0;n=int(promote.sum());row=np.arange(100);chosen_delta=np.zeros((100,4),np.float32);chosen_delta[promote]=bank["delta"][row[promote],selected[promote]-1];d=int(((chosen_delta[promote,0]<-1e-7)&(chosen_delta[promote,3]<-1e-7)).sum());h=int(((chosen_delta[promote,0]>1e-7)|(chosen_delta[promote,3]>1e-7)).sum());return {"promotions":n,"coverage":n/100,"dual":d/n if n else None,"wilson":wilson_lower(d,n),"harm":h/n if n else None}

def main():
 a=parse();result_path=a.output_dir/"result.json"
 if result_path.exists():raise FileExistsError(result_path)
 banks=[]
 for l in "abcdef":
  r=a.root/f"bank_{l}";f=next((r/"features").glob(f"recon_target_bank_{l}_v*_features.npz"));banks.append(load(f"bank_{l}",r,f,r/"universal_scores.npz",r/"source_labeled.npz"))
 banks.append(load("bank_g",a.bank_g,a.bank_g/"features/recon_target_confirmation_v29_features.npz",a.bank_g/"universal_scores.npz",a.bank_g/"source_labeled.npz"));banks.append(load("bank_h",a.bank_h,a.bank_h/"features/recon_target_cem_confirmation_v35_features.npz",a.bank_h/"universal_scores.npz",a.bank_h/"source_labeled.npz"))
 names=[b["name"] for b in banks];cohort=np.concatenate([np.full(100,n,dtype="<U16") for n in names]);group=np.concatenate([b["route"] for b in banks]);dual=np.concatenate([b["dual"] for b in banks]);harm=np.concatenate([b["harm"] for b in banks]);record=np.concatenate([b["record"] for b in banks]);a.output_dir.mkdir(parents=True,exist_ok=False);predictions={};checkpoints={}
 for vn,variant in enumerate(("state_universal_support","state_motion_universal_support")):
  x=np.concatenate([b["feature"][variant] for b in banks]);pd=np.zeros(800,np.float32);ph=np.zeros(800,np.float32);pds=np.zeros(800,np.float32);phs=np.zeros(800,np.float32);checkpoints[variant]={}
  for fi,heldout in enumerate(names):
   ti=np.flatnonzero(cohort!=heldout);hi=np.flatnonzero(cohort==heldout);md=[];mh=[];checkpoints[variant][heldout]=[]
   for m in range(a.ensemble_members):
    seed=a.seed+vn*1000000+fi*100000+m*1009;model,mean,scale=train_fast(x,dual,harm,group,ti,a,seed);d,h=predict(model,mean,scale,x[hi],a.device);md.append(d);mh.append(h);path=a.output_dir/f"{variant}_exclude_{heldout}_member_{m}.pt";torch.save({"schema":"recon_target_hashfold_v36_member","excluded_bank":heldout,"input_dim":x.shape[1],"hidden_dim":a.hidden_dim,"dropout":a.dropout,"feature_mean":mean,"feature_scale":scale,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"seed":seed},path);checkpoints[variant][heldout].append({"path":str(path),"sha256":file_sha256(path),"seed":seed});del model
   ds=np.stack(md);hs=np.stack(mh);pd[hi]=ds.mean(0);ph[hi]=hs.mean(0);pds[hi]=ds.std(0);phs[hi]=hs.std(0)
  pp=a.output_dir/f"{variant}_oof.npz";write_npz(pp,record_key=record,cohort=cohort,p_dual=pd,p_harm=ph,p_dual_std=pds,p_harm_std=phs);predictions[variant]={"path":str(pp),"sha256":file_sha256(pp),"p_dual":pd,"p_harm":ph}
 feasible=[];diagnostic=[]
 for blend in (.5,.6,.7,.8,.9,1.0):
  for hw in (.25,.5,1.,2.,4.):
   risk=blend*predictions["state_universal_support"]["p_dual"]+(1-blend)*predictions["state_motion_universal_support"]["p_dual"]-hw*(blend*predictions["state_universal_support"]["p_harm"]+(1-blend)*predictions["state_motion_universal_support"]["p_harm"]);thresholds=np.unique(risk)
   for t in thresholds:
    per={};ok=True
    for i,b in enumerate(banks):
     r=risk[i*100:(i+1)*100];selected=np.where(r>t,b["challenger"],0);m=metrics(selected,b);ok=ok and m["coverage"]>=.4 and m["dual"] is not None and m["dual"]>=.9 and m["wilson"] is not None and m["wilson"]>=.8 and m["harm"] is not None and m["harm"]<=.1;per[b["name"]]=m
    if ok:feasible.append({"blend":blend,"harm_weight":hw,"threshold":float(t),"banks":per,"coverage_sum":sum(v["coverage"] for v in per.values())})
   top={}
   for i,b in enumerate(banks):
    r=risk[i*100:(i+1)*100];order=np.argsort(-r,kind="stable")[:40];s=np.zeros(100,np.int64);s[order]=b["challenger"][order];top[b["name"]]=metrics(s,b)
   diagnostic.append({"blend":blend,"harm_weight":hw,"top40":top})
 best=max(feasible,key=lambda x:x["coverage_sum"]) if feasible else None;report={"schema":"recon_target_hashfold_v36","status":"eight_bank_whole_bank_oof_complete","banks":names,"states":800,"routes":200,"cem":False,"checkpoints":checkpoints,"predictions":{k:{"path":v["path"],"sha256":v["sha256"]} for k,v in predictions.items()},"feasible_count":len(feasible),"best_policy":best,"top40_diagnostic":diagnostic,"deployment_router":"sha256(route_group) modulo 8","formal_claim_allowed":False};write_json(result_path,report);print(json.dumps({"feasible_count":len(feasible),"best":best,"best_top40":max(diagnostic,key=lambda x:sum(v["dual"] or 0 for v in x["top40"].values()))},indent=2,sort_keys=True))

if __name__=="__main__":main()
