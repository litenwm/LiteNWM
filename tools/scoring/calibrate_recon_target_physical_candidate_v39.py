#!/usr/bin/env python3
"""Calibrate physical K17 candidate filtering plus frozen universal ranking."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
import numpy as np
REPO_ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO_ROOT));sys.path.insert(0,str(REPO_ROOT/"src"))
from scoring.dense_meta_v1 import wilson_lower
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256,write_json,write_npz

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--bank-g",type=Path,required=True);p.add_argument("--bank-h",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);return p.parse_args()
def load(name,feature,score,source):
 with np.load(feature,allow_pickle=False) as p:path=p["candidates"].astype(np.float32);record=p["record_key"].astype(str);support=p["majority_support"][:,1:].astype(bool)
 with np.load(score,allow_pickle=False) as p:std=p["std"][:,1:].astype(np.float32);dual=p["dual_probability"][:,1:].astype(np.float32);harm=p["harm_probability"][:,1:].astype(np.float32);valid=p["valid"][:,1:].astype(bool) if "valid" in p.files else support
 with np.load(source,allow_pickle=False) as p:targets=p["targets"][:,1:,:4].astype(np.float32)
 diff=path[:,1:]-path[:,:1];rmse=np.sqrt(np.mean(diff*diff,axis=(2,3)));endpoint=np.linalg.norm(diff[:,:,-1],axis=2);ic=np.linalg.norm(np.diff(path[:,:1],n=2,axis=2),axis=3).mean(2);cc=np.linalg.norm(np.diff(path[:,1:],n=2,axis=2),axis=3).mean(2);curve=np.abs(cc-ic);return {"name":name,"record":record,"std":std,"dual_score":dual,"harm_score":harm,"valid":valid,"rmse":rmse,"endpoint":endpoint,"curve":curve,"target":targets}
def met(sel,b):
 pro=sel!=0;n=int(pro.sum());r=np.arange(100);dlt=np.zeros((100,4),np.float32);dlt[pro]=b["target"][r[pro],sel[pro]-1];d=int(((dlt[pro,0]<-1e-7)&(dlt[pro,3]<-1e-7)).sum());h=int(((dlt[pro,0]>1e-7)|(dlt[pro,3]>1e-7)).sum());return {"promotions":n,"coverage":n/100,"dual":d/n if n else None,"wilson":wilson_lower(d,n),"harm":h/n if n else None}
def main():
 a=parse();out=a.output_dir/"result.json"
 if out.exists():raise FileExistsError(out)
 banks=[]
 for l in "abcdef":
  r=a.root/f"bank_{l}";banks.append(load(f"bank_{l}",next((r/"features").glob(f"recon_target_bank_{l}_v*_features.npz")),r/"universal_scores.npz",r/"source_labeled.npz"))
 banks.append(load("bank_g",a.bank_g/"features/recon_target_confirmation_v29_features.npz",a.bank_g/"universal_scores.npz",a.bank_g/"source_labeled.npz"));banks.append(load("bank_h",a.bank_h/"features/recon_target_cem_confirmation_v35_features.npz",a.bank_h/"universal_scores.npz",a.bank_h/"source_labeled.npz"));rmse=np.concatenate([b["rmse"] for b in banks]);endpoint=np.concatenate([b["endpoint"] for b in banks]);curve=np.concatenate([b["curve"] for b in banks]);q=(.3,.5,.7,.85,1.);rc=np.unique(np.quantile(rmse,q));ec=np.unique(np.quantile(endpoint,q));cc=np.unique(np.quantile(curve,q));feasible=[];best_diag=None
 for hw in (.25,.5,1.,2.):
  for uw in (0.,.25,.5,1.):
   for rcap in rc:
    for ecap in ec:
     for ccap in cc:
      conf=[];choice=[]
      for b in banks:
       score=b["dual_score"]-hw*b["harm_score"]-uw*(b["std"][...,0]+b["std"][...,3]);ok=b["valid"]&(b["rmse"]<=rcap)&(b["endpoint"]<=ecap)&(b["curve"]<=ccap);masked=np.where(ok,score,-np.inf);ch=masked.argmax(1);co=masked[np.arange(100),ch];conf.append(co);choice.append(ch+1)
      if any(np.isfinite(x).sum()<40 for x in conf):continue
      threshold=min(np.sort(x[np.isfinite(x)])[-40] for x in conf)-1e-9;per={};passed=True
      for b,co,ch in zip(banks,conf,choice):
       sel=np.where(co>threshold,ch,0);m=met(sel,b);passed=passed and m["coverage"]>=.4 and m["dual"] is not None and m["dual"]>=.9 and m["wilson"] is not None and m["wilson"]>=.8 and m["harm"] is not None and m["harm"]<=.1;per[b["name"]]=m
      item={"harm_weight":hw,"uncertainty_weight":uw,"rmse_cap":float(rcap),"endpoint_cap":float(ecap),"curvature_cap":float(ccap),"threshold":float(threshold),"banks":per,"coverage_sum":sum(v["coverage"] for v in per.values())}
      if passed:feasible.append(item)
      if best_diag is None or min(v["dual"] or 0 for v in per.values())>min(v["dual"] or 0 for v in best_diag["banks"].values()):best_diag=item
 best=max(feasible,key=lambda x:x["coverage_sum"]) if feasible else None;a.output_dir.mkdir(parents=True,exist_ok=False);report={"schema":"recon_target_physical_candidate_v39","status":"eight_bank_physical_candidate_calibration_complete","cem":False,"feature_contract":"candidate-vs-incumbent path RMSE, endpoint displacement, curvature change then frozen universal ranking","future_windows_used":False,"feasible_count":len(feasible),"best_policy":best,"best_diagnostic":best_diag,"formal_claim_allowed":False};write_json(out,report);print(json.dumps({"feasible_count":len(feasible),"best":best,"best_diagnostic":best_diag},indent=2,sort_keys=True))
if __name__=="__main__":main()
