#!/usr/bin/env python3
"""Calibrate causal physical-distance guards for v32 CEM replacements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--root",type=Path,required=True); p.add_argument("--v32-result",type=Path,required=True); p.add_argument("--policy-result",type=Path,required=True); p.add_argument("--hashfold-result",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--bootstrap-replicates",type=int,default=10000); p.add_argument("--seed",type=int,default=20261005); return p.parse_args()


def fast(selected:np.ndarray,absolute:np.ndarray)->dict:
    row=np.arange(len(selected)); chosen=absolute[row,selected]; inc=absolute[:,0]; delta=chosen-inc; promote=selected!=0; n=int(promote.sum()); d=int(((delta[promote,0]<-1e-7)&(delta[promote,3]<-1e-7)).sum()); h=int(((delta[promote,0]>1e-7)|(delta[promote,3]>1e-7)).sum()); return {"coverage":n/len(selected),"dual":d/n if n else None,"wilson":wilson_lower(d,n),"harm":h/n if n else None,"absolute":{name:{"mean_m":float(chosen[:,i].mean()),"p90_m":float(np.quantile(chosen[:,i],.9))} for i,name in enumerate(NAMES)}}


def passes(v:dict,b:dict)->bool:
    safety=v["coverage"]>=.40 and v["dual"] is not None and v["dual"]>=.90 and v["wilson"] is not None and v["wilson"]>=.80 and v["harm"] is not None and v["harm"]<=.10
    quality=all(v["absolute"][n]["mean_m"]<=b["absolute"][n]["mean_m"]+1e-9 for n in NAMES) and all(v["absolute"][n]["p90_m"]<=b["absolute"][n]["p90_m"]+1e-9 for n in ("ate","objective")) and v["absolute"]["ate"]["mean_m"]<b["absolute"]["ate"]["mean_m"]-1e-9 and v["absolute"]["objective"]["mean_m"]<b["absolute"]["objective"]["mean_m"]-1e-9
    return safety and quality


def main()->None:
    a=parse_args(); result_path=a.output_dir/"result.json"
    if result_path.exists(): raise FileExistsError(result_path)
    v32=json.loads(a.v32_result.read_text()); pred_path=Path(v32["prediction"]["path"])
    if file_sha256(pred_path)!=v32["prediction"]["sha256"]: raise RuntimeError("v32 prediction SHA mismatch")
    with np.load(pred_path,allow_pickle=False) as p: record=p["record_key"].astype(str); cohort=p["cohort"].astype(str); pg=p["p_good"].astype(np.float32); ph=p["p_harm"].astype(np.float32); valid=p["valid"].astype(bool)
    policy=json.loads(a.policy_result.read_text())["policy"]; hf=json.loads(a.hashfold_result.read_text()); bp={}
    for n,item in hf["predictions"].items():
        with np.load(Path(item["path"]),allow_pickle=False) as p: bp[n]={k:p[k].copy() for k in p.files}
    blend=float(policy["blend"]); risk=blend*bp["state_universal_support"]["p_dual"]+(1-blend)*bp["state_motion_universal_support"]["p_dual"]-float(policy["harm_weight"])*(blend*bp["state_universal_support"]["p_harm"]+(1-blend)*bp["state_motion_universal_support"]["p_harm"]); base_all=np.where(risk>float(policy["threshold"]),bp["state_universal_support"]["challenger_index"].astype(np.int64),0)
    banks=[]; rmse_all=[]; endpoint_all=[]; curve_all=[]
    for letter in "abcdef":
        name=f"bank_{letter}"; mask=cohort==name; bank=a.root/name
        with np.load(next((bank/"features").glob(f"recon_target_bank_{letter}_v*_features.npz")),allow_pickle=False) as p: base_path=p["candidates"].astype(np.float32)
        with np.load(bank/"cem_run_v30/iteration_02/cem_features.npz",allow_pickle=False) as p: cem_path=p["candidates"][:,1:].astype(np.float32)
        with np.load(bank/"cem_run_v30/development_outcomes.npz",allow_pickle=False) as p: union=p["union_absolute_metrics"].astype(np.float32); route=p["route_group"].astype(str); trajectory=p["trajectory"].astype(str)
        base=base_all[mask]; chosen=base_path[np.arange(100),base]; diff=cem_path-chosen[:,None]; rmse=np.sqrt(np.mean(diff*diff,axis=(2,3))); endpoint=np.linalg.norm(diff[:,:,-1],axis=2); base_curve=np.linalg.norm(np.diff(chosen,n=2,axis=1),axis=2).mean(1); cem_curve=np.linalg.norm(np.diff(cem_path,n=2,axis=2),axis=3).mean(2); curve=np.abs(cem_curve-base_curve[:,None]); banks.append({"name":name,"mask":mask,"base":base,"union":union,"route":route,"trajectory":trajectory,"rmse":rmse,"endpoint":endpoint,"curve":curve}); rmse_all.append(rmse); endpoint_all.append(endpoint); curve_all.append(curve)
    rmse=np.concatenate(rmse_all); endpoint=np.concatenate(endpoint_all); curve=np.concatenate(curve_all); score=np.where(valid,pg-.5*ph,-np.inf); best=score.argmax(1); row=np.arange(600); best_score=score[row,best]; br=rmse[row,best]; be=endpoint[row,best]; bc=curve[row,best]; finite=np.isfinite(best_score); thresholds=np.unique(np.quantile(best_score[finite],np.linspace(0,.95,20))); q=np.asarray((.30,.50,.70,.80,.90,.95,1.0)); rmse_caps=np.unique(np.quantile(br[finite],q)); endpoint_caps=np.unique(np.quantile(be[finite],q)); curve_caps=np.unique(np.quantile(bc[finite],q)); feasible=[]
    for t in thresholds:
      for rc in rmse_caps:
       for ec in endpoint_caps:
        for cc in curve_caps:
         accept=finite&(best_score>=t)&(br<=rc)&(be<=ec)&(bc<=cc); per={}; ok=True; improve=0.; total=0
         for b in banks:
          local=accept[b["mask"]]; selected=np.where(local,best[b["mask"]]+17,b["base"]); v=fast(selected,b["union"]); base=fast(b["base"],b["union"]); ok=ok and passes(v,base); improve+=base["absolute"]["objective"]["mean_m"]-v["absolute"]["objective"]["mean_m"]; total+=int(local.sum()); per[b["name"]]={"replacements":int(local.sum()),"fast":v}
         if ok: feasible.append({"threshold":float(t),"rmse_cap":float(rc),"endpoint_cap":float(ec),"curvature_change_cap":float(cc),"objective_improvement":improve,"total_replacements":total,"accept":accept,"banks":per})
    best_policy=max(feasible,key=lambda x:(x["objective_improvement"],x["total_replacements"])) if feasible else None; detailed=None
    a.output_dir.mkdir(parents=True,exist_ok=False)
    if best_policy:
      detailed={k:v for k,v in best_policy.items() if k not in("accept","banks")}; detailed["banks"]={}; selected_all=np.zeros(600,np.int64)
      for i,b in enumerate(banks):
       local=best_policy["accept"][b["mask"]]; selected=np.where(local,best[b["mask"]]+17,b["base"]); selected_all[b["mask"]]=selected; detailed["banks"][b["name"]]={"replacements":int(local.sum()),"base":method_metrics(b["base"],b["union"],b["trajectory"],b["route"],replicates=a.bootstrap_replicates,seed=a.seed+i*100),"cem_true":method_metrics(selected,b["union"],b["trajectory"],b["route"],replicates=a.bootstrap_replicates,seed=a.seed+i*100+10)}
      sp=a.output_dir/"selected_index_oof.npz"; write_npz(sp,record_key=record,cohort=cohort,selected_index=selected_all,replacement=selected_all>=17); detailed["selected_index_sha256"]=file_sha256(sp)
    report={"schema":"recon_target_cem_physical_guard_v35","status":"causal_physical_guard_calibration_complete","evidence_role":"development_only_six_bank_oof","cem":True,"feature_contract":"candidate-vs-base path RMSE, endpoint displacement, curvature change; current actions only","future_windows_used":False,"feasible_count":len(feasible),"best_policy":detailed,"formal_claim_allowed":False}; write_json(result_path,report); print(json.dumps({"feasible_count":len(feasible),"best":None if detailed is None else {k:v for k,v in detailed.items() if k!="banks"},"banks":None if detailed is None else {n:{"replacements":v["replacements"],"dual":v["cem_true"]["dual_precision"],"wilson":v["cem_true"]["dual_wilson_lower_95"],"harm":v["cem_true"]["harm_rate_per_promotion"],"absolute":v["cem_true"]["absolute"]} for n,v in detailed["banks"].items()}},indent=2,sort_keys=True))


if __name__=="__main__": main()
