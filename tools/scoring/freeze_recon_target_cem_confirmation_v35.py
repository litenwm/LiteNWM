#!/usr/bin/env python3
"""Freeze v35 CEM/base selections on a still-unlabeled confirmation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO_ROOT))
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_state_gate_v24 import StateRisk,predict
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256,write_json,write_npz


def args()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--feature-name",required=True);p.add_argument("--policy-freeze",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--device",choices=("cuda","cpu"),default="cuda");return p.parse_args()


def fold(route:str,n:int)->int:return int.from_bytes(hashlib.sha256(route.encode()).digest()[:8],"big")%n

def compact(feature_path:Path,score_path:Path)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
 with np.load(feature_path,allow_pickle=False) as p: static=p["static_features"][:,1:].astype(np.float32);raw=p["raw_ltwm_features"].astype(np.float32);action=p["actions_h4"].astype(np.float32);candidate=p["candidates"].astype(np.float32);support=p["majority_support"][:,1:].astype(bool);record=p["record_key"].astype(str)
 with np.load(score_path,allow_pickle=False) as p:
  score_record=p["record_key"].astype(str) if "record_key" in p.files else record
  if not np.array_equal(record,score_record):raise RuntimeError("feature/score identity mismatch")
  universal=np.concatenate((p["mean"][:,1:].astype(np.float32),p["std"][:,1:].astype(np.float32),p["dual_probability"][:,1:,None].astype(np.float32),p["harm_probability"][:,1:,None].astype(np.float32),p["rank_logit"][:,1:,None].astype(np.float32)),2)
  valid=p["valid"][:,1:].astype(bool) if "valid" in p.files else support
 ac=action[:,1:];ad=ac-action[:,:1];pc=candidate[:,1:];pd=pc-candidate[:,:1];feature=np.concatenate((static,raw[:,1:],raw[:,1:]-raw[:,:1],ac.reshape(len(record),16,-1),ad.reshape(len(record),16,-1),pc.reshape(len(record),16,-1),pd.reshape(len(record),16,-1),universal,support[...,None].astype(np.float32)),2).astype(np.float32);return feature,valid,record,candidate


def ensemble(entries:list[dict],feature:np.ndarray,device:str)->tuple[np.ndarray,np.ndarray]:
 ds=[];hs=[]
 for item in entries:
  path=Path(item["path"])
  if file_sha256(path)!=item["sha256"]:raise RuntimeError(f"checkpoint SHA mismatch: {path}")
  p=torch.load(path,map_location="cpu",weights_only=False);model=StateRisk(int(p["input_dim"]),int(p["hidden_dim"]),float(p["dropout"])).to(device);model.load_state_dict(p["model"],strict=True);model.eval();d,h=predict(model,np.asarray(p["feature_mean"],np.float32),np.asarray(p["feature_scale"],np.float32),feature,device);ds.append(d);hs.append(h);del model
 return np.stack(ds).mean(0),np.stack(hs).mean(0)


def main()->None:
 a=args();result_path=a.output_dir/"freeze.json"
 if result_path.exists():raise FileExistsError(result_path)
 if (a.root/"source_labeled.npz").exists():raise RuntimeError("confirmation outcomes already opened")
 cem_policy=json.loads(a.policy_freeze.read_text());base_policy_path=Path(cem_policy["base_policy"]);base_policy=json.loads(base_policy_path.read_text());order=cem_policy["replacement_router"]["fold_order"]
 base_feature=a.root/"features"/a.feature_name;base_score=a.root/"universal_scores.npz";cem_feature=a.root/"cem_run/iteration_02/cem_features.npz";cem_score=a.root/"cem_run/iteration_02/frozen_scores.npz"
 bf,bvalid,record,base_paths=compact(base_feature,base_score);cf,cvalid,cem_record,cem_paths=compact(cem_feature,cem_score)
 with np.load(a.root/"source_label_free.npz",allow_pickle=False) as p:route=p["route_group"].astype(str);trajectory=p["trajectory"].astype(str)
 if not np.array_equal(record,cem_record):raise RuntimeError("base/CEM identity mismatch")
 assigned=np.asarray([fold(r,len(order)) for r in route],np.int64)
 with np.load(base_score,allow_pickle=False) as p:score={"std":p["std"][:,1:].astype(np.float32),"dual":p["dual_probability"][:,1:].astype(np.float32),"harm":p["harm_probability"][:,1:].astype(np.float32)}
 challenger=frozen_challenger(score,bvalid);chosen=bf[np.arange(100),challenger];variants={"state_universal_support":chosen[:,474:486],"state_motion_universal_support":chosen[:,434:486]};prob={}
 for name,feature in variants.items():
  pd=np.zeros(100,np.float32);ph=np.zeros(100,np.float32)
  for fi,fn in enumerate(order):
   idx=np.flatnonzero(assigned==fi)
   if len(idx):pd[idx],ph[idx]=ensemble(base_policy["checkpoints"][name][fn],feature[idx],a.device)
  prob[name]=(pd,ph)
 bp=base_policy["policy"];blend=float(bp["blend"]);pdual=blend*prob["state_universal_support"][0]+(1-blend)*prob["state_motion_universal_support"][0];pharm=blend*prob["state_universal_support"][1]+(1-blend)*prob["state_motion_universal_support"][1];base_risk=pdual-float(bp["harm_weight"])*pharm;base_selected=np.where(base_risk>float(bp["threshold"]),challenger+1,0).astype(np.int64)
 with np.load(cem_score,allow_pickle=False) as p:energy=np.nan_to_num(p["energy"][:,1:].astype(np.float32),posinf=10.,neginf=-10.)
 slot=np.maximum(base_selected-1,0);base_motion=bf[np.arange(100),slot,434:486];cem_motion=cf[:,:,434:486];replacement_feature=np.concatenate((cem_motion,np.repeat(base_motion[:,None],16,1),cem_motion-base_motion[:,None],np.repeat(base_risk[:,None,None],16,1),energy[:,:,None]),2).astype(np.float32);pg=np.zeros((100,16),np.float32);ph=np.zeros_like(pg)
 for fi,fn in enumerate(order):
  idx=np.flatnonzero(assigned==fi)
  if len(idx):
   d,h=ensemble(cem_policy["replacement_checkpoints"][fn],replacement_feature[idx].reshape(-1,158),a.device);pg[idx]=d.reshape(len(idx),16);ph[idx]=h.reshape(len(idx),16)
 score=np.where(cvalid,pg-.5*ph,-np.inf);best=score.argmax(1);best_score=score[np.arange(100),best];base_chosen_path=base_paths[np.arange(100),base_selected];cem_challenger_path=cem_paths[np.arange(100),best+1];diff=cem_challenger_path-base_chosen_path;rmse=np.sqrt(np.mean(diff*diff,axis=(1,2)));endpoint=np.linalg.norm(diff[:,-1],axis=1);bc=np.linalg.norm(np.diff(base_chosen_path,n=2,axis=1),axis=2).mean(1);cc=np.linalg.norm(np.diff(cem_challenger_path,n=2,axis=1),axis=2).mean(1);curve=np.abs(cc-bc);rp=cem_policy["replacement_policy"];replace=(base_selected!=0)&np.isfinite(best_score)&(best_score>=float(rp["threshold"]))&(rmse<=float(rp["rmse_cap"]))&(endpoint<=float(rp["endpoint_cap"]))&(curve<=float(rp["curvature_change_cap"]));cem_selected=np.where(replace,best+17,base_selected).astype(np.int64)
 with np.load(a.root/"cem_run/selected_actions.npz",allow_pickle=False) as p:native=p["native_scorer_v2_selected_index"].astype(np.int64)
 a.output_dir.mkdir(parents=True,exist_ok=False);sp=a.output_dir/"selected_index.npz";write_npz(sp,record_key=record,trajectory=trajectory,route_group=route,assigned_fold_index=assigned,base_selected_index=base_selected,cem_selected_union_index=cem_selected,native_scorer_v2_selected_index=native,replacement=replace,cem_challenger_index=best+1,base_risk=base_risk,replacement_score=best_score,path_rmse=rmse,endpoint_displacement=endpoint,curvature_change=curve)
 report={"schema":"recon_target_cem_confirmation_v35_selection_freeze","status":"cem_true_and_false_selected_before_confirmation_outcomes","states":100,"routes":25,"cem":True,"base_promotions_label_blind":int((base_selected!=0).sum()),"base_coverage_label_blind":float((base_selected!=0).mean()),"cem_promotions_label_blind":int((cem_selected!=0).sum()),"cem_coverage_label_blind":float((cem_selected!=0).mean()),"cem_replacements_label_blind":int(replace.sum()),"fold_state_counts":{order[i]:int((assigned==i).sum()) for i in range(6)},"policy_freeze":str(a.policy_freeze),"policy_freeze_sha256":file_sha256(a.policy_freeze),"artifacts":{"manifest_sha256":file_sha256(a.root/"manifest.windows.jsonl"),"k17_sha256":file_sha256(a.root/"k17.npz"),"source_sha256":file_sha256(a.root/"source_label_free.npz"),"base_features_sha256":file_sha256(base_feature),"base_scores_sha256":file_sha256(base_score),"cem_freeze_sha256":file_sha256(a.root/"cem_run/prelabel_freeze.json"),"cem_features_sha256":file_sha256(cem_feature),"cem_scores_sha256":file_sha256(cem_score),"selected_index_sha256":file_sha256(sp)},"selected_index":str(sp),"online_causal":True,"future_route_quota":False,"labels_opened":False,"outcomes_accessed":False};write_json(result_path,report);print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__":main()
