#!/usr/bin/env python3
"""Open and evaluate the prelabel-frozen v35 real-CEM confirmation once."""

from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(REPO_ROOT))
from tools.evaluation.build_recon100_k17_source_v1 import metric_labels
from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256,write_json,write_npz


NAMES=("ate","rpe","endpoint","objective")


def parse()->argparse.Namespace:
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--selection-freeze",type=Path,required=True);p.add_argument("--trajectory-root",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--bootstrap-replicates",type=int,default=10000);p.add_argument("--seed",type=int,default=20261006);return p.parse_args()


def main()->None:
 a=parse();result_path=a.output_dir/"result.json";labeled=a.root/"source_labeled.npz";audit_path=a.root/"outcome_open.audit.json"
 if result_path.exists() or labeled.exists() or audit_path.exists():raise FileExistsError(result_path)
 freeze=json.loads(a.selection_freeze.read_text())
 if freeze["status"]!="cem_true_and_false_selected_before_confirmation_outcomes" or freeze.get("labels_opened") or freeze.get("outcomes_accessed") or not freeze.get("cem"):raise RuntimeError("selection freeze invalid")
 sp=Path(freeze["selected_index"])
 if file_sha256(sp)!=freeze["artifacts"]["selected_index_sha256"]:raise RuntimeError("selection changed after freeze")
 manifest=a.root/"manifest.windows.jsonl";source=a.root/"source_label_free.npz";rows=[json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
 if file_sha256(manifest)!=freeze["artifacts"]["manifest_sha256"] or file_sha256(source)!=freeze["artifacts"]["source_sha256"]:raise RuntimeError("confirmation artifacts changed")
 with np.load(source,allow_pickle=False) as p:arrays={k:p[k].copy() for k in p.files};record=p["record_key"].astype(str);trajectory=p["trajectory"].astype(str);route=p["route_group"].astype(str);base_candidates=p["candidates"].astype(np.float32)
 with np.load(a.root/"cem_run/selected_actions.npz",allow_pickle=False) as p:cem_candidates=p["final_cem_feature_waypoint_units"].astype(np.float32)
 with np.load(sp,allow_pickle=False) as p:
  if not np.array_equal(p["record_key"].astype(str),record):raise RuntimeError("selection identity mismatch")
  base_selected=p["base_selected_index"].astype(np.int64);cem_selected=p["cem_selected_union_index"].astype(np.int64);native=p["native_scorer_v2_selected_index"].astype(np.int64)
 ground=np.stack([local_ground_truth(a.trajectory_root/str(r["trajectory_id"])/"traj_data.pkl",int(r["current_t"])) for r in rows]).astype(np.float32);base_labels=metric_labels(base_candidates,ground,.25,.4,.6);cem_labels=metric_labels(cem_candidates,ground,.25,.4,.6);base_abs=base_labels["values"].astype(np.float32);cem_abs=cem_labels["values"].astype(np.float32);union=np.concatenate((base_abs,cem_abs[:,1:]),1);write_npz(labeled,**arrays,targets=base_labels["targets"],absolute_metrics=base_abs,ground_truth_local_m=ground)
 row=np.arange(100);base_obj=base_abs[row,base_selected,3];best=cem_abs[:,1:,3].argmin(1)+1;oracle=np.where(cem_abs[row,best,3]<base_obj,best+16,base_selected);indices={"incumbent":np.zeros(100,np.int64),"native_scorer_v2":native,"hashfold_v29_cem_false":base_selected,"physical_guard_v35_cem_true":cem_selected,"posthoc_cem_replacement_oracle":oracle};methods={name:method_metrics(index,union,trajectory,route,replicates=a.bootstrap_replicates,seed=a.seed+i*10) for i,(name,index) in enumerate(indices.items())};b=methods["hashfold_v29_cem_false"];c=methods["physical_guard_v35_cem_true"]
 safety={"coverage_at_least_0.40":c["promotion_fraction"]>=.40,"dual_at_least_0.90":c["dual_precision"] is not None and c["dual_precision"]>=.90,"wilson_at_least_0.80":c["dual_wilson_lower_95"] is not None and c["dual_wilson_lower_95"]>=.80,"harm_at_most_0.10":c["harm_rate_per_promotion"] is not None and c["harm_rate_per_promotion"]<=.10,"dual_not_below_cem_false":c["dual_precision"]>=b["dual_precision"]-1e-12,"wilson_not_below_cem_false":c["dual_wilson_lower_95"]>=b["dual_wilson_lower_95"]-1e-12,"harm_not_above_cem_false":c["harm_rate_per_promotion"]<=b["harm_rate_per_promotion"]+1e-12};quality={"all_means_not_worse_than_cem_false":all(c["absolute"][n]["mean_m"]<=b["absolute"][n]["mean_m"]+1e-9 for n in NAMES),"all_p90_not_worse_than_cem_false":all(c["absolute"][n]["p90_m"]<=b["absolute"][n]["p90_m"]+1e-9 for n in NAMES),"ate_mean_strictly_better":c["absolute"]["ate"]["mean_m"]<b["absolute"]["ate"]["mean_m"]-1e-9,"objective_mean_strictly_better":c["absolute"]["objective"]["mean_m"]<b["absolute"]["objective"]["mean_m"]-1e-9};checks={**safety,**quality,"all_safety_pass":all(safety.values()),"all_quality_pass":all(quality.values()),"development_confirmation_good_result":all(safety.values()) and all(quality.values())};audit={"schema":"recon_target_cem_confirmation_v35_outcome_open","status":"outcomes_opened_once_after_cem_true_false_freeze","opened_at":datetime.now(timezone.utc).isoformat(),"selection_freeze_sha256":file_sha256(a.selection_freeze),"selected_index_sha256":file_sha256(sp),"source_label_free_sha256":file_sha256(source),"source_labeled_sha256":file_sha256(labeled),"labels_opened":True,"outcomes_accessed":True,"cem":True};write_json(audit_path,audit);a.output_dir.mkdir(parents=True,exist_ok=False);report={"schema":"recon_target_cem_confirmation_v35_result","status":"fresh_route_disjoint_real_cem_confirmation_evaluated","evidence_role":"fresh RECON H4-100 real-CEM confirmation","states":100,"routes":25,"trajectories":25,"cem":True,"methods":methods,"cem_true_vs_false_checks":checks,"selection_freeze":str(a.selection_freeze),"selection_freeze_sha256":file_sha256(a.selection_freeze),"selected_index_sha256":file_sha256(sp),"labeled_source_sha256":file_sha256(labeled),"bootstrap_replicates":a.bootstrap_replicates,"formal_recon_claim_allowed":checks["development_confirmation_good_result"]};write_json(result_path,report);print(json.dumps({"checks":checks,"methods":{n:{"promotions":v["promotions"],"coverage":v["promotion_fraction"],"dual":v["dual_precision"],"wilson":v["dual_wilson_lower_95"],"harm":v["harm_rate_per_promotion"],"absolute":v["absolute"]} for n,v in methods.items()}},indent=2,sort_keys=True))


if __name__=="__main__":main()
