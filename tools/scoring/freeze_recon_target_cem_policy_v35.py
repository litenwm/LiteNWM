#!/usr/bin/env python3
"""Freeze v35 real-CEM deployment policy before a new confirmation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--v35-result",type=Path,required=True); p.add_argument("--v32-result",type=Path,required=True); p.add_argument("--base-policy",type=Path,required=True); p.add_argument("--qualification-result",type=Path,required=True); p.add_argument("--output",type=Path,required=True); return p.parse_args()


def main()->None:
    a=parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    v35=json.loads(a.v35_result.read_text()); v32=json.loads(a.v32_result.read_text()); base=json.loads(a.base_policy.read_text()); qualification=json.loads(a.qualification_result.read_text())
    if v35.get("feasible_count",0)<=0 or v35.get("best_policy") is None: raise RuntimeError("v35 has no feasible policy")
    checks=qualification["methods"]["hashfold_v29_cem_false"]["deployment_checks"]
    if not checks["all_safety_constraints_pass"]: raise RuntimeError("RECON fixed-pool qualification did not pass")
    checkpoint_audit={}
    for fold,members in v32["checkpoints"].items():
        checkpoint_audit[fold]=[]
        for item in members:
            path=Path(item["path"])
            if file_sha256(path)!=item["sha256"]: raise RuntimeError(f"v32 checkpoint SHA mismatch: {path}")
            checkpoint_audit[fold].append(item)
    best=v35["best_policy"]
    policy={key:best[key] for key in ("threshold","rmse_cap","endpoint_cap","curvature_change_cap")}
    policy.update({"candidate_score":"p_good - 0.5*p_harm","replacement_only_when_base_promotes":True,"strict_threshold":False})
    report={"schema":"recon_target_cem_policy_v35","status":"real_cem_policy_frozen_before_new_confirmation","cem":True,"generator":{"iterations":3,"population":16,"elite_fraction":0.25,"proposal_scale":0.70,"support_quantile":0.999},"base_policy":str(a.base_policy),"base_policy_sha256":file_sha256(a.base_policy),"fixed_pool_qualification":str(a.qualification_result),"fixed_pool_qualification_sha256":file_sha256(a.qualification_result),"replacement_policy":policy,"replacement_checkpoints":checkpoint_audit,"replacement_router":{"algorithm":"int.from_bytes(sha256(route_group).digest()[:8], 'big') modulo 6","fold_order":["bank_a","bank_b","bank_c","bank_d","bank_e","bank_f"],"uses_outcome":False,"uses_future_windows":False},"development_result":str(a.v35_result),"development_result_sha256":file_sha256(a.v35_result),"development_feasible_count":v35["feasible_count"],"development_selected_index_sha256":best["selected_index_sha256"],"new_confirmation_accessed":False,"labels_opened":False,"outcomes_accessed":False,"formal_claim_allowed":False}
    a.output.parent.mkdir(parents=True,exist_ok=True); write_json(a.output,report); print(json.dumps(report,indent=2,sort_keys=True))


if __name__=="__main__": main()
