#!/usr/bin/env python3
"""Whole-bank-OOF quantile regression for conservative CEM replacement."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


NAMES = ("ate", "rpe", "endpoint", "objective")


class QuantileModel(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 4))
    def forward(self, value: torch.Tensor) -> torch.Tensor: return self.net(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--policy-result", type=Path, required=True); parser.add_argument("--hashfold-result", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=240); parser.add_argument("--hidden-dim", type=int, default=64); parser.add_argument("--dropout", type=float, default=0.35); parser.add_argument("--learning-rate", type=float, default=3.0e-4); parser.add_argument("--weight-decay", type=float, default=1.0e-2); parser.add_argument("--quantile", type=float, default=0.80); parser.add_argument("--ensemble-members", type=int, default=3); parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda"); parser.add_argument("--bootstrap-replicates", type=int, default=10000); parser.add_argument("--seed", type=int, default=20261004)
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train(feature: np.ndarray, target: np.ndarray, group: np.ndarray, index: np.ndarray, args: argparse.Namespace, seed: int) -> tuple[QuantileModel, np.ndarray, np.ndarray]:
    seed_all(seed); mean = feature[index].mean(0).astype(np.float32); scale = np.maximum(feature[index].std(0), 1.0e-4).astype(np.float32); x = torch.from_numpy(((feature[index] - mean) / scale).astype(np.float32)).to(args.device); y = torch.from_numpy((target[index] * 20.0).astype(np.float32)).to(args.device); unique, gid = np.unique(group[index], return_inverse=True); gid = torch.from_numpy(gid).to(args.device); counts = torch.bincount(gid, minlength=len(unique)).float(); top = max(1, math.ceil(0.20 * len(unique))); model = QuantileModel(feature.shape[1], args.hidden_dim, args.dropout).to(args.device); optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay); q = float(args.quantile); weights = torch.tensor([2.0, 0.5, 0.5, 2.0], device=args.device)
    for _ in range(args.epochs):
        pred = model(x); error = y - pred; loss_metric = torch.maximum(q * error, (q - 1.0) * error) * weights; state_loss = loss_metric.sum(1); sums = torch.zeros(len(unique), device=args.device).scatter_add_(0, gid, state_loss); group_loss = sums / counts; loss = 0.5 * group_loss.mean() + 0.5 * torch.topk(group_loss, top).values.mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    model.eval(); return model, mean, scale


def predict(model: QuantileModel, mean: np.ndarray, scale: np.ndarray, value: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode(): return (model(torch.from_numpy(((value - mean) / scale).astype(np.float32)).to(device)).cpu().numpy() / 20.0).astype(np.float32)


def fast(selected: np.ndarray, absolute: np.ndarray) -> dict:
    row = np.arange(len(selected)); chosen = absolute[row, selected]; inc = absolute[:, 0]; delta = chosen - inc; promote = selected != 0; count = int(promote.sum()); dual = int(((delta[promote, 0] < -1e-7) & (delta[promote, 3] < -1e-7)).sum()); harm = int(((delta[promote, 0] > 1e-7) | (delta[promote, 3] > 1e-7)).sum()); return {"coverage": count / len(selected), "dual": dual / count if count else None, "wilson": wilson_lower(dual, count), "harm": harm / count if count else None, "absolute": {name: {"mean_m": float(chosen[:, i].mean()), "p90_m": float(np.quantile(chosen[:, i], .9))} for i, name in enumerate(NAMES)}}


def passes(value: dict, base: dict) -> bool:
    safety = value["coverage"] >= .40 and value["dual"] is not None and value["dual"] >= .90 and value["wilson"] is not None and value["wilson"] >= .80 and value["harm"] is not None and value["harm"] <= .10
    quality = all(value["absolute"][name]["mean_m"] <= base["absolute"][name]["mean_m"] + 1e-9 for name in NAMES) and all(value["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1e-9 for name in ("ate", "objective")) and value["absolute"]["ate"]["mean_m"] < base["absolute"]["ate"]["mean_m"] - 1e-9 and value["absolute"]["objective"]["mean_m"] < base["absolute"]["objective"]["mean_m"] - 1e-9
    return safety and quality


def main() -> None:
    args = parse_args(); result_path = args.output_dir / "result.json"
    if result_path.exists(): raise FileExistsError(result_path)
    policy = json.loads(args.policy_result.read_text(encoding="utf-8"))["policy"]; hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8")); bp = {}
    for name, item in hashfold["predictions"].items():
        with np.load(Path(item["path"]), allow_pickle=False) as payload: bp[name] = {key: payload[key].copy() for key in payload.files}
    blend = float(policy["blend"]); risk = blend * bp["state_universal_support"]["p_dual"] + (1-blend) * bp["state_motion_universal_support"]["p_dual"] - float(policy["harm_weight"]) * (blend * bp["state_universal_support"]["p_harm"] + (1-blend) * bp["state_motion_universal_support"]["p_harm"]); base_selected_all = np.where(risk > float(policy["threshold"]), bp["state_universal_support"]["challenger_index"].astype(np.int64), 0)
    banks = []
    for letter in "abcdef":
        name = f"bank_{letter}"; bank = args.root / name; original_feature = next((bank / "features").glob(f"recon_target_bank_{letter}_v*_features.npz")); base_compact, _ = compact_base_feature(original_feature, bank / "universal_scores.npz", np.empty((100,0), np.float32)); cem_compact, _ = compact_base_feature(bank / "cem_run_v30/iteration_02/cem_features.npz", bank / "cem_run_v30/iteration_02/frozen_scores_record_v31.npz", np.empty((100,0), np.float32))
        with np.load(bank / "cem_run_v30/iteration_02/frozen_scores_record_v31.npz", allow_pickle=False) as payload: record = payload["record_key"].astype(str); valid = payload["valid"][:,1:].astype(bool); energy = np.nan_to_num(payload["energy"][:,1:].astype(np.float32), posinf=10.0, neginf=-10.0)
        with np.load(bank / "cem_run_v30/development_outcomes.npz", allow_pickle=False) as payload: union = payload["union_absolute_metrics"].astype(np.float32); base_abs = payload["base_absolute_metrics"].astype(np.float32); cem_abs = payload["cem_absolute_metrics"].astype(np.float32); route = payload["route_group"].astype(str); trajectory = payload["trajectory"].astype(str)
        mask = bp["state_universal_support"]["cohort"].astype(str) == name; base_selected = base_selected_all[mask]; promoted = base_selected != 0; slot = np.maximum(base_selected - 1, 0); base_motion = base_compact[np.arange(100), slot, 434:486]; cem_motion = cem_compact[:,:,434:486]; feature = np.concatenate((cem_motion, np.repeat(base_motion[:,None],16,1), cem_motion-base_motion[:,None], np.repeat(risk[mask,None,None],16,1), energy[:,:,None]),2).astype(np.float32); base_value = base_abs[np.arange(100), base_selected]; target = cem_abs[:,1:] - base_value[:,None]; banks.append({"name":name,"record":record,"feature":feature,"target":target,"valid":valid & promoted[:,None],"route":route,"trajectory":trajectory,"union":union,"base":base_selected})
    feature = np.concatenate([b["feature"] for b in banks]).reshape(-1,158); target = np.concatenate([b["target"] for b in banks]).reshape(-1,4); valid = np.concatenate([b["valid"] for b in banks]); group_state = np.concatenate([np.asarray([f"{b['name']}::{r}" for r in b["route"]]) for b in banks]); cohort_state = np.concatenate([np.full(100,b["name"],dtype="<U16") for b in banks]); group = np.repeat(group_state,16); cohort=np.repeat(cohort_state,16); prediction=np.zeros_like(target); pred_std=np.zeros_like(target); checkpoints={}; args.output_dir.mkdir(parents=True,exist_ok=False)
    for fold, heldout in enumerate(np.unique(cohort)):
        train_index=np.flatnonzero(cohort!=heldout); heldout_index=np.flatnonzero(cohort==heldout); members=[]; checkpoints[heldout]=[]
        for member in range(args.ensemble_members):
            seed=args.seed+fold*100000+member*1009; model,mean,scale=train(feature,target,group,train_index,args,seed); members.append(predict(model,mean,scale,feature[heldout_index],args.device)); path=args.output_dir/f"quantile_exclude_{heldout}_member_{member}.pt"; torch.save({"schema":"recon_target_cem_quantile_v34_member","excluded_bank":heldout,"input_dim":158,"hidden_dim":args.hidden_dim,"dropout":args.dropout,"quantile":args.quantile,"feature_mean":mean,"feature_scale":scale,"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"seed":seed},path); checkpoints[heldout].append({"path":str(path),"sha256":file_sha256(path),"seed":seed}); del model
        stack=np.stack(members); prediction[heldout_index]=stack.mean(0); pred_std[heldout_index]=stack.std(0)
    prediction=prediction.reshape(600,16,4); pred_std=pred_std.reshape(600,16,4); candidates=[]
    for z in (0.0,.5,1.0,1.5,2.0):
        upper=prediction+z*pred_std; utility=upper[...,0]+upper[...,3]+.25*upper[...,1]+.25*upper[...,2]; utility=np.where(valid,utility,np.inf); best=utility.argmin(1); row=np.arange(600); best_upper=upper[row,best]
        for margin in (0.0,.001,.002,.005,.010):
            for tolerance in (0.0,.001,.002,.005,.010,.020):
                accept=(best_upper[:,0] < -margin)&(best_upper[:,3] < -margin)&(best_upper[:,1] <= tolerance)&(best_upper[:,2] <= tolerance)&np.isfinite(utility[row,best]); per={}; all_pass=True; improvement=0.; total=0
                offset=0
                for b in banks:
                    local=accept[offset:offset+100]; selected=np.where(local,best[offset:offset+100]+17,b["base"]); value=fast(selected,b["union"]); base_value=fast(b["base"],b["union"]); all_pass=all_pass and passes(value,base_value); improvement += base_value["absolute"]["objective"]["mean_m"]-value["absolute"]["objective"]["mean_m"]; total+=int(local.sum()); per[b["name"]]={"replacements":int(local.sum()),"fast":value}; offset+=100
                if all_pass:candidates.append({"z":z,"margin":margin,"tolerance":tolerance,"objective_improvement":improvement,"total_replacements":total,"banks":per,"best":best,"accept":accept})
    best_policy=max(candidates,key=lambda x:(x["objective_improvement"],x["total_replacements"])) if candidates else None; detailed=None
    if best_policy:
        detailed={k:v for k,v in best_policy.items() if k not in("best","accept","banks")}; detailed["banks"]={}; selected_all=np.zeros(600,np.int64); offset=0
        for number,b in enumerate(banks):
            local=best_policy["accept"][offset:offset+100]; selected=np.where(local,best_policy["best"][offset:offset+100]+17,b["base"]); selected_all[offset:offset+100]=selected; detailed["banks"][b["name"]]={"replacements":int(local.sum()),"base":method_metrics(b["base"],b["union"],b["trajectory"],b["route"],replicates=args.bootstrap_replicates,seed=args.seed+number*100),"cem_true":method_metrics(selected,b["union"],b["trajectory"],b["route"],replicates=args.bootstrap_replicates,seed=args.seed+number*100+10)}; offset+=100
        selected_path=args.output_dir/"selected_index_oof.npz"; write_npz(selected_path,record_key=np.concatenate([b["record"] for b in banks]),cohort=cohort_state,selected_index=selected_all,replacement=selected_all>=17); detailed["selected_index_sha256"]=file_sha256(selected_path)
    pred_path=args.output_dir/"quantile_oof.npz"; write_npz(pred_path,record_key=np.concatenate([b["record"] for b in banks]),cohort=cohort_state,prediction=prediction,prediction_std=pred_std,valid=valid); report={"schema":"recon_target_cem_quantile_v34","status":"whole_bank_oof_quantile_replacement_complete","evidence_role":"development_only","cem":True,"quantile":args.quantile,"feature_contract":"candidate/base motion-universal differences plus base risk and CEM energy","checkpoints":checkpoints,"prediction":{"path":str(pred_path),"sha256":file_sha256(pred_path)},"feasible_count":len(candidates),"best_policy":detailed,"formal_claim_allowed":False}; write_json(result_path,report); print(json.dumps({"feasible_count":len(candidates),"best":None if detailed is None else {k:v for k,v in detailed.items() if k!="banks"},"banks":None if detailed is None else {n:{"replacements":v["replacements"],"dual":v["cem_true"]["dual_precision"],"wilson":v["cem_true"]["dual_wilson_lower_95"],"harm":v["cem_true"]["harm_rate_per_promotion"],"absolute":v["cem_true"]["absolute"]} for n,v in detailed["banks"].items()}},indent=2,sort_keys=True))


if __name__=="__main__": main()
