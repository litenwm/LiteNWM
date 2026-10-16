#!/usr/bin/env python3
"""Train whole-bank-OOF candidate-level safe CEM replacement models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.screen_recon_target_state_gate_v24 import predict
from tools.scoring.train_recon_target_state_gate_hashfold_v29 import train_fast
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--hashfold-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261002)
    return parser.parse_args()


def quality(candidate: dict, base: dict) -> dict[str, bool]:
    names = ("ate", "rpe", "endpoint", "objective")
    mean = [candidate["absolute"][name]["mean_m"] <= base["absolute"][name]["mean_m"] + 1.0e-9 for name in names]
    p90 = [candidate["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1.0e-9 for name in names]
    primary_strict = candidate["absolute"]["ate"]["mean_m"] < base["absolute"]["ate"]["mean_m"] - 1.0e-9 and candidate["absolute"]["objective"]["mean_m"] < base["absolute"]["objective"]["mean_m"] - 1.0e-9
    return {"all_means_nondegraded": all(mean), "all_p90_nondegraded": all(p90), "ate_objective_mean_strict": primary_strict, "strict_all_metric_pass": all(mean) and all(p90) and primary_strict}


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    policy_result = json.loads(args.policy_result.read_text(encoding="utf-8"))
    policy = policy_result["policy"]
    hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8"))
    base_pred = {}
    for name, item in hashfold["predictions"].items():
        with np.load(Path(item["path"]), allow_pickle=False) as payload:
            base_pred[name] = {key: payload[key].copy() for key in payload.files}
    blend = float(policy["blend"])
    base_risk_all = blend * base_pred["state_universal_support"]["p_dual"] + (1.0 - blend) * base_pred["state_motion_universal_support"]["p_dual"] - float(policy["harm_weight"]) * (blend * base_pred["state_universal_support"]["p_harm"] + (1.0 - blend) * base_pred["state_motion_universal_support"]["p_harm"])
    base_challenger_all = base_pred["state_universal_support"]["challenger_index"].astype(np.int64)
    base_selected_all = np.where(base_risk_all > float(policy["threshold"]), base_challenger_all, 0).astype(np.int64)

    banks = []
    for letter in "abcdef":
        name = f"bank_{letter}"
        bank = args.root / name
        original_feature = next((bank / "features").glob(f"recon_target_bank_{letter}_v*_features.npz"))
        original_score = bank / "universal_scores.npz"
        base_compact, _ = compact_base_feature(original_feature, original_score, np.empty((100, 0), dtype=np.float32))
        cem_feature = bank / "cem_run_v30/iteration_02/cem_features.npz"
        cem_score = bank / "cem_run_v30/iteration_02/frozen_scores_record_v31.npz"
        cem_compact, _ = compact_base_feature(cem_feature, cem_score, np.empty((100, 0), dtype=np.float32))
        with np.load(cem_score, allow_pickle=False) as payload:
            record = payload["record_key"].astype(str)
            valid = payload["valid"][:, 1:].astype(bool)
            energy = payload["energy"][:, 1:].astype(np.float32)
        with np.load(bank / "cem_run_v30/development_outcomes.npz", allow_pickle=False) as payload:
            union = payload["union_absolute_metrics"].astype(np.float32)
            base_absolute = payload["base_absolute_metrics"].astype(np.float32)
            cem_absolute = payload["cem_absolute_metrics"].astype(np.float32)
            route = payload["route_group"].astype(str)
            trajectory = payload["trajectory"].astype(str)
        cohort = base_pred["state_universal_support"]["cohort"].astype(str) == name
        if cohort.sum() != 100 or not np.array_equal(base_pred["state_universal_support"]["record_key"].astype(str)[cohort], record):
            raise RuntimeError(f"base/CEM identity mismatch: {name}")
        base_selected = base_selected_all[cohort]
        base_risk = base_risk_all[cohort]
        promoted = base_selected != 0
        base_slot = np.maximum(base_selected - 1, 0)
        base_chosen = base_compact[np.arange(100), base_slot, 434:486]
        cem_motion = cem_compact[:, :, 434:486]
        feature = np.concatenate((cem_motion, np.repeat(base_chosen[:, None], 16, axis=1), cem_motion - base_chosen[:, None], np.repeat(base_risk[:, None, None], 16, axis=1), energy[:, :, None]), axis=2).astype(np.float32)
        row = np.arange(100)
        base_value = base_absolute[row, base_selected]
        cem_value = cem_absolute[:, 1:]
        dual_inc = (cem_value[..., 0] < base_absolute[:, :1, 0] - 1.0e-7) & (cem_value[..., 3] < base_absolute[:, :1, 3] - 1.0e-7)
        dominates = (cem_value[..., 0] <= base_value[:, None, 0] + 1.0e-7) & (cem_value[..., 3] <= base_value[:, None, 3] + 1.0e-7) & ((cem_value[..., 0] < base_value[:, None, 0] - 1.0e-7) | (cem_value[..., 3] < base_value[:, None, 3] - 1.0e-7))
        good = dual_inc & dominates & promoted[:, None] & valid
        harm = ((cem_value[..., 0] > base_absolute[:, :1, 0] + 1.0e-7) | (cem_value[..., 3] > base_absolute[:, :1, 3] + 1.0e-7)) & promoted[:, None]
        banks.append({"name": name, "record": record, "feature": feature, "valid": valid & promoted[:, None], "good": good.astype(np.float32), "harm": harm.astype(np.float32), "route": route, "trajectory": trajectory, "union": union, "base_selected": base_selected})

    state_feature = np.concatenate([bank["feature"] for bank in banks])
    valid = np.concatenate([bank["valid"] for bank in banks])
    good = np.concatenate([bank["good"] for bank in banks])
    harm = np.concatenate([bank["harm"] for bank in banks])
    group_state = np.concatenate([np.asarray([f"{bank['name']}::{value}" for value in bank["route"]]) for bank in banks])
    cohort_state = np.concatenate([np.full(100, bank["name"], dtype="<U16") for bank in banks])
    flat_feature = state_feature.reshape(-1, state_feature.shape[-1])
    flat_good = good.reshape(-1)
    flat_harm = harm.reshape(-1)
    flat_group = np.repeat(group_state, 16)
    flat_cohort = np.repeat(cohort_state, 16)
    p_good = np.zeros_like(flat_good)
    p_harm = np.zeros_like(flat_harm)
    p_good_std = np.zeros_like(flat_good)
    p_harm_std = np.zeros_like(flat_harm)
    checkpoints = {}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for fold_number, heldout in enumerate(np.unique(flat_cohort)):
        train_index = np.flatnonzero(flat_cohort != heldout)
        heldout_index = np.flatnonzero(flat_cohort == heldout)
        member_good, member_harm = [], []
        checkpoints[heldout] = []
        for member in range(args.ensemble_members):
            seed = args.seed + fold_number * 100000 + member * 1009
            model, mean, scale = train_fast(flat_feature, flat_good, flat_harm, flat_group, train_index, args, seed)
            pred = predict(model, mean, scale, flat_feature[heldout_index], args.device)
            member_good.append(pred[0]); member_harm.append(pred[1])
            path = args.output_dir / f"candidate_replacement_exclude_{heldout}_member_{member}.pt"
            torch.save({"schema": "recon_target_cem_replacement_v32_member", "excluded_bank": heldout, "input_dim": flat_feature.shape[1], "hidden_dim": args.hidden_dim, "dropout": args.dropout, "feature_mean": mean, "feature_scale": scale, "model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "seed": seed}, path)
            checkpoints[heldout].append({"path": str(path), "sha256": file_sha256(path), "seed": seed})
            del model
        good_stack = np.stack(member_good); harm_stack = np.stack(member_harm)
        p_good[heldout_index] = good_stack.mean(0); p_harm[heldout_index] = harm_stack.mean(0)
        p_good_std[heldout_index] = good_stack.std(0); p_harm_std[heldout_index] = harm_stack.std(0)
    p_good = p_good.reshape(600, 16); p_harm = p_harm.reshape(600, 16); p_good_std = p_good_std.reshape(600, 16); p_harm_std = p_harm_std.reshape(600, 16)

    reports = []
    for harm_weight in (0.5, 1.0, 2.0, 4.0):
        for z in (0.0, 0.5, 1.0):
            score = p_good - z * p_good_std - harm_weight * (p_harm + z * p_harm_std)
            score = np.where(valid, score, -np.inf)
            best = score.argmax(1)
            best_score = score[np.arange(600), best]
            finite = best_score[np.isfinite(best_score)]
            if not len(finite):
                continue
            thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 0.95, 20)))
            for threshold in thresholds:
                per_bank = {}
                all_safety = True
                all_quality = True
                offset = 0
                for bank_number, bank in enumerate(banks):
                    local_best = best[offset:offset + 100]
                    local_score = best_score[offset:offset + 100]
                    accept = np.isfinite(local_score) & (local_score >= threshold)
                    selected = np.where(accept, local_best + 17, bank["base_selected"])
                    base_metrics = method_metrics(bank["base_selected"], bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + bank_number * 100)
                    metrics = method_metrics(selected, bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + bank_number * 100 + 10)
                    safety = metrics["promotion_fraction"] >= 0.40 and metrics["dual_precision"] is not None and metrics["dual_precision"] >= 0.90 and metrics["dual_wilson_lower_95"] is not None and metrics["dual_wilson_lower_95"] >= 0.80 and metrics["harm_rate_per_promotion"] is not None and metrics["harm_rate_per_promotion"] <= 0.10
                    q = quality(metrics, base_metrics)
                    all_safety = all_safety and safety; all_quality = all_quality and q["strict_all_metric_pass"]
                    per_bank[bank["name"]] = {"replacements": int(accept.sum()), "safety_pass": safety, "quality": q, "base": base_metrics, "cem_true": metrics}
                    offset += 100
                reports.append({"harm_weight": harm_weight, "uncertainty_scale": z, "threshold": float(threshold), "all_banks_safety": all_safety, "all_banks_strict_quality": all_quality, "all_banks_pass": all_safety and all_quality, "banks": per_bank})
    feasible = [item for item in reports if item["all_banks_pass"]]
    safety_feasible = [item for item in reports if item["all_banks_safety"]]
    best = max(safety_feasible, key=lambda item: (sum(value["quality"]["strict_all_metric_pass"] for value in item["banks"].values()), sum(value["replacements"] for value in item["banks"].values()))) if safety_feasible else None
    prediction_path = args.output_dir / "candidate_replacement_oof.npz"
    write_npz(prediction_path, record_key=np.concatenate([bank["record"] for bank in banks]), cohort=cohort_state, p_good=p_good, p_harm=p_harm, p_good_std=p_good_std, p_harm_std=p_harm_std, valid=valid)
    report = {"schema": "recon_target_cem_replacement_v32", "status": "candidate_level_whole_bank_oof_complete", "evidence_role": "development_only", "cem": True, "feature_contract": "CEM motion+universal+support 52, v29 base 52, delta 52, base risk, frozen CEM energy", "target_contract": "candidate dual-safe versus incumbent and ATE/objective dominates v29 base; separate harm head", "checkpoints": checkpoints, "prediction": {"path": str(prediction_path), "sha256": file_sha256(prediction_path)}, "policies_tested": len(reports), "feasible_count": len(feasible), "safety_feasible_count": len(safety_feasible), "best_safety_feasible": best, "formal_claim_allowed": False}
    write_json(result_path, report)
    compact = {"feasible_count": len(feasible), "safety_feasible_count": len(safety_feasible), "best": None if best is None else {"harm_weight": best["harm_weight"], "uncertainty_scale": best["uncertainty_scale"], "threshold": best["threshold"], "banks": {name: {"replacements": value["replacements"], "dual": value["cem_true"]["dual_precision"], "wilson": value["cem_true"]["dual_wilson_lower_95"], "harm": value["cem_true"]["harm_rate_per_promotion"], "quality": value["quality"]} for name, value in best["banks"].items()}}}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
