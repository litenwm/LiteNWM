#!/usr/bin/env python3
"""Screen OOF v29 safety-gated CEM replacements on six development banks."""

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
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_state_gate_v24 import StateRisk, predict
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--hashfold-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261001)
    return parser.parse_args()


def load_member(path: Path, expected_sha: str, feature: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"checkpoint SHA mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = StateRisk(int(payload["input_dim"]), int(payload["hidden_dim"]), float(payload["dropout"])).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    value = predict(model, np.asarray(payload["feature_mean"], dtype=np.float32), np.asarray(payload["feature_scale"], dtype=np.float32), feature, device)
    del model
    return value


def quality_not_worse(candidate: dict, base: dict) -> dict[str, bool]:
    means = [candidate["absolute"][name]["mean_m"] <= base["absolute"][name]["mean_m"] + 1.0e-9 for name in ("ate", "rpe", "endpoint", "objective")]
    p90 = [candidate["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1.0e-9 for name in ("ate", "rpe", "endpoint", "objective")]
    strict = any(candidate["absolute"][name]["mean_m"] < base["absolute"][name]["mean_m"] - 1.0e-9 for name in ("ate", "rpe", "endpoint", "objective"))
    return {"all_means_not_worse": all(means), "all_p90_not_worse": all(p90), "at_least_one_mean_better": strict, "pass": all(means) and all(p90) and strict}


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    policy_result = json.loads(args.policy_result.read_text(encoding="utf-8"))
    policy = policy_result["policy"]
    hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8"))
    base_prediction = {}
    for name, item in hashfold["predictions"].items():
        path = Path(item["path"])
        if file_sha256(path) != item["sha256"]:
            raise RuntimeError("base OOF SHA mismatch")
        with np.load(path, allow_pickle=False) as payload:
            base_prediction[name] = {key: payload[key].copy() for key in payload.files}
    blend = float(policy["blend"])
    base_dual = blend * base_prediction["state_universal_support"]["p_dual"] + (1.0 - blend) * base_prediction["state_motion_universal_support"]["p_dual"]
    base_harm = blend * base_prediction["state_universal_support"]["p_harm"] + (1.0 - blend) * base_prediction["state_motion_universal_support"]["p_harm"]
    base_risk_all = base_dual - float(policy["harm_weight"]) * base_harm
    base_challenger_all = base_prediction["state_universal_support"]["challenger_index"].astype(np.int64)
    base_selected_all = np.where(base_risk_all > float(policy["threshold"]), base_challenger_all, 0).astype(np.int64)

    banks = {}
    prediction_audit = {}
    for letter in "abcdef":
        name = f"bank_{letter}"
        bank = args.root / name
        feature_path = next((bank / "cem_run_v30/iteration_02").glob("cem_features.npz"))
        score_path = bank / "cem_run_v30/iteration_02/frozen_scores.npz"
        compact, _ = compact_base_feature(feature_path, score_path, np.empty((100, 0), dtype=np.float32))
        with np.load(score_path, allow_pickle=False) as payload:
            record = payload["record_key"].astype(str)
            score = {"std": payload["std"][:, 1:].astype(np.float32), "dual": payload["dual_probability"][:, 1:].astype(np.float32), "harm": payload["harm_probability"][:, 1:].astype(np.float32)}
            valid = payload["valid"][:, 1:].astype(bool)
        challenger = frozen_challenger(score, valid)
        chosen = compact[np.arange(100), challenger]
        variants = {"state_universal_support": chosen[:, 474:486], "state_motion_universal_support": chosen[:, 434:486]}
        probability = {}
        prediction_audit[name] = {}
        for variant, feature in variants.items():
            member_dual, member_harm = [], []
            prediction_audit[name][variant] = []
            for item in hashfold["checkpoints"][variant][name]:
                dual, harm = load_member(Path(item["path"]), item["sha256"], feature, args.device)
                member_dual.append(dual); member_harm.append(harm)
                prediction_audit[name][variant].append({"path": item["path"], "sha256": item["sha256"]})
            probability[variant] = {"dual": np.stack(member_dual).mean(0), "harm": np.stack(member_harm).mean(0)}
        cem_dual = blend * probability["state_universal_support"]["dual"] + (1.0 - blend) * probability["state_motion_universal_support"]["dual"]
        cem_harm = blend * probability["state_universal_support"]["harm"] + (1.0 - blend) * probability["state_motion_universal_support"]["harm"]
        cem_risk = cem_dual - float(policy["harm_weight"]) * cem_harm
        with np.load(bank / "cem_run_v30/development_outcomes.npz", allow_pickle=False) as payload:
            if not np.array_equal(payload["record_key"].astype(str), record):
                raise RuntimeError(f"CEM outcome identity mismatch: {name}")
            union = payload["union_absolute_metrics"].astype(np.float32)
            route = payload["route_group"].astype(str)
            trajectory = payload["trajectory"].astype(str)
        cohort = base_prediction["state_universal_support"]["cohort"].astype(str) == name
        if cohort.sum() != 100 or not np.array_equal(base_prediction["state_universal_support"]["record_key"].astype(str)[cohort], record):
            raise RuntimeError(f"base/CEM OOF identity mismatch: {name}")
        banks[name] = {"record": record, "route": route, "trajectory": trajectory, "union": union, "base_selected": base_selected_all[cohort], "base_risk": base_risk_all[cohort], "cem_challenger": challenger + 17, "cem_risk": cem_risk, "cem_dual": cem_dual, "cem_harm": cem_harm}

    margins = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25)
    variants = {}
    for margin in margins:
        per_bank = {}
        all_pass = True
        for bank_number, (name, bank) in enumerate(banks.items()):
            base = bank["base_selected"]
            accept = (base != 0) & (bank["cem_risk"] > float(policy["threshold"])) & (bank["cem_risk"] > bank["base_risk"] + margin)
            selected = np.where(accept, bank["cem_challenger"], base)
            base_metrics = method_metrics(base, bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + bank_number * 100)
            metrics = method_metrics(selected, bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + bank_number * 100 + 10)
            safety = {"coverage": metrics["promotion_fraction"] >= 0.40, "dual": metrics["dual_precision"] is not None and metrics["dual_precision"] >= 0.90, "wilson": metrics["dual_wilson_lower_95"] is not None and metrics["dual_wilson_lower_95"] >= 0.80, "harm": metrics["harm_rate_per_promotion"] is not None and metrics["harm_rate_per_promotion"] <= 0.10}
            quality = quality_not_worse(metrics, base_metrics)
            passed = all(safety.values()) and quality["pass"]
            all_pass = all_pass and passed
            per_bank[name] = {"replacements": int(accept.sum()), "base": base_metrics, "cem_true": metrics, "safety": safety, "quality": quality, "pass": passed}
        variants[f"margin_{margin:.2f}"] = {"margin": margin, "all_banks_pass": all_pass, "banks": per_bank}
    feasible = [value for value in variants.values() if value["all_banks_pass"]]
    report = {"schema": "recon_target_cem_hashfold_v31_screen", "status": "oof_cem_replacement_screen_complete", "evidence_role": "development_only_six_bank_whole_bank_oof", "cem": True, "candidate": "frozen universal conservative challenger from real final CEM population", "replacement": "only replace an existing v29 base promotion when OOF CEM risk exceeds threshold and base risk by fixed margin", "policy": policy, "variants": variants, "feasible_count": len(feasible), "prediction_audit": prediction_audit, "formal_claim_allowed": False}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(result_path, report)
    print(json.dumps({key: {"all_banks_pass": value["all_banks_pass"], "banks": {bank: {"replacements": item["replacements"], "dual": item["cem_true"]["dual_precision"], "wilson": item["cem_true"]["dual_wilson_lower_95"], "harm": item["cem_true"]["harm_rate_per_promotion"], "quality": item["quality"]["pass"]} for bank, item in value["banks"].items()}} for key, value in variants.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
