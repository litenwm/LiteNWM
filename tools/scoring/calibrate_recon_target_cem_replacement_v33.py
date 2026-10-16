#!/usr/bin/env python3
"""Calibrate hard harm/uncertainty guards for v32 OOF CEM replacements."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v32-result", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--hashfold-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261003)
    return parser.parse_args()


def fast(selected: np.ndarray, absolute: np.ndarray) -> dict:
    row = np.arange(len(selected)); chosen = absolute[row, selected]; incumbent = absolute[:, 0]; delta = chosen - incumbent; promote = selected != 0; count = int(promote.sum())
    dual_count = int(((delta[promote, 0] < -1.0e-7) & (delta[promote, 3] < -1.0e-7)).sum()); harm_count = int(((delta[promote, 0] > 1.0e-7) | (delta[promote, 3] > 1.0e-7)).sum())
    return {"promotions": count, "coverage": count / len(selected), "dual": dual_count / count if count else None, "wilson": wilson_lower(dual_count, count), "harm": harm_count / count if count else None, "absolute": {name: {"mean_m": float(chosen[:, index].mean()), "p90_m": float(np.quantile(chosen[:, index], 0.90))} for index, name in enumerate(NAMES)}}


def quality(value: dict, base: dict) -> bool:
    return all(value["absolute"][name]["mean_m"] <= base["absolute"][name]["mean_m"] + 1.0e-9 for name in NAMES) and all(value["absolute"][name]["p90_m"] <= base["absolute"][name]["p90_m"] + 1.0e-9 for name in NAMES) and value["absolute"]["ate"]["mean_m"] < base["absolute"]["ate"]["mean_m"] - 1.0e-9 and value["absolute"]["objective"]["mean_m"] < base["absolute"]["objective"]["mean_m"] - 1.0e-9


def main() -> None:
    args = parse_args(); result_path = args.output_dir / "result.json"
    if result_path.exists(): raise FileExistsError(result_path)
    v32 = json.loads(args.v32_result.read_text(encoding="utf-8")); prediction_path = Path(v32["prediction"]["path"])
    if file_sha256(prediction_path) != v32["prediction"]["sha256"]: raise RuntimeError("v32 prediction SHA mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str); cohort = payload["cohort"].astype(str); p_good = payload["p_good"].astype(np.float32); p_harm = payload["p_harm"].astype(np.float32); p_good_std = payload["p_good_std"].astype(np.float32); p_harm_std = payload["p_harm_std"].astype(np.float32); valid = payload["valid"].astype(bool)
    policy = json.loads(args.policy_result.read_text(encoding="utf-8"))["policy"]
    hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8")); base = {}
    for name, item in hashfold["predictions"].items():
        with np.load(Path(item["path"]), allow_pickle=False) as payload: base[name] = {key: payload[key].copy() for key in payload.files}
    blend = float(policy["blend"]); base_risk = blend * base["state_universal_support"]["p_dual"] + (1.0 - blend) * base["state_motion_universal_support"]["p_dual"] - float(policy["harm_weight"]) * (blend * base["state_universal_support"]["p_harm"] + (1.0 - blend) * base["state_motion_universal_support"]["p_harm"])
    base_selected = np.where(base_risk > float(policy["threshold"]), base["state_universal_support"]["challenger_index"].astype(np.int64), 0)
    if not np.array_equal(base["state_universal_support"]["record_key"].astype(str), record): raise RuntimeError("v32/base record mismatch")
    banks = []
    for letter in "abcdef":
        name = f"bank_{letter}"; mask = cohort == name; bank = args.root / name
        with np.load(bank / "cem_run_v30/development_outcomes.npz", allow_pickle=False) as payload:
            if not np.array_equal(payload["record_key"].astype(str), record[mask]): raise RuntimeError(f"outcome identity mismatch: {name}")
            union = payload["union_absolute_metrics"].astype(np.float32); route = payload["route_group"].astype(str); trajectory = payload["trajectory"].astype(str)
        banks.append({"name": name, "mask": mask, "union": union, "route": route, "trajectory": trajectory, "base": base_selected[mask], "base_fast": fast(base_selected[mask], union)})

    policies = []
    for harm_weight in (0.5, 1.0, 2.0, 4.0):
        for z in (0.0, 0.5, 1.0):
            raw = p_good - z * p_good_std - harm_weight * (p_harm + z * p_harm_std); raw = np.where(valid, raw, -np.inf); best_index = raw.argmax(1); row = np.arange(600); best_score = raw[row, best_index]; best_good = p_good[row, best_index]; best_harm = p_harm[row, best_index]; best_unc = p_good_std[row, best_index] + p_harm_std[row, best_index]
            finite = best_score[np.isfinite(best_score)]; thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 0.95, 20)))
            for harm_cap in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
                for good_floor in (0.30, 0.40, 0.50, 0.60, 0.70):
                    for unc_cap in (0.10, 0.20, 0.30, 0.50):
                        guard = (best_harm <= harm_cap) & (best_good >= good_floor) & (best_unc <= unc_cap)
                        for threshold in thresholds:
                            accept_all = guard & np.isfinite(best_score) & (best_score >= threshold); per_bank = {}; all_pass = True; total_replacements = 0; improvement = 0.0
                            for bank in banks:
                                accept = accept_all[bank["mask"]]; selected = np.where(accept, best_index[bank["mask"]] + 17, bank["base"]); value = fast(selected, bank["union"]); safety = value["coverage"] >= 0.40 and value["dual"] is not None and value["dual"] >= 0.90 and value["wilson"] is not None and value["wilson"] >= 0.80 and value["harm"] is not None and value["harm"] <= 0.10; q = quality(value, bank["base_fast"]); all_pass = all_pass and safety and q; total_replacements += int(accept.sum()); improvement += bank["base_fast"]["absolute"]["objective"]["mean_m"] - value["absolute"]["objective"]["mean_m"]; per_bank[bank["name"]] = {"replacements": int(accept.sum()), "safety": safety, "quality": q, "fast": value}
                            if all_pass: policies.append({"harm_weight": harm_weight, "uncertainty_scale": z, "threshold": float(threshold), "harm_cap": harm_cap, "good_floor": good_floor, "uncertainty_cap": unc_cap, "total_replacements": total_replacements, "objective_improvement_sum": improvement, "banks": per_bank, "best_index": best_index, "accept_all": accept_all})
    best_policy = max(policies, key=lambda value: (value["objective_improvement_sum"], value["total_replacements"])) if policies else None
    detailed = None
    if best_policy is not None:
        detailed = {key: value for key, value in best_policy.items() if key not in ("best_index", "accept_all", "banks")}; detailed["banks"] = {}; selected_all = np.zeros(600, dtype=np.int64)
        for number, bank in enumerate(banks):
            accept = best_policy["accept_all"][bank["mask"]]; selected = np.where(accept, best_policy["best_index"][bank["mask"]] + 17, bank["base"]); selected_all[bank["mask"]] = selected
            base_metrics = method_metrics(bank["base"], bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + number * 100)
            metrics = method_metrics(selected, bank["union"], bank["trajectory"], bank["route"], replicates=args.bootstrap_replicates, seed=args.seed + number * 100 + 10)
            detailed["banks"][bank["name"]] = {"replacements": int(accept.sum()), "base": base_metrics, "cem_true": metrics}
        selected_path = args.output_dir / "selected_index_oof.npz"; args.output_dir.mkdir(parents=True, exist_ok=False); write_npz(selected_path, record_key=record, cohort=cohort, selected_index=selected_all, replacement=(selected_all >= 17)); detailed["selected_index"] = str(selected_path); detailed["selected_index_sha256"] = file_sha256(selected_path)
    else: args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema": "recon_target_cem_replacement_v33", "status": "hard_guard_calibration_complete", "evidence_role": "development_only_six_bank_oof", "cem": True, "feasible_count": len(policies), "best_policy": detailed, "v32_result_sha256": file_sha256(args.v32_result), "formal_claim_allowed": False}
    write_json(result_path, report)
    compact = {"feasible_count": len(policies), "best": None if detailed is None else {key: value for key, value in detailed.items() if key != "banks"}}
    if detailed is not None: compact["banks"] = {name: {"replacements": value["replacements"], "dual": value["cem_true"]["dual_precision"], "wilson": value["cem_true"]["dual_wilson_lower_95"], "harm": value["cem_true"]["harm_rate_per_promotion"], "absolute": value["cem_true"]["absolute"]} for name, value in detailed["banks"].items()}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__": main()
