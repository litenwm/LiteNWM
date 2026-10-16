#!/usr/bin/env python3
"""Evaluate frozen real-CEM pools on the six labeled target-development banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--policy-result", type=Path, required=True)
    parser.add_argument("--hashfold-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260930)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    policy_result = json.loads(args.policy_result.read_text(encoding="utf-8"))
    policy = policy_result["policy"]
    hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8"))
    predictions = {}
    for name, value in hashfold["predictions"].items():
        path = Path(value["path"])
        if file_sha256(path) != value["sha256"]:
            raise RuntimeError("v29 OOF prediction SHA mismatch")
        with np.load(path, allow_pickle=False) as payload:
            predictions[name] = {key: payload[key].copy() for key in payload.files}
    if not np.array_equal(
        predictions["state_universal_support"]["record_key"],
        predictions["state_motion_universal_support"]["record_key"],
    ):
        raise RuntimeError("v29 OOF identity mismatch")
    p_dual = (
        float(policy["blend"]) * predictions["state_universal_support"]["p_dual"]
        + (1.0 - float(policy["blend"])) * predictions["state_motion_universal_support"]["p_dual"]
    )
    p_harm = (
        float(policy["blend"]) * predictions["state_universal_support"]["p_harm"]
        + (1.0 - float(policy["blend"])) * predictions["state_motion_universal_support"]["p_harm"]
    )
    risk = p_dual - float(policy["harm_weight"]) * p_harm
    challenger = predictions["state_universal_support"]["challenger_index"].astype(np.int64)
    base_selected_all = np.where(risk > float(policy["threshold"]), challenger, 0).astype(np.int64)

    reports = {}
    outcome_artifacts = {}
    for bank_number, letter in enumerate("abcdef"):
        name = f"bank_{letter}"
        bank = args.root / name
        freeze_path = bank / "cem_run_v30/prelabel_freeze.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze["status"] != "cem_true_actions_and_selections_frozen_before_outcomes" or freeze.get("outcomes_used_by_cem") or freeze.get("future_pose_outcomes_opened") or not freeze.get("cem"):
            raise RuntimeError(f"invalid CEM freeze: {name}")
        selected_path = Path(freeze["selected_actions"])
        if file_sha256(selected_path) != freeze["selected_actions_sha256"]:
            raise RuntimeError(f"CEM selected actions changed: {name}")
        with np.load(bank / "source_labeled.npz", allow_pickle=False) as payload:
            record = payload["record_key"].astype(str)
            route = payload["route_group"].astype(str)
            trajectory = payload["trajectory"].astype(str)
            base_absolute = payload["absolute_metrics"].astype(np.float32)
            ground_truth = payload["ground_truth_local_m"].astype(np.float32)
        with np.load(selected_path, allow_pickle=False) as payload:
            if not np.array_equal(payload["record_key"].astype(str), record):
                raise RuntimeError(f"CEM identity mismatch: {name}")
            cem_candidates = payload["final_cem_feature_waypoint_units"].astype(np.float32)
            native = payload["native_scorer_v2_selected_index"].astype(np.int64)
            universal_cem = payload["universal_d_cem_true_selected_index"].astype(np.int64)
            energy = payload["final_cem_energy"].astype(np.float32)
            valid = payload["final_cem_valid"].astype(bool)
        cem_absolute = metric_labels(cem_candidates, ground_truth, 0.25, 0.4, 0.6)["values"].astype(np.float32)
        cohort = predictions["state_universal_support"]["cohort"].astype(str) == name
        if cohort.sum() != 100 or not np.array_equal(predictions["state_universal_support"]["record_key"].astype(str)[cohort], record):
            raise RuntimeError(f"OOF/bank identity mismatch: {name}")
        base_selected = base_selected_all[cohort]
        # Union: base K17 indices 0..16, then CEM challenger indices 17..32.
        union_absolute = np.concatenate((base_absolute, cem_absolute[:, 1:]), axis=1)
        cem_fallback = np.where(universal_cem != 0, universal_cem + 16, base_selected)
        row = np.arange(100)
        base_objective = base_absolute[row, base_selected, 3]
        cem_best = cem_absolute[:, 1:, 3].argmin(1) + 1
        cem_best_objective = cem_absolute[row, cem_best, 3]
        oracle_replace = np.where(cem_best_objective < base_objective, cem_best + 16, base_selected)
        methods_index = {
            "incumbent": np.zeros(100, dtype=np.int64),
            "native_scorer_v2": native,
            "hashfold_v29_cem_false": base_selected,
            "universal_d_cem_true_with_v29_fallback": cem_fallback,
            "posthoc_cem_replacement_oracle": oracle_replace,
        }
        methods = {
            method: method_metrics(index, union_absolute, trajectory, route, replicates=args.bootstrap_replicates, seed=args.seed + bank_number * 100 + method_number * 10)
            for method_number, (method, index) in enumerate(methods_index.items())
        }
        reports[name] = {
            "methods": methods,
            "cem_candidate_diagnostics": {
                "valid_fraction": float(valid[:, 1:].mean()),
                "windows_any_valid": float(valid[:, 1:].any(1).mean()),
                "finite_energy_fraction": float(np.isfinite(energy[:, 1:]).mean()),
                "oracle_replacements": int((oracle_replace != base_selected).sum()),
            },
            "cem_freeze": str(freeze_path),
            "cem_freeze_sha256": file_sha256(freeze_path),
        }
        outcome_path = bank / "cem_run_v30/development_outcomes.npz"
        if outcome_path.exists():
            raise FileExistsError(outcome_path)
        write_npz(
            outcome_path, record_key=record, route_group=route, trajectory=trajectory,
            base_absolute_metrics=base_absolute, cem_absolute_metrics=cem_absolute,
            union_absolute_metrics=union_absolute, base_selected_index=base_selected,
            universal_cem_with_base_fallback_index=cem_fallback,
            posthoc_cem_replacement_oracle_index=oracle_replace,
        )
        outcome_artifacts[name] = {"path": str(outcome_path), "sha256": file_sha256(outcome_path)}
    report = {
        "schema": "recon_target_cem_v30_six_bank_development_evaluation",
        "status": "real_cem_development_outcomes_opened_after_candidate_freeze",
        "evidence_role": "development_only_six_spent_target_banks",
        "cem": True,
        "config": {"iterations": 3, "population": 16, "elite_fraction": 0.25, "proposal_scale": 0.70, "support_quantile": 0.999},
        "policy_result_sha256": file_sha256(args.policy_result),
        "hashfold_result_sha256": file_sha256(args.hashfold_result),
        "banks": reports,
        "outcome_artifacts": outcome_artifacts,
        "formal_claim_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(result_path, report)
    compact = {bank: {method: {"promotions": value["promotions"], "coverage": value["promotion_fraction"], "dual": value["dual_precision"], "wilson": value["dual_wilson_lower_95"], "harm": value["harm_rate_per_promotion"], "absolute": value["absolute"]} for method, value in bank_result["methods"].items()} for bank, bank_result in reports.items()}
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
