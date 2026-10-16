#!/usr/bin/env python3
"""Calibrate final-model score scale using only spent A-F development banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.evaluate_recon_frozen_universal_gate_v23 import load_score
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.screen_recon_target_state_gate_v24 import StateRisk, predict
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary, optional
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-freeze", type=Path, required=True)
    parser.add_argument("--preflight-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def passed(metrics: dict) -> bool:
    return (
        metrics["coverage"] >= 0.40
        and optional(metrics["dual_precision"], 0.0) >= 0.90
        and optional(metrics["dual_wilson_lower_95"], 0.0) >= 0.80
        and optional(metrics["harm_rate"], 1.0) <= 0.10
    )


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    freeze = json.loads(args.model_freeze.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight_freeze.read_text(encoding="utf-8"))
    if preflight.get("outcomes_accessed") or preflight.get("labels_opened"):
        raise RuntimeError("failed preflight unexpectedly opened outcomes")
    if preflight["coverage_label_blind"] >= 0.40:
        raise RuntimeError("preflight was not an automatic coverage failure")
    banks = [load_bank(args.root, value) for value in "abcdef"]
    scores = [load_score(args.root, value) for value in "abcdef"]
    full = np.concatenate([bank["feature"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    challenger_values = []
    for bank, score in zip(banks, scores):
        challenger_values.append(frozen_challenger(score, bank["valid"]))
    challenger = np.concatenate(challenger_values)
    chosen = full[np.arange(600), challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
    }
    probabilities = {}
    for name, feature in variants.items():
        dual_values = []
        harm_values = []
        for value in freeze["checkpoints"][name]:
            path = Path(value["path"])
            if file_sha256(path) != value["sha256"]:
                raise RuntimeError("v27 checkpoint SHA mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            model = StateRisk(
                config["input_dim"], config["hidden_dim"], config["dropout"]
            ).to(args.device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            dual, harm = predict(
                model,
                np.asarray(payload["feature_mean"], dtype=np.float32),
                np.asarray(payload["feature_scale"], dtype=np.float32),
                feature,
                args.device,
            )
            dual_values.append(dual)
            harm_values.append(harm)
            del model
        probabilities[name] = {
            "dual": np.stack(dual_values).mean(0),
            "harm": np.stack(harm_values).mean(0),
        }
    old_policy = freeze["policy"]
    blend = float(old_policy["blend"])
    p_dual = (
        blend * probabilities["state_universal_support"]["dual"]
        + (1.0 - blend) * probabilities["state_motion_universal_support"]["dual"]
    )
    p_harm = (
        blend * probabilities["state_universal_support"]["harm"]
        + (1.0 - blend) * probabilities["state_motion_universal_support"]["harm"]
    )
    risk = p_dual - float(old_policy["harm_weight"]) * p_harm
    eligible = challenger >= 0
    thresholds = np.unique(np.quantile(risk[eligible], np.linspace(0.0, 0.80, 1601)))
    policies = []
    for threshold in thresholds:
        selected = np.where(
            eligible & (risk >= threshold), challenger + 1, 0
        ).astype(np.int64)
        per_bank = {}
        for name in np.unique(cohort):
            mask = cohort == name
            per_bank[name] = method_summary(selected[mask], delta[mask], absolute[mask])
        policies.append({
            "threshold": float(threshold), "selected": selected,
            "per_bank": per_bank,
            "strict": all(passed(value) for value in per_bank.values()),
        })
    feasible = [value for value in policies if value["strict"]]
    def key(value: dict):
        metrics = list(value["per_bank"].values())
        return (
            sum(passed(metric) for metric in metrics),
            min(optional(metric["dual_wilson_lower_95"], 0.0) for metric in metrics),
            min(optional(metric["dual_precision"], 0.0) for metric in metrics),
            -max(optional(metric["harm_rate"], 1.0) for metric in metrics),
            min(metric["coverage"] for metric in metrics),
            sum(metric["coverage"] for metric in metrics),
        )
    best = max(feasible if feasible else policies, key=key)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = args.output_dir / "development_selected_index.npz"
    write_npz(
        selected_path, record_key=record, cohort=cohort,
        risk=risk, selected_index=best["selected"],
    )
    policy = {**old_policy, "threshold": best["threshold"], "strict": best["strict"]}
    report = {
        "schema": "recon_target_state_gate_final_v27_calibration",
        "status": (
            "final_score_scale_calibrated_freeze_for_new_confirmation"
            if feasible else "final_score_scale_calibration_failed"
        ),
        "evidence_role": "spent_A_to_F_development_only",
        "policy": policy,
        "strict": best["strict"],
        "strict_feasible_thresholds": len(feasible),
        "searched_thresholds": len(policies),
        "per_bank": best["per_bank"],
        "development_selected_index": str(selected_path),
        "development_selected_index_sha256": file_sha256(selected_path),
        "model_freeze": str(args.model_freeze),
        "model_freeze_sha256": file_sha256(args.model_freeze),
        "preflight_downgrade": {
            "selection_freeze": str(args.preflight_freeze),
            "selection_freeze_sha256": file_sha256(args.preflight_freeze),
            "role": "label_blind_preflight_spent_due_to_observed_coverage",
            "coverage": preflight["coverage_label_blind"],
            "labels_opened": False,
            "outcomes_accessed": False,
        },
        "online_causal": True,
        "future_route_quota": False,
        "new_confirmation_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
