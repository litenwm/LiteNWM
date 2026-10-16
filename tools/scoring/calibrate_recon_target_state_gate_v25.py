#!/usr/bin/env python3
"""Calibrate one causal fixed threshold for the v25 state-gate blend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary, optional
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    banks = [load_bank(args.root, value) for value in "abcdef"]
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    record = np.concatenate([bank["record"] for bank in banks])
    paths = [
        args.prediction_dir / "state_universal_support_oof.npz",
        args.prediction_dir / "state_motion_universal_support_oof.npz",
    ]
    values = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(record, payload["record_key"].astype(str)):
                raise RuntimeError("v25 calibration identity mismatch")
            values.append({
                "dual": payload["p_dual"].astype(np.float32),
                "harm": payload["p_harm"].astype(np.float32),
                "challenger": payload["challenger_index"].astype(np.int64),
            })
    if not np.array_equal(values[0]["challenger"], values[1]["challenger"]):
        raise RuntimeError("v25 challenger mismatch")
    challenger = values[0]["challenger"]
    p_dual = 0.75 * values[0]["dual"] + 0.25 * values[1]["dual"]
    p_harm = 0.75 * values[0]["harm"] + 0.25 * values[1]["harm"]
    risk = p_dual - 0.5 * p_harm
    risk = np.where(challenger > 0, risk, -np.inf)
    finite = risk[np.isfinite(risk)]
    thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 0.80, 801)))
    policies = []
    for threshold in thresholds:
        selected = np.where(
            np.isfinite(risk) & (risk >= threshold), challenger, 0
        ).astype(np.int64)
        per_bank = {}
        for bank_name in np.unique(cohort):
            mask = cohort == bank_name
            per_bank[bank_name] = method_summary(
                selected[mask], delta[mask], absolute[mask]
            )
        strict = all(passed(value) for value in per_bank.values())
        policies.append({
            "threshold": float(threshold), "selected": selected,
            "per_bank": per_bank, "strict": strict,
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
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path, record_key=record, cohort=cohort,
        risk=risk, selected_index=best["selected"],
    )
    report = {
        "schema": "recon_target_state_gate_v25_fixed_calibration",
        "status": (
            "fixed_threshold_development_passed_freeze_for_new_confirmation"
            if feasible else "fixed_threshold_development_failed_continue_causal_gate_research"
        ),
        "evidence_role": "six_bank_oof_development_calibration",
        "online_causal": True,
        "future_route_quota": False,
        "policy": {
            "universal_state_weight": 0.75,
            "motion_state_weight": 0.25,
            "harm_weight": 0.5,
            "threshold": best["threshold"],
        },
        "strict_feasible": bool(feasible),
        "strict_feasible_thresholds": len(feasible),
        "searched_thresholds": len(policies),
        "per_bank": best["per_bank"],
        "strict": best["strict"],
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "prediction_artifacts": [
            {"path": str(path), "sha256": file_sha256(path)} for path in paths
        ],
        "final_reserve_accessed": False,
        "final_reserve_remains_sealed": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
