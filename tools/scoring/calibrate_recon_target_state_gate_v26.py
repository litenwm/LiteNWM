#!/usr/bin/env python3
"""Search a single causal fixed threshold over v24 OOF state ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary, wilson_lower
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    banks = [load_bank(args.root, value) for value in "abcdef"]
    record = np.concatenate([bank["record"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    paths = [
        args.prediction_dir / "state_universal_support_oof.npz",
        args.prediction_dir / "state_motion_universal_support_oof.npz",
    ]
    predictions = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            if not np.array_equal(record, payload["record_key"].astype(str)):
                raise RuntimeError("v26 record identity mismatch")
            predictions.append({key: payload[key].astype(np.float32) for key in (
                "p_dual", "p_harm", "p_dual_std", "p_harm_std",
            )})
            challenger = payload["challenger_index"].astype(np.int64)
    eligible = challenger > 0
    candidate = np.maximum(challenger - 1, 0)
    row = np.arange(600)
    chosen_delta = delta[row, candidate]
    dual_label = eligible & (chosen_delta[:, 0] < -1.0e-7) & (chosen_delta[:, 3] < -1.0e-7)
    harm_label = eligible & ((chosen_delta[:, 0] > 1.0e-7) | (chosen_delta[:, 3] > 1.0e-7))
    bank_masks = [cohort == value for value in np.unique(cohort)]
    best = None
    feasible_count = 0
    searched = 0
    for blend in np.linspace(0.0, 1.0, 21):
        p_dual = blend * predictions[0]["p_dual"] + (1.0 - blend) * predictions[1]["p_dual"]
        p_harm = blend * predictions[0]["p_harm"] + (1.0 - blend) * predictions[1]["p_harm"]
        p_dual_std = blend * predictions[0]["p_dual_std"] + (1.0 - blend) * predictions[1]["p_dual_std"]
        p_harm_std = blend * predictions[0]["p_harm_std"] + (1.0 - blend) * predictions[1]["p_harm_std"]
        for z in (0.0, 0.5, 1.0, 1.5, 2.0):
            dual_lower = p_dual - z * p_dual_std
            harm_upper = p_harm + z * p_harm_std
            for harm_weight in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
                risk = np.where(eligible, dual_lower - harm_weight * harm_upper, -np.inf)
                finite = risk[eligible]
                thresholds = np.unique(np.quantile(finite, np.linspace(0.0, 0.80, 401)))
                promote = risk[:, None] >= thresholds[None, :]
                coverage_values = []
                dual_values = []
                wilson_values = []
                harm_values = []
                strict = np.ones(len(thresholds), dtype=bool)
                for mask in bank_masks:
                    count = promote[mask].sum(0)
                    dual_count = (promote[mask] & dual_label[mask, None]).sum(0)
                    harm_count = (promote[mask] & harm_label[mask, None]).sum(0)
                    coverage = count / 100.0
                    dual_precision = np.divide(
                        dual_count, count, out=np.zeros_like(count, dtype=float), where=count > 0
                    )
                    harm_rate = np.divide(
                        harm_count, count, out=np.ones_like(count, dtype=float), where=count > 0
                    )
                    wilson = np.asarray([
                        wilson_lower(int(success), int(total)) or 0.0
                        for success, total in zip(dual_count, count)
                    ])
                    strict &= (
                        (coverage >= 0.40) & (dual_precision >= 0.90)
                        & (wilson >= 0.80) & (harm_rate <= 0.10)
                    )
                    coverage_values.append(coverage)
                    dual_values.append(dual_precision)
                    wilson_values.append(wilson)
                    harm_values.append(harm_rate)
                searched += len(thresholds)
                for index in range(len(thresholds)):
                    is_feasible = bool(strict[index])
                    feasible_count += int(is_feasible)
                    key = (
                        int(is_feasible),
                        min(value[index] for value in wilson_values),
                        min(value[index] for value in dual_values),
                        -max(value[index] for value in harm_values),
                        min(value[index] for value in coverage_values),
                        sum(value[index] for value in coverage_values),
                    )
                    if best is None or key > best[0]:
                        best = (key, {
                            "blend": float(blend), "uncertainty_scale": z,
                            "harm_weight": harm_weight,
                            "threshold": float(thresholds[index]),
                            "risk": risk, "strict": is_feasible,
                        })
    assert best is not None
    policy = best[1]
    selected = np.where(
        eligible & (policy["risk"] >= policy["threshold"]), challenger, 0
    ).astype(np.int64)
    per_bank = {}
    for bank_name in np.unique(cohort):
        mask = cohort == bank_name
        per_bank[bank_name] = method_summary(selected[mask], delta[mask], absolute[mask])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path, record_key=record, cohort=cohort,
        risk=policy.pop("risk"), selected_index=selected,
    )
    report = {
        "schema": "recon_target_state_gate_v26_fixed_calibration",
        "status": (
            "fixed_threshold_development_passed_freeze_for_new_confirmation"
            if policy["strict"] else "fixed_threshold_development_failed"
        ),
        "evidence_role": "six_bank_oof_development_calibration",
        "online_causal": True, "future_route_quota": False,
        "policy": policy,
        "searched_policies": searched,
        "strict_feasible_policies": feasible_count,
        "per_bank": per_bank,
        "strict": policy["strict"],
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
