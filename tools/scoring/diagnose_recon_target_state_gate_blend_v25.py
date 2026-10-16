#!/usr/bin/env python3
"""Find one cross-bank blend of v24 state-risk predictions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary, optional
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json


NAMES = (
    "state_universal_support",
    "state_motion_universal_support",
    "state_static_universal_support",
    "state_universal_support_spatial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    banks = [load_bank(args.root, value) for value in "abcdef"]
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    prediction = {}
    reference_record = None
    challenger = None
    paths = {}
    for name in NAMES:
        path = args.prediction_dir / f"{name}_oof.npz"
        with np.load(path, allow_pickle=False) as payload:
            record = payload["record_key"].astype(str)
            value = {
                "dual": payload["p_dual"].astype(np.float32),
                "harm": payload["p_harm"].astype(np.float32),
                "dual_std": payload["p_dual_std"].astype(np.float32),
                "harm_std": payload["p_harm_std"].astype(np.float32),
            }
            candidate = payload["challenger_index"].astype(np.int64)
        if reference_record is None:
            reference_record = record
            challenger = candidate
        elif not np.array_equal(reference_record, record) or not np.array_equal(challenger, candidate):
            raise RuntimeError("v25 prediction identity mismatch")
        prediction[name] = value
        paths[name] = {"path": str(path), "sha256": file_sha256(path)}
    assert challenger is not None
    candidates = []
    for left, right in itertools.combinations_with_replacement(NAMES, 2):
        for blend in np.linspace(0.0, 1.0, 21):
            dual = blend * prediction[left]["dual"] + (1.0 - blend) * prediction[right]["dual"]
            harm = blend * prediction[left]["harm"] + (1.0 - blend) * prediction[right]["harm"]
            dual_std = blend * prediction[left]["dual_std"] + (1.0 - blend) * prediction[right]["dual_std"]
            harm_std = blend * prediction[left]["harm_std"] + (1.0 - blend) * prediction[right]["harm_std"]
            for z in (0.0, 0.5, 1.0, 1.5, 2.0):
                for risk_weight in (0.5, 1.0, 2.0, 4.0):
                    risk = dual - z * dual_std - risk_weight * (harm + z * harm_std)
                    risk = np.where(challenger > 0, risk, -np.inf)
                    for promotions in (40, 45, 50, 55, 60):
                        per_bank = {}
                        selected = np.zeros(600, dtype=np.int64)
                        for bank_name in np.unique(cohort):
                            index = np.flatnonzero(cohort == bank_name)
                            order = index[np.argsort(-risk[index], kind="stable")[:promotions]]
                            order = order[np.isfinite(risk[order])]
                            selected[order] = challenger[order]
                            per_bank[bank_name] = method_summary(
                                selected[index], delta[index], absolute[index]
                            )
                        strict = all(passed(value) for value in per_bank.values())
                        candidates.append({
                            "policy": {
                                "left": left, "right": right, "blend": float(blend),
                                "uncertainty_scale": z, "risk_weight": risk_weight,
                                "diagnostic_promotions_per_bank": promotions,
                            },
                            "per_bank": per_bank,
                            "strict": strict,
                        })
    feasible = [value for value in candidates if value["strict"]]
    def key(value: dict):
        metrics = list(value["per_bank"].values())
        return (
            sum(passed(metric) for metric in metrics),
            min(optional(metric["dual_wilson_lower_95"], 0.0) for metric in metrics),
            min(optional(metric["dual_precision"], 0.0) for metric in metrics),
            -max(optional(metric["harm_rate"], 1.0) for metric in metrics),
            sum(metric["coverage"] for metric in metrics),
        )
    best = max(feasible if feasible else candidates, key=key)
    report = {
        "schema": "recon_target_state_gate_blend_v25_diagnostic",
        "evidence_role": "development_only_noncausal_equal_per_bank_coverage_diagnostic",
        "strict_feasible": bool(feasible),
        "strict_feasible_policies": len(feasible),
        "searched_policies": len(candidates),
        "best": best,
        "prediction_artifacts": paths,
        "next_step": (
            "nested five-bank fixed-threshold calibration"
            if feasible else "feature family still cannot satisfy all six banks"
        ),
        "final_reserve_accessed": False,
        "cem": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
