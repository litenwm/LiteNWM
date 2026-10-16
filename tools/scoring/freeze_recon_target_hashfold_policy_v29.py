#!/usr/bin/env python3
"""Freeze the causal v29 hash-fold deployment policy before confirmation data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hashfold-result", type=Path, required=True)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--old-preflight-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashfold = json.loads(args.hashfold_result.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration_result.read_text(encoding="utf-8"))
    preflight = json.loads(args.old_preflight_freeze.read_text(encoding="utf-8"))
    if hashfold["status"] != "six_hashfold_model_groups_and_oof_predictions_frozen":
        raise RuntimeError("unexpected v29 status")
    if hashfold.get("new_confirmation_accessed") or hashfold.get("preflight_outcomes_accessed"):
        raise RuntimeError("v29 result is not outcome-sealed")
    if preflight.get("outcomes_accessed") or preflight.get("labels_opened"):
        raise RuntimeError("old preflight outcome seal was broken")
    policy = calibration["policy"]
    checkpoints = {}
    for variant, folds in hashfold["checkpoints"].items():
        checkpoints[variant] = {}
        for fold, members in folds.items():
            checkpoints[variant][fold] = []
            for member in members:
                path = Path(member["path"])
                observed = file_sha256(path)
                if observed != member["sha256"]:
                    raise RuntimeError(f"checkpoint SHA mismatch: {path}")
                checkpoints[variant][fold].append({**member, "sha256": observed})
    report = {
        "schema": "recon_target_hashfold_policy_v29",
        "status": "deployment_policy_frozen_before_new_confirmation",
        "cem": False,
        "online_causal": True,
        "future_route_quota": False,
        "router": {
            "input": "raw route_group available at inference",
            "algorithm": "int.from_bytes(sha256(route_group).digest()[:8], 'big') modulo 6",
            "fold_order": hashfold["fold_order"],
            "route_constant": True,
            "uses_outcome": False,
            "uses_future_windows": False,
        },
        "challenger": "argmax(p_dual - 0.5*p_harm - 0.5*(std_ate+std_objective)); incumbent when support invalid",
        "support_quantile": 0.999,
        "policy": policy,
        "development_banks": ["bank_a", "bank_b", "bank_c", "bank_d", "bank_e", "bank_f"],
        "development_metrics": calibration.get("banks", calibration.get("per_bank")),
        "hashfold_result": str(args.hashfold_result),
        "hashfold_result_sha256": file_sha256(args.hashfold_result),
        "calibration_result": str(args.calibration_result),
        "calibration_result_sha256": file_sha256(args.calibration_result),
        "oof_predictions": hashfold["predictions"],
        "checkpoints": checkpoints,
        "old_preflight": {
            "freeze": str(args.old_preflight_freeze),
            "freeze_sha256": file_sha256(args.old_preflight_freeze),
            "status": "label_blind_preflight_spent",
            "outcomes_accessed": False,
        },
        "new_confirmation_accessed": False,
        "labels_opened": False,
        "outcomes_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
