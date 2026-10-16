#!/usr/bin/env python3
"""Evaluate existing Arm-D route-OOF ensembles on SCAND scorer-cal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import build_universal_targets  # noqa: E402
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as v66  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    choose_candidate_level_cem_frontier,
)
from tools.scoring.run_train_dualdomain_cem_tail_v67_finalize import (  # noqa: E402
    ensemble_predict_compatible,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    ensemble_predict,
    sigmoid,
)


implementation = v66.implementation
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v67_head_hard_negative/"
    "oof_ensemble_diagnosis.json"
)
TEMP_GRID = implementation.TEMP_GRID


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(
        min(TEMP_GRID, key=lambda value: binary_nll(logit, target, float(value)))
    )


def audit(
    name: str,
    records: list[dict[str, Any]],
    predictor: Callable[..., dict[str, np.ndarray]],
    part: dict[str, np.ndarray],
) -> dict[str, Any]:
    prediction = predictor(
        records, part["feature"], part["domain"], device="cuda"
    )
    targets = build_universal_targets(part["targets_raw"])
    dual_temperature = fit_temperature(prediction["dual_logit"], targets["dual"])
    harm_temperature = fit_temperature(prediction["harm_logit"], targets["harm"])
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    contract = {
        "minimum_coverage": 0.4,
        "target_dual_precision": 0.85,
        "target_wilson_lower_95": 0.80,
        "maximum_harm_rate": 0.15,
    }
    result = choose_candidate_level_cem_frontier(
        {"scand": {
            "prediction": prediction,
            "p_dual": p_dual,
            "p_harm": p_harm,
            "valid": part["valid"],
            "targets": part["targets_raw"],
        }},
        contract,
        shared_policy=False,
    )["domains"]["scand"]
    coverage = [
        row
        for row in result["frontier"]
        if row["metrics"]["coverage"] >= 0.40
    ]
    best_dual = max(
        coverage,
        key=lambda row: (
            row["metrics"]["dual_precision"] or -1.0,
            -(row["metrics"]["harm_rate"] or 1.0),
            row["metrics"]["coverage"],
        ),
        default=None,
    )
    return {
        "name": name,
        "members": len(records),
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "selected_frontier_feasible": result["feasible"],
        "selected_policy": result["policy"],
        "selected_metrics": result["metrics"],
        "feasible_policy_count": int(sum(row["feasible"] for row in result["frontier"])),
        "best_dual_at_coverage": best_dual,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = read_json(implementation.BASE_RESULT)
    v66_result = read_json(implementation.OUTPUT / "result.json")
    v67_path = (
        ROOT
        / "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v67_head_hard_negative/training/result.json"
    )
    v67_result = read_json(v67_path)
    envelopes = implementation.load_envelopes(Path(base["support"]))
    part = implementation.load_domain_split("scand", "scorer_cal", envelopes)
    values = [
        audit("original_arm_d_oof5", base["arms"]["D"]["checkpoints"]["oof"], ensemble_predict, part),
        audit("v66_all_parameter_oof5", v66_result["checkpoints"]["oof"], ensemble_predict, part),
        audit("v67_head_hard_oof5", v67_result["checkpoints"]["oof"], ensemble_predict_compatible, part),
    ]
    payload = {
        "schema": "dualdomain_cem_tail_oof_ensemble_diagnosis",
        "status": "complete_without_A_to_H_or_reserve",
        "models": values,
        "OOF_ensemble_is_diagnostic_not_frozen_for_replay": True,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
