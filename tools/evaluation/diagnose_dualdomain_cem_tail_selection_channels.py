#!/usr/bin/env python3
"""Test existing SCAND prediction channels as safe-candidate ranking signals."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.planning.run_cem_safe_elite_v55 import candidate_safety_mask  # noqa: E402
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as v66  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import candidate_energy  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v67_finalize import ensemble_predict_compatible  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    policy_grid,
    safety_metrics,
    sigmoid,
)


implementation = v66.implementation
MODELS = {
    "v66_all_parameter": (
        ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/training/result.json",
        ensemble_predict,
    ),
    "v67_scand_head_hard_negative": (
        ROOT
        / "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v67_head_hard_negative/training/result.json",
        ensemble_predict_compatible,
    ),
}
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/dualdomain_cem_tail_v67_head_hard_negative/"
    "selection_channel_diagnosis.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_for_score(score: np.ndarray, safe: np.ndarray) -> np.ndarray:
    masked = np.where(safe, score, np.inf)
    selected = masked.argmin(axis=1).astype(np.int64)
    return np.where(safe.any(axis=1), selected, 0).astype(np.int64)


def rank_z(rank: np.ndarray) -> np.ndarray:
    scale = np.maximum(rank[:, 1:].std(axis=1, keepdims=True), 1.0e-3)
    return (rank - rank[:, :1]) / scale


def channel_scores(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, float],
) -> dict[str, np.ndarray]:
    z = float(policy["z"])
    ucb = prediction["mean"] + z * prediction["std"]
    scores = {
        "conservative_energy": candidate_energy(
            prediction, p_dual, p_harm, valid, policy
        ),
        "harm_probability": p_harm,
        "negative_dual_probability": -p_dual,
        "probability_risk_gap": p_harm - p_dual,
        "max_ate_objective_ucb": np.maximum(ucb[..., 0], ucb[..., 3]),
        "ate_plus_half_objective_ucb": ucb[..., 0] + 0.5 * ucb[..., 3],
        "negative_rank_z": -rank_z(prediction["rank_logit"]),
    }
    for value in scores.values():
        value[:, 0] = np.inf
    return scores


def summarize_channel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    coverage = [row for row in rows if row["metrics"]["coverage"] >= 0.40]
    feasible = [
        row
        for row in coverage
        if row["metrics"]["dual_precision"] is not None
        and row["metrics"]["dual_precision"] >= 0.85
        and row["metrics"]["harm_rate"] is not None
        and row["metrics"]["harm_rate"] <= 0.15
        and row["metrics"]["dual_wilson_lower_95"] is not None
        and row["metrics"]["dual_wilson_lower_95"] >= 0.80
    ]
    best = max(
        coverage,
        key=lambda row: (
            row["metrics"]["dual_precision"] or -1.0,
            -(row["metrics"]["harm_rate"] or 1.0),
            row["metrics"]["coverage"],
        ),
        default=None,
    )
    best_feasible = max(
        feasible,
        key=lambda row: (
            row["metrics"]["coverage"],
            -row["metrics"]["net_delta_per_state_m"]["objective"],
        ),
        default=None,
    )
    return {
        "policies_with_coverage_at_least_0_40": len(coverage),
        "feasible_policies": len(feasible),
        "best_dual_at_coverage": best,
        "maximum_coverage_feasible": best_feasible,
    }


def audit_model(
    result_path: Path,
    predictor: Callable[..., dict[str, np.ndarray]],
    part: dict[str, np.ndarray],
) -> dict[str, Any]:
    result = read_json(result_path)
    if result.get("A_to_H_accessed") or result.get("reserve_accessed"):
        raise RuntimeError(f"forbidden access reported by {result_path}")
    calibration = result["calibration"]["scand"]
    prediction = predictor(
        result["checkpoints"]["full"],
        part["feature"],
        part["domain"],
        device="cuda",
    )
    p_dual = sigmoid(
        prediction["dual_logit"] / float(calibration["dual_temperature"])
    )
    p_harm = sigmoid(
        prediction["harm_logit"] / float(calibration["harm_temperature"])
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    for policy in policy_grid():
        safe = candidate_safety_mask(
            prediction, p_dual, p_harm, part["valid"], policy
        )
        for channel, score in channel_scores(
            prediction, p_dual, p_harm, part["valid"], policy
        ).items():
            selected = selected_for_score(score, safe)
            rows.setdefault(channel, []).append({
                "policy": policy,
                "metrics": safety_metrics(selected, part["targets_raw"]),
            })
    return {
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "dual_temperature": calibration["dual_temperature"],
        "harm_temperature": calibration["harm_temperature"],
        "channels": {
            name: summarize_channel(values) for name, values in rows.items()
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = read_json(implementation.BASE_RESULT)
    envelopes = implementation.load_envelopes(Path(base["support"]))
    part = implementation.load_domain_split("scand", "scorer_cal", envelopes)
    models = {
        name: audit_model(path, predictor, part)
        for name, (path, predictor) in MODELS.items()
    }
    payload = {
        "schema": "dualdomain_cem_tail_selection_channel_diagnosis",
        "status": "complete_without_A_to_H_or_reserve",
        "models": models,
        "selection_rule_role": "diagnostic_only_not_frozen_for_replay",
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
