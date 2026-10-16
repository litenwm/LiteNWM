#!/usr/bin/env python3
"""Route-crossfit SCAND proposal-tail adaptation with RECON bit-exact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    UniversalDualDomainRawV2,
    UniversalModelConfig,
    build_universal_targets,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring import run_train_dualdomain_cem_tail_v66_frozen as v66  # noqa: E402
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as head_training  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    choose_candidate_level_cem_frontier,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    ensemble_predict,
    predict_member,
    sigmoid,
)


implementation = v66.implementation
BASE_RESULT = implementation.BASE_RESULT
PREREG = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v69_scand_route_crossfit_preregistration.json"
)
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v69_scand_route_crossfit"
)
TEMP_GRID = implementation.TEMP_GRID
HARD_MULTIPLIER = 3.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def combined_state_weights(group: np.ndarray, set_type: np.ndarray) -> np.ndarray:
    result = np.zeros(len(group), dtype=np.float64)
    group = group.astype(str)
    set_type = set_type.astype(str)
    for name, total in implementation.SET_WEIGHTS.items():
        mask = set_type == name
        routes = np.unique(group[mask])
        if not len(routes):
            raise RuntimeError(f"missing combined set type {name}")
        for route in routes:
            local = mask & (group == route)
            result[local] = float(total) / len(routes) / int(local.sum())
    if not np.isclose(result.sum(), 1.0):
        raise RuntimeError(f"combined state weights sum to {result.sum()}")
    return result


def load_combined(envelopes):
    train = implementation.load_domain_split("scand", "train_fit", envelopes)
    adaptation = implementation.load_domain_split("scand", "scorer_cal", envelopes)
    combined = {
        key: np.concatenate([train[key], adaptation[key]])
        for key in train
    }
    targets = build_universal_targets(combined["targets_raw"])
    train_targets = build_universal_targets(train["targets_raw"])
    train_hard, train_audit = head_training.load_scand_hard_negatives(
        train, train_targets
    )
    adaptation_targets = build_universal_targets(adaptation["targets_raw"])
    states = len(adaptation["feature"]) // 4
    adaptation_hard = np.zeros_like(adaptation_targets["harm"], dtype=bool)
    adaptation_audit = []
    for iteration in range(3):
        score_path = (
            implementation.HARVEST
            / "scand_scorer_cal"
            / f"iteration_{iteration:02d}/scores.npz"
        )
        score = load_npz(score_path)
        start = (iteration + 1) * states
        stop = start + states
        local = (
            score["safe"].astype(bool)
            & adaptation_targets["harm"][start:stop].astype(bool)
        )
        local[:, 0] = False
        adaptation_hard[start:stop] = local
        adaptation_audit.append({
            "iteration": iteration,
            "false_safe_harm_candidates": int(local[:, 1:].sum()),
            "score_sha256": file_sha256(score_path),
        })
    hard = np.concatenate([train_hard, adaptation_hard])
    audit = {
        "multiplier": HARD_MULTIPLIER,
        "train_fit": train_audit,
        "adaptation_total": int(adaptation_hard.sum()),
        "adaptation_per_iteration": adaptation_audit,
        "combined_total": int(hard.sum()),
    }
    return train, adaptation, combined, targets, hard, audit


def load_crossfit_member(record, device):
    path = Path(record["path"])
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"checkpoint hash changed: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "dualdomain_cem_tail_v67_scand_head_member":
        raise RuntimeError(f"unexpected crossfit checkpoint schema: {path}")
    if payload.get("hard_negative_multiplier") != HARD_MULTIPLIER:
        raise RuntimeError(f"crossfit hard-negative multiplier changed: {path}")
    model = UniversalDualDomainRawV2(
        UniversalModelConfig(**payload["model_config"])
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    normalization = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["normalization"].items()
    }
    target_norm = {
        key: np.asarray(value, dtype=np.float32)
        for key, value in payload["target_normalization"].items()
    }
    return model, normalization, target_norm


def predict_crossfit_ensemble(records, feature, domain, *, device):
    predictions = []
    for record in records:
        member = load_crossfit_member(record, device)
        predictions.append(
            predict_member(member, feature, domain, device=device)
        )
        del member
        if device == "cuda":
            torch.cuda.empty_cache()
    means = np.stack([value["mean"] for value in predictions])
    variances = np.stack([value["variance"] for value in predictions])
    mean = means.mean(axis=0)
    total_variance = (
        variances.mean(axis=0)
        + np.square(means).mean(axis=0)
        - np.square(mean)
    )
    return {
        "mean": mean.astype(np.float32),
        "std": np.sqrt(np.maximum(total_variance, 1.0e-8)).astype(np.float32),
        **{
            key: np.stack([value[key] for value in predictions]).mean(axis=0)
            for key in (
                "helpful_logit",
                "dual_logit",
                "harm_logit",
                "rank_logit",
            )
        },
    }


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(
        min(TEMP_GRID, key=lambda value: binary_nll(logit, target, float(value)))
    )


def frontier(prediction, adaptation, targets):
    dual_temperature = fit_temperature(
        prediction["dual_logit"], targets["dual"]
    )
    harm_temperature = fit_temperature(
        prediction["harm_logit"], targets["harm"]
    )
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    contract = {
        "minimum_coverage": 0.4,
        "target_dual_precision": 0.85,
        "target_wilson_lower_95": 0.80,
        "maximum_harm_rate": 0.15,
    }
    value = choose_candidate_level_cem_frontier(
        {"scand": {
            "prediction": prediction,
            "p_dual": p_dual,
            "p_harm": p_harm,
            "valid": adaptation["valid"],
            "targets": adaptation["targets_raw"],
        }},
        contract,
        shared_policy=False,
    )["domains"]["scand"]
    return {
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "policy": value["policy"],
        "metrics": value["metrics"],
        "feasible": value["feasible"],
        "selected": value["selected"],
        "feasible_policy_count": int(
            sum(row["feasible"] for row in value["frontier"])
        ),
    }


def crossfit(args, base, envelopes):
    result_path = args.output_dir / "crossfit_result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    train, adaptation, combined, targets, hard, hard_audit = load_combined(
        envelopes
    )
    weights = combined_state_weights(combined["group"], combined["set_type"])
    train_count = len(train["feature"])
    adaptation_group = adaptation["group"].astype(str)
    unique_routes = np.unique(adaptation_group)
    if len(unique_routes) != 5:
        raise RuntimeError(f"expected five exact adaptation routes, got {len(unique_routes)}")
    prediction = {
        key: np.empty(
            (
                len(adaptation["feature"]),
                adaptation["feature"].shape[1],
                4,
            )
            if key in {"mean", "std"}
            else (
                len(adaptation["feature"]),
                adaptation["feature"].shape[1],
            ),
            dtype=np.float32,
        )
        for key in (
            "mean",
            "std",
            "helpful_logit",
            "dual_logit",
            "harm_logit",
            "rank_logit",
        )
    }
    fold_records = []
    base_records = base["arms"]["D"]["checkpoints"]["full"]
    old_multiplier = head_training.HARD_MULTIPLIER
    head_training.HARD_MULTIPLIER = HARD_MULTIPLIER
    try:
        for fold, route in enumerate(unique_routes):
            heldout_local = np.flatnonzero(adaptation_group == route)
            heldout_combined = train_count + heldout_local
            train_index = np.setdiff1d(
                np.arange(len(combined["feature"]), dtype=np.int64),
                heldout_combined,
                assume_unique=True,
            )
            records = []
            for member_index, initial in enumerate(base_records):
                records.append(head_training.fine_tune_member(
                    initial=initial,
                    feature=combined["feature"],
                    targets=targets,
                    hard_negative=hard,
                    base_weight=weights,
                    train_index=train_index,
                    seed=20261210 + 10 * fold + member_index,
                    device=args.device,
                    checkpoint=(
                        args.output_dir
                        / "crossfit_checkpoints"
                        / f"fold{fold}_seed{member_index}.pt"
                    ),
                    role=f"route_crossfit_{fold}_seed_{member_index}",
                    epochs=args.epochs,
                ))
            local_prediction = predict_crossfit_ensemble(
                records,
                adaptation["feature"][heldout_local],
                adaptation["domain"][heldout_local],
                device=args.device,
            )
            for key in prediction:
                prediction[key][heldout_local] = local_prediction[key]
            fold_records.append({
                "fold": fold,
                "heldout_route": str(route),
                "heldout_states": int(len(heldout_local)),
                "members": records,
            })
    finally:
        head_training.HARD_MULTIPLIER = old_multiplier
    adaptation_targets = build_universal_targets(adaptation["targets_raw"])
    calibration = frontier(prediction, adaptation, adaptation_targets)
    prediction_path = args.output_dir / "crossfit_oof_predictions.npz"
    write_npz_atomic(
        prediction_path,
        **prediction,
        record_key=adaptation["record_key"],
        route_group=adaptation_group,
        selected_index=calibration.pop("selected"),
    )
    result = {
        "schema": "dualdomain_cem_tail_v69_route_crossfit_result",
        "status": (
            "route_crossfit_scorer_cal_feasible"
            if calibration["feasible"]
            else "route_crossfit_scorer_cal_infeasible_A_to_H_sealed"
        ),
        "folds": fold_records,
        "adaptation_routes": int(len(unique_routes)),
        "train_fit_states": int(len(train["feature"])),
        "adaptation_states": int(len(adaptation["feature"])),
        "hard_negative_audit": hard_audit,
        "calibration": calibration,
        "crossfit_oof_predictions": str(prediction_path),
        "crossfit_oof_predictions_sha256": file_sha256(prediction_path),
        "preregistration": str(PREREG),
        "preregistration_sha256": file_sha256(PREREG),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(result_path, result)
    print(json.dumps({
        "stage": "v69_route_crossfit",
        "status": result["status"],
        "calibration": calibration,
        "output": str(result_path),
        "sha256": file_sha256(result_path),
    }, indent=2))


def main():
    args = parse_args()
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_before_route_crossfit_training":
        raise RuntimeError("v69 preregistration changed")
    if prereg.get("A_to_H_accessed") or prereg.get("reserve_accessed"):
        raise RuntimeError("v69 preregistration reports forbidden access")
    base = read_json(BASE_RESULT)
    envelopes = implementation.load_envelopes(Path(base["support"]))
    if args.stage == "crossfit":
        crossfit(args, base, envelopes)
    else:
        raise NotImplementedError("final training is enabled only after feasible crossfit")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("crossfit", "finalize"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    main()
