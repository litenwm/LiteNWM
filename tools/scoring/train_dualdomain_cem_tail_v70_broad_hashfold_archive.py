#!/usr/bin/env python3
"""Broad route-hash OOF calibration for the v70 archived CEM decision."""

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

from scoring.universal_dualdomain_raw_v2 import build_universal_targets, group_fold  # noqa: E402
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
)
from tools.scoring import train_dualdomain_cem_tail_v67_head_hard_negative as head_training  # noqa: E402
from tools.scoring import train_dualdomain_cem_tail_v69_scand_route_crossfit as v69  # noqa: E402
from tools.scoring.run_train_dualdomain_cem_tail_v66_candidate_level import (  # noqa: E402
    candidate_energy,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    binary_nll,
    policy_grid,
    safety_metrics,
    sigmoid,
)


PREREG = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v70_broad_hashfold_archive_preregistration.json"
)
OUTPUT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v70_broad_hashfold_archive"
)
TEMPERATURES = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_temperature(logit: np.ndarray, target: np.ndarray) -> float:
    return float(
        min(
            TEMPERATURES,
            key=lambda value: binary_nll(logit, target, float(value)),
        )
    )


def archive_selection(
    *,
    prediction,
    p_dual,
    p_harm,
    valid,
    policy,
    targets,
    record_key,
    group,
):
    energy = candidate_energy(prediction, p_dual, p_harm, valid, policy)
    safe = candidate_safety_mask(
        prediction, p_dual, p_harm, valid, policy
    )
    unique, first, inverse, counts = np.unique(
        record_key.astype(str),
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if not np.all(counts == 4):
        raise RuntimeError(
            f"archive calibration expects four sets per state, got {np.unique(counts)}"
        )
    if any(len(np.unique(group[inverse == index])) != 1 for index in range(len(unique))):
        raise RuntimeError("archive state crosses exact route groups")
    best_energy = np.full(len(unique), np.inf, dtype=np.float32)
    best_row = np.full(len(unique), -1, dtype=np.int64)
    best_candidate = np.zeros(len(unique), dtype=np.int64)
    for row in range(len(record_key)):
        local = np.flatnonzero(safe[row])
        local = local[local != 0]
        if not len(local):
            continue
        candidate = int(local[np.argmin(energy[row, local])])
        state = int(inverse[row])
        if energy[row, candidate] < best_energy[state]:
            best_energy[state] = energy[row, candidate]
            best_row[state] = row
            best_candidate[state] = candidate
    aggregate_targets = np.empty((len(unique), 2, 4), dtype=np.float32)
    aggregate_targets[:, 0] = targets[first, 0, :4]
    aggregate_targets[:, 1] = aggregate_targets[:, 0]
    promoted = best_row >= 0
    states = np.flatnonzero(promoted)
    aggregate_targets[states, 1] = targets[
        best_row[states], best_candidate[states], :4
    ]
    selected = promoted.astype(np.int64)
    state_group = group[first].astype(str)
    return {
        "metrics": safety_metrics(selected, aggregate_targets),
        "record_key": unique,
        "group": state_group,
        "selected": selected,
        "selected_row": best_row,
        "selected_candidate": best_candidate,
        "targets": aggregate_targets,
    }


def policy_is_pooled_feasible(metrics):
    return bool(
        metrics["coverage"] >= 0.40
        and metrics["dual_precision"] is not None
        and metrics["dual_precision"] >= 0.85
        and metrics["dual_wilson_lower_95"] is not None
        and metrics["dual_wilson_lower_95"] >= 0.80
        and metrics["harm_rate"] is not None
        and metrics["harm_rate"] <= 0.15
    )


def robust_summary(fold_metrics):
    wilson = [
        -1.0 if value["dual_wilson_lower_95"] is None
        else float(value["dual_wilson_lower_95"])
        for value in fold_metrics.values()
    ]
    harm = [
        1.0 if value["harm_rate"] is None else float(value["harm_rate"])
        for value in fold_metrics.values()
    ]
    coverage = [float(value["coverage"]) for value in fold_metrics.values()]
    return {
        "minimum_fold_dual_wilson_lower_95": min(wilson),
        "maximum_fold_harm_rate": max(harm),
        "minimum_fold_coverage": min(coverage),
    }


def calibrate(prediction, combined, targets):
    dual_temperature = fit_temperature(
        prediction["dual_logit"], targets["dual"]
    )
    harm_temperature = fit_temperature(
        prediction["harm_logit"], targets["harm"]
    )
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    fold_id = np.asarray(
        [group_fold("scand", value, 5) for value in combined["group"].astype(str)],
        dtype=np.int64,
    )
    frontier = []
    selections = []
    for policy in policy_grid():
        archived = archive_selection(
            prediction=prediction,
            p_dual=p_dual,
            p_harm=p_harm,
            valid=combined["valid"],
            policy=policy,
            targets=combined["targets_raw"],
            record_key=combined["record_key"],
            group=combined["group"],
        )
        state_fold = np.asarray(
            [group_fold("scand", value, 5) for value in archived["group"]],
            dtype=np.int64,
        )
        fold_metrics = {}
        for fold in range(5):
            local = state_fold == fold
            fold_metrics[str(fold)] = safety_metrics(
                archived["selected"][local], archived["targets"][local]
            )
        pooled = archived["metrics"]
        summary = robust_summary(fold_metrics)
        feasible = policy_is_pooled_feasible(pooled)
        frontier.append({
            "policy": policy,
            "pooled_metrics": pooled,
            "fold_metrics": fold_metrics,
            "robust_summary": summary,
            "pooled_feasible": feasible,
        })
        selections.append(archived)
    feasible_indices = [
        index for index, value in enumerate(frontier)
        if value["pooled_feasible"]
    ]
    if not feasible_indices:
        selected_index = min(
            range(len(frontier)),
            key=lambda index: (
                -frontier[index]["pooled_metrics"]["coverage"],
                frontier[index]["pooled_metrics"]["net_delta_per_state_m"]["objective"],
            ),
        )
    else:
        selected_index = min(
            feasible_indices,
            key=lambda index: (
                -frontier[index]["robust_summary"]["minimum_fold_dual_wilson_lower_95"],
                frontier[index]["robust_summary"]["maximum_fold_harm_rate"],
                -frontier[index]["robust_summary"]["minimum_fold_coverage"],
                -frontier[index]["pooled_metrics"]["coverage"],
                frontier[index]["pooled_metrics"]["net_delta_per_state_m"]["objective"],
            ),
        )
    chosen = frontier[selected_index]
    selection = selections[selected_index]
    return {
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "feasible": bool(feasible_indices),
        "feasible_policy_count": len(feasible_indices),
        "selected_frontier_index": selected_index,
        "policy": chosen["policy"],
        "metrics": chosen["pooled_metrics"],
        "fold_metrics": chosen["fold_metrics"],
        "robust_summary": chosen["robust_summary"],
        "frontier": frontier,
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    prereg = read_json(PREREG)
    if prereg.get("status") != "frozen_after_v69_spent_failure_before_v70_training":
        raise RuntimeError("v70 preregistration changed")
    if prereg.get("A_to_H_used_by_v70_training_or_calibration"):
        raise RuntimeError("v70 preregistration permits forbidden A-H use")
    if prereg.get("reserve_accessed"):
        raise RuntimeError("v70 preregistration reports reserve access")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    base = read_json(v69.BASE_RESULT)
    v69_result = read_json(
        ROOT
        / "outputs/cem_development_v1/"
        "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json"
    )
    envelopes = v69.implementation.load_envelopes(Path(base["support"]))
    train, adaptation, combined, targets, hard, hard_audit = v69.load_combined(
        envelopes
    )
    groups = combined["group"].astype(str)
    folds = np.asarray(
        [group_fold("scand", value, 5) for value in groups], dtype=np.int64
    )
    exact_routes = np.unique(groups)
    broad_routes = np.unique([value.split("::bag::")[0] for value in groups])
    if len(exact_routes) != 36 or len(broad_routes) != 33:
        raise RuntimeError(
            f"v70 route universe changed: exact={len(exact_routes)}, broad={len(broad_routes)}"
        )
    weights = v69.combined_state_weights(groups, combined["set_type"])
    prediction = {
        key: np.empty(
            (
                len(combined["feature"]),
                combined["feature"].shape[1],
                4,
            )
            if key in {"mean", "std"}
            else (
                len(combined["feature"]),
                combined["feature"].shape[1],
            ),
            dtype=np.float32,
        )
        for key in (
            "mean", "std", "helpful_logit", "dual_logit", "harm_logit", "rank_logit"
        )
    }
    base_oof = base["arms"]["D"]["checkpoints"]["oof"]
    if len(base_oof) != 5:
        raise RuntimeError("original Arm-D OOF checkpoint count changed")
    checkpoint_records = []
    old_multiplier = head_training.HARD_MULTIPLIER
    head_training.HARD_MULTIPLIER = v69.HARD_MULTIPLIER
    try:
        for fold in range(5):
            heldout = folds == fold
            train_index = np.flatnonzero(~heldout)
            records = []
            for member in range(3):
                records.append(head_training.fine_tune_member(
                    initial=base_oof[fold],
                    feature=combined["feature"],
                    targets=targets,
                    hard_negative=hard,
                    base_weight=weights,
                    train_index=train_index,
                    seed=20261400 + 10 * fold + member,
                    device=args.device,
                    checkpoint=(
                        args.output_dir
                        / "oof_checkpoints"
                        / f"fold{fold}_seed{member}.pt"
                    ),
                    role=f"broad_hashfold_{fold}_seed_{member}",
                    epochs=args.epochs,
                ))
            local = v69.predict_crossfit_ensemble(
                records,
                combined["feature"][heldout],
                combined["domain"][heldout],
                device=args.device,
            )
            for key in prediction:
                prediction[key][heldout] = local[key]
            checkpoint_records.append({
                "heldout_fold": fold,
                "heldout_states": int(heldout.sum()),
                "heldout_exact_routes": int(len(np.unique(groups[heldout]))),
                "members": records,
            })
    finally:
        head_training.HARD_MULTIPLIER = old_multiplier

    calibration = calibrate(prediction, combined, targets)
    selection = calibration.pop("selection")
    frontier = calibration.pop("frontier")
    prediction_path = args.output_dir / "broad_hashfold_oof_predictions.npz"
    write_npz_atomic(
        prediction_path,
        **prediction,
        record_key=combined["record_key"],
        route_group=groups,
        fold_id=folds,
    )
    selection_path = args.output_dir / "calibration_archive_selection.npz"
    write_npz_atomic(
        selection_path,
        record_key=selection["record_key"],
        route_group=selection["group"],
        selected=selection["selected"],
        selected_row=selection["selected_row"],
        selected_candidate=selection["selected_candidate"],
    )
    frontier_path = args.output_dir / "calibration_frontier.json"
    write_json_atomic(frontier_path, {
        "schema": "v70_broad_hashfold_archive_frontier",
        "selection_contract": prereg["calibration_contract"],
        "frontier": frontier,
        "selected_frontier_index": calibration["selected_frontier_index"],
        "A_to_H_used": False,
        "fresh_scand_used": False,
        "reserve_accessed": False,
    })
    result = {
        "schema": "dualdomain_cem_tail_v70_broad_hashfold_archive_result",
        "status": (
            "frozen_after_v69_spent_failure_before_v70_spent_replay"
            if calibration["feasible"]
            else "broad_hashfold_archive_calibration_infeasible"
        ),
        "preregistration": str(PREREG),
        "preregistration_sha256": file_sha256(PREREG),
        "route_universe": {
            "exact_routes": int(len(exact_routes)),
            "broad_routes": int(len(broad_routes)),
            "fold_states": np.bincount(folds, minlength=5).tolist(),
            "fold_exact_routes": [
                int(len(np.unique(groups[folds == fold]))) for fold in range(5)
            ],
        },
        "train_fit_states": int(len(train["feature"])),
        "scorer_cal_states": int(len(adaptation["feature"])),
        "combined_set_states": int(len(combined["feature"])),
        "combined_archive_states": int(len(selection["record_key"])),
        "hard_negative_audit": hard_audit,
        "calibration": {
            "recon": v69_result["calibration"]["recon"],
            "scand": {
                **calibration,
                "policy_source": "36-route five-hashfold OOF cross-round archive",
            },
        },
        "checkpoints": {
            "calibration_oof": checkpoint_records,
            "full": v69_result["checkpoints"]["full"],
        },
        "full_checkpoint_source": str(
            ROOT
            / "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json"
        ),
        "full_checkpoint_source_sha256": file_sha256(
            ROOT
            / "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/training/result.json"
        ),
        "oof_predictions": str(prediction_path),
        "oof_predictions_sha256": file_sha256(prediction_path),
        "archive_selection": str(selection_path),
        "archive_selection_sha256": file_sha256(selection_path),
        "frontier": str(frontier_path),
        "frontier_sha256": file_sha256(frontier_path),
        "recon_output_bit_exact": True,
        "A_to_H_already_spent_by_v69": True,
        "A_to_H_used_by_v70_training_or_calibration": False,
        "fresh_scand_already_spent_by_v69": True,
        "fresh_scand_used_by_v70_training_or_calibration": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(result_path, result)
    print(json.dumps({
        "status": result["status"],
        "route_universe": result["route_universe"],
        "calibration": result["calibration"],
        "result": str(result_path),
        "result_sha256": file_sha256(result_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
