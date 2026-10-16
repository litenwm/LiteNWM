#!/usr/bin/env python3
"""Train the frozen six-arm SCAND support/raw domain-adaptation experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import (  # noqa: E402
    DenseHeadConfig,
    MODES,
    build_meta_features,
    build_targets,
    empty_prediction,
    ensemble_predict,
    family_fold,
    feature_by_mode,
    fit_fold,
    fit_ridge,
    predict,
    predict_ridge,
    residual_rows,
    selected_indices,
    selection_metrics,
)
from scoring.safe_promotion_v1 import (  # noqa: E402
    conservative_utility,
    paired_relative_gains,
)
from scoring.spacing_support_v1 import (  # noqa: E402
    fit_spacing_envelope,
    serializable_envelope,
    support_mask,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
    scorer_scales,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    store_prediction,
    write_json,
    write_npz,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/scand/scand_domain_adaptation_v1.json",
    )
    parser.add_argument(
        "--scand-train-features",
        type=Path,
        default=DEFAULT_ROOT
        / "features/train_fit/scand_inner_train_fit_spacing038_features.npz",
    )
    parser.add_argument(
        "--scand-cal-features",
        type=Path,
        default=DEFAULT_ROOT
        / "features/scorer_cal/scand_inner_scorer_cal_spacing038_features.npz",
    )
    parser.add_argument(
        "--recon-train-features",
        type=Path,
        default=REPO_ROOT
        / "outputs/recon_dense_meta_dinov3_v1/features/dense_train_features.npz",
    )
    parser.add_argument(
        "--scorer-checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints/project/scorer_v2",
    )
    parser.add_argument(
        "--spent-manifest",
        type=Path,
        default=REPO_ROOT
        / "data/evaluation/scand/strict_test_h4_100_seed17/manifest.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_ROOT / "training_v1"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def trajectory_id(record_key: str) -> str:
    trajectory, separator, timestep = str(record_key).rpartition(":")
    if not separator or not trajectory or not timestep.isdigit():
        raise ValueError(f"invalid record key: {record_key}")
    return trajectory


def spent_trajectories(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["trajectory_id"]) for row in value["records"]}


def state_subset(data: dict[str, Any], index: np.ndarray) -> dict[str, np.ndarray]:
    keys = (
        "candidates",
        "actions_h4",
        "action_features",
        "static_features",
        "raw_ltwm_features",
        "targets",
        "family",
        "record_key",
    )
    return {key: np.asarray(data[key])[index] for key in keys}


def route_balanced_sample(family: np.ndarray, size: int, seed: int) -> np.ndarray:
    families = np.asarray(family).astype(str)
    unique = sorted(np.unique(families).tolist())
    if size > len(families):
        raise ValueError("balanced sample exceeds SCAND train-fit")
    rng = np.random.default_rng(seed)
    queues: dict[str, list[int]] = {}
    for name in unique:
        indices = np.flatnonzero(families == name)
        rng.shuffle(indices)
        queues[name] = indices.tolist()
    order = unique.copy()
    rng.shuffle(order)
    chosen: list[int] = []
    while len(chosen) < size:
        progressed = False
        for name in order:
            if queues[name]:
                chosen.append(queues[name].pop())
                progressed = True
                if len(chosen) == size:
                    break
        if not progressed:
            raise RuntimeError("route-balanced sampler exhausted unexpectedly")
    return np.asarray(sorted(chosen), dtype=np.int64)


def concatenate_domain_data(
    recon: dict[str, Any], scand: dict[str, Any], scand_index: np.ndarray
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    recon_index = np.arange(len(recon["family"]), dtype=np.int64)
    recon_data = state_subset(recon, recon_index)
    scand_data = state_subset(scand, scand_index)
    result: dict[str, np.ndarray] = {}
    for key in (
        "candidates",
        "actions_h4",
        "action_features",
        "static_features",
        "raw_ltwm_features",
        "targets",
        "record_key",
    ):
        result[key] = np.concatenate((recon_data[key], scand_data[key]), axis=0)
    result["family"] = np.concatenate(
        (
            np.asarray([f"recon::{value}" for value in recon_data["family"].astype(str)]),
            scand_data["family"].astype(str),
        )
    )
    spacing = np.concatenate(
        (
            np.full(len(recon_index), 0.25, dtype=np.float32),
            np.full(len(scand_index), 0.38, dtype=np.float32),
        )
    )
    return result, spacing


def metrics_without_delta(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "selected_delta"}


def augment_safety_metrics(selected: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    result = selection_metrics(selected, targets)
    delta = result["selected_delta"]
    promote = np.asarray(selected) != 0
    harmed = promote & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    result["harm_count"] = int(harmed.sum())
    result["harm_rate_per_promotion"] = (
        float(harmed.sum() / promote.sum()) if promote.any() else None
    )
    result["mean_harm_magnitude_m"] = (
        float(np.maximum(delta[harmed, 0], delta[harmed, 3]).mean())
        if harmed.any()
        else 0.0
    )
    return metrics_without_delta(result)


def feasible(metrics: dict[str, Any], calibration: dict[str, Any]) -> bool:
    return bool(
        metrics["promotions"] >= int(calibration["minimum_promotions"])
        and metrics["dual_precision"] is not None
        and metrics["dual_precision"] >= float(calibration["target_dual_precision"])
        and metrics["dual_wilson_lower_95"] is not None
        and metrics["dual_wilson_lower_95"]
        >= float(calibration["target_dual_wilson_lower_95"])
    )


def choose_calibration(candidates: list[dict[str, Any]], calibration: dict[str, Any]) -> dict[str, Any]:
    passing = [value for value in candidates if feasible(value["metrics"], calibration)]
    if passing:
        return min(
            passing,
            key=lambda value: (
                value["metrics"]["net_ate_delta_per_state_m"]
                + value["metrics"]["net_objective_delta_per_state_m"],
                -value["metrics"]["promotions"],
                canonical_json(jsonable(value.get("tie_break", {}))),
            ),
        )
    return min(
        candidates,
        key=lambda value: (
            -(
                value["metrics"]["dual_wilson_lower_95"]
                if value["metrics"]["dual_wilson_lower_95"] is not None
                else -1.0
            ),
            value["metrics"]["net_objective_delta_per_state_m"],
            -value["metrics"]["promotions"],
            canonical_json(jsonable(value.get("tie_break", {}))),
        ),
    )


def threshold_grid(score: np.ndarray, quantiles: list[float]) -> list[float]:
    finite = np.asarray(score, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [1.0]
    return sorted(
        set(
            float(np.quantile(finite, value, method="higher"))
            for value in quantiles
        )
    )


def train_groupfirst_regime(
    name: str,
    train: dict[str, np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
    device: str,
    train_sha: str,
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]],
    dict[str, Any],
]:
    training = config["training"]
    folds = int(training["folds"])
    dense_config = DenseHeadConfig(
        hidden=int(training["hidden_dim"]),
        dropout=float(training["dropout"]),
        epochs=int(training["epochs"]),
        batch_groups=max(1, int(round(int(training["batch_size"]) / 17))),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        target_temperature=0.05,
    )
    dense_config.validate()
    families = train["family"].astype(str)
    family_to_fold = {
        family: family_fold(family, folds) for family in np.unique(families)
    }
    fold = np.asarray([family_to_fold[value] for value in families], dtype=np.int64)
    family_counts = [
        int(sum(value == index for value in family_to_fold.values()))
        for index in range(folds)
    ]
    if min(family_counts) == 0:
        raise RuntimeError(f"{name} has an empty family fold: {family_counts}")
    targets = build_targets(train["targets"], dense_config.target_temperature)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    members: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]] = {}
    records: dict[str, list[dict[str, Any]]] = {}
    for mode_index, mode in enumerate(MODES):
        feature = feature_by_mode(train, mode, variant="raw")
        oof = empty_prediction(len(feature), feature.shape[1])
        members[mode] = []
        records[mode] = []
        for heldout in range(folds):
            fit_index = np.flatnonzero(fold != heldout)
            oof_index = np.flatnonzero(fold == heldout)
            seed = int(training["seed"]) + mode_index * 1000 + heldout
            model, stats = fit_fold(
                feature,
                targets,
                fit_index,
                dense_config,
                seed=seed,
                device=device,
            )
            store_prediction(
                oof,
                oof_index,
                predict(model, stats, feature, oof_index, device=device),
            )
            checkpoint = output_dir / "checkpoints" / f"{name}_{mode}_fold{heldout}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": "scand_domain_adaptation_groupfirst_member_v1",
                    "domain_regime": name,
                    "mode": mode,
                    "feature_variant": "raw",
                    "scorer_v2_predictions_in_features": False,
                    "heldout_family_fold": heldout,
                    "seed": seed,
                    "input_dim": int(feature.shape[-1]),
                    "config": asdict(dense_config),
                    "normalization": stats,
                    "state_dict": model.state_dict(),
                    "training_feature_sha256": train_sha,
                    "family_fold_counts": family_counts,
                    "fixed_pool_only": True,
                    "cem": False,
                },
                checkpoint,
            )
            members[mode].append((model, stats))
            records[mode].append(
                {
                    "fold": heldout,
                    "path": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                }
            )
            print(
                canonical_json(
                    {
                        "stage": "fit_groupfirst",
                        "domain_regime": name,
                        "mode": mode,
                        "fold": heldout,
                        "folds": folds,
                    }
                ),
                flush=True,
            )
        predictions[mode] = oof
    audit = {
        "states": int(len(families)),
        "families": int(len(np.unique(families))),
        "family_fold_counts": family_counts,
        "config": asdict(dense_config),
        "checkpoints": records,
    }
    return predictions, members, audit


def ensemble_on_cal(
    members: dict[str, list[tuple[torch.nn.Module, dict[str, np.ndarray]]]],
    cal: dict[str, Any],
    device: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        mode: ensemble_predict(
            members[mode],
            feature_by_mode(cal, mode, variant="raw"),
            device=device,
        )
        for mode in MODES
    }


def calibrate_mode(
    action: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    targets: np.ndarray,
    valid: np.ndarray,
    mode: str,
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    configs = (
        [(0.0, "linear", "action")]
        if mode == "action"
        else [
            (alpha, gate, risk)
            for alpha in (0.25, 0.50, 0.75, 1.0)
            for gate in ("linear", "min_confirm", "geometric")
            for risk in ("action", "residual", "both", "blend")
        ]
    )
    candidates: list[dict[str, Any]] = []
    for alpha, gate, risk in configs:
        rows = residual_rows(
            action,
            residual,
            alpha=alpha,
            gate=gate,
            risk=risk,
            true_target=targets,
            valid=valid,
        )
        eligible = (
            (rows["challenger"] != 0)
            & (rows["predicted_delta"][:, 0] < 0.0)
            & (rows["predicted_delta"][:, 3] < 0.0)
        )
        thresholds = threshold_grid(
            rows["p_gate"][eligible],
            [float(value) for value in calibration["promotion_threshold_quantiles"]],
        )
        for threshold in thresholds:
            selected = selected_indices(rows, threshold)
            metrics = augment_safety_metrics(selected, targets)
            candidates.append(
                {
                    "metrics": metrics,
                    "alpha": float(alpha),
                    "gate": gate,
                    "risk": risk,
                    "threshold": float(threshold),
                    "rows": rows,
                    "selected": selected,
                    "tie_break": {
                        "alpha": alpha,
                        "gate": gate,
                        "risk": risk,
                        "threshold": threshold,
                    },
                }
            )
    best = choose_calibration(candidates, calibration)
    policy = {
        "alpha": best["alpha"],
        "gate": best["gate"],
        "risk": best["risk"],
        "threshold": best["threshold"],
        "calibration_metrics": best["metrics"],
        "feasible_gate": feasible(best["metrics"], calibration),
    }
    summary = [
        {
            key: value
            for key, value in candidate.items()
            if key not in {"rows", "selected"}
        }
        for candidate in candidates
    ]
    return policy, best["rows"], summary


def calibrate_meta(
    predictions: dict[str, dict[str, np.ndarray]],
    rows: dict[str, dict[str, np.ndarray]],
    selections: dict[str, np.ndarray],
    cal: dict[str, Any],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    data = {
        key: np.asarray(cal[key])
        for key in ("action_features", "static_features", "raw_ltwm_features")
    }
    feature = build_meta_features(predictions, rows, data, variant="raw")
    targets = cal["targets"]
    row = np.arange(len(targets))
    static_delta = targets[row, selections["static"], :4]
    full_delta = targets[row, selections["full"], :4]
    static_cost = static_delta[:, 0] + 0.5 * static_delta[:, 3]
    full_cost = full_delta[:, 0] + 0.5 * full_delta[:, 3]
    target = (full_cost < static_cost).astype(np.float32)
    families = cal["family"].astype(str)
    unique = sorted(np.unique(families).tolist())
    if len(unique) < 3:
        raise RuntimeError("meta calibration needs at least three route groups")
    mapping = {family: index for index, family in enumerate(unique)}
    fold = np.asarray([mapping[value] for value in families], dtype=np.int64)
    candidates: list[dict[str, Any]] = []
    for regularization in [float(value) for value in calibration["meta_ridge_alphas"]]:
        oof = np.empty(len(feature), dtype=np.float32)
        for heldout in range(len(unique)):
            fit_index = np.flatnonzero(fold != heldout)
            test_index = np.flatnonzero(fold == heldout)
            model = fit_ridge(
                feature[fit_index], target[fit_index], regularization=regularization
            )
            oof[test_index] = predict_ridge(model, feature[test_index])
        for threshold in [float(value) for value in calibration["meta_thresholds"]]:
            selected = np.where(
                oof >= threshold, selections["full"], selections["static"]
            ).astype(np.int64)
            metrics = augment_safety_metrics(selected, targets)
            candidates.append(
                {
                    "metrics": metrics,
                    "regularization": regularization,
                    "threshold": threshold,
                    "oof_score": oof,
                    "selected": selected,
                    "tie_break": {
                        "regularization": regularization,
                        "threshold": threshold,
                    },
                }
            )
    best = choose_calibration(candidates, calibration)
    final_model = fit_ridge(
        feature, target, regularization=float(best["regularization"])
    )
    policy = {
        "feature_mean": np.asarray(final_model["feature_mean"]).tolist(),
        "feature_scale": np.asarray(final_model["feature_scale"]).tolist(),
        "coefficient": np.asarray(final_model["coefficient"]).tolist(),
        "intercept": float(final_model["intercept"]),
        "regularization": float(best["regularization"]),
        "threshold": float(best["threshold"]),
        "target": "full_has_lower_ATE_plus_half_objective_delta_than_static",
        "route_group_oof_folds": len(unique),
        "calibration_metrics": best["metrics"],
        "feasible_gate": feasible(best["metrics"], calibration),
    }
    summary = {
        "route_groups": unique,
        "candidate_count": len(candidates),
        "selected_regularization": float(best["regularization"]),
        "selected_threshold": float(best["threshold"]),
        "selected_metrics": best["metrics"],
    }
    return policy, best["selected"], summary


def calibrate_raw_branch(
    predictions: dict[str, dict[str, np.ndarray]],
    cal: dict[str, Any],
    valid: np.ndarray,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    policies: dict[str, Any] = {}
    rows: dict[str, dict[str, np.ndarray]] = {}
    searches: dict[str, Any] = {}
    selections: dict[str, np.ndarray] = {}
    for mode in MODES:
        policy, mode_rows, search = calibrate_mode(
            predictions["action"],
            predictions[mode],
            cal["targets"],
            valid,
            mode,
            calibration,
        )
        policies[mode] = policy
        rows[mode] = mode_rows
        searches[mode] = search
        selections[mode] = selected_indices(mode_rows, float(policy["threshold"]))
    meta_policy, selected, meta_summary = calibrate_meta(
        predictions, rows, selections, cal, calibration
    )
    return {
        "policies": policies,
        "meta": meta_policy,
        "meta_summary": meta_summary,
        "selected": selected,
        "metrics": augment_safety_metrics(selected, cal["targets"]),
        "search_points": {mode: len(searches[mode]) for mode in MODES},
    }


def support_outcome_audit(valid: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    proposal_valid = valid[:, 1:]
    proposal_target = targets[:, 1:, :4]
    harmful = (proposal_target[..., 0] > 1.0e-7) | (
        proposal_target[..., 3] > 1.0e-7
    )
    dual_benefit = (proposal_target[..., 0] < -1.0e-7) & (
        proposal_target[..., 3] < -1.0e-7
    )
    accepted = int(proposal_valid.sum())
    rejected_opportunity_window = ((~proposal_valid) & dual_benefit).any(axis=1)
    return {
        "accepted_candidates": accepted,
        "accepted_harm_rate": (
            float(harmful[proposal_valid].mean()) if accepted else None
        ),
        "rejected_oracle_opportunity_window_rate": float(
            rejected_opportunity_window.mean()
        ),
    }


def native_selection(
    cal: dict[str, Any],
    valid: np.ndarray,
    ate_scale: float,
    rtp_scale: float,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    gains = paired_relative_gains(
        cal["ate_by_seed"], cal["rtp_by_seed"], uncertainty_beta=1.0
    )
    utility = conservative_utility(
        gains, ate_scale=ate_scale, rtp_scale=rtp_scale
    )
    allowed = np.asarray(valid, dtype=bool).copy()
    allowed[:, 0] = False
    ranked = np.where(allowed, utility, -np.inf)
    challenger = ranked.argmax(axis=1).astype(np.int64)
    exists = allowed.any(axis=1)
    challenger = np.where(exists, challenger, 0)
    row = np.arange(len(challenger))
    score = np.where(exists, utility[row, challenger], -np.inf)
    candidates = []
    for threshold in threshold_grid(
        score[exists],
        [float(value) for value in calibration["promotion_threshold_quantiles"]],
    ):
        applied = max(0.0, float(threshold))
        selected = np.where(exists & (score >= applied), challenger, 0).astype(np.int64)
        metrics = augment_safety_metrics(selected, cal["targets"])
        candidates.append(
            {
                "metrics": metrics,
                "threshold": applied,
                "selected": selected,
                "tie_break": {"threshold": applied},
            }
        )
    best = choose_calibration(candidates, calibration)
    return {
        "threshold": float(best["threshold"]),
        "uncertainty_beta": 1.0,
        "selected": best["selected"],
        "metrics": best["metrics"],
        "feasible_gate": feasible(best["metrics"], calibration),
        "search_points": len(candidates),
    }


def save_support_checkpoint(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(value)
        for key, value in model.items()
        if isinstance(value, np.ndarray)
    }
    arrays["metadata"] = np.asarray(
        canonical_json(
            {
                key: value
                for key, value in model.items()
                if not isinstance(value, np.ndarray)
            }
        )
    )
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def calibrate_regime(
    name: str,
    train: dict[str, np.ndarray],
    spacing: np.ndarray,
    cal: dict[str, Any],
    predictions: dict[str, dict[str, np.ndarray]],
    config: dict[str, Any],
    ate_scale: float,
    rtp_scale: float,
    output_dir: Path,
) -> dict[str, Any]:
    calibration = config["calibration"]
    support_config = config["support"]
    envelope = fit_spacing_envelope(
        train["actions_h4"],
        spacing,
        physical_guard_quantile=float(support_config["physical_guard_quantile"]),
    )
    support_path = output_dir / "support" / f"{name}_spacing_envelope.npz"
    save_support_checkpoint(support_path, envelope)
    cal_spacing = np.full(len(cal["family"]), 0.38, dtype=np.float32)
    support_candidates: list[dict[str, Any]] = []
    raw_candidates: list[dict[str, Any]] = []
    for quantile in [float(value) for value in support_config["acceptance_quantiles"]]:
        valid, acceptance = support_mask(
            envelope, cal["actions_h4"], cal_spacing, quantile=quantile
        )
        acceptance.update(support_outcome_audit(valid, cal["targets"]))
        native = native_selection(
            cal, valid, ate_scale, rtp_scale, calibration
        )
        support_candidates.append(
            {
                "metrics": native["metrics"],
                "quantile": quantile,
                "support_audit": acceptance,
                "native": native,
                "selected": native["selected"],
                "tie_break": {"quantile": quantile},
            }
        )
        raw = calibrate_raw_branch(predictions, cal, valid, calibration)
        raw_candidates.append(
            {
                "metrics": raw["metrics"],
                "quantile": quantile,
                "support_audit": acceptance,
                "raw": raw,
                "selected": raw["selected"],
                "tie_break": {"quantile": quantile},
            }
        )
        print(
            canonical_json(
                {
                    "stage": "calibrate_spacing_support",
                    "domain_regime": name,
                    "quantile": quantile,
                    "candidate_acceptance": acceptance["candidate_acceptance"],
                    "no_supported_proposal_windows": acceptance[
                        "no_supported_proposal_windows"
                    ],
                }
            ),
            flush=True,
        )
    support_best = choose_calibration(support_candidates, calibration)
    support_raw_best = choose_calibration(raw_candidates, calibration)
    raw_only = calibrate_raw_branch(
        predictions, cal, cal["majority_support"], calibration
    )
    raw_only_support_audit = support_outcome_audit(
        cal["majority_support"], cal["targets"]
    )
    raw_only_support_audit.update(
        {
            "candidate_acceptance": float(cal["majority_support"][:, 1:].mean()),
            "window_any_proposal_acceptance": float(
                cal["majority_support"][:, 1:].any(axis=1).mean()
            ),
            "no_supported_proposal_windows": int(
                (~cal["majority_support"][:, 1:].any(axis=1)).sum()
            ),
            "source": "frozen_scorer_v2_majority_support",
        }
    )
    arms = {
        "support_only": {
            "role": "support_gain_ablation",
            "ranker": "frozen_native_scorer_v2",
            "support_quantile": float(support_best["quantile"]),
            "support_audit": support_best["support_audit"],
            "promotion_threshold": float(support_best["native"]["threshold"]),
            "metrics": support_best["metrics"],
            "feasible_gate": feasible(support_best["metrics"], calibration),
            "selected": support_best["selected"],
        },
        "raw_only": {
            "role": "raw_model_gain_ablation",
            "ranker": "scand_adapted_full_meta_raw",
            "support": "frozen_scorer_v2_majority_support",
            "support_audit": raw_only_support_audit,
            "policies": raw_only["policies"],
            "meta": raw_only["meta"],
            "metrics": raw_only["metrics"],
            "feasible_gate": feasible(raw_only["metrics"], calibration),
            "selected": raw_only["selected"],
        },
        "support_plus_raw": {
            "role": "primary",
            "ranker": "scand_adapted_full_meta_raw",
            "support_quantile": float(support_raw_best["quantile"]),
            "support_audit": support_raw_best["support_audit"],
            "policies": support_raw_best["raw"]["policies"],
            "meta": support_raw_best["raw"]["meta"],
            "metrics": support_raw_best["metrics"],
            "feasible_gate": feasible(support_raw_best["metrics"], calibration),
            "selected": support_raw_best["selected"],
        },
    }
    selected_path = output_dir / "calibration" / f"{name}_selected_index.npz"
    write_npz(
        selected_path,
        record_key=cal["record_key"].astype(str),
        family=cal["family"].astype(str),
        support_only=arms["support_only"].pop("selected"),
        raw_only=arms["raw_only"].pop("selected"),
        support_plus_raw=arms["support_plus_raw"].pop("selected"),
    )
    return {
        "support_checkpoint": str(support_path),
        "support_checkpoint_sha256": file_sha256(support_path),
        "support_parameters": serializable_envelope(envelope),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "arms": arms,
        "support_search": [
            {
                "quantile": value["quantile"],
                "support_audit": value["support_audit"],
                "metrics": value["metrics"],
            }
            for value in support_candidates
        ],
        "support_plus_raw_search": [
            {
                "quantile": value["quantile"],
                "support_audit": value["support_audit"],
                "metrics": value["metrics"],
            }
            for value in raw_candidates
        ],
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "scand_domain_adaptation_v1":
        raise RuntimeError("training protocol mismatch")
    if config["candidate_pool"]["cem"] or config["reporting"]["cem"]:
        raise RuntimeError("CEM must remain disabled")
    expected = config["data_contract"]
    if file_sha256(args.spent_manifest) != expected["spent_strict100"]["sha256"]:
        raise RuntimeError("spent strict100 manifest hash changed")
    if file_sha256(args.config) != "8917ddecaff8e93a54889eb920c5308b9117e3d931c92f3a162b0444051cfb20":
        raise RuntimeError("frozen protocol hash changed")
    scand = load_features(args.scand_train_features, labels_expected=True)
    cal = load_features(args.scand_cal_features, labels_expected=True)
    recon = load_features(args.recon_train_features, labels_expected=True)
    spent = spent_trajectories(args.spent_manifest)
    for label, data in (("scand_train_fit", scand), ("scand_scorer_cal", cal)):
        overlap = spent.intersection(
            trajectory_id(value) for value in data["record_key"].astype(str)
        )
        if overlap:
            raise RuntimeError(f"{label} overlaps spent strict100: {sorted(overlap)}")
    if set(scand["family"].astype(str)) & set(cal["family"].astype(str)):
        raise RuntimeError("SCAND train-fit/scorer-cal route families overlap")
    if len(scand["family"]) != int(expected["train_fit"]["windows"]):
        raise RuntimeError("SCAND train-fit state count changed")
    if len(cal["family"]) != int(expected["scorer_cal"]["windows"]):
        raise RuntimeError("SCAND scorer-cal state count changed")
    args.output_dir.mkdir(parents=True)
    balanced_index = route_balanced_sample(
        scand["family"],
        len(recon["family"]),
        int(config["training"]["seed"]),
    )
    write_npz(
        args.output_dir / "domain_balanced_scand_index.npz",
        index=balanced_index,
        record_key=scand["record_key"][balanced_index].astype(str),
        family=scand["family"][balanced_index].astype(str),
    )
    scand_data = state_subset(scand, np.arange(len(scand["family"])))
    scand_spacing = np.full(len(scand_data["family"]), 0.38, dtype=np.float32)
    balanced_data, balanced_spacing = concatenate_domain_data(
        recon, scand, balanced_index
    )
    regimes = {
        "scand_only": (scand_data, scand_spacing),
        "domain_balanced": (balanced_data, balanced_spacing),
    }
    ate_scale, rtp_scale, scorer_records = scorer_scales(
        args.scorer_checkpoint_root
    )
    preflight = {
        "schema": "scand_domain_adaptation_preflight_v1",
        "status": "frozen_before_groupfirst_fit",
        "protocol": str(args.config),
        "protocol_sha256": file_sha256(args.config),
        "feature_sha256": {
            "scand_train_fit": file_sha256(args.scand_train_features),
            "scand_scorer_cal": file_sha256(args.scand_cal_features),
            "recon_train": file_sha256(args.recon_train_features),
        },
        "states": {
            "scand_train_fit": int(len(scand["family"])),
            "scand_scorer_cal": int(len(cal["family"])),
            "recon_train": int(len(recon["family"])),
            "domain_balanced": int(len(balanced_data["family"])),
        },
        "domain_balanced_scand_index_sha256": file_sha256(
            args.output_dir / "domain_balanced_scand_index.npz"
        ),
        "spent_strict100_overlap": 0,
        "cem_cal_accessed": False,
        "internal_holdout_accessed": False,
        "scorer_v2_predictions_in_primary_features": False,
        "fixed_pool_only": True,
        "cem": False,
    }
    write_json(args.output_dir / "preflight.json", preflight)
    result: dict[str, Any] = {
        **preflight,
        "schema": "scand_domain_adaptation_calibrated_v1",
        "status": "training",
        "scorer_v2": {
            "ate_scale": ate_scale,
            "rtp_scale": rtp_scale,
            "checkpoints": scorer_records,
        },
        "regimes": {},
    }
    for regime_index, (name, (train, spacing)) in enumerate(regimes.items()):
        train_fingerprint = array_sha256(
            np.concatenate(
                (
                    train["record_key"].astype("U").view(np.uint32).reshape(len(train["record_key"]), -1),
                    np.full((len(train["record_key"]), 1), regime_index, dtype=np.uint32),
                ),
                axis=1,
            )
        )
        _, members, training_audit = train_groupfirst_regime(
            name,
            train,
            config,
            args.output_dir,
            args.device,
            train_fingerprint,
        )
        cal_predictions = ensemble_on_cal(members, cal, args.device)
        calibration_result = calibrate_regime(
            name,
            train,
            spacing,
            cal,
            cal_predictions,
            config,
            ate_scale,
            rtp_scale,
            args.output_dir,
        )
        result["regimes"][name] = {
            "training": training_audit,
            "calibration": calibration_result,
        }
        result["status"] = f"completed_{name}"
        write_json(args.output_dir / "progress.json", jsonable(result))
        del members, cal_predictions
        if args.device == "cuda":
            torch.cuda.empty_cache()
    result["status"] = "calibrated_six_arm_fixed_pool_cem_false"
    result["calibration_labels_opened"] = True
    result["calibration_label_scope"] = "SCAND inner scorer-cal only"
    result["spent_strict100_outcomes_used"] = False
    result["cem_cal_accessed"] = False
    result["internal_holdout_accessed"] = False
    write_json(args.output_dir / "result.json", jsonable(result))
    print(json.dumps(jsonable(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
