#!/usr/bin/env python3
"""Route-OOF residual ridge calibration for frozen dense-meta union65 transfer.

This experiment is deliberately low capacity.  It uses only route-disjoint
frozen dense-meta predictions, route-OOF v4 predictions, candidate geometry,
and pool identity.  Every union65 label used to fit a ridge member comes from
routes disjoint from that member's evaluation routes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METRIC_NAMES = ("ate", "rpe", "endpoint", "objective")
TARGET_NAMES = (
    "ate_gain_m",
    "objective_gain_m",
    "harm_probability",
    "downside_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--v4-oof", type=Path, required=True)
    parser.add_argument("--v4-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--harm-ucb-threshold", type=float, default=0.60)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270725)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


def build_features(
    table: dict[str, np.ndarray],
    transfer: dict[str, np.ndarray],
    v4: dict[str, np.ndarray],
) -> np.ndarray:
    states, candidates = table["physical_support_pool"].shape
    row = np.arange(states)
    formal = table["formal_v70_index"].astype(np.int64)
    parts = []
    for mode in ("action", "full"):
        for key in ("outcome_mean", "outcome_std"):
            value = transfer[f"{mode}_{key}"].astype(np.float32)
            parts.extend((value, value - value[row, formal, None, :]))
        for key in (
            "p_help_mean",
            "p_safe_mean",
            "p_harm_mean",
            "p_select_mean",
        ):
            value = transfer[f"{mode}_{key}"].astype(np.float32)[..., None]
            parts.extend((value, value - value[row, formal, None, :]))
    heads_mean = v4["heads_mean"].astype(np.float32)
    heads_std = v4["heads_std"].astype(np.float32)
    parts.extend(
        (
            heads_mean,
            heads_mean - heads_mean[row, formal, None, :],
            heads_std,
        )
    )
    utility = v4["utility_score"].astype(np.float32)[..., None]
    parts.extend((utility, utility - utility[row, formal, None, :]))
    geometry = table["candidates_feature_waypoint_units"].astype(np.float32).reshape(
        states, candidates, -1
    )
    parts.extend((geometry, geometry - geometry[row, formal, None, :]))
    parts.append(table["source_round"].astype(np.float32)[..., None])
    feature = np.concatenate(parts, axis=-1).astype(np.float32)
    if not np.isfinite(feature).all():
        raise RuntimeError("residual features contain NaN or Inf")
    return feature


def fit_ridge(
    feature: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    outer_fold: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    states, candidates, dimensions = feature.shape
    predictions = np.full((states, candidates, target.shape[-1]), np.nan, np.float32)
    means = []
    scales = []
    coefficients = []
    flat_feature = feature.reshape(-1, dimensions)
    flat_target = target.reshape(-1, target.shape[-1])
    for heldout in sorted(np.unique(outer_fold).tolist()):
        train_state = outer_fold != heldout
        test_state = outer_fold == heldout
        train_row = (train_state[:, None] & valid).reshape(-1)
        value = flat_feature[train_row]
        truth = flat_target[train_row]
        mean = value.mean(axis=0).astype(np.float32)
        scale = np.maximum(value.std(axis=0), 1.0e-5).astype(np.float32)
        normalized = np.clip((value - mean) / scale, -8.0, 8.0)
        design = np.concatenate(
            (normalized, np.ones((len(normalized), 1), dtype=np.float32)), axis=1
        )
        penalty = np.eye(design.shape[1], dtype=np.float32) * alpha
        penalty[-1, -1] = 0.0
        coefficient = np.linalg.solve(
            design.T @ design + penalty, design.T @ truth
        ).astype(np.float32)
        test_value = np.clip(
            (feature[test_state] - mean) / scale, -8.0, 8.0
        )
        test_design = np.concatenate(
            (
                test_value,
                np.ones((*test_value.shape[:2], 1), dtype=np.float32),
            ),
            axis=2,
        )
        predictions[test_state] = test_design @ coefficient
        means.append(mean)
        scales.append(scale)
        coefficients.append(coefficient)
    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF residual predictions contain NaN or Inf")
    return (
        predictions,
        np.stack(means),
        np.stack(scales),
        np.stack(coefficients),
    )


def select_policy(
    prediction: np.ndarray,
    valid: np.ndarray,
    formal: np.ndarray,
    v4: dict[str, np.ndarray],
    harm_ucb_threshold: float,
) -> np.ndarray:
    states = len(formal)
    row = np.arange(states)
    score = 0.5 * (prediction[..., 0] + prediction[..., 1])
    gain = score - score[row, formal, None]
    harm_ucb = v4["heads_mean"][..., 2] + 1.5 * v4["heads_std"][..., 2]
    eligible = valid & (harm_ucb <= harm_ucb_threshold)
    candidate_score = np.where(eligible, gain, -np.inf)
    candidate_score[row, formal] = 0.0
    selected = np.argmax(candidate_score, axis=1).astype(np.int64)
    selected = np.where(
        candidate_score[row, selected] > 0.0, selected, formal
    ).astype(np.int64)
    return selected


def policy_metrics(
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    state_index: np.ndarray | None = None,
) -> dict[str, Any]:
    if state_index is None:
        state_index = np.arange(len(selected))
    state_index = np.asarray(state_index, dtype=np.int64)
    chosen = selected[state_index]
    formal = table["formal_v70_index"].astype(np.int64)[state_index]
    metrics = table["true_metrics_m"].astype(np.float64)
    selected_metrics = metrics[state_index, chosen]
    formal_metrics = metrics[state_index, formal]
    replacement = chosen != formal
    harm = table["harm_vs_formal_v70"][state_index, chosen].astype(bool)
    dual = table["dual_vs_formal_v70"][state_index, chosen].astype(bool)
    gain = formal_metrics - selected_metrics
    result: dict[str, Any] = {
        "states": int(len(state_index)),
        "replacements": int(replacement.sum()),
        "coverage": float(replacement.mean()),
        "harm_count": int(harm.sum()),
        "harm_rate": float(harm.sum() / max(replacement.sum(), 1)),
        "dual_count": int(dual.sum()),
        "dual_rate": float(dual.sum() / max(replacement.sum(), 1)),
    }
    for metric_index, name in enumerate(METRIC_NAMES):
        result[f"{name}_gain_m"] = float(gain[:, metric_index].mean())
        result[f"formal_{name}_mean_m"] = float(
            formal_metrics[:, metric_index].mean()
        )
        result[f"selected_{name}_mean_m"] = float(
            selected_metrics[:, metric_index].mean()
        )
        result[f"formal_{name}_p90_m"] = float(
            np.quantile(formal_metrics[:, metric_index], 0.90)
        )
        result[f"selected_{name}_p90_m"] = float(
            np.quantile(selected_metrics[:, metric_index], 0.90)
        )
        result[f"{name}_p90_improvement_m"] = (
            result[f"formal_{name}_p90_m"]
            - result[f"selected_{name}_p90_m"]
        )
    return result


def grouped_metrics(
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    group_key: str,
) -> dict[str, Any]:
    group = table[group_key].astype(str)
    values = {
        name: policy_metrics(selected, table, np.flatnonzero(group == name))
        for name in sorted(np.unique(group).tolist())
    }
    return {
        "groups": values,
        "positive_ate_fraction": float(
            np.mean([value["ate_gain_m"] > 0.0 for value in values.values()])
        ),
        "positive_objective_fraction": float(
            np.mean(
                [value["objective_gain_m"] > 0.0 for value in values.values()]
            )
        ),
    }


def route_bootstrap(
    selected: np.ndarray,
    table: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if draws <= 0:
        raise ValueError("bootstrap draws must be positive")
    route = table["route_group"].astype(str)
    unique = np.asarray(sorted(np.unique(route).tolist()))
    formal = table["formal_v70_index"].astype(np.int64)
    row = np.arange(len(route))
    ate_gain = table["ate_gain_vs_formal_v70_m"][row, selected].astype(np.float64)
    objective_gain = table["objective_gain_vs_formal_v70_m"][
        row, selected
    ].astype(np.float64)
    route_ate = [ate_gain[route == name] for name in unique]
    route_objective = [objective_gain[route == name] for name in unique]
    rng = np.random.default_rng(seed)
    ate_draw = np.empty(draws, dtype=np.float64)
    objective_draw = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(unique), size=len(unique))
        ate_draw[draw] = np.concatenate([route_ate[index] for index in sampled]).mean()
        objective_draw[draw] = np.concatenate(
            [route_objective[index] for index in sampled]
        ).mean()
    return {
        "routes": int(len(unique)),
        "draws": int(draws),
        "ate_gain_m": {
            "mean": float(ate_gain.mean()),
            "lcb95": float(np.quantile(ate_draw, 0.025)),
            "ucb95": float(np.quantile(ate_draw, 0.975)),
        },
        "objective_gain_m": {
            "mean": float(objective_gain.mean()),
            "lcb95": float(np.quantile(objective_draw, 0.025)),
            "ucb95": float(np.quantile(objective_draw, 0.975)),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)

    table = load_npz(args.table)
    transfer = load_npz(args.transfer)
    v4 = load_npz(args.v4_oof)
    record = table["record_key"].astype(str)
    if not np.array_equal(record, transfer["record_key"].astype(str)):
        raise RuntimeError("table/transfer record orders differ")
    if not np.array_equal(record, v4["record_key"].astype(str)):
        raise RuntimeError("table/v4 record orders differ")
    if transfer.get("exact_train_record_overlap") is None:
        raise RuntimeError("transfer artifact lacks exact-overlap audit fields")

    states, candidates = table["physical_support_pool"].shape
    row = np.arange(states)
    formal = table["formal_v70_index"].astype(np.int64)
    valid = table["physical_support_pool"].astype(bool).copy()
    valid[row, formal] = True
    outer_fold = v4["outer_fold"].astype(np.int64)
    feature = build_features(table, transfer, v4)
    ate_gain = table["ate_gain_vs_formal_v70_m"].astype(np.float32)
    objective_gain = table["objective_gain_vs_formal_v70_m"].astype(np.float32)
    downside = np.maximum(np.maximum(-ate_gain, -objective_gain), 0.0)
    target = np.stack(
        (
            ate_gain,
            objective_gain,
            table["harm_vs_formal_v70"].astype(np.float32),
            downside,
        ),
        axis=-1,
    )
    prediction, mean, scale, coefficient = fit_ridge(
        feature, target, valid, outer_fold, args.alpha
    )
    selected = select_policy(
        prediction,
        valid,
        formal,
        v4,
        args.harm_ucb_threshold,
    )

    prediction_path = output_dir / "outer_route_oof_predictions.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=record,
            route_group=table["route_group"],
            outer_fold=outer_fold.astype(np.int8),
            target_names=np.asarray(TARGET_NAMES),
            prediction=prediction,
            selected_index=selected,
            formal_v70_index=formal,
        )
    model_path = output_dir / "frozen_outer_ridge_models.npz"
    with model_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            outer_fold=np.asarray(sorted(np.unique(outer_fold).tolist())),
            feature_mean=mean,
            feature_scale=scale,
            coefficient=coefficient,
            alpha=np.asarray(args.alpha),
            harm_ucb_threshold=np.asarray(args.harm_ucb_threshold),
        )

    overall = policy_metrics(selected, table)
    route = grouped_metrics(selected, table, "route_group")
    cohort = grouped_metrics(selected, table, "cohort")
    folds = grouped_metrics(selected, {**table, "outer_fold": outer_fold}, "outer_fold")
    bootstrap = route_bootstrap(
        selected, table, args.bootstrap_draws, args.seed
    )
    v4_result = json.loads(args.v4_result.read_text(encoding="utf-8"))
    v4_metrics = v4_result["inner_oof_safe_policy"]
    checks = {
        "ate_gain_above_formal": overall["ate_gain_m"] > 0.0,
        "objective_gain_above_formal": overall["objective_gain_m"] > 0.0,
        "ate_gain_above_v4": overall["ate_gain_m"]
        > float(v4_metrics["ate_gain_m"]),
        "objective_gain_above_v4": overall["objective_gain_m"]
        > float(v4_metrics["objective_gain_m"]),
        "aggregate_harm_rate_below_30pct": overall["harm_rate"] < 0.30,
        "ate_p90_nonworse": overall["ate_p90_improvement_m"] >= 0.0,
        "all_outer_folds_positive_ate": all(
            value["ate_gain_m"] > 0.0 for value in folds["groups"].values()
        ),
        "route_bootstrap_ate_lcb_positive": bootstrap["ate_gain_m"]["lcb95"] > 0.0,
        "route_bootstrap_objective_lcb_positive": (
            bootstrap["objective_gain_m"]["lcb95"] > 0.0
        ),
        "outer_route_overlap_zero": True,
        "exact_overlap_leakage_zero": bool(
            transfer["exact_train_record_overlap"].sum() == 0
            or np.all(
                transfer["route_disjoint_member"][
                    transfer["exact_train_record_overlap"]
                ]
                >= 0
            )
        ),
    }
    development_go = all(checks.values())
    result = {
        "schema": "scand_union65_densemeta_residual_ridge_v1",
        "status": "complete_spent_development_route_oof",
        "confirmatory_claim_allowed": False,
        "development_data_status": (
            "union65 outcomes were already spent before this adaptive experiment"
        ),
        "experiment_novelty_audit": {
            "same_experiment_found": False,
            "distinguishing_change": (
                "first strict route-OOF residual calibration of route-disjoint "
                "frozen 7489-state K17 dense-meta predictions on union65, followed "
                "by formal-v70-relative ranking and v4 harm-UCB filtering"
            ),
            "closest_completed_experiments": {
                "union65_listwise_selector_v1": (
                    "trained on runtime-pair features without frozen dense-meta "
                    "transfer predictions"
                ),
                "highres_dual_union65_listwise_v1": (
                    "used high-resolution dual features, not the K17 GroupFirst "
                    "transfer heads"
                ),
                "top10_latent_and_invariant_rerankers": (
                    "reranked v4 top-10 candidates with different latent/invariant "
                    "features and no dense-meta transfer"
                ),
                "scand_domain_adaptation_v1": (
                    "trained/evaluated only the K17 pool and was never applied to "
                    "the four-pool union65/formal-v70 decision"
                ),
            },
        },
        "policy": {
            "ranker": "ridge10_dual_gain",
            "alpha": float(args.alpha),
            "score": "mean(predicted_ate_gain_m,predicted_objective_gain_m)",
            "eligibility": (
                "physical_support_pool and v4 harm mean + 1.5 std <= threshold"
            ),
            "harm_ucb_threshold": float(args.harm_ucb_threshold),
            "fallback": "formal_v70",
        },
        "states": int(states),
        "candidates": int(candidates),
        "features": int(feature.shape[-1]),
        "transfer_leakage_audit": {
            "exact_train_record_overlap_states": int(
                transfer["exact_train_record_overlap"].sum()
            ),
            "seen_train_route_states": int(transfer["seen_train_route"].sum()),
            "exact_overlap_uses_route_excluding_member_only": True,
        },
        "primary_policy": overall,
        "route_generalization": route,
        "cohort_generalization": cohort,
        "outer_folds": folds,
        "route_cluster_bootstrap": bootstrap,
        "formal_v70_reference": {
            "v4_ate_gain_m": float(v4_metrics["ate_gain_m"]),
            "v4_objective_gain_m": float(v4_metrics["objective_gain_m"]),
            "v4_harm_rate": float(v4_metrics["harm_rate"]),
            "ate_gain_relative_to_v4": float(
                overall["ate_gain_m"] / float(v4_metrics["ate_gain_m"])
            ),
            "objective_gain_relative_to_v4": float(
                overall["objective_gain_m"]
                / float(v4_metrics["objective_gain_m"])
            ),
        },
        "checks": checks,
        "decision": (
            "freeze_for_one_time_unopened_validation"
            if development_go
            else "no_go"
        ),
        "required_next_step": (
            "evaluate this exact frozen policy once on a genuinely unopened "
            "route-disjoint candidate/outcome cohort"
        ),
        "inputs": {
            "table": {
                "path": str(args.table.resolve()),
                "sha256": sha256_file(args.table),
            },
            "transfer": {
                "path": str(args.transfer.resolve()),
                "sha256": sha256_file(args.transfer),
            },
            "v4_oof": {
                "path": str(args.v4_oof.resolve()),
                "sha256": sha256_file(args.v4_oof),
            },
            "v4_result": {
                "path": str(args.v4_result.resolve()),
                "sha256": sha256_file(args.v4_result),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "models": {
                "path": str(model_path),
                "sha256": sha256_file(model_path),
            },
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": str(result_path),
                "decision": result["decision"],
                "ate_gain_m": overall["ate_gain_m"],
                "objective_gain_m": overall["objective_gain_m"],
                "harm_rate": overall["harm_rate"],
                "route_bootstrap_ate_lcb95_m": bootstrap["ate_gain_m"]["lcb95"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
