#!/usr/bin/env python3
"""One delta-only LTWM selector optimized for generalized ATE gain.

The frozen rich cache and frozen union65 table are reused without modification.
Every reranker token is a challenger-minus-formal difference from the same
state.  Absolute scene, goal, and formal features are deliberately excluded.
The only learned target is continuous ATE gain.  Model fitting and the single
fallback margin are evaluated with strict nested route out-of-fold splits.
Sealed validation is never read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_scand_top10_latent_reranker_v1 as base  # noqa: E402
from tools.evaluation import run_scand_union65_listwise_selector_v1 as coarse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def resolve_and_verify(entry: dict[str, Any]) -> Path:
    path = (ROOT / entry["path"]).resolve()
    require("sealed_validation" not in str(path).lower(), f"sealed input rejected: {path}")
    require(path.is_file(), f"missing input: {path}")
    require(sha256_file(path) == entry["sha256"], f"input hash changed: {path}")
    return path


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def gather(value: np.ndarray, states: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return value[states[:, None], candidates]


def invariant_token_batch(
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    formal = data["formal_v70_index"][states].astype(np.int64)
    pieces = []
    for name in ("rollout_grid", "latent_scalar", "logvar_grid", "decoded_action"):
        candidate = gather(rich[name], states, candidates).astype(np.float32)
        reference = rich[name][states, formal].astype(np.float32)[:, None]
        pieces.append(candidate - reference)
    candidate_action = gather(actions, states, candidates).astype(np.float32)
    reference_action = actions[states, formal].astype(np.float32)[:, None]
    action_delta = candidate_action - reference_action
    action_delta = np.broadcast_to(
        action_delta[:, :, None],
        (*action_delta.shape[:2], 3, *action_delta.shape[2:]),
    )
    pieces.append(action_delta)
    token = np.concatenate(pieces, axis=-1).astype(np.float32)
    require(token.shape[-1] == 541, f"invariant token dimension changed: {token.shape}")
    require(
        np.allclose(token[:, 0], 0.0, atol=0.0, rtol=0.0),
        "formal token is not identically zero",
    )
    return token


def ate_target(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    require(
        contract["target"]["name"] == "continuous_ate_gain_vs_formal_v70_m",
        "target changed",
    )
    target = gather(data["ate_gain_vs_formal_v70_m"], states, candidates).astype(
        np.float32
    )
    target[:, 0] = 0.0
    return target


def invariant_stream_normalizer(
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    route: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    weight = coarse.state_weights(route, states)
    total = 0.0
    total_sum = None
    total_square = None
    token_dimension = 0
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        token = invariant_token_batch(
            rich,
            actions,
            data,
            states[start:end],
            candidates[start:end],
        ).astype(np.float64)
        token_dimension = token.shape[-1]
        local_weight = weight[start:end, None, None, None, None]
        normalizer = token.shape[1] * token.shape[2] * token.shape[3]
        local_sum = (token * local_weight).sum(axis=(0, 1, 2, 3))
        local_square = ((token * token) * local_weight).sum(axis=(0, 1, 2, 3))
        local_total = float(weight[start:end].sum() * normalizer)
        total_sum = local_sum if total_sum is None else total_sum + local_sum
        total_square = (
            local_square if total_square is None else total_square + local_square
        )
        total += local_total
    require(total_sum is not None and total_square is not None, "empty normalizer")
    mean = total_sum / total
    variance = np.maximum(total_square / total - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32), token_dimension


def add_ate_route_stats(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    selected: np.ndarray,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = coarse.selection_stats(data, states, selected) if stats is None else stats
    formal = data["formal_v70_index"][states].astype(np.int64)
    gain = data["ate_gain_vs_formal_v70_m"][states, selected].astype(np.float64)
    gain[selected == formal] = 0.0
    route_values, positive = coarse.route_direction(
        gain,
        data["route_group"][states].astype(str),
    )
    result["positive_ate_route_fraction"] = positive
    result["route_ate_gain_m"] = route_values
    return result


def inner_ate_curve_and_margin(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    contract: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[str, dict[str, Any]]]:
    _, advantage = coarse.top_challenger(scores, data, states)
    margins = {0.0}
    for value in contract["evaluation"]["coverage_curve"]:
        margins.add(coarse.margin_for_coverage(advantage, float(value)))
    rules = contract["inner_margin_selection"]
    records: dict[str, dict[str, Any]] = {}
    feasible = []
    for margin in sorted(margins):
        selected = coarse.select_with_margin(scores, data, states, margin)
        stats = add_ate_route_stats(data, states, selected)
        harm = stats["harm_rate"]
        accepted = bool(
            stats["replacements"] >= int(rules["minimum_replacements"])
            and stats["objective_gain_m"]
            > float(rules["minimum_all_state_objective_gain_m"])
            and stats["ate_gain_m"] > float(rules["minimum_all_state_ate_gain_m"])
            and harm is not None
            and harm <= float(rules["maximum_empirical_harm_rate"])
            and stats["positive_ate_route_fraction"]
            >= float(rules["minimum_positive_ate_route_fraction"])
        )
        records[f"{margin:.9g}"] = {
            "margin": float(margin),
            "feasible": accepted,
            **{
                name: value
                for name, value in stats.items()
                if name not in ("route_objective_gain_m", "route_ate_gain_m")
            },
        }
        if accepted:
            feasible.append((margin, stats))
    if feasible:
        chosen_margin, chosen = sorted(
            feasible,
            key=lambda item: (
                -item[1]["ate_gain_m"],
                -item[1]["objective_gain_m"],
                item[1]["harm_rate"],
                -item[1]["replacements"],
                item[0],
            ),
        )[0]
        reason = "maximum_inner_ate_gain_among_empirically_feasible_margins"
    else:
        chosen_margin = 0.0
        chosen = add_ate_route_stats(
            data,
            states,
            coarse.select_with_margin(scores, data, states, 0.0),
        )
        reason = "no_inner_candidate_met_constraints_zero_margin_used"
    return float(chosen_margin), {"reason": reason, **chosen}, records


def ate_oracle(
    data: dict[str, np.ndarray],
    states: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = coarse.choice_mask(data, states)
    gain = data["ate_gain_vs_formal_v70_m"][states].astype(np.float32)
    selected = np.argmax(np.where(mask, gain, -np.inf), axis=1).astype(np.int64)
    return selected, add_ate_route_stats(data, states, selected)


def ate_ranking_stats(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    top_k: list[int],
) -> dict[str, Any]:
    mask = coarse.choice_mask(data, states)
    gain = data["ate_gain_vs_formal_v70_m"][states].astype(np.float32)
    masked_gain = np.where(mask, gain, -np.inf)
    oracle_index = np.argmax(masked_gain, axis=1)
    oracle_gain = masked_gain[np.arange(len(states)), oracle_index]
    opportunity = oracle_gain > 0.0
    ordered = np.argsort(np.where(mask, scores, -np.inf), axis=1)[:, ::-1]
    result = {}
    for k in top_k:
        local = ordered[:, :k]
        exact = (local == oracle_index[:, None]).any(axis=1)
        top_gain = np.max(np.take_along_axis(masked_gain, local, axis=1), axis=1)
        ratio = np.divide(
            np.maximum(top_gain, 0.0),
            oracle_gain,
            out=np.ones_like(oracle_gain),
            where=oracle_gain > 0.0,
        )
        result[str(k)] = {
            "exact_ate_oracle_candidate_recall": (
                float(exact[opportunity].mean()) if opportunity.any() else None
            ),
            "mean_best_ate_gain_fraction": (
                float(ratio[opportunity].mean()) if opportunity.any() else None
            ),
            "opportunity_states": int(opportunity.sum()),
        }
    return result


def cohort_stats(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, Any]:
    result = {}
    cohorts = data["cohort"].astype(str)
    for name in np.unique(cohorts):
        states = np.flatnonzero(cohorts == name)
        stats = add_ate_route_stats(data, states, selected[states])
        _, oracle = ate_oracle(data, states)
        stats["ate_oracle_gain_m"] = oracle["ate_gain_m"]
        stats["ate_oracle_gain_ratio"] = (
            stats["ate_gain_m"] / oracle["ate_gain_m"]
            if oracle["ate_gain_m"] > 0.0
            else None
        )
        result[name] = stats
    return result


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    contract = read_json(contract_path)
    require(
        contract.get("status") == "frozen_before_single_strict_nested_route_oof",
        "contract not frozen",
    )
    input_paths = {
        name: resolve_and_verify(entry) for name, entry in contract["inputs"].items()
    }
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    data = coarse.load_npz(input_paths["runtime_table"])
    require(
        data["runtime_pair_feature"].shape == (800, 65, 351),
        "frozen runtime table changed",
    )
    routes = data["route_group"].astype(str)
    require(len(np.unique(routes)) == 42, "route count changed")
    coarse_contract = read_json(input_paths["coarse_contract"])
    rich = base.load_rich(input_paths["rich_ltwm_cache"], data["record_key"])
    actions = base.actions_from_data(data)
    base.token_batch = invariant_token_batch
    base.stream_normalizer = invariant_stream_normalizer
    base.selected_target = ate_target
    folds = int(contract["evaluation"]["outer_route_folds"])
    fold, fold_audit = coarse.assign_folds(routes, folds)
    states_all = np.arange(len(routes))
    rerank_score_oof = np.full((len(routes), 65), -np.inf, dtype=np.float32)
    coarse_score_oof = np.full((len(routes), 65), np.nan, dtype=np.float32)
    formal = data["formal_v70_index"].astype(np.int64)
    selected_calibrated = formal.copy()
    curve_selected = {
        str(float(value)): formal.copy()
        for value in contract["evaluation"]["coverage_curve"]
    }
    parameters: dict[str, np.ndarray] = {}
    outer_records = {}
    seed = int(contract["model"]["seed"])
    started = time.perf_counter()
    for outer in range(folds):
        outer_train = states_all[fold != outer]
        outer_test = states_all[fold == outer]
        require(
            not (set(routes[outer_train]) & set(routes[outer_test])),
            "outer route leakage",
        )
        inner_scores = np.full((len(routes), 65), -np.inf, dtype=np.float32)
        inner_fits = []
        for inner in range(folds):
            if inner == outer:
                continue
            inner_train = states_all[(fold != outer) & (fold != inner)]
            inner_test = states_all[fold == inner]
            require(
                not (set(routes[inner_train]) & set(routes[inner_test])),
                "inner route leakage",
            )
            coarse_model, coarse_mean, coarse_std, coarse_fit = coarse.fit_model(
                data,
                inner_train,
                routes,
                coarse_contract,
                device,
                seed + outer * 1000 + inner * 10,
            )
            coarse_train_score = coarse.predict(
                coarse_model,
                coarse_mean,
                coarse_std,
                data["runtime_pair_feature"][inner_train],
                device,
            )
            coarse_test_score = coarse.predict(
                coarse_model,
                coarse_mean,
                coarse_std,
                data["runtime_pair_feature"][inner_test],
                device,
            )
            train_short = base.shortlist(coarse_train_score, data, inner_train)
            test_short = base.shortlist(coarse_test_score, data, inner_test)
            reranker, mean, std, rerank_fit = base.fit_reranker(
                rich,
                actions,
                data,
                inner_train,
                train_short,
                routes,
                contract,
                device,
                seed + outer * 1000 + inner * 10 + 1,
            )
            local_score = base.predict_reranker(
                reranker,
                mean,
                std,
                rich,
                actions,
                data,
                inner_test,
                test_short,
                device,
                int(contract["model"]["state_batch_size"]),
            )
            inner_scores[inner_test[:, None], test_short] = local_score
            inner_fits.append(
                {"inner_fold": inner, "coarse": coarse_fit, "reranker": rerank_fit}
            )
            del coarse_model, reranker
            if device.type == "cuda":
                torch.cuda.empty_cache()
        require(
            np.isfinite(inner_scores[outer_train]).any(axis=1).all(),
            "inner reranker predictions incomplete",
        )
        margin, margin_stats, inner_curve = inner_ate_curve_and_margin(
            inner_scores[outer_train],
            data,
            outer_train,
            contract,
        )
        coarse_model, coarse_mean, coarse_std, coarse_fit = coarse.fit_model(
            data,
            outer_train,
            routes,
            coarse_contract,
            device,
            seed + outer * 1000 + 990,
        )
        coarse_train_score = coarse.predict(
            coarse_model,
            coarse_mean,
            coarse_std,
            data["runtime_pair_feature"][outer_train],
            device,
        )
        coarse_test_score = coarse.predict(
            coarse_model,
            coarse_mean,
            coarse_std,
            data["runtime_pair_feature"][outer_test],
            device,
        )
        train_short = base.shortlist(coarse_train_score, data, outer_train)
        test_short = base.shortlist(coarse_test_score, data, outer_test)
        reranker, mean, std, rerank_fit = base.fit_reranker(
            rich,
            actions,
            data,
            outer_train,
            train_short,
            routes,
            contract,
            device,
            seed + outer * 1000 + 991,
        )
        outer_score = base.predict_reranker(
            reranker,
            mean,
            std,
            rich,
            actions,
            data,
            outer_test,
            test_short,
            device,
            int(contract["model"]["state_batch_size"]),
        )
        rerank_score_oof[outer_test[:, None], test_short] = outer_score
        coarse_score_oof[outer_test] = coarse_test_score
        selected_calibrated[outer_test] = coarse.select_with_margin(
            rerank_score_oof[outer_test],
            data,
            outer_test,
            margin,
        )
        _, inner_advantage = coarse.top_challenger(
            inner_scores[outer_train],
            data,
            outer_train,
        )
        curve_margins = {}
        for value in contract["evaluation"]["coverage_curve"]:
            coverage = float(value)
            local_margin = coarse.margin_for_coverage(inner_advantage, coverage)
            curve_selected[str(coverage)][outer_test] = coarse.select_with_margin(
                rerank_score_oof[outer_test],
                data,
                outer_test,
                local_margin,
            )
            curve_margins[str(coverage)] = float(local_margin)
        for name, tensor in coarse_model.state_dict().items():
            parameters[f"outer_{outer}__coarse__{name}"] = (
                tensor.detach().cpu().numpy()
            )
        parameters[f"outer_{outer}__coarse_mean"] = coarse_mean
        parameters[f"outer_{outer}__coarse_std"] = coarse_std
        for name, tensor in reranker.state_dict().items():
            parameters[f"outer_{outer}__reranker__{name}"] = (
                tensor.detach().cpu().numpy()
            )
        parameters[f"outer_{outer}__reranker_mean"] = mean
        parameters[f"outer_{outer}__reranker_std"] = std
        parameters[f"outer_{outer}__margin"] = np.asarray(margin, dtype=np.float32)
        outer_records[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "route_overlap": 0,
            "inner_selected_margin": margin,
            "inner_selected_margin_stats": {
                name: value
                for name, value in margin_stats.items()
                if name not in ("route_objective_gain_m", "route_ate_gain_m")
            },
            "inner_margin_curve": inner_curve,
            "inner_coverage_curve_margins": curve_margins,
            "inner_fits": inner_fits,
            "final_coarse_fit": coarse_fit,
            "final_reranker_fit": rerank_fit,
        }
        print(
            canonical_json(
                {
                    "stage": "nested_invariant_ate_reranker",
                    "outer_fold": outer,
                    "margin": margin,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del coarse_model, reranker
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(np.isfinite(coarse_score_oof).all(), "coarse OOF scores incomplete")
    require(
        np.isfinite(rerank_score_oof).any(axis=1).all(),
        "reranker OOF scores incomplete",
    )
    pure_selected = coarse.select_with_margin(
        rerank_score_oof,
        data,
        states_all,
        0.0,
    )
    pure_stats = add_ate_route_stats(data, states_all, pure_selected)
    calibrated_stats = add_ate_route_stats(data, states_all, selected_calibrated)
    _, oracle_stats = ate_oracle(data, states_all)
    pure_stats["ate_oracle_gain_ratio"] = (
        pure_stats["ate_gain_m"] / oracle_stats["ate_gain_m"]
    )
    calibrated_stats["ate_oracle_gain_ratio"] = (
        calibrated_stats["ate_gain_m"] / oracle_stats["ate_gain_m"]
    )
    top_k = [int(value) for value in contract["evaluation"]["top_k"]]
    rerank_ranking = ate_ranking_stats(
        rerank_score_oof,
        data,
        states_all,
        top_k,
    )
    coarse_ranking = ate_ranking_stats(
        coarse_score_oof,
        data,
        states_all,
        top_k,
    )
    curve = {}
    for coverage, selected in curve_selected.items():
        stats = add_ate_route_stats(data, states_all, selected)
        curve[coverage] = {
            name: stats[name]
            for name in (
                "replacements",
                "coverage",
                "objective_gain_m",
                "ate_gain_m",
                "dual_rate",
                "harm_rate",
                "positive_ate_route_fraction",
            )
        }
    cohorts = cohort_stats(data, selected_calibrated)
    ate_gain = data["ate_gain_vs_formal_v70_m"][
        states_all,
        selected_calibrated,
    ].astype(np.float64)
    ate_gain[selected_calibrated == formal] = 0.0
    bootstrap = coarse.bootstrap_gain(ate_gain, routes)
    mainrgb = cohorts["spent_new_mainrgb_development"]
    gate = contract["prototype_go_gate"]
    checks = {
        "overall_ate_gain": calibrated_stats["ate_gain_m"]
        >= float(gate["minimum_overall_ate_gain_m"]),
        "new_mainrgb_ate_gain": mainrgb["ate_gain_m"]
        >= float(gate["minimum_new_mainrgb_ate_gain_m"]),
        "overall_objective_gain": calibrated_stats["objective_gain_m"]
        >= float(gate["minimum_overall_objective_gain_m"]),
        "harm_rate": calibrated_stats["harm_rate"] is not None
        and calibrated_stats["harm_rate"]
        <= float(gate["maximum_empirical_harm_rate"]),
        "new_mainrgb_ate_oracle_gain_ratio": mainrgb["ate_oracle_gain_ratio"]
        >= float(gate["minimum_new_mainrgb_ate_oracle_gain_ratio"]),
        "route_bootstrap_ate_lcb": bootstrap["lcb95_m"]
        > float(gate["route_bootstrap_ate_lcb_min_m"]),
    }
    go = all(checks.values())
    prediction_path = output / "outer_route_oof_predictions.npz"
    parameter_path = output / "frozen_outer_pipeline.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"],
            route_group=routes,
            outer_fold=fold,
            coarse_score=coarse_score_oof,
            reranker_score=rerank_score_oof,
            pure_selected_index=pure_selected,
            calibrated_selected_index=selected_calibrated,
            formal_v70_index=formal,
        )
    with parameter_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    result = {
        "schema": "scand_top10_invariant_ate_reranker_nested_route_oof_result_v1",
        "status": "complete_adaptive_spent_development_performance_prototype",
        "decision": (
            "go_freeze_pipeline_worth_one_time_sealed_validation"
            if go
            else "no_go_delta_only_ltwm_signal_did_not_generalize_for_ate"
        ),
        "prototype_go": go,
        "prototype_checks": checks,
        "states": int(len(routes)),
        "routes": int(len(np.unique(routes))),
        "pure_top1_including_formal": pure_stats,
        "inner_calibrated_fallback": calibrated_stats,
        "ate_oracle": oracle_stats,
        "cohort_results": cohorts,
        "coarse_ate_ranking": coarse_ranking,
        "reranker_ate_ranking": rerank_ranking,
        "performance_harm_curve": curve,
        "route_cluster_bootstrap_ate_gain": bootstrap,
        "outer_folds": outer_records,
        "fold_audit": fold_audit,
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in input_paths.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "frozen_outer_pipeline": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "sealed_validation_opened": False,
        "formal_v70_mutated": False,
        "cem_mutated": False,
        "ltwm_weights_mutated": False,
    }
    write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "overall_ate_gain_m": calibrated_stats["ate_gain_m"],
                "overall_ate_oracle_gain_ratio": calibrated_stats[
                    "ate_oracle_gain_ratio"
                ],
                "overall_objective_gain_m": calibrated_stats["objective_gain_m"],
                "harm_rate": calibrated_stats["harm_rate"],
                "new_mainrgb_ate_gain_m": mainrgb["ate_gain_m"],
                "new_mainrgb_ate_oracle_gain_ratio": mainrgb[
                    "ate_oracle_gain_ratio"
                ],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            }
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
