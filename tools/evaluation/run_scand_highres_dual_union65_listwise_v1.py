#!/usr/bin/env python3
"""Single strict nested route-OOF listwise test with frozen dual-LTWM features."""

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
from tools.evaluation import run_scand_union65_listwise_selector_v1 as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_inputs(contract: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    table_path = resolve(contract["base_table"]["path"])
    feature_path = resolve(contract["highres_dual_features"]["path"])
    base.require("validation" not in str(table_path).lower(), "validation table rejected")
    base.require(
        sha256_file(table_path) == contract["base_table"]["sha256"],
        "frozen base table hash changed",
    )
    base.require(
        sha256_file(feature_path) == contract["highres_dual_features"]["sha256"],
        "frozen highres feature hash changed",
    )
    data = base.load_npz(table_path)
    highres = base.load_npz(feature_path)
    base.require(
        data["runtime_pair_feature"].shape == (800, 65, 351),
        "base feature shape changed",
    )
    base.require(
        highres["highres_dual_pair_feature"].shape == (800, 65, 80),
        "highres feature shape changed",
    )
    base.require(
        np.array_equal(data["record_key"].astype(str), highres["record_key"].astype(str)),
        "record binding changed",
    )
    base.require(
        np.array_equal(
            data["formal_v70_index"].astype(np.int64),
            highres["formal_v70_index"].astype(np.int64),
        ),
        "formal binding changed",
    )
    metadata = json.loads(str(highres["metadata"].item()))
    base.require(metadata["future_images_read"] is False, "future-image leakage")
    base.require(metadata["candidate_outcomes_read"] is False, "outcome leakage")
    base.require(metadata["sealed_validation_opened"] is False, "sealed validation opened")
    data["runtime_pair_feature"] = np.concatenate(
        (
            data["runtime_pair_feature"].astype(np.float32),
            highres["highres_dual_pair_feature"].astype(np.float32),
        ),
        axis=-1,
    )
    base.require(
        data["runtime_pair_feature"].shape == (800, 65, 431),
        "combined feature shape changed",
    )
    return data, {
        "table": {"path": str(table_path), "sha256": sha256_file(table_path)},
        "highres_dual_features": {
            "path": str(feature_path),
            "sha256": sha256_file(feature_path),
            "metadata": metadata,
        },
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    output = args.output_dir.resolve()
    base.require(not output.exists(), f"refusing to overwrite: {output}")
    contract = base.read_json(contract_path)
    base.require(
        contract.get("status") == "frozen_before_single_strict_nested_route_oof",
        "contract not frozen",
    )
    data, input_artifacts = load_inputs(contract)
    feature = data["runtime_pair_feature"].astype(np.float32)
    routes = data["route_group"].astype(str)
    base.require(len(np.unique(routes)) == 42, "route count changed")
    device = torch.device(args.device)
    base.require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")

    folds = int(contract["evaluation"]["outer_route_folds"])
    fold, fold_audit = base.assign_folds(routes, folds)
    states_all = np.arange(len(routes))
    scores_oof = np.full(feature.shape[:2], np.nan, dtype=np.float32)
    selected_calibrated = data["formal_v70_index"].astype(np.int64).copy()
    selected_curve = {
        str(float(value)): data["formal_v70_index"].astype(np.int64).copy()
        for value in contract["evaluation"]["coverage_curve"]
    }
    outer_records: dict[str, Any] = {}
    parameters: dict[str, np.ndarray] = {}
    seed = int(contract["model"]["seed"])
    started = time.perf_counter()

    for outer in range(folds):
        outer_train = states_all[fold != outer]
        outer_test = states_all[fold == outer]
        base.require(
            not (set(routes[outer_train]) & set(routes[outer_test])),
            "outer route leakage",
        )
        inner_scores = np.full(feature.shape[:2], np.nan, dtype=np.float32)
        inner_fits = []
        for inner in range(folds):
            if inner == outer:
                continue
            inner_train = states_all[(fold != outer) & (fold != inner)]
            inner_test = states_all[fold == inner]
            base.require(
                not (set(routes[inner_train]) & set(routes[inner_test])),
                "inner route leakage",
            )
            model, mean, std, fit = base.fit_model(
                data,
                inner_train,
                routes,
                contract,
                device,
                seed + outer * 100 + inner,
            )
            inner_scores[inner_test] = base.predict(
                model,
                mean,
                std,
                feature[inner_test],
                device,
            )
            inner_fits.append({"inner_fold": inner, **fit})
            del model
        base.require(
            np.isfinite(inner_scores[outer_train]).all(),
            "inner OOF scores incomplete",
        )
        margin, margin_stats, inner_curve = base.inner_curve_and_margin(
            inner_scores[outer_train],
            data,
            outer_train,
            contract,
        )
        model, mean, std, final_fit = base.fit_model(
            data,
            outer_train,
            routes,
            contract,
            device,
            seed + outer * 100 + 99,
        )
        outer_scores = base.predict(
            model,
            mean,
            std,
            feature[outer_test],
            device,
        )
        scores_oof[outer_test] = outer_scores
        selected_calibrated[outer_test] = base.select_with_margin(
            outer_scores,
            data,
            outer_test,
            margin,
        )
        _, inner_advantage = base.top_challenger(
            inner_scores[outer_train],
            data,
            outer_train,
        )
        curve_margins = {}
        for value in contract["evaluation"]["coverage_curve"]:
            coverage = float(value)
            curve_margin = base.margin_for_coverage(inner_advantage, coverage)
            selected_curve[str(coverage)][outer_test] = base.select_with_margin(
                outer_scores,
                data,
                outer_test,
                curve_margin,
            )
            curve_margins[str(coverage)] = float(curve_margin)
        for name, tensor in model.state_dict().items():
            parameters[f"outer_{outer}__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"outer_{outer}__mean"] = mean
        parameters[f"outer_{outer}__std"] = std
        parameters[f"outer_{outer}__margin"] = np.asarray(
            margin,
            dtype=np.float32,
        )
        outer_records[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_route_overlap": 0,
            "inner_fits": inner_fits,
            "inner_selected_margin": margin,
            "inner_selected_margin_stats": {
                name: value
                for name, value in margin_stats.items()
                if name != "route_objective_gain_m"
            },
            "inner_margin_curve": inner_curve,
            "inner_coverage_curve_margins": curve_margins,
            "final_fit": final_fit,
        }
        print(
            canonical_json(
                {
                    "outer_fold": outer,
                    "margin": margin,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del model

    base.require(np.isfinite(scores_oof).all(), "outer OOF scores incomplete")
    pure_top1 = base.select_with_margin(scores_oof, data, states_all, 0.0)
    pure_stats = base.selection_stats(data, states_all, pure_top1)
    calibrated_stats = base.selection_stats(
        data,
        states_all,
        selected_calibrated,
    )
    oracle_index, oracle_stats = base.objective_oracle(data, states_all)
    ranking = base.ranking_stats(
        scores_oof,
        data,
        states_all,
        [int(value) for value in contract["evaluation"]["top_k"]],
    )
    oracle_gain = float(oracle_stats["objective_gain_m"])
    for stats in (pure_stats, calibrated_stats):
        stats["oracle_gain_ratio"] = (
            stats["objective_gain_m"] / oracle_gain
            if oracle_gain > 0.0
            else None
        )
    performance_harm_curve = {}
    for coverage, selected in selected_curve.items():
        stats = base.selection_stats(data, states_all, selected)
        performance_harm_curve[coverage] = {
            name: value
            for name, value in stats.items()
            if name
            in {
                "replacements",
                "coverage",
                "objective_gain_m",
                "ate_gain_m",
                "dual_count",
                "harm_count",
                "dual_rate",
                "harm_rate",
                "positive_route_fraction",
            }
        }
    formal = data["formal_v70_index"].astype(np.int64)
    rows = np.arange(len(formal))
    selected_gain = data["objective_gain_vs_formal_v70_m"][
        rows,
        selected_calibrated,
    ].astype(np.float64)
    selected_gain[selected_calibrated == formal] = 0.0
    bootstrap = base.bootstrap_gain(selected_gain, routes)
    gate = contract["prototype_go_gate"]
    harm_rate = calibrated_stats["harm_rate"]
    checks = {
        "objective_gain_exceeds_existing_4p33mm": calibrated_stats[
            "objective_gain_m"
        ] > float(gate["minimum_all_state_objective_gain_m"]),
        "ate_gain_positive": calibrated_stats["ate_gain_m"]
        > float(gate["minimum_all_state_ate_gain_m"]),
        "harm_not_uncontrolled": harm_rate is not None
        and harm_rate <= float(gate["maximum_empirical_harm_rate"]),
        "oracle_gain_ratio": calibrated_stats["oracle_gain_ratio"] is not None
        and calibrated_stats["oracle_gain_ratio"]
        >= float(gate["minimum_oracle_gain_ratio"]),
        "route_direction": calibrated_stats["positive_route_fraction"]
        >= float(gate["minimum_positive_route_fraction"]),
    }
    go = all(checks.values())

    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "outer_route_oof_predictions.npz"
    parameter_path = output / "frozen_outer_models.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"],
            route_group=routes,
            outer_fold=fold,
            utility_score=scores_oof,
            pure_top1_selected_index=pure_top1,
            calibrated_selected_index=selected_calibrated,
            objective_oracle_index=oracle_index,
            formal_v70_index=formal,
        )
    with parameter_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    result = {
        "schema": "scand_highres_dual_union65_nested_route_oof_result_v1",
        "status": "complete_adaptive_spent_development_performance_prototype",
        "decision": (
            "go_freeze_highres_dual_selector_worth_one_time_validation"
            if go
            else "no_go_highres_dual_features_do_not_close_selector_oracle_gap"
        ),
        "prototype_go": go,
        "prototype_checks": checks,
        "states": 800,
        "route_groups": 42,
        "candidate_rows": 52000,
        "base_feature_dimensions": 351,
        "highres_dual_feature_dimensions": 80,
        "combined_feature_dimensions": 431,
        "model_parameters_per_fold": int(
            sum(
                value.numel()
                for value in base.UtilityNetwork(431).parameters()
            )
        ),
        "pure_top1_including_formal_fallback": pure_stats,
        "inner_calibrated_fallback_margin": calibrated_stats,
        "objective_oracle": oracle_stats,
        "ranking": ranking,
        "performance_harm_curve": performance_harm_curve,
        "route_cluster_bootstrap_objective_gain": bootstrap,
        "fold_assignment": fold_audit,
        "outer_folds": outer_records,
        "comparison_reference": {
            "formal_v70_objective_gain_m": 0.0,
            "preexisting_selector_realized_gain_m": 0.00433,
            "prior_351d_listwise_gain_m": 0.0053311675367876885,
            "prior_351d_listwise_harm_rate": 0.41245791245791247
        },
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            **input_artifacts,
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
            "frozen_outer_models": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "sealed_validation_opened": False,
        "formal_v70_mutated": False,
        "cem_mutated": False,
        "ltwm_trained_during_selector_test": False,
        "outer_labels_selected_model_or_margin": False,
    }
    base.write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "pure_top1_objective_gain_m": pure_stats["objective_gain_m"],
                "calibrated_objective_gain_m": calibrated_stats[
                    "objective_gain_m"
                ],
                "calibrated_ate_gain_m": calibrated_stats["ate_gain_m"],
                "calibrated_harm_rate": calibrated_stats["harm_rate"],
                "calibrated_replacements": calibrated_stats["replacements"],
                "oracle_gain_m": oracle_gain,
                "oracle_gain_ratio": calibrated_stats["oracle_gain_ratio"],
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
