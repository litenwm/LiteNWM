#!/usr/bin/env python3
"""Open v60 G/H outcomes once after the complete CEM selection freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import local_ground_truth  # noqa: E402
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import write_json, write_npz  # noqa: E402


METRICS = ("ate", "rpe", "endpoint", "objective")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chosen(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    return values[np.arange(len(index)), index]


def route_bootstrap_comparison(
    fixed_index: np.ndarray,
    combined_index: np.ndarray,
    values: np.ndarray,
    routes: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    fixed_value = chosen(values, fixed_index)
    combined_value = chosen(values, combined_index)
    delta = combined_value - fixed_value
    coverage_delta = (combined_index != 0).astype(np.float64) - (
        fixed_index != 0
    ).astype(np.float64)
    unique = sorted(np.unique(routes.astype(str)).tolist())
    metric_by_route = np.stack([delta[routes == route].mean(axis=0) for route in unique])
    coverage_by_route = np.asarray(
        [coverage_delta[routes == route].mean() for route in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(replicates, len(unique)))
    metric_draw = metric_by_route[draw].mean(axis=1)
    coverage_draw = coverage_by_route[draw].mean(axis=1)
    return {
        "coverage_gain": float(coverage_delta.mean()),
        "coverage_gain_ci95": [
            float(np.quantile(coverage_draw, 0.025)),
            float(np.quantile(coverage_draw, 0.975)),
        ],
        "quality_delta_combined_minus_fixed_m": {
            name: {
                "mean": float(delta[:, metric].mean()),
                "ci95_low": float(np.quantile(metric_draw[:, metric], 0.025)),
                "ci95_high": float(np.quantile(metric_draw[:, metric], 0.975)),
            }
            for metric, name in enumerate(METRICS)
        },
    }


def route_method_ci(
    index: np.ndarray,
    values: np.ndarray,
    routes: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float] | None]:
    delta = chosen(values, index) - values[:, 0]
    promote = index != 0
    dual = promote & (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = promote & ((delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7))
    unique = sorted(np.unique(routes.astype(str)).tolist())
    counts = np.asarray(
        [
            [
                int((routes == route).sum()),
                int((promote & (routes == route)).sum()),
                int((dual & (routes == route)).sum()),
                int((harm & (routes == route)).sum()),
            ]
            for route in unique
        ],
        dtype=np.int64,
    )
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(unique), size=(replicates, len(unique)))
    totals = counts[draw].sum(axis=1)
    coverage = totals[:, 1] / totals[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        dual_precision = totals[:, 2] / totals[:, 1]
        harm_rate = totals[:, 3] / totals[:, 1]

    def interval(value: np.ndarray) -> list[float] | None:
        finite = value[np.isfinite(value)]
        if not len(finite):
            return None
        return [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]

    return {
        "coverage_ci95": interval(coverage),
        "dual_precision_ci95": interval(dual_precision),
        "harm_rate_ci95": interval(harm_rate),
    }


def route_counts(
    fixed_index: np.ndarray,
    combined_index: np.ndarray,
    values: np.ndarray,
    routes: np.ndarray,
) -> list[dict[str, Any]]:
    fixed_delta = chosen(values, fixed_index) - values[:, 0]
    combined_delta = chosen(values, combined_index) - values[:, 0]
    result = []
    for route in sorted(np.unique(routes.astype(str)).tolist()):
        mask = routes == route
        fixed_promote = mask & (fixed_index != 0)
        combined_promote = mask & (combined_index != 0)
        result.append(
            {
                "route": route,
                "states": int(mask.sum()),
                "fixed_promotions": int(fixed_promote.sum()),
                "combined_promotions": int(combined_promote.sum()),
                "net_added": int(combined_promote.sum() - fixed_promote.sum()),
                "combined_dual_successes": int(
                    (
                        combined_promote
                        & (combined_delta[:, 0] < -1.0e-7)
                        & (combined_delta[:, 3] < -1.0e-7)
                    ).sum()
                ),
                "combined_harms": int(
                    (
                        combined_promote
                        & (
                            (combined_delta[:, 0] > 1.0e-7)
                            | (combined_delta[:, 3] > 1.0e-7)
                        )
                    ).sum()
                ),
            }
        )
    return result


def marginal_subset(
    fixed_index: np.ndarray,
    combined_index: np.ndarray,
    values: np.ndarray,
) -> dict[str, Any]:
    fixed = chosen(values, fixed_index)
    combined = chosen(values, combined_index)
    incumbent = values[:, 0]
    added = (fixed_index == 0) & (combined_index != 0)
    replaced = (fixed_index != 0) & (combined_index != fixed_index)

    def subset(mask: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n": 0}
        delta_incumbent = combined[mask] - incumbent[mask]
        delta_baseline = combined[mask] - baseline[mask]
        dual = (delta_incumbent[:, 0] < -1.0e-7) & (
            delta_incumbent[:, 3] < -1.0e-7
        )
        harm = (delta_incumbent[:, 0] > 1.0e-7) | (
            delta_incumbent[:, 3] > 1.0e-7
        )
        return {
            "n": int(mask.sum()),
            "dual_precision_vs_incumbent": float(dual.mean()),
            "harm_rate_vs_incumbent": float(harm.mean()),
            "mean_delta_vs_baseline_m": {
                name: float(delta_baseline[:, metric].mean())
                for metric, name in enumerate(METRICS)
            },
        }

    return {
        "new_coverage": subset(added, incumbent),
        "fixed_action_replacements": subset(replaced, fixed),
        "all_selection_changes": subset(combined_index != fixed_index, fixed),
    }


def quality_nondegraded(combined: dict[str, Any], fixed: dict[str, Any]) -> bool:
    return all(
        combined["absolute"][name]["mean_m"]
        <= fixed["absolute"][name]["mean_m"] + 1.0e-9
        and combined["absolute"][name]["p90_m"]
        <= fixed["absolute"][name]["p90_m"] + 1.0e-9
        for name in METRICS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261065)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("route bootstrap requires at least 1000 replicates")
    output_root = args.root / "evaluation_v60"
    result_path = output_root / "result.json"
    outcome_audit_path = output_root / "outcome_open_and_spent_audit.json"
    if output_root.exists():
        raise FileExistsError(output_root)

    freeze_path = args.root / "GH_preoutcome_selection_freeze_v60.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "policy_and_GH_selected_indices_frozen_before_any_outcome"
        or freeze.get("labels_opened")
        or freeze.get("outcomes_accessed")
        or freeze.get("reserve_accessed")
        or freeze.get("post_outcome_tuning_or_regrouping_allowed")
    ):
        raise RuntimeError("v60 pre-outcome selection freeze is invalid")
    reserve_path = Path(freeze["sealed_25route_reserve"])
    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    if (
        sha256(reserve_path) != freeze["sealed_25route_reserve_sha256"]
        or reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
    ):
        raise RuntimeError("sealed 25-route reserve changed or was accessed")

    bank_payloads: dict[str, dict[str, Any]] = {}
    for bank_number, bank in enumerate(("bank_g_v60", "bank_h_v60")):
        bank_freeze = freeze["banks"][bank]
        archive_path = Path(bank_freeze["archive"])
        manifest_path = Path(bank_freeze["manifest"])
        if (
            sha256(archive_path) != bank_freeze["archive_sha256"]
            or sha256(manifest_path) != bank_freeze["manifest_sha256"]
        ):
            raise RuntimeError(f"{bank}: frozen archive or manifest changed")
        manifest_rows = read_rows(manifest_path)
        record = np.asarray(
            [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in manifest_rows]
        )
        trajectory = np.asarray([str(row["trajectory_id"]) for row in manifest_rows])
        route = np.asarray([str(row["route_id"]) for row in manifest_rows])
        with np.load(archive_path, allow_pickle=False) as payload:
            candidates = payload["candidates_waypoint_units"].astype(np.float32)
            archive_record = payload["record_key"].astype(str)
            archive_route = payload["route_id"].astype(str)
            fixed_index = payload["fixed_selected_index"].astype(np.int64)
            combined_index = payload["combined_selected_index"].astype(np.int64)
        if (
            candidates.shape != (44, 65, 4, 2)
            or not np.array_equal(record, archive_record)
            or not np.array_equal(route, archive_route)
            or array_sha256(candidates) != bank_freeze["candidate_array_sha256"]
            or array_sha256(fixed_index) != bank_freeze["fixed_selected_index_sha256"]
            or array_sha256(combined_index)
            != bank_freeze["combined_selected_index_sha256"]
        ):
            raise RuntimeError(f"{bank}: frozen identity/selection contract failed")

        # First and only outcome access for these 44 states.
        ground_truth = np.stack(
            [
                local_ground_truth(
                    args.trajectory_root / str(row["trajectory_id"]) / "traj_data.pkl",
                    int(row["current_t"]),
                )
                for row in manifest_rows
            ]
        ).astype(np.float32)
        labels = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)
        values = labels["values"].astype(np.float32)
        outcome_path = output_root / f"{bank}_outcomes.npz"
        write_npz(
            outcome_path,
            record_key=record,
            trajectory=trajectory,
            route_id=route,
            ground_truth_local_m=ground_truth,
            absolute_metrics=values,
            targets=labels["targets"].astype(np.float32),
            fixed_selected_index=fixed_index,
            combined_selected_index=combined_index,
        )

        selected_actions_path = args.root / bank / "cem_run_v60" / "selected_actions.npz"
        with np.load(selected_actions_path, allow_pickle=False) as selected_actions:
            native_index = selected_actions["native_scorer_v2_selected_index"].astype(
                np.int64
            )
        methods_index = {
            "incumbent": np.zeros(44, dtype=np.int64),
            "native_scorer_v2": native_index,
            "cem_false_fixed": fixed_index,
            "cem_true_unified": combined_index,
            "posthoc_objective_K65_oracle": values[..., 3].argmin(axis=1).astype(np.int64),
        }
        methods = {
            name: method_metrics(
                index,
                values,
                trajectory,
                route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + bank_number * 100 + method_number * 10,
            )
            for method_number, (name, index) in enumerate(methods_index.items())
        }
        methods["cem_false_fixed"]["route_clustered_safety_ci95"] = route_method_ci(
            fixed_index,
            values,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + bank_number * 100 + 60,
        )
        methods["cem_true_unified"]["route_clustered_safety_ci95"] = route_method_ci(
            combined_index,
            values,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + bank_number * 100 + 70,
        )
        comparison = route_bootstrap_comparison(
            fixed_index,
            combined_index,
            values,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + bank_number * 100 + 80,
        )
        bank_payloads[bank] = {
            "record": record,
            "trajectory": trajectory,
            "route": route,
            "values": values,
            "fixed_index": fixed_index,
            "combined_index": combined_index,
            "native_index": native_index,
            "methods": methods,
            "comparison": comparison,
            "marginal": marginal_subset(fixed_index, combined_index, values),
            "route_counts": route_counts(fixed_index, combined_index, values, route),
            "outcome_path": outcome_path,
        }

    pooled_values = np.concatenate(
        [bank_payloads[bank]["values"] for bank in bank_payloads], axis=0
    )
    pooled_trajectory = np.concatenate(
        [bank_payloads[bank]["trajectory"] for bank in bank_payloads]
    )
    pooled_route = np.concatenate([bank_payloads[bank]["route"] for bank in bank_payloads])
    pooled_fixed = np.concatenate(
        [bank_payloads[bank]["fixed_index"] for bank in bank_payloads]
    )
    pooled_combined = np.concatenate(
        [bank_payloads[bank]["combined_index"] for bank in bank_payloads]
    )
    pooled_native = np.concatenate(
        [bank_payloads[bank]["native_index"] for bank in bank_payloads]
    )
    pooled_methods_index = {
        "incumbent": np.zeros(88, dtype=np.int64),
        "native_scorer_v2": pooled_native,
        "cem_false_fixed": pooled_fixed,
        "cem_true_unified": pooled_combined,
        "posthoc_objective_K65_oracle": pooled_values[..., 3].argmin(axis=1).astype(
            np.int64
        ),
    }
    pooled_methods = {
        name: method_metrics(
            index,
            pooled_values,
            pooled_trajectory,
            pooled_route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 500 + number * 10,
        )
        for number, (name, index) in enumerate(pooled_methods_index.items())
    }
    pooled_methods["cem_false_fixed"]["route_clustered_safety_ci95"] = route_method_ci(
        pooled_fixed,
        pooled_values,
        pooled_route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 560,
    )
    pooled_methods["cem_true_unified"]["route_clustered_safety_ci95"] = route_method_ci(
        pooled_combined,
        pooled_values,
        pooled_route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 570,
    )
    pooled_comparison = route_bootstrap_comparison(
        pooled_fixed,
        pooled_combined,
        pooled_values,
        pooled_route,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 580,
    )

    gate = freeze["success_gate"]
    pooled_fixed_metrics = pooled_methods["cem_false_fixed"]
    pooled_combined_metrics = pooled_methods["cem_true_unified"]
    net_added = (
        pooled_combined_metrics["promotions"] - pooled_fixed_metrics["promotions"]
    )
    bank_checks: dict[str, dict[str, bool]] = {}
    for bank, payload in bank_payloads.items():
        fixed = payload["methods"]["cem_false_fixed"]
        combined = payload["methods"]["cem_true_unified"]
        bank_checks[bank] = {
            "coverage_gain_positive": (
                combined["promotion_fraction"] - fixed["promotion_fraction"] > 0
            ),
            "dual_at_least_0.75": (
                combined["dual_precision"] is not None
                and combined["dual_precision"] >= float(gate["each_bank_dual_min"])
            ),
            "harm_at_most_0.25": (
                combined["harm_rate_per_promotion"] is not None
                and combined["harm_rate_per_promotion"] <= float(gate["each_bank_harm_max"])
            ),
            "all_four_mean_and_p90_nondegraded": quality_nondegraded(combined, fixed),
        }
        bank_checks[bank]["direction_consistent"] = all(bank_checks[bank].values())
    pooled_checks = {
        "net_added_at_least_9": net_added >= int(gate["combined_added_windows_min"]),
        "coverage_gain_at_least_0.10": pooled_comparison["coverage_gain"]
        >= float(gate["combined_coverage_gain_min"]),
        "dual_at_least_0.85": (
            pooled_combined_metrics["dual_precision"] is not None
            and pooled_combined_metrics["dual_precision"]
            >= float(gate["pooled_dual_min"])
        ),
        "harm_at_most_0.15": (
            pooled_combined_metrics["harm_rate_per_promotion"] is not None
            and pooled_combined_metrics["harm_rate_per_promotion"]
            <= float(gate["pooled_harm_max"])
        ),
        "all_four_mean_and_p90_nondegraded": quality_nondegraded(
            pooled_combined_metrics, pooled_fixed_metrics
        ),
    }
    overall_pass = all(pooled_checks.values()) and all(
        value["direction_consistent"] for value in bank_checks.values()
    )

    output_root.mkdir(parents=True, exist_ok=False)
    # The NPZ files were staged before mkdir by write_npz, which creates parents.
    bank_reports = {
        bank: {
            "states": 44,
            "routes": 11,
            "methods": payload["methods"],
            "cem_true_minus_false": payload["comparison"],
            "marginal_cem_contribution": payload["marginal"],
            "route_clustered_counts": payload["route_counts"],
            "direction_checks": bank_checks[bank],
            "outcome_artifact": str(payload["outcome_path"]),
            "outcome_artifact_sha256": sha256(payload["outcome_path"]),
        }
        for bank, payload in bank_payloads.items()
    }
    spent_routes = sorted(np.unique(pooled_route.astype(str)).tolist())
    audit = {
        "schema": "recon_small_fresh_GH_v60_outcome_open_spent",
        "status": "all_22_routes_spent_after_one_time_outcome_evaluation",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "selection_freeze": str(freeze_path),
        "selection_freeze_sha256": sha256(freeze_path),
        "labels_opened": True,
        "outcomes_accessed": True,
        "spent_routes": spent_routes,
        "spent_route_count": len(spent_routes),
        "post_outcome_tuning_or_regrouping_allowed": False,
        "sealed_25route_reserve": str(reserve_path),
        "sealed_25route_reserve_sha256_after_evaluation": sha256(reserve_path),
        "sealed_25route_reserve_labels_opened": False,
        "sealed_25route_reserve_outcomes_accessed": False,
        "reserve_accessed": False,
    }
    write_json(outcome_audit_path, audit)
    report = {
        "schema": "recon_small_fresh_GH_v60_result",
        "status": (
            "promising_development_confirmation_passed"
            if overall_pass
            else "small_fresh_development_confirmation_failed_frozen_gate"
        ),
        "evidence_role": "small_fresh_development_confirmation_not_H4_100_not_production_safe",
        "protocol_is_original_25route_H4_100": False,
        "states": 88,
        "routes": 22,
        "banks": bank_reports,
        "pooled": {
            "methods": pooled_methods,
            "cem_true_minus_false": pooled_comparison,
            "marginal_cem_contribution": marginal_subset(
                pooled_fixed, pooled_combined, pooled_values
            ),
            "route_clustered_counts": route_counts(
                pooled_fixed, pooled_combined, pooled_values, pooled_route
            ),
            "success_checks": pooled_checks,
            "net_added_windows": int(net_added),
        },
        "bank_direction_checks": bank_checks,
        "overall_frozen_success_gate_pass": overall_pass,
        "success_gate": gate,
        "small_sample_limitation": (
            "each bank has only 11 independent routes and 44 states; route-clustered "
            "intervals are correspondingly wide"
        ),
        "selection_freeze": str(freeze_path),
        "selection_freeze_sha256": sha256(freeze_path),
        "outcome_open_audit": str(outcome_audit_path),
        "outcome_open_audit_sha256": sha256(outcome_audit_path),
        "all_22_routes_now_spent": True,
        "post_outcome_tuning_or_regrouping_allowed": False,
        "sealed_25route_reserve_accessed": False,
        "production_safe_claim_allowed": False,
        "cem": True,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    write_json(result_path, report)
    compact = {
        "status": report["status"],
        "overall_pass": overall_pass,
        "pooled_checks": pooled_checks,
        "bank_checks": bank_checks,
        "pooled": {
            name: {
                "promotions": value["promotions"],
                "coverage": value["promotion_fraction"],
                "dual": value["dual_precision"],
                "wilson": value["dual_wilson_lower_95"],
                "harm": value["harm_rate_per_promotion"],
                "absolute": value["absolute"],
            }
            for name, value in pooled_methods.items()
        },
        "net_added": int(net_added),
        "comparison": pooled_comparison,
        "marginal": report["pooled"]["marginal_cem_contribution"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
