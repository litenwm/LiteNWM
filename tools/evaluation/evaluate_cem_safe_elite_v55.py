#!/usr/bin/env python3
"""Evaluate frozen v55 CEM true/false and per-iteration elite utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.build_recon100_k17_source_v1 import metric_labels  # noqa: E402
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRICS = ("ATE", "RPE", "Endpoint", "Objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261055)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def elite_true_utility(
    *,
    freeze_dir: Path,
    ground_truth: np.ndarray,
) -> list[dict[str, object]]:
    reports = []
    for iteration in range(3):
        directory = freeze_dir / f"iteration_{iteration:02d}"
        with np.load(directory / "population.npz", allow_pickle=False) as payload:
            candidates = payload["candidates_feature_waypoint_units"].astype(
                np.float32
            )
        with np.load(directory / "frozen_scores.npz", allow_pickle=False) as payload:
            energy = payload["energy"].astype(np.float32)
            safe = payload["safe"].astype(bool)
            elite = payload["elite_index_for_update"].astype(np.int64)
            updated = payload["proposal_updated"].astype(bool)
        absolute = metric_labels(candidates, ground_truth, 0.25, 0.4, 0.6)[
            "values"
        ]
        state_index = np.repeat(np.flatnonzero(updated), 4)
        candidate_index = elite[updated].reshape(-1)
        if len(state_index):
            delta = absolute[state_index, candidate_index] - absolute[
                state_index, 0
            ]
            scorer_utility = -energy[state_index, candidate_index]
            true_utility = -0.5 * (delta[:, 0] + delta[:, 3])
            dual = (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
            harm = (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
            correlation = (
                float(np.corrcoef(scorer_utility, true_utility)[0, 1])
                if scorer_utility.std() > 0 and true_utility.std() > 0
                else None
            )
            delta_report = {
                name: {
                    "mean": float(delta[:, index].mean()),
                    "p90": float(np.quantile(delta[:, index], 0.9)),
                }
                for index, name in enumerate(METRICS)
            }
            report = {
                "iteration": iteration,
                "updated_windows": int(updated.sum()),
                "elite_candidates": int(len(state_index)),
                "safe_candidates": int(safe[:, 1:].sum()),
                "predicted_scorer_utility_mean": float(scorer_utility.mean()),
                "true_utility_definition": "-0.5*(delta_ATE+delta_Objective)",
                "true_utility_mean": float(true_utility.mean()),
                "true_utility_p10": float(np.quantile(true_utility, 0.1)),
                "true_utility_p50": float(np.quantile(true_utility, 0.5)),
                "predicted_true_utility_correlation": correlation,
                "elite_dual_rate": float(dual.mean()),
                "elite_harm_rate": float(harm.mean()),
                "elite_delta": delta_report,
            }
        else:
            report = {
                "iteration": iteration,
                "updated_windows": 0,
                "elite_candidates": 0,
                "safe_candidates": int(safe[:, 1:].sum()),
                "predicted_scorer_utility_mean": None,
                "true_utility_mean": None,
                "elite_dual_rate": None,
                "elite_harm_rate": None,
            }
        reports.append(report)
    return reports


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        freeze.get("schema") != "universal_raw_d_cem_safe_elite_v55"
        or freeze.get("labels_opened")
        or freeze.get("outcomes_used_by_cem")
        or freeze.get("state_gate_used")
        or contract.get("status")
        != "safe_gate_proposals_and_paired_comparison_frozen_before_true_utility"
        or contract.get("labels_opened")
        or contract.get("outcomes_accessed")
        or contract.get("state_gate_used")
    ):
        raise RuntimeError("pre-utility contract is invalid")
    if file_sha256(args.freeze) != contract["prelabel_freeze_sha256"]:
        raise RuntimeError("prelabel freeze changed")
    if file_sha256(args.selected) != freeze["selected_actions_sha256"]:
        raise RuntimeError("selected actions changed")
    if file_sha256(args.manifest) != freeze["manifest_sha256"]:
        raise RuntimeError("manifest changed")

    rows = read_rows(args.manifest)
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    ground_truth = np.stack(
        [
            local_ground_truth(
                args.trajectory_root
                / str(row["trajectory_id"])
                / "traj_data.pkl",
                int(row["current_t"]),
            )
            for row in rows
        ]
    ).astype(np.float32)
    trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    route = np.asarray([str(row["route_id"]) for row in rows])
    if set(route) != set(contract["paired_comparison"]["routes"]):
        raise RuntimeError("contract/manifest route identity mismatch")

    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected_record):
            raise RuntimeError("selected/manifest record mismatch")
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        native = payload["native_scorer_v2_selected_index"].astype(np.int64)
        cem_false = payload["universal_d_cem_false_selected_index"].astype(
            np.int64
        )
        cem_true_local = payload["universal_d_cem_true_selected_index"].astype(
            np.int64
        )
    if base.shape != (100, 17, 4, 2) or final.shape != (100, 17, 4, 2):
        raise RuntimeError("candidate shape changed")
    combined = np.concatenate((base, final[:, 1:]), axis=1)
    absolute = metric_labels(combined, ground_truth, 0.25, 0.4, 0.6)["values"]
    cem_true = np.where(cem_true_local == 0, 0, 16 + cem_true_local)
    methods_index = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "native_scorer_v2": native,
        "safe_cem_false": cem_false,
        "safe_cem_true": cem_true,
        "posthoc_combined_pool_oracle": absolute[..., 3].argmin(axis=1),
    }
    methods = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index * 100,
        )
        for index, (name, selected) in enumerate(methods_index.items())
    }
    reference = methods["safe_cem_false"]
    cem = methods["safe_cem_true"]
    quality_checks = {
        "all_means_nondegraded": all(
            cem["absolute"][name]["mean_m"]
            <= reference["absolute"][name]["mean_m"] + 1.0e-12
            for name in ("ate", "rpe", "endpoint", "objective")
        ),
        "all_p90_nondegraded": all(
            cem["absolute"][name]["p90_m"]
            <= reference["absolute"][name]["p90_m"] + 1.0e-12
            for name in ("ate", "rpe", "endpoint", "objective")
        ),
    }
    relaxed = {
        "coverage_at_least_0.40": cem["promotion_fraction"] >= 0.40,
        "dual_at_least_0.85": (
            cem["dual_precision"] is not None and cem["dual_precision"] >= 0.85
        ),
        "harm_at_most_0.15": (
            cem["harm_rate_per_promotion"] is not None
            and cem["harm_rate_per_promotion"] <= 0.15
        ),
        **quality_checks,
    }
    relaxed["cohort_pass"] = all(relaxed.values())
    iteration_utility = elite_true_utility(
        freeze_dir=args.freeze.parent,
        ground_truth=ground_truth,
    )

    outcome_path = args.output.with_suffix(".outcomes.npz")
    write_npz(
        outcome_path,
        record_key=expected_record,
        trajectory=trajectory,
        route_group=route,
        combined_absolute_metrics=absolute,
        safe_cem_false_selected_index=cem_false,
        safe_cem_true_combined_index=cem_true,
    )
    report = {
        "schema": "cem_safe_elite_v55_development_result",
        "status": "spent_development_evaluated_after_preutility_contract",
        "evidence_role": "A_to_T_spent_development_only",
        "blind_confirmation": False,
        "production_safe_claim_allowed": False,
        "cem": True,
        "state_gate_used": False,
        "prelabel_freeze": str(args.freeze),
        "prelabel_freeze_sha256": file_sha256(args.freeze),
        "preutility_contract": str(args.contract),
        "preutility_contract_sha256": file_sha256(args.contract),
        "methods": methods,
        "iteration_elite_predicted_vs_true": iteration_utility,
        "relaxed_development_checks": relaxed,
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "outcomes_used_inside_cem": False,
        "reserve_accessed": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "methods": {
                    name: {
                        "promotions": value["promotions"],
                        "coverage": value["promotion_fraction"],
                        "dual": value["dual_precision"],
                        "wilson": value["dual_wilson_lower_95"],
                        "harm": value["harm_rate_per_promotion"],
                        "absolute": value["absolute"],
                    }
                    for name, value in methods.items()
                },
                "iteration_elite_predicted_vs_true": iteration_utility,
                "relaxed_development_checks": relaxed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
