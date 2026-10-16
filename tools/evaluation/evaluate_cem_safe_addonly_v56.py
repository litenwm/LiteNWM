#!/usr/bin/env python3
"""Evaluate frozen fixed-safe plus CEM-safe add-only development selection."""

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
from tools.evaluation.evaluate_cem_safe_elite_v55 import (  # noqa: E402
    elite_true_utility,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261056)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def added_subset_report(
    *,
    added: np.ndarray,
    selected: np.ndarray,
    absolute: np.ndarray,
) -> dict[str, object]:
    row = np.flatnonzero(added)
    index = selected[added]
    chosen = absolute[row, index]
    incumbent = absolute[row, 0]
    delta = chosen - incumbent
    dual = (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
    quality = {}
    for metric, name in enumerate(NAMES):
        quality[name] = {
            "absolute_mean_m": float(chosen[:, metric].mean()),
            "absolute_p90_m": float(np.quantile(chosen[:, metric], 0.9)),
            "delta_vs_incumbent_mean_m": float(delta[:, metric].mean()),
            "delta_vs_incumbent_p90_m": float(
                np.quantile(delta[:, metric], 0.9)
            ),
        }
    return {
        "n": int(len(row)),
        "dual": float(dual.mean()) if len(row) else None,
        "harm": float(harm.mean()) if len(row) else None,
        "quality": quality,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if (
        freeze.get("status")
        != "hybrid_addonly_selection_frozen_before_outcomes"
        or freeze.get("fixed_pool_action_replacement_allowed")
        or freeze.get("labels_opened")
        or freeze.get("outcomes_accessed")
        or freeze.get("state_gate_used")
        or freeze.get("reserve_accessed")
    ):
        raise RuntimeError("v56 pre-outcome freeze contract is invalid")
    if file_sha256(args.selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("v56 selected index changed")
    cem_freeze_path = Path(freeze["cem_freeze"])
    cem_freeze = json.loads(cem_freeze_path.read_text(encoding="utf-8"))
    if file_sha256(cem_freeze_path) != freeze["cem_freeze_sha256"]:
        raise RuntimeError("CEM freeze changed")
    if file_sha256(args.manifest) != cem_freeze["manifest_sha256"]:
        raise RuntimeError("manifest changed")

    rows = read_rows(args.manifest)
    expected = np.asarray(
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
    with np.load(args.selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), expected):
            raise RuntimeError("selection/manifest identity mismatch")
        base = payload["base_k17_feature_waypoint_units"].astype(np.float32)
        final = payload["final_cem_feature_waypoint_units"].astype(np.float32)
        cem_false = payload["safe_cem_false_selected_index"].astype(np.int64)
        cem_local = payload["final_cem_safe_selected_local_index"].astype(
            np.int64
        )
        hybrid = payload["hybrid_combined_selected_index"].astype(np.int64)
        added = payload["cem_added_window"].astype(bool)
    if not np.array_equal(hybrid[cem_false != 0], cem_false[cem_false != 0]):
        raise RuntimeError("fixed-pool promotion was replaced")
    if not np.array_equal(added, (cem_false == 0) & (cem_local != 0)):
        raise RuntimeError("CEM attribution mask changed")

    combined = np.concatenate((base, final[:, 1:]), axis=1)
    absolute = metric_labels(combined, ground_truth, 0.25, 0.4, 0.6)["values"]
    methods_index = {
        "incumbent": np.zeros(len(rows), dtype=np.int64),
        "safe_cem_false": cem_false,
        "safe_fixed_plus_cem_addonly": hybrid,
    }
    methods = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 100 * index,
        )
        for index, (name, selected) in enumerate(methods_index.items())
    }
    reference = methods["safe_cem_false"]
    hybrid_metrics = methods["safe_fixed_plus_cem_addonly"]
    relative = {
        "coverage_delta": float(
            hybrid_metrics["promotion_fraction"] - reference["promotion_fraction"]
        ),
        "dual_delta": float(
            hybrid_metrics["dual_precision"] - reference["dual_precision"]
        ),
        "harm_delta": float(
            hybrid_metrics["harm_rate_per_promotion"]
            - reference["harm_rate_per_promotion"]
        ),
        "absolute_delta_hybrid_minus_cem_false": {
            name: {
                "mean_m": float(
                    hybrid_metrics["absolute"][name]["mean_m"]
                    - reference["absolute"][name]["mean_m"]
                ),
                "p90_m": float(
                    hybrid_metrics["absolute"][name]["p90_m"]
                    - reference["absolute"][name]["p90_m"]
                ),
            }
            for name in NAMES
        },
    }
    quality_nondegraded = all(
        relative["absolute_delta_hybrid_minus_cem_false"][name][summary]
        <= 1.0e-12
        for name in NAMES
        for summary in ("mean_m", "p90_m")
    )
    checks = {
        "coverage_at_least_0.40": hybrid_metrics["promotion_fraction"] >= 0.40,
        "dual_at_least_0.85": hybrid_metrics["dual_precision"] >= 0.85,
        "harm_at_most_0.15": hybrid_metrics["harm_rate_per_promotion"] <= 0.15,
        "all_mean_and_p90_nondegraded": quality_nondegraded,
    }
    checks["cohort_pass"] = all(checks.values())
    marginal = added_subset_report(
        added=added, selected=hybrid, absolute=absolute
    )
    iteration_utility = elite_true_utility(
        freeze_dir=cem_freeze_path.parent,
        ground_truth=ground_truth,
    )

    outcome_path = args.output.with_suffix(".outcomes.npz")
    write_npz(
        outcome_path,
        record_key=expected,
        trajectory=trajectory,
        route_group=route,
        combined_absolute_metrics=absolute,
        safe_cem_false_selected_index=cem_false,
        hybrid_combined_selected_index=hybrid,
        cem_added_window=added,
    )
    report = {
        "schema": "recon_cem_safe_addonly_v56_development_result",
        "status": "B_or_C_spent_development_evaluated_after_v56_freeze",
        "evidence_role": "spent_development_confirmation_not_blind_not_production",
        "cem": True,
        "state_gate_used": False,
        "fixed_pool_action_replacement_allowed": False,
        "freeze": str(args.freeze),
        "freeze_sha256": file_sha256(args.freeze),
        "methods": methods,
        "hybrid_relative_to_cem_false": relative,
        "cem_marginal_added_windows": marginal,
        "iteration_elite_predicted_vs_true": iteration_utility,
        "promising_development_checks": checks,
        "outcomes": str(outcome_path),
        "outcomes_sha256": file_sha256(outcome_path),
        "outcomes_used_inside_cem_or_selector": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "methods": methods,
                "hybrid_relative_to_cem_false": relative,
                "cem_marginal_added_windows": marginal,
                "promising_development_checks": checks,
                "iteration_elite_predicted_vs_true": iteration_utility,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
