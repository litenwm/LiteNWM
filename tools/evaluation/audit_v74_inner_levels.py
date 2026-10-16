#!/usr/bin/env python3
"""Summarize frozen v74 inner-only policy levels without changing selection."""

from __future__ import annotations

import json

import numpy as np

from tools.evaluation import run_v72_proposal_tail_reranker as v72
from tools.evaluation import run_v74_direct_delta_nested as v74


def main() -> None:
    v74.verify_contract()
    data, audit = v74.load_data()
    rows = []
    for outer in range(5):
        cached = v72.load_npz(v74.outer_cache_path(outer))
        for method in v74.METHODS:
            prediction = v74.expand_inner(data, cached, method)
            for domain in ("recon", "scand"):
                states = np.flatnonzero(
                    (data["fold"] != outer)
                    & (data["domain"].astype(str) == domain)
                )
                policy = v74.fit_policy(data, states, prediction, method)
                for level in policy["all_levels"]:
                    metrics = level["metrics"]
                    rows.append(
                        {
                            "outer_fold": outer,
                            "domain": domain,
                            "method": method,
                            **level["parameters"],
                            "replacements": metrics["replacements"],
                            "dual_precision": metrics["dual_precision"],
                            "dual_lcb": metrics["dual_wilson_95"]["lower"],
                            "harm_rate": metrics["harm_rate"],
                            "harm_ucb": metrics["harm_wilson_95"]["upper"],
                            "mean_objective_gain_m": metrics["mean_gain_m"]["objective"],
                            "feasible": level["feasible"],
                        }
                    )
    nonzero = [row for row in rows if row["replacements"] > 0]
    summary = {
        "schema": "v74_inner_level_audit_v1",
        "selection_impact": "none_posthoc_inner_only_diagnostic",
        "outer_truth_used": False,
        "levels": len(rows),
        "nonzero_replacement_levels": len(nonzero),
        "dual_wilson_lcb_passes": sum(row["dual_lcb"] >= 0.85 for row in nonzero),
        "harm_wilson_ucb_passes": sum(row["harm_ucb"] <= 0.15 for row in nonzero),
        "joint_wilson_passes": sum(
            row["dual_lcb"] >= 0.85 and row["harm_ucb"] <= 0.15
            for row in nonzero
        ),
        "fully_feasible_levels": sum(row["feasible"] for row in rows),
        "maximum_replacements": max((row["replacements"] for row in rows), default=0),
        "best_nonzero_by_dual_lcb": sorted(
            nonzero,
            key=lambda row: (
                row["dual_lcb"],
                -row["harm_ucb"],
                row["mean_objective_gain_m"],
            ),
            reverse=True,
        )[:10],
        "best_nonzero_by_objective_gain": sorted(
            nonzero,
            key=lambda row: (
                row["mean_objective_gain_m"],
                row["dual_lcb"],
                -row["harm_ucb"],
            ),
            reverse=True,
        )[:10],
        "outer_assignment_sha256": audit["base_v72_data_audit"]["route_folds"]["assignment_sha256"],
        "route_overlap_violations": 0,
        "reserve_accessed": False,
    }
    v72.write_json(v74.OUTPUT / "inner_level_audit.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
