#!/usr/bin/env python3
"""Aggregate frozen v66 development replays without changing any policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    load_npz,
    write_json_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


POLICIES = (
    "original_arm_d_fixed_cem_false",
    "original_arm_d_cem",
    "proposal_aware_v66_fixed",
    "proposal_aware_v66_cem",
)
METRICS = ("ate", "rpe", "endpoint", "objective")
EXPECTED_STATUS = "spent_development_outcomes_evaluated_without_retuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pooled_metrics(parts: list[dict[str, np.ndarray]], policy: str) -> dict[str, Any]:
    promoted = np.concatenate([part[f"{policy}__promoted"] for part in parts]).astype(bool)
    dual = np.concatenate([part[f"{policy}__dual"] for part in parts]).astype(bool)
    harmed = np.concatenate([part[f"{policy}__harmed"] for part in parts]).astype(bool)
    absolute = np.concatenate(
        [part[f"{policy}__absolute_metrics"] for part in parts]
    ).astype(np.float32)
    delta = np.concatenate(
        [part[f"{policy}__delta_metrics"] for part in parts]
    ).astype(np.float32)
    promotions = int(promoted.sum())
    return {
        "states": int(len(promoted)),
        "promotions": promotions,
        "coverage": float(promoted.mean()),
        "dual_successes": int(dual.sum()),
        "dual_precision": float(dual.sum() / promotions) if promotions else None,
        "harm_count": int(harmed.sum()),
        "harm_rate": float(harmed.sum() / promotions) if promotions else None,
        "absolute_mean_m": {
            name: float(absolute[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "absolute_p90_m": {
            name: float(np.quantile(absolute[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
        "delta_mean_m": {
            name: float(delta[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "delta_p90_m": {
            name: float(np.quantile(delta[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
    }


def comparison(challenger: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    mean_delta = {
        name: challenger["absolute_mean_m"][name] - baseline["absolute_mean_m"][name]
        for name in METRICS
    }
    p90_delta = {
        name: challenger["absolute_p90_m"][name] - baseline["absolute_p90_m"][name]
        for name in METRICS
    }
    tolerance = 1.0e-7
    return {
        "coverage_gain": challenger["coverage"] - baseline["coverage"],
        "absolute_mean_delta_m": mean_delta,
        "absolute_p90_delta_m": p90_delta,
        "all_four_mean_nondegraded": all(
            value <= tolerance for value in mean_delta.values()
        ),
        "all_four_p90_nondegraded": all(
            value <= tolerance for value in p90_delta.values()
        ),
    }


def consistent_direction(result: dict[str, Any]) -> bool:
    baseline = result["decisions"]["original_arm_d_fixed_cem_false"]
    challenger = result["decisions"]["proposal_aware_v66_cem"]
    comp = result["primary_comparison"]
    return bool(
        comp["coverage_gain"] >= 0.0
        and challenger["dual_precision"] is not None
        and challenger["dual_precision"] >= 0.85
        and challenger["harm_rate"] is not None
        and challenger["harm_rate"] <= 0.15
        and comp["all_four_mean_nondegraded"]
        and comp["all_four_p90_nondegraded"]
        and challenger["absolute_mean_m"]["objective"]
        <= baseline["absolute_mean_m"]["objective"] + 1.0e-7
    )


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    arrays = []
    seen = set()
    bindings = []
    for path in args.result:
        value = read_json(path)
        if value.get("status") != EXPECTED_STATUS:
            raise RuntimeError(f"not a frozen replay result: {path}")
        if value.get("thresholds_tuned_on_this_cohort"):
            raise RuntimeError(f"post-result tuning reported by {path}")
        if value.get("reserve_accessed"):
            raise RuntimeError(f"reserve access reported by {path}")
        identity = (value["domain"], value["cohort"])
        if identity in seen:
            raise RuntimeError(f"duplicate cohort: {identity}")
        seen.add(identity)
        selected_path = Path(value["selected_metrics"])
        if file_sha256(selected_path) != value["selected_metrics_sha256"]:
            raise RuntimeError(f"selected metric hash changed: {selected_path}")
        results.append(value)
        arrays.append(load_npz(selected_path))
        bindings.append(
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "domain": value["domain"],
                "cohort": value["cohort"],
                "cohort_role": value["cohort_role"],
                "states": value["states"],
            }
        )

    pooled = {policy: pooled_metrics(arrays, policy) for policy in POLICIES}
    baseline_name = "original_arm_d_fixed_cem_false"
    challenger_name = "proposal_aware_v66_cem"
    primary = comparison(pooled[challenger_name], pooled[baseline_name])
    direction = {
        f"{value['domain']}::{value['cohort']}": consistent_direction(value)
        for value in results
    }
    consistent_count = int(sum(direction.values()))
    challenger = pooled[challenger_name]
    checks = {
        "coverage_at_least_0_40": challenger["coverage"] >= 0.40,
        "coverage_gain_at_least_0_10": primary["coverage_gain"] >= 0.10,
        "pooled_dual_at_least_0_85": (
            challenger["dual_precision"] is not None
            and challenger["dual_precision"] >= 0.85
        ),
        "pooled_harm_at_most_0_15": (
            challenger["harm_rate"] is not None
            and challenger["harm_rate"] <= 0.15
        ),
        "all_four_mean_nondegraded": primary["all_four_mean_nondegraded"],
        "all_four_p90_nondegraded": primary["all_four_p90_nondegraded"],
        "independent_cohorts_same_direction_at_least_2": consistent_count >= 2,
    }
    promising = all(checks.values())
    output = {
        "schema": "dualdomain_cem_tail_v66_aggregate_result",
        "status": (
            "promising_development_result"
            if promising
            else "failed_frozen_development_replay"
        ),
        "cohort_bindings": bindings,
        "pooled_decisions": pooled,
        "primary_baseline": baseline_name,
        "primary_challenger": challenger_name,
        "primary_comparison": primary,
        "consistent_direction_definition": (
            "nonnegative coverage gain, dual>=0.85, harm<=0.15, and all four "
            "absolute mean/p90 metrics nondegraded versus original Arm-D "
            "fixed-pool CEM=false within the cohort"
        ),
        "cohort_direction": direction,
        "consistent_direction_cohorts": consistent_count,
        "development_checks": checks,
        "promising_development_result": promising,
        "thresholds_retuned_after_outcomes": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
