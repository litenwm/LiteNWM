#!/usr/bin/env python3
"""Archive the precommitted v56-style add-only decomposition for v60 G/H.

This reads only the already-written v60 outcome artifacts.  It does not reopen
trajectory outcomes, alter the failed unified primary, tune a rule, or access
the sealed reserve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.evaluate_recon_cem_boundary_gh_v60 import (  # noqa: E402
    METRICS,
    marginal_subset,
    quality_nondegraded,
    route_bootstrap_comparison,
    route_counts,
    route_method_ci,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import method_metrics  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import write_json, write_npz  # noqa: E402


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


def relative_summary(
    addonly: dict[str, Any], fixed: dict[str, Any]
) -> dict[str, Any]:
    return {
        "coverage_delta": float(
            addonly["promotion_fraction"] - fixed["promotion_fraction"]
        ),
        "dual_delta": float(addonly["dual_precision"] - fixed["dual_precision"]),
        "harm_delta": float(
            addonly["harm_rate_per_promotion"]
            - fixed["harm_rate_per_promotion"]
        ),
        "absolute_delta_addonly_minus_fixed": {
            name: {
                "mean_m": float(
                    addonly["absolute"][name]["mean_m"]
                    - fixed["absolute"][name]["mean_m"]
                ),
                "p90_m": float(
                    addonly["absolute"][name]["p90_m"]
                    - fixed["absolute"][name]["p90_m"]
                ),
            }
            for name in METRICS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--v56-preregistration",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_cem_safe_addonly_v56/preregistration.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261066)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1000:
        raise ValueError("route bootstrap requires at least 1000 replicates")

    evaluation_root = args.root / "evaluation_v60"
    primary_path = evaluation_root / "result.json"
    secondary_path = evaluation_root / "addonly_secondary_decomposition.json"
    selected_path = evaluation_root / "addonly_secondary_selected_index.npz"
    if secondary_path.exists() or selected_path.exists():
        raise FileExistsError(secondary_path)
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    freeze_path = Path(primary["selection_freeze"])
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    outcome_audit_path = Path(primary["outcome_open_audit"])
    outcome_audit = json.loads(outcome_audit_path.read_text(encoding="utf-8"))
    prereg = json.loads(args.v56_preregistration.read_text(encoding="utf-8"))
    reserve_path = Path(freeze["sealed_25route_reserve"])
    reserve = json.loads(reserve_path.read_text(encoding="utf-8"))
    if (
        primary.get("status")
        != "small_fresh_development_confirmation_failed_frozen_gate"
        or primary.get("overall_frozen_success_gate_pass")
        or primary.get("sealed_25route_reserve_accessed")
        or freeze.get("status")
        != "policy_and_GH_selected_indices_frozen_before_any_outcome"
        or outcome_audit.get("status")
        != "all_22_routes_spent_after_one_time_outcome_evaluation"
        or outcome_audit.get("spent_route_count") != 22
        or prereg.get("selection_rule", {}).get(
            "fixed_pool_action_replacement_allowed"
        )
        or reserve.get("labels_opened")
        or reserve.get("outcomes_accessed")
        or sha256(reserve_path) != freeze["sealed_25route_reserve_sha256"]
    ):
        raise RuntimeError("primary/add-only/reserve archive contract failed")
    if sha256(freeze_path) != primary["selection_freeze_sha256"]:
        raise RuntimeError("primary selection freeze changed")
    if sha256(outcome_audit_path) != primary["outcome_open_audit_sha256"]:
        raise RuntimeError("primary outcome audit changed")

    banks: dict[str, dict[str, Any]] = {}
    selection_arrays: dict[str, np.ndarray] = {}
    pooled_arrays: dict[str, list[np.ndarray]] = {
        "values": [],
        "trajectory": [],
        "route": [],
        "fixed": [],
        "unified": [],
        "addonly": [],
        "record": [],
    }
    for bank_number, bank in enumerate(("bank_g_v60", "bank_h_v60")):
        bank_primary = primary["banks"][bank]
        bank_freeze = freeze["banks"][bank]
        outcome_path = Path(bank_primary["outcome_artifact"])
        archive_path = Path(bank_freeze["archive"])
        if (
            sha256(outcome_path) != bank_primary["outcome_artifact_sha256"]
            or sha256(archive_path) != bank_freeze["archive_sha256"]
        ):
            raise RuntimeError(f"{bank}: primary outcome/archive changed")
        with np.load(outcome_path, allow_pickle=False) as outcome:
            record = outcome["record_key"].astype(str)
            trajectory = outcome["trajectory"].astype(str)
            route = outcome["route_id"].astype(str)
            values = outcome["absolute_metrics"].astype(np.float32)
            frozen_fixed = outcome["fixed_selected_index"].astype(np.int64)
            frozen_unified = outcome["combined_selected_index"].astype(np.int64)
        with np.load(archive_path, allow_pickle=False) as archive:
            fixed = archive["fixed_selected_index"].astype(np.int64)
            unified = archive["combined_selected_index"].astype(np.int64)
            archive_record = archive["record_key"].astype(str)
            archive_route = archive["route_id"].astype(str)
        if (
            values.shape != (44, 65, 4)
            or not np.array_equal(record, archive_record)
            or not np.array_equal(route, archive_route)
            or not np.array_equal(fixed, frozen_fixed)
            or not np.array_equal(unified, frozen_unified)
            or array_sha256(fixed) != bank_freeze["fixed_selected_index_sha256"]
            or array_sha256(unified)
            != bank_freeze["combined_selected_index_sha256"]
        ):
            raise RuntimeError(f"{bank}: pre-frozen selection identity mismatch")

        # v56 structural rule applied to the nine pre-frozen v60 CEM identities:
        # keep every fixed-safe action; use the already-selected CEM identity only
        # where fixed abstained.  This mapping does not inspect metric values.
        added = (fixed == 0) & (unified != 0)
        addonly = np.where(fixed != 0, fixed, unified).astype(np.int64)
        if not np.array_equal(addonly[fixed != 0], fixed[fixed != 0]):
            raise RuntimeError(f"{bank}: add-only replaced a fixed-safe action")
        if not np.array_equal(added, (fixed == 0) & (addonly != 0)):
            raise RuntimeError(f"{bank}: add-only attribution changed")

        methods_index = {
            "cem_false_fixed": fixed,
            "cem_true_unified_primary_failed": unified,
            "cem_addonly_secondary": addonly,
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
        methods["cem_addonly_secondary"]["route_clustered_safety_ci95"] = (
            route_method_ci(
                addonly,
                values,
                route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + bank_number * 100 + 60,
            )
        )
        comparison = route_bootstrap_comparison(
            fixed,
            addonly,
            values,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + bank_number * 100 + 70,
        )
        relative = relative_summary(
            methods["cem_addonly_secondary"], methods["cem_false_fixed"]
        )
        direction = {
            "coverage_increased": relative["coverage_delta"] > 0.0,
            "dual_not_worse": relative["dual_delta"] >= -1.0e-12,
            "harm_not_worse": relative["harm_delta"] <= 1.0e-12,
            "all_four_mean_and_p90_nondegraded": quality_nondegraded(
                methods["cem_addonly_secondary"], methods["cem_false_fixed"]
            ),
        }
        direction["consistent_addonly_direction"] = all(direction.values())
        banks[bank] = {
            "states": 44,
            "routes": 11,
            "pre_frozen_added_windows": int(added.sum()),
            "fixed_safe_actions_preserved": int((fixed != 0).sum()),
            "methods": methods,
            "addonly_minus_fixed": comparison,
            "relative": relative,
            "marginal_pre_frozen_added_subset": marginal_subset(
                fixed, addonly, values
            )["new_coverage"],
            "route_clustered_counts": route_counts(
                fixed, addonly, values, route
            ),
            "direction_checks": direction,
        }
        selection_arrays[f"{bank}_record_key"] = record
        selection_arrays[f"{bank}_fixed_selected_index"] = fixed
        selection_arrays[f"{bank}_unified_primary_selected_index"] = unified
        selection_arrays[f"{bank}_addonly_secondary_selected_index"] = addonly
        selection_arrays[f"{bank}_pre_frozen_added_window"] = added
        for key, value in (
            ("values", values),
            ("trajectory", trajectory),
            ("route", route),
            ("fixed", fixed),
            ("unified", unified),
            ("addonly", addonly),
            ("record", record),
        ):
            pooled_arrays[key].append(value)

    pooled = {key: np.concatenate(value, axis=0) for key, value in pooled_arrays.items()}
    if int(((pooled["fixed"] == 0) & (pooled["addonly"] != 0)).sum()) != 9:
        raise RuntimeError("pooled pre-frozen add-only count changed from nine")
    pooled_methods_index = {
        "cem_false_fixed": pooled["fixed"],
        "cem_true_unified_primary_failed": pooled["unified"],
        "cem_addonly_secondary": pooled["addonly"],
    }
    pooled_methods = {
        name: method_metrics(
            index,
            pooled["values"],
            pooled["trajectory"],
            pooled["route"],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 500 + method_number * 10,
        )
        for method_number, (name, index) in enumerate(pooled_methods_index.items())
    }
    pooled_methods["cem_addonly_secondary"]["route_clustered_safety_ci95"] = (
        route_method_ci(
            pooled["addonly"],
            pooled["values"],
            pooled["route"],
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 560,
        )
    )
    pooled_comparison = route_bootstrap_comparison(
        pooled["fixed"],
        pooled["addonly"],
        pooled["values"],
        pooled["route"],
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 570,
    )
    pooled_relative = relative_summary(
        pooled_methods["cem_addonly_secondary"], pooled_methods["cem_false_fixed"]
    )
    addonly_metrics = pooled_methods["cem_addonly_secondary"]
    fixed_metrics = pooled_methods["cem_false_fixed"]
    pooled_checks = {
        "net_added_equals_9": (
            addonly_metrics["promotions"] - fixed_metrics["promotions"] == 9
        ),
        "coverage_gain_at_least_9_over_88": pooled_comparison["coverage_gain"]
        >= 9.0 / 88.0 - 1.0e-12,
        "dual_at_least_0.85": addonly_metrics["dual_precision"] >= 0.85,
        "harm_at_most_0.15": addonly_metrics["harm_rate_per_promotion"] <= 0.15,
        "all_four_mean_and_p90_nondegraded": quality_nondegraded(
            addonly_metrics, fixed_metrics
        ),
        "G_H_consistent_direction": all(
            value["direction_checks"]["consistent_addonly_direction"]
            for value in banks.values()
        ),
    }
    addonly_substantial_gain = all(pooled_checks.values())
    conclusion = (
        "CEM add-only obtained substantial gain, while replacement failed"
        if addonly_substantial_gain
        else "only the pre-frozen CEM added subset succeeded; unified replacement failed"
    )

    write_npz(selected_path, **selection_arrays)
    report = {
        "schema": "recon_small_fresh_GH_v60_addonly_secondary_decomposition",
        "status": "secondary_addonly_decomposition_archived_without_rule_tuning",
        "primary_unified_policy_status_unchanged": primary["status"],
        "primary_unified_policy_pass": False,
        "evidence_role": (
            "secondary decomposition of pre-frozen added identities under the "
            "existing v56 add-only rule; not a rewrite of the v60 primary"
        ),
        "selection_rule": (
            "Preserve every fixed-safe selected identity. Only where fixed abstained, "
            "use the corresponding pre-frozen v60 CEM selected identity."
        ),
        "outcome_dependent_rule_or_selection_tuning": False,
        "banks": banks,
        "pooled": {
            "states": 88,
            "routes": 22,
            "pre_frozen_added_windows": 9,
            "methods": pooled_methods,
            "addonly_minus_fixed": pooled_comparison,
            "relative": pooled_relative,
            "marginal_pre_frozen_added_subset": marginal_subset(
                pooled["fixed"], pooled["addonly"], pooled["values"]
            )["new_coverage"],
            "route_clustered_counts": route_counts(
                pooled["fixed"],
                pooled["addonly"],
                pooled["values"],
                pooled["route"],
            ),
            "success_checks": pooled_checks,
        },
        "addonly_substantial_gain": addonly_substantial_gain,
        "conclusion": conclusion,
        "primary_result": str(primary_path),
        "primary_result_sha256": sha256(primary_path),
        "primary_selection_freeze": str(freeze_path),
        "primary_selection_freeze_sha256": sha256(freeze_path),
        "primary_outcome_audit": str(outcome_audit_path),
        "primary_outcome_audit_sha256": sha256(outcome_audit_path),
        "v56_preregistration": str(args.v56_preregistration),
        "v56_preregistration_sha256": sha256(args.v56_preregistration),
        "derived_selected_index": str(selected_path),
        "derived_selected_index_sha256": sha256(selected_path),
        "all_22_routes_remain_spent": True,
        "post_outcome_rule_tuning_allowed": False,
        "sealed_25route_reserve_accessed": False,
        "sealed_25route_reserve_sha256": sha256(reserve_path),
        "production_safe_claim_allowed": False,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
    }
    write_json(secondary_path, report)
    compact = {
        "conclusion": conclusion,
        "addonly_substantial_gain": addonly_substantial_gain,
        "pooled_checks": pooled_checks,
        "pooled_methods": {
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
        "pooled_comparison": pooled_comparison,
        "bank_direction": {
            name: value["direction_checks"] for name, value in banks.items()
        },
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
