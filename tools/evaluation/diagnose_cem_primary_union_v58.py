#!/usr/bin/env python3
"""Spent E/F attribution: CEM-only and one-gate fixed/CEM union arbitration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
)
from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.planning.run_cem_development_v1 import domain_policy  # noqa: E402
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    select_lowest_energy_safe,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    sigmoid,
)


NAMES = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ef-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261058)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def v57_score_base(
    *,
    base_features: Path,
    training_result: Path,
    gate: dict[str, float],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    base = load_features(base_features, labels_expected=False)
    training = read_json(training_result)
    if training.get("cem") is not False:
        raise RuntimeError("base scorer is not the frozen CEM=false model")
    envelopes = load_envelopes(Path(str(training["support"])))
    envelope = envelopes["recon"]
    threshold = threshold_for_quantile(envelope, 0.999)
    feature, support_score, physical_guard = build_universal_features(
        base, spacing_m=0.25, support_envelope=envelope
    )
    valid = (
        physical_guard
        & (support_score <= threshold)
        & np.asarray(base["majority_support"], dtype=bool)
    )
    valid[:, 0] = True
    domain_id = np.zeros(len(base["record_key"]), dtype=np.int64)
    prediction = ensemble_predict(
        training["arms"]["D"]["checkpoints"]["full"],
        feature,
        domain_id,
        device=device,
    )
    _, dual_temperature, harm_temperature = domain_policy(training, "recon")
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    mean = prediction["mean"]
    std = prediction["std"]
    benefit_lcb = -0.5 * (mean[..., 0] + mean[..., 3]) - float(
        gate["beta"]
    ) * np.sqrt(std[..., 0] ** 2 + std[..., 3] ** 2)
    energy = -benefit_lcb + 0.25 * p_harm - 0.10 * p_dual
    energy = np.where(valid, energy, np.inf).astype(np.float32)
    energy[:, 0] = np.inf
    safe = (
        valid
        & (benefit_lcb >= float(gate["benefit_threshold"]))
        & (p_dual >= float(gate["dual_threshold"]))
        & (p_harm <= float(gate["harm_threshold"]))
    )
    safe[:, 0] = False
    return energy, safe


def replacement_report(
    *,
    fixed: np.ndarray,
    union: np.ndarray,
    absolute: np.ndarray,
) -> dict[str, object]:
    replaced = (fixed != 0) & (union != fixed)
    row = np.flatnonzero(replaced)
    delta = absolute[row, union[replaced]] - absolute[row, fixed[replaced]]
    dual = (delta[:, 0] < -1.0e-7) & (delta[:, 3] < -1.0e-7)
    harm = (delta[:, 0] > 1.0e-7) | (delta[:, 3] > 1.0e-7)
    utility = -0.5 * (delta[:, 0] + delta[:, 3])
    return {
        "n": int(replaced.sum()),
        "fixed_to_CEM_replacements": int(replaced.sum()),
        "true_dual_better_than_fixed": float(dual.mean()) if len(row) else None,
        "true_harm_vs_fixed": float(harm.mean()) if len(row) else None,
        "net_utility_mean_m": float(utility.mean()) if len(row) else None,
        "net_utility_p10_m": (
            float(np.quantile(utility, 0.1)) if len(row) else None
        ),
        "metric_delta_union_minus_fixed": {
            name: {
                "mean_m": float(delta[:, index].mean()) if len(row) else None,
                "p90_m": (
                    float(np.quantile(delta[:, index], 0.9))
                    if len(row)
                    else None
                ),
            }
            for index, name in enumerate(NAMES)
        },
    }


def relative_to_addonly(
    method: dict[str, object], addonly: dict[str, object]
) -> dict[str, object]:
    return {
        "coverage_delta": float(
            method["promotion_fraction"] - addonly["promotion_fraction"]
        ),
        "dual_delta": float(
            method["dual_precision"] - addonly["dual_precision"]
        ),
        "harm_delta": float(
            method["harm_rate_per_promotion"]
            - addonly["harm_rate_per_promotion"]
        ),
        "absolute_delta": {
            name: {
                summary: float(
                    method["absolute"][name][summary]
                    - addonly["absolute"][name][summary]
                )
                for summary in ("mean_m", "p90_m")
            }
            for name in NAMES
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    ef = read_json(args.ef_result)
    if (
        ef.get("status")
        != "frozen_one_time_EF_development_failed_substantial_gain_rule"
        or ef.get("production_safe_claim_allowed")
        or ef.get("reserve_accessed")
        or not ef.get("reserve_still_sealed")
    ):
        raise RuntimeError("E/F spent-development attribution contract invalid")
    pre_path = Path(str(ef["preoutcome_freeze"]))
    if file_sha256(pre_path) != ef["preoutcome_freeze_sha256"]:
        raise RuntimeError("pre-outcome E/F freeze changed")
    pre = read_json(pre_path)
    gate = pre["candidate_gate"]

    pooled_absolute = []
    pooled_trajectory = []
    pooled_route = []
    pooled_index = {
        "incumbent": [],
        "safe_cem_false": [],
        "addonly": [],
        "cem_only": [],
        "unified_union": [],
    }
    cohorts = {}
    replacement_parts = []
    score_identity_audit = {}

    for bank in ("bank_e", "bank_f"):
        frozen = pre["cohorts"][bank]
        contract_path = Path(str(frozen["preutility_contract"]))
        contract = read_json(contract_path)
        prelabel_path = Path(str(contract["prelabel_freeze"]))
        prelabel = read_json(prelabel_path)
        selector_dir = Path(str(frozen["addonly_freeze"])).parent
        result = read_json(selector_dir / "result.json")
        outcomes_path = Path(str(result["outcomes"]))
        with np.load(outcomes_path, allow_pickle=False) as payload:
            absolute = payload["combined_absolute_metrics"].astype(np.float32)
            trajectory = payload["trajectory"].astype(str)
            route = payload["route_group"].astype(str)
            fixed = payload["safe_cem_false_selected_index"].astype(np.int64)
            addonly = payload["hybrid_combined_selected_index"].astype(np.int64)
        cem_selected_path = Path(str(frozen["cem_selected_actions"]))
        with np.load(cem_selected_path, allow_pickle=False) as payload:
            saved_base_safe = payload["base_safe"].astype(bool)
            final_local = payload[
                "universal_d_cem_true_selected_index"
            ].astype(np.int64)
            final_energy = payload["final_cem_energy"].astype(np.float32)
            final_safe = payload["final_cem_safe"].astype(bool)

        training_path = Path(str(prelabel["training_result"]))
        base_features_path = Path(str(prelabel["base_features"]))
        base_energy, base_safe = v57_score_base(
            base_features=base_features_path,
            training_result=training_path,
            gate=gate,
            device=args.device,
        )
        base_selected = select_lowest_energy_safe(base_energy, base_safe)
        final_selected = select_lowest_energy_safe(final_energy, final_safe)
        base_safe_equal = np.array_equal(base_safe, saved_base_safe)
        fixed_equal = np.array_equal(base_selected, fixed)
        final_equal = np.array_equal(final_selected, final_local)
        if not (base_safe_equal and fixed_equal and final_equal):
            raise RuntimeError(f"{bank} frozen-score identity mismatch")

        row = np.arange(len(fixed))
        best_base_energy = base_energy[row, base_selected]
        best_final_energy = final_energy[row, final_selected]
        has_base = base_selected != 0
        has_final = final_selected != 0
        choose_final = has_final & (
            (~has_base) | (best_final_energy < best_base_energy)
        )
        union = base_selected.copy()
        union[choose_final] = 16 + final_selected[choose_final]
        cem_only = np.where(has_final, 16 + final_selected, 0).astype(np.int64)
        if not np.array_equal((union != 0), (addonly != 0)):
            raise RuntimeError(f"{bank} union/add-only coverage identity changed")

        method_index = {
            "incumbent": np.zeros(len(fixed), dtype=np.int64),
            "safe_cem_false": fixed,
            "addonly": addonly,
            "cem_only": cem_only,
            "unified_union": union,
        }
        methods = {
            name: method_metrics(
                selected,
                absolute,
                trajectory,
                route,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + 1000 * (0 if bank == "bank_e" else 1)
                + 10 * index,
            )
            for index, (name, selected) in enumerate(method_index.items())
        }
        replacement = replacement_report(
            fixed=fixed, union=union, absolute=absolute
        )
        replacement_parts.append(
            {
                "fixed": fixed,
                "union": union,
                "absolute": absolute,
            }
        )
        cohorts[bank] = {
            "methods": methods,
            "cem_only_relative_to_addonly": relative_to_addonly(
                methods["cem_only"], methods["addonly"]
            ),
            "unified_union_relative_to_addonly": relative_to_addonly(
                methods["unified_union"], methods["addonly"]
            ),
            "unified_union_replacement_subset": replacement,
            "score_identity_audit": {
                "recomputed_base_safe_matches_frozen": base_safe_equal,
                "recomputed_base_selection_matches_frozen": fixed_equal,
                "final_selection_matches_frozen": final_equal,
                "same_v57_gate_and_LCB_used_for_base_and_CEM": True,
                "selection_outcome_free": True,
            },
        }
        score_identity_audit[bank] = cohorts[bank]["score_identity_audit"]
        pooled_absolute.append(absolute)
        pooled_trajectory.append(trajectory)
        pooled_route.append(route)
        for name, selected in method_index.items():
            pooled_index[name].append(selected)

    absolute = np.concatenate(pooled_absolute)
    trajectory = np.concatenate(pooled_trajectory)
    route = np.concatenate(pooled_route)
    selected = {name: np.concatenate(parts) for name, parts in pooled_index.items()}
    methods = {
        name: method_metrics(
            index,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + 100 * offset,
        )
        for offset, (name, index) in enumerate(selected.items())
    }
    fixed = selected["safe_cem_false"]
    union = selected["unified_union"]
    replacement = replacement_report(fixed=fixed, union=union, absolute=absolute)
    comparisons = {
        name: relative_to_addonly(methods[name], methods["addonly"])
        for name in ("cem_only", "unified_union")
    }

    def materially_better(name: str) -> bool:
        comparison = comparisons[name]
        quality = all(
            comparison["absolute_delta"][metric][summary] <= 1.0e-12
            for metric in NAMES
            for summary in ("mean_m", "p90_m")
        )
        safety_material = (
            comparison["dual_delta"] >= 0.02
            or comparison["harm_delta"] <= -0.02
        )
        return bool(
            quality
            and safety_material
            and comparison["coverage_delta"] >= -0.05
            and comparison["dual_delta"] >= -0.02
            and comparison["harm_delta"] <= 0.02
        )

    diagnostic_flags = {
        name: materially_better(name)
        for name in ("cem_only", "unified_union")
    }
    preferred = next(
        (name for name in ("unified_union", "cem_only") if diagnostic_flags[name]),
        None,
    )
    policy_draft = None
    if preferred == "unified_union":
        policy_draft = {
            "name": "v58_unified_union_single_gate_draft",
            "rule": (
                "Run true 3x16 safe-elite CEM. Score the original K17 and final "
                "CEM candidates with the same frozen candidate gate and LCB; "
                "select the lowest-energy safe candidate over their union, else "
                "incumbent. Do not privilege or preserve the old fixed action."
            ),
            "new_cohort_run_authorized_by_this_diagnostic": False,
            "must_be_preregistered_and_frozen_before_any_new_cohort": True,
        }
    elif preferred == "cem_only":
        policy_draft = {
            "name": "v58_cem_primary_single_gate_draft",
            "rule": (
                "Run true 3x16 safe-elite CEM and select the lowest-LCB final "
                "safe CEM candidate, otherwise incumbent; the K17 pool only "
                "initializes q and is not directly selected."
            ),
            "new_cohort_run_authorized_by_this_diagnostic": False,
            "must_be_preregistered_and_frozen_before_any_new_cohort": True,
        }

    report = {
        "schema": "recon_cem_primary_union_v58_spent_EF_diagnostic",
        "status": "spent_EF_attribution_complete_no_new_cohort_run",
        "evidence_role": "outcome_aware_attribution_only_not_validation",
        "selection_formula_frozen_before_diagnostic": {
            "candidate_gate": gate,
            "formula": pre["formula"],
            "cem_only": "final safe CEM candidate, otherwise incumbent",
            "unified_union": (
                "lowest LCB energy among v57-safe original K17 and final CEM "
                "candidates, otherwise incumbent"
            ),
        },
        "cohorts": cohorts,
        "pooled_methods": methods,
        "pooled_relative_to_addonly": comparisons,
        "pooled_unified_union_replacement_subset": replacement,
        "score_identity_audit": score_identity_audit,
        "materially_better_diagnostic_rule": {
            "all_four_mean_and_p90_nondegraded": True,
            "dual_gain_at_least_0.02_or_harm_drop_at_least_0.02": True,
            "coverage_drop_at_most_0.05": True,
            "dual_drop_at_most_0.02": True,
            "harm_increase_at_most_0.02": True,
        },
        "materially_better_than_addonly": diagnostic_flags,
        "preferred_next_strategy_diagnostic": preferred,
        "single_candidate_strategy_draft": policy_draft,
        "E_F_outcomes_were_already_spent": True,
        "new_cohort_opened_or_run": False,
        "post_E_F_tuning_performed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
