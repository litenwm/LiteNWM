#!/usr/bin/env python3
"""Audit the v62 72-vs-v61 67 selection discrepancy without outcomes.

The only candidate-safety authority in this audit is the score/support state that
was written before G/H outcomes were opened.  The v62 CPU recomputation is an
object of the audit; it is never allowed to redefine eligibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


BANKS = ("bank_g_v60", "bank_h_v60")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def scalar(value: Any) -> float | int | bool | str:
    if isinstance(value, np.generic):
        return value.item()
    return value


def candidate_location(global_index: int) -> tuple[int, int]:
    if not 17 <= global_index < 65:
        raise ValueError(f"candidate {global_index} is not a three-round CEM identity")
    offset = global_index - 17
    return offset // 16, offset % 16 + 1


def audit_candidate(
    *,
    bank: str,
    state_index: int,
    global_index: int,
    record_key: str,
    bank_root: Path,
    archive: dict[str, np.ndarray],
    freeze: dict[str, Any],
) -> dict[str, Any]:
    iteration, local_index = candidate_location(global_index)
    iteration_dir = bank_root / "cem_run_v60" / f"iteration_{iteration:02d}"
    population_path = iteration_dir / "population.npz"
    scores_path = iteration_dir / "frozen_scores.npz"
    features_path = iteration_dir / "cem_features.npz"
    iteration_freeze = freeze["iteration_audit"][iteration]

    with np.load(population_path, allow_pickle=False) as population:
        population_keys = list(population.files)
        population_record = population["record_key"].astype(str)
        population_candidate = population[
            "candidates_feature_waypoint_units"
        ][state_index, local_index].astype(np.float32)
    with np.load(scores_path, allow_pickle=False) as scores:
        score_keys = list(scores.files)
        mean = scores["mean"][state_index, local_index].astype(np.float64)
        std = scores["std"][state_index, local_index].astype(np.float64)
        raw = {
            "energy": scalar(scores["energy"][state_index, local_index]),
            "safe": scalar(scores["safe"][state_index, local_index]),
            "valid": scalar(scores["valid"][state_index, local_index]),
            "p_dual": scalar(
                scores["dual_probability"][state_index, local_index]
            ),
            "p_harm": scalar(
                scores["harm_probability"][state_index, local_index]
            ),
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    with np.load(features_path, allow_pickle=False) as features:
        feature_keys = list(features.files)
        feature_record = features["record_key"].astype(str)
        feature_candidate = features["candidates"][state_index, local_index].astype(
            np.float32
        )
        support = {
            "support_features": features[
                "support_features"
            ][state_index, local_index].astype(np.float64).tolist(),
            "support_by_seed": features[
                "support_by_seed"
            ][:, state_index, local_index].astype(np.float64).tolist(),
            "majority_support": scalar(
                features["majority_support"][state_index, local_index]
            ),
        }

    archive_candidate = archive["candidates_waypoint_units"][
        state_index, global_index
    ].astype(np.float32)
    policy = freeze["policy"]
    ucb_ate = float(mean[0] + float(policy["z"]) * std[0])
    ucb_objective = float(mean[3] + float(policy["z"]) * std[3])
    reconstructed_checks = {
        "valid": bool(raw["valid"]),
        "majority_support": bool(support["majority_support"]),
        "p_dual_at_least_frozen_threshold": (
            float(raw["p_dual"]) >= float(policy["dual_threshold"])
        ),
        "p_harm_at_most_frozen_threshold": (
            float(raw["p_harm"]) <= float(policy["harm_threshold"])
        ),
        "ate_ucb_below_negative_margin": (
            ucb_ate < -float(policy["margin_ate"])
        ),
        "objective_ucb_below_negative_margin": (
            ucb_objective < -float(policy["margin_objective"])
        ),
    }
    identity_checks = {
        "record_matches_population": population_record[state_index] == record_key,
        "record_matches_features": feature_record[state_index] == record_key,
        "population_matches_archive": bool(
            np.array_equal(population_candidate, archive_candidate)
        ),
        "features_match_archive": bool(
            np.array_equal(feature_candidate, archive_candidate)
        ),
        "population_sha_matches_preoutcome_freeze": (
            sha256(population_path) == iteration_freeze["population_sha256"]
        ),
        "scores_sha_matches_preoutcome_freeze": (
            sha256(scores_path) == iteration_freeze["frozen_scores_sha256"]
        ),
        "features_sha_matches_preoutcome_freeze": (
            sha256(features_path) == iteration_freeze["feature_sha256"]
        ),
    }
    archived_safe = bool(archive["safe"][state_index, global_index])
    raw_safe = bool(raw["safe"])
    legal = (
        all(identity_checks.values())
        and archived_safe
        and raw_safe
        and bool(raw["valid"])
        and bool(support["majority_support"])
    )
    failed_gate_checks = [
        name for name, passed in reconstructed_checks.items() if not passed
    ]
    return {
        "bank": bank,
        "state_index": state_index,
        "record_key": record_key,
        "candidate_global_index": global_index,
        "iteration": iteration,
        "iteration_local_candidate_index": local_index,
        "preoutcome_authority": {
            "archive_safe": archived_safe,
            "archive_energy": scalar(archive["energy"][state_index, global_index]),
            "raw_frozen_score": raw,
            "support": support,
            "policy": policy,
            "ucb_ate": ucb_ate,
            "ucb_objective": ucb_objective,
            "reconstructed_gate_checks": reconstructed_checks,
            "failed_gate_checks": failed_gate_checks,
        },
        "identity_checks": identity_checks,
        "preoutcome_artifacts": {
            "population": str(population_path),
            "population_sha256": sha256(population_path),
            "frozen_scores": str(scores_path),
            "frozen_scores_sha256": sha256(scores_path),
            "cem_features": str(features_path),
            "cem_features_sha256": sha256(features_path),
            "population_keys": population_keys,
            "frozen_score_keys": score_keys,
            "feature_keys": feature_keys,
        },
        "contract_legal": legal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gh-root", type=Path, required=True)
    parser.add_argument("--v61-secondary", type=Path, required=True)
    parser.add_argument("--v62-selected", type=Path, required=True)
    parser.add_argument("--v61-result", type=Path, required=True)
    parser.add_argument("--v62-result", type=Path, required=True)
    parser.add_argument("--reserve-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)

    reserve_sha_before = sha256(args.reserve_freeze)
    reserve = read_json(args.reserve_freeze)
    if reserve.get("labels_opened") or reserve.get("outcomes_accessed"):
        raise RuntimeError("sealed reserve has been opened")

    with np.load(args.v61_secondary, allow_pickle=False) as v61_payload:
        v61 = {key: v61_payload[key].copy() for key in v61_payload.files}
    with np.load(args.v62_selected, allow_pickle=False) as v62_payload:
        v62 = {key: v62_payload[key].copy() for key in v62_payload.files}

    introduced: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    per_bank: dict[str, Any] = {}
    authoritative_fallback: dict[str, np.ndarray] = {}
    for bank in BANKS:
        bank_root = args.gh_root / bank
        freeze_path = bank_root / "cem_run_v60" / "prelabel_freeze.json"
        freeze = read_json(freeze_path)
        if (
            freeze.get("labels_opened")
            or freeze.get("future_pose_outcomes_opened")
            or freeze.get("outcomes_used_by_cem")
            or freeze.get("reserve_accessed")
            or not freeze.get("online_causal")
        ):
            raise RuntimeError(f"{bank}: pre-outcome causal contract failed")
        archive_path = bank_root / "combined_archive_preoutcome_v60.npz"
        with np.load(archive_path, allow_pickle=False) as archive_payload:
            archive = {key: archive_payload[key].copy() for key in archive_payload.files}

        records = v61[f"{bank}_record_key"].astype(str)
        fixed = v61[f"{bank}_fixed_selected_index"].astype(np.int64)
        authority = v61[
            f"{bank}_addonly_secondary_selected_index"
        ].astype(np.int64)
        v62_mask = v62["bank"].astype(str) == bank
        proposed_records = v62["record_key"][v62_mask].astype(str)
        proposed_fixed = v62["fixed_selected_index"][v62_mask].astype(np.int64)
        proposed = v62["cumulative_3_selected_index"][v62_mask].astype(np.int64)
        if not (
            np.array_equal(records, proposed_records)
            and np.array_equal(fixed, proposed_fixed)
            and np.array_equal(records, archive["record_key"].astype(str))
            and np.array_equal(fixed, archive["fixed_selected_index"].astype(np.int64))
        ):
            raise RuntimeError(f"{bank}: record/fixed identity mapping mismatch")
        expected_authority = np.where(
            fixed != 0, fixed, archive["combined_selected_index"].astype(np.int64)
        )
        if not np.array_equal(authority, expected_authority):
            raise RuntimeError(f"{bank}: v61 secondary is not the pre-outcome archive authority")

        introduced_mask = (authority == 0) & (proposed != 0)
        removed_mask = (authority != 0) & (proposed == 0)
        changed_mask = (authority != 0) & (proposed != 0) & (authority != proposed)
        bank_introduced = []
        for state_index in np.flatnonzero(introduced_mask):
            detail = audit_candidate(
                bank=bank,
                state_index=int(state_index),
                global_index=int(proposed[state_index]),
                record_key=str(records[state_index]),
                bank_root=bank_root,
                archive=archive,
                freeze=freeze,
            )
            introduced.append(detail)
            bank_introduced.append(detail)
        for state_index in np.flatnonzero(removed_mask):
            detail = audit_candidate(
                bank=bank,
                state_index=int(state_index),
                global_index=int(authority[state_index]),
                record_key=str(records[state_index]),
                bank_root=bank_root,
                archive=archive,
                freeze=freeze,
            )
            detail["v62_removed_preoutcome_authority_identity"] = True
            removed.append(detail)
        for state_index in np.flatnonzero(changed_mask):
            changed.append(
                {
                    "bank": bank,
                    "state_index": int(state_index),
                    "record_key": str(records[state_index]),
                    "v61_candidate_global_index": int(authority[state_index]),
                    "v62_candidate_global_index": int(proposed[state_index]),
                }
            )
        authoritative_fallback[f"{bank}_record_key"] = records
        authoritative_fallback[f"{bank}_fixed_selected_index"] = fixed
        authoritative_fallback[f"{bank}_selected_index"] = authority
        per_bank[bank] = {
            "states": int(len(records)),
            "v61_promotions": int((authority != 0).sum()),
            "v62_promotions": int((proposed != 0).sum()),
            "introduced_count": int(introduced_mask.sum()),
            "removed_count": int(removed_mask.sum()),
            "changed_nonzero_candidate_count": int(changed_mask.sum()),
            "net_promotion_change": int((proposed != 0).sum() - (authority != 0).sum()),
            "introduced_contract_legal": [
                bool(item["contract_legal"]) for item in bank_introduced
            ],
            "preoutcome_freeze": str(freeze_path),
            "preoutcome_freeze_sha256": sha256(freeze_path),
            "combined_archive": str(archive_path),
            "combined_archive_sha256": sha256(archive_path),
        }

    all_introduced_legal = bool(introduced) and all(
        item["contract_legal"] for item in introduced
    )
    exact_pure_five_addition = (
        len(introduced) == 5 and len(removed) == 0 and len(changed) == 0
    )
    promote_v62_72 = all_introduced_legal and exact_pure_five_addition
    decision = "accept_72" if promote_v62_72 else "fallback_to_frozen_67"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "authoritative_fallback_gh_selected_index.npz"
    np.savez_compressed(selected_path, **authoritative_fallback)
    v61_result = read_json(args.v61_result)
    v62_result = read_json(args.v62_result)
    reserve_sha_after = sha256(args.reserve_freeze)
    if reserve_sha_after != reserve_sha_before:
        raise RuntimeError("reserve changed during identity audit")

    result = {
        "schema": "recon_cem_72_vs_67_identity_audit_v63",
        "status": "complete",
        "decision": decision,
        "selection_rule": (
            "Any introduced identity failing original pre-outcome safe/support authority, "
            "or any non-pure +5 identity change, forces the complete policy back to v61 67."
        ),
        "identity_accounting": {
            "claimed_net_difference": 5,
            "introduced_identity_count": len(introduced),
            "removed_identity_count": len(removed),
            "changed_nonzero_candidate_count": len(changed),
            "net_promotion_change": len(introduced) - len(removed),
            "is_pure_five_addition": exact_pure_five_addition,
            "all_introduced_contract_legal": all_introduced_legal,
        },
        "why_v61_did_not_select_them": (
            "The v61 G/H add-only identities are exactly the pre-outcome combined archive. "
            "v62 rebuilt eligibility on CPU with a different benefit/p_dual/p_harm rule and "
            "then minimized over the merged archive. Therefore this is a gate redefinition/"
            "archive reselection issue, not final-round-only omission or record mapping loss."
        ),
        "per_bank": per_bank,
        "introduced_identities": introduced,
        "removed_preoutcome_authority_identities": removed,
        "changed_nonzero_candidate_identities": changed,
        "authoritative_fallback_selected_index": str(selected_path),
        "authoritative_fallback_selected_index_sha256": sha256(selected_path),
        "source_artifacts": {
            "v61_secondary": str(args.v61_secondary),
            "v61_secondary_sha256": sha256(args.v61_secondary),
            "v62_selected": str(args.v62_selected),
            "v62_selected_sha256": sha256(args.v62_selected),
            "v61_result": str(args.v61_result),
            "v61_result_sha256": sha256(args.v61_result),
            "v62_result": str(args.v62_result),
            "v62_result_sha256": sha256(args.v62_result),
        },
        "v61_frozen_67_summary": v61_result.get("cross_fitted_pooled", {}).get(
            "frozen_addonly"
        ),
        "v62_72_summary_is_non_authoritative": v62_result.get("curve", {}).get(
            "cumulative_iterations_3"
        ),
        "reserve": {
            "path": str(args.reserve_freeze),
            "sha256_before": reserve_sha_before,
            "sha256_after": reserve_sha_after,
            "labels_opened": bool(reserve.get("labels_opened", False)),
            "outcomes_accessed": bool(reserve.get("outcomes_accessed", False)),
        },
        "outcomes_used_for_identity_decision": False,
        "thresholds_tuned": False,
        "candidates_generated": False,
        "model_trained": False,
    }
    result_path = args.output_dir / "result.json"
    write_json(result_path, result)
    print(json.dumps({
        "decision": decision,
        "introduced": len(introduced),
        "removed": len(removed),
        "changed": len(changed),
        "legal_introduced": sum(item["contract_legal"] for item in introduced),
        "result": str(result_path),
        "selected_sha256": sha256(selected_path),
    }, indent=2))


if __name__ == "__main__":
    main()
