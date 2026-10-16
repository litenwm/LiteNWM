#!/usr/bin/env python3
"""Freeze v60 boundary rule with the audited non-identical tie addendum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation.audit_cem_existing_score_frontier_v59 import (  # noqa: E402
    BANKS,
    raw_base_scores,
    select_rule,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    write_json_atomic,
    write_npz_atomic,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--tie-addendum", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def eligible_entries(
    frontier: dict[str, object], amendment: dict[str, object]
) -> list[dict[str, object]]:
    requirement = amendment["development_boundary_eligibility"]
    guard = requirement["single_cohort_guard"]
    result = []
    for entry in frontier["frontier"]:
        pooled = entry["pooled"]
        per_bank = entry["per_bank"]
        bank_ok = all(
            per_bank[bank]["combined_coverage"] >= float(guard["coverage_min"])
            and per_bank[bank]["coverage_gain"]
            >= float(guard["coverage_gain_min"])
            and per_bank[bank]["dual"] >= float(guard["dual_min"])
            and per_bank[bank]["harm"] <= float(guard["harm_max"])
            for bank in BANKS
        )
        if (
            pooled["combined"]["dual"] >= float(requirement["pooled_dual_min"])
            and pooled["combined"]["harm"] <= float(requirement["pooled_harm_max"])
            and pooled["relative"]["all_quality_nondegraded"]
            and bank_ok
        ):
            result.append(entry)
    return result


def score_archive(
    root: Path, bank: str, device: str
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, object]]:
    run_name = (
        "cem_run_v57_lobo_gate"
        if bank in ("bank_e", "bank_f")
        else "cem_run_v55_safe_elite"
    )
    run_dir = root / bank / run_name
    freeze_path = run_dir / "prelabel_freeze.json"
    freeze = load(freeze_path)
    base_path = Path(str(freeze["base_features"]))
    base = raw_base_scores(
        base_features=base_path,
        training_result=Path(str(freeze["training_result"])),
        device=device,
    )
    arrays = {
        "mean": [base["mean"]],
        "std": [base["std"]],
        "p_dual": [base["p_dual"]],
        "p_harm": [base["p_harm"]],
        "valid": [base["valid"]],
    }
    score_sha = []
    for iteration in range(3):
        path = run_dir / f"iteration_{iteration:02d}" / "frozen_scores.npz"
        with np.load(path, allow_pickle=False) as payload:
            arrays["mean"].append(payload["mean"][:, 1:].astype(np.float32))
            arrays["std"].append(payload["std"][:, 1:].astype(np.float32))
            arrays["p_dual"].append(
                payload["dual_probability"][:, 1:].astype(np.float32)
            )
            arrays["p_harm"].append(
                payload["harm_probability"][:, 1:].astype(np.float32)
            )
            arrays["valid"].append(payload["valid"][:, 1:].astype(bool))
        score_sha.append(file_sha256(path))
    scores = {key: np.concatenate(value, axis=1) for key, value in arrays.items()}
    provenance = {
        "prelabel_freeze_sha256": file_sha256(freeze_path),
        "base_features_sha256": file_sha256(base_path),
        "iteration_score_sha256": score_sha,
    }
    return scores, base["record_key"], provenance


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    frontier = load(args.frontier)
    amendment = load(args.amendment)
    addendum = load(args.tie_addendum)
    if (
        amendment.get("G_H_outcomes_accessed")
        or amendment.get("grid_expansion_allowed")
        or addendum.get("status")
        != "tie_resolution_frozen_after_A_F_identity_audit_and_before_new_GH_manifests"
        or addendum.get("new_GH_manifests_created")
        or addendum.get("G_H_outcomes_accessed")
        or addendum.get("grid_expansion_allowed")
        or frontier.get("G_H_outcomes_accessed")
        or frontier.get("grid_rules") != 36
    ):
        raise RuntimeError("invalid v60 boundary contract")
    eligible = eligible_entries(frontier, amendment)
    maximum = max(entry["pooled"]["relative"]["coverage_gain"] for entry in eligible)
    ties = [
        entry
        for entry in eligible
        if abs(entry["pooled"]["relative"]["coverage_gain"] - maximum) <= 1e-12
    ]
    tie_ids = sorted(int(entry["config"]["id"]) for entry in ties)
    if tie_ids != amendment["expected_boundary"]["candidate_grid_ids"]:
        raise RuntimeError(f"tie ids changed: {tie_ids}")
    selected = max(
        ties,
        key=lambda item: (
            float(item["config"]["dual_threshold"]),
            -float(item["config"]["harm_threshold"]),
            float(item["config"]["benefit_threshold"]),
            float(item["config"]["dominance_margin"]),
            float(item["config"]["beta"]),
            -int(item["config"]["id"]),
        ),
    )
    selected_id = int(selected["config"]["id"])
    if selected_id != int(addendum["expected_unique_id"]):
        raise RuntimeError("unique tie result changed")

    bank_scores = {}
    records = []
    names = []
    provenance = {}
    for bank in BANKS:
        scores, record, source = score_archive(args.root, bank, args.device)
        bank_scores[bank] = scores
        records.append(record)
        names.append(np.full(len(record), bank))
        provenance[bank] = source
    tie_selection = {}
    for entry in ties:
        fixed_parts = []
        combined_parts = []
        for bank in BANKS:
            fixed, combined = select_rule(bank_scores[bank], entry["config"])
            fixed_parts.append(fixed)
            combined_parts.append(combined)
        tie_selection[int(entry["config"]["id"])] = (
            np.concatenate(fixed_parts),
            np.concatenate(combined_parts),
        )
    differences = {
        str(value): {
            "fixed_vs_selected_count": int(
                (tie_selection[value][0] != tie_selection[selected_id][0]).sum()
            ),
            "combined_vs_selected_count": int(
                (tie_selection[value][1] != tie_selection[selected_id][1]).sum()
            ),
        }
        for value in tie_ids
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "AF_selected_index.npz"
    write_npz_atomic(
        selected_path,
        record_key=np.concatenate(records),
        bank=np.concatenate(names),
        fixed_selected_index=tie_selection[selected_id][0],
        combined_selected_index=tie_selection[selected_id][1],
    )
    freeze = {
        "schema": "recon_cem_boundary_rule_v60_freeze",
        "status": "unique_boundary_rule_frozen_before_new_GH_manifests",
        "evidence_role": "A_F_spent_development_boundary_for_small_fresh_GH",
        "frontier": str(args.frontier),
        "frontier_sha256": file_sha256(args.frontier),
        "amendment": str(args.amendment),
        "amendment_sha256": file_sha256(args.amendment),
        "tie_addendum": str(args.tie_addendum),
        "tie_addendum_sha256": file_sha256(args.tie_addendum),
        "eligible_rule_count": len(eligible),
        "max_coverage_tie_ids": tie_ids,
        "tie_selection_differences": differences,
        "tie_break_order": [
            "higher dual_threshold",
            "lower harm_threshold",
            "higher benefit_threshold",
            "larger dominance_margin",
            "larger beta",
            "lower grid id",
        ],
        "selected_rule": selected["config"],
        "selected_development_metrics": selected["pooled"],
        "formula": {
            "benefit_lcb": "-0.5*(mean_delta_ATE+mean_delta_Objective)-beta*sqrt(std_ATE^2+std_Objective^2)",
            "safe": "support_valid AND benefit_lcb>=benefit_threshold AND p_dual>=dual_threshold AND p_harm<=harm_threshold",
            "energy": "-benefit_lcb + 0.25*p_harm - 0.10*p_dual",
            "selection": "best safe fixed; best safe over all three CEM archives; replace fixed only if E_cem + dominance_margin < E_fixed; otherwise fixed; if neither safe use incumbent",
        },
        "AF_selected_index": str(selected_path),
        "AF_selected_index_sha256": file_sha256(selected_path),
        "source_sha256": provenance,
        "new_GH": {
            "names": ["bank_g_v60", "bank_h_v60"],
            "routes_each": 11,
            "states_each": 44,
            "outcomes_accessed": False,
        },
        "grid_expanded": False,
        "state_gate_used": False,
        "G_H_outcomes_accessed": False,
        "post_GH_tuning_allowed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(args.output_dir / "boundary_freeze.json", freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
