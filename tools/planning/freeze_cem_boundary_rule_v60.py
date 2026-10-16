#!/usr/bin/env python3
"""Freeze the unique v59 safe boundary rule and A-F selection identity."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    frontier = read_json(args.frontier)
    amendment = read_json(args.amendment)
    if (
        amendment.get("status")
        != "boundary_selection_and_GH_success_rule_frozen_before_new_GH_manifests"
        or amendment.get("grid_expansion_allowed")
        or amendment.get("G_H_outcomes_accessed")
        or amendment.get("reserve_accessed")
        or frontier.get("grid_rules") != 36
        or frontier.get("G_H_outcomes_accessed")
    ):
        raise RuntimeError("invalid boundary amendment/frontier contract")

    requirement = amendment["development_boundary_eligibility"]
    guard = requirement["single_cohort_guard"]
    eligible = []
    for entry in frontier["frontier"]:
        pooled = entry["pooled"]
        per_bank = entry["per_bank"]
        guards = all(
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
            and guards
        ):
            eligible.append(entry)
    if not eligible:
        raise RuntimeError("no amended boundary-eligible v59 rule")
    maximum_gain = max(item["pooled"]["relative"]["coverage_gain"] for item in eligible)
    ties = [
        item
        for item in eligible
        if abs(item["pooled"]["relative"]["coverage_gain"] - maximum_gain)
        <= 1.0e-12
    ]
    observed_ids = sorted(int(item["config"]["id"]) for item in ties)
    expected_ids = amendment["expected_boundary"]["candidate_grid_ids"]
    if observed_ids != expected_ids:
        raise RuntimeError(f"boundary tie set changed: {observed_ids}")

    bank_scores = {}
    record_parts = []
    bank_parts = []
    source_sha = {}
    for bank in BANKS:
        run_name = (
            "cem_run_v57_lobo_gate"
            if bank in ("bank_e", "bank_f")
            else "cem_run_v55_safe_elite"
        )
        run_dir = args.root / bank / run_name
        freeze_path = run_dir / "prelabel_freeze.json"
        freeze = read_json(freeze_path)
        base_path = Path(str(freeze["base_features"]))
        training_path = Path(str(freeze["training_result"]))
        base = raw_base_scores(
            base_features=base_path,
            training_result=training_path,
            device=args.device,
        )
        mean = [base["mean"]]
        std = [base["std"]]
        p_dual = [base["p_dual"]]
        p_harm = [base["p_harm"]]
        valid = [base["valid"]]
        iterations = []
        for iteration in range(3):
            path = run_dir / f"iteration_{iteration:02d}" / "frozen_scores.npz"
            with np.load(path, allow_pickle=False) as payload:
                mean.append(payload["mean"][:, 1:].astype(np.float32))
                std.append(payload["std"][:, 1:].astype(np.float32))
                p_dual.append(
                    payload["dual_probability"][:, 1:].astype(np.float32)
                )
                p_harm.append(
                    payload["harm_probability"][:, 1:].astype(np.float32)
                )
                valid.append(payload["valid"][:, 1:].astype(bool))
            iterations.append(file_sha256(path))
        bank_scores[bank] = {
            "mean": np.concatenate(mean, axis=1),
            "std": np.concatenate(std, axis=1),
            "p_dual": np.concatenate(p_dual, axis=1),
            "p_harm": np.concatenate(p_harm, axis=1),
            "valid": np.concatenate(valid, axis=1),
        }
        record_parts.append(base["record_key"])
        bank_parts.append(np.full(len(base["record_key"]), bank))
        source_sha[bank] = {
            "prelabel_freeze_sha256": file_sha256(freeze_path),
            "base_features_sha256": file_sha256(base_path),
            "iteration_score_sha256": iterations,
        }

    tie_selections = {}
    for entry in ties:
        config = entry["config"]
        fixed = []
        combined = []
        for bank in BANKS:
            bank_fixed, bank_combined = select_rule(bank_scores[bank], config)
            fixed.append(bank_fixed)
            combined.append(bank_combined)
        tie_selections[int(config["id"])] = (
            np.concatenate(fixed),
            np.concatenate(combined),
        )
    reference_id = observed_ids[0]
    ties_identical = all(
        np.array_equal(tie_selections[reference_id][0], tie_selections[value][0])
        and np.array_equal(
            tie_selections[reference_id][1], tie_selections[value][1]
        )
        for value in observed_ids[1:]
    )
    if not ties_identical:
        raise RuntimeError("coverage-tied rules do not have identical A-F selections")

    selected_entry = max(
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
    selected_id = int(selected_entry["config"]["id"])
    expected_id = int(amendment["expected_boundary"]["expected_unique_id_after_tie_break"])
    if selected_id != expected_id:
        raise RuntimeError(f"tie break changed: {selected_id}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = args.output_dir / "AF_selected_index.npz"
    write_npz_atomic(
        selected_path,
        record_key=np.concatenate(record_parts),
        bank=np.concatenate(bank_parts),
        fixed_selected_index=tie_selections[selected_id][0],
        combined_selected_index=tie_selections[selected_id][1],
    )
    freeze = {
        "schema": "recon_cem_boundary_rule_v60_freeze",
        "status": "unique_boundary_rule_frozen_before_new_GH_manifests",
        "evidence_role": "A_F_spent_development_boundary_for_one_time_new_GH",
        "frontier": str(args.frontier),
        "frontier_sha256": file_sha256(args.frontier),
        "amendment": str(args.amendment),
        "amendment_sha256": file_sha256(args.amendment),
        "eligible_rule_count": len(eligible),
        "max_coverage_tie_ids": observed_ids,
        "tie_selected_indices_identical": ties_identical,
        "tie_break_order": [
            "higher dual_threshold",
            "lower harm_threshold",
            "higher benefit_threshold",
            "larger dominance_margin",
            "larger beta",
            "lower grid id",
        ],
        "selected_rule": selected_entry["config"],
        "selected_development_metrics": selected_entry["pooled"],
        "formula": {
            "benefit_lcb": "-0.5*(mean_delta_ATE+mean_delta_Objective)-beta*sqrt(std_ATE^2+std_Objective^2)",
            "safe": "support_valid AND benefit_lcb>=benefit_threshold AND p_dual>=dual_threshold AND p_harm<=harm_threshold",
            "energy": "-benefit_lcb + 0.25*p_harm - 0.10*p_dual",
            "selection": "best safe fixed; best safe over all three CEM archives; replace fixed only if E_cem + dominance_margin < E_fixed; otherwise fixed; if neither safe use incumbent",
        },
        "AF_selected_index": str(selected_path),
        "AF_selected_index_sha256": file_sha256(selected_path),
        "source_sha256": source_sha,
        "new_GH": {
            "names": ["bank_g_v60", "bank_h_v60"],
            "states_each": 44,
            "routes_each": 11,
            "outcomes_accessed": False,
            "selected_index_frozen": False,
        },
        "grid_expanded": False,
        "state_gate_used": False,
        "G_H_outcomes_accessed": False,
        "post_GH_tuning_allowed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    freeze_path = args.output_dir / "boundary_freeze.json"
    write_json_atomic(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
