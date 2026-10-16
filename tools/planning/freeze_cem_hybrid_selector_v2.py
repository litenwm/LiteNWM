#!/usr/bin/env python3
"""Freeze the spent-dev-tuned hybrid CEM selector on an outcome-sealed cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument("--cem-freeze", type=Path, required=True)
    parser.add_argument("--selected-actions", type=Path, required=True)
    parser.add_argument("--frozen-scores", type=Path, required=True)
    parser.add_argument("--selector-tuning", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--confirmation-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cem = json.loads(args.cem_freeze.read_text(encoding="utf-8"))
    tuning = json.loads(args.selector_tuning.read_text(encoding="utf-8"))
    confirmation = json.loads(
        args.confirmation_audit.read_text(encoding="utf-8")
    )
    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if (
        cem.get("status")
        != "cem_true_actions_and_selections_frozen_before_outcomes"
        or cem.get("domain") != args.domain
        or cem.get("labels_opened")
        or cem.get("future_pose_outcomes_opened")
        or not cem.get("cem")
    ):
        raise RuntimeError("CEM confirmation actions are not outcome-sealed")
    if (
        tuning.get("evidence_role")
        != "in_sample_development_only_not_outer_evidence"
        or tuning.get("domain") != args.domain
        or not tuning.get("hyperparameters_selected_with_outcomes")
    ):
        raise RuntimeError("selector tuning provenance is invalid")
    if (
        confirmation.get("status") != "label_sealed_confirmation_frozen"
        or confirmation.get("domain") != args.domain
        or confirmation.get("states") != 100
        or any(confirmation["development_overlap"].values())
        or confirmation.get("labels_opened")
        or confirmation.get("outcomes_opened")
    ):
        raise RuntimeError("confirmation subset is not strictly sealed/disjoint")
    if file_sha256(args.selected_actions) != cem["selected_actions_sha256"]:
        raise RuntimeError("CEM selected actions changed")
    with np.load(args.selected_actions, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        baseline = payload[
            "universal_d_cem_false_selected_index"
        ].astype(np.int64)
    with np.load(args.frozen_scores, allow_pickle=False) as payload:
        energy = payload["energy"].astype(np.float32)
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
        p_dual = payload["dual_probability"].astype(np.float32)
        p_harm = payload["harm_probability"].astype(np.float32)
        valid = payload["valid"].astype(bool)
    if energy.shape != (100, 17):
        raise RuntimeError("hybrid confirmation score shape changed")

    policy = training["arms"]["D"]["calibration"]["frontier"]["domains"][
        args.domain
    ]["policy"]
    parameters = tuning["best"]["parameters"]
    row = np.arange(100)
    candidate = np.argmin(energy, axis=1).astype(np.int64)
    exists = np.isfinite(energy[row, candidate]) & (candidate != 0)
    z = float(policy["z"])
    ucb_ate = mean[row, candidate, 0] + z * std[row, candidate, 0]
    ucb_objective = mean[row, candidate, 3] + z * std[row, candidate, 3]
    allow = (
        exists
        & valid[row, candidate]
        & (
            p_dual[row, candidate]
            >= min(
                0.999,
                float(policy["dual_threshold"])
                + float(parameters["dual_threshold_offset"]),
            )
        )
        & (
            p_harm[row, candidate]
            <= float(policy["harm_threshold"])
            * float(parameters["harm_threshold_scale"])
        )
        & (
            ucb_ate
            < -(
                float(policy["margin_ate"])
                + float(parameters["ate_margin_extra_m"])
            )
        )
        & (
            ucb_objective
            < -(
                float(policy["margin_objective"])
                + float(parameters["objective_margin_extra_m"])
            )
        )
        & (
            energy[row, candidate]
            <= float(parameters["conservative_energy_cap"])
        )
    )
    combined = baseline.copy()
    combined[allow] = 16 + candidate[allow]
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path,
        record_key=record_key,
        universal_d_cem_false_selected_index=baseline,
        cem_hybrid_v2_selected_combined_index=combined,
        cem_candidate_local_index=candidate,
        cem_replacement_allowed=allow,
    )
    freeze_path = args.output_dir / "prelabel_freeze.json"
    freeze = {
        "schema": "cem_hybrid_selector_v2_confirmation_freeze",
        "status": "confirmation_selector_and_actions_frozen_before_outcomes",
        "evidence_role": "route_trajectory_disjoint_cem_cal_confirmation",
        "domain": args.domain,
        "states": 100,
        "cem": True,
        "cem_budget": cem["config"],
        "selector_parameters": parameters,
        "selection_rule": tuning["selection_rule"],
        "fixed_pool_promotion_never_removed": True,
        "hybrid_changed_windows": int((combined != baseline).sum()),
        "hybrid_promotions": int((combined != 0).sum()),
        "cem_freeze": str(args.cem_freeze),
        "cem_freeze_sha256": file_sha256(args.cem_freeze),
        "selected_actions": str(args.selected_actions),
        "selected_actions_sha256": file_sha256(args.selected_actions),
        "frozen_scores": str(args.frozen_scores),
        "frozen_scores_sha256": file_sha256(args.frozen_scores),
        "selector_tuning": str(args.selector_tuning),
        "selector_tuning_sha256": file_sha256(args.selector_tuning),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "confirmation_audit": str(args.confirmation_audit),
        "confirmation_audit_sha256": file_sha256(args.confirmation_audit),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "outcomes_used_by_selector_on_confirmation": False,
        "formal_outer_claim_allowed": False,
    }
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
