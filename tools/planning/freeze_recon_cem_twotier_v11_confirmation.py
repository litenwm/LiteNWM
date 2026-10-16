#!/usr/bin/env python3
"""Freeze v11 base-only and base+CEM selections before confirmation outcomes."""

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
from tools.scoring.tune_recon_cem_combined_frozen_universal_v8 import (  # noqa: E402
    combined_prediction,
)
from tools.scoring.tune_recon_cem_combined_twotier_v11 import (  # noqa: E402
    select_twotier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--base-scores", type=Path, required=True)
    parser.add_argument("--cem-features", type=Path, required=True)
    parser.add_argument("--cem-scores", type=Path, required=True)
    parser.add_argument("--cem-freeze", type=Path, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    freeze_path = args.output_dir / "prelabel_freeze.json"
    selection_path = args.output_dir / "selected_actions.npz"
    if freeze_path.exists() or selection_path.exists():
        raise FileExistsError(args.output_dir)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError("v11 confirmation must remain H4-100")
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    development = json.loads(
        args.development_result.read_text(encoding="utf-8")
    )
    search = development["policy_search"]
    if (
        development.get("schema") != "recon_cem_combined_twotier_v11"
        or not search.get("strict_feasible")
        or not search.get("strict_safety_all_cohorts")
        or not search.get("quality_all_cohorts")
    ):
        raise RuntimeError("v11 development policy was not strictly feasible")
    policy = search["policy"]
    prediction = combined_prediction(
        args.base_scores, args.cem_scores, expected_record
    )
    false_mask = np.zeros((100, 33), dtype=bool)
    false_mask[:, :17] = True
    true_mask = np.ones((100, 33), dtype=bool)
    false_selected = select_twotier(
        prediction, false_mask, policy
    )
    true_selected = select_twotier(
        prediction, true_mask, policy
    )

    with np.load(args.base_features, allow_pickle=False) as payload:
        base_record = payload["record_key"].astype(str)
        base_candidates = payload["candidates"].astype(np.float32)
        base_actions = payload["actions_h4"].astype(np.float32)
    with np.load(args.cem_features, allow_pickle=False) as payload:
        cem_record = payload["record_key"].astype(str)
        cem_candidates = payload["candidates"].astype(np.float32)
        cem_actions = payload["actions_h4"].astype(np.float32)
    if not (
        np.array_equal(base_record, expected_record)
        and np.array_equal(cem_record, expected_record)
    ):
        raise RuntimeError("v11 confirmation feature identity mismatch")
    combined_candidates = np.concatenate(
        (base_candidates, cem_candidates[:, 1:]), axis=1
    )
    combined_actions = np.concatenate(
        (base_actions, cem_actions[:, 1:]), axis=1
    )
    row = np.arange(100)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_npz(
        selection_path,
        record_key=expected_record,
        trajectory=np.asarray(
            [str(row_value["trajectory_id"]) for row_value in rows]
        ),
        route_group=np.asarray(
            [str(row_value["route_group"]) for row_value in rows]
        ),
        cem_false_selected_index=false_selected,
        cem_true_selected_index=true_selected,
        cem_false_candidates_waypoint_units=combined_candidates[
            row, false_selected
        ],
        cem_true_candidates_waypoint_units=combined_candidates[
            row, true_selected
        ],
        cem_false_actions_h4=combined_actions[row, false_selected],
        cem_true_actions_h4=combined_actions[row, true_selected],
    )
    freeze = {
        "schema": "recon_cem_twotier_v11_confirmation_freeze",
        "status": "selections_frozen_before_confirmation_outcomes",
        "evidence_role": "new_route_disjoint_development_confirmation",
        "domain": "recon",
        "cem": True,
        "states": 100,
        "trajectories": len(
            {str(row_value["trajectory_id"]) for row_value in rows}
        ),
        "route_groups": len(
            {str(row_value["route_group"]) for row_value in rows}
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate_pool": str(args.candidate_pool),
        "candidate_pool_sha256": file_sha256(args.candidate_pool),
        "base_features": str(args.base_features),
        "base_features_sha256": file_sha256(args.base_features),
        "base_scores": str(args.base_scores),
        "base_scores_sha256": file_sha256(args.base_scores),
        "cem_features": str(args.cem_features),
        "cem_features_sha256": file_sha256(args.cem_features),
        "cem_scores": str(args.cem_scores),
        "cem_scores_sha256": file_sha256(args.cem_scores),
        "cem_freeze": str(args.cem_freeze),
        "cem_freeze_sha256": file_sha256(args.cem_freeze),
        "development_result": str(args.development_result),
        "development_result_sha256": file_sha256(
            args.development_result
        ),
        "policy": policy,
        "policy_source_strict_feasible_count": search[
            "strict_feasible_policies"
        ],
        "selected_actions": str(selection_path),
        "selected_actions_sha256": file_sha256(selection_path),
        "cem_false_promotions": int((false_selected != 0).sum()),
        "cem_true_promotions": int((true_selected != 0).sum()),
        "cem_true_selected_from_cem_pool": int(
            (true_selected > 16).sum()
        ),
        "online_causal": True,
        "future_route_windows_used": False,
        "outcomes_accessed": False,
        "labels_opened": False,
        "formal_outer_claim_allowed": False,
    }
    write_json(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
