from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from planning.proposal_tail_sidecar_v1 import ProposalTailSidecar, SidecarBindings


ROOT = Path(__file__).resolve().parents[2]


def test_live_sidecar_serializes_without_mutating_formal_arrays(tmp_path: Path) -> None:
    policy = {
        "z": 2.0,
        "margin_ate": 0.01,
        "margin_objective": 0.01,
        "dual_threshold": 0.5,
        "harm_threshold": 0.1,
    }
    rows = [
        {
            "route_group": "synthetic_spent_route",
            "bag_id": "synthetic_spent_bag",
            "trajectory_id": "synthetic_spent_trajectory",
            "window_uid": "synthetic_spent_window",
            "current_t": 4,
            "goal_t": 8,
            "history_timesteps": [1, 2, 3, 4],
        }
    ]
    manifest_path = tmp_path / "input.jsonl"
    manifest_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    bindings = SidecarBindings(
        manifest=manifest_path,
        preregistration=ROOT / "configs/proposal_tail_collection/scand_proposal_tail_collection_v1_preregistration.json",
        schema=ROOT / "configs/proposal_tail_collection/scand_proposal_tail_collection_v1.schema.json",
        support_checkpoint=manifest_path,
        model_result=manifest_path,
        runner_v66=manifest_path,
        runner_v70=manifest_path,
        sampler=manifest_path,
    )
    sidecar = ProposalTailSidecar(
        tmp_path / "sidecar",
        cohort_role="spent_smoke",
        cohort="synthetic_spent_smoke",
        rows=rows,
        bindings=bindings,
        policy=policy,
        support_threshold=4.0,
    )
    original_energy = None
    for source_round in (-1, 0, 1, 2):
        candidates = np.zeros((1, 17, 4, 2), dtype=np.float32)
        candidates[:, 1:, :, 0] = 0.01 * (source_round + 2)
        actions = np.zeros((1, 17, 4, 3), dtype=np.float32)
        support = np.zeros((1, 17), dtype=np.float32)
        guard = np.ones((1, 17), dtype=bool)
        energy = np.broadcast_to(np.arange(17, dtype=np.float32), (1, 17)).copy()
        energy[:, 0] = np.inf
        if source_round == -1:
            original_energy = energy.copy()
        prediction = {
            "mean": np.full((1, 17, 4), -0.03, dtype=np.float32),
            "std": np.full((1, 17, 4), 0.001, dtype=np.float32),
            "helpful_logit": np.zeros((1, 17), dtype=np.float32),
            "dual_logit": np.ones((1, 17), dtype=np.float32),
            "harm_logit": -np.ones((1, 17), dtype=np.float32),
            "rank_logit": np.zeros((1, 17), dtype=np.float32),
        }
        feature = {
            "action_features": np.zeros((1, 17, 242), dtype=np.float32),
            "static_features": np.zeros((1, 17, 210), dtype=np.float32),
            "raw_ltwm_features": np.zeros((1, 17, 112), dtype=np.float32),
            "scorer_v2_aux_features": np.zeros((1, 17, 10), dtype=np.float32),
            "ate_by_seed": np.zeros((3, 1, 17), dtype=np.float32),
            "rtp_by_seed": np.zeros((3, 1, 17), dtype=np.float32),
            "support_by_seed": np.ones((3, 1, 17), dtype=bool),
            "majority_support": np.ones((1, 17), dtype=bool),
        }
        if source_round >= 0:
            sidecar.begin_iteration(
                source_round,
                proposal_mean=np.zeros((1, 8), dtype=np.float32),
                proposal_covariance=np.eye(8, dtype=np.float32)[None],
                incumbent=candidates[:, 0],
                reference=candidates[:, 1:],
            )
            sidecar.capture_sampling_event(
                {
                    "event": "draw_batch",
                    "state_index": 0,
                    "draw_sequence": np.asarray([0]),
                    "raw_draw_unclipped": np.zeros((1, 8)),
                    "draw_clipped": np.zeros((1, 8), dtype=np.float32),
                    "clip_mask": np.zeros((1, 8), dtype=bool),
                    "actions_h4": np.zeros((1, 4, 3), dtype=np.float32),
                    "normalized_action_vector": np.zeros((1, 12), dtype=np.float32),
                    "physical_coordinate_margin": np.ones((1, 12), dtype=np.float32),
                    "physical_coordinate_mask": np.ones((1, 12), dtype=bool),
                    "physical_guard": np.ones(1, dtype=bool),
                    "support_score": np.zeros(1, dtype=np.float32),
                    "support_margin": np.ones(1, dtype=np.float32),
                    "support_mask": np.ones(1, dtype=bool),
                    "accepted_mask": np.ones(1, dtype=bool),
                    "final_candidate_index": np.asarray([1]),
                }
            )
            sidecar.capture_sampling_event(
                {
                    "event": "state_complete",
                    "state_index": 0,
                    "attempted_draws": 1,
                    "accepted_draws_before_truncation": 16,
                    "fallback_candidates": 0,
                    "dedup_applied": False,
                }
            )
        sidecar.record_pool(
            source_round,
            candidates=candidates,
            actions_h4=actions,
            support_score=support,
            physical_guard=guard,
            support_mask=guard,
            valid=guard,
            safe=guard,
            energy=energy,
            prediction=prediction,
            p_dual=np.full((1, 17), 0.9, dtype=np.float32),
            p_harm=np.full((1, 17), 0.05, dtype=np.float32),
            feature_data=feature,
        )
    assert np.array_equal(original_energy, np.where(np.arange(17)[None] == 0, np.inf, np.arange(17)[None]).astype(np.float32))
    selected_path = tmp_path / "selected_actions.npz"
    np.savez(
        selected_path,
        record_key=np.asarray(["synthetic_spent_trajectory:4"]),
        final_cem_selected_index=np.asarray([1]),
        final_cem_energy=np.arange(17, dtype=np.float32)[None],
        archive_energy=np.asarray([1.0], dtype=np.float32),
        archive_source_round=np.asarray([2]),
        archive_source_candidate=np.asarray([1]),
    )
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}\n", encoding="utf-8")
    selected_before = selected_path.read_bytes()
    manifest_path = sidecar.finalize(
        formal_selected_path=selected_path,
        formal_freeze_path=freeze_path,
    )
    assert selected_path.read_bytes() == selected_before
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["formal_behavior_influence"] is False
    table = np.load(tmp_path / "sidecar/candidate_table.npz", allow_pickle=False)
    assert table["candidates_feature_waypoint_units"].shape == (1, 65, 4, 2)
    fixed = np.load(
        tmp_path / "sidecar/fixed_pool/candidates_and_scores.npz",
        allow_pickle=False,
    )
    assert fixed["raw_ltwm_features"].shape == (1, 17, 112)
    assert fixed["scorer_v2_aux_features"].shape == (1, 17, 10)
