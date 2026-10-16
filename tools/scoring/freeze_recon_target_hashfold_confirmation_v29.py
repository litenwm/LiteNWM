#!/usr/bin/env python3
"""Freeze a v29 route-hash fixed-pool selection before confirmation outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_state_gate_v24 import StateRisk, predict
from tools.scoring.train_recon_cem_latent_base_risk_v16 import compact_base_feature
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--policy-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def route_fold(route: str, fold_order: list[str]) -> int:
    digest = hashlib.sha256(route.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % len(fold_order)


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "freeze.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    for forbidden in ("source_labeled.npz", "outcomes.npz", "targets.npz"):
        if (args.confirmation_root / forbidden).exists():
            raise RuntimeError(f"confirmation outcome artifact exists before freeze: {forbidden}")
    policy_freeze = json.loads(args.policy_freeze.read_text(encoding="utf-8"))
    if policy_freeze["status"] != "deployment_policy_frozen_before_new_confirmation":
        raise RuntimeError("unexpected v29 policy status")
    if policy_freeze.get("labels_opened") or policy_freeze.get("outcomes_accessed"):
        raise RuntimeError("policy freeze references confirmation outcomes")

    feature_path = args.confirmation_root / "features" / args.feature_name
    rollout_path = args.confirmation_root / "spatial_rollout_v17.npz"
    score_path = args.confirmation_root / "universal_scores.npz"
    source_path = args.confirmation_root / "source_label_free.npz"
    compact, valid = compact_base_feature(
        feature_path, score_path, np.empty((100, 0), dtype=np.float32)
    )
    with np.load(source_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
    with np.load(feature_path, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
        feature_metadata = json.loads(str(payload["metadata"]))
    with np.load(rollout_path, allow_pickle=False) as payload:
        rollout = payload["spatial_rollout_features"].astype(np.float32)
        rollout_record = payload["record_key"].astype(str)
        rollout_metadata = json.loads(str(payload["metadata"]))
    with np.load(score_path, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        score = {
            "std": payload["std"][:, 1:].astype(np.float32),
            "dual": payload["dual_probability"][:, 1:].astype(np.float32),
            "harm": payload["harm_probability"][:, 1:].astype(np.float32),
        }
    if len(record) != 100:
        raise RuntimeError(f"expected 100 confirmation states, got {len(record)}")
    if not all(np.array_equal(record, value) for value in (
        feature_record, rollout_record, score_record,
    )):
        raise RuntimeError("confirmation feature identity mismatch")
    if (
        feature_metadata.get("labels_in_output")
        or feature_metadata.get("labels_accessed")
        or rollout_metadata.get("labels_in_output")
        or rollout_metadata.get("outcomes_accessed")
    ):
        raise RuntimeError("confirmation feature export accessed outcomes")

    full = np.concatenate((compact, rollout[:, 1:]), axis=2)
    challenger = frozen_challenger(score, valid)
    chosen = full[np.arange(len(record)), challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
    }
    fold_order = list(policy_freeze["router"]["fold_order"])
    assigned = np.asarray([route_fold(value, fold_order) for value in route], dtype=np.int64)
    probabilities: dict[str, dict[str, np.ndarray]] = {}
    checkpoint_audit = {}
    for name, feature in variants.items():
        p_dual = np.zeros(len(record), dtype=np.float32)
        p_harm = np.zeros(len(record), dtype=np.float32)
        checkpoint_audit[name] = {}
        for fold_index, fold_name in enumerate(fold_order):
            index = np.flatnonzero(assigned == fold_index)
            members = policy_freeze["checkpoints"][name][fold_name]
            member_dual = []
            member_harm = []
            checkpoint_audit[name][fold_name] = []
            for member in members:
                path = Path(member["path"])
                if file_sha256(path) != member["sha256"]:
                    raise RuntimeError(f"v29 checkpoint SHA mismatch: {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model = StateRisk(
                    int(payload["input_dim"]), int(payload["hidden_dim"]),
                    float(payload["dropout"]),
                ).to(args.device)
                model.load_state_dict(payload["model"], strict=True)
                model.eval()
                dual, harm = predict(
                    model,
                    np.asarray(payload["feature_mean"], dtype=np.float32),
                    np.asarray(payload["feature_scale"], dtype=np.float32),
                    feature[index], args.device,
                )
                member_dual.append(dual)
                member_harm.append(harm)
                checkpoint_audit[name][fold_name].append({
                    "path": str(path), "sha256": member["sha256"],
                })
                del model
            if len(index):
                p_dual[index] = np.stack(member_dual).mean(0)
                p_harm[index] = np.stack(member_harm).mean(0)
        probabilities[name] = {"dual": p_dual, "harm": p_harm}

    policy = policy_freeze["policy"]
    blend = float(policy["blend"])
    p_dual = (
        blend * probabilities["state_universal_support"]["dual"]
        + (1.0 - blend) * probabilities["state_motion_universal_support"]["dual"]
    )
    p_harm = (
        blend * probabilities["state_universal_support"]["harm"]
        + (1.0 - blend) * probabilities["state_motion_universal_support"]["harm"]
    )
    risk = p_dual - float(policy["harm_weight"]) * p_harm
    eligible = challenger >= 0
    comparison = risk > float(policy["threshold"]) if policy.get("strict") else risk >= float(policy["threshold"])
    selected = np.where(eligible & comparison, challenger + 1, 0).astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path,
        record_key=record,
        route_group=route,
        assigned_fold_index=assigned,
        assigned_fold_name=np.asarray([fold_order[value] for value in assigned]),
        challenger_index=np.where(eligible, challenger + 1, 0),
        eligible=eligible,
        p_dual=p_dual,
        p_harm=p_harm,
        risk=risk,
        selected_index=selected,
    )
    unique_fold, fold_count = np.unique(assigned, return_counts=True)
    report = {
        "schema": "recon_target_hashfold_confirmation_v29_selection_freeze",
        "status": "selected_index_frozen_before_confirmation_outcomes",
        "states": len(record),
        "routes": int(len(np.unique(route))),
        "promotions_label_blind": int((selected != 0).sum()),
        "coverage_label_blind": float((selected != 0).mean()),
        "support_eligible_states": int(eligible.sum()),
        "fold_state_counts": {
            fold_order[int(key)]: int(value) for key, value in zip(unique_fold, fold_count)
        },
        "router": policy_freeze["router"],
        "policy": policy,
        "policy_freeze": str(args.policy_freeze),
        "policy_freeze_sha256": file_sha256(args.policy_freeze),
        "checkpoints": checkpoint_audit,
        "artifacts": {
            "manifest_sha256": file_sha256(args.confirmation_root / "manifest.windows.jsonl"),
            "candidate_k17_sha256": file_sha256(args.confirmation_root / "k17.npz"),
            "source_label_free_sha256": file_sha256(source_path),
            "features_sha256": file_sha256(feature_path),
            "spatial_rollout_sha256": file_sha256(rollout_path),
            "universal_scores_sha256": file_sha256(score_path),
            "selected_index_sha256": file_sha256(selected_path),
        },
        "selected_index": str(selected_path),
        "online_causal": True,
        "future_route_quota": False,
        "labels_opened": False,
        "outcomes_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
