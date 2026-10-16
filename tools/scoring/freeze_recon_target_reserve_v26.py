#!/usr/bin/env python3
"""Freeze v26 selections on the still-unlabeled final RECON reserve."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--reserve-root", type=Path, required=True)
    parser.add_argument("--model-freeze", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "freeze.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if (args.reserve_root / "source_labeled.npz").exists():
        raise RuntimeError("reserve labels were already opened before selection freeze")
    freeze = json.loads(args.model_freeze.read_text(encoding="utf-8"))
    if freeze.get("final_reserve_outcomes_accessed"):
        raise RuntimeError("model freeze already references reserve outcomes")
    feature_path = (
        args.reserve_root / "features" / "recon_target_final_reserve_v26_features.npz"
    )
    rollout_path = args.reserve_root / "spatial_rollout_v17.npz"
    score_path = args.reserve_root / "universal_scores.npz"
    source_path = args.reserve_root / "source_label_free.npz"
    compact, valid = compact_base_feature(
        feature_path, score_path, np.empty((100, 0), dtype=np.float32)
    )
    with np.load(rollout_path, allow_pickle=False) as payload:
        rollout = payload["spatial_rollout_features"].astype(np.float32)
        rollout_record = payload["record_key"].astype(str)
        rollout_metadata = json.loads(str(payload["metadata"]))
    with np.load(source_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
    with np.load(feature_path, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
        feature_metadata = json.loads(str(payload["metadata"]))
    with np.load(score_path, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
        score = {
            "std": payload["std"][:, 1:].astype(np.float32),
            "dual": payload["dual_probability"][:, 1:].astype(np.float32),
            "harm": payload["harm_probability"][:, 1:].astype(np.float32),
        }
    if not all(np.array_equal(record, value) for value in (
        rollout_record, feature_record, score_record,
    )):
        raise RuntimeError("reserve feature identity mismatch")
    if (
        feature_metadata.get("labels_in_output")
        or feature_metadata.get("labels_accessed")
        or rollout_metadata.get("labels_in_output")
        or rollout_metadata.get("outcomes_accessed")
    ):
        raise RuntimeError("reserve feature export accessed labels")
    full = np.concatenate((compact, rollout[:, 1:]), axis=2)
    challenger = frozen_challenger(score, valid)
    chosen = full[np.arange(100), challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
    }
    probabilities = {}
    checkpoint_audit = {}
    for name, feature in variants.items():
        dual_values = []
        harm_values = []
        checkpoint_audit[name] = []
        for value in freeze["checkpoints"][name]:
            path = Path(value["path"])
            if file_sha256(path) != value["sha256"]:
                raise RuntimeError("v26 checkpoint SHA mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            config = payload["config"]
            model = StateRisk(
                config["input_dim"], config["hidden_dim"], config["dropout"]
            ).to(args.device)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            dual, harm = predict(
                model,
                np.asarray(payload["feature_mean"], dtype=np.float32),
                np.asarray(payload["feature_scale"], dtype=np.float32),
                feature,
                args.device,
            )
            dual_values.append(dual)
            harm_values.append(harm)
            checkpoint_audit[name].append({
                "path": str(path), "sha256": value["sha256"],
            })
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
        probabilities[name] = {
            "dual": np.stack(dual_values).mean(0),
            "harm": np.stack(harm_values).mean(0),
        }
    policy = freeze["policy"]
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
    selected = np.where(
        eligible & (risk >= float(policy["threshold"])), challenger + 1, 0
    ).astype(np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = args.output_dir / "selected_index.npz"
    write_npz(
        selected_path,
        record_key=record,
        challenger_index=np.where(eligible, challenger + 1, 0),
        eligible=eligible,
        p_dual=p_dual,
        p_harm=p_harm,
        risk=risk,
        selected_index=selected,
    )
    report = {
        "schema": "recon_target_reserve_v26_selection_freeze",
        "status": "selected_index_frozen_before_outcomes",
        "states": 100,
        "promotions_label_blind": int((selected != 0).sum()),
        "coverage_label_blind": float((selected != 0).mean()),
        "support_eligible_states": int(eligible.sum()),
        "policy": policy,
        "model_freeze": str(args.model_freeze),
        "model_freeze_sha256": file_sha256(args.model_freeze),
        "checkpoints": checkpoint_audit,
        "artifacts": {
            "source_label_free_sha256": file_sha256(source_path),
            "features_sha256": file_sha256(feature_path),
            "spatial_rollout_sha256": file_sha256(rollout_path),
            "universal_scores_sha256": file_sha256(score_path),
            "candidate_k17_sha256": file_sha256(args.reserve_root / "k17.npz"),
            "manifest_sha256": file_sha256(args.reserve_root / "manifest.windows.jsonl"),
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
