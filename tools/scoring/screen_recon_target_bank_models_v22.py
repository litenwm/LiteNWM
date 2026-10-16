#!/usr/bin/env python3
"""Six-bank whole-cohort OOF screen after expanding target development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.screen_recon_target_bank_models_v21 import top40
from tools.scoring.train_recon_cem_latent_base_risk_v16 import (
    TailConfig, compact_base_feature, predict_tail,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz
from tools.scoring.train_recon_target_bank_base_v20 import train_group_dro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-3)
    parser.add_argument("--dro-eta", type=float, default=0.03)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def feature_path(bank: Path, letter: str) -> Path:
    values = sorted((bank / "features").glob(f"recon_target_bank_{letter}_v*_features.npz"))
    if len(values) != 1:
        raise RuntimeError(f"bank_{letter} feature ambiguity: {values}")
    return values[0]


def load_bank(root: Path, letter: str) -> dict:
    bank = root / f"bank_{letter}"
    feature = feature_path(bank, letter)
    score = bank / "universal_scores.npz"
    rollout_path = bank / "spatial_rollout_v17.npz"
    source = bank / "source_labeled.npz"
    compact, valid = compact_base_feature(
        feature, score, np.empty((100, 0), dtype=np.float32)
    )
    with np.load(source, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        targets = payload["targets"].astype(np.float32)
        absolute = payload["absolute_metrics"].astype(np.float32)
    with np.load(feature, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
    with np.load(score, allow_pickle=False) as payload:
        score_record = payload["record_key"].astype(str)
    with np.load(rollout_path, allow_pickle=False) as payload:
        rollout_record = payload["record_key"].astype(str)
        rollout = payload["spatial_rollout_features"].astype(np.float32)
        metadata = json.loads(str(payload["metadata"]))
    if not all(np.array_equal(record, value) for value in (
        feature_record, score_record, rollout_record,
    )):
        raise RuntimeError(f"bank_{letter} record identity mismatch")
    if rollout.shape != (100, 17, 258) or not np.isfinite(rollout).all():
        raise RuntimeError(f"bank_{letter} rollout contract failed")
    if metadata.get("outcomes_accessed") or metadata.get("labels_in_output"):
        raise RuntimeError(f"bank_{letter} rollout leaked labels")
    full = np.concatenate((compact, rollout[:, 1:]), axis=2)
    delta = targets[:, 1:, :4]
    dual = ((delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)).astype(np.float32)
    harm = ((delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)).astype(np.float32)
    return {
        "name": f"bank_{letter}", "feature": full, "valid": valid,
        "delta": delta, "absolute": absolute, "dual": dual, "harm": harm,
        "record": record,
        "route": np.asarray([f"bank_{letter}::{value}" for value in route]),
        "trajectory": trajectory,
        "sha": {
            "feature": file_sha256(feature), "score": file_sha256(score),
            "rollout": file_sha256(rollout_path), "source": file_sha256(source),
        },
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    banks = [load_bank(args.root, value) for value in "abcdef"]
    record_sets = [set(bank["record"]) for bank in banks]
    route_sets = [set(bank["route"]) for bank in banks]
    trajectory_sets = [set(bank["trajectory"]) for bank in banks]
    for left in range(len(banks)):
        for right in range(left + 1, len(banks)):
            if (
                record_sets[left] & record_sets[right]
                or route_sets[left] & route_sets[right]
                or trajectory_sets[left] & trajectory_sets[right]
            ):
                raise RuntimeError("six-bank identity leakage")
    reserve = args.root / "final_confirmation_reserve" / "source_label_free.npz"
    with np.load(reserve, allow_pickle=False) as payload:
        reserve_record = set(payload["record_key"].astype(str))
        reserve_route = set(payload["route_group"].astype(str))
    if set.union(*record_sets) & reserve_record:
        raise RuntimeError("six-bank/reserve record leakage")
    if any({value.split("::", 1)[-1] for value in routes} & reserve_route for routes in route_sets):
        raise RuntimeError("six-bank/reserve route leakage")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full = np.concatenate([bank["feature"] for bank in banks])
    valid = np.concatenate([bank["valid"] for bank in banks])
    dual = np.concatenate([bank["dual"] for bank in banks])
    harm = np.concatenate([bank["harm"] for bank in banks])
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    group = np.concatenate([bank["route"] for bank in banks])
    cohort = np.concatenate([
        np.full(len(bank["record"]), bank["name"], dtype="<U16") for bank in banks
    ])
    variants = {
        "universal_support": full[..., 474:486],
        "motion_universal_support": full[..., 434:486],
    }
    reports = {}
    for variant_number, (name, feature) in enumerate(variants.items()):
        config = TailConfig(
            input_dim=feature.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout
        )
        p_dual = np.zeros((len(feature), 16), dtype=np.float32)
        p_harm = np.zeros_like(p_dual)
        p_dual_std = np.zeros_like(p_dual)
        p_harm_std = np.zeros_like(p_dual)
        for fold_number, heldout in enumerate(np.unique(cohort)):
            train_index = np.flatnonzero(cohort != heldout)
            heldout_index = np.flatnonzero(cohort == heldout)
            member_dual = []
            member_harm = []
            for member in range(args.ensemble_members):
                seed = args.seed + variant_number * 1000000 + fold_number * 100000 + member * 1009
                model, mean, scale, _ = train_group_dro(
                    feature=feature, dual=dual, harm=harm, group=group,
                    train_index=train_index, config=config, args=args, seed=seed,
                )
                values = predict_tail(model, mean, scale, feature[heldout_index], args.device)
                member_dual.append(values[0])
                member_harm.append(values[1])
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            dual_stack = np.stack(member_dual)
            harm_stack = np.stack(member_harm)
            p_dual[heldout_index] = dual_stack.mean(0)
            p_harm[heldout_index] = harm_stack.mean(0)
            p_dual_std[heldout_index] = dual_stack.std(0)
            p_harm_std[heldout_index] = harm_stack.std(0)
        per_bank = {}
        for bank_name in np.unique(cohort):
            mask = cohort == bank_name
            per_bank[bank_name] = top40(
                p_dual[mask], p_harm[mask], p_dual_std[mask], p_harm_std[mask],
                valid[mask], delta[mask], absolute[mask],
            )
        prediction = args.output_dir / f"{name}_oof.npz"
        write_npz(
            prediction, record_key=record, cohort=cohort, route_group=group,
            p_dual=p_dual, p_harm=p_harm, p_dual_std=p_dual_std,
            p_harm_std=p_harm_std, valid=valid,
        )
        reports[name] = {
            "feature_dim": int(feature.shape[-1]),
            "noncausal_top40_diagnostic": per_bank,
            "all_banks_top40_safety_pass": all(
                value["metrics"]["dual_precision"] >= 0.90
                and value["metrics"]["dual_wilson_lower_95"] >= 0.80
                and value["metrics"]["harm_rate"] <= 0.10
                for value in per_bank.values()
            ),
            "predictions": str(prediction), "predictions_sha256": file_sha256(prediction),
        }
    report = {
        "schema": "recon_target_bank_models_v22_screen",
        "states": 600, "routes": 150, "banks": 6,
        "evidence_role": "expanded_target_development_whole_bank_oof_screen",
        "warning": "per-bank top40 uses whole heldout-bank ordering for separability only",
        "identity_audit": {
            "all_six_banks_pairwise_record_route_trajectory_disjoint": True,
            "all_six_banks_reserve_record_route_disjoint": True,
            "reserve_sealed": not (args.root / "final_confirmation_reserve" / "source_labeled.npz").exists(),
        },
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "dro_eta": args.dro_eta,
            "ensemble_members": args.ensemble_members,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout, "seed": args.seed,
        },
        "variants": reports,
        "bank_artifact_sha": {bank["name"]: bank["sha"] for bank in banks},
        "reserve_source_sha256": file_sha256(reserve),
        "final_reserve_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
