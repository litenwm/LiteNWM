#!/usr/bin/env python3
"""Screen low-capacity feature families with whole-bank cross-fit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_cem_latent_base_risk_v16 import TailConfig, predict_tail
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz
from tools.scoring.train_recon_target_bank_base_v20 import load_bank, train_group_dro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=5.0e-3)
    parser.add_argument("--dro-eta", type=float, default=0.05)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def variants(feature: np.ndarray) -> dict[str, np.ndarray]:
    # compact_base_feature v16 layout: static210, raw112, raw-delta112,
    # action12, action-delta12, path8, path-delta8, universal11, support1,
    # followed here by spatial rollout258.
    if feature.shape[-1] != 744:
        raise RuntimeError(f"unexpected v20 feature layout: {feature.shape}")
    return {
        "universal_support": feature[..., 474:486],
        "motion_universal_support": feature[..., 434:486],
        "universal_support_spatial": np.concatenate(
            (feature[..., 474:486], feature[..., 486:744]), axis=2
        ),
        "raw_motion_universal": feature[..., 210:486],
    }


def top40(
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    p_dual_std: np.ndarray,
    p_harm_std: np.ndarray,
    valid: np.ndarray,
    delta: np.ndarray,
    absolute: np.ndarray,
) -> dict:
    best = None
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        for weight in (0.5, 1.0, 2.0, 4.0):
            score = np.where(
                valid,
                p_dual - z * p_dual_std - weight * (p_harm + z * p_harm_std),
                -np.inf,
            )
            candidate = score.argmax(1)
            row = np.arange(len(candidate))
            order = np.argsort(-score[row, candidate], kind="stable")[:40]
            selected = np.zeros(len(candidate), dtype=np.int64)
            selected[order] = candidate[order] + 1
            metrics = method_summary(selected, delta, absolute)
            item = {"z": z, "risk_weight": weight, "metrics": metrics}
            key = (
                metrics["dual_precision"] or 0.0,
                -(metrics["harm_rate"] if metrics["harm_rate"] is not None else 1.0),
            )
            if best is None or key > best[0]:
                best = (key, item)
    assert best is not None
    return best[1]


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    banks = [load_bank(args.root, value) for value in ("a", "b", "c")]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full_feature = np.concatenate([bank["feature"] for bank in banks])
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
    feature_values = variants(full_feature)
    variant_reports = {}
    for variant_number, (name, feature) in enumerate(feature_values.items()):
        config = TailConfig(
            input_dim=feature.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout
        )
        p_dual = np.zeros((len(feature), 16), dtype=np.float32)
        p_harm = np.zeros_like(p_dual)
        p_dual_std = np.zeros_like(p_dual)
        p_harm_std = np.zeros_like(p_dual)
        fit_stats = []
        for fold_number, heldout_name in enumerate(np.unique(cohort)):
            train_index = np.flatnonzero(cohort != heldout_name)
            heldout_index = np.flatnonzero(cohort == heldout_name)
            members_dual = []
            members_harm = []
            for member in range(args.ensemble_members):
                seed = (
                    args.seed + variant_number * 1000000
                    + fold_number * 100000 + member * 1009
                )
                model, mean, scale, stats = train_group_dro(
                    feature=feature, dual=dual, harm=harm, group=group,
                    train_index=train_index, config=config, args=args, seed=seed,
                )
                value_dual, value_harm = predict_tail(
                    model, mean, scale, feature[heldout_index], args.device
                )
                members_dual.append(value_dual)
                members_harm.append(value_harm)
                fit_stats.append({"heldout": heldout_name, "seed": seed, **stats})
                del model
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            stack_dual = np.stack(members_dual)
            stack_harm = np.stack(members_harm)
            p_dual[heldout_index] = stack_dual.mean(0)
            p_harm[heldout_index] = stack_harm.mean(0)
            p_dual_std[heldout_index] = stack_dual.std(0)
            p_harm_std[heldout_index] = stack_harm.std(0)
        per_bank = {}
        for bank_name in np.unique(cohort):
            mask = cohort == bank_name
            per_bank[bank_name] = top40(
                p_dual[mask], p_harm[mask], p_dual_std[mask], p_harm_std[mask],
                valid[mask], delta[mask], absolute[mask],
            )
        prediction_path = args.output_dir / f"{name}_oof.npz"
        write_npz(
            prediction_path, record_key=record, cohort=cohort, route_group=group,
            p_dual=p_dual, p_harm=p_harm, p_dual_std=p_dual_std,
            p_harm_std=p_harm_std, valid=valid,
        )
        variant_reports[name] = {
            "feature_dim": int(feature.shape[-1]),
            "noncausal_top40_diagnostic": per_bank,
            "all_banks_top40_safety_pass": all(
                value["metrics"]["dual_precision"] >= 0.90
                and value["metrics"]["dual_wilson_lower_95"] >= 0.80
                and value["metrics"]["harm_rate"] <= 0.10
                for value in per_bank.values()
            ),
            "predictions": str(prediction_path),
            "predictions_sha256": file_sha256(prediction_path),
            "fit_stats": fit_stats,
        }
    report = {
        "schema": "recon_target_bank_models_v21_screen",
        "evidence_role": "development_only_whole_bank_oof_separability_screen",
        "warning": "per-bank top40 is a noncausal upper-bound diagnostic, not a deployable gate",
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "dro_eta": args.dro_eta,
            "ensemble_members": args.ensemble_members,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            "seed": args.seed,
        },
        "variants": variant_reports,
        "base_result": str(args.base_result),
        "base_result_sha256": file_sha256(args.base_result),
        "final_reserve_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
