#!/usr/bin/env python3
"""Strongly regularized linear state gates with six-bank LOBO predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.evaluate_recon_frozen_universal_gate_v23 import load_score
from tools.scoring.run_screen_recon_target_state_gate_v24 import frozen_challenger
from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_predict(
    feature: np.ndarray, dual: np.ndarray, harm: np.ndarray,
    train_index: np.ndarray, predict_index: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    set_seed(seed)
    mean = feature[train_index].mean(0).astype(np.float32)
    scale = np.maximum(feature[train_index].std(0), 1.0e-4).astype(np.float32)
    x = torch.from_numpy(((feature[train_index] - mean) / scale).astype(np.float32)).to(args.device)
    y_dual = torch.from_numpy(dual[train_index]).to(args.device)
    y_harm = torch.from_numpy(harm[train_index]).to(args.device)
    model = nn.Linear(feature.shape[1], 2).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    for _ in range(args.epochs):
        logits = model(x)
        dual_loss = F.binary_cross_entropy_with_logits(
            logits[:, 0], y_dual, reduction="none"
        ) * torch.where(
            y_dual > 0.5, torch.ones_like(y_dual), torch.full_like(y_dual, 4.0)
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1], y_harm, reduction="none"
        ) * torch.where(
            y_harm > 0.5, torch.full_like(y_harm, 5.0), torch.ones_like(y_harm)
        )
        loss = dual_loss.mean() + 1.5 * harm_loss.mean() + 0.01 * logits.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    value = torch.from_numpy(((feature[predict_index] - mean) / scale).astype(np.float32)).to(args.device)
    with torch.inference_mode():
        probability = torch.sigmoid(model(value)).cpu().numpy()
    return probability[:, 0], probability[:, 1]


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    banks = [load_bank(args.root, value) for value in "abcdef"]
    scores = [load_score(args.root, value) for value in "abcdef"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full = np.concatenate([bank["feature"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
    group = np.concatenate([bank["route"] for bank in banks])
    challenger_values = []
    dual_values = []
    harm_values = []
    for bank, score in zip(banks, scores):
        challenger = frozen_challenger(score, bank["valid"])
        row = np.arange(100)
        challenger_values.append(challenger)
        dual_values.append(bank["dual"][row, challenger])
        harm_values.append(bank["harm"][row, challenger])
    challenger = np.concatenate(challenger_values)
    dual = np.concatenate(dual_values).astype(np.float32)
    harm = np.concatenate(harm_values).astype(np.float32)
    chosen = full[np.arange(600), challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
    }
    reports = {}
    for variant_number, (name, feature) in enumerate(variants.items()):
        p_dual = np.zeros(600, dtype=np.float32)
        p_harm = np.zeros(600, dtype=np.float32)
        p_dual_std = np.zeros(600, dtype=np.float32)
        p_harm_std = np.zeros(600, dtype=np.float32)
        for fold_number, heldout in enumerate(np.unique(cohort)):
            train_index = np.flatnonzero(cohort != heldout)
            predict_index = np.flatnonzero(cohort == heldout)
            member_dual = []
            member_harm = []
            for member in range(args.ensemble_members):
                seed = args.seed + variant_number * 1000000 + fold_number * 100000 + member * 1009
                values = fit_predict(
                    feature, dual, harm, train_index, predict_index, args, seed
                )
                member_dual.append(values[0])
                member_harm.append(values[1])
            dual_stack = np.stack(member_dual)
            harm_stack = np.stack(member_harm)
            p_dual[predict_index] = dual_stack.mean(0)
            p_harm[predict_index] = harm_stack.mean(0)
            p_dual_std[predict_index] = dual_stack.std(0)
            p_harm_std[predict_index] = harm_stack.std(0)
        path = args.output_dir / f"{name}_oof.npz"
        write_npz(
            path, record_key=record, cohort=cohort, route_group=group,
            challenger_index=challenger + 1,
            p_dual=p_dual, p_harm=p_harm,
            p_dual_std=p_dual_std, p_harm_std=p_harm_std,
        )
        reports[name] = {"path": str(path), "sha256": file_sha256(path)}
    report = {
        "schema": "recon_target_state_gate_linear_v28",
        "status": "linear_lobo_predictions_frozen_for_fixed_threshold_calibration",
        "states": 600, "routes": 150, "banks": 6,
        "model": "linear_logistic_dual_harm",
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members, "seed": args.seed,
        },
        "predictions": reports,
        "candidate_ranking_learned": False,
        "final_reserve_preflight_outcomes_accessed": False,
        "new_confirmation_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
