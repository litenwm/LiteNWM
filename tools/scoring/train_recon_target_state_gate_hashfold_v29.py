#!/usr/bin/env python3
"""Freeze deployable route-hashed models trained by whole-bank exclusion."""

from __future__ import annotations

import argparse
import json
import math
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
from tools.scoring.screen_recon_target_state_gate_v24 import StateRisk, predict
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256, write_json, write_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.40)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_fast(
    feature: np.ndarray, dual: np.ndarray, harm: np.ndarray,
    group: np.ndarray, train_index: np.ndarray,
    args: argparse.Namespace, seed: int,
) -> tuple[StateRisk, np.ndarray, np.ndarray]:
    set_seed(seed)
    mean = feature[train_index].mean(0).astype(np.float32)
    scale = np.maximum(feature[train_index].std(0), 1.0e-4).astype(np.float32)
    x = torch.from_numpy(((feature[train_index] - mean) / scale).astype(np.float32)).to(args.device)
    y_dual = torch.from_numpy(dual[train_index]).to(args.device)
    y_harm = torch.from_numpy(harm[train_index]).to(args.device)
    unique, group_id = np.unique(group[train_index], return_inverse=True)
    group_id = torch.from_numpy(group_id).to(args.device)
    counts = torch.bincount(group_id, minlength=len(unique)).float()
    top_count = max(1, math.ceil(0.20 * len(unique)))
    model = StateRisk(feature.shape[1], args.hidden_dim, args.dropout).to(args.device)
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
        state_loss = dual_loss + 1.5 * harm_loss
        group_sum = torch.zeros(len(unique), device=args.device).scatter_add_(
            0, group_id, state_loss
        )
        group_loss = group_sum / counts
        robust = torch.topk(group_loss, top_count).values.mean()
        loss = 0.50 * group_loss.mean() + 0.50 * robust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, mean, scale


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
    checkpoint_report = {}
    prediction_report = {}
    unique_cohort = np.unique(cohort)
    for variant_number, (name, feature) in enumerate(variants.items()):
        p_dual = np.zeros(600, dtype=np.float32)
        p_harm = np.zeros(600, dtype=np.float32)
        p_dual_std = np.zeros(600, dtype=np.float32)
        p_harm_std = np.zeros(600, dtype=np.float32)
        checkpoint_report[name] = {}
        for fold_number, heldout in enumerate(unique_cohort):
            train_index = np.flatnonzero(cohort != heldout)
            predict_index = np.flatnonzero(cohort == heldout)
            member_dual = []
            member_harm = []
            checkpoint_report[name][heldout] = []
            for member in range(args.ensemble_members):
                seed = args.seed + variant_number * 1000000 + fold_number * 100000 + member * 1009
                model, mean, scale = train_fast(
                    feature, dual, harm, group, train_index, args, seed
                )
                values = predict(model, mean, scale, feature[predict_index], args.device)
                member_dual.append(values[0])
                member_harm.append(values[1])
                path = args.output_dir / f"{name}_exclude_{heldout}_member_{member}.pt"
                torch.save(
                    {
                        "schema": "recon_target_state_gate_hashfold_v29_member",
                        "variant": name, "excluded_bank": heldout,
                        "input_dim": feature.shape[1],
                        "hidden_dim": args.hidden_dim, "dropout": args.dropout,
                        "feature_mean": mean, "feature_scale": scale,
                        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "seed": seed,
                    },
                    path,
                )
                checkpoint_report[name][heldout].append({
                    "path": str(path), "sha256": file_sha256(path), "seed": seed,
                })
                del model
            dual_stack = np.stack(member_dual)
            harm_stack = np.stack(member_harm)
            p_dual[predict_index] = dual_stack.mean(0)
            p_harm[predict_index] = harm_stack.mean(0)
            p_dual_std[predict_index] = dual_stack.std(0)
            p_harm_std[predict_index] = harm_stack.std(0)
        prediction_path = args.output_dir / f"{name}_oof.npz"
        write_npz(
            prediction_path, record_key=record, cohort=cohort, route_group=group,
            challenger_index=challenger + 1,
            p_dual=p_dual, p_harm=p_harm,
            p_dual_std=p_dual_std, p_harm_std=p_harm_std,
        )
        prediction_report[name] = {
            "path": str(prediction_path), "sha256": file_sha256(prediction_path),
        }
    report = {
        "schema": "recon_target_state_gate_hashfold_v29",
        "status": "six_hashfold_model_groups_and_oof_predictions_frozen",
        "deployment_router": (
            "sha256(route_group) modulo 6 selects one excluded-bank model group; "
            "label-free, route-constant, no future-window quota"
        ),
        "fold_order": unique_cohort.tolist(),
        "checkpoints": checkpoint_report,
        "predictions": prediction_report,
        "states": 600, "routes": 150,
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout,
            "seed": args.seed,
        },
        "online_causal": True, "future_route_quota": False,
        "preflight_outcomes_accessed": False,
        "new_confirmation_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
