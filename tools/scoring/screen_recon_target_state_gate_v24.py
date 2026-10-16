#!/usr/bin/env python3
"""Screen state-level safety gates with a frozen universal challenger."""

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
from tools.scoring.screen_recon_target_bank_models_v22 import load_bank
from tools.scoring.train_recon_cem_state_gate_v6 import method_summary
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


class StateRisk(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def frozen_challenger(score: dict, valid: np.ndarray) -> np.ndarray:
    uncertainty = score["std"][..., 0] + score["std"][..., 3]
    value = score["dual"] - 0.5 * score["harm"] - 0.5 * uncertainty
    value = np.where(valid, value, -np.inf)
    candidate = value.argmax(1)
    if not np.isfinite(value[np.arange(len(candidate)), candidate]).all():
        raise RuntimeError("state without valid frozen challenger")
    return candidate


def train_model(
    feature: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    group: np.ndarray,
    train_index: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[StateRisk, np.ndarray, np.ndarray]:
    set_seed(seed)
    mean = feature[train_index].mean(0).astype(np.float32)
    scale = np.maximum(feature[train_index].std(0), 1.0e-4).astype(np.float32)
    x = torch.from_numpy(((feature[train_index] - mean) / scale).astype(np.float32)).to(args.device)
    dual_target = torch.from_numpy(dual[train_index]).to(args.device)
    harm_target = torch.from_numpy(harm[train_index]).to(args.device)
    train_group = group[train_index]
    unique, group_id = np.unique(train_group, return_inverse=True)
    group_id = torch.from_numpy(group_id).to(args.device)
    model = StateRisk(feature.shape[1], args.hidden_dim, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    top_count = max(1, math.ceil(0.20 * len(unique)))
    for _ in range(args.epochs):
        model.train()
        logits = model(x)
        dual_loss = F.binary_cross_entropy_with_logits(
            logits[:, 0], dual_target, reduction="none"
        ) * torch.where(
            dual_target > 0.5, torch.ones_like(dual_target),
            torch.full_like(dual_target, 4.0),
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            logits[:, 1], harm_target, reduction="none"
        ) * torch.where(
            harm_target > 0.5, torch.full_like(harm_target, 5.0),
            torch.ones_like(harm_target),
        )
        state_loss = dual_loss + 1.5 * harm_loss
        group_loss = torch.stack([
            state_loss[group_id == value].mean() for value in range(len(unique))
        ])
        robust = torch.topk(group_loss, top_count).values.mean()
        loss = 0.50 * group_loss.mean() + 0.50 * robust
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    return model, mean, scale


def predict(
    model: StateRisk, mean: np.ndarray, scale: np.ndarray,
    feature: np.ndarray, device: str,
) -> tuple[np.ndarray, np.ndarray]:
    value = ((feature - mean) / scale).astype(np.float32)
    with torch.inference_mode():
        probability = torch.sigmoid(model(torch.from_numpy(value).to(device))).cpu().numpy()
    return probability[:, 0], probability[:, 1]


def top40(
    p_dual: np.ndarray, p_harm: np.ndarray,
    p_dual_std: np.ndarray, p_harm_std: np.ndarray,
    challenger: np.ndarray, delta: np.ndarray, absolute: np.ndarray,
) -> dict:
    best = None
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        for weight in (0.5, 1.0, 2.0, 4.0):
            risk = p_dual - z * p_dual_std - weight * (p_harm + z * p_harm_std)
            order = np.argsort(-risk, kind="stable")[:40]
            selected = np.zeros(len(challenger), dtype=np.int64)
            selected[order] = challenger[order] + 1
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
    banks = [load_bank(args.root, value) for value in "abcdef"]
    scores = [load_score(args.root, value) for value in "abcdef"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    full = np.concatenate([bank["feature"] for bank in banks])
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    group = np.concatenate([bank["route"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, bank["name"], dtype="<U16") for bank in banks
    ])
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
    row = np.arange(len(full))
    chosen = full[row, challenger]
    variants = {
        "state_universal_support": chosen[:, 474:486],
        "state_motion_universal_support": chosen[:, 434:486],
        "state_static_universal_support": np.concatenate(
            (chosen[:, :210], chosen[:, 474:486]), axis=1
        ),
        "state_universal_support_spatial": np.concatenate(
            (chosen[:, 474:486], chosen[:, 486:744]), axis=1
        ),
    }
    reports = {}
    for variant_number, (name, feature) in enumerate(variants.items()):
        p_dual = np.zeros(len(feature), dtype=np.float32)
        p_harm = np.zeros(len(feature), dtype=np.float32)
        p_dual_std = np.zeros(len(feature), dtype=np.float32)
        p_harm_std = np.zeros(len(feature), dtype=np.float32)
        for fold_number, heldout in enumerate(np.unique(cohort)):
            train_index = np.flatnonzero(cohort != heldout)
            heldout_index = np.flatnonzero(cohort == heldout)
            member_dual = []
            member_harm = []
            for member in range(args.ensemble_members):
                seed = args.seed + variant_number * 1000000 + fold_number * 100000 + member * 1009
                model, mean, scale = train_model(
                    feature, dual, harm, group, train_index, args, seed
                )
                values = predict(model, mean, scale, feature[heldout_index], args.device)
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
                challenger[mask], delta[mask], absolute[mask],
            )
        prediction = args.output_dir / f"{name}_oof.npz"
        write_npz(
            prediction, record_key=record, cohort=cohort, route_group=group,
            challenger_index=challenger + 1,
            p_dual=p_dual, p_harm=p_harm,
            p_dual_std=p_dual_std, p_harm_std=p_harm_std,
        )
        reports[name] = {
            "feature_dim": int(feature.shape[1]),
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
        "schema": "recon_target_state_gate_v24_screen",
        "states": 600, "routes": 150, "banks": 6,
        "challenger_contract": (
            "frozen universal p_dual - 0.5*p_harm - 0.5*(std_ate+std_objective)"
        ),
        "candidate_ranking_learned": False,
        "state_gate_only": True,
        "route_robust_loss": "0.5 route mean + 0.5 worst-route-CVaR20",
        "evidence_role": "development_only_whole_bank_oof_state_gate_screen",
        "warning": "per-bank top40 is noncausal separability diagnostic only",
        "variants": reports,
        "config": {
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members,
            "hidden_dim": args.hidden_dim, "dropout": args.dropout, "seed": args.seed,
        },
        "final_reserve_accessed": False,
        "final_reserve_remains_sealed": True,
        "cem": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
