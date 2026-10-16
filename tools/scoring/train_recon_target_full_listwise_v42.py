#!/usr/bin/env python3
"""Full-feature setwise/listwise candidate ranker, eleven-bank LOBO screen.

This is a development separability experiment.  Models are whole-bank OOF;
the final common threshold scan uses all development OOF labels and therefore
is not a confirmation or deployment claim.  A feasible result must still be
re-run with a strict outer/inner gate before any reserve can be touched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.scoring.train_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    compact_base_feature,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRICS = ("ATE", "RPE", "Endpoint", "Objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank-g", type=Path, required=True)
    parser.add_argument("--bank-h", type=Path, required=True)
    parser.add_argument("--bank-ijk-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261015)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bank(
    name: str, feature: Path, score: Path, source: Path, spatial: Path
) -> dict:
    compact, valid = compact_base_feature(
        feature, score, np.empty((100, 0), np.float32)
    )
    with np.load(spatial, allow_pickle=False) as payload:
        rollout = payload["spatial_rollout_features"][:, 1:].astype(np.float32)
    if rollout.shape != (100, 16, 258):
        raise RuntimeError(f"spatial feature contract changed for {name}")
    x = np.concatenate((compact, rollout), axis=2).astype(np.float32)
    with np.load(source, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
        targets = payload["targets"].astype(np.float32)
        absolute = payload["absolute_metrics"].astype(np.float32)
    delta = targets[:, 1:, :4]
    dual = ((delta[..., 0] < -1e-7) &
            (delta[..., 3] < -1e-7)).astype(np.float32)
    harm = ((delta[..., 0] > 1e-7) |
            (delta[..., 3] > 1e-7)).astype(np.float32)
    return {
        "name": name,
        "x": x,
        "valid": valid.astype(bool),
        "delta": delta,
        "dual": dual,
        "harm": harm,
        "absolute": absolute,
        "record": record,
        "route": np.asarray([f"{name}::{value}" for value in route]),
        "artifacts": {
            "feature": str(feature), "feature_sha256": file_sha256(feature),
            "score": str(score), "score_sha256": file_sha256(score),
            "source": str(source), "source_sha256": file_sha256(source),
            "spatial": str(spatial), "spatial_sha256": file_sha256(spatial),
        },
    }


def load_banks(args: argparse.Namespace) -> list[dict]:
    banks = []
    for letter in "abcdef":
        root = args.root / f"bank_{letter}"
        banks.append(load_bank(
            f"bank_{letter}",
            next((root / "features").glob(
                f"recon_target_bank_{letter}_v*_features.npz"
            )),
            root / "universal_scores.npz",
            root / "source_labeled.npz",
            root / "spatial_rollout_v17.npz",
        ))
    banks.append(load_bank(
        "bank_g",
        args.bank_g / "features/recon_target_confirmation_v29_features.npz",
        args.bank_g / "universal_scores.npz",
        args.bank_g / "source_labeled.npz",
        args.bank_g / "spatial_rollout_v17.npz",
    ))
    banks.append(load_bank(
        "bank_h",
        args.bank_h / "features/recon_target_cem_confirmation_v35_features.npz",
        args.bank_h / "universal_scores.npz",
        args.bank_h / "source_labeled.npz",
        args.bank_h / "spatial_rollout_v17.npz",
    ))
    for letter in "ijk":
        root = args.bank_ijk_root / f"bank_{letter}"
        banks.append(load_bank(
            f"bank_{letter}",
            root / "features" / f"recon_target_bank_{letter}_v40_features.npz",
            root / "universal_scores.npz",
            root / "source_labeled.npz",
            root / "spatial_rollout_v17.npz",
        ))
    return banks


@dataclass(frozen=True)
class Config:
    input_dim: int
    hidden_dim: int
    dropout: float


class Ranker(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.head = nn.Linear(config.hidden_dim, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(value))


def train_model(
    x: np.ndarray,
    valid: np.ndarray,
    delta: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    cohort: np.ndarray,
    route: np.ndarray,
    train_state: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[Ranker, np.ndarray, np.ndarray, dict]:
    set_seed(seed)
    train_valid = valid[train_state]
    flat = x[train_state][train_valid]
    mean = flat.mean(0).astype(np.float32)
    scale = np.maximum(flat.std(0), 1e-4).astype(np.float32)
    model = Ranker(Config(x.shape[-1], args.hidden_dim, args.dropout)).to(
        args.device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    xt = torch.from_numpy(((x[train_state] - mean) / scale).astype(np.float32)).to(args.device)
    vt = torch.from_numpy(valid[train_state]).to(args.device)
    dt = torch.from_numpy(delta[train_state][..., (0, 3)]).to(args.device)
    dual_t = torch.from_numpy(dual[train_state]).to(args.device)
    harm_t = torch.from_numpy(harm[train_state]).to(args.device)
    local_cohort = cohort[train_state]
    local_route = route[train_state]
    unique_route, route_inverse, route_count = np.unique(
        local_route, return_inverse=True, return_counts=True
    )
    state_weight = (1.0 / route_count[route_inverse]).astype(np.float32)
    for name in np.unique(local_cohort):
        mask = local_cohort == name
        state_weight[mask] *= mask.sum() / state_weight[mask].sum()
    state_weight /= state_weight.mean()
    sw = torch.from_numpy(state_weight).to(args.device)
    cohort_masks = [
        torch.from_numpy(np.flatnonzero(local_cohort == name)).to(args.device)
        for name in np.unique(local_cohort)
    ]
    last = {}
    for _ in range(args.epochs):
        model.train()
        output = model(xt)
        pred_delta = output[..., :2]
        dual_logit = output[..., 2]
        harm_logit = output[..., 3]
        regression = F.smooth_l1_loss(
            pred_delta, dt, reduction="none", beta=0.03
        ).sum(-1)
        dual_bce = F.binary_cross_entropy_with_logits(
            dual_logit, dual_t, reduction="none"
        ) * torch.where(dual_t > 0.5, 1.0, 4.0)
        harm_bce = F.binary_cross_entropy_with_logits(
            harm_logit, harm_t, reduction="none"
        ) * torch.where(harm_t > 0.5, 5.0, 1.0)
        candidate_loss = regression + dual_bce + 1.5 * harm_bce
        candidate_loss = torch.where(vt, candidate_loss, torch.zeros_like(candidate_loss))
        state_loss = candidate_loss.sum(1) / vt.sum(1).clamp_min(1)
        cohort_loss = torch.stack([
            (state_loss[index] * sw[index]).sum() / sw[index].sum()
            for index in cohort_masks
        ])
        adversarial = torch.softmax(
            3.0 * (cohort_loss.detach() - cohort_loss.detach().mean()), dim=0
        )
        classification = (
            (0.75 * adversarial + 0.25 / len(cohort_masks)) * cohort_loss
        ).sum()
        rank_score = (
            -pred_delta[..., 0] - pred_delta[..., 1]
            + 0.35 * dual_logit - 0.50 * harm_logit
        ).masked_fill(~vt, -1e9)
        safe_utility = -(delta[train_state, :, 0] + delta[train_state, :, 3])
        safe_utility = np.where(
            valid[train_state] & (dual[train_state] > 0.5),
            safe_utility, -np.inf,
        )
        best = torch.from_numpy(safe_utility.argmax(1).astype(np.int64)).to(args.device)
        has_safe = torch.from_numpy(np.isfinite(safe_utility).any(1)).to(args.device)
        listwise = F.cross_entropy(rank_score[has_safe], best[has_safe])
        probability = torch.softmax(rank_score / 0.5, dim=1)
        expected_harm = (probability * harm_t).sum(1)
        harm_risk = (expected_harm * sw).sum() / sw.sum()
        loss = classification + 0.75 * listwise + 1.5 * harm_risk
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "classification": float(classification.detach().cpu()),
            "listwise": float(listwise.detach().cpu()),
            "expected_harm": float(harm_risk.detach().cpu()),
        }
    return model, mean, scale, last


@torch.no_grad()
def predict_model(
    model: Ranker, mean: np.ndarray, scale: np.ndarray,
    x: np.ndarray, device: str,
) -> np.ndarray:
    model.eval()
    value = torch.from_numpy(((x - mean) / scale).astype(np.float32)).to(device)
    return model(value).cpu().numpy().astype(np.float32)


def metrics(selected: np.ndarray, bank: dict) -> dict:
    promote = selected != 0
    count = int(promote.sum())
    row = np.arange(100)
    chosen_delta = np.zeros((100, 4), np.float32)
    chosen_delta[promote] = bank["delta"][row[promote], selected[promote] - 1]
    dual = int(((chosen_delta[promote, 0] < -1e-7) &
                (chosen_delta[promote, 3] < -1e-7)).sum())
    harm = int(((chosen_delta[promote, 0] > 1e-7) |
                (chosen_delta[promote, 3] > 1e-7)).sum())
    chosen_absolute = bank["absolute"][row, selected]
    return {
        "promotions": count,
        "coverage": count / 100.0,
        "dual": dual / count if count else None,
        "wilson": wilson_lower(dual, count),
        "harm": harm / count if count else None,
        "quality": {
            name: {
                "mean": float(chosen_absolute[:, index].mean()),
                "p90": float(np.quantile(chosen_absolute[:, index], 0.9)),
            }
            for index, name in enumerate(METRICS)
        },
    }


def passes(value: dict) -> bool:
    return bool(
        value["coverage"] >= 0.40
        and value["dual"] is not None and value["dual"] >= 0.90
        and value["wilson"] is not None and value["wilson"] >= 0.80
        and value["harm"] is not None and value["harm"] <= 0.10
    )


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    banks = load_banks(args)
    names = [bank["name"] for bank in banks]
    x = np.concatenate([bank["x"] for bank in banks])
    valid = np.concatenate([bank["valid"] for bank in banks])
    delta = np.concatenate([bank["delta"] for bank in banks])
    dual = np.concatenate([bank["dual"] for bank in banks])
    harm = np.concatenate([bank["harm"] for bank in banks])
    cohort = np.concatenate([
        np.full(100, name, dtype="<U16") for name in names
    ])
    route = np.concatenate([bank["route"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    members = []
    checkpoints = {}
    for fold, name in enumerate(names):
        train_state = np.flatnonzero(cohort != name)
        heldout = np.flatnonzero(cohort == name)
        fold_members = []
        checkpoints[name] = []
        for member in range(args.ensemble_members):
            seed = args.seed + fold * 100_000 + member * 1009
            model, mean, scale, loss = train_model(
                x, valid, delta, dual, harm, cohort, route,
                train_state, args, seed,
            )
            prediction = predict_model(model, mean, scale, x[heldout], args.device)
            fold_members.append(prediction)
            path = args.output_dir / f"exclude_{name}_member_{member}.pt"
            torch.save({
                "schema": "recon_target_full_listwise_v42_member",
                "excluded_bank": name,
                "input_dim": x.shape[-1],
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "feature_mean": mean,
                "feature_scale": scale,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": seed,
                "loss": loss,
            }, path)
            checkpoints[name].append({
                "path": str(path), "sha256": file_sha256(path),
                "seed": seed, "loss": loss,
            })
            del model
        members.append(np.stack(fold_members))
        print(json.dumps({"heldout": name, "members": len(fold_members)}), flush=True)
    member_prediction = np.concatenate(members, axis=1)
    prediction = member_prediction.mean(0)
    prediction_std = member_prediction.std(0)
    pred_delta = prediction[..., :2]
    p_dual = 1.0 / (1.0 + np.exp(-np.clip(prediction[..., 2], -30, 30)))
    p_harm = 1.0 / (1.0 + np.exp(-np.clip(prediction[..., 3], -30, 30)))
    p_dual_member = 1.0 / (1.0 + np.exp(-np.clip(member_prediction[..., 2], -30, 30)))
    p_harm_member = 1.0 / (1.0 + np.exp(-np.clip(member_prediction[..., 3], -30, 30)))
    p_dual_std = p_dual_member.std(0)
    p_harm_std = p_harm_member.std(0)
    prediction_path = args.output_dir / "oof_predictions.npz"
    write_npz(
        prediction_path, record_key=record, cohort=cohort,
        pred_delta=pred_delta, pred_delta_std=prediction_std[..., :2],
        p_dual=p_dual, p_harm=p_harm,
        p_dual_std=p_dual_std, p_harm_std=p_harm_std,
        valid=valid,
    )
    feasible, top40 = [], []
    for harm_weight in (0.25, 0.5, 1.0, 2.0, 4.0):
        for utility_weight in (0.0, 0.5, 1.0, 2.0, 4.0):
            for z in (0.0, 0.5, 1.0, 2.0):
                score = (
                    p_dual - harm_weight * p_harm
                    + utility_weight * (-pred_delta[..., 0] - pred_delta[..., 1])
                    - z * (p_dual_std + p_harm_std + prediction_std[..., :2].sum(-1))
                )
                score = np.where(valid, score, -np.inf)
                best = score.argmax(1)
                confidence = score[np.arange(len(score)), best]
                for threshold in np.unique(confidence):
                    per = {}
                    for index, bank in enumerate(banks):
                        local = slice(index * 100, (index + 1) * 100)
                        selected = np.where(
                            confidence[local] > threshold,
                            best[local] + 1, 0,
                        )
                        per[bank["name"]] = metrics(selected, bank)
                    if all(passes(value) for value in per.values()):
                        feasible.append({
                            "harm_weight": harm_weight,
                            "utility_weight": utility_weight,
                            "z": z,
                            "threshold": float(threshold),
                            "banks": per,
                            "coverage_sum": float(sum(
                                value["coverage"] for value in per.values()
                            )),
                        })
                per_top = {}
                for index, bank in enumerate(banks):
                    local = slice(index * 100, (index + 1) * 100)
                    local_confidence = confidence[local]
                    order = np.argsort(-local_confidence, kind="stable")[:40]
                    selected = np.zeros(100, np.int64)
                    selected[order] = best[local][order] + 1
                    per_top[bank["name"]] = metrics(selected, bank)
                top40.append({
                    "harm_weight": harm_weight,
                    "utility_weight": utility_weight,
                    "z": z,
                    "banks": per_top,
                })
    best_policy = max(feasible, key=lambda value: value["coverage_sum"]) if feasible else None
    best_top40 = max(
        top40,
        key=lambda value: (
            min((metric["wilson"] or 0.0) for metric in value["banks"].values()),
            min((metric["dual"] or 0.0) for metric in value["banks"].values()),
            -max((metric["harm"] or 1.0) for metric in value["banks"].values()),
        ),
    )
    report = {
        "schema": "recon_target_full_listwise_v42",
        "status": "eleven_bank_full_feature_listwise_lobo_development_screen_complete",
        "development_only": True,
        "formal_claim_allowed": False,
        "strict_nested_gate": False,
        "cem": False,
        "feature_contract": (
            "candidate static + raw LTWM + candidate-incumbent raw delta + "
            "action/path physical and delta + frozen universal uncertainty + "
            "support OOD + real candidate-conditioned spatial LTWM rollout"
        ),
        "input_dim": x.shape[-1],
        "states": len(x),
        "routes": len(set(route.tolist())),
        "banks": names,
        "inputs": {bank["name"]: bank["artifacts"] for bank in banks},
        "checkpoints": checkpoints,
        "prediction": {
            "path": str(prediction_path),
            "sha256": file_sha256(prediction_path),
        },
        "policies_tested": 5 * 5 * 4,
        "feasible_count": len(feasible),
        "best_policy": best_policy,
        "best_top40": best_top40,
        "reserve_accessed": False,
    }
    write_json(result_path, report)
    print(json.dumps({
        "feasible_count": len(feasible),
        "best_policy": best_policy,
        "best_top40": best_top40,
        "prediction_sha256": report["prediction"]["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
