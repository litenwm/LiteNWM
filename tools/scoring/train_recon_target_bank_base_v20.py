#!/usr/bin/env python3
"""Nested whole-bank cross-fit for the RECON target-distribution base gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring.train_recon_cem_latent_base_risk_v16 import (  # noqa: E402
    TailConfig,
    TailRiskModel,
    calibration,
    compact_base_feature,
    normalization,
    predict_tail,
)
from tools.scoring.train_recon_cem_state_gate_v6 import (  # noqa: E402
    method_summary,
    optional,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning-rate", type=float, default=4.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-3)
    parser.add_argument("--dro-eta", type=float, default=0.08)
    parser.add_argument("--ensemble-members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_bank(root: Path, letter: str) -> dict[str, Any]:
    bank = root / f"bank_{letter}"
    feature_path = (
        bank / "features" / f"recon_target_bank_{letter}_v20_features.npz"
    )
    score_path = bank / "universal_scores.npz"
    rollout_path = bank / "spatial_rollout_v17.npz"
    source_path = bank / "source_labeled.npz"
    empty_latent = np.empty((100, 0), dtype=np.float32)
    feature, valid = compact_base_feature(
        feature_path, score_path, empty_latent
    )
    with np.load(rollout_path, allow_pickle=False) as payload:
        rollout_record = payload["record_key"].astype(str)
        rollout = payload["spatial_rollout_features"].astype(np.float32)
        rollout_meta = json.loads(str(payload["metadata"]))
    with np.load(source_path, allow_pickle=False) as payload:
        record = payload["record_key"].astype(str)
        route = payload["route_group"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        targets = payload["targets"].astype(np.float32)
        absolute = payload["absolute_metrics"].astype(np.float32)
    with np.load(feature_path, allow_pickle=False) as payload:
        feature_record = payload["record_key"].astype(str)
    if not (
        np.array_equal(record, feature_record)
        and np.array_equal(record, rollout_record)
        and rollout.shape == (len(record), 17, 258)
        and not rollout_meta.get("outcomes_accessed")
        and not rollout_meta.get("labels_in_output")
    ):
        raise RuntimeError(f"bank_{letter} identity/rollout contract failed")
    feature = np.concatenate((feature, rollout[:, 1:]), axis=2)
    delta = targets[:, 1:, :4]
    dual = ((delta[..., 0] < -1.0e-7) & (delta[..., 3] < -1.0e-7)).astype(
        np.float32
    )
    harm = ((delta[..., 0] > 1.0e-7) | (delta[..., 3] > 1.0e-7)).astype(
        np.float32
    )
    if not np.isfinite(feature).all():
        raise RuntimeError(f"bank_{letter} non-finite model feature")
    return {
        "name": f"bank_{letter}",
        "feature": feature,
        "valid": valid,
        "delta": delta,
        "absolute": absolute,
        "dual": dual,
        "harm": harm,
        "record": record,
        "route": np.asarray([f"bank_{letter}::{value}" for value in route]),
        "trajectory": trajectory,
        "artifacts": {
            "feature": str(feature_path),
            "feature_sha256": file_sha256(feature_path),
            "score": str(score_path),
            "score_sha256": file_sha256(score_path),
            "rollout": str(rollout_path),
            "rollout_sha256": file_sha256(rollout_path),
            "source": str(source_path),
            "source_sha256": file_sha256(source_path),
        },
    }


def train_group_dro(
    *,
    feature: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    group: np.ndarray,
    train_index: np.ndarray,
    config: TailConfig,
    args: argparse.Namespace,
    seed: int,
) -> tuple[TailRiskModel, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    mean, scale = normalization(feature, train_index)
    normalized = ((feature[train_index] - mean) / scale).astype(np.float32)
    x = torch.from_numpy(normalized).to(args.device)
    dual_target = torch.from_numpy(dual[train_index]).to(args.device)
    harm_target = torch.from_numpy(harm[train_index]).to(args.device)
    train_group = group[train_index]
    unique_group, group_id = np.unique(train_group, return_inverse=True)
    group_id_tensor = torch.from_numpy(group_id).to(args.device)
    q = torch.full(
        (len(unique_group),), 1.0 / len(unique_group), device=args.device
    )
    model = TailRiskModel(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    last_group_loss = None
    for _ in range(args.epochs):
        model.train()
        logits = model(x)
        dual_loss = F.binary_cross_entropy_with_logits(
            logits[..., 0], dual_target, reduction="none"
        )
        dual_loss *= torch.where(
            dual_target > 0.5,
            torch.ones_like(dual_target),
            torch.full_like(dual_target, 3.0),
        )
        harm_loss = F.binary_cross_entropy_with_logits(
            logits[..., 1], harm_target, reduction="none"
        )
        harm_loss *= torch.where(
            harm_target > 0.5,
            torch.full_like(harm_target, 4.0),
            torch.ones_like(harm_target),
        )
        safety = logits[..., 0] - logits[..., 1]
        chosen = safety.argmax(dim=1)
        row = torch.arange(len(train_index), device=args.device)
        hard_dual = F.binary_cross_entropy_with_logits(
            logits[row, chosen, 0], dual_target[row, chosen], reduction="none"
        )
        hard_harm = F.binary_cross_entropy_with_logits(
            logits[row, chosen, 1], harm_target[row, chosen], reduction="none"
        )
        state_loss = (dual_loss + 1.5 * harm_loss).mean(1) + 3.0 * (
            hard_dual + hard_harm
        )
        group_loss = torch.stack(
            [state_loss[group_id_tensor == value].mean() for value in range(len(unique_group))]
        )
        with torch.no_grad():
            centered = torch.clamp(group_loss - group_loss.mean(), -5.0, 5.0)
            q *= torch.exp(args.dro_eta * centered)
            q /= q.sum()
        loss = (q.detach() * group_loss).sum() + 0.10 * state_loss.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_group_loss = group_loss.detach().cpu().numpy()
    model.eval()
    assert last_group_loss is not None
    stats = {
        "groups": int(len(unique_group)),
        "group_loss_mean": float(last_group_loss.mean()),
        "group_loss_p90": float(np.quantile(last_group_loss, 0.90)),
        "group_loss_max": float(last_group_loss.max()),
        "dro_weight_max": float(q.max().item()),
        "dro_effective_groups": float(1.0 / torch.sum(q * q).item()),
    }
    return model, mean, scale, stats


def ensemble_predict(
    *,
    feature: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    group: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    config: TailConfig,
    args: argparse.Namespace,
    stage: str,
    seed_offset: int,
    output_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    dual_values = []
    harm_values = []
    checkpoints = []
    dro_stats = []
    for member in range(args.ensemble_members):
        seed = args.seed + seed_offset + member * 1009
        model, mean, scale, stats = train_group_dro(
            feature=feature,
            dual=dual,
            harm=harm,
            group=group,
            train_index=train_index,
            config=config,
            args=args,
            seed=seed,
        )
        p_dual, p_harm = predict_tail(
            model, mean, scale, feature[predict_index], args.device
        )
        dual_values.append(p_dual)
        harm_values.append(p_harm)
        checkpoint = output_dir / f"{stage}_member_{member}.pt"
        torch.save(
            {
                "schema": "recon_target_bank_base_v20_member",
                "stage": stage,
                "config": asdict(config),
                "feature_mean": mean,
                "feature_scale": scale,
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "seed": seed,
                "group_dro": stats,
            },
            checkpoint,
        )
        checkpoints.append({
            "path": str(checkpoint), "sha256": file_sha256(checkpoint), "seed": seed,
        })
        dro_stats.append(stats)
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
    dual_stack = np.stack(dual_values)
    harm_stack = np.stack(harm_values)
    return (
        dual_stack.mean(0), harm_stack.mean(0),
        dual_stack.std(0), harm_stack.std(0), checkpoints, dro_stats,
    )


def checks(metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "coverage_at_least_0.40": metrics["coverage"] >= 0.40,
        "dual_at_least_0.90": optional(metrics["dual_precision"], 0.0) >= 0.90,
        "wilson_at_least_0.80": optional(metrics["dual_wilson_lower_95"], 0.0) >= 0.80,
        "harm_at_most_0.10": optional(metrics["harm_rate"], 1.0) <= 0.10,
    }


def conformal_higher(values: np.ndarray, alpha: float = 0.10) -> float:
    if len(values) == 0:
        return 0.0
    level = min(1.0, math.ceil((len(values) + 1) * (1.0 - alpha)) / len(values))
    return float(np.quantile(values, level, method="higher"))


def route_conformal(
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    p_dual_std: np.ndarray,
    p_harm_std: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    valid: np.ndarray,
    group: np.ndarray,
    z: float,
    risk_weight: float,
) -> tuple[float, float]:
    raw_dual = p_dual - z * p_dual_std
    raw_harm = p_harm + z * p_harm_std
    score = np.where(valid, raw_dual - risk_weight * raw_harm, -np.inf)
    candidate = score.argmax(1)
    row = np.arange(len(candidate))
    dual_residual = raw_dual[row, candidate] - dual[row, candidate]
    harm_residual = harm[row, candidate] - raw_harm[row, candidate]
    route_dual = []
    route_harm = []
    for value in np.unique(group):
        mask = group == value
        route_dual.append(float(dual_residual[mask].max()))
        route_harm.append(float(harm_residual[mask].max()))
    return conformal_higher(np.asarray(route_dual)), conformal_higher(np.asarray(route_harm))


def apply_policy(
    policy: dict[str, float],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    p_dual_std: np.ndarray,
    p_harm_std: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    dual_lower = p_dual - policy["uncertainty_scale"] * p_dual_std - policy["dual_conformal"]
    harm_upper = p_harm + policy["uncertainty_scale"] * p_harm_std + policy["harm_conformal"]
    score = np.where(valid, dual_lower - policy["risk_weight"] * harm_upper, -np.inf)
    candidate = score.argmax(1)
    row = np.arange(len(candidate))
    promote = (
        (dual_lower[row, candidate] >= policy["dual_threshold"])
        & (harm_upper[row, candidate] <= policy["harm_threshold"])
        & np.isfinite(score[row, candidate])
    )
    return np.where(promote, candidate + 1, 0).astype(np.int64)


def calibrate_policy(
    *,
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    p_dual_std: np.ndarray,
    p_harm_std: np.ndarray,
    dual: np.ndarray,
    harm: np.ndarray,
    valid: np.ndarray,
    group: np.ndarray,
    cohort: np.ndarray,
    delta: np.ndarray,
    absolute: np.ndarray,
) -> dict[str, Any]:
    policies = []
    for z in (0.0, 0.5, 1.0, 1.5, 2.0):
        for risk_weight in (0.5, 1.0, 2.0, 4.0):
            dual_q, harm_q = route_conformal(
                p_dual, p_harm, p_dual_std, p_harm_std,
                dual, harm, valid, group, z, risk_weight,
            )
            base = {
                "uncertainty_scale": z,
                "risk_weight": risk_weight,
                "dual_conformal": dual_q,
                "harm_conformal": harm_q,
            }
            raw_dual = p_dual - z * p_dual_std - dual_q
            raw_harm = p_harm + z * p_harm_std + harm_q
            rank = np.where(valid, raw_dual - risk_weight * raw_harm, -np.inf)
            challenger = rank.argmax(1)
            row = np.arange(len(challenger))
            dvalue = raw_dual[row, challenger]
            hvalue = raw_harm[row, challenger]
            dual_grid = np.unique(np.quantile(dvalue, np.linspace(0.0, 0.85, 18)))
            harm_grid = np.unique(np.quantile(hvalue, np.linspace(0.15, 1.0, 18)))
            for dual_threshold in dual_grid:
                for harm_threshold in harm_grid:
                    policy = {
                        **base,
                        "dual_threshold": float(dual_threshold),
                        "harm_threshold": float(harm_threshold),
                    }
                    selected = apply_policy(
                        policy, p_dual, p_harm, p_dual_std, p_harm_std, valid
                    )
                    per_cohort = {}
                    for name in np.unique(cohort):
                        mask = cohort == name
                        metric = method_summary(selected[mask], delta[mask], absolute[mask])
                        per_cohort[name] = {"metrics": metric, "checks": checks(metric)}
                    strict = all(
                        all(value["checks"].values()) for value in per_cohort.values()
                    )
                    policies.append({
                        "policy": policy,
                        "selected": selected,
                        "per_cohort": per_cohort,
                        "strict": strict,
                    })
    feasible = [value for value in policies if value["strict"]]
    def rank_key(value: dict[str, Any]):
        metrics = [item["metrics"] for item in value["per_cohort"].values()]
        return (
            sum(sum(item["checks"].values()) for item in value["per_cohort"].values()),
            min(optional(metric["dual_wilson_lower_95"], 0.0) for metric in metrics),
            min(optional(metric["dual_precision"], 0.0) for metric in metrics),
            -max(optional(metric["harm_rate"], 1.0) for metric in metrics),
            min(metric["coverage"] for metric in metrics),
            sum(metric["coverage"] for metric in metrics),
        )
    best = max(feasible if feasible else policies, key=rank_key)
    return {
        "strict_feasible": bool(feasible),
        "strict_feasible_policies": len(feasible),
        "searched_policies": len(policies),
        "policy": best["policy"],
        "inner_per_cohort": best["per_cohort"],
    }


def main() -> None:
    args = parse_args()
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not audit["reserve"]["sealed"]:
        raise RuntimeError("v20 cache/reserve audit did not pass")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    banks = [load_bank(args.root, letter) for letter in ("a", "b", "c")]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    feature = np.concatenate([bank["feature"] for bank in banks])
    valid = np.concatenate([bank["valid"] for bank in banks])
    delta = np.concatenate([bank["delta"] for bank in banks])
    absolute = np.concatenate([bank["absolute"] for bank in banks])
    dual = np.concatenate([bank["dual"] for bank in banks])
    harm = np.concatenate([bank["harm"] for bank in banks])
    record = np.concatenate([bank["record"] for bank in banks])
    group = np.concatenate([bank["route"] for bank in banks])
    cohort = np.concatenate([
        np.full(len(bank["record"]), bank["name"], dtype="<U16") for bank in banks
    ])
    config = TailConfig(
        input_dim=feature.shape[-1], hidden_dim=args.hidden_dim, dropout=args.dropout
    )
    selected_oof = np.zeros(len(feature), dtype=np.int64)
    p_dual_oof = np.zeros((len(feature), 16), dtype=np.float32)
    p_harm_oof = np.zeros_like(p_dual_oof)
    p_dual_std_oof = np.zeros_like(p_dual_oof)
    p_harm_std_oof = np.zeros_like(p_dual_oof)
    outer_reports = {}
    all_checkpoints = []
    all_dro = []
    for outer_number, outer_name in enumerate(np.unique(cohort)):
        outer_index = np.flatnonzero(cohort == outer_name)
        remaining_names = [name for name in np.unique(cohort) if name != outer_name]
        inner_index = np.flatnonzero(np.isin(cohort, remaining_names))
        inner_p_dual = np.zeros((len(inner_index), 16), dtype=np.float32)
        inner_p_harm = np.zeros_like(inner_p_dual)
        inner_p_dual_std = np.zeros_like(inner_p_dual)
        inner_p_harm_std = np.zeros_like(inner_p_dual)
        inner_cohort = cohort[inner_index]
        for inner_number, inner_name in enumerate(remaining_names):
            local_predict = np.flatnonzero(inner_cohort == inner_name)
            global_predict = inner_index[local_predict]
            global_train = inner_index[inner_cohort != inner_name]
            values = ensemble_predict(
                feature=feature, dual=dual, harm=harm, group=group,
                train_index=global_train, predict_index=global_predict,
                config=config, args=args,
                stage=f"outer_{outer_name}_inner_{inner_name}",
                seed_offset=outer_number * 100000 + inner_number * 10000,
                output_dir=args.output_dir,
            )
            inner_p_dual[local_predict], inner_p_harm[local_predict] = values[:2]
            inner_p_dual_std[local_predict], inner_p_harm_std[local_predict] = values[2:4]
            all_checkpoints.extend(values[4])
            all_dro.extend(values[5])
        calibrated = calibrate_policy(
            p_dual=inner_p_dual, p_harm=inner_p_harm,
            p_dual_std=inner_p_dual_std, p_harm_std=inner_p_harm_std,
            dual=dual[inner_index], harm=harm[inner_index], valid=valid[inner_index],
            group=group[inner_index], cohort=cohort[inner_index],
            delta=delta[inner_index], absolute=absolute[inner_index],
        )
        values = ensemble_predict(
            feature=feature, dual=dual, harm=harm, group=group,
            train_index=inner_index, predict_index=outer_index,
            config=config, args=args, stage=f"outer_{outer_name}_final",
            seed_offset=outer_number * 100000 + 50000, output_dir=args.output_dir,
        )
        p_dual_oof[outer_index], p_harm_oof[outer_index] = values[:2]
        p_dual_std_oof[outer_index], p_harm_std_oof[outer_index] = values[2:4]
        all_checkpoints.extend(values[4])
        all_dro.extend(values[5])
        selected = apply_policy(
            calibrated["policy"], values[0], values[1], values[2], values[3], valid[outer_index]
        )
        selected_oof[outer_index] = selected
        metric = method_summary(selected, delta[outer_index], absolute[outer_index])
        outer_reports[outer_name] = {
            "outer_metrics": metric,
            "outer_checks": checks(metric),
            "inner_calibration": calibrated,
            "calibration": {
                "dual": calibration(values[0], dual[outer_index]),
                "harm": calibration(values[1], harm[outer_index]),
            },
            "train_states": int(len(inner_index)),
            "train_routes": int(len(np.unique(group[inner_index]))),
            "heldout_states": int(len(outer_index)),
            "heldout_routes": int(len(np.unique(group[outer_index]))),
        }
    strict = all(all(value["outer_checks"].values()) for value in outer_reports.values())
    pooled = method_summary(selected_oof, delta, absolute)
    prediction_path = args.output_dir / "nested_oof_predictions.npz"
    write_npz(
        prediction_path,
        record_key=record,
        cohort=cohort,
        route_group=group,
        p_dual=p_dual_oof,
        p_harm=p_harm_oof,
        p_dual_std=p_dual_std_oof,
        p_harm_std=p_harm_std_oof,
        valid=valid,
        selected_index=selected_oof,
    )
    worst = {
        "coverage": min(value["outer_metrics"]["coverage"] for value in outer_reports.values()),
        "dual_precision": min(optional(value["outer_metrics"]["dual_precision"], 0.0) for value in outer_reports.values()),
        "dual_wilson_lower_95": min(optional(value["outer_metrics"]["dual_wilson_lower_95"], 0.0) for value in outer_reports.values()),
        "harm_rate": max(optional(value["outer_metrics"]["harm_rate"], 1.0) for value in outer_reports.values()),
    }
    report = {
        "schema": "recon_target_bank_base_v20",
        "status": (
            "base_all_outer_banks_passed_cem_replacement_may_train"
            if strict else "base_outer_bank_safety_failed_continue_target_bank_research"
        ),
        "evidence_role": "target_distribution_development_nested_whole_bank_crossfit",
        "states": int(len(feature)),
        "routes": int(len(np.unique(group))),
        "feature_dim": int(feature.shape[-1]),
        "fixed_pool": "K17_trimmed_mean75_incumbent_plus_fixed_nomad_x16",
        "cem": False,
        "nested_protocol": (
            "outer whole-bank holdout; policy and route-block conformal correction "
            "fit only from inner whole-bank cross-predictions"
        ),
        "group_dro": {"enabled": True, "eta": args.dro_eta, "route_is_group": True},
        "strict_gates": {
            "coverage": 0.40, "dual_precision": 0.90,
            "dual_wilson_lower_95": 0.80, "harm_rate": 0.10,
        },
        "config": {
            **asdict(config), "epochs": args.epochs,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "ensemble_members": args.ensemble_members, "seed": args.seed,
        },
        "outer_folds": outer_reports,
        "worst_outer_fold": worst,
        "pooled_nested_oof": pooled,
        "strict_all_outer_folds": strict,
        "bank_artifacts": {bank["name"]: bank["artifacts"] for bank in banks},
        "audit": str(args.audit),
        "audit_sha256": file_sha256(args.audit),
        "checkpoints": all_checkpoints,
        "group_dro_fit_summaries": all_dro,
        "predictions": str(prediction_path),
        "predictions_sha256": file_sha256(prediction_path),
        "final_reserve_accessed": False,
        "final_reserve_remains_sealed": True,
        "cem_replacement_trained": False,
        "formal_claim_allowed": False,
    }
    write_json(result_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
