#!/usr/bin/env python3
"""Strict 20-bank LOBO screen of native NoMaD state-level goal signals.

The candidate challenger is frozen from the existing v47 whole-bank-OOF
ranker before labels are loaded.  This script then fits only low-capacity
state gates and evaluates a fixed top-40-per-bank separability diagnostic.
That quota is intentionally reported as a batch upper bound, not an online
deployment policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scoring.dense_meta_v1 import wilson_lower  # noqa: E402
from tools.evaluation.export_recon_nomad_goal_condition_v53 import (  # noqa: E402
    bank_roots,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


METRICS = ("ATE", "RPE", "Endpoint", "Objective")
VARIANTS = (
    "distance_only",
    "condition_pca8",
    "condition_pca8_plus_distance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("freeze", "screen"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/"
        "goal_condition_at.npz",
    )
    parser.add_argument(
        "--candidate-oof",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_target_candidate_spatial_v47/"
        "candidate_universal_support_oof.npz",
    )
    parser.add_argument(
        "--valid-oof",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_target_diffusion_setwise_v50/"
        "oof_predictions.npz",
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/"
        "preregistration.json",
    )
    parser.add_argument(
        "--preregistration-amendment",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/"
        "preregistration_amendment_no_sklearn.json",
    )
    return parser.parse_args()


def freeze_challenger(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    with np.load(args.feature, allow_pickle=False) as feature:
        record = feature["record_key"].astype(str)
        bank = feature["bank"].astype(str)
    with np.load(args.candidate_oof, allow_pickle=False) as prediction:
        candidate_record = prediction["record_key"].astype(str)
        cohort = prediction["cohort"].astype(str)
        dual = prediction["dual"].astype(np.float32)
        harm = prediction["harm"].astype(np.float32)
    with np.load(args.valid_oof, allow_pickle=False) as valid_payload:
        valid_record = valid_payload["record_key"].astype(str)
        valid = valid_payload["valid"].astype(bool)
    if not (
        np.array_equal(record, candidate_record)
        and np.array_equal(record, valid_record)
    ):
        raise RuntimeError("feature/v47/v50 record identity mismatch")
    if not np.array_equal(np.char.add("bank_", bank), cohort):
        raise RuntimeError("feature/v47 bank identity mismatch")
    if dual.shape != (2000, 16) or harm.shape != (2000, 16):
        raise RuntimeError("v47 candidate prediction contract changed")
    if valid.shape != (2000, 16):
        raise RuntimeError("candidate valid-mask contract changed")
    state_eligible = valid.any(1)
    score = np.where(valid, dual - 2.0 * harm, -np.inf)
    challenger = score.argmax(1).astype(np.int64) + 1
    confidence = score[np.arange(len(score)), challenger - 1].astype(np.float32)
    selected_path = args.output_dir / "frozen_challenger.npz"
    write_npz(
        selected_path,
        record_key=record,
        bank=bank,
        challenger_index=challenger,
        challenger_score=confidence,
        state_eligible=state_eligible,
    )
    audit = {
        "schema": "recon_nomad_goal_state_gate_v53_challenger_freeze",
        "status": "challenger_frozen_before_state_gate_label_screen",
        "states": 2000,
        "banks": 20,
        "candidate_ranker": "existing v47 candidate_universal_support whole-bank-OOF",
        "candidate_ranker_retrained": False,
        "formula": "argmax over valid candidates of p_dual - 2.0*p_harm",
        "state_support_guard": (
            "states with no valid challenger are ineligible for promotion; "
            "the incumbent remains selected"
        ),
        "ineligible_states": int((~state_eligible).sum()),
        "ineligible_states_by_bank": {
            letter: int(((bank == letter) & ~state_eligible).sum())
            for letter in "abcdefghijklmnopqrst"
        },
        "feature": str(args.feature),
        "feature_sha256": file_sha256(args.feature),
        "candidate_oof": str(args.candidate_oof),
        "candidate_oof_sha256": file_sha256(args.candidate_oof),
        "valid_oof": str(args.valid_oof),
        "valid_oof_sha256": file_sha256(args.valid_oof),
        "preregistration_sha256": file_sha256(args.preregistration),
        "preregistration_amendment_sha256": file_sha256(
            args.preregistration_amendment
        ),
        "frozen_challenger": str(selected_path),
        "frozen_challenger_sha256": file_sha256(selected_path),
        "outcomes_accessed": False,
        "final_confirmation_reserve_accessed": False,
        "cem": False,
    }
    write_json(args.output_dir / "challenger_freeze.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


def load_labels(
    record: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routes, targets, absolutes = [], [], []
    offset = 0
    for letter, root in sorted(bank_roots().items()):
        source = root / "source_labeled.npz"
        with np.load(source, allow_pickle=False) as payload:
            local_record = payload["record_key"].astype(str)
            route = payload["route_group"].astype(str)
            target = payload["targets"].astype(np.float32)
            absolute = payload["absolute_metrics"].astype(np.float32)
        local = slice(offset, offset + 100)
        if not np.array_equal(record[local], local_record):
            raise RuntimeError(f"bank {letter} state-label record mismatch")
        if target.shape[0:2] != (100, 17) or absolute.shape != (100, 17, 4):
            raise RuntimeError(f"bank {letter} state-label shape changed")
        routes.append(route)
        targets.append(target)
        absolutes.append(absolute)
        offset += 100
    route = np.concatenate(routes)
    if len(np.unique(route)) != 500:
        raise RuntimeError(
            f"expected 500 route-disjoint groups across A--T, got {len(np.unique(route))}"
        )
    return route, np.concatenate(targets), np.concatenate(absolutes)


def route_balanced_weight(route: np.ndarray, bank: np.ndarray) -> np.ndarray:
    _, inverse, count = np.unique(route, return_inverse=True, return_counts=True)
    weight = 1.0 / count[inverse]
    for letter in np.unique(bank):
        local = bank == letter
        weight[local] *= local.sum() / weight[local].sum()
    return (weight / weight.mean()).astype(np.float64)


def standardize(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(0, keepdims=True)
    scale = np.maximum(train.std(0, keepdims=True), 1e-6)
    return (train - mean) / scale, (test - mean) / scale


def fold_features(
    condition_train: np.ndarray,
    condition_test: np.ndarray,
    distance_train: np.ndarray,
    distance_test: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    centered = condition_train - condition_train.mean(0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:8]
    train_pca = centered @ components.T
    test_pca = (
        condition_test - condition_train.mean(0, keepdims=True)
    ) @ components.T
    train_pca, test_pca = standardize(train_pca, test_pca)
    train_distance, test_distance = standardize(
        distance_train[:, None], distance_test[:, None]
    )
    explained = float(
        np.square(singular[:8]).sum() / np.square(singular).sum()
    )
    return {
        "distance_only": (train_distance, test_distance, explained),
        "condition_pca8": (train_pca, test_pca, explained),
        "condition_pca8_plus_distance": (
            np.concatenate((train_pca, train_distance), axis=1),
            np.concatenate((test_pca, test_distance), axis=1),
            explained,
        ),
    }


def fit_linear_logistic(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    if len(np.unique(y)) != 2:
        raise RuntimeError("state-gate training fold lost a binary class")
    class_count = np.bincount(y.astype(np.int64), minlength=2)
    class_factor = len(y) / (2.0 * class_count)
    weight = sample_weight * class_factor[y.astype(np.int64)]
    xt = torch.from_numpy(x.astype(np.float64))
    yt = torch.from_numpy(y.astype(np.float64))
    wt = torch.from_numpy(weight.astype(np.float64))
    model = nn.Linear(x.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logit = model(xt).reshape(-1)
        bce = F.binary_cross_entropy_with_logits(logit, yt, reduction="none")
        loss = (bce * wt).sum() / wt.sum()
        loss = loss + 0.5 * 0.1 * model.weight.square().sum()
        loss.backward()
        return loss

    loss = float(optimizer.step(closure).detach())
    return (
        model.weight.detach().numpy().reshape(-1),
        float(model.bias.detach().item()),
        loss,
    )


def probability(x: np.ndarray, weight: np.ndarray, bias: float) -> np.ndarray:
    logit = np.clip(x @ weight + bias, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)


def auc_score(y: np.ndarray, score: np.ndarray) -> float | None:
    positive = int(y.sum())
    negative = len(y) - positive
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score, kind="stable")
    rank = np.empty(len(score), np.float64)
    rank[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((rank[y.astype(bool)].sum() - positive * (positive + 1) / 2) /
                 (positive * negative))


def quality(value: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(value[:, index].mean()),
            "p90": float(np.quantile(value[:, index], 0.9)),
        }
        for index, name in enumerate(METRICS)
    }


def screen(args: argparse.Namespace) -> None:
    result_path = args.output_dir / "result.json"
    selected_path = args.output_dir / "selected_index_top40.npz"
    if result_path.exists() or selected_path.exists():
        raise FileExistsError(result_path)
    freeze_path = args.output_dir / "challenger_freeze.json"
    challenger_path = args.output_dir / "frozen_challenger.npz"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["frozen_challenger_sha256"] != file_sha256(challenger_path):
        raise RuntimeError("frozen challenger SHA changed")
    with np.load(args.feature, allow_pickle=False) as feature:
        record = feature["record_key"].astype(str)
        bank = feature["bank"].astype(str)
        condition = feature["nomad_goal_condition"].astype(np.float64)
        distance = feature["nomad_goal_distance_prediction"].astype(np.float64)
    with np.load(challenger_path, allow_pickle=False) as payload:
        if not np.array_equal(record, payload["record_key"].astype(str)):
            raise RuntimeError("feature/frozen challenger record mismatch")
        challenger = payload["challenger_index"].astype(np.int64)
        eligible = payload["state_eligible"].astype(bool)

    route, targets, absolute = load_labels(record)
    row = np.arange(2000)
    chosen_delta = targets[row, challenger, :4]
    dual = ((chosen_delta[:, 0] < -1e-7) &
            (chosen_delta[:, 3] < -1e-7)).astype(np.int64)
    harm = ((chosen_delta[:, 0] > 1e-7) |
            (chosen_delta[:, 3] > 1e-7)).astype(np.int64)
    incumbent_absolute = absolute[:, 0]
    challenger_absolute = absolute[row, challenger]
    selected_by_variant = {
        variant: np.zeros(2000, dtype=np.int64) for variant in VARIANTS
    }
    score_by_variant = {
        variant: np.full(2000, np.nan, dtype=np.float32) for variant in VARIANTS
    }
    fold_fit: dict[str, dict[str, object]] = {variant: {} for variant in VARIANTS}
    for letter in "abcdefghijklmnopqrst":
        heldout = bank == letter
        train = ~heldout & eligible
        train_weight = route_balanced_weight(route[train], bank[train])
        feature_variants = fold_features(
            condition[train], condition[heldout], distance[train], distance[heldout]
        )
        for variant, (x_train, x_test, explained) in feature_variants.items():
            dual_weight, dual_bias, dual_loss = fit_linear_logistic(
                x_train, dual[train], train_weight
            )
            harm_weight, harm_bias, harm_loss = fit_linear_logistic(
                x_train, harm[train], train_weight
            )
            p_dual = probability(x_test, dual_weight, dual_bias)
            p_harm = probability(x_test, harm_weight, harm_bias)
            score = p_dual - p_harm
            heldout_index = np.flatnonzero(heldout)
            eligible_local = eligible[heldout]
            if int(eligible_local.sum()) < 40:
                raise RuntimeError(
                    f"bank {letter} has fewer than 40 eligible challengers"
                )
            order = np.argsort(-score[eligible_local], kind="stable")[:40]
            chosen = heldout_index[eligible_local][order]
            selected_by_variant[variant][chosen] = challenger[chosen]
            score_by_variant[variant][heldout_index] = np.where(
                eligible_local, score, -np.inf
            )
            fold_fit[variant][letter] = {
                "train_banks": 19,
                "train_states": int(train.sum()),
                "heldout_states": int(heldout.sum()),
                "input_dim": int(x_train.shape[1]),
                "condition_pca8_explained_variance": explained,
                "train_dual_rate": float(dual[train].mean()),
                "train_harm_rate": float(harm[train].mean()),
                "dual_loss": dual_loss,
                "harm_loss": harm_loss,
                "heldout_dual_auc": auc_score(dual[heldout], p_dual),
                "heldout_harm_auc": auc_score(harm[heldout], p_harm),
            }

    write_npz(
        selected_path,
        record_key=record,
        bank=bank,
        challenger_index=challenger,
        **{
            f"selected_{variant}": selected_by_variant[variant]
            for variant in VARIANTS
        },
        **{f"score_{variant}": score_by_variant[variant] for variant in VARIANTS},
    )

    variant_reports = {}
    for variant in VARIANTS:
        per_bank = {}
        for letter in "abcdefghijklmnopqrst":
            local = bank == letter
            selected = selected_by_variant[variant][local]
            promote = selected != 0
            count = int(promote.sum())
            local_dual = int(dual[local][promote].sum())
            local_harm = int(harm[local][promote].sum())
            chosen_absolute = incumbent_absolute[local].copy()
            chosen_absolute[promote] = challenger_absolute[local][promote]
            incumbent_quality = quality(incumbent_absolute[local])
            selected_quality = quality(chosen_absolute)
            quality_pass = all(
                selected_quality[name][summary]
                <= incumbent_quality[name][summary] + 1e-9
                for name in METRICS
                for summary in ("mean", "p90")
            )
            metrics = {
                "promotions": count,
                "coverage": count / 100.0,
                "dual": local_dual / count,
                "wilson": wilson_lower(local_dual, count),
                "harm": local_harm / count,
                "quality": selected_quality,
                "incumbent_quality": incumbent_quality,
                "quality_non_degradation": quality_pass,
            }
            metrics["safety_pass"] = bool(
                metrics["coverage"] >= 0.40
                and metrics["dual"] >= 0.90
                and metrics["wilson"] >= 0.80
                and metrics["harm"] <= 0.10
            )
            per_bank[letter] = metrics
        variant_reports[variant] = {
            "banks": per_bank,
            "fold_fit": fold_fit[variant],
            "all_20_safety_pass": all(
                value["safety_pass"] for value in per_bank.values()
            ),
            "all_20_quality_non_degradation": all(
                value["quality_non_degradation"] for value in per_bank.values()
            ),
            "worst": {
                "dual": min(value["dual"] for value in per_bank.values()),
                "wilson": min(value["wilson"] for value in per_bank.values()),
                "harm": max(value["harm"] for value in per_bank.values()),
                "dual_auc": min(
                    value["heldout_dual_auc"] or 0.0
                    for value in fold_fit[variant].values()
                ),
                "harm_auc": min(
                    value["heldout_harm_auc"] or 0.0
                    for value in fold_fit[variant].values()
                ),
            },
        }
    candidate_oracle = {}
    for letter in "abcdefghijklmnopqrst":
        local = bank == letter
        local_eligible = local & eligible
        helpful = int(dual[local_eligible].sum())
        harmful = int(harm[local_eligible].sum())
        eligible_count = int(local_eligible.sum())
        candidate_oracle[letter] = {
            "eligible_fixed_challenger_states": eligible_count,
            "fixed_challenger_dual_states": helpful,
            "fixed_challenger_harm_states": harmful,
            "best_possible_top40_dual": min(helpful, 40) / 40.0,
            "best_possible_top40_harm": max(
                0, 40 - (eligible_count - harmful)
            ) / 40.0,
        }
    feasible = [
        variant
        for variant, value in variant_reports.items()
        if value["all_20_safety_pass"]
    ]
    report = {
        "schema": "recon_nomad_goal_state_gate_v53",
        "status": (
            "native_nomad_goal_signal_fixed40_pass"
            if feasible
            else "native_nomad_goal_signal_fixed40_failed_stop"
        ),
        "development_only": True,
        "formal_claim_allowed": False,
        "diagnostic_is_online_deployable": False,
        "noncausal_reason": (
            "fixed top-40 ranking sees the full held-out bank feature batch; "
            "used only as a separability upper-bound diagnostic"
        ),
        "strict_lobo": True,
        "banks": 20,
        "states": 2000,
        "routes": int(len(np.unique(route))),
        "route_overlap_across_banks": False,
        "candidate_ranker_retrained": False,
        "challenger_freeze": str(freeze_path),
        "challenger_freeze_sha256": file_sha256(freeze_path),
        "frozen_challenger_sha256": file_sha256(challenger_path),
        "feature": str(args.feature),
        "feature_sha256": file_sha256(args.feature),
        "selected_index": str(selected_path),
        "selected_index_sha256": file_sha256(selected_path),
        "variants": variant_reports,
        "fixed_challenger_availability_oracle": candidate_oracle,
        "feasible_variants": feasible,
        "stop_rule_triggered": not feasible,
        "next_action": (
            "freeze a causal threshold before any new confirmation"
            if feasible
            else "stop; do not add another architecture or alter WM"
        ),
        "final_confirmation_reserve_accessed": False,
        "cem": False,
    }
    write_json(result_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "feasible_variants": feasible,
                "selected_index_sha256": report["selected_index_sha256"],
                "worst": {
                    variant: value["worst"]
                    for variant, value in variant_reports.items()
                },
                "failed_banks": {
                    variant: [
                        letter
                        for letter, metrics in value["banks"].items()
                        if not metrics["safety_pass"]
                    ]
                    for variant, value in variant_reports.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def main() -> None:
    args = parse_args()
    if args.stage == "freeze":
        freeze_challenger(args)
    else:
        screen(args)


if __name__ == "__main__":
    main()
