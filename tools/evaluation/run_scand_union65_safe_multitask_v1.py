#!/usr/bin/env python3
"""Strict nested route-OOF safe multi-task selector on the frozen union65 table.

This experiment tests one untried hypothesis: candidate-set context and direct
multi-task prediction of objective gain, ATE gain, harm, and dual benefit can
turn the useful coarse union65 ranking into a conservative selector.  Policy
selection uses inner route-OOF labels only.  If no inner policy satisfies all
safety and generalization constraints, the corresponding outer fold uses
formal-v70 for every state.

The sealed validation split is never read.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools" / "evaluation"),
]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from run_scand_union65_listwise_selector_v1 import (  # noqa: E402
    assign_folds,
    bootstrap_gain,
    choice_mask,
    fit_normalizer,
    load_npz,
    objective_oracle,
    ranking_stats,
    read_json,
    require,
    selection_stats,
    state_weights,
    write_json_atomic,
)


HEADS = ("objective_gain_m", "ate_gain_m", "harm_probability", "dual_probability")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the same leakage checks with one epoch/member and reduced bootstraps.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SetContextMultiTaskNetwork(nn.Module):
    """Shared candidate encoder with permutation-invariant state context."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dimension, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 3, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Linear(32, 4),
        )

    def forward(
        self,
        feature: torch.Tensor,
        mask: torch.Tensor,
        formal_index: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(feature)
        weight = mask.to(encoded.dtype).unsqueeze(-1)
        context = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        batch = torch.arange(len(feature), device=feature.device)
        formal = encoded[batch, formal_index]
        state_context = torch.cat([context, formal], dim=-1)
        expanded = state_context[:, None, :].expand(-1, encoded.shape[1], -1)
        return self.head(torch.cat([encoded, expanded], dim=-1))


def masked_state_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def fit_model(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    routes: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
    epochs_override: int | None = None,
) -> tuple[SetContextMultiTaskNetwork, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    config = contract["model"]
    loss_config = config["loss"]
    mean, std = fit_normalizer(data, states, routes)
    feature = data["runtime_pair_feature"][states].astype(np.float32)
    feature = ((feature - mean[None, None]) / std[None, None]).astype(np.float32)
    mask = choice_mask(data, states)
    formal = data["formal_v70_index"][states].astype(np.int64)
    objective_scale = float(config["objective_gain_scale_m"])
    ate_scale = float(config["ate_gain_scale_m"])
    objective = (
        data["objective_gain_vs_formal_v70_m"][states].astype(np.float32)
        / objective_scale
    )
    ate = data["ate_gain_vs_formal_v70_m"][states].astype(np.float32) / ate_scale
    harm = data["harm_vs_formal_v70"][states].astype(np.float32)
    dual = data["dual_vs_formal_v70"][states].astype(np.float32)
    route_weight = state_weights(routes, states)

    model = SetContextMultiTaskNetwork(feature.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(epochs_override or config["epochs"])
    batch_size = int(config["state_batch_size"])
    temperature = float(loss_config["softmax_temperature_m"])
    harm_penalty = float(loss_config["harm_penalty_m"])
    rng = np.random.default_rng(seed)

    tensors = {
        "feature": torch.from_numpy(feature),
        "mask": torch.from_numpy(mask),
        "formal": torch.from_numpy(formal),
        "objective": torch.from_numpy(objective),
        "ate": torch.from_numpy(ate),
        "harm": torch.from_numpy(harm),
        "dual": torch.from_numpy(dual),
        "weight": torch.from_numpy(route_weight),
    }
    final = {
        "total": math.nan,
        "listwise": math.nan,
        "objective": math.nan,
        "ate": math.nan,
        "harm": math.nan,
        "dual": math.nan,
    }
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(states))
        totals = {name: 0.0 for name in final}
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            xb = tensors["feature"][index].to(device)
            mb = tensors["mask"][index].to(device)
            fb = tensors["formal"][index].to(device)
            ob = tensors["objective"][index].to(device)
            ab = tensors["ate"][index].to(device)
            hb = tensors["harm"][index].to(device)
            db = tensors["dual"][index].to(device)
            wb = tensors["weight"][index].to(device)

            raw = model(xb, mb, fb)
            predicted_objective_m = raw[..., 0] * objective_scale
            predicted_ate_m = raw[..., 1] * ate_scale
            predicted_harm = torch.sigmoid(raw[..., 2])
            predicted_utility = (
                0.7 * predicted_objective_m
                + 0.3 * predicted_ate_m
                - harm_penalty * predicted_harm
            )
            true_utility = (
                0.7 * ob * objective_scale
                + 0.3 * ab * ate_scale
                - harm_penalty * hb
            )
            prediction_logits = (predicted_utility / temperature).masked_fill(
                ~mb, -1.0e9
            )
            target_logits = (true_utility / temperature).masked_fill(~mb, -1.0e9)
            soft_target = torch.softmax(target_logits, dim=1)
            listwise_per_state = -(
                soft_target * torch.log_softmax(prediction_logits, dim=1)
            ).sum(dim=1)

            objective_element = F.smooth_l1_loss(
                raw[..., 0], ob, reduction="none", beta=1.0
            )
            ate_element = F.smooth_l1_loss(
                raw[..., 1], ab, reduction="none", beta=1.0
            )
            harm_element = F.binary_cross_entropy_with_logits(
                raw[..., 2], hb, reduction="none"
            )
            dual_element = F.binary_cross_entropy_with_logits(
                raw[..., 3], db, reduction="none"
            )
            components = {
                "listwise": listwise_per_state,
                "objective": masked_state_mean(objective_element, mb),
                "ate": masked_state_mean(ate_element, mb),
                "harm": masked_state_mean(harm_element, mb),
                "dual": masked_state_mean(dual_element, mb),
            }
            per_state = (
                float(loss_config["listwise_weight"]) * components["listwise"]
                + float(loss_config["objective_regression_weight"])
                * components["objective"]
                + float(loss_config["ate_regression_weight"]) * components["ate"]
                + float(loss_config["harm_bce_weight"]) * components["harm"]
                + float(loss_config["dual_bce_weight"]) * components["dual"]
            )
            loss = (per_state * wb).sum() / wb.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            local_weight = float(wb.sum().detach().cpu())
            totals["total"] += float(loss.detach().cpu()) * local_weight
            for name, value in components.items():
                weighted = (value * wb).sum() / wb.sum()
                totals[name] += float(weighted.detach().cpu()) * local_weight
            total_weight += local_weight
        final = {name: value / max(total_weight, 1.0) for name, value in totals.items()}

    return model, mean, std, {
        "states": int(len(states)),
        "routes": int(len(np.unique(routes[states]))),
        "epochs": epochs,
        **{f"last_epoch_{name}_loss": float(value) for name, value in final.items()},
    }


def predict_model(
    model: SetContextMultiTaskNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    feature: np.ndarray,
    formal_index: np.ndarray,
    mask: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    config = contract["model"]
    normalized = ((feature.astype(np.float32) - mean) / std).astype(np.float32)
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), 64):
            end = min(start + 64, len(normalized))
            raw = model(
                torch.from_numpy(normalized[start:end]).to(device),
                torch.from_numpy(mask[start:end]).to(device),
                torch.from_numpy(formal_index[start:end]).to(device),
            )
            raw = raw.cpu().numpy().astype(np.float32)
            raw[..., 0] *= float(config["objective_gain_scale_m"])
            raw[..., 1] *= float(config["ate_gain_scale_m"])
            raw[..., 2:] = 1.0 / (1.0 + np.exp(-np.clip(raw[..., 2:], -30.0, 30.0)))
            result.append(raw)
    prediction = np.concatenate(result, axis=0)
    row = np.arange(len(prediction))
    prediction[row, formal_index, :] = 0.0
    return prediction


def fit_predict_ensemble(
    data: dict[str, np.ndarray],
    train_states: np.ndarray,
    test_states: np.ndarray,
    routes: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
    seed_base: int,
    ensemble_size: int,
    epochs_override: int | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    predictions = []
    fits = []
    parameters = []
    feature = data["runtime_pair_feature"][test_states].astype(np.float32)
    formal = data["formal_v70_index"][test_states].astype(np.int64)
    mask = choice_mask(data, test_states)
    for member in range(ensemble_size):
        seed = seed_base + member
        model, mean, std, fit = fit_model(
            data,
            train_states,
            routes,
            contract,
            device,
            seed,
            epochs_override=epochs_override,
        )
        predictions.append(
            predict_model(
                model,
                mean,
                std,
                feature,
                formal,
                mask,
                contract,
                device,
            )
        )
        fits.append({"member": member, "seed": seed, **fit})
        state = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in model.state_dict().items()
        }
        state["feature_mean"] = mean
        state["feature_std"] = std
        parameters.append(state)
        del model
    stacked = np.stack(predictions, axis=0)
    return (
        stacked.mean(axis=0).astype(np.float32),
        stacked.std(axis=0).astype(np.float32),
        fits,
        parameters,
    )


def conservative_candidates(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    beta: float,
    harm_limit: float,
    contract: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = contract["model"]["loss"]
    requirements = contract["policy_grid"]["candidate_requirements"]
    mask = choice_mask(data, states)
    formal = data["formal_v70_index"][states].astype(np.int64)
    objective_lcb = mean[..., 0] - beta * std[..., 0]
    ate_lcb = mean[..., 1] - beta * std[..., 1]
    harm_ucb = np.clip(mean[..., 2] + beta * std[..., 2], 0.0, 1.0)
    uncertainty = (
        0.7 * std[..., 0]
        + 0.3 * std[..., 1]
        + float(config["harm_penalty_m"]) * std[..., 2]
    )
    conservative_utility = (
        0.7 * mean[..., 0]
        + 0.3 * mean[..., 1]
        - float(config["harm_penalty_m"]) * mean[..., 2]
        - beta * uncertainty
    )
    challenger_mask = (
        mask
        & (objective_lcb > float(requirements["objective_gain_lcb_m"]))
        & (ate_lcb > float(requirements["ate_gain_lcb_m"]))
        & (harm_ucb <= harm_limit)
        & (
            conservative_utility
            > float(requirements["conservative_utility_advantage_m"])
        )
    )
    challenger_mask[np.arange(len(states)), formal] = False
    masked = np.where(challenger_mask, conservative_utility, -np.inf)
    candidate = np.argmax(masked, axis=1).astype(np.int64)
    advantage = masked[np.arange(len(states)), candidate].astype(np.float32)
    eligible = challenger_mask.any(axis=1)
    candidate[~eligible] = formal[~eligible]
    advantage[~eligible] = -np.inf
    return candidate, advantage, eligible


def threshold_for_coverage(
    advantage: np.ndarray,
    target_coverage: float,
) -> float:
    positive = np.sort(advantage[np.isfinite(advantage) & (advantage > 0.0)])[::-1]
    requested = int(math.ceil(target_coverage * len(advantage)))
    if not len(positive) or requested <= 0:
        return float("inf")
    if requested >= len(positive):
        return 0.0
    return float(np.nextafter(positive[requested - 1], -np.inf))


def apply_policy(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    policy: dict[str, Any],
    contract: dict[str, Any],
) -> np.ndarray:
    formal = data["formal_v70_index"][states].astype(np.int64)
    if policy.get("fallback", False):
        return formal.copy()
    candidate, advantage, eligible = conservative_candidates(
        mean,
        std,
        data,
        states,
        float(policy["uncertainty_beta"]),
        float(policy["harm_probability_limit"]),
        contract,
    )
    selected = formal.copy()
    replace = eligible & (advantage > float(policy["advantage_threshold_m"]))
    selected[replace] = candidate[replace]
    return selected


def selected_gain(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    selected: np.ndarray,
    key: str,
) -> np.ndarray:
    formal = data["formal_v70_index"][states].astype(np.int64)
    gain = data[key][states, selected].astype(np.float64)
    gain[selected == formal] = 0.0
    return gain


def positive_route_fraction(gain: np.ndarray, routes: np.ndarray) -> float:
    values = [gain[routes == name].mean() for name in np.unique(routes.astype(str))]
    return float(np.mean(np.asarray(values) > 0.0))


def fast_cluster_bootstrap(
    gain: np.ndarray,
    routes: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float]:
    names, inverse = np.unique(routes.astype(str), return_inverse=True)
    count = np.bincount(inverse).astype(np.float64)
    total = np.bincount(inverse, weights=gain.astype(np.float64))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(names), size=(draws, len(names)))
    value = total[sampled].sum(axis=1) / count[sampled].sum(axis=1)
    return {
        "mean_m": float(gain.mean()),
        "lcb95_m": float(np.quantile(value, 0.025)),
        "ucb95_m": float(np.quantile(value, 0.975)),
        "clusters": int(len(names)),
        "draws": int(draws),
    }


def augmented_stats(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    stats = selection_stats(data, states, selected)
    ate = selected_gain(
        data, states, selected, "ate_gain_vs_formal_v70_m"
    )
    stats["positive_ate_route_fraction"] = positive_route_fraction(
        ate, data["route_group"][states].astype(str)
    )
    return stats


def choose_inner_policy(
    mean: np.ndarray,
    std: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    contract: dict[str, Any],
    smoke: bool,
    outer_fold: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    grid = contract["policy_grid"]
    rules = contract["inner_policy_selection"]
    routes = data["route_group"][states].astype(str)
    draws = 100 if smoke else int(rules["bootstrap_draws"])
    records = []
    feasible = []
    for beta in [float(value) for value in grid["uncertainty_betas"]]:
        for harm_limit in [
            float(value) for value in grid["harm_probability_limits"]
        ]:
            _, advantage, _ = conservative_candidates(
                mean, std, data, states, beta, harm_limit, contract
            )
            for coverage in [float(value) for value in grid["target_coverages"]]:
                threshold = threshold_for_coverage(advantage, coverage)
                policy = {
                    "fallback": False,
                    "uncertainty_beta": beta,
                    "harm_probability_limit": harm_limit,
                    "target_inner_coverage": coverage,
                    "advantage_threshold_m": threshold,
                }
                selected = apply_policy(
                    mean, std, data, states, policy, contract
                )
                stats = augmented_stats(data, states, selected)
                harm = stats["harm_rate"]
                basic = bool(
                    stats["replacements"] >= int(rules["minimum_replacements"])
                    and stats["objective_gain_m"]
                    > float(rules["minimum_all_state_objective_gain_m"])
                    and stats["ate_gain_m"]
                    > float(rules["minimum_all_state_ate_gain_m"])
                    and harm is not None
                    and harm <= float(rules["maximum_empirical_harm_rate"])
                    and stats["positive_route_fraction"]
                    >= float(rules["minimum_positive_objective_route_fraction"])
                    and stats["positive_ate_route_fraction"]
                    >= float(rules["minimum_positive_ate_route_fraction"])
                )
                objective_bootstrap = None
                ate_bootstrap = None
                if basic:
                    objective_gain = selected_gain(
                        data,
                        states,
                        selected,
                        "objective_gain_vs_formal_v70_m",
                    )
                    ate_gain = selected_gain(
                        data,
                        states,
                        selected,
                        "ate_gain_vs_formal_v70_m",
                    )
                    objective_bootstrap = fast_cluster_bootstrap(
                        objective_gain,
                        routes,
                        draws,
                        20260725 + outer_fold * 1000,
                    )
                    ate_bootstrap = fast_cluster_bootstrap(
                        ate_gain,
                        routes,
                        draws,
                        20260726 + outer_fold * 1000,
                    )
                bootstrap_ok = bool(
                    basic
                    and (
                        not rules[
                            "require_objective_route_cluster_bootstrap_lcb_positive"
                        ]
                        or objective_bootstrap["lcb95_m"] > 0.0
                    )
                    and (
                        not rules[
                            "require_ate_route_cluster_bootstrap_lcb_positive"
                        ]
                        or ate_bootstrap["lcb95_m"] > 0.0
                    )
                )
                compact = {
                    **policy,
                    "feasible": bootstrap_ok,
                    "replacements": stats["replacements"],
                    "coverage": stats["coverage"],
                    "objective_gain_m": stats["objective_gain_m"],
                    "ate_gain_m": stats["ate_gain_m"],
                    "harm_rate": harm,
                    "positive_objective_route_fraction": stats[
                        "positive_route_fraction"
                    ],
                    "positive_ate_route_fraction": stats[
                        "positive_ate_route_fraction"
                    ],
                    "objective_bootstrap_lcb95_m": (
                        objective_bootstrap["lcb95_m"]
                        if objective_bootstrap is not None
                        else None
                    ),
                    "ate_bootstrap_lcb95_m": (
                        ate_bootstrap["lcb95_m"]
                        if ate_bootstrap is not None
                        else None
                    ),
                }
                records.append(compact)
                if bootstrap_ok:
                    feasible.append((policy, compact))
    if not feasible:
        fallback = {
            "fallback": True,
            "reason": "no_inner_route_oof_policy_met_all_constraints",
        }
        selected = apply_policy(mean, std, data, states, fallback, contract)
        return fallback, augmented_stats(data, states, selected), records
    policy, compact = sorted(
        feasible,
        key=lambda item: (
            -item[1]["ate_gain_m"],
            -item[1]["objective_gain_m"],
            item[1]["harm_rate"],
            -item[1]["replacements"],
            item[0]["uncertainty_beta"],
        ),
    )[0]
    policy = {
        **policy,
        "reason": "best_ate_then_objective_among_inner_route_oof_safe_policies",
    }
    selected = apply_policy(mean, std, data, states, policy, contract)
    return policy, augmented_stats(data, states, selected), records


def ate_oracle(
    data: dict[str, np.ndarray],
    states: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = choice_mask(data, states)
    gain = data["ate_gain_vs_formal_v70_m"][states].astype(np.float32)
    selected = np.argmax(np.where(mask, gain, -np.inf), axis=1).astype(np.int64)
    return selected, augmented_stats(data, states, selected)


def utility_score(mean: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    harm_penalty = float(contract["model"]["loss"]["harm_penalty_m"])
    return (
        0.7 * mean[..., 0]
        + 0.3 * mean[..., 1]
        - harm_penalty * mean[..., 2]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    table_path = args.table.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require("validation" not in str(table_path).lower(), "validation-like path rejected")
    contract = read_json(contract_path)
    require(
        contract.get("status") == "frozen_before_strict_nested_route_oof",
        "contract not frozen",
    )
    require(
        sha256_file(table_path) == contract["table"]["sha256"],
        "frozen 351-feature table hash changed",
    )
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    data = load_npz(table_path)
    feature = data["runtime_pair_feature"].astype(np.float32)
    require(feature.shape == (800, 65, 351), f"table shape changed: {feature.shape}")
    routes = data["route_group"].astype(str)
    require(len(np.unique(routes)) == 42, "route count changed")

    folds = int(contract["evaluation"]["outer_route_folds"])
    fold, fold_audit = assign_folds(routes, folds)
    states_all = np.arange(len(routes))
    heads_mean_oof = np.full((*feature.shape[:2], 4), np.nan, dtype=np.float32)
    heads_std_oof = np.full_like(heads_mean_oof, np.nan)
    selected_safe = data["formal_v70_index"].astype(np.int64).copy()
    outer_records = {}
    parameters: dict[str, np.ndarray] = {}
    ensemble_size = 1 if args.smoke else int(contract["model"]["ensemble_size"])
    epochs_override = 1 if args.smoke else None
    seed = int(contract["model"]["seed"])
    started = time.perf_counter()

    for outer in range(folds):
        outer_train = states_all[fold != outer]
        outer_test = states_all[fold == outer]
        require(
            not (set(routes[outer_train]) & set(routes[outer_test])),
            "outer route leakage",
        )
        inner_mean = np.full((*feature.shape[:2], 4), np.nan, dtype=np.float32)
        inner_std = np.full_like(inner_mean, np.nan)
        inner_fits = []
        for inner in range(folds):
            if inner == outer:
                continue
            inner_train = states_all[(fold != outer) & (fold != inner)]
            inner_test = states_all[fold == inner]
            require(
                not (set(routes[inner_train]) & set(routes[inner_test])),
                "inner route leakage",
            )
            predicted_mean, predicted_std, fits, _ = fit_predict_ensemble(
                data,
                inner_train,
                inner_test,
                routes,
                contract,
                device,
                seed + outer * 10000 + inner * 100,
                ensemble_size,
                epochs_override,
            )
            inner_mean[inner_test] = predicted_mean
            inner_std[inner_test] = predicted_std
            inner_fits.append({"inner_fold": inner, "members": fits})
        require(
            np.isfinite(inner_mean[outer_train]).all(),
            "inner OOF mean predictions incomplete",
        )
        require(
            np.isfinite(inner_std[outer_train]).all(),
            "inner OOF uncertainty incomplete",
        )
        policy, policy_stats, policy_grid = choose_inner_policy(
            inner_mean[outer_train],
            inner_std[outer_train],
            data,
            outer_train,
            contract,
            args.smoke,
            outer,
        )
        predicted_mean, predicted_std, final_fits, final_parameters = (
            fit_predict_ensemble(
                data,
                outer_train,
                outer_test,
                routes,
                contract,
                device,
                seed + outer * 10000 + 9900,
                ensemble_size,
                epochs_override,
            )
        )
        heads_mean_oof[outer_test] = predicted_mean
        heads_std_oof[outer_test] = predicted_std
        selected_safe[outer_test] = apply_policy(
            predicted_mean,
            predicted_std,
            data,
            outer_test,
            policy,
            contract,
        )
        for member, state in enumerate(final_parameters):
            for name, value in state.items():
                parameters[f"outer_{outer}__member_{member}__{name}"] = value
        outer_records[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_route_overlap": 0,
            "inner_fits": inner_fits,
            "inner_selected_policy": policy,
            "inner_selected_policy_stats": {
                name: value
                for name, value in policy_stats.items()
                if name != "route_objective_gain_m"
            },
            "inner_policy_grid": policy_grid,
            "final_fits": final_fits,
            "outer_test_selection": augmented_stats(
                data, outer_test, selected_safe[outer_test]
            ),
        }
        print(
            canonical_json(
                {
                    "outer_fold": outer,
                    "policy": policy,
                    "outer_replacements": outer_records[str(outer)][
                        "outer_test_selection"
                    ]["replacements"],
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    require(np.isfinite(heads_mean_oof).all(), "outer OOF means incomplete")
    require(np.isfinite(heads_std_oof).all(), "outer OOF uncertainties incomplete")
    scores = utility_score(heads_mean_oof, contract)
    formal = data["formal_v70_index"].astype(np.int64)
    mask = choice_mask(data, states_all)
    pure = np.argmax(np.where(mask, scores, -np.inf), axis=1).astype(np.int64)
    pure_stats = augmented_stats(data, states_all, pure)
    safe_stats = augmented_stats(data, states_all, selected_safe)
    objective_oracle_index, objective_oracle_stats = objective_oracle(
        data, states_all
    )
    ate_oracle_index, ate_oracle_stats = ate_oracle(data, states_all)
    ranking = ranking_stats(
        scores,
        data,
        states_all,
        [int(value) for value in contract["evaluation"]["top_k"]],
    )
    objective_gain = selected_gain(
        data,
        states_all,
        selected_safe,
        "objective_gain_vs_formal_v70_m",
    )
    ate_gain = selected_gain(
        data,
        states_all,
        selected_safe,
        "ate_gain_vs_formal_v70_m",
    )
    final_draws = (
        200 if args.smoke else int(contract["evaluation"]["final_bootstrap_draws"])
    )
    objective_bootstrap = fast_cluster_bootstrap(
        objective_gain, routes, final_draws, 20260725
    )
    ate_bootstrap = fast_cluster_bootstrap(
        ate_gain, routes, final_draws, 20260726
    )
    # Preserve comparability with the previous runner's exact bootstrap as an audit.
    legacy_objective_bootstrap = bootstrap_gain(
        objective_gain, routes, draws=final_draws
    )
    ate_oracle_gain = float(ate_oracle_stats["ate_gain_m"])
    ate_oracle_ratio = (
        safe_stats["ate_gain_m"] / ate_oracle_gain if ate_oracle_gain > 0.0 else None
    )
    gate = contract["prototype_go_gate"]
    harm_rate = safe_stats["harm_rate"]
    checks = {
        "minimum_replacements": safe_stats["replacements"]
        >= int(gate["minimum_replacements"]),
        "objective_gain_positive": safe_stats["objective_gain_m"]
        > float(gate["minimum_all_state_objective_gain_m"]),
        "ate_gain_positive": safe_stats["ate_gain_m"]
        > float(gate["minimum_all_state_ate_gain_m"]),
        "harm_rate_controlled": harm_rate is not None
        and harm_rate <= float(gate["maximum_empirical_harm_rate"]),
        "objective_route_direction": safe_stats["positive_route_fraction"]
        >= float(gate["minimum_positive_objective_route_fraction"]),
        "ate_route_direction": safe_stats["positive_ate_route_fraction"]
        >= float(gate["minimum_positive_ate_route_fraction"]),
        "objective_route_cluster_lcb_positive": (
            not gate["require_objective_route_cluster_bootstrap_lcb_positive"]
            or objective_bootstrap["lcb95_m"] > 0.0
        ),
        "ate_route_cluster_lcb_positive": (
            not gate["require_ate_route_cluster_bootstrap_lcb_positive"]
            or ate_bootstrap["lcb95_m"] > 0.0
        ),
        "ate_oracle_gain_ratio": ate_oracle_ratio is not None
        and ate_oracle_ratio >= float(gate["minimum_ate_oracle_gain_ratio"]),
        "ate_p90_not_worse": (
            not gate["require_ate_p90_not_worse"]
            or safe_stats["selected_p90_m"]["ate"]
            <= safe_stats["formal_p90_m"]["ate"] + 1.0e-12
        ),
    }
    go = bool(not args.smoke and all(checks.values()))

    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "outer_route_oof_predictions.npz"
    parameter_path = output / "frozen_outer_models.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"],
            route_group=routes,
            outer_fold=fold,
            head_names=np.asarray(HEADS),
            heads_mean=heads_mean_oof,
            heads_std=heads_std_oof,
            utility_score=scores,
            pure_multitask_selected_index=pure,
            safe_selected_index=selected_safe,
            objective_oracle_index=objective_oracle_index,
            ate_oracle_index=ate_oracle_index,
            formal_v70_index=formal,
        )
    with parameter_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    helper_path = ROOT / "tools" / "evaluation" / "run_scand_union65_listwise_selector_v1.py"
    result = {
        "schema": "scand_union65_safe_multitask_nested_route_oof_result_v1",
        "status": (
            "complete_smoke_not_for_decision"
            if args.smoke
            else "complete_adaptive_spent_development_performance_prototype"
        ),
        "decision": (
            "smoke_only"
            if args.smoke
            else (
                "go_safe_multitask_selector_reliably_exceeds_formal_v70_on_route_oof"
                if go
                else "no_go_this_safe_multitask_selector_not_reliably_above_formal_v70"
            )
        ),
        "prototype_go": go,
        "prototype_checks": checks,
        "smoke": bool(args.smoke),
        "states": int(len(routes)),
        "route_groups": int(len(np.unique(routes))),
        "candidate_rows": int(np.prod(feature.shape[:2])),
        "feature_dimensions": int(feature.shape[-1]),
        "model_parameters_per_member": int(
            sum(
                value.numel()
                for value in SetContextMultiTaskNetwork(feature.shape[-1]).parameters()
            )
        ),
        "ensemble_size": ensemble_size,
        "pure_multitask_top1": pure_stats,
        "inner_oof_safe_policy": safe_stats,
        "objective_oracle": objective_oracle_stats,
        "ate_oracle": ate_oracle_stats,
        "ate_oracle_gain_ratio": ate_oracle_ratio,
        "ranking": ranking,
        "route_cluster_bootstrap_objective_gain": objective_bootstrap,
        "route_cluster_bootstrap_ate_gain": ate_bootstrap,
        "legacy_route_cluster_bootstrap_objective_gain": legacy_objective_bootstrap,
        "outer_folds": outer_records,
        "fold_audit": fold_audit,
        "elapsed_seconds": float(time.perf_counter() - started),
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "table": {
                "path": str(table_path),
                "sha256": sha256_file(table_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "helper_runner": {
                "path": str(helper_path),
                "sha256": sha256_file(helper_path),
            },
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "frozen_outer_models": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
            },
        },
        "sealed_validation_opened": False,
    }
    write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "decision": result["decision"],
                "safe_objective_gain_m": safe_stats["objective_gain_m"],
                "safe_ate_gain_m": safe_stats["ate_gain_m"],
                "safe_harm_rate": safe_stats["harm_rate"],
                "objective_lcb95_m": objective_bootstrap["lcb95_m"],
                "ate_lcb95_m": ate_bootstrap["lcb95_m"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
