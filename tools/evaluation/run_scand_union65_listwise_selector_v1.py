#!/usr/bin/env python3
"""One direct listwise selector experiment on the frozen SCAND union65 table.

The experiment has one model family and one target.  It trains a shared
candidate utility network with a state-grouped soft ranking loss, keeps the
formal-v70 action inside every choice set as the zero-gain fallback, and uses
strict nested route-OOF for the only tunable policy value: a fallback margin.
Sealed validation is never read.
"""

from __future__ import annotations

import argparse
import hashlib
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


TOL = 1.0e-12
METRICS = ("ate", "rpe", "endpoint", "objective")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_folds(route: np.ndarray, folds: int) -> tuple[np.ndarray, dict[str, Any]]:
    unique, counts = np.unique(route.astype(str), return_counts=True)
    records = sorted(
        zip(unique.tolist(), counts.tolist()),
        key=lambda item: (-item[1], hashlib.sha256(item[0].encode()).hexdigest()),
    )
    loads = [0] * folds
    mapping: dict[str, int] = {}
    for name, count in records:
        fold = min(range(folds), key=lambda value: (loads[value], value))
        mapping[name] = fold
        loads[fold] += count
    assignment = np.asarray([mapping[name] for name in route.astype(str)], dtype=np.int8)
    return assignment, {
        "fold_state_loads": loads,
        "route_to_fold": dict(sorted(mapping.items())),
        "route_overlap": 0,
    }


class UtilityNetwork(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def choice_mask(data: dict[str, np.ndarray], states: np.ndarray) -> np.ndarray:
    mask = data["physical_support_pool"][states].astype(bool).copy()
    formal = data["formal_v70_index"][states].astype(np.int64)
    mask[np.arange(len(states)), formal] = True
    return mask


def state_weights(route: np.ndarray, states: np.ndarray) -> np.ndarray:
    local = route[states].astype(str)
    unique, counts = np.unique(local, return_counts=True)
    count = dict(zip(unique.tolist(), counts.tolist()))
    weight = np.asarray([1.0 / count[name] for name in local], dtype=np.float32)
    return weight / weight.mean()


def fit_normalizer(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    route: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature = data["runtime_pair_feature"][states].astype(np.float32)
    mask = choice_mask(data, states)
    per_state = state_weights(route, states)
    valid_count = np.maximum(mask.sum(axis=1), 1)
    row_weight = (per_state[:, None] / valid_count[:, None]) * mask
    weight = row_weight.reshape(-1).astype(np.float64)
    x = feature.reshape(-1, feature.shape[-1]).astype(np.float64)
    total = float(weight.sum())
    mean = (x * weight[:, None]).sum(axis=0) / total
    variance = (((x - mean) ** 2) * weight[:, None]).sum(axis=0) / total
    std = np.maximum(np.sqrt(variance), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32)


def target_utility(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    target = contract["target"]
    utility = (
        float(target["objective_gain_weight"])
        * data["objective_gain_vs_formal_v70_m"][states].astype(np.float32)
        + float(target["ate_gain_weight"])
        * data["ate_gain_vs_formal_v70_m"][states].astype(np.float32)
        - float(target["harm_penalty_m"])
        * data["harm_vs_formal_v70"][states].astype(np.float32)
    )
    formal = data["formal_v70_index"][states].astype(np.int64)
    utility[np.arange(len(states)), formal] = 0.0
    return utility


def fit_model(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    route: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[UtilityNetwork, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    model_config = contract["model"]
    mean, std = fit_normalizer(data, states, route)
    feature = data["runtime_pair_feature"][states].astype(np.float32)
    feature = ((feature - mean[None, None]) / std[None, None]).astype(np.float32)
    mask = choice_mask(data, states)
    utility = target_utility(data, states, contract)
    weight = state_weights(route, states)
    model = UtilityNetwork(feature.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    temperature = float(contract["target"]["softmax_temperature_m"])
    epochs = int(model_config["epochs"])
    batch_size = int(model_config["state_batch_size"])
    rng = np.random.default_rng(seed)
    x = torch.from_numpy(feature)
    mask_tensor = torch.from_numpy(mask)
    utility_tensor = torch.from_numpy(utility)
    weight_tensor = torch.from_numpy(weight)
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(states))
        weighted_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            xb = x[index].to(device)
            mb = mask_tensor[index].to(device)
            ub = utility_tensor[index].to(device)
            wb = weight_tensor[index].to(device)
            logits = model(xb).masked_fill(~mb, -1.0e9)
            target_logits = (ub / temperature).masked_fill(~mb, -1.0e9)
            soft_target = torch.softmax(target_logits, dim=1)
            per_state = -(soft_target * torch.log_softmax(logits, dim=1)).sum(dim=1)
            loss = (per_state * wb).sum() / wb.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu()) * float(wb.sum().cpu())
            total_weight += float(wb.sum().cpu())
        final_loss = weighted_loss / max(total_weight, 1.0)
    return model, mean, std, {
        "states": int(len(states)),
        "routes": int(len(np.unique(route[states]))),
        "last_epoch_route_balanced_listwise_loss": float(final_loss),
    }


def predict(
    model: UtilityNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    feature: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    shape = feature.shape[:2]
    x = ((feature.reshape(-1, feature.shape[-1]).astype(np.float32) - mean) / std).astype(np.float32)
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 8192):
            result.append(
                model(torch.from_numpy(x[start : start + 8192]).to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return np.concatenate(result).reshape(shape)


def top_challenger(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = choice_mask(data, states)
    formal = data["formal_v70_index"][states].astype(np.int64)
    challenger_mask = mask.copy()
    challenger_mask[np.arange(len(states)), formal] = False
    masked = np.where(challenger_mask, scores, -np.inf)
    candidate = np.argmax(masked, axis=1).astype(np.int64)
    advantage = (
        scores[np.arange(len(states)), candidate]
        - scores[np.arange(len(states)), formal]
    ).astype(np.float32)
    absent = ~challenger_mask.any(axis=1)
    candidate[absent] = formal[absent]
    advantage[absent] = -np.inf
    return candidate, advantage


def select_with_margin(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    margin: float,
) -> np.ndarray:
    formal = data["formal_v70_index"][states].astype(np.int64)
    challenger, advantage = top_challenger(scores, data, states)
    selected = formal.copy()
    replace = advantage > float(margin)
    selected[replace] = challenger[replace]
    return selected


def route_direction(
    gain: np.ndarray,
    route: np.ndarray,
) -> tuple[dict[str, float], float]:
    values = {
        name: float(gain[route == name].mean())
        for name in np.unique(route.astype(str))
    }
    positive = float(np.mean(np.asarray(list(values.values())) > 0.0))
    return dict(sorted(values.items())), positive


def selection_stats(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    selected: np.ndarray,
) -> dict[str, Any]:
    formal = data["formal_v70_index"][states].astype(np.int64)
    replacement = selected != formal
    row = states
    objective = data["objective_gain_vs_formal_v70_m"][row, selected].astype(np.float64)
    ate = data["ate_gain_vs_formal_v70_m"][row, selected].astype(np.float64)
    objective[~replacement] = 0.0
    ate[~replacement] = 0.0
    count = int(replacement.sum())
    dual = int((data["dual_vs_formal_v70"][row, selected] & replacement).sum())
    harm = int((data["harm_vs_formal_v70"][row, selected] & replacement).sum())
    routes, positive_fraction = route_direction(
        objective,
        data["route_group"][states].astype(str),
    )
    formal_metrics = data["true_metrics_m"][row, formal].astype(np.float64)
    selected_metrics = data["true_metrics_m"][row, selected].astype(np.float64)
    return {
        "replacements": count,
        "coverage": count / len(states),
        "objective_gain_m": float(objective.mean()),
        "ate_gain_m": float(ate.mean()),
        "dual_count": dual,
        "harm_count": harm,
        "dual_rate": dual / count if count else None,
        "harm_rate": harm / count if count else None,
        "positive_route_fraction": positive_fraction,
        "route_objective_gain_m": routes,
        "formal_mean_m": {
            name: float(formal_metrics[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "selected_mean_m": {
            name: float(selected_metrics[:, index].mean())
            for index, name in enumerate(METRICS)
        },
        "formal_p90_m": {
            name: float(np.quantile(formal_metrics[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
        "selected_p90_m": {
            name: float(np.quantile(selected_metrics[:, index], 0.9))
            for index, name in enumerate(METRICS)
        },
    }


def margin_for_coverage(advantage: np.ndarray, target: float) -> float:
    finite = advantage[np.isfinite(advantage)]
    if not len(finite):
        return 0.0
    positive = np.sort(finite[finite > 0.0])[::-1]
    requested = int(math.ceil(float(target) * len(advantage)))
    if requested <= 0:
        return float(np.nextafter(positive[0], np.inf)) if len(positive) else 0.0
    if requested >= len(positive):
        return 0.0
    cutoff = positive[requested - 1]
    return float(np.nextafter(cutoff, -np.inf))


def inner_curve_and_margin(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    contract: dict[str, Any],
) -> tuple[float, dict[str, Any], dict[str, dict[str, Any]]]:
    _, advantage = top_challenger(scores, data, states)
    coverages = [float(value) for value in contract["evaluation"]["coverage_curve"]]
    margins = {0.0}
    for coverage in coverages:
        margins.add(margin_for_coverage(advantage, coverage))
    records: dict[str, dict[str, Any]] = {}
    candidates = []
    rules = contract["inner_margin_selection"]
    for margin in sorted(margins):
        selected = select_with_margin(scores, data, states, margin)
        stats = selection_stats(data, states, selected)
        harm = stats["harm_rate"]
        feasible = bool(
            stats["replacements"] >= int(rules["minimum_replacements"])
            and stats["objective_gain_m"] > float(rules["minimum_all_state_objective_gain_m"])
            and stats["ate_gain_m"] > float(rules["minimum_all_state_ate_gain_m"])
            and harm is not None
            and harm <= float(rules["maximum_empirical_harm_rate"])
            and stats["positive_route_fraction"]
            >= float(rules["minimum_positive_route_fraction"])
        )
        key = f"{margin:.9g}"
        records[key] = {
            "margin": float(margin),
            "feasible": feasible,
            **{name: value for name, value in stats.items() if name != "route_objective_gain_m"},
        }
        if feasible:
            candidates.append((margin, stats))
    if not candidates:
        chosen_margin = 0.0
        chosen = selection_stats(
            data,
            states,
            select_with_margin(scores, data, states, 0.0),
        )
        reason = "no_inner_candidate_met_empirical_constraints_zero_margin_used"
    else:
        chosen_margin, chosen = sorted(
            candidates,
            key=lambda item: (
                -item[1]["objective_gain_m"],
                item[1]["harm_rate"],
                -item[1]["replacements"],
                item[0],
            ),
        )[0]
        reason = "maximum_inner_objective_gain_among_empirically_feasible_margins"
    return float(chosen_margin), {"reason": reason, **chosen}, records


def objective_oracle(
    data: dict[str, np.ndarray],
    states: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = choice_mask(data, states)
    gain = data["objective_gain_vs_formal_v70_m"][states].astype(np.float32)
    masked = np.where(mask, gain, -np.inf)
    selected = np.argmax(masked, axis=1).astype(np.int64)
    return selected, selection_stats(data, states, selected)


def ranking_stats(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    top_k: list[int],
) -> dict[str, Any]:
    mask = choice_mask(data, states)
    objective = data["objective_gain_vs_formal_v70_m"][states].astype(np.float32)
    masked_objective = np.where(mask, objective, -np.inf)
    oracle_index = np.argmax(masked_objective, axis=1)
    oracle_gain = masked_objective[np.arange(len(states)), oracle_index]
    opportunity = oracle_gain > 0.0
    ordered = np.argsort(np.where(mask, scores, -np.inf), axis=1)[:, ::-1]
    result = {}
    for k in top_k:
        local = ordered[:, :k]
        exact = (local == oracle_index[:, None]).any(axis=1)
        top_gain = np.max(
            np.take_along_axis(masked_objective, local, axis=1),
            axis=1,
        )
        ratio = np.divide(
            np.maximum(top_gain, 0.0),
            oracle_gain,
            out=np.ones_like(oracle_gain),
            where=oracle_gain > 0.0,
        )
        result[str(k)] = {
            "exact_oracle_candidate_recall": float(exact[opportunity].mean())
            if opportunity.any()
            else None,
            "mean_best_gain_fraction": float(ratio[opportunity].mean())
            if opportunity.any()
            else None,
            "opportunity_states": int(opportunity.sum()),
        }
    return result


def bootstrap_gain(
    gain: np.ndarray,
    route: np.ndarray,
    draws: int = 5000,
) -> dict[str, float]:
    names = np.unique(route.astype(str))
    indices = {name: np.flatnonzero(route.astype(str) == name) for name in names}
    rng = np.random.default_rng(20260724)
    value = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.choice(names, size=len(names), replace=True)
        joined = np.concatenate([indices[name] for name in sampled])
        value[draw] = gain[joined].mean()
    return {
        "mean_m": float(gain.mean()),
        "lcb95_m": float(np.quantile(value, 0.025)),
        "ucb95_m": float(np.quantile(value, 0.975)),
        "clusters": int(len(names)),
        "draws": draws,
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    table_path = args.table.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require("validation" not in str(table_path).lower(), "validation-like table path rejected")
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
    require(len(np.unique(data["route_group"].astype(str))) == 42, "route count changed")
    routes = data["route_group"].astype(str)
    folds = int(contract["evaluation"]["outer_route_folds"])
    fold, fold_audit = assign_folds(routes, folds)
    states_all = np.arange(len(routes))
    scores_oof = np.full(feature.shape[:2], np.nan, dtype=np.float32)
    selected_calibrated = data["formal_v70_index"].astype(np.int64).copy()
    selected_curve = {
        str(float(value)): data["formal_v70_index"].astype(np.int64).copy()
        for value in contract["evaluation"]["coverage_curve"]
    }
    outer_records = {}
    parameters: dict[str, np.ndarray] = {}
    seed = int(contract["model"]["seed"])
    started = time.perf_counter()
    for outer in range(folds):
        outer_train = states_all[fold != outer]
        outer_test = states_all[fold == outer]
        require(
            not (set(routes[outer_train]) & set(routes[outer_test])),
            "outer route leakage",
        )
        inner_scores = np.full(feature.shape[:2], np.nan, dtype=np.float32)
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
            model, mean, std, fit = fit_model(
                data,
                inner_train,
                routes,
                contract,
                device,
                seed + outer * 100 + inner,
            )
            inner_scores[inner_test] = predict(
                model,
                mean,
                std,
                feature[inner_test],
                device,
            )
            inner_fits.append({"inner_fold": inner, **fit})
            del model
        require(np.isfinite(inner_scores[outer_train]).all(), "inner OOF scores incomplete")
        margin, margin_stats, inner_curve = inner_curve_and_margin(
            inner_scores[outer_train],
            data,
            outer_train,
            contract,
        )
        model, mean, std, final_fit = fit_model(
            data,
            outer_train,
            routes,
            contract,
            device,
            seed + outer * 100 + 99,
        )
        outer_scores = predict(model, mean, std, feature[outer_test], device)
        scores_oof[outer_test] = outer_scores
        selected_calibrated[outer_test] = select_with_margin(
            outer_scores,
            data,
            outer_test,
            margin,
        )
        _, inner_advantage = top_challenger(
            inner_scores[outer_train],
            data,
            outer_train,
        )
        curve_margins = {}
        for value in contract["evaluation"]["coverage_curve"]:
            coverage = float(value)
            curve_margin = margin_for_coverage(inner_advantage, coverage)
            selected_curve[str(coverage)][outer_test] = select_with_margin(
                outer_scores,
                data,
                outer_test,
                curve_margin,
            )
            curve_margins[str(coverage)] = float(curve_margin)
        for name, tensor in model.state_dict().items():
            parameters[f"outer_{outer}__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"outer_{outer}__mean"] = mean
        parameters[f"outer_{outer}__std"] = std
        parameters[f"outer_{outer}__margin"] = np.asarray(margin, dtype=np.float32)
        outer_records[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "outer_route_overlap": 0,
            "inner_fits": inner_fits,
            "inner_selected_margin": margin,
            "inner_selected_margin_stats": {
                name: value
                for name, value in margin_stats.items()
                if name != "route_objective_gain_m"
            },
            "inner_margin_curve": inner_curve,
            "inner_coverage_curve_margins": curve_margins,
            "final_fit": final_fit,
        }
        print(
            canonical_json(
                {
                    "outer_fold": outer,
                    "margin": margin,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del model
    require(np.isfinite(scores_oof).all(), "outer OOF scores incomplete")
    pure_top1 = select_with_margin(scores_oof, data, states_all, 0.0)
    pure_stats = selection_stats(data, states_all, pure_top1)
    calibrated_stats = selection_stats(data, states_all, selected_calibrated)
    oracle_index, oracle_stats = objective_oracle(data, states_all)
    ranking = ranking_stats(
        scores_oof,
        data,
        states_all,
        [int(value) for value in contract["evaluation"]["top_k"]],
    )
    oracle_gain = float(oracle_stats["objective_gain_m"])
    pure_stats["oracle_gain_ratio"] = (
        pure_stats["objective_gain_m"] / oracle_gain if oracle_gain > 0.0 else None
    )
    calibrated_stats["oracle_gain_ratio"] = (
        calibrated_stats["objective_gain_m"] / oracle_gain
        if oracle_gain > 0.0
        else None
    )
    performance_harm_curve = {}
    for coverage, selected in selected_curve.items():
        stats = selection_stats(data, states_all, selected)
        performance_harm_curve[coverage] = {
            name: value
            for name, value in stats.items()
            if name
            in {
                "replacements",
                "coverage",
                "objective_gain_m",
                "ate_gain_m",
                "dual_count",
                "harm_count",
                "dual_rate",
                "harm_rate",
                "positive_route_fraction",
            }
        }
    formal = data["formal_v70_index"].astype(np.int64)
    rows = np.arange(len(formal))
    selected_gain = data["objective_gain_vs_formal_v70_m"][
        rows,
        selected_calibrated,
    ].astype(np.float64)
    selected_gain[selected_calibrated == formal] = 0.0
    bootstrap = bootstrap_gain(selected_gain, routes)
    gate = contract["prototype_go_gate"]
    harm_rate = calibrated_stats["harm_rate"]
    checks = {
        "objective_gain_exceeds_existing_4p33mm": calibrated_stats["objective_gain_m"]
        > float(gate["minimum_all_state_objective_gain_m"]),
        "ate_gain_positive": calibrated_stats["ate_gain_m"]
        > float(gate["minimum_all_state_ate_gain_m"]),
        "harm_not_uncontrolled": harm_rate is not None
        and harm_rate <= float(gate["maximum_empirical_harm_rate"]),
        "oracle_gain_ratio": calibrated_stats["oracle_gain_ratio"]
        is not None
        and calibrated_stats["oracle_gain_ratio"]
        >= float(gate["minimum_oracle_gain_ratio"]),
        "route_direction": calibrated_stats["positive_route_fraction"]
        >= float(gate["minimum_positive_route_fraction"]),
    }
    go = all(checks.values())
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
            utility_score=scores_oof,
            pure_top1_selected_index=pure_top1,
            calibrated_selected_index=selected_calibrated,
            objective_oracle_index=oracle_index,
            formal_v70_index=formal,
        )
    with parameter_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    result = {
        "schema": "scand_union65_listwise_selector_nested_route_oof_result_v1",
        "status": "complete_adaptive_spent_development_performance_prototype",
        "decision": (
            "go_freeze_selector_and_margin_worth_one_time_validation"
            if go
            else "no_go_current_351_runtime_representation_cannot_reliably_select_oracle_actions"
        ),
        "prototype_go": go,
        "prototype_checks": checks,
        "states": int(len(routes)),
        "route_groups": int(len(np.unique(routes))),
        "candidate_rows": int(np.prod(feature.shape[:2])),
        "feature_dimensions": int(feature.shape[-1]),
        "model_parameters_per_fold": int(
            sum(value.numel() for value in UtilityNetwork(feature.shape[-1]).parameters())
        ),
        "pure_top1_including_formal_fallback": pure_stats,
        "inner_calibrated_fallback_margin": calibrated_stats,
        "objective_oracle": oracle_stats,
        "ranking": ranking,
        "performance_harm_curve": performance_harm_curve,
        "route_cluster_bootstrap_objective_gain": bootstrap,
        "outer_folds": outer_records,
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
        "runtime_seconds": float(time.perf_counter() - started),
        "sealed_validation_opened": False,
        "formal_v70_mutated": False,
        "cem_mutated": False,
        "physical_support_pool_mutated": False,
    }
    write_json_atomic(result_path, result)
    summary = {
        "status": result["status"],
        "decision": result["decision"],
        "pure_top1_objective_gain_m": pure_stats["objective_gain_m"],
        "calibrated_objective_gain_m": calibrated_stats["objective_gain_m"],
        "calibrated_ate_gain_m": calibrated_stats["ate_gain_m"],
        "calibrated_harm_rate": calibrated_stats["harm_rate"],
        "calibrated_replacements": calibrated_stats["replacements"],
        "oracle_gain_m": oracle_gain,
        "oracle_gain_ratio": calibrated_stats["oracle_gain_ratio"],
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
    }
    print(canonical_json(summary))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
