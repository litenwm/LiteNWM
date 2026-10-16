#!/usr/bin/env python3
"""Test one structured candidate-vs-formal temporal representation.

The experiment is deliberately limited to the already-spent four-route SCAND
development pilot.  It answers one feasibility question before any new route
collection: does preserving the horizon and three-seed order-statistic axes
improve route-held-out identification of candidates that safely beat the
frozen formal-v70 action?

This experiment is not eligible to open sealed validation or promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.planning import build_proposal_tail_temporal_pair_v1 as pair  # noqa: E402


GAIN_SCALE_M = 0.05
TOLERANCE = 1.0e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--temporal-features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_formal(value: np.ndarray, formal_index: np.ndarray) -> np.ndarray:
    selected = value[np.arange(len(value)), formal_index]
    return np.expand_dims(selected, axis=1)


def triple(value: np.ndarray, formal_index: np.ndarray) -> np.ndarray:
    candidate = np.asarray(value, dtype=np.float32).reshape(*value.shape[:2], -1)
    formal = expand_formal(candidate, formal_index)
    formal = np.broadcast_to(formal, candidate.shape)
    return np.concatenate((candidate, formal, candidate - formal), axis=-1)


def build_features(
    temporal: dict[str, np.ndarray],
    formal_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    order = temporal["ltwm_temporal_order_statistics"].astype(np.float32)
    if order.shape[2:] != (3, 7, 4):
        raise RuntimeError(f"seed-order/horizon contract changed: {order.shape}")
    # [state, candidate, seed-order, metric, horizon] -> horizon tokens.
    candidate_temporal = order.transpose(0, 1, 4, 2, 3).reshape(
        *order.shape[:2], 4, 21
    )
    formal_temporal = expand_formal(candidate_temporal, formal_index)
    formal_temporal = np.broadcast_to(formal_temporal, candidate_temporal.shape)

    actions = temporal["actions_h4_m_rad"].astype(np.float32)
    if actions.shape[2:] != (4, 3):
        raise RuntimeError(f"action horizon contract changed: {actions.shape}")
    formal_actions = expand_formal(actions, formal_index)
    formal_actions = np.broadcast_to(formal_actions, actions.shape)
    token = np.concatenate(
        (
            candidate_temporal,
            formal_temporal,
            candidate_temporal - formal_temporal,
            actions,
            formal_actions,
            actions - formal_actions,
        ),
        axis=-1,
    ).astype(np.float32)

    context_blocks = (
        triple(temporal["action_structure"], formal_index),
        triple(temporal["ate_by_seed"], formal_index),
        triple(temporal["rtp_by_seed"], formal_index),
        triple(temporal["support_by_seed"].astype(np.float32), formal_index),
        triple(temporal["scorer_v2_aux"], formal_index),
        triple(temporal["support_margin"], formal_index),
        triple(temporal["dual_probability"], formal_index),
        triple(temporal["harm_probability"], formal_index),
        triple(temporal["rank_logit"], formal_index),
        triple(temporal["support_score"], formal_index),
    )
    context = np.concatenate(context_blocks, axis=-1).astype(np.float32)
    if not np.isfinite(token).all() or not np.isfinite(context).all():
        raise RuntimeError("temporal pair features contain NaN or Inf")
    return token, context, {
        "token_dimension": int(token.shape[-1]),
        "context_dimension": int(context.shape[-1]),
        "horizons": int(token.shape[-2]),
        "seed_order_statistics": int(order.shape[2]),
        "latent_metrics": int(order.shape[3]),
    }


class TemporalPairEncoder(nn.Module):
    def __init__(
        self,
        token_dimension: int,
        context_dimension: int,
        width: int,
    ) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(token_dimension, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.horizon = nn.GRU(width, width, batch_first=True)
        self.context = nn.Sequential(
            nn.Linear(context_dimension, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 5),
        )

    def forward(self, token: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.token(token)
        _, hidden = self.horizon(encoded)
        state = torch.cat((hidden[-1], self.context(context)), dim=-1)
        return self.head(state)


def normalize_fit(
    token: np.ndarray,
    context: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    token_mean = token.mean(axis=(0, 1))
    token_std = np.maximum(token.std(axis=(0, 1)), 1.0e-5)
    context_mean = context.mean(axis=0)
    context_std = np.maximum(context.std(axis=0), 1.0e-5)
    return token_mean, token_std, context_mean, context_std


def flatten_rows(
    token: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state, candidate = np.nonzero(mask)
    route_name = route[state]
    counts = {name: int(np.sum(route_name == name)) for name in np.unique(route_name)}
    weight = np.asarray([1.0 / counts[name] for name in route_name], dtype=np.float32)
    weight /= weight.mean()
    return token[state, candidate], context[state, candidate], targets[state, candidate], weight


def fit_fold(
    token: np.ndarray,
    context: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    train_state: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[TemporalPairEncoder, tuple[np.ndarray, ...], dict[str, float]]:
    set_seed(seed)
    train_mask = mask & train_state[:, None]
    x_token, x_context, y, weight = flatten_rows(
        token,
        context,
        targets,
        train_mask,
        route,
    )
    token_mean, token_std, context_mean, context_std = normalize_fit(
        x_token,
        x_context,
    )
    x_token = (x_token - token_mean[None, None]) / token_std[None, None]
    x_context = (x_context - context_mean[None]) / context_std[None]

    model = TemporalPairEncoder(
        token_dimension=token.shape[-1],
        context_dimension=context.shape[-1],
        width=int(contract["model"]["width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(contract["model"]["learning_rate"]),
        weight_decay=float(contract["model"]["weight_decay"]),
    )
    tensor_token = torch.from_numpy(x_token).to(device)
    tensor_context = torch.from_numpy(x_context).to(device)
    tensor_target = torch.from_numpy(y).to(device)
    tensor_weight = torch.from_numpy(weight).to(device)
    epochs = int(contract["model"]["epochs"])
    model.train()
    final_loss = math.nan
    for _ in range(epochs):
        output = model(tensor_token, tensor_context)
        regression = (
            F.smooth_l1_loss(
                output[:, 0],
                tensor_target[:, 0] / GAIN_SCALE_M,
                reduction="none",
            )
            + F.smooth_l1_loss(
                output[:, 1],
                tensor_target[:, 1] / GAIN_SCALE_M,
                reduction="none",
            )
        )
        binary = sum(
            F.binary_cross_entropy_with_logits(
                output[:, index],
                tensor_target[:, index],
                reduction="none",
            )
            for index in range(2, 5)
        )
        loss = ((regression + binary) * tensor_weight).sum() / (
            5.0 * tensor_weight.sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()
    return model, (
        token_mean,
        token_std,
        context_mean,
        context_std,
    ), {
        "training_rows": int(len(y)),
        "final_loss": final_loss,
    }


def predict(
    model: TemporalPairEncoder,
    normalization: tuple[np.ndarray, ...],
    token: np.ndarray,
    context: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    token_mean, token_std, context_mean, context_std = normalization
    shape = token.shape[:2]
    x_token = (
        (token.reshape(-1, *token.shape[2:]) - token_mean[None, None])
        / token_std[None, None]
    ).astype(np.float32)
    x_context = (
        (context.reshape(-1, context.shape[-1]) - context_mean[None])
        / context_std[None]
    ).astype(np.float32)
    batches = []
    with torch.inference_mode():
        for start in range(0, len(x_token), 4096):
            output = model(
                torch.from_numpy(x_token[start : start + 4096]).to(device),
                torch.from_numpy(x_context[start : start + 4096]).to(device),
            )
            batches.append(output.cpu().numpy().astype(np.float32))
    raw = np.concatenate(batches).reshape(*shape, 5)
    result = raw.copy()
    result[..., :2] *= GAIN_SCALE_M
    result[..., 2:] = 1.0 / (1.0 + np.exp(-result[..., 2:]))
    return result


def average_ranks(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and value[order[end]] == value[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    return ranks


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    label = np.asarray(label, dtype=bool)
    positive = int(label.sum())
    negative = int((~label).sum())
    if positive == 0 or negative == 0:
        return math.nan
    ranks = average_ranks(np.asarray(score, dtype=np.float64))
    return float(
        (ranks[label].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def wilson(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def policy_score(prediction: np.ndarray) -> np.ndarray:
    objective = np.maximum(prediction[..., 0], 0.0)
    ate = np.maximum(prediction[..., 1], 0.0)
    return (
        objective
        * (ate > 0.0)
        * prediction[..., 3]
        * (1.0 - prediction[..., 4])
    )


def selection_metrics(
    selected: np.ndarray,
    formal_index: np.ndarray,
    true_metrics: np.ndarray,
    route: np.ndarray,
) -> dict[str, Any]:
    row = np.arange(len(selected))
    formal_metrics = true_metrics[row, formal_index]
    chosen = true_metrics[row, selected]
    gain = formal_metrics - chosen
    changed = selected != formal_index
    dual = changed & (gain[:, 0] > TOLERANCE) & (gain[:, 3] > TOLERANCE)
    harm = changed & ((gain[:, 0] < -TOLERANCE) | (gain[:, 3] < -TOLERANCE))
    replacements = int(changed.sum())
    by_route = {}
    for name in sorted(np.unique(route)):
        local = route == name
        by_route[str(name)] = {
            "replacements": int(changed[local].sum()),
            "objective_gain_m": float(gain[local, 3].mean()),
            "dual": int(dual[local].sum()),
            "harm": int(harm[local].sum()),
        }
    return {
        "replacements": replacements,
        "coverage": replacements / len(selected),
        "objective_gain_m": float(gain[:, 3].mean()),
        "ate_gain_m": float(gain[:, 0].mean()),
        "dual": {
            "count": int(dual.sum()),
            "rate": None if replacements == 0 else float(dual.sum() / replacements),
            "wilson95": wilson(int(dual.sum()), replacements),
        },
        "harm": {
            "count": int(harm.sum()),
            "rate": None if replacements == 0 else float(harm.sum() / replacements),
            "wilson95": wilson(int(harm.sum()), replacements),
        },
        "routes_with_replacement": int(
            sum(value["replacements"] > 0 for value in by_route.values())
        ),
        "routes_with_positive_objective_gain": int(
            sum(value["objective_gain_m"] > 0.0 for value in by_route.values())
        ),
        "by_route": by_route,
    }


def evaluate(
    prediction: np.ndarray,
    true_metrics: np.ndarray,
    valid: np.ndarray,
    formal_index: np.ndarray,
    route: np.ndarray,
    contract: dict[str, Any],
) -> dict[str, Any]:
    row = np.arange(len(true_metrics))
    formal_metrics = true_metrics[row, formal_index]
    gain = formal_metrics[:, None] - true_metrics
    dual = (gain[..., 0] > TOLERANCE) & (gain[..., 3] > TOLERANCE)
    harm = (gain[..., 0] < -TOLERANCE) | (gain[..., 3] < -TOLERANCE)
    candidate_mask = valid.copy()
    candidate_mask[row, formal_index] = False
    score = policy_score(prediction)

    valid_score = score[candidate_mask]
    candidate_layer = {
        "dual_auroc": auroc(valid_score, dual[candidate_mask]),
        "harm_auroc": auroc(-valid_score, harm[candidate_mask]),
        "objective_gain_pearson": float(
            np.corrcoef(prediction[..., 0][candidate_mask], gain[..., 3][candidate_mask])[0, 1]
        ),
    }
    opportunity = 0
    recalled = 0
    for state in range(len(true_metrics)):
        local = np.flatnonzero(candidate_mask[state])
        if not dual[state, local].any():
            continue
        opportunity += 1
        top = local[np.argsort(score[state, local])[-5:]]
        recalled += int(dual[state, top].any())
    candidate_layer["top5_dual_state_recall"] = (
        math.nan if opportunity == 0 else recalled / opportunity
    )
    candidate_layer["dual_opportunity_states"] = opportunity

    strict = (
        candidate_mask
        & (prediction[..., 0] > 0.0)
        & (prediction[..., 1] > 0.0)
        & (prediction[..., 3] >= float(contract["strict_gate"]["dual_probability_min"]))
        & (prediction[..., 4] <= float(contract["strict_gate"]["harm_probability_max"]))
    )
    strict_selected = formal_index.copy()
    for state in range(len(strict_selected)):
        local = np.flatnonzero(strict[state])
        if len(local):
            strict_selected[state] = local[np.argmax(score[state, local])]

    fixed_selected = formal_index.copy()
    choice = np.full(len(fixed_selected), -1, dtype=np.int64)
    confidence = np.full(len(fixed_selected), -np.inf, dtype=np.float32)
    positive = candidate_mask & (prediction[..., 0] > 0.0) & (prediction[..., 1] > 0.0)
    for state in range(len(fixed_selected)):
        local = np.flatnonzero(positive[state])
        if len(local):
            best = local[np.argmax(score[state, local])]
            choice[state] = best
            confidence[state] = min(
                float(prediction[state, best, 3]),
                float(1.0 - prediction[state, best, 4]),
            )
    coverage = float(contract["diagnostic"]["fixed_route_coverage"])
    for name in np.unique(route):
        states = np.flatnonzero(route == name)
        eligible = states[choice[states] >= 0]
        count = min(len(eligible), max(1, int(round(coverage * len(states)))))
        if count:
            chosen_state = eligible[np.argsort(confidence[eligible])[-count:]]
            fixed_selected[chosen_state] = choice[chosen_state]

    fixed_metrics = selection_metrics(
        fixed_selected,
        formal_index,
        true_metrics,
        route,
    )
    strict_metrics = selection_metrics(
        strict_selected,
        formal_index,
        true_metrics,
        route,
    )
    thresholds = contract["feasibility_gate"]
    checks = {
        "dual_auroc": candidate_layer["dual_auroc"] >= float(thresholds["dual_auroc_min"]),
        "harm_auroc": candidate_layer["harm_auroc"] >= float(thresholds["harm_auroc_min"]),
        "objective_pearson": candidate_layer["objective_gain_pearson"]
        >= float(thresholds["objective_pearson_min"]),
        "top5_recall": candidate_layer["top5_dual_state_recall"]
        >= float(thresholds["top5_dual_state_recall_min"]),
        "fixed_coverage_dual": (fixed_metrics["dual"]["rate"] or 0.0)
        >= float(thresholds["fixed_coverage_dual_rate_min"]),
        "fixed_coverage_harm": (fixed_metrics["harm"]["rate"] or 1.0)
        <= float(thresholds["fixed_coverage_harm_rate_max"]),
        "fixed_coverage_objective": fixed_metrics["objective_gain_m"] > 0.0,
        "route_consistency": fixed_metrics["routes_with_positive_objective_gain"]
        >= int(thresholds["minimum_positive_routes"]),
    }
    return {
        "candidate_layer": candidate_layer,
        "fixed_route_coverage_diagnostic": fixed_metrics,
        "strict_uncalibrated_gate_diagnostic": strict_metrics,
        "feasibility_checks": checks,
        "feasibility_passed": bool(all(checks.values())),
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    sidecar = args.sidecar.resolve()
    temporal_path = args.temporal_features.resolve()
    labels_path = args.labels.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema") != "scand_temporal_pair_encoder_feasibility_contract_v1":
        raise RuntimeError("wrong experiment contract")
    expected_inputs = {
        "temporal_features": temporal_path,
        "candidate_table": sidecar / "candidate_table.npz",
        "formal_snapshot": sidecar / "formal_selection_snapshot.npz",
        "labels": labels_path,
        "runner": Path(__file__).resolve(),
    }
    for name, path in expected_inputs.items():
        expected = contract["inputs"][name]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen input changed: {name}")

    temporal = load_npz(temporal_path)
    table = load_npz(sidecar / "candidate_table.npz")
    formal = load_npz(sidecar / "formal_selection_snapshot.npz")
    labels = load_npz(labels_path)
    if not np.array_equal(
        table["candidate_hash"].astype(str),
        temporal["candidate_hash"].astype(str),
    ):
        raise RuntimeError("candidate/temporal binding mismatch")
    if not np.array_equal(
        table["candidate_hash"].astype(str),
        labels["candidate_hash"].astype(str),
    ):
        raise RuntimeError("candidate/label binding mismatch")

    formal_index = pair.formal_union_index(table, formal)
    token, context, feature_contract = build_features(temporal, formal_index)
    if feature_contract != contract["feature_contract"]:
        raise RuntimeError(
            f"feature contract changed: {feature_contract} != {contract['feature_contract']}"
        )

    pools = {
        source: load_npz(sidecar / relative)
        for source, relative in pair.POOL_PATH.items()
    }
    physical, _ = pair.gather_union(table, pools, ("physical_mask",))
    support, _ = pair.gather_union(table, pools, ("support_mask",))
    if physical is None or support is None:
        raise RuntimeError("candidate validity masks unavailable")
    valid = physical.astype(bool) & support.astype(bool)
    valid[np.arange(len(valid)), formal_index] = True

    state_rows = [
        json.loads(line)
        for line in (sidecar / "states.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    route = np.asarray([str(row["route"]) for row in state_rows])
    true_metrics = labels["true_metrics_m"].astype(np.float32)
    formal_metrics = true_metrics[np.arange(len(true_metrics)), formal_index]
    gain = formal_metrics[:, None] - true_metrics
    targets = np.stack(
        (
            gain[..., 3],
            gain[..., 0],
            gain[..., 3] > TOLERANCE,
            (gain[..., 0] > TOLERANCE) & (gain[..., 3] > TOLERANCE),
            (gain[..., 0] < -TOLERANCE) | (gain[..., 3] < -TOLERANCE),
        ),
        axis=-1,
    ).astype(np.float32)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prediction = np.full((*valid.shape, 5), np.nan, dtype=np.float32)
    fold_records = []
    routes = sorted(np.unique(route))
    base_seed = int(contract["model"]["seed"])
    for fold, held_route in enumerate(routes):
        test_state = route == held_route
        train_state = ~test_state
        model, normalization, training = fit_fold(
            token,
            context,
            targets,
            valid,
            route,
            train_state,
            contract,
            device,
            base_seed + fold,
        )
        prediction[test_state] = predict(
            model,
            normalization,
            token[test_state],
            context[test_state],
            device,
        )
        fold_records.append(
            {
                "fold": fold,
                "held_route": str(held_route),
                **training,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(prediction).all():
        raise RuntimeError("OOF prediction is incomplete")

    result = {
        "schema": "scand_temporal_pair_encoder_feasibility_result_v1",
        "status": "development_only_complete",
        "evidence_role": "spent_four_route_representation_feasibility_only",
        "states": int(len(route)),
        "routes": len(routes),
        "candidates": int(valid.shape[1]),
        "feature_contract": feature_contract,
        "model": {
            **contract["model"],
            "parameters": int(
                sum(
                    parameter.numel()
                    for parameter in TemporalPairEncoder(
                        feature_contract["token_dimension"],
                        feature_contract["context_dimension"],
                        int(contract["model"]["width"]),
                    ).parameters()
                )
            ),
        },
        "folds": fold_records,
        **evaluate(
            prediction,
            true_metrics,
            valid,
            formal_index,
            route,
            contract,
        ),
        "decision": (
            "representation_is_worth_independent_route_collection"
            if evaluate(
                prediction,
                true_metrics,
                valid,
                formal_index,
                route,
                contract,
            )["feasibility_passed"]
            else "representation_did_not_clear_precollection_feasibility_gate"
        ),
        "formal_v70_mutated": False,
        "sealed_validation_opened": False,
        "inputs": contract["inputs"],
        "contract_sha256": sha256_file(contract_path),
    }
    prediction_path = output_dir / "outer_route_oof_predictions.npz"
    result_path = output_dir / "result.json"
    write_npz_atomic(
        prediction_path,
        prediction=prediction,
        formal_index=formal_index.astype(np.int64),
        valid=valid.astype(bool),
        route=route.astype(str),
    )
    result["prediction_sha256"] = sha256_file(prediction_path)
    write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
                "candidate_layer": result["candidate_layer"],
                "fixed_route_coverage_diagnostic": result[
                    "fixed_route_coverage_diagnostic"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
