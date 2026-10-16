#!/usr/bin/env python3
"""Train plain future-path imitation on 7.5k existing SCAND log windows.

This spent-data diagnostic uses no counterfactual action outcomes.  It predicts
the logged H4 future path from existing runtime-safe K17 feature summaries,
trains on train-fit routes, and evaluates on scorer-cal routes with zero route,
trajectory, and bag overlap.
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
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


SPLITS = {
    "train_fit": {
        "source": ROOT / "outputs/scand_domain_adaptation_v1/sources/train_fit_source.npz",
        "feature": ROOT
        / "outputs/scand_domain_adaptation_v1/features/train_fit"
        / "scand_inner_train_fit_spacing038_features.npz",
    },
    "scorer_cal": {
        "source": ROOT
        / "outputs/scand_domain_adaptation_v1/sources/scorer_cal_source.npz",
        "feature": ROOT
        / "outputs/scand_domain_adaptation_v1/features/scorer_cal"
        / "scand_inner_scorer_cal_spacing038_features.npz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--ensemble-size", type=int, default=2)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npz(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        names = payload.files if keys is None else keys
        return {name: np.asarray(payload[name]).copy() for name in names}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_candidates(value: np.ndarray) -> np.ndarray:
    flat = value.reshape(value.shape[0], value.shape[1], -1).astype(np.float32)
    return np.concatenate(
        (flat[:, 0], flat.mean(axis=1), flat.std(axis=1)),
        axis=1,
    ).astype(np.float32)


def load_split(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = SPLITS[name]
    source = load_npz(
        paths["source"],
        (
            "candidates",
            "ground_truth_local_m",
            "record_key",
            "route_group",
            "trajectory",
            "bag_id",
        ),
    )
    feature = load_npz(
        paths["feature"],
        (
            "candidates",
            "action_features",
            "static_features",
            "raw_ltwm_features",
            "scorer_v2_aux_features",
            "ate_by_seed",
            "rtp_by_seed",
            "support_by_seed",
            "majority_support",
            "record_key",
        ),
    )
    require(
        np.array_equal(source["record_key"].astype(str), feature["record_key"].astype(str)),
        f"{name} source/feature identity mismatch",
    )
    require(
        np.allclose(source["candidates"], feature["candidates"], atol=1.0e-7),
        f"{name} candidate mismatch",
    )
    seed_blocks = [
        np.moveaxis(feature[key], 0, -1)
        for key in ("ate_by_seed", "rtp_by_seed", "support_by_seed")
    ]
    blocks = {
        "candidate_geometry": summarize_candidates(feature["candidates"]),
        "action_history": summarize_candidates(feature["action_features"]),
        "static_visual_action": summarize_candidates(feature["static_features"]),
        "ltwm_rollout": summarize_candidates(feature["raw_ltwm_features"]),
        "frozen_scorer": summarize_candidates(feature["scorer_v2_aux_features"]),
        "seed_signals": summarize_candidates(np.concatenate(seed_blocks, axis=-1)),
        "support_fraction": feature["majority_support"].mean(axis=1, keepdims=True).astype(
            np.float32
        ),
    }
    descriptor = np.concatenate(list(blocks.values()), axis=1).astype(np.float32)
    require(np.isfinite(descriptor).all(), f"{name} descriptor nonfinite")
    source["descriptor"] = descriptor
    source["support"] = feature["majority_support"].astype(bool)
    return source, {
        "source": {"path": str(paths["source"]), "sha256": sha256_file(paths["source"])},
        "feature": {
            "path": str(paths["feature"]),
            "sha256": sha256_file(paths["feature"]),
        },
        "states": int(len(descriptor)),
        "routes": int(len(np.unique(source["route_group"].astype(str)))),
        "trajectories": int(len(np.unique(source["trajectory"].astype(str)))),
        "descriptor_dimension": int(descriptor.shape[1]),
        "descriptor_blocks": {key: int(value.shape[1]) for key, value in blocks.items()},
    }


class DensePathNetwork(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 8),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


def route_weights(route: np.ndarray) -> np.ndarray:
    unique, count = np.unique(route.astype(str), return_counts=True)
    lookup = dict(zip(unique.tolist(), count.tolist()))
    weight = np.asarray([1.0 / lookup[name] for name in route.astype(str)], np.float32)
    return weight / weight.mean()


def fit_model(
    train: dict[str, np.ndarray],
    device: torch.device,
    seed: int,
    epochs: int,
) -> tuple[DensePathNetwork, np.ndarray, np.ndarray, dict[str, float]]:
    set_seed(seed)
    x = train["descriptor"].astype(np.float32)
    mean = x.mean(axis=0)
    std = np.maximum(x.std(axis=0), 1.0e-5)
    x = ((x - mean) / std).astype(np.float32)
    candidates_m = train["candidates"].astype(np.float32) * 0.25
    baseline_path = candidates_m[:, 0]
    y = (
        (train["ground_truth_local_m"].astype(np.float32) - baseline_path).reshape(
            len(x), 8
        )
        / 0.1
    ).astype(np.float32)
    weight = route_weights(train["route_group"])
    model = DensePathNetwork(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6.0e-4, weight_decay=5.0e-4)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    weight_tensor = torch.from_numpy(weight)
    rng = np.random.default_rng(seed)
    batch_size = 256
    final_loss = math.nan
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(x))
        total = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = torch.from_numpy(order[start : start + batch_size])
            xb = x_tensor[index].to(device)
            yb = y_tensor[index].to(device)
            wb = weight_tensor[index].to(device)
            prediction = model(xb)
            per_state = F.smooth_l1_loss(
                prediction, yb, reduction="none", beta=0.5
            ).mean(dim=1)
            loss = (per_state * wb).sum() / wb.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            local_weight = float(wb.sum().detach().cpu())
            total += float(loss.detach().cpu()) * local_weight
            total_weight += local_weight
        final_loss = total / max(total_weight, 1.0)
    return model, mean.astype(np.float32), std.astype(np.float32), {
        "seed": seed,
        "epochs": epochs,
        "last_epoch_route_balanced_loss": float(final_loss),
    }


def predict(
    model: DensePathNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    split: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    x = ((split["descriptor"].astype(np.float32) - mean) / std).astype(np.float32)
    baseline = split["candidates"][:, 0].astype(np.float32) * 0.25
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), 512):
            result.append(
                model(torch.from_numpy(x[start : start + 512]).to(device))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    residual = np.concatenate(result).reshape(-1, 4, 2) * 0.1
    return baseline + residual


def ate(candidates_m: np.ndarray, path_m: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.mean(np.sum((candidates_m - path_m[:, None]) ** 2, axis=-1), axis=-1)
    ).astype(np.float32)


def evaluate(
    split: dict[str, np.ndarray],
    predicted_path: np.ndarray,
    support_only: bool,
) -> dict[str, Any]:
    candidates_m = split["candidates"].astype(np.float32) * 0.25
    truth = split["ground_truth_local_m"].astype(np.float32)
    predicted_distance = ate(candidates_m, predicted_path)
    true_distance = ate(candidates_m, truth)
    mask = (
        split["support"].astype(bool).copy()
        if support_only
        else np.ones(true_distance.shape, dtype=bool)
    )
    mask[:, 0] = True
    selected = np.argmin(np.where(mask, predicted_distance, np.inf), axis=1)
    oracle = np.argmin(np.where(mask, true_distance, np.inf), axis=1)
    row = np.arange(len(selected))
    baseline_ate = true_distance[:, 0]
    selected_ate = true_distance[row, selected]
    oracle_ate = true_distance[row, oracle]
    gain = baseline_ate - selected_ate
    oracle_gain = baseline_ate - oracle_ate
    changed = selected != 0
    harm = changed & (gain < 0.0)
    routes = split["route_group"].astype(str)
    route_gain = {
        name: float(gain[routes == name].mean()) for name in np.unique(routes)
    }
    return {
        "states": int(len(selected)),
        "support_only": support_only,
        "baseline_mean_ate_m": float(baseline_ate.mean()),
        "selected_mean_ate_m": float(selected_ate.mean()),
        "oracle_mean_ate_m": float(oracle_ate.mean()),
        "ate_gain_m": float(gain.mean()),
        "oracle_ate_gain_m": float(oracle_gain.mean()),
        "oracle_gain_ratio": float(gain.mean() / oracle_gain.mean())
        if oracle_gain.mean() > 0.0
        else None,
        "baseline_p90_ate_m": float(np.quantile(baseline_ate, 0.9)),
        "selected_p90_ate_m": float(np.quantile(selected_ate, 0.9)),
        "coverage": float(changed.mean()),
        "harm_rate": float(harm.sum() / changed.sum()) if changed.any() else None,
        "exact_oracle_recall": float((selected == oracle).mean()),
        "positive_route_fraction": float(
            np.mean(np.asarray(list(route_gain.values())) > 0.0)
        ),
        "route_ate_gain_m": dict(sorted(route_gain.items())),
        "path_rmse_m": float(
            np.sqrt(np.mean(np.sum((predicted_path - truth) ** 2, axis=-1)))
        ),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    train, train_audit = load_split("train_fit")
    test, test_audit = load_split("scorer_cal")
    overlap = {}
    for key in ("route_group", "trajectory", "bag_id"):
        overlap[key] = sorted(
            set(train[key].astype(str).tolist()) & set(test[key].astype(str).tolist())
        )
        require(not overlap[key], f"train/test {key} overlap")
    started = time.perf_counter()
    train_predictions = []
    test_predictions = []
    fits = []
    for member in range(args.ensemble_size):
        model, mean, std, fit = fit_model(
            train, device, 20260750 + member, args.epochs
        )
        train_predictions.append(predict(model, mean, std, train, device))
        test_predictions.append(predict(model, mean, std, test, device))
        fits.append(fit)
        del model
    train_prediction = np.mean(np.stack(train_predictions), axis=0)
    test_prediction = np.mean(np.stack(test_predictions), axis=0)
    evaluations = {
        "train_resubstitution_all_candidates": evaluate(
            train, train_prediction, False
        ),
        "train_resubstitution_supported_candidates": evaluate(
            train, train_prediction, True
        ),
        "unseen_route_test_all_candidates": evaluate(test, test_prediction, False),
        "unseen_route_test_supported_candidates": evaluate(test, test_prediction, True),
    }
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "path_predictions.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_record_key=train["record_key"],
            train_predicted_path_m=train_prediction,
            test_record_key=test["record_key"],
            test_predicted_path_m=test_prediction,
        )
    result = {
        "schema": "scand_dense_path_imitation_diagnostic_v1",
        "status": "complete_spent_development_diagnostic_only",
        "promotion_allowed": False,
        "counterfactual_action_outcomes_used": False,
        "train_test_identity_overlap": overlap,
        "train": train_audit,
        "test": test_audit,
        "fits": fits,
        "evaluations": evaluations,
        "runtime_seconds": float(time.perf_counter() - started),
        "inputs": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            }
        },
        "artifacts": {
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            }
        },
    }
    write = result_path.with_name(result_path.name + ".tmp")
    write.write_text(canonical_json(result) + "\n", encoding="utf-8")
    write.replace(result_path)
    print(
        canonical_json(
            {
                "train_states": train_audit["states"],
                "test_states": test_audit["states"],
                "test_routes": test_audit["routes"],
                "test_all_candidate_ate_gain_m": evaluations[
                    "unseen_route_test_all_candidates"
                ]["ate_gain_m"],
                "test_all_candidate_oracle_ratio": evaluations[
                    "unseen_route_test_all_candidates"
                ]["oracle_gain_ratio"],
                "test_supported_ate_gain_m": evaluations[
                    "unseen_route_test_supported_candidates"
                ]["ate_gain_m"],
                "result": str(result_path),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
