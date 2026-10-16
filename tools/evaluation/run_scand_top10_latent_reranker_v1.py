#!/usr/bin/env python3
"""Export one rich LTWM cache and test one nested top-10 latent reranker."""

from __future__ import annotations

import argparse
import gc
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


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.costs import latent_trajectory_features  # noqa: E402
from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from recon_latents.models_v2 import LiteTokenFormerConfig, LiteTokenFormerWMV2  # noqa: E402
from tools.evaluation import run_scand_union65_listwise_selector_v1 as coarse  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wm-state-batch", type=int, default=4)
    parser.add_argument("--wm-candidate-chunk", type=int, default=13)
    return parser.parse_args()


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


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


def resolve_and_verify(entry: dict[str, Any]) -> Path:
    path = (ROOT / entry["path"]).resolve()
    require("validation" not in str(path).lower(), f"validation-like input rejected: {path}")
    require(path.is_file(), f"missing input: {path}")
    require(sha256_file(path) == entry["sha256"], f"input hash changed: {path}")
    return path


def reconstruct_actions(data: dict[str, np.ndarray]) -> np.ndarray:
    names = data["feature_names"].astype(str).tolist()
    delta = [names.index(f"delta_actions_h4_{index}") for index in range(12)]
    formal = [names.index(f"formal_actions_h4_{index}") for index in range(12)]
    feature = data["runtime_pair_feature"].astype(np.float32)
    actions = feature[..., delta] + feature[..., formal]
    actions = actions.reshape(800, 65, 4, 3).astype(np.float32)
    formal_index = data["formal_v70_index"].astype(np.int64)
    formal_action = feature[..., formal].reshape(800, 65, 4, 3)
    expected = formal_action[np.arange(800), formal_index]
    require(
        np.array_equal(actions[np.arange(800), formal_index], expected),
        "formal action reconstruction changed",
    )
    return actions


def spatial_channel_pool(value: torch.Tensor) -> torch.Tensor:
    prefix = value.shape[:-2]
    pooled = value.reshape(*prefix, 16, 16, 32, 12).mean(dim=-1)
    pooled = pooled.reshape(*prefix, 4, 4, 4, 4, 32).mean(dim=(-4, -2))
    return pooled.flatten(-3)


def spatial_logvar_pool(value: torch.Tensor) -> torch.Tensor:
    prefix = value.shape[:-1]
    pooled = value.reshape(*prefix, 16, 16)
    pooled = pooled.reshape(*prefix, 4, 4, 4, 4).mean(dim=(-3, -1))
    return pooled.flatten(-2)


def export_rich_features(
    contract: dict[str, Any],
    data: dict[str, np.ndarray],
    output: Path,
    device: torch.device,
    state_batch: int,
    candidate_chunk: int,
) -> dict[str, Any]:
    require(not output.exists(), f"refusing to overwrite: {output}")
    actions = reconstruct_actions(data)
    states, candidates = actions.shape[:2]
    grid = np.empty((states, candidates, 3, 4, 512), dtype=np.float16)
    scalar = np.empty((states, candidates, 3, 4, 7), dtype=np.float16)
    logvar = np.empty((states, candidates, 3, 4, 16), dtype=np.float16)
    decoded = np.empty((states, candidates, 3, 4, 3), dtype=np.float16)
    goal_grid = np.empty((states, 512), dtype=np.float16)
    current_grid = np.empty((states, 512), dtype=np.float16)
    latent_entries = contract["inputs"]["latent_caches"]
    latent_paths = [resolve_and_verify(entry) for entry in latent_entries]
    offsets = np.cumsum([0] + [int(entry["states"]) for entry in latent_entries])
    require(offsets[-1] == states, "latent cache state total changed")
    checkpoints = contract["inputs"]["ltwm_checkpoints"]
    started = time.perf_counter()
    for seed_position, entry in enumerate(checkpoints):
        checkpoint_path = resolve_and_verify(entry)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**checkpoint["config"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().requires_grad_(False)
        for cohort_position, latent_path in enumerate(latent_paths):
            with np.load(latent_path, allow_pickle=False) as payload:
                history_all = np.asarray(payload["history"])
                goal_all = np.asarray(payload["goal"])
            global_offset = int(offsets[cohort_position])
            expected_states = int(offsets[cohort_position + 1] - global_offset)
            require(len(history_all) == expected_states, "latent segment size changed")
            for local_start in range(0, expected_states, state_batch):
                local_end = min(local_start + state_batch, expected_states)
                global_start = global_offset + local_start
                global_end = global_offset + local_end
                history = torch.from_numpy(
                    history_all[local_start:local_end].astype(np.float32)
                ).to(device)
                goal = torch.from_numpy(
                    goal_all[local_start:local_end].astype(np.float32)
                ).to(device)
                local_actions = torch.from_numpy(actions[global_start:global_end]).to(device)
                if seed_position == 0:
                    with torch.inference_mode():
                        goal_grid[global_start:global_end] = (
                            spatial_channel_pool(goal).cpu().numpy().astype(np.float16)
                        )
                        current_grid[global_start:global_end] = (
                            spatial_channel_pool(history[:, -1]).cpu().numpy().astype(np.float16)
                        )
                for candidate_start in range(0, candidates, candidate_chunk):
                    candidate_end = min(candidate_start + candidate_chunk, candidates)
                    chunk = local_actions[:, candidate_start:candidate_end]
                    autocast = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device.type == "cuda"
                        else torch.autocast("cpu", enabled=False)
                    )
                    with torch.inference_mode(), autocast:
                        prediction = model(history, chunk)
                    future = prediction["predicted_tokens"].float()
                    with torch.inference_mode():
                        local_scalar, _ = latent_trajectory_features(
                            future,
                            goal,
                            history[:, -1],
                        )
                        local_grid = spatial_channel_pool(future)
                        local_logvar = spatial_logvar_pool(
                            prediction["predicted_logvar"].float()
                        )
                    destination = (
                        slice(global_start, global_end),
                        slice(candidate_start, candidate_end),
                        seed_position,
                    )
                    grid[destination] = local_grid.cpu().numpy().astype(np.float16)
                    scalar[destination] = (
                        local_scalar.reshape(
                            global_end - global_start,
                            candidate_end - candidate_start,
                            7,
                            4,
                        )
                        .transpose(0, 1, 3, 2)
                        .cpu()
                        .numpy()
                        .astype(np.float16)
                    )
                    logvar[destination] = local_logvar.cpu().numpy().astype(np.float16)
                    decoded[destination] = (
                        prediction["decoded_action"].float().cpu().numpy().astype(np.float16)
                    )
                print(
                    canonical_json(
                        {
                            "stage": "rich_ltwm_export",
                            "seed": int(entry["seed"]),
                            "completed_states": global_end,
                            "states": states,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(
        all(
            np.isfinite(value).all()
            for value in (grid, scalar, logvar, decoded, goal_grid, current_grid)
        ),
        "rich features contain NaN or Inf",
    )
    metadata = {
        "schema": "scand_union65_rich_ltwm_runtime_cache_v1",
        "states": states,
        "candidates": candidates,
        "seeds": [int(entry["seed"]) for entry in checkpoints],
        "labels_included": False,
        "runtime_table_sha256": contract["inputs"]["runtime_table"]["sha256"],
        "feature_contract": contract["feature"],
    }
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            record_key=data["record_key"],
            rollout_grid=grid,
            latent_scalar=scalar,
            logvar_grid=logvar,
            decoded_action=decoded,
            goal_grid=goal_grid,
            current_grid=current_grid,
            metadata=np.asarray(canonical_json(metadata)),
        )
    temporary.replace(output)
    return {
        "path": str(output),
        "sha256": sha256_file(output),
        "runtime_seconds": time.perf_counter() - started,
        "labels_included": False,
    }


def load_rich(path: Path, record_key: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        result = {name: np.asarray(payload[name]).copy() for name in payload.files}
    require(
        np.array_equal(result["record_key"].astype(str), record_key.astype(str)),
        "rich cache record binding changed",
    )
    return result


def shortlist(
    scores: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    challengers: int = 10,
) -> np.ndarray:
    mask = coarse.choice_mask(data, states)
    formal = data["formal_v70_index"][states].astype(np.int64)
    mask[np.arange(len(states)), formal] = False
    masked = np.where(mask, scores, -np.inf)
    ordered = np.argsort(-masked, axis=1, kind="stable")[:, :challengers]
    require(np.isfinite(np.take_along_axis(masked, ordered, axis=1)).all(), "shortlist incomplete")
    return np.concatenate((formal[:, None], ordered.astype(np.int64)), axis=1)


def gather_candidate(value: np.ndarray, states: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return value[states[:, None], candidates]


def actions_from_data(data: dict[str, np.ndarray]) -> np.ndarray:
    return reconstruct_actions(data)


def token_batch(
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    grid = gather_candidate(rich["rollout_grid"], states, candidates).astype(np.float32)
    scalar = gather_candidate(rich["latent_scalar"], states, candidates).astype(np.float32)
    logvar = gather_candidate(rich["logvar_grid"], states, candidates).astype(np.float32)
    decoded = gather_candidate(rich["decoded_action"], states, candidates).astype(np.float32)
    local_actions = gather_candidate(actions, states, candidates).astype(np.float32)
    formal = data["formal_v70_index"][states].astype(np.int64)
    formal_grid = rich["rollout_grid"][states, formal].astype(np.float32)[:, None]
    formal_scalar = rich["latent_scalar"][states, formal].astype(np.float32)[:, None]
    formal_logvar = rich["logvar_grid"][states, formal].astype(np.float32)[:, None]
    formal_decoded = rich["decoded_action"][states, formal].astype(np.float32)[:, None]
    formal_actions = actions[states, formal].astype(np.float32)[:, None, None]
    goal = rich["goal_grid"][states].astype(np.float32)[:, None, None, None]
    formal_grid = np.broadcast_to(formal_grid, grid.shape)
    formal_scalar = np.broadcast_to(formal_scalar, scalar.shape)
    formal_logvar = np.broadcast_to(formal_logvar, logvar.shape)
    formal_decoded = np.broadcast_to(formal_decoded, decoded.shape)
    expanded_action = np.broadcast_to(
        local_actions[:, :, None],
        (*local_actions.shape[:2], 3, *local_actions.shape[2:]),
    )
    formal_actions = np.broadcast_to(formal_actions, expanded_action.shape)
    return np.concatenate(
        (
            grid - formal_grid,
            grid - goal,
            formal_grid - goal,
            scalar,
            formal_scalar,
            scalar - formal_scalar,
            logvar,
            formal_logvar,
            logvar - formal_logvar,
            decoded,
            formal_decoded,
            decoded - formal_decoded,
            expanded_action,
            formal_actions,
            expanded_action - formal_actions,
        ),
        axis=-1,
    ).astype(np.float32)


class LatentReranker(nn.Module):
    def __init__(self, token_dimension: int, token_width: int, horizon_width: int) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(token_dimension, token_width),
            nn.LayerNorm(token_width),
            nn.GELU(),
        )
        self.horizon = nn.GRU(2 * token_width, horizon_width, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(horizon_width, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # [B,C,S,H,D]
        encoded = self.token(value)
        seed_mean = encoded.mean(dim=2)
        seed_std = encoded.std(dim=2, unbiased=False)
        horizon = torch.cat((seed_mean, seed_std), dim=-1)
        batch, candidates = horizon.shape[:2]
        _, hidden = self.horizon(horizon.reshape(batch * candidates, 4, -1))
        return self.head(hidden[-1]).reshape(batch, candidates)


def stream_normalizer(
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    route: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    weight = coarse.state_weights(route, states)
    total = 0.0
    total_sum = None
    total_square = None
    token_dimension = 0
    for start in range(0, len(states), batch_size):
        end = min(start + batch_size, len(states))
        token = token_batch(
            rich,
            actions,
            data,
            states[start:end],
            candidates[start:end],
        ).astype(np.float64)
        token_dimension = token.shape[-1]
        local_weight = weight[start:end, None, None, None]
        normalizer = token.shape[1] * token.shape[2] * token.shape[3]
        local_sum = (token * local_weight).sum(axis=(0, 1, 2, 3))
        local_square = ((token * token) * local_weight).sum(axis=(0, 1, 2, 3))
        local_total = float(weight[start:end].sum() * normalizer)
        total_sum = local_sum if total_sum is None else total_sum + local_sum
        total_square = local_square if total_square is None else total_square + local_square
        total += local_total
    assert total_sum is not None and total_square is not None
    mean = total_sum / total
    variance = np.maximum(total_square / total - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1.0e-5)
    return mean.astype(np.float32), std.astype(np.float32), token_dimension


def selected_target(
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    contract: dict[str, Any],
) -> np.ndarray:
    target = contract["target"]
    objective = gather_candidate(
        data["objective_gain_vs_formal_v70_m"],
        states,
        candidates,
    )
    ate = gather_candidate(data["ate_gain_vs_formal_v70_m"], states, candidates)
    harm = gather_candidate(
        data["harm_vs_formal_v70"].astype(np.float32),
        states,
        candidates,
    )
    utility = (
        float(target["objective_gain_weight"]) * objective
        + float(target["ate_gain_weight"]) * ate
        - float(target["harm_penalty_m"]) * harm
    ).astype(np.float32)
    utility[:, 0] = 0.0
    return utility


def fit_reranker(
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    route: np.ndarray,
    contract: dict[str, Any],
    device: torch.device,
    seed: int,
) -> tuple[LatentReranker, np.ndarray, np.ndarray, dict[str, Any]]:
    set_seed(seed)
    config = contract["model"]
    batch_size = int(config["state_batch_size"])
    mean, std, token_dimension = stream_normalizer(
        rich,
        actions,
        data,
        states,
        candidates,
        route,
        batch_size,
    )
    model = LatentReranker(
        token_dimension,
        int(config["token_width"]),
        int(config["horizon_width"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    target = selected_target(data, states, candidates, contract)
    state_weight = coarse.state_weights(route, states)
    temperature = float(contract["target"]["softmax_temperature_m"])
    rng = np.random.default_rng(seed)
    final_loss = math.nan
    model.train()
    for _ in range(int(config["epochs"])):
        order = rng.permutation(len(states))
        total_loss = 0.0
        total_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            token = token_batch(
                rich,
                actions,
                data,
                states[index],
                candidates[index],
            )
            token = ((token - mean) / std).astype(np.float32)
            x = torch.from_numpy(token).to(device)
            y = torch.from_numpy(target[index]).to(device)
            weight = torch.from_numpy(state_weight[index]).to(device)
            logits = model(x)
            soft_target = torch.softmax(y / temperature, dim=1)
            per_state = -(soft_target * torch.log_softmax(logits, dim=1)).sum(dim=1)
            loss = (per_state * weight).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * float(weight.sum().cpu())
            total_weight += float(weight.sum().cpu())
        final_loss = total_loss / max(total_weight, 1.0)
    return model, mean, std, {
        "states": int(len(states)),
        "routes": int(len(np.unique(route[states]))),
        "token_dimension": int(token_dimension),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "last_epoch_route_balanced_listwise_loss": float(final_loss),
    }


def predict_reranker(
    model: LatentReranker,
    mean: np.ndarray,
    std: np.ndarray,
    rich: dict[str, np.ndarray],
    actions: np.ndarray,
    data: dict[str, np.ndarray],
    states: np.ndarray,
    candidates: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            token = token_batch(
                rich,
                actions,
                data,
                states[start:end],
                candidates[start:end],
            )
            token = ((token - mean) / std).astype(np.float32)
            result.append(model(torch.from_numpy(token).to(device)).cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def map_short_scores(
    states: np.ndarray,
    candidates: np.ndarray,
    scores: np.ndarray,
    total_states: int,
) -> np.ndarray:
    result = np.full((total_states, 65), -np.inf, dtype=np.float32)
    result[states[:, None], candidates] = scores
    return result


def cohort_stats(
    data: dict[str, np.ndarray],
    selected: np.ndarray,
) -> dict[str, Any]:
    result = {}
    cohorts = data["cohort"].astype(str)
    for name in np.unique(cohorts):
        mask = cohorts == name
        states = np.flatnonzero(mask)
        stats = coarse.selection_stats(data, states, selected[mask])
        _, oracle = coarse.objective_oracle(data, states)
        stats["objective_oracle_gain_m"] = oracle["objective_gain_m"]
        stats["oracle_gain_ratio"] = (
            stats["objective_gain_m"] / oracle["objective_gain_m"]
            if oracle["objective_gain_m"] > 0.0
            else None
        )
        result[name] = stats
    return result


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    contract = read_json(contract_path)
    require(
        contract.get("status") == "frozen_before_feature_export_and_nested_route_oof",
        "contract not frozen",
    )
    device = torch.device(args.device)
    require(device.type != "cuda" or torch.cuda.is_available(), "CUDA unavailable")
    table_path = resolve_and_verify(contract["inputs"]["runtime_table"])
    coarse_contract_path = resolve_and_verify(contract["inputs"]["coarse_contract"])
    data = coarse.load_npz(table_path)
    require(data["runtime_pair_feature"].shape == (800, 65, 351), "runtime table changed")
    coarse_contract = read_json(coarse_contract_path)
    output.mkdir(parents=True, exist_ok=False)
    rich_path = output / "rich_ltwm_runtime_features.npz"
    rich_audit = export_rich_features(
        contract,
        data,
        rich_path,
        device,
        args.wm_state_batch,
        args.wm_candidate_chunk,
    )
    rich = load_rich(rich_path, data["record_key"])
    actions = actions_from_data(data)
    routes = data["route_group"].astype(str)
    folds = int(contract["evaluation"]["outer_route_folds"])
    fold, fold_audit = coarse.assign_folds(routes, folds)
    states_all = np.arange(len(routes))
    rerank_score_oof = np.full((len(routes), 65), -np.inf, dtype=np.float32)
    coarse_score_oof = np.full((len(routes), 65), np.nan, dtype=np.float32)
    selected_calibrated = data["formal_v70_index"].astype(np.int64).copy()
    curve_selected = {
        str(float(value)): data["formal_v70_index"].astype(np.int64).copy()
        for value in contract["evaluation"]["coverage_curve"]
    }
    parameters: dict[str, np.ndarray] = {}
    outer_records = {}
    seed = int(contract["model"]["seed"])
    started = time.perf_counter()
    for outer in range(folds):
        outer_train = states_all[fold != outer]
        outer_test = states_all[fold == outer]
        require(
            not (set(routes[outer_train]) & set(routes[outer_test])),
            "outer route leakage",
        )
        inner_scores = np.full((len(routes), 65), -np.inf, dtype=np.float32)
        inner_fits = []
        for inner in range(folds):
            if inner == outer:
                continue
            inner_train = states_all[(fold != outer) & (fold != inner)]
            inner_test = states_all[fold == inner]
            coarse_model, coarse_mean, coarse_std, coarse_fit = coarse.fit_model(
                data,
                inner_train,
                routes,
                coarse_contract,
                device,
                seed + outer * 1000 + inner * 10,
            )
            coarse_train_score = coarse.predict(
                coarse_model,
                coarse_mean,
                coarse_std,
                data["runtime_pair_feature"][inner_train],
                device,
            )
            coarse_test_score = coarse.predict(
                coarse_model,
                coarse_mean,
                coarse_std,
                data["runtime_pair_feature"][inner_test],
                device,
            )
            train_short = shortlist(coarse_train_score, data, inner_train)
            test_short = shortlist(coarse_test_score, data, inner_test)
            reranker, mean, std, rerank_fit = fit_reranker(
                rich,
                actions,
                data,
                inner_train,
                train_short,
                routes,
                contract,
                device,
                seed + outer * 1000 + inner * 10 + 1,
            )
            local_score = predict_reranker(
                reranker,
                mean,
                std,
                rich,
                actions,
                data,
                inner_test,
                test_short,
                device,
                int(contract["model"]["state_batch_size"]),
            )
            inner_scores[inner_test[:, None], test_short] = local_score
            inner_fits.append(
                {
                    "inner_fold": inner,
                    "coarse": coarse_fit,
                    "reranker": rerank_fit,
                }
            )
            del coarse_model, reranker
            if device.type == "cuda":
                torch.cuda.empty_cache()
        require(
            np.isfinite(inner_scores[outer_train]).any(axis=1).all(),
            "inner reranker predictions incomplete",
        )
        margin, margin_stats, inner_curve = coarse.inner_curve_and_margin(
            inner_scores[outer_train],
            data,
            outer_train,
            contract,
        )
        coarse_model, coarse_mean, coarse_std, coarse_fit = coarse.fit_model(
            data,
            outer_train,
            routes,
            coarse_contract,
            device,
            seed + outer * 1000 + 990,
        )
        coarse_train_score = coarse.predict(
            coarse_model,
            coarse_mean,
            coarse_std,
            data["runtime_pair_feature"][outer_train],
            device,
        )
        coarse_test_score = coarse.predict(
            coarse_model,
            coarse_mean,
            coarse_std,
            data["runtime_pair_feature"][outer_test],
            device,
        )
        train_short = shortlist(coarse_train_score, data, outer_train)
        test_short = shortlist(coarse_test_score, data, outer_test)
        reranker, mean, std, rerank_fit = fit_reranker(
            rich,
            actions,
            data,
            outer_train,
            train_short,
            routes,
            contract,
            device,
            seed + outer * 1000 + 991,
        )
        outer_score = predict_reranker(
            reranker,
            mean,
            std,
            rich,
            actions,
            data,
            outer_test,
            test_short,
            device,
            int(contract["model"]["state_batch_size"]),
        )
        rerank_score_oof[outer_test[:, None], test_short] = outer_score
        coarse_score_oof[outer_test] = coarse_test_score
        selected_calibrated[outer_test] = coarse.select_with_margin(
            rerank_score_oof[outer_test],
            data,
            outer_test,
            margin,
        )
        _, inner_advantage = coarse.top_challenger(
            inner_scores[outer_train],
            data,
            outer_train,
        )
        curve_margins = {}
        for value in contract["evaluation"]["coverage_curve"]:
            coverage = float(value)
            local_margin = coarse.margin_for_coverage(inner_advantage, coverage)
            curve_selected[str(coverage)][outer_test] = coarse.select_with_margin(
                rerank_score_oof[outer_test],
                data,
                outer_test,
                local_margin,
            )
            curve_margins[str(coverage)] = float(local_margin)
        for name, tensor in coarse_model.state_dict().items():
            parameters[f"outer_{outer}__coarse__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"outer_{outer}__coarse_mean"] = coarse_mean
        parameters[f"outer_{outer}__coarse_std"] = coarse_std
        for name, tensor in reranker.state_dict().items():
            parameters[f"outer_{outer}__reranker__{name}"] = tensor.detach().cpu().numpy()
        parameters[f"outer_{outer}__reranker_mean"] = mean
        parameters[f"outer_{outer}__reranker_std"] = std
        parameters[f"outer_{outer}__margin"] = np.asarray(margin, dtype=np.float32)
        outer_records[str(outer)] = {
            "outer_train_states": int(len(outer_train)),
            "outer_test_states": int(len(outer_test)),
            "route_overlap": 0,
            "inner_selected_margin": margin,
            "inner_selected_margin_stats": {
                key: value
                for key, value in margin_stats.items()
                if key != "route_objective_gain_m"
            },
            "inner_margin_curve": inner_curve,
            "inner_coverage_curve_margins": curve_margins,
            "inner_fits": inner_fits,
            "final_coarse_fit": coarse_fit,
            "final_reranker_fit": rerank_fit,
        }
        print(
            canonical_json(
                {
                    "stage": "nested_reranker",
                    "outer_fold": outer,
                    "margin": margin,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        del coarse_model, reranker
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(np.isfinite(coarse_score_oof).all(), "coarse OOF scores incomplete")
    require(np.isfinite(rerank_score_oof).any(axis=1).all(), "reranker OOF scores incomplete")
    pure_selected = coarse.select_with_margin(
        rerank_score_oof,
        data,
        states_all,
        0.0,
    )
    pure_stats = coarse.selection_stats(data, states_all, pure_selected)
    calibrated_stats = coarse.selection_stats(
        data,
        states_all,
        selected_calibrated,
    )
    _, oracle_stats = coarse.objective_oracle(data, states_all)
    pure_stats["oracle_gain_ratio"] = (
        pure_stats["objective_gain_m"] / oracle_stats["objective_gain_m"]
    )
    calibrated_stats["oracle_gain_ratio"] = (
        calibrated_stats["objective_gain_m"] / oracle_stats["objective_gain_m"]
    )
    rerank_ranking = coarse.ranking_stats(
        rerank_score_oof,
        data,
        states_all,
        [int(value) for value in contract["evaluation"]["top_k"]],
    )
    coarse_ranking = coarse.ranking_stats(
        coarse_score_oof,
        data,
        states_all,
        [int(value) for value in contract["evaluation"]["top_k"]],
    )
    curve = {}
    for coverage, selected in curve_selected.items():
        stats = coarse.selection_stats(data, states_all, selected)
        curve[coverage] = {
            key: stats[key]
            for key in (
                "replacements",
                "coverage",
                "objective_gain_m",
                "ate_gain_m",
                "dual_rate",
                "harm_rate",
                "positive_route_fraction",
            )
        }
    cohorts = cohort_stats(data, selected_calibrated)
    formal = data["formal_v70_index"].astype(np.int64)
    gain = data["objective_gain_vs_formal_v70_m"][
        states_all,
        selected_calibrated,
    ].astype(np.float64)
    gain[selected_calibrated == formal] = 0.0
    bootstrap = coarse.bootstrap_gain(gain, routes)
    mainrgb = cohorts["spent_new_mainrgb_development"]
    gate = contract["prototype_go_gate"]
    harm_rate = calibrated_stats["harm_rate"]
    checks = {
        "overall_objective_gain": calibrated_stats["objective_gain_m"]
        >= float(gate["minimum_overall_objective_gain_m"]),
        "new_mainrgb_objective_gain": mainrgb["objective_gain_m"]
        >= float(gate["minimum_new_mainrgb_objective_gain_m"]),
        "overall_ate_gain": calibrated_stats["ate_gain_m"]
        > float(gate["minimum_overall_ate_gain_m"]),
        "harm_rate": harm_rate is not None
        and harm_rate <= float(gate["maximum_empirical_harm_rate"]),
        "new_mainrgb_oracle_gain_ratio": mainrgb["oracle_gain_ratio"]
        >= float(gate["minimum_new_mainrgb_oracle_gain_ratio"]),
        "route_bootstrap_lcb": bootstrap["lcb95_m"]
        > float(gate["route_bootstrap_lcb_min_m"]),
    }
    go = all(checks.values())
    prediction_path = output / "outer_route_oof_predictions.npz"
    parameter_path = output / "frozen_outer_pipeline.npz"
    result_path = output / "result.json"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            record_key=data["record_key"],
            route_group=routes,
            outer_fold=fold,
            coarse_score=coarse_score_oof,
            reranker_score=rerank_score_oof,
            pure_selected_index=pure_selected,
            calibrated_selected_index=selected_calibrated,
            formal_v70_index=formal,
        )
    with parameter_path.open("wb") as handle:
        np.savez_compressed(handle, **parameters)
    result = {
        "schema": "scand_top10_latent_reranker_nested_route_oof_result_v1",
        "status": "complete_adaptive_spent_development_performance_prototype",
        "decision": (
            "go_freeze_pipeline_worth_one_time_sealed_validation"
            if go
            else "no_go_rich_top10_reranker_did_not_reliably_improve_performance"
        ),
        "prototype_go": go,
        "prototype_checks": checks,
        "states": int(len(routes)),
        "routes": int(len(np.unique(routes))),
        "pure_top1_including_formal": pure_stats,
        "inner_calibrated_fallback": calibrated_stats,
        "objective_oracle": oracle_stats,
        "cohort_results": cohorts,
        "coarse_ranking": coarse_ranking,
        "reranker_ranking": rerank_ranking,
        "performance_harm_curve": curve,
        "route_cluster_bootstrap_objective_gain": bootstrap,
        "outer_folds": outer_records,
        "fold_audit": fold_audit,
        "inputs": {
            "contract": {
                "path": str(contract_path),
                "sha256": sha256_file(contract_path),
            },
            "runtime_table": {
                "path": str(table_path),
                "sha256": sha256_file(table_path),
            },
            "coarse_contract": {
                "path": str(coarse_contract_path),
                "sha256": sha256_file(coarse_contract_path),
            },
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "artifacts": {
            "rich_features": rich_audit,
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "frozen_outer_pipeline": {
                "path": str(parameter_path),
                "sha256": sha256_file(parameter_path),
            },
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "sealed_validation_opened": False,
        "formal_v70_mutated": False,
        "cem_mutated": False,
    }
    write_json_atomic(result_path, result)
    print(
        canonical_json(
            {
                "status": result["status"],
                "decision": result["decision"],
                "overall_objective_gain_m": calibrated_stats["objective_gain_m"],
                "overall_ate_gain_m": calibrated_stats["ate_gain_m"],
                "harm_rate": calibrated_stats["harm_rate"],
                "new_mainrgb_objective_gain_m": mainrgb["objective_gain_m"],
                "new_mainrgb_oracle_gain_ratio": mainrgb["oracle_gain_ratio"],
                "result": str(result_path),
                "result_sha256": sha256_file(result_path),
            }
        )
    )


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
