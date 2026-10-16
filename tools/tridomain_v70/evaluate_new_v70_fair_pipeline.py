#!/usr/bin/env python3
"""Run the new tri-seed LTWM/scorer inside the frozen fair-V70 H4 protocol.

Selection is deliberately a separate subcommand from metric computation.  The
``select`` stage only opens history/goal DINOv3 latents and a frozen nested
NoMaD K32 pool.  It freezes K16/K32 actions before ``metrics`` opens poses.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from planning.cem_sampler import (  # noqa: E402
    CEMConfig,
    CEMState,
    MultiModalCEM,
    paired_conservative_joint_cost,
)
from planning.nomad_action_adapter import (  # noqa: E402
    NWM_WAYPOINT_SPACING_METERS,
    integrate_local_actions,
    nomad_waypoints_to_local_actions,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)


SEEDS = (1701, 1702, 1703)
HORIZON = 4
ACTION_DIM = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = value.get("records") or value.get("windows") or value.get("rows")
        else:
            rows = None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("manifest has no records")
    return rows


def current_time(row: dict[str, Any]) -> int:
    value = row.get("current_t", row.get("curr_time"))
    if value is None:
        raise KeyError("manifest row misses current_t/curr_time")
    return int(value)


def record_keys(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [f"{row['trajectory_id']}:{current_time(row)}" for row in rows]
    )


def stable_window_seed(base_seed: int, record_key: str, arm_offset: int) -> int:
    digest = hashlib.sha256(record_key.encode("utf-8")).digest()
    return int((base_seed + arm_offset + int.from_bytes(digest[:8], "little")) % (2**63 - 1))


class StatewiseSeedCEM(MultiModalCEM):
    """Batch-vectorized CEM with independent, batching-invariant window RNGs."""

    def __init__(self, config: CEMConfig, window_seeds: list[int]) -> None:
        super().__init__(config)
        self.window_seeds = window_seeds

    def sample(self, state: CEMState) -> torch.Tensor:
        expected_tail = (self.config.modes, HORIZON, ACTION_DIM)
        if tuple(state.means.shape[1:]) != expected_tail:
            raise ValueError("CEM state mean shape is invalid")
        if len(self.window_seeds) != state.means.shape[0]:
            raise ValueError("window seed count differs from CEM batch")
        shape = (
            self.config.modes,
            self.config.samples_per_mode,
            HORIZON,
            ACTION_DIM,
        )
        values = []
        for batch_index, seed in enumerate(self.window_seeds):
            generator = torch.Generator(device=state.means.device)
            generator.manual_seed(seed + state.iteration)
            white = torch.randn(
                shape,
                generator=generator,
                device=state.means.device,
                dtype=state.means.dtype,
            )
            rho = self.config.temporal_correlation
            correlated = white.clone()
            innovation = math.sqrt(1.0 - rho * rho)
            for horizon_index in range(1, HORIZON):
                correlated[..., horizon_index, :] = (
                    rho * correlated[..., horizon_index - 1, :]
                    + innovation * white[..., horizon_index, :]
                )
            values.append(
                state.means[batch_index, :, None]
                + state.stds[batch_index, :, None] * correlated
            )
        return torch.stack(values)


def load_pairs(
    run_root: Path,
    device: torch.device,
) -> tuple[list[tuple[int, LiteTokenFormerWMV2, TrajectoryQualityScorerV2]], dict[str, Any]]:
    pairs = []
    audit = {"seeds": []}
    config_names = {item.name for item in fields(LiteTokenFormerConfig)}
    for seed in SEEDS:
        wm_path = run_root / f"ltwm/seed{seed}/checkpoints/best.pt"
        scorer_path = run_root / f"scorer/seed{seed}/best.pt"
        wm_payload = torch.load(wm_path, map_location="cpu", weights_only=False)
        wm_config_payload = dict(wm_payload.get("config", {}))
        wm_config = LiteTokenFormerConfig(
            **{
                key: value
                for key, value in wm_config_payload.items()
                if key in config_names
            }
        )
        wm = LiteTokenFormerWMV2(wm_config)
        wm_result = wm.load_state_dict(
            wm_payload.get("model", wm_payload),
            strict=True,
        )
        if wm_result.missing_keys or wm_result.unexpected_keys:
            raise RuntimeError(f"world-model state mismatch for seed {seed}")
        wm = wm.eval().requires_grad_(False).to(device)

        scorer_payload = torch.load(
            scorer_path,
            map_location="cpu",
            weights_only=False,
        )
        scorer_config = TrajectoryQualityScorerConfig(
            **dict(scorer_payload["model_config"])
        )
        scorer = TrajectoryQualityScorerV2(scorer_config)
        scorer_result = scorer.load_state_dict(
            scorer_payload["model"],
            strict=True,
        )
        if scorer_result.missing_keys or scorer_result.unexpected_keys:
            raise RuntimeError(f"scorer state mismatch for seed {seed}")
        scorer = scorer.eval().requires_grad_(False).to(device)
        pairs.append((seed, wm, scorer))
        audit["seeds"].append(
            {
                "seed": seed,
                "world_model": str(wm_path.resolve()),
                "world_model_sha256": sha256(wm_path),
                "scorer": str(scorer_path.resolve()),
                "scorer_sha256": sha256(scorer_path),
                "scorer_world_model_sha256": scorer_payload.get(
                    "world_model_checkpoint_sha256"
                ),
                "selection_config": scorer_payload.get("selection_config"),
                "loss_config": scorer_payload["loss_config"],
                "action_support": scorer_payload["action_support"],
            }
        )
    wm_hashes = {
        row["seed"]: row["world_model_sha256"] for row in audit["seeds"]
    }
    for row in audit["seeds"]:
        bound = row["scorer_world_model_sha256"]
        if bound is not None and bound != wm_hashes[row["seed"]]:
            raise RuntimeError(
                f"seed {row['seed']} scorer is not bound to its world model"
            )
    return pairs, audit


def load_latents(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    encoded_cache: Path | None,
    latent_cache: Path | None,
    latent_identity_cache: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    expected = record_keys(rows)
    if (encoded_cache is None) == (latent_cache is None):
        raise ValueError("provide exactly one of --encoded-cache/--latent-cache")
    if encoded_cache is not None:
        payload = torch.load(
            encoded_cache,
            map_location="cpu",
            weights_only=False,
        )
        inputs = payload["inputs"]
        history = inputs["history"].contiguous()
        goal = inputs["target"][:, -1].contiguous()
        observed = np.asarray(
            [
                f"{str(uid).split('|')[1]}:{str(uid).split('|')[2]}"
                for uid in inputs["record_uids"]
            ]
        )
        if not np.array_equal(observed.astype(str), expected.astype(str)):
            raise RuntimeError("encoded cache record order differs from manifest")
        audit = {
            "kind": "strict_encoded_input",
            "path": str(encoded_cache.resolve()),
            "sha256": sha256(encoded_cache),
            "binding": payload.get("binding"),
            "goal_source": "available_goal_frame_only",
        }
    else:
        payload = load_npz(latent_cache)
        history = torch.from_numpy(payload["history"].copy())
        goal = torch.from_numpy(payload["goal"].copy())
        if latent_identity_cache is None:
            raise ValueError("--latent-identity-cache is required with --latent-cache")
        identity = load_npz(latent_identity_cache)
        observed = identity["record_key"].astype(str)
        if not np.array_equal(observed, expected.astype(str)):
            raise RuntimeError("latent identity cache differs from manifest")
        audit = {
            "kind": "fair_v70_latent_cache",
            "path": str(latent_cache.resolve()),
            "sha256": sha256(latent_cache),
            "identity_path": str(latent_identity_cache.resolve()),
            "identity_sha256": sha256(latent_identity_cache),
            "metadata": str(payload.get("metadata", np.asarray("")).item()),
            "goal_source": "available_goal_frame_only",
        }
    if tuple(history.shape) != (len(rows), 4, 256, 384):
        raise RuntimeError(f"history latent shape differs: {tuple(history.shape)}")
    if tuple(goal.shape) != (len(rows), 256, 384):
        raise RuntimeError(f"goal latent shape differs: {tuple(goal.shape)}")
    return history, goal, audit


@torch.inference_mode()
def predict_pairs(
    pairs: list[tuple[int, LiteTokenFormerWMV2, TrajectoryQualityScorerV2]],
    history: torch.Tensor,
    goal: torch.Tensor,
    actions: torch.Tensor,
    *,
    max_flat_batch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 4 or tuple(actions.shape[-2:]) != (4, 3):
        raise ValueError("actions must have shape [B,K,4,3]")
    batch, candidate_count = actions.shape[:2]
    if history.shape[0] != batch or goal.shape[0] != batch:
        raise ValueError("latent/action batch dimensions differ")
    device = actions.device
    current = history[:, -1]
    ate_rows = []
    rtp_rows = []
    per_window_chunk = max(1, max_flat_batch // batch)
    for _seed, wm, scorer in pairs:
        ate_chunks = []
        rtp_chunks = []
        for start in range(0, candidate_count, per_window_chunk):
            stop = min(start + per_window_chunk, candidate_count)
            chunk = actions[:, start:stop]
            local_k = stop - start
            expanded_history = (
                history[:, None]
                .expand(-1, local_k, -1, -1, -1)
                .reshape(batch * local_k, 4, 256, 384)
            )
            flat_actions = chunk.reshape(batch * local_k, 4, 3)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                future = wm(expanded_history, flat_actions)[
                    "predicted_tokens"
                ][:, 0].reshape(batch, local_k, 4, 256, 384)
                prediction = scorer(
                    current_latent=current,
                    goal_latent=goal,
                    candidate_actions=chunk,
                    predicted_future=future,
                )
            ate_chunks.append(prediction.ate.float())
            rtp_chunks.append(prediction.rtp.float())
        ate_rows.append(torch.cat(ate_chunks, dim=1))
        rtp_rows.append(torch.cat(rtp_chunks, dim=1))
    return torch.stack(ate_rows), torch.stack(rtp_rows)


def common_scales_and_bounds(pair_audit: dict[str, Any]) -> tuple[float, float, np.ndarray, np.ndarray]:
    loss = [row["loss_config"] for row in pair_audit["seeds"]]
    ate_scales = np.asarray([row["ate_scale"] for row in loss], dtype=np.float64)
    rtp_scales = np.asarray([row["rtp_scale"] for row in loss], dtype=np.float64)
    if not np.allclose(ate_scales, ate_scales[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("ATE scales differ across scorer seeds")
    if not np.allclose(rtp_scales, rtp_scales[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("RTP scales differ across scorer seeds")
    def action_bound(row: dict[str, Any], *, lower: bool) -> Any:
        support = row["action_support"]
        current_key = "lower" if lower else "upper"
        legacy_key = "lower_p01" if lower else "upper_p99"
        value = support.get(current_key, support.get(legacy_key))
        if value is None:
            raise RuntimeError(
                f"action support is missing both {current_key!r} and {legacy_key!r}"
            )
        return value

    lowers = np.asarray(
        [action_bound(row, lower=True) for row in pair_audit["seeds"]],
        dtype=np.float32,
    )
    uppers = np.asarray(
        [action_bound(row, lower=False) for row in pair_audit["seeds"]],
        dtype=np.float32,
    )
    lower = lowers.max(axis=0)
    upper = uppers.min(axis=0)
    if np.any(lower >= upper):
        raise RuntimeError("tri-seed action-support intersection is empty")
    return float(ate_scales[0]), float(rtp_scales[0]), lower, upper


def common_planning_config(pair_audit: dict[str, Any]) -> dict[str, Any]:
    configs = [row.get("selection_config") for row in pair_audit["seeds"]]
    if all(config is None for config in configs):
        return {"mode": "pareto_min", "rtp_weight": None}
    if any(config is None for config in configs):
        raise RuntimeError("selection config is missing for some scorer seeds")
    modes = {str(config.get("mode")) for config in configs}
    if modes != {"ate_primary_weighted_rtp"}:
        raise RuntimeError(f"unsupported or inconsistent selection modes: {modes}")
    weights = np.asarray(
        [float(config["rtp_weight"]) for config in configs],
        dtype=np.float64,
    )
    if not np.allclose(weights, weights[0], rtol=0.0, atol=1.0e-12):
        raise RuntimeError("RTP selection weights differ across scorer seeds")
    return {"mode": modes.pop(), "rtp_weight": float(weights[0])}


def refine_arm(
    *,
    pairs: list[tuple[int, LiteTokenFormerWMV2, TrajectoryQualityScorerV2]],
    history: torch.Tensor,
    goal: torch.Tensor,
    parents: torch.Tensor,
    record_key: list[str],
    arm_offset: int,
    ate_scale: float,
    rtp_scale: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    uncertainty_beta: float,
    rtp_weight: float | None,
    base_seed: int,
    max_flat_batch: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reference = parents[:, :1]
    reference_ate, reference_rtp = predict_pairs(
        pairs,
        history,
        goal,
        reference,
        max_flat_batch=max_flat_batch,
    )

    def cost_function(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ate, rtp = predict_pairs(
            pairs,
            history,
            goal,
            actions,
            max_flat_batch=max_flat_batch,
        )
        if rtp_weight is None:
            value = paired_conservative_joint_cost(
                ate,
                rtp,
                reference_ate,
                reference_rtp,
                ate_scale=ate_scale,
                rtp_scale=rtp_scale,
                uncertainty_beta=uncertainty_beta,
            )
            return value.cost, value.valid
        ate_gain = reference_ate - ate
        rtp_gain = rtp - reference_rtp
        lcb_ate = ate_gain.mean(dim=0) - uncertainty_beta * ate_gain.std(
            dim=0, unbiased=False
        )
        lcb_rtp = rtp_gain.mean(dim=0) - uncertainty_beta * rtp_gain.std(
            dim=0, unbiased=False
        )
        utility = lcb_ate / ate_scale + rtp_weight * lcb_rtp / rtp_scale
        valid = torch.isfinite(utility)
        cost = torch.where(valid, -utility, torch.full_like(utility, torch.inf))
        return cost, valid

    def generated_constraint(actions: torch.Tensor) -> torch.Tensor:
        return (
            torch.isfinite(actions).all(dim=(-1, -2))
            & ((actions >= lower) & (actions <= upper)).all(dim=(-1, -2))
        )

    config = CEMConfig(
        modes=4,
        population_size=16,
        elites_per_mode=1,
        iterations=3,
        mean_update_rate=0.50,
        std_update_rate=0.50,
        std_floor=0.02,
        std_ceiling=0.50,
        temporal_correlation=0.60,
        kmedoids_iterations=8,
        seed=base_seed,
    )
    seeds = [
        stable_window_seed(base_seed, value, arm_offset) for value in record_key
    ]
    planner = StatewiseSeedCEM(config, seeds)
    result = planner.refine(
        parents,
        cost_function,
        constraint_function=generated_constraint,
    )
    return (
        result.selected_actions,
        result.selected_costs,
        result.selected_archive_indices,
    )


def select(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    rows = load_manifest(args.manifest)
    expected = record_keys(rows)
    pool = load_npz(args.pool)
    if not np.array_equal(pool["record_key"].astype(str), expected.astype(str)):
        raise RuntimeError("NoMaD pool record identity/order differs")
    proposals = pool["nomad_x32_actions_h4"].astype(np.float32)
    if proposals.shape != (len(rows), 32, 4, 3):
        raise RuntimeError(f"NoMaD action pool shape differs: {proposals.shape}")
    history_cpu, goal_cpu, latent_audit = load_latents(
        dataset=args.dataset,
        rows=rows,
        encoded_cache=args.encoded_cache,
        latent_cache=args.latent_cache,
        latent_identity_cache=args.latent_identity_cache,
    )
    device = torch.device(args.device)
    pairs, pair_audit = load_pairs(args.run_root, device)
    ate_scale, rtp_scale, lower_np, upper_np = common_scales_and_bounds(pair_audit)
    planning_config = common_planning_config(pair_audit)
    lower = torch.as_tensor(lower_np, device=device)
    upper = torch.as_tensor(upper_np, device=device)

    incumbent16 = nomad_waypoints_to_local_actions(
        pool["nomad_k16_incumbent_waypoint_units"],
        dataset=args.dataset,
    ).astype(np.float32)
    incumbent32 = nomad_waypoints_to_local_actions(
        pool["nomad_k32_incumbent_waypoint_units"],
        dataset=args.dataset,
    ).astype(np.float32)
    parent_arrays = {
        "k16": np.concatenate((incumbent16[:, None], proposals[:, :16]), axis=1),
        "k32_batch0": np.concatenate(
            (incumbent32[:, None], proposals[:, :16]), axis=1
        ),
        "k32_batch1": np.concatenate(
            (incumbent32[:, None], proposals[:, 16:32]), axis=1
        ),
    }
    selected: dict[str, list[torch.Tensor]] = {
        "k16_action": [],
        "k16_cost": [],
        "k16_archive_index": [],
        "k32_action": [],
        "k32_cost": [],
        "k32_archive_index": [],
        "k32_source_batch": [],
    }
    for start in range(0, len(rows), args.state_batch_size):
        stop = min(start + args.state_batch_size, len(rows))
        history = history_cpu[start:stop].to(device=device, dtype=torch.float32)
        goal = goal_cpu[start:stop].to(device=device, dtype=torch.float32)
        keys = expected[start:stop].astype(str).tolist()
        arm_results = {}
        for arm_index, arm in enumerate(("k16", "k32_batch0", "k32_batch1")):
            parents = torch.from_numpy(parent_arrays[arm][start:stop]).to(device)
            arm_results[arm] = refine_arm(
                pairs=pairs,
                history=history,
                goal=goal,
                parents=parents,
                record_key=keys,
                arm_offset=10_000 * (arm_index + 1),
                ate_scale=ate_scale,
                rtp_scale=rtp_scale,
                lower=lower,
                upper=upper,
                uncertainty_beta=args.uncertainty_beta,
                rtp_weight=planning_config["rtp_weight"],
                base_seed=args.cem_seed,
                max_flat_batch=args.max_flat_batch,
            )
        k16_action, k16_cost, k16_index = arm_results["k16"]
        b0_action, b0_cost, b0_index = arm_results["k32_batch0"]
        b1_action, b1_cost, b1_index = arm_results["k32_batch1"]
        choose_b1 = b1_cost < b0_cost
        selected["k16_action"].append(k16_action.cpu())
        selected["k16_cost"].append(k16_cost.cpu())
        selected["k16_archive_index"].append(k16_index.cpu())
        selected["k32_action"].append(
            torch.where(choose_b1[:, None, None], b1_action, b0_action).cpu()
        )
        selected["k32_cost"].append(torch.minimum(b0_cost, b1_cost).cpu())
        selected["k32_archive_index"].append(
            torch.where(choose_b1, b1_index, b0_index).cpu()
        )
        selected["k32_source_batch"].append(choose_b1.to(torch.int64).cpu())
        print(
            json.dumps(
                {
                    "stage": "select",
                    "cohort": args.cohort_id,
                    "completed": stop,
                    "total": len(rows),
                    "gpu_memory_allocated_mib": (
                        torch.cuda.memory_allocated(device) / 1024**2
                        if device.type == "cuda"
                        else 0.0
                    ),
                }
            ),
            flush=True,
        )

    arrays = {
        name: torch.cat(parts).numpy()
        for name, parts in selected.items()
    }
    k16_path = integrate_local_actions(arrays["k16_action"]).astype(np.float32)
    k32_path = integrate_local_actions(arrays["k32_action"]).astype(np.float32)
    args.output_dir.mkdir(parents=True)
    selection_path = args.output_dir / "selected_actions.npz"
    write_npz_atomic(
        selection_path,
        record_key=expected,
        k16_actions_h4=arrays["k16_action"].astype(np.float32),
        k16_path_m=k16_path,
        k16_cost=arrays["k16_cost"].astype(np.float32),
        k16_archive_index=arrays["k16_archive_index"].astype(np.int64),
        k32_actions_h4=arrays["k32_action"].astype(np.float32),
        k32_path_m=k32_path,
        k32_cost=arrays["k32_cost"].astype(np.float32),
        k32_archive_index=arrays["k32_archive_index"].astype(np.int64),
        k32_source_batch=arrays["k32_source_batch"].astype(np.int64),
    )
    audit = {
        "schema": "new_v70_triseed_fair_h4_selection_v1",
        "created_at": utc_now(),
        "status": "actions_frozen_before_future_poses",
        "cohort_id": args.cohort_id,
        "dataset": args.dataset,
        "states": len(rows),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "pool": str(args.pool.resolve()),
        "pool_sha256": sha256(args.pool),
        "latent_input": latent_audit,
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256(selection_path),
        "world_model_scorer_pairs": pair_audit,
        "ensemble": {
            "member_seeds": list(SEEDS),
            "paired_world_model_and_scorer_per_seed": True,
            "aggregation": "paired seed-wise ATE/RTP gains",
            "uncertainty_beta": args.uncertainty_beta,
            "ate_scale": ate_scale,
            "rtp_scale": rtp_scale,
            "planning_config": planning_config,
        },
        "candidate_protocol": {
            "nested_nomad_k16_k32": True,
            "k16": "trimmed-mean incumbent plus first 16 NoMaD samples",
            "k32": (
                "shared K32 trimmed-mean incumbent plus disjoint NoMaD "
                "samples 0:16 and 16:32; lower-cost frozen branch retained"
            ),
        },
        "cem": {
            "iterations": 3,
            "population": 16,
            "modes": 4,
            "elites_per_mode": 1,
            "generated_candidates_per_state_per_branch": 48,
            "base_seed": args.cem_seed,
            "per_window_seed": "SHA256(record_key), batching invariant",
            "action_support": "intersection of three train-fit 1%-99% bounds",
            "action_support_lower": lower_np.tolist(),
            "action_support_upper": upper_np.tolist(),
        },
        "future_pose_opened": False,
        "test_metrics_opened": False,
    }
    write_json_atomic(args.output_dir / "selection_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def load_fair_metric_module() -> Any:
    path = (
        REPO_ROOT
        / "experiment_prep/fair_v70_benchmark_20260723/"
        "evaluate_fair_benchmark.py"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("fair_v70_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fair metric module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    selection_audit_path = args.selection_dir / "selection_audit.json"
    selection_audit = json.loads(selection_audit_path.read_text(encoding="utf-8"))
    if selection_audit.get("status") != "actions_frozen_before_future_poses":
        raise RuntimeError("selection was not frozen before pose access")
    if selection_audit.get("future_pose_opened") is not False:
        raise RuntimeError("selection audit reports future-pose access")
    selection_path = args.selection_dir / "selected_actions.npz"
    if sha256(selection_path) != selection_audit["selection_sha256"]:
        raise RuntimeError("selection hash differs from prelabel audit")
    rows = load_manifest(args.manifest)
    expected = record_keys(rows)
    selected = load_npz(selection_path)
    if not np.array_equal(selected["record_key"].astype(str), expected.astype(str)):
        raise RuntimeError("selected action record order differs from manifest")
    fair = load_fair_metric_module()
    truth, truth_yaw = fair.ground_truth(rows, args.trajectory_root)
    spacing = float(NWM_WAYPOINT_SPACING_METERS[args.dataset])
    values = {}
    summaries = {}
    clusters = np.asarray([str(row["trajectory_id"]) for row in rows])
    rng = np.random.default_rng(args.bootstrap_seed)
    for key, action_name, path_name in (
        ("new_v70_triseed_k16", "k16_actions_h4", "k16_path_m"),
        ("new_v70_triseed_k32", "k32_actions_h4", "k32_path_m"),
    ):
        actions = selected[action_name].astype(np.float32)
        prediction = selected[path_name].astype(np.float32)
        predicted_yaw = fair.wrap_to_pi(actions[..., 2].sum(axis=1))
        value = fair.metric_values(
            prediction,
            predicted_yaw,
            truth,
            truth_yaw,
            spacing,
            0.4,
            0.6,
        )
        values[key] = value
        summaries[key] = {}
        for metric in (
            "ate_nwm",
            "rte_nwm",
            "fde_nwm",
            "success_fde_1step",
            "success_fde_2step",
            "success_pose_1step_30deg",
            "success_pose_2step_30deg",
        ):
            summaries[key][metric] = fair.summarize(
                value[metric],
                clusters,
                args.bootstrap,
                rng,
                binary=metric.startswith("success_"),
            )
    args.output_dir.mkdir(parents=True)
    import csv

    per_state_path = args.output_dir / "per_state_metrics.csv"
    with per_state_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "record_key",
                "trajectory_id",
                "method",
                "ate_nwm",
                "rte_nwm",
                "fde_nwm",
                "success_fde_1step",
                "success_fde_2step",
                "success_pose_1step_30deg",
                "success_pose_2step_30deg",
            ]
        )
        for method, value in values.items():
            for index, row in enumerate(rows):
                writer.writerow(
                    [
                        expected[index],
                        row["trajectory_id"],
                        method,
                        *[
                            float(value[name][index])
                            for name in (
                                "ate_nwm",
                                "rte_nwm",
                                "fde_nwm",
                                "success_fde_1step",
                                "success_fde_2step",
                                "success_pose_1step_30deg",
                                "success_pose_2step_30deg",
                            )
                        ],
                    ]
                )
    result = {
        "schema": "new_v70_triseed_fair_h4_metrics_v1",
        "created_at": utc_now(),
        "status": "complete_after_frozen_selection",
        "cohort_id": args.cohort_id,
        "dataset": args.dataset,
        "states": len(rows),
        "spacing_m": spacing,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "selection_audit": str(selection_audit_path.resolve()),
        "selection_audit_sha256": sha256(selection_audit_path),
        "selection": str(selection_path.resolve()),
        "selection_sha256": sha256(selection_path),
        "label_boundary": (
            "future poses opened only after K16/K32 tri-seed selections and "
            "hash bindings were frozen"
        ),
        "metric_compatibility": {
            "ate": "EVO APE translation RMSE, NWM average-step units",
            "rte": "EVO RPE translation RMSE, delta=1 frame",
            "fde": "H4 endpoint displacement in NWM average-step units",
            "sr_at_1": "FDE <= 1 average step",
            "sr_at_2": "FDE <= 2 average steps",
        },
        "summaries": summaries,
        "bootstrap": {
            "iterations": args.bootstrap,
            "seed": args.bootstrap_seed,
            "unit": "trajectory cluster",
        },
        "per_state_metrics": str(per_state_path.resolve()),
        "per_state_metrics_sha256": sha256(per_state_path),
    }
    write_json_atomic(args.output_dir / "results.json", result)
    compact = {}
    for method, summary in summaries.items():
        compact[method] = {
            "ATE": summary["ate_nwm"]["mean"],
            "RTE": summary["rte_nwm"]["mean"],
            "FDE": summary["fde_nwm"]["mean"],
            "SR@1": summary["success_fde_1step"]["rate"],
            "SR@2": summary["success_fde_2step"]["rate"],
        }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument(
        "--dataset",
        choices=("recon", "scand", "sacson"),
        required=True,
    )
    select_parser.add_argument("--cohort-id", required=True)
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--pool", type=Path, required=True)
    select_parser.add_argument("--encoded-cache", type=Path)
    select_parser.add_argument("--latent-cache", type=Path)
    select_parser.add_argument("--latent-identity-cache", type=Path)
    select_parser.add_argument("--run-root", type=Path, required=True)
    select_parser.add_argument("--output-dir", type=Path, required=True)
    select_parser.add_argument("--state-batch-size", type=int, default=4)
    select_parser.add_argument("--max-flat-batch", type=int, default=32)
    select_parser.add_argument("--uncertainty-beta", type=float, default=2.0)
    select_parser.add_argument("--cem-seed", type=int, default=17)
    select_parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    select_parser.set_defaults(handler=select)

    metric_parser = subparsers.add_parser("metrics")
    metric_parser.add_argument(
        "--dataset",
        choices=("recon", "scand", "sacson"),
        required=True,
    )
    metric_parser.add_argument("--cohort-id", required=True)
    metric_parser.add_argument("--manifest", type=Path, required=True)
    metric_parser.add_argument("--trajectory-root", type=Path, required=True)
    metric_parser.add_argument("--selection-dir", type=Path, required=True)
    metric_parser.add_argument("--output-dir", type=Path, required=True)
    metric_parser.add_argument("--bootstrap", type=int, default=10_000)
    metric_parser.add_argument("--bootstrap-seed", type=int, default=20_260_728)
    metric_parser.set_defaults(handler=metrics)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "state_batch_size", 1) <= 0:
        raise ValueError("state batch size must be positive")
    if getattr(args, "max_flat_batch", 1) <= 0:
        raise ValueError("max flat batch must be positive")
    if getattr(args, "uncertainty_beta", 0.0) < 0:
        raise ValueError("uncertainty beta must be non-negative")
    if getattr(args, "bootstrap", 10_000) < 1_000:
        raise ValueError("bootstrap iterations must be at least 1000")
    args.handler(args)


if __name__ == "__main__":
    main()
