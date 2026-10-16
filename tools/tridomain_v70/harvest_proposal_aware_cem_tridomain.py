#!/usr/bin/env python3
"""Mine old-V70 CEM proposals and hard negatives on internal three-domain data.

The planner is the original candidate-level safe-elite CEM protocol:
three iterations, sixteen challengers, four safety elites, q=0.999 spacing
support, and incumbent retention when a safe elite set is unavailable.
The proposal-aware 815-D model is scored before any outcome label is opened.
Only after a pre-outcome seal is written are internal train/calibration labels
used to construct regression/classification targets and false-safe hard
negatives.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.costs import latent_trajectory_features  # noqa: E402
from planning.nomad_action_adapter import (  # noqa: E402
    nomad_waypoints_to_local_actions,
)
from scoring.spacing_support_v1 import (  # noqa: E402
    score_spacing_envelope,
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
    build_universal_targets,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    metric_labels,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    pool_static_features,
)
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    conservative_energy,
    sample_population,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
    initial_proposal,
    proposal_summary,
    safe_elite_update,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
)
from tools.tridomain_v70.prepare_proposal_aware_cem_fixed_pool import (  # noqa: E402
    MODEL_SEEDS,
    ShardedLatents,
    load_world_models,
    read_rows,
)


DATASETS = ("recon", "scand", "sacson")
MODEL_DOMAIN_INDEX = {"recon": 0, "scand": 1, "sacson": 2}
SPACING_M = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}
CANDIDATES = 17
ITERATIONS = 3
POPULATION = 16
ELITE_COUNT = 4
SUPPORT_QUANTILE = 0.999
PROPOSAL_SCALE = 0.70
COVARIANCE_FLOOR = 0.015


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument(
        "--partition",
        choices=("train-fit", "scorer-cal"),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixed-features", type=Path, required=True)
    parser.add_argument("--fixed-targets", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--stage1-result", type=Path, required=True)
    for seed in MODEL_SEEDS:
        parser.add_argument(f"--wm-{seed}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wm-batch-size", type=int, default=128)
    parser.add_argument("--candidate-chunk", type=int, default=17)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path, required: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise RuntimeError(f"{path} missing arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in archive.files}


def reconstructed_history(
    candidates: np.ndarray,
    base_action_features: np.ndarray,
) -> np.ndarray:
    """Recompute the exact old-V70 38-D causal motion-history block."""

    raw = np.asarray(base_action_features, dtype=np.float32)[..., :121]
    state = raw[:, 0, 83:101].astype(np.float32)
    reference_specific = raw[:, 1:, 101:111].astype(np.float32)
    center = np.median(reference_specific, axis=1, keepdims=True)
    scale = np.maximum(reference_specific.std(axis=1, keepdims=True), 1.0e-5)

    candidates_m = np.asarray(candidates, dtype=np.float32) * np.float32(0.25)
    count, population, horizon, _ = candidates_m.shape
    local_velocity = state[:, :6].reshape(count, 3, 2)
    speed = state[:, 6:9]
    speed_weighted = state[:, 14]
    yaw_rate = state[:, 17]
    weights = np.asarray((0.2, 0.3, 0.5), dtype=np.float32)
    velocity_weighted = (local_velocity * weights[None, :, None]).sum(axis=1)
    steps = np.arange(1, horizon + 1, dtype=np.float32)[None, :, None]
    expected_last = steps * local_velocity[:, -1:, :]
    expected_weighted = steps * velocity_weighted[:, None, :]
    expected_twist = np.zeros((count, horizon, 2), dtype=np.float32)
    for future_step in range(horizon):
        heading = (future_step + 0.5) * yaw_rate
        increment = speed_weighted[:, None] * np.stack(
            (np.cos(heading), np.sin(heading)), axis=-1
        )
        if future_step:
            expected_twist[:, future_step] = expected_twist[:, future_step - 1]
        expected_twist[:, future_step] += increment

    origin = np.zeros((count, population, 1, 2), dtype=np.float32)
    candidate_steps = np.diff(
        np.concatenate((origin, candidates_m), axis=2), axis=2
    )
    candidate_speed = np.linalg.norm(candidate_steps, axis=-1)

    def residual(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(candidates_m - expected[:, None], axis=-1)
        return distance.mean(axis=2), distance[:, :, -1]

    last_mean, last_final = residual(expected_last)
    weighted_mean, weighted_final = residual(expected_weighted)
    twist_mean, twist_final = residual(expected_twist)
    expected_direction = velocity_weighted / np.maximum(
        np.linalg.norm(velocity_weighted, axis=1, keepdims=True), 1.0e-6
    )
    candidate_direction = candidates_m[:, :, -1] / np.maximum(
        np.linalg.norm(candidates_m[:, :, -1], axis=-1, keepdims=True),
        1.0e-6,
    )
    specifics = np.stack(
        (
            last_mean,
            last_final,
            weighted_mean,
            weighted_final,
            twist_mean,
            twist_final,
            np.abs(candidate_speed.mean(axis=2) - speed_weighted[:, None]),
            np.abs(candidate_speed[:, :, 0] - speed[:, -1, None]),
            np.abs(
                np.linalg.norm(candidates_m[:, :, -1], axis=-1) / horizon
                - speed_weighted[:, None]
            ),
            1.0
            - np.einsum(
                "nkd,nd->nk", candidate_direction, expected_direction
            ),
        ),
        axis=-1,
    ).astype(np.float32)
    relative = np.clip((specifics - center) / scale, -8.0, 8.0)
    return np.concatenate(
        (
            np.repeat(state[:, None], population, axis=1),
            specifics,
            relative,
        ),
        axis=-1,
    ).astype(np.float32)


def reconstruct_action_features(
    candidates: np.ndarray,
    reference: np.ndarray,
    base_action_features: np.ndarray,
) -> np.ndarray:
    action_set = action_set_features(candidates, reference)
    history = reconstructed_history(candidates, base_action_features)
    raw = np.concatenate((action_set, history), axis=-1).astype(np.float32)
    result = np.concatenate((raw, raw - raw[:, :1]), axis=-1)
    if result.shape != (len(candidates), CANDIDATES, 242):
        raise RuntimeError(f"old V70 action feature shape changed: {result.shape}")
    return result.astype(np.float32)


def candidate_ltwm_features(
    rows: list[dict[str, Any]],
    candidate_actions: np.ndarray,
    *,
    latent_root: Path,
    models: dict[int, torch.nn.Module],
    device: str,
    batch_size: int,
    candidate_chunk: int,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    loader = ShardedLatents(latent_root)
    static_parts: list[np.ndarray] = []
    scalar_parts: dict[int, list[np.ndarray]] = {
        seed: [] for seed in MODEL_SEEDS
    }
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        history_np, goal_np = loader.batch(rows[start:end])
        history = torch.from_numpy(history_np).to(device)
        goal = torch.from_numpy(goal_np).to(device)
        actions = torch.from_numpy(candidate_actions[start:end]).to(device)
        with torch.inference_mode():
            static_parts.append(
                pool_static_features(history, goal)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            for seed in MODEL_SEEDS:
                local_parts = []
                for candidate_start in range(
                    0, CANDIDATES, candidate_chunk
                ):
                    candidate_end = min(
                        candidate_start + candidate_chunk, CANDIDATES
                    )
                    context = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if device == "cuda"
                        else torch.autocast("cpu", enabled=False)
                    )
                    with context:
                        future = models[seed](
                            history,
                            actions[:, candidate_start:candidate_end],
                        )["predicted_tokens"]
                    scalar, _ = latent_trajectory_features(
                        future.float(), goal, history[:, -1]
                    )
                    local_parts.append(
                        scalar.cpu().numpy().astype(np.float32)
                    )
                local = np.concatenate(local_parts, axis=1)
                if local.shape != (
                    end - start,
                    CANDIDATES,
                    28,
                ):
                    raise RuntimeError(
                        f"LTWM scalar shape changed: {local.shape}"
                    )
                scalar_parts[seed].append(local)
        print(
            canonical_json(
                {
                    "stage": stage,
                    "completed": end,
                    "total": len(rows),
                    "elapsed_sec": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    stack = np.stack(
        [
            np.concatenate(scalar_parts[seed])
            for seed in MODEL_SEEDS
        ]
    )
    aggregate = np.concatenate(
        (stack.mean(0), stack.std(0), stack.min(0), stack.max(0)),
        axis=-1,
    ).astype(np.float32)
    if aggregate.shape != (len(rows), CANDIDATES, 112):
        raise RuntimeError(f"LTWM aggregate shape changed: {aggregate.shape}")
    return np.concatenate(static_parts), aggregate


def make_feature_artifact(
    *,
    rows: list[dict[str, Any]],
    candidates: np.ndarray,
    reference: np.ndarray,
    base_action_features: np.ndarray,
    latent_root: Path,
    models: dict[int, torch.nn.Module],
    dataset: str,
    partition: str,
    iteration: int,
    envelope: dict[str, Any],
    device: str,
    batch_size: int,
    candidate_chunk: int,
    output: Path,
) -> dict[str, np.ndarray]:
    spacing = SPACING_M[dataset]
    bridge = np.float32(spacing / 0.25)
    native_candidates = candidates / bridge
    actions_h4 = nomad_waypoints_to_local_actions(
        native_candidates, dataset=dataset
    ).astype(np.float32)
    action_features = reconstruct_action_features(
        candidates, reference, base_action_features
    )
    static, raw_ltwm = candidate_ltwm_features(
        rows,
        actions_h4,
        latent_root=latent_root,
        models=models,
        device=device,
        batch_size=batch_size,
        candidate_chunk=candidate_chunk,
        stage=f"cem_{dataset}_{partition}_iter{iteration}",
    )
    support, guard = score_spacing_envelope(
        envelope,
        actions_h4,
        np.full(len(rows), spacing, dtype=np.float32),
    )
    threshold = threshold_for_quantile(envelope, SUPPORT_QUANTILE)
    majority_support = guard & (support <= threshold)
    majority_support[:, 0] = True
    support_by_seed = np.repeat(
        majority_support[None], len(MODEL_SEEDS), axis=0
    )
    record_key = np.asarray(
        [
            f"{row['trajectory_id']}:{int(row['current_t'])}"
            for row in rows
        ]
    )
    family = np.asarray(
        [
            f"{dataset}::{row.get('route_id', row.get('route_group', row.get('bag_id', row['trajectory_id'])))}"
            for row in rows
        ]
    )
    metadata = {
        "schema": "recon_dense_dinov3_feature_artifact_v1",
        "created_at": utc_now(),
        "split": f"{dataset}_{partition}_proposal_aware_v70_cem_iter{iteration}",
        "states": len(rows),
        "candidates": CANDIDATES,
        "candidate_pool": "incumbent_plus_continuous_cem_population16",
        "feature_dimensions": {
            "action": 242,
            "static": 210,
            "raw_ltwm": 112,
            "scorer_v2_aux": 10,
            "support": 1,
        },
        "labels_accessed": False,
        "labels_in_output": False,
        "generic_ate_rtp_scorer_used_as_input": False,
        "fixed_pool_only": False,
        "cem": True,
        "cem_iteration": iteration,
        "dataset": dataset,
        'model_domain': dataset,
        'model_domain_index': MODEL_DOMAIN_INDEX[dataset],
        "waypoint_spacing_m": spacing,
    }
    zeros = np.zeros(
        (len(MODEL_SEEDS), len(rows), CANDIDATES), dtype=np.float32
    )
    arrays = {
        "candidates": candidates.astype(np.float32),
        "actions_h4": actions_h4,
        "action_features": action_features,
        "static_features": np.repeat(
            static[:, None], CANDIDATES, axis=1
        ).astype(np.float32),
        "raw_ltwm_features": raw_ltwm,
        "scorer_v2_aux_features": np.zeros(
            (len(rows), CANDIDATES, 10), dtype=np.float32
        ),
        "support_features": majority_support[..., None].astype(np.float32),
        "ate_by_seed": zeros,
        "rtp_by_seed": zeros.copy(),
        "support_by_seed": support_by_seed,
        "majority_support": majority_support,
        "record_key": record_key,
        "family": family,
        "metadata": np.asarray(canonical_json(metadata)),
    }
    write_npz(output, **arrays)
    write_json(
        output.with_suffix(".audit.json"),
        {
            **metadata,
            "output": str(output),
            "output_sha256": file_sha256(output),
        },
    )
    return arrays


def score_artifact(
    data: dict[str, np.ndarray],
    *,
    checkpoints: list[dict[str, Any]],
    policy: dict[str, float],
    dual_temperature: float,
    harm_temperature: float,
    envelope: dict[str, Any],
    dataset: str,
    device: str,
) -> dict[str, np.ndarray]:
    feature, support, guard = build_universal_features(
        data,
        spacing_m=SPACING_M[dataset],
        support_envelope=envelope,
    )
    valid = guard & (
        support <= threshold_for_quantile(envelope, SUPPORT_QUANTILE)
    )
    valid[:, 0] = True
    domain = np.full(
        len(feature), MODEL_DOMAIN_INDEX[dataset], dtype=np.int64
    )
    prediction = ensemble_predict(
        checkpoints, feature, domain, device=device
    )
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    safe = candidate_safety_mask(
        prediction, p_dual, p_harm, valid, policy
    )
    return {
        **prediction,
        "energy": energy,
        "p_dual": p_dual,
        "p_harm": p_harm,
        "valid": valid,
        "safe": safe,
        "support": support,
        "guard": guard,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    if min(args.wm_batch_size, args.candidate_chunk) <= 0:
        raise ValueError("batch and chunk values must be positive")
    if args.candidate_chunk > CANDIDATES:
        raise ValueError("candidate chunk cannot exceed 17")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    wm_paths = {
        seed: getattr(args, f'wm_{seed}') for seed in MODEL_SEEDS
    }
    wm_bindings = {
        str(seed): {
            'path': str(wm_paths[seed].resolve(strict=True)),
            'sha256': file_sha256(wm_paths[seed]),
        }
        for seed in MODEL_SEEDS
    }
    result = json.loads(args.stage1_result.read_text(encoding="utf-8"))
    if (
        result.get("decision")
        != "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_FROZEN"
    ):
        raise RuntimeError("stage1 proposal-aware ensemble is not frozen")
    if result.get("test_accessed") or result.get("outer_val_accessed"):
        raise RuntimeError("stage1 seal reports forbidden evaluation access")
    expected_domain_index = {'recon': 0, 'scand': 1, 'sacson': 2}
    if result.get('model_domain_index') != expected_domain_index:
        raise RuntimeError(
            'stage1 checkpoint is not an independently conditioned '
            'three-domain model'
        )
    calibration = result["calibration"][args.dataset]
    # Stage 1 is a mining policy, not a deployment policy.  An infeasible
    # frontier is allowed here specifically so its predicted-safe/true-harm
    # mistakes become hard negatives.  Final V70 deployment still requires
    # per-domain feasibility.
    checkpoints = result["checkpoints"]["full"]
    support_path = Path(result["support"])
    envelope = load_envelopes(support_path)[args.dataset]
    rows = read_rows(args.manifest)
    base = load_features(args.fixed_features, labels_expected=False)
    candidate = load_npz(
        args.candidate_pool,
        (
            "candidates_waypoint_units",
            "nomad_x16_waypoint_units",
            "record_key",
        ),
    )
    expected = np.asarray(
        [
            f"{row['trajectory_id']}:{int(row['current_t'])}"
            for row in rows
        ]
    )
    if not (
        np.array_equal(base["record_key"].astype(str), expected)
        and np.array_equal(candidate["record_key"].astype(str), expected)
    ):
        raise RuntimeError("manifest/fixed-feature/candidate identity mismatch")
    spacing = SPACING_M[args.dataset]
    bridge = np.float32(spacing / 0.25)
    incumbent = (
        candidate["candidates_waypoint_units"][:, 0].astype(np.float32)
        * bridge
    )
    reference = (
        candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge
    )
    base_candidates = np.concatenate(
        (incumbent[:, None], reference), axis=1
    )
    if not np.array_equal(
        base_candidates, base["candidates"].astype(np.float32)
    ):
        raise RuntimeError("fixed feature/candidate waypoint bridge mismatch")

    args.output_dir.mkdir(parents=True)
    models = load_world_models(wm_paths, args.device)
    fixed_score = score_artifact(
        base,
        checkpoints=checkpoints,
        policy=calibration["policy"],
        dual_temperature=float(calibration["dual_temperature"]),
        harm_temperature=float(calibration["harm_temperature"]),
        envelope=envelope,
        dataset=args.dataset,
        device=args.device,
    )
    mean, covariance = initial_proposal(
        reference,
        floor=COVARIANCE_FLOOR,
        scale=PROPOSAL_SCALE,
    )
    mean, covariance, elite, updated, safe_count = safe_elite_update(
        base_candidates,
        fixed_score["energy"],
        fixed_score["safe"],
        previous_mean=mean,
        previous_covariance=covariance,
        elite_count=ELITE_COUNT,
        floor=COVARIANCE_FLOOR,
        scale=PROPOSAL_SCALE,
    )
    rng = np.random.default_rng(
        20260730
        + {"recon": 0, "scand": 100000, "sacson": 200000}[args.dataset]
        + (0 if args.partition == "train-fit" else 10000)
    )
    iteration_records = []
    for iteration in range(ITERATIONS):
        directory = args.output_dir / f"iteration_{iteration:02d}"
        directory.mkdir()
        before = proposal_summary(mean, covariance)
        candidates, sampling = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            population=POPULATION,
            rng=rng,
            support_envelope=envelope,
            support_threshold=threshold_for_quantile(
                envelope, SUPPORT_QUANTILE
            ),
            spacing_m=spacing,
        )
        population_path = directory / "population.npz"
        write_npz(
            population_path,
            candidates_feature_waypoint_units=candidates,
            candidates_native_waypoint_units=candidates / bridge,
            proposal_mean=mean,
            proposal_covariance=covariance,
            elite_index_from_previous=elite,
            record_key=expected,
        )
        feature_path = directory / "base_features.npz"
        feature = make_feature_artifact(
            rows=rows,
            candidates=candidates,
            reference=reference,
            base_action_features=base["action_features"],
            latent_root=args.latent_root,
            models=models,
            dataset=args.dataset,
            partition=args.partition,
            iteration=iteration,
            envelope=envelope,
            device=args.device,
            batch_size=args.wm_batch_size,
            candidate_chunk=args.candidate_chunk,
            output=feature_path,
        )
        score = score_artifact(
            feature,
            checkpoints=checkpoints,
            policy=calibration["policy"],
            dual_temperature=float(calibration["dual_temperature"]),
            harm_temperature=float(calibration["harm_temperature"]),
            envelope=envelope,
            dataset=args.dataset,
            device=args.device,
        )
        next_mean, next_covariance, next_elite, next_updated, next_safe_count = (
            safe_elite_update(
                candidates,
                score["energy"],
                score["safe"],
                previous_mean=mean,
                previous_covariance=covariance,
                elite_count=ELITE_COUNT,
                floor=COVARIANCE_FLOOR,
                scale=PROPOSAL_SCALE,
            )
        )
        score_path = directory / "frozen_scores.npz"
        write_npz(
            score_path,
            energy=score["energy"],
            mean=score["mean"],
            std=score["std"],
            dual_probability=score["p_dual"],
            harm_probability=score["p_harm"],
            rank_logit=score["rank_logit"],
            valid=score["valid"],
            safe=score["safe"],
            support=score["support"],
            physical_guard=score["guard"],
            elite_index_for_update=next_elite,
            proposal_updated=next_updated,
            safe_count=next_safe_count,
            record_key=expected,
        )
        iteration_records.append(
            {
                "iteration": iteration,
                "population": str(population_path),
                "population_sha256": file_sha256(population_path),
                "features": str(feature_path),
                "features_sha256": file_sha256(feature_path),
                "frozen_scores": str(score_path),
                "frozen_scores_sha256": file_sha256(score_path),
                "sampling": sampling,
                "valid_challengers": int(score["valid"][:, 1:].sum()),
                "safe_challengers": int(score["safe"][:, 1:].sum()),
                "updated_windows": int(next_updated.sum()),
                "safe_shortfall_windows": int((~next_updated).sum()),
                "proposal_before": before,
                "proposal_after": proposal_summary(
                    next_mean, next_covariance
                ),
            }
        )
        mean, covariance, elite, updated, safe_count = (
            next_mean,
            next_covariance,
            next_elite,
            next_updated,
            next_safe_count,
        )
        del feature, score
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    freeze_path = args.output_dir / "preoutcome_freeze.json"
    freeze = {
        "schema": "proposal_aware_v70_tridomain_cem_preoutcome_freeze_v1",
        "status": "candidates_scores_and_proposals_frozen_before_internal_labels",
        "created_at": utc_now(),
        "dataset": args.dataset,
        "partition": args.partition,
        "states": len(rows),
        "stage1_result": str(args.stage1_result),
        "stage1_result_sha256": file_sha256(args.stage1_result),
        'model_domain_index': MODEL_DOMAIN_INDEX[args.dataset],
        'world_models': wm_bindings,
        "stage1_policy": calibration["policy"],
        "stage1_calibration_feasible": bool(calibration.get("feasible")),
        "dual_temperature": calibration["dual_temperature"],
        "harm_temperature": calibration["harm_temperature"],
        "support": str(support_path),
        "support_sha256": file_sha256(support_path),
        "iterations": iteration_records,
        "outcomes_used_by_cem": False,
        "fixed_targets_opened_before_freeze": False,
        "outer_val_accessed": False,
        "test_accessed": False,
    }
    write_json(freeze_path, freeze)

    target = load_npz(
        args.fixed_targets,
        ("record_key", "ground_truth_local_m"),
    )
    if not np.array_equal(target["record_key"].astype(str), expected):
        raise RuntimeError("fixed target identity mismatch")
    truth = target["ground_truth_local_m"].astype(np.float32)
    hard_total = 0
    for record in iteration_records:
        directory = args.output_dir / f"iteration_{record['iteration']:02d}"
        population = load_npz(
            directory / "population.npz",
            ("candidates_feature_waypoint_units", "record_key"),
        )
        score = load_npz(
            directory / "frozen_scores.npz",
            ("safe", "record_key"),
        )
        if not (
            np.array_equal(population["record_key"].astype(str), expected)
            and np.array_equal(score["record_key"].astype(str), expected)
        ):
            raise RuntimeError("frozen artifact identity mismatch")
        labels = metric_labels(
            population["candidates_feature_waypoint_units"],
            truth,
            0.25,
            0.4,
            0.6,
        )
        universal = build_universal_targets(labels["targets"])
        hard = score["safe"].astype(bool) & universal["harm"].astype(bool)
        hard[:, 0] = False
        target_path = directory / "targets.npz"
        write_npz(
            target_path,
            record_key=expected,
            family=base["family"].astype(str),
            targets=labels["targets"],
            absolute_metrics=labels["values"],
            metadata=np.asarray(
                canonical_json(
                    {
                        "schema": "proposal_aware_v70_tridomain_targets_v1",
                        "dataset": args.dataset,
                        "partition": args.partition,
                        "cem_iteration": record["iteration"],
                        "preoutcome_freeze_sha256": file_sha256(freeze_path),
                        "RTP_used_by_proposal_aware_model": False,
                        "external_test_accessed": False,
                    }
                )
            ),
        )
        hard_path = directory / "hard_negatives.npz"
        write_npz(
            hard_path,
            record_key=expected,
            hard_negative=hard,
            metadata=np.asarray(
                canonical_json(
                    {
                        "definition": (
                            "stage1 predicted safe AND true proposal-aware harm"
                        ),
                        "multiplier_for_final_training": 6.0,
                    }
                )
            ),
        )
        record["targets"] = str(target_path)
        record["targets_sha256"] = file_sha256(target_path)
        record["hard_negatives"] = str(hard_path)
        record["hard_negatives_sha256"] = file_sha256(hard_path)
        record["false_safe_harm_candidates"] = int(hard.sum())
        hard_total += int(hard.sum())

    final = {
        "schema": "proposal_aware_v70_tridomain_cem_harvest_result_v1",
        "status": "complete",
        "created_at": utc_now(),
        "dataset": args.dataset,
        "partition": args.partition,
        "states": len(rows),
        'model_domain_index': MODEL_DOMAIN_INDEX[args.dataset],
        'world_models': wm_bindings,
        "iterations": iteration_records,
        "hard_negative_definition": (
            "stage1 predicted safe AND true proposal-aware harm"
        ),
        "hard_negative_total": hard_total,
        "preoutcome_freeze": str(freeze_path),
        "preoutcome_freeze_sha256": file_sha256(freeze_path),
        "labels_opened_only_after_freeze": True,
        "generic_absolute_ate_rtp_scorer_used": False,
        "RTP_used_by_dual_or_harm_supervision": False,
        "outer_val_accessed": False,
        "test_accessed": False,
        "decision": "CEM_HARD_NEGATIVES_READY",
    }
    write_json(args.output_dir / "result.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
