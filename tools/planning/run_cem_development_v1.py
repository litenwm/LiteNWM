#!/usr/bin/env python3
"""Run label-free, development-only CEM with the frozen universal raw scorer.

Each CEM population contains the incumbent plus sixteen continuous challengers,
so the frozen setwise scorer sees the same K17 cardinality used in training.
The optimizer never reads future poses or outcome labels.  It repeatedly:

1. samples waypoint paths from a per-window Gaussian;
2. rejects paths outside the frozen spacing-aware physical support;
3. runs the three frozen LTWM rollouts and the frozen universal-D ensemble;
4. updates the proposal from the lowest conservative-risk elites.

The final actions, selections, manifests, model inputs, and SHA-256 bindings are
written before a separate outcome evaluator is allowed to open future poses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from planning.nomad_action_adapter import nomad_waypoints_to_local_actions  # noqa: E402
from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from scoring.spacing_support_v1 import (  # noqa: E402
    normalized_action_vector,
    score_spacing_envelope,
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    canonical_json,
    checkpoint_pairs,
    export_features,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.evaluation.freeze_recon_fresh_outer_universal_v2 import (  # noqa: E402
    arm_policy,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    selected_for_policy,
    sigmoid,
)


DEFAULT_ROOT = REPO_ROOT / "outputs/cem_development_v1"
TRAINING_RESULT = (
    REPO_ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
)
NATIVE_POLICY = (
    REPO_ROOT
    / "outputs/recon_dense_meta_dinov3_v1/three_arm_training/frozen_policy.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, default=TRAINING_RESULT)
    parser.add_argument("--native-policy", type=Path, default=NATIVE_POLICY)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--proposal-scale", type=float, default=0.70)
    parser.add_argument("--covariance-floor", type=float, default=0.015)
    parser.add_argument("--support-quantile", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wm-batch-size", type=int, default=4)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(canonical_json(list(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def domain_policy(
    training: dict[str, Any], domain: str
) -> tuple[dict[str, float], float, float]:
    frontier = training["arms"]["D"]["calibration"]["frontier"]["domains"][domain]
    temperature = training["arms"]["D"]["temperature"]["domains"][domain]
    return (
        frontier["policy"],
        float(temperature["deployment_dual_temperature"]),
        float(temperature["deployment_harm_temperature"]),
    )


def conservative_energy(
    prediction: dict[str, np.ndarray],
    *,
    policy: dict[str, float],
    dual_temperature: float,
    harm_temperature: float,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a frozen-scorer-only conservative objective; lower is better."""

    z = float(policy["z"])
    ucb = prediction["mean"] + z * prediction["std"]
    p_dual = sigmoid(prediction["dual_logit"] / dual_temperature)
    p_harm = sigmoid(prediction["harm_logit"] / harm_temperature)
    rank = prediction["rank_logit"].astype(np.float64)
    rank_scale = np.maximum(rank[:, 1:].std(axis=1, keepdims=True), 1.0e-3)
    rank_z = (rank - rank[:, :1]) / rank_scale
    energy = (
        ucb[..., 0]
        + 0.25 * ucb[..., 1]
        + 0.25 * np.maximum(ucb[..., 2], 0.0)
        + 0.50 * ucb[..., 3]
        + 0.15 * p_harm
        - 0.05 * p_dual
        - 0.03 * rank_z
    )
    energy = np.where(valid, energy, np.inf)
    energy[:, 0] = np.inf
    return energy.astype(np.float32), p_dual.astype(np.float32), p_harm.astype(np.float32)


def regularized_covariance(
    elite: np.ndarray,
    *,
    fallback: np.ndarray,
    floor: float,
    scale: float,
) -> np.ndarray:
    dim = elite.shape[-1]
    if len(elite) >= 2:
        covariance = np.cov(elite, rowvar=False, ddof=1)
    else:
        covariance = np.diag(np.maximum(np.var(fallback, axis=0), floor * floor))
    covariance = np.asarray(covariance, dtype=np.float64).reshape(dim, dim)
    covariance = 0.5 * (covariance + covariance.T)
    covariance *= scale * scale
    diagonal_floor = np.maximum(np.var(fallback, axis=0) * 0.02, floor * floor)
    covariance.flat[:: dim + 1] += diagonal_floor
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    eigenvalue = np.maximum(eigenvalue, floor * floor)
    return ((eigenvector * eigenvalue) @ eigenvector.T).astype(np.float32)


def fit_proposal(
    candidates: np.ndarray,
    energy: np.ndarray,
    *,
    elite_count: int,
    floor: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = len(candidates)
    flat = candidates[:, 1:].reshape(states, candidates.shape[1] - 1, -1)
    mean = np.empty((states, flat.shape[-1]), dtype=np.float32)
    covariance = np.empty((states, flat.shape[-1], flat.shape[-1]), dtype=np.float32)
    elite_index = np.empty((states, elite_count), dtype=np.int64)
    for state in range(states):
        order = np.argsort(energy[state, 1:], kind="stable")
        finite = order[np.isfinite(energy[state, 1:][order])]
        if not len(finite):
            finite = np.arange(flat.shape[1])
        chosen = np.resize(finite[:elite_count], elite_count)
        elite_index[state] = chosen + 1
        elite = flat[state, chosen]
        mean[state] = elite.mean(axis=0)
        covariance[state] = regularized_covariance(
            elite,
            fallback=flat[state],
            floor=floor,
            scale=scale,
        )
    return mean, covariance, elite_index


def sample_population(
    *,
    incumbent: np.ndarray,
    reference: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    population: int,
    rng: np.random.Generator,
    support_envelope: dict[str, Any],
    support_threshold: float,
    spacing_m: float,
    additional_support_mask: Callable[[np.ndarray], np.ndarray] | None = None,
    observer: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    states = len(incumbent)
    dim = mean.shape[-1]
    result = np.empty((states, population + 1, 4, 2), dtype=np.float32)
    result[:, 0] = incumbent
    reference_flat = reference.reshape(states, reference.shape[1], dim)
    lower = reference_flat.min(axis=1) - 0.35 * reference_flat.std(axis=1)
    upper = reference_flat.max(axis=1) + 0.35 * reference_flat.std(axis=1)
    accepted_total = 0
    attempted_total = 0
    fallback_total = 0
    additional_accepted_total = 0
    for state in range(states):
        accepted: list[np.ndarray] = []
        attempts = 0
        maximum_attempts = population * 24
        while len(accepted) < population and attempts < maximum_attempts:
            draw_count = min(64, maximum_attempts - attempts)
            draw = rng.multivariate_normal(
                mean[state].astype(np.float64),
                covariance[state].astype(np.float64),
                size=draw_count,
                check_valid="raise",
            )
            raw_draw = draw.copy() if observer is not None else None
            draw = np.clip(draw, lower[state], upper[state]).astype(np.float32)
            paths = draw.reshape(draw_count, 4, 2)
            pool = np.concatenate((incumbent[state : state + 1], paths), axis=0)[
                None
            ]
            actions = nomad_waypoints_to_local_actions(pool, dataset="recon")
            score, guard = score_spacing_envelope(
                support_envelope,
                actions,
                np.asarray([spacing_m], dtype=np.float32),
            )
            support_valid = score[0, 1:] <= support_threshold
            central_valid = guard[0, 1:] & support_valid
            additional = np.zeros_like(guard)
            if additional_support_mask is not None:
                additional = np.asarray(
                    additional_support_mask(actions), dtype=bool
                )
                if additional.shape != guard.shape:
                    raise RuntimeError(
                        "additional support mask shape mismatch: "
                        f"expected {guard.shape}, got {additional.shape}"
                    )
            additional_valid = additional[0, 1:]
            valid = central_valid | additional_valid
            accepted_before = len(accepted)
            accepted.extend(paths[valid])
            if observer is not None:
                vector = normalized_action_vector(
                    actions,
                    np.asarray([spacing_m], dtype=np.float32),
                )[0, 1:]
                coordinate_limit = np.asarray(
                    support_envelope["coordinate_abs_limit"], dtype=np.float64
                )
                coordinate_margin = coordinate_limit[None] - np.abs(vector)
                final_candidate_index = np.full(draw_count, -1, dtype=np.int64)
                used = np.flatnonzero(valid)[: max(0, population - accepted_before)]
                final_candidate_index[used] = (
                    1 + accepted_before + np.arange(len(used), dtype=np.int64)
                )
                rng_state_before_observer = repr(rng.bit_generator.state)
                observer(
                    {
                        "event": "draw_batch",
                        "state_index": int(state),
                        "draw_sequence": attempts + np.arange(
                            draw_count, dtype=np.int64
                        ),
                        "raw_draw_unclipped": raw_draw,
                        "draw_clipped": draw.copy(),
                        "clip_mask": (raw_draw != draw).reshape(draw_count, dim),
                        "actions_h4": actions[0, 1:].copy(),
                        "normalized_action_vector": vector.astype(np.float32),
                        "physical_coordinate_margin": coordinate_margin.astype(
                            np.float32
                        ),
                        "physical_coordinate_mask": coordinate_margin >= 0.0,
                        "physical_guard": guard[0, 1:].copy(),
                        "support_score": score[0, 1:].copy(),
                        "support_margin": (
                            support_threshold - score[0, 1:]
                        ).astype(np.float32),
                        "additional_support_mask": additional_valid.copy(),
                        "support_mask": support_valid.copy(),
                        "accepted_mask": valid.copy(),
                        "final_candidate_index": final_candidate_index,
                    }
                )
                if repr(rng.bit_generator.state) != rng_state_before_observer:
                    raise RuntimeError(
                        "proposal-tail observer changed planner RNG state"
                    )
            attempts += draw_count
            attempted_total += draw_count
            accepted_total += int(valid.sum())
            additional_accepted_total += int(
                (additional_valid & ~central_valid).sum()
            )
        accepted_draw_count = len(accepted)
        fallback_count = max(0, population - accepted_draw_count)
        if len(accepted) < population:
            missing = population - len(accepted)
            fallback_total += missing
            accepted.extend(reference[state, :missing])
        result[state, 1:] = np.asarray(accepted[:population], dtype=np.float32)
        if observer is not None:
            observer(
                {
                    "event": "state_complete",
                    "state_index": int(state),
                    "attempted_draws": int(attempts),
                    "accepted_draws_before_truncation": int(accepted_draw_count),
                    "fallback_candidates": int(fallback_count),
                    "dedup_applied": False,
                }
            )
    return result, {
        "attempted_samples": int(attempted_total),
        "support_accepted_samples": int(accepted_total),
        "additional_support_accepted_samples": int(additional_accepted_total),
        "support_acceptance": (
            float(accepted_total / attempted_total) if attempted_total else 0.0
        ),
        "fallback_samples": int(fallback_total),
        "online_causal": True,
        "future_route_windows_used": False,
    }


def build_source(
    *,
    candidates: np.ndarray,
    reference: np.ndarray,
    rows: list[dict[str, Any]],
    trajectory_root: Path,
    domain: str,
    output: Path,
) -> dict[str, np.ndarray]:
    legacy_rows = [
        {"traj_name": row["trajectory_id"], "curr_time": int(row["current_t"])}
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    motion = history_features(candidates, reference, legacy_rows, trajectory_root)
    raw = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action = np.concatenate((raw, raw - raw[:, :1]), axis=-1).astype(np.float32)
    if action.shape != (len(rows), candidates.shape[1], 242):
        raise RuntimeError(f"CEM action feature contract changed: {action.shape}")
    family_key = "route_id" if domain == "recon" else "route_group"
    source = {
        "candidates": candidates.astype(np.float32),
        "action_features": action,
        "record_key": np.asarray(
            [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
        ),
        "family": np.asarray(
            [f"{domain}::{row[family_key]}" for row in rows]
        ),
    }
    write_npz_atomic(output, **source)
    return source


def mark_cem_feature_artifact(path: Path, *, iteration: int) -> dict[str, Any]:
    arrays = load_npz(path)
    metadata = json.loads(str(arrays["metadata"].item()))
    metadata.update(
        {
            "candidate_pool": "incumbent_plus_continuous_cem_population16",
            "fixed_pool_only": False,
            "cem": True,
            "cem_iteration": int(iteration),
            "labels_accessed": False,
            "labels_in_output": False,
        }
    )
    arrays["metadata"] = np.asarray(canonical_json(metadata))
    write_npz_atomic(path, **arrays)
    metadata["output"] = str(path)
    metadata["output_sha256"] = file_sha256(path)
    write_json_atomic(path.with_suffix(".audit.json"), metadata)
    return metadata


def main() -> None:
    args = parse_args()
    if args.population != 16:
        raise ValueError("population must remain 16 to preserve the frozen K17 set contract")
    if args.iterations < 1 or not 0.0 < args.elite_fraction <= 0.5:
        raise ValueError("invalid CEM iteration/elite configuration")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"CEM output directory is not empty: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest)
    base = load_features(args.base_features, labels_expected=False)
    with np.load(args.candidate, allow_pickle=False) as payload:
        incumbent_native = payload["candidates_waypoint_units"][:, 0].astype(np.float32)
        reference_native = payload["nomad_x16_waypoint_units"].astype(np.float32)
        record_key = payload["record_key"].astype(str)
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if (
        len(rows) != 100
        or not np.array_equal(record_key, expected_record)
        or not np.array_equal(base["record_key"].astype(str), expected_record)
    ):
        raise RuntimeError("cem-cal manifest/candidate/feature identity mismatch")
    bridge = np.float32(1.0 if args.domain == "recon" else 0.38 / 0.25)
    spacing_m = 0.25 if args.domain == "recon" else 0.38
    incumbent = incumbent_native * bridge
    reference = reference_native * bridge

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("cem") is not False:
        raise RuntimeError("frozen base scorer must have been trained with CEM=false")
    envelopes = load_envelopes(Path(training["support"]))
    envelope = envelopes[args.domain]
    threshold = threshold_for_quantile(envelope, args.support_quantile)
    feature, support_score, physical_guard = build_universal_features(
        base,
        spacing_m=spacing_m,
        support_envelope=envelope,
    )
    valid = (
        physical_guard
        & (support_score <= threshold)
        & np.asarray(base["majority_support"], dtype=bool)
    )
    valid[:, 0] = True
    domain_id = np.full(len(rows), 0 if args.domain == "recon" else 1, dtype=np.int64)
    prediction = ensemble_predict(
        training["arms"]["D"]["checkpoints"]["full"],
        feature,
        domain_id,
        device=args.device,
    )
    policy, dual_temperature, harm_temperature = domain_policy(training, args.domain)
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    fixed_selected = selected_for_policy(
        prediction,
        p_dual,
        p_harm,
        valid,
        policy,
    )
    native_policy = json.loads(args.native_policy.read_text(encoding="utf-8"))
    native = native_policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(
        base["ate_by_seed"],
        base["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]),
        rtp_scale=float(native["rtp_scale"]),
        valid=base["majority_support"],
    )

    elite_count = max(2, int(round(args.population * args.elite_fraction)))
    base_candidates = np.concatenate((incumbent[:, None], reference), axis=1)
    mean, covariance, elite_index = fit_proposal(
        base_candidates,
        energy,
        elite_count=elite_count,
        floor=args.covariance_floor,
        scale=args.proposal_scale,
    )
    rng = np.random.default_rng(args.seed + (0 if args.domain == "recon" else 100000))
    iteration_audit: list[dict[str, Any]] = []
    final_candidates = base_candidates
    final_prediction = prediction
    final_valid = valid
    final_energy = energy

    for iteration in range(args.iterations):
        candidates, sampling_audit = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            population=args.population,
            rng=rng,
            support_envelope=envelope,
            support_threshold=threshold,
            spacing_m=spacing_m,
        )
        iteration_dir = args.output_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=False)
        population_path = iteration_dir / "population.npz"
        write_npz_atomic(
            population_path,
            candidates_feature_waypoint_units=candidates,
            candidates_native_waypoint_units=candidates / bridge,
            proposal_mean=mean,
            proposal_covariance=covariance,
            elite_index_from_previous=elite_index,
            record_key=expected_record,
        )
        source_path = iteration_dir / "source_label_free.npz"
        source = build_source(
            candidates=candidates,
            reference=reference,
            rows=rows,
            trajectory_root=args.trajectory_root,
            domain=args.domain,
            output=source_path,
        )
        feature_path = iteration_dir / "cem_features.npz"
        export_features(
            source,
            load_npz(args.latent_cache),
            pairs=checkpoint_pairs(REPO_ROOT / "checkpoints/project"),
            split=f"{args.domain}_cem_cal_dev100_cem_iter{iteration}",
            source=source_path,
            latent_path=args.latent_cache,
            output=feature_path,
            include_labels=False,
            device=torch.device(args.device),
            batch_size=args.wm_batch_size,
            candidate_chunk=17,
        )
        feature_metadata = mark_cem_feature_artifact(
            feature_path, iteration=iteration
        )
        iteration_data = load_features(feature_path, labels_expected=False)
        iteration_feature, iteration_support, iteration_guard = build_universal_features(
            iteration_data,
            spacing_m=spacing_m,
            support_envelope=envelope,
        )
        iteration_valid = (
            iteration_guard
            & (iteration_support <= threshold)
            & np.asarray(iteration_data["majority_support"], dtype=bool)
        )
        iteration_valid[:, 0] = True
        iteration_prediction = ensemble_predict(
            training["arms"]["D"]["checkpoints"]["full"],
            iteration_feature,
            domain_id,
            device=args.device,
        )
        iteration_energy, iteration_p_dual, iteration_p_harm = conservative_energy(
            iteration_prediction,
            policy=policy,
            dual_temperature=dual_temperature,
            harm_temperature=harm_temperature,
            valid=iteration_valid,
        )
        write_npz_atomic(
            iteration_dir / "frozen_scores.npz",
            energy=iteration_energy,
            mean=iteration_prediction["mean"],
            std=iteration_prediction["std"],
            dual_probability=iteration_p_dual,
            harm_probability=iteration_p_harm,
            rank_logit=iteration_prediction["rank_logit"],
            valid=iteration_valid,
        )
        mean, covariance, elite_index = fit_proposal(
            candidates,
            iteration_energy,
            elite_count=elite_count,
            floor=args.covariance_floor,
            scale=args.proposal_scale,
        )
        finite_challenger = np.isfinite(iteration_energy[:, 1:])
        iteration_audit.append(
            {
                "iteration": iteration,
                **sampling_audit,
                "population_sha256": file_sha256(population_path),
                "source_sha256": file_sha256(source_path),
                "feature_sha256": feature_metadata["output_sha256"],
                "frozen_scores_sha256": file_sha256(
                    iteration_dir / "frozen_scores.npz"
                ),
                "scorer_valid_candidate_acceptance": float(
                    iteration_valid[:, 1:].mean()
                ),
                "windows_with_any_valid_challenger": float(
                    iteration_valid[:, 1:].any(axis=1).mean()
                ),
                "finite_energy_candidates": int(finite_challenger.sum()),
                "best_energy_mean": float(
                    np.min(iteration_energy[:, 1:], axis=1).mean()
                ),
            }
        )
        final_candidates = candidates
        final_prediction = iteration_prediction
        final_valid = iteration_valid
        final_energy = iteration_energy

    final_p_dual = sigmoid(final_prediction["dual_logit"] / dual_temperature)
    final_p_harm = sigmoid(final_prediction["harm_logit"] / harm_temperature)
    cem_selected = selected_for_policy(
        final_prediction,
        final_p_dual,
        final_p_harm,
        final_valid,
        policy,
    )
    selected_path = args.output_dir / "selected_actions.npz"
    write_npz_atomic(
        selected_path,
        record_key=expected_record,
        incumbent_feature_waypoint_units=incumbent,
        base_k17_feature_waypoint_units=base["candidates"].astype(np.float32),
        final_cem_feature_waypoint_units=final_candidates,
        native_scorer_v2_selected_index=native_selected,
        universal_d_cem_false_selected_index=fixed_selected,
        universal_d_cem_true_selected_index=cem_selected,
        final_cem_energy=final_energy,
        final_cem_valid=final_valid,
    )
    freeze_path = args.output_dir / "prelabel_freeze.json"
    freeze = {
        "schema": "universal_raw_d_cem_development_v1",
        "status": "cem_true_actions_and_selections_frozen_before_outcomes",
        "evidence_role": "development_only_cem_cal",
        "domain": args.domain,
        "states": len(rows),
        "cem": True,
        "formal_fixed_pool_qualification": False,
        "future_pose_outcomes_opened": False,
        "outcomes_used_by_cem": False,
        "scorer_generator_separation": (
            "Gaussian CEM proposes actions; frozen LTWM plus universal-D scorer "
            "and uncertainty rank them. No outcome enters a window optimization."
        ),
        "online_causal": True,
        "future_route_window_ranking": False,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "base_features": str(args.base_features),
        "base_features_sha256": file_sha256(args.base_features),
        "latent_cache": str(args.latent_cache),
        "latent_cache_sha256": file_sha256(args.latent_cache),
        "training_result": str(args.training_result),
        "training_result_sha256": file_sha256(args.training_result),
        "native_policy": str(args.native_policy),
        "native_policy_sha256": file_sha256(args.native_policy),
        "support_checkpoint": str(training["support"]),
        "support_checkpoint_sha256": file_sha256(Path(training["support"])),
        "support_quantile": args.support_quantile,
        "support_threshold": threshold,
        "config": {
            "iterations": args.iterations,
            "population_challengers": args.population,
            "set_cardinality_with_incumbent": args.population + 1,
            "elite_fraction": args.elite_fraction,
            "elite_count": elite_count,
            "proposal_scale": args.proposal_scale,
            "covariance_floor_waypoint_units": args.covariance_floor,
            "proposal_covariance": "per-window full 8x8, eigenvalue floored",
            "seed": args.seed,
            "spacing_m": spacing_m,
        },
        "base_selection": {
            "native_scorer_v2_promotions": int((native_selected != 0).sum()),
            "universal_d_cem_false_promotions": int((fixed_selected != 0).sum()),
        },
        "cem_true_promotions": int((cem_selected != 0).sum()),
        "cem_true_coverage": float((cem_selected != 0).mean()),
        "iteration_audit": iteration_audit,
        "selected_actions": str(selected_path),
        "selected_actions_sha256": file_sha256(selected_path),
        "final_candidates_sha256": array_sha256(final_candidates),
        "labels_opened": False,
    }
    write_json_atomic(freeze_path, freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
