#!/usr/bin/env python3
"""Run true iterative CEM with a frozen candidate-level safety elite gate.

The optimizer is outcome-free.  Every iteration samples a new K16 proposal,
runs the existing LTWM/DINO feature export and frozen universal-D scorer, then
updates only from candidates that pass the complete frozen promotion gate.  A
window with fewer safe candidates than the elite count keeps its previous
proposal unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.safe_promotion_v1 import greedy_fixed_pool_selection  # noqa: E402
from scoring.spacing_support_v1 import (  # noqa: E402
    threshold_for_quantile,
)
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    checkpoint_pairs,
    export_features,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.evaluation.freeze_recon_fresh_outer_universal_v2 import (  # noqa: E402,F401
    arm_policy,
)
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    NATIVE_POLICY,
    TRAINING_RESULT,
    array_sha256,
    build_source,
    conservative_energy,
    domain_policy,
    mark_cem_feature_artifact,
    read_rows,
    regularized_covariance,
    sample_population,
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
    sigmoid,
)


AMENDMENT = (
    ROOT
    / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/"
    "amendment_v54.json"
)
TREND_RESULT = (
    ROOT
    / "outputs/cem_development_v1/recon_nomad_goal_condition_v53/"
    "amendment_v54_result.json"
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
    parser.add_argument("--route-amendment", type=Path, default=AMENDMENT)
    parser.add_argument("--v53-trend-result", type=Path, default=TREND_RESULT)
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


def candidate_safety_mask(
    prediction: dict[str, np.ndarray],
    p_dual: np.ndarray,
    p_harm: np.ndarray,
    valid: np.ndarray,
    policy: dict[str, float],
) -> np.ndarray:
    """Apply the frozen deployment risk policy to every candidate."""

    z = float(policy["z"])
    ucb = prediction["mean"] + z * prediction["std"]
    safe = (
        np.asarray(valid, dtype=bool)
        & (ucb[..., 0] < -float(policy["margin_ate"]))
        & (ucb[..., 3] < -float(policy["margin_objective"]))
        & (p_dual >= float(policy["dual_threshold"]))
        & (p_harm <= float(policy["harm_threshold"]))
    )
    safe[:, 0] = False
    return safe


def select_lowest_energy_safe(
    energy: np.ndarray,
    safe: np.ndarray,
) -> np.ndarray:
    masked = np.where(safe, energy, np.inf)
    selected = masked.argmin(axis=1).astype(np.int64)
    return np.where(safe.any(axis=1), selected, 0).astype(np.int64)


def initial_proposal(
    reference: np.ndarray,
    *,
    floor: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    flat = reference.reshape(len(reference), reference.shape[1], -1)
    mean = flat.mean(axis=1).astype(np.float32)
    covariance = np.stack(
        [
            regularized_covariance(
                row,
                fallback=row,
                floor=floor,
                scale=scale,
            )
            for row in flat
        ]
    )
    return mean, covariance


def safe_elite_update(
    candidates: np.ndarray,
    energy: np.ndarray,
    safe: np.ndarray,
    *,
    previous_mean: np.ndarray,
    previous_covariance: np.ndarray,
    elite_count: int,
    floor: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Update q only where at least ``elite_count`` safe candidates exist."""

    flat = candidates.reshape(len(candidates), candidates.shape[1], -1)
    mean = previous_mean.copy()
    covariance = previous_covariance.copy()
    elite_index = np.full((len(candidates), elite_count), -1, dtype=np.int64)
    updated = np.zeros(len(candidates), dtype=bool)
    safe_count = (safe & np.isfinite(energy)).sum(axis=1).astype(np.int64)
    for state in range(len(candidates)):
        if safe_count[state] < elite_count:
            continue
        order = np.argsort(energy[state], kind="stable")
        chosen = order[(safe[state] & np.isfinite(energy[state]))[order]][
            :elite_count
        ]
        elite_index[state] = chosen
        elite = flat[state, chosen]
        mean[state] = elite.mean(axis=0)
        covariance[state] = regularized_covariance(
            elite,
            fallback=flat[state, 1:],
            floor=floor,
            scale=scale,
        )
        updated[state] = True
    return mean, covariance, elite_index, updated, safe_count


def proposal_summary(mean: np.ndarray, covariance: np.ndarray) -> dict[str, float]:
    dimension = covariance.shape[-1]
    _, logdet = np.linalg.slogdet(covariance.astype(np.float64))
    entropy = 0.5 * (
        dimension * (1.0 + np.log(2.0 * np.pi)) + logdet
    )
    diagonal = np.diagonal(covariance, axis1=1, axis2=2)
    return {
        "mean_l2_mean": float(np.linalg.norm(mean, axis=1).mean()),
        "mean_l2_std": float(np.linalg.norm(mean, axis=1).std()),
        "variance_mean": float(diagonal.mean()),
        "variance_min": float(diagonal.min()),
        "variance_max": float(diagonal.max()),
        "gaussian_entropy_mean": float(entropy.mean()),
        "gaussian_entropy_min": float(entropy.min()),
        "gaussian_entropy_max": float(entropy.max()),
    }


def main() -> None:
    args = parse_args()
    if args.population != 16 or args.iterations != 3:
        raise ValueError("v55 budget is frozen at 3 iterations and population 16")
    if abs(args.elite_fraction - 0.25) > 1.0e-12:
        raise ValueError("v55 elite fraction is frozen at 0.25")
    if abs(args.proposal_scale - 0.70) > 1.0e-12:
        raise ValueError("v55 proposal scale is frozen at 0.70")
    if abs(args.support_quantile - 0.999) > 1.0e-12:
        raise ValueError("v55 support quantile is frozen at 0.999")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    amendment = json.loads(args.route_amendment.read_text(encoding="utf-8"))
    trend = json.loads(args.v53_trend_result.read_text(encoding="utf-8"))
    if trend.get("v53_terminal") is not True or trend.get("reserve_accessed"):
        raise RuntimeError("v53 amendment result contract is invalid")
    state_gate_enabled = bool(trend.get("useful_trend_variants"))
    if state_gate_enabled:
        raise RuntimeError(
            "v55 has no frozen causal state-gate artifact; a diagnostic top-40 "
            "gate may not be used inside online CEM"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.manifest)
    base = load_features(args.base_features, labels_expected=False)
    with np.load(args.candidate, allow_pickle=False) as payload:
        incumbent_native = payload["candidates_waypoint_units"][:, 0].astype(
            np.float32
        )
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
        raise RuntimeError("manifest/candidate/base-feature identity mismatch")

    bridge = np.float32(1.0 if args.domain == "recon" else 0.38 / 0.25)
    spacing_m = 0.25 if args.domain == "recon" else 0.38
    incumbent = incumbent_native * bridge
    reference = reference_native * bridge
    base_candidates = np.concatenate((incumbent[:, None], reference), axis=1)

    training = json.loads(args.training_result.read_text(encoding="utf-8"))
    if training.get("cem") is not False:
        raise RuntimeError("base scorer must be frozen from CEM=false training")
    envelopes = load_envelopes(Path(training["support"]))
    envelope = envelopes[args.domain]
    threshold = threshold_for_quantile(envelope, args.support_quantile)
    base_feature, support_score, physical_guard = build_universal_features(
        base, spacing_m=spacing_m, support_envelope=envelope
    )
    valid = (
        physical_guard
        & (support_score <= threshold)
        & np.asarray(base["majority_support"], dtype=bool)
    )
    valid[:, 0] = True
    domain_id = np.full(
        len(rows), 0 if args.domain == "recon" else 1, dtype=np.int64
    )
    prediction = ensemble_predict(
        training["arms"]["D"]["checkpoints"]["full"],
        base_feature,
        domain_id,
        device=args.device,
    )
    policy, dual_temperature, harm_temperature = domain_policy(
        training, args.domain
    )
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    fixed_selected = select_lowest_energy_safe(energy, safe)

    native_policy = json.loads(args.native_policy.read_text(encoding="utf-8"))
    native = native_policy["arms"]["dinov3_scorer_v2_native"]
    native_selected = greedy_fixed_pool_selection(
        base["ate_by_seed"],
        base["rtp_by_seed"],
        ate_scale=float(native["ate_scale"]),
        rtp_scale=float(native["rtp_scale"]),
        valid=base["majority_support"],
    )

    elite_count = int(round(args.population * args.elite_fraction))
    mean, covariance = initial_proposal(
        reference, floor=args.covariance_floor, scale=args.proposal_scale
    )
    mean, covariance, elite_index, updated, safe_count = safe_elite_update(
        base_candidates,
        energy,
        safe,
        previous_mean=mean,
        previous_covariance=covariance,
        elite_count=elite_count,
        floor=args.covariance_floor,
        scale=args.proposal_scale,
    )
    base_initialization = {
        "valid_candidates": int(valid[:, 1:].sum()),
        "safe_candidates": int(safe[:, 1:].sum()),
        "elite_candidates": int(updated.sum() * elite_count),
        "updated_windows": int(updated.sum()),
        "safe_shortfall_fallback_windows": int((~updated).sum()),
        "safe_count_min": int(safe_count.min()),
        "safe_count_mean": float(safe_count.mean()),
        "proposal": proposal_summary(mean, covariance),
    }

    rng = np.random.default_rng(
        args.seed + (0 if args.domain == "recon" else 100000)
    )
    iteration_audit: list[dict[str, Any]] = []
    final_candidates = base_candidates
    final_prediction = prediction
    final_valid = valid
    final_safe = safe
    final_energy = energy

    for iteration in range(args.iterations):
        proposal_before = proposal_summary(mean, covariance)
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
            pairs=checkpoint_pairs(ROOT / "checkpoints/project"),
            split=f"{args.domain}_cem_safe_elite_v55_iter{iteration}",
            source=source_path,
            latent_path=args.latent_cache,
            output=feature_path,
            include_labels=False,
            device=torch.device(args.device),
            batch_size=args.wm_batch_size,
            candidate_chunk=17,
        )
        feature_metadata = mark_cem_feature_artifact(feature_path, iteration=iteration)
        data = load_features(feature_path, labels_expected=False)
        features, support, guard = build_universal_features(
            data, spacing_m=spacing_m, support_envelope=envelope
        )
        iteration_valid = (
            guard
            & (support <= threshold)
            & np.asarray(data["majority_support"], dtype=bool)
        )
        iteration_valid[:, 0] = True
        iteration_prediction = ensemble_predict(
            training["arms"]["D"]["checkpoints"]["full"],
            features,
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
        iteration_safe = candidate_safety_mask(
            iteration_prediction,
            iteration_p_dual,
            iteration_p_harm,
            iteration_valid,
            policy,
        )
        next_mean, next_covariance, next_elite, next_updated, next_safe_count = (
            safe_elite_update(
                candidates,
                iteration_energy,
                iteration_safe,
                previous_mean=mean,
                previous_covariance=covariance,
                elite_count=elite_count,
                floor=args.covariance_floor,
                scale=args.proposal_scale,
            )
        )
        score_path = iteration_dir / "frozen_scores.npz"
        write_npz_atomic(
            score_path,
            energy=iteration_energy,
            mean=iteration_prediction["mean"],
            std=iteration_prediction["std"],
            dual_probability=iteration_p_dual,
            harm_probability=iteration_p_harm,
            rank_logit=iteration_prediction["rank_logit"],
            valid=iteration_valid,
            safe=iteration_safe,
            elite_index_for_update=next_elite,
            proposal_updated=next_updated,
            safe_count=next_safe_count,
        )
        elite_energy = np.asarray(
            [
                iteration_energy[state, index]
                for state, index in enumerate(next_elite)
                if next_updated[state]
            ],
            dtype=np.float32,
        ).reshape(-1)
        iteration_audit.append(
            {
                "iteration": iteration,
                **sampling_audit,
                "population_sha256": file_sha256(population_path),
                "source_sha256": file_sha256(source_path),
                "feature_sha256": feature_metadata["output_sha256"],
                "frozen_scores_sha256": file_sha256(score_path),
                "valid_candidates": int(iteration_valid[:, 1:].sum()),
                "safe_candidates": int(iteration_safe[:, 1:].sum()),
                "elite_candidates": int(next_updated.sum() * elite_count),
                "ood_or_invalid_candidates": int(
                    iteration_valid[:, 1:].size
                    - iteration_valid[:, 1:].sum()
                ),
                "windows_with_any_valid_challenger": int(
                    iteration_valid[:, 1:].any(axis=1).sum()
                ),
                "windows_with_any_safe_challenger": int(
                    iteration_safe[:, 1:].any(axis=1).sum()
                ),
                "proposal_updated_windows": int(next_updated.sum()),
                "safe_shortfall_fallback_windows": int((~next_updated).sum()),
                "safe_count_min": int(next_safe_count.min()),
                "safe_count_mean": float(next_safe_count.mean()),
                "elite_scorer_energy_mean": (
                    float(elite_energy.mean()) if elite_energy.size else None
                ),
                "elite_scorer_utility_mean": (
                    float((-elite_energy).mean()) if elite_energy.size else None
                ),
                "proposal_before": proposal_before,
                "proposal_after": proposal_summary(next_mean, next_covariance),
                "true_utility_available_before_outcomes": False,
            }
        )
        mean, covariance, elite_index, updated, safe_count = (
            next_mean,
            next_covariance,
            next_elite,
            next_updated,
            next_safe_count,
        )
        final_candidates = candidates
        final_prediction = iteration_prediction
        final_valid = iteration_valid
        final_safe = iteration_safe
        final_energy = iteration_energy

    cem_selected = select_lowest_energy_safe(final_energy, final_safe)
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
        base_safe= safe,
        final_cem_energy=final_energy,
        final_cem_valid=final_valid,
        final_cem_safe=final_safe,
    )
    freeze_path = args.output_dir / "prelabel_freeze.json"
    freeze = {
        "schema": "universal_raw_d_cem_safe_elite_v55",
        "status": "cem_true_safe_elite_actions_and_selections_frozen_before_outcomes",
        "evidence_role": "development_only_cem_cal",
        "domain": args.domain,
        "states": len(rows),
        "cem": True,
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "outcomes_used_by_cem": False,
        "online_causal": True,
        "future_route_window_ranking": False,
        "candidate_level_safety_elite": True,
        "insufficient_safe_elite_rule": (
            "retain previous proposal unchanged when safe_count < elite_count"
        ),
        "state_gate_used": False,
        "state_gate_reason": (
            "v53 had no useful trend; CEM proceeds under the supervisory amendment "
            "using candidate-level safety only"
        ),
        "route_amendment": str(args.route_amendment),
        "route_amendment_sha256": file_sha256(args.route_amendment),
        "v53_trend_result": str(args.v53_trend_result),
        "v53_trend_result_sha256": file_sha256(args.v53_trend_result),
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
        "policy": policy,
        "config": {
            "iterations": args.iterations,
            "population_challengers": args.population,
            "set_cardinality_with_incumbent": args.population + 1,
            "elite_fraction": args.elite_fraction,
            "elite_count": elite_count,
            "proposal_scale": args.proposal_scale,
            "covariance_floor_waypoint_units": args.covariance_floor,
            "seed": args.seed,
            "spacing_m": spacing_m,
        },
        "base_initialization": base_initialization,
        "base_selection": {
            "native_scorer_v2_promotions": int((native_selected != 0).sum()),
            "safe_cem_false_promotions": int((fixed_selected != 0).sum()),
        },
        "cem_true_promotions": int((cem_selected != 0).sum()),
        "cem_true_coverage": float((cem_selected != 0).mean()),
        "iteration_audit": iteration_audit,
        "selected_actions": str(selected_path),
        "selected_actions_sha256": file_sha256(selected_path),
        "final_candidates_sha256": array_sha256(final_candidates),
        "reserve_accessed": False,
    }
    write_json_atomic(freeze_path, freeze)
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
