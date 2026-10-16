#!/usr/bin/env python3
"""Continue frozen v70 CEM from cached depth 3 to depths 4 and 5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import build_universal_features  # noqa: E402
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    checkpoint_pairs,
    export_features,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
)
from tools.planning.run_cem_development_v1 import (  # noqa: E402
    array_sha256,
    build_source,
    conservative_energy,
    mark_cem_feature_artifact,
    read_rows,
    sample_population,
)
from tools.planning.run_cem_safe_elite_v55 import (  # noqa: E402
    candidate_safety_mask,
    proposal_summary,
    safe_elite_update,
    select_lowest_energy_safe,
)
from tools.planning import run_dualdomain_cem_tail_v70_archive_replay as v70  # noqa: E402
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import load_envelopes  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import load_features  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


EXPECTED_STATUS = "candidate_level_safe_elite_actions_frozen_before_outcomes"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--base-replay-dir", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wm-batch-size", type=int, default=32)
    return parser.parse_args()


def assert_no_reserve(path: Path, label: str) -> None:
    lowered = str(path.resolve()).lower()
    if "final_confirmation_reserve" in lowered or "reserve_confirmation" in lowered:
        raise RuntimeError(f"spent reserve is forbidden in depth pilot ({label}): {path}")


def archive_counts(source_round: np.ndarray, maximum_round: int) -> dict[str, int]:
    return {
        "none": int((source_round == -2).sum()),
        "fixed_pool": int((source_round == -1).sum()),
        **{
            f"iteration_{iteration:02d}": int((source_round == iteration).sum())
            for iteration in range(maximum_round + 1)
        },
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for path, label in (
        (args.preregistration, "preregistration"),
        (args.base_replay_dir, "base replay"),
        (args.trajectory_root, "trajectory root"),
        (args.output_dir, "output"),
    ):
        assert_no_reserve(path, label)

    prereg = read_json(args.preregistration)
    if prereg.get("status") != "frozen_before_depth4_or_depth5_candidate_evaluation":
        raise RuntimeError("CEM-depth pilot is not preregistered")
    runner_binding = prereg["source_bindings"]["continuation_runner"]
    if runner_binding["sha256"] != file_sha256(Path(__file__)):
        raise RuntimeError("continuation runner changed after preregistration")

    freeze_path = args.base_replay_dir / "prelabel_freeze.json"
    freeze = read_json(freeze_path)
    cohort = str(freeze.get("cohort"))
    if cohort not in ("recon_bank_g", "scand_fresh_outer"):
        raise RuntimeError(f"cohort is outside the preregistered pilot: {cohort}")
    binding = prereg["cohorts"][cohort]
    if file_sha256(freeze_path) != binding["depth3_freeze_sha256"]:
        raise RuntimeError("depth-3 freeze changed after preregistration")
    if freeze.get("schema") != "dualdomain_cem_tail_v70_archive_prelabel_replay":
        raise RuntimeError("depth-3 replay is not the v70 archive runner")
    if freeze.get("status") != EXPECTED_STATUS:
        raise RuntimeError("depth-3 replay is not frozen before outcomes")
    if freeze.get("reserve_accessed") or freeze.get("labels_opened") or freeze.get("outcomes_used_by_cem"):
        raise RuntimeError("depth-3 replay reports forbidden outcome/reserve use")
    if freeze.get("model", {}).get("model_version") != "v70_broad_hashfold_archive":
        raise RuntimeError("depth-3 replay is not the frozen v70 model")
    config = freeze["config"]
    expected_config = prereg["frozen_cem_contract"]
    for key, value in expected_config.items():
        if key == "cem_iterations":
            continue
        observed_key = {
            "population": "population_challengers",
            "proposal_scale": "proposal_scale",
            "support_quantile": "support_quantile",
            "random_seed": "seed",
        }.get(key, key)
        if observed_key in config and config[observed_key] != value:
            raise RuntimeError(f"frozen CEM contract changed for {key}")
    if config.get("iterations") != 3 or config.get("population_challengers") != 16:
        raise RuntimeError("base replay is not the frozen three-round population-16 run")
    if config.get("elite_count") != 4 or config.get("proposal_scale") != 0.7:
        raise RuntimeError("base replay elite/proposal contract changed")
    if not freeze.get("cross_round_safe_archive") or not freeze.get("candidate_level_safety_elite"):
        raise RuntimeError("safe archive or candidate-level safety is disabled")

    for key in ("manifest", "candidate_pool", "base_features", "latent_cache", "selected_actions"):
        assert_no_reserve(Path(freeze[key]), key)
    for key, sha_key in (
        ("manifest", "manifest_sha256"),
        ("candidate_pool", "candidate_pool_sha256"),
        ("base_features", "base_features_sha256"),
        ("latent_cache", "latent_cache_sha256"),
        ("selected_actions", "selected_actions_sha256"),
    ):
        if file_sha256(Path(freeze[key])) != freeze[sha_key]:
            raise RuntimeError(f"cached depth-3 artifact changed: {key}")

    domain = str(freeze["domain"])
    rows = read_rows(Path(freeze["manifest"]))
    candidate = load_npz(Path(freeze["candidate_pool"]))
    base_feature = load_features(Path(freeze["base_features"]), labels_expected=False)
    depth3 = load_npz(Path(freeze["selected_actions"]))
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not np.array_equal(depth3["record_key"].astype(str), expected_record):
        raise RuntimeError("depth-3 selected-action identity mismatch")
    if not np.array_equal(candidate["record_key"].astype(str), expected_record):
        raise RuntimeError("candidate identity mismatch")
    if not np.array_equal(base_feature["record_key"].astype(str), expected_record):
        raise RuntimeError("base-feature identity mismatch")

    incumbent = depth3["incumbent_feature_waypoint_units"].astype(np.float32)
    reference = depth3["base_k17_feature_waypoint_units"][:, 1:].astype(np.float32)
    bridge = np.float32(1.0 if domain == "recon" else 0.38 / 0.25)
    spacing_m = float(config["spacing_m"])
    base_result_path = Path(prereg["model_bindings"]["original_arm_d_result"]["path"])
    v70_result_path = Path(prereg["model_bindings"]["v70_result"]["path"])
    checkpoints, policy, dual_temperature, harm_temperature, model_audit = (
        v70.frozen_model_contract(
            role="proposal_aware_v66",
            domain=domain,
            base_result_path=base_result_path,
            v66_result_path=v70_result_path,
        )
    )
    if policy != freeze["policy"]:
        raise RuntimeError("model policy differs from cached depth-3 replay")
    if dual_temperature != freeze["dual_temperature"] or harm_temperature != freeze["harm_temperature"]:
        raise RuntimeError("model temperatures differ from cached depth-3 replay")
    base_result = read_json(base_result_path)
    support_path = Path(base_result["support"])
    envelope = load_envelopes(support_path)[domain]
    support_threshold = threshold_for_quantile(envelope, config["support_quantile"])
    if float(support_threshold) != float(config["support_threshold"]):
        raise RuntimeError("support threshold differs from cached depth-3 replay")
    domain_id = np.full(len(rows), 0 if domain == "recon" else 1, dtype=np.int64)

    population2 = load_npz(args.base_replay_dir / "iteration_02/population.npz")
    score2 = load_npz(args.base_replay_dir / "iteration_02/frozen_scores.npz")
    mean, covariance, elite_index, updated, safe_count = safe_elite_update(
        population2["candidates_feature_waypoint_units"].astype(np.float32),
        score2["energy"].astype(np.float32),
        score2["safe"].astype(bool),
        previous_mean=population2["proposal_mean"].astype(np.float32),
        previous_covariance=population2["proposal_covariance"].astype(np.float32),
        elite_count=4,
        floor=float(config["covariance_floor_waypoint_units"]),
        scale=float(config["proposal_scale"]),
    )
    if not np.array_equal(elite_index, score2["elite_index_for_update"].astype(np.int64)):
        raise RuntimeError("cached q3 elite reconstruction failed")
    if not np.array_equal(updated, score2["proposal_updated"].astype(bool)):
        raise RuntimeError("cached q3 update reconstruction failed")
    if not np.array_equal(safe_count, score2["safe_count"].astype(np.int64)):
        raise RuntimeError("cached q3 safe-count reconstruction failed")

    rng = np.random.default_rng(
        int(config["seed"]) + (0 if domain == "recon" else 100000)
    )
    cached_rng_verification = []
    for iteration in range(3):
        cached_population = load_npz(
            args.base_replay_dir / f"iteration_{iteration:02d}/population.npz"
        )
        regenerated, sampling_audit = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=cached_population["proposal_mean"].astype(np.float32),
            covariance=cached_population["proposal_covariance"].astype(np.float32),
            population=16,
            rng=rng,
            support_envelope=envelope,
            support_threshold=float(support_threshold),
            spacing_m=spacing_m,
        )
        if not np.array_equal(
            regenerated,
            cached_population["candidates_feature_waypoint_units"].astype(np.float32),
        ):
            raise RuntimeError(f"RNG continuation is not exact at cached iteration {iteration}")
        cached_rng_verification.append(
            {
                "iteration": iteration,
                "population_sha256": file_sha256(
                    args.base_replay_dir / f"iteration_{iteration:02d}/population.npz"
                ),
                "candidate_array_sha256": array_sha256(regenerated),
                "sampling_audit": sampling_audit,
                "ltwm_recomputed": False,
                "scorer_recomputed": False,
            }
        )

    archive_energy = depth3["archive_energy"].astype(np.float32).copy()
    archive_candidate = depth3["final_cem_feature_waypoint_units"][:, 1].astype(np.float32).copy()
    archive_source_round = depth3["archive_source_round"].astype(np.int64).copy()
    archive_source_candidate = depth3["archive_source_candidate"].astype(np.int64).copy()
    latents = load_npz(Path(freeze["latent_cache"]))
    pairs = checkpoint_pairs(ROOT / "checkpoints/project")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    added_iteration_audit = []
    depth_artifacts: dict[str, Any] = {}

    for iteration in (3, 4):
        round_started = time.perf_counter()
        proposal_before = proposal_summary(mean, covariance)
        sampling_started = time.perf_counter()
        candidates, sampling_audit = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            population=16,
            rng=rng,
            support_envelope=envelope,
            support_threshold=float(support_threshold),
            spacing_m=spacing_m,
        )
        sampling_seconds = time.perf_counter() - sampling_started
        directory = args.output_dir / f"iteration_{iteration:02d}"
        directory.mkdir(parents=True, exist_ok=False)
        population_path = directory / "population.npz"
        write_npz_atomic(
            population_path,
            candidates_feature_waypoint_units=candidates,
            candidates_native_waypoint_units=candidates / bridge,
            proposal_mean=mean,
            proposal_covariance=covariance,
            elite_index_from_previous=elite_index,
            record_key=expected_record,
        )

        source_started = time.perf_counter()
        source_path = directory / "source_label_free.npz"
        source = build_source(
            candidates=candidates,
            reference=reference,
            rows=rows,
            trajectory_root=args.trajectory_root,
            domain=domain,
            output=source_path,
        )
        source_seconds = time.perf_counter() - source_started
        feature_started = time.perf_counter()
        feature_path = directory / "cem_features.npz"
        export_features(
            source,
            latents,
            pairs=pairs,
            split=f"v70_cem_depth_pilot_{domain}_{cohort}_iter{iteration}",
            source=source_path,
            latent_path=Path(freeze["latent_cache"]),
            output=feature_path,
            include_labels=False,
            device=torch.device(args.device),
            batch_size=args.wm_batch_size,
            candidate_chunk=17,
        )
        feature_metadata = mark_cem_feature_artifact(feature_path, iteration=iteration)
        feature_seconds = time.perf_counter() - feature_started

        scoring_started = time.perf_counter()
        data = load_features(feature_path, labels_expected=False)
        local_feature, local_support, local_guard = build_universal_features(
            data, spacing_m=spacing_m, support_envelope=envelope
        )
        local_valid = local_guard & (local_support <= support_threshold)
        local_valid[:, 0] = True
        prediction = v70.compatible_ensemble_predict(
            checkpoints, local_feature, domain_id, device=args.device
        )
        energy, p_dual, p_harm = conservative_energy(
            prediction,
            policy=policy,
            dual_temperature=dual_temperature,
            harm_temperature=harm_temperature,
            valid=local_valid,
        )
        safe = candidate_safety_mask(prediction, p_dual, p_harm, local_valid, policy)
        next_mean, next_covariance, next_elite, next_updated, next_safe_count = (
            safe_elite_update(
                candidates,
                energy,
                safe,
                previous_mean=mean,
                previous_covariance=covariance,
                elite_count=4,
                floor=float(config["covariance_floor_waypoint_units"]),
                scale=float(config["proposal_scale"]),
            )
        )
        scorer_selected = select_lowest_energy_safe(energy, safe)
        row = np.arange(len(rows))
        archive_update = (
            (scorer_selected != 0)
            & (energy[row, scorer_selected] < archive_energy)
        )
        local = np.flatnonzero(archive_update)
        archive_energy[local] = energy[local, scorer_selected[local]]
        archive_candidate[local] = candidates[local, scorer_selected[local]]
        archive_source_round[local] = iteration
        archive_source_candidate[local] = scorer_selected[local]
        scoring_seconds = time.perf_counter() - scoring_started

        score_path = directory / "frozen_scores.npz"
        write_npz_atomic(
            score_path,
            energy=energy,
            mean=prediction["mean"],
            std=prediction["std"],
            dual_probability=p_dual,
            harm_probability=p_harm,
            rank_logit=prediction["rank_logit"],
            valid=local_valid,
            safe=safe,
            elite_index_for_update=next_elite,
            proposal_updated=next_updated,
            safe_count=next_safe_count,
            archive_updated=archive_update,
            record_key=expected_record,
        )
        depth = iteration + 1
        selected_candidates = candidates.copy()
        archived = np.isfinite(archive_energy)
        selected_candidates[:, 1] = archive_candidate
        selected_path = args.output_dir / f"selected_depth_{depth}.npz"
        write_npz_atomic(
            selected_path,
            record_key=expected_record,
            base_k17_feature_waypoint_units=depth3["base_k17_feature_waypoint_units"],
            fixed_pool_selected_index=depth3["fixed_pool_selected_index"],
            final_cem_feature_waypoint_units=selected_candidates,
            final_cem_selected_index=archived.astype(np.int64),
            archive_energy=archive_energy,
            archive_source_round=archive_source_round,
            archive_source_candidate=archive_source_candidate,
            final_round_energy=energy,
            final_round_valid=local_valid,
            final_round_safe=safe,
        )
        total_seconds = time.perf_counter() - round_started
        audit = {
            "iteration": iteration,
            "depth": depth,
            **sampling_audit,
            "population_sha256": file_sha256(population_path),
            "source_sha256": file_sha256(source_path),
            "feature_sha256": feature_metadata["output_sha256"],
            "frozen_scores_sha256": file_sha256(score_path),
            "selected_actions_sha256": file_sha256(selected_path),
            "valid_candidates": int(local_valid[:, 1:].sum()),
            "safe_candidates": int(safe[:, 1:].sum()),
            "elite_candidates": int(next_updated.sum() * 4),
            "proposal_updated_windows": int(next_updated.sum()),
            "safe_shortfall_fallback_windows": int((~next_updated).sum()),
            "safe_count_min": int(next_safe_count.min()),
            "safe_count_mean": float(next_safe_count.mean()),
            "archive_updates": int(archive_update.sum()),
            "archive_source_counts": archive_counts(archive_source_round, iteration),
            "proposal_before": proposal_before,
            "proposal_after": proposal_summary(next_mean, next_covariance),
            "timing_seconds": {
                "sampling": sampling_seconds,
                "source": source_seconds,
                "ltwm_feature_export": feature_seconds,
                "scoring_and_update": scoring_seconds,
                "end_to_end": total_seconds,
                "end_to_end_per_state": total_seconds / len(rows),
            },
            "extra_new_challenger_evaluations": int(len(rows) * 16),
            "set_candidate_evaluations": int(len(rows) * 17),
            "ltwm_seed_candidate_rollouts": int(len(rows) * 17 * 3),
            "cached_rounds_recomputed": 0,
            "outcomes_available": False,
        }
        added_iteration_audit.append(audit)
        depth_artifacts[str(depth)] = {
            "selected_actions": str(selected_path.resolve()),
            "selected_actions_sha256": file_sha256(selected_path),
            "archive_source_counts": audit["archive_source_counts"],
        }
        mean, covariance, elite_index = next_mean, next_covariance, next_elite

    continuation = {
        "schema": "v70_cem_depth_continuation_preoutcome_freeze",
        "status": "depth4_and_depth5_actions_frozen_before_spent_outcome_diagnostics",
        "domain": domain,
        "cohort": cohort,
        "states": len(rows),
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": file_sha256(args.preregistration),
        "depth3_freeze": str(freeze_path.resolve()),
        "depth3_freeze_sha256": file_sha256(freeze_path),
        "depth3_selected_actions": str(Path(freeze["selected_actions"]).resolve()),
        "depth3_selected_actions_sha256": file_sha256(Path(freeze["selected_actions"])),
        "depth_artifacts": depth_artifacts,
        "model": model_audit,
        "policy": policy,
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "config_unchanged_except_iterations": True,
        "depths": [3, 4, 5],
        "added_iterations": [3, 4],
        "cached_rng_verification": cached_rng_verification,
        "cached_ltwm_or_scorer_rounds_recomputed": 0,
        "added_iteration_audit": added_iteration_audit,
        "total_extra_new_challenger_evaluations": int(len(rows) * 16 * 2),
        "total_extra_ltwm_seed_candidate_rollouts": int(len(rows) * 17 * 3 * 2),
        "labels_opened": False,
        "outcomes_used_by_cem": False,
        "reserve_accessed": False,
        "go2_accessed": False,
        "production_safe_claim_allowed": False,
    }
    continuation_path = args.output_dir / "continuation_freeze.json"
    write_json_atomic(continuation_path, continuation)
    print(json.dumps(continuation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
