#!/usr/bin/env python3
"""Resume the sealed SCAND v66 harvest after an iteration-boundary CUDA OOM."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.scoring import run_dualdomain_cem_tail_v66_universal_valid as universal  # noqa: E402


implementation = universal.implementation
OUTPUT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/harvest/scand_train_fit"
RESUME_SEAL = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66_scand_train_fit_oom_resume.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_resume_seal(output: Path) -> None:
    seal = read_json(RESUME_SEAL)
    if seal.get("status") != "frozen_before_resume":
        raise RuntimeError("OOM resume seal changed")
    preserved = seal["preserved_artifacts"]
    paths = {
        "iteration_0_population_sha256": output / "iteration_00/population.npz",
        "iteration_0_features_sha256": output / "iteration_00/features.npz",
        "iteration_0_scores_sha256": output / "iteration_00/scores.npz",
        "iteration_1_population_sha256": output / "iteration_01/population.npz",
        "iteration_1_source_label_free_sha256": output / "iteration_01/source_label_free.npz",
    }
    for key, path in paths.items():
        if implementation.sha256(path) != preserved[key]:
            raise RuntimeError(f"preserved OOM-resume artifact changed: {path}")


def context(output: Path) -> dict[str, Any]:
    source, candidate, base, artifact = implementation.load_inputs("scand", "train_fit")
    training = implementation.read_json(implementation.TRAINING)
    envelopes = implementation.load_envelopes(Path(training["support"]))
    envelope = envelopes["scand"]
    bridge = np.float32(0.38 / 0.25)
    return {
        "output": output,
        "source": source,
        "candidate": candidate,
        "base": base,
        "artifact": artifact,
        "training": training,
        "envelope": envelope,
        "bridge": bridge,
        "spacing": 0.38,
        "incumbent": candidate["candidates_waypoint_units"][:, 0].astype(np.float32) * bridge,
        "reference": candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge,
        "family": source["family"].astype(str),
    }


def rng_for_scand_train_fit() -> np.random.Generator:
    return np.random.default_rng(20260730 + 100000)


def replay_population(
    ctx: dict[str, Any], rng: np.random.Generator, iteration: int
) -> dict[str, Any]:
    directory = ctx["output"] / f"iteration_{iteration:02d}"
    population = implementation.load_npz(directory / "population.npz")
    candidates, sampling = implementation.sample_population(
        incumbent=ctx["incumbent"],
        reference=ctx["reference"],
        mean=population["proposal_mean"].astype(np.float32),
        covariance=population["proposal_covariance"].astype(np.float32),
        population=16,
        rng=rng,
        support_envelope=ctx["envelope"],
        support_threshold=implementation.threshold_for_quantile(ctx["envelope"], 0.999),
        spacing_m=ctx["spacing"],
    )
    stored = population["candidates_feature_waypoint_units"].astype(np.float32)
    if not np.array_equal(candidates, stored):
        difference = np.abs(candidates - stored)
        raise RuntimeError(
            f"iteration {iteration} RNG replay changed: max_abs={difference.max()}"
        )
    return {"population": population, "sampling": sampling}


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_and_write(
    ctx: dict[str, Any], directory: Path, fold_id: np.ndarray
) -> dict[str, np.ndarray]:
    feature_path = directory / "features.npz"
    features = implementation.load_npz(feature_path)
    score = implementation.score_population(
        features,
        domain="scand",
        split="train_fit",
        training=ctx["training"],
        envelope=ctx["envelope"],
        fold_id=fold_id,
        device="cuda",
    )
    population = implementation.load_npz(directory / "population.npz")
    previous_mean = population["proposal_mean"].astype(np.float32)
    previous_covariance = population["proposal_covariance"].astype(np.float32)
    _, _, elite, updated, safe_count = implementation.safe_elite_update(
        population["candidates_feature_waypoint_units"].astype(np.float32),
        score["energy"],
        score["safe"],
        previous_mean=previous_mean,
        previous_covariance=previous_covariance,
        elite_count=4,
        floor=0.015,
        scale=0.70,
    )
    score_path = directory / "scores.npz"
    if score_path.exists():
        raise FileExistsError(score_path)
    implementation.write_npz_atomic(
        score_path,
        energy=score["energy"],
        mean=score["mean"],
        std=score["std"],
        dual_probability=score["p_dual"],
        harm_probability=score["p_harm"],
        rank_logit=score["rank_logit"],
        valid=score["valid"],
        safe=score["safe"],
        elite_index_for_update=elite,
        proposal_updated=updated,
        safe_count=safe_count,
        record_key=ctx["source"]["record_key"],
        fold_id=fold_id,
    )
    return score


def generate_iteration_two(ctx: dict[str, Any]) -> None:
    directory = ctx["output"] / "iteration_02"
    if directory.exists():
        raise FileExistsError(directory)
    rng = rng_for_scand_train_fit()
    replay_population(ctx, rng, 0)
    previous = replay_population(ctx, rng, 1)["population"]
    previous_score = implementation.load_npz(ctx["output"] / "iteration_01/scores.npz")
    mean, covariance, elite, _, _ = implementation.safe_elite_update(
        previous["candidates_feature_waypoint_units"].astype(np.float32),
        previous_score["energy"].astype(np.float32),
        previous_score["safe"].astype(bool),
        previous_mean=previous["proposal_mean"].astype(np.float32),
        previous_covariance=previous["proposal_covariance"].astype(np.float32),
        elite_count=4,
        floor=0.015,
        scale=0.70,
    )
    candidates, _ = implementation.sample_population(
        incumbent=ctx["incumbent"],
        reference=ctx["reference"],
        mean=mean,
        covariance=covariance,
        population=16,
        rng=rng,
        support_envelope=ctx["envelope"],
        support_threshold=implementation.threshold_for_quantile(ctx["envelope"], 0.999),
        spacing_m=ctx["spacing"],
    )
    directory.mkdir()
    fold_id = previous["fold_id"].astype(np.int64)
    implementation.write_npz_atomic(
        directory / "population.npz",
        candidates_feature_waypoint_units=candidates,
        candidates_native_waypoint_units=candidates / ctx["bridge"],
        proposal_mean=mean,
        proposal_covariance=covariance,
        elite_index_from_previous=elite,
        record_key=ctx["source"]["record_key"],
        route_group=ctx["source"]["route_group"],
        fold_id=fold_id,
    )
    action = implementation.reconstruct_action_features(
        candidates, ctx["reference"], ctx["base"]["action_features"]
    )
    implementation.write_npz_atomic(
        directory / "source_label_free.npz",
        candidates=candidates.astype(np.float32),
        action_features=action,
        record_key=ctx["source"]["record_key"],
        family=ctx["family"],
    )


def complete_iteration(iteration: int, output: Path, batch_size: int) -> None:
    if iteration not in (1, 2):
        raise ValueError("OOM resume only supports remaining iterations 1 and 2")
    verify_resume_seal(output)
    ctx = context(output)
    rng = rng_for_scand_train_fit()
    for existing in range(iteration):
        replay_population(ctx, rng, existing)
    if iteration == 2:
        if not (output / "iteration_01/scores.npz").exists():
            raise RuntimeError("complete iteration 1 before iteration 2")
        generate_iteration_two(ctx)
    directory = output / f"iteration_{iteration:02d}"
    feature_path = directory / "features.npz"
    if feature_path.exists():
        raise FileExistsError(feature_path)
    source_data = implementation.load_npz(directory / "source_label_free.npz")
    latents = implementation.load_npz(ctx["artifact"]["latent"])
    clear_cuda()
    implementation.export_features(
        source_data,
        latents,
        pairs=implementation.checkpoint_pairs(ROOT / "checkpoints/project"),
        split=f"dualdomain_cem_tail_v66_scand_train_fit_iter{iteration}",
        source=directory / "source_label_free.npz",
        latent_path=ctx["artifact"]["latent"],
        output=feature_path,
        include_labels=False,
        device=torch.device("cuda"),
        batch_size=batch_size,
        candidate_chunk=17,
    )
    del source_data, latents
    clear_cuda()
    population = implementation.load_npz(directory / "population.npz")
    score = score_and_write(ctx, directory, population["fold_id"].astype(np.int64))
    clear_cuda()
    print(
        json.dumps(
            {
                "stage": "resume_complete_iteration",
                "iteration": iteration,
                "batch_size": batch_size,
                "features_sha256": implementation.sha256(feature_path),
                "scores_sha256": implementation.sha256(directory / "scores.npz"),
                "valid_candidates": int(score["valid"][:, 1:].sum()),
                "safe_candidates": int(score["safe"][:, 1:].sum()),
                "outcomes_opened": False,
                "A_to_H_accessed": False,
                "reserve_accessed": False,
            },
            indent=2,
        )
    )


def iteration_record(ctx: dict[str, Any], rng: np.random.Generator, iteration: int) -> dict[str, Any]:
    directory = ctx["output"] / f"iteration_{iteration:02d}"
    replay = replay_population(ctx, rng, iteration)
    population = replay["population"]
    score = implementation.load_npz(directory / "scores.npz")
    next_mean, next_covariance, _, next_updated, _ = implementation.safe_elite_update(
        population["candidates_feature_waypoint_units"].astype(np.float32),
        score["energy"].astype(np.float32),
        score["safe"].astype(bool),
        previous_mean=population["proposal_mean"].astype(np.float32),
        previous_covariance=population["proposal_covariance"].astype(np.float32),
        elite_count=4,
        floor=0.015,
        scale=0.70,
    )
    paths = {
        "population": directory / "population.npz",
        "source": directory / "source_label_free.npz",
        "features": directory / "features.npz",
        "scores": directory / "scores.npz",
    }
    return {
        "iteration": iteration,
        **{
            key: str(path)
            for key, path in paths.items()
        },
        **{
            f"{key}_sha256": implementation.sha256(path)
            for key, path in paths.items()
        },
        "sampling": replay["sampling"],
        "valid_candidates": int(score["valid"][:, 1:].sum()),
        "safe_candidates": int(score["safe"][:, 1:].sum()),
        "updated_windows": int(next_updated.sum()),
        "safe_shortfall_windows": int((~next_updated).sum()),
        "proposal_before": implementation.proposal_summary(
            population["proposal_mean"].astype(np.float32),
            population["proposal_covariance"].astype(np.float32),
        ),
        "proposal_after": implementation.proposal_summary(next_mean, next_covariance),
    }


def finalize(output: Path) -> None:
    verify_resume_seal(output)
    ctx = context(output)
    freeze_path = output / "preoutcome_freeze.json"
    result_path = output / "result.json"
    if freeze_path.exists() or result_path.exists():
        raise FileExistsError(freeze_path if freeze_path.exists() else result_path)
    for iteration in range(3):
        directory = output / f"iteration_{iteration:02d}"
        for name in ("population.npz", "source_label_free.npz", "features.npz", "scores.npz"):
            if not (directory / name).exists():
                raise FileNotFoundError(directory / name)
        if (directory / "targets.npz").exists():
            raise RuntimeError("targets were opened before resume finalization")
    rng = rng_for_scand_train_fit()
    records = [iteration_record(ctx, rng, iteration) for iteration in range(3)]
    freeze = {
        "schema": "dualdomain_cem_tail_v66_preoutcome_freeze",
        "status": "candidates_scores_and_proposals_frozen_before_training_labels",
        "domain": "scand",
        "split": "train_fit",
        "states": int(len(ctx["source"]["record_key"])),
        "route_groups": int(len(np.unique(ctx["source"]["route_group"].astype(str)))),
        "behavior_policy": "original Arm-D route-OOF",
        "iterations": records,
        "source_artifacts": {
            key: {"path": str(path), "sha256": implementation.sha256(path)}
            for key, path in ctx["artifact"].items()
        },
        "training_result": str(implementation.TRAINING),
        "training_result_sha256": implementation.sha256(implementation.TRAINING),
        "preregistration": str(implementation.PREREG),
        "preregistration_sha256": implementation.sha256(implementation.PREREG),
        "oom_resume_seal": str(RESUME_SEAL),
        "oom_resume_seal_sha256": implementation.sha256(RESUME_SEAL),
        "resume_rng_replay_bit_exact": True,
        "outcomes_used_by_cem": False,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    implementation.write_json_atomic(freeze_path, freeze)
    fold_id = implementation.load_npz(output / "iteration_00/population.npz")["fold_id"].astype(np.int64)
    for iteration, record in enumerate(records):
        directory = output / f"iteration_{iteration:02d}"
        candidates = implementation.load_npz(directory / "population.npz")[
            "candidates_feature_waypoint_units"
        ].astype(np.float32)
        labels = implementation.metric_labels(
            candidates,
            ctx["source"]["ground_truth_local_m"],
            0.25,
            0.4,
            0.6,
        )
        target_path = directory / "targets.npz"
        implementation.write_npz_atomic(
            target_path,
            targets=labels["targets"],
            absolute_metrics=labels["values"],
            record_key=ctx["source"]["record_key"],
            route_group=ctx["source"]["route_group"],
            family=ctx["family"],
            fold_id=fold_id,
            iteration=np.asarray(iteration, dtype=np.int64),
            preoutcome_freeze_sha256=np.asarray(implementation.sha256(freeze_path)),
        )
        record["targets"] = str(target_path)
        record["targets_sha256"] = implementation.sha256(target_path)
    result = {
        "schema": "dualdomain_cem_tail_v66_harvest_result",
        "status": "complete",
        "domain": "scand",
        "split": "train_fit",
        "states": int(len(ctx["source"]["record_key"])),
        "iterations": records,
        "preoutcome_freeze": str(freeze_path),
        "preoutcome_freeze_sha256": implementation.sha256(freeze_path),
        "labels_opened_only_after_freeze": True,
        "oom_resume_protocol_preserving": True,
        "oom_resume_seal": str(RESUME_SEAL),
        "oom_resume_seal_sha256": implementation.sha256(RESUME_SEAL),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    implementation.write_json_atomic(result_path, result)
    print(
        json.dumps(
            {
                "stage": "resume_finalize",
                "status": "complete",
                "output": str(result_path),
                "sha256": implementation.sha256(result_path),
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("complete", "finalize"))
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "complete":
        if args.iteration is None:
            raise ValueError("complete requires --iteration")
        complete_iteration(args.iteration, args.output, args.batch_size)
    else:
        finalize(args.output)


if __name__ == "__main__":
    main()
