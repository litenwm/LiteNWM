#!/usr/bin/env python3
"""Harvest outcome-free route-OOF CEM proposal populations for v66 training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scoring.spacing_support_v1 import threshold_for_quantile  # noqa: E402
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
    group_fold,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    metric_labels,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    checkpoint_pairs,
    export_features,
    load_npz,
    write_json_atomic,
    write_npz_atomic,
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
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
    load_member,
    predict_member,
)


PREREG = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66_preregistration.json"
TRAINING = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
OUTPUT_ROOT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/harvest"


def paths(domain: str, split: str) -> dict[str, Path]:
    if domain == "recon":
        prefix = "recon_train_fit" if split == "train_fit" else "recon_scorer_cal"
        return {
            "source": ROOT / "outputs/universal_dualdomain_raw_v2/sources" / f"{prefix}_source.npz",
            "candidate": ROOT / "outputs/universal_dualdomain_raw_v2/candidates" / f"{prefix}_k17.npz",
            "feature": ROOT / "outputs/universal_dualdomain_raw_v2/features" / prefix / f"{prefix}_universal_v2_features.npz",
            "latent": ROOT / "outputs/universal_dualdomain_raw_v2/features" / prefix / f"{prefix}_universal_v2_latents.npz",
        }
    stem = "train_fit" if split == "train_fit" else "scorer_cal"
    feature_stem = f"scand_inner_{stem}_spacing038"
    return {
        "source": ROOT / "outputs/scand_domain_adaptation_v1/sources" / f"{stem}_source.npz",
        "candidate": ROOT / "outputs/scand_domain_adaptation_v1/candidates" / f"{stem}_k17.npz",
        "feature": ROOT / "outputs/scand_domain_adaptation_v1/features" / stem / f"{feature_stem}_features.npz",
        "latent": ROOT / "outputs/scand_domain_adaptation_v1/features" / stem / f"{feature_stem}_latents.npz",
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstructed_history(
    candidates: np.ndarray,
    base_action_features: np.ndarray,
) -> np.ndarray:
    """Recompute the 38-D history block from the cached causal state block."""

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
        increment = speed_weighted[:, None] * np.stack((np.cos(heading), np.sin(heading)), axis=-1)
        if future_step:
            expected_twist[:, future_step] = expected_twist[:, future_step - 1]
        expected_twist[:, future_step] += increment

    origin = np.zeros((count, population, 1, 2), dtype=np.float32)
    candidate_steps = np.diff(np.concatenate((origin, candidates_m), axis=2), axis=2)
    candidate_speed = np.linalg.norm(candidate_steps, axis=-1)

    def residual(expected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distance = np.linalg.norm(candidates_m - expected[:, None], axis=-1)
        return distance.mean(axis=2), distance[:, :, -1]

    last_mean, last_final = residual(expected_last)
    weighted_mean, weighted_final = residual(expected_weighted)
    twist_mean, twist_final = residual(expected_twist)
    expected_direction = velocity_weighted / np.maximum(np.linalg.norm(velocity_weighted, axis=1, keepdims=True), 1.0e-6)
    candidate_direction = candidates_m[:, :, -1] / np.maximum(np.linalg.norm(candidates_m[:, :, -1], axis=-1, keepdims=True), 1.0e-6)
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
            np.abs(np.linalg.norm(candidates_m[:, :, -1], axis=-1) / horizon - speed_weighted[:, None]),
            1.0 - np.einsum("nkd,nd->nk", candidate_direction, expected_direction),
        ),
        axis=-1,
    ).astype(np.float32)
    relative = np.clip((specifics - center) / scale, -8.0, 8.0).astype(np.float32)
    return np.concatenate((np.repeat(state[:, None], population, axis=1), specifics, relative), axis=-1).astype(np.float32)


def reconstruct_action_features(
    candidates: np.ndarray,
    reference: np.ndarray,
    base_action_features: np.ndarray,
) -> np.ndarray:
    action_set = action_set_features(candidates, reference)
    history = reconstructed_history(candidates, base_action_features)
    raw = np.concatenate((action_set, history), axis=-1).astype(np.float32)
    result = np.concatenate((raw, raw - raw[:, :1]), axis=-1).astype(np.float32)
    if result.shape[-1] != 242:
        raise RuntimeError("reconstructed action feature dimension changed")
    return result


def oof_predict(
    training: dict[str, Any],
    feature: np.ndarray,
    domain_id: np.ndarray,
    fold_id: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    result = {
        key: np.empty((len(feature), feature.shape[1], 4) if key in {"mean", "std"} else (len(feature), feature.shape[1]), dtype=np.float32)
        for key in ("mean", "std", "helpful_logit", "dual_logit", "harm_logit", "rank_logit")
    }
    records = training["arms"]["D"]["checkpoints"]["oof"]
    for heldout, record in enumerate(records):
        index = np.flatnonzero(fold_id == heldout)
        if not len(index):
            continue
        member = load_member(record, device)
        prediction = predict_member(member, feature[index], domain_id[index], device=device)
        for key in result:
            result[key][index] = np.sqrt(np.maximum(prediction["variance"], 1.0e-8)) if key == "std" else prediction[key]
        del member
        if device == "cuda":
            torch.cuda.empty_cache()
    return result


def score_population(
    data: dict[str, np.ndarray],
    *,
    domain: str,
    split: str,
    training: dict[str, Any],
    envelope: dict[str, Any],
    fold_id: np.ndarray,
    device: str,
) -> dict[str, np.ndarray]:
    spacing = 0.25 if domain == "recon" else 0.38
    feature, support, guard = build_universal_features(data, spacing_m=spacing, support_envelope=envelope)
    valid = guard & (support <= threshold_for_quantile(envelope, 0.999)) & np.asarray(data["majority_support"], dtype=bool)
    valid[:, 0] = True
    domain_id = np.full(len(feature), 0 if domain == "recon" else 1, dtype=np.int64)
    prediction = (
        oof_predict(training, feature, domain_id, fold_id, device)
        if split == "train_fit"
        else ensemble_predict(training["arms"]["D"]["checkpoints"]["full"], feature, domain_id, device=device)
    )
    policy = training["arms"]["D"]["calibration"]["frontier"]["domains"][domain]["policy"]
    temperature = training["arms"]["D"]["temperature"]["domains"][domain]
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=float(temperature["deployment_dual_temperature"]),
        harm_temperature=float(temperature["deployment_harm_temperature"]),
        valid=valid,
    )
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    return {**prediction, "energy": energy, "p_dual": p_dual, "p_harm": p_harm, "valid": valid, "safe": safe}


def load_inputs(domain: str, split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Path]]:
    artifact = paths(domain, split)
    source = load_npz(artifact["source"])
    candidate = load_npz(artifact["candidate"])
    base = load_npz(artifact["feature"])
    if not np.array_equal(source["record_key"].astype(str), base["record_key"].astype(str)) or not np.array_equal(source["record_key"].astype(str), candidate["record_key"].astype(str)):
        raise RuntimeError(f"{domain}/{split}: identity mismatch")
    bridge = np.float32(1.0 if domain == "recon" else 0.38 / 0.25)
    expected_candidates = candidate["candidates_waypoint_units"].astype(np.float32) * bridge
    if not np.array_equal(expected_candidates, base["candidates"].astype(np.float32)):
        raise RuntimeError(f"{domain}/{split}: feature/candidate bridge mismatch")
    return source, candidate, base, artifact


def preflight(output_root: Path) -> None:
    result = {"schema": "dualdomain_cem_tail_v66_preflight", "checks": {}}
    for domain in ("recon", "scand"):
        source, candidate, base, artifact = load_inputs(domain, "train_fit")
        count = min(128, len(base["record_key"]))
        bridge = np.float32(1.0 if domain == "recon" else 0.38 / 0.25)
        reference = candidate["nomad_x16_waypoint_units"][:count].astype(np.float32) * bridge
        candidates = base["candidates"][:count].astype(np.float32)
        reconstructed = reconstruct_action_features(candidates, reference, base["action_features"][:count])
        difference = np.abs(reconstructed - base["action_features"][:count])
        check = {
            "states": count,
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "allclose_at_2e_5": bool(np.allclose(reconstructed, base["action_features"][:count], atol=2.0e-5, rtol=1.0e-5)),
            "artifacts": {key: {"path": str(path), "sha256": sha256(path)} for key, path in artifact.items()},
            "source_targets_available": "targets" in source,
            "ground_truth_available": "ground_truth_local_m" in source,
        }
        if not check["allclose_at_2e_5"]:
            raise RuntimeError(f"{domain}: cached causal action reconstruction failed: {check}")
        result["checks"][domain] = check
    result.update({
        "status": "passed",
        "preregistration": str(PREREG),
        "preregistration_sha256": sha256(PREREG),
        "outcomes_opened": False,
        "candidate_generation_started": False,
        "reserve_accessed": False,
    })
    path = output_root.parent / "preflight.json"
    if path.exists():
        raise FileExistsError(path)
    write_json_atomic(path, result)
    print(json.dumps({"stage": "preflight", "status": "passed", "output": str(path), "sha256": sha256(path)}, indent=2))


def harvest(domain: str, split: str, output_root: Path, device: str, batch_size: int) -> None:
    output = output_root / f"{domain}_{split}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    source, candidate, base, artifact = load_inputs(domain, split)
    training = read_json(TRAINING)
    envelopes = load_envelopes(Path(training["support"]))
    envelope = envelopes[domain]
    bridge = np.float32(1.0 if domain == "recon" else 0.38 / 0.25)
    spacing = 0.25 if domain == "recon" else 0.38
    incumbent = candidate["candidates_waypoint_units"][:, 0].astype(np.float32) * bridge
    reference = candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge
    family = source["family"].astype(str)
    fold_id = np.asarray([group_fold(domain, value, 5) for value in family], dtype=np.int64)

    base_score = score_population(base, domain=domain, split=split, training=training, envelope=envelope, fold_id=fold_id, device=device)
    mean, covariance = initial_proposal(reference, floor=0.015, scale=0.70)
    mean, covariance, elite, updated, safe_count = safe_elite_update(
        base["candidates"], base_score["energy"], base_score["safe"],
        previous_mean=mean, previous_covariance=covariance, elite_count=4, floor=0.015, scale=0.70,
    )
    rng = np.random.default_rng(20260730 + (0 if domain == "recon" else 100000) + (0 if split == "train_fit" else 10000))
    iteration_records = []
    label_pending = []
    pairs = checkpoint_pairs(ROOT / "checkpoints/project")
    latents = load_npz(artifact["latent"])
    for iteration in range(3):
        directory = output / f"iteration_{iteration:02d}"
        directory.mkdir()
        before = proposal_summary(mean, covariance)
        candidates, sampling = sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            population=16,
            rng=rng,
            support_envelope=envelope,
            support_threshold=threshold_for_quantile(envelope, 0.999),
            spacing_m=spacing,
        )
        population_path = directory / "population.npz"
        write_npz_atomic(
            population_path,
            candidates_feature_waypoint_units=candidates,
            candidates_native_waypoint_units=candidates / bridge,
            proposal_mean=mean,
            proposal_covariance=covariance,
            elite_index_from_previous=elite,
            record_key=source["record_key"],
            route_group=source["route_group"],
            fold_id=fold_id,
        )
        action = reconstruct_action_features(candidates, reference, base["action_features"])
        source_path = directory / "source_label_free.npz"
        source_data = {
            "candidates": candidates.astype(np.float32),
            "action_features": action,
            "record_key": source["record_key"],
            "family": family,
        }
        write_npz_atomic(source_path, **source_data)
        feature_path = directory / "features.npz"
        export_features(
            source_data,
            latents,
            pairs=pairs,
            split=f"dualdomain_cem_tail_v66_{domain}_{split}_iter{iteration}",
            source=source_path,
            latent_path=artifact["latent"],
            output=feature_path,
            include_labels=False,
            device=torch.device(device),
            batch_size=batch_size,
            candidate_chunk=17,
        )
        features = load_npz(feature_path)
        score = score_population(features, domain=domain, split=split, training=training, envelope=envelope, fold_id=fold_id, device=device)
        next_mean, next_covariance, next_elite, next_updated, next_safe_count = safe_elite_update(
            candidates, score["energy"], score["safe"],
            previous_mean=mean, previous_covariance=covariance, elite_count=4, floor=0.015, scale=0.70,
        )
        score_path = directory / "scores.npz"
        write_npz_atomic(
            score_path,
            energy=score["energy"], mean=score["mean"], std=score["std"],
            dual_probability=score["p_dual"], harm_probability=score["p_harm"],
            rank_logit=score["rank_logit"], valid=score["valid"], safe=score["safe"],
            elite_index_for_update=next_elite, proposal_updated=next_updated,
            safe_count=next_safe_count, record_key=source["record_key"], fold_id=fold_id,
        )
        iteration_records.append({
            "iteration": iteration,
            "population": str(population_path), "population_sha256": sha256(population_path),
            "source": str(source_path), "source_sha256": sha256(source_path),
            "features": str(feature_path), "features_sha256": sha256(feature_path),
            "scores": str(score_path), "scores_sha256": sha256(score_path),
            "sampling": sampling,
            "valid_candidates": int(score["valid"][:, 1:].sum()),
            "safe_candidates": int(score["safe"][:, 1:].sum()),
            "updated_windows": int(next_updated.sum()),
            "safe_shortfall_windows": int((~next_updated).sum()),
            "proposal_before": before,
            "proposal_after": proposal_summary(next_mean, next_covariance),
        })
        label_pending.append((directory, candidates))
        mean, covariance, elite, updated, safe_count = next_mean, next_covariance, next_elite, next_updated, next_safe_count

    freeze_path = output / "preoutcome_freeze.json"
    freeze = {
        "schema": "dualdomain_cem_tail_v66_preoutcome_freeze",
        "status": "candidates_scores_and_proposals_frozen_before_training_labels",
        "domain": domain,
        "split": split,
        "states": int(len(source["record_key"])),
        "route_groups": int(len(np.unique(source["route_group"].astype(str)))),
        "behavior_policy": "original Arm-D route-OOF" if split == "train_fit" else "original Arm-D full ensemble",
        "iterations": iteration_records,
        "source_artifacts": {key: {"path": str(path), "sha256": sha256(path)} for key, path in artifact.items()},
        "training_result": str(TRAINING),
        "training_result_sha256": sha256(TRAINING),
        "preregistration": str(PREREG),
        "preregistration_sha256": sha256(PREREG),
        "outcomes_used_by_cem": False,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    write_json_atomic(freeze_path, freeze)

    for iteration, (directory, candidates) in enumerate(label_pending):
        labels = metric_labels(candidates, source["ground_truth_local_m"], 0.25, 0.4, 0.6)
        target_path = directory / "targets.npz"
        write_npz_atomic(
            target_path,
            targets=labels["targets"],
            absolute_metrics=labels["values"],
            record_key=source["record_key"],
            route_group=source["route_group"],
            family=family,
            fold_id=fold_id,
            iteration=np.asarray(iteration, dtype=np.int64),
            preoutcome_freeze_sha256=np.asarray(sha256(freeze_path)),
        )
        iteration_records[iteration]["targets"] = str(target_path)
        iteration_records[iteration]["targets_sha256"] = sha256(target_path)

    result_path = output / "result.json"
    result = {
        "schema": "dualdomain_cem_tail_v66_harvest_result",
        "status": "complete",
        "domain": domain,
        "split": split,
        "states": int(len(source["record_key"])),
        "iterations": iteration_records,
        "preoutcome_freeze": str(freeze_path),
        "preoutcome_freeze_sha256": sha256(freeze_path),
        "labels_opened_only_after_freeze": True,
        "A_to_H_accessed": False,
        "reserve_accessed": False,
    }
    write_json_atomic(result_path, result)
    print(json.dumps({"stage": "harvest", "domain": domain, "split": split, "states": len(source["record_key"]), "output": str(result_path), "sha256": sha256(result_path)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "harvest"))
    parser.add_argument("--domain", choices=("recon", "scand"))
    parser.add_argument("--split", choices=("train_fit", "scorer_cal"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if read_json(PREREG).get("status") != "frozen_before_proposal_harvest":
        raise RuntimeError("v66 preregistration status mismatch")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if args.stage == "preflight":
        preflight(args.output_root)
    else:
        if args.domain is None or args.split is None:
            raise ValueError("harvest requires --domain and --split")
        harvest(args.domain, args.split, args.output_root, args.device, args.batch_size)


if __name__ == "__main__":
    main()
