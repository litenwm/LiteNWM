#!/usr/bin/env python3
"""Freeze outcome-free CEM replays for the original Arm-D or v66 scorer.

This entrypoint deliberately stops before opening future-pose outcomes.  It
supports two already-frozen model contracts:

* ``original_arm_d``: the fixed-pool/CEM=false-trained Arm-D baseline;
* ``proposal_aware_v66``: the proposal-aware Arm-D fine-tune.

Both roles use the same universal validity contract and the same three-round
candidate-level safe-elite CEM protocol.  Labels are opened only by a separate
evaluator after both replays and their SHA-256 bindings have been frozen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.checkpoint_io import load_checkpoint
from planning.proposal_tail_sidecar_v1 import (  # noqa: E402
    ProposalTailSidecar,
    SidecarBindings,
)
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
    initial_proposal,
    proposal_summary,
    safe_elite_update,
    select_lowest_energy_safe,
)
from tools.scoring.finetune_universal_dualdomain_safe_rank_v2 import (  # noqa: E402
    load_envelopes,
)
from tools.scoring.train_recon_dense_meta_dinov3_three_arm_v1 import (  # noqa: E402
    load_features,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    ensemble_predict,
)


BASE_RESULT = ROOT / "outputs/universal_dualdomain_raw_v2/training_v2/result.json"
V66_RESULT = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66/training/result.json"
V66_PREREG = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v66_preregistration.json"
PROPOSAL_TAIL_PREREG = (
    ROOT
    / "configs/proposal_tail_collection/scand_proposal_tail_collection_v1_preregistration.json"
)
PROPOSAL_TAIL_SCHEMA = (
    ROOT
    / "configs/proposal_tail_collection/scand_proposal_tail_collection_v1.schema.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("recon", "scand"), required=True)
    parser.add_argument(
        "--model-role",
        choices=("original_arm_d", "proposal_aware_v66"),
        required=True,
    )
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--latent-cache", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, default=BASE_RESULT)
    parser.add_argument("--v66-result", type=Path, default=V66_RESULT)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--proposal-scale", type=float, default=0.70)
    parser.add_argument("--covariance-floor", type=float, default=0.015)
    parser.add_argument("--support-quantile", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--wm-batch-size", type=int, default=64)
    parser.add_argument(
        "--proposal-tail-sidecar-dir",
        type=Path,
        default=None,
        help="Opt-in decision-isolated SCAND proposal-tail output; default off.",
    )
    parser.add_argument(
        "--proposal-tail-cohort-role",
        choices=("spent_smoke", "development", "one_time_validation"),
        default=None,
    )
    return parser.parse_args()


def frozen_model_contract(
    *,
    role: str,
    domain: str,
    base_result_path: Path,
    v66_result_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, float], float, float, dict[str, Any]]:
    base = read_json(base_result_path)
    if base.get("cem") is not False:
        raise RuntimeError("original Arm-D result is not the frozen CEM=false model")
    if role == "original_arm_d":
        arm = base["arms"]["D"]
        policy = arm["calibration"]["frontier"]["domains"][domain]["policy"]
        temperature = arm["temperature"]["domains"][domain]
        audit = {
            "model_role": role,
            "model_result": str(base_result_path),
            "model_result_sha256": file_sha256(base_result_path),
            "model_trained_with_cem": False,
            "calibration_feasible": bool(arm["calibration"]["feasible"]),
        }
        return (
            arm["checkpoints"]["full"],
            policy,
            float(temperature["deployment_dual_temperature"]),
            float(temperature["deployment_harm_temperature"]),
            audit,
        )

    v66 = read_json(v66_result_path)
    if v66.get("status") != "trained_and_frozen_before_A_to_H_replay":
        raise RuntimeError("v66 model/gate is not frozen before A-H replay")
    if v66.get("A_to_H_accessed") or v66.get("reserve_accessed"):
        raise RuntimeError("v66 training seal reports forbidden evaluation access")
    calibration = v66["calibration"][domain]
    if not calibration.get("feasible"):
        raise RuntimeError(f"v66 {domain} scorer-cal policy is infeasible; do not spend A-H")
    audit = {
        "model_role": role,
        "model_result": str(v66_result_path),
        "model_result_sha256": file_sha256(v66_result_path),
        "model_trained_with_cem": True,
        "calibration_feasible": True,
    }
    return (
        v66["checkpoints"]["full"],
        calibration["policy"],
        float(calibration["dual_temperature"]),
        float(calibration["harm_temperature"]),
        audit,
    )


def validate_protocol(args: argparse.Namespace) -> None:
    if args.iterations != 3 or args.population != 16:
        raise ValueError("v66 replay is frozen at 3 iterations and population 16")
    if abs(args.elite_fraction - 0.25) > 1.0e-12:
        raise ValueError("v66 replay elite fraction is frozen at 0.25")
    if abs(args.proposal_scale - 0.70) > 1.0e-12:
        raise ValueError("v66 replay proposal scale is frozen at 0.70")
    if abs(args.support_quantile - 0.999) > 1.0e-12:
        raise ValueError("v66 replay support quantile is frozen at 0.999")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.cohort):
        raise ValueError("cohort must be a filesystem-safe identifier")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(args.output_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    prereg = read_json(V66_PREREG)
    if prereg.get("status") != "frozen_before_proposal_harvest":
        raise RuntimeError("v66 preregistration seal changed")
    sidecar_enabled = args.proposal_tail_sidecar_dir is not None
    if sidecar_enabled != (args.proposal_tail_cohort_role is not None):
        raise ValueError("sidecar directory and cohort role must be supplied together")
    if sidecar_enabled:
        if args.domain != "scand":
            raise ValueError("proposal-tail sidecar is SCAND-only")
        formal_root = args.output_dir.resolve()
        sidecar_root = args.proposal_tail_sidecar_dir.resolve()
        if sidecar_root == formal_root or formal_root in sidecar_root.parents:
            raise ValueError("sidecar output must be outside the formal replay directory")
        proposal_prereg = read_json(PROPOSAL_TAIL_PREREG)
        if proposal_prereg.get("status") != "frozen_before_implementation_and_smoke":
            raise RuntimeError("proposal-tail preregistration seal changed")


def main() -> None:
    args = parse_args()
    validate_protocol(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest)
    base = load_features(args.base_features, labels_expected=False)
    candidate = load_npz(args.candidate)
    expected_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not len(rows):
        raise RuntimeError("empty replay cohort")
    if not (
        np.array_equal(candidate["record_key"].astype(str), expected_record)
        and np.array_equal(base["record_key"].astype(str), expected_record)
    ):
        raise RuntimeError("manifest/candidate/base-feature identity mismatch")

    bridge = np.float32(1.0 if args.domain == "recon" else 0.38 / 0.25)
    spacing_m = 0.25 if args.domain == "recon" else 0.38
    incumbent = candidate["candidates_waypoint_units"][:, 0].astype(np.float32) * bridge
    reference = candidate["nomad_x16_waypoint_units"].astype(np.float32) * bridge
    base_candidates = np.concatenate((incumbent[:, None], reference), axis=1)
    expected_candidates = base["candidates"].astype(np.float32)
    sacson_alias = os.environ.get("TRIDOMAIN_RUNTIME_DATASET") == "sacson"
    bridge_matches = (
        (
            base_candidates.shape == expected_candidates.shape
            and np.allclose(
                base_candidates,
                expected_candidates,
                rtol=0.0,
                atol=4.0 * np.finfo(np.float32).eps,
            )
        )
        if sacson_alias
        else np.array_equal(base_candidates, expected_candidates)
    )
    if not bridge_matches:
        raise RuntimeError("base feature/candidate waypoint bridge mismatch")

    checkpoints, policy, dual_temperature, harm_temperature, model_audit = (
        frozen_model_contract(
            role=args.model_role,
            domain=args.domain,
            base_result_path=args.base_result,
            v66_result_path=args.v66_result,
        )
    )
    base_training = read_json(args.base_result)
    support_path = Path(base_training["support"])
    envelope = load_envelopes(support_path)[args.domain]
    threshold = threshold_for_quantile(envelope, args.support_quantile)
    domain_id = np.full(len(rows), 0 if args.domain == "recon" else 1, dtype=np.int64)

    feature, support, guard = build_universal_features(
        base, spacing_m=spacing_m, support_envelope=envelope
    )
    valid = guard & (support <= threshold)
    valid[:, 0] = True
    prediction = ensemble_predict(checkpoints, feature, domain_id, device=args.device)
    energy, p_dual, p_harm = conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    safe = candidate_safety_mask(prediction, p_dual, p_harm, valid, policy)
    fixed_selected = select_lowest_energy_safe(energy, safe)
    sidecar = None
    if args.proposal_tail_sidecar_dir is not None:
        sidecar = ProposalTailSidecar(
            args.proposal_tail_sidecar_dir,
            cohort_role=args.proposal_tail_cohort_role,
            cohort=args.cohort,
            rows=rows,
            bindings=SidecarBindings(
                manifest=args.manifest,
                preregistration=PROPOSAL_TAIL_PREREG,
                schema=PROPOSAL_TAIL_SCHEMA,
                support_checkpoint=support_path,
                model_result=Path(model_audit["model_result"]),
                runner_v66=Path(__file__),
                runner_v70=ROOT / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py",
                sampler=ROOT / "tools/planning/run_cem_development_v1.py",
            ),
            policy=policy,
            support_threshold=threshold,
            support_quantile=args.support_quantile,
        )
        sidecar.record_pool(
            -1,
            candidates=base_candidates,
            actions_h4=base["actions_h4"],
            support_score=support,
            physical_guard=guard,
            support_mask=support <= threshold,
            valid=valid,
            safe=safe,
            energy=energy,
            prediction=prediction,
            p_dual=p_dual,
            p_harm=p_harm,
            feature_data=base,
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
    final_energy = energy
    final_valid = valid
    final_safe = safe
    latents = load_npz(args.latent_cache)
    pairs = checkpoint_pairs(ROOT / "checkpoints/project")

    for iteration in range(args.iterations):
        proposal_before = proposal_summary(mean, covariance)
        if sidecar is not None:
            sidecar.begin_iteration(
                iteration,
                proposal_mean=mean,
                proposal_covariance=covariance,
                incumbent=incumbent,
                reference=reference,
            )
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
            observer=(
                sidecar.capture_sampling_event if sidecar is not None else None
            ),
        )
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
        source_path = directory / "source_label_free.npz"
        source = build_source(
            candidates=candidates,
            reference=reference,
            rows=rows,
            trajectory_root=args.trajectory_root,
            domain=args.domain,
            output=source_path,
        )
        feature_path = directory / "cem_features.npz"
        export_features(
            source,
            latents,
            pairs=pairs,
            split=(
                f"dualdomain_cem_tail_v66_{args.domain}_{args.cohort}_"
                f"{args.model_role}_iter{iteration}"
            ),
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
        local_feature, local_support, local_guard = build_universal_features(
            data, spacing_m=spacing_m, support_envelope=envelope
        )
        local_valid = local_guard & (local_support <= threshold)
        local_valid[:, 0] = True
        local_prediction = ensemble_predict(
            checkpoints, local_feature, domain_id, device=args.device
        )
        local_energy, local_p_dual, local_p_harm = conservative_energy(
            local_prediction,
            policy=policy,
            dual_temperature=dual_temperature,
            harm_temperature=harm_temperature,
            valid=local_valid,
        )
        local_safe = candidate_safety_mask(
            local_prediction, local_p_dual, local_p_harm, local_valid, policy
        )
        next_mean, next_covariance, next_elite, next_updated, next_safe_count = (
            safe_elite_update(
                candidates,
                local_energy,
                local_safe,
                previous_mean=mean,
                previous_covariance=covariance,
                elite_count=elite_count,
                floor=args.covariance_floor,
                scale=args.proposal_scale,
            )
        )
        if sidecar is not None:
            sidecar.record_pool(
                iteration,
                candidates=candidates,
                actions_h4=data["actions_h4"],
                support_score=local_support,
                physical_guard=local_guard,
                support_mask=local_support <= threshold,
                valid=local_valid,
                safe=local_safe,
                energy=local_energy,
                prediction=local_prediction,
                p_dual=local_p_dual,
                p_harm=local_p_harm,
                feature_data=data,
                elite_index=next_elite,
                proposal_updated=next_updated,
                safe_count=next_safe_count,
            )
        score_path = directory / "frozen_scores.npz"
        write_npz_atomic(
            score_path,
            energy=local_energy,
            mean=local_prediction["mean"],
            std=local_prediction["std"],
            dual_probability=local_p_dual,
            harm_probability=local_p_harm,
            rank_logit=local_prediction["rank_logit"],
            valid=local_valid,
            safe=local_safe,
            elite_index_for_update=next_elite,
            proposal_updated=next_updated,
            safe_count=next_safe_count,
            record_key=expected_record,
        )
        elite_energy = np.asarray(
            [
                local_energy[state, index]
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
                "valid_candidates": int(local_valid[:, 1:].sum()),
                "safe_candidates": int(local_safe[:, 1:].sum()),
                "elite_candidates": int(next_updated.sum() * elite_count),
                "ood_or_invalid_candidates": int(
                    local_valid[:, 1:].size - local_valid[:, 1:].sum()
                ),
                "windows_with_any_valid_challenger": int(
                    local_valid[:, 1:].any(axis=1).sum()
                ),
                "windows_with_any_safe_challenger": int(
                    local_safe[:, 1:].any(axis=1).sum()
                ),
                "proposal_updated_windows": int(next_updated.sum()),
                "safe_shortfall_fallback_windows": int((~next_updated).sum()),
                "safe_count_min": int(next_safe_count.min()),
                "safe_count_mean": float(next_safe_count.mean()),
                "elite_scorer_energy_mean": (
                    float(elite_energy.mean()) if elite_energy.size else None
                ),
                "proposal_before": proposal_before,
                "proposal_after": proposal_summary(next_mean, next_covariance),
                "true_utility_available_before_outcomes": False,
            }
        )
        mean, covariance, elite_index = next_mean, next_covariance, next_elite
        final_candidates = candidates
        final_energy = local_energy
        final_valid = local_valid
        final_safe = local_safe

    cem_selected = select_lowest_energy_safe(final_energy, final_safe)
    selected_path = args.output_dir / "selected_actions.npz"
    write_npz_atomic(
        selected_path,
        record_key=expected_record,
        incumbent_feature_waypoint_units=incumbent,
        base_k17_feature_waypoint_units=base_candidates,
        final_cem_feature_waypoint_units=final_candidates,
        fixed_pool_selected_index=fixed_selected,
        final_cem_selected_index=cem_selected,
        base_valid=valid,
        base_safe=safe,
        final_cem_energy=final_energy,
        final_cem_valid=final_valid,
        final_cem_safe=final_safe,
    )
    freeze_path = args.output_dir / "prelabel_freeze.json"
    freeze = {
        "schema": "dualdomain_cem_tail_v66_prelabel_replay",
        "status": "candidate_level_safe_elite_actions_frozen_before_outcomes",
        "evidence_role": "spent_development_replay_only",
        "domain": args.domain,
        "cohort": args.cohort,
        "states": len(rows),
        "model": model_audit,
        "labels_opened": False,
        "future_pose_outcomes_opened": False,
        "outcomes_used_by_cem": False,
        "online_causal": True,
        "candidate_level_safety_elite": True,
        "universal_valid_contract": "physical_guard AND spacing_support<=q0.999",
        "legacy_majority_support_used": False,
        "safe_shortfall_rule": "retain previous proposal unchanged",
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "candidate_pool": str(args.candidate),
        "candidate_pool_sha256": file_sha256(args.candidate),
        "base_features": str(args.base_features),
        "base_features_sha256": file_sha256(args.base_features),
        "latent_cache": str(args.latent_cache),
        "latent_cache_sha256": file_sha256(args.latent_cache),
        "support_checkpoint": str(support_path),
        "support_checkpoint_sha256": file_sha256(support_path),
        "v66_preregistration": str(V66_PREREG),
        "v66_preregistration_sha256": file_sha256(V66_PREREG),
        "policy": policy,
        "dual_temperature": dual_temperature,
        "harm_temperature": harm_temperature,
        "config": {
            "iterations": args.iterations,
            "population_challengers": args.population,
            "set_cardinality_with_incumbent": args.population + 1,
            "elite_fraction": args.elite_fraction,
            "elite_count": elite_count,
            "proposal_scale": args.proposal_scale,
            "covariance_floor_waypoint_units": args.covariance_floor,
            "support_quantile": args.support_quantile,
            "support_threshold": threshold,
            "seed": args.seed,
            "spacing_m": spacing_m,
        },
        "base_initialization": base_initialization,
        "fixed_pool_promotions": int((fixed_selected != 0).sum()),
        "fixed_pool_coverage": float((fixed_selected != 0).mean()),
        "cem_promotions": int((cem_selected != 0).sum()),
        "cem_coverage": float((cem_selected != 0).mean()),
        "iteration_audit": iteration_audit,
        "selected_actions": str(selected_path),
        "selected_actions_sha256": file_sha256(selected_path),
        "final_candidates_sha256": array_sha256(final_candidates),
        "A_to_H_accessed": False,
        "reserve_accessed": False,
        "production_safe_claim_allowed": False,
    }
    write_json_atomic(freeze_path, freeze)
    if sidecar is not None:
        sidecar.finalize(
            formal_selected_path=selected_path,
            formal_freeze_path=freeze_path,
        )
    print(json.dumps(freeze, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
