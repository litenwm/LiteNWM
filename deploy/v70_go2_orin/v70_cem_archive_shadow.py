#!/usr/bin/env python3
"""Three-round v70 archive-CEM shadow runner for the current live state.

Reads only cached shadow artifacts. It creates no ROS publishers and issues no
Unitree motion commands.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(DEPLOY_DIR)]

from planning.costs import latent_trajectory_features  # noqa: E402
from planning.nomad_action_adapter import (  # noqa: E402
    nomad_waypoints_to_local_actions,
)
from planning.trajectory_constraints import (  # noqa: E402
    evaluate_trajectory_constraints,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.trajectory_quality_scorer_v2 import (  # noqa: E402
    TrajectoryQualityScorerConfig,
    TrajectoryQualityScorerV2,
)
from scoring.universal_dualdomain_raw_v2 import (  # noqa: E402
    build_universal_features,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
)
from tools.planning import (  # noqa: E402
    run_dualdomain_cem_tail_v66_replay as replay,
)
from tools.planning import (  # noqa: E402
    run_dualdomain_cem_tail_v70_archive_replay as v70,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
)

from v70_armd_fixed_pool_shadow import (  # noqa: E402
    localize_checkpoint_records,
)
from v70_ltwm_fixed_pool_shadow import support_stats  # noqa: E402
from v70_pose_features import (  # noqa: E402
    PoseSample,
    build_history_features,
)

DOMAIN = "recon"
DOMAIN_ID = 0
SPACING_M = 0.25
SUPPORT_QUANTILE = 0.999
EXPECTED_THRESHOLD = 3.783139705657959

SEEDS = (1701, 1702, 1703)
K = 17
HORIZON = 4
TOKEN_COUNT = 256
LATENT_DIM = 384

POPULATION = 16
ELITE_FRACTION = 0.25
ELITE_COUNT = 4
ITERATIONS = 3
PROPOSAL_SCALE = 0.7
COVARIANCE_FLOOR = 0.015
RNG_SEED = 20260730


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-features",
        type=Path,
        default=Path("/home/unitree/v70_online_features_shadow.npz"),
    )
    parser.add_argument(
        "--dino-features",
        type=Path,
        default=Path("/home/unitree/v70_dino_live_shadow.npz"),
    )
    parser.add_argument(
        "--fixed-armd",
        type=Path,
        default=Path("/home/unitree/v70_armd_fixed_pool_shadow.npz"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/project",
    )
    parser.add_argument(
        "--support",
        type=Path,
        default=(
            ROOT
            / "outputs/universal_dualdomain_raw_v2/"
            "training_v2/support/"
            "dualdomain_spacing_envelopes.npz"
        ),
    )
    parser.add_argument(
        "--v70-result",
        type=Path,
        default=(
            ROOT
            / "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v70_broad_hashfold_archive/"
            "result.json"
        ),
    )
    parser.add_argument(
        "--armd-checkpoint-dir",
        type=Path,
        default=(
            ROOT
            / "outputs/cem_development_v1/"
            "dualdomain_cem_tail_v69_scand_route_crossfit/"
            "training/checkpoints"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_cem_archive_shadow.npz"),
    )
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name].copy() for name in payload.files}


def require_shape(
    name: str,
    value: np.ndarray,
    expected: tuple[int, ...],
) -> None:
    if value.shape != expected:
        raise RuntimeError(
            f"{name}: expected shape {expected}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def load_live_context(
    online_path: Path,
    dino_path: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[PoseSample],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    online = load_npz(online_path)
    dino = load_npz(dino_path)

    required_online = {
        "nomad_x16_waypoint_units",
        "candidates_waypoint_units",
        "pose_samples",
    }
    required_dino = {
        "history_latent",
        "goal_latent",
        "static_features",
    }

    missing_online = required_online.difference(online)
    missing_dino = required_dino.difference(dino)
    if missing_online:
        raise RuntimeError(
            f"online artifact missing: {sorted(missing_online)}"
        )
    if missing_dino:
        raise RuntimeError(
            f"DINO artifact missing: {sorted(missing_dino)}"
        )

    reference = online["nomad_x16_waypoint_units"].astype(np.float32)[None]
    base_candidates = online["candidates_waypoint_units"].astype(np.float32)[None]
    pose_array = online["pose_samples"].astype(np.float64)

    history = dino["history_latent"].astype(np.float32)
    goal = dino["goal_latent"].astype(np.float32)
    state_static = dino["static_features"].astype(np.float32)

    require_shape("reference", reference, (1, 16, 4, 2))
    require_shape("base_candidates", base_candidates, (1, 17, 4, 2))
    require_shape("pose_samples", pose_array, (4, 4))
    require_shape("history", history, (1, 4, 256, 384))
    require_shape("goal", goal, (1, 256, 384))
    require_shape("state_static", state_static, (1, 210))

    poses = [
        PoseSample(
            receive_time=float(row[0]),
            x=float(row[1]),
            y=float(row[2]),
            yaw=float(row[3]),
        )
        for row in pose_array
    ]
    return (
        reference,
        base_candidates,
        poses,
        history,
        goal,
        state_static,
    )


def build_action_inputs(
    candidates: np.ndarray,
    reference: np.ndarray,
    poses: list[PoseSample],
    state_static: np.ndarray,
) -> dict[str, np.ndarray]:
    require_shape("candidate pool", candidates, (1, 17, 4, 2))
    require_shape("reference pool", reference, (1, 16, 4, 2))

    action_set = action_set_features(
        candidates,
        reference,
    ).astype(np.float32)
    require_shape("action_set", action_set, (1, 17, 83))

    history_features, _ = build_history_features(
        candidates[0],
        reference[0],
        poses,
    )
    history_features = history_features[None].astype(np.float32)
    require_shape("history_features", history_features, (1, 17, 38))

    raw = np.concatenate(
        (action_set, history_features),
        axis=-1,
    ).astype(np.float32)
    require_shape("raw action/history", raw, (1, 17, 121))

    action_features = np.concatenate(
        (raw, raw - raw[:, :1]),
        axis=-1,
    ).astype(np.float32)
    require_shape("action_features", action_features, (1, 17, 242))

    actions_h4 = nomad_waypoints_to_local_actions(
        candidates,
        dataset="recon",
    ).astype(np.float32)
    require_shape("actions_h4", actions_h4, (1, 17, 4, 3))

    static_features = np.repeat(
        state_static[:, None, :],
        17,
        axis=1,
    ).astype(np.float32)
    require_shape("static_features", static_features, (1, 17, 210))

    return {
        "candidates": candidates,
        "actions_h4": actions_h4,
        "action_features": action_features,
        "static_features": static_features,
    }


def compute_ltwm_features(
    actions_h4: np.ndarray,
    history_np: np.ndarray,
    goal_np: np.ndarray,
    checkpoint_root: Path,
    device: torch.device,
) -> dict[str, np.ndarray]:
    actions_cpu = torch.from_numpy(actions_h4.astype(np.float32))
    history = torch.from_numpy(history_np).to(device)
    goal = torch.from_numpy(goal_np).to(device)

    scalar_by_seed: list[np.ndarray] = []
    ate_by_seed: list[np.ndarray] = []
    rtp_by_seed: list[np.ndarray] = []
    support_by_seed: list[np.ndarray] = []

    for seed in SEEDS:
        wm_path = (
            checkpoint_root
            / "ltwm_expansion"
            / f"seed{seed}"
            / "best.pt"
        )
        scorer_path = (
            checkpoint_root
            / "scorer_v2"
            / f"seed{seed}"
            / "best.pt"
        )
        if not wm_path.is_file():
            raise FileNotFoundError(wm_path)
        if not scorer_path.is_file():
            raise FileNotFoundError(scorer_path)

        wm_hash = file_sha256(wm_path)
        wm_checkpoint = torch.load(
            wm_path,
            map_location="cpu",
            weights_only=False,
        )
        scorer_checkpoint = torch.load(
            scorer_path,
            map_location="cpu",
            weights_only=False,
        )

        paired_hash = scorer_checkpoint.get(
            "world_model_checkpoint_sha256"
        )
        if paired_hash != wm_hash:
            raise RuntimeError(
                f"seed {seed}: scorer/WM checkpoint hash mismatch"
            )

        wm = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**wm_checkpoint["config"])
        )
        wm.load_state_dict(wm_checkpoint["model"], strict=True)
        wm = wm.eval().to(device).requires_grad_(False)

        scorer = TrajectoryQualityScorerV2(
            TrajectoryQualityScorerConfig(
                **scorer_checkpoint["model_config"]
            )
        )
        scorer.load_state_dict(
            scorer_checkpoint["model"],
            strict=True,
        )
        scorer = scorer.eval().to(device).requires_grad_(False)

        support = evaluate_trajectory_constraints(
            actions_cpu,
            support_stats(scorer_checkpoint["action_support"]),
            parent_actions=actions_cpu,
        ).valid.cpu().numpy()
        support[:, 0] = True
        support_by_seed.append(support.astype(bool))

        actions = actions_cpu.to(device)
        autocast = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            )
            if device.type == "cuda"
            else torch.autocast(
                device_type="cpu",
                enabled=False,
            )
        )

        with torch.inference_mode(), autocast:
            future = wm(
                history,
                actions,
            )["predicted_tokens"]

        future = future.float()

        with torch.inference_mode():
            scalar, _ = latent_trajectory_features(
                future,
                goal,
                history[:, -1],
            )
            prediction = scorer(
                current_latent=history[:, -1],
                goal_latent=goal,
                candidate_actions=actions,
                predicted_future=future,
            )

        scalar_np = scalar.cpu().numpy().astype(np.float32)
        ate_np = prediction.ate.cpu().numpy().astype(np.float32)
        rtp_np = prediction.rtp.cpu().numpy().astype(np.float32)

        require_shape(f"seed {seed} scalar", scalar_np, (1, 17, 28))
        require_shape(f"seed {seed} ATE", ate_np, (1, 17))
        require_shape(f"seed {seed} RTP", rtp_np, (1, 17))

        scalar_by_seed.append(scalar_np)
        ate_by_seed.append(ate_np)
        rtp_by_seed.append(rtp_np)

        del (
            wm,
            scorer,
            wm_checkpoint,
            scorer_checkpoint,
            actions,
            future,
            scalar,
            prediction,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scalar_seed = np.stack(scalar_by_seed, axis=0)
    ate_seed = np.stack(ate_by_seed, axis=0)
    rtp_seed = np.stack(rtp_by_seed, axis=0)
    support_seed = np.stack(support_by_seed, axis=0)

    raw_ltwm = np.concatenate(
        (
            scalar_seed.mean(axis=0),
            scalar_seed.std(axis=0),
            scalar_seed.min(axis=0),
            scalar_seed.max(axis=0),
        ),
        axis=-1,
    ).astype(np.float32)
    require_shape("raw_ltwm_features", raw_ltwm, (1, 17, 112))

    scorer_aux = np.concatenate(
        (
            np.moveaxis(ate_seed, 0, -1),
            np.moveaxis(rtp_seed, 0, -1),
            ate_seed.mean(axis=0)[..., None],
            ate_seed.std(axis=0)[..., None],
            rtp_seed.mean(axis=0)[..., None],
            rtp_seed.std(axis=0)[..., None],
        ),
        axis=-1,
    ).astype(np.float32)
    require_shape("scorer_v2_aux", scorer_aux, (1, 17, 10))

    majority_support = support_seed.sum(axis=0) >= 2
    majority_support[:, 0] = True

    return {
        "raw_ltwm_features": raw_ltwm,
        "scorer_v2_aux_features": scorer_aux,
        "support_by_seed": support_seed,
        "majority_support": majority_support,
    }


def evaluate_pool(
    *,
    candidates: np.ndarray,
    reference: np.ndarray,
    poses: list[PoseSample],
    history: np.ndarray,
    goal: np.ndarray,
    state_static: np.ndarray,
    checkpoint_root: Path,
    envelope: dict[str, Any],
    threshold: float,
    checkpoint_records: list[dict],
    policy: dict[str, float],
    dual_temperature: float,
    harm_temperature: float,
    device: torch.device,
) -> dict[str, np.ndarray]:
    action_data = build_action_inputs(
        candidates,
        reference,
        poses,
        state_static,
    )

    ltwm_data = compute_ltwm_features(
        action_data["actions_h4"],
        history,
        goal,
        checkpoint_root,
        device,
    )

    model_data = {
        "action_features": action_data["action_features"],
        "actions_h4": action_data["actions_h4"],
        "static_features": action_data["static_features"],
        "raw_ltwm_features": ltwm_data["raw_ltwm_features"],
    }

    universal, support_score, physical_guard = build_universal_features(
        model_data,
        spacing_m=SPACING_M,
        support_envelope=envelope,
    )
    require_shape("universal", universal, (1, 17, 815))
    require_shape("support_score", support_score, (1, 17))
    require_shape(
        "physical_guard",
        physical_guard.astype(np.float32),
        (1, 17),
    )

    valid = physical_guard & (support_score <= threshold)
    valid[:, 0] = True

    prediction = v70.compatible_ensemble_predict(
        checkpoint_records,
        universal,
        np.asarray([DOMAIN_ID], dtype=np.int64),
        device=device.type,
    )

    energy, p_dual, p_harm = replay.conservative_energy(
        prediction,
        policy=policy,
        dual_temperature=dual_temperature,
        harm_temperature=harm_temperature,
        valid=valid,
    )
    safe = replay.candidate_safety_mask(
        prediction,
        p_dual,
        p_harm,
        valid,
        policy,
    )

    return {
        **action_data,
        **ltwm_data,
        "universal_features": universal,
        "support_score": support_score,
        "physical_guard": physical_guard,
        "valid": valid,
        "prediction_mean": prediction["mean"],
        "prediction_std": prediction["std"],
        "dual_probability": p_dual,
        "harm_probability": p_harm,
        "rank_logit": prediction["rank_logit"],
        "energy": energy,
        "safe": safe,
    }


def update_archive(
    *,
    round_index: int,
    candidates: np.ndarray,
    energy: np.ndarray,
    safe: np.ndarray,
    archive: dict[str, Any],
) -> None:
    selected = replay.select_lowest_energy_safe(
        energy,
        safe,
    )
    index = int(selected[0])
    if index == 0:
        return

    value = float(energy[0, index])
    if not np.isfinite(value):
        return

    if value < float(archive["energy"]):
        archive["energy"] = value
        archive["candidate"] = candidates[0, index].copy()
        archive["source_round"] = int(round_index)
        archive["source_candidate"] = int(index)


def main() -> None:
    args = parse_args()

    print("===== V70 3-round archive-CEM shadow =====")
    print("No ROS publisher is created.")
    print("No SportClient command is issued.")
    print()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)

    (
        reference,
        base_candidates,
        poses,
        history,
        goal,
        state_static,
    ) = load_live_context(
        args.online_features,
        args.dino_features,
    )
    incumbent = base_candidates[:, 0].copy()

    fixed = load_npz(args.fixed_armd)
    required_fixed = {
        "candidates_waypoint_units",
        "energy",
        "safe",
        "valid",
    }
    missing_fixed = required_fixed.difference(fixed)
    if missing_fixed:
        raise RuntimeError(
            f"fixed Arm-D artifact missing: {sorted(missing_fixed)}"
        )

    fixed_candidates = fixed["candidates_waypoint_units"].astype(np.float32)
    fixed_energy = fixed["energy"].astype(np.float32)
    fixed_safe = fixed["safe"].astype(bool)
    fixed_valid = fixed["valid"].astype(bool)

    require_shape("fixed candidates", fixed_candidates, (1, 17, 4, 2))
    # +inf is intentional for the incumbent and invalid candidates.
    # Reject only shape errors, NaN, and negative infinity.
    if fixed_energy.shape != (1, 17):
        raise RuntimeError(
            f"fixed energy: expected shape (1, 17), "
            f"got {fixed_energy.shape}"
        )
    if np.isnan(fixed_energy).any():
        raise RuntimeError("fixed energy contains NaN")
    if np.isneginf(fixed_energy).any():
        raise RuntimeError("fixed energy contains -Inf")
    if not np.isposinf(fixed_energy[0, 0]):
        raise RuntimeError(
            "candidate 0 must have +Inf energy because it is "
            "the incumbent fallback"
        )
    require_shape(
        "fixed safe",
        fixed_safe.astype(np.float32),
        (1, 17),
    )
    require_shape(
        "fixed valid",
        fixed_valid.astype(np.float32),
        (1, 17),
    )

    if not np.allclose(
        fixed_candidates,
        base_candidates,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            "fixed Arm-D artifact does not match current K17 candidate pool"
        )

    envelopes = replay.load_envelopes(args.support)
    envelope = envelopes[DOMAIN]
    threshold = replay.threshold_for_quantile(
        envelope,
        SUPPORT_QUANTILE,
    )
    if not np.isclose(
        threshold,
        EXPECTED_THRESHOLD,
        atol=1.0e-6,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"support threshold changed: {threshold}"
        )

    v70_result = json.loads(
        args.v70_result.read_text(encoding="utf-8")
    )
    calibration = v70_result["calibration"][DOMAIN]
    if not bool(calibration.get("feasible")):
        raise RuntimeError("v70 RECON calibration is infeasible")

    policy = dict(calibration["policy"])
    dual_temperature = float(calibration["dual_temperature"])
    harm_temperature = float(calibration["harm_temperature"])

    checkpoint_records = localize_checkpoint_records(
        list(v70_result["checkpoints"]["full"]),
        args.armd_checkpoint_dir,
    )

    archive: dict[str, Any] = {
        "energy": np.inf,
        "candidate": incumbent[0].copy(),
        "source_round": -2,
        "source_candidate": 0,
    }

    # Formal base fixed-pool archive/update: source round -1.
    update_archive(
        round_index=-1,
        candidates=base_candidates,
        energy=fixed_energy,
        safe=fixed_safe,
        archive=archive,
    )

    mean, covariance = replay.initial_proposal(
        reference,
        floor=COVARIANCE_FLOOR,
        scale=PROPOSAL_SCALE,
    )
    (
        mean,
        covariance,
        elite_index,
        updated,
        safe_count,
    ) = replay.safe_elite_update(
        base_candidates,
        fixed_energy,
        fixed_safe,
        previous_mean=mean,
        previous_covariance=covariance,
        elite_count=ELITE_COUNT,
        floor=COVARIANCE_FLOOR,
        scale=PROPOSAL_SCALE,
    )

    print("base fixed pool:")
    print("  valid challengers:", int(fixed_valid[:, 1:].sum()), "/ 16")
    print("  safe challengers:", int(fixed_safe[:, 1:].sum()), "/ 16")
    print("  proposal updated:", bool(updated[0]))
    print("  safe count:", int(safe_count[0]))
    print("  archive source round:", int(archive["source_round"]))
    print("  archive energy:", float(archive["energy"]))
    print()

    rng = np.random.default_rng(RNG_SEED)
    rounds: list[dict[str, Any]] = []

    for iteration in range(ITERATIONS):
        print(f"===== CEM iteration {iteration} =====")
        started = time.monotonic()

        candidates, sampling_audit = replay.sample_population(
            incumbent=incumbent,
            reference=reference,
            mean=mean,
            covariance=covariance,
            population=POPULATION,
            rng=rng,
            support_envelope=envelope,
            support_threshold=threshold,
            spacing_m=SPACING_M,
        )
        require_shape(
            f"iteration {iteration} candidates",
            candidates,
            (1, 17, 4, 2),
        )

        evaluation = evaluate_pool(
            candidates=candidates,
            reference=reference,
            poses=poses,
            history=history,
            goal=goal,
            state_static=state_static,
            checkpoint_root=args.checkpoint_root,
            envelope=envelope,
            threshold=threshold,
            checkpoint_records=checkpoint_records,
            policy=policy,
            dual_temperature=dual_temperature,
            harm_temperature=harm_temperature,
            device=device,
        )

        update_archive(
            round_index=iteration,
            candidates=candidates,
            energy=evaluation["energy"],
            safe=evaluation["safe"],
            archive=archive,
        )

        (
            next_mean,
            next_covariance,
            next_elite,
            next_updated,
            next_safe_count,
        ) = replay.safe_elite_update(
            candidates,
            evaluation["energy"],
            evaluation["safe"],
            previous_mean=mean,
            previous_covariance=covariance,
            elite_count=ELITE_COUNT,
            floor=COVARIANCE_FLOOR,
            scale=PROPOSAL_SCALE,
        )

        elapsed = time.monotonic() - started

        print(
            "  sampling accepted:",
            sampling_audit.get("accepted_total", "n/a"),
        )
        print(
            "  sampling fallback:",
            sampling_audit.get("fallback_total", "n/a"),
        )
        print(
            "  valid challengers:",
            int(evaluation["valid"][:, 1:].sum()),
            "/ 16",
        )
        print(
            "  safe challengers:",
            int(evaluation["safe"][:, 1:].sum()),
            "/ 16",
        )
        print("  safe count:", int(next_safe_count[0]))
        print("  proposal updated:", bool(next_updated[0]))
        print("  elite indices:", next_elite[0].tolist())
        print("  archive source round:", int(archive["source_round"]))
        print(
            "  archive source candidate:",
            int(archive["source_candidate"]),
        )
        print("  archive energy:", float(archive["energy"]))
        print("  round seconds:", round(elapsed, 3))
        print()

        rounds.append(
            {
                "iteration": iteration,
                "candidates": candidates,
                "evaluation": evaluation,
                "proposal_mean_before": mean.copy(),
                "proposal_covariance_before": covariance.copy(),
                "elite_index": next_elite.copy(),
                "proposal_updated": next_updated.copy(),
                "safe_count": next_safe_count.copy(),
            }
        )

        mean = next_mean
        covariance = next_covariance
        elite_index = next_elite

    archived = np.isfinite(float(archive["energy"]))
    selected_candidate = (
        np.asarray(archive["candidate"], dtype=np.float32)
        if archived
        else incumbent[0].astype(np.float32)
    )
    selected_source = (
        "fixed_pool"
        if int(archive["source_round"]) == -1
        else (
            f"cem_round_{int(archive['source_round'])}"
            if int(archive["source_round"]) >= 0
            else "incumbent_fallback"
        )
    )

    selected_actions = nomad_waypoints_to_local_actions(
        selected_candidate[None],
        dataset="recon",
    ).astype(np.float32)[0]
    require_shape("selected_actions", selected_actions, (4, 3))

    save_data: dict[str, np.ndarray] = {
        "reference_nomad_x16": reference,
        "incumbent_waypoint_units": incumbent,
        "final_proposal_mean": mean,
        "final_proposal_covariance": covariance,
        "archive_candidate_waypoint_units": selected_candidate,
        "archive_energy": np.asarray(
            float(archive["energy"]),
            dtype=np.float32,
        ),
        "archive_source_round": np.asarray(
            int(archive["source_round"]),
            dtype=np.int64,
        ),
        "archive_source_candidate": np.asarray(
            int(archive["source_candidate"]),
            dtype=np.int64,
        ),
        "selected_actions_h4": selected_actions,
        "selected_first_action": selected_actions[0],
    }

    for record in rounds:
        prefix = f"round_{record['iteration']}"
        evaluation = record["evaluation"]
        save_data[f"{prefix}_candidates"] = record["candidates"]
        save_data[f"{prefix}_energy"] = evaluation["energy"]
        save_data[f"{prefix}_safe"] = evaluation["safe"]
        save_data[f"{prefix}_valid"] = evaluation["valid"]
        save_data[f"{prefix}_support_score"] = evaluation["support_score"]
        save_data[f"{prefix}_physical_guard"] = evaluation["physical_guard"]
        save_data[f"{prefix}_elite_index"] = record["elite_index"]
        save_data[f"{prefix}_proposal_updated"] = record["proposal_updated"]
        save_data[f"{prefix}_safe_count"] = record["safe_count"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save_data)

    print("===== RESULT =====")
    print("iterations:", ITERATIONS)
    print("population per round:", POPULATION, "+ incumbent")
    print("elite count:", ELITE_COUNT)
    print("archive source:", selected_source)
    print(
        "archive source candidate:",
        int(archive["source_candidate"]),
    )
    print("archive energy:", float(archive["energy"]))
    print("selected waypoints:")
    print(
        np.array2string(
            selected_candidate,
            precision=6,
            suppress_small=True,
        )
    )
    print("selected actions H4:")
    print(
        np.array2string(
            selected_actions,
            precision=6,
            suppress_small=True,
        )
    )
    print("selected first action:", selected_actions[0])
    print("saved:", args.output)
    print()
    print("V70 3-ROUND ARCHIVE-CEM SHADOW: PASS")


if __name__ == "__main__":
    main()
