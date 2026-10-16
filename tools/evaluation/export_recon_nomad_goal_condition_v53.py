#!/usr/bin/env python3
"""Replay frozen NoMaD encodings and export state-level goal condition signals.

This exporter deliberately does not run or read any navigation outcome.  It
recovers the pooled observation-to-goal conditioning vector already computed
by the deployed NoMaD forward pass and its native distance-head prediction.
The frozen K17 record, seed, and action identities are audited against the
original progress cache and the consolidated A--T rollout before writing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from planning.nomad_backend import (  # noqa: E402
    add_external_paths,
    load_config,
    load_nomad_model,
    transform_images,
)
from tools.planning.build_scand100_nomad_k17_v1 import (  # noqa: E402
    image_for,
    sha256,
    write_json,
    write_npz,
)


def bank_roots() -> dict[str, Path]:
    return {
        **{
            letter: ROOT
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{letter}"
            for letter in "abcdef"
        },
        "g": ROOT / "outputs/cem_development_v1/recon_target_confirmation_v29",
        "h": ROOT
        / "outputs/cem_development_v1/recon_target_cem_confirmation_v35",
        **{
            letter: ROOT
            / "outputs/cem_development_v1/recon_target_development_v40"
            / f"bank_{letter}"
            for letter in "ijk"
        },
        **{
            letter: ROOT
            / "outputs/cem_development_v1/recon_target_development_v44"
            / f"bank_{letter}"
            for letter in "lmnop"
        },
        **{
            letter: ROOT
            / "outputs/cem_development_v1/recon_target_development_v46"
            / f"bank_{letter}"
            for letter in "qrst"
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states-per-bank", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--trajectory-root",
        type=Path,
        default=Path("/home/rainbow/nwm/data/recon"),
    )
    parser.add_argument(
        "--visualnav-root",
        type=Path,
        default=Path("/home/rainbow/visualnav-transformer"),
    )
    parser.add_argument(
        "--diffusion-policy-root",
        type=Path,
        default=Path("/home/rainbow/diffusion_policy"),
    )
    parser.add_argument(
        "--nomad-config",
        type=Path,
        default=Path("/home/rainbow/visualnav-transformer/train/config/nomad.yaml"),
    )
    parser.add_argument(
        "--nomad-checkpoint",
        type=Path,
        default=Path(
            "/home/rainbow/visualnav-transformer/deployment/model_weights/nomad.pth"
        ),
    )
    parser.add_argument(
        "--legacy-site-packages",
        type=Path,
        default=Path(
            "/home/rainbow/miniconda3/envs/vint_data/lib/python3.8/site-packages"
        ),
    )
    parser.add_argument(
        "--reference-rollout",
        type=Path,
        default=ROOT
        / "outputs/cem_development_v1/recon_target_ltwm_adaptation_v52/rollouts/"
        "hard_t_strong_action_plus_late_blocks_at.npz",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 manifest rows in {path}, got {len(rows)}")
    for row in rows:
        history = list(map(int, row["history_timesteps"]))
        current = int(row["current_t"])
        if history != list(range(current - 3, current + 1)):
            raise RuntimeError(f"history contract changed in {path}")
        if int(row["goal_t"]) != current + 4:
            raise RuntimeError(f"H4 goal contract changed in {path}")
    return rows


def load_identity(
    letter: str,
    root: Path,
    reference_record: np.ndarray,
    reference_actions: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], dict[str, object]]:
    manifest = root / "manifest.windows.jsonl"
    k17 = root / "k17.npz"
    progress = root / "k17.progress.npz"
    rows = read_rows(manifest)
    with np.load(k17, allow_pickle=False) as payload:
        arrays = {
            key: payload[key].copy()
            for key in (
                "record_key",
                "candidate_actions_h4",
                "nomad_x16_waypoint_units",
                "per_window_seed",
                "route_id",
            )
        }
    manifest_record = np.asarray(
        [f"{row['trajectory_id']}:{int(row['current_t'])}" for row in rows]
    )
    if not np.array_equal(arrays["record_key"].astype(str), manifest_record):
        raise RuntimeError(f"bank {letter} manifest/K17 record mismatch")
    if not np.array_equal(arrays["record_key"].astype(str), reference_record):
        raise RuntimeError(f"bank {letter} K17/consolidated record mismatch")
    action_error = float(
        np.max(np.abs(arrays["candidate_actions_h4"] - reference_actions))
    )
    if action_error != 0.0:
        raise RuntimeError(
            f"bank {letter} K17/consolidated action mismatch: {action_error}"
        )
    with np.load(progress, allow_pickle=False) as payload:
        completed = int(payload["completed"].item())
        progress_seed = payload["per_window_seed"].astype(np.int64)
        progress_waypoint = payload["nomad_x16_waypoint_units"].astype(np.float32)
    if completed != 100:
        raise RuntimeError(f"bank {letter} K17 progress incomplete: {completed}")
    if not np.array_equal(arrays["per_window_seed"], progress_seed):
        raise RuntimeError(f"bank {letter} K17/progress seed mismatch")
    waypoint_error = float(
        np.max(np.abs(arrays["nomad_x16_waypoint_units"] - progress_waypoint))
    )
    if waypoint_error != 0.0:
        raise RuntimeError(
            f"bank {letter} K17/progress waypoint mismatch: {waypoint_error}"
        )
    audit = {
        "bank": letter,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "k17": str(k17),
        "k17_sha256": sha256(k17),
        "progress": str(progress),
        "progress_sha256": sha256(progress),
        "record_identity": True,
        "seed_identity": True,
        "action_identity": True,
        "max_action_identity_error": action_error,
        "max_waypoint_identity_error": waypoint_error,
    }
    return rows, arrays, audit


@torch.inference_mode()
def encode_batch(
    model: torch.nn.Module,
    observations: torch.Tensor,
    goals: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    observations = observations.to(device)
    goals = goals.to(device)
    goal_mask = torch.zeros(
        observations.shape[0], dtype=torch.long, device=device
    )
    condition = model(
        "vision_encoder",
        obs_img=observations,
        goal_img=goals,
        input_goal_mask=goal_mask,
    )
    distance = model("dist_pred_net", obsgoal_cond=condition)
    return (
        condition.float().cpu().numpy().astype(np.float32),
        distance.reshape(-1).float().cpu().numpy().astype(np.float32),
    )


def main() -> None:
    args = parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    with np.load(args.reference_rollout, allow_pickle=False) as payload:
        reference_record = payload["record_key"].astype(str)
        reference_bank = payload["bank"].astype(str)
        reference_actions = payload["actions_h4"].astype(np.float32)
    if reference_record.shape != (2000,) or reference_actions.shape != (
        2000,
        17,
        4,
        3,
    ):
        raise RuntimeError("consolidated A--T rollout contract changed")

    legacy = args.legacy_site_packages.resolve(strict=True)
    if str(legacy) not in sys.path:
        sys.path.append(str(legacy))
    add_external_paths(args.visualnav_root, args.diffusion_policy_root)
    config = load_config(args.nomad_config)
    device = torch.device(args.device)
    model, load_report = load_nomad_model(args.nomad_checkpoint, config, device)
    image_root = args.trajectory_root.resolve(strict=True)

    records, banks, routes, seeds, conditions, distances = [], [], [], [], [], []
    bank_audits = []
    for bank_index, (letter, root) in enumerate(sorted(bank_roots().items())):
        local = slice(bank_index * 100, (bank_index + 1) * 100)
        if not np.all(reference_bank[local] == letter):
            raise RuntimeError(f"consolidated bank order changed at {letter}")
        rows, arrays, identity = load_identity(
            letter,
            root,
            reference_record[local],
            reference_actions[local],
        )
        count = 100
        if args.max_states_per_bank is not None:
            count = min(count, args.max_states_per_bank)
        local_conditions, local_distances = [], []
        for start in range(0, count, args.batch_size):
            batch_rows = rows[start : min(start + args.batch_size, count)]
            observation = torch.cat(
                [
                    transform_images(
                        [
                            image_for(
                                image_root,
                                str(row["trajectory_id"]),
                                int(timestep),
                            )
                            for timestep in row["history_timesteps"]
                        ],
                        list(config["image_size"]),
                    )
                    for row in batch_rows
                ],
                dim=0,
            )
            goal = torch.cat(
                [
                    transform_images(
                        [
                            image_for(
                                image_root,
                                str(row["trajectory_id"]),
                                int(row["goal_t"]),
                            )
                        ],
                        list(config["image_size"]),
                    )
                    for row in batch_rows
                ],
                dim=0,
            )
            condition, distance = encode_batch(model, observation, goal, device)
            local_conditions.append(condition)
            local_distances.append(distance)
        local_condition = np.concatenate(local_conditions)
        local_distance = np.concatenate(local_distances)
        expected_dim = int(config["encoding_size"])
        if local_condition.shape != (count, expected_dim):
            raise RuntimeError(
                f"bank {letter} condition shape changed: {local_condition.shape}"
            )
        if local_distance.shape != (count,):
            raise RuntimeError(
                f"bank {letter} distance shape changed: {local_distance.shape}"
            )
        if not np.isfinite(local_condition).all() or not np.isfinite(
            local_distance
        ).all():
            raise RuntimeError(f"bank {letter} non-finite NoMaD goal signal")
        records.append(arrays["record_key"][:count].astype(str))
        banks.append(np.full(count, letter))
        routes.append(arrays["route_id"][:count].astype(str))
        seeds.append(arrays["per_window_seed"][:count].astype(np.int64))
        conditions.append(local_condition)
        distances.append(local_distance)
        identity["states_exported"] = count
        bank_audits.append(identity)
        print(
            json.dumps(
                {
                    "stage": "nomad_goal_condition_v53",
                    "bank": letter,
                    "states": count,
                }
            ),
            flush=True,
        )

    value = np.concatenate(conditions).astype(np.float32)
    distance = np.concatenate(distances).astype(np.float32)
    write_npz(
        args.output,
        record_key=np.concatenate(records),
        bank=np.concatenate(banks),
        route_id=np.concatenate(routes),
        per_window_seed=np.concatenate(seeds),
        nomad_goal_condition=value,
        nomad_goal_distance_prediction=distance,
    )
    audit = {
        "schema": "recon_nomad_goal_condition_v53",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "outcome_free_native_nomad_goal_signal_export_complete",
        "states": int(len(value)),
        "banks": len(bank_audits),
        "condition_shape": list(value.shape),
        "distance_shape": list(distance.shape),
        "all_finite": True,
        "feature_contract": (
            "frozen NoMaD pooled observation-to-goal condition embedding plus "
            "the checkpoint's native DenseNetwork distance-head prediction"
        ),
        "input_time": "history t-3..t and goal image t+4",
        "online_deployable": True,
        "native_uncertainty_available": False,
        "diffusion_90d_reused": False,
        "outcomes_accessed": False,
        "labels_in_output": False,
        "candidate_ranker_run_or_trained": False,
        "reference_rollout": str(args.reference_rollout),
        "reference_rollout_sha256": sha256(args.reference_rollout),
        "nomad_checkpoint": str(args.nomad_checkpoint),
        "nomad_checkpoint_sha256": sha256(args.nomad_checkpoint),
        "nomad_config": str(args.nomad_config),
        "nomad_config_sha256": sha256(args.nomad_config),
        "load_report": load_report,
        "identity": bank_audits,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "final_confirmation_reserve_accessed": False,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
