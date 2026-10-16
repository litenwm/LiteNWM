#!/usr/bin/env python3
"""Build old-V70-compatible K17 fixed-pool artifacts for three-domain training.

The learned feature contract is exactly the proposal-aware V70 contract:

* 242-D incumbent/challenger action and past-motion features;
* 210-D candidate-independent DINOv3 route/history features;
* 112-D three-seed LTWM rollout statistics (28-D per seed summarized by
  mean/std/min/max);
* the unchanged 12-D spacing/support diagnostics consumed by
  ``build_universal_features``.

The generic absolute ATE/RTP scorer is deliberately excluded.  Existing
K16 caches are used only to recover the exact per-seed 28-D LTWM scalar block.
The trimmed-mean incumbent is new, so its LTWM features are evaluated once per
window and per world-model seed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    integrate_local_actions,
    nomad_waypoints_to_local_actions,
)
from recon_latents.models_v2 import (  # noqa: E402
    LiteTokenFormerConfig,
    LiteTokenFormerWMV2,
)
from scoring.spacing_support_v1 import (  # noqa: E402
    fit_spacing_envelope,
    support_mask,
)
from scoring.trajectory_artifact_contract_v2 import (  # noqa: E402
    local_future_positions,
)
from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    action_set_features,
    history_features,
    metric_labels,
    trimmed_mean75,
)
from tools.evaluation.export_recon_dense_dinov3_features_v1 import (  # noqa: E402
    pool_static_features,
)
from tools.scoring.build_trajectory_quality_labels_v2 import (  # noqa: E402
    load_pose,
)
from tools.scoring.train_universal_dualdomain_raw_v2 import (  # noqa: E402
    save_support_bundle,
)


HISTORY = 4
HORIZON = 4
CHALLENGERS = 16
CANDIDATES = 17
SCALAR_START = 3 * HORIZON * 64
SCALAR_STOP = SCALAR_START + 7 * HORIZON
MODEL_SEEDS = (1701, 1702, 1703)
DOMAIN_SPACING = {"recon": 0.25, "scand": 0.38, "sacson": 0.255}
MODEL_DOMAIN = {"recon": "recon", "scand": "scand", "sacson": "sacson"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DOMAIN_SPACING), required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--train-proposals", type=Path, required=True)
    parser.add_argument("--calibration-proposals", type=Path, required=True)
    for seed in MODEL_SEEDS:
        parser.add_argument(f"--wm-{seed}", type=Path, required=True)
        parser.add_argument(f"--train-cache-{seed}", type=Path, required=True)
        parser.add_argument(
            f"--calibration-cache-{seed}", type=Path, required=True
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--support-quantile", type=float, default=0.999)
    parser.add_argument("--rtp-minimum-distance", type=float, default=0.05)
    parser.add_argument("--rtp-distance-quantile", type=float, default=0.01)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
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
        np.savez(handle, **arrays)
    temporary.replace(path)


def load_npz(path: Path, keys: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        selected = archive.files if keys is None else keys
        missing = set(selected).difference(archive.files)
        if missing:
            raise RuntimeError(f"{path} is missing arrays: {sorted(missing)}")
        return {key: archive[key].copy() for key in selected}


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = parsed.get("records") if isinstance(parsed, dict) else parsed
        if not isinstance(rows, list):
            raise RuntimeError(f"manifest has no record list: {path}")
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def validate_rows(
    rows: list[dict[str, Any]],
    proposal: dict[str, np.ndarray],
    *,
    dataset: str,
    partition: str,
) -> None:
    count = len(rows)
    if proposal["actions_h4"].shape != (count, CHALLENGERS, HORIZON, 3):
        raise RuntimeError("proposal action shape is not [N,16,4,3]")
    expected_uid = np.asarray([str(row["window_uid"]) for row in rows])
    expected_trajectory = np.asarray([str(row["trajectory_id"]) for row in rows])
    expected_current = np.asarray(
        [int(row["current_t"]) for row in rows], dtype=np.int64
    )
    for key, expected in (
        ("window_uid", expected_uid),
        ("trajectory_id", expected_trajectory),
        ("current_t", expected_current),
    ):
        actual = proposal[key]
        if actual.dtype.kind in {"U", "S"}:
            actual = actual.astype(str)
        if not np.array_equal(actual, expected):
            raise RuntimeError(f"{dataset}/{partition}: proposal {key} mismatch")
    if "dataset" in proposal and str(proposal["dataset"].item()) != dataset:
        raise RuntimeError("proposal dataset mismatch")
    if "partition" in proposal and str(proposal["partition"].item()) != partition:
        raise RuntimeError("proposal partition mismatch")


def k17_actions_and_waypoints(
    challenger_actions: np.ndarray,
    *,
    dataset: str,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    integrated = integrate_local_actions(challenger_actions).astype(np.float32)
    challenger_native = integrated[..., :2] / np.float32(spacing_m)
    incumbent_native = np.stack(
        [trimmed_mean75(pool) for pool in challenger_native]
    ).astype(np.float32)
    candidate_native = np.concatenate(
        (incumbent_native[:, None], challenger_native), axis=1
    )
    incumbent_actions = nomad_waypoints_to_local_actions(
        incumbent_native[:, None], dataset=dataset
    ).astype(np.float32)
    candidate_actions = np.concatenate(
        (incumbent_actions, challenger_actions.astype(np.float32)), axis=1
    )
    reconstructed = integrate_local_actions(candidate_actions)[..., :2]
    expected = candidate_native * np.float32(spacing_m)
    if not np.allclose(reconstructed, expected, rtol=2.0e-5, atol=2.0e-5):
        raise RuntimeError("K17 waypoint/action round-trip failed")
    return candidate_native, candidate_actions, incumbent_actions


class ShardedLatents:
    def __init__(self, root: Path) -> None:
        self.root = root
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        self.shards = {
            str(row["trajectory_id"]): str(row["shard_file"])
            for row in metadata["trajectories"]
        }
        self.cache_id: str | None = None
        self.cache_value: np.ndarray | None = None

    def batch(
        self, rows: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        history = []
        goal = []
        for row in rows:
            trajectory = str(row["trajectory_id"])
            if trajectory != self.cache_id:
                with np.load(
                    self.root / "shards" / self.shards[trajectory],
                    allow_pickle=False,
                ) as shard:
                    self.cache_value = shard["patch_latents"].copy()
                self.cache_id = trajectory
            assert self.cache_value is not None
            times = [int(value) for value in row["history_timesteps"]]
            history.append(
                self.cache_value[times]
                .astype(np.float32)
                .reshape(HISTORY, 256, 384)
            )
            goal.append(
                self.cache_value[int(row["goal_t"])]
                .astype(np.float32)
                .reshape(256, 384)
            )
        return np.stack(history), np.stack(goal)


def load_world_models(
    checkpoints: dict[int, Path], device: str
) -> dict[int, LiteTokenFormerWMV2]:
    result = {}
    for seed, path in checkpoints.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("temporal_protocol") != {
            "history_frames": HISTORY,
            "future_frames": HORIZON,
        }:
            raise RuntimeError(f"world-model temporal protocol mismatch: {path}")
        model = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**payload["config"])
        ).to(device)
        model.load_state_dict(payload["model"], strict=True)
        result[seed] = model.eval().requires_grad_(False)
    return result


def latent_features(
    rows: list[dict[str, Any]],
    incumbent_actions: np.ndarray,
    *,
    latent_root: Path,
    models: dict[int, LiteTokenFormerWMV2],
    device: str,
    batch_size: int,
    stage: str,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    loader = ShardedLatents(latent_root)
    static_parts: list[np.ndarray] = []
    incumbent_parts: dict[int, list[np.ndarray]] = {
        seed: [] for seed in MODEL_SEEDS
    }
    started = time.perf_counter()
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        history_np, goal_np = loader.batch(rows[start:end])
        history = torch.from_numpy(history_np).to(device)
        goal = torch.from_numpy(goal_np).to(device)
        actions = torch.from_numpy(incumbent_actions[start:end]).to(device)
        with torch.inference_mode():
            static_parts.append(
                pool_static_features(history, goal).cpu().numpy().astype(np.float32)
            )
            for seed in MODEL_SEEDS:
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device == "cuda"
                    else torch.autocast("cpu", enabled=False)
                )
                with context:
                    future = models[seed](history, actions)["predicted_tokens"]
                scalar, _ = latent_trajectory_features(
                    future.float(), goal, history[:, -1]
                )
                if scalar.shape[-1] != 28:
                    raise RuntimeError("LTWM scalar feature dimension changed")
                incumbent_parts[seed].append(
                    scalar[:, 0].cpu().numpy().astype(np.float32)
                )
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
    return (
        np.concatenate(static_parts),
        {
            seed: np.concatenate(incumbent_parts[seed])
            for seed in MODEL_SEEDS
        },
    )


def challenger_scalars(
    cache_paths: dict[int, Path],
    expected_uid: np.ndarray,
) -> dict[int, np.ndarray]:
    result = {}
    for seed, path in cache_paths.items():
        data = load_npz(path, ("window_uid", "features"))
        if not np.array_equal(data["window_uid"].astype(str), expected_uid):
            raise RuntimeError(f"cache UID mismatch: {path}")
        features = data["features"]
        if features.shape[:2] != (len(expected_uid), CHALLENGERS):
            raise RuntimeError(f"cache candidate shape mismatch: {path}")
        if features.shape[-1] < SCALAR_STOP:
            raise RuntimeError(f"cache feature dimension mismatch: {path}")
        result[seed] = (
            features[..., SCALAR_START:SCALAR_STOP].astype(np.float32)
        )
    return result


def aggregate_ltwm(
    incumbent: dict[int, np.ndarray],
    challenger: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    per_seed = []
    for seed in MODEL_SEEDS:
        value = np.concatenate(
            (incumbent[seed][:, None], challenger[seed]), axis=1
        ).astype(np.float32)
        if value.shape[-1] != 28:
            raise RuntimeError("per-seed LTWM scalar width changed")
        per_seed.append(value)
    stack = np.stack(per_seed)
    aggregate = np.concatenate(
        (stack.mean(0), stack.std(0), stack.min(0), stack.max(0)), axis=-1
    ).astype(np.float32)
    if aggregate.shape[-1] != 112:
        raise RuntimeError("three-seed LTWM aggregate width changed")
    return aggregate, stack


def ground_truth(
    rows: list[dict[str, Any]], trajectory_root: Path
) -> np.ndarray:
    pose_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    result = []
    resolved = trajectory_root.resolve(strict=True)
    for row in rows:
        trajectory = str(row["trajectory_id"])
        if trajectory not in pose_cache:
            path = (resolved / trajectory / "traj_data.pkl").resolve(strict=True)
            path.relative_to(resolved)
            pose_cache[trajectory] = load_pose(path)
        positions, yaw = pose_cache[trajectory]
        result.append(
            local_future_positions(
                positions,
                yaw,
                current_t=int(row["current_t"]),
                horizon=HORIZON,
            )
        )
    return np.stack(result).astype(np.float32)


def rtp_labels(
    candidate_actions: np.ndarray,
    truth: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = integrate_local_actions(candidate_actions)[..., :2].astype(np.float64)
    truth64 = truth.astype(np.float64)
    initial_distance = np.linalg.norm(truth64[:, -1], axis=-1)
    final_distance = np.linalg.norm(
        path[:, :, -1] - truth64[:, None, -1], axis=-1
    )
    valid = initial_distance >= threshold
    value = np.full(final_distance.shape, np.nan, dtype=np.float32)
    value[valid] = (
        (
            initial_distance[valid, None] - final_distance[valid]
        )
        / initial_distance[valid, None]
    ).astype(np.float32)
    return value, valid, initial_distance.astype(np.float32)


def robust_scale(value: np.ndarray) -> dict[str, float]:
    finite = np.asarray(value, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise RuntimeError("no finite values for robust scale")
    median = float(np.median(finite))
    mad_scale = float(1.4826 * np.median(np.abs(finite - median)))
    q25, q75 = np.quantile(finite, (0.25, 0.75))
    iqr_scale = float((q75 - q25) / 1.349)
    scale = max(mad_scale, iqr_scale, 1.0e-3)
    return {
        "median": median,
        "mad_scale": mad_scale,
        "iqr_scale": iqr_scale,
        "robust_scale": scale,
    }


def family_for(row: dict[str, Any], dataset: str) -> str:
    for key in ("route_id", "route_group", "bag_id"):
        if key in row and str(row[key]):
            return f"{dataset}::{row[key]}"
    return f"{dataset}::{row['trajectory_id']}"


def build_partition(
    *,
    dataset: str,
    partition: str,
    rows: list[dict[str, Any]],
    proposal_path: Path,
    cache_paths: dict[int, Path],
    trajectory_root: Path,
    latent_root: Path,
    models: dict[int, LiteTokenFormerWMV2],
    output_dir: Path,
    spacing_m: float,
    rtp_threshold: float,
    support_envelope: dict[str, Any] | None,
    support_quantile: float,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    proposal = load_npz(
        proposal_path,
        (
            "window_uid",
            "trajectory_id",
            "current_t",
            "actions_h4",
            "dataset",
            "partition",
        ),
    )
    validate_rows(
        rows, proposal, dataset=dataset, partition=partition
    )
    uid = proposal["window_uid"].astype(str)
    native, actions_h4, incumbent_actions = k17_actions_and_waypoints(
        proposal["actions_h4"], dataset=dataset, spacing_m=spacing_m
    )
    static, incumbent_scalar = latent_features(
        rows,
        incumbent_actions,
        latent_root=latent_root,
        models=models,
        device=device,
        batch_size=batch_size,
        stage=f"proposal_aware_incumbent_{dataset}_{partition}",
    )
    challenger_scalar = challenger_scalars(cache_paths, uid)
    raw_ltwm, raw_ltwm_by_seed = aggregate_ltwm(
        incumbent_scalar, challenger_scalar
    )

    bridge = np.float32(spacing_m / 0.25)
    candidates = native * bridge
    reference = candidates[:, 1:]
    legacy_rows = [
        {
            "traj_name": str(row["trajectory_id"]),
            "curr_time": int(row["current_t"]),
        }
        for row in rows
    ]
    action_set = action_set_features(candidates, reference)
    motion = history_features(
        candidates, reference, legacy_rows, trajectory_root
    )
    raw_action = np.concatenate((action_set, motion), axis=-1).astype(np.float32)
    action_features = np.concatenate(
        (raw_action, raw_action - raw_action[:, :1]), axis=-1
    ).astype(np.float32)
    if action_features.shape != (len(rows), CANDIDATES, 242):
        raise RuntimeError("old V70 action feature contract changed")

    truth = ground_truth(rows, trajectory_root)
    labels = metric_labels(candidates, truth, 0.25, 0.4, 0.6)
    rtp, rtp_valid, initial_distance = rtp_labels(
        actions_h4, truth, threshold=rtp_threshold
    )
    record_key = np.asarray(
        [
            f"{row['trajectory_id']}:{int(row['current_t'])}"
            for row in rows
        ]
    )
    family = np.asarray([family_for(row, dataset) for row in rows])
    source = {
        "candidates": candidates.astype(np.float32),
        "actions_h4": actions_h4.astype(np.float32),
        "action_features": action_features,
        "static_features": np.repeat(
            static[:, None], CANDIDATES, axis=1
        ).astype(np.float32),
        "raw_ltwm_features": raw_ltwm,
        "record_key": record_key,
        "family": family,
    }
    if support_envelope is None:
        support_envelope = fit_spacing_envelope(
            actions_h4,
            np.full(len(rows), spacing_m, dtype=np.float32),
        )
    majority_support, support_audit = support_mask(
        support_envelope,
        actions_h4,
        np.full(len(rows), spacing_m, dtype=np.float32),
        quantile=support_quantile,
    )
    support_by_seed = np.repeat(
        majority_support[None], len(MODEL_SEEDS), axis=0
    )
    scorer_aux = np.zeros((len(rows), CANDIDATES, 10), dtype=np.float32)
    zero_seed = np.zeros(
        (len(MODEL_SEEDS), len(rows), CANDIDATES), dtype=np.float32
    )
    metadata = {
        "schema": "recon_dense_dinov3_feature_artifact_v1",
        "created_at": utc_now(),
        "split": f"{dataset}_{partition}_proposal_aware_v70",
        "states": len(rows),
        "candidates": CANDIDATES,
        "candidate_pool": "trimmed_mean75_incumbent_plus_nomad_x16",
        "feature_dimensions": {
            "action": 242,
            "static": 210,
            "raw_ltwm": 112,
            "scorer_v2_aux": 10,
            "support": 1,
        },
        "feature_separation": {
            "raw_ltwm_features": (
                "three-seed LTWM rollout scalar features summarized by "
                "mean/std/min/max"
            ),
            "scorer_v2_aux_features": (
                "zero placeholder; excluded from proposal-aware V70 input"
            ),
            "support_features": (
                "domain train-fit spacing-envelope acceptance; excluded from "
                "the learned feature contract"
            ),
        },
        "labels_accessed": False,
        "labels_in_output": False,
        "ltwm_future_features_recomputed_for_incumbent": True,
        "ltwm_challenger_features_reused_from_bound_cache": True,
        "generic_ate_rtp_scorer_used_as_input": False,
        "fixed_pool_only": True,
        "cem": False,
        "dataset": dataset,
        "model_domain": MODEL_DOMAIN[dataset],
        "waypoint_spacing_m": spacing_m,
        "proposal_sha256": file_sha256(proposal_path),
        "manifest_window_uid_sha256": array_sha256(uid),
        "seeds": [
            {
                "seed": seed,
                "cache_sha256": file_sha256(cache_paths[seed]),
            }
            for seed in MODEL_SEEDS
        ],
    }
    feature_path = output_dir / partition / "base_features.npz"
    write_npz(
        feature_path,
        **source,
        scorer_v2_aux_features=scorer_aux,
        support_features=majority_support[..., None].astype(np.float32),
        ate_by_seed=zero_seed,
        rtp_by_seed=zero_seed.copy(),
        support_by_seed=support_by_seed,
        majority_support=majority_support,
        metadata=np.asarray(canonical_json(metadata)),
    )
    target_path = output_dir / partition / "targets.npz"
    write_npz(
        target_path,
        record_key=record_key,
        family=family,
        targets=labels["targets"],
        absolute_metrics=labels["values"],
        ground_truth_local_m=truth,
        rtp=rtp,
        rtp_valid=rtp_valid,
        start_goal_distance_m=initial_distance,
        metadata=np.asarray(
            canonical_json(
                {
                    "schema": "proposal_aware_v70_tridomain_targets_v1",
                    "dataset": dataset,
                    "partition": partition,
                    "rtp_definition": (
                        "(start_goal_distance-final_goal_distance)/"
                        "start_goal_distance"
                    ),
                    "rtp_validity_threshold_m": rtp_threshold,
                    "rtp_invalid_values_stored_as_nan": True,
                    "rtp_used_by_proposal_aware_model": False,
                    "primary_targets": [
                        "delta_ate",
                        "delta_rpe",
                        "delta_endpoint",
                        "delta_objective",
                        "dual",
                        "harm",
                    ],
                    "external_test_accessed": False,
                }
            )
        ),
    )
    candidate_path = output_dir / partition / "candidate_pool.npz"
    write_npz(
        candidate_path,
        candidates_waypoint_units=native,
        nomad_x16_waypoint_units=native[:, 1:],
        candidate_actions_h4=actions_h4,
        record_key=record_key,
        window_uid=uid,
        trajectory_id=np.asarray(
            [str(row["trajectory_id"]) for row in rows]
        ),
        curr_time=np.asarray(
            [int(row["current_t"]) for row in rows], dtype=np.int64
        ),
    )
    audit = {
        "dataset": dataset,
        "partition": partition,
        "states": len(rows),
        "base_features": str(feature_path),
        "base_features_sha256": file_sha256(feature_path),
        "targets": str(target_path),
        "targets_sha256": file_sha256(target_path),
        "candidate_pool": str(candidate_path),
        "candidate_pool_sha256": file_sha256(candidate_path),
        "support": support_audit,
        "rtp_valid_windows": int(rtp_valid.sum()),
        "rtp_invalid_windows": int((~rtp_valid).sum()),
        "rtp_raw_min_valid": (
            float(np.nanmin(rtp)) if np.isfinite(rtp).any() else None
        ),
        "rtp_raw_max_valid": (
            float(np.nanmax(rtp)) if np.isfinite(rtp).any() else None
        ),
        "generic_scorer_excluded": True,
        "external_test_accessed": False,
    }
    write_json(output_dir / partition / "audit.json", audit)
    return audit, support_envelope, {
        "rtp": rtp,
        "rtp_valid": rtp_valid,
        "start_goal_distance": initial_distance,
        "raw_ltwm_by_seed": raw_ltwm_by_seed,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    spacing_m = DOMAIN_SPACING[args.dataset]
    train_rows = read_rows(args.train_manifest)
    calibration_rows = read_rows(args.calibration_manifest)
    train_proposal = load_npz(
        args.train_proposals,
        (
            "window_uid",
            "trajectory_id",
            "current_t",
            "actions_h4",
            "dataset",
            "partition",
        ),
    )
    validate_rows(
        train_rows,
        train_proposal,
        dataset=args.dataset,
        partition="train-fit",
    )
    truth = ground_truth(train_rows, args.trajectory_root)
    distance = np.linalg.norm(truth[:, -1], axis=-1)
    positive = distance[distance > 0.0]
    if not len(positive):
        raise RuntimeError("train-fit has no positive start-goal distance")
    learned_floor = float(
        np.quantile(
            positive,
            args.rtp_distance_quantile,
            method="higher",
        )
    )
    rtp_threshold = max(args.rtp_minimum_distance, learned_floor)

    checkpoints = {
        seed: getattr(args, f"wm_{seed}") for seed in MODEL_SEEDS
    }
    models = load_world_models(checkpoints, args.device)
    train_cache = {
        seed: getattr(args, f"train_cache_{seed}") for seed in MODEL_SEEDS
    }
    calibration_cache = {
        seed: getattr(args, f"calibration_cache_{seed}")
        for seed in MODEL_SEEDS
    }
    args.output_dir.mkdir(parents=True)
    train_audit, envelope, train_extra = build_partition(
        dataset=args.dataset,
        partition="train-fit",
        rows=train_rows,
        proposal_path=args.train_proposals,
        cache_paths=train_cache,
        trajectory_root=args.trajectory_root,
        latent_root=args.latent_root,
        models=models,
        output_dir=args.output_dir,
        spacing_m=spacing_m,
        rtp_threshold=rtp_threshold,
        support_envelope=None,
        support_quantile=args.support_quantile,
        device=args.device,
        batch_size=args.batch_size,
    )
    calibration_audit, _, calibration_extra = build_partition(
        dataset=args.dataset,
        partition="scorer-cal",
        rows=calibration_rows,
        proposal_path=args.calibration_proposals,
        cache_paths=calibration_cache,
        trajectory_root=args.trajectory_root,
        latent_root=args.latent_root,
        models=models,
        output_dir=args.output_dir,
        spacing_m=spacing_m,
        rtp_threshold=rtp_threshold,
        support_envelope=envelope,
        support_quantile=args.support_quantile,
        device=args.device,
        batch_size=args.batch_size,
    )
    support_path = args.output_dir / "support_envelope.npz"
    save_support_bundle(support_path, {args.dataset: envelope})
    train_rtp = train_extra["rtp"]
    train_valid = train_extra["rtp_valid"]
    rtp_delta = train_rtp[train_valid] - train_rtp[train_valid, :1]
    scale = robust_scale(rtp_delta)
    final = {
        "schema": "proposal_aware_v70_tridomain_fixed_pool_audit_v1",
        "created_at": utc_now(),
        "dataset": args.dataset,
        "model_domain": MODEL_DOMAIN[args.dataset],
        "spacing_m": spacing_m,
        "world_models": {
            str(seed): {
                "path": str(checkpoints[seed]),
                "sha256": file_sha256(checkpoints[seed]),
            }
            for seed in MODEL_SEEDS
        },
        "train": train_audit,
        "calibration": calibration_audit,
        "support_envelope": str(support_path),
        "support_envelope_sha256": file_sha256(support_path),
        "rtp": {
            "threshold_source": "train-fit only",
            "absolute_minimum_distance_m": args.rtp_minimum_distance,
            "positive_distance_quantile": args.rtp_distance_quantile,
            "learned_distance_floor_m": learned_floor,
            "validity_threshold_m": rtp_threshold,
            "no_value_clipping": True,
            "invalid_values_excluded": True,
            "robust_scale_source": "train-fit valid incumbent-relative RTP only",
            **scale,
            "calibration_valid_windows": int(
                calibration_extra["rtp_valid"].sum()
            ),
        },
        "generic_absolute_ate_rtp_scorer_used_by_model": False,
        "external_test_accessed": False,
        "decision": "OLD_V70_K17_FIXED_POOL_READY",
    }
    write_json(args.output_dir / "FIXED_POOL_COMPLETE.json", final)
    del models
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
