#!/usr/bin/env python3
"""Build a train-only coherent-turn tail support bank for V70 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECON = (
    REPO_ROOT
    / "outputs/universal_dualdomain_raw_v2/candidates/recon_train_fit_k17.npz"
)
DEFAULT_SCAND = (
    REPO_ROOT
    / "outputs/scand_domain_adaptation_v1/candidates/train_fit_k17.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runtime/v70_tridomain/action_support/coherent_turn_tail_v1.npz"
)
EXPECTED_SOURCE_SHA256 = {
    "recon": "398e9977a0bf8dbea44409e9ac3f7fb559a4e30e4a0d5f705bae2d8f08acdad7",
    "scand": "9b0119ef7cb93df089af244ee0f0599131181e89d972ea761f0ab0285ad74a43",
}
COMPATIBLE_SCORER_SHA256 = {
    1701: "e7cef7ce68e97ab74db2a07474b0e0880eff06c3d4f9e2803be3a861461930ef",
    1702: "ee8f65aae6c117e76cc3e59ecc9f2b831e28ccc3625ca4db6bcf681ce0328f09",
    1703: "49be8e8c5022ffcc9b8fc0a6298320711070f80c77909b27f799744a7ed14754",
}
COMPATIBLE_SPACING_SUPPORT_SHA256 = (
    "da86c3f8286da60ac96fd0c3a9cd612e"
    "cdc3ed4198a82345868fa24592a6bfd9"
)

# Seed-1701 tri-domain scorer bounds are used only for frozen normalization
# and to identify candidates outside the central per-coordinate box.
NORMALIZATION_LOWER = np.asarray(
    [
        [0.0757758543, -0.0524755269, -0.1751965582],
        [0.0659731254, -0.0980592594, -0.3542641401],
        [0.0582701080, -0.0809485689, -0.3087089658],
        [0.0523282140, -0.0735953078, -0.2817817330],
    ],
    dtype=np.float32,
)
NORMALIZATION_UPPER = np.asarray(
    [
        [0.5025255084, 0.0587636121, 0.1955545098],
        [0.5008600354, 0.1016835868, 0.3504380584],
        [0.5008883476, 0.0877432749, 0.3297919333],
        [0.5006397367, 0.0830194205, 0.3177157640],
    ],
    dtype=np.float32,
)
COMPATIBLE_ACTION_SUPPORT_LOWER = np.asarray(
    [
        NORMALIZATION_LOWER,
        [
            [0.0734155998, -0.0536438152, -0.1837540716],
            [0.0640645176, -0.1002518833, -0.3703424931],
            [0.0554685853, -0.0841913223, -0.3213452101],
            [0.0532634258, -0.0741315186, -0.2817462981],
        ],
        [
            [0.0766008571, -0.0541715436, -0.1819976121],
            [0.0652752891, -0.1008251235, -0.3706398606],
            [0.0565295815, -0.0827065557, -0.3233909607],
            [0.0525851175, -0.0749822631, -0.2905083895],
        ],
    ],
    dtype=np.float32,
)
COMPATIBLE_ACTION_SUPPORT_UPPER = np.asarray(
    [
        NORMALIZATION_UPPER,
        [
            [0.5030031204, 0.0582314171, 0.1956209689],
            [0.5013756156, 0.1001180187, 0.3541660011],
            [0.5016536713, 0.0883893073, 0.3346362710],
            [0.4998815060, 0.0831966102, 0.3264000416],
        ],
        [
            [0.5036292672, 0.0582776703, 0.1967417002],
            [0.5020352006, 0.1003283635, 0.3522523642],
            [0.5016479492, 0.0872603208, 0.3287881315],
            [0.5014229417, 0.0812634975, 0.3175317347],
        ],
    ],
    dtype=np.float32,
)

MIN_FORWARD_M = 0.04
MIN_STEP_LENGTH_M = 0.05
MAX_STEP_LENGTH_M = 0.35
MAX_ABS_DYAW_RAD = 0.75
MAX_ABS_YAW_SUM_RAD = 1.70
YAW_SIGN_TOLERANCE_RAD = 0.03
LOO_DISTANCE_QUANTILE = 0.99


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recon", type=Path, default=DEFAULT_RECON)
    parser.add_argument("--scand", type=Path, default=DEFAULT_SCAND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_actions(path: Path, *, domain: str) -> np.ndarray:
    actual_sha256 = sha256(path)
    expected_sha256 = EXPECTED_SOURCE_SHA256[domain]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{domain} candidate pool hash mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    with np.load(path, allow_pickle=False) as archive:
        actions = np.asarray(archive["candidate_actions_h4"], dtype=np.float32)
    if actions.ndim != 4 or actions.shape[-2:] != (4, 3):
        raise RuntimeError(f"invalid {domain} action shape: {actions.shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError(f"{domain} actions contain NaN or Inf")
    return actions.reshape(-1, 4, 3)


def coherent_tail_mask(actions: np.ndarray) -> np.ndarray:
    central = np.all(
        (actions >= NORMALIZATION_LOWER)
        & (actions <= NORMALIZATION_UPPER),
        axis=(1, 2),
    )
    translation = np.linalg.norm(actions[..., :2], axis=-1)
    yaw = actions[..., 2]
    same_turn = np.all(yaw >= -YAW_SIGN_TOLERANCE_RAD, axis=1) | np.all(
        yaw <= YAW_SIGN_TOLERANCE_RAD, axis=1
    )
    coherent = (
        (actions[..., 0].min(axis=1) >= MIN_FORWARD_M)
        & (translation.min(axis=1) >= MIN_STEP_LENGTH_M)
        & (translation.max(axis=1) <= MAX_STEP_LENGTH_M)
        & (np.abs(yaw).max(axis=1) <= MAX_ABS_DYAW_RAD)
        & (np.abs(yaw).sum(axis=1) <= MAX_ABS_YAW_SUM_RAD)
        & same_turn
    )
    return coherent & ~central


def leave_one_out_distance(templates: np.ndarray) -> np.ndarray:
    scale = (NORMALIZATION_UPPER - NORMALIZATION_LOWER).reshape(1, 1, 12)
    normalized = templates.reshape(-1, 1, 12) / scale
    distance = np.sqrt(
        np.mean(
            (normalized - np.swapaxes(normalized, 0, 1)) ** 2,
            axis=-1,
        )
    )
    np.fill_diagonal(distance, np.inf)
    return distance.min(axis=1).astype(np.float32)


def main() -> None:
    args = parse_args()
    domain_actions = {
        "recon": load_actions(args.recon.resolve(strict=True), domain="recon"),
        "scand": load_actions(args.scand.resolve(strict=True), domain="scand"),
    }
    templates = np.concatenate(list(domain_actions.values()), axis=0)
    templates = templates[coherent_tail_mask(templates)]
    if len(templates) < 2:
        raise RuntimeError("coherent tail template bank is empty")
    unique = np.unique(templates.reshape(len(templates), -1), axis=0)
    if len(unique) != len(templates):
        raise RuntimeError("coherent tail template bank contains duplicates")
    templates = unique.reshape(-1, 4, 3).astype(np.float32)
    loo_distance = leave_one_out_distance(templates)
    distance_max = float(np.quantile(loo_distance, LOO_DISTANCE_QUANTILE))

    metadata = {
        "schema": "v70_coherent_turn_tail_support_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fit_partition": "train-fit only",
        "domains": ["recon", "scand"],
        "source_paths": {
            "recon": str(args.recon.resolve()),
            "scand": str(args.scand.resolve()),
        },
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "compatible_scorer_sha256": {
            str(seed): value
            for seed, value in COMPATIBLE_SCORER_SHA256.items()
        },
        "compatible_spacing_support_sha256": (
            COMPATIBLE_SPACING_SUPPORT_SHA256
        ),
        "candidate_counts": {
            domain: int(len(actions))
            for domain, actions in domain_actions.items()
        },
        "tail_template_count": int(len(templates)),
        "tail_knn": {
            "metric": "RMS over 12 dimensions normalized by frozen P01-P99 span",
            "neighbors": 1,
            "leave_one_out_quantile": LOO_DISTANCE_QUANTILE,
            "distance_max": distance_max,
            "loo_p50": float(np.quantile(loo_distance, 0.50)),
            "loo_p90": float(np.quantile(loo_distance, 0.90)),
            "loo_p99": float(np.quantile(loo_distance, 0.99)),
            "loo_max": float(loo_distance.max()),
        },
        "physical_guard": {
            "min_forward_m": MIN_FORWARD_M,
            "min_step_length_m": MIN_STEP_LENGTH_M,
            "max_step_length_m": MAX_STEP_LENGTH_M,
            "max_abs_dyaw_rad": MAX_ABS_DYAW_RAD,
            "max_abs_yaw_sum_rad": MAX_ABS_YAW_SUM_RAD,
            "yaw_sign_tolerance_rad": YAW_SIGN_TOLERANCE_RAD,
        },
        "central_support_behavior": "unchanged scorer-member P01-P99 bounds",
        "tail_behavior": (
            "outside-central candidates require both physical coherence and "
            "train-fit trajectory-level nearest-neighbor support"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        tail_support_actions=templates.reshape(-1, 12),
        tail_normalization_lower=NORMALIZATION_LOWER,
        tail_normalization_upper=NORMALIZATION_UPPER,
        tail_knn_distance_max=np.asarray(distance_max, dtype=np.float32),
        tail_min_forward_m=np.asarray(MIN_FORWARD_M, dtype=np.float32),
        tail_min_step_length_m=np.asarray(MIN_STEP_LENGTH_M, dtype=np.float32),
        tail_max_step_length_m=np.asarray(MAX_STEP_LENGTH_M, dtype=np.float32),
        tail_max_abs_dyaw_rad=np.asarray(MAX_ABS_DYAW_RAD, dtype=np.float32),
        tail_max_abs_yaw_sum_rad=np.asarray(
            MAX_ABS_YAW_SUM_RAD, dtype=np.float32
        ),
        tail_yaw_sign_tolerance_rad=np.asarray(
            YAW_SIGN_TOLERANCE_RAD, dtype=np.float32
        ),
        compatible_action_support_seeds=np.asarray(
            sorted(COMPATIBLE_SCORER_SHA256), dtype=np.int64
        ),
        compatible_spacing_support_sha256=np.asarray(
            COMPATIBLE_SPACING_SUPPORT_SHA256
        ),
        compatible_action_support_lower=COMPATIBLE_ACTION_SUPPORT_LOWER,
        compatible_action_support_upper=COMPATIBLE_ACTION_SUPPORT_UPPER,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    )
    print(json.dumps({**metadata, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
