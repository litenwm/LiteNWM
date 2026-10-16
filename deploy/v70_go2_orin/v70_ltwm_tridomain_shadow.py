#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.costs import latent_trajectory_features  # noqa: E402
from planning.trajectory_constraints import (  # noqa: E402
    ActionSupportStats,
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
from numpy_pickle_compat import install_numpy_pickle_compat  # noqa: E402

SEEDS = (1701, 1702, 1703)
CANDIDATES = 17
HISTORY = 4
HORIZON = 4
TOKENS = 256
LATENT_DIM = 384


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
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints/project",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/unitree/v70_ltwm_tridomain_shadow.npz"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_stats(value: dict[str, Any]) -> ActionSupportStats:
    return ActionSupportStats(
        lower=torch.as_tensor(
            value["lower_p01"],
            dtype=torch.float32,
        ),
        upper=torch.as_tensor(
            value["upper_p99"],
            dtype=torch.float32,
        ),
        mean=torch.as_tensor(
            value["mean"],
            dtype=torch.float32,
        ),
        covariance_inverse=torch.as_tensor(
            value["covariance_inverse"],
            dtype=torch.float32,
        ),
        mahalanobis_squared_max=float(
            value["mahalanobis_squared_p99"]
        ),
        support_actions=torch.as_tensor(
            value["support_actions"],
            dtype=torch.float32,
        ),
        knn_k=int(value["knn_k"]),
        knn_distance_max=float(value["knn_distance_p99"]),
        roughness_max=float(value["roughness_p99"]),
        jerk_max=float(value["jerk_p99"]),
        curvature_max=float(value["curvature_p99"]),
        parent_trust_radius=float(value["parent_trust_radius"]),
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as payload:
        return {
            key: payload[key].copy()
            for key in payload.files
        }


def require_shape(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> None:
    if value.shape != shape:
        raise RuntimeError(
            f"{name} shape mismatch: "
            f"expected={shape}, actual={value.shape}"
        )

    if not np.isfinite(value).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def main() -> None:
    args = parse_args()
    install_numpy_pickle_compat()

    print("===== V70 tri-domain LTWM fixed-pool shadow =====")
    print("No ROS publisher is created.")
    print("No SportClient command is issued.")
    print()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    online = load_npz(args.online_features)
    dino = load_npz(args.dino_features)

    required_online = {
        "candidates_waypoint_units",
        "actions_h4",
        "action_features",
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
            f"online feature file missing: {sorted(missing_online)}"
        )
    if missing_dino:
        raise RuntimeError(
            f"DINO feature file missing: {sorted(missing_dino)}"
        )

    candidates = online["candidates_waypoint_units"].astype(
        np.float32
    )
    actions = online["actions_h4"].astype(np.float32)
    action_features = online["action_features"].astype(np.float32)

    history_np = dino["history_latent"].astype(np.float32)
    goal_np = dino["goal_latent"].astype(np.float32)
    state_static = dino["static_features"].astype(np.float32)

    require_shape(
        "candidates_waypoint_units",
        candidates,
        (CANDIDATES, HORIZON, 2),
    )
    require_shape(
        "actions_h4",
        actions,
        (CANDIDATES, HORIZON, 3),
    )
    require_shape(
        "action_features",
        action_features,
        (CANDIDATES, 242),
    )
    require_shape(
        "history_latent",
        history_np,
        (1, HISTORY, TOKENS, LATENT_DIM),
    )
    require_shape(
        "goal_latent",
        goal_np,
        (1, TOKENS, LATENT_DIM),
    )
    require_shape(
        "state_static",
        state_static,
        (1, 210),
    )

    # 增加 batch 维度：[17,4,3] -> [1,17,4,3]
    actions_np = actions[None]
    action_features_np = action_features[None]

    # 静态视觉特征对同一个 K17 候选池重复。
    static_features = np.repeat(
        state_static[:, None, :],
        CANDIDATES,
        axis=1,
    ).astype(np.float32)

    require_shape(
        "static_features",
        static_features,
        (1, CANDIDATES, 210),
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    history = torch.from_numpy(history_np).to(device)
    goal = torch.from_numpy(goal_np).to(device)
    actions_cpu = torch.from_numpy(actions_np)

    scalar_by_seed: list[np.ndarray] = []
    ate_by_seed: list[np.ndarray] = []
    rtp_by_seed: list[np.ndarray] = []
    support_by_seed: list[np.ndarray] = []

    for seed in SEEDS:
        print()
        print(f"===== seed {seed} =====")

        wm_path = (
            args.checkpoint_root
            / "ltwm_expansion"
            / f"seed{seed}"
            / "best.pt"
        )
        scorer_path = (
            args.checkpoint_root
            / "scorer_v2"
            / f"seed{seed}"
            / "best.pt"
        )

        if not wm_path.is_file():
            raise FileNotFoundError(wm_path)
        if not scorer_path.is_file():
            raise FileNotFoundError(scorer_path)

        wm_sha256 = file_sha256(wm_path)

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
        if paired_hash != wm_sha256:
            raise RuntimeError(
                f"seed {seed} scorer/WM provenance mismatch:\n"
                f"scorer expects={paired_hash}\n"
                f"actual WM={wm_sha256}"
            )

        wm = LiteTokenFormerWMV2(
            LiteTokenFormerConfig(**wm_checkpoint["config"])
        )
        wm.load_state_dict(
            wm_checkpoint["model"],
            strict=True,
        )
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

        require_shape(
            f"support seed {seed}",
            support.astype(np.float32),
            (1, CANDIDATES),
        )

        # incumbent 始终可作为 fallback。
        support[:, 0] = True
        support_by_seed.append(support.astype(bool))

        actions_device = actions_cpu.to(device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.monotonic()

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                future = wm(
                    history,
                    actions_device,
                )["predicted_tokens"]

            future = future.float()

            scalar, _ = latent_trajectory_features(
                future,
                goal,
                history[:, -1],
            )

            prediction = scorer(
                current_latent=history[:, -1],
                goal_latent=goal,
                candidate_actions=actions_device,
                predicted_future=future,
            )

        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        peak_gib = (
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 3)
        )

        scalar_np = scalar.cpu().numpy().astype(np.float32)
        ate_np = prediction.ate.cpu().numpy().astype(np.float32)
        rtp_np = prediction.rtp.cpu().numpy().astype(np.float32)

        require_shape(
            f"scalar seed {seed}",
            scalar_np,
            (1, CANDIDATES, 28),
        )
        require_shape(
            f"ATE seed {seed}",
            ate_np,
            (1, CANDIDATES),
        )
        require_shape(
            f"RTP seed {seed}",
            rtp_np,
            (1, CANDIDATES),
        )

        scalar_by_seed.append(scalar_np)
        ate_by_seed.append(ate_np)
        rtp_by_seed.append(rtp_np)

        print("future:", tuple(future.shape))
        print("scalar:", scalar_np.shape)
        print("ATE range:", float(ate_np.min()), float(ate_np.max()))
        print("RTP range:", float(rtp_np.min()), float(rtp_np.max()))
        print(
            "support accepted:",
            int(support.sum()),
            "/",
            CANDIDATES,
        )
        print("inference seconds:", round(elapsed, 3))
        print(
            "peak CUDA allocated GiB:",
            round(peak_gib, 3),
        )

        del (
            wm,
            scorer,
            wm_checkpoint,
            scorer_checkpoint,
            future,
            scalar,
            prediction,
            actions_device,
        )
        gc.collect()
        torch.cuda.empty_cache()

    scalar_seed = np.stack(
        scalar_by_seed,
        axis=0,
    ).astype(np.float32)
    ate_seed = np.stack(
        ate_by_seed,
        axis=0,
    ).astype(np.float32)
    rtp_seed = np.stack(
        rtp_by_seed,
        axis=0,
    ).astype(np.float32)
    support_seed = np.stack(
        support_by_seed,
        axis=0,
    ).astype(bool)

    require_shape(
        "scalar_by_seed",
        scalar_seed,
        (3, 1, CANDIDATES, 28),
    )
    require_shape(
        "ate_by_seed",
        ate_seed,
        (3, 1, CANDIDATES),
    )
    require_shape(
        "rtp_by_seed",
        rtp_seed,
        (3, 1, CANDIDATES),
    )

    majority_support = (
        support_seed.sum(axis=0) >= 2
    )
    majority_support[:, 0] = True

    # 正式 112D LTWM 特征：
    # 28D × (mean, std, min, max)。
    raw_ltwm_features = np.concatenate(
        (
            scalar_seed.mean(axis=0),
            scalar_seed.std(axis=0),
            scalar_seed.min(axis=0),
            scalar_seed.max(axis=0),
        ),
        axis=-1,
    ).astype(np.float32)

    # scorer-v2 只作为独立辅助信息：
    # ATE×3 + RTP×3 + ATE mean/std + RTP mean/std = 10D。
    scorer_v2_aux_features = np.concatenate(
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

    support_features = support_seed.mean(
        axis=0
    )[..., None].astype(np.float32)

    require_shape(
        "raw_ltwm_features",
        raw_ltwm_features,
        (1, CANDIDATES, 112),
    )
    require_shape(
        "scorer_v2_aux_features",
        scorer_v2_aux_features,
        (1, CANDIDATES, 10),
    )
    require_shape(
        "support_features",
        support_features,
        (1, CANDIDATES, 1),
    )
    require_shape(
        "majority_support",
        majority_support.astype(np.float32),
        (1, CANDIDATES),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        args.output,
        candidates_waypoint_units=candidates[None],
        actions_h4=actions_np,
        action_features=action_features_np,
        static_features=static_features,
        raw_ltwm_features=raw_ltwm_features,
        scorer_v2_aux_features=scorer_v2_aux_features,
        support_features=support_features,
        ate_by_seed=ate_seed,
        rtp_by_seed=rtp_seed,
        support_by_seed=support_seed,
        majority_support=majority_support,
        history_latent=history_np.astype(np.float16),
        goal_latent=goal_np.astype(np.float16),
    )

    print()
    print("===== RESULT =====")
    print("actions_h4:", actions_np.shape)
    print("action_features:", action_features_np.shape)
    print("static_features:", static_features.shape)
    print("raw_ltwm_features:", raw_ltwm_features.shape)
    print(
        "scorer_v2_aux_features:",
        scorer_v2_aux_features.shape,
    )
    print("support_features:", support_features.shape)
    print("ate_by_seed:", ate_seed.shape)
    print("rtp_by_seed:", rtp_seed.shape)
    print("support_by_seed:", support_seed.shape)
    print(
        "majority support:",
        int(majority_support.sum()),
        "/",
        CANDIDATES,
    )
    print(
        "all finite:",
        bool(
            np.isfinite(raw_ltwm_features).all()
            and np.isfinite(scorer_v2_aux_features).all()
        ),
    )
    print("saved:", args.output)
    print()
    print("V70 TRI-DOMAIN LTWM FIXED-POOL SHADOW: PASS")


if __name__ == "__main__":
    main()
