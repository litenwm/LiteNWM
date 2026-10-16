#!/usr/bin/env python3
"""Held-route retrieval test for the joystick-to-motion-to-latent chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from run_scand_command_motion_probe_v1 import Ridge  # noqa: E402
from run_scand_two_stage_latent_probe_v1 import (  # noqa: E402
    HISTORY,
    HORIZON,
    flatten_tokens,
    latent_nmse,
    load_route,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-pair",
        nargs=2,
        metavar=("BAG", "LATENT"),
        action="append",
        required=True,
    )
    parser.add_argument(
        "--held-pair",
        nargs=2,
        metavar=("BAG", "LATENT"),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ridge", type=float, default=1.0e-2)
    return parser.parse_args()


def candidate_indices(index: int, count: int, candidates: int) -> np.ndarray:
    if candidates < 2 or candidates > count:
        raise ValueError("candidate count must be between 2 and held window count")
    stride = 17
    values = [index]
    step = 1
    while len(values) < candidates:
        candidate = (index + step * stride) % count
        if candidate not in values:
            values.append(candidate)
        step += 1
    return np.asarray(values, dtype=np.int64)


def rank_metrics(scores: np.ndarray) -> dict[str, float]:
    true = scores[:, :1]
    rank = 1 + (scores[:, 1:] < true).sum(axis=1)
    ties = (scores[:, 1:] == true).sum(axis=1)
    rank = rank + 0.5 * ties
    return {
        "top1": float((rank == 1).mean()),
        "pairwise_win_rate": float((true < scores[:, 1:]).mean()),
        "mean_rank": float(rank.mean()),
        "mean_reciprocal_rank": float((1.0 / rank).mean()),
    }


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    train = [
        load_route(Path(bag), Path(latent))
        for bag, latent in args.train_pair
    ]
    held = load_route(Path(args.held_pair[0]), Path(args.held_pair[1]))
    train_x = np.concatenate([route["full"] for route in train])
    train_y = np.concatenate([route["target"] for route in train])
    controller = Ridge(args.ridge).fit(train_x, train_y)
    state_dim = held["state"].shape[1]
    held_count = len(held["target"])

    predicted_candidates = np.zeros(
        (held_count, args.candidates, HORIZON, 3),
        dtype=np.float32,
    )
    achieved_candidates = np.zeros_like(predicted_candidates)
    source_indices = np.zeros((held_count, args.candidates), dtype=np.int64)
    for index in range(held_count):
        sources = candidate_indices(index, held_count, args.candidates)
        source_indices[index] = sources
        features = np.concatenate(
            (
                np.repeat(held["state"][index : index + 1], args.candidates, axis=0),
                held["full"][sources, state_dim:],
            ),
            axis=1,
        )
        predicted_candidates[index] = controller.predict(features).reshape(
            args.candidates,
            HORIZON,
            3,
        )
        achieved_candidates[index] = held["target"][sources].reshape(
            args.candidates,
            HORIZON,
            3,
        )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = HighResDualLatentWMConfig(**payload["config"])
    model = HighResMultiLayerDualLatentWM(config).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)

    predicted_score_rows = []
    achieved_score_rows = []
    starts = held["starts"]
    with torch.no_grad():
        for offset in range(0, held_count, args.batch_size):
            batch_starts = starts[offset : offset + args.batch_size]
            batch = len(batch_starts)
            semantic_history = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(held["semantic"][start : start + HISTORY])
                        for start in batch_starts
                    ]
                )
            ).cuda()
            geometry_history = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(held["geometry"][start : start + HISTORY])
                        for start in batch_starts
                    ]
                )
            ).cuda()
            semantic_future = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(
                            held["semantic"][
                                start + HISTORY : start + HISTORY + HORIZON
                            ]
                        )
                        for start in batch_starts
                    ]
                )
            ).cuda()
            geometry_future = torch.from_numpy(
                np.stack(
                    [
                        flatten_tokens(
                            held["geometry"][
                                start + HISTORY : start + HISTORY + HORIZON
                            ]
                        )
                        for start in batch_starts
                    ]
                )
            ).cuda()
            action_groups = {
                "predicted": torch.from_numpy(
                    predicted_candidates[offset : offset + batch]
                ).cuda(),
                "achieved": torch.from_numpy(
                    achieved_candidates[offset : offset + batch]
                ).cuda(),
            }
            for name, actions in action_groups.items():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(semantic_history, geometry_history, actions)
                semantic_target = semantic_future[:, None].expand_as(
                    output["predicted_semantic_tokens"]
                )
                geometry_target = geometry_future[:, None].expand_as(
                    output["predicted_geometry_tokens"]
                )
                score = 0.5 * (
                    latent_nmse(
                        output["predicted_semantic_tokens"],
                        semantic_target,
                    )[:, :, -1]
                    + latent_nmse(
                        output["predicted_geometry_tokens"],
                        geometry_target,
                    )[:, :, -1]
                )
                if name == "predicted":
                    predicted_score_rows.append(score.cpu().numpy())
                else:
                    achieved_score_rows.append(score.cpu().numpy())

    predicted_scores = np.concatenate(predicted_score_rows)
    achieved_scores = np.concatenate(achieved_score_rows)
    predicted_metrics = rank_metrics(predicted_scores)
    achieved_metrics = rank_metrics(achieved_scores)
    random_top1 = 1.0 / args.candidates
    status = (
        "GO_TWO_STAGE_ACTION_RETRIEVAL"
        if predicted_metrics["top1"] >= 2.0 * random_top1
        and predicted_metrics["pairwise_win_rate"] >= 0.6
        else "NO_GO_TWO_STAGE_ACTION_RETRIEVAL"
    )
    result = {
        "status": status,
        "interpretation": (
            "true joystick is recoverable from held-route future latents through the two-stage chain"
            if status.startswith("GO_")
            else "the two-stage chain improves factual prediction but does not retrieve the true action reliably"
        ),
        "protocol": {
            "train_pairs": args.train_pair,
            "held_pair": args.held_pair,
            "checkpoint": str(args.checkpoint),
            "candidates_including_true": args.candidates,
            "negative_source_stride": 17,
            "state_held_fixed_while_joystick_sequence_changes": True,
            "ate_or_objective_loaded": False,
            "sealed_validation_opened": False,
            "formal_v70_modified": False,
            "frozen_highres_dual_modified": False,
        },
        "windows": held_count,
        "random_top1": random_top1,
        "predicted_motion_from_joystick": predicted_metrics,
        "achieved_motion_upper_bound": achieved_metrics,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
