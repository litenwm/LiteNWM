#!/usr/bin/env python3
"""Test metric-depth forward splatting on the frozen strong dual-latent LTWM.

The pilot uses only SCAND development train/validation routes and true future
DINO latents.  A frozen Depth Pro model estimates current-frame metric depth
and focal length.  Current semantic/geometry tokens are then forward-splatted
under each H1-H4 action using the previously validated Spot camera axes.

Two scalar residual weights are fit on train routes by closed-form least
squares and frozen before validation inference.  No navigation outcome, ATE,
objective, union65 label, or sealed test route is read.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(Path(__file__).parent)]

import run_raw_resolution_paired_pilot_v1 as evaluation  # noqa: E402
from recon_latents.models_highres_dual_v1 import (  # noqa: E402
    HighResDualLatentWMConfig,
    HighResMultiLayerDualLatentWM,
)
from recon_latents.models_v2 import integrate_se2_prefix  # noqa: E402


SELF_CAL_FOCAL_448_PX = 298.47442166440477
IMAGE_WIDTH = 448
IMAGE_HEIGHT = 336
GRID_WIDTH = 28
GRID_HEIGHT = 21
PRINCIPAL_X = 224.0
PRINCIPAL_Y = 168.0
BODY_FORWARD_AXIS_CAMERA = (
    0.013540377093979815,
    -0.05269835610248765,
    0.9985186735621164,
)
BODY_LATERAL_AXIS_CAMERA = (
    0.9995075688849837,
    -0.02755676943468673,
    -0.01500813779098614,
)
CAMERA_YAW_AXIS = (
    0.028306853053325053,
    0.9982301877434978,
    0.05229927674259504,
)


@dataclass(frozen=True)
class WindowRef:
    trajectory_id: str
    route_id: str
    shard_path: Path
    timestep: int
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=ROOT / "data/highres_dual_latents/scand_v1",
    )
    parser.add_argument(
        "--depth-model",
        type=Path,
        default=ROOT / "assets/third_party/depthpro_hf",
    )
    parser.add_argument(
        "--strong-checkpoint",
        type=Path,
        default=(
            ROOT
            / "outputs/highres_dual_ltwm_v1/frozen/"
            "highres_multilayer_dual_ltwm_scand_v1.pt"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs/highres_dual_ltwm_v1/"
            "metric_depth_forward_splat_pilot_v1/result.json"
        ),
    )
    parser.add_argument("--fit-windows", type=int, default=48)
    parser.add_argument("--eval-windows", type=int, default=96)
    parser.add_argument("--model-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026072402)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def route_id(trajectory_id: str) -> str:
    value = re.sub(r"_\d+$", "", trajectory_id)
    require(value != trajectory_id, f"unrecognized trajectory id: {trajectory_id}")
    return value


def select_evenly(rows: list[WindowRef], count: int) -> list[WindowRef]:
    require(count > 0 and rows, "invalid deterministic sample request")
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    return [rows[int(index)] for index in indices]


def collect_refs(
    latent_root: Path,
    split: str,
    total: int,
) -> list[WindowRef]:
    metadata = read_json(latent_root / "metadata.json")
    splits = read_json(latent_root / "splits.json")
    allowed = set(map(str, splits[split]))
    grouped: dict[str, list[WindowRef]] = defaultdict(list)
    for row in metadata["trajectories"]:
        trajectory = str(row["trajectory_id"])
        if trajectory not in allowed or "_Spot_" not in trajectory:
            continue
        path = latent_root / "shards" / str(row["shard_file"])
        with np.load(path, allow_pickle=False) as shard:
            image_paths = shard["image_paths"].astype(str)
        for timestep in range(3, int(row["num_frames"]) - 4):
            grouped[route_id(trajectory)].append(
                WindowRef(
                    trajectory_id=trajectory,
                    route_id=route_id(trajectory),
                    shard_path=path,
                    timestep=timestep,
                    image_path=Path(image_paths[timestep]),
                )
            )
    require(grouped, f"no Spot routes for split={split}")
    routes = sorted(grouped)
    per_route = max(1, total // len(routes))
    selected = []
    for route in routes:
        local = sorted(
            grouped[route],
            key=lambda row: (row.trajectory_id, row.timestep),
        )
        selected.extend(select_evenly(local, per_route))
    if len(selected) < total:
        selected_set = set(selected)
        remainder = [
            row
            for route in routes
            for row in grouped[route]
            if row not in selected_set
        ]
        selected.extend(select_evenly(remainder, total - len(selected)))
    selected = sorted(
        selected[:total],
        key=lambda row: (row.route_id, row.trajectory_id, row.timestep),
    )
    require(
        len({row.route_id for row in selected}) >= min(4, len(routes)),
        f"insufficient route coverage for split={split}",
    )
    return selected


def load_window(
    row: WindowRef,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(row.shard_path, allow_pickle=False) as shard:
        semantic = shard["semantic_latents"].astype(np.float32)
        geometry = shard["geometry_latents"].astype(np.float32)
        actions = shard["actions_continuous"].astype(np.float32)
    t = row.timestep
    semantic_history = semantic[t - 3 : t + 1].reshape(4, -1, 384)
    geometry_history = geometry[t - 3 : t + 1].reshape(4, -1, 192)
    semantic_future = semantic[t + 1 : t + 5].reshape(4, -1, 384)
    geometry_future = geometry[t + 1 : t + 5].reshape(4, -1, 192)
    return tuple(
        torch.from_numpy(value)
        for value in (
            semantic_history,
            geometry_history,
            semantic_future,
            geometry_future,
            actions[t : t + 4],
        )
    )


def load_batch(
    rows: list[WindowRef],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = [load_window(row) for row in rows]
    return tuple(
        torch.stack([row[index] for row in values], dim=0)
        for index in range(5)
    )


def infer_depth(
    rows: list[WindowRef],
    model_dir: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    from transformers import (
        DepthProForDepthEstimation,
        DepthProImageProcessorFast,
    )

    processor = DepthProImageProcessorFast.from_pretrained(
        model_dir,
        local_files_only=True,
    )
    model = DepthProForDepthEstimation.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    depth_rows = []
    focal_rows = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with Image.open(row.image_path) as image:
                image = image.convert("RGB")
                original_width, original_height = image.size
                inputs = processor(
                    images=image,
                    return_tensors="pt",
                ).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(**inputs)
            processed = processor.post_process_depth_estimation(
                output,
                target_sizes=[(original_height, original_width)],
            )[0]
            depth = processed["predicted_depth"].float()[None, None]
            depth = F.interpolate(
                depth,
                size=(GRID_HEIGHT, GRID_WIDTH),
                mode="area",
            )[0, 0]
            focal = float(processed["focal_length"])
            focal *= IMAGE_WIDTH / float(original_width)
            depth_rows.append(depth.cpu())
            focal_rows.append(focal)
            if (index + 1) % 16 == 0 or index + 1 == len(rows):
                print(
                    json.dumps(
                        {
                            "stage": "depth",
                            "images": index + 1,
                            "total": len(rows),
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
    del model
    torch.cuda.empty_cache()
    depth = torch.stack(depth_rows)
    focal = torch.tensor(focal_rows, dtype=torch.float32)
    summary = {
        "images": len(rows),
        "depth_m": {
            "q10": float(torch.quantile(depth, 0.10)),
            "q50": float(torch.quantile(depth, 0.50)),
            "q90": float(torch.quantile(depth, 0.90)),
            "minimum": float(depth.min()),
            "maximum": float(depth.max()),
        },
        "focal_448_px": {
            "q10": float(torch.quantile(focal, 0.10)),
            "q50": float(torch.quantile(focal, 0.50)),
            "q90": float(torch.quantile(focal, 0.90)),
            "selfcalibration_reference": SELF_CAL_FOCAL_448_PX,
            "median_relative_difference": float(
                abs(float(torch.median(focal)) - SELF_CAL_FOCAL_448_PX)
                / SELF_CAL_FOCAL_448_PX
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    return depth, focal, summary


def normalized_axis(value: tuple[float, float, float]) -> torch.Tensor:
    tensor = torch.tensor(value, dtype=torch.float32)
    return tensor / torch.linalg.vector_norm(tensor).clamp_min(1.0e-8)


def camera_rotation(yaw: torch.Tensor) -> torch.Tensor:
    axis = normalized_axis(CAMERA_YAW_AXIS).to(yaw)
    x, y, z = axis.unbind()
    zero = torch.zeros((), device=yaw.device, dtype=yaw.dtype)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    outer = axis[:, None] * axis[None, :]
    identity = torch.eye(3, device=yaw.device, dtype=yaw.dtype)
    cosine = torch.cos(yaw)[..., None, None]
    sine = torch.sin(yaw)[..., None, None]
    return cosine * identity + (1.0 - cosine) * outer + sine * skew


def forward_splat(
    reference: torch.Tensor,
    actions: torch.Tensor,
    depth: torch.Tensor,
    focal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-splat current tokens using current-frame metric depth."""
    if actions.ndim == 3:
        actions = actions[:, None]
    batch, candidates, horizon = actions.shape[:3]
    tokens, channels = reference.shape[-2:]
    require(tokens == GRID_HEIGHT * GRID_WIDTH, "token grid changed")
    cumulative = integrate_se2_prefix(actions.float())
    yaw = cumulative[..., 2]
    rotation = camera_rotation(yaw)
    forward = normalized_axis(BODY_FORWARD_AXIS_CAMERA).to(reference)
    lateral = normalized_axis(BODY_LATERAL_AXIS_CAMERA).to(reference)
    lateral = lateral - torch.dot(lateral, forward) * forward
    lateral = lateral / torch.linalg.vector_norm(lateral).clamp_min(1.0e-8)
    translation = (
        cumulative[..., 0:1].to(reference) * forward
        + cumulative[..., 1:2].to(reference) * lateral
    )
    cell_width = IMAGE_WIDTH / GRID_WIDTH
    cell_height = IMAGE_HEIGHT / GRID_HEIGHT
    x = (
        torch.arange(GRID_WIDTH, device=reference.device, dtype=torch.float32)
        + 0.5
    ) * cell_width
    y = (
        torch.arange(GRID_HEIGHT, device=reference.device, dtype=torch.float32)
        + 0.5
    ) * cell_height
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.reshape(1, tokens).expand(batch, -1)
    yy = yy.reshape(1, tokens).expand(batch, -1)
    focal = focal.to(reference).reshape(batch, 1)
    rays = torch.stack(
        (
            (xx - PRINCIPAL_X) / focal,
            (yy - PRINCIPAL_Y) / focal,
            torch.ones_like(xx),
        ),
        dim=-1,
    )
    points = rays * depth.to(reference)[..., None]
    shifted = points[:, None, None] - translation[..., None, :]
    future = torch.einsum("bkhij,bkhtj->bkhti", rotation, shifted)
    z = future[..., 2]
    valid_z = z > 0.05
    projected_x = focal[:, None, None] * future[..., 0] / z.clamp_min(0.05)
    projected_x = projected_x + PRINCIPAL_X
    projected_y = focal[:, None, None] * future[..., 1] / z.clamp_min(0.05)
    projected_y = projected_y + PRINCIPAL_Y
    grid_x = projected_x / cell_width - 0.5
    grid_y = projected_y / cell_height - 0.5
    x0 = torch.floor(grid_x)
    y0 = torch.floor(grid_y)
    wx = grid_x - x0
    wy = grid_y - y0
    source = reference[:, None, None].expand(
        batch,
        candidates,
        horizon,
        tokens,
        channels,
    )
    flat_count = batch * candidates * horizon
    source = source.reshape(flat_count, tokens, channels)
    output = torch.zeros(
        flat_count,
        tokens,
        channels,
        device=reference.device,
        dtype=reference.dtype,
    )
    weight_sum = torch.zeros(
        flat_count,
        tokens,
        1,
        device=reference.device,
        dtype=reference.dtype,
    )
    for x_index, y_index, weight in (
        (x0, y0, (1.0 - wx) * (1.0 - wy)),
        (x0 + 1.0, y0, wx * (1.0 - wy)),
        (x0, y0 + 1.0, (1.0 - wx) * wy),
        (x0 + 1.0, y0 + 1.0, wx * wy),
    ):
        valid = (
            valid_z
            & (x_index >= 0)
            & (x_index < GRID_WIDTH)
            & (y_index >= 0)
            & (y_index < GRID_HEIGHT)
        )
        target = (
            y_index.clamp(0, GRID_HEIGHT - 1).long() * GRID_WIDTH
            + x_index.clamp(0, GRID_WIDTH - 1).long()
        ).reshape(flat_count, tokens)
        local_weight = (weight * valid).reshape(
            flat_count,
            tokens,
            1,
        ).to(reference)
        output.scatter_add_(
            1,
            target[..., None].expand(-1, -1, channels),
            source * local_weight,
        )
        weight_sum.scatter_add_(1, target[..., None], local_weight)
    reference_expanded = reference[:, None, None].expand(
        batch,
        candidates,
        horizon,
        tokens,
        channels,
    ).reshape(flat_count, tokens, channels)
    visible = weight_sum > 1.0e-5
    warped = torch.where(
        visible,
        output / weight_sum.clamp_min(1.0e-5),
        reference_expanded,
    )
    return (
        warped.reshape(
            batch,
            candidates,
            horizon,
            tokens,
            channels,
        ),
        visible.float().mean(dim=(1, 2)).reshape(
            batch,
            candidates,
            horizon,
        ),
    )


def load_strong_model(
    checkpoint: Path,
    device: torch.device,
) -> HighResMultiLayerDualLatentWM:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = HighResDualLatentWMConfig(**payload["config"])
    model = HighResMultiLayerDualLatentWM(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


def build_action_bank(
    rows: list[WindowRef],
    device: torch.device,
) -> torch.Tensor:
    return torch.stack([load_window(row)[-1] for row in rows]).to(device)


@torch.no_grad()
def fit_scales(
    model: HighResMultiLayerDualLatentWM,
    rows: list[WindowRef],
    depth: torch.Tensor,
    focal: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    numerator = torch.zeros(2, device=device, dtype=torch.float64)
    denominator = torch.zeros(2, device=device, dtype=torch.float64)
    visibility = []
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            value.to(device) for value in load_batch(local)
        )
        output = model(semantic_history, geometry_history, actions)
        semantic_warp, visible = forward_splat(
            semantic_history[:, -1],
            actions,
            depth[start : start + len(local)].to(device),
            focal[start : start + len(local)].to(device),
        )
        geometry_warp, _ = forward_splat(
            geometry_history[:, -1],
            actions,
            depth[start : start + len(local)].to(device),
            focal[start : start + len(local)].to(device),
        )
        semantic_transport = (
            semantic_warp[:, 0] - semantic_history[:, None, -1]
        )
        geometry_transport = (
            geometry_warp[:, 0] - geometry_history[:, None, -1]
        )
        for index, (prediction, target, transport) in enumerate(
            (
                (
                    output["predicted_semantic_tokens"][:, 0],
                    semantic_future,
                    semantic_transport,
                ),
                (
                    output["predicted_geometry_tokens"][:, 0],
                    geometry_future,
                    geometry_transport,
                ),
            )
        ):
            residual = target.float() - prediction.float()
            transport = transport.float()
            numerator[index] += (residual * transport).sum().double()
            denominator[index] += transport.square().sum().double()
        visibility.append(visible[:, 0].cpu())
    raw = numerator / denominator.clamp_min(1.0e-12)
    clipped = raw.clamp(-0.5, 0.5).float()
    return clipped, {
        "raw_least_squares_scales": raw.cpu().tolist(),
        "frozen_clipped_scales": clipped.cpu().tolist(),
        "clip_bounds": [-0.5, 0.5],
        "visibility_fraction": float(torch.cat(visibility).mean()),
    }


def per_candidate_dual_error(
    semantic_prediction: torch.Tensor,
    geometry_prediction: torch.Tensor,
    semantic_future: torch.Tensor,
    geometry_future: torch.Tensor,
) -> torch.Tensor:
    semantic_target = semantic_future[:, None]
    geometry_target = geometry_future[:, None]
    semantic_energy = semantic_target.float().square().mean(
        dim=(-1, -2)
    ).clamp_min(1.0e-8)
    geometry_energy = geometry_target.float().square().mean(
        dim=(-1, -2)
    ).clamp_min(1.0e-8)
    semantic_error = (
        (semantic_prediction.float() - semantic_target)
        .square()
        .mean(dim=(-1, -2))
        / semantic_energy
    )
    geometry_error = (
        (geometry_prediction.float() - geometry_target)
        .square()
        .mean(dim=(-1, -2))
        / geometry_energy
    )
    return 0.5 * (semantic_error + geometry_error)


def summarize_errors(
    semantic_h4: list[torch.Tensor],
    geometry_h4: list[torch.Tensor],
    correct: list[torch.Tensor],
    negative: list[torch.Tensor],
    visibility: list[torch.Tensor],
) -> dict[str, Any]:
    semantic = torch.cat(semantic_h4).numpy()
    geometry = torch.cat(geometry_h4).numpy()
    correct_value = torch.cat(correct).numpy()
    negative_value = torch.cat(negative).numpy()
    rank = 1 + (negative_value < correct_value[:, None]).sum(axis=1)
    gap = float(negative_value.mean() - correct_value.mean())
    return {
        "windows": int(len(correct_value)),
        "candidate_count_including_true": 12,
        "semantic_h4_nmse": float(semantic.mean()),
        "geometry_h4_nmse": float(geometry.mean()),
        "dual_h4_nmse": float(0.5 * (semantic.mean() + geometry.mean())),
        "correct_action_dual_nmse": float(correct_value.mean()),
        "negative_action_dual_nmse": float(negative_value.mean()),
        "negative_action_gap": gap,
        "negative_gap_fraction_of_true_error": float(
            gap / max(float(correct_value.mean()), 1.0e-8)
        ),
        "pairwise_action_win_rate": float(
            (correct_value[:, None] < negative_value).mean()
        ),
        "retrieval_top1": float(np.mean(rank == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank)),
        "mean_rank": float(rank.mean()),
        "visibility_fraction": float(torch.cat(visibility).mean()),
    }


@torch.no_grad()
def evaluate(
    model: HighResMultiLayerDualLatentWM,
    rows: list[WindowRef],
    depth: torch.Tensor,
    focal: torch.Tensor,
    scales: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    action_bank = build_action_bank(rows, device)
    base_values = ([], [], [], [], [])
    metric_values = ([], [], [], [], [])
    for start in range(0, len(rows), batch_size):
        local = rows[start : start + batch_size]
        semantic_history, geometry_history, semantic_future, geometry_future, actions = (
            value.to(device) for value in load_batch(local)
        )
        negatives = evaluation.validation_negatives(actions, action_bank, start)
        candidate_actions = torch.cat((actions[:, None], negatives), dim=1)
        output = model(
            semantic_history,
            geometry_history,
            candidate_actions,
        )
        local_depth = depth[start : start + len(local)].to(device)
        local_focal = focal[start : start + len(local)].to(device)
        semantic_warp, visible = forward_splat(
            semantic_history[:, -1],
            candidate_actions,
            local_depth,
            local_focal,
        )
        geometry_warp, _ = forward_splat(
            geometry_history[:, -1],
            candidate_actions,
            local_depth,
            local_focal,
        )
        semantic_transport = (
            semantic_warp - semantic_history[:, None, None, -1]
        )
        geometry_transport = (
            geometry_warp - geometry_history[:, None, None, -1]
        )
        base_semantic = output["predicted_semantic_tokens"]
        base_geometry = output["predicted_geometry_tokens"]
        metric_semantic = base_semantic + scales[0] * semantic_transport
        metric_geometry = base_geometry + scales[1] * geometry_transport
        for values, semantic_prediction, geometry_prediction in (
            (base_values, base_semantic, base_geometry),
            (metric_values, metric_semantic, metric_geometry),
        ):
            semantic_h4 = evaluation.h4_nmse(
                semantic_prediction[:, 0, -1],
                semantic_future[:, -1],
            )
            geometry_h4 = evaluation.h4_nmse(
                geometry_prediction[:, 0, -1],
                geometry_future[:, -1],
            )
            error = per_candidate_dual_error(
                semantic_prediction,
                geometry_prediction,
                semantic_future,
                geometry_future,
            ).mean(dim=-1)
            values[0].append(semantic_h4.cpu())
            values[1].append(geometry_h4.cpu())
            values[2].append(error[:, 0].cpu())
            values[3].append(error[:, 1:].cpu())
            values[4].append(visible.cpu())
    return (
        summarize_errors(*base_values),
        summarize_errors(*metric_values),
    )


def changes(base: dict[str, Any], metric: dict[str, Any]) -> dict[str, float]:
    names = (
        "semantic_h4_nmse",
        "geometry_h4_nmse",
        "dual_h4_nmse",
        "negative_gap_fraction_of_true_error",
        "pairwise_action_win_rate",
        "retrieval_top1",
        "mean_rank",
    )
    return {
        f"{name}_change": float(metric[name] - base[name])
        for name in names
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"refusing to overwrite: {output}")
    require(torch.cuda.is_available(), "CUDA required")
    latent_root = args.latent_root.resolve()
    depth_model = args.depth_model.resolve()
    strong_checkpoint = args.strong_checkpoint.resolve()
    for path in (
        latent_root / "metadata.json",
        latent_root / "splits.json",
        depth_model / "config.json",
        depth_model / "preprocessor_config.json",
        depth_model / "model.safetensors",
        strong_checkpoint,
    ):
        require(path.is_file(), f"missing input: {path}")
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    fit_rows = collect_refs(latent_root, "train", args.fit_windows)
    eval_rows = collect_refs(latent_root, "val", args.eval_windows)
    require(
        not ({row.route_id for row in fit_rows} & {row.route_id for row in eval_rows}),
        "fit/eval route overlap",
    )
    all_rows = fit_rows + eval_rows
    device = torch.device("cuda")
    depth, focal, depth_summary = infer_depth(all_rows, depth_model, device)
    fit_depth = depth[: len(fit_rows)]
    eval_depth = depth[len(fit_rows) :]
    fit_focal = focal[: len(fit_rows)]
    eval_focal = focal[len(fit_rows) :]
    strong = load_strong_model(strong_checkpoint, device)
    scales, fit_summary = fit_scales(
        strong,
        fit_rows,
        fit_depth,
        fit_focal,
        device,
        args.model_batch_size,
    )
    base, metric = evaluate(
        strong,
        eval_rows,
        eval_depth,
        eval_focal,
        scales.to(device),
        device,
        args.model_batch_size,
    )
    pooled_changes = changes(base, metric)
    per_route = {}
    dual_improved = 0
    pairwise_improved = 0
    top1_not_worse = 0
    for route in sorted({row.route_id for row in eval_rows}):
        indices = [
            index for index, row in enumerate(eval_rows) if row.route_id == route
        ]
        local_rows = [eval_rows[index] for index in indices]
        local_depth = eval_depth[indices]
        local_focal = eval_focal[indices]
        route_base, route_metric = evaluate(
            strong,
            local_rows,
            local_depth,
            local_focal,
            scales.to(device),
            device,
            args.model_batch_size,
        )
        local_changes = changes(route_base, route_metric)
        dual_improved += int(local_changes["dual_h4_nmse_change"] < 0.0)
        pairwise_improved += int(
            local_changes["pairwise_action_win_rate_change"] > 0.0
        )
        top1_not_worse += int(local_changes["retrieval_top1_change"] >= 0.0)
        per_route[route] = {
            "strong_baseline": route_base,
            "metric_depth_forward_splat": route_metric,
            "changes": local_changes,
        }
    route_count = len(per_route)
    checks = {
        "depth_noncollapsed": depth_summary["depth_m"]["q90"]
        > depth_summary["depth_m"]["q10"] + 0.5,
        "focal_matches_rotation_selfcal_within_25_percent": (
            depth_summary["focal_448_px"]["median_relative_difference"] <= 0.25
        ),
        "fitted_scales_not_clipped": all(
            abs(value) < 0.5
            for value in fit_summary["raw_least_squares_scales"]
        ),
        "pooled_dual_h4_improved": pooled_changes["dual_h4_nmse_change"] < 0.0,
        "pooled_geometry_h4_not_worse": (
            pooled_changes["geometry_h4_nmse_change"] <= 0.0
        ),
        "pooled_pairwise_improved": (
            pooled_changes["pairwise_action_win_rate_change"] > 0.0
        ),
        "pooled_top1_not_worse": (
            pooled_changes["retrieval_top1_change"] >= 0.0
        ),
        "pooled_mean_rank_improved": pooled_changes["mean_rank_change"] < 0.0,
        "majority_routes_dual_improved": dual_improved > route_count / 2,
        "majority_routes_pairwise_improved": (
            pairwise_improved > route_count / 2
        ),
        "majority_routes_top1_not_worse": top1_not_worse > route_count / 2,
    }
    go = all(checks.values())
    result = {
        "schema": "metric_depth_forward_splat_strong_ltwm_pilot_v1",
        "status": (
            "GO_METRIC_DEPTH_FORWARD_SPLAT"
            if go
            else "NO_GO_METRIC_DEPTH_FORWARD_SPLAT"
        ),
        "decision": (
            "freeze depth transport and run one external-bag confirmation"
            if go
            else "retain original frozen highres LTWM; metric depth does not solve action ordering"
        ),
        "protocol": {
            "depth_model_frozen": True,
            "strong_ltwm_frozen": True,
            "transport_scales_closed_form_fit_on_train_routes": True,
            "fit_eval_route_disjoint": True,
            "outcome_labels_loaded": False,
            "ate_or_objective_gradients": False,
            "sealed_validation_opened": False,
            "union65_rescored": False,
            "formal_v70_modified": False,
        },
        "sample": {
            "fit_windows": len(fit_rows),
            "fit_routes": sorted({row.route_id for row in fit_rows}),
            "eval_windows": len(eval_rows),
            "eval_routes": sorted({row.route_id for row in eval_rows}),
        },
        "depth": depth_summary,
        "fit": fit_summary,
        "pooled": {
            "strong_baseline": base,
            "metric_depth_forward_splat": metric,
            "changes": pooled_changes,
        },
        "per_route": per_route,
        "route_consistency": {
            "routes": route_count,
            "dual_improved_routes": dual_improved,
            "pairwise_improved_routes": pairwise_improved,
            "top1_not_worse_routes": top1_not_worse,
        },
        "checks": checks,
        "inputs": {
            "latent_root": str(latent_root),
            "latent_metadata_sha256": sha256_file(latent_root / "metadata.json"),
            "latent_splits_sha256": sha256_file(latent_root / "splits.json"),
            "depth_model": str(depth_model),
            "depth_model_sha256": sha256_file(
                depth_model / "model.safetensors"
            ),
            "depth_config_sha256": sha256_file(depth_model / "config.json"),
            "strong_checkpoint": str(strong_checkpoint),
            "strong_checkpoint_sha256": sha256_file(strong_checkpoint),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": checks,
                "pooled": result["pooled"],
                "route_consistency": result["route_consistency"],
                "output": str(output),
                "output_sha256": sha256_file(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                }
            ),
            file=sys.stderr,
        )
        raise
