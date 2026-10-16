#!/usr/bin/env python3
"""Compare deployed V70, NoMaD, and NoMaD+NWM-S on one Go2 snapshot.

The comparison deliberately shares the exact NoMaD x16 pool produced inside
the V70 planner.  Pure NoMaD is the formal trimmed-mean75 incumbent.  NWM-S
reranks the same x16 pool with the original CDiT-S/2 100k EMA checkpoint,
250-step rollouts, and final-frame LPIPS-Alex.  Nothing in this script can
publish a robot command.
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEPLOY = ROOT / "deploy/v70_go2_orin"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(DEPLOY)]


def install_ros_import_stubs() -> None:
    """Allow importing V70's pure feature functions without a ROS install."""
    try:
        __import__("rclpy")
        return
    except ModuleNotFoundError:
        pass

    class OfflineNode:
        pass

    class OfflineQoSProfile:
        def __init__(self, **_: Any) -> None:
            pass

    class OfflineDurabilityPolicy:
        VOLATILE = 0

    class OfflineHistoryPolicy:
        KEEP_LAST = 0

    class OfflineReliabilityPolicy:
        RELIABLE = 0

    rclpy = types.ModuleType("rclpy")
    rclpy_node = types.ModuleType("rclpy.node")
    rclpy_qos = types.ModuleType("rclpy.qos")
    nav_msgs = types.ModuleType("nav_msgs")
    nav_msgs_msg = types.ModuleType("nav_msgs.msg")
    rclpy_node.Node = OfflineNode
    rclpy_qos.DurabilityPolicy = OfflineDurabilityPolicy
    rclpy_qos.HistoryPolicy = OfflineHistoryPolicy
    rclpy_qos.QoSProfile = OfflineQoSProfile
    rclpy_qos.ReliabilityPolicy = OfflineReliabilityPolicy
    nav_msgs_msg.Odometry = type("OfflineOdometry", (), {})
    nav_msgs.msg = nav_msgs_msg
    rclpy.node = rclpy_node
    rclpy.qos = rclpy_qos
    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.node": rclpy_node,
            "rclpy.qos": rclpy_qos,
            "nav_msgs": nav_msgs,
            "nav_msgs.msg": nav_msgs_msg,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--nwm-root", type=Path, default=Path("/home/rainbow/rrnwm"))
    parser.add_argument(
        "--nwm-config",
        type=Path,
        default=Path("/home/rainbow/rrnwm/config/nwm_cdit_s.yaml"),
    )
    parser.add_argument(
        "--nwm-checkpoint",
        type=Path,
        default=Path(
            "/home/rainbow/rrnwm/logs/nwm_cdit_s/checkpoints/100000.pth.tar"
        ),
    )
    parser.add_argument(
        "--nwm-vae",
        type=Path,
        default=Path("/home/rainbow/rrnwm/stabilityai/sd-vae-ft-ema"),
    )
    parser.add_argument("--nwm-seed", type=int, default=0)
    parser.add_argument("--nwm-chunk-size", type=int, default=2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def frame_paths(frame_dir: Path) -> list[Path]:
    result = []
    for index in range(4):
        candidates = [
            frame_dir / f"history_{index}.png",
            frame_dir / f"history_{index}.jpg",
            frame_dir / f"history_{index}.jpeg",
        ]
        match = next((path for path in candidates if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"missing history_{index} in {frame_dir}")
        result.append(match)
    return result


def open_rgb(paths: Sequence[Path]) -> list[Image.Image]:
    output = []
    for path in paths:
        with Image.open(path) as image:
            output.append(image.convert("RGB").copy())
    return output


def direction_label(value: float, tolerance: float = 0.03) -> str:
    if value > tolerance:
        return "left"
    if value < -tolerance:
        return "right"
    return "straight"


def method_payload(
    name: str,
    actions: np.ndarray,
    waypoints: np.ndarray,
    **extra: Any,
) -> dict[str, Any]:
    action = np.asarray(actions, dtype=np.float32)
    waypoint = np.asarray(waypoints, dtype=np.float32)
    endpoint = waypoint[-1]
    return {
        "name": name,
        "actions_h4_m_rad": action.tolist(),
        "waypoints_h4_m": waypoint.tolist(),
        "first_step": {
            "forward_m": float(action[0, 0]),
            "lateral_left_m": float(action[0, 1]),
            "yaw_left_rad": float(action[0, 2]),
            "yaw_left_deg": math.degrees(float(action[0, 2])),
            "direction": direction_label(float(action[0, 1])),
        },
        "endpoint": {
            "forward_m": float(endpoint[0]),
            "lateral_left_m": float(endpoint[1]),
            "direction": direction_label(float(endpoint[1])),
        },
        **extra,
    }


def nwm_deltas_from_nomad(proposals: np.ndarray) -> torch.Tensor:
    """Exact original NWM-S NoMaD action adapter used in matched reranking."""
    from planning.nomad_action_adapter import nomad_waypoints_to_local_actions

    proposals = np.asarray(proposals, dtype=np.float32)
    minimum = np.asarray([-2.5, -4.0], dtype=np.float32)
    maximum = np.asarray([5.0, 4.0], dtype=np.float32)
    normalized_xy = (proposals - minimum) / (maximum - minimum)
    normalized_xy = normalized_xy * np.float32(2.0) - np.float32(1.0)
    local_actions = nomad_waypoints_to_local_actions(
        proposals,
        dataset="recon",
        horizon=proposals.shape[1],
    )
    cumulative_yaw = np.cumsum(local_actions[..., 2], axis=1)
    cumulative = np.concatenate(
        (normalized_xy, cumulative_yaw[..., None]),
        axis=-1,
    )
    origin = np.zeros((len(proposals), 1, 3), dtype=np.float32)
    return torch.from_numpy(
        np.diff(np.concatenate((origin, cumulative), axis=1), axis=1)
    )


def run_nwm_s(
    *,
    args: argparse.Namespace,
    observations: Sequence[Image.Image],
    goal: Image.Image,
    proposals: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Run original CDiT-S/2 250-step final-frame LPIPS reranking."""
    nwm_root = args.nwm_root.resolve()
    sys.path.insert(0, str(nwm_root))
    old_cwd = Path.cwd()
    os.chdir(nwm_root)
    try:
        import lpips
        import misc
        from diffusers.models import AutoencoderKL
        from diffusion import create_diffusion
        from isolated_nwm_infer import model_forward_wrapper
        from models import CDiT_models

        config = yaml.safe_load(args.nwm_config.read_text(encoding="utf-8"))
        latent_size = int(config["image_size"]) // 8
        model = CDiT_models[config["model"]](
            context_size=4,
            input_size=latent_size,
        )
        checkpoint = torch.load(
            args.nwm_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["ema"], strict=True)
        device = torch.device("cuda:0")
        model.eval().to(device)
        diffusion = create_diffusion("250")
        vae = AutoencoderKL.from_pretrained(
            str(args.nwm_vae),
            local_files_only=True,
        ).to(device)
        vae.eval()
        loss_fn = lpips.LPIPS(net="alex").to(device)
        loss_fn.eval()

        observation = torch.stack([misc.transform(image) for image in observations])
        goal_tensor = misc.transform(goal).to(device)
        deltas = nwm_deltas_from_nomad(proposals)

        torch.manual_seed(args.nwm_seed)
        torch.cuda.manual_seed_all(args.nwm_seed)
        losses = []
        started = time.monotonic()
        for start in range(0, len(deltas), args.nwm_chunk_size):
            chunk = deltas[start : start + args.nwm_chunk_size].to(device)
            current = observation.unsqueeze(0).repeat(len(chunk), 1, 1, 1, 1)
            current = current.to(device)
            for step in range(chunk.shape[1]):
                prediction = model_forward_wrapper(
                    (model, diffusion, vae),
                    current,
                    chunk[:, step : step + 1],
                    1,
                    latent_size,
                    num_cond=4,
                    device=device,
                )
                current = torch.cat(
                    (current[:, 1:], prediction.unsqueeze(1)),
                    dim=1,
                )
            with torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=torch.bfloat16,
            ):
                target = goal_tensor.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
                loss = loss_fn(prediction, target).flatten().float().cpu()
            losses.append(loss)
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        loss_array = torch.cat(losses).detach().numpy().astype(np.float32)
        selected = int(np.argmin(loss_array))
        return loss_array, selected, elapsed
    finally:
        os.chdir(old_cwd)


def svg_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def svg_path(waypoints: Sequence[Sequence[float]]) -> str:
    points = [(960.0, 1000.0)]
    for forward, lateral in waypoints:
        x = 960.0 - float(lateral) * 650.0
        y = 1000.0 - float(forward) * 470.0
        points.append((x, y))
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
        for index, (x, y) in enumerate(points)
    )


def write_overlay(
    path: Path,
    *,
    current_frame: Path,
    goal: Path,
    methods: Sequence[dict[str, Any]],
) -> None:
    colors = {
        "V70": "#00E5FF",
        "NoMaD": "#FFD54F",
        "NoMaD+NWM-S": "#FF4FD8",
    }
    paths = []
    dots = []
    legend = []
    for index, method in enumerate(methods):
        name = str(method["name"])
        color = colors[name]
        path_value = svg_path(method["waypoints_h4_m"])
        paths.append(
            f'<path d="{path_value}" fill="none" stroke="{color}" '
            'stroke-width="12" stroke-linecap="round" stroke-linejoin="round" '
            'marker-end="url(#arrow)" opacity="0.94"/>'
        )
        for forward, lateral in method["waypoints_h4_m"][:-1]:
            x = 960.0 - float(lateral) * 650.0
            y = 1000.0 - float(forward) * 470.0
            dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}"/>'
            )
        endpoint = method["endpoint"]
        legend.append(
            f'<rect x="1100" y="{690 + index * 78}" width="32" height="10" '
            f'rx="5" fill="{color}"/>'
            f'<text x="1150" y="{710 + index * 78}" class="legend">{name}: '
            f'x={endpoint["forward_m"]:.2f} m, '
            f'y={endpoint["lateral_left_m"]:+.2f} m</text>'
        )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080">
  <defs>
    <filter id="shadow"><feDropShadow dx="0" dy="3" stdDeviation="5" flood-opacity="0.75"/></filter>
    <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
      <path d="M 0 0 L 12 6 L 0 12 z" fill="context-stroke"/>
    </marker>
    <style>
      .title {{ font: 700 34px Arial, sans-serif; fill: white; }}
      .sub {{ font: 500 23px Arial, sans-serif; fill: #E9F1F7; }}
      .legend {{ font: 700 25px Arial, sans-serif; fill: white; }}
      .small {{ font: 500 20px Arial, sans-serif; fill: #E9F1F7; }}
    </style>
  </defs>
  <image href="{svg_data_uri(current_frame)}" x="0" y="0" width="1920" height="1080" preserveAspectRatio="xMidYMid slice"/>
  <rect x="0" y="0" width="1920" height="1080" fill="#00131F" opacity="0.13"/>
  <rect x="45" y="40" width="1120" height="175" rx="24" fill="#061A27" opacity="0.84" filter="url(#shadow)"/>
  <text x="80" y="93" class="title">Same Go2 snapshot · predicted H4 motion trends</text>
  <text x="80" y="137" class="sub">Background: current view (history_3, reset heading). Curves are schematic ground-plane projections.</text>
  <text x="80" y="178" class="small">Image left = robot left · image right = robot right · arrows start at the robot.</text>
  <rect x="1065" y="645" width="810" height="305" rx="24" fill="#061A27" opacity="0.84" filter="url(#shadow)"/>
  {''.join(legend)}
  <image href="{svg_data_uri(goal)}" x="1510" y="45" width="360" height="203" preserveAspectRatio="xMidYMid slice"/>
  <rect x="1510" y="45" width="360" height="203" fill="none" stroke="white" stroke-width="5"/>
  <rect x="1510" y="210" width="360" height="38" fill="#061A27" opacity="0.88"/>
  <text x="1530" y="237" class="small">Correct goal · SHA 4bd80f…</text>
  <circle cx="960" cy="1000" r="22" fill="white" stroke="#061A27" stroke-width="7"/>
  {''.join(paths)}
  {''.join(dots)}
  <path d="M 1000 1015 L 1420 1015" fill="none" stroke="#6BFF95" stroke-width="9" stroke-dasharray="22 14" marker-end="url(#arrow)" opacity="0.96"/>
  <text x="1080" y="1055" style="font:700 22px Arial,sans-serif;fill:#6BFF95">Visual goal cue (not model output): RIGHT</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for required in (
        args.frame_dir,
        args.manifest,
        args.goal,
        args.nwm_root,
        args.nwm_config,
        args.nwm_checkpoint,
        args.nwm_vae,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.nwm_chunk_size < 1:
        raise ValueError("--nwm-chunk-size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    args.output_dir.mkdir(parents=True)
    images_path = frame_paths(args.frame_dir)
    observations = open_rgb(images_path)
    with Image.open(args.goal) as image:
        goal_image = image.convert("RGB").copy()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    from planning.nomad_action_adapter import (
        integrate_local_actions,
        nomad_waypoints_to_local_actions,
    )
    from tools.evaluation.build_recon100_k17_source_v1 import trimmed_mean75
    install_ros_import_stubs()
    from v70_online_feature_shadow import PoseSample
    from v70_persistent_planner import PersistentV70Planner

    poses = [
        PoseSample(
            receive_time=float(frame["pose"]["receive_time_monotonic"]),
            x=float(frame["pose"]["x_m"]),
            y=float(frame["pose"]["y_m"]),
            yaw=float(frame["pose"]["yaw_rad"]),
        )
        for frame in manifest["frames"]
    ]

    v70_started = time.monotonic()
    planner = PersistentV70Planner()
    v70_result = planner.plan(
        observations=observations,
        goal_image=goal_image,
        poses=poses,
    )
    torch.cuda.synchronize()
    v70_elapsed = time.monotonic() - v70_started
    references = np.asarray(v70_result["nomad"]["candidates"], dtype=np.float32)[
        :16, :4, :2
    ]
    np.save(args.output_dir / "shared_nomad_x16_waypoint_units.npy", references)

    incumbent = trimmed_mean75(references)
    nomad_actions = nomad_waypoints_to_local_actions(
        incumbent,
        dataset="recon",
    )
    nomad_waypoints = integrate_local_actions(nomad_actions)
    v70_actions = np.asarray(v70_result["selected_actions_h4"], dtype=np.float32)
    v70_waypoints = integrate_local_actions(v70_actions)

    v70_method = method_payload(
        "V70",
        v70_actions,
        v70_waypoints,
        abstained=bool(v70_result["abstained"]),
        selection_reason=str(v70_result["selection_reason"]),
        archive_energy=float(v70_result["archive_energy"]),
        archive_source_round=int(v70_result["archive_source_round"]),
        wall_seconds=float(v70_elapsed),
    )
    nomad_method = method_payload(
        "NoMaD",
        nomad_actions,
        nomad_waypoints,
        selection_rule="trimmed_mean75_of_shared_nomad_x16",
    )

    # Release the larger V70 stack before loading the original NWM-S model.
    del planner
    gc.collect()
    torch.cuda.empty_cache()

    nwm_losses, nwm_index, nwm_elapsed = run_nwm_s(
        args=args,
        observations=observations,
        goal=goal_image,
        proposals=references,
    )
    nwm_waypoint_units = references[nwm_index]
    nwm_actions = nomad_waypoints_to_local_actions(
        nwm_waypoint_units,
        dataset="recon",
    )
    nwm_waypoints = integrate_local_actions(nwm_actions)
    ordered_losses = np.sort(nwm_losses)
    nwm_method = method_payload(
        "NoMaD+NWM-S",
        nwm_actions,
        nwm_waypoints,
        selected_nomad_index=int(nwm_index),
        selected_lpips=float(nwm_losses[nwm_index]),
        lpips_margin_to_second=float(ordered_losses[1] - ordered_losses[0]),
        nwm_wall_seconds=float(nwm_elapsed),
        protocol="CDiT-S/2_100k_EMA_250step_final_frame_LPIPS_Alex_repeat1",
    )

    methods = [v70_method, nomad_method, nwm_method]
    endpoint_signs = [direction_label(item["endpoint"]["lateral_left_m"]) for item in methods]
    first_signs = [direction_label(item["first_step"]["lateral_left_m"], 0.03) for item in methods]
    payload = {
        "schema": "go2_snapshot_planner_comparison_v1",
        "safety": "offline_only_no_robot_interface_or_command_publisher",
        "inputs": {
            "goal": str(args.goal.resolve()),
            "goal_sha256": sha256(args.goal),
            "frames": [str(path.resolve()) for path in images_path],
            "frame_sha256": [sha256(path) for path in images_path],
            "manifest": str(args.manifest.resolve()),
            "current_frame": str(images_path[-1].resolve()),
        },
        "shared_nomad_pool": {
            "count": 16,
            "shape": list(references.shape),
            "seed": int(v70_result["nomad"].get("seed", 20260730)),
        },
        "methods": methods,
        "agreement": {
            "first_step_directions": first_signs,
            "endpoint_directions": endpoint_signs,
            "all_first_step_agree": len(set(first_signs)) == 1,
            "all_endpoint_agree": len(set(endpoint_signs)) == 1,
        },
        "nwm_s_losses": nwm_losses.tolist(),
    }
    atomic_json(args.output_dir / "comparison.json", payload)
    np.savez_compressed(
        args.output_dir / "comparison_arrays.npz",
        shared_nomad_x16=references,
        v70_actions_h4=v70_actions,
        nomad_actions_h4=nomad_actions,
        nomad_nwm_s_actions_h4=nwm_actions,
        nwm_s_lpips=nwm_losses,
    )
    write_overlay(
        args.output_dir / "current_frame_action_trends.svg",
        current_frame=images_path[-1],
        goal=args.goal,
        methods=methods,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
