"""Run the deployed RGB planner on four saved images, without ROS/control."""
from __future__ import annotations
import argparse
import json
import os
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def json_safe(value):
    """Represent missing finite-safe energy as null, not nonstandard Infinity."""
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if hasattr(value, "tolist"): return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", nargs=4, required=True, type=Path)
    parser.add_argument("--goal", required=True, type=Path)
    parser.add_argument("--poses", required=True, type=Path,
                        help="JSON list of four {receive_time,x,y,yaw} observations")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cem-iterations", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asset-root", type=Path, default=ROOT,
                        help="Root of an already SHA-verified installed asset layout")
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "deploy/v70_go2_orin")]
    from PIL import Image
    import numpy as np
    from v70_persistent_planner_fast import PersistentV70Planner, PersistentPlannerPaths
    from v70_pose_features import PoseSample
    poses = [PoseSample(**m) for m in json.loads(args.poses.read_text(encoding="utf-8"))]
    if len(poses) != 4:
        raise ValueError("Exactly four aligned poses required")
    observations = [Image.open(path).convert("RGB") for path in args.history]
    goal = Image.open(args.goal).convert("RGB")
    asset_root = args.asset_root.expanduser().resolve()
    os.environ.setdefault("V70_ACTION_TAIL_SUPPORT", str(
        asset_root / "runtime/v70_tridomain/action_support/coherent_turn_tail_v1.npz"))
    paths = PersistentPlannerPaths.defaults(root=asset_root)
    planner = PersistentV70Planner(paths=paths, device=args.device, adaptive_cem=True,
                                  enable_fixed_pool_fast_path=True,
                                  enable_dino_feature_cache=True)
    result = planner.plan(observations=observations, goal_image=goal, poses=poses,
                          maximum_cem_iterations=args.cem_iterations)
    def encode(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    # Do not serialize internal live CUDA tensors returned for the ROS node.
    result = planner.summary(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(json_safe(result), default=encode, indent=2,
                                      allow_nan=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
