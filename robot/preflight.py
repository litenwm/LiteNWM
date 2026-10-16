"""Read-only deployment checks. Never imports a Unitree control client."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def check_assets(root: Path) -> list[str]:
    errors = []
    spec = json.loads((root / "robot/assets_manifest.json").read_text(encoding="utf-8"))
    for item in spec["artifacts"]:
        path = root / item["destination"]
        if not path.is_file(): errors.append("Missing asset: " + item["destination"])
        elif sha256(path) != item["sha256"]: errors.append("Asset SHA mismatch: " + item["destination"])
    config_path = root / "configs/local_assets.json"
    if not config_path.is_file():
        return errors + ["Missing configs/local_assets.json; run robot/install_assets.py"]
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for name in ["visualnav_root", "diffusion_policy_root"]:
            value = Path(config["nomad"][name]).expanduser()
            path = value if value.is_absolute() else config_path.parent / value
            if not path.is_dir(): errors.append("Missing external code: " + name)
    except (KeyError, ValueError) as e:
        errors.append("Invalid local asset configuration: " + str(e))
    return errors


def check_route(path: Path, mode: str) -> list[str]:
    sys.path.insert(0, str(ROOT / "deploy/v70_go2_orin"))
    from v70_closed_loop_core import RouteManifest
    errors = []
    try:
        route = RouteManifest.load(path, require_calibrated=(
            mode == "live" and os.environ.get("V70_OPERATOR_STOP_ONLY", "0") != "1"))
        if len(route.goals) != 1: errors.append("Public runner requires one image goal")
        # The node still checks calibration/explicit manual-stop authorization.
    except Exception as e:
        errors.append("Invalid route: " + str(e))
    return errors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", action="store_true")
    p.add_argument("--hardware", action="store_true")
    p.add_argument("--route", type=Path)
    p.add_argument("--mode", choices=("shadow", "live"), default="shadow")
    args = p.parse_args()
    errors = check_assets(ROOT) if args.assets else []
    if args.route: errors += check_route(args.route, args.mode)
    if args.hardware:
        if not sys.platform.startswith("linux"): errors.append("Go2 runtime requires Linux")
        for name in ["rclpy", "nav_msgs.msg", "geometry_msgs.msg", "sensor_msgs.msg",
                     "sensor_msgs_py.point_cloud2", "unitree_sdk2py.core.channel",
                     "unitree_sdk2py.go2.video.video_client", "cv2", "torchvision",
                     "diffusers", "transformers", "efficientnet_pytorch", "einops"]:
            try:
                importlib.import_module(name)
            except (ImportError, OSError, RuntimeError) as exc:
                errors.append("Runtime import failed: " + name + ": " + str(exc))
        try:
            from ament_index_python.packages import get_package_prefix
            for middleware in {os.environ.get("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"),
                               "rmw_fastrtps_cpp"}:
                try: get_package_prefix(middleware)
                except LookupError: errors.append("Missing ROS middleware: " + middleware)
        except ImportError:
            errors.append("ROS ament_index_python unavailable; source ROS setup first")
        if importlib.util.find_spec("torch") is None:
            errors.append("Missing board PyTorch")
        else:
            import torch
            if not torch.cuda.is_available(): errors.append("Go2/Orin runtime requires CUDA-enabled board PyTorch")
        interface = os.environ.get("V70_INTERFACE", "enP8p1s0")
        if not (Path("/sys/class/net") / interface).is_dir(): errors.append("Missing network interface: " + interface)
        stop = Path(os.environ.get("GO2_NAV_STOP_FILE", "/tmp/go2_nav_stop"))
        arm = Path(os.environ.get("GO2_MOTION_ARM_FILE", "/run/go2-safety/ARMED"))
        if not stop.is_file(): errors.append("STOP file must already exist: " + str(stop))
        if arm.exists(): errors.append("ARM file must be absent at startup: " + str(arm))
        if args.mode == "live" and (not arm.parent.is_dir() or not os.access(arm.parent, os.W_OK)):
            errors.append("ARM parent missing or not writable: " + str(arm.parent) +
                          "; configure GO2_MOTION_ARM_FILE to a prepared writable directory")
    print(json.dumps({"passed": not errors, "errors": errors,
                      "read_only": True, "motion_api_imported": False}, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
