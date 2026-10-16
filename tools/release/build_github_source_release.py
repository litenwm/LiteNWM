#!/usr/bin/env python3
"""Export a standalone GitHub source tree and ZIP, never model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = Path(__file__).parent / "github_templates"
NAME = "dinov3-ltwm-scorer-github"
TREES = (
    "src", "configs", "tests", "docs", "deploy", "robot",
    "tools", "data/split_lock",
    "data/inner_manifests", "data_splits",
)
ROOT_FILES = ("pipeline_contract.json", "EXTERNAL_DEPENDENCIES.md", ".gitattributes")
BLOCKED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "node_modules",
    "checkpoints", "outputs", "downloads", "artifacts", "external", "remote",
}
BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".engine",
    ".npz", ".npy", ".bag", ".db3", ".mcap", ".zip", ".gz", ".tar",
    ".pyc", ".pyo", ".orig", ".rej", ".log",
}
ALLOWED_SUFFIXES = {".py", ".sh", ".bash", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".toml", ".svg", ".mmd", ".drawio", ".sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acceptable(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if path.is_symlink() or any(p in BLOCKED_PARTS or p.startswith(".codex") for p in rel.parts):
        return False
    if path.name in {"local_assets.json", "inference.local.json"} or ":" in path.name:
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    # These are frozen evaluation indices, not model/data tensors.
    if path.suffix == ".pkl":
        return rel.parts[0] == "data_splits" and path.name == "navigation_eval.pkl"
    return path.suffix.lower() in ALLOWED_SUFFIXES or path.name in ROOT_FILES


def build(output: Path) -> dict:
    output = output.resolve()
    staging = output / NAME
    archive = output / f"{NAME}.zip"
    if staging.exists() or archive.exists():
        raise FileExistsError(f"Release already exists: {output}; choose a new --output directory")
    if not TEMPLATES.is_dir():
        raise FileNotFoundError(TEMPLATES)
    excluded = 0
    output.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    for tree in TREES:
        source = ROOT / tree
        if not source.is_dir():
            continue
        # Do not follow linked directories into machine-local assets.
        import os
        for directory, dirs, names in os.walk(source, followlinks=False):
            dirs[:] = [d for d in dirs if d not in BLOCKED_PARTS and not (Path(directory) / d).is_symlink()]
            for name in sorted(names):
                path = Path(directory) / name
                if not acceptable(path):
                    excluded += 1
                    continue
                if path.stat().st_size > 50 * 1024 * 1024:
                    raise ValueError(f"Unexpected large source file: {path.relative_to(ROOT)}")
                dest = staging / path.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
    for name in ROOT_FILES:
        if (ROOT / name).is_file():
            shutil.copy2(ROOT / name, staging / name)
    # Preserve historical prose, but make the new docs authoritative.
    for name in ("CHECKPOINTS.md", "LOCAL_ASSETS.md"):
        source = staging / "docs" / name
        if source.is_file():
            target = staging / "docs/archive" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copytree(TEMPLATES, staging, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # UTF-8 BOMs in two historical Python scripts are valid on disk, but
    # normalize them for linters and source consumers.
    for path in staging.rglob("*.py"):
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            path.write_bytes(raw[3:])
    (staging / "docs/LOCAL_ASSETS.md").write_text(
        "# Local assets\n\nSee [RUNNING.md](RUNNING.md) and [CHECKPOINTS.md](CHECKPOINTS.md).\n"
        "The historical server instructions are in [archive/LOCAL_ASSETS.md](archive/LOCAL_ASSETS.md).\n", encoding="utf-8")
    for directory, content in {
        "checkpoints": "# Checkpoints\n\nAll weights are external. See `docs/CHECKPOINTS.md`.\n",
        "runtime": "# Runtime assets\n\nSupport/calibration arrays are external. See `docs/RUNNING.md`.\n",
    }.items():
        (staging / directory).mkdir(exist_ok=True)
        (staging / directory / "README.md").write_text(content, encoding="utf-8")
    manifest_dir = staging / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    if (ROOT / "manifests/checkpoints.json").is_file():
        shutil.copy2(ROOT / "manifests/checkpoints.json", manifest_dir / "checkpoints.json")
    cfg = json.loads((staging / "configs/local_assets.example.json").read_text(encoding="utf-8"))
    cfg["dinov3"]["model_dir"] = "../external/dinov3-vitl16-pretrain-lvd1689m"
    for key, value in {
        "visualnav_root": "../external/visualnav-transformer",
        "diffusion_policy_root": "../external/diffusion_policy",
        "checkpoint": "../external/nomad/nomad.pth",
        "config": "../external/nomad/nomad.yaml",
    }.items():
        cfg["nomad"][key] = value
    for dataset in ("recon", "scand"):
        cfg["datasets"][dataset]["trajectory_root"] = f"../datasets/{dataset}"
    (staging / "configs/local_assets.example.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    (staging / "configs/inference.example.json").write_text(json.dumps({
        "wm_checkpoint": "../checkpoints/latest/ltwm_expansion/seed1701/best.pt",
        "scorer_checkpoint": "../checkpoints/latest/scorer_v2/seed1701/best.pt",
        "wm_sha256": "8970c387cd642027913c1cb01d5e2d2a2bfd6b58c94d1432e62880d3a434ac66",
        "scorer_sha256": "e7cef7ce68e97ab74db2a07474b0e0880eff06c3d4f9e2803be3a861461930ef",
        "inputs": "../inputs.npz",
    }, indent=2) + "\n", encoding="utf-8")
    import build_v70_tridomain_go2_orin_bundle_v2 as latest_bundle
    weights = []
    for path, expected_hash in latest_bundle.EXPECTED.items():
        if path.is_relative_to(latest_bundle.EXPORT / "ltwm"):
            destination = "checkpoints/latest/ltwm_expansion/" + path.relative_to(latest_bundle.EXPORT / "ltwm").as_posix()
        elif path.is_relative_to(latest_bundle.SCORER_ROOT):
            seed = path.name.split("_")[0]
            destination = f"checkpoints/latest/scorer_v2/{seed}/best.pt"
        else:
            destination = "runtime/v70_tridomain/" + path.relative_to(latest_bundle.EXPORT).as_posix()
        weights.append({"destination": destination, "sha256": expected_hash, "included": False})
    (manifest_dir / "latest_weights.json").write_text(json.dumps({
        "version": "sealed_tridomain_v70_20260729", "artifacts": weights,
        "split_lock_sha256": "66294673129bd20355e3aef55757bdf47bd915b54881c8c3ccfb01e057ca5614",
    }, indent=2) + "\n", encoding="utf-8")
    downloader = staging / "tools/release/download_dinov3_vitl16.sh"
    downloader.write_text(downloader.read_text(encoding="utf-8").replace(
        "/home/rainbow/models/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "external/dinov3-vitl16-pretrain-lvd1689m"), encoding="utf-8")
    # The original verifier assumes checkpoint payloads are bundled.
    shutil.copy2(TEMPLATES / "verify_source_release.py", staging / "tools/release/verify_repository.py")
    (staging / "verify_source_release.py").unlink()
    report = {
        "format": "github_source_only_v1", "weights_included": False,
        "raw_datasets_included": False, "robot_implementation": "placeholder",
        "excluded_files_in_selected_trees": excluded,
        "excluded_trees": ["artifacts", "checkpoints", "outputs", "downloads", "server_exports", "tools/remote", "qa", "results"],
        "research_scripts": "Preserved as historical references; some require external assets.",
        "source_version": "current working tree, including uncommitted source changes",
        "default_model_version": "sealed_tridomain_v70_20260729",
    }
    (staging / "RELEASE_INFO.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = sorted(p for p in staging.rglob("*") if p.is_file())
    bad = [p for p in files if p.suffix.lower() in BLOCKED_SUFFIXES]
    if bad:
        raise RuntimeError(f"Forbidden release payload: {bad}")
    records = [{"path": p.relative_to(staging).as_posix(), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]
    (manifest_dir / "source_release.json").write_text(json.dumps({"files": records}, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                zip_file.write(path, f"{NAME}/{path.relative_to(staging).as_posix()}")
    checksum = sha256(archive)
    (output / "SHA256SUMS.txt").write_text(f"{checksum}  {archive.name}\n", encoding="utf-8")
    return {"directory": str(staging), "archive": str(archive), "files": len(records) + 1, "zip_bytes": archive.stat().st_size, "sha256": checksum}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "deliverables/github_release_20261010")
    print(json.dumps(build(parser.parse_args().output), indent=2))
