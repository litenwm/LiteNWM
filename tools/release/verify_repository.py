"""Verify the source release inventory without any third-party dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    records = json.loads((root / "manifests/source_release.json").read_text(encoding="utf-8"))["files"]
    failures = []
    for item in records:
        path = root / item["path"]
        if not path.is_file():
            failures.append(f"Missing: {item['path']}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != item["size_bytes"] or digest != item["sha256"]:
            failures.append(f"Changed: {item['path']}")
    blocked = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".onnx", ".engine", ".npz", ".npy"}
    for path in root.rglob("*"):
        if any(part in {".git", ".venv", "__pycache__", ".tmp", ".pytest_cache"} or part.endswith(".egg-info") for part in path.relative_to(root).parts):
            continue
        if path.is_file() and path.suffix.lower() in blocked:
            failures.append(f"External weight/array present: {path.relative_to(root)}")
    if failures:
        print("FAIL\n" + "\n".join(failures))
        return 1
    print(f"PASS: {len(records)} source files verified; no model weights or tensor arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
