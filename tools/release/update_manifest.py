"""Refresh the maintainer's source-only inventory after intentional code edits."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SKIP = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
        ".tmp", "outputs", "external", "build", "dist"}


def main() -> int:
    records = []
    for file in sorted(ROOT.rglob("*")):
        if not file.is_file(): continue
        rel = file.relative_to(ROOT)
        if any(p in SKIP or p.endswith(".egg-info") for p in rel.parts): continue
        if rel.as_posix() in {"manifests/source_release.json", "configs/local_assets.json", "configs/inference.local.json"}: continue
        if rel.parts[0] in {"runtime", "checkpoints"} and rel.name != "README.md": continue
        records.append({"path": rel.as_posix(), "size_bytes": file.stat().st_size,
                        "sha256": hashlib.sha256(file.read_bytes()).hexdigest()})
    output = {"format": "github_source_only_v2", "model_version": "sealed_tridomain_v70_20260729",
              "robot_execution": "fresh_h1_once_no_action_reuse", "files": records}
    (ROOT / "manifests/source_release.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Updated", len(records), "source records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
