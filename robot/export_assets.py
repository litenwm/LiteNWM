"""Author tool: export verified runtime assets from the desktop backup ZIP."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def destination(relative: str) -> str | None:
    if relative.startswith(("external/", "runtime/v70_tridomain/")):
        return relative
    if relative.startswith("checkpoints/project/dinov3_adapter/"):
        return relative
    for kind in ["ltwm_expansion", "scorer_v2"]:
        prefix = "checkpoints/project/" + kind + "/"
        if relative.startswith(prefix):
            return "checkpoints/latest/" + kind + "/" + relative[len(prefix):]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-prefix", default="/home/unitree/dinov3_ltwm_v70_tridomain_go2_orin_20260730_v5/")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Asset archive must be outside the GitHub source checkout")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    if partial.exists(): raise FileExistsError(partial)
    records = []
    with zipfile.ZipFile(args.backup_archive) as source:
        index = "audit/final_manifest.json" if "audit/final_manifest.json" in source.namelist() else "audit/manifest.json"
        manifest = json.loads(source.read(index))
        with tarfile.open(partial, "w") as target:
            for record in manifest:
                if not record["source"].startswith(args.source_prefix): continue
                name = destination(record["source"][len(args.source_prefix):])
                if name is None: continue
                raw = source.read(record["storage"])
                if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                    raise ValueError("Backup asset SHA mismatch: " + name)
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                target.addfile(info, io.BytesIO(raw))
                records.append({"destination": name, "sha256": record["sha256"],
                                "bytes": len(raw)})
    partial.rename(args.output)
    spec = {"schema": "litenwm_go2_assets_v1", "model_version": "sealed_tridomain_v70_20260729",
            "weights_in_git": False, "artifacts": records}
    (ROOT / "robot/assets_manifest.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(json.dumps({"asset_archive": str(args.output), "files": len(records),
                      "bytes": args.output.stat().st_size, "sha256_verified": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
