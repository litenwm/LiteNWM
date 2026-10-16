"""Install an author-provided asset TAR into a checkout and verify its hashes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

try:
    from .preflight import check_assets
except ImportError:
    from preflight import check_assets

ROOT = Path(__file__).resolve().parents[1]


def install(archive: Path, root: Path) -> None:
    # Never extract with arbitrary absolute paths, links or special files.
    with tarfile.open(archive, "r|") as t:
        for member in t:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe asset member: " + member.name)
            if not member.isfile():
                raise ValueError("Only regular asset files allowed: " + member.name)
            if rel.parts[0] not in {"checkpoints", "runtime", "external"}:
                raise ValueError("Unexpected asset destination: " + member.name)
            destination = root.joinpath(*rel.parts)
            # Prevent traversal through an existing symlink in the checkout.
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError("Destination leaves checkout: " + member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(member) as src, destination.open("wb") as target:
                shutil.copyfileobj(src, target, 8 * 1024 * 1024)
    example = root / "robot/local_assets.example.json"
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs/local_assets.json").write_bytes(example.read_bytes())
    errors = check_assets(root)
    if errors:
        raise RuntimeError("Asset verification failed:\n" + "\n".join(errors))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    args = p.parse_args()
    install(args.archive, ROOT)
    print("Installed and SHA-verified Go2 assets; no motion API imported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
