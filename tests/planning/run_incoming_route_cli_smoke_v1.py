"""Exercise the builder and readiness checker through their CLI entrypoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tests.planning.test_incoming_route_preflight_v1 import registry, write_route


ROOT = Path(__file__).resolve().parents[2]


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ".:src"
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="incoming_route_cli_smoke_v1_"))
    development_route = write_route(
        root / "dev", role="development", route="cli_dev", trajectory="cli_dev_traj"
    )
    validation_route = write_route(
        root / "val",
        role="one_time_validation",
        route="cli_val",
        trajectory="cli_val_traj",
    )
    development_manifest = root / "dev.windows.jsonl"
    validation_manifest = root / "val.windows.jsonl"
    for route, manifest in (
        (development_route, development_manifest),
        (validation_route, validation_manifest),
    ):
        run(
            "tools/planning/build_incoming_scand_route_manifest_v1.py",
            "--mode",
            "build",
            "--route-dir",
            str(route),
            "--manifest",
            str(manifest),
        )
        run(
            "tools/planning/build_incoming_scand_route_manifest_v1.py",
            "--mode",
            "validate",
            "--route-dir",
            str(route),
            "--manifest",
            str(manifest),
        )
    used = registry(root / "used.json")
    forbidden = registry(root / "forbidden.json")
    output = root / "readiness.json"
    completed = run(
        "tools/planning/check_proposal_tail_readiness_v1.py",
        "--development-manifest",
        str(development_manifest),
        "--validation-manifest",
        str(validation_manifest),
        "--used-identity-registry",
        str(used),
        "--forbidden-identity-registry",
        str(forbidden),
        "--output",
        str(output),
    )
    result = json.loads(completed.stdout)
    assert result["status"] == "ready_prelabel"
    assert json.loads(output.read_text(encoding="utf-8"))["ready"] is True
    print("CLI builder build+validate and readiness roundtrip passed")


if __name__ == "__main__":
    main()
