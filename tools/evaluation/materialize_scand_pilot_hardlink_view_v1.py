#!/usr/bin/env python3
"""Build an in-root hardlink image view for the strict-identity SCAND pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v2_strict_identity"
SOURCE_VIEW = ROOT / "outputs/cem_development_v1/scand_new_routefit10_v4_development_pilot_v1/trajectory_view"


def main() -> None:
    manifest = BASE / "pilot_manifest.windows.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = BASE / "trajectory_hardlink_view"
    if output.exists():
        raise FileExistsError(output)
    count = 0
    for row in rows:
        trajectory = row["trajectory_id"]
        directory = output / trajectory
        directory.mkdir(parents=True, exist_ok=True)
        timesteps = set(map(int, row["history_timesteps"])) | {int(row["current_t"]), int(row["goal_t"])}
        for timestep in timesteps:
            destination = directory / f"{timestep}.jpg"
            if destination.exists():
                continue
            source = (SOURCE_VIEW / trajectory / f"{timestep}.jpg").resolve(strict=True)
            os.link(source, destination)
            if source.stat().st_ino != destination.stat().st_ino:
                raise RuntimeError("hardlink inode mismatch")
            count += 1
    audit = {
        "schema": "scand_pilot_hardlink_view_v1",
        "status": "complete_label_free_zero_copy",
        "root": str(output),
        "files": count,
        "states": len(rows),
        "all_files_resolve_inside_root": all(
            (output / row["trajectory_id"] / f"{int(t)}.jpg").resolve().is_relative_to(output.resolve())
            for row in rows
            for t in [*row["history_timesteps"], row["goal_t"]]
        ),
        "labels_or_outcomes_opened": False,
        "reserve_accessed": False,
    }
    audit_path = BASE / "trajectory_hardlink_view.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
