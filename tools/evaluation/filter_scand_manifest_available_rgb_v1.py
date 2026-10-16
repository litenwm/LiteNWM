#!/usr/bin/env python3
"""Freeze a label-blind SCAND inner manifest filtered only by RGB availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.planning.build_scand100_nomad_k17_v1 import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--trajectory-root", type=Path, default=Path("/home/rainbow/nwm/data/scand")
    )
    args = parser.parse_args()
    audit_path = args.output.with_suffix(".audit.json")
    if args.output.exists() or audit_path.exists():
        raise FileExistsError(args.output)
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    kept = []
    excluded = []
    for row in rows:
        missing = []
        for timestep in [*map(int, row["history_timesteps"]), int(row["goal_t"])]:
            path = args.trajectory_root / str(row["trajectory_id"]) / f"{timestep}.jpg"
            if not path.is_file():
                missing.append(str(path))
        if missing:
            excluded.append(
                {
                    "window_uid": str(row["window_uid"]),
                    "trajectory_id": str(row["trajectory_id"]),
                    "current_t": int(row["current_t"]),
                    "missing_rgb": missing,
                }
            )
        else:
            kept.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in kept),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    audit = {
        "schema": "scand_inner_rgb_available_manifest_v1",
        "input": str(args.input.resolve()),
        "input_sha256": sha256(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "input_windows": len(rows),
        "kept_windows": len(kept),
        "excluded_windows": len(excluded),
        "exclusions": excluded,
        "filter_inputs": "filesystem RGB existence for history t-3..t and goal t+4 only",
        "labels_opened": False,
        "pose_opened": False,
        "model_outputs_used": False,
        "cem": False,
    }
    write_json(audit_path, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
