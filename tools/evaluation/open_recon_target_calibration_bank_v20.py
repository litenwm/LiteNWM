#!/usr/bin/env python3
"""Open outcomes for preregistered development banks while sealing reserve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.evaluation.build_recon100_k17_source_v1 import (  # noqa: E402
    metric_labels,
)
from tools.evaluation.build_recon_inner_k17_source_v2 import (  # noqa: E402
    local_ground_truth,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
    write_npz,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs=3, required=True)
    parser.add_argument("--sources", type=Path, nargs=3, required=True)
    parser.add_argument("--outputs", type=Path, nargs=3, required=True)
    parser.add_argument("--reserve-manifest", type=Path, required=True)
    parser.add_argument("--reserve-source", type=Path, required=True)
    parser.add_argument("--trajectory-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    if args.audit.exists() or any(path.exists() for path in args.outputs):
        raise FileExistsError(args.audit)
    reserve_rows = rows(args.reserve_manifest)
    reserve_record = {
        f"{value['trajectory_id']}:{int(value['current_t'])}"
        for value in reserve_rows
    }
    with np.load(args.reserve_source, allow_pickle=False) as payload:
        reserve_source_record = set(payload["record_key"].astype(str))
    if reserve_record != reserve_source_record or len(reserve_record) != 100:
        raise RuntimeError("reserve identity contract changed")

    records = []
    used_records: set[str] = set()
    used_routes: set[str] = set()
    for index, (manifest, source, output) in enumerate(
        zip(args.manifests, args.sources, args.outputs)
    ):
        bank_rows = rows(manifest)
        if len(bank_rows) != 100:
            raise RuntimeError("target calibration bank is not H4-100")
        record_key = np.asarray(
            [
                f"{value['trajectory_id']}:{int(value['current_t'])}"
                for value in bank_rows
            ]
        )
        route = {
            str(value["route_group"]) for value in bank_rows
        }
        if (
            set(record_key) & reserve_record
            or set(record_key) & used_records
            or route & used_routes
        ):
            raise RuntimeError("bank/reserve identity overlap")
        with np.load(source, allow_pickle=False) as payload:
            if not np.array_equal(
                record_key, payload["record_key"].astype(str)
            ):
                raise RuntimeError("bank source identity mismatch")
            source_arrays = {
                key: payload[key].copy() for key in payload.files
            }
        ground_truth = np.stack(
            [
                local_ground_truth(
                    args.trajectory_root
                    / str(value["trajectory_id"])
                    / "traj_data.pkl",
                    int(value["current_t"]),
                )
                for value in bank_rows
            ]
        ).astype(np.float32)
        labels = metric_labels(
            source_arrays["candidates"],
            ground_truth,
            0.25,
            0.4,
            0.6,
        )
        write_npz(
            output,
            **source_arrays,
            targets=labels["targets"],
            absolute_metrics=labels["values"],
            ground_truth_local_m=ground_truth,
        )
        records.append(
            {
                "bank": index,
                "manifest": str(manifest),
                "manifest_sha256": file_sha256(manifest),
                "source_label_free": str(source),
                "source_label_free_sha256": file_sha256(source),
                "source_labeled": str(output),
                "source_labeled_sha256": file_sha256(output),
                "states": len(record_key),
                "routes": len(route),
            }
        )
        used_records.update(record_key.tolist())
        used_routes.update(route)

    audit = {
        "schema": "recon_target_calibration_bank_v20_outcome_open",
        "status": "development_bank_labels_opened_reserve_sealed",
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "states": len(used_records),
        "routes": len(used_routes),
        "banks": records,
        "reserve": {
            "manifest": str(args.reserve_manifest),
            "manifest_sha256": file_sha256(args.reserve_manifest),
            "source_label_free": str(args.reserve_source),
            "source_label_free_sha256": file_sha256(args.reserve_source),
            "states": len(reserve_record),
            "labels_opened": False,
            "outcomes_accessed": False,
            "role": "final_confirmation_reserved_do_not_touch",
        },
        "bank_reserve_record_overlap": 0,
        "bank_reserve_route_overlap": 0,
        "formal_claim_allowed": False,
        "cem": False,
    }
    write_json(args.audit, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
