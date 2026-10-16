#!/usr/bin/env python3
"""Evaluate frozen universal-v2 selections on spent SCAND H4-100 outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.evaluation.evaluate_scand_fresh_outer_v1 import (  # noqa: E402
    method_metrics,
)
from tools.scoring.train_recon_dense_meta_dinov3_v1 import (  # noqa: E402
    file_sha256,
    write_json,
)


SCAND_ROOT = REPO_ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
UNIVERSAL_ROOT = REPO_ROOT / "outputs/universal_dualdomain_raw_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze",
        type=Path,
        default=UNIVERSAL_ROOT / "scand_spent_replay_v2/freeze/freeze.json",
    )
    parser.add_argument(
        "--universal-selected",
        type=Path,
        default=UNIVERSAL_ROOT
        / "scand_spent_replay_v2/freeze/selected_index.npz",
    )
    parser.add_argument(
        "--legacy-selected",
        type=Path,
        default=SCAND_ROOT / "prelabel_freeze/selected_index.npz",
    )
    parser.add_argument(
        "--native-selected",
        type=Path,
        default=SCAND_ROOT / "prelabel_freeze/native_v2_selected_index.npz",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=SCAND_ROOT / "outcomes/fresh_outer_outcomes.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=UNIVERSAL_ROOT / "scand_spent_replay_v2/result.json",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260721)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_replicates < 1000:
        raise ValueError("clustered bootstrap needs at least 1000 replicates")
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    if freeze.get("status") != "selections_frozen_for_spent_development_replay":
        raise RuntimeError("universal spent selection freeze is incomplete")
    if freeze.get("evidence_role") != (
        "historically spent development; never fresh outer"
    ):
        raise RuntimeError("spent evidence role changed")
    if file_sha256(args.universal_selected) != freeze["selected_index_sha256"]:
        raise RuntimeError("universal selections changed after freeze")
    if freeze.get("cem") is not False:
        raise RuntimeError("spent replay unexpectedly enables CEM")

    with np.load(args.outcomes, allow_pickle=False) as payload:
        record_key = payload["record_key"].astype(str)
        trajectory = payload["trajectory"].astype(str)
        route = payload["route_group"].astype(str)
        absolute = payload["absolute_metrics"].astype(np.float32)
        oracle = payload["objective_oracle_selected_index"].astype(np.int64)
    if len(record_key) != 100 or absolute.shape != (100, 17, 4):
        raise RuntimeError("spent outcome contract changed")

    methods: dict[str, np.ndarray] = {
        "incumbent": np.zeros(100, dtype=np.int64),
        "posthoc_objective_pool_oracle": oracle,
    }
    with np.load(args.native_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("native selection identity mismatch")
        methods["native_scorer_v2"] = payload[
            "native_v2_selected_index"
        ].astype(np.int64)
    with np.load(args.legacy_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("legacy selection identity mismatch")
        methods["legacy_balanced_support_plus_raw"] = payload[
            "domain_balanced__support_plus_raw"
        ].astype(np.int64)
    with np.load(args.universal_selected, allow_pickle=False) as payload:
        if not np.array_equal(payload["record_key"].astype(str), record_key):
            raise RuntimeError("universal selection identity mismatch")
        for key in payload.files:
            if key != "record_key":
                methods[key] = payload[key].astype(np.int64)

    metrics: dict[str, Any] = {
        name: method_metrics(
            selected,
            absolute,
            trajectory,
            route,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed + index * 10,
        )
        for index, (name, selected) in enumerate(methods.items())
    }
    report = {
        "schema": "universal_dualdomain_raw_v2_scand_spent_replay",
        "status": "spent_development_outcomes_evaluated",
        "evidence_role": "historically spent development; never fresh outer",
        "states": 100,
        "trajectories": int(len(np.unique(trajectory))),
        "route_groups": int(len(np.unique(route))),
        "selection_freeze": str(args.freeze),
        "selection_freeze_sha256": file_sha256(args.freeze),
        "universal_selected_sha256": file_sha256(args.universal_selected),
        "legacy_selected_sha256": file_sha256(args.legacy_selected),
        "native_selected_sha256": file_sha256(args.native_selected),
        "outcomes_sha256": file_sha256(args.outcomes),
        "methods": metrics,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "fixed_pool_only": True,
        "cem_cal_accessed": False,
        "cem": False,
    }
    write_json(args.output, report)
    compact = {
        name: {
            "promotions": value["promotions"],
            "coverage": value["promotion_fraction"],
            "dual_precision": value["dual_precision"],
            "wilson": value["dual_wilson_lower_95"],
            "harm_rate": value["harm_rate_per_promotion"],
            "absolute": value["absolute"],
        }
        for name, value in metrics.items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
