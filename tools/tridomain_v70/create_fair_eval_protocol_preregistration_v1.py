#!/usr/bin/env python3
"""Freeze the tri-domain V70 fair-evaluation protocol before action selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seal-record", type=Path, required=True)
    parser.add_argument("--active-eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    result_path = args.result.resolve(strict=True)
    seal_path = args.seal_record.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    result = read_json(result_path)
    seal = read_json(seal_path)
    if result.get("seal_status") != "RELAXED_FEASIBLE_V1":
        raise RuntimeError("result is not the fixed relaxed V1 seal")
    if result.get("strict_all_domains_feasible"):
        raise RuntimeError("relaxed fallback incorrectly claims strict feasibility")
    if not result.get("relaxed_all_domains_feasible_v1"):
        raise RuntimeError("result does not satisfy fixed relaxed V1")
    if result.get("test_accessed") or result.get("outer_val_accessed"):
        raise RuntimeError("result seal reports external label access")
    if seal.get("status") != "RELAXED_FEASIBLE_V1" or seal.get("test_accessed"):
        raise RuntimeError("seal record is not clean")
    active_root = args.active_eval_root.resolve()
    selection_outputs = []
    if active_root.exists():
        selection_outputs = [
            str(path)
            for pattern in ("selected_actions.npz", "results.json", "compact_results.json")
            for path in active_root.rglob(pattern)
        ]
    if selection_outputs:
        raise RuntimeError(
            f"action selection or metrics already exist: {selection_outputs}"
        )
    value = {
        "schema": "tridomain_v70_fair_eval_protocol_preregistration_v1",
        "status": "frozen_before_external_fair_selection_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_result": str(result_path),
        "frozen_result_sha256": sha256(result_path),
        "seal_record": str(seal_path),
        "seal_record_sha256": sha256(seal_path),
        "seal_status": "RELAXED_FEASIBLE_V1",
        "strict_feasible": False,
        "relaxed_feasible_v1": True,
        "model_domain_index": {"recon": 0, "scand": 1, "sacson": 2},
        "cem_contract": {
            "iterations": 3,
            "population": 16,
            "elite_fraction": 0.25,
            "proposal_scale": 0.70,
            "support_quantile": 0.999,
            "candidate_level_safety_before_elite": True,
            "cross_round_safe_archive": True,
            "retain_incumbent_when_no_safe_candidate": True,
        },
        "evaluation_contract": {
            "cohorts": ["recon100", "scand_common100", "sacson100"],
            "candidate_counts": [16, 32],
            "primary_metric": "ATE",
            "secondary_metrics": ["RTE", "FDE"],
            "checkpoint_selection_uses_external_metrics": False,
            "old_fair_v70_definitions_reused": True,
        },
        "selection_outputs_exist_at_freeze": False,
        "external_test_labels_accessed_before_freeze": False,
        "external_metrics_accessed_before_freeze": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
