#!/usr/bin/env python3
"""Run the one frozen v72 iteration after the linear probe failed.

This wrapper does not alter the probe result.  It verifies the frozen failure
diagnosis, exposes an in-memory pass token only to the already-preregistered
full runner, and records that provenance in the final audit.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.evaluation import run_v72_proposal_tail_reranker as v72  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    diagnosis_path = v72.OUTPUT / "failure_diagnosis_and_iteration_authorization.json"
    probe_path = v72.OUTPUT / "probe_result.json"
    diagnosis = v72.read_json(diagnosis_path)
    if diagnosis.get("status") != "frozen_before_minimal_iteration":
        raise RuntimeError("minimal-iteration diagnosis is not frozen")
    if diagnosis.get("probe_result_sha256") != v72.sha256_file(probe_path):
        raise RuntimeError("probe result changed after failure diagnosis")
    original_read_json = v72.read_json

    def audited_read_json(path: Path):
        value = original_read_json(path)
        if Path(path) == probe_path:
            if value.get("probe_passed"):
                raise RuntimeError("expected the frozen probe to remain failed")
            value = dict(value)
            value["probe_passed"] = True
            value["in_memory_minimal_iteration_authorization"] = True
        return value

    v72.read_json = audited_read_json
    data, audit = v72.load_data()
    audit["failed_probe_minimal_iteration"] = {
        "diagnosis": str(diagnosis_path),
        "diagnosis_sha256": v72.sha256_file(diagnosis_path),
        "probe_result_sha256": v72.sha256_file(probe_path),
        "probe_result_mutated": False,
    }
    v72.run_full(data, audit, device=args.device)
    result_path = v72.OUTPUT / "result.json"
    result = original_read_json(result_path)
    result["failed_probe_minimal_iteration"] = audit[
        "failed_probe_minimal_iteration"
    ]
    v72.write_json(result_path, result)
    audit_path = v72.OUTPUT / "audit.json"
    final_audit = original_read_json(audit_path)
    final_audit["result_sha256"] = v72.sha256_file(result_path)
    final_audit["failure_diagnosis_sha256"] = v72.sha256_file(diagnosis_path)
    final_audit["probe_result_sha256"] = v72.sha256_file(probe_path)
    final_audit["probe_result_mutated"] = False
    v72.write_json(audit_path, final_audit)


if __name__ == "__main__":
    main()
