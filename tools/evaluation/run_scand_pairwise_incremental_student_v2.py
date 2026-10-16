#!/usr/bin/env python3
"""Amendment-guarded entrypoint for the single PyTorch pairwise student."""

from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import run_scand_pairwise_incremental_student_v1 as runner


AMENDMENT = ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/pairwise_student_v1/preregistration.model_amendment.v2.json"
EXPECTED_SHA256 = "2a691e166d6a28b35cf639fb69fc8d2d81046015547dc4d74d8cbd5ace763f1f"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_amendment() -> None:
    if not AMENDMENT.is_file() or sha256_file(AMENDMENT) != EXPECTED_SHA256:
        raise RuntimeError("model-family amendment missing or changed")
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if value["status"] != "frozen_before_any_model_fit_or_nested_outer_prediction":
        raise RuntimeError("model-family amendment is not frozen")
    if value["implementation_evidence"]["extratrees_training_runs"] != 0:
        raise RuntimeError("ExtraTrees was run; single-model contract violated")
    if not value["disposition"]["future_extratrees_run_for_comparison_or_selection_forbidden"]:
        raise RuntimeError("future ExtraTrees prohibition missing")
    if value["sealed_validation_opened"] or value["reserve_accessed"]:
        raise RuntimeError("forbidden cohort access reported")


if __name__ == "__main__":
    verify_amendment()
    runner.main()
