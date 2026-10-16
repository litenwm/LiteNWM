#!/usr/bin/env python3
"""Single adaptive-development 351+210 static-context student experiment."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402
from tools.evaluation import run_scand_pairwise_incremental_student_v1 as core  # noqa: E402


STATIC_ROOT = (
    ROOT
    / "outputs/cem_development_v1/incremental_outcome_student_v1/static_context_student_v1"
)
OUTPUT = STATIC_ROOT / "pairwise_student_v1"
TABLE = STATIC_ROOT / "spent_runtime_pair_static_context_table.npz"
TABLE_AUDIT = STATIC_ROOT / "spent_runtime_pair_static_context_table.audit.v3.json"
STRUCTURE_AUDIT = OUTPUT / "structure_and_determinism.audit.json"
AMENDMENT = OUTPUT / "representation_amendment.json"
PREREG = OUTPUT / "preregistration.json"
PREVIOUS_ROOT = (
    ROOT / "outputs/cem_development_v1/incremental_outcome_student_v1/pairwise_student_v1"
)
PREVIOUS_PREREG = PREVIOUS_ROOT / "preregistration.json"
PREVIOUS_RESULT = PREVIOUS_ROOT / "result.json"
PREVIOUS_DIAGNOSTIC = PREVIOUS_ROOT / "failure_diagnostic.json"
MODEL_AMENDMENT = PREVIOUS_ROOT / "preregistration.model_amendment.v2.json"

EXPECTED = {
    "table": "a0fb034ff89a5cf1bb17e2039e9c7f904baae3c24357fb175fbea6065d19f917",
    "table_audit": "8a1b26d119812dcdb93511bacc1275ddeb77fc3553d8cde327352a5105d7f634",
    "structure": "769934420bd35634691221a8bcc2d59d892c8a95fb05c9f3854cde5a34aed0fd",
    "previous_prereg": "ca30840ac782eab52c12a6a0ced8b9fa589e971e3a2ff314ea4044797f99c355",
    "previous_result": "01233bb70316deb38efdfe46fbb8ad60c054a0b56a442a3c394503f3bda1b786",
    "previous_diagnostic": "b9698383bfda2f46f1808e7ed2e72ca690b8b370376beec4ccd613b3ba391994",
    "model_amendment": "2a691e166d6a28b35cf639fb69fc8d2d81046015547dc4d74d8cbd5ace763f1f",
}


def contract_sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def relative_input(path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}


def bind_core_paths() -> None:
    core.OUTPUT = OUTPUT
    core.PREREG = PREREG
    core.RESULT = OUTPUT / "result.json"
    core.AUDIT = OUTPUT / "audit.json"
    core.PREDICTIONS = OUTPUT / "outer_oof_predictions.npz"
    core.PARAMETERS = OUTPUT / "model_parameters.npz"
    core.TABLE = TABLE
    core.TABLE_AUDIT = TABLE_AUDIT


def verify_expected_hashes() -> None:
    paths = {
        "table": TABLE,
        "table_audit": TABLE_AUDIT,
        "structure": STRUCTURE_AUDIT,
        "previous_prereg": PREVIOUS_PREREG,
        "previous_result": PREVIOUS_RESULT,
        "previous_diagnostic": PREVIOUS_DIAGNOSTIC,
        "model_amendment": MODEL_AMENDMENT,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing frozen input: {name}: {path}")
        observed = sha256_file(path)
        if observed != EXPECTED[name]:
            raise RuntimeError(
                f"frozen input changed: {name}: expected={EXPECTED[name]} observed={observed}"
            )


def preregister() -> None:
    bind_core_paths()
    verify_expected_hashes()
    if AMENDMENT.exists() or PREREG.exists():
        raise RuntimeError("refusing to overwrite representation amendment/preregistration")
    previous = core.read_json(PREVIOUS_PREREG)
    table_audit = core.read_json(TABLE_AUDIT)
    structure = core.read_json(STRUCTURE_AUDIT)
    if table_audit["status"] != "complete_adaptive_development_alignment_verified":
        raise RuntimeError("static table audit is not complete")
    if not table_audit["representation"]["candidate_independent_exact"]:
        raise RuntimeError("static context contract not exact")
    if structure["status"] != "pass_before_preregistration_and_true_label_training":
        raise RuntimeError("structure check not complete")
    if table_audit["sealed_validation"]["trajectory_overlap_with_development"]:
        raise RuntimeError("sealed validation trajectory overlap")

    frozen_contracts = {
        "target": previous["target"],
        "nested_route_oof": previous["nested_route_oof"],
        "candidate_policy": previous["candidate_policy"],
        "inner_profile_selection": previous["inner_profile_selection"],
        "final_success_gate": previous["final_success_gate"],
    }
    amendment = {
        "schema": "scand_static_context_representation_amendment_v1",
        "status": "frozen_before_any_561_feature_true_label_training_or_outer_prediction",
        "experiment_role": "adaptive_development_experiment_not_unbiased_outer_certification",
        "selection_disclosure": {
            "prior_42_route_outer_labels_and_results_influenced_representation_family_choice": True,
            "basis": (
                "351-feature student had useful average candidate ranking but no "
                "route-stable high-precision risk tail; append the already cached "
                "DINOv3 history/goal static context to test missing state/visual context."
            ),
            "previous_result": relative_input(PREVIOUS_RESULT),
            "previous_failure_diagnostic": relative_input(PREVIOUS_DIAGNOSTIC),
            "same_outer_folds_are_unbiased_generalization_estimate": False,
        },
        "representation": {
            "only_allowed_change": (
                "append exactly the existing 210-dimensional static_features block "
                "to the frozen 351-dimensional pair table"
            ),
            "input_dimensions": 561,
            "base_dimensions": 351,
            "added_dimensions": 210,
            "subsets_fusions_or_alternative_contexts_allowed": False,
            "static_cache_source_hashes": table_audit["alignment"]["source_hashes"],
            "all_52000_record_candidate_rows_exactly_aligned": True,
            "alignment_sha256": table_audit["alignment"][
                "record_candidate_static_alignment_sha256"
            ],
            "runtime_safe": True,
            "truth_route_record_features_forbidden": True,
        },
        "unchanged_contract_hashes": {
            name: contract_sha(value) for name, value in frozen_contracts.items()
        },
        "model": {
            "family": "the single previously frozen PyTorch pairwise multitask student",
            "architecture_change": "input width only: 351->561; hidden [64,32] and five heads unchanged",
            "targets_epochs_optimizer_loss_profiles_gates_unchanged": True,
            "ExtraTrees": (
                "permanently abandoned unrun before all earlier training because "
                "scikit-learn was absent; installation and later comparison forbidden"
            ),
            "prior_model_amendment": relative_input(MODEL_AMENDMENT),
        },
        "inputs": {
            "table": relative_input(TABLE),
            "table_audit": relative_input(TABLE_AUDIT),
            "structure_audit": relative_input(STRUCTURE_AUDIT),
            "runner_wrapper": relative_input(Path(__file__).resolve()),
            "runner_core": relative_input(Path(core.__file__).resolve()),
            "previous_preregistration": relative_input(PREVIOUS_PREREG),
        },
        "sealed_validation": {
            "opened": False,
            "trajectory_overlap": [],
            "route_group_name_overlap": table_audit["sealed_validation"][
                "route_group_name_overlap_with_development"
            ],
            "manifest_sha256": table_audit["sealed_validation"]["manifest_sha256"],
            "rule": (
                "only if every unchanged development gate passes, freeze code/config/"
                "model/profile/gate/SHA before one-time validation; never tune back"
            ),
        },
        "reserve_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    core.write_json(AMENDMENT, amendment)

    payload = deepcopy(previous)
    payload["schema"] = "scand_static_context_pairwise_student_preregistration_v1"
    payload["status"] = "frozen_before_nested_outer_oof_predictions"
    payload["evidence_role"] = (
        "adaptive_spent_development_after_prior_outer_results_not_unbiased_certification"
    )
    payload["features"] = {
        "source": (
            "frozen 351-dimensional runtime-safe pair table plus exactly 210-dimensional "
            "candidate-independent cached DINOv3 history/goal static_features"
        ),
        "dimensions": 561,
        "only_change_from_previous_experiment": "append all 210 static_features",
        "forbidden": [
            "route ID",
            "record ID",
            "trajectory ID",
            "ground truth",
            "validation metadata",
            "post-result feature subset selection",
        ],
    }
    payload["model"]["architecture"] = [561, 64, 32, 5]
    payload["model"]["all_other_fields_unchanged_from_previous_preregistration"] = True
    payload["adaptive_development_caveat"] = {
        "previous_outer_results_affected_representation_choice": True,
        "current_outer_metric_is_unbiased_generalization_estimate": False,
        "independent_evidence_requires_unopened_one_time_validation": True,
    }
    payload["inputs"] = {
        "table": relative_input(TABLE),
        "table_audit": relative_input(TABLE_AUDIT),
        "structure_audit": relative_input(STRUCTURE_AUDIT),
        "representation_amendment": relative_input(AMENDMENT),
        "runner_wrapper": relative_input(Path(__file__).resolve()),
        "runner_core": relative_input(Path(core.__file__).resolve()),
        "previous_preregistration": relative_input(PREVIOUS_PREREG),
    }
    payload["sealed_validation_opened"] = False
    payload["reserve_accessed"] = False
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    core.write_json(PREREG, payload)
    print(
        canonical_json(
            {
                "status": payload["status"],
                "evidence_role": payload["evidence_role"],
                "representation_amendment": str(AMENDMENT),
                "representation_amendment_sha256": sha256_file(AMENDMENT),
                "preregistration": str(PREREG),
                "preregistration_sha256": sha256_file(PREREG),
                "sealed_validation_opened": False,
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preregister", "run"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    bind_core_paths()
    if args.phase == "preregister":
        preregister()
    else:
        verify_expected_hashes()
        core.run(args.device)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            canonical_json(
                {
                    "status": "fail_closed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "sealed_validation_opened": False,
                    "reserve_accessed": False,
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
