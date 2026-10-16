#!/usr/bin/env python3
"""Write the final code/data SHA audit for proposal-tail collection v1."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from planning.proposal_tail_sidecar_v1 import canonical_json, sha256_file  # noqa: E402


FORMAL = ROOT / "outputs/cem_development_v1/dualdomain_cem_tail_v70_broad_hashfold_archive/replays/scand_fresh_outer/proposal_aware_v70"
SMOKE = ROOT / "outputs/proposal_tail_collection_v1/pipeline_smoke_spent_nonreserve_state0"
OUTPUT = ROOT / "outputs/proposal_tail_collection_v1/implementation_audit.json"


IMPLEMENTATION = (
    "configs/proposal_tail_collection/scand_proposal_tail_collection_v1_preregistration.json",
    "configs/proposal_tail_collection/scand_proposal_tail_collection_v1.schema.json",
    "configs/proposal_tail_collection/reserve_identity_hashes.template.json",
    "src/planning/proposal_tail_sidecar_v1.py",
    "tools/planning/run_cem_development_v1.py",
    "tools/planning/run_dualdomain_cem_tail_v66_replay.py",
    "tools/planning/build_proposal_tail_identity_registry_v1.py",
    "tools/planning/validate_proposal_tail_dataset_v1.py",
    "tools/planning/build_proposal_tail_labels_v1.py",
    "tools/planning/dry_run_proposal_tail_sidecar_v1.py",
    "tests/planning/test_proposal_tail_sidecar_v1.py",
    "tests/planning/test_proposal_tail_sidecar_serialization_v1.py",
    "tests/planning/test_proposal_tail_sampler_legacy_parity_v1.py",
    "docs/SCAND_PROPOSAL_TAIL_COLLECTION_V1.md",
)


def record(relative: str) -> dict[str, str]:
    path = ROOT / relative
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    selected = FORMAL / "selected_actions.npz"
    prelabel = FORMAL / "prelabel_freeze.json"
    expected = {
        "selected_actions": "95b6fcf79f8540ebb33f692571c335745d540e67998534b461629937e91b6ac0",
        "prelabel_freeze": "7b25a44e769cfb0f7f0eb705256256cbe371c88b62c0765b06ecc6f3ddcda7fc",
    }
    actual = {
        "selected_actions": sha256_file(selected),
        "prelabel_freeze": sha256_file(prelabel),
    }
    if actual != expected:
        raise RuntimeError("formal v70 artifact hash changed")
    smoke_paths = (
        SMOKE / "collection_manifest.json",
        SMOKE / "validation.audit.json",
        SMOKE / "labels_spent_smoke.npz",
        SMOKE / "labels_spent_smoke.audit.json",
        ROOT / "outputs/proposal_tail_collection_v1/used_v70_v77_identities.json",
    )
    audit = {
        "schema": "scand_proposal_tail_collection_implementation_audit_v1",
        "status": "pipeline_ready_no_collection_started",
        "preregistration_frozen_before_implementation": True,
        "logger_default_off": True,
        "formal_behavior_influence": False,
        "formal_v70_hashes_expected": expected,
        "formal_v70_hashes_after": actual,
        "formal_v70_bitwise_unchanged": True,
        "implementation": {relative: record(relative) for relative in IMPLEMENTATION},
        "smoke_artifacts": {
            str(path.relative_to(ROOT)): {"path": str(path), "sha256": sha256_file(path)}
            for path in smoke_paths
        },
        "regression": {
            "syntax_compile": "pass",
            "pytest_runner": "unavailable_in_current_python_environment",
            "manual_assertion_functions": 4,
            "logger_off_vs_preobserver_sampler_candidates": "bitwise_equal",
            "logger_off_vs_preobserver_sampler_rng_state": "equal",
            "logger_on_vs_off_candidates_and_order": "bitwise_equal",
            "logger_on_vs_off_rng_state": "equal",
            "live_sidecar_formal_selected_bytes": "unchanged",
            "cached_formal_snapshot_selected_energy_archive": "all_keys_exact",
            "cached_candidate_order_fixed_plus_r1_r2_r3": "exact",
            "validator": "pass",
            "offline_label_alignment": "pass_spent_shape_only",
        },
        "smoke_limits": {
            "states": 1,
            "cohort": "spent_scand_fresh_outer_state0",
            "raw_prefilter_draws": "historically_unobservable_not_fabricated",
            "sampling_rng_state": "historically_unobservable; covered by synthetic parity regression",
            "performance_or_training_evidence": False,
        },
        "known_unavailable_live_fields": {
            "seed_specific_rpe": "frozen feature exposes RTP, not metric RPE by seed",
            "seed_specific_endpoint": "not emitted by frozen LTWM feature contract",
            "seed_specific_objective": "not emitted by frozen LTWM feature contract",
            "seed_specific_logits": "frozen predictor emits member-averaged logits only",
        },
        "future_collection_blockers": [
            "complete hash-only forbidden identity registry must be supplied by reserve custodian",
            "new manifest must provide collection day, raw timestamps, resolution, encoding, image file size and SHA",
            "new routes must be absent from v70-v77 used route/trajectory/window hashes",
        ],
        "bugs_fixed_during_smoke": [
            "forbidden-path check now tokenizes path components so explicit nonreserve smoke names are not false positives"
        ],
        "reserve_accessed": False,
        "new_routes_collected": False,
        "model_trained": False,
        "automatic_next_step": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(canonical_json(audit) + "\n", encoding="utf-8")
    digest = sha256_file(OUTPUT)
    OUTPUT.with_suffix(".sha256").write_text(digest + "  " + OUTPUT.name + "\n", encoding="utf-8")
    print(canonical_json({"output": str(OUTPUT), "sha256": digest, "status": audit["status"]}))


if __name__ == "__main__":
    main()
