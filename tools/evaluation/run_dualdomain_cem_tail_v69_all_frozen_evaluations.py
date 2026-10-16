#!/usr/bin/env python3
"""Audit every replay pair, then open outcomes once and aggregate v69 results."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tools.evaluation import evaluate_dualdomain_cem_tail_v66_replay as evaluator  # noqa: E402
from tools.scoring.train_recon_dense_meta_dinov3_v1 import file_sha256  # noqa: E402


PYTHON = Path("/home/rainbow/miniconda3/envs/nwm/bin/python")
BASE = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v69_scand_route_crossfit"
)


def specifications():
    for bank in "abcdef":
        source = (
            ROOT
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{bank}/source_labeled.npz"
        )
        yield f"recon_bank_{bank}", "a_to_h", source
    for bank in "gh":
        source = (
            ROOT
            / "outputs/cem_development_v1/recon_cem_boundary_v60/evaluation_v60"
            / f"bank_{bank}_v60_outcomes.npz"
        )
        yield f"recon_bank_{bank}", "a_to_h", source
    source = (
        ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/"
        "outcomes/fresh_outer_outcomes.npz"
    )
    yield "scand_fresh_outer", "spent_scand", source


def load_freeze(path, role):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != evaluator.EXPECTED_STATUS:
        raise RuntimeError(f"replay not frozen: {path}")
    if value.get("model", {}).get("model_role") != role:
        raise RuntimeError(f"wrong role: {path}")
    if value.get("labels_opened") or value.get("future_pose_outcomes_opened"):
        raise RuntimeError(f"early outcome access: {path}")
    if value.get("reserve_accessed"):
        raise RuntimeError(f"reserve access: {path}")
    selected = Path(value["selected_actions"])
    if file_sha256(selected) != value["selected_actions_sha256"]:
        raise RuntimeError(f"selected action hash changed: {selected}")
    return value


def audit_all_pairs(specs):
    summary = []
    for cohort, _, _ in specs:
        replay = BASE / "replays" / cohort
        original = load_freeze(
            replay / "original_arm_d/prelabel_freeze.json", "original_arm_d"
        )
        v69 = load_freeze(
            replay / "proposal_aware_v69/prelabel_freeze.json", "proposal_aware_v66"
        )
        evaluator.bind_pair(original, v69)
        summary.append({
            "cohort": cohort,
            "domain": original["domain"],
            "states": original["states"],
            "original_fixed_coverage": original["fixed_pool_coverage"],
            "original_cem_coverage": original["cem_coverage"],
            "v69_fixed_coverage": v69["fixed_pool_coverage"],
            "v69_cem_coverage": v69["cem_coverage"],
            "selected_actions_bit_exact": (
                original["selected_actions_sha256"]
                == v69["selected_actions_sha256"]
            ),
        })
    print(json.dumps({
        "status": "all_18_replays_paired_and_frozen_before_any_outcome_open",
        "pairs": summary,
        "outcomes_opened": False,
        "reserve_accessed": False,
    }, indent=2, sort_keys=True), flush=True)


def evaluate_one(cohort, role, source):
    replay = BASE / "replays" / cohort
    output = BASE / "evaluations" / cohort
    result = output / "result.json"
    if result.is_file():
        print(f"SKIP_EVALUATED {cohort}", flush=True)
        return result
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"partial evaluation requires inspection: {output}")
    command = [
        str(PYTHON),
        "-m", "tools.evaluation.evaluate_dualdomain_cem_tail_v66_replay_frozen",
        "--original-replay-dir", str(replay / "original_arm_d"),
        "--v66-replay-dir", str(replay / "proposal_aware_v69"),
        "--ground-truth-source", str(source),
        "--cohort-role", role,
        "--output-dir", str(output),
    ]
    print(f"OPEN_OUTCOME_ONCE {cohort}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        print(completed.stdout, flush=True)
        raise RuntimeError(f"evaluation failed ({completed.returncode}): {cohort}")
    value = json.loads(result.read_text(encoding="utf-8"))
    challenger = value["decisions"]["proposal_aware_v66_cem"]
    print(json.dumps({
        "cohort": cohort,
        "coverage": challenger["coverage"],
        "dual_precision": challenger["dual_precision"],
        "harm_rate": challenger["harm_rate"],
        "cohort_pass": value["development_checks"]["cohort_pass"],
    }, sort_keys=True), flush=True)
    return result


def aggregate(results):
    output = BASE / "aggregate_result.json"
    if output.is_file():
        print(f"SKIP_AGGREGATED {output}", flush=True)
        return
    command = [
        str(PYTHON),
        "-m", "tools.evaluation.aggregate_dualdomain_cem_tail_v66_replays_frozen",
    ]
    for result in results:
        command.extend(("--result", str(result)))
    command.extend(("--output", str(output)))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        print(completed.stdout, flush=True)
        raise RuntimeError(f"aggregation failed ({completed.returncode})")
    value = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def main() -> None:
    specs = list(specifications())
    audit_all_pairs(specs)
    results = [evaluate_one(*spec) for spec in specs]
    aggregate(results)


if __name__ == "__main__":
    main()
