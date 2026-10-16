#!/usr/bin/env python3
"""Audit all v70 pairs, evaluate spent outcomes, and aggregate once."""

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
    "dualdomain_cem_tail_v70_broad_hashfold_archive"
)


def specifications():
    for bank in "abcdef":
        yield (
            f"recon_bank_{bank}",
            "a_to_h",
            ROOT
            / "outputs/cem_development_v1/recon_target_calibration_v20"
            / f"bank_{bank}/source_labeled.npz",
        )
    for bank in "gh":
        yield (
            f"recon_bank_{bank}",
            "a_to_h",
            ROOT
            / "outputs/cem_development_v1/recon_cem_boundary_v60/evaluation_v60"
            / f"bank_{bank}_v60_outcomes.npz",
        )
    yield (
        "scand_fresh_outer",
        "spent_scand",
        ROOT
        / "outputs/scand_domain_adaptation_v1/fresh_outer_v1/"
        "outcomes/fresh_outer_outcomes.npz",
    )


def load_freeze(path, role):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != evaluator.EXPECTED_STATUS:
        raise RuntimeError(f"replay not frozen: {path}")
    if value.get("model", {}).get("model_role") != role:
        raise RuntimeError(f"wrong model role: {path}")
    if not value.get("cross_round_safe_archive"):
        raise RuntimeError(f"archive missing: {path}")
    if value.get("outcomes_used_by_cem") or value.get("reserve_accessed"):
        raise RuntimeError(f"forbidden outcome/reserve use: {path}")
    selected = Path(value["selected_actions"])
    if file_sha256(selected) != value["selected_actions_sha256"]:
        raise RuntimeError(f"selected action hash changed: {selected}")
    return value


def audit(specs):
    rows = []
    for cohort, _, _ in specs:
        root = BASE / "replays" / cohort
        original = load_freeze(
            root / "original_arm_d/prelabel_freeze.json", "original_arm_d"
        )
        v70 = load_freeze(
            root / "proposal_aware_v70/prelabel_freeze.json", "proposal_aware_v66"
        )
        evaluator.bind_pair(original, v70)
        bit_exact = original["selected_actions_sha256"] == v70["selected_actions_sha256"]
        if original["domain"] == "recon" and not bit_exact:
            raise RuntimeError(f"RECON pair not bit-exact: {cohort}")
        rows.append({
            "cohort": cohort,
            "domain": original["domain"],
            "states": original["states"],
            "fixed_coverage": original["fixed_pool_coverage"],
            "original_archive_coverage": original["cem_coverage"],
            "v70_archive_coverage": v70["cem_coverage"],
            "selected_actions_bit_exact": bit_exact,
        })
    print(json.dumps({
        "status": "all_v70_pairs_frozen_and_bound",
        "pairs": rows,
        "reserve_accessed": False,
    }, indent=2, sort_keys=True), flush=True)


def evaluate_one(cohort, role, source):
    output = BASE / "evaluations" / cohort
    result = output / "result.json"
    if result.is_file():
        value = json.loads(result.read_text(encoding="utf-8"))
    else:
        command = [
            str(PYTHON),
            "-m", "tools.evaluation.evaluate_dualdomain_cem_tail_v66_replay_frozen",
            "--original-replay-dir", str(BASE / "replays" / cohort / "original_arm_d"),
            "--v66-replay-dir", str(BASE / "replays" / cohort / "proposal_aware_v70"),
            "--ground-truth-source", str(source),
            "--cohort-role", role,
            "--output-dir", str(output),
        ]
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
    if output.exists():
        raise FileExistsError(output)
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
    print(output.read_text(encoding="utf-8"), flush=True)


def main() -> None:
    specs = list(specifications())
    audit(specs)
    results = [evaluate_one(*spec) for spec in specs]
    aggregate(results)


if __name__ == "__main__":
    main()
