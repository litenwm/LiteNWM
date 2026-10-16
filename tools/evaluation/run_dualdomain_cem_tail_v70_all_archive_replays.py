#!/usr/bin/env python3
"""Run every missing original/v70 archived replay pair without outcomes."""

from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/rainbow/miniconda3/envs/nwm/bin/python")
RUNNER = ROOT / "tools/planning/run_dualdomain_cem_tail_v70_archive_replay.py"
RESULT = (
    ROOT
    / "outputs/cem_development_v1/"
    "dualdomain_cem_tail_v70_broad_hashfold_archive/result.json"
)
OUTPUT_ROOT = RESULT.parent / "replays"


def specifications():
    for bank in "abcdefgh":
        if bank < "g":
            base = (
                ROOT
                / "outputs/cem_development_v1/recon_target_calibration_v20"
                / f"bank_{bank}"
            )
            features = (
                base
                / "features_cem_v30"
                / f"recon_target_bank_{bank}_cem_v30_features.npz"
            )
            latents = (
                base
                / "features_cem_v30"
                / f"recon_target_bank_{bank}_cem_v30_latents.npz"
            )
        else:
            base = (
                ROOT
                / "outputs/cem_development_v1/recon_cem_boundary_v60"
                / f"bank_{bank}_v60"
            )
            features = base / "features" / f"recon_bank_{bank}_v60_features.npz"
            latents = base / "features" / f"recon_bank_{bank}_v60_latents.npz"
        yield {
            "domain": "recon",
            "cohort": f"recon_bank_{bank}",
            "manifest": base / "manifest.windows.jsonl",
            "candidate": base / "k17.npz",
            "features": features,
            "latents": latents,
            "trajectory_root": Path("/home/rainbow/nwm/data/recon"),
        }
    base = ROOT / "outputs/scand_domain_adaptation_v1/fresh_outer_v1"
    yield {
        "domain": "scand",
        "cohort": "scand_fresh_outer",
        "manifest": base / "manifest.windows.jsonl",
        "candidate": base / "fresh_outer_k17.npz",
        "features": base
        / "features/scand_fresh_outer_h4_100_seed20260720_features.npz",
        "latents": base
        / "features/scand_fresh_outer_h4_100_seed20260720_latents.npz",
        "trajectory_root": Path("/home/rainbow/nwm/data/scand"),
    }


def run_one(spec, role, directory_name):
    output = OUTPUT_ROOT / spec["cohort"] / directory_name
    freeze = output / "prelabel_freeze.json"
    if freeze.is_file():
        print(f"SKIP_FROZEN {spec['cohort']} {directory_name}", flush=True)
        return
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"partial replay requires inspection: {output}")
    command = [
        str(PYTHON), str(RUNNER),
        "--domain", spec["domain"],
        "--model-role", role,
        "--cohort", spec["cohort"],
        "--manifest", str(spec["manifest"]),
        "--candidate", str(spec["candidate"]),
        "--base-features", str(spec["features"]),
        "--latent-cache", str(spec["latents"]),
        "--trajectory-root", str(spec["trajectory_root"]),
        "--v66-result", str(RESULT),
        "--output-dir", str(output),
        "--wm-batch-size", "32",
    ]
    print(f"START {spec['cohort']} {directory_name}", flush=True)
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
        raise RuntimeError(
            f"replay failed ({completed.returncode}): {spec['cohort']} {directory_name}"
        )
    if not freeze.is_file():
        raise RuntimeError(f"replay returned without freeze: {output}")
    progress = [
        line for line in completed.stdout.splitlines()
        if '"completed_states"' in line
    ]
    print(progress[-1] if progress else "NO_PROGRESS_LINE", flush=True)
    print(f"DONE {spec['cohort']} {directory_name}", flush=True)


def main() -> None:
    for spec in specifications():
        run_one(spec, "original_arm_d", "original_arm_d")
        run_one(spec, "proposal_aware_v66", "proposal_aware_v70")
    print("ALL_18_V70_ARCHIVE_REPLAYS_FROZEN", flush=True)


if __name__ == "__main__":
    main()
