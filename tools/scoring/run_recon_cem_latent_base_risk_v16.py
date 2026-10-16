#!/usr/bin/env python3
"""Run v16 against the four audited spent-development base-score artifacts."""

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.scoring import train_recon_cem_latent_base_risk_v16 as v16


BASE_SCORE_BY_RESULT = {
    "outputs/cem_development_v1/recon/cem_run_v1/result.json": Path(
        "outputs/cem_development_v1/recon/base_scores_dev1.npz"
    ),
    (
        "outputs/cem_development_v1/recon_confirmation/"
        "cem_hybrid_v2_confirmation_result.json"
    ): Path(
        "outputs/cem_development_v1/recon_confirmation/"
        "base_scores_confirm2.npz"
    ),
    (
        "outputs/cem_development_v1/recon_fresh_confirmation_v3/"
        "cem_addonly_v3_result.json"
    ): Path(
        "outputs/cem_development_v1/recon_fresh_confirmation_v3/"
        "base_scores_fresh_spent3.npz"
    ),
    (
        "outputs/cem_development_v1/recon_v11_replication/"
        "v11_confirmation_result.json"
    ): Path(
        "outputs/cem_development_v1/recon_v11_replication/base_scores.npz"
    ),
}


def audited_base_score(result_path: Path) -> Path:
    key = str(result_path)
    if key not in BASE_SCORE_BY_RESULT:
        raise RuntimeError(f"unaudited v16 base-score result: {key}")
    path = BASE_SCORE_BY_RESULT[key]
    if not path.exists():
        raise FileNotFoundError(path)
    return path


if __name__ == "__main__":
    v16.resolve_base_score = audited_base_score
    v16.main()
