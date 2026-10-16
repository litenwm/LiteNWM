#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1/feature_checkpoint_root}"
WM_ROOT=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56/ltwm
SCORER_ROOT=/root/autodl-tmp/dwm_runs/v70_tridomain_scorer_ate_primary_rtpbounded_v1/scorer

test ! -e "${ROOT}"
for seed in 1701 1702 1703; do
  mkdir -p \
    "${ROOT}/ltwm_expansion/seed${seed}" \
    "${ROOT}/scorer_v2/seed${seed}"
  ln -s \
    "${WM_ROOT}/seed${seed}/checkpoints/best.pt" \
    "${ROOT}/ltwm_expansion/seed${seed}/best.pt"
  ln -s \
    "${SCORER_ROOT}/seed${seed}/best.pt" \
    "${ROOT}/scorer_v2/seed${seed}/best.pt"
done

find "${ROOT}" -type l -ls
