#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO_ROOT="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
STATUS_LOG="${RUN_ROOT}/fair_eval_dispatch_status.log"

printf '%s waiting for final CEM-trained V70 seal\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
until test -f "${RUN_ROOT}/FINAL_CEM_TRAINING_COMPLETE"; do
  sleep 30
done

printf '%s final V70 sealed; waiting for an idle GPU\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
while true; do
  read -r utilization memory_used < <(
    nvidia-smi \
      --query-gpu=utilization.gpu,memory.used \
      --format=csv,noheader,nounits |
      tr -d ','
  )
  printf '%s gpu_utilization_pct=%s gpu_memory_used_mib=%s\n' \
    "$(date -u +%FT%TZ)" "${utilization}" "${memory_used}" >> "${STATUS_LOG}"
  if (( utilization <= 20 && memory_used <= 1024 )); then
    break
  fi
  sleep 20
done

printf '%s launching sealed three-cohort fair evaluation\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
exec bash "${REPO_ROOT}/tools/tridomain_v70/run_fair_eval_final_v70_tridomain.sh" \
  "${RUN_ROOT}" "${REPO_ROOT}"
