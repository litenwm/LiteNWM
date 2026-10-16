#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO_ROOT="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
STATUS_LOG="${RUN_ROOT}/final_training_dispatch_status.log"

printf '%s waiting for complete three-domain CEM harvest\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
until test -f "${RUN_ROOT}/CEM_HARVEST_COMPLETE.json"; do
  sleep 30
done

printf '%s CEM harvest sealed; waiting for an idle GPU\n' \
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

printf '%s launching final proposal-aware V70 training\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
exec bash "${REPO_ROOT}/tools/tridomain_v70/run_train_proposal_aware_cem_final.sh" \
  "${RUN_ROOT}" "${REPO_ROOT}"
