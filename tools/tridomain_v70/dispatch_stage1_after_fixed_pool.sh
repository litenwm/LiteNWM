#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO_ROOT="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
OLD_REPO_ROOT="${3:-/root/autodl-tmp/dinov3-nomad-ltwm-v70}"
STATUS_LOG="${RUN_ROOT}/stage1_tridomain_v2_dispatch_status.log"
MASTER_LOG="${RUN_ROOT}/stage1_tridomain_v2_dispatch_master.log"

if test -e "${RUN_ROOT}/stage1_tridomain_v2_training"; then
  echo "stage1 output already exists; refusing a second dispatch" >&2
  exit 1
fi

printf '%s waiting for all fixed-pool completion seals\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
until \
  test -f "${RUN_ROOT}/fixed/recon/FIXED_POOL_COMPLETE.json" &&
  test -f "${RUN_ROOT}/fixed/scand/FIXED_POOL_COMPLETE.json" &&
  test -f "${RUN_ROOT}/fixed/sacson/FIXED_POOL_COMPLETE.json"; do
  sleep 20
done

printf '%s fixed pools complete; waiting for an idle GPU\n' \
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

printf '%s launching old-V70-compatible fixed-pool training\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
exec bash "${REPO_ROOT}/tools/tridomain_v70/run_train_proposal_aware_fixed_stage.sh" \
  "${RUN_ROOT}" "${REPO_ROOT}" "${OLD_REPO_ROOT}" \
  >> "${MASTER_LOG}" 2>&1
