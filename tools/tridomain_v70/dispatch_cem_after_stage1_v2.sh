#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO_ROOT="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
STAGE1="${3:-${RUN_ROOT}/stage1_tridomain_v2_training/result.json}"
STATUS_LOG="${RUN_ROOT}/cem_tridomain_v2_dispatch_status.log"
MASTER_LOG="${RUN_ROOT}/cem_tridomain_v2_dispatch_master.log"
exec >> "${MASTER_LOG}" 2>&1

printf '%s waiting for stage1 proposal-aware training\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
while pgrep -f \
  "train_proposal_aware_cem_risk_tridomain.py.*stage1_tridomain_v2_training" \
  > /dev/null; do
  sleep 20
done

if ! test -f "${STAGE1}"; then
  printf '%s stage1 process ended without result.json\n' \
    "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
  exit 1
fi

/root/DWM/.venv/bin/python - "${STAGE1}" <<'PY'
import json
from pathlib import Path
import sys

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if value.get("decision") != "PROPOSAL_AWARE_V70_TRIDOMAIN_ENSEMBLE_FROZEN":
    raise SystemExit("stage1 decision is not frozen")
if value.get("test_accessed") or value.get("outer_val_accessed"):
    raise SystemExit("stage1 reports forbidden evaluation access")
PY

printf '%s stage1 sealed; waiting for an idle GPU\n' \
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

printf '%s launching full three-domain CEM hard-negative harvest\n' \
  "$(date -u +%FT%TZ)" >> "${STATUS_LOG}"
exec bash "${REPO_ROOT}/tools/tridomain_v70/run_harvest_proposal_aware_cem_all.sh" \
  "${RUN_ROOT}" "${REPO_ROOT}" "${STAGE1}"
