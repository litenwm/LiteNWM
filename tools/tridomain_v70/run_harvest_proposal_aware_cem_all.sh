#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

RUN="${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}"
REPO="${2:-/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain}"
STAGE1="${3:-${RUN}/stage1_tridomain_v2_training/result.json}"
SOURCE=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
CACHE=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56
DATA=/root/autodl-tmp/dwm_datasets
PY=/root/DWM/.venv/bin/python
OUTPUT="${RUN}/cem_harvest"
LOG_ROOT=${RUN}/cem_harvest_logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -f "${STAGE1}"
test ! -e "${OUTPUT}"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
mkdir -p "${OUTPUT}" "${LOG_ROOT}"
cd "${REPO}"

wm_args=()
for seed in 1701 1702 1703; do
  wm_args+=(
    "--wm-${seed}"
    "${CACHE}/ltwm/seed${seed}/checkpoints/best.pt"
  )
done

trajectory_root () {
  case "$1" in
    recon) printf '%s\n' "${DATA}/raw/recon_full_v71" ;;
    scand) printf '%s\n' "${DATA}/raw/scand_full_v71" ;;
    sacson) printf '%s\n' "${DATA}/processed/sacson_processed" ;;
    *) return 2 ;;
  esac
}

latent_root () {
  printf '%s\n' "${DATA}/processed_v70_tridomain/$1_dinov3_ltwm_h4"
}

run_partition () {
  local dataset=$1
  local partition=$2
  local out="${OUTPUT}/${dataset}/${partition}"
  local log="${LOG_ROOT}/${dataset}_${partition}.log"
  "${PY}" tools/tridomain_v70/harvest_proposal_aware_cem_tridomain.py \
    --dataset "${dataset}" \
    --partition "${partition}" \
    --manifest "${SOURCE}/scorer_manifests/${dataset}/${partition}.windows.jsonl" \
    --fixed-features "${RUN}/fixed/${dataset}/${partition}/base_features.npz" \
    --fixed-targets "${RUN}/fixed/${dataset}/${partition}/targets.npz" \
    --candidate-pool "${RUN}/fixed/${dataset}/${partition}/candidate_pool.npz" \
    --latent-root "$(latent_root "${dataset}")" \
    --stage1-result "${STAGE1}" \
    "${wm_args[@]}" \
    --output-dir "${out}" \
    --wm-batch-size 64 \
    --candidate-chunk 8 \
    --device cuda \
    > "${log}" 2>&1
  test -f "${out}/result.json"
}

# The smallest cohort runs first and acts as a real-contract CUDA smoke test.
run_partition recon scorer-cal
run_partition recon train-fit
run_partition scand scorer-cal
run_partition scand train-fit
run_partition sacson scorer-cal
run_partition sacson train-fit

"${PY}" - "${OUTPUT}" "${RUN}/CEM_HARVEST_COMPLETE.json" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
domains = {}
for dataset in ("recon", "scand", "sacson"):
    domains[dataset] = {}
    for partition in ("train-fit", "scorer-cal"):
        path = root / dataset / partition / "result.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        domains[dataset][partition] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "states": value["states"],
            "hard_negative_total": value["hard_negative_total"],
        }
value = {
    "schema": "proposal_aware_v70_tridomain_cem_harvest_complete_v1",
    "status": "complete",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "domains": domains,
    "external_test_accessed": False,
}
output.write_text(
    json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "PROPOSAL_AWARE_CEM_HARVEST_COMPLETE"
