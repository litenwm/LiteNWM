#!/usr/bin/env bash
set -Eeuo pipefail
umask 022

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
SOURCE=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
CACHE=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56
RUN=${1:-/root/autodl-tmp/dwm_runs/v70_proposal_aware_cem_tridomain_v1}
DATA=/root/autodl-tmp/dwm_datasets
PY=/root/DWM/.venv/bin/python

test ! -e "$RUN" || {
  echo "refusing overwrite: $RUN" >&2
  exit 98
}
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"

mkdir -p "$RUN/logs" "$RUN/fixed"
cd "$REPO"

wm_args=()
for seed in 1701 1702 1703; do
  wm_args+=(
    "--wm-$seed"
    "$CACHE/ltwm/seed$seed/checkpoints/best.pt"
  )
done

run_domain () {
  local domain=$1
  local trajectory=$2
  local latent=$3
  local out="$RUN/fixed/$domain"
  local cache_args=()
  for seed in 1701 1702 1703; do
    cache_args+=(
      "--train-cache-$seed"
      "$CACHE/selector_artifacts/seed$seed/$domain/train-fit/cache.npz"
      "--calibration-cache-$seed"
      "$CACHE/selector_artifacts/seed$seed/$domain/scorer-cal/cache.npz"
    )
  done
  "$PY" tools/tridomain_v70/prepare_proposal_aware_cem_fixed_pool.py \
    --dataset "$domain" \
    --trajectory-root "$trajectory" \
    --latent-root "$latent" \
    --train-manifest "$SOURCE/scorer_manifests/$domain/train-fit.windows.jsonl" \
    --calibration-manifest "$SOURCE/scorer_manifests/$domain/scorer-cal.windows.jsonl" \
    --train-proposals "$SOURCE/selector_artifacts/$domain/train-fit/proposals.npz" \
    --calibration-proposals "$SOURCE/selector_artifacts/$domain/scorer-cal/proposals.npz" \
    "${wm_args[@]}" \
    "${cache_args[@]}" \
    --output-dir "$out" \
    --batch-size 64 \
    --support-quantile 0.999 \
    --rtp-minimum-distance 0.05 \
    --rtp-distance-quantile 0.01 \
    --device cuda
}

run_domain \
  recon \
  "$DATA/raw/recon_full_v71" \
  "$DATA/processed_v70_tridomain/recon_dinov3_ltwm_h4" \
  > "$RUN/logs/recon.log" 2>&1

run_domain \
  scand \
  "$DATA/raw/scand_full_v71" \
  "$DATA/processed_v70_tridomain/scand_dinov3_ltwm_h4" \
  > "$RUN/logs/scand.log" 2>&1

run_domain \
  sacson \
  "$DATA/processed/sacson_processed" \
  "$DATA/processed_v70_tridomain/sacson_dinov3_ltwm_h4" \
  > "$RUN/logs/sacson.log" 2>&1

"$PY" - "$RUN" <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
domains = {}
for name in ("recon", "scand", "sacson"):
    path = root / "fixed" / name / "FIXED_POOL_COMPLETE.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    domains[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "train_states": value["train"]["states"],
        "calibration_states": value["calibration"]["states"],
    }
output = {
    "schema": "proposal_aware_v70_tridomain_fixed_pool_complete_v1",
    "status": "complete",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "domains": domains,
    "external_test_accessed": False,
}
(root / "FIXED_POOL_COMPLETE.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

echo "PROPOSAL_AWARE_FIXED_POOL_COMPLETE"
