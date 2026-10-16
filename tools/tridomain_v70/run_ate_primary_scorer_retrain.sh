#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
SOURCE_RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56
RUN=${1:-/root/autodl-tmp/dwm_runs/v70_tridomain_scorer_ate_primary_rtpbounded_v1}
PY=/root/DWM/.venv/bin/python
TRAINER=tools/tridomain_v70/train_scorer_v2_tridomain.py
SPLIT_LOCK="$SOURCE_RUN/split_lock.json"

if [[ -e "$RUN" ]]; then
  echo "refusing to overwrite existing run: $RUN" >&2
  exit 2
fi
if [[ ! -f "$SPLIT_LOCK" ]]; then
  echo "missing split lock: $SPLIT_LOCK" >&2
  exit 2
fi
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
if [[ -n "$GPU_PROCESSES" ]]; then
  echo "refusing to start while GPU compute processes exist:" >&2
  echo "$GPU_PROCESSES" >&2
  exit 3
fi

mkdir -p "$RUN/logs" "$RUN/scorer"
ln -s "$SOURCE_RUN/ltwm" "$RUN/ltwm"
cp "$SPLIT_LOCK" "$RUN/split_lock.json"

cd "$REPO"
for seed in 1701 1702 1703; do
  CACHE_ROOT="$SOURCE_RUN/selector_artifacts/seed$seed"
  OUT="$RUN/scorer/seed$seed"
  LOG="$RUN/logs/scorer_seed$seed.log"
  "$PY" "$TRAINER" \
    --recon-train "$CACHE_ROOT/recon/train-fit/cache.npz" \
    --recon-calibration "$CACHE_ROOT/recon/scorer-cal/cache.npz" \
    --scand-train "$CACHE_ROOT/scand/train-fit/cache.npz" \
    --scand-calibration "$CACHE_ROOT/scand/scorer-cal/cache.npz" \
    --sacson-train "$CACHE_ROOT/sacson/train-fit/cache.npz" \
    --sacson-calibration "$CACHE_ROOT/sacson/scorer-cal/cache.npz" \
    --split-lock "$SPLIT_LOCK" \
    --seed "$seed" \
    --output-dir "$OUT" \
    --epochs 20 \
    --batch-size 96 \
    --learning-rate 0.0001 \
    --weight-decay 0.0001 \
    --patience 5 \
    --rtp-target-lower -1.0 \
    --rtp-target-upper 1.0 \
    --robust-quantile-lower 0.01 \
    --robust-quantile-upper 0.99 \
    --rtp-selection-weight 0.15 \
    --rtp-regression-penalty 0.05 \
    --worst-domain-ate-penalty 2.0 \
    --device cuda \
    > "$LOG" 2>&1
done

"$PY" - "$RUN" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
run = Path(sys.argv[1])
seeds = {}
for seed in (1701, 1702, 1703):
    path = run / f"scorer/seed{seed}/best.pt"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = json.loads((path.parent / "summary.json").read_text())
    seeds[str(seed)] = {"checkpoint": str(path), "sha256": digest, "summary": summary}
audit = {
    "schema": "v70_tridomain_ate_primary_rtpbounded_retrain_v1",
    "status": "complete",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_world_models": str(run / "ltwm"),
    "seeds": seeds,
    "outer_test_accessed": False,
}
(run / "TRAINING_COMPLETE.json").write_text(json.dumps(audit, indent=2) + "\n")
PY
