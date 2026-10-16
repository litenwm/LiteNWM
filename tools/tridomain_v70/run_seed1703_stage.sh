#!/usr/bin/env bash
set -euo pipefail

stage="${1:?stage number required}"
seed="${2:?seed required}"
warm="${3:?warm-start checkpoint required}"
batches="${4:-6000}"

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
STAGE_DIR="$RUN/ltwm/seed1703_stage${stage}"

mkdir -p "$STAGE_DIR/checkpoints" "$STAGE_DIR/report"
cd "$REPO"

runtime/python tools/tridomain_v71/train_litetokenformer_wm_v2_tridomain.py \
  --latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/recon_dinov3_ltwm_h4 \
  --scand-latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/scand_dinov3_ltwm_h4 \
  --sacson-latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/sacson_dinov3_ltwm_h4 \
  --split-lock "$RUN/splits/split_lock.json" \
  --warm-start-parent-split-lock "$REPO/data_splits/scand/project_strict_test_h4_100_seed17/split_provenance/split_lock.json" \
  --domain-schedule recon,recon,scand,sacson \
  --batch-size 4 \
  --eval-batch-size 4 \
  --no-train-drop-last \
  --stream-cache-shards 16 \
  --num-workers 0 \
  --amp bf16 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --checkpoint-dir "$STAGE_DIR/checkpoints" \
  --report-dir "$STAGE_DIR/report" \
  --seed "$seed" \
  --epochs 1 \
  --batches-per-epoch "$batches" \
  --eval-batches-per-epoch 512 \
  --lr 0.00002 \
  --contrastive-every 4 \
  --consistency-every 8 \
  --objective-revision action_sensitive \
  --recon-val-h4-nmse-limit 0.1763303837103789 \
  --warm-start-checkpoint "$warm" \
  > "$RUN/logs/ltwm_train_seed1703_stage${stage}.log" 2>&1
