#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
WARM_PARENT="$REPO/data_splits/scand/project_strict_test_h4_100_seed17/split_provenance/split_lock.json"

cd "$REPO"

baseline="$RUN/ltwm/baseline_seed1703"
guard=$(runtime/python -c "import json; p='$baseline/report/train_summary.json'; x=json.load(open(p)); print(x['history'][0]['val']['recon']['true']['nmse'][3] * 1.05)")

runtime/python tools/tridomain_v71/train_litetokenformer_wm_v2_tridomain.py \
  --latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/recon_dinov3_ltwm_h4 \
  --scand-latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/scand_dinov3_ltwm_h4 \
  --sacson-latent-root /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/sacson_dinov3_ltwm_h4 \
  --split-lock "$RUN/splits/split_lock.json" \
  --warm-start-parent-split-lock "$WARM_PARENT" \
  --domain-schedule recon,recon,scand,sacson \
  --batch-size 4 \
  --eval-batch-size 4 \
  --no-train-drop-last \
  --stream-cache-shards 16 \
  --num-workers 0 \
  --amp bf16 \
  --weight-decay 0.0001 \
  --grad-clip 1.0 \
  --checkpoint-dir "$RUN/ltwm/seed1703/checkpoints" \
  --report-dir "$RUN/ltwm/seed1703/report" \
  --seed 1703 \
  --epochs 3 \
  --batches-per-epoch 6000 \
  --eval-batches-per-epoch 512 \
  --lr 0.00002 \
  --contrastive-every 4 \
  --consistency-every 8 \
  --objective-revision action_sensitive \
  --recon-val-h4-nmse-limit "$guard" \
  --warm-start-checkpoint "$REPO/checkpoints/project/ltwm_expansion/seed1703/best.pt" \
  > "$RUN/logs/ltwm_train_seed1703.log" 2>&1

touch "$RUN/LTWM_PIPELINE_COMPLETE"
