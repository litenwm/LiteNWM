#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v71-tridomain
DATA=/root/autodl-tmp/dwm_datasets
RUN=/root/autodl-tmp/dwm_runs/tridomain_v71
RECON_RAW="$DATA/raw/recon_full_v71"
RECON_LATENT="$DATA/processed_v71/recon_dinov3_ltwm_h4"
SCAND_LATENT="$DATA/processed_v71/scand_dinov3_ltwm_h4"
SACSON_LATENT="$DATA/processed_v71/sacson_dinov3_ltwm_h4"
SPLIT_LOCK="$RUN/splits/split_lock.json"
RECON_SPLITS="$RUN/splits/recon_strict_splits.json"
MODEL_DIR="$DATA/deps/dinov3-vitl16-pretrain-lvd1689m"
ADAPTER="$REPO/checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt"
ADAPTER_PARENT="$REPO/data/parent_split_lock/split_lock.json"
WARM_PARENT="$REPO/data_splits/scand/project_strict_test_h4_100_seed17/split_provenance/split_lock.json"

mkdir -p "$RUN/logs" "$RUN/ltwm"
cd "$REPO"

while pgrep -f '[e]ncode_dinov3_adapted_tridomain.py.*partition-count 4' >/dev/null; do
  sleep 30
done

for index in 0 1 2 3; do
  log="$RUN/logs/encode_recon_part${index}.log"
  grep -q '"status": "PASS"' "$log"
  if grep -q 'Traceback' "$log"; then
    echo "partition $index contains a traceback" >&2
    exit 1
  fi
done

# A single resume pass validates every atomic shard and writes complete metadata.
runtime/python tools/tridomain_v71/encode_dinov3_adapted_tridomain.py \
  --trajectory-root "$RECON_RAW" \
  --output-root "$RECON_LATENT" \
  --strict-splits "$RECON_SPLITS" \
  --split-lock "$SPLIT_LOCK" \
  --dataset recon \
  --model-dir "$MODEL_DIR" \
  --adapter-checkpoint "$ADAPTER" \
  --adapter-parent-split-lock "$ADAPTER_PARENT" \
  --include-splits train,val \
  --batch-size 24 \
  --device cuda \
  --resume > "$RUN/logs/encode_recon_consolidate.log" 2>&1

common_args=(
  --latent-root "$RECON_LATENT"
  --scand-latent-root "$SCAND_LATENT"
  --sacson-latent-root "$SACSON_LATENT"
  --split-lock "$SPLIT_LOCK"
  --warm-start-parent-split-lock "$WARM_PARENT"
  --domain-schedule recon,recon,scand,sacson
  --batch-size 4
  --eval-batch-size 4
  --no-train-drop-last
  --stream-cache-shards 16
  --num-workers 0
  --amp bf16
  --weight-decay 0.0001
  --grad-clip 1.0
)

for seed in 1701 1702 1703; do
  warm="$REPO/checkpoints/project/ltwm_expansion/seed${seed}/best.pt"
  baseline="$RUN/ltwm/baseline_seed${seed}"
  mkdir -p "$baseline/checkpoints" "$baseline/report"
  runtime/python tools/tridomain_v71/train_litetokenformer_wm_v2_tridomain.py \
    "${common_args[@]}" \
    --checkpoint-dir "$baseline/checkpoints" \
    --report-dir "$baseline/report" \
    --seed "$seed" \
    --epochs 1 \
    --batches-per-epoch 1 \
    --eval-batches-per-epoch 512 \
    --lr 0 \
    --objective-revision base \
    --warm-start-checkpoint "$warm" > "$RUN/logs/ltwm_baseline_seed${seed}.log" 2>&1

  guard=$(runtime/python -c "import json; p='$baseline/report/train_summary.json'; x=json.load(open(p)); print(x['history'][0]['val']['recon']['true']['nmse'][3] * 1.05)")
  output="$RUN/ltwm/seed${seed}"
  mkdir -p "$output/checkpoints" "$output/report"
  runtime/python tools/tridomain_v71/train_litetokenformer_wm_v2_tridomain.py \
    "${common_args[@]}" \
    --checkpoint-dir "$output/checkpoints" \
    --report-dir "$output/report" \
    --seed "$seed" \
    --epochs 3 \
    --batches-per-epoch 6000 \
    --eval-batches-per-epoch 512 \
    --lr 0.00002 \
    --contrastive-every 4 \
    --consistency-every 8 \
    --objective-revision action_sensitive \
    --recon-val-h4-nmse-limit "$guard" \
    --warm-start-checkpoint "$warm" > "$RUN/logs/ltwm_train_seed${seed}.log" 2>&1
done

touch "$RUN/LTWM_PIPELINE_COMPLETE"
