#!/usr/bin/env bash
set -u
REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
DATA=/root/autodl-tmp/dwm_datasets
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56
PY=/root/DWM/.venv/bin/python
TRAINER=$REPO/tools/recon/train_litetokenformer_wm_v2_4ctx.py
SPLIT_LOCK=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights/splits/split_lock.json
LATENT_RECON=$DATA/processed_v70_tridomain/recon_dinov3_ltwm_h4
LATENT_SCAND=$DATA/processed_v70_tridomain/scand_dinov3_ltwm_h4
LATENT_SACSON=$DATA/processed_v70_tridomain/sacson_dinov3_ltwm_h4
SOURCE_RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/logs" "$RUN/ltwm"
cd "$REPO"
sha256sum "$TRAINER" > "$RUN/trainer.sha256"
cp "$SPLIT_LOCK" "$RUN/split_lock.json"
for seed in 1701 1702 1703; do
  OUT=$RUN/ltwm/seed$seed
  mkdir -p "$OUT/checkpoints" "$OUT/report"
  SUMMARY=$OUT/report/train_summary.json
  if [[ -f "$SUMMARY" ]] && grep -q '"status": "complete"' "$SUMMARY"; then
    echo "seed $seed already complete"
    continue
  fi
  WARM=$SOURCE_RUN/ltwm/seed$seed/checkpoints/best_tridomain.pt
  GUARD=$("$PY" -c "import torch; p=torch.load('$WARM', map_location='cpu', weights_only=False); print(float(p['history'][-1]['val']['recon']['true']['nmse'][3])*1.05)")
  attempt=0
  while true; do
    attempt=$((attempt + 1))
    LOG=$RUN/logs/ltwm_seed${seed}_attempt${attempt}.log
    if [[ -f "$OUT/checkpoints/latest.pt" ]]; then
      RESUME_ARG="--resume-checkpoint $OUT/checkpoints/latest.pt"
    else
      RESUME_ARG="--warm-start-checkpoint $WARM"
    fi
    echo "seed=$seed attempt=$attempt guard=$GUARD resume=$RESUME_ARG" | tee "$LOG"
    "$PY" "$TRAINER"       --latent-root "$LATENT_RECON"       --scand-latent-root "$LATENT_SCAND"       --sacson-latent-root "$LATENT_SACSON"       --split-lock "$SPLIT_LOCK"       --domain-schedule recon:4,scand:1,sacson:2       --checkpoint-dir "$OUT/checkpoints"       --report-dir "$OUT/report"       --seed "$seed"       --epochs 1       --batches-per-epoch 9600       --eval-batches-per-epoch 512       --batch-size 56       --eval-batch-size 16       --no-train-drop-last       --lr 0.00006       --lr-schedule warmup_cosine       --warmup-steps 300       --min-lr-ratio 0.1       --stream-cache-shards 64       --num-workers 4       --prefetch-factor 2       --amp bf16       --contrastive-every 4       --consistency-every 8       --objective-revision action_norm_balanced       --recon-val-h4-nmse-limit "$GUARD"       --checkpoint-every-steps 250       $RESUME_ARG >> "$LOG" 2>&1
    code=$?
    if [[ $code -eq 0 ]] && [[ -f "$SUMMARY" ]] && grep -q '"status": "complete"' "$SUMMARY"; then
      echo "seed $seed complete"
      break
    fi
    if [[ $attempt -ge 5 ]]; then
      echo "seed $seed failed after $attempt attempts (exit $code)" >&2
      exit "$code"
    fi
    echo "seed $seed interrupted/failed (exit $code); resuming from latest.pt" >&2
    sleep 10
  done
done
touch "$RUN/WORLD_MODEL_COMPLETE"
echo "WORLD_MODEL_COMPLETE"
