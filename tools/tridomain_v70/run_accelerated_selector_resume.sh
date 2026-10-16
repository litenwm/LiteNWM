#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_strict_full_window_bs56
SOURCE=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
DATA=/root/autodl-tmp/dwm_datasets
PY=/root/DWM/.venv/bin/python
VISUALNAV=/root/baselines/nomad_official
DIFFUSION=/root/autodl-tmp/dwm_datasets/deps/diffusion_policy
NOMAD_CONFIG=/root/baselines/nomad_official/train/config/nomad.yaml
NOMAD_CHECKPOINT=/root/baselines/nomad_official/deployment/model_weights/nomad.pth
TRAJECTORY_ROOT=/root/autodl-tmp/dwm_datasets/processed/sacson_processed
MANIFEST_ROOT=$SOURCE/scorer_manifests/sacson
ARTIFACT_ROOT=$SOURCE/selector_artifacts/sacson
SPLIT_LOCK=$RUN/split_lock.json

cd "$REPO"
mkdir -p "$RUN/logs" "$ARTIFACT_ROOT"

for partition in train-fit scorer-cal; do
  output=$ARTIFACT_ROOT/$partition/proposals.npz
  if [[ ! -f "$output" ]]; then
    mkdir -p "$(dirname "$output")"
    "$PY" tools/tridomain_v70/run_batched_sacson_proposals.py \
      --partition "$partition" \
      --trajectory-root "$TRAJECTORY_ROOT" \
      --inner-manifest "$MANIFEST_ROOT/$partition.windows.jsonl" \
      --split-lock "$SPLIT_LOCK" \
      --visualnav-root "$VISUALNAV" \
      --diffusion-policy-root "$DIFFUSION" \
      --nomad-config "$NOMAD_CONFIG" \
      --nomad-checkpoint "$NOMAD_CHECKPOINT" \
      --output "$output" \
      --seed 1701 \
      --window-batch-size 32 \
      --save-every 1024 \
      --verify-windows 32 \
      --equivalence-atol 0.02 \
      --device cuda \
      > "$RUN/logs/proposals_sacson_${partition}_batched.log" 2>&1
  fi
done

exec bash tools/tridomain_v70/run_strict_full_window_selector.sh
