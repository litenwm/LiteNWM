#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/dinov3-nomad-ltwm-v70-tridomain
RUN=/root/autodl-tmp/dwm_runs/v70_tridomain_new_weights
DATA=/root/autodl-tmp/dwm_datasets
PY=/root/DWM/.venv/bin/python
VISUALNAV=/root/baselines/nomad_official
DIFFUSION=/root/autodl-tmp/dwm_datasets/deps/diffusion_policy
NOMAD_CONFIG=/root/baselines/nomad_official/train/config/nomad.yaml
NOMAD_CHECKPOINT=/root/baselines/nomad_official/deployment/model_weights/nomad.pth
SPLIT_LOCK="$RUN/splits/split_lock.json"
MANIFEST_ROOT="$RUN/scorer_manifests"
ARTIFACT_ROOT="$RUN/selector_artifacts"

mkdir -p "$RUN/logs" "$ARTIFACT_ROOT"
cd "$REPO"

while [[ ! -f "$RUN/LTWM_PIPELINE_COMPLETE" ]]; do
  sleep 60
done

run_proposals_and_labels() {
  local domain="$1"
  local root="$2"
  local partition="$3"
  local manifest="$MANIFEST_ROOT/$domain/$partition.windows.jsonl"
  local out="$ARTIFACT_ROOT/$domain/$partition"
  mkdir -p "$out"
  if [[ ! -f "$out/proposals.npz" ]]; then
    if [[ "$domain" == "sacson" ]]; then
      "$PY" tools/tridomain_v70/run_sacson_artifact.py proposals \
        --dataset sacson --partition "$partition" \
        --trajectory-root "$root" --inner-manifest "$manifest" \
        --split-lock "$SPLIT_LOCK" --visualnav-root "$VISUALNAV" \
        --diffusion-policy-root "$DIFFUSION" --nomad-config "$NOMAD_CONFIG" \
        --nomad-checkpoint "$NOMAD_CHECKPOINT" \
        --output "$out/proposals.npz" --seed 1701 --save-every 32 --resume \
        --device cuda > "$RUN/logs/proposals_${domain}_${partition}.log" 2>&1
    else
      "$PY" tools/planning/build_nomad_h4_proposals_v2.py \
        --dataset "$domain" --partition "$partition" \
        --trajectory-root "$root" --inner-manifest "$manifest" \
        --split-lock "$SPLIT_LOCK" --visualnav-root "$VISUALNAV" \
        --diffusion-policy-root "$DIFFUSION" --nomad-config "$NOMAD_CONFIG" \
        --nomad-checkpoint "$NOMAD_CHECKPOINT" \
        --output "$out/proposals.npz" --seed 1701 --save-every 32 --resume \
        --device cuda > "$RUN/logs/proposals_${domain}_${partition}.log" 2>&1
    fi
  fi
  if [[ ! -f "$out/labels.npz" ]]; then
    if [[ "$domain" == "sacson" ]]; then
      "$PY" tools/tridomain_v70/run_sacson_artifact.py labels \
        --dataset sacson --partition "$partition" \
        --trajectory-root "$root" --inner-manifest "$manifest" \
        --proposals "$out/proposals.npz" --split-lock "$SPLIT_LOCK" \
        --output "$out/labels.npz" \
        > "$RUN/logs/labels_${domain}_${partition}.log" 2>&1
    else
      "$PY" tools/scoring/build_trajectory_quality_labels_v2.py \
        --dataset "$domain" --partition "$partition" \
        --trajectory-root "$root" --inner-manifest "$manifest" \
        --proposals "$out/proposals.npz" --split-lock "$SPLIT_LOCK" \
        --output "$out/labels.npz" \
        > "$RUN/logs/labels_${domain}_${partition}.log" 2>&1
    fi
  fi
}

for spec in \
  "recon /root/autodl-tmp/dwm_datasets/raw/recon_full_v71" \
  "scand /root/autodl-tmp/dwm_datasets/raw/scand_full_v71" \
  "sacson /root/autodl-tmp/dwm_datasets/processed/sacson_processed"; do
  read -r domain root <<<"$spec"
  for partition in train-fit scorer-cal; do
    run_proposals_and_labels "$domain" "$root" "$partition"
  done
done

for seed in 1701 1702 1703; do
  WM="$RUN/ltwm/seed${seed}/checkpoints/best_tridomain.pt"
  for spec in \
    "recon /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/recon_dinov3_ltwm_h4" \
    "scand /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/scand_dinov3_ltwm_h4" \
    "sacson /root/autodl-tmp/dwm_datasets/processed_v70_tridomain/sacson_dinov3_ltwm_h4"; do
    read -r domain latent <<<"$spec"
    for partition in train-fit scorer-cal; do
      out="$ARTIFACT_ROOT/seed${seed}/$domain/$partition"
      mkdir -p "$out"
      if [[ ! -f "$out/cache.npz" ]]; then
        if [[ "$domain" == "sacson" ]]; then
          "$PY" tools/tridomain_v70/run_sacson_artifact.py cache \
            --dataset sacson --partition "$partition" \
            --latent-root "$latent" \
            --proposals "$ARTIFACT_ROOT/$domain/$partition/proposals.npz" \
            --labels "$ARTIFACT_ROOT/$domain/$partition/labels.npz" \
            --wm-checkpoint "$WM" \
            --inner-manifest "$MANIFEST_ROOT/$domain/$partition.windows.jsonl" \
            --split-lock "$SPLIT_LOCK" --output "$out/cache.npz" \
            --batch-size 2 --candidate-chunk 4 --device cuda \
            > "$RUN/logs/cache_seed${seed}_${domain}_${partition}.log" 2>&1
        else
          "$PY" tools/scoring/cache_trajectory_quality_features_v2.py \
            --dataset "$domain" --partition "$partition" \
            --latent-root "$latent" \
            --proposals "$ARTIFACT_ROOT/$domain/$partition/proposals.npz" \
            --labels "$ARTIFACT_ROOT/$domain/$partition/labels.npz" \
            --wm-checkpoint "$WM" \
            --inner-manifest "$MANIFEST_ROOT/$domain/$partition.windows.jsonl" \
            --split-lock "$SPLIT_LOCK" --output "$out/cache.npz" \
            --batch-size 2 --candidate-chunk 4 --device cuda \
            > "$RUN/logs/cache_seed${seed}_${domain}_${partition}.log" 2>&1
        fi
      fi
    done
  done
  out="$RUN/scorer/seed${seed}"
  if [[ ! -f "$out/summary.json" ]]; then
    mkdir -p "$out"
    "$PY" tools/tridomain_v70/train_scorer_v2_tridomain.py \
      --recon-train "$ARTIFACT_ROOT/seed${seed}/recon/train-fit/cache.npz" \
      --recon-calibration "$ARTIFACT_ROOT/seed${seed}/recon/scorer-cal/cache.npz" \
      --scand-train "$ARTIFACT_ROOT/seed${seed}/scand/train-fit/cache.npz" \
      --scand-calibration "$ARTIFACT_ROOT/seed${seed}/scand/scorer-cal/cache.npz" \
      --sacson-train "$ARTIFACT_ROOT/seed${seed}/sacson/train-fit/cache.npz" \
      --sacson-calibration "$ARTIFACT_ROOT/seed${seed}/sacson/scorer-cal/cache.npz" \
      --split-lock "$SPLIT_LOCK" --seed "$seed" --output-dir "$out" \
      --epochs 40 --batch-size 96 --learning-rate 0.0002 \
      --weight-decay 0.0001 --patience 8 --device cuda \
      > "$RUN/logs/scorer_seed${seed}.log" 2>&1
  fi
done

touch "$RUN/SCORER_PIPELINE_COMPLETE"
