#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_REVISION="dd0a398fa8e84f2a37179332f6c561d20276300b"
EXPECTED_SHA256="dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
MODEL_DIR="${1:-external/dinov3-vitl16-pretrain-lvd1689m}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing 'hf' CLI. Install it with: python3 -m pip install --upgrade huggingface_hub" >&2
  exit 2
fi

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not authenticated. Accept the model license in a browser, then run: hf auth login" >&2
  exit 2
fi

mkdir -p "$MODEL_DIR"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

hf download "$MODEL_ID" \
  config.json \
  preprocessor_config.json \
  model.safetensors \
  LICENSE.md \
  README.md \
  --revision "$MODEL_REVISION" \
  --local-dir "$MODEL_DIR"

ACTUAL_SHA256="$(sha256sum "$MODEL_DIR/model.safetensors" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "DINOv3 model.safetensors SHA-256 mismatch" >&2
  echo "expected: $EXPECTED_SHA256" >&2
  echo "actual:   $ACTUAL_SHA256" >&2
  exit 1
fi

test -s "$MODEL_DIR/config.json"
test -s "$MODEL_DIR/preprocessor_config.json"
echo "DINOv3 download verified: $MODEL_DIR"
