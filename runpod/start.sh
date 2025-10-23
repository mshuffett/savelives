#!/usr/bin/env bash
set -euo pipefail

echo "=== RunPod Continuous Command: Canary‑Qwen Medical LoRA ==="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

# Defaults
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/workspace/checkpoints/canary-qwen-medical-lora}"
export DATA_DIR="${DATA_DIR:-/workspace/data}"
export LOG_DIR="${LOG_DIR:-/workspace/logs}"

mkdir -p "$CHECKPOINT_DIR" "$DATA_DIR" "$LOG_DIR"

backoff=10
max_backoff=300

while true; do
  echo "\n--- Setup environment (idempotent) ---"
  bash scripts/setup_environment.sh || true

  echo "\n--- Healthcheck ---"
  bash runpod/healthcheck.sh || true

  echo "\n--- Ensure dataset subset present ---"
  python scripts/download_data.py --dataset hani89 --max-samples 2000 || true

  echo "\n--- Train / Resume ---"
  if python src/train.py --config config/salm_lora.yaml --resume; then
    echo "\n✓ Training completed successfully. Running evaluation..."
    CKPT_ARG="nvidia/canary-qwen-2.5b"
    if [[ -f "$CHECKPOINT_DIR/last.ckpt" ]]; then
      CKPT_ARG="$CHECKPOINT_DIR/last.ckpt"
    fi
    python src/evaluation/evaluate.py --checkpoint "$CKPT_ARG" --limit 200 || true
    echo "All done. Sleeping 10 minutes before next check..."
    sleep 600
    backoff=10
  else
    echo "\n✗ Training failed. Backing off for ${backoff}s..."
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    if (( backoff > max_backoff )); then backoff=$max_backoff; fi
  fi
done
