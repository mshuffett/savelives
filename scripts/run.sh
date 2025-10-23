#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

echo "=== Setup ==="
bash scripts/setup_environment.sh

echo "=== Data ==="
python scripts/download_data.py --dataset hani89 --max-samples 2000

echo "=== Train ==="
python src/train.py --config config/salm_lora.yaml --resume

echo "=== Eval ==="
python src/evaluation/evaluate.py --limit 200

