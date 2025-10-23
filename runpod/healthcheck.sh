#!/usr/bin/env bash
set -euo pipefail

echo "[healthcheck] GPU: $(python - <<'PY'
import torch
print('available' if torch.cuda.is_available() else 'missing')
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
)"

echo "[healthcheck] CUDA: $(python - <<'PY'
import torch
print(torch.version.cuda)
PY
)"

echo "[healthcheck] NeMo: $(python - <<'PY'
import nemo
print(nemo.__version__)
PY
)"

echo "[healthcheck] W&B: $(python - <<'PY'
import os
print('configured' if os.getenv('WANDB_API_KEY') else 'not-set')
PY
)"

for d in /workspace/checkpoints /workspace/data /workspace/logs; do
  [[ -d "$d" ]] || mkdir -p "$d"
  echo "[healthcheck] dir ok: $d"
done

