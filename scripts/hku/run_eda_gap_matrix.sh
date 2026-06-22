#!/usr/bin/env bash
# Execute eda_gap_matrix.ipynb headlessly on the GPU node (papermill).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f data/base.csv ]]; then
  bash scripts/hku/download_baf_data.sh
fi

if [[ ! -d .venv ]]; then
  echo "Creating remote venv..."
  bash scripts/hku/setup_remote_env.sh
fi
source .venv/bin/activate
pip install -q papermill jupyterlab ipykernel huggingface_hub 2>/dev/null || true

echo "=== Node / GPU ==="
hostname
nvidia-smi || echo "WARNING: nvidia-smi failed"

echo "=== XGBoost compute ==="
python -c "from src.modeling.xgb_runtime import resolve_xgb_compute; print(resolve_xgb_compute(prefer_gpu=True))"

NOTEBOOK="notebooks/experiments/eda_gap_matrix.ipynb"
OUTPUT="results/eda_matrix/eda_gap_matrix_executed.ipynb"
LOG="results/eda_matrix/run.log"
mkdir -p results/eda_matrix

echo "=== Running notebook (papermill) ==="
echo "Log: $LOG"
papermill "$NOTEBOOK" "$OUTPUT" -k fraud-detection 2>&1 | tee "$LOG"

echo "Done. Output notebook: $OUTPUT"
echo "Artifacts: results/eda_matrix/"
