#!/usr/bin/env bash
# One-time (or occasional) environment setup on HKU GPU farm compute node.
# Usage: bash scripts/hku/setup_remote_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
pip install jupyterlab ipykernel papermill huggingface_hub

python -m ipykernel install --user --name fraud-detection --display-name "Fraud-Detection (GPU)"
mkdir -p data results/eda_matrix

echo ""
echo "Setup complete in $REPO_ROOT"
echo "Next: place data/base.csv in data/ then run scripts/hku/start_jupyter_gpu.sh"
