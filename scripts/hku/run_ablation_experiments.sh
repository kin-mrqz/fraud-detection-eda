#!/usr/bin/env bash
# Run Stage 2 and Stage 4 ablations on top of completed eda_gap_matrix_executed.ipynb.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d .venv ]]; then
  echo "Run setup_remote_env.sh first."
  exit 1
fi
source .venv/bin/activate
pip install -q nbclient pyarrow 2>/dev/null || true

mkdir -p results/eda_matrix
python scripts/hku/run_ablation_experiments.py 2>&1 | tee results/eda_matrix/ablation.log
