#!/bin/bash
#SBATCH --job-name=eda-gap-matrix
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/Fraud-Detection}"

source .venv/bin/activate

echo "=== Node / GPU ==="
hostname
nvidia-smi

echo "=== XGBoost compute ==="
python -c "from src.modeling.xgb_runtime import resolve_xgb_compute; print(resolve_xgb_compute(prefer_gpu=True))"

NOTEBOOK="notebooks/experiments/eda_gap_matrix.ipynb"
OUTPUT="results/eda_matrix/eda_gap_matrix_executed.ipynb"

mkdir -p results/eda_matrix data

if [[ ! -f data/base.csv ]]; then
  echo "ERROR: data/base.csv missing. Upload before submitting batch job."
  exit 1
fi

# Headless execute; set eda_max_plots in notebook CONFIG if full factor plots are too slow
papermill "$NOTEBOOK" "$OUTPUT" -k fraud-detection

echo "Done. Artifacts in results/eda_matrix/"
