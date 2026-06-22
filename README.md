# BAF EDA Gap Matrix (GPU)

Account-level EDA comparison matrix for the NeurIPS 2022 BAF (Bank Account Fraud) dataset: exploratory plots, temporal kNN graph features, property-graph embeddings, vanilla XGBoost training, business cost metrics, and SHAP summaries.

**Primary metric:** PR-AUC on test month 7 (threshold tuned on validation month 6).

## Prerequisites (GPU machine)

- NVIDIA GPU with working driver (`nvidia-smi` succeeds)
- Python 3.11+ recommended
- Git
- CUDA runtime compatible with your XGBoost wheel (CUDA 12.x is typical for recent RTX cards)

## Quick start

```bash
git clone https://github.com/kin-mrqz/fraud-detection-eda.git
cd fraud-detection-eda
bash scripts/hku/setup_remote_env.sh
bash scripts/hku/download_baf_data.sh
```

Verify GPU-backed XGBoost:

```bash
source .venv/bin/activate
python -c "from src.modeling.xgb_runtime import resolve_xgb_compute; print(resolve_xgb_compute())"
# Expect: {'tree_method': 'hist', 'device': 'cuda'}
```

## Run the notebook

**Interactive (Jupyter):**

1. Open `notebooks/experiments/eda_gap_matrix.ipynb`
2. Select kernel: `.venv/bin/python` (or **Fraud-Detection (GPU)** if registered)
3. Run All

**Headless (papermill):**

```bash
bash scripts/hku/run_eda_gap_matrix.sh
```

Outputs are written locally to `results/eda_matrix/` (not tracked in git).

## Data

`scripts/hku/download_baf_data.sh` fetches BAF CSVs from Hugging Face (`jAEhEEkIM/operationbench-baf-raw`) into `data/`:

- `base.csv`, `variant_1.csv` … `variant_5.csv`
- `variant_6.csv` is created as a copy of `base.csv` (matches the notebook `CONFIG` slot)

Each CSV must include `month` and `fraud_bool`.

## Time split (fixed)

| Split | Months |
|-------|--------|
| Train | 0–5 |
| Validation | 6 |
| Test | 7 |

Graph edges use temporal kNN only: `neighbor_month < query_month`.

## Project layout

```text
notebooks/experiments/eda_gap_matrix.ipynb   # main workflow
src/eda/                                     # BAF feature plots
src/graph/                                   # temporal kNN + property graph
src/modeling/                                # XGBoost training, metrics, cost model
src/preprocessing/baf_preprocessor.py        # time split + encoding
scripts/hku/                                 # env setup, data download, papermill runner
```

## Optional notebook extras

`setup_remote_env.sh` also installs: `jupyterlab`, `ipykernel`, `papermill`, `huggingface_hub`.

Manual install:

```bash
pip install jupyterlab ipykernel papermill huggingface_hub
python -m ipykernel install --user --name fraud-detection --display-name "Fraud-Detection (GPU)"
```

## What is not in this repo

- Raw CSVs (`data/`) and experiment outputs (`results/`) — download/regenerate locally
- Retriever / Postgres / inference API / LLM components
