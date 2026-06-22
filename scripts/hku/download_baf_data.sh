#!/usr/bin/env bash
# Download BAF Base.csv (and optional variants) into data/ for eda_gap_matrix.ipynb.
# Source: Hugging Face mirror of NeurIPS 2022 BAF suite (upstream: Kaggle sgpjesus/...).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$REPO_ROOT/data"
RAW_DIR="$DATA_DIR/raw/baf"
mkdir -p "$RAW_DIR"

if [[ -f "$DATA_DIR/base.csv" ]] && head -1 "$DATA_DIR/base.csv" | grep -q 'fraud_bool'; then
  echo "data/base.csv already exists ($(wc -l < "$DATA_DIR/base.csv") lines). Skipping download."
  exit 0
fi
if [[ -f "$DATA_DIR/base.csv" ]]; then
  echo "data/base.csv exists but is missing fraud_bool; re-normalizing from raw/ ..."
fi

echo "Installing huggingface_hub (if needed)..."
python3 -m pip install -q --upgrade pip
python3 -m pip install -q huggingface_hub

python3 <<'PY'
from pathlib import Path
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download

repo_id = "jAEhEEkIM/operationbench-baf-raw"
raw_dir = Path("data/raw/baf")
raw_dir.mkdir(parents=True, exist_ok=True)
data_dir = Path("data")

files = {
    "Base.csv": "base.csv",
    "Variant I.csv": "variant_1.csv",
    "Variant II.csv": "variant_2.csv",
    "Variant III.csv": "variant_3.csv",
    "Variant IV.csv": "variant_4.csv",
    "Variant V.csv": "variant_5.csv",
}

def normalize_baf_csv(src: Path, dest: Path) -> None:
    # Raw BAF CSVs lead with fraud_bool; do not treat it as a row index.
    df = pd.read_csv(src)
    if "fraud_bool" not in df.columns:
        raise ValueError(f"{src} missing fraud_bool column after read")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(df):,} rows, {len(df.columns)} cols)")

for remote_name, local_name in files.items():
    dest = data_dir / local_name
    raw_path = raw_dir / remote_name
    needs_normalize = (
        not dest.exists()
        or "fraud_bool" not in dest.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    )
    if not needs_normalize:
        print(f"Skip existing {dest}")
        continue
    if not raw_path.exists():
        print(f"Downloading {remote_name} ...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_name,
            repo_type="dataset",
            local_dir=str(raw_dir),
        )
        raw_path = Path(path)
    else:
        print(f"Re-normalizing {remote_name} from {raw_path} ...")
    normalize_baf_csv(raw_path, dest)

# Notebook CONFIG lists variant_6; duplicate base for optional Stage 4 gate.
variant_6 = data_dir / "variant_6.csv"
if not variant_6.exists() or "fraud_bool" not in variant_6.read_text(encoding="utf-8", errors="ignore").splitlines()[0]:
    shutil.copy2(data_dir / "base.csv", variant_6)
    print(f"Wrote {variant_6} (copy of base.csv for variant_6 slot)")

if not (data_dir / "base.csv").exists():
    raise SystemExit("ERROR: data/base.csv missing after download")
PY

echo "BAF data ready under $DATA_DIR"
