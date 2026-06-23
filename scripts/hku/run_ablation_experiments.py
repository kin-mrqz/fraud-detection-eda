#!/usr/bin/env python3
"""Continue eda_gap_matrix_executed.ipynb from Stage 2/4 ablations without re-running heavy cells."""
from __future__ import annotations

import copy
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_core.utils import run_sync
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_NB = REPO_ROOT / "notebooks" / "experiments" / "eda_gap_matrix.ipynb"
EXECUTED_NB = REPO_ROOT / "results" / "eda_matrix" / "eda_gap_matrix_executed.ipynb"
NOTEBOOK_COPY = REPO_ROOT / "notebooks" / "experiments" / "eda_gap_matrix_executed.ipynb"
RESULTS_DIR = REPO_ROOT / "results" / "eda_matrix"


def _code_cell_indices(nb: nbformat.NotebookNode) -> list[int]:
    return [i for i, cell in enumerate(nb.cells) if cell.cell_type == "code"]


def _stage2_code_index(nb: nbformat.NotebookNode) -> int:
    for i in _code_cell_indices(nb):
        if "RUN_STAGE2" in "".join(nb.cells[i].source):
            return i
    raise RuntimeError("Stage 2 code cell not found")


def _strip_papermill_error_cells(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    nb = copy.deepcopy(nb)
    nb.cells = [
        c
        for c in nb.cells
        if "papermill-error-cell-tag" not in c.get("metadata", {}).get("tags", [])
    ]
    return nb


def _preflight() -> None:
    if not EXECUTED_NB.exists():
        raise SystemExit(f"Missing executed notebook: {EXECUTED_NB}")
    if not (RESULTS_DIR / "stage1_results.csv").exists():
        raise SystemExit(f"Missing Stage 1 results: {RESULTS_DIR / 'stage1_results.csv'}")
    if not (REPO_ROOT / "data" / "base.csv").exists():
        raise SystemExit("Missing data/base.csv — run download_baf_data.sh first.")


def _preserve_early_cells(src_nb: nbformat.NotebookNode, exe_nb: nbformat.NotebookNode) -> int:
    exe_nb = _strip_papermill_error_cells(exe_nb)
    src_codes = _code_cell_indices(src_nb)
    exe_codes = _code_cell_indices(exe_nb)
    start = src_codes.index(_stage2_code_index(src_nb))
    if start > len(exe_codes):
        raise RuntimeError("Executed notebook missing code cells to preserve")

    for j in range(start):
        src_i = src_codes[j]
        exe_i = exe_codes[j]
        exe_cell = exe_nb.cells[exe_i]
        src_nb.cells[src_i].source = copy.deepcopy(exe_cell.source)
        src_nb.cells[src_i].outputs = copy.deepcopy(exe_cell.outputs)
        src_nb.cells[src_i].execution_count = exe_cell.execution_count

    for i in src_codes[start:]:
        src_nb.cells[i].outputs = []
        src_nb.cells[i].execution_count = None

    return start


def _as_source_list(text: str) -> list[str]:
    return text.splitlines(keepends=True) if text else []


def _patch_ablation_flags(nb: nbformat.NotebookNode) -> None:
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = "".join(cell.source) if isinstance(cell.source, list) else cell.source
        if "RUN_STAGE2" in src:
            src = re.sub(r"RUN_STAGE2\s*=\s*False", "RUN_STAGE2 = True", src)
        if "RUN_STAGE4" in src:
            src = re.sub(r"RUN_STAGE4\s*=\s*False", "RUN_STAGE4 = True", src)
        cell.source = _as_source_list(src)


def _run_kernel_code(kc, code: str) -> None:
    async def _run() -> None:
        msg_id = kc.execute(code)
        while True:
            msg = await kc.get_iopub_msg()
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            if msg["msg_type"] == "error":
                raise RuntimeError("\n".join(msg["content"]["traceback"]))
            if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
                break

    run_sync(_run)()


def _bootstrap_kernel(kc) -> None:
    bootstrap = f'''
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path({str(REPO_ROOT)!r})
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.ablation_summary import (
    best_preprocess_config,
    format_conclusion_markdown,
    load_ablation_artifacts,
)
from src.modeling.train_vanilla import train_vanilla

CONFIG = {{
    "seed": 42,
    "data_path": REPO_ROOT / "data" / "base.csv",
    "variant_paths": [REPO_ROOT / "data" / f"variant_{{i}}.csv" for i in range(1, 7)],
    "results_dir": REPO_ROOT / "results" / "eda_matrix",
    "stage1_variant": REPO_ROOT / "data" / "base.csv",
    "use_smote_stage1": False,
    "use_yeo_johnson": True,
}}

DATA_PATH = Path(CONFIG["stage1_variant"])
results_dir = CONFIG["results_dir"]

stage1_df = pd.read_csv(results_dir / "stage1_results.csv")
graph_stats = json.loads((results_dir / "graph_stats.json").read_text())
knn_stats = graph_stats["temporal_knn"]
prop_stats = graph_stats["property_graph"]
cost_assumptions = graph_stats.get("cost_assumptions") or json.loads(
    (results_dir / "cost_assumptions.json").read_text()
)
optimal = json.loads((results_dir / "optimal_cutoffs.json").read_text())
if "expected_profit_hkd" not in optimal and "max_profit" in optimal:
    optimal["expected_profit_hkd"] = optimal["max_profit"]

all_feature_cols = [None] * graph_stats["n_raw_features"]
feature_names = [None] * graph_stats["n_preprocessed_features"]
colors = {{"none": "gray", "temporal_kNN": "coral", "property": "seagreen"}}
stage2_results = []
variant_rows = []
RUN_STAGE2 = False
RUN_STAGE4 = False
stage2_df = None
variant_df = None
'''
    _run_kernel_code(kc, bootstrap)


def _write_notebooks(nb: nbformat.NotebookNode, status: str) -> None:
    nb.metadata.setdefault("papermill", {})
    nb.metadata["papermill"].update(
        {
            "input_path": str(SOURCE_NB),
            "output_path": str(EXECUTED_NB),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
    )
    EXECUTED_NB.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTED_NB.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    NOTEBOOK_COPY.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_COPY.open("w", encoding="utf-8") as fh:
        nbformat.write(nb, fh)
    print(f"Wrote {EXECUTED_NB}")
    print(f"Wrote {NOTEBOOK_COPY} (status={status})")


def main() -> None:
    _preflight()

    try:
        from src.modeling.train_vanilla import train_vanilla  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}") from exc

    with SOURCE_NB.open(encoding="utf-8") as fh:
        nb = nbformat.read(fh, as_version=4)
    with EXECUTED_NB.open(encoding="utf-8") as fh:
        old_nb = nbformat.read(fh, as_version=4)

    start_code_ord = _preserve_early_cells(nb, old_nb)
    _patch_ablation_flags(nb)
    code_indices = _code_cell_indices(nb)
    start_cell_idx = code_indices[start_code_ord]

    print(
        f"Preserved {start_code_ord} code cells; "
        f"executing ablations from cell {start_cell_idx} (Stage 2)"
    )

    client = NotebookClient(
        nb,
        kernel_name="fraud-detection",
        timeout=-1,
        log_output=True,
        record_timing=True,
    )
    status = "completed"
    try:
        with client.setup_kernel():
            _bootstrap_kernel(client.kc)
            for i in range(start_cell_idx, len(nb.cells)):
                cell = nb.cells[i]
                if cell.cell_type != "code":
                    continue
                if isinstance(cell.source, list):
                    cell.source = "".join(cell.source)
                print(f"Executing cell {i} ...")
                client.execute_cell(cell, i)
    except CellExecutionError as exc:
        print(exc)
        status = "failed"
        _write_notebooks(nb, status)
        raise

    _write_notebooks(nb, status)


if __name__ == "__main__":
    main()
