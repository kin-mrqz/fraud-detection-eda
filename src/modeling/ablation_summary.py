"""Plain-language summaries for EDA gap matrix ablation results."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_ablation_artifacts(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    artifacts: dict = {"results_dir": str(results_dir)}

    stage1_path = results_dir / "stage1_results.csv"
    if stage1_path.exists():
        artifacts["stage1_df"] = pd.read_csv(stage1_path)

    stage2_path = results_dir / "stage2_ablations.csv"
    if stage2_path.exists():
        artifacts["stage2_df"] = pd.read_csv(stage2_path)

    stage4_path = results_dir / "stage4_variants.csv"
    if stage4_path.exists():
        artifacts["stage4_df"] = pd.read_csv(stage4_path)

    champion_path = results_dir / "champion_config.json"
    if champion_path.exists():
        artifacts["champion"] = json.loads(champion_path.read_text())

    optimal_path = results_dir / "optimal_cutoffs.json"
    if optimal_path.exists():
        artifacts["optimal"] = json.loads(optimal_path.read_text())

    return artifacts


def _fmt_preprocess_label(use_yj: bool, use_smote: bool) -> str:
    yj = "Yeo-Johnson on" if use_yj else "Yeo-Johnson off"
    smote = "SMOTE on" if use_smote else "SMOTE off"
    return f"{yj}, {smote}"


def format_conclusion_markdown(artifacts: dict) -> str:
    lines = ["## What we found", ""]

    stage1 = artifacts.get("stage1_df")
    if stage1 is not None and not stage1.empty:
        champion_row = stage1.sort_values("pr_auc", ascending=False).iloc[0]
        anchor_row = stage1[stage1["graph"] == "none"].iloc[0]
        lines.append(
            f"- **Best graph setup:** {champion_row['graph']} "
            f"(PR-AUC {champion_row['pr_auc']:.3f} on test month 7), "
            f"compared with tabular anchor PR-AUC {anchor_row['pr_auc']:.3f}."
        )
        if champion_row["graph"] == "property":
            prop_row = stage1[stage1["graph"] == "property"].iloc[0]
            lines.append(
                f"- **Property graph** caught more fraud (recall {prop_row['recall']:.1%}) "
                f"than the anchor ({anchor_row['recall']:.1%}), with a small rise in alerts flagged."
            )
        knn_row = stage1[stage1["graph"] == "temporal_kNN"]
        if not knn_row.empty and knn_row.iloc[0]["pr_auc"] < anchor_row["pr_auc"]:
            lines.append(
                "- **Temporal kNN** did not beat the tabular baseline on PR-AUC in this run."
            )
    else:
        lines.append("- Stage 1 results were not found on disk.")

    stage2 = artifacts.get("stage2_df")
    if stage2 is not None and not stage2.empty:
        best2 = stage2.sort_values("pr_auc", ascending=False).iloc[0]
        lines.append(
            f"- **Preprocessing ablation:** best combo was "
            f"{_fmt_preprocess_label(bool(best2['use_yeo_johnson']), bool(best2['use_smote']))} "
            f"(PR-AUC {best2['pr_auc']:.3f})."
        )
    else:
        lines.append("- **Preprocessing ablation:** not run yet.")

    stage4 = artifacts.get("stage4_df")
    if stage4 is not None and not stage4.empty:
        pr = stage4["pr_auc"]
        lines.append(
            f"- **Variant benchmark:** PR-AUC ranged from {pr.min():.3f} to {pr.max():.3f} "
            f"across {len(stage4)} BAF variants (std {pr.std():.3f})."
        )
    else:
        lines.append("- **Variant benchmark:** not run yet.")

    optimal = artifacts.get("optimal")
    if optimal is not None:
        profit = optimal.get("expected_profit_hkd", optimal.get("max_profit"))
        lines.append(
            f"- **Cost model (anchor scores):** optimal cutoffs block at {optimal['block_cutoff']:.2f} "
            f"and alert at {optimal['alert_cutoff']:.2f}. "
            f"Expected profit is about HKD {profit:.2f} on test month 7 — a rough proxy, not real bank savings."
        )

    lines.append("")
    lines.append("## What to do next")
    lines.append("")
    if stage1 is not None and not stage1.empty:
        champ = stage1.sort_values("pr_auc", ascending=False).iloc[0]
        if champ["graph"] != "none":
            lines.append(
                f"- Use **{champ['graph']}** graph features as the default for the next modeling stage."
            )
        else:
            lines.append("- Graph features did not clearly win; keep the tabular model as the default.")
    lines.append("- Replace the fraud revenue proxy with real loss data before trusting the cost model.")
    lines.append("- Re-run ablations if you change preprocessing or download new BAF variants.")

    return "\n".join(lines)


def best_preprocess_config(stage2_df: pd.DataFrame) -> dict:
    best = stage2_df.sort_values("pr_auc", ascending=False).iloc[0]
    return {
        "use_yeo_johnson": bool(best["use_yeo_johnson"]),
        "use_smote": bool(best["use_smote"]),
        "pr_auc": float(best["pr_auc"]),
    }
