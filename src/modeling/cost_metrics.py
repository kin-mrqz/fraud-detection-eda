from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


COST_DISCLAIMER = (
    "BAF has no currency column. Exposure uses dataset_amount_units from a numeric "
    "proxy column (not HKD). Ops labor costs use hypothetical HKD clerk rates for the "
    "HK demo only. Absolute HKD savings are not identified from BAF. "
    "See docs/baf_cost_units.md."
)


@dataclass(frozen=True)
class CostConfig:
    """Unit-honest cost config: BAF exposure units vs hypothetical HKD ops labor."""

    clerk_salary_annual: float = 400_000.0
    hours_per_month: float = 160.0
    hours_to_unlock_fp: float = 32.0
    hours_to_check_fp: float = 8.0
    fraud_volume_proxy_col: str = "intended_balcon_amount"
    exposure_unit: str = "dataset_amount_units"
    ops_currency: str = "HKD"
    ops_currency_is_hypothetical: bool = True
    ops_narrative: str = "HK_demo"

    # Back-compat alias used by older notebook cells
    @property
    def clerk_salary_hkd_annual(self) -> float:
        return self.clerk_salary_annual


def hourly_clerk_rate(config: CostConfig) -> float:
    return config.clerk_salary_annual / 12.0 / config.hours_per_month


def hourly_clerk_rate_hkd(config: CostConfig) -> float:
    return hourly_clerk_rate(config)


def cost_unlock_fp_per_account(config: CostConfig) -> float:
    return hourly_clerk_rate(config) * config.hours_to_unlock_fp


def cost_check_fp_per_account(config: CostConfig) -> float:
    return hourly_clerk_rate(config) * config.hours_to_check_fp


def compute_exposure_tp_per_account(
    train_df: pd.DataFrame,
    target_col: str = "fraud_bool",
    volume_col: str = "intended_balcon_amount",
) -> float:
    """
    Relative fraud exposure per TP account in dataset_amount_units.
    Not a currency; privacy-perturbed BAF numeric proxy.
    """
    fraud_df = train_df[train_df[target_col] == 1]
    if fraud_df.empty:
        return 0.0
    if volume_col not in fraud_df.columns:
        raise ValueError(f"Volume proxy column '{volume_col}' not in training data.")
    volumes = fraud_df[volume_col].replace(-1, np.nan).fillna(0.0)
    return float(volumes.sum() / len(fraud_df))


def compute_revenue_tp_per_account(
    train_df: pd.DataFrame,
    target_col: str = "fraud_bool",
    volume_col: str = "intended_balcon_amount",
) -> float:
    """Alias for compute_exposure_tp_per_account (legacy name)."""
    return compute_exposure_tp_per_account(train_df, target_col=target_col, volume_col=volume_col)


def assumptions_dict(
    config: CostConfig,
    exposure_tp_per_account: float,
    cost_unlock_fp: float,
    cost_check_fp: float,
) -> dict:
    return {
        "clerk_salary_annual": config.clerk_salary_annual,
        "hours_per_month": config.hours_per_month,
        "hours_to_unlock_fp": config.hours_to_unlock_fp,
        "hours_to_check_fp": config.hours_to_check_fp,
        "fraud_volume_proxy_col": config.fraud_volume_proxy_col,
        "exposure_unit": config.exposure_unit,
        "exposure_tp_per_account": exposure_tp_per_account,
        "ops_currency": config.ops_currency,
        "ops_currency_is_hypothetical": config.ops_currency_is_hypothetical,
        "ops_narrative": config.ops_narrative,
        "cost_unlock_fp": cost_unlock_fp,
        "cost_check_fp": cost_check_fp,
        "disclaimer": COST_DISCLAIMER,
        "citations": [
            "Jesus et al., Turning the Tables…, NeurIPS 2022",
            "docs/baf_cost_units.md",
        ],
    }


def model_metrics_cut(
    cutoff: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure_tp: float,
    cost_unlock_fp: float,
    cost_check_fp: float,
) -> pd.Series:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= cutoff).astype(int)

    predicted = int(y_pred.sum())
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    correct = precision * predicted
    incorrect = predicted - correct
    n = len(y_true)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = incorrect
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    base_rate = float(np.mean(y_true)) if n else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0
    fpr = float(fp / max(tn + fp, 1))
    alert_yield = float(predicted / max(n, 1))

    return pd.Series(
        {
            "Cutoff": cutoff,
            "Predicted": predicted,
            "Correct": correct,
            "Incorrect": incorrect,
            "FP": float(fp),
            "TN": float(tn),
            "FN": float(fn),
            "Precision": precision,
            "Recall": recall,
            "FPR": fpr,
            "Alert_Yield": alert_yield,
            "Lift": lift,
            "Exposure_Correct": correct * exposure_tp,
            "Cost_Unlock_Incorrect": incorrect * cost_unlock_fp,
            "Cost_Check_Incorrect": incorrect * cost_check_fp,
            # Legacy aliases for older notebook/plot code
            "Revenue_Correct": correct * exposure_tp,
        }
    )


def build_cutoff_metrics_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    revenue_tp: float,
    cost_unlock_fp: float,
    cost_check_fp: float,
    cutoffs: np.ndarray | None = None,
) -> pd.DataFrame:
    if cutoffs is None:
        cutoffs = np.linspace(0.01, 0.99, 50)
    rows = [
        model_metrics_cut(c, y_true, y_score, revenue_tp, cost_unlock_fp, cost_check_fp)
        for c in cutoffs
    ]
    return pd.DataFrame(rows).sort_values("Cutoff", ascending=False).reset_index(drop=True)


def calc_profit(metrics_df: pd.DataFrame, block_idx: int, alert_idx: int, cost_check_fp: float) -> float:
    """
    Dual-cutoff relative index: exposure_units at ALERT band minus HKD labor costs.
    Not interpretable as real bank HKD P&L — see COST_DISCLAIMER.
    """
    rev_tp = metrics_df.loc[alert_idx, "Revenue_Correct"] - (
        metrics_df.loc[alert_idx, "Correct"] - metrics_df.loc[block_idx, "Correct"]
    ) * cost_check_fp
    cost_ul = metrics_df.loc[block_idx, "Cost_Unlock_Incorrect"]
    cost_ch = (
        metrics_df.loc[alert_idx, "Cost_Check_Incorrect"]
        - metrics_df.loc[block_idx, "Cost_Check_Incorrect"]
    )
    return float(rev_tp - cost_ul - cost_ch)


def maximize_profit(metrics_df: pd.DataFrame, cost_check_fp: float, config: CostConfig | None = None) -> dict:
    max_profit = float("-inf")
    best_block = 0
    best_alert = 0
    n = len(metrics_df)
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            profit = calc_profit(metrics_df, i, j, cost_check_fp)
            if profit > max_profit:
                max_profit = profit
                best_block = i
                best_alert = j
    block_row = metrics_df.iloc[best_block]
    alert_row = metrics_df.iloc[best_alert]
    clerks = 0.0
    if config is not None:
        alert_band = max(int(alert_row["Predicted"]) - int(block_row["Predicted"]), 0)
        clerks = alert_band * config.hours_to_check_fp / config.hours_per_month
    return {
        "relative_profit_index": float(max_profit),
        "max_profit": float(max_profit),  # legacy key
        "block_idx": int(best_block),
        "alert_idx": int(best_alert),
        "block_cutoff": float(block_row["Cutoff"]),
        "alert_cutoff": float(alert_row["Cutoff"]),
        "block_predicted": int(block_row["Predicted"]),
        "alert_predicted": int(alert_row["Predicted"]),
        "block_fp": float(block_row["Incorrect"]),
        "alert_fp": float(alert_row["Incorrect"]),
        "clerks_per_month_estimate": float(clerks),
        "disclaimer": COST_DISCLAIMER,
    }


def false_alarm_summary_at_cutoff(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cutoff: float,
    exposure_tp: float,
    cost_unlock_fp: float,
    cost_check_fp: float,
    label: str = "operating",
) -> dict:
    row = model_metrics_cut(cutoff, y_true, y_score, exposure_tp, cost_unlock_fp, cost_check_fp)
    return {
        "label": label,
        "cutoff": float(cutoff),
        "fp_count": float(row["Incorrect"]),
        "fpr": float(row["FPR"]),
        "alert_yield": float(row["Alert_Yield"]),
        "precision": float(row["Precision"]),
        "recall": float(row["Recall"]),
        "tp_exposure_dataset_units": float(row["Exposure_Correct"]),
        "fp_unlock_cost_ops_currency": float(row["Cost_Unlock_Incorrect"]),
        "fp_check_cost_ops_currency": float(row["Cost_Check_Incorrect"]),
        "disclaimer": COST_DISCLAIMER,
    }


def build_false_alarm_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    exposure_tp: float,
    cost_unlock_fp: float,
    cost_check_fp: float,
    cutoffs: dict[str, float],
    config: CostConfig | None = None,
) -> pd.DataFrame:
    rows = [
        false_alarm_summary_at_cutoff(
            y_true, y_score, c, exposure_tp, cost_unlock_fp, cost_check_fp, label=name
        )
        for name, c in cutoffs.items()
    ]
    df = pd.DataFrame(rows)
    if config is not None:
        df["ops_currency"] = config.ops_currency
        df["exposure_unit"] = config.exposure_unit
    return df


def plot_cutoff_economics(
    metrics_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    ops_currency: str = "HKD",
    exposure_unit: str = "dataset_amount_units",
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    effect = metrics_df[
        ["Cutoff", "Revenue_Correct", "Cost_Unlock_Incorrect", "Cost_Check_Incorrect", "Incorrect"]
    ].copy()
    effect["Profit_Check"] = effect["Revenue_Correct"] - effect["Cost_Check_Incorrect"]
    effect["Profit_Lock"] = effect["Revenue_Correct"] - effect["Cost_Unlock_Incorrect"]
    ax.plot(effect["Cutoff"], effect["Revenue_Correct"], label=f"Exposure_Correct ({exposure_unit})")
    ax.plot(
        effect["Cutoff"],
        effect["Cost_Unlock_Incorrect"],
        label=f"Cost_Unlock_FP ({ops_currency}, hypothetical)",
    )
    ax.plot(
        effect["Cutoff"],
        effect["Cost_Check_Incorrect"],
        label=f"Cost_Check_FP ({ops_currency}, hypothetical)",
    )
    ax.plot(effect["Cutoff"], effect["Profit_Check"], label="Relative_index_check", linestyle="--")
    ax.plot(effect["Cutoff"], effect["Profit_Lock"], label="Relative_index_lock", linestyle="--")
    ax.set_xlabel("Score cutoff")
    ax.set_ylabel(f"Mixed units: {exposure_unit} vs {ops_currency} labor")
    ax.legend(fontsize=8)
    ax.set_title("Cutoff economics (unit-honest; not real HKD P&L)")
    return ax


def plot_false_alarm_rate(metrics_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(metrics_df["Cutoff"], metrics_df["FPR"], label="FPR")
    ax.plot(metrics_df["Cutoff"], metrics_df["Alert_Yield"], label="Alert yield")
    ax.plot(metrics_df["Cutoff"], 1.0 - metrics_df["Precision"], label="1 - Precision (among alerts)")
    ax.set_xlabel("Score cutoff")
    ax.set_ylabel("Rate")
    ax.legend()
    ax.set_title("False-alarm / alert rates vs cutoff (currency-free)")
    return ax
