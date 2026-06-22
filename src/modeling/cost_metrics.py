from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


@dataclass(frozen=True)
class CostConfig:
    clerk_salary_hkd_annual: float = 400_000.0
    hours_per_month: float = 160.0
    hours_to_unlock_fp: float = 32.0
    hours_to_check_fp: float = 8.0
    fraud_volume_proxy_col: str = "intended_balcon_amount"


def hourly_clerk_rate_hkd(config: CostConfig) -> float:
    return config.clerk_salary_hkd_annual / 12.0 / config.hours_per_month


def cost_unlock_fp_per_account(config: CostConfig) -> float:
    return hourly_clerk_rate_hkd(config) * config.hours_to_unlock_fp


def cost_check_fp_per_account(config: CostConfig) -> float:
    return hourly_clerk_rate_hkd(config) * config.hours_to_check_fp


def compute_revenue_tp_per_account(
    train_df: pd.DataFrame,
    target_col: str = "fraud_bool",
    volume_col: str = "intended_balcon_amount",
) -> float:
    """
    Revenue recovered per correctly blocked fraud account.
    Proxy: total fraud exposure on train / number of fraud accounts.
    """
    fraud_df = train_df[train_df[target_col] == 1]
    if fraud_df.empty:
        return 0.0
    if volume_col not in fraud_df.columns:
        raise ValueError(f"Volume proxy column '{volume_col}' not in training data.")
    volumes = fraud_df[volume_col].replace(-1, np.nan).fillna(0.0)
    return float(volumes.sum() / len(fraud_df))


def model_metrics_cut(
    cutoff: float,
    y_true: np.ndarray,
    y_score: np.ndarray,
    revenue_tp: float,
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
    base_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    lift = precision / base_rate if base_rate > 0 else 0.0

    return pd.Series(
        {
            "Cutoff": cutoff,
            "Predicted": predicted,
            "Correct": correct,
            "Incorrect": incorrect,
            "Precision": precision,
            "Recall": recall,
            "Lift": lift,
            "Revenue_Correct": correct * revenue_tp,
            "Cost_Unlock_Incorrect": incorrect * cost_unlock_fp,
            "Cost_Check_Incorrect": incorrect * cost_check_fp,
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
    Dual-cutoff profit: BLOCK at block_idx (higher cutoff), ALERT at alert_idx (lower cutoff).
    alert_idx should correspond to a lower score cutoff than block_idx (higher index in descending table).
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
        "max_profit": float(max_profit),
        "block_idx": int(best_block),
        "alert_idx": int(best_alert),
        "block_cutoff": float(block_row["Cutoff"]),
        "alert_cutoff": float(alert_row["Cutoff"]),
        "block_predicted": int(block_row["Predicted"]),
        "alert_predicted": int(alert_row["Predicted"]),
        "clerks_per_month_estimate": float(clerks),
    }


def plot_cutoff_economics(metrics_df: pd.DataFrame, ax: plt.Axes | None = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    effect = metrics_df[
        ["Cutoff", "Revenue_Correct", "Cost_Unlock_Incorrect", "Cost_Check_Incorrect", "Incorrect"]
    ].copy()
    effect["Profit_Check"] = effect["Revenue_Correct"] - effect["Cost_Check_Incorrect"]
    effect["Profit_Lock"] = effect["Revenue_Correct"] - effect["Cost_Unlock_Incorrect"]
    ax.plot(effect["Cutoff"], effect["Revenue_Correct"], label="Revenue_Correct")
    ax.plot(effect["Cutoff"], effect["Cost_Unlock_Incorrect"], label="Cost_Unlock_Incorrect")
    ax.plot(effect["Cutoff"], effect["Cost_Check_Incorrect"], label="Cost_Check_Incorrect")
    ax.plot(effect["Cutoff"], effect["Profit_Check"], label="Profit_Check", linestyle="--")
    ax.plot(effect["Cutoff"], effect["Profit_Lock"], label="Profit_Lock", linestyle="--")
    ax.set_xlabel("Score cutoff")
    ax.set_ylabel("HKD (cumulative at cutoff)")
    ax.legend()
    ax.set_title("Cutoff economics")
    return ax
