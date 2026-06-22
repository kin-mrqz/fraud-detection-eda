from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NUMERICAL_COLS = [
    "income",
    "name_email_similarity",
    "prev_address_months_count",
    "current_address_months_count",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "bank_months_count",
    "proposed_credit_limit",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
    "device_fraud_count",
    "month",
]

CATEGORICAL_COLS = [
    "payment_type",
    "employment_status",
    "email_is_free",
    "housing_status",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "foreign_request",
    "source",
    "device_os",
    "keep_alive_session",
]


def list_baf_feature_columns(
    df: pd.DataFrame,
    target_col: str = "fraud_bool",
    month_col: str = "month",
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) present in df, excluding target."""
    numeric = [c for c in NUMERICAL_COLS if c in df.columns and c not in {target_col, month_col}]
    categorical = [c for c in CATEGORICAL_COLS if c in df.columns]
    # Include any extra columns not in canonical lists
    for col in df.columns:
        if col in {target_col, month_col}:
            continue
        if col in numeric or col in categorical:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def _percent_within_class(df: pd.DataFrame, feature: str, target_col: str) -> pd.DataFrame:
    tmp = df[[target_col, feature]].copy()
    tmp[feature] = tmp[feature].astype(str)
    agg = tmp.groupby([target_col, feature]).size().reset_index(name="count")
    agg["pct"] = agg.groupby(target_col)["count"].transform(lambda x: x / x.sum())
    agg[target_col] = agg[target_col].map({0: "Non-Fraud", 1: "Fraud"})
    return agg


def draw_features_baf(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    target_col: str = "fraud_bool",
    max_plots: int | None = None,
) -> None:
    """
    Colab-style factor plots: % distribution of each feature category by fraud label.
    Numeric features are binned with qcut before plotting.
    """
    numeric, categorical = list_baf_feature_columns(df, target_col=target_col)
    if feature_cols is None:
        feature_cols = categorical + numeric
    if max_plots is not None:
        feature_cols = feature_cols[:max_plots]

    for feat in feature_cols:
        plot_df = df.copy()
        if feat in numeric:
            series = plot_df[feat].replace(-1, np.nan)
            try:
                plot_df[feat] = pd.qcut(series, q=min(10, series.nunique()), duplicates="drop")
            except ValueError:
                plot_df[feat] = series.astype(str)
        agg = _percent_within_class(plot_df, feat, target_col)
        plt.figure(figsize=(12, 4))
        sns.barplot(data=agg, x=feat, y="pct", hue=target_col, palette={"Fraud": "darkorange", "Non-Fraud": "steelblue"})
        plt.title(f"{feat} by fraud_bool — % within class")
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Share within fraud / non-fraud")
        plt.tight_layout()
        plt.show()
