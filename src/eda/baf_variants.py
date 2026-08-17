"""Canonical BAF suite naming: Base + Variants I–V (NeurIPS 2022).

Repo `variant_6.csv` is a duplicate of `base.csv` for legacy Stage-4 slots —
never present it as a distinct bias regime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BAFVariantSpec:
    deck_label: str
    file_stem: str
    bias_axis: str
    alias_of: str | None = None


# Official suite only (Base + I–V). variant_6 is tracked separately as Base alias.
BAF_SUITE: tuple[BAFVariantSpec, ...] = (
    BAFVariantSpec("Base", "base", "Representative baseline"),
    BAFVariantSpec(
        "Variant I",
        "variant_1",
        "Aggravated group-size disparity; equal prevalence across groups",
    ),
    BAFVariantSpec("Variant II", "variant_2", "Higher prevalence disparity"),
    BAFVariantSpec("Variant III", "variant_3", "Separability disparity"),
    BAFVariantSpec("Variant IV", "variant_4", "Temporal prevalence disparity"),
    BAFVariantSpec("Variant V", "variant_5", "Temporal separability drift"),
)

VARIANT_6_ALIAS = BAFVariantSpec(
    "Base (alias)",
    "variant_6",
    "Duplicate of base.csv — not an official bias type",
    alias_of="base",
)

# Approximate overall fraud rates from Jesus et al. NeurIPS 2022 Table 1 narrative
# (group-weighted where needed). Empirical CSV rates override these for reporting.
LITERATURE_OVERALL_FRAUD_RATE: dict[str, float] = {
    "Base": 0.011,
    "Variant I": 0.011,
    "Variant II": 0.0115,  # ~0.5*(0.004+0.019)
    "Variant III": 0.011,
    "Variant IV": 0.010,  # train prevalence disparity; overall ~1%
    "Variant V": 0.011,
}


def suite_mapping_table() -> pd.DataFrame:
    rows = [
        {
            "deck_label": s.deck_label,
            "file": f"{s.file_stem}.csv",
            "bias_axis": s.bias_axis,
            "alias_of": s.alias_of,
        }
        for s in BAF_SUITE
    ]
    rows.append(
        {
            "deck_label": VARIANT_6_ALIAS.deck_label,
            "file": f"{VARIANT_6_ALIAS.file_stem}.csv",
            "bias_axis": VARIANT_6_ALIAS.bias_axis,
            "alias_of": VARIANT_6_ALIAS.alias_of,
        }
    )
    return pd.DataFrame(rows)


def official_eval_paths(data_dir: Path | str) -> list[tuple[str, Path]]:
    """Ordered (deck_label, path) for Base + Variants I–V only."""
    root = Path(data_dir)
    return [(s.deck_label, root / f"{s.file_stem}.csv") for s in BAF_SUITE]


def deck_label_for_stem(stem: str) -> str:
    for s in BAF_SUITE:
        if s.file_stem == stem:
            return s.deck_label
    if stem == "variant_6":
        return VARIANT_6_ALIAS.deck_label
    return stem


def compute_variant_base_rates(
    data_dir: Path | str,
    target_col: str = "fraud_bool",
    month_col: str = "month",
    include_alias: bool = False,
) -> pd.DataFrame:
    """
    Empirical fraud prevalence per official BAF variant (and optionally variant_6 alias).
    """
    root = Path(data_dir)
    specs = list(BAF_SUITE)
    if include_alias:
        specs.append(VARIANT_6_ALIAS)

    rows: list[dict] = []
    for spec in specs:
        path = root / f"{spec.file_stem}.csv"
        if not path.exists():
            rows.append(
                {
                    "deck_label": spec.deck_label,
                    "file": path.name,
                    "exists": False,
                    "n_rows": None,
                    "fraud_rate_overall": None,
                    "fraud_rate_train_0_5": None,
                    "fraud_rate_valid_6": None,
                    "fraud_rate_test_7": None,
                    "literature_fraud_rate_approx": LITERATURE_OVERALL_FRAUD_RATE.get(spec.deck_label),
                    "alias_of": spec.alias_of,
                }
            )
            continue
        df = pd.read_csv(path, usecols=lambda c: c in {target_col, month_col})
        overall = float(df[target_col].mean())
        by_month = df.groupby(month_col)[target_col].mean()
        train = df[df[month_col].isin([0, 1, 2, 3, 4, 5])][target_col].mean()
        valid = df[df[month_col] == 6][target_col].mean() if 6 in by_month.index else None
        test = df[df[month_col] == 7][target_col].mean() if 7 in by_month.index else None
        lit_key = "Base" if spec.alias_of == "base" else spec.deck_label
        rows.append(
            {
                "deck_label": spec.deck_label,
                "file": path.name,
                "exists": True,
                "n_rows": int(len(df)),
                "fraud_rate_overall": overall,
                "fraud_rate_train_0_5": float(train) if pd.notna(train) else None,
                "fraud_rate_valid_6": float(valid) if valid is not None and pd.notna(valid) else None,
                "fraud_rate_test_7": float(test) if test is not None and pd.notna(test) else None,
                "literature_fraud_rate_approx": LITERATURE_OVERALL_FRAUD_RATE.get(lit_key),
                "alias_of": spec.alias_of,
            }
        )
    return pd.DataFrame(rows)


def attach_lift(base_rates: pd.DataFrame, pr_auc_by_label: dict[str, float]) -> pd.DataFrame:
    """Add champion PR-AUC and lift = PR-AUC / empirical overall prevalence."""
    out = base_rates.copy()
    out["pr_auc"] = out["deck_label"].map(pr_auc_by_label)
    out["random_pr_baseline"] = out["fraud_rate_overall"]
    out["lift_vs_prevalence"] = out.apply(
        lambda r: (r["pr_auc"] / r["fraud_rate_overall"])
        if r.get("pr_auc") is not None and r.get("fraud_rate_overall")
        else None,
        axis=1,
    )
    return out
