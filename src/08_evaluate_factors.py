from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# Config
# =========================================================
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from config import (
    PROCESSED_DIR,
    OUTPUT_DIR,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. Load processed factors
# =========================================================
df = pd.read_parquet(
    PROCESSED_DIR
    / "factor_features_processed.parquet"
)


df["date"] = pd.to_datetime(
    df["date"]
)


df["code"] = (
    df["code"]
    .astype(str)
    .str.zfill(6)
)


df = (
    df
    .sort_values(
        ["date", "code"]
    )
    .reset_index(drop=True)
)


# =========================================================
# 2. Factors to evaluate
# =========================================================
factors = [
    "factor_momentum",
    "factor_volatility",
    "factor_liquidity",
    "factor_quality",
    "factor_growth",
    "factor_low_leverage",
    "reversal_5d_final",
]


targets = [
    "fwd_ret_5d",
    "fwd_ret_20d",
]


# =========================================================
# 3. Create month-end research snapshots
#
# We deliberately do not use every trading day
# for the main IC summary because forward-return
# windows overlap heavily.
# =========================================================
df["month"] = (
    df["date"]
    .dt.to_period("M")
)


month_end_dates = (
    df
    .groupby("month")["date"]
    .max()
)


month_end = (
    df[
        df["date"].isin(
            month_end_dates.values
        )
    ]
    .copy()
)


print("=" * 75)
print("FACTOR EVALUATION")
print("=" * 75)


print(
    f"Full panel rows: "
    f"{len(df):,}"
)

print(
    f"Month-end rows: "
    f"{len(month_end):,}"
)

print(
    f"Month-end dates: "
    f"{month_end['date'].nunique()}"
)


# =========================================================
# 4. Evaluate one cross-section
#
# Metrics:
# - Rank IC
# - Top quintile future return
# - Bottom quintile future return
# - Top-minus-bottom spread
# =========================================================
def evaluate_cross_section(
    data,
    factor,
    target,
    min_stocks=30,
):

    x = (
        data[
            [
                factor,
                target,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )


    if (
        len(x) < min_stocks
        or x[factor].nunique() < 5
    ):

        return None


    # ---------------------------------------------
    # Rank IC = Spearman correlation
    # ---------------------------------------------
    rank_ic = (
        x[factor]
        .corr(
            x[target],
            method="spearman",
        )
    )


    # ---------------------------------------------
    # Quintile spread
    # ---------------------------------------------
    q20 = (
        x[factor]
        .quantile(0.20)
    )

    q80 = (
        x[factor]
        .quantile(0.80)
    )


    bottom = (
        x.loc[
            x[factor] <= q20,
            target,
        ]
        .mean()
    )


    top = (
        x.loc[
            x[factor] >= q80,
            target,
        ]
        .mean()
    )


    spread = (
        top - bottom
    )


    return {
        "n": len(x),
        "rank_ic": rank_ic,
        "bottom_return": bottom,
        "top_return": top,
        "top_bottom_spread": spread,
    }


# =========================================================
# 5. Run month-end evaluation
# =========================================================
rows = []


for date, day in month_end.groupby(
    "date",
    sort=True,
):

    for factor in factors:

        for target in targets:

            result = (
                evaluate_cross_section(
                    day,
                    factor,
                    target,
                )
            )


            if result is None:
                continue


            rows.append(
                {
                    "date": date,
                    "factor": factor,
                    "target": target,
                    **result,
                }
            )


results = pd.DataFrame(
    rows
)


if results.empty:

    raise RuntimeError(
        "No valid factor evaluation results."
    )


# =========================================================
# 6. Aggregate factor summary
# =========================================================
summary = (
    results
    .groupby(
        [
            "factor",
            "target",
        ]
    )
    .agg(
        periods=(
            "rank_ic",
            "count",
        ),

        mean_rank_ic=(
            "rank_ic",
            "mean",
        ),

        median_rank_ic=(
            "rank_ic",
            "median",
        ),

        ic_std=(
            "rank_ic",
            "std",
        ),

        positive_ic_ratio=(
            "rank_ic",
            lambda x:
                (x > 0).mean(),
        ),

        mean_top_return=(
            "top_return",
            "mean",
        ),

        mean_bottom_return=(
            "bottom_return",
            "mean",
        ),

        mean_top_bottom_spread=(
            "top_bottom_spread",
            "mean",
        ),

        positive_spread_ratio=(
            "top_bottom_spread",
            lambda x:
                (x > 0).mean(),
        ),

        avg_n=(
            "n",
            "mean",
        ),
    )
    .reset_index()
)


# =========================================================
# 7. ICIR
#
# Monthly ICIR:
# mean(IC) / std(IC)
#
# Annualized version:
# sqrt(12) * mean / std
# =========================================================
summary[
    "ic_ir"
] = (
    summary[
        "mean_rank_ic"
    ]
    /
    summary[
        "ic_std"
    ]
)


summary[
    "ic_ir_annualized"
] = (
    np.sqrt(12)
    *
    summary[
        "ic_ir"
    ]
)


summary = (
    summary
    .sort_values(
        [
            "target",
            "mean_rank_ic",
        ],
        ascending=[
            True,
            False,
        ]
    )
)


# =========================================================
# 8. Year-by-year IC stability
# =========================================================
results["year"] = (
    results["date"]
    .dt.year
)


yearly_ic = (
    results
    .groupby(
        [
            "year",
            "factor",
            "target",
        ]
    )
    .agg(
        periods=(
            "rank_ic",
            "count",
        ),

        mean_rank_ic=(
            "rank_ic",
            "mean",
        ),

        positive_ic_ratio=(
            "rank_ic",
            lambda x:
                (x > 0).mean(),
        ),

        mean_spread=(
            "top_bottom_spread",
            "mean",
        ),
    )
    .reset_index()
)


# =========================================================
# 9. Factor correlation
#
# Use month-end observations to see whether
# different factors are largely duplicating
# one another.
# =========================================================
factor_corr = (
    month_end[
        factors
    ]
    .corr(
        method="spearman"
    )
)


# =========================================================
# 10. Print 20D results first
# =========================================================
print(
    "\n20-DAY FORWARD RETURN RESULTS"
)

print("-" * 75)


summary_20 = (
    summary[
        summary["target"]
        == "fwd_ret_20d"
    ]
    .copy()
)


display_cols = [
    "factor",
    "periods",
    "mean_rank_ic",
    "median_rank_ic",
    "ic_ir",
    "ic_ir_annualized",
    "positive_ic_ratio",
    "mean_top_bottom_spread",
    "positive_spread_ratio",
    "avg_n",
]


print(
    summary_20[
        display_cols
    ]
    .to_string(
        index=False,
        formatters={
            "mean_rank_ic":
                lambda x: f"{x:.4f}",

            "median_rank_ic":
                lambda x: f"{x:.4f}",

            "ic_ir":
                lambda x: f"{x:.3f}",

            "ic_ir_annualized":
                lambda x: f"{x:.3f}",

            "positive_ic_ratio":
                lambda x: f"{x:.1%}",

            "mean_top_bottom_spread":
                lambda x: f"{x:.2%}",

            "positive_spread_ratio":
                lambda x: f"{x:.1%}",

            "avg_n":
                lambda x: f"{x:.1f}",
        }
    )
)


# =========================================================
# 11. Print 5D results
# =========================================================
print(
    "\n5-DAY FORWARD RETURN RESULTS"
)

print("-" * 75)


summary_5 = (
    summary[
        summary["target"]
        == "fwd_ret_5d"
    ]
    .copy()
)


print(
    summary_5[
        display_cols
    ]
    .to_string(
        index=False,
        formatters={
            "mean_rank_ic":
                lambda x: f"{x:.4f}",

            "median_rank_ic":
                lambda x: f"{x:.4f}",

            "ic_ir":
                lambda x: f"{x:.3f}",

            "ic_ir_annualized":
                lambda x: f"{x:.3f}",

            "positive_ic_ratio":
                lambda x: f"{x:.1%}",

            "mean_top_bottom_spread":
                lambda x: f"{x:.2%}",

            "positive_spread_ratio":
                lambda x: f"{x:.1%}",

            "avg_n":
                lambda x: f"{x:.1f}",
        }
    )
)


# =========================================================
# 12. Correlation matrix
# =========================================================
print(
    "\nFACTOR SPEARMAN CORRELATION"
)

print("-" * 75)


print(
    factor_corr
    .round(3)
    .to_string()
)


# =========================================================
# 13. Save outputs
# =========================================================
results.to_csv(
    OUTPUT_DIR
    / "factor_monthly_ic_detail.csv",
    index=False,
)


summary.to_csv(
    OUTPUT_DIR
    / "factor_evaluation_summary.csv",
    index=False,
)


yearly_ic.to_csv(
    OUTPUT_DIR
    / "factor_yearly_stability.csv",
    index=False,
)


factor_corr.to_csv(
    OUTPUT_DIR
    / "factor_correlation_matrix.csv"
)


print(
    f"\nOutputs saved to: "
    f"{OUTPUT_DIR}"
)


print(
    "\nPASS: Factor evaluation completed."
)