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
    exist_ok=True,
)


# =========================================================
# 1. Load extended factor panel
# =========================================================
df = pd.read_parquet(
    PROCESSED_DIR
    / "factor_features_extended.parquet"
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
# 2. Factor list
# =========================================================
factors = [
    "factor_size",
    "factor_value",

    # Compare original vs Size-neutral liquidity
    "factor_liquidity",
    "factor_liquidity_size_neutral",

    "factor_momentum",
    "reversal_5d_final",
    "factor_volatility",
    "factor_quality",
    "factor_growth",
    "factor_low_leverage",
]


targets = [
    "fwd_ret_5d",
    "fwd_ret_20d",
]


# =========================================================
# 3. Month-end research snapshots
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


print("=" * 80)
print("EXTENDED FACTOR EVALUATION")
print("=" * 80)

print(
    f"Full panel rows: "
    f"{len(df):,}"
)

print(
    f"Month-end dates: "
    f"{month_end['date'].nunique()}"
)


# =========================================================
# 4. Evaluate one cross-section
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


    # Rank IC
    rank_ic = (
        x[factor]
        .corr(
            x[target],
            method="spearman",
        )
    )


    # Top / bottom quintiles
    q20 = (
        x[factor]
        .quantile(0.20)
    )

    q80 = (
        x[factor]
        .quantile(0.80)
    )


    bottom_return = (
        x.loc[
            x[factor] <= q20,
            target,
        ]
        .mean()
    )


    top_return = (
        x.loc[
            x[factor] >= q80,
            target,
        ]
        .mean()
    )


    return {
        "n": len(x),

        "rank_ic":
            rank_ic,

        "bottom_return":
            bottom_return,

        "top_return":
            top_return,

        "top_bottom_spread":
            top_return
            - bottom_return,
    }


# =========================================================
# 5. Run evaluation
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


# =========================================================
# 6. Aggregate
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


# =========================================================
# 7. Function for printing
# =========================================================
def print_summary(
    target,
    title,
):

    x = (
        summary[
            summary["target"]
            == target
        ]
        .sort_values(
            "mean_rank_ic",
            ascending=False,
        )
        .copy()
    )


    print("\n")
    print(title)
    print("-" * 80)


    cols = [
        "factor",
        "periods",
        "mean_rank_ic",
        "median_rank_ic",
        "ic_ir_annualized",
        "positive_ic_ratio",
        "mean_top_bottom_spread",
        "positive_spread_ratio",
        "avg_n",
    ]


    print(
        x[
            cols
        ]
        .to_string(
            index=False,

            formatters={
                "mean_rank_ic":
                    lambda v:
                        f"{v:.4f}",

                "median_rank_ic":
                    lambda v:
                        f"{v:.4f}",

                "ic_ir_annualized":
                    lambda v:
                        f"{v:.3f}",

                "positive_ic_ratio":
                    lambda v:
                        f"{v:.1%}",

                "mean_top_bottom_spread":
                    lambda v:
                        f"{v:.2%}",

                "positive_spread_ratio":
                    lambda v:
                        f"{v:.1%}",

                "avg_n":
                    lambda v:
                        f"{v:.1f}",
            }
        )
    )


print_summary(
    "fwd_ret_20d",
    "20-DAY FORWARD RETURN",
)


print_summary(
    "fwd_ret_5d",
    "5-DAY FORWARD RETURN",
)


# =========================================================
# 8. Liquidity-specific comparison
# =========================================================
liquidity_comparison = (
    summary[
        summary["factor"].isin(
            [
                "factor_liquidity",
                "factor_liquidity_size_neutral",
                "factor_size",
            ]
        )
    ]
    .sort_values(
        [
            "target",
            "factor",
        ]
    )
    .copy()
)


print("\n")
print("=" * 80)
print("LIQUIDITY / SIZE DECOMPOSITION")
print("=" * 80)


print(
    liquidity_comparison[
        [
            "factor",
            "target",
            "mean_rank_ic",
            "positive_ic_ratio",
            "mean_top_bottom_spread",
        ]
    ]
    .to_string(
        index=False,

        formatters={
            "mean_rank_ic":
                lambda v:
                    f"{v:.4f}",

            "positive_ic_ratio":
                lambda v:
                    f"{v:.1%}",

            "mean_top_bottom_spread":
                lambda v:
                    f"{v:.2%}",
        }
    )
)


# =========================================================
# 9. Daily Size-Liquidity correlation diagnostics
#
# OLS residualization should make daily PEARSON
# correlation with Size approximately zero.
# =========================================================
corr_rows = []


for date, day in df.groupby(
    "date",
    sort=True,
):

    raw = (
        day[
            [
                "factor_liquidity",
                "factor_size",
            ]
        ]
        .dropna()
    )

    neutral = (
        day[
            [
                "factor_liquidity_size_neutral",
                "factor_size",
            ]
        ]
        .dropna()
    )


    if len(raw) >= 20:

        raw_pearson = (
            raw[
                "factor_liquidity"
            ]
            .corr(
                raw[
                    "factor_size"
                ],
                method="pearson",
            )
        )

        raw_spearman = (
            raw[
                "factor_liquidity"
            ]
            .corr(
                raw[
                    "factor_size"
                ],
                method="spearman",
            )
        )

    else:

        raw_pearson = np.nan
        raw_spearman = np.nan


    if len(neutral) >= 20:

        neutral_pearson = (
            neutral[
                "factor_liquidity_size_neutral"
            ]
            .corr(
                neutral[
                    "factor_size"
                ],
                method="pearson",
            )
        )

        neutral_spearman = (
            neutral[
                "factor_liquidity_size_neutral"
            ]
            .corr(
                neutral[
                    "factor_size"
                ],
                method="spearman",
            )
        )

    else:

        neutral_pearson = np.nan
        neutral_spearman = np.nan


    corr_rows.append(
        {
            "date": date,

            "raw_pearson":
                raw_pearson,

            "raw_spearman":
                raw_spearman,

            "neutral_pearson":
                neutral_pearson,

            "neutral_spearman":
                neutral_spearman,
        }
    )


corr_df = pd.DataFrame(
    corr_rows
)


print("\n")
print("=" * 80)
print("DAILY SIZE / LIQUIDITY CORRELATION")
print("=" * 80)


for col in [
    "raw_pearson",
    "raw_spearman",
    "neutral_pearson",
    "neutral_spearman",
]:

    print(
        f"{col:25s}: "
        f"{corr_df[col].mean():.4f}"
    )


# =========================================================
# 10. Factor correlation at month-end
# =========================================================
corr_factors = [
    "factor_size",
    "factor_value",
    "factor_liquidity",
    "factor_liquidity_size_neutral",
    "factor_momentum",
    "reversal_5d_final",
]


factor_corr = (
    month_end[
        corr_factors
    ]
    .corr(
        method="spearman"
    )
)


print("\n")
print("=" * 80)
print("EXTENDED FACTOR CORRELATION")
print("=" * 80)


print(
    factor_corr
    .round(3)
    .to_string()
)


# =========================================================
# 11. Save
# =========================================================
results.to_csv(
    OUTPUT_DIR
    / "extended_factor_ic_detail.csv",
    index=False,
)


summary.to_csv(
    OUTPUT_DIR
    / "extended_factor_evaluation_summary.csv",
    index=False,
)


corr_df.to_csv(
    OUTPUT_DIR
    / "liquidity_size_daily_correlation.csv",
    index=False,
)


factor_corr.to_csv(
    OUTPUT_DIR
    / "extended_factor_correlation.csv"
)


print(
    f"\nOutputs saved to: "
    f"{OUTPUT_DIR}"
)


print(
    "\nPASS: "
    "Extended factor evaluation completed."
)