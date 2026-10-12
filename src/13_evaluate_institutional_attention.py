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
# 1. Load alternative-data panel
# =========================================================
df = pd.read_parquet(
    PROCESSED_DIR
    / "institutional_attention_features.parquet"
)

df["date"] = pd.to_datetime(
    df["date"]
)

df["code"] = (
    df["code"]
    .astype(str)
    .str.zfill(6)
)


# Only use mature alternative-data history
df = (
    df[
        df["alt_data_eligible"]
    ]
    .copy()
    .sort_values(
        [
            "date",
            "code",
        ]
    )
    .reset_index(drop=True)
)


print("=" * 80)
print("INSTITUTIONAL ATTENTION EVALUATION")
print("=" * 80)

print(
    f"Eligible rows: "
    f"{len(df):,}"
)

print(
    f"Stocks: "
    f"{df['code'].nunique()}"
)

print(
    f"Date range: "
    f"{df['date'].min()} "
    f"-> "
    f"{df['date'].max()}"
)


# =========================================================
# 2. Define raw alternative-data factors
# =========================================================

# Already log1p(institution_count_30d)
df[
    "alt_attention_30d_raw"
] = df[
    "attention_intensity_30d"
]


# Longer-term institutional attention
df[
    "alt_attention_90d_raw"
] = np.log1p(
    df[
        "institution_count_90d"
    ]
)


# Is recent attention accelerating?
df[
    "alt_acceleration_raw"
] = df[
    "attention_acceleration"
]


# Number of separate research events,
# not just number of participating institutions
df[
    "alt_event_frequency_raw"
] = np.log1p(
    df[
        "research_events_30d"
    ]
)


raw_alt_factors = [
    "alt_attention_30d_raw",
    "alt_attention_90d_raw",
    "alt_acceleration_raw",
    "alt_event_frequency_raw",
]


# =========================================================
# 3. Cross-sectional winsorization
# =========================================================
def winsorize_cross_section(
    series,
    lower=0.01,
    upper=0.99,
):

    valid = series.dropna()

    if len(valid) < 10:
        return series

    low = valid.quantile(
        lower
    )

    high = valid.quantile(
        upper
    )

    return series.clip(
        lower=low,
        upper=high,
    )


# =========================================================
# 4. Cross-sectional z-score
# =========================================================
def zscore_cross_section(
    series
):

    mean = series.mean()
    std = series.std()

    if (
        pd.isna(std)
        or std == 0
    ):

        return pd.Series(
            np.nan,
            index=series.index,
        )

    return (
        series - mean
    ) / std


for factor in raw_alt_factors:

    win_col = (
        f"{factor}_win"
    )

    z_col = (
        f"{factor}_z"
    )

    df[
        win_col
    ] = (
        df
        .groupby("date")[
            factor
        ]
        .transform(
            winsorize_cross_section
        )
    )

    df[
        z_col
    ] = (
        df
        .groupby("date")[
            win_col
        ]
        .transform(
            zscore_cross_section
        )
    )


# =========================================================
# 5. Industry neutralization
# =========================================================
def neutralize_one_day(
    day,
    factor_col,
):

    result = (
        day[
            factor_col
        ]
        .copy()
    )

    for industry, idx in (
        day
        .groupby("industry")
        .groups
        .items()
    ):

        values = day.loc[
            idx,
            factor_col,
        ]

        valid = (
            values
            .dropna()
        )

        # Avoid unstable adjustment
        # for tiny industries
        if len(valid) >= 3:

            result.loc[
                valid.index
            ] = (
                valid
                - valid.mean()
            )

    return result


for factor in raw_alt_factors:

    z_col = (
        f"{factor}_z"
    )

    neutral_col = (
        f"{factor}_industry_neutral"
    )

    pieces = []

    for date, day in df.groupby(
        "date",
        sort=False,
    ):

        pieces.append(
            neutralize_one_day(
                day,
                z_col,
            )
        )

    df[
        neutral_col
    ] = (
        pd.concat(
            pieces
        )
        .sort_index()
    )


# =========================================================
# 6. Re-standardize
# =========================================================
for factor in raw_alt_factors:

    neutral_col = (
        f"{factor}_industry_neutral"
    )

    final_col = (
        f"{factor}_final"
    )

    df[
        final_col
    ] = (
        df
        .groupby("date")[
            neutral_col
        ]
        .transform(
            zscore_cross_section
        )
    )


# =========================================================
# 7. Friendly factor names
# =========================================================
df[
    "factor_attention_30d"
] = df[
    "alt_attention_30d_raw_final"
]


df[
    "factor_attention_90d"
] = df[
    "alt_attention_90d_raw_final"
]


df[
    "factor_attention_acceleration"
] = df[
    "alt_acceleration_raw_final"
]


df[
    "factor_research_frequency"
] = df[
    "alt_event_frequency_raw_final"
]


# =========================================================
# 8. Remove Size + Liquidity exposure
#
# Institutional attention may naturally be higher
# for large and liquid stocks.
#
# Regress:
#
# Attention =
# alpha
# + beta1 * Size
# + beta2 * Size-neutral Liquidity
# + residual
#
# Residual = incremental attention not explained
# by these conventional market characteristics.
# =========================================================
def residualize_attention(
    day
):

    result = pd.Series(
        np.nan,
        index=day.index,
        dtype=float,
    )

    cols = [
        "factor_attention_30d",
        "factor_size",
        "factor_liquidity_size_neutral",
    ]

    x = (
        day[
            cols
        ]
        .dropna()
    )

    if len(x) < 30:

        return result


    X = np.column_stack(
        [
            np.ones(
                len(x)
            ),

            x[
                "factor_size"
            ].values,

            x[
                "factor_liquidity_size_neutral"
            ].values,
        ]
    )


    y = x[
        "factor_attention_30d"
    ].values


    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]


    residual = (
        y
        -
        X @ beta
    )


    result.loc[
        x.index
    ] = residual


    return result


pieces = []


for date, day in df.groupby(
    "date",
    sort=False,
):

    pieces.append(
        residualize_attention(
            day
        )
    )


df[
    "attention_incremental_raw"
] = (
    pd.concat(
        pieces
    )
    .sort_index()
)


df[
    "factor_attention_incremental"
] = (
    df
    .groupby("date")[
        "attention_incremental_raw"
    ]
    .transform(
        zscore_cross_section
    )
)


# =========================================================
# 9. Check exposure before / after controls
# =========================================================
corr_cols = [
    "factor_attention_30d",
    "factor_attention_incremental",
    "factor_size",
    "factor_liquidity_size_neutral",
    "factor_value",
]


factor_corr = (
    df[
        corr_cols
    ]
    .corr(
        method="spearman"
    )
)


print("\n")
print("ALTERNATIVE FACTOR CORRELATION")
print("-" * 80)

print(
    factor_corr
    .round(3)
    .to_string()
)


# =========================================================
# 10. Month-end snapshots
#
# NOTE:
# Only ~8 months of mature alternative data.
# This is EXPLORATORY, not production validation.
# =========================================================
df[
    "month"
] = (
    df["date"]
    .dt.to_period("M")
)


month_end_dates = (
    df
    .groupby("month")[
        "date"
    ]
    .max()
)


month_end = (
    df[
        df["date"]
        .isin(
            month_end_dates.values
        )
    ]
    .copy()
)


print(
    f"\nMature month-end dates: "
    f"{month_end['date'].nunique()}"
)


# =========================================================
# 11. Evaluation function
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


    rank_ic = (
        x[factor]
        .corr(
            x[target],
            method="spearman",
        )
    )


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
        "n":
            len(x),

        "rank_ic":
            rank_ic,

        "top_return":
            top_return,

        "bottom_return":
            bottom_return,

        "top_bottom_spread":
            top_return
            - bottom_return,
    }


# =========================================================
# 12. Evaluate factors
# =========================================================
alt_factors = [
    "factor_attention_30d",
    "factor_attention_90d",
    "factor_attention_acceleration",
    "factor_research_frequency",
    "factor_attention_incremental",
]


targets = [
    "fwd_ret_5d",
    "fwd_ret_20d",
]


rows = []


for date, day in month_end.groupby(
    "date",
    sort=True,
):

    for factor in alt_factors:

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
                    "date":
                        date,

                    "factor":
                        factor,

                    "target":
                        target,

                    **result,
                }
            )


results = pd.DataFrame(
    rows
)


# =========================================================
# 13. Aggregate summary
#
# IMPORTANT:
# We intentionally do NOT annualize ICIR here.
#
# The mature sample is too short.
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
    "ic_ir_unannualized"
] = (
    summary[
        "mean_rank_ic"
    ]
    /
    summary[
        "ic_std"
    ]
)


# =========================================================
# 14. Print
# =========================================================
def print_result(
    target,
):

    temp = (
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
    print(
        f"{target.upper()} - EXPLORATORY"
    )

    print("-" * 80)


    print(
        temp[
            [
                "factor",
                "periods",
                "mean_rank_ic",
                "median_rank_ic",
                "positive_ic_ratio",
                "mean_top_bottom_spread",
                "positive_spread_ratio",
                "avg_n",
            ]
        ]
        .to_string(
            index=False,

            formatters={
                "mean_rank_ic":
                    lambda x:
                        f"{x:.4f}",

                "median_rank_ic":
                    lambda x:
                        f"{x:.4f}",

                "positive_ic_ratio":
                    lambda x:
                        f"{x:.1%}",

                "mean_top_bottom_spread":
                    lambda x:
                        f"{x:.2%}",

                "positive_spread_ratio":
                    lambda x:
                        f"{x:.1%}",

                "avg_n":
                    lambda x:
                        f"{x:.1f}",
            }
        )
    )


print_result(
    "fwd_ret_5d"
)


print_result(
    "fwd_ret_20d"
)


# =========================================================
# 15. Save
# =========================================================
results.to_csv(
    OUTPUT_DIR
    / "institutional_attention_ic_detail.csv",
    index=False,
)


summary.to_csv(
    OUTPUT_DIR
    / "institutional_attention_evaluation.csv",
    index=False,
)


factor_corr.to_csv(
    OUTPUT_DIR
    / "institutional_attention_factor_correlation.csv"
)


processed_file = (
    PROCESSED_DIR
    / "factor_features_with_alternative_data.parquet"
)


df.to_parquet(
    processed_file,
    index=False,
)


print(
    f"\nOutputs saved to: "
    f"{OUTPUT_DIR}"
)


print(
    f"Processed panel saved to: "
    f"{processed_file}"
)


print(
    "\nPASS: "
    "Exploratory alternative-data "
    "evaluation completed."
)