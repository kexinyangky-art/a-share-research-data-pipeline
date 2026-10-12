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

from config import PROCESSED_DIR


# =========================================================
# 1. Load raw factor panel
# =========================================================
df = pd.read_parquet(
    PROCESSED_DIR / "factor_features_raw.parquet"
)

df["date"] = pd.to_datetime(df["date"])

df["code"] = (
    df["code"]
    .astype(str)
    .str.zfill(6)
)

df = (
    df
    .sort_values(["date", "code"])
    .reset_index(drop=True)
)


# =========================================================
# 2. Define raw factors
# =========================================================
raw_factors = [
    "mom_20d",
    "mom_60d",
    "reversal_5d",
    "vol_20d",
    "turnover_20d",
    "amihud_20d",
    "quality_roe_raw",
    "quality_cash_raw",
    "growth_profit_raw",
    "leverage_raw",
]


# =========================================================
# 3. Cross-sectional winsorization
#
# For each trading date:
# clip factor values to 1st / 99th percentile
# =========================================================
def winsorize_cross_section(
    series,
    lower=0.01,
    upper=0.99
):

    valid = series.dropna()

    if len(valid) < 10:
        return series

    lo = valid.quantile(lower)
    hi = valid.quantile(upper)

    return series.clip(
        lower=lo,
        upper=hi
    )


for factor in raw_factors:

    df[
        f"{factor}_win"
    ] = (
        df
        .groupby("date")[
            factor
        ]
        .transform(
            winsorize_cross_section
        )
    )


# =========================================================
# 4. Cross-sectional z-score
#
# z = (x - mean) / std
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
            index=series.index
        )

    return (
        series - mean
    ) / std


for factor in raw_factors:

    win_col = (
        f"{factor}_win"
    )

    z_col = (
        f"{factor}_z"
    )

    df[z_col] = (
        df
        .groupby("date")[
            win_col
        ]
        .transform(
            zscore_cross_section
        )
    )


# =========================================================
# 5. Simple industry neutralization
#
# Idea:
# subtract same-day industry mean
#
# Only neutralize industries with >= 3 valid stocks.
# Small groups remain unchanged.
# =========================================================
def neutralize_one_day(
    day_df,
    factor_col
):

    result = day_df[
        factor_col
    ].copy()

    if (
        "industry"
        not in day_df.columns
    ):
        return result

    for industry, idx in (
        day_df
        .groupby("industry")
        .groups
        .items()
    ):

        values = day_df.loc[
            idx,
            factor_col
        ]

        valid = values.dropna()

        if len(valid) >= 3:

            result.loc[
                valid.index
            ] = (
                valid
                - valid.mean()
            )

    return result


# Fundamental and style factors for which
# industry effects may matter
neutralize_factors = [
    "mom_20d",
    "mom_60d",
    "vol_20d",
    "turnover_20d",
    "amihud_20d",
    "quality_roe_raw",
    "quality_cash_raw",
    "growth_profit_raw",
    "leverage_raw",
]


for factor in neutralize_factors:

    z_col = f"{factor}_z"

    neut_col = (
        f"{factor}_industry_neutral"
    )

    pieces = []

    for date, day in df.groupby(
        "date",
        sort=False
    ):

        neutralized = (
            neutralize_one_day(
                day,
                z_col
            )
        )

        pieces.append(
            neutralized
        )

    df[neut_col] = pd.concat(
        pieces
    ).sort_index()


# =========================================================
# 6. Re-standardize neutralized factors
#
# After industry demeaning,
# make cross-sectional scale comparable again.
# =========================================================
for factor in neutralize_factors:

    neut_col = (
        f"{factor}_industry_neutral"
    )

    final_col = (
        f"{factor}_final"
    )

    df[final_col] = (
        df
        .groupby("date")[
            neut_col
        ]
        .transform(
            zscore_cross_section
        )
    )


# reversal:
# only cross-sectional z-score for now
df[
    "reversal_5d_final"
] = df[
    "reversal_5d_z"
]


# =========================================================
# 7. Create interpretable factor directions
#
# Higher final score should generally mean
# "more of that characteristic"
# =========================================================

# Momentum:
# higher = stronger recent return
df[
    "factor_momentum"
] = df[
    "mom_20d_final"
]


# Volatility:
# higher = more volatile
df[
    "factor_volatility"
] = df[
    "vol_20d_final"
]


# Liquidity:
# Amihud high = illiquid
#
# We flip sign so:
# higher factor_liquidity = MORE liquid
df[
    "factor_liquidity"
] = (
    -df[
        "amihud_20d_final"
    ]
)


# Quality:
# First version:
# use ROE only as the cleanest universal signal
df[
    "factor_quality"
] = df[
    "quality_roe_raw_final"
]


# Growth:
# higher = stronger net profit growth
df[
    "factor_growth"
] = df[
    "growth_profit_raw_final"
]


# Leverage:
# flip sign so:
# higher = lower leverage
df[
    "factor_low_leverage"
] = (
    -df[
        "leverage_raw_final"
    ]
)


# =========================================================
# 8. Coverage / distribution summary
# =========================================================
final_factors = [
    "factor_momentum",
    "factor_volatility",
    "factor_liquidity",
    "factor_quality",
    "factor_growth",
    "factor_low_leverage",
    "reversal_5d_final",
]


print("=" * 70)
print("PROCESSED FACTOR SUMMARY")
print("=" * 70)


print(
    f"Rows: {len(df):,}"
)

print(
    f"Stocks: "
    f"{df['code'].nunique()}"
)


print("\nFinal factor coverage:")

for col in final_factors:

    print(
        f"{col:25s} "
        f"{df[col].notna().mean():.2%}"
    )


print("\nFinal factor distribution:")

summary = (
    df[
        final_factors
    ]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
    .T
)

print(
    summary.to_string()
)


# =========================================================
# 9. Daily mean / std diagnostic
#
# Properly standardized factors should have
# daily cross-sectional mean near 0
# and std near 1.
# =========================================================
diagnostics = []

for col in final_factors:

    daily = (
        df
        .groupby("date")[col]
        .agg(
            daily_mean="mean",
            daily_std="std",
            n="count",
        )
    )

    diagnostics.append(
        {
            "factor": col,
            "avg_abs_daily_mean":
                daily[
                    "daily_mean"
                ].abs().mean(),

            "avg_daily_std":
                daily[
                    "daily_std"
                ].mean(),

            "avg_n":
                daily["n"].mean(),
        }
    )


diagnostics = pd.DataFrame(
    diagnostics
)


print("\nStandardization diagnostics:")

print(
    diagnostics.to_string(
        index=False
    )
)


# =========================================================
# 10. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "factor_features_processed.parquet"
)


df.to_parquet(
    output_file,
    index=False
)


summary.to_csv(
    PROCESSED_DIR
    / "processed_factor_distribution.csv"
)


diagnostics.to_csv(
    PROCESSED_DIR
    / "factor_standardization_diagnostics.csv",
    index=False
)


print(
    f"\nSaved to: "
    f"{output_file}"
)

print(
    "\nPASS: "
    "Winsorization, z-score and "
    "industry neutralization completed."
)