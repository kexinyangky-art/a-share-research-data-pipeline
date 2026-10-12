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
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. Load valuation data
# =========================================================
valuation = pd.read_parquet(
    RAW_DIR / "valuation_daily.parquet"
)

valuation["date"] = pd.to_datetime(
    valuation["date"]
)

valuation["code"] = (
    valuation["code"]
    .astype(str)
    .str.zfill(6)
)

valuation = (
    valuation
    .sort_values(
        ["code", "date"]
    )
    .reset_index(drop=True)
)


# =========================================================
# 2. Load processed factor panel
# =========================================================
factor_panel = pd.read_parquet(
    PROCESSED_DIR
    / "factor_features_processed.parquet"
)

factor_panel["date"] = pd.to_datetime(
    factor_panel["date"]
)

factor_panel["code"] = (
    factor_panel["code"]
    .astype(str)
    .str.zfill(6)
)


# =========================================================
# 3. Implied float shares
#
# Turnover (%)
# = Volume / Float Shares * 100
#
# therefore:
#
# Float Shares
# = Volume / (Turnover / 100)
# =========================================================
valid_turnover = (
    valuation["turnover_pct"]
    > 0
)

valid_volume = (
    valuation["volume_share"]
    > 0
)


valuation[
    "float_shares_implied_raw"
] = np.where(
    valid_turnover
    & valid_volume,

    valuation["volume_share"]
    /
    (
        valuation[
            "turnover_pct"
        ]
        / 100
    ),

    np.nan,
)


# =========================================================
# 4. Stabilize implied float shares
#
# Daily implied shares can move slightly because
# turnover is rounded by the data vendor.
#
# Use rolling 20-day median.
# =========================================================
valuation[
    "float_shares_implied"
] = (
    valuation
    .groupby("code")[
        "float_shares_implied_raw"
    ]
    .rolling(
        window=20,
        min_periods=5,
    )
    .median()
    .reset_index(
        level=0,
        drop=True,
    )
)


# =========================================================
# 5. Free-float market-cap proxy
#
# IMPORTANT:
# use UNADJUSTED price here.
# =========================================================
valuation[
    "float_market_cap_proxy"
] = (
    valuation["close_unadj"]
    *
    valuation["float_shares_implied"]
)


# Remove impossible / invalid values
valuation.loc[
    valuation[
        "float_market_cap_proxy"
    ] <= 0,
    "float_market_cap_proxy"
] = np.nan


# =========================================================
# 6. Raw Size factor
#
# Standard equity style-factor convention:
# log(market cap)
# =========================================================
valuation[
    "size_raw"
] = np.log(
    valuation[
        "float_market_cap_proxy"
    ]
)


# =========================================================
# 7. Raw Value factor
#
# Book-to-Price proxy:
#
# B/P = 1 / P/B
#
# PB must be positive.
# =========================================================
valuation[
    "value_bp_raw"
] = np.where(
    valuation["pb_mrq"] > 0,

    1
    /
    valuation["pb_mrq"],

    np.nan,
)


# =========================================================
# 8. Optional Earnings Yield
#
# Keep for analysis,
# but do NOT include in core Value yet.
# =========================================================
valuation[
    "earnings_yield_raw"
] = np.where(
    valuation["pe_ttm"] > 0,

    1
    /
    valuation["pe_ttm"],

    np.nan,
)


# =========================================================
# 9. Merge into factor panel
# =========================================================
valuation_cols = [
    "date",
    "code",
    "close_unadj",
    "pb_mrq",
    "pe_ttm",
    "float_shares_implied",
    "float_market_cap_proxy",
    "size_raw",
    "value_bp_raw",
    "earnings_yield_raw",
]


df = factor_panel.merge(
    valuation[
        valuation_cols
    ],
    on=[
        "date",
        "code",
    ],
    how="left",
    validate="one_to_one",
)


# =========================================================
# 10. Cross-sectional winsorization
# =========================================================
def winsorize_cross_section(
    series,
    lower=0.01,
    upper=0.99,
):

    valid = series.dropna()

    if len(valid) < 10:
        return series

    lo = valid.quantile(lower)
    hi = valid.quantile(upper)

    return series.clip(
        lower=lo,
        upper=hi,
    )


# =========================================================
# 11. Cross-sectional z-score
# =========================================================
def zscore_cross_section(series):

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


for factor in [
    "size_raw",
    "value_bp_raw",
]:

    win_col = (
        f"{factor}_win"
    )

    z_col = (
        f"{factor}_z"
    )

    df[win_col] = (
        df
        .groupby("date")[
            factor
        ]
        .transform(
            winsorize_cross_section
        )
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
# 12. Industry neutralization
#
# Subtract same-day industry mean
# only when industry has >=3 valid stocks.
# =========================================================
def neutralize_one_day(
    day_df,
    factor_col,
):

    result = (
        day_df[
            factor_col
        ]
        .copy()
    )

    for industry, idx in (
        day_df
        .groupby("industry")
        .groups
        .items()
    ):

        values = day_df.loc[
            idx,
            factor_col,
        ]

        valid = (
            values
            .dropna()
        )

        if len(valid) >= 3:

            result.loc[
                valid.index
            ] = (
                valid
                - valid.mean()
            )

    return result


for factor in [
    "size_raw",
    "value_bp_raw",
]:

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

        x = neutralize_one_day(
            day,
            z_col,
        )

        pieces.append(x)

    df[neutral_col] = (
        pd.concat(
            pieces
        )
        .sort_index()
    )


# =========================================================
# 13. Re-standardize
# =========================================================
for factor in [
    "size_raw",
    "value_bp_raw",
]:

    neutral_col = (
        f"{factor}_industry_neutral"
    )

    final_col = (
        f"{factor}_final"
    )

    df[final_col] = (
        df
        .groupby("date")[
            neutral_col
        ]
        .transform(
            zscore_cross_section
        )
    )


# =========================================================
# 14. Interpretable final factors
#
# High factor_size = larger
# High factor_value = cheaper / higher B/P
# =========================================================
df[
    "factor_size"
] = df[
    "size_raw_final"
]


df[
    "factor_value"
] = df[
    "value_bp_raw_final"
]


# =========================================================
# 15. Size-neutral Liquidity
#
# Current factor_liquidity already:
# - winsorized
# - standardized
# - industry adjusted
#
# Now regress it cross-sectionally on Size:
#
# Liquidity = a + b * Size + residual
#
# Residual = liquidity independent of Size.
# =========================================================
def residualize_on_size(day):

    result = pd.Series(
        np.nan,
        index=day.index,
        dtype=float,
    )

    x = day[
        [
            "factor_liquidity",
            "factor_size",
        ]
    ].dropna()

    if len(x) < 20:

        return result

    X = np.column_stack(
        [
            np.ones(len(x)),
            x[
                "factor_size"
            ].values,
        ]
    )

    y = x[
        "factor_liquidity"
    ].values

    beta = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )[0]

    fitted = (
        X @ beta
    )

    residual = (
        y - fitted
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

    residual = residualize_on_size(
        day
    )

    pieces.append(
        residual
    )


df[
    "liquidity_size_residual_raw"
] = (
    pd.concat(
        pieces
    )
    .sort_index()
)


# Re-standardize residual
df[
    "factor_liquidity_size_neutral"
] = (
    df
    .groupby("date")[
        "liquidity_size_residual_raw"
    ]
    .transform(
        zscore_cross_section
    )
)


# =========================================================
# 16. Diagnostics
# =========================================================
print("=" * 72)

print(
    "SIZE / VALUE FACTOR SUMMARY"
)

print("=" * 72)


print(
    f"Rows: "
    f"{len(df):,}"
)

print(
    f"Stocks: "
    f"{df['code'].nunique()}"
)


print(
    "\nCoverage:"
)


for col in [
    "float_market_cap_proxy",
    "factor_size",
    "factor_value",
    "factor_liquidity",
    "factor_liquidity_size_neutral",
]:

    print(
        f"{col:35s} "
        f"{df[col].notna().mean():.2%}"
    )


# =========================================================
# 17. Size sanity check
# =========================================================
print(
    "\nFloat market-cap proxy distribution:"
)


print(
    df[
        "float_market_cap_proxy"
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
)


print(
    "\nPB / Value distribution:"
)


print(
    df[
        [
            "pb_mrq",
            "value_bp_raw",
        ]
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


# =========================================================
# 18. Key correlation:
#
# Does liquidity contain Size exposure?
# =========================================================
corr_before = (
    df[
        [
            "factor_liquidity",
            "factor_size",
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[
        0,
        1
    ]
)


corr_after = (
    df[
        [
            "factor_liquidity_size_neutral",
            "factor_size",
        ]
    ]
    .corr(
        method="spearman"
    )
    .iloc[
        0,
        1
    ]
)


print(
    "\nLiquidity vs Size Spearman correlation:"
)

print(
    f"Before size neutralization: "
    f"{corr_before:.4f}"
)

print(
    f"After size neutralization:  "
    f"{corr_after:.4f}"
)


# =========================================================
# 19. Daily standardization diagnostics
# =========================================================
final_factors = [
    "factor_size",
    "factor_value",
    "factor_liquidity_size_neutral",
]


print(
    "\nDaily standardization diagnostics:"
)


for factor in final_factors:

    daily = (
        df
        .groupby("date")[
            factor
        ]
        .agg(
            mean="mean",
            std="std",
            n="count",
        )
    )

    print(
        f"\n{factor}"
    )

    print(
        f"avg abs mean: "
        f"{daily['mean'].abs().mean():.6f}"
    )

    print(
        f"avg std:      "
        f"{daily['std'].mean():.6f}"
    )

    print(
        f"avg n:        "
        f"{daily['n'].mean():.1f}"
    )


# =========================================================
# 20. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "factor_features_extended.parquet"
)


df.to_parquet(
    output_file,
    index=False,
)


print(
    f"\nSaved to: "
    f"{output_file}"
)

print(
    "\nPASS: "
    "Size, Value and size-neutral "
    "Liquidity factors created."
)