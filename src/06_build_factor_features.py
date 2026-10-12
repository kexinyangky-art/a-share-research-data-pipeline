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
# 1. Load PIT research panel
# =========================================================
df = pd.read_parquet(
    PROCESSED_DIR / "research_panel_base.parquet"
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
        ["code", "date"]
    )
    .reset_index(drop=True)
)


# =========================================================
# 2. Daily return
#
# Because close is adjusted price,
# pct_change is suitable for return calculations.
# =========================================================
df["ret_1d"] = (
    df
    .groupby("code")["close"]
    .pct_change()
)


# =========================================================
# 3. Momentum Factors
# =========================================================

# Past 20 trading-day return
df["mom_20d"] = (
    df["close"]
    /
    df.groupby("code")[
        "close"
    ].shift(20)
    - 1
)


# Past 60 trading-day return
df["mom_60d"] = (
    df["close"]
    /
    df.groupby("code")[
        "close"
    ].shift(60)
    - 1
)


# =========================================================
# 4. Short-Term Reversal
#
# If a stock rose strongly in the past 5 days,
# reversal factor becomes negative.
# =========================================================
df["ret_5d"] = (
    df["close"]
    /
    df.groupby("code")[
        "close"
    ].shift(5)
    - 1
)


df["reversal_5d"] = (
    -df["ret_5d"]
)


# =========================================================
# 5. Volatility
#
# 20-day rolling std of daily returns,
# annualized with sqrt(252)
# =========================================================
df["vol_20d"] = (
    df
    .groupby("code")[
        "ret_1d"
    ]
    .rolling(
        window=20,
        min_periods=15
    )
    .std()
    .reset_index(
        level=0,
        drop=True
    )
    * np.sqrt(252)
)


# =========================================================
# 6. Turnover / Liquidity
# =========================================================
df["turnover_20d"] = (
    df
    .groupby("code")[
        "turnover_pct"
    ]
    .rolling(
        window=20,
        min_periods=15
    )
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)


# =========================================================
# 7. Amihud Illiquidity
#
# |daily return| / trading amount
#
# Larger value:
# same trading amount causes larger price movement
# -> less liquid
#
# Scale by 1e9 only for readability.
# =========================================================
df["amihud_daily"] = (
    df["ret_1d"].abs()
    /
    df["amount"].replace(
        0,
        np.nan
    )
    * 1e9
)


df["amihud_20d"] = (
    df
    .groupby("code")[
        "amihud_daily"
    ]
    .rolling(
        window=20,
        min_periods=15
    )
    .mean()
    .reset_index(
        level=0,
        drop=True
    )
)


# =========================================================
# 8. Fundamental factor candidates
#
# These already came through the PIT join,
# so no future financial information is used.
# =========================================================

# Quality
df["quality_roe_raw"] = (
    df["roe_avg"]
)

df["quality_cash_raw"] = (
    df["cfo_net_profit_ratio"]
)


# Growth
df["growth_profit_raw"] = (
    df["net_profit_yoy"]
)


# Leverage
df["leverage_raw"] = (
    df["debt_asset_ratio_clean"]
)


# =========================================================
# 9. Forward returns
#
# IMPORTANT:
# These are LABELS for evaluation,
# NOT input factors.
# =========================================================
df["fwd_ret_5d"] = (
    df.groupby("code")[
        "close"
    ].shift(-5)
    /
    df["close"]
    - 1
)


df["fwd_ret_20d"] = (
    df.groupby("code")[
        "close"
    ].shift(-20)
    /
    df["close"]
    - 1
)


# =========================================================
# 10. Feature coverage summary
# =========================================================
factor_cols = [
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


print("=" * 70)
print("FACTOR FEATURE SUMMARY")
print("=" * 70)


print(
    f"Rows: {len(df):,}"
)

print(
    f"Stocks: {df['code'].nunique()}"
)


print("\nFeature coverage:")


for col in factor_cols:

    coverage = (
        df[col]
        .notna()
        .mean()
    )

    print(
        f"{col:25s} "
        f"{coverage:.2%}"
    )


# =========================================================
# 11. Distribution summary
# =========================================================
print(
    "\nRaw factor distribution:"
)


summary = (
    df[
        factor_cols
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
# 12. Quality Checks
# =========================================================

# No duplicate stock-date rows
duplicates = (
    df
    .duplicated(
        ["code", "date"]
    )
    .sum()
)


# Forward return must not be accidentally used
# as a factor input.
print(
    f"\nDuplicate code-date rows: "
    f"{duplicates}"
)


assert (
    duplicates == 0
)


# =========================================================
# 13. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "factor_features_raw.parquet"
)


df.to_parquet(
    output_file,
    index=False
)


summary.to_csv(
    PROCESSED_DIR
    / "factor_feature_distribution.csv"
)


print(
    f"\nSaved to: "
    f"{output_file}"
)

print(
    "\nPASS: Raw factor features created."
)