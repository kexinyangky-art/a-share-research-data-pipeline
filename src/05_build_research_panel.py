from pathlib import Path
import sys

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
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. Load datasets
# =========================================================
prices = pd.read_parquet(
    RAW_DIR / "prices_hfq.parquet"
)

profiles = pd.read_parquet(
    RAW_DIR / "profiles.parquet"
)

fund = pd.read_parquet(
    PROCESSED_DIR / "fundamentals_clean.parquet"
)


# =========================================================
# 2. Standardize code
# =========================================================
for df in [
    prices,
    profiles,
    fund,
]:

    df["code"] = (
        df["code"]
        .astype(str)
        .str.zfill(6)
    )


# =========================================================
# 3. Standardize dates
# =========================================================
prices["date"] = pd.to_datetime(
    prices["date"]
)

fund["report_date"] = pd.to_datetime(
    fund["report_date"]
)

fund["available_date"] = pd.to_datetime(
    fund["available_date"]
)


# =========================================================
# 4. Add stock metadata to daily market data
# =========================================================
profile_cols = [
    "code",
    "name",
    "industry",
    "listing_date",
]


profile_cols = [
    c
    for c in profile_cols
    if c in profiles.columns
]


prices = prices.merge(
    profiles[
        profile_cols
    ],
    on="code",
    how="left",
)


# =========================================================
# 5. PIT join:
#
# For every trading date, use the most recent
# fundamental observation whose available_date
# is <= trading date.
#
# Do this stock by stock.
# =========================================================
panel_parts = []


for code, px in prices.groupby(
    "code",
    sort=False
):

    px = (
        px
        .sort_values("date")
        .copy()
    )


    f = (
        fund[
            fund["code"] == code
        ]
        .drop(
            columns=[
                "code",
                "industry",
            ],
            errors="ignore",
        )
        .sort_values(
            "available_date"
        )
        .copy()
    )


    # No fundamentals for this stock
    if f.empty:

        panel_parts.append(
            px
        )

        continue


    merged = pd.merge_asof(
        px,
        f,

        left_on="date",

        right_on="available_date",

        direction="backward",

        allow_exact_matches=True,
    )


    panel_parts.append(
        merged
    )


# =========================================================
# 6. Combine all stocks
# =========================================================
panel = pd.concat(
    panel_parts,
    ignore_index=True
)


panel = (
    panel
    .sort_values(
        [
            "date",
            "code",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 7. PIT quality checks
# =========================================================

# ---------------------------------------------
# A. Row count should not change
# ---------------------------------------------
input_rows = len(prices)
output_rows = len(panel)


# ---------------------------------------------
# B. No duplicate stock-date
# ---------------------------------------------
duplicate_rows = (
    panel
    .duplicated(
        [
            "code",
            "date",
        ]
    )
    .sum()
)


# ---------------------------------------------
# C. No future fundamental information
# ---------------------------------------------
future_leak = (
    (
        panel["available_date"].notna()
    )
    &
    (
        panel["available_date"]
        >
        panel["date"]
    )
).sum()


# ---------------------------------------------
# D. Fundamental coverage
#
# Early 2022 dates may legitimately have no
# quarterly fundamental information yet.
# ---------------------------------------------
fund_coverage = (
    panel[
        "available_date"
    ]
    .notna()
    .mean()
)


# ---------------------------------------------
# E. Industry coverage
# ---------------------------------------------
industry_coverage = (
    panel[
        "industry"
    ]
    .notna()
    .mean()
)


# =========================================================
# 8. Calculate age of fundamental information
# =========================================================
panel[
    "fundamental_age_days"
] = (
    panel["date"]
    -
    panel["available_date"]
).dt.days


# =========================================================
# 9. Report-date quarter label
# =========================================================
panel[
    "fundamental_period"
] = (
    panel[
        "report_date"
    ]
    .dt.to_period("Q")
    .astype("string")
)


# =========================================================
# 10. Quality Summary
# =========================================================
print("=" * 70)

print(
    "POINT-IN-TIME RESEARCH PANEL"
)

print("=" * 70)


print(
    f"Input market rows: "
    f"{input_rows:,}"
)

print(
    f"Output panel rows: "
    f"{output_rows:,}"
)

print(
    f"Stocks: "
    f"{panel['code'].nunique()}"
)

print(
    f"Date range: "
    f"{panel['date'].min()} "
    f"-> "
    f"{panel['date'].max()}"
)


print(
    f"\nDuplicate code-date rows: "
    f"{duplicate_rows}"
)

print(
    f"Future fundamental leaks: "
    f"{future_leak}"
)

print(
    f"Fundamental coverage: "
    f"{fund_coverage:.2%}"
)

print(
    f"Industry coverage: "
    f"{industry_coverage:.2%}"
)


# =========================================================
# 11. Fundamental age distribution
# =========================================================
print(
    "\nFundamental age distribution:"
)


print(
    panel[
        "fundamental_age_days"
    ]
    .describe(
        percentiles=[
            0.05,
            0.50,
            0.95,
            0.99,
        ]
    )
)


# =========================================================
# 12. Critical assertions
# =========================================================
assert (
    input_rows
    ==
    output_rows
), (
    "Row count changed during PIT join."
)


assert (
    duplicate_rows
    == 0
), (
    "Duplicate code-date rows found."
)


assert (
    future_leak
    == 0
), (
    "LOOK-AHEAD BIAS DETECTED."
)


# =========================================================
# 13. Sample around a fundamental update
# =========================================================
sample_code = "300033"


sample = (
    panel[
        panel["code"]
        == sample_code
    ]
    [
        [
            "date",
            "code",
            "name",
            "close",
            "report_date",
            "available_date",
            "fundamental_period",
            "roe_avg",
            "net_profit_yoy",
            "debt_asset_ratio_clean",
            "cfo_net_profit_ratio",
        ]
    ]
    .tail(30)
)


print(
    f"\nSample - {sample_code}:"
)


print(
    sample.to_string(
        index=False
    )
)


# =========================================================
# 14. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "research_panel_base.parquet"
)


panel.to_parquet(
    output_file,
    index=False
)


print(
    f"\nSaved to: "
    f"{output_file}"
)

print(
    "\nPASS: "
    "Point-in-time research panel "
    "successfully built."
)