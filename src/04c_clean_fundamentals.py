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

from config import RAW_DIR, PROCESSED_DIR


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. Load raw fundamental data
# =========================================================
df = pd.read_parquet(
    RAW_DIR / "fundamentals_quarterly.parquet"
)


df["code"] = (
    df["code"]
    .astype(str)
    .str.zfill(6)
)


df["report_date"] = pd.to_datetime(
    df["report_date"]
)

df["available_date"] = pd.to_datetime(
    df["available_date"]
)


# =========================================================
# 2. Reconstruct debt / asset ratio
#
# asset_equity_multiplier
# = Assets / Equity
#
# Debt / Asset
# = 1 - Equity / Assets
# = 1 - 1 / (Assets / Equity)
# =========================================================
multiplier = pd.to_numeric(
    df["asset_equity_multiplier"],
    errors="coerce"
)


df[
    "debt_asset_ratio_derived"
] = np.where(
    multiplier > 0,
    1 - 1 / multiplier,
    np.nan
)


# =========================================================
# 3. Compare vendor field vs derived value
# =========================================================
df[
    "debt_ratio_diff"
] = (
    df["debt_asset_ratio"]
    -
    df["debt_asset_ratio_derived"]
).abs()


# Flag large disagreements
df[
    "debt_ratio_vendor_flag"
] = (
    df["debt_ratio_diff"]
    > 0.05
)


# Prefer derived ratio when valid.
# Otherwise fall back to raw vendor field.
df[
    "debt_asset_ratio_clean"
] = (
    df[
        "debt_asset_ratio_derived"
    ]
    .where(
        df[
            "debt_asset_ratio_derived"
        ].between(
            0,
            1.5
        )
    )
    .fillna(
        df[
            "debt_asset_ratio"
        ]
    )
)


# =========================================================
# 4. Revenue quality flag
#
# We keep the raw field for traceability,
# but do NOT use it as a research feature because
# Q1/Q3 coverage is structurally poor.
# =========================================================
df[
    "revenue_usable"
] = False


# =========================================================
# 5. Financial-sector gross margin applicability
# =========================================================
profiles = pd.read_parquet(
    RAW_DIR / "profiles.parquet"
)


profiles["code"] = (
    profiles["code"]
    .astype(str)
    .str.zfill(6)
)


df = df.merge(
    profiles[
        [
            "code",
            "industry",
        ]
    ],
    on="code",
    how="left"
)


df[
    "is_financial"
] = (
    df["industry"]
    .fillna("")
    .str.startswith(
        (
            "J66",
            "J67",
        )
    )
)


# Gross margin is not treated as a universal
# feature for financial companies.
df[
    "gross_margin_clean"
] = df[
    "gross_margin"
]


df.loc[
    df["is_financial"],
    "gross_margin_clean"
] = np.nan


# =========================================================
# 6. Select research-ready fields
#
# Keep raw fields as well where useful,
# but explicitly define the inputs we trust.
# =========================================================
research_cols = [
    "code",
    "report_date",
    "available_date",
    "industry",

    # profitability
    "roe_avg",
    "net_margin",
    "gross_margin_clean",
    "net_profit",
    "eps_ttm",

    # growth
    "net_profit_yoy",
    "equity_yoy",
    "asset_yoy",
    "eps_basic_yoy",

    # leverage
    "debt_asset_ratio_clean",
    "current_ratio",
    "quick_ratio",

    # cash-flow quality
    "cfo_net_profit_ratio",
    "cfo_revenue_ratio",

    # audit / traceability
    "debt_asset_ratio",
    "debt_asset_ratio_derived",
    "debt_ratio_diff",
    "debt_ratio_vendor_flag",

    # raw revenue retained for traceability
    "revenue",
]


research_cols = [
    c
    for c in research_cols
    if c in df.columns
]


clean = (
    df[
        research_cols
    ]
    .sort_values(
        [
            "code",
            "report_date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 7. Quality summary
# =========================================================
print("=" * 60)
print("CLEAN FUNDAMENTAL SUMMARY")
print("=" * 60)


print(
    f"Rows: {len(clean):,}"
)

print(
    f"Stocks: "
    f"{clean['code'].nunique()}"
)


print(
    "\nDebt ratio vendor discrepancies "
    "(absolute difference > 5pp):"
)

print(
    int(
        clean[
            "debt_ratio_vendor_flag"
        ].sum()
    )
)


print(
    "\nClean debt ratio distribution:"
)

print(
    clean[
        "debt_asset_ratio_clean"
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
    "\nKey missing rates:"
)


for col in [
    "roe_avg",
    "gross_margin_clean",
    "net_profit_yoy",
    "debt_asset_ratio_clean",
    "cfo_net_profit_ratio",
]:

    if col in clean.columns:

        print(
            f"{col:28s} "
            f"{clean[col].isna().mean():.2%}"
        )


# =========================================================
# 8. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "fundamentals_clean.parquet"
)


clean.to_parquet(
    output_file,
    index=False
)


print(
    f"\nSaved to: "
    f"{output_file}"
)