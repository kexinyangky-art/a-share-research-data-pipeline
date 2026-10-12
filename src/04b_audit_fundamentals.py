from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# 项目配置
# =========================================================
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from config import RAW_DIR, OUTPUT_DIR


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# 1. 读取数据
# =========================================================
fund = pd.read_parquet(
    RAW_DIR / "fundamentals_quarterly.parquet"
)

universe = pd.read_parquet(
    RAW_DIR / "universe.parquet"
)

profiles = pd.read_parquet(
    RAW_DIR / "profiles.parquet"
)


fund["code"] = (
    fund["code"]
    .astype(str)
    .str.zfill(6)
)

universe["code"] = (
    universe["code"]
    .astype(str)
    .str.zfill(6)
)

profiles["code"] = (
    profiles["code"]
    .astype(str)
    .str.zfill(6)
)


fund["report_date"] = pd.to_datetime(
    fund["report_date"]
)

fund["available_date"] = pd.to_datetime(
    fund["available_date"]
)


print("=" * 70)
print("FUNDAMENTAL DATA AUDIT")
print("=" * 70)


# =========================================================
# 2. 基础覆盖
# =========================================================
print("\n1. BASIC COVERAGE")
print("-" * 70)

print(
    f"Rows: {len(fund):,}"
)

print(
    f"Stocks: {fund['code'].nunique()}"
)

print(
    f"Duplicate code-report_date: "
    f"{fund.duplicated(['code', 'report_date']).sum()}"
)

print(
    f"Missing available_date: "
    f"{fund['available_date'].isna().sum()}"
)


# =========================================================
# 3. 构造理论 stock-quarter grid
# =========================================================
expected_dates = pd.to_datetime(
    [
        "2022-03-31",
        "2022-06-30",
        "2022-09-30",
        "2022-12-31",
        "2023-03-31",
        "2023-06-30",
        "2023-09-30",
        "2023-12-31",
        "2024-03-31",
        "2024-06-30",
        "2024-09-30",
        "2024-12-31",
        "2025-03-31",
        "2025-06-30",
        "2025-09-30",
        "2025-12-31",
        "2026-03-31",
        "2026-06-30",
    ]
)


expected = (
    pd.MultiIndex.from_product(
        [
            universe["code"].unique(),
            expected_dates,
        ],
        names=[
            "code",
            "report_date",
        ]
    )
    .to_frame(
        index=False
    )
)


actual_keys = (
    fund[
        [
            "code",
            "report_date",
        ]
    ]
    .drop_duplicates()
)


missing_periods = (
    expected
    .merge(
        actual_keys,
        on=[
            "code",
            "report_date",
        ],
        how="left",
        indicator=True,
    )
)


missing_periods = (
    missing_periods[
        missing_periods["_merge"]
        == "left_only"
    ]
    .drop(
        columns="_merge"
    )
)


# =========================================================
# 4. 判断缺失季度是否发生在上市前
# =========================================================
profiles["listing_date"] = pd.to_datetime(
    profiles["listing_date"],
    errors="coerce"
)


missing_periods = (
    missing_periods
    .merge(
        profiles[
            [
                "code",
                "name",
                "industry",
                "listing_date",
            ]
        ],
        on="code",
        how="left"
    )
)


missing_periods[
    "before_listing"
] = (
    missing_periods[
        "listing_date"
    ].notna()
    &
    (
        missing_periods[
            "report_date"
        ]
        <
        missing_periods[
            "listing_date"
        ]
    )
)


print("\n2. MISSING STOCK-QUARTERS")
print("-" * 70)

print(
    f"Expected rows: "
    f"{len(expected):,}"
)

print(
    f"Actual rows: "
    f"{len(actual_keys):,}"
)

print(
    f"Missing stock-quarters: "
    f"{len(missing_periods):,}"
)

print(
    f"Missing before listing: "
    f"{missing_periods['before_listing'].sum():,}"
)

print(
    f"Missing after listing: "
    f"{(~missing_periods['before_listing']).sum():,}"
)


if not missing_periods.empty:

    print("\nMissing period details:")

    print(
        missing_periods[
            [
                "code",
                "name",
                "industry",
                "listing_date",
                "report_date",
                "before_listing",
            ]
        ]
        .to_string(
            index=False
        )
    )


# =========================================================
# 5. 字段 missing rate
# =========================================================
key_fields = [
    "roe_avg",
    "net_margin",
    "gross_margin",
    "net_profit",
    "revenue",
    "net_profit_yoy",
    "debt_asset_ratio",
    "cfo_net_profit_ratio",
]


print("\n3. FIELD COMPLETENESS")
print("-" * 70)


missing_summary = []

for col in key_fields:

    if col not in fund.columns:
        continue

    missing_count = (
        fund[col]
        .isna()
        .sum()
    )

    missing_rate = (
        fund[col]
        .isna()
        .mean()
    )

    missing_summary.append(
        {
            "field": col,
            "missing_count":
                missing_count,
            "missing_rate":
                missing_rate,
        }
    )


missing_summary = pd.DataFrame(
    missing_summary
)


print(
    missing_summary
    .to_string(
        index=False,
        formatters={
            "missing_rate":
                lambda x: f"{x:.2%}"
        }
    )
)


# =========================================================
# 6. Revenue缺失：按季度
# =========================================================
if "revenue" in fund.columns:

    revenue_by_period = (
        fund
        .assign(
            revenue_missing=
            fund["revenue"].isna()
        )
        .groupby(
            "report_date"
        )
        .agg(
            rows=(
                "code",
                "size"
            ),
            missing=(
                "revenue_missing",
                "sum"
            ),
            missing_rate=(
                "revenue_missing",
                "mean"
            ),
        )
        .reset_index()
    )


    print("\n4. REVENUE MISSING BY REPORT PERIOD")
    print("-" * 70)

    print(
        revenue_by_period
        .to_string(
            index=False,
            formatters={
                "missing_rate":
                    lambda x: f"{x:.2%}"
            }
        )
    )


# =========================================================
# 7. Revenue缺失：按行业
# =========================================================
fund_with_industry = (
    fund
    .merge(
        profiles[
            [
                "code",
                "industry",
            ]
        ],
        on="code",
        how="left"
    )
)


if "revenue" in fund_with_industry.columns:

    revenue_by_industry = (
        fund_with_industry
        .assign(
            revenue_missing=
            fund_with_industry[
                "revenue"
            ].isna()
        )
        .groupby(
            "industry",
            dropna=False
        )
        .agg(
            rows=(
                "code",
                "size"
            ),
            missing=(
                "revenue_missing",
                "sum"
            ),
            missing_rate=(
                "revenue_missing",
                "mean"
            ),
        )
        .sort_values(
            [
                "missing_rate",
                "rows",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .reset_index()
    )


    print("\n5. REVENUE MISSING BY INDUSTRY")
    print("-" * 70)

    print(
        revenue_by_industry
        .head(30)
        .to_string(
            index=False,
            formatters={
                "missing_rate":
                    lambda x: f"{x:.2%}"
            }
        )
    )


# =========================================================
# 8. Gross Margin缺失按行业
# =========================================================
if "gross_margin" in fund_with_industry.columns:

    gm_by_industry = (
        fund_with_industry
        .assign(
            gross_margin_missing=
            fund_with_industry[
                "gross_margin"
            ].isna()
        )
        .groupby(
            "industry",
            dropna=False
        )
        .agg(
            rows=(
                "code",
                "size"
            ),
            missing=(
                "gross_margin_missing",
                "sum"
            ),
            missing_rate=(
                "gross_margin_missing",
                "mean"
            ),
        )
        .query(
            "missing > 0"
        )
        .sort_values(
            [
                "missing_rate",
                "missing",
            ],
            ascending=[
                False,
                False,
            ]
        )
        .reset_index()
    )


    print("\n6. GROSS MARGIN MISSING BY INDUSTRY")
    print("-" * 70)

    print(
        gm_by_industry
        .head(30)
        .to_string(
            index=False,
            formatters={
                "missing_rate":
                    lambda x: f"{x:.2%}"
            }
        )
    )


# =========================================================
# 9. 描述统计
# =========================================================
audit_fields = [
    "roe_avg",
    "net_margin",
    "gross_margin",
    "net_profit_yoy",
    "debt_asset_ratio",
    "cfo_net_profit_ratio",
]


existing_audit_fields = [
    x
    for x in audit_fields
    if x in fund.columns
]


describe = (
    fund[
        existing_audit_fields
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


print("\n7. DISTRIBUTION SUMMARY")
print("-" * 70)

print(
    describe
    .to_string()
)


# =========================================================
# 10. Flag极端值
#
# 这里只flag，不删除！
# =========================================================
flags = fund[
    [
        "code",
        "report_date",
    ]
].copy()


if "roe_avg" in fund.columns:

    flags[
        "flag_roe"
    ] = (
        fund["roe_avg"]
        .abs()
        > 1
    )


if "gross_margin" in fund.columns:

    flags[
        "flag_gross_margin"
    ] = (
        (
            fund["gross_margin"]
            < -1
        )
        |
        (
            fund["gross_margin"]
            > 1
        )
    )


if "debt_asset_ratio" in fund.columns:

    flags[
        "flag_debt_asset"
    ] = (
        (
            fund[
                "debt_asset_ratio"
            ]
            < 0
        )
        |
        (
            fund[
                "debt_asset_ratio"
            ]
            > 1.5
        )
    )


if "cfo_net_profit_ratio" in fund.columns:

    flags[
        "flag_cfo_np"
    ] = (
        fund[
            "cfo_net_profit_ratio"
        ]
        .abs()
        > 10
    )


flag_cols = [
    c
    for c in flags.columns
    if c.startswith(
        "flag_"
    )
]


flags[
    "any_flag"
] = (
    flags[
        flag_cols
    ]
    .any(
        axis=1
    )
)


extreme_rows = (
    fund
    .loc[
        flags[
            "any_flag"
        ]
    ]
    .merge(
        profiles[
            [
                "code",
                "name",
                "industry",
            ]
        ],
        on="code",
        how="left"
    )
)


print("\n8. EXTREME VALUE FLAGS")
print("-" * 70)

print(
    f"Flagged rows: "
    f"{len(extreme_rows):,}"
)


if not extreme_rows.empty:

    show_cols = [
        "code",
        "name",
        "industry",
        "report_date",
        "roe_avg",
        "gross_margin",
        "debt_asset_ratio",
        "cfo_net_profit_ratio",
    ]

    show_cols = [
        c
        for c in show_cols
        if c in extreme_rows.columns
    ]

    print(
        extreme_rows[
            show_cols
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


# =========================================================
# 11. 保存Audit输出
# =========================================================
missing_summary.to_csv(
    OUTPUT_DIR
    / "fundamental_missing_summary.csv",
    index=False
)


missing_periods.to_csv(
    OUTPUT_DIR
    / "fundamental_missing_periods.csv",
    index=False
)


if "revenue" in fund.columns:

    revenue_by_period.to_csv(
        OUTPUT_DIR
        / "revenue_missing_by_period.csv",
        index=False
    )

    revenue_by_industry.to_csv(
        OUTPUT_DIR
        / "revenue_missing_by_industry.csv",
        index=False
    )


if "gross_margin" in fund.columns:

    gm_by_industry.to_csv(
        OUTPUT_DIR
        / "gross_margin_missing_by_industry.csv",
        index=False
    )


describe.to_csv(
    OUTPUT_DIR
    / "fundamental_distribution_summary.csv"
)


extreme_rows.to_csv(
    OUTPUT_DIR
    / "fundamental_extreme_rows.csv",
    index=False
)


print("\n")
print("=" * 70)

print(
    "AUDIT COMPLETE"
)

print("=" * 70)

print(
    f"Outputs saved to: "
    f"{OUTPUT_DIR}"
)