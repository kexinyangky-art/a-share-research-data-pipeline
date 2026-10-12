from pathlib import Path
import sys
import time

import baostock as bs
import pandas as pd
from tqdm import tqdm


# =========================================================
# 项目配置
# =========================================================
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from config import RAW_DIR


# =========================================================
# 1. 读取研究股票池
# =========================================================
universe = pd.read_parquet(
    RAW_DIR / "universe.parquet"
)

universe["code"] = (
    universe["code"]
    .astype(str)
    .str.zfill(6)
)

all_codes = universe["code"].tolist()

print(
    f"Universe stocks: {len(all_codes)}"
)


# =========================================================
# 2. 财务数据时间范围
#
# 行情从2022开始，所以财务也从2022Q1开始
# 当前做到2026Q2
# =========================================================
periods = []

for year in range(2022, 2026):

    for quarter in [1, 2, 3, 4]:

        periods.append(
            (year, quarter)
        )


# 2026年目前使用Q1、Q2
periods.extend(
    [
        (2026, 1),
        (2026, 2),
    ]
)

print(
    f"Financial periods: {len(periods)}"
)


# =========================================================
# 3. 股票代码转换
#
# 000001 -> sz.000001
# 600000 -> sh.600000
# =========================================================
def to_bs_code(code):

    if code.startswith(
        ("0", "3")
    ):

        return f"sz.{code}"

    elif code.startswith(
        ("6", "9")
    ):

        return f"sh.{code}"

    elif code.startswith(
        ("4", "8")
    ):

        return f"bj.{code}"

    else:

        raise ValueError(
            f"Unknown stock code: {code}"
        )


# =========================================================
# 4. 登录 / 登出辅助函数
# =========================================================
def login():

    result = bs.login()

    if result.error_code != "0":

        raise RuntimeError(
            "BaoStock login failed: "
            + result.error_msg
        )


def safe_logout():

    try:

        bs.logout()

    except Exception:

        pass


# =========================================================
# 5. BaoStock ResultSet -> DataFrame
# =========================================================
def result_to_df(rs):

    rows = []

    while (
        rs.error_code == "0"
        and rs.next()
    ):

        rows.append(
            rs.get_row_data()
        )


    # 如果next过程中连接出错
    if rs.error_code != "0":

        raise RuntimeError(
            rs.error_msg
        )


    return pd.DataFrame(
        rows,
        columns=rs.fields
    )


# =========================================================
# 6. 带retry的查询函数
# =========================================================
def query_with_retry(
    query_func,
    bs_code,
    year,
    quarter,
    max_attempts=3
):

    for attempt in range(
        max_attempts
    ):

        try:

            rs = query_func(
                code=bs_code,
                year=year,
                quarter=quarter
            )


            if rs.error_code != "0":

                raise RuntimeError(
                    rs.error_msg
                )


            return result_to_df(
                rs
            )


        except Exception as e:

            print(
                f"\n[WARN] "
                f"{bs_code} "
                f"{year}Q{quarter} "
                f"attempt "
                f"{attempt + 1}/"
                f"{max_attempts} failed"
            )

            print(
                f"Error: {e}"
            )


            if (
                attempt
                < max_attempts - 1
            ):

                # 连接出问题以后重新登录
                safe_logout()

                time.sleep(
                    2 * (
                        attempt + 1
                    )
                )

                login()


    raise RuntimeError(
        f"{bs_code} "
        f"{year}Q{quarter} "
        f"failed after "
        f"{max_attempts} attempts"
    )


# =========================================================
# 7. 盈利能力数据清洗
# =========================================================
def clean_profit(df):

    if df.empty:
        return df


    df = df.rename(
        columns={
            "pubDate":
                "profit_pub_date",

            "statDate":
                "report_date",

            "roeAvg":
                "roe_avg",

            "npMargin":
                "net_margin",

            "gpMargin":
                "gross_margin",

            "netProfit":
                "net_profit",

            "epsTTM":
                "eps_ttm",

            "MBRevenue":
                "revenue",

            "totalShare":
                "total_shares",

            "liqaShare":
                "float_shares",
        }
    )


    return df


# =========================================================
# 8. 成长能力数据清洗
# =========================================================
def clean_growth(df):

    if df.empty:
        return df


    df = df.rename(
        columns={
            "pubDate":
                "growth_pub_date",

            "statDate":
                "report_date",

            "YOYEquity":
                "equity_yoy",

            "YOYAsset":
                "asset_yoy",

            "YOYNI":
                "net_profit_yoy",

            "YOYEPSBasic":
                "eps_basic_yoy",

            "YOYEPSDiluted":
                "eps_diluted_yoy",
        }
    )


    return df


# =========================================================
# 9. 偿债 / 资本结构数据清洗
# =========================================================
def clean_balance(df):

    if df.empty:
        return df


    df = df.rename(
        columns={
            "pubDate":
                "balance_pub_date",

            "statDate":
                "report_date",

            "currentRatio":
                "current_ratio",

            "quickRatio":
                "quick_ratio",

            "cashRatio":
                "cash_ratio",

            "YOYLiability":
                "liability_yoy",

            "liabilityToAsset":
                "debt_asset_ratio",

            "assetToEquity":
                "asset_equity_multiplier",
        }
    )


    return df


# =========================================================
# 10. 现金流指标清洗
# =========================================================
def clean_cashflow(df):

    if df.empty:
        return df


    df = df.rename(
        columns={
            "pubDate":
                "cashflow_pub_date",

            "statDate":
                "report_date",

            "CAToAsset":
                "current_asset_ratio",

            "NCAToAsset":
                "noncurrent_asset_ratio",

            "tangibleAssetToAsset":
                "tangible_asset_ratio",

            "ebitToInterest":
                "ebit_interest_coverage",

            "CFOToOR":
                "cfo_revenue_ratio",

            "CFOToNP":
                "cfo_net_profit_ratio",

            "CFOToGr":
                "cfo_gross_revenue_ratio",
        }
    )


    return df


# =========================================================
# 11. 合并同一季度四组指标
# =========================================================
def merge_period_frames(
    frames
):

    usable = [
        df
        for df in frames
        if not df.empty
    ]


    if not usable:

        return pd.DataFrame()


    result = usable[0]


    for df in usable[1:]:

        result = result.merge(
            df,
            on=[
                "code",
                "report_date",
            ],
            how="outer",
        )


    return result


# =========================================================
# 12. Checkpoint
#
# 中途断网时不用从头重新跑
# =========================================================
checkpoint_file = (
    RAW_DIR
    / "fundamentals_checkpoint.parquet"
)


completed_file = (
    RAW_DIR
    / "fundamentals_completed_codes.txt"
)


if checkpoint_file.exists():

    existing = pd.read_parquet(
        checkpoint_file
    )

else:

    existing = pd.DataFrame()


if completed_file.exists():

    completed_codes = set(
        completed_file
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

else:

    completed_codes = set()


remaining_codes = [
    code
    for code in all_codes
    if code not in completed_codes
]


print(
    f"Already completed: "
    f"{len(completed_codes)}"
)

print(
    f"Remaining stocks: "
    f"{len(remaining_codes)}"
)


# =========================================================
# 13. 开始抓财务数据
# =========================================================
all_new_rows = []

failed_stocks = []


for code in tqdm(
    remaining_codes,
    desc="Fetching fundamentals"
):

    bs_code = to_bs_code(
        code
    )

    stock_rows = []

    stock_success = True


    # -----------------------------------------------------
    # 每只股票重新建立一次BaoStock连接
    # -----------------------------------------------------
    try:

        login()

    except Exception as e:

        print(
            f"\n[FAILED LOGIN] "
            f"{code}: {e}"
        )

        failed_stocks.append(
            code
        )

        continue


    # -----------------------------------------------------
    # 一只股票的18个季度
    # -----------------------------------------------------
    for year, quarter in periods:

        try:

            # =============================================
            # A. 盈利能力
            # =============================================
            profit = (
                query_with_retry(
                    bs.query_profit_data,
                    bs_code,
                    year,
                    quarter
                )
            )

            profit = clean_profit(
                profit
            )


            # =============================================
            # B. 成长能力
            # =============================================
            growth = (
                query_with_retry(
                    bs.query_growth_data,
                    bs_code,
                    year,
                    quarter
                )
            )

            growth = clean_growth(
                growth
            )


            # =============================================
            # C. 偿债能力
            # =============================================
            balance = (
                query_with_retry(
                    bs.query_balance_data,
                    bs_code,
                    year,
                    quarter
                )
            )

            balance = clean_balance(
                balance
            )


            # =============================================
            # D. 现金流质量
            # =============================================
            cashflow = (
                query_with_retry(
                    bs.query_cash_flow_data,
                    bs_code,
                    year,
                    quarter
                )
            )

            cashflow = clean_cashflow(
                cashflow
            )


            # =============================================
            # E. 合并当前季度
            # =============================================
            period_df = (
                merge_period_frames(
                    [
                        profit,
                        growth,
                        balance,
                        cashflow,
                    ]
                )
            )


            if not period_df.empty:

                period_df[
                    "requested_year"
                ] = year

                period_df[
                    "requested_quarter"
                ] = quarter

                stock_rows.append(
                    period_df
                )


            # 稍微降低请求速度
            time.sleep(0.03)


        except Exception as e:

            print(
                f"\n[FAILED] "
                f"{code} "
                f"{year}Q{quarter}"
            )

            print(
                f"Error: {e}"
            )

            stock_success = False

            break


    safe_logout()


    # =====================================================
    # 14. 只有整只股票全部季度成功
    #     才写入completed
    # =====================================================
    if stock_success:

        if stock_rows:

            stock_df = pd.concat(
                stock_rows,
                ignore_index=True
            )

            all_new_rows.append(
                stock_df
            )


        completed_codes.add(
            code
        )


        # =================================================
        # 每完成一只股票就保存checkpoint
        # =================================================
        frames = []


        if not existing.empty:

            frames.append(
                existing
            )


        if all_new_rows:

            frames.append(
                pd.concat(
                    all_new_rows,
                    ignore_index=True
                )
            )


        if frames:

            checkpoint = pd.concat(
                frames,
                ignore_index=True
            )


            checkpoint.to_parquet(
                checkpoint_file,
                index=False
            )


        completed_file.write_text(
            "\n".join(
                sorted(
                    completed_codes
                )
            ),
            encoding="utf-8"
        )


    else:

        failed_stocks.append(
            code
        )


# =========================================================
# 15. 读取最终checkpoint
# =========================================================
if checkpoint_file.exists():

    fundamentals = (
        pd.read_parquet(
            checkpoint_file
        )
    )

else:

    raise RuntimeError(
        "No fundamental data fetched."
    )


# =========================================================
# 16. 股票代码格式统一
#
# sz.000001 -> 000001
# =========================================================
fundamentals["code"] = (
    fundamentals["code"]
    .str.split(".")
    .str[-1]
    .str.zfill(6)
)


# =========================================================
# 17. 日期格式
# =========================================================
date_cols = [
    "report_date",
    "profit_pub_date",
    "growth_pub_date",
    "balance_pub_date",
    "cashflow_pub_date",
]


for col in date_cols:

    if col in fundamentals.columns:

        fundamentals[col] = (
            pd.to_datetime(
                fundamentals[col],
                errors="coerce"
            )
        )


# =========================================================
# 18. available_date
#
# Conservative Point-in-Time rule:
#
# 四组财务信息都已经公开以后，
# 这一整行数据才允许进入research layer。
# =========================================================
pub_cols = [
    c
    for c in [
        "profit_pub_date",
        "growth_pub_date",
        "balance_pub_date",
        "cashflow_pub_date",
    ]
    if c in fundamentals.columns
]


fundamentals[
    "available_date"
] = fundamentals[
    pub_cols
].max(
    axis=1
)


# =========================================================
# 19. 数值字段转 numeric
# =========================================================
non_numeric_cols = {
    "code",
    "report_date",
    "profit_pub_date",
    "growth_pub_date",
    "balance_pub_date",
    "cashflow_pub_date",
    "available_date",
}


for col in fundamentals.columns:

    if (
        col
        not in non_numeric_cols
    ):

        fundamentals[col] = (
            pd.to_numeric(
                fundamentals[col],
                errors="coerce"
            )
        )


# =========================================================
# 20. 去重
# =========================================================
fundamentals = (
    fundamentals
    .sort_values(
        [
            "code",
            "report_date",
        ]
    )
    .drop_duplicates(
        [
            "code",
            "report_date",
        ],
        keep="last"
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 21. 保存正式数据
# =========================================================
output_file = (
    RAW_DIR
    / "fundamentals_quarterly.parquet"
)


fundamentals.to_parquet(
    output_file,
    index=False
)


# =========================================================
# 22. Quality Summary
# =========================================================
print("\n")

print("=" * 60)

print(
    "FUNDAMENTAL DATA SUMMARY"
)

print("=" * 60)


print(
    f"Rows: "
    f"{len(fundamentals):,}"
)


print(
    f"Stocks: "
    f"{fundamentals['code'].nunique()}"
)


print(
    f"Report date range: "
    f"{fundamentals['report_date'].min()} "
    f"-> "
    f"{fundamentals['report_date'].max()}"
)


print(
    f"Available date range: "
    f"{fundamentals['available_date'].min()} "
    f"-> "
    f"{fundamentals['available_date'].max()}"
)


# =========================================================
# 23. 核心字段完整度
# =========================================================
important_cols = [
    "roe_avg",
    "net_margin",
    "gross_margin",
    "net_profit",
    "revenue",
    "net_profit_yoy",
    "debt_asset_ratio",
    "cfo_net_profit_ratio",
]


print(
    "\nMissing rate:"
)


for col in important_cols:

    if col in fundamentals.columns:

        print(
            f"{col:25s}: "
            f"{fundamentals[col].isna().mean():.2%}"
        )


# =========================================================
# 24. publication date基础时间检查
# =========================================================
invalid_pub_dates = (
    fundamentals[
        "available_date"
    ]
    <
    fundamentals[
        "report_date"
    ]
).sum()


print(
    f"\nInvalid publication dates: "
    f"{invalid_pub_dates}"
)


# =========================================================
# 25. Sample
# =========================================================
print(
    "\nSample:"
)


sample_cols = [
    "code",
    "report_date",
    "available_date",
    "roe_avg",
    "net_profit_yoy",
    "debt_asset_ratio",
    "cfo_net_profit_ratio",
]


print(
    fundamentals[
        [
            c
            for c in sample_cols
            if c in fundamentals.columns
        ]
    ]
    .tail(15)
    .to_string(
        index=False
    )
)


print(
    f"\nSaved to: "
    f"{output_file}"
)


if failed_stocks:

    print(
        "\nStocks requiring retry:"
    )

    print(
        sorted(
            set(
                failed_stocks
            )
        )
    )

else:

    print(
        "\nAll stocks completed."
    )