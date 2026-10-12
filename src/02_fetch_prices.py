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

from config import (
    RAW_DIR,
    START_DATE,
    END_DATE,
)


# =========================================================
# 1. 读取股票池
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
# 2. AKShare日期格式：
#       20220101
#
#    BaoStock需要：
#       2022-01-01
# =========================================================
start_date = pd.to_datetime(
    START_DATE
).strftime("%Y-%m-%d")

end_date = pd.to_datetime(
    END_DATE
).strftime("%Y-%m-%d")


# =========================================================
# 3. A股代码转换
#
# 深圳：
# 000001 -> sz.000001
#
# 上海：
# 600000 -> sh.600000
#
# 北京：
# 8xxxxx / 4xxxxx -> bj.xxxxxx
# =========================================================
def convert_code(code):

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
# 4. 登录 BaoStock
# =========================================================
login_result = bs.login()

if login_result.error_code != "0":

    raise RuntimeError(
        "BaoStock login failed: "
        + login_result.error_msg
    )


print("BaoStock login success.")


# =========================================================
# 5. 抓取行情
# =========================================================
all_rows = []

failed = []


fields = (
    "date,"
    "code,"
    "open,"
    "high,"
    "low,"
    "close,"
    "preclose,"
    "volume,"
    "amount,"
    "adjustflag,"
    "turn,"
    "tradestatus,"
    "pctChg,"
    "isST"
)


for code in tqdm(
    all_codes,
    desc="Fetching prices"
):

    bs_code = convert_code(code)

    try:

        rs = bs.query_history_k_data_plus(
            bs_code,
            fields,
            start_date=start_date,
            end_date=end_date,

            # d = daily
            frequency="d",

            # 1 = 后复权
            adjustflag="1",
        )


        if rs.error_code != "0":

            raise RuntimeError(
                rs.error_msg
            )


        rows = []

        while (
            rs.error_code == "0"
            and rs.next()
        ):

            rows.append(
                rs.get_row_data()
            )


        if not rows:

            raise ValueError(
                "Empty response"
            )


        df = pd.DataFrame(
            rows,
            columns=rs.fields
        )


        # =============================================
        # BaoStock格式：
        #
        # sz.000001
        #
        # 转成：
        #
        # 000001
        # =============================================
        df["code"] = (
            df["code"]
            .str.split(".")
            .str[-1]
        )


        df["date"] = pd.to_datetime(
            df["date"]
        )


        # =============================================
        # 数值字段转换
        # =============================================
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
        ]


        for col in numeric_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )


        # =============================================
        # 字段重命名
        # 与我们后续pipeline保持一致
        # =============================================
        df = df.rename(
            columns={
                "volume": "volume_share",

                "turn":
                    "turnover_pct",

                "pctChg":
                    "return_pct_vendor",

                "tradestatus":
                    "trade_status",

                "isST":
                    "is_st",
            }
        )


        # =============================================
        # 计算价格变化
        # =============================================
        df["price_change"] = (
            df["close"]
            - df["preclose"]
        )


        # =============================================
        # 振幅
        # =============================================
        df["amplitude_pct"] = (
            (
                df["high"]
                - df["low"]
            )
            / df["preclose"]
            * 100
        )


        all_rows.append(df)


        print(
            f"\n[OK] {code}: "
            f"{len(df):,} rows"
        )


        # 稍微放慢即可
        time.sleep(0.2)


    except Exception as e:

        print(
            f"\n[FAILED] "
            f"{code}: {e}"
        )

        failed.append(
            {
                "code": code,
                "error": str(e),
            }
        )


# =========================================================
# 6. 登出
# =========================================================
bs.logout()


# =========================================================
# 7. 检查
# =========================================================
if not all_rows:

    raise RuntimeError(
        "No price data fetched."
    )


# =========================================================
# 8. 合并
# =========================================================
prices = pd.concat(
    all_rows,
    ignore_index=True
)


prices = (
    prices
    .sort_values(
        ["code", "date"]
    )
    .drop_duplicates(
        ["code", "date"]
    )
    .reset_index(drop=True)
)


# =========================================================
# 9. 保存
# =========================================================
output_file = (
    RAW_DIR
    / "prices_hfq.parquet"
)

prices.to_parquet(
    output_file,
    index=False
)


# =========================================================
# 10. Failed股票
# =========================================================
failed_file = (
    RAW_DIR
    / "failed_prices.csv"
)


if failed:

    pd.DataFrame(
        failed
    ).to_csv(
        failed_file,
        index=False
    )

elif failed_file.exists():

    failed_file.unlink()


# =========================================================
# 11. Summary
# =========================================================
downloaded_stocks = (
    prices["code"]
    .nunique()
)

coverage = (
    downloaded_stocks
    / len(all_codes)
)


print("\n")
print("=" * 50)

print("DOWNLOAD SUMMARY")

print("=" * 50)


print(
    f"Rows: "
    f"{len(prices):,}"
)

print(
    f"Stocks downloaded: "
    f"{downloaded_stocks}"
)

print(
    f"Universe stocks: "
    f"{len(all_codes)}"
)

print(
    f"Coverage: "
    f"{coverage:.2%}"
)

print(
    f"Saved to: "
    f"{output_file}"
)


if failed:

    print(
        f"\nFailed stocks: "
        f"{len(failed)}"
    )

    print(
        pd.DataFrame(failed)
    )

else:

    print(
        "\nAll stocks successfully "
        "downloaded."
    )