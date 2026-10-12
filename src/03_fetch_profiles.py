from pathlib import Path
import sys

import baostock as bs
import pandas as pd


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

target_codes = set(
    universe["code"]
)

print(
    f"Universe stocks: "
    f"{len(target_codes)}"
)


# =========================================================
# 2. 登录 BaoStock
# =========================================================
login_result = bs.login()

if login_result.error_code != "0":

    raise RuntimeError(
        "BaoStock login failed: "
        + login_result.error_msg
    )

print("BaoStock login success.")


# =========================================================
# 3. 获取行业分类
# =========================================================
industry_rs = (
    bs.query_stock_industry()
)

if industry_rs.error_code != "0":

    raise RuntimeError(
        "Industry query failed: "
        + industry_rs.error_msg
    )


industry_rows = []

while (
    industry_rs.error_code == "0"
    and industry_rs.next()
):

    industry_rows.append(
        industry_rs.get_row_data()
    )


industry = pd.DataFrame(
    industry_rows,
    columns=industry_rs.fields
)


print(
    f"Raw industry records: "
    f"{len(industry):,}"
)


# =========================================================
# 4. BaoStock代码：
#
# sh.600000
# sz.000001
#
# 转成：
#
# 600000
# 000001
# =========================================================
industry["code"] = (
    industry["code"]
    .str.split(".")
    .str[-1]
    .str.zfill(6)
)


# 只保留我们的研究股票池
industry = industry[
    industry["code"].isin(
        target_codes
    )
].copy()


# =========================================================
# 5. 字段统一
# =========================================================
industry = industry.rename(
    columns={
        "code_name": "name",
        "industryClassification":
            "industry_classification",
        "updateDate":
            "industry_update_date",
    }
)


# =========================================================
# 6. 获取股票基础信息
# =========================================================
basic_rows = []


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
        return None


for code in sorted(target_codes):

    bs_code = to_bs_code(code)

    if bs_code is None:
        continue

    rs = bs.query_stock_basic(
        code=bs_code
    )

    if rs.error_code != "0":
        continue

    while (
        rs.error_code == "0"
        and rs.next()
    ):

        row = dict(
            zip(
                rs.fields,
                rs.get_row_data()
            )
        )

        basic_rows.append(row)


basic = pd.DataFrame(
    basic_rows
)


if not basic.empty:

    basic["code"] = (
        basic["code"]
        .str.split(".")
        .str[-1]
        .str.zfill(6)
    )

    basic = basic.rename(
        columns={
            "code_name":
                "basic_name",

            "ipoDate":
                "listing_date",

            "outDate":
                "delisting_date",

            "type":
                "security_type",

            "status":
                "listing_status",
        }
    )


# =========================================================
# 7. 合并行业 + 股票基础信息
# =========================================================
profiles = industry.merge(
    basic,
    on="code",
    how="left"
)


# 优先使用行业表里的名字
if "basic_name" in profiles.columns:

    profiles["name"] = (
        profiles["name"]
        .fillna(
            profiles["basic_name"]
        )
    )


# =========================================================
# 8. 保留需要字段
# =========================================================
keep_cols = [
    "code",
    "name",
    "industry",
    "industry_classification",
    "industry_update_date",
    "listing_date",
    "delisting_date",
    "security_type",
    "listing_status",
]


keep_cols = [
    c
    for c in keep_cols
    if c in profiles.columns
]


profiles = profiles[
    keep_cols
].copy()


profiles = (
    profiles
    .drop_duplicates(
        "code",
        keep="last"
    )
    .sort_values(
        "code"
    )
    .reset_index(drop=True)
)


# =========================================================
# 9. 登出
# =========================================================
bs.logout()


# =========================================================
# 10. 数据质量检查
# =========================================================
industry_coverage = (
    profiles["industry"]
    .notna()
    .mean()
)

stock_coverage = (
    profiles["code"]
    .nunique()
    / len(target_codes)
)


print("\n")
print("=" * 50)

print("PROFILE SUMMARY")

print("=" * 50)

print(
    f"Profile stocks: "
    f"{profiles['code'].nunique()}"
)

print(
    f"Universe stocks: "
    f"{len(target_codes)}"
)

print(
    f"Stock coverage: "
    f"{stock_coverage:.2%}"
)

print(
    f"Industry coverage: "
    f"{industry_coverage:.2%}"
)


print("\nIndustry examples:")

print(
    profiles[
        [
            "code",
            "name",
            "industry",
        ]
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# =========================================================
# 11. 保存
# =========================================================
output_file = (
    RAW_DIR
    / "profiles.parquet"
)

profiles.to_parquet(
    output_file,
    index=False
)

print(
    f"\nSaved to: "
    f"{output_file}"
)