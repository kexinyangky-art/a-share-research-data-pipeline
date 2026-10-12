from pathlib import Path
import sys
import time

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
# 1. 读取现有 profiles
# =========================================================
profile_file = (
    RAW_DIR / "profiles.parquet"
)

profiles = pd.read_parquet(
    profile_file
)

profiles["code"] = (
    profiles["code"]
    .astype(str)
    .str.zfill(6)
)


# =========================================================
# 2. 找出 basic info 缺失的股票
# =========================================================
missing_codes = (
    profiles.loc[
        profiles["listing_date"].isna(),
        "code"
    ]
    .tolist()
)

print(
    f"Missing basic info stocks: "
    f"{len(missing_codes)}"
)

print(missing_codes)


# =========================================================
# 3. 转换 BaoStock 股票代码
# =========================================================
def to_bs_code(code):

    if code.startswith(("0", "3")):
        return f"sz.{code}"

    elif code.startswith(("6", "9")):
        return f"sh.{code}"

    elif code.startswith(("4", "8")):
        return f"bj.{code}"

    else:
        raise ValueError(
            f"Unknown stock code: {code}"
        )


# =========================================================
# 4. 逐只重新获取
#
# 因为只有3只，我们每只都重新登录，
# 避免之前 broken pipe 的连接继续影响后面的请求。
# =========================================================
recovered = []

failed = []


for code in missing_codes:

    print(
        f"\nFetching {code}..."
    )

    success = False

    for attempt in range(3):

        try:

            # ---------------------------------------------
            # 每次重新建立连接
            # ---------------------------------------------
            login_result = bs.login()

            if login_result.error_code != "0":

                raise RuntimeError(
                    login_result.error_msg
                )


            bs_code = to_bs_code(code)

            rs = bs.query_stock_basic(
                code=bs_code
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


            # query_stock_basic理论上这里只返回一行
            row = dict(
                zip(
                    rs.fields,
                    rows[0]
                )
            )


            recovered.append(
                {
                    "code": code,

                    "listing_date":
                        row.get(
                            "ipoDate"
                        ),

                    "delisting_date":
                        row.get(
                            "outDate"
                        ),

                    "security_type":
                        row.get(
                            "type"
                        ),

                    "listing_status":
                        row.get(
                            "status"
                        ),
                }
            )


            print(
                f"[OK] {code}"
            )

            print(
                recovered[-1]
            )


            success = True

            bs.logout()

            break


        except Exception as e:

            print(
                f"[WARN] {code} "
                f"attempt {attempt + 1}/3 failed"
            )

            print(
                f"Error: {e}"
            )

            try:
                bs.logout()
            except Exception:
                pass

            time.sleep(
                3 * (attempt + 1)
            )


    if not success:

        failed.append(code)

        print(
            f"[FAILED] {code}"
        )


# =========================================================
# 5. 把补回来的信息更新到原 profiles
# =========================================================
recovered_df = pd.DataFrame(
    recovered
)


if not recovered_df.empty:

    recovered_df = (
        recovered_df
        .set_index("code")
    )

    profiles = (
        profiles
        .set_index("code")
    )


    for col in [
        "listing_date",
        "delisting_date",
        "security_type",
        "listing_status",
    ]:

        profiles.loc[
            recovered_df.index,
            col
        ] = recovered_df[col]


    profiles = (
        profiles
        .reset_index()
    )


# =========================================================
# 6. 保存回原文件
# =========================================================
profiles.to_parquet(
    profile_file,
    index=False
)


# =========================================================
# 7. 检查结果
# =========================================================
print("\n")
print("=" * 50)

print("REPAIR SUMMARY")

print("=" * 50)


print(
    profiles[
        profiles["code"].isin(
            missing_codes
        )
    ][
        [
            "code",
            "name",
            "listing_date",
            "delisting_date",
            "security_type",
            "listing_status",
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nNull counts:"
)

print(
    profiles[
        [
            "listing_date",
            "delisting_date",
            "security_type",
            "listing_status",
        ]
    ]
    .isna()
    .sum()
)


if failed:

    print(
        f"\nStill failed: "
        f"{failed}"
    )

else:

    print(
        "\nAll missing basic info repaired."
    )