from pathlib import Path
import sys
import time

import baostock as bs
import pandas as pd
from tqdm import tqdm


# =========================================================
# Config
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
# 1. Load universe
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
# 2. Date format
#
# config:
# 20220101
#
# BaoStock:
# 2022-01-01
# =========================================================
start_date = (
    pd.to_datetime(START_DATE)
    .strftime("%Y-%m-%d")
)

end_date = (
    pd.to_datetime(END_DATE)
    .strftime("%Y-%m-%d")
)


# =========================================================
# 3. Convert stock code
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
# 4. Login helper
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
# 5. Checkpoint
# =========================================================
checkpoint_file = (
    RAW_DIR /
    "valuation_daily_checkpoint.parquet"
)

completed_file = (
    RAW_DIR /
    "valuation_completed_codes.txt"
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
# 6. Fields
#
# Important:
# close here will be UNADJUSTED price.
# =========================================================
fields = (
    "date,"
    "code,"
    "close,"
    "volume,"
    "amount,"
    "turn,"
    "peTTM,"
    "pbMRQ,"
    "psTTM,"
    "pcfNcfTTM,"
    "tradestatus,"
    "isST"
)


# =========================================================
# 7. Fetch one stock
# =========================================================
def fetch_one_stock(
    code,
    max_attempts=3
):

    bs_code = to_bs_code(code)


    for attempt in range(
        max_attempts
    ):

        try:

            login()


            rs = bs.query_history_k_data_plus(
                bs_code,
                fields,

                start_date=start_date,
                end_date=end_date,

                frequency="d",

                # 3 = 不复权
                adjustflag="3",
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


            if rs.error_code != "0":

                raise RuntimeError(
                    rs.error_msg
                )


            if not rows:

                raise ValueError(
                    "Empty response"
                )


            df = pd.DataFrame(
                rows,
                columns=rs.fields
            )


            safe_logout()


            # ---------------------------------------------
            # Standardize code
            # ---------------------------------------------
            df["code"] = (
                df["code"]
                .str.split(".")
                .str[-1]
                .str.zfill(6)
            )


            df["date"] = pd.to_datetime(
                df["date"]
            )


            # ---------------------------------------------
            # Rename
            # ---------------------------------------------
            df = df.rename(
                columns={
                    "close":
                        "close_unadj",

                    "volume":
                        "volume_share",

                    "turn":
                        "turnover_pct",

                    "peTTM":
                        "pe_ttm",

                    "pbMRQ":
                        "pb_mrq",

                    "psTTM":
                        "ps_ttm",

                    "pcfNcfTTM":
                        "pcf_ncf_ttm",

                    "tradestatus":
                        "trade_status",

                    "isST":
                        "is_st",
                }
            )


            # ---------------------------------------------
            # Numeric conversion
            # ---------------------------------------------
            numeric_cols = [
                "close_unadj",
                "volume_share",
                "amount",
                "turnover_pct",
                "pe_ttm",
                "pb_mrq",
                "ps_ttm",
                "pcf_ncf_ttm",
                "trade_status",
                "is_st",
            ]


            for col in numeric_cols:

                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )


            return df


        except Exception as e:

            print(
                f"\n[WARN] {code} "
                f"attempt "
                f"{attempt + 1}/"
                f"{max_attempts} failed"
            )

            print(
                f"Error: {e}"
            )


            safe_logout()


            if (
                attempt
                < max_attempts - 1
            ):

                time.sleep(
                    3 * (
                        attempt + 1
                    )
                )


    return None


# =========================================================
# 8. Fetch all remaining stocks
# =========================================================
new_rows = []

failed = []


for code in tqdm(
    remaining_codes,
    desc="Fetching valuation data"
):

    df = fetch_one_stock(
        code
    )


    if df is None:

        failed.append(
            code
        )

        print(
            f"\n[FAILED] {code}"
        )

        continue


    new_rows.append(
        df
    )


    completed_codes.add(
        code
    )


    print(
        f"\n[OK] {code}: "
        f"{len(df):,} rows"
    )


    # =====================================================
    # Save checkpoint after every successful stock
    # =====================================================
    frames = []


    if not existing.empty:

        frames.append(
            existing
        )


    if new_rows:

        frames.append(
            pd.concat(
                new_rows,
                ignore_index=True
            )
        )


    checkpoint = pd.concat(
        frames,
        ignore_index=True
    )


    checkpoint = (
        checkpoint
        .sort_values(
            [
                "code",
                "date",
            ]
        )
        .drop_duplicates(
            [
                "code",
                "date",
            ],
            keep="last"
        )
        .reset_index(
            drop=True
        )
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


    time.sleep(0.15)


# =========================================================
# 9. Load completed checkpoint
# =========================================================
if not checkpoint_file.exists():

    raise RuntimeError(
        "No valuation data fetched."
    )


valuation = pd.read_parquet(
    checkpoint_file
)


valuation["code"] = (
    valuation["code"]
    .astype(str)
    .str.zfill(6)
)


valuation["date"] = pd.to_datetime(
    valuation["date"]
)


valuation = (
    valuation
    .sort_values(
        [
            "code",
            "date",
        ]
    )
    .drop_duplicates(
        [
            "code",
            "date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =========================================================
# 10. Basic quality checks
# =========================================================
duplicates = (
    valuation
    .duplicated(
        [
            "code",
            "date",
        ]
    )
    .sum()
)


pb_positive_rate = (
    (
        valuation[
            "pb_mrq"
        ]
        > 0
    )
    .mean()
)


pe_positive_rate = (
    (
        valuation[
            "pe_ttm"
        ]
        > 0
    )
    .mean()
)


print("\n")
print("=" * 65)

print(
    "VALUATION DATA SUMMARY"
)

print("=" * 65)


print(
    f"Rows: "
    f"{len(valuation):,}"
)

print(
    f"Stocks: "
    f"{valuation['code'].nunique()}"
)

print(
    f"Date range: "
    f"{valuation['date'].min()} "
    f"-> "
    f"{valuation['date'].max()}"
)

print(
    f"Duplicate code-date: "
    f"{duplicates}"
)

print(
    f"Positive PB coverage: "
    f"{pb_positive_rate:.2%}"
)

print(
    f"Positive PE coverage: "
    f"{pe_positive_rate:.2%}"
)


print(
    "\nValuation missing rates:"
)


for col in [
    "close_unadj",
    "turnover_pct",
    "pb_mrq",
    "pe_ttm",
    "ps_ttm",
    "pcf_ncf_ttm",
]:

    print(
        f"{col:20s} "
        f"{valuation[col].isna().mean():.2%}"
    )


print(
    "\nPB distribution:"
)

print(
    valuation[
        "pb_mrq"
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


# =========================================================
# 11. Save formal output
# =========================================================
output_file = (
    RAW_DIR /
    "valuation_daily.parquet"
)


valuation.to_parquet(
    output_file,
    index=False
)


print(
    f"\nSaved to: "
    f"{output_file}"
)


if failed:

    print(
        "\nStocks requiring retry:"
    )

    print(
        failed
    )

else:

    print(
        "\nAll stocks completed."
    )