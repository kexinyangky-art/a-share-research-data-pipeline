from pathlib import Path
import sys
import time

import akshare as ak
import pandas as pd


# =========================================================
# Config
# =========================================================
sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from config import RAW_DIR


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

target_codes = set(
    universe["code"]
)


print(
    f"Universe stocks: "
    f"{len(target_codes)}"
)


# =========================================================
# 2. Fetch institutional research data
#
# AKShare / Eastmoney:
# one request returns historical records
# from the requested starting date
# =========================================================
max_attempts = 5

raw = None


for attempt in range(max_attempts):

    try:

        print(
            f"Fetching institutional research "
            f"data... attempt "
            f"{attempt + 1}/{max_attempts}"
        )

        raw = ak.stock_jgdy_tj_em(
            date="20220101"
        )

        if (
            raw is None
            or raw.empty
        ):

            raise ValueError(
                "Empty response"
            )

        break


    except Exception as e:

        print(
            f"[WARN] Fetch failed: {e}"
        )

        if (
            attempt
            < max_attempts - 1
        ):

            wait = (
                10
                * (
                    attempt + 1
                )
            )

            print(
                f"Waiting {wait}s..."
            )

            time.sleep(wait)


if (
    raw is None
    or raw.empty
):

    raise RuntimeError(
        "Institutional research "
        "data fetch failed."
    )


print(
    f"Raw rows fetched: "
    f"{len(raw):,}"
)


# =========================================================
# 3. Rename
# =========================================================
rename_map = {
    "代码":
        "code",

    "名称":
        "name",

    "接待机构数量":
        "institution_count",

    "接待方式":
        "reception_method",

    "接待人员":
        "reception_staff",

    "接待地点":
        "reception_location",

    "接待日期":
        "research_date",

    "公告日期":
        "announcement_date",
}


raw = raw.rename(
    columns=rename_map
)


# =========================================================
# 4. Standardize code
# =========================================================
raw["code"] = (
    raw["code"]
    .astype(str)
    .str.zfill(6)
)


# =========================================================
# 5. Filter to our universe
# =========================================================
research = (
    raw[
        raw["code"]
        .isin(
            target_codes
        )
    ]
    .copy()
)


# =========================================================
# 6. Dates
# =========================================================
research[
    "research_date"
] = pd.to_datetime(
    research[
        "research_date"
    ],
    errors="coerce",
)


research[
    "announcement_date"
] = pd.to_datetime(
    research[
        "announcement_date"
    ],
    errors="coerce",
)


# =========================================================
# 7. Institution count numeric
# =========================================================
research[
    "institution_count"
] = pd.to_numeric(
    research[
        "institution_count"
    ],
    errors="coerce",
)


# =========================================================
# 8. PIT logic
#
# Research event itself may happen before
# public disclosure.
#
# For research use, information becomes
# available on announcement_date.
# =========================================================
research[
    "available_date"
] = research[
    "announcement_date"
]


# =========================================================
# 9. Basic quality checks
# =========================================================
research = (
    research
    .dropna(
        subset=[
            "code",
            "available_date",
        ]
    )
    .sort_values(
        [
            "available_date",
            "code",
        ]
    )
    .reset_index(
        drop=True
    )
)


duplicate_events = (
    research
    .duplicated(
        [
            "code",
            "research_date",
            "announcement_date",
        ]
    )
    .sum()
)


invalid_dates = (
    research[
        "available_date"
    ]
    <
    research[
        "research_date"
    ]
).sum()


# =========================================================
# 10. Summary
# =========================================================
print("\n")
print("=" * 65)

print(
    "INSTITUTIONAL RESEARCH DATA SUMMARY"
)

print("=" * 65)


print(
    f"Universe-matched rows: "
    f"{len(research):,}"
)


print(
    f"Stocks with research events: "
    f"{research['code'].nunique()}"
)


print(
    f"Research date range: "
    f"{research['research_date'].min()} "
    f"-> "
    f"{research['research_date'].max()}"
)


print(
    f"Announcement date range: "
    f"{research['available_date'].min()} "
    f"-> "
    f"{research['available_date'].max()}"
)


print(
    f"Duplicate event keys: "
    f"{duplicate_events}"
)


print(
    f"Announcement before research date: "
    f"{invalid_dates}"
)


print(
    "\nInstitution count distribution:"
)


print(
    research[
        "institution_count"
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
    "\nMost researched stocks:"
)


top = (
    research
    .groupby(
        [
            "code",
            "name",
        ]
    )
    .agg(
        events=(
            "available_date",
            "size",
        ),

        institutions=(
            "institution_count",
            "sum",
        ),
    )
    .sort_values(
        "institutions",
        ascending=False,
    )
    .head(15)
)


print(
    top.to_string()
)


# =========================================================
# 11. Save
# =========================================================
output_file = (
    RAW_DIR
    / "institutional_research_raw.parquet"
)


research.to_parquet(
    output_file,
    index=False,
)


print(
    f"\nSaved to: "
    f"{output_file}"
)

print(
    "\nPASS: "
    "Institutional research "
    "raw data created."
)