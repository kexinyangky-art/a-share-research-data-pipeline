from pathlib import Path
import sys
import time

import akshare as ak
import pandas as pd

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from config import RAW_DIR


# =========================================================
# 1. Universe
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
    f"Universe stocks: {len(target_codes)}"
)


# =========================================================
# 2. Fetch detail data
# =========================================================
raw = None
max_attempts = 5


for attempt in range(max_attempts):

    try:

        print(
            f"Fetching institutional research detail "
            f"attempt {attempt + 1}/{max_attempts}..."
        )

        raw = ak.stock_jgdy_detail_em(
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

        if attempt < max_attempts - 1:

            wait = 15 * (
                attempt + 1
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
        "Institutional research detail fetch failed."
    )


print(
    f"Raw rows fetched: {len(raw):,}"
)


# =========================================================
# 3. Rename
# =========================================================
rename_map = {
    "代码":
        "code",

    "名称":
        "name",

    "调研机构":
        "institution",

    "机构类型":
        "institution_type",

    "调研人员":
        "researcher",

    "接待方式":
        "reception_method",

    "接待人员":
        "reception_staff",

    "接待地点":
        "reception_location",

    "调研日期":
        "research_date",

    "公告日期":
        "announcement_date",
}


raw = raw.rename(
    columns=rename_map
)


# =========================================================
# 4. Standardize
# =========================================================
raw["code"] = (
    raw["code"]
    .astype(str)
    .str.zfill(6)
)


detail = (
    raw[
        raw["code"]
        .isin(
            target_codes
        )
    ]
    .copy()
)


detail["research_date"] = pd.to_datetime(
    detail["research_date"],
    errors="coerce",
)


detail["announcement_date"] = pd.to_datetime(
    detail["announcement_date"],
    errors="coerce",
)


detail["available_date"] = (
    detail["announcement_date"]
)


# =========================================================
# 5. Clean
# =========================================================
detail = (
    detail
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
    .reset_index(drop=True)
)


# =========================================================
# 6. Summary
# =========================================================
print("\n")
print("=" * 70)

print(
    "INSTITUTIONAL RESEARCH DETAIL SUMMARY"
)

print("=" * 70)


print(
    f"Universe-matched rows: "
    f"{len(detail):,}"
)

print(
    f"Stocks covered: "
    f"{detail['code'].nunique()}"
)

print(
    f"Research date range: "
    f"{detail['research_date'].min()} "
    f"-> "
    f"{detail['research_date'].max()}"
)

print(
    f"Announcement date range: "
    f"{detail['available_date'].min()} "
    f"-> "
    f"{detail['available_date'].max()}"
)


if "institution" in detail.columns:

    print(
        f"Unique institutions: "
        f"{detail['institution'].nunique()}"
    )


if "institution_type" in detail.columns:

    print(
        "\nInstitution types:"
    )

    print(
        detail[
            "institution_type"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )


# Unique stock-event count
event_keys = (
    detail[
        [
            "code",
            "research_date",
            "announcement_date",
        ]
    ]
    .drop_duplicates()
)


print(
    f"\nUnique stock-research events: "
    f"{len(event_keys):,}"
)


print(
    "\nTop researched stocks:"
)


top = (
    detail
    .groupby(
        [
            "code",
            "name",
        ]
    )
    .agg(
        detail_rows=(
            "available_date",
            "size",
        ),

        unique_events=(
            "research_date",
            "nunique",
        ),

        unique_institutions=(
            "institution",
            "nunique",
        ),
    )
    .sort_values(
        "unique_institutions",
        ascending=False,
    )
    .head(15)
)


print(
    top.to_string()
)


# =========================================================
# 7. Save
# =========================================================
output_file = (
    RAW_DIR
    / "institutional_research_detail_raw.parquet"
)


detail.to_parquet(
    output_file,
    index=False,
)


print(
    f"\nSaved to: "
    f"{output_file}"
)