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

from config import (
    RAW_DIR,
    PROCESSED_DIR,
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 1. Load research panel
# =========================================================
panel = pd.read_parquet(
    PROCESSED_DIR
    / "factor_features_extended.parquet"
)

panel["date"] = pd.to_datetime(
    panel["date"]
)

panel["code"] = (
    panel["code"]
    .astype(str)
    .str.zfill(6)
)

panel = (
    panel
    .sort_values(
        ["code", "date"]
    )
    .reset_index(drop=True)
)


# =========================================================
# 2. Load institutional research raw data
# =========================================================
research = pd.read_parquet(
    RAW_DIR
    / "institutional_research_raw.parquet"
)

research["code"] = (
    research["code"]
    .astype(str)
    .str.zfill(6)
)

research[
    "available_date"
] = pd.to_datetime(
    research["available_date"]
)

research[
    "research_date"
] = pd.to_datetime(
    research["research_date"]
)

research[
    "institution_count"
] = pd.to_numeric(
    research["institution_count"],
    errors="coerce",
)


# =========================================================
# 3. Source coverage
#
# IMPORTANT:
#
# Although we requested data from 2022,
# the vendor only returned data beginning
# around 2025-10.
#
# Therefore:
#
# dates BEFORE source_start are UNKNOWN,
# not zero-attention periods.
# =========================================================
source_start = (
    research[
        "available_date"
    ]
    .min()
)

source_end = (
    research[
        "available_date"
    ]
    .max()
)


print(
    f"Alternative-data source range: "
    f"{source_start.date()} "
    f"-> "
    f"{source_end.date()}"
)


# =========================================================
# 4. Map announcement date to next trading day
#
# Example:
#
# announcement = Saturday
#       ↓
# first usable date = Monday trading day
# =========================================================
trade_dates = np.array(
    sorted(
        panel["date"]
        .unique()
    ),
    dtype="datetime64[ns]",
)


announcement_values = (
    research[
        "available_date"
    ]
    .values
    .astype(
        "datetime64[ns]"
    )
)


positions = np.searchsorted(
    trade_dates,
    announcement_values,
    side="left",
)


valid = (
    positions
    < len(trade_dates)
)


research = (
    research.loc[
        valid
    ]
    .copy()
)


positions = positions[
    valid
]


research[
    "effective_trade_date"
] = pd.to_datetime(
    trade_dates[
        positions
    ]
)


# =========================================================
# 5. Aggregate multiple research events
#    occurring on the same usable trading date
# =========================================================
event_daily = (
    research
    .groupby(
        [
            "code",
            "effective_trade_date",
        ],
        as_index=False,
    )
    .agg(
        research_events_day=(
            "institution_count",
            "size",
        ),

        institution_count_day=(
            "institution_count",
            "sum",
        ),

        max_institutions_event_day=(
            "institution_count",
            "max",
        ),
    )
)


event_daily = (
    event_daily
    .rename(
        columns={
            "effective_trade_date":
                "date"
        }
    )
)


# =========================================================
# 6. Merge daily event data into market panel
# =========================================================
df = panel.merge(
    event_daily,
    on=[
        "code",
        "date",
    ],
    how="left",
    validate="one_to_one",
)


event_cols = [
    "research_events_day",
    "institution_count_day",
    "max_institutions_event_day",
]


for col in event_cols:

    df[col] = (
        df[col]
        .fillna(0)
    )


# =========================================================
# 7. Build rolling alternative-data features
#
# Use CALENDAR-day windows:
# 30D / 60D / 90D
# =========================================================
parts = []


for code, stock in df.groupby(
    "code",
    sort=False,
):

    stock = (
        stock
        .sort_values("date")
        .copy()
    )

    stock = stock.set_index(
        "date"
    )


    # -----------------------------------------------------
    # Event frequency
    # -----------------------------------------------------
    stock[
        "research_events_30d"
    ] = (
        stock[
            "research_events_day"
        ]
        .rolling("30D")
        .sum()
    )


    stock[
        "research_events_90d"
    ] = (
        stock[
            "research_events_day"
        ]
        .rolling("90D")
        .sum()
    )


    # -----------------------------------------------------
    # Institutional participation
    # -----------------------------------------------------
    stock[
        "institution_count_30d"
    ] = (
        stock[
            "institution_count_day"
        ]
        .rolling("30D")
        .sum()
    )


    stock[
        "institution_count_60d"
    ] = (
        stock[
            "institution_count_day"
        ]
        .rolling("60D")
        .sum()
    )


    stock[
        "institution_count_90d"
    ] = (
        stock[
            "institution_count_day"
        ]
        .rolling("90D")
        .sum()
    )


    # -----------------------------------------------------
    # Maximum event size
    # -----------------------------------------------------
    stock[
        "max_institutions_30d"
    ] = (
        stock[
            "max_institutions_event_day"
        ]
        .rolling("30D")
        .max()
    )


    # -----------------------------------------------------
    # Previous 30-day institutional attention
    #
    # roughly:
    #
    # previous 30D
    # = rolling 60D - rolling 30D
    # -----------------------------------------------------
    stock[
        "institution_count_prev30d"
    ] = (
        stock[
            "institution_count_60d"
        ]
        -
        stock[
            "institution_count_30d"
        ]
    )


    stock[
        "institution_count_prev30d"
    ] = (
        stock[
            "institution_count_prev30d"
        ]
        .clip(
            lower=0
        )
    )


    # -----------------------------------------------------
    # Attention acceleration
    #
    # Positive:
    # recent attention > previous attention
    # -----------------------------------------------------
    stock[
        "attention_acceleration"
    ] = (
        np.log1p(
            stock[
                "institution_count_30d"
            ]
        )
        -
        np.log1p(
            stock[
                "institution_count_prev30d"
            ]
        )
    )


    # -----------------------------------------------------
    # Log-scaled attention intensity
    #
    # Raw institution count is highly skewed:
    # median 4, max ~394.
    # -----------------------------------------------------
    stock[
        "attention_intensity_30d"
    ] = np.log1p(
        stock[
            "institution_count_30d"
        ]
    )


    # -----------------------------------------------------
    # Days since latest research event
    # -----------------------------------------------------
    current_dates = pd.Series(
        stock.index,
        index=stock.index,
    )


    event_dates = (
        current_dates
        .where(
            stock[
                "research_events_day"
            ]
            > 0
        )
    )


    last_event_date = (
        event_dates
        .ffill()
    )


    stock[
        "days_since_last_research"
    ] = (
        current_dates
        -
        last_event_date
    ).dt.days


    # -----------------------------------------------------
    # Recency score
    #
    # event today -> 1
    # 30 days old -> exp(-1) ≈ 0.37
    # older events decay toward zero
    # -----------------------------------------------------
    stock[
        "research_recency_score"
    ] = np.exp(
        -stock[
            "days_since_last_research"
        ]
        / 30
    )


    stock = stock.reset_index()

    parts.append(
        stock
    )


df = pd.concat(
    parts,
    ignore_index=True,
)


# =========================================================
# 8. Left-censoring protection
#
# We only have vendor history from source_start.
#
# A 90-day feature is not fully observable
# until 90 days later.
#
# Before then:
# feature = UNKNOWN, not zero.
# =========================================================
eligible_start = (
    source_start
    +
    pd.Timedelta(
        days=90
    )
)


df[
    "alt_data_eligible"
] = (
    df["date"]
    >= eligible_start
)


feature_cols = [
    "research_events_30d",
    "research_events_90d",
    "institution_count_30d",
    "institution_count_90d",
    "max_institutions_30d",
    "days_since_last_research",
    "attention_acceleration",
    "attention_intensity_30d",
    "research_recency_score",
]


df.loc[
    ~df[
        "alt_data_eligible"
    ],
    feature_cols,
] = np.nan


# =========================================================
# 9. Do NOT treat "no prior research event"
# as missing attention for the count features
#
# Once source history is mature,
# zero really means no event in the lookback window.
#
# days_since_last_research remains NaN until
# a first event has actually occurred.
# =========================================================


# =========================================================
# 10. Quality summary
# =========================================================
eligible = df[
    df[
        "alt_data_eligible"
    ]
].copy()


print("\n")
print("=" * 72)
print(
    "INSTITUTIONAL ATTENTION FEATURE SUMMARY"
)
print("=" * 72)


print(
    f"Source starts: "
    f"{source_start.date()}"
)

print(
    f"90-day mature feature start: "
    f"{eligible_start.date()}"
)

print(
    f"Eligible panel rows: "
    f"{len(eligible):,}"
)

print(
    f"Eligible stocks: "
    f"{eligible['code'].nunique()}"
)


print(
    "\nFeature coverage within eligible period:"
)


for col in feature_cols:

    print(
        f"{col:32s} "
        f"{eligible[col].notna().mean():.2%}"
    )


# =========================================================
# 11. Distribution
# =========================================================
distribution_cols = [
    "research_events_30d",
    "institution_count_30d",
    "institution_count_90d",
    "attention_acceleration",
    "attention_intensity_30d",
    "research_recency_score",
]


print(
    "\nFeature distribution:"
)


print(
    eligible[
        distribution_cols
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
    .to_string()
)


# =========================================================
# 12. Most institutionally active stocks
# =========================================================
latest_date = (
    eligible["date"].max()
)


latest = (
    eligible[
        eligible["date"]
        == latest_date
    ]
    .sort_values(
        "institution_count_90d",
        ascending=False,
    )
)


print(
    f"\nTop institutional attention "
    f"on {latest_date.date()}:"
)


show_cols = [
    "code",
    "name",
    "industry",
    "research_events_30d",
    "institution_count_30d",
    "institution_count_90d",
    "attention_acceleration",
    "days_since_last_research",
]


print(
    latest[
        show_cols
    ]
    .head(15)
    .to_string(
        index=False
    )
)


# =========================================================
# 13. PIT sanity check
#
# Event cannot affect a trading date
# before its announcement.
# Mapping already guarantees that.
# =========================================================
duplicate_rows = (
    df
    .duplicated(
        [
            "code",
            "date",
        ]
    )
    .sum()
)


print(
    f"\nDuplicate code-date rows: "
    f"{duplicate_rows}"
)


assert (
    duplicate_rows
    == 0
)


# =========================================================
# 14. Save
# =========================================================
output_file = (
    PROCESSED_DIR
    / "institutional_attention_features.parquet"
)


df.to_parquet(
    output_file,
    index=False,
)


print(
    f"\nSaved to: "
    f"{output_file}"
)


print(
    "\nPASS: "
    "Institutional alternative-data "
    "features created."
)