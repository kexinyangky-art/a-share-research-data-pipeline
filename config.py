from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"

INDEX_CODE = "000300"          # 沪深300
N_STOCKS = 80                  # 第一版先跑80只，稳定后可扩到300
START_DATE = "20220101"
END_DATE = "20260831"

# 财报期：先覆盖2022Q1-2026Q2
REPORT_DATES = [
    "20220331", "20220630", "20220930", "20221231",
    "20230331", "20230630", "20230930", "20231231",
    "20240331", "20240630", "20240930", "20241231",
    "20250331", "20250630", "20250930", "20251231",
    "20260331", "20260630",
]
