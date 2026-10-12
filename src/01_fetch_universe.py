\
from pathlib import Path
import sys
import pandas as pd
import akshare as ak

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, INDEX_CODE, N_STOCKS

RAW_DIR.mkdir(parents=True, exist_ok=True)

df = ak.index_stock_cons_csindex(symbol=INDEX_CODE).copy()

rename_map = {
    "成分券代码": "code",
    "成分券名称": "name",
    "交易所": "exchange",
    "日期": "snapshot_date",
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

if "code" not in df.columns:
    raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

df["code"] = df["code"].astype(str).str.zfill(6)
df = df.drop_duplicates("code").head(N_STOCKS).reset_index(drop=True)

out = RAW_DIR / "universe.parquet"
df.to_parquet(out, index=False)

print(df.head(10))
print(f"\nSaved {len(df)} stocks to {out}")
