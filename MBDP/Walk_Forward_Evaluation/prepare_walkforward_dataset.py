from pathlib import Path

import numpy as np
import pandas as pd

def _repo_data(name):
    """Locate a dataset in the repository data/ folder (or next to the script)."""
    here = Path(__file__).resolve().parent
    for base in (here, *here.parents):
        for candidate in (base / "data" / name, base / name):
            if candidate.is_file():
                return str(candidate)
    return str(here.parent / "data" / name)


src = Path(_repo_data("djia30_full_2009_2022.csv"))
dst = Path("djia30_full_2009_2022_1.csv")

raw = pd.read_csv(src)
raw.columns = raw.columns.str.strip()
raw["Ticker"] = raw["Ticker"].astype(str).str.strip()

excluded = ["DOW", "WBA"]
clean = raw.loc[~raw["Ticker"].isin(excluded)].copy()

clean["Date"] = pd.to_datetime(clean["Date"], errors="raise")
clean["Adj Close"] = pd.to_numeric(clean["Adj Close"], errors="raise")

if clean.empty or clean["Date"].isna().any():
    raise ValueError("Empty data or missing dates.")

if clean.duplicated(["Date", "Ticker"]).any():
    raise ValueError("Duplicate date/ticker rows.")

panel = clean.pivot(
    index="Date", columns="Ticker", values="Adj Close"
).sort_index()

values = panel.to_numpy(dtype=float)
if not np.isfinite(values).all() or (values <= 0).any():
    raise ValueError("Remaining panel still has missing or invalid prices.")

if panel.shape[1] != 28:
    raise ValueError(f"Expected 28 symbols, found {panel.shape[1]}.")

clean.sort_values(["Date", "Ticker"]).to_csv(dst, index=False, mode="x")

print("Saved:", dst)
print("Explicitly excluded:", excluded)
print("Dates / symbols:", panel.shape)
print("Range:", panel.index.min(), "to", panel.index.max())
