"""Prepare the PMG subsurface-gas xlsx as a harness-ready sample CSV.

Renames `date` -> `time` (the validator hardcodes parse_dates=['time']) and
writes `pmg_sample.csv` next to the source file.
"""
import glob
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
src = glob.glob(os.path.join(HERE, "uspgasmon1weach.00.*.xlsx"))[0]

df = pd.read_excel(src)
df = df.rename(columns={"date": "time"})
df["time"] = pd.to_datetime(df["time"], errors="coerce")

out = os.path.join(HERE, "pmg_sample.csv")
df.to_csv(out, index=False)
print("rows:", len(df))
print("columns:", list(df.columns))
print(df.head(3).to_string())
