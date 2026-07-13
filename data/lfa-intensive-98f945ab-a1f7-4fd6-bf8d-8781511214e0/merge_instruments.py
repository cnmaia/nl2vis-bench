"""Merge Campinas L3 30min instrument tables into one wide aerosol dataset.

Produces `campinas_merged_sample.csv` with a single `time` column (UTC) and the
summary variables from each instrument, using technical/opaque column names
(the point of the enrichment test). The SMPS per-bin size distribution
(~100 columns) is intentionally excluded here and kept for the schema-scaling
study; only its summary columns are merged.
"""
import glob
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# instrument file glob -> {raw-substring: clean_name}
SPECS = {
    "CPC_3772": {"conc Mean": "conc"},
    "MAAP_5012": {"BC637": "BC637", "Abs637": "Abs637"},
    "Nephelometer_Aurora_Ecotech_3000": {
        "scat450": "scat450", "scat525": "scat525", "scat635": "scat635",
        "Angstrom": "Angstrom", "SSA637": "SSA637",
    },
    "SMPS_TSI_3082": {
        "Total concentration": "total_conc_smps",
        "Total volume": "total_volume",
        "Mean diameter": "mean_diameter",
    },
}


def load_instrument(instrument, mapping):
    path = glob.glob(os.path.join(HERE, f"Campinas_*_{instrument}_L3_30min.csv"))[0]
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]          # strip leading/trailing spaces
    df = df.loc[:, [c for c in df.columns if c]]          # drop unnamed/empty column
    cols = {"Time(UTC)": "time"}
    for raw_sub, clean in mapping.items():
        match = [c for c in df.columns if raw_sub in c]
        if not match:
            raise KeyError(f"{instrument}: no column matching '{raw_sub}' in {list(df.columns)}")
        cols[match[0]] = clean
    df = df[list(cols)].rename(columns=cols)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.dropna(subset=["time"])


merged = None
for instrument, mapping in SPECS.items():
    d = load_instrument(instrument, mapping)
    merged = d if merged is None else merged.merge(d, on="time", how="outer")

merged = merged.sort_values("time").reset_index(drop=True)
out = os.path.join(HERE, "campinas_merged_sample.csv")
merged.to_csv(out, index=False)
print("rows:", len(merged))
print("columns:", list(merged.columns))
print(merged.head(3).to_string())
