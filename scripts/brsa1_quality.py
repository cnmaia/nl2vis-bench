"""BR-Sa1 metadata-quality reference builder.
Curator reference = CF long_name/units embedded in the deposited NetCDF (CF-1.x), which is
author-independent standard documentation. Aligns each generated description (column_metadata.csv)
with its CF long_name so the authors can rate correct/partial/wrong."""
import csv, os
import xarray as xr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = ROOT + "/data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d"
NC = D + "/NEE-METEORS-AMAZON_BRSa1_2002_2011_with_qc_flags.nc"
CSV = D + "/column_metadata.csv"

ds = xr.open_dataset(NC, decode_times=False)
cf = {}
for name, var in list(ds.variables.items()):
    cf[name] = {
        "long_name": var.attrs.get("long_name", ""),
        "standard_name": var.attrs.get("standard_name", ""),
        "units": var.attrs.get("units", ""),
    }

gen = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        gen[row["column_name"]] = row["description"]

n_long = sum(1 for c in cf.values() if c["long_name"])
n_units = sum(1 for c in cf.values() if c["units"])
print(f"CF coverage: long_name {n_long}/{len(cf)}  units {n_units}/{len(cf)}\n")

rows = []
for col, desc in gen.items():
    ref = cf.get(col, {})
    rows.append((col, ref.get("long_name", ""), ref.get("units", ""), desc))

# print aligned for author rating
for col, ln, un, desc in rows:
    print(f"### {col}   [units: {un or '-'}]")
    print(f"  CF long_name : {ln or '(none)'}")
    print(f"  generated    : {desc[:180]}")
    print()

out = D + "/evaluation/cf_reference.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["column_name", "cf_long_name", "cf_standard_name", "units", "generated_description"])
    for col, desc in gen.items():
        ref = cf.get(col, {})
        w.writerow([col, ref.get("long_name", ""), ref.get("standard_name", ""), ref.get("units", ""), desc])
print(f"\nwrote {out}  ({len(gen)} columns; CF long_name on {n_long})")
