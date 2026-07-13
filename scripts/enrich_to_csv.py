"""Standalone enrichment: build a column_metadata.csv (with LLM descriptions)
from a sample CSV, in the exact schema the validator's load_schema_from_csv expects.

Mirrors the metadata-extractor's batch description prompt, but runs locally with
only an API key (no DB / MinIO). Use the SAME enrichment model across all datasets
for methodological consistency (the original NEE descriptions were gemini-2.5-flash).

Usage:
  python scripts/enrich_to_csv.py --data <sample.csv> --model gemini-2.5-flash \
      --output <dir>/column_metadata.csv --dataset-name <name>
"""
import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nl2vis_bench.validator.cli import create_llm_provider  # noqa: E402

BATCH_SIZE = 10
BATCH_PROMPT = """You are a data analyst. Given multiple columns from an environmental dataset file "{file_name}", generate a brief description for each.

Columns:
{columns_info}

For EACH column, respond with exactly one line in this format:
COLUMN_NAME: One sentence description

Be specific and mention units if apparent from the values. Include ALL columns listed above."""


def profile(df):
    rows = []
    for name in df.columns:
        s = df[name]
        numeric = pd.api.types.is_numeric_dtype(s)
        samples = [str(v) for v in s.dropna().unique()[:3]]
        rows.append({
            "column_name": name,
            "dtype": str(s.dtype),
            "min_value": float(s.min()) if numeric and s.notna().any() else "",
            "max_value": float(s.max()) if numeric and s.notna().any() else "",
            "mean_value": float(s.mean()) if numeric and s.notna().any() else "",
            "null_count": int(s.isna().sum()),
            "sample_values": "; ".join(samples),
        })
    return rows


def format_col(r):
    return (f"- {r['column_name']} (type: {r['dtype']}, samples: {r['sample_values'] or 'N/A'}, "
            f"min: {r['min_value'] or 'N/A'}, max: {r['max_value'] or 'N/A'}, mean: {r['mean_value'] or 'N/A'})")


def parse(resp):
    out = {}
    for line in resp.strip().splitlines():
        m = re.match(r"^\s*[-*]?\s*([^:]+):\s*(.+)$", line.strip())
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset-name", default="dataset")
    args = ap.parse_args()

    llm = create_llm_provider(args.model)
    df = pd.read_csv(args.data)
    rows = profile(df)
    file_name = os.path.basename(args.data)

    descriptions = {}
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        info = "\n".join(format_col(r) for r in batch)
        resp = llm.complete(BATCH_PROMPT.format(file_name=file_name, columns_info=info))
        descriptions.update(parse(resp))
        print(f"  batch {i // BATCH_SIZE + 1}: {len(batch)} cols")

    for r in rows:
        r["description"] = descriptions.get(r["column_name"], "")
        r["llm_provider"] = llm.model_name()
        if not r["description"]:
            print(f"  WARNING: no description parsed for '{r['column_name']}'")

    cols = ["column_name", "dtype", "description", "min_value", "max_value",
            "mean_value", "null_count", "sample_values", "llm_provider"]
    out = pd.DataFrame(rows)[cols]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} columns -> {args.output}")


if __name__ == "__main__":
    main()
