"""Objective metadata-quality metric: cosine similarity between LLM-generated and
human-authored column descriptions (complements the manual rubric in §4.5).

Usage (needs OPENAI_API_KEY; reads local.env if present):
  python scripts/metadata_quality_eval.py \
    --schema data/<dataset>/column_metadata.csv \
    --human  data/<dataset>/evaluation/human_descriptions.yaml
"""
import argparse
import csv
import math
import os

import yaml


def embed(texts, model="text-embedding-3-small"):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    out = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in out.data]


def cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--human", required=True)
    args = ap.parse_args()

    gen = {r["column_name"]: (r.get("description") or "").strip()
           for r in csv.DictReader(open(args.schema))}
    human = yaml.safe_load(open(args.human))["descriptions"]

    cols = [c for c in human if c in gen and gen[c]]
    gen_emb = embed([gen[c] for c in cols])
    hum_emb = embed([human[c] for c in cols])

    sims = []
    print(f"{'column':>16}  {'cosine':>7}")
    for c, g, h in zip(cols, gen_emb, hum_emb):
        s = cos(g, h)
        sims.append(s)
        print(f"{c:>16}  {s:>7.3f}")
    print(f"\nmean cosine: {sum(sims) / len(sims):.3f}  (n={len(sims)})")


if __name__ == "__main__":
    main()
