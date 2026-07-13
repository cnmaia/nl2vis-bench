"""Findability / retrieval experiment (R2#1): does enrichment make opaque columns
retrievable by a natural-language query? For each question we rank columns by cosine
similarity between the question embedding and the column representation, under two
representations: BARE (column name only, what a schema-only catalog exposes) vs
ENRICHED (name + generated description). Metrics: MRR and Recall@k over the ground-truth
expected_columns. Run per-collection and POOLED (all 109 columns from 3 collections in
one index = repository-scale findability). Embeddings: OpenAI text-embedding-3-small."""
import csv, os, math
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVF = ROOT + "/local.env"
if os.path.exists(ENVF):
    for line in open(ENVF):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

COLL = {
    "BR-Sa1": "data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d",
    "PMG": "data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185",
    "Campinas": "data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0",
}

from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
_cache = {}
def embed(texts):
    todo = [t for t in texts if t not in _cache]
    for i in range(0, len(todo), 256):
        chunk = todo[i:i+256]
        for t, d in zip(chunk, client.embeddings.create(model="text-embedding-3-small", input=chunk).data):
            _cache[t] = d.embedding
    return [_cache[t] for t in texts]

def cos(a, b):
    dot = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na and nb else 0.0

def load(coll):
    d = ROOT + "/" + COLL[coll]
    cols = {}
    for r in csv.DictReader(open(d + "/column_metadata.csv")):
        cols[r["column_name"]] = r["description"]
    qs = yaml.safe_load(open(d + "/evaluation/questions.yaml"))["questions"]
    return cols, qs

def rank_eval(questions, colnames, colrep_emb):
    """colrep_emb: {colname: embedding}. Returns (MRR, recall@1, @3, @5)."""
    qtexts = [q["question"] for q in questions]
    qembs = embed(qtexts)
    mrr = 0.0; rec = {1:0.0, 3:0.0, 5:0.0}
    for q, qe in zip(questions, qembs):
        gt = set(q.get("expected_columns") or [])
        gt = {g for g in gt if g in colrep_emb}
        if not gt:
            continue
        ranked = sorted(colnames, key=lambda c: cos(qe, colrep_emb[c]), reverse=True)
        first = next((i+1 for i, c in enumerate(ranked) if c in gt), None)
        if first:
            mrr += 1.0/first
        for k in rec:
            hit = len(gt & set(ranked[:k]))
            rec[k] += hit/len(gt)   # recall@k = fraction of GT columns in top-k
    n = sum(1 for q in questions if (set(q.get("expected_columns") or []) & set(colnames)))
    return mrr/n, rec[1]/n, rec[3]/n, rec[5]/n, n

def build_emb(colmap, mode):
    reps = {c: (c if mode == "bare" else f"{c}: {desc}") for c, desc in colmap.items()}
    embs = embed(list(reps.values()))
    return {c: e for c, e in zip(reps.keys(), embs)}

# ---- per-collection ----
print(f"{'collection':10s} {'cond':8s}  {'MRR':>5} {'R@1':>5} {'R@3':>5} {'R@5':>5}  n")
allcols = {}; allq = []
for coll in COLL:
    cols, qs = load(coll)
    allcols.update({f"{coll}::{c}": d for c, d in cols.items()})
    for q in qs:
        q2 = dict(q); q2["expected_columns"] = [f"{coll}::{c}" for c in (q.get("expected_columns") or [])]
        allq.append(q2)
    for mode in ["bare", "enriched"]:
        ce = build_emb(cols, mode)
        mrr, r1, r3, r5, n = rank_eval(qs, list(cols), ce)
        print(f"{coll:10s} {mode:8s}  {mrr:5.2f} {r1:5.2f} {r3:5.2f} {r5:5.2f}  {n}")

# ---- pooled (repository-scale: all 109 columns in one index) ----
print("\n-- POOLED across all 3 collections (109 columns, repository-scale findability) --")
for mode in ["bare", "enriched"]:
    ce = build_emb(allcols, mode)
    mrr, r1, r3, r5, n = rank_eval(allq, list(allcols), ce)
    print(f"{'POOLED':10s} {mode:8s}  {mrr:5.2f} {r1:5.2f} {r3:5.2f} {r5:5.2f}  {n}")
