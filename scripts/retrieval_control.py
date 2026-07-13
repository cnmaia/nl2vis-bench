"""Retrieval CONTROL experiment: separate 'any text helps retrieval'
from 'THESE descriptions help'. Conditions per column representation:
  bare       : column name only
  generic    : name + one fixed content-neutral sentence (same for every column)
  shuffled   : name + another column's real LLM description (length-matched, WRONG content)
  enriched   : name + its own LLM description
  cf_longname: name + embedded CF long_name  (BR-Sa1 only; curator baseline)
Metric: MRR / recall@k over expected_columns, per collection and pooled (109 cols).
Embeddings: OpenAI text-embedding-3-small (same as the main retrieval experiment)."""
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
GENERIC = "This column stores recorded numeric measurement values from the dataset."

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
    cf = {}
    ref = d + "/evaluation/cf_reference.csv"
    if os.path.exists(ref):
        for r in csv.DictReader(open(ref)):
            if r.get("cf_long_name", "").strip():
                cf[r["column_name"]] = r["cf_long_name"].strip()
    qs = yaml.safe_load(open(d + "/evaluation/questions.yaml"))["questions"]
    return cols, cf, qs

def reps(cols, cf, mode):
    names = list(cols)
    if mode == "bare":       return {c: c for c in names}
    if mode == "generic":    return {c: f"{c}: {GENERIC}" for c in names}
    if mode == "enriched":   return {c: f"{c}: {cols[c]}" for c in names}
    if mode == "shuffled":   # each column gets the NEXT column's description (deterministic, wrong)
        n = len(names)
        return {names[i]: f"{names[i]}: {cols[names[(i+1) % n]]}" for i in range(n)}
    if mode == "cf_longname":return {c: (f"{c}: {cf[c]}" if c in cf else c) for c in names}
    raise ValueError(mode)

def rank_eval(questions, colnames, emb):
    qembs = embed([q["question"] for q in questions])
    mrr = 0.0; rec = {1:0.0,3:0.0,5:0.0}; n = 0
    for q, qe in zip(questions, qembs):
        gt = {g for g in (q.get("expected_columns") or []) if g in colnames}
        if not gt: continue
        n += 1
        ranked = sorted(colnames, key=lambda c: cos(qe, emb[c]), reverse=True)
        first = next((i+1 for i, c in enumerate(ranked) if c in gt), None)
        if first: mrr += 1.0/first
        for k in rec: rec[k] += len(gt & set(ranked[:k]))/len(gt)
    return mrr/n, rec[1]/n, rec[3]/n, rec[5]/n, n

def emb_for(cols, cf, mode):
    r = reps(cols, cf, mode); e = embed(list(r.values()))
    return {c: v for c, v in zip(r.keys(), e)}

MODES = ["bare", "generic", "shuffled", "enriched", "cf_longname"]
print(f"{'collection':10s} {'mode':11s}  {'MRR':>5} {'R@1':>5} {'R@3':>5} {'R@5':>5}  n")
allcols, allcf, allq = {}, {}, []
for coll in COLL:
    cols, cf, qs = load(coll)
    allcols.update({f"{coll}::{c}": d for c, d in cols.items()})
    allcf.update({f"{coll}::{c}": v for c, v in cf.items()})
    for q in qs:
        q2 = dict(q); q2["expected_columns"] = [f"{coll}::{c}" for c in (q.get("expected_columns") or [])]
        allq.append(q2)
    for mode in MODES:
        if mode == "cf_longname" and not cf: continue
        e = emb_for(cols, cf, mode)
        print(f"{coll:10s} {mode:11s}  %5.2f %5.2f %5.2f %5.2f  %d" % rank_eval(qs, list(cols), e))

print("\n-- POOLED (109 columns) --")
for mode in MODES:
    if mode == "cf_longname":  # only BR-Sa1 has cf; skip pooled cf
        continue
    e = emb_for(allcols, allcf, mode)
    print(f"{'POOLED':10s} {mode:11s}  %5.2f %5.2f %5.2f %5.2f  %d" % rank_eval(allq, list(allcols), e))
