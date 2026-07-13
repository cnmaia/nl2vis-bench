"""Correct-column accuracy, strict vs equivalence-relaxed, from existing logs.
TA (execution_success) = code runs; it does NOT check semantic column choice and does NOT
use the equivalence classes. Here we instead parse the columns each generated query actually
references and score them against the questions' expected_columns, under two rules:
  loose  = query uses the expected columns OR their declared equivalents
  strict = query uses the exact expected columns (no equivalence credit)
Majority vote over 5 runs, per (collection, model, condition). Shows how much of the
column-mapping success depends on the loose equivalence classes."""
import csv, json, os, re, yaml
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = {
    "BR-Sa1/GPT":      ("nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d",
                         "brsa1-paper-canonical-5runs-new/detailed_logs", "gpt-4o-mini"),
    "Campinas/Gemini": ("lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0",
                         "detailed_logs", "gemini-2.5-flash"),
}

def load(coll_dir):
    d = f"{ROOT}/data/{coll_dir}"
    cols = [r["column_name"] for r in csv.DictReader(open(d + "/column_metadata.csv"))]
    qs = {q["id"]: [c for c in (q.get("expected_columns") or [])]
          for q in yaml.safe_load(open(d + "/evaluation/questions.yaml"))["questions"]}
    equiv = defaultdict(set)
    ep = d + "/evaluation/semantic_equivalence.yaml"
    if os.path.exists(ep):
        for pair in (yaml.safe_load(open(ep)).get("equivalent_pairs") or []):
            g = pair["columns"]
            for c in g: equiv[c] |= set(g)
    return cols, qs, equiv

def used_columns(code, schema):
    """Real schema columns referenced in the generated pandas code."""
    return {c for c in schema
            if re.search(r"['\"]" + re.escape(c) + r"['\"]", code) or re.search(r"\bdf\." + re.escape(c) + r"\b", code)}

def majority(rows_by_q):
    return {q: (sum(v) >= 3) for q, v in rows_by_q.items()}

for name, (cdir, logsub, model) in CELLS.items():
    schema, expected, equiv = load(cdir)
    logdir = f"{ROOT}/data/{cdir}/runs/{logsub}"
    # only questions whose expected columns are real schema columns (skip pure-derived)
    scored_q = {q: e for q, e in expected.items() if e and all(c in schema for c in e)}
    equiv_q = [q for q, e in scored_q.items() if any(equiv.get(c) for c in e)]
    print(f"\n=== {name} === ({len(scored_q)} column-scored questions; {len(equiv_q)} touch an equivalence pair: {equiv_q})")
    for cond, label in [("False", "minimal"), ("True", "enriched")]:
        loose_v, strict_v = defaultdict(list), defaultdict(list)
        for run in range(1, 6):
            f = f"{logdir}/{model}_{cond}_run{run}.jsonl"
            for line in open(f):
                r = json.loads(line)
                q = r["question_id"]
                if q not in scored_q: continue
                code = r.get("generated_code", "") or ""
                exok = str(r["execution_success"]) == "True"
                use = used_columns(code, schema)
                exp = scored_q[q]
                loose_ok = exok and all((use & (equiv.get(c, {c}) | {c})) for c in exp)
                strict_ok = exok and all(c in use for c in exp)
                loose_v[q].append(int(loose_ok)); strict_v[q].append(int(strict_ok))
        L, S = majority(loose_v), majority(strict_v)
        n = len(scored_q)
        la, sa = 100*sum(L.values())/n, 100*sum(S.values())/n
        flips = [q for q in L if L[q] and not S[q]]
        print(f"  {label:9s}: correct-col loose {la:.0f}%  strict {sa:.0f}%  (gap {la-sa:.0f} pts; equivalence-credited Qs: {flips})")
