"""Per-question paired analysis (T1.1) + flip analysis (T1.2).
Per (collection, model): majority-vote each question's TA (execution_success) over the 5 runs,
without vs with enrichment; McNemar exact test on discordant pairs; list fail->pass / pass->fail
questions and their categories."""
import json, math, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = {
    "BR-Sa1": ROOT + "/data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d/runs/brsa1-paper-canonical-5runs-new/detailed_logs",
    "PMG": ROOT + "/data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185/runs/detailed_logs",
    "Campinas": ROOT + "/data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0/runs/detailed_logs",
}
CAT = {"comp": "computed", "temp": "temporal", "dist": "distribution",
       "corr": "correlation", "grp": "grouping", "flt": "filtering"}


def majority(logdir, model, enrich, field="execution_success"):
    """Return {question_id: bool} majority over 5 runs."""
    votes = {}
    for run in range(1, 6):
        f = f"{logdir}/{model}_{enrich}_run{run}.jsonl"
        for line in open(f):
            r = json.loads(line)
            votes.setdefault(r["question_id"], []).append(int(r[field]))
    return {q: (sum(v) >= 3) for q, v in votes.items()}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


for cell, logdir in CELLS.items():
    for model in ["gpt-4o-mini", "gemini-2.5-flash"]:
        try:
            no = majority(logdir, model, "False")
            yes = majority(logdir, model, "True")
        except FileNotFoundError:
            print(f"{cell} {model}: logs missing"); continue
        qs = sorted(no)
        b = [q for q in qs if no[q] and not yes[q]]        # pass->fail
        c = [q for q in qs if not no[q] and yes[q]]        # fail->pass
        p = mcnemar_exact(len(b), len(c))
        ta_no = 100 * sum(no.values()) / len(qs)
        ta_yes = 100 * sum(yes.values()) / len(qs)
        cats_pass = Counter(CAT.get(q.split("_")[0], "?") for q in c)
        cats_fail = Counter(CAT.get(q.split("_")[0], "?") for q in b)
        print(f"\n{cell} / {model}: TA {ta_no:.0f}->{ta_yes:.0f}  "
              f"fail->pass={len(c)} pass->fail={len(b)}  McNemar p={p:.3f}")
        print(f"   fail->pass {dict(cats_pass)}  {c}")
        print(f"   pass->fail {dict(cats_fail)}  {b}")
