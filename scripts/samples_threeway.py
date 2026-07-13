"""Three-way per-question analysis for the samples-in-prompt ablation.
Per (collection, model): majority-vote TA (execution_success) over 5 runs for three conditions:
  minimal    = no enrichment (name+type)            -> {model}_False_run{n}.jsonl in ENRICHED dir
  samplesonly= name+type+range+sample values, no NL -> {model}_True_run{n}.jsonl  in SAMPLESONLY dir
  enriched   = name+type+NL description             -> {model}_True_run{n}.jsonl  in ENRICHED dir
McNemar exact test on each pairwise comparison, plus flip categories for enriched-vs-samplesonly."""
import json, math, os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = {
    "BR-Sa1": ROOT + "/data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d/runs",
    "PMG": ROOT + "/data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185/runs",
    "Campinas": ROOT + "/data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0/runs",
}
ENRICHED_SUB = {"BR-Sa1": "brsa1-paper-canonical-5runs-new/detailed_logs",
                "PMG": "detailed_logs", "Campinas": "detailed_logs"}
CAT = {"comp": "computed", "temp": "temporal", "dist": "distribution",
       "corr": "correlation", "grp": "grouping", "flt": "filtering"}


def majority(logdir, model, label, field="execution_success"):
    votes = {}
    for run in range(1, 6):
        f = f"{logdir}/{model}_{label}_run{run}.jsonl"
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


def compare(a, b):
    qs = sorted(set(a) & set(b))
    pf = [q for q in qs if a[q] and not b[q]]   # a pass -> b fail
    fp = [q for q in qs if not a[q] and b[q]]   # a fail -> b pass
    return fp, pf, mcnemar_exact(len(pf), len(fp)), qs


for cell, base in D.items():
    edir = f"{base}/{ENRICHED_SUB[cell]}"
    sdir = f"{base}/detailed_logs_samplesonly"
    for model in ["gpt-4o-mini", "gemini-2.5-flash"]:
        try:
            mini = majority(edir, model, "False")
            enr = majority(edir, model, "True")
            samp = majority(sdir, model, "True")
        except FileNotFoundError as e:
            print(f"{cell} {model}: missing {e.filename}"); continue
        n = len(set(mini) & set(enr) & set(samp))
        ta = lambda d: 100 * sum(d[q] for q in (set(mini)&set(enr)&set(samp))) / n
        print(f"\n=== {cell} / {model}  (n={n}) ===")
        print(f"  TA: minimal {ta(mini):.1f}  ->  samplesonly {ta(samp):.1f}  ->  enriched {ta(enr):.1f}")
        # samplesonly vs minimal
        fp, pf, p, _ = compare(mini, samp)
        print(f"  samplesonly vs minimal : +{len(fp)}/-{len(pf)}  McNemar p={p:.3f}")
        # enriched vs minimal
        fp, pf, p, _ = compare(mini, enr)
        print(f"  enriched   vs minimal  : +{len(fp)}/-{len(pf)}  McNemar p={p:.3f}")
        # enriched vs samplesonly  (the decisive one: does the NL description add value beyond raw evidence?)
        fp, pf, p, _ = compare(samp, enr)
        cats_gain = Counter(CAT.get(q.split("_")[0], "?") for q in fp)
        print(f"  enriched   vs samplesonly: +{len(fp)}/-{len(pf)}  McNemar p={p:.3f}   (NL adds beyond samples)")
        print(f"      description-only wins by category: {dict(cats_gain)}")
