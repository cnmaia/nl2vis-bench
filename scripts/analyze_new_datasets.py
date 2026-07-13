"""Analyze the JCDL multi-dataset ablation: TA/VA deltas + t-tests per dataset x model.
Reads the per-dataset runs/ YAMLs (and the BR-Sa1 canonical run) and prints a table."""
import glob
import math
import os
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs(path, metric):
    d = yaml.safe_load(open(path))
    return d["aggregated"][metric]["runs"]


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, s


def ttest_ind(a, b):
    """Student's two-sample t-test (pooled), returns (t, df, p)."""
    na, nb = len(a), len(b)
    ma, sa = mean_std(a)
    mb, sb = mean_std(b)
    df = na + nb - 2
    sp2 = (((na - 1) * sa ** 2) + ((nb - 1) * sb ** 2)) / df
    denom = math.sqrt(sp2 * (1 / na + 1 / nb)) if sp2 > 0 else 0.0
    if denom == 0:
        return float("inf") if (ma - mb) != 0 else 0.0, df, (0.0 if (ma - mb) != 0 else 1.0)
    t = (ma - mb) / denom
    try:
        from scipy import stats
        p = 2 * stats.t.sf(abs(t), df)
    except Exception:
        # normal approximation fallback
        p = 2 * 0.5 * math.erfc(abs(t) / math.sqrt(2))
    return t, df, p


def row(no_path, yes_path, model):
    out = {}
    for metric, key in [("execution_success_rate", "TA"), ("viz_accuracy", "VA")]:
        no_r, yes_r = runs(no_path, metric), runs(yes_path, metric)
        mno, sno = mean_std(no_r)
        myes, syes = mean_std(yes_r)
        t, df, p = ttest_ind(yes_r, no_r)
        out[key] = (mno, sno, myes, syes, (myes - mno) * 100, t, df, p)
    return model, out


def pf(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def report(name, pairs):
    """pairs: list of (model_label, no_path, yes_path)."""
    print(f"\n=== {name} ===")
    print(f"{'model':<18}{'TA no':>9}{'TA yes':>9}{'dTA pp':>8}{'p(TA)':>9}   {'VA no':>9}{'VA yes':>9}{'dVA pp':>8}{'p(VA)':>9}")
    for model, no_p, yes_p in pairs:
        if not (os.path.exists(no_p) and os.path.exists(yes_p)):
            miss = no_p if not os.path.exists(no_p) else yes_p
            print(f"{model:<18}  (missing: {os.path.basename(miss)})")
            continue
        _, o = row(no_p, yes_p, model)
        ta, va = o["TA"], o["VA"]
        print(f"{model:<18}{ta[0]*100:>8.1f}%{ta[2]*100:>8.1f}%{ta[4]:>+8.1f}{pf(ta[7]):>9}   "
              f"{va[0]*100:>8.1f}%{va[2]*100:>8.1f}%{va[4]:>+8.1f}{pf(va[7]):>9}")


PMG = ROOT + "/data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185/runs"
CAMP = ROOT + "/data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0/runs"
NEE = ROOT + "/data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d/runs/brsa1-paper-canonical-5runs-new"

report("BR-Sa1 (carbon flux, 81 cols)", [
    ("gpt-4o-mini", f"{NEE}/gpt_no_enrichment_5runs_new.yaml", f"{NEE}/gpt_with_enrichment_5runs_new.yaml"),
    ("gemini-2.5-flash", f"{NEE}/gemini_no_enrichment_5runs_new.yaml", f"{NEE}/gemini_with_enrichment_5runs_new.yaml"),
])
report("PMG (subsurface gas, 16 cols)", [
    ("gpt-4o-mini", f"{PMG}/gpt-4o-mini_false.yaml", f"{PMG}/gpt-4o-mini_true.yaml"),
    ("gemini-2.5-flash", f"{PMG}/gemini-2.5-flash_false.yaml", f"{PMG}/gemini-2.5-flash_true.yaml"),
])
report("Campinas (aerosols, 11 cols)", [
    ("gpt-4o-mini", f"{CAMP}/gpt-4o-mini_false.yaml", f"{CAMP}/gpt-4o-mini_true.yaml"),
    ("gemini-2.5-flash", f"{CAMP}/gemini-2.5-flash_false.yaml", f"{CAMP}/gemini-2.5-flash_true.yaml"),
])
