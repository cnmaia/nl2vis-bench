"""Per-category accuracy and error decomposition from a detailed-log JSONL
(one representative run). TA = execution_success; VA = overall_correct."""
import json
import sys
from collections import defaultdict

CATS = {"comp": "Computed", "temp": "Temporal", "dist": "Distribution",
        "corr": "Correlation", "grp": "Grouping", "flt": "Filtering"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def per_category(rows):
    agg = defaultdict(lambda: [0, 0, 0])  # n, ta, va
    for r in rows:
        c = CATS.get(r["question_id"].split("_")[0], "Other")
        agg[c][0] += 1
        agg[c][1] += int(r["execution_success"])
        agg[c][2] += int(r["overall_correct"])
    return agg


def errors(rows):
    correct = execfail = vizwrong = axeswrong = 0
    for r in rows:
        if r["overall_correct"]:
            correct += 1
        elif not r["execution_success"]:
            execfail += 1
        elif not r["viz_type_correct"]:
            vizwrong += 1
        else:
            axeswrong += 1
    return correct, execfail, vizwrong, axeswrong


for path in sys.argv[1:]:
    rows = load(path)
    print(f"\n=== {path}  (n={len(rows)}) ===")
    agg = per_category(rows)
    order = ["Computed", "Temporal", "Distribution", "Correlation", "Grouping", "Filtering"]
    print(f"{'category':<13}{'N':>3}{'TA%':>7}{'VA%':>7}")
    for c in order:
        n, ta, va = agg[c]
        if n:
            print(f"{c:<13}{n:>3}{100*ta/n:>7.0f}{100*va/n:>7.0f}")
    c, ef, vw, aw = errors(rows)
    tot = len(rows)
    print(f"decomp: correct={c} exec_fail={ef} viz_type_wrong={vw} axes_wrong={aw}  (of {tot})")
