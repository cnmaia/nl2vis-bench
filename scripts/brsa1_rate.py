"""BR-Sa1 metadata-quality rating vs CF long_name (curator reference embedded in the deposit).
Ratings are the authors' rubric (correct/partial/wrong), encoded per column for reproducibility.
Cosine (OpenAI text-embedding-3-small) computed generated-vs-CF-long_name for corroboration,
consistent with the PMG/Campinas evaluation."""
import csv, os, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = ROOT + "/data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d"
REF = D + "/evaluation/cf_reference.csv"
ENVF = ROOT + "/local.env"
if os.path.exists(ENVF):
    for line in open(ENVF):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Authors' rubric. Default = correct; only the non-correct columns are listed.
PARTIAL = {
    "Clear_sky_days", "Clear_sky_nights",           # "days/nights" vs clear-sky daylight/night HOURS
    "aluvd_mean", "aluvd_minimum", "aluvd_std", "aluvd_variance",  # UV-visible albedo -> gives only UV or only visible / muddled spectrum
    "alnid_maximum",                                # dropped NIR/diffuse, vague "or related reflectivity"
    "adj_sfc_sw_diff_pri_daily", "adj_sfc_par_diff_pri_daily",     # pristine-sky -> "primary/specific component"
    "adj_sfc_sw_diff_naer_daily",                   # no-aerosol -> vague "non-aerosol scattering"
    "e_median", "e_minimum", "e_variance",          # evaporation named but hedged with latent heat / energy-water flux
    "ro_variance",                                  # runoff hedged with "surface roughness"
}
WRONG = {
    "e_mean",       # "mean surface emissivity" -- it is evaporation
    "e_maximum",    # "emissivity or an energy flux component" -- evaporation
    "e_std",        # "environmental variable 'e', units not available" -- no identification
}

rows = list(csv.DictReader(open(REF)))
def rate(c):
    return "wrong" if c in WRONG else ("partial" if c in PARTIAL else "correct")

counts = {"correct": 0, "partial": 0, "wrong": 0}
for r in rows:
    counts[rate(r["column_name"])] += 1
n = len(rows)
print(f"BR-Sa1 rubric over {n} columns: "
      f"correct {counts['correct']}  partial {counts['partial']}  wrong {counts['wrong']}  "
      f"({100*counts['correct']/n:.0f}% correct)")
print("  WRONG:", sorted(WRONG))
print("  PARTIAL:", sorted(PARTIAL))

# cosine corroboration (skip columns without a CF long_name, e.g. time)
pairs = [(r["column_name"], r["generated_description"], r["cf_long_name"])
         for r in rows if r["cf_long_name"].strip()]
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def emb(texts):
        return [d.embedding for d in client.embeddings.create(
            model="text-embedding-3-small", input=texts).data]
    g = emb([p[1] for p in pairs]); h = emb([p[2] for p in pairs])
    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
        return dot/(na*nb)
    sims = [(pairs[i][0], cos(g[i], h[i])) for i in range(len(pairs))]
    mean = sum(s for _, s in sims)/len(sims)
    print(f"\nmean cosine (gen vs CF long_name, n={len(sims)}): {mean:.3f}")
    print("  lowest 8 cosines:")
    for c, s in sorted(sims, key=lambda x: x[1])[:8]:
        print(f"    {c:22s} {s:.2f}  [{rate(c)}]")
except Exception as e:
    print(f"\ncosine skipped: {e}")
