# NL2Vis Enrichment Benchmark & Harness

Anonymized artifact for a double-blind submission. This repository contains the evaluation
harness, analysis scripts, and benchmark used to study **LLM-based semantic metadata enrichment
for findable scientific data collections**: generating natural-language descriptions for opaque
scientific column names at ingestion, and measuring whether they improve column findability and
downstream natural-language-to-visualization (NL2Vis) accuracy.

> Author, institution, and repository details are withheld for review. No API keys are included;
> supply your own via environment variables (below).

## What's here

```
nl2vis_bench/        # the harness: schema loading, translation (NL->pandas), execution, viz selection, validator
scripts/            # experiment + analysis scripts (see "Reproducing the paper")
tests/              # unit/integration tests
data/               # the benchmark: 3 collections (BR-Sa1 flux, PMG subsurface gas, Campinas aerosols)
  <collection>/
    column_metadata.csv              # enriched descriptions (name, type, stats, LLM description)
    column_metadata_samplesonly.csv  # samples-in-prompt condition (no NL description)
    column_metadata_gpt.csv          # GPT-4o-mini-authored descriptions (cross-enrichment)
    column_metadata_cflongname.csv   # (BR-Sa1) embedded CF long_name as description (extract baseline)
    *_sample.csv                     # a data sample the harness runs queries against
    evaluation/
      questions.yaml                 # 50 NL discovery questions + ground truth (expected columns/query/chart)
      semantic_equivalence.yaml      # equivalence classes fixed before running
      cf_reference.csv               # (BR-Sa1) CF long_name curator references for the quality rubric
    runs/                            # cached run outputs (*.yaml) + per-question detailed_logs (*.jsonl)
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,netcdf]"
export OPENAI_API_KEY=...     # for GPT-4o-mini queries + text-embedding-3-small (retrieval/cosine)
export GEMINI_API_KEY=...     # for Gemini 2.5 Flash queries
```

## Running the harness

The validator runs the with/without-enrichment ablation for one collection and model:

```bash
python -m nl2vis_bench.validator \
  --model gpt-4o-mini --enrichment true --runs 5 --detailed-logs \
  --schema data/<collection>/column_metadata.csv \
  --data   data/<collection>/<collection>_sample.csv \
  --questions data/<collection>/evaluation/questions.yaml \
  --dataset-name <NAME> --output data/<collection>/runs/<model>_true.yaml
```

`--enrichment false` gives the name+type-only baseline. Translation accuracy (TA) is the fraction
of questions whose generated code executes; visualization accuracy (VA) checks chart type and axes.

## Reproducing the paper

All scripts read the cached run outputs in `data/*/runs/` and print the reported numbers.

| Result | Script |
|---|---|
| Findability retrieval (bare vs enriched, MRR/recall) | `python scripts/retrieval_experiment.py` |
| Retrieval content controls (generic / shuffled / CF long_name) | `python scripts/retrieval_control.py` |
| Per-question McNemar + flip analysis | `python scripts/mcnemar_analysis.py` |
| Samples-in-prompt ablation (three-way) | `bash scripts/run_samples_only.sh` then `python scripts/samples_threeway.py` |
| Extract-then-enrich + cross-enrichment | `bash scripts/run_review_experiments.sh` |
| Strict vs equivalence-relaxed column accuracy | `python scripts/strict_columns.py` |
| Metadata-quality rubric + cosine | `python scripts/brsa1_rate.py`, `scripts/metadata_quality_eval.py` |
| Regenerate descriptions with a given model | `python scripts/enrich_to_csv.py --data <sample.csv> --model <m> --output <out.csv>` |

The `runs/` outputs are included so the analysis scripts reproduce the paper's tables without
re-issuing API calls; re-running the validator/`run_*.sh` scripts requires the API keys above.

## Notes

- Models are referenced by API alias (`gpt-4o-mini`, `gemini-2.5-flash`); exact dated snapshots
  are not recoverable from the alias.
- Only schema, statistics, and a data sample are needed; the large raw source files are not included.
