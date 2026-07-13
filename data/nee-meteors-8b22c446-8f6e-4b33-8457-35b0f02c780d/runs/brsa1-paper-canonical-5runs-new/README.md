# Canonical paper run — BR-Sa1, 5 runs (datetime-bugfix, "new")

These are the **authoritative result files cited in the paper** (originally BRACIS 2026, now being revised for JCDL 2026). Every numeric value in the paper traces to the files here.

## What this run is
- **Dataset:** NEE-METEORS Amazon, BR-Sa1 flux tower (2002–2011), 3,652 daily records, 81 columns.
- **Questions:** 50 (`data/evaluation/questions.yaml`) — 10 computed, 10 temporal, 10 distribution, 10 correlation, 5 grouping, 5 filtering.
- **Design:** ablation (with / without semantic enrichment) × 2 models (GPT-4o-mini, Gemini 2.5 Flash), **5 independent runs** each.
- **"_new" suffix:** these are the corrected runs after the datetime-parsing fix. The earlier, pre-bugfix single-run numbers (GPT 60%→82%) in the old `BENCHMARK_SUMMARY.md` are **superseded** and must not be used.

## Files
| File | Content |
|---|---|
| `gpt_no_enrichment_5runs_new.yaml` | GPT-4o-mini, no enrichment — TA 0.776±0.0089, VA 0.192±0.011 |
| `gpt_with_enrichment_5runs_new.yaml` | GPT-4o-mini, enriched — TA 0.932±0.0179, VA 0.380±0.020 |
| `gemini_no_enrichment_5runs_new.yaml` | Gemini 2.5 Flash, no enrichment — TA 1.000±0.000, VA 0.440±0.0245 |
| `gemini_with_enrichment_5runs_new.yaml` | Gemini 2.5 Flash, enriched — TA 0.980±0.0141, VA 0.444±0.0261 |
| `detailed_logs/{model}_{True\|False}_run{1..5}.jsonl` | per-question logs (20 files) |
| `paper_verification_report.md` | automated cross-check of all 48 paper values vs. these YAMLs (0 discrepancies) |

## Mapping to paper tables
- **Abstract / Ablation (Table 5) / LLM comparison (Table 6):** the four YAML aggregates above.
- **Per-category (Table 7) and Error analysis (Table 8):** `detailed_logs/gpt-4o-mini_True_run1.jsonl` (representative run 1).
- **t-tests** (GPT TA t(8)=17.4 p<.001; GPT VA t(8)=18.4 p<.001; Gemini VA t(8)=0.25 p=.81): reproduce via `python scripts/analyze_results.py` over these YAMLs.

## Reproduce
```
python -m nl2vis_bench.validator --model gpt-4o-mini     --enrichment false --runs 5 --output results/.../gpt_no_enrichment_5runs_new.yaml
python -m nl2vis_bench.validator --model gpt-4o-mini     --enrichment true  --runs 5 --output results/.../gpt_with_enrichment_5runs_new.yaml
python -m nl2vis_bench.validator --model gemini-2.5-flash --enrichment false --runs 5 --output results/.../gemini_no_enrichment_5runs_new.yaml
python -m nl2vis_bench.validator --model gemini-2.5-flash --enrichment true  --runs 5 --output results/.../gemini_with_enrichment_5runs_new.yaml
```
(Other, non-canonical/exploratory runs remain under `nl2vis-bench/legacy_run/`.)
