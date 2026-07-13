#!/usr/bin/env bash
# Samples-in-prompt ablation: query model sees name+type+range+sample values
# but NO natural-language description. Uses column_metadata_samplesonly.csv (description field
# replaced by raw samples/range) with --enrichment true, so no translator code changes.
# Compare its TA to the existing no-enrichment (*_false) and enriched (*_true) runs.
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -f local.env ] && { set -a; source local.env; set +a; }

run() {
  local D="$1" DATA="$2" NAME="$3" EQUIV="$4"
  [ -n "$EQUIV" ] && { mkdir -p data/evaluation; cp "$EQUIV" data/evaluation/semantic_equivalence.yaml; }
  mkdir -p "$D/runs"
  for M in gpt-4o-mini gemini-2.5-flash; do
    echo ">>> $NAME samples-only | model=$M"
    python -m nl2vis_bench.validator --model "$M" --enrichment true --runs 5 --detailed-logs \
      --schema "$D/column_metadata_samplesonly.csv" --data "$D/$DATA" \
      --questions "$D/evaluation/questions.yaml" --dataset-name "$NAME" \
      --output "$D/runs/${M}_samplesonly.yaml"
  done
  if ls results/detailed_logs/*.jsonl >/dev/null 2>&1; then
    mkdir -p "$D/runs/detailed_logs_samplesonly"
    mv results/detailed_logs/*.jsonl "$D/runs/detailed_logs_samplesonly/"
  fi
}

NEE=data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d
PMG=data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185
CAMP=data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0
run "$NEE"  nee_sample.csv              NEE-METEORS "$NEE/evaluation/semantic_equivalence.yaml"
run "$PMG"  pmg_sample.csv              PMG         "$PMG/evaluation/semantic_equivalence.yaml"
run "$CAMP" campinas_merged_sample.csv  CAMPINAS    "$CAMP/evaluation/semantic_equivalence.yaml"
echo "DONE -> */runs/{gpt-4o-mini,gemini-2.5-flash}_samplesonly.yaml"
