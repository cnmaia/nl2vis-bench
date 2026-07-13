#!/usr/bin/env bash
# Two robustness controls, same swap-the-schema pattern as run_samples_only.sh
# (no translator code change; --enrichment true reads whatever description column the CSV carries).
#
#  A) EXTRACT-THEN-ENRICH (R2#3, R3#1): BR-Sa1 only, description = embedded CF long_name
#     (zero-LLM). Answers "how much of the gain does free embedded metadata recover?"
#     -> */runs/{model}_cflongname.yaml
#  B) CROSS-ENRICHMENT (R1#4): Gemini queries with GPT-4o-mini-authored descriptions, all 3
#     collections. De-confounds the self-enrichment in the Campinas/Gemini flagship.
#     -> */runs/{model}_crossenrich.yaml
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
[ -f local.env ] && { set -a; source local.env; set +a; }

NEE=data/nee-meteors-8b22c446-8f6e-4b33-8457-35b0f02c780d
PMG=data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185
CAMP=data/lfa-intensive-98f945ab-a1f7-4fd6-bf8d-8781511214e0

run() { # dir schema data name model tag equiv
  local D="$1" SCHEMA="$2" DATA="$3" NAME="$4" M="$5" TAG="$6" EQUIV="$7"
  [ -n "$EQUIV" ] && { mkdir -p data/evaluation; cp "$EQUIV" data/evaluation/semantic_equivalence.yaml; }
  mkdir -p "$D/runs"
  echo ">>> $NAME | $TAG | model=$M"
  python -m nl2vis_bench.validator --model "$M" --enrichment true --runs 5 --detailed-logs \
    --schema "$D/$SCHEMA" --data "$D/$DATA" \
    --questions "$D/evaluation/questions.yaml" --dataset-name "$NAME" \
    --output "$D/runs/${M}_${TAG}.yaml"
  if ls results/detailed_logs/*.jsonl >/dev/null 2>&1; then
    mkdir -p "$D/runs/detailed_logs_${TAG}"
    mv results/detailed_logs/*.jsonl "$D/runs/detailed_logs_${TAG}/"
  fi
}

echo "===== A) EXTRACT-THEN-ENRICH (BR-Sa1, CF long_name as description) ====="
for M in gpt-4o-mini gemini-2.5-flash; do
  run "$NEE" column_metadata_cflongname.csv nee_sample.csv NEE-METEORS "$M" cflongname "$NEE/evaluation/semantic_equivalence.yaml"
done

echo "===== B) CROSS-ENRICHMENT (Gemini queries, GPT-authored descriptions) ====="
run "$NEE"  column_metadata_gpt.csv nee_sample.csv              NEE-METEORS gemini-2.5-flash crossenrich "$NEE/evaluation/semantic_equivalence.yaml"
run "$PMG"  column_metadata_gpt.csv pmg_sample.csv              PMG         gemini-2.5-flash crossenrich "$PMG/evaluation/semantic_equivalence.yaml"
run "$CAMP" column_metadata_gpt.csv campinas_merged_sample.csv  CAMPINAS    gemini-2.5-flash crossenrich "$CAMP/evaluation/semantic_equivalence.yaml"

echo "DONE -> *_cflongname.yaml (BR-Sa1 x2 models) and *_crossenrich.yaml (Gemini x3 collections)"
