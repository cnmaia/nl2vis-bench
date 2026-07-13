#!/usr/bin/env bash
# Run the full ablation matrix (2 models x enrichment off/on x 5 runs) for one dataset.
#
# Usage:
#   bash scripts/run_eval_matrix.sh <dataset_dir> <data_filename> <DATASET_NAME> [equivalence_yaml]
#
# Example (PMG):
#   bash scripts/run_eval_matrix.sh \
#     data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185 pmg_sample.csv PMG \
#     data/subsurface-e32f5fe6-f6f5-4675-82fc-b93ac50ac185/evaluation/semantic_equivalence.yaml
#
# Reads API keys from ./local.env automatically. Outputs to <dataset_dir>/runs/<model>_<enrichment>.yaml
set -euo pipefail

DIR="$1"; DATA="$2"; NAME="$3"; EQUIV="${4:-}"
SCHEMA="$DIR/column_metadata.csv"
DATA_PATH="$DIR/$DATA"
QUESTIONS="$DIR/evaluation/questions.yaml"

[ -f "$SCHEMA" ]    || { echo "missing schema: $SCHEMA"; exit 1; }
[ -f "$DATA_PATH" ] || { echo "missing data: $DATA_PATH"; exit 1; }
[ -f "$QUESTIONS" ] || { echo "missing questions: $QUESTIONS"; exit 1; }

# Load API keys
if [ -f local.env ]; then set -a; source local.env; set +a; fi

# Activate this dataset's semantic-equivalence at the path the validator reads
if [ -n "$EQUIV" ]; then
  mkdir -p data/evaluation
  cp "$EQUIV" data/evaluation/semantic_equivalence.yaml
  echo "equivalence: $EQUIV -> data/evaluation/semantic_equivalence.yaml"
fi

mkdir -p "$DIR/runs"
for E in false true; do
  for M in gpt-4o-mini gemini-2.5-flash; do
    echo ">>> $NAME | model=$M | enrichment=$E"
    python -m nl2vis_bench.validator --model "$M" --enrichment "$E" --runs 5 --detailed-logs \
      --schema "$SCHEMA" --data "$DATA_PATH" --questions "$QUESTIONS" \
      --dataset-name "$NAME" --output "$DIR/runs/${M}_${E}.yaml"
  done
done
# The validator writes per-question logs to a shared results/detailed_logs/; relocate
# them into this dataset's folder so a later dataset's run cannot overwrite them.
if ls results/detailed_logs/*.jsonl >/dev/null 2>&1; then
  mkdir -p "$DIR/runs/detailed_logs"
  mv results/detailed_logs/*.jsonl "$DIR/runs/detailed_logs/"
fi
echo "DONE: $NAME -> $DIR/runs/"
