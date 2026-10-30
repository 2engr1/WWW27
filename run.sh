#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# Modular Dense / NS-Retrieval experiment runner
#
# Fixed protocol:
#   Dense retrieval : Top-1000
#   NS filtering    : Top-50
#   Saved ranking  : Top-50
#   Generation     : Top-1
#   Evaluation     : Recall / MRR / nDCG @5, @10, @20, @50
# ============================================================

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON="${PYTHON:-python}"
DEVICE="${1:-cuda:6}"

# Project entry points
MAIN="$PROJECT_DIR/main.py"
INDEX="$PROJECT_DIR/index.py"
EVALUATE="$PROJECT_DIR/evaluate.py"
SUMMARIZE="$PROJECT_DIR/summarize_ns_metrics.py"

# Existing prepared datasets and local models
DATA_ROOT="/hybrid_data_chunk"
MODEL_ROOT="/ns-retrieval"

EMBED_MODEL="$MODEL_ROOT/e5-base-v2"
RERANK_MODEL="$MODEL_ROOT/ms-marco-MiniLM-L-6-v2"

# Six dataset scales
SCALES=(20000 50000 100000 200000 400000 full)

# Five NS experiment modes
MODES=(
  ns
  ns-ce
  ns-retrieval
  ns-suf
  ns-nec
)

# Fixed retrieval protocol: do not change these values independently.
CANDIDATES=1000
NS_CANDIDATES=50
SAVE_TOP_K=50
GENERATION_TOP_K=1

# Model and indexing batch sizes
ENCODE_BATCH_SIZE=64
UPLOAD_BATCH_SIZE=256
TOKEN_BATCH_SIZE=16
RERANK_BATCH_SIZE=32

# NS-Rerank local evidence window settings
WINDOW_TOKENS=256
WINDOW_STRIDE=128

# Optional LLM generation / judging.
# Retrieval context is always selected from the final Top-1.
# Set RUN_GENERATION=1 to actually call the LLM.
# Set RUN_JUDGE=1 to run judging; this also requires RUN_GENERATION=1.
RUN_GENERATION="${RUN_GENERATION:-0}"
RUN_JUDGE="${RUN_JUDGE:-0}"
LLM_MODEL="${LLM_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

if [[ "$RUN_GENERATION" == "1" && "$RUN_JUDGE" == "0" ]]; then
  GENERATION_FLAGS=(--generate)
elif [[ "$RUN_GENERATION" == "1" && "$RUN_JUDGE" == "1" ]]; then
  GENERATION_FLAGS=(--generate --judge --llm-model "$LLM_MODEL" --judge-model "$JUDGE_MODEL")
elif [[ "$RUN_GENERATION" == "0" && "$RUN_JUDGE" == "0" ]]; then
  GENERATION_FLAGS=()
else
  echo "Invalid options: RUN_JUDGE=1 requires RUN_GENERATION=1." >&2
  exit 1
fi

# Unique output directory for this experiment run.
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
OUT_ROOT="$PROJECT_DIR/outputs/$RUN_ID"
mkdir -p "$OUT_ROOT"

# ------------------------------------------------------------
# Preflight checks
# ------------------------------------------------------------

for path in "$MAIN" "$INDEX" "$EVALUATE" "$SUMMARIZE" "$EMBED_MODEL" "$RERANK_MODEL"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing file or directory: $path" >&2
    exit 1
  fi
done

for scale in "${SCALES[@]}"; do
  dataset="$DATA_ROOT/$scale"

  for filename in dataset.json corpus.jsonl queries.jsonl id_map.jsonl; do
    path="$dataset/$filename"
    if [[ ! -f "$path" ]]; then
      echo "Missing prepared dataset file: $path" >&2
      exit 1
    fi
  done
done

echo "============================================================"
echo "Project       : $PROJECT_DIR"
echo "Python        : $PYTHON"
echo "Device        : $DEVICE"
echo "Data root     : $DATA_ROOT"
echo "Embedding     : $EMBED_MODEL"
echo "Reranker      : $RERANK_MODEL"
echo "Output root   : $OUT_ROOT"
echo "Modes         : ${MODES[*]}"
echo "Dense Top-K   : $CANDIDATES"
echo "NS Top-K      : $NS_CANDIDATES"
echo "Saved Top-K   : $SAVE_TOP_K"
echo "Generation K  : $GENERATION_TOP_K"
echo "Generation    : $RUN_GENERATION"
echo "Judge         : $RUN_JUDGE"
echo "============================================================"

TOTAL=$(( ${#SCALES[@]} * ${#MODES[@]} ))
CURRENT=0

# ------------------------------------------------------------
# Main experiment loop
# ------------------------------------------------------------

for scale in "${SCALES[@]}"; do
  dataset="$DATA_ROOT/$scale"
  scale_out="$OUT_ROOT/$scale"
  mkdir -p "$scale_out"

  echo
  echo "========== Scale $scale: build / reuse E5 + Qdrant index =========="
  echo "Start time: $(date '+%F %T')"

  "$PYTHON" -u "$INDEX" index \
    --dataset "$dataset" \
    --backend dense \
    --embed-model "$EMBED_MODEL" \
    --device "$DEVICE" \
    --encode-batch-size "$ENCODE_BATCH_SIZE" \
    --upload-batch-size "$UPLOAD_BATCH_SIZE" \
    2>&1 | tee "$scale_out/index.log"

  echo "Index stage completed: $(date '+%F %T')"

  for mode in "${MODES[@]}"; do
    CURRENT=$((CURRENT + 1))

    raw_out="$scale_out/$mode.jsonl"
    eval_log="$scale_out/$mode.evaluate.log"

    echo
    echo "========== [$CURRENT/$TOTAL] Scale $scale / Mode $mode =========="
    echo "Start time: $(date '+%F %T')"

    "$PYTHON" -u "$MAIN" run \
      --dataset "$dataset" \
      --mode "$mode" \
      --device "$DEVICE" \
      --embed-model "$EMBED_MODEL" \
      --rerank-model "$RERANK_MODEL" \
      --candidates "$CANDIDATES" \
      --ns-candidates "$NS_CANDIDATES" \
      --save-top-k "$SAVE_TOP_K" \
      --generation-top-k "$GENERATION_TOP_K" \
      --token-batch-size "$TOKEN_BATCH_SIZE" \
      --rerank-batch-size "$RERANK_BATCH_SIZE" \
      --window-tokens "$WINDOW_TOKENS" \
      --window-stride "$WINDOW_STRIDE" \
      "${GENERATION_FLAGS[@]}" \
      --out "$raw_out" \
      2>&1 | tee "$scale_out/$mode.log"

    echo "Retrieval completed: $(date '+%F %T')"

    # Evaluate each run immediately. This creates:
    #   MODE.jsonl.evaluated.jsonl
    #   MODE.jsonl.summary.json
    "$PYTHON" -u "$EVALUATE" \
      --input "$raw_out" \
      2>&1 | tee "$eval_log"

    echo "Evaluation completed: $(date '+%F %T')"
  done
done

# ------------------------------------------------------------
# Aggregate the five NS modes across the six scales.
# Dense and Dense+CE summaries remain available in each scale folder.
# ------------------------------------------------------------

SUMMARY_OUT="$OUT_ROOT/summary_tables"

echo
echo "========== Summarize five NS modes across six scales =========="
"$PYTHON" -u "$SUMMARIZE" \
  --root "$OUT_ROOT" \
  --out "$SUMMARY_OUT" \
  2>&1 | tee "$OUT_ROOT/summarize.log"

echo
echo "============================================================"
echo "All $TOTAL retrieval runs and evaluations completed."
echo "Experiment output : $OUT_ROOT"
echo "NS summary tables : $SUMMARY_OUT"
echo "Finished          : $(date '+%F %T')"
echo "============================================================"
