#!/usr/bin/env bash
# run_canon.sh — canonical single-model launcher for the USVB benchmark.
#
# Runs both canon presets (canon_no_distractor, canon_unified) for a
# single subject model, in parallel, with the gemini-3-flash judge
# inline. Settings match the inference contract documented in INFERENCE.md.
#
# Usage:
#   bash scripts/run_canon.sh --model <slug>
#   bash scripts/run_canon.sh --model qwen/qwen3.5-9b --reasoning off --model-tag reasoning-off
#   bash scripts/run_canon.sh --model openai/gpt-oss-20b --reasoning default
#
# Flags:
#   --model        OpenRouter slug (required), e.g. qwen/qwen3.5-9b,
#                  openai/gpt-oss-120b, deepseek/deepseek-v4-pro
#   --reasoning    default | off | low
#                  - default = no override; the model uses its
#                    system-default thinking behavior (the setting
#                    used for the canon runs of frontier models)
#                  - off     = inject reasoning.enabled=false (used by
#                    the no-thinking Qwen ladder)
#                  - low     = legacy reasoning.effort=low (kept for
#                    back-compat with earlier OpenAI/Gemini runs)
#   --model-tag    optional suffix appended to the on-disk dir name
#                  (NOT to the API call). Used to keep multiple runs
#                  of the same model under separate viewer dirs (e.g.
#                  reasoning-on vs reasoning-off ablations).
#   --concurrency  per-preset OR concurrency cap (default 100)
#   --max-tokens   per-call max output tokens (default 30000)
#
# Outputs:
#   data/runs/<preset>/<model_dir>/<run_id>/results.tsv
#   pipeline/api_logs/{costs.csv, raw_io.*.csv.gz}
#   logs/<model_dir>_<preset>.log
#
# Cost-aware: relies on the runner's HTTP-402 detection
# (`is_insufficient_credits`) to abort cleanly if account credits run
# out. Per-request prompt-token-cap rejections are skipped, not aborted.

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Defaults ──────────────────────────────────────────────────────────
MODEL=""
REASONING="default"
MODEL_TAG=""
CONCURRENCY=100
MAX_TOKENS=30000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --reasoning)    REASONING="$2"; shift 2 ;;
        --model-tag)    MODEL_TAG="$2"; shift 2 ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --max-tokens)   MAX_TOKENS="$2"; shift 2 ;;
        -h|--help)
            head -45 "$0" | sed 's|^# \?||'
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "✗ --model is required"
    exit 1
fi

case "$REASONING" in
    default|off|low) ;;
    *) echo "✗ --reasoning must be one of: default, off, low"; exit 1 ;;
esac

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MODEL_FS="${MODEL//\//_}"
DIR_TAG="$MODEL_FS"
[[ -n "$MODEL_TAG" ]] && DIR_TAG="${MODEL_FS}-${MODEL_TAG}"

echo "════════════════════════════════════════════════════════════════════════"
echo "▶ run_canon.sh — $MODEL  ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "════════════════════════════════════════════════════════════════════════"
echo "  reasoning:    $REASONING"
echo "  model-tag:    ${MODEL_TAG:-<none>}"
echo "  concurrency:  $CONCURRENCY"
echo "  max-tokens:   $MAX_TOKENS"
echo "  output dir:   data/runs/<preset>/${DIR_TAG}/<run_id>/"
echo

# ── Launch both presets in parallel ───────────────────────────────────
PIDS=()
run_id="$(date +%Y%m%d_%H%M%S)"
[[ -n "$MODEL_TAG" ]] && run_id="${run_id}_${MODEL_TAG}"

for preset in canon_no_distractor canon_unified; do
    logfile="${LOG_DIR}/${DIR_TAG}_${preset}.log"
    nohup python3 pipeline/run.py \
        --prompts-dir "generated/${preset}" \
        --model "$MODEL" \
        --reasoning "$REASONING" \
        ${MODEL_TAG:+--model-tag "$MODEL_TAG"} \
        --concurrency "$CONCURRENCY" \
        --max-tokens "$MAX_TOKENS" \
        --run-id "$run_id" \
        --run \
        >> "$logfile" 2>&1 &
    PIDS+=("$!")
    echo "  ▶ ${preset} → PID $! → $logfile"
    sleep 2
done

echo
echo "  Waiting for ${#PIDS[@]} jobs to finish..."
for pid in "${PIDS[@]}"; do
    wait "$pid" || echo "  PID $pid exited with non-zero status (check log)"
done
echo "  ✓ $MODEL complete at $(date '+%H:%M:%S')"
echo

# ── Per-model summary card ────────────────────────────────────────────
if [[ -f scripts/per_model_card.py ]]; then
    echo "  --- per-model card ---"
    python3 scripts/per_model_card.py --model "$MODEL" \
        ${MODEL_TAG:+--tag "$MODEL_TAG"} || true
fi
