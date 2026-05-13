#!/usr/bin/env bash
# status.sh — auto-discover progress for any in-flight or completed
# canon runs. Iterates every model_dir under data/runs/canon_*/ and
# reports row counts vs expected sizes.
#
# Usage:
#   bash scripts/status.sh            # one-shot snapshot
#   bash scripts/status.sh --loop     # auto-refresh every 15s
#   bash scripts/status.sh --loop 5   # custom interval

set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--loop" ]]; then
    interval="${2:-15}"
    while true; do
        clear
        "$0"
        sleep "$interval"
    done
fi

PRESETS=(canon_no_distractor:2122 canon_unified:6366)

printf "USVB run status — %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
printf "%-2s %-32s %-22s %12s   %s\n" "" "MODEL_DIR" "PRESET" "ROWS / TOTAL" "PROGRESS"
printf -- "----------------------------------------------------------------------------------------------------\n"

TOTAL_DONE=0
TOTAL_TARGET=0

for entry in "${PRESETS[@]}"; do
    preset="${entry%:*}"
    target="${entry#*:}"
    base="data/runs/${preset}"
    [[ -d "$base" ]] || continue
    for model_dir in "$base"/*/; do
        [[ -d "$model_dir" ]] || continue
        name=$(basename "$model_dir")
        latest_tsv=$(ls -t "$model_dir"/*/results.tsv 2>/dev/null | head -1)
        if [[ -n "$latest_tsv" && -f "$latest_tsv" ]]; then
            n=$(($(wc -l < "$latest_tsv" 2>/dev/null) - 1))
            (( n < 0 )) && n=0
        else
            n=0
        fi
        pct=$(awk "BEGIN {printf \"%.1f\", $n*100/$target}")
        filled=$(awk "BEGIN {printf \"%d\", $n*30/$target}")
        (( filled > 30 )) && filled=30
        bar=$(printf "%${filled}s" "" | tr ' ' '#')
        empty=$(( 30 - filled ))
        bar="${bar}$(printf "%${empty}s" "" | tr ' ' '.')"
        mark=" "
        (( n >= target )) && mark="OK"
        printf "%-2s %-32s %-22s %5d / %5d   [%s] %5s%%\n" "$mark" "$name" "$preset" "$n" "$target" "$bar" "$pct"
        TOTAL_DONE=$(( TOTAL_DONE + n ))
        TOTAL_TARGET=$(( TOTAL_TARGET + target ))
    done
done

printf -- "----------------------------------------------------------------------------------------------------\n"
if (( TOTAL_TARGET > 0 )); then
    overall_pct=$(awk "BEGIN {printf \"%.1f\", $TOTAL_DONE*100/$TOTAL_TARGET}")
else
    overall_pct="0.0"
fi
printf "TOTAL %-52s %5d / %5d              %5s%%\n" "" "$TOTAL_DONE" "$TOTAL_TARGET" "$overall_pct"

# Process count
alive=$(pgrep -f "pipeline/run.py.*--model " 2>/dev/null | wc -l | tr -d ' ' || echo 0)
printf "\nalive runner processes: %s\n" "$alive"
