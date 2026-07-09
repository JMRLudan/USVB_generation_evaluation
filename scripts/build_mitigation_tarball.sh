#!/usr/bin/env bash
# build_mitigation_tarball.sh — package the 2026-07 mitigation-era inputs +
# outputs (naive-prompt arms, memsum summaries + grades, memcond evals, RAG
# diagnostic) into release tarballs for GitHub Releases.
#
# Output (repo root; all match the gitignored *vb-data-*.tar.gz pattern):
#   usvb-data-mitigation-v1-runs.tar.gz                    results.tsv for all new
#                                                          presets + grades +
#                                                          analysis_rag scores +
#                                                          batch manifests + INTEGRITY
#   usvb-data-mitigation-v1-prompts-mit-nd.tar.gz          ND mit prompt dirs (3)
#   usvb-data-mitigation-v1-prompts-mit-unified-<pos>.tar.gz  one per position (3)
#   usvb-data-mitigation-v1-prompts-memcond.tar.gz         generated/memcond_* (3)
# (unified prompt dirs are packaged one-per-part so each part compresses within
#  constrained-shell time budgets; every part extracts in-place independently)
#
# Each part extracts in-place over a repo clone, independently of the others.
# The draw-0 subset dirs (generated/_subset1of3_*) are NOT packaged — they are
# byte-identical copies of the *_d0_* files in the corresponding mit dirs.
# RAG embedding vectors (analysis_rag/emb_*.jsonl, ~1.2GB) are NOT packaged —
# regenerable via scripts/rag_embedding_analysis.py embed (~$4).
#
# Usage: bash scripts/build_mitigation_tarball.sh [--version v1]

set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="v1"
[[ "${1:-}" == "--version" ]] && VERSION="$2"
PREFIX="usvb-data-mitigation-${VERSION}"

MIT_PRESETS="canon_unified_mit_systop canon_unified_mit_sysbottom canon_unified_mit_querytop \
canon_no_distractor_mit_systop canon_no_distractor_mit_sysbottom canon_no_distractor_mit_querytop"
MEM_PRESETS="memsum_profile memsum_persn memsum_safety \
memsum_profile_safe memsum_persn_safe memsum_safety_safe \
memcond_profile memcond_persn memcond_safety"

echo "▶ MITIGATION_INTEGRITY.json"
python3 - <<PY > MITIGATION_INTEGRITY.json
import csv, sys, json, datetime
from pathlib import Path
csv.field_size_limit(sys.maxsize)
presets = "${MIT_PRESETS} ${MEM_PRESETS}".split()
man = {"version": "${VERSION}",
       "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
       "presets": {}}
for preset in presets:
    base = Path(f"data/runs/{preset}")
    if not base.exists(): continue
    man["presets"][preset] = {}
    for md in sorted(p for p in base.iterdir() if p.is_dir()):
        runs = sorted((d for d in md.iterdir() if (d/"results.tsv").exists()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs: continue
        rows = list(csv.DictReader(open(runs[0]/"results.tsv"), delimiter="\t"))
        n_err = sum(1 for r in rows if (r.get("raw_response") or "").startswith("ERROR"))
        man["presets"][preset][md.name] = {
            "run_id": runs[0].name, "n_total": len(rows), "n_err": n_err}
print(json.dumps(man, indent=2))
PY

echo "▶ staging runs part"
STG="${PREFIX}-runs"; rm -rf "$STG"; mkdir -p "$STG"
for preset in $MIT_PRESETS $MEM_PRESETS; do
    base="data/runs/${preset}"; [[ -d "$base" ]] || continue
    for md in "$base"/*/; do
        latest=$(ls -t "$md" | head -1)
        [[ -f "${md}${latest}/results.tsv" ]] || continue
        mkdir -p "${STG}/${md}${latest}"
        cp "${md}${latest}/results.tsv" "${STG}/${md}${latest}/"
    done
done
mkdir -p "${STG}/analysis_memsum" "${STG}/analysis_rag" "${STG}/batch_manifests"
cp analysis_memsum/grades_*.tsv "${STG}/analysis_memsum/" 2>/dev/null || true
cp analysis_rag/scores_*.tsv analysis_rag/prompts.jsonl analysis_rag/corpus.jsonl \
   "${STG}/analysis_rag/" 2>/dev/null || true
for pat in "mitnv*" "st_*_0708*" "memsum_*" "memcond_*" "judge__*mit_*" "judge__*memcond_*"; do
    cp batch_manifests/${pat}.json "${STG}/batch_manifests/" 2>/dev/null || true
done
cp MITIGATION_INTEGRITY.json "${STG}/"
tar -I 'gzip -1' -cf "${PREFIX}-runs.tar.gz" -C "$STG" .
rm -rf "$STG"

echo "▶ prompts (mit, ND)"
tar -I 'gzip -1' -cf "${PREFIX}-prompts-mit-nd.tar.gz" \
    generated/canon_no_distractor_mit_systop \
    generated/canon_no_distractor_mit_sysbottom \
    generated/canon_no_distractor_mit_querytop

for pos in systop sysbottom querytop; do
    echo "▶ prompts (mit, unified ${pos})"
    tar -I 'gzip -1' -cf "${PREFIX}-prompts-mit-unified-${pos}.tar.gz" \
        "generated/canon_unified_mit_${pos}"
done

echo "▶ prompts (memcond)"
tar -I 'gzip -1' -cf "${PREFIX}-prompts-memcond.tar.gz" \
    generated/memcond_profile generated/memcond_persn generated/memcond_safety

ls -lh ${PREFIX}-*.tar.gz
echo "✓ done — upload all parts to the GitHub release; each extracts in-place."
