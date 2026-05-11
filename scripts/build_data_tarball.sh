#!/usr/bin/env bash
# build_data_tarball.sh — package the canonical results + prompts into a
# single tarball for GitHub Releases.
#
# Output: lcvb-data-v2.tar.gz at repo root (~244MB).
#
# Contents:
#   data/runs/canon_*/<model>/<run_id>/results.tsv   (all canonical TSVs)
#   generated/canon_*/                                (all rendered prompts)
#   INTEGRITY.json                                    (per-(model,preset) row counts)
#   README.md                                         (extraction instructions)
#
# Reader UX after publishing:
#   git clone <repo>
#   curl -OL <release-url>/lcvb-data-v1.tar.gz
#   tar -xzvf lcvb-data-v1.tar.gz
#   python3 viewer/app.py
#
# Usage:
#   bash scripts/build_data_tarball.sh
#   bash scripts/build_data_tarball.sh --version v1.1
#   bash scripts/build_data_tarball.sh --dry-run   # show what would be packed

set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="v2"
DRY=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help)
            head -25 "$0" | sed 's|^# \?||'
            exit 0 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

OUT="lcvb-data-${VERSION}.tar.gz"
STAGING="lcvb-data-${VERSION}"

echo "▶ Building ${OUT}"

# ── Generate INTEGRITY.json ───────────────────────────────────────────
echo "  Generating INTEGRITY.json..."
python3 - <<PY > /tmp/INTEGRITY.json
import csv, sys, json
from pathlib import Path
csv.field_size_limit(sys.maxsize)

PRESETS_TARGET = {"canon_direct": 2122, "canon_no_distractor": 2122, "canon_unified": 6366}

manifest = {
    "version": "${VERSION}",
    "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "presets": {},
    "models": {},
}

for preset, target in PRESETS_TARGET.items():
    base = Path(f"data/runs/{preset}")
    if not base.exists(): continue
    manifest["presets"][preset] = {"expected_rows_per_model": target, "models": {}}
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir(): continue
        latest = sorted([d for d in model_dir.iterdir() if d.is_dir() and (d/"results.tsv").exists()],
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if not latest: continue
        run = latest[0]
        with open(run/"results.tsv") as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        n_total = len(rows)
        n_err   = sum(1 for r in rows if (r.get("raw_response","") or "").startswith("ERROR"))
        n_pe    = sum(1 for r in rows if str(r.get("parse_error","")).strip() in ("1","True","true"))
        n_ok    = n_total - n_err - n_pe
        manifest["presets"][preset]["models"][model_dir.name] = {
            "run_id": run.name,
            "n_total": n_total, "n_ok": n_ok,
            "n_err": n_err, "n_pe": n_pe,
            "completeness_pct": round(100 * n_ok / target, 2),
        }
        m = manifest["models"].setdefault(model_dir.name, {"presets": []})
        m["presets"].append(preset)

print(json.dumps(manifest, indent=2))
PY
echo "  ✓ INTEGRITY.json generated ($(wc -c < /tmp/INTEGRITY.json) bytes)"

# ── Stage the tarball contents ────────────────────────────────────────
if [[ -n "$DRY" ]]; then
    echo "[dry-run] Would stage:"
    echo "  - data/runs/canon_*/...results.tsv (only the most-recent run_id per (preset, model))"
    echo "  - generated/canon_*/*.json"
    echo "  - INTEGRITY.json"
    echo "  - README.md (tarball-internal)"
    echo "  - lcvb-data-${VERSION}.tar.gz at repo root"
    echo "Total inputs:"
    find data/runs -name "results.tsv" 2>/dev/null | wc -l | xargs echo "  results.tsv files:"
    find generated -name "*.json" 2>/dev/null | wc -l | xargs echo "  prompt JSONs:"
    exit 0
fi

echo "  Staging contents..."
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Copy results.tsv files (only the latest run_id per model_dir)
for preset_dir in data/runs/canon_direct data/runs/canon_no_distractor data/runs/canon_unified; do
    [[ -d "$preset_dir" ]] || continue
    for model_dir in "$preset_dir"/*/; do
        [[ -d "$model_dir" ]] || continue
        latest=$(ls -t "$model_dir" | head -1)
        if [[ -f "${model_dir}${latest}/results.tsv" ]]; then
            target_dir="${STAGING}/${model_dir}${latest}"
            mkdir -p "$target_dir"
            cp "${model_dir}${latest}/results.tsv" "$target_dir/"
            # Also copy meta.json + checkpoint.txt if present
            for aux in meta.json checkpoint.txt; do
                [[ -f "${model_dir}${latest}/${aux}" ]] && cp "${model_dir}${latest}/${aux}" "$target_dir/"
            done
        fi
    done
done

# Copy the rendered prompts. The -L flag dereferences symlinks, which
# matters because canon_direct and canon_no_distractor in this repo are
# symlinks pointing at directories outside the tracked tree. Without -L
# the tarball would ship dangling symlinks and consumers would see no
# prompt files for those two presets.
mkdir -p "${STAGING}/generated"
cp -rL generated/canon_direct generated/canon_no_distractor generated/canon_unified "${STAGING}/generated/" 2>/dev/null || true

# Manifest + tarball-internal README
cp /tmp/INTEGRITY.json "${STAGING}/INTEGRITY.json"

cat > "${STAGING}/README.md" <<EOF
# LCVB data archive ${VERSION}

This tarball contains the canonical LCVB results across the full
model roster, plus the rendered prompt sets used to produce them.
Extracted in-place over a clone of
\`JMRLudan/LCVB_generation_evaluation\`, it provides:

- \`data/runs/canon_<preset>/<model>/<run_id>/results.tsv\` — per-row judged outputs
- \`generated/canon_<preset>/*.json\` — the prompt files (system + user message + metadata)
- \`INTEGRITY.json\` — per-(model, preset) row counts and error tallies

## Quickstart

\`\`\`bash
git clone https://github.com/JMRLudan/LCVB_generation_evaluation.git
cd LCVB_generation_evaluation
tar -xzvf /path/to/lcvb-data-${VERSION}.tar.gz   # extracts in-place
python3 viewer/app.py
\`\`\`

The viewer's Frontier tab includes a "Baseline vs vigilance" chart
that places every model in the roster on a single grouped-bar
display, alongside per-preset overall and per-variant metrics in the
Charts and Scenarios tabs.

## Provenance

See \`INFERENCE.md\` in the code repo for the exact API parameters used for
every model. Aggregate metrics are scenario-macro-averaged per
\`SCORING.md\`.

## License

CC-BY-4.0 for the result TSVs and rendered prompts in this tarball.
The underlying scenarios + distractor pool license is documented in the
code repo (\`data/distractors/LICENSE\`).
EOF

echo "  ✓ Staging directory built"

# ── Build the tarball ─────────────────────────────────────────────────
echo "  Compressing → $OUT..."
tar -czf "$OUT" "$STAGING"
SIZE=$(du -h "$OUT" | cut -f1)
echo "  ✓ Built $OUT ($SIZE)"

# Cleanup staging
rm -rf "$STAGING"

# ── Summary ───────────────────────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════════════"
echo "▶ Tarball ready: $OUT ($SIZE)"
echo "════════════════════════════════════════════════════════════════════════"
echo
echo "To publish:"
echo "  1. Tag the release: git tag ${VERSION} && git push origin ${VERSION}"
echo "  2. Attach $OUT to the GitHub release at:"
echo "     https://github.com/JMRLudan/LCVB_generation_evaluation/releases/new"
echo "  3. Update README.md's curl URL if the slug changed"
