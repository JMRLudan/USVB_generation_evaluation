#!/usr/bin/env python3
"""
per_model_card.py — quick per-model headline metrics card, one row per
preset. Useful immediately after a run lands to verify it's healthy.

Outputs to stdout AND appends a single JSON line to
`analysis/per_model_cards.jsonl` so we can grep history later.

Usage:
    python3 scripts/per_model_card.py --model qwen/qwen3.5-9b --tag reasoning-off
    python3 scripts/per_model_card.py --model openai/gpt-oss-120b
    python3 scripts/per_model_card.py --all   # iterate every model dir
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

csv.field_size_limit(sys.maxsize)

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "data" / "runs"
JOURNAL_PATH = REPO / "analysis" / "per_model_cards.jsonl"

PRESETS = ["canon_no_distractor", "canon_unified"]


def _is_yes(v): return str(v).strip().upper() == "YES"
def _is_true(v): return str(v) in ("1", "True", "true")


def metrics_for(tsv: Path) -> dict | None:
    if not tsv.exists():
        return None
    with open(tsv) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows: return None

    n_total = len(rows)
    n_err = sum(1 for r in rows if (r.get("raw_response","") or "").startswith("ERROR"))
    n_pe  = sum(1 for r in rows if str(r.get("parse_error","")).strip() in ("1","True","true"))
    n_ok  = n_total - n_err - n_pe

    def pct(num, den):
        return round(100 * num / den, 2) if den else None

    def _sr_value(r):
        # Prefer the v2 column; fall back to the legacy "vigilance" for
        # rows written before the rename.
        return r.get("vigilance_refuse_only") or r.get("vigilance")

    sr  = sum(1 for r in rows
              if _is_true(_sr_value(r))
              and not (r.get("raw_response","") or "").startswith("ERROR"))
    cm_yes = sum(1 for r in rows if _is_yes(r.get("constraint_mentioned")))
    cm_set = sum(1 for r in rows if str(r.get("constraint_mentioned","")).upper() in ("YES","NO"))
    sm_yes = sum(1 for r in rows if _is_yes(r.get("sufficiently_modified")))
    sm_set = sum(1 for r in rows if str(r.get("sufficiently_modified","")).upper() in ("YES","NO"))
    ab  = sum(1 for r in rows if _is_true(r.get("abstained")))

    # Wilson 95% CI for SR
    import math
    if n_ok:
        p = sr / n_ok
        z = 1.96; n = n_ok
        denom = 1 + z*z/n
        center = (p + z*z/(2*n)) / denom
        margin = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
        ci = (round(100*(center - margin), 1), round(100*(center + margin), 1))
    else:
        ci = (None, None)

    return {
        "n_total": n_total, "n_ok": n_ok, "n_err": n_err, "n_pe": n_pe,
        "SR": pct(sr, n_ok),
        "SR_ci95_lo": ci[0], "SR_ci95_hi": ci[1],
        "CM": pct(cm_yes, cm_set),
        "SM": pct(sm_yes, sm_set) if sm_set else None,
        "abstain": pct(ab, n_ok),
    }


def card_for_model_dir(model_dir: str) -> dict:
    """model_dir is the on-disk dir name like `qwen_qwen3.5-9b-reasoning-off`."""
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_dir": model_dir,
        "presets": {},
    }
    for preset in PRESETS:
        d = RUNS_DIR / preset / model_dir
        if not d.exists(): continue
        # Pick the most recent run_id
        runs = sorted([p for p in d.iterdir() if p.is_dir() and (p/"results.tsv").exists()],
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs: continue
        run = runs[0]
        m = metrics_for(run / "results.tsv")
        if m is None: continue
        out["presets"][preset] = {**m, "run_id": run.name}
    # Vigilance gap
    u = out["presets"].get("canon_unified", {}).get("SR")
    n = out["presets"].get("canon_no_distractor", {}).get("SR")
    out["vigilance_gap_no_dist_vs_unified"] = round(n - u, 2) if (n is not None and u is not None) else None
    return out


def print_card(card: dict):
    print(f"\n{'='*70}")
    print(f"  {card['model_dir']}  @ {card['timestamp']}")
    print(f"{'='*70}")
    print(f"  {'PRESET':<22} {'N':>6} {'SR%':>7} {'CI95':>14} {'CM%':>6} {'SM%':>6} {'AB%':>6}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*14} {'-'*6} {'-'*6} {'-'*6}")
    for preset in PRESETS:
        m = card["presets"].get(preset)
        if not m:
            print(f"  {preset:<22} (no data)")
            continue
        ci = f"[{m['SR_ci95_lo']:.1f}, {m['SR_ci95_hi']:.1f}]" if m['SR_ci95_lo'] is not None else "—"
        sm = f"{m['SM']:.1f}" if m['SM'] is not None else "—"
        print(f"  {preset:<22} {m['n_ok']:>6} {m['SR']:>6.2f}% {ci:>14} {m['CM']:>5.1f}% {sm:>5}% {m['abstain']:>5.1f}%")
    if card['vigilance_gap_no_dist_vs_unified'] is not None:
        print(f"  Vigilance gap (no_dist − unified):  {card['vigilance_gap_no_dist_vs_unified']:+.2f} pp")


def append_jsonl(card: dict):
    JOURNAL_PATH.parent.mkdir(exist_ok=True)
    with JOURNAL_PATH.open("a") as f:
        f.write(json.dumps(card) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="OR-style slug like qwen/qwen3.5-9b or openai/gpt-oss-120b")
    p.add_argument("--tag",   default="", help="model_tag suffix (e.g. reasoning-off)")
    p.add_argument("--model-dir", help="Direct model_dir name (alternative to --model + --tag)")
    p.add_argument("--all", action="store_true", help="Iterate every model dir under data/runs/canon_unified/")
    p.add_argument("--no-jsonl", action="store_true", help="Skip the jsonl history append")
    args = p.parse_args()

    targets: list[str] = []
    if args.all:
        seen = set()
        for preset in PRESETS:
            d = RUNS_DIR / preset
            if d.exists():
                for md in d.iterdir():
                    if md.is_dir() and md.name not in seen:
                        seen.add(md.name)
                        targets.append(md.name)
        targets.sort()
    elif args.model_dir:
        targets = [args.model_dir]
    elif args.model:
        md = args.model.replace("/", "_")
        if args.tag:
            md = f"{md}-{args.tag}"
        targets = [md]
    else:
        p.error("Must pass --model, --model-dir, or --all")

    for md in targets:
        card = card_for_model_dir(md)
        print_card(card)
        if not args.no_jsonl:
            append_jsonl(card)


if __name__ == "__main__":
    main()
