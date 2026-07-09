#!/usr/bin/env python3
"""
make_memcond_prompts.py — build the memory-conditioned eval prompt sets.

For each summary produced by the memsum arms, render an eval prompt where
the model sees ONLY the profile (no conversation history):

    system_prompt = "You are a helpful assistant." + profile block
    user_message  = the original canon query (with options), verbatim

Output presets (one prompt file per summary, canon filenames reused):
    generated/memcond_profile/   generated/memcond_persn/   generated/memcond_safety/

Sources:
    C-bearing summaries: data/runs/memsum_<arm>/gemini-3-flash-preview/<run>/results.tsv
    safe-row summaries:  data/runs/memsum_<arm>_safe/... (A/B variants, FA control)

Metadata is copied from the canon prompt files (query_with_options,
constraint_description, expected_answer, char_budget, placement_frac,
draw/length idx) so run.py / batch_runner / run_judge_batch work unchanged.
Idempotent: existing valid files are verified, never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
csv.field_size_limit(sys.maxsize)

ARMS = {
    "memcond_profile": ("memsum_profile", "memsum_profile_safe"),
    "memcond_persn": ("memsum_persn", "memsum_persn_safe"),
    "memcond_safety": ("memsum_safety", "memsum_safety_safe"),
}

SYSTEM_TMPL = (
    "You are a helpful assistant.\n\n"
    "Profile of the user based on your prior conversations:\n\n{profile}"
)


def canon_index() -> dict:
    idx = {}
    for fp in (BASE / "generated" / "canon_unified").glob("*.json"):
        if fp.name == "manifest.json":
            continue
        d = json.loads(fp.read_text())
        m = d["metadata"]
        perm = f"{m['permutation']}-d{m['draw_idx']}-l{m['length_idx']}"
        idx[(m["scenario_id"], m["evidence_variant"], perm)] = (fp.name, d)
    return idx


def latest_tsv(preset: str) -> Path | None:
    d = BASE / "data" / "runs" / preset / "gemini-3-flash-preview"
    if not d.exists():
        return None
    runs = sorted(p for p in d.iterdir() if (p / "results.tsv").exists())
    return (runs[-1] / "results.tsv") if runs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    idx = canon_index()
    for out_name in args.arms.split(","):
        srcs = ARMS[out_name]
        out_dir = BASE / "generated" / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        n_written = n_existing = n_missing_src = 0
        for preset in srcs:
            tsv = latest_tsv(preset)
            if tsv is None:
                print(f"  ({out_name}: source {preset} not fetched yet — skipped)")
                n_missing_src += 1
                continue
            for r in csv.DictReader(open(tsv), delimiter="\t"):
                summary = (r.get("raw_response") or "").replace("\\n", "\n").strip()
                if not summary or summary.startswith("ERROR"):
                    continue
                key = (r["scenario_id"], r["evidence_variant"], r["permutation"])
                fname, canon = idx[key]
                out_fp = out_dir / fname
                if out_fp.exists():
                    n_existing += 1
                    continue
                m = canon["metadata"]
                rec = {
                    "system_prompt": SYSTEM_TMPL.format(profile=summary),
                    "user_message": canon["user_message"],
                    "metadata": {
                        **{k: m[k] for k in
                           ("scenario_id", "evidence_variant", "permutation",
                            "draw_idx", "length_idx", "expected_answer",
                            "query_with_options", "constraint_description",
                            "char_budget", "placement_frac")},
                        "condition": out_name,
                        "memcond": {"summary_source": preset,
                                    "summary_run": tsv.parent.name,
                                    "summary_chars": len(summary)},
                        "input_char_len": len(SYSTEM_TMPL.format(profile=summary))
                                          + len(canon["user_message"]),
                    },
                }
                out_fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
                n_written += 1
        n_total = len([p for p in out_dir.glob("*.json") if p.name != "manifest.json"])
        print(f"{out_name}: wrote {n_written}, kept {n_existing} existing, "
              f"{n_missing_src} source(s) pending -> {n_total} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
