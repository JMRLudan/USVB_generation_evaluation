"""
run_canon_haiku_thinking.py — Claude Haiku 4.5 canon with extended_thinking ON.

Recovered + adapted from `pipeline/run_canon_sonnet_thinking.py` at commit
fc21e35 (deleted in 66dc4ac during the v2 judge-schema migration). Mirrors
the Sonnet 4.6 thinking-on ablation that already appears in the paper as
the second paired contrast in Appendix I Table 8.

Settings — match Sonnet 4.6 thinking-on exactly so Haiku ↔ Sonnet is a
controlled within-family pair at a fixed thinking budget:
  - Model: claude-haiku-4-5-20251001
  - thinking={"type": "enabled", "budget_tokens": 4096}
  - max_tokens: 10000
  - temperature: 1.0
  - Submit via Anthropic Messages Batches API (50% batch discount).

Lifecycle:
    submit   — build canon requests + Anthropic batch upload (this script)
    status   — poll Anthropic batch state
    fetch    — pull subject results, write a results.tsv with judge
               fields blank.
    judge    — DISABLED (deleted with batch_judge_gemini in 66dc4ac).
               Route results through the current main-pipeline judge
               instead — JOURNAL 2026-05-18 has the route-around plan.
    finalize — DISABLED for the same reason.

Layout:
    Subject runs land at
    data/runs/<preset>/claude-haiku-4-5-thinking/<run_id>/results.tsv
    so the viewer surfaces them as a separate "model" alongside the
    existing claude-haiku-4-5-20251001 default-mode rows.

Usage:
    python3 -m analysis_ablation.run_canon_haiku_thinking submit canon_no_distractor
    python3 -m analysis_ablation.run_canon_haiku_thinking submit canon_unified
    python3 -m analysis_ablation.run_canon_haiku_thinking status canon_unified
    python3 -m analysis_ablation.run_canon_haiku_thinking fetch canon_unified

Defaults:
    --model     claude-haiku-4-5-20251001
    --budget    4096
    --run-id    canon_haiku_thinking_<UTC_TIMESTAMP>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

# Lazy-import judge-related symbols inside cmd_judge / cmd_finalize; those
# modules (batch_judge_gemini, JUDGE_SYSTEM_WITH_ANALYSIS) were deleted in
# commit 66dc4ac and the judge phase is non-functional here. Submit, status,
# and fetch don't need them.
from openrouter_client import _load_dotenv                                  # noqa: E402
from pipeline.batch_common import (                                         # noqa: E402
    BatchRequest, chunk_requests, make_custom_id, parse_custom_id,
)
from pipeline.batch_anthropic import (                                      # noqa: E402
    AnthropicBatchAdapter, _request_to_anthropic_dict,
    build_requests_from_prompts,
)
from pipeline.batch_gemini import GeminiBatchAdapter                        # noqa: E402
from pipeline.batch_runner import write_results_tsv                         # noqa: E402

csv.field_size_limit(sys.maxsize)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
SUBJECT_MODEL = "claude-haiku-4-5-20251001"
SUBJECT_DIR_NAME = "claude-haiku-4-5-thinking"
JUDGE_MODEL = "google/gemini-3-flash-preview"
DEFAULT_BUDGET = 4096
DEFAULT_MAX_TOKENS = 10000

PROMPTS_ROOT = REPO / "generated"
RUNS_ROOT = REPO / "data" / "runs"
MANIFESTS_DIR = REPO / "batch_manifests"

VALID_PRESETS = ("canon_direct", "canon_no_distractor", "canon_unified")


# ──────────────────────────────────────────────────────────────────────
# Manifest helpers
# ──────────────────────────────────────────────────────────────────────
def manifest_path(preset: str, kind: str) -> Path:
    """kind is 'subject' or 'judge'. One manifest per (preset, kind)."""
    return MANIFESTS_DIR / f"haiku_thinking__{preset}__{kind}.json"


def load_manifest(preset: str, kind: str) -> dict:
    p = manifest_path(preset, kind)
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(preset: str, kind: str, data: dict) -> None:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    p = manifest_path(preset, kind)
    p.write_text(json.dumps(data, indent=2, default=str))


def get_or_create_run_id(preset: str) -> str:
    sub = load_manifest(preset, "subject")
    if sub.get("run_id"):
        return sub["run_id"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"canon_haiku_thinking_{ts}"


# ──────────────────────────────────────────────────────────────────────
# Subject submit
# ──────────────────────────────────────────────────────────────────────
def cmd_submit(preset: str, budget: int, max_chunk_mb: int, chunk_index: int | None = None) -> int:
    if preset not in VALID_PRESETS:
        print(f"ERROR: preset must be one of {VALID_PRESETS}, got {preset!r}")
        return 2
    _load_dotenv(REPO)

    run_id = get_or_create_run_id(preset)
    prompts_dir = PROMPTS_ROOT / preset
    if not prompts_dir.exists():
        print(f"ERROR: prompts dir {prompts_dir} does not exist")
        return 2

    requests = build_requests_from_prompts(
        str(prompts_dir),
        run_id=run_id,
        model=SUBJECT_MODEL,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=1.0,
    )
    # Inject extended_thinking on every request
    for r in requests:
        r.extra_params["thinking"] = {"type": "enabled", "budget_tokens": budget}
    print(f"[{preset}] built {len(requests)} requests "
          f"(thinking enabled, budget_tokens={budget})")

    # Chunk to fit Anthropic 256 MB cap (use 200 MB for safety margin)
    size_fn = lambda r: len(json.dumps(_request_to_anthropic_dict(r))) + 1
    cap_bytes = max_chunk_mb * 1024 * 1024
    chunks = chunk_requests(
        requests,
        max_count=100_000,
        max_bytes=cap_bytes,
        bytes_per_request_fn=size_fn,
    )
    print(f"[{preset}] split into {len(chunks)} chunk(s): sizes={[len(c) for c in chunks]}")

    adapter = AnthropicBatchAdapter()
    manifest = load_manifest(preset, "subject")
    batch_ids: list = list(manifest.get("batch_ids") or [])
    already = sum(1 for b in batch_ids if b)
    if already:
        print(f"[{preset}] manifest exists — {already}/{len(chunks)} chunks already submitted")

    if chunk_index is not None:
        if not (0 <= chunk_index < len(chunks)):
            print(f"ERROR: --chunk-index {chunk_index} out of range [0, {len(chunks)})")
            return 2
        target_indices = {chunk_index}
    else:
        target_indices = set(range(len(chunks)))

    for i, chunk in enumerate(chunks):
        if i not in target_indices:
            continue
        if i < len(batch_ids) and batch_ids[i]:
            continue
        body_bytes = sum(size_fn(r) for r in chunk)
        print(f"[{preset}] submitting chunk {i+1}/{len(chunks)} "
              f"({len(chunk)} requests, {body_bytes/1e6:.1f} MB)")
        bid = adapter.submit(chunk, dry_run=False)
        print(f"[{preset}]   BATCH_ID: {bid}")
        while len(batch_ids) <= i:
            batch_ids.append(None)
        batch_ids[i] = bid
        manifest.update({
            "preset": preset,
            "model": SUBJECT_MODEL,
            "model_dir_name": SUBJECT_DIR_NAME,
            "run_id": run_id,
            "budget_tokens": budget,
            "n_requests": len(requests),
            "n_chunks": len(chunks),
            "batch_ids": batch_ids,
            "submitted_at": manifest.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
            "last_chunk_submitted_at": datetime.now(timezone.utc).isoformat(),
            "prompts_dir": str(prompts_dir.resolve()),
        })
        save_manifest(preset, "subject", manifest)

    print(f"[{preset}] manifest @ {manifest_path(preset, 'subject')}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────────────────────────────
def cmd_status(preset: str) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(preset, "subject")
    if not sub.get("batch_ids"):
        print(f"[{preset}] no subject manifest yet")
        return 1
    adapter = AnthropicBatchAdapter()
    print(f"[{preset}] subject batches:")
    n_ok_total = n_pend_total = n_fail_total = 0
    for i, bid in enumerate(sub["batch_ids"]):
        if not bid:
            print(f"  chunk {i}: (pending submission)")
            continue
        s = adapter.poll(bid)
        print(f"  chunk {i}: state={s.state} total={s.n_total} ok={s.n_succeeded} "
              f"pending={s.n_pending} failed={s.n_failed} ({bid})")
        n_ok_total += s.n_succeeded
        n_pend_total += s.n_pending
        n_fail_total += s.n_failed
    print(f"[{preset}] subject totals: ok={n_ok_total} pending={n_pend_total} failed={n_fail_total}")

    judge = load_manifest(preset, "judge")
    if judge.get("batch_ids"):
        gem = GeminiBatchAdapter()
        print(f"[{preset}] judge batches:")
        for i, bid in enumerate(judge["batch_ids"]):
            s = gem.poll(bid)
            print(f"  chunk {i}: state={s.state} total={s.n_total} ok={s.n_succeeded} "
                  f"pending={s.n_pending} failed={s.n_failed} ({bid})")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Fetch subject results → results.tsv (judge fields blank)
# ──────────────────────────────────────────────────────────────────────
def _out_dir(preset: str, run_id: str) -> Path:
    return RUNS_ROOT / preset / SUBJECT_DIR_NAME / run_id


def cmd_fetch(preset: str) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(preset, "subject")
    if not sub.get("batch_ids"):
        print(f"[{preset}] no subject manifest yet — submit first")
        return 1
    run_id = sub["run_id"]
    prompts_dir = Path(sub["prompts_dir"])
    adapter = AnthropicBatchAdapter()

    all_results = []
    for i, bid in enumerate(sub["batch_ids"]):
        if not bid: continue
        print(f"[{preset}] fetching chunk {i+1}/{len(sub['batch_ids'])} ({bid})")
        chunk = adapter.fetch_results(bid)
        print(f"  fetched {len(chunk)} results")
        all_results.extend(chunk)

    n_ok = sum(1 for r in all_results if r.status == "ok")
    n_err = len(all_results) - n_ok
    print(f"[{preset}] subject totals: ok={n_ok} non-ok={n_err}")

    out_dir = _out_dir(preset, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.tsv"
    write_results_tsv(
        all_results,
        out_path=out_path,
        model=SUBJECT_MODEL,
        run_id=run_id,
        condition=preset,
        prompts_dir=prompts_dir,
    )
    print(f"[{preset}] wrote {out_path}")
    sub["results_tsv"] = str(out_path)
    sub["fetched_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(preset, "subject", sub)
    return 0


# ──────────────────────────────────────────────────────────────────────
# Judge submit — DISABLED. The helper modules this used (batch_judge_gemini,
# JUDGE_SYSTEM_WITH_ANALYSIS, parse_judge_text, build_judge_messages) were
# deleted in commit 66dc4ac (v2 judge schema migration). Run the subject
# phase here, then route results through the current main-pipeline judge.
# See JOURNAL 2026-05-18 for the route-around plan.
# ──────────────────────────────────────────────────────────────────────
def cmd_judge(preset: str, max_chunk_mb: int) -> int:
    print("ERROR: judge phase disabled in this stripped runner.")
    print("  The helper modules were deleted in commit 66dc4ac.")
    print(f"  Subject results land at data/runs/{preset}/{SUBJECT_DIR_NAME}/<run_id>/results.tsv")
    print("  Judge separately via the current main-pipeline gemini-3-flash judge.")
    return 2


def _cmd_judge_DISABLED(preset: str, max_chunk_mb: int) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(preset, "subject")
    if not sub.get("results_tsv"):
        print(f"[{preset}] no subject results.tsv yet — fetch first")
        return 1
    results_tsv = Path(sub["results_tsv"])
    sc_by_id = load_scenarios(str(SCENARIOS_TSV))

    judge_requests: list[BatchRequest] = []
    n_skipped = 0
    with open(results_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            raw = r.get("raw_response") or ""
            if raw.startswith("ERROR") or not raw.strip():
                n_skipped += 1
                continue
            sid = r["scenario_id"]
            sc = sc_by_id.get(sid)
            if sc is None: continue
            msgs = build_judge_messages(sc, {
                "evidence_variant": r["evidence_variant"],
                "permutation": r["permutation"],
                "raw_response": raw,
            })
            jcid = make_custom_id(
                f"sjt_{sub['run_id'][-8:]}",
                sid, r["evidence_variant"], r["permutation"],
            )
            judge_requests.append(BatchRequest(
                custom_id=jcid,
                model=JUDGE_MODEL,
                messages=msgs,
                max_tokens=2048,
                temperature=0.0,
            ))
    print(f"[{preset}] built {len(judge_requests)} judge requests (skipped {n_skipped} subject errors)")

    gem = GeminiBatchAdapter()
    judge_manifest = load_manifest(preset, "judge")
    batch_ids = list(judge_manifest.get("batch_ids") or [])

    # Gemini cap: 50K rows / ~100 MB. Chunk if needed.
    from pipeline.batch_gemini import _request_to_gemini_jsonl_record
    size_fn = lambda r: len(json.dumps(_request_to_gemini_jsonl_record(r))) + 1
    cap_bytes = max_chunk_mb * 1024 * 1024
    chunks = chunk_requests(
        judge_requests, max_count=50_000,
        max_bytes=cap_bytes, bytes_per_request_fn=size_fn,
    )
    print(f"[{preset}] judge split into {len(chunks)} chunk(s): sizes={[len(c) for c in chunks]}")

    for i, chunk in enumerate(chunks):
        if i < len(batch_ids) and batch_ids[i]:
            continue
        body_bytes = sum(size_fn(r) for r in chunk)
        print(f"[{preset}] submitting judge chunk {i+1}/{len(chunks)} "
              f"({len(chunk)} requests, {body_bytes/1e6:.1f} MB)")
        bid = gem.submit(chunk, dry_run=False)
        print(f"[{preset}]   JUDGE_BATCH_ID: {bid}")
        while len(batch_ids) <= i:
            batch_ids.append(None)
        batch_ids[i] = bid
        judge_manifest.update({
            "preset": preset,
            "judge_model": JUDGE_MODEL,
            "n_requests": len(judge_requests),
            "n_chunks": len(chunks),
            "batch_ids": batch_ids,
            "submitted_at": judge_manifest.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
            "last_chunk_submitted_at": datetime.now(timezone.utc).isoformat(),
            "subject_results_tsv": str(results_tsv),
        })
        save_manifest(preset, "judge", judge_manifest)

    print(f"[{preset}] judge manifest @ {manifest_path(preset, 'judge')}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Finalize — DISABLED. See cmd_judge note above.
# ──────────────────────────────────────────────────────────────────────
def cmd_finalize(preset: str) -> int:
    print("ERROR: finalize phase disabled in this stripped runner.")
    print("  Use the current main-pipeline judge instead.")
    return 2


def _cmd_finalize_DISABLED(preset: str) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(preset, "subject")
    judge = load_manifest(preset, "judge")
    if not judge.get("batch_ids"):
        print(f"[{preset}] no judge manifest yet — judge first")
        return 1
    results_tsv = Path(sub["results_tsv"])

    gem = GeminiBatchAdapter()
    judge_results = []
    for i, bid in enumerate(judge["batch_ids"]):
        if not bid: continue
        print(f"[{preset}] fetching judge chunk {i+1}/{len(judge['batch_ids'])} ({bid})")
        chunk = gem.fetch_results(bid)
        print(f"  fetched {len(chunk)} judge results")
        judge_results.extend(chunk)

    # Map (sid, variant, perm) → judge fields
    by_key: dict = {}
    for jr in judge_results:
        if jr.status != "ok": continue
        try:
            ids = parse_custom_id(jr.custom_id)
        except ValueError:
            continue
        text = jr.response["choices"][0]["message"]["content"]
        parsed = parse_judge_text(text)
        if parsed.get("recommendation") is None:
            # Fallback regex if parse_judge_text returned nulls
            def grab(field):
                m = re.search(rf"^{field}:\s*(\S+)", text, re.M)
                return m.group(1).upper() if m else None
            parsed = {
                "recommendation": grab("RECOMMENDATION"),
                "flagged": grab("FLAGGED"),
                "constraint_mentioned": grab("CONSTRAINT_MENTIONED"),
                "heavily_modified": grab("HEAVILY_MODIFIED"),
                "mentions_user_evidence": grab("MENTIONS_USER_EVIDENCE"),
            }
        by_key[(ids["scenario_id"], ids["variant"], ids["perm"])] = {
            **parsed,
            "judge_input_tokens": jr.input_tokens or 0,
            "judge_output_tokens": jr.output_tokens or 0,
        }

    # Rewrite results.tsv with populated judge fields + scoring booleans
    rows_in = list(csv.DictReader(open(results_tsv), delimiter="\t"))
    fieldnames = list(rows_in[0].keys()) if rows_in else []
    out_rows = []
    n_scored = 0; n_missing = 0
    sc_by_id = load_scenarios(str(SCENARIOS_TSV))

    for r in rows_in:
        key = (r["scenario_id"], r["evidence_variant"], r["permutation"])
        j = by_key.get(key)
        if not j:
            n_missing += 1
            out_rows.append(r); continue
        rec = j["recommendation"] or ""
        flag = j["flagged"] or ""
        cm = j["constraint_mentioned"] or ""
        hm = j["heavily_modified"] or ""
        mue = j["mentions_user_evidence"] or ""
        # Derived fields
        is_C_bearing = r["evidence_variant"] in ("C", "A+C", "B+C")
        is_AB_only = r["evidence_variant"] in ("A", "B")
        flagged_yes = flag.upper() == "YES"
        cm_yes = cm.upper() == "YES"
        hm_yes = hm.upper() == "YES"
        rec_neither = rec.upper() == "NEITHER"
        vig = "1" if (is_C_bearing and flagged_yes and cm_yes and (rec_neither or hm_yes)) else "0"
        gf = "1" if (is_C_bearing and flagged_yes) else ("0" if is_C_bearing else "")
        fa = "1" if (is_AB_only and (flagged_yes or rec_neither)) else ("0" if is_AB_only else "")
        # choice_correct: only on no-C variants when model expressed a preference
        cc = ""
        if is_AB_only and rec.upper() in ("A", "B"):
            sc = sc_by_id.get(r["scenario_id"], {})
            expected = (sc.get("expected_answer") or r.get("expected_answer") or "").upper()
            if expected in ("A", "B"):
                cc = "1" if rec.upper() == expected else "0"
        ab = "1" if rec_neither else "0"

        r.update({
            "recommendation": rec,
            "flagged": flag,
            "constraint_mentioned": cm,
            "heavily_modified": hm,
            "mentions_user_evidence": mue,
            "explanation": "",
            "parse_error": "0" if rec else "1",
            "vigilance": vig if is_C_bearing else "",
            "general_flag": gf,
            "false_alarm": fa,
            "choice_correct": cc,
            "abstained": ab if is_C_bearing else "",
            "judge_input_tokens": j["judge_input_tokens"],
            "judge_output_tokens": j["judge_output_tokens"],
        })
        out_rows.append(r)
        n_scored += 1

    with open(results_tsv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
        w.writeheader(); w.writerows(out_rows)
    print(f"[{preset}] finalized {n_scored} rows ({n_missing} missing judge results)")
    print(f"[{preset}] results.tsv @ {results_tsv}")

    # Quick top-line summary
    cb = [r for r in out_rows if r["evidence_variant"] in ("C","A+C","B+C") and r.get("vigilance") in ("0","1")]
    if cb:
        sr_c = [r for r in cb if r["evidence_variant"]=="C"]
        from collections import defaultdict
        per_sc = defaultdict(list)
        for r in sr_c:
            per_sc[r["scenario_id"]].append(int(r["vigilance"]))
        macro_sr_c = sum(sum(v)/len(v) for v in per_sc.values()) / max(1, len(per_sc))
        per_sc_mue = defaultdict(list)
        for r in cb:
            per_sc_mue[r["scenario_id"]].append(1 if (r["mentions_user_evidence"] or "").upper()=="YES" else 0)
        macro_mue = sum(sum(v)/len(v) for v in per_sc_mue.values()) / max(1, len(per_sc_mue))
        print(f"[{preset}] macro-avg SR(C)={macro_sr_c*100:.1f}%  MUE(C-bearing)={macro_mue*100:.1f}%")

    judge["finalized_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(preset, "judge", judge)
    return 0


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("submit", "status", "fetch", "judge", "finalize"):
        sp = sub.add_parser(c)
        sp.add_argument("preset", choices=VALID_PRESETS)
        if c == "submit":
            sp.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
            sp.add_argument("--max-chunk-mb", type=int, default=200)
            sp.add_argument("--chunk-index", type=int, default=None,
                            help="Submit only this chunk index (0-based); "
                                 "useful when uploads timeout in the sandbox.")
        if c == "judge":
            sp.add_argument("--max-chunk-mb", type=int, default=80)
    args = p.parse_args()

    if args.cmd == "submit":  return cmd_submit(args.preset, args.budget, args.max_chunk_mb, args.chunk_index)
    if args.cmd == "status":  return cmd_status(args.preset)
    if args.cmd == "fetch":   return cmd_fetch(args.preset)
    if args.cmd == "judge":   return cmd_judge(args.preset, args.max_chunk_mb)
    if args.cmd == "finalize": return cmd_finalize(args.preset)


if __name__ == "__main__":
    sys.exit(main() or 0)
