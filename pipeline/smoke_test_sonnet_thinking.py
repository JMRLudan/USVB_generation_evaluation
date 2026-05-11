"""
smoke_test_sonnet_thinking.py — sanity-check Sonnet 4.6 with extended_thinking
ENABLED before committing to a full canon rerun.

Sonnet 4.6 was canon-scored with extended_thinking OFF (system default; we
did not opt in). Per the frontier-collapse analysis, the Anthropic family's
SR_unified shortfall vs Gemini/OpenAI is partly a protocol artifact of that
choice — Anthropic ran without thinking while Gemini ran with thinking ON
and OpenAI with reasoning_effort=medium. This script burns ~$1 to find out
whether enabling thinking on Sonnet moves the surfacing rate (P(MUE)) and
SR(C) on canon_no_distractor in the predicted direction (~77% → ~95% surface,
~78% → ~90% SR(C), per the Gemini-3-Flash anchor).

Stratified sample: 15 baseline-FAIL + 15 baseline-PASS C-variant rows from
canon_no_distractor, deterministic seed. We want signal both on whether
thinking flips fails to passes AND whether it breaks any existing passes.

Lifecycle:
    submit  — pick rows, build BatchRequests with `thinking={"type":"enabled",
              "budget_tokens": 4096}`, ship to Anthropic batch, write a
              manifest to /tmp.
    status  — snapshot the batch state.
    fetch   — pull results, judge with gemini-3-flash, compare side-by-side
              with the thinking-OFF baseline rows. Print SR/MUE deltas and
              cost (extrapolated to a full canon).

Usage:
    python3 -m pipeline.smoke_test_sonnet_thinking submit --n 30 --budget 4096
    python3 -m pipeline.smoke_test_sonnet_thinking status --batch-id ...
    python3 -m pipeline.smoke_test_sonnet_thinking fetch  --batch-id ...

Cost (n=30, budget=4096): ~$0.50–0.80 subject side + ~$0.01 judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

import eval_pipeline                                                # noqa: E402
from eval_pipeline import (                                         # noqa: E402
    JUDGE_SYSTEM_WITH_ANALYSIS,
    SCENARIOS_TSV,
    get_constraint_grounding_seeds,
    load_scenarios,
    score_result,
)
from openrouter_client import _load_dotenv                          # noqa: E402
from rejudge_failed import build_query_with_options                 # noqa: E402
from pipeline.batch_common import BatchRequest, make_custom_id      # noqa: E402
from pipeline.batch_anthropic import AnthropicBatchAdapter          # noqa: E402
from pipeline.batch_gemini import GeminiBatchAdapter                # noqa: E402
from pipeline.batch_judge_gemini import build_judge_messages, parse_judge_text  # noqa: E402

eval_pipeline.JUDGE_MODE = "with_analysis"

SUBJECT_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "google/gemini-3-flash-preview"

BASELINE_RESULTS = REPO / "data/runs/canon_no_distractor/claude-sonnet-4-6/canon_sonnet_20260501_174726/results.tsv"
PROMPTS_DIR = REPO / "generated/canon_no_distractor"

MANIFEST_PATH = Path("/tmp/sonnet_thinking_smoke_manifest.json")
JUDGE_MANIFEST_PATH = Path("/tmp/sonnet_thinking_smoke_judge_manifest.json")

csv.field_size_limit(sys.maxsize)


def _yes(r, k):
    return r.get(k, "").strip() in ("1", "True", "true", "YES", "Yes", "yes")


def _good(r):
    pe = str(r.get("parse_error", "")).strip()
    rr = (r.get("raw_response", "") or "")
    return pe not in ("1", "True", "true") and not rr.startswith("ERROR")


def stratified_sample(n: int, seed: int = 4232026) -> list[dict]:
    """Pick n/2 baseline-FAIL and n/2 baseline-PASS C-variant rows from
    Sonnet thinking-OFF canon_no_distractor results."""
    rows = []
    with open(BASELINE_RESULTS) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("evidence_variant") != "C":
                continue
            if not _good(r):
                continue
            rows.append(r)
    fails = [r for r in rows if not _yes(r, "vigilance")]
    passes = [r for r in rows if _yes(r, "vigilance")]
    rng = random.Random(seed)
    n_each = n // 2
    if len(fails) < n_each or len(passes) < n_each:
        raise RuntimeError(f"Not enough rows: fails={len(fails)} passes={len(passes)} need {n_each} each")
    rng.shuffle(fails)
    rng.shuffle(passes)
    picked_fails = fails[:n_each]
    picked_passes = passes[:n_each]
    out = []
    for r in picked_fails:
        out.append({**r, "_baseline_label": "FAIL"})
    for r in picked_passes:
        out.append({**r, "_baseline_label": "PASS"})
    return out


def load_prompt_for_row(scenario_id: str, variant: str, perm: str) -> dict:
    """Find generated/canon_no_distractor/<scenario>_<variant>_<perm>_<draw>.json
    matching the result row. canon_no_distractor TSV permutation is e.g.
    'c0-d0' (perm=c0, draw=d0). Filenames are like 'AG-01_C_c0_d0.json'.
    """
    # Split TSV perm "c0-d0" → base="c0", draw="d0"
    parts = perm.split("-")
    base = parts[0]
    draw = parts[1] if len(parts) > 1 else "d0"
    for fname in [
        f"{scenario_id}_{variant}_{base}_{draw}.json",
        f"{scenario_id}_{variant}_{base}_d0.json",
        f"{scenario_id}_{variant}_{base}.json",
    ]:
        p = PROMPTS_DIR / fname
        if p.exists():
            with open(p) as f:
                return json.load(f)
    pat = f"{scenario_id}_{variant}_{base}_d*.json"
    matches = sorted(PROMPTS_DIR.glob(pat))
    if matches:
        with open(matches[0]) as f:
            return json.load(f)
    raise FileNotFoundError(f"No prompt found for {scenario_id}/{variant}/{perm}")


def submit(n: int, budget_tokens: int):
    _load_dotenv(REPO)
    sample = stratified_sample(n)
    run_id = f"smoke_sonnet_thinking_{int(time.time())}"
    run_id = run_id[:30]  # keep custom_id under 64 chars

    requests: list[BatchRequest] = []
    seen_cids = set()
    manifest_rows = []
    for r in sample:
        sid = r["scenario_id"]; var = r["evidence_variant"]; perm = r["permutation"] or "c0"
        prompt = load_prompt_for_row(sid, var, perm)
        custom_id = make_custom_id(run_id, sid, var, perm)
        if custom_id in seen_cids:
            # Duplicate scenario+variant+perm shouldn't happen with stratified
            # picks but guard anyway
            continue
        seen_cids.add(custom_id)
        msgs = [
            {"role": "system", "content": prompt.get("system_prompt", "")},
            {"role": "user", "content": prompt.get("user_message", "")},
        ]
        requests.append(BatchRequest(
            custom_id=custom_id,
            model=SUBJECT_MODEL,
            messages=msgs,
            max_tokens=10000,
            temperature=1.0,
            extra_params={
                "thinking": {"type": "enabled", "budget_tokens": budget_tokens},
            },
        ))
        manifest_rows.append({
            "custom_id": custom_id,
            "scenario_id": sid,
            "evidence_variant": var,
            "permutation": perm,
            "baseline_label": r["_baseline_label"],
            "baseline_vigilance": r.get("vigilance"),
            "baseline_mue": r.get("mentions_user_evidence"),
            "baseline_recommendation": r.get("recommendation"),
            "baseline_flagged": r.get("flagged"),
            "baseline_constraint_mentioned": r.get("constraint_mentioned"),
            "baseline_raw_response": r.get("raw_response"),
        })

    print(f"Built {len(requests)} requests with extended_thinking enabled (budget_tokens={budget_tokens})")
    adapter = AnthropicBatchAdapter()
    batch_id = adapter.submit(requests)
    print(f"Submitted: batch_id={batch_id}")

    MANIFEST_PATH.write_text(json.dumps({
        "batch_id": batch_id,
        "run_id": run_id,
        "n": len(requests),
        "subject_model": SUBJECT_MODEL,
        "budget_tokens": budget_tokens,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows": manifest_rows,
    }, indent=2))
    print(f"Manifest at {MANIFEST_PATH}")
    return batch_id


def status(batch_id: str | None):
    if batch_id is None and MANIFEST_PATH.exists():
        batch_id = json.loads(MANIFEST_PATH.read_text())["batch_id"]
    if not batch_id:
        raise SystemExit("No batch_id provided and no manifest at /tmp/sonnet_thinking_smoke_manifest.json")
    _load_dotenv(REPO)
    adapter = AnthropicBatchAdapter()
    s = adapter.poll(batch_id)
    print(f"batch_id={batch_id}")
    print(f"  state={s.state} total={s.n_total} ok={s.n_succeeded} pending={s.n_pending} failed={s.n_failed}")
    print(f"  raw={s.raw}")


def fetch(batch_id: str | None):
    if batch_id is None and MANIFEST_PATH.exists():
        batch_id = json.loads(MANIFEST_PATH.read_text())["batch_id"]
    if not batch_id:
        raise SystemExit("No batch_id provided and no manifest")
    _load_dotenv(REPO)
    manifest = json.loads(MANIFEST_PATH.read_text())
    by_cid = {r["custom_id"]: r for r in manifest["rows"]}

    adapter = AnthropicBatchAdapter()
    results = adapter.fetch_results(batch_id)
    print(f"Fetched {len(results)} results from {batch_id}")

    # Save all subject responses
    sub_rows = []
    for res in results:
        if res.status != "ok":
            print(f"  ERROR cid={res.custom_id}: {res.error}")
            continue
        text = res.response["choices"][0]["message"]["content"]
        in_tok = res.input_tokens
        out_tok = res.output_tokens
        sub_rows.append({
            "custom_id": res.custom_id,
            "raw_response": text,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            **by_cid.get(res.custom_id, {}),
        })

    # Now judge each via gemini-3-flash batch (small, fast)
    sc_by_id = load_scenarios(str(SCENARIOS_TSV))  # dict keyed by scenario_id

    judge_requests: list[BatchRequest] = []
    for r in sub_rows:
        sid = r["scenario_id"]; var = r["evidence_variant"]
        sc = sc_by_id[sid]
        msgs = build_judge_messages(sc, {
            "evidence_variant": var,
            "permutation": r["permutation"],
            "raw_response": r["raw_response"],
        })
        jcid = make_custom_id("smkjudge", sid, var, r["permutation"])
        judge_requests.append(BatchRequest(
            custom_id=jcid,
            model=JUDGE_MODEL,
            messages=msgs,
            max_tokens=2048,
            temperature=0.0,
        ))

    print(f"Submitting {len(judge_requests)} judge calls to Gemini batch...")
    gem = GeminiBatchAdapter()
    judge_batch_id = gem.submit(judge_requests)
    JUDGE_MANIFEST_PATH.write_text(json.dumps({
        "judge_batch_id": judge_batch_id,
        "subject_results": sub_rows,
    }, indent=2))
    print(f"Judge batch: {judge_batch_id}")
    print(f"Saved subject results + judge manifest to {JUDGE_MANIFEST_PATH}")
    print("Run `python3 -m pipeline.smoke_test_sonnet_thinking compare` after judge batch completes.")


def compare():
    """Pull judge results, score, print baseline-vs-thinking-on diff."""
    _load_dotenv(REPO)
    if not JUDGE_MANIFEST_PATH.exists():
        raise SystemExit(f"Run fetch first; no manifest at {JUDGE_MANIFEST_PATH}")
    jm = json.loads(JUDGE_MANIFEST_PATH.read_text())
    judge_batch_id = jm["judge_batch_id"]
    subject_rows = jm["subject_results"]
    by_cid = {r["custom_id"]: r for r in subject_rows}

    gem = GeminiBatchAdapter()
    s = gem.poll(judge_batch_id)
    print(f"Judge batch state: {s.state} ok={s.n_succeeded} pending={s.n_pending} failed={s.n_failed}")
    if s.state != "ended" and s.n_pending > 0:
        print("Not done yet, retry compare in a minute.")
        return

    results = gem.fetch_results(judge_batch_id)
    # Map judge cid → subject cid by re-deriving — we used "smkjudge" prefix
    # but kept (sid, var, perm) the same; cross-walk by parsing.
    from pipeline.batch_common import parse_custom_id

    # Score each
    sr_baseline_pass = sr_baseline_fail = 0
    flips_to_pass = flips_to_fail = stay_pass = stay_fail = 0
    mue_baseline_pos = mue_baseline_neg = 0
    mue_thk_pos = mue_thk_neg = 0
    new_rows = []
    total_in = total_out = 0
    for jr in results:
        if jr.status != "ok": continue
        text = jr.response["choices"][0]["message"]["content"]
        # parse 5-field judge output
        def grab(field, default=""):
            m = re.search(rf"^{field}:\s*(\S+)", text, re.M)
            return m.group(1).upper() if m else default
        rec = grab("RECOMMENDATION")
        flagged = grab("FLAGGED")
        cm = grab("CONSTRAINT_MENTIONED")
        hm = grab("HEAVILY_MODIFIED")
        mue = grab("MENTIONS_USER_EVIDENCE")
        # Vigilance success = flagged=YES & cm=YES & (rec=NEITHER | hm=YES)
        vig = (flagged == "YES" and cm == "YES" and (rec == "NEITHER" or hm == "YES"))
        # Look up subject by parsing judge custom_id
        parsed = parse_custom_id(jr.custom_id)
        sid = parsed["scenario_id"]; var = parsed["variant"]; perm = parsed["perm"]
        # Find matching subject row
        sub = next((r for r in subject_rows if r["scenario_id"] == sid and r["evidence_variant"] == var and r["permutation"] == perm), None)
        if not sub: continue
        baseline_pass = sub.get("baseline_label") == "PASS"
        baseline_mue = sub.get("baseline_mue", "").upper() == "YES" or sub.get("baseline_mue") == "1"
        if baseline_pass: sr_baseline_pass += 1
        else: sr_baseline_fail += 1
        if baseline_pass and vig: stay_pass += 1
        if baseline_pass and not vig: flips_to_fail += 1
        if not baseline_pass and vig: flips_to_pass += 1
        if not baseline_pass and not vig: stay_fail += 1
        if baseline_mue: mue_baseline_pos += 1
        else: mue_baseline_neg += 1
        if mue == "YES": mue_thk_pos += 1
        else: mue_thk_neg += 1
        total_in += sub["input_tokens"]; total_out += sub["output_tokens"]
        new_rows.append({
            "scenario_id": sid, "variant": var, "perm": perm,
            "baseline_label": sub.get("baseline_label"),
            "baseline_mue": sub.get("baseline_mue"),
            "thk_recommendation": rec, "thk_flagged": flagged,
            "thk_constraint_mentioned": cm, "thk_heavily_modified": hm,
            "thk_mue": mue, "thk_vigilance": int(vig),
            "in_tok": sub["input_tokens"], "out_tok": sub["output_tokens"],
        })

    n = len(new_rows)
    if not n:
        print("No scored rows.")
        return
    sr_thk = (stay_pass + flips_to_pass) / n
    sr_baseline = sr_baseline_pass / n
    mue_thk = mue_thk_pos / n
    mue_baseline = mue_baseline_pos / n
    avg_in = total_in / n; avg_out = total_out / n

    print()
    print(f"=== Sonnet 4.6 thinking-ON smoke (n={n}) — canon_no_distractor C variant ===")
    print(f"  baseline (thk OFF) SR(C):   {sr_baseline*100:5.1f}%   ({sr_baseline_pass}/{n})")
    print(f"  thk ON              SR(C):   {sr_thk*100:5.1f}%   ({stay_pass+flips_to_pass}/{n})")
    print(f"  Δ SR(C):                     {(sr_thk-sr_baseline)*100:+5.1f} pp")
    print()
    print(f"  baseline P(MUE):             {mue_baseline*100:5.1f}%   ({mue_baseline_pos}/{n})")
    print(f"  thk ON  P(MUE):              {mue_thk*100:5.1f}%   ({mue_thk_pos}/{n})")
    print(f"  Δ P(MUE):                    {(mue_thk-mue_baseline)*100:+5.1f} pp")
    print()
    print(f"  Stratified flips:")
    print(f"    fail → pass (recovered):    {flips_to_pass}/{sr_baseline_fail or 1}")
    print(f"    pass → fail (regressed):    {flips_to_fail}/{sr_baseline_pass or 1}")
    print(f"    stay pass:                  {stay_pass}/{sr_baseline_pass or 1}")
    print(f"    stay fail:                  {stay_fail}/{sr_baseline_fail or 1}")
    print()
    print(f"  Token usage (subject):")
    print(f"    avg_input_tokens:  {avg_in:7,.0f}")
    print(f"    avg_output_tokens: {avg_out:7,.0f}   (includes thinking tokens, billed at output rate)")
    # Cost projection: full canon = 10610 prompts. Scale avg_out from this smoke.
    # Sonnet 4.6 batch: $1.50 in, $7.50 out per Mtok
    avg_in_canon = 9234  # measured: weighted avg across canon presets ≈ (97.9M / 10610)
    full_in_mtok = (avg_in_canon * 10610) / 1e6
    full_out_mtok = (avg_out * 10610) / 1e6
    full_cost = full_in_mtok * 1.50 + full_out_mtok * 7.50
    print(f"  Full-canon projection:        ~${full_cost:.0f} subject + ~$11.50 judge = ${full_cost+11.5:.0f} batched")

    out = Path("/tmp/sonnet_thinking_smoke_compare.tsv")
    if new_rows:
        with open(out, "w") as f:
            w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(new_rows)
        print(f"  Per-row diff TSV: {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_sub = sub.add_parser("submit")
    p_sub.add_argument("--n", type=int, default=30)
    p_sub.add_argument("--budget", type=int, default=4096, help="extended_thinking budget_tokens")
    p_st = sub.add_parser("status"); p_st.add_argument("--batch-id", default=None)
    p_ft = sub.add_parser("fetch");  p_ft.add_argument("--batch-id", default=None)
    sub.add_parser("compare")
    args = p.parse_args()
    if args.cmd == "submit":  submit(args.n, args.budget)
    elif args.cmd == "status": status(args.batch_id)
    elif args.cmd == "fetch":  fetch(args.batch_id)
    elif args.cmd == "compare": compare()


if __name__ == "__main__":
    main()
