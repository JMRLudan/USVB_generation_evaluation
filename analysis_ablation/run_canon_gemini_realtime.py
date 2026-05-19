"""
run_canon_gemini_realtime.py — Gemini 3.x ablation via native real-time API.

The batch API rejects `thinkingLevel` (semantic) and silently downgrades large
`thinkingBudget` values to dynamic-default — both verified empirically in this
session. The only way to actually force `thinking_level="high"` on Gemini is
via the real-time `generateContent` endpoint.

This script does that, async, with concurrency control + incremental TSV
writes + resume from checkpoint. Mirrors the schema of the existing canon
runs so the same judge-batch tooling works on the output.

Bills against GEMINI_API_KEY (not OpenRouter), so the OR key's $750 self-cap
is irrelevant.

Usage:
    python3 -m analysis_ablation.run_canon_gemini_realtime \
        --model gemini-3-flash-preview --level high \
        --preset canon_no_distractor --concurrency 30

    python3 -m analysis_ablation.run_canon_gemini_realtime \
        --model gemini-3-flash-preview --level high \
        --preset canon_unified --concurrency 30 --limit 0

Output:
    data/runs/<preset>/<model>-thinking-rt-<level>/<run_id>/results.tsv
    data/runs/<preset>/<model>-thinking-rt-<level>/<run_id>/checkpoint.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pipeline"))

from openrouter_client import _load_dotenv                                  # noqa: E402

VALID_LEVELS = ("minimal", "low", "medium", "high")
VALID_PRESETS = ("canon_no_distractor", "canon_unified")

DEFAULT_MAX_OUTPUT = 16000
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CONCURRENCY = 20
DEFAULT_TIMEOUT = 300.0

PROMPTS_ROOT = REPO / "generated"
RUNS_ROOT = REPO / "data" / "runs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _item_key(sid: str, variant: str, perm: str) -> str:
    return f"{sid}|{variant}|{perm}"


# TSV columns — match the schema used by analysis_ablation/run_judge_batch.py
# so the same fetch/merge tooling works.
COLUMNS = [
    "run_id", "condition", "scenario_id", "evidence_variant", "permutation",
    "expected_answer",
    "raw_response",
    "recommendation", "constraint_mentioned", "sufficiently_modified",
    "explanation", "parse_error",
    "vigilance_refuse_only", "abstain_type", "choice_correct", "abstained",
    "input_tokens", "output_tokens", "judge_input_tokens", "judge_output_tokens",
    "latency_ms",
]


def load_completed_keys(checkpoint: Path) -> set[str]:
    if not checkpoint.exists():
        return set()
    with open(checkpoint) as f:
        return {line.strip() for line in f if line.strip()}


def load_prompts(preset: str, limit: int) -> list[dict]:
    pdir = PROMPTS_ROOT / preset
    items: list[dict] = []
    for fp in sorted(pdir.iterdir()):
        if fp.suffix != ".json":
            continue
        try:
            j = json.loads(fp.read_text())
        except Exception:
            continue
        items.append(j)
        if limit and len(items) >= limit:
            break
    return items


async def call_once(client, model: str, system_prompt: str, user_message: str,
                    thinking_level: str, max_output: int,
                    max_retries: int = 5) -> dict:
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=DEFAULT_TEMPERATURE,
        max_output_tokens=max_output,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        system_instruction=system_prompt,
    )
    last_err = ""
    for attempt in range(max_retries):
        t0 = time.monotonic()
        try:
            resp = await client.aio.models.generate_content(
                model=model,
                contents=user_message,
                config=config,
            )
            text = resp.text or ""
            usage = resp.usage_metadata
            in_tok = int(getattr(usage, "prompt_token_count", 0) or 0)
            cand_tok = int(getattr(usage, "candidates_token_count", 0) or 0)
            thought_tok = int(getattr(usage, "thoughts_token_count", 0) or 0)
            out_tok = cand_tok + thought_tok
            latency_ms = int((time.monotonic() - t0) * 1000)
            return {
                "ok": True,
                "raw_response": text,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency_ms,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            msg = str(e).lower()
            transient = any(t in msg for t in ("rate", "429", "503", "500",
                                                "deadline", "timeout", "unavailable"))
            if transient and attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt + 1)
                continue
            return {
                "ok": False,
                "raw_response": f"ERROR {last_err}",
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": int((time.monotonic() - t0) * 1000),
            }
    return {
        "ok": False,
        "raw_response": f"ERROR retries_exhausted: {last_err}",
        "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
    }


async def main_async(args: argparse.Namespace) -> int:
    _load_dotenv(REPO)
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set")
        return 2

    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    # Output dir
    model_slug = args.model
    dir_name = f"{model_slug}-thinking-rt-{args.level}"
    run_id = args.run_id or f"canon_rt_{args.level}_{_ts()}"
    out_dir = RUNS_ROOT / args.preset / dir_name / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "results.tsv"
    checkpoint = out_dir / "checkpoint.txt"

    # Items
    items = load_prompts(args.preset, args.limit)
    print(f"loaded {len(items)} prompts from generated/{args.preset}/")
    completed = load_completed_keys(checkpoint)
    print(f"completed (resume): {len(completed)} keys")

    todo = []
    for it in items:
        m = it.get("metadata", {})
        key = _item_key(m["scenario_id"], m["evidence_variant"], m["permutation"])
        if key in completed:
            continue
        todo.append(it)
    print(f"todo: {len(todo)} items at concurrency={args.concurrency}")

    if not todo:
        print("nothing to do")
        return 0

    # Open TSV in append mode (write header if new)
    tsv_new = not out_tsv.exists()
    tsv_f = open(out_tsv, "a", newline="")
    tsv_w = csv.DictWriter(tsv_f, fieldnames=COLUMNS, delimiter="\t",
                           quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
    if tsv_new:
        tsv_w.writeheader()
        tsv_f.flush()
    ck_f = open(checkpoint, "a")

    sem = asyncio.Semaphore(args.concurrency)
    done_count = [0]
    err_count = [0]
    n_total = len(todo)
    t_start = time.monotonic()

    async def worker(item: dict):
        m = item["metadata"]
        sid, var, perm = m["scenario_id"], m["evidence_variant"], m["permutation"]
        key = _item_key(sid, var, perm)
        async with sem:
            r = await call_once(
                client, args.model,
                item.get("system_prompt", ""),
                item.get("user_message", ""),
                thinking_level=args.level,
                max_output=args.max_tokens,
            )
        row = {
            "run_id": run_id,
            "condition": args.preset,
            "scenario_id": sid,
            "evidence_variant": var,
            "permutation": perm,
            "expected_answer": m.get("expected_answer", ""),
            "raw_response": r["raw_response"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "latency_ms": r["latency_ms"],
        }
        tsv_w.writerow(row); tsv_f.flush()
        ck_f.write(key + "\n"); ck_f.flush()
        done_count[0] += 1
        if not r["ok"]:
            err_count[0] += 1
        if done_count[0] % 100 == 0 or done_count[0] == n_total:
            elapsed = time.monotonic() - t_start
            rate = done_count[0] / elapsed if elapsed > 0 else 0
            eta = (n_total - done_count[0]) / rate if rate > 0 else 0
            print(f"  {done_count[0]:>5}/{n_total}  err={err_count[0]}  "
                  f"rate={rate:.1f}/s  eta={int(eta)}s")

    tasks = [asyncio.create_task(worker(it)) for it in todo]
    try:
        await asyncio.gather(*tasks)
    finally:
        tsv_f.close()
        ck_f.close()

    elapsed = time.monotonic() - t_start
    print(f"\n✓ {args.preset}/{dir_name}: {done_count[0]} done  "
          f"err={err_count[0]} in {int(elapsed)}s")
    print(f"   → {out_tsv}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="Gemini model id (e.g. gemini-3-flash-preview)")
    p.add_argument("--level", required=True, choices=VALID_LEVELS,
                   help="thinking_level")
    p.add_argument("--preset", required=True, choices=VALID_PRESETS)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_OUTPUT)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap items processed (0 = all)")
    p.add_argument("--run-id", default=None,
                   help="Reuse a specific run_id (for resume)")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)) or 0)


if __name__ == "__main__":
    main()
