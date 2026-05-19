"""
run_canon_gemini_thinking.py — Gemini 3.x canon with explicit thinkingLevel.

Companion to run_canon_haiku_thinking.py for the Gemini 3.x ablations.
Body-roster Gemini runs passed no `thinkingConfig`, relying on each model's
provider default. Empirical default thinking (from body-roster output token
volumes vs no-think baseline):
  - gemini-3-flash-preview:         ~4× no-think baseline (moderate)
  - gemini-3.1-flash-lite-preview:  ~2× no-think baseline (near-minimal)

This script re-runs canon_no_distractor + canon_unified for one Gemini model
with thinkingLevel explicitly set ("high" by default → upward ablation).

Settings:
  - Submit via Google AI Files Batch API (50% batch discount).
  - generationConfig.temperature = 1.0
  - generationConfig.maxOutputTokens = 10000
  - generationConfig.thinkingConfig.thinkingLevel = <CLI arg, default "high">
  - All other params provider-default.

Lifecycle:
    submit  — build canon requests + Gemini batch upload (this script)
    status  — poll Gemini batch state
    fetch   — pull subject results, write a results.tsv with judge fields
              blank. Judge separately via current main-pipeline judge.

Layout:
    Subject runs land at
    data/runs/<preset>/<model-slug>-thinking-<level>/<run_id>/results.tsv
    so the viewer surfaces them as a separate "model" alongside the
    existing default-mode rows.

Usage:
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        submit canon_no_distractor --model gemini-3-flash-preview --level high
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        submit canon_unified --model gemini-3-flash-preview --level high
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        submit canon_no_distractor --model gemini-3.1-flash-lite-preview --level high
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        submit canon_unified --model gemini-3.1-flash-lite-preview --level high

Status / fetch take the same --model and --level so the right manifest +
output dir are found:
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        status canon_unified --model gemini-3-flash-preview --level high
    python3 -m analysis_ablation.run_canon_gemini_thinking \
        fetch canon_unified --model gemini-3-flash-preview --level high

Valid thinking levels per Gemini 3.x docs: "minimal", "low", "medium", "high".
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

from openrouter_client import _load_dotenv                                  # noqa: E402
from pipeline.batch_common import (                                         # noqa: E402
    BatchRequest, chunk_requests,
)
from pipeline.batch_anthropic import build_requests_from_prompts            # noqa: E402
from pipeline.batch_gemini import (                                         # noqa: E402
    GeminiBatchAdapter, _request_to_gemini_jsonl_record,
)
from pipeline.batch_runner import write_results_tsv                         # noqa: E402

csv.field_size_limit(sys.maxsize)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 10000
DEFAULT_TEMPERATURE = 1.0
VALID_LEVELS = ("minimal", "low", "medium", "high")

# Gemini batch API rejects `thinkingLevel` (semantic string) but accepts
# `thinkingBudget` (numeric). Real-time accepts both. Map our semantic levels
# to numeric budgets that approximate the same thinking depth. Per Google's
# docs for Gemini 3 Flash: max thinkingBudget is 24576; -1 = dynamic (model
# decides); 0 = off.
LEVEL_TO_BUDGET = {
    "minimal": 128,
    "low":     2048,
    "medium":  8192,
    "high":    24576,
}
VALID_PRESETS = ("canon_no_distractor", "canon_unified")

PROMPTS_ROOT = REPO / "generated"
RUNS_ROOT = REPO / "data" / "runs"
MANIFESTS_DIR = REPO / "batch_manifests"


def _slug(model: str) -> str:
    """Full slug — used for dir names + manifest filenames (no length cap)."""
    return model


def _short_slug(model: str) -> str:
    """Compact slug for use inside run_id (must keep custom_id <= 64 chars)."""
    return {
        "gemini-3-flash-preview": "g3f",
        "gemini-3.1-flash-lite-preview": "g31fl",
        "gemini-3.1-pro-preview": "g31p",
    }.get(model, model[:8])


def _dir_name(model: str, level: str) -> str:
    return f"{_slug(model)}-thinking-{level}"


def _manifest_tag(model: str, level: str) -> str:
    return f"gemini_thinking__{_slug(model)}__{level}"


def manifest_path(model: str, level: str, preset: str, kind: str) -> Path:
    return MANIFESTS_DIR / f"{_manifest_tag(model, level)}__{preset}__{kind}.json"


def load_manifest(model: str, level: str, preset: str, kind: str) -> dict:
    p = manifest_path(model, level, preset, kind)
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(model: str, level: str, preset: str, kind: str, data: dict) -> None:
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    p = manifest_path(model, level, preset, kind)
    p.write_text(json.dumps(data, indent=2, default=str))


def get_or_create_run_id(model: str, level: str, preset: str) -> str:
    sub = load_manifest(model, level, preset, "subject")
    if sub.get("run_id"):
        return sub["run_id"]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"canon_{_short_slug(model)}_th{level[:3]}_{ts}"


# ──────────────────────────────────────────────────────────────────────
# Subject submit
# ──────────────────────────────────────────────────────────────────────
def cmd_submit(model: str, level: str, preset: str, max_chunk_mb: int,
               chunk_index: int | None = None) -> int:
    if preset not in VALID_PRESETS:
        print(f"ERROR: preset must be one of {VALID_PRESETS}, got {preset!r}")
        return 2
    if level not in VALID_LEVELS:
        print(f"ERROR: level must be one of {VALID_LEVELS}, got {level!r}")
        return 2
    _load_dotenv(REPO)

    run_id = get_or_create_run_id(model, level, preset)
    prompts_dir = PROMPTS_ROOT / preset
    if not prompts_dir.exists():
        print(f"ERROR: prompts dir {prompts_dir} does not exist")
        return 2

    requests = build_requests_from_prompts(
        str(prompts_dir),
        run_id=run_id,
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
    )
    # Inject thinking by overriding generationConfig — this REPLACES the
    # default generationConfig that batch_gemini built. We restore
    # temperature + maxOutputTokens explicitly + add thinkingConfig.
    # NB: batch endpoint accepts thinkingBudget (numeric) but rejects
    # thinkingLevel (semantic). See JOURNAL 2026-05-19.
    budget = LEVEL_TO_BUDGET[level]
    for r in requests:
        r.extra_params["generationConfig"] = {
            "temperature": DEFAULT_TEMPERATURE,
            "maxOutputTokens": DEFAULT_MAX_TOKENS,
            "thinkingConfig": {"thinkingBudget": budget},
        }
    print(f"[{preset}] built {len(requests)} requests "
          f"(model={model}, thinkingLevel={level} → thinkingBudget={budget})")

    # Gemini batch caps: 50K rows / ~100 MB per file
    size_fn = lambda r: len(json.dumps(_request_to_gemini_jsonl_record(r))) + 1
    cap_bytes = max_chunk_mb * 1024 * 1024
    chunks = chunk_requests(
        requests, max_count=50_000,
        max_bytes=cap_bytes, bytes_per_request_fn=size_fn,
    )
    print(f"[{preset}] split into {len(chunks)} chunk(s): sizes={[len(c) for c in chunks]}")

    adapter = GeminiBatchAdapter()
    manifest = load_manifest(model, level, preset, "subject")
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
            "model": model,
            "thinking_level": level,
            "model_dir_name": _dir_name(model, level),
            "run_id": run_id,
            "n_requests": len(requests),
            "n_chunks": len(chunks),
            "batch_ids": batch_ids,
            "submitted_at": manifest.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
            "last_chunk_submitted_at": datetime.now(timezone.utc).isoformat(),
            "prompts_dir": str(prompts_dir.resolve()),
        })
        save_manifest(model, level, preset, "subject", manifest)

    print(f"[{preset}] manifest @ {manifest_path(model, level, preset, 'subject')}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Status
# ──────────────────────────────────────────────────────────────────────
def cmd_status(model: str, level: str, preset: str) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(model, level, preset, "subject")
    if not sub.get("batch_ids"):
        print(f"[{preset}] no subject manifest yet")
        return 1
    adapter = GeminiBatchAdapter()
    print(f"[{preset}] subject batches ({model}, thinkingLevel={level}):")
    n_ok = n_pend = n_fail = 0
    for i, bid in enumerate(sub["batch_ids"]):
        if not bid:
            print(f"  chunk {i}: (pending submission)")
            continue
        s = adapter.poll(bid)
        print(f"  chunk {i}: state={s.state} total={s.n_total} ok={s.n_succeeded} "
              f"pending={s.n_pending} failed={s.n_failed} ({bid})")
        n_ok += s.n_succeeded; n_pend += s.n_pending; n_fail += s.n_failed
    print(f"[{preset}] subject totals: ok={n_ok} pending={n_pend} failed={n_fail}")
    return 0


# ──────────────────────────────────────────────────────────────────────
# Fetch subject results → results.tsv (judge fields blank)
# ──────────────────────────────────────────────────────────────────────
def cmd_fetch(model: str, level: str, preset: str) -> int:
    _load_dotenv(REPO)
    sub = load_manifest(model, level, preset, "subject")
    if not sub.get("batch_ids"):
        print(f"[{preset}] no subject manifest yet — submit first")
        return 1
    run_id = sub["run_id"]
    prompts_dir = Path(sub["prompts_dir"])
    adapter = GeminiBatchAdapter()

    all_results = []
    for i, bid in enumerate(sub["batch_ids"]):
        if not bid:
            continue
        print(f"[{preset}] fetching chunk {i+1}/{len(sub['batch_ids'])} ({bid})")
        chunk = adapter.fetch_results(bid)
        print(f"  fetched {len(chunk)} results")
        all_results.extend(chunk)

    n_ok = sum(1 for r in all_results if r.status == "ok")
    n_err = len(all_results) - n_ok
    print(f"[{preset}] subject totals: ok={n_ok} non-ok={n_err}")

    out_dir = RUNS_ROOT / preset / _dir_name(model, level) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.tsv"
    write_results_tsv(
        all_results,
        out_path=out_path,
        model=model,
        run_id=run_id,
        condition=preset,
        prompts_dir=prompts_dir,
    )
    print(f"[{preset}] wrote {out_path}")
    sub["results_tsv"] = str(out_path)
    sub["fetched_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(model, level, preset, "subject", sub)
    return 0


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("submit", "status", "fetch"):
        sp = sub.add_parser(c)
        sp.add_argument("preset", choices=VALID_PRESETS)
        sp.add_argument("--model", required=True,
                        help="Gemini model (e.g., gemini-3-flash-preview, gemini-3.1-flash-lite-preview)")
        sp.add_argument("--level", required=True, choices=VALID_LEVELS,
                        help="thinkingLevel to inject")
        if c == "submit":
            sp.add_argument("--max-chunk-mb", type=int, default=80,
                            help="Per-chunk JSONL size cap (Gemini limit ~100 MB)")
            sp.add_argument("--chunk-index", type=int, default=None)
    args = p.parse_args()

    if args.cmd == "submit":
        return cmd_submit(args.model, args.level, args.preset, args.max_chunk_mb, args.chunk_index)
    if args.cmd == "status":
        return cmd_status(args.model, args.level, args.preset)
    if args.cmd == "fetch":
        return cmd_fetch(args.model, args.level, args.preset)


if __name__ == "__main__":
    sys.exit(main() or 0)
