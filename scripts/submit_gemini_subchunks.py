#!/usr/bin/env python3
"""
submit_gemini_subchunks.py — sandbox-safe resumable Gemini batch submission.

Rationale: a single ~100MB Files-API upload exceeds the sandbox's 45s
per-call budget, and a mid-upload kill leaves an orphan batch whose id
was never recorded (observed 2026-07-08, mitigation-arm submission).
This driver submits small sub-chunks (default ≤25MB) in a loop with a
self-imposed wall-clock deadline, persisting every batch_id to a
sidecar state file IMMEDIATELY after each submit. Re-run the same
command until it prints DONE; each invocation resumes from the sidecar.

On completion it writes a batch_runner-compatible manifest
(batch_manifests/<run_id>__<preset>.json) so the normal
`python3 -m pipeline.batch_runner fetch <run_id>__<preset>` flow works
unchanged (fetch unions all batch_ids in the manifest).

Usage:
    python3 scripts/submit_gemini_subchunks.py \
        --prompts-dir generated/canon_unified_mit_sysbottom \
        --run-id mitnv_20260708_171704 \
        [--model gemini-3-flash-preview] [--sub-mb 25] [--deadline-s 30] \
        [--adopt <batch_id>:<n_requests>]   # record an orphan covering
                                            # the NEXT n requests in order
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from pipeline.batch_runner import build_requests, get_adapter  # noqa: E402
from pipeline.batch_gemini import _request_to_gemini_jsonl_record  # noqa: E402

# Same .env auto-load as batch_runner.main() — the Gemini adapter reads
# the key from the process environment.
try:
    from pipeline.openrouter_client import _load_dotenv
    _load_dotenv(BASE)
except Exception:
    pass

STATE_DIR = BASE / "batch_manifests"


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--provider", choices=["gemini", "anthropic"], default="gemini")
    ap.add_argument("--model", default="gemini-3-flash-preview")
    ap.add_argument("--max-tokens", type=int, default=10000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sub-mb", type=float, default=25.0)
    ap.add_argument("--deadline-s", type=float, default=30.0)
    ap.add_argument("--adopt", default=None,
                    help="<batch_id>:<n_requests> — record an already-submitted "
                         "orphan batch as covering the next n requests.")
    ap.add_argument("--replace-user-message", default=None,
                    help="Replace the final user message of every request with "
                         "this text (e.g. a summarization instruction). The "
                         "history-bearing system prompt is untouched.")
    ap.add_argument("--variants", default=None,
                    help="Comma-separated evidence-variant filter matched "
                         "against the custom_id (normalized: A+C -> AC). "
                         "e.g. 'C,AC,BC' keeps only C-bearing prompts.")
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    preset = prompts_dir.name
    stem = f"{args.run_id}__{preset}"
    state_path = STATE_DIR / f"{stem}.subchunks.json"
    manifest_path = STATE_DIR / f"{stem}.json"
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    state = (json.loads(state_path.read_text())
             if state_path.exists()
             else {"covered": 0, "batch_ids": [], "sub_sizes": []})

    if args.adopt:
        bid, n = args.adopt.rsplit(":", 1)
        state["batch_ids"].append(bid)
        state["sub_sizes"].append(int(n))
        state["covered"] += int(n)
        state.setdefault("adopted", []).append(bid)
        state_path.write_text(json.dumps(state, indent=2))
        print(f"adopted {bid} covering {n} requests; covered={state['covered']}")

    requests = build_requests(
        prompts_dir, run_id=args.run_id, model=args.model,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    if args.variants:
        keep = {v.strip() for v in args.variants.split(",")}
        requests = [r for r in requests
                    if r.custom_id.split("__")[2] in keep]
    if args.replace_user_message:
        for r in requests:
            for msg in reversed(r.messages):
                if msg["role"] == "user":
                    msg["content"] = args.replace_user_message
                    break
    total = len(requests)
    print(f"{preset}: {total} requests total; covered={state['covered']}")

    adapter = get_adapter(args.provider)
    if args.provider == "anthropic":
        from pipeline.batch_anthropic import _request_to_anthropic_dict
        size_fn = lambda r: len(json.dumps(_request_to_anthropic_dict(r))) + 1
    else:
        size_fn = lambda r: len(json.dumps(_request_to_gemini_jsonl_record(r))) + 1
    max_bytes = int(args.sub_mb * 1024 * 1024)

    while state["covered"] < total:
        if time.time() - t0 > args.deadline_s:
            print(f"DEADLINE — stopping cleanly at covered={state['covered']}/{total}. "
                  f"Re-run the same command to continue.")
            return 0
        # Build the next sub-chunk under the byte cap (min 1 request).
        sub, size = [], 0
        for r in requests[state["covered"]:]:
            b = size_fn(r)
            if sub and size + b > max_bytes:
                break
            sub.append(r)
            size += b
        print(f"submitting requests [{state['covered']}:{state['covered']+len(sub)}] "
              f"(~{size/1e6:.1f} MB)…", flush=True)
        bid = adapter.submit(sub, dry_run=False)
        # Persist BEFORE anything else can fail.
        state["batch_ids"].append(bid)
        state["sub_sizes"].append(len(sub))
        state["covered"] += len(sub)
        state_path.write_text(json.dumps(state, indent=2))
        print(f"  BATCH_ID: {bid}  (covered {state['covered']}/{total})", flush=True)

    # Complete — write the batch_runner-compatible manifest.
    manifest = {
        "batch_id": state["batch_ids"][0],
        "batch_ids": state["batch_ids"],
        "provider": args.provider,
        "model": args.model,
        "run_id": args.run_id,
        "prompts_dir": str(prompts_dir.resolve()),
        "n_requests": total,
        "n_chunks": len(state["batch_ids"]),
        "sub_sizes": state["sub_sizes"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "last_chunk_submitted_at": datetime.now(timezone.utc).isoformat(),
        "note": "submitted via scripts/submit_gemini_subchunks.py (sandbox-safe sub-chunks)",
    }
    if state.get("adopted"):
        manifest["adopted_orphans"] = state["adopted"]
    if args.replace_user_message:
        manifest["replaced_user_message"] = args.replace_user_message
    if args.variants:
        manifest["variant_filter"] = args.variants
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"DONE — all {total} requests submitted across "
          f"{len(state['batch_ids'])} batches.\nManifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
