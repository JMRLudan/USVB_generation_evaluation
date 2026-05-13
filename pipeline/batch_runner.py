"""
batch_runner.py — top-level orchestrator for provider batch runs.
==================================================================

Glue between the per-provider adapters (batch_anthropic, batch_openai,
batch_gemini) and the existing eval pipeline. A single CLI entry
point lets you submit a batch, wait for it, and fetch results into a
results.tsv compatible with the rest of the repo (viewer, judge
re-scoring, vigilance breakdown).

Modes:
  --dry-run       Build the JSONL bodies that WOULD be submitted, write
                  them to disk, exit without calling the provider API.
                  Use this to inspect message shapes before paying.

  --submit        Submit the batch and print the batch_id. Returns
                  immediately. Use this from the user's terminal so
                  the launch shell can exit while the batch runs.

  --status BATCH  Poll a batch by id and print the snapshot.

  --fetch BATCH   Fetch finished results, write a results.tsv to
                  data/runs/<preset>/<model_dir>/<run_id>/results.tsv,
                  and append cost rows to api_logs/costs.csv.

The orchestrator does not poll-and-fetch in one shot by default
because the launching shell would have to stay alive for hours. Use
the schedule skill if you want auto-fetch on completion.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .batch_common import (
    BatchRequest, BatchResult, BatchStatus,
    parse_custom_id, infer_provider, synthesize_cost_rows,
)


# ──────────────────────────────────────────────────────────────────────
# Adapter dispatch
# ──────────────────────────────────────────────────────────────────────
def get_adapter(provider: str):
    """Lazy-import the provider adapter so a missing optional SDK
    doesn't block the other providers from running."""
    if provider == "anthropic":
        from .batch_anthropic import AnthropicBatchAdapter
        return AnthropicBatchAdapter()
    if provider == "openai":
        from .batch_openai import OpenAIBatchAdapter
        return OpenAIBatchAdapter()
    if provider == "gemini":
        from .batch_gemini import GeminiBatchAdapter
        return GeminiBatchAdapter()
    raise ValueError(f"Unknown provider {provider!r}")


def build_requests(
    prompts_dir: Path,
    *,
    run_id: str,
    model: str,
    max_tokens: int = 10000,
    temperature: float = 1.0,
) -> list[BatchRequest]:
    """Provider-agnostic request builder. Identical to
    `batch_anthropic.build_requests_from_prompts`, but kept here so
    runners that don't import the Anthropic adapter can still use it."""
    from .batch_anthropic import build_requests_from_prompts
    return build_requests_from_prompts(
        prompts_dir, run_id=run_id, model=model,
        max_tokens=max_tokens, temperature=temperature,
    )


# ──────────────────────────────────────────────────────────────────────
# Results sink — write to the same TSV shape the existing pipeline uses
# ──────────────────────────────────────────────────────────────────────
# Mirrors data/runs/<preset>/<model>/<run_id>/results.tsv as written by
# multi_model_runner. Order matches the on-disk header.
RESULTS_HEADER = [
    "run_id", "condition",
    "scenario_id", "evidence_variant", "permutation",
    "expected_answer",
    "raw_response",
    "recommendation", "constraint_mentioned", "sufficiently_modified",
    "explanation",
    "parse_error",
    "vigilance_refuse_only", "abstain_type",
    "choice_correct", "abstained",
    "input_tokens", "output_tokens",
    "judge_input_tokens", "judge_output_tokens",
    "latency_ms",
]

# Batch-judge rows start with parse_error=1 + explanation="(judge error)"
# so a downstream judge pass picks them up.
_JUDGE_PENDING_EXPLANATION = "(judge error — pending batch judge pass)"


def _load_prompt_metadata(prompts_dir: Path) -> dict[str, dict]:
    """Build {custom_id_tail → metadata} for joining batch results back
    to scenario/condition/expected_answer.

    The custom_id tail used here is `(scenario_id, evidence_variant,
    permutation)` — same triple that uniquely keys a prompt within a
    preset.
    """
    out: dict[tuple[str, str, str], dict] = {}
    for f in sorted(prompts_dir.glob("*.json")):
        if f.name == "manifest.json":
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
        except json.JSONDecodeError:
            continue
        meta = d.get("metadata") or {}
        if "scenario_id" not in meta or "evidence_variant" not in meta:
            continue
        key = (
            meta["scenario_id"],
            meta["evidence_variant"],
            str(meta.get("permutation") or "0"),
        )
        out[key] = meta
    return out


def write_results_tsv(
    results: list[BatchResult],
    *,
    out_path: Path,
    model: str,
    run_id: str,
    condition: str,
    prompts_dir: Path | None = None,
) -> None:
    """Write a results.tsv with the same 24-col schema multi_model_runner
    produces. Subject-side fields populated from the BatchResult; judge
    fields left blank with `parse_error=1` so a downstream judge pass
    fills them in.

    `prompts_dir` is optional — when provided, we pull
    `expected_answer`, `condition`, and a few other fields straight from
    the prompt-file metadata so the row is byte-identical to a
    real-time write. When omitted, those fields fall back to ``""`` and
    `condition` from the arg.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_lookup: dict[tuple[str, str, str], dict] = {}
    if prompts_dir is not None:
        meta_lookup = _load_prompt_metadata(prompts_dir)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=RESULTS_HEADER, delimiter="\t",
            quoting=csv.QUOTE_MINIMAL, extrasaction="ignore",
        )
        w.writeheader()
        for r in results:
            try:
                ids = parse_custom_id(r.custom_id)
            except ValueError:
                ids = {
                    "run_id": run_id, "scenario_id": r.custom_id,
                    "variant": "?", "perm": "?",
                }
            sid = ids.get("scenario_id", "")
            variant = ids.get("variant", "")
            perm = ids.get("perm", "")

            # Strip the synthetic `-d{n}-l{n}-pX` suffix from perm to
            # match the meta-lookup key (the renderer's metadata uses
            # the bare `c{n}_a{n}` permutation form).
            base_perm = perm.split("-", 1)[0] if "-" in perm else perm

            meta = meta_lookup.get((sid, variant, base_perm), {})
            expected = meta.get("expected_answer", "")
            cond = meta.get("condition", condition)

            content = ""
            if r.response:
                try:
                    content = r.response["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    content = ""

            # Match how the real-time path stores raw_response — newlines
            # escaped so the TSV stays valid for naive split-on-\n
            # parsers. multi_model_runner does the same thing in
            # `apply_judge_result`.
            content_escaped = content.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")

            row = {
                "run_id": run_id,
                "condition": cond,
                "scenario_id": sid,
                "evidence_variant": variant,
                "permutation": perm,
                "expected_answer": expected,
                "raw_response": (f"ERROR: {r.error}" if r.status != "ok" else content_escaped),
                "recommendation": "",
                "constraint_mentioned": "",
                "sufficiently_modified": "",
                "explanation": _JUDGE_PENDING_EXPLANATION if r.status == "ok" else "(subject error)",
                "parse_error": "1",
                "vigilance_refuse_only": "",
                "abstain_type": "",
                "choice_correct": "",
                "abstained": "",
                "input_tokens": r.input_tokens or 0,
                "output_tokens": r.output_tokens or 0,
                "judge_input_tokens": 0,
                "judge_output_tokens": 0,
                "latency_ms": "",
            }
            w.writerow(row)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
def cmd_dry_run(args: argparse.Namespace) -> int:
    provider = args.provider or infer_provider(args.model)
    requests = build_requests(
        Path(args.prompts_dir), run_id=args.run_id, model=args.model,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    print(f"Built {len(requests)} BatchRequests for model={args.model}, "
          f"provider={provider}")
    if args.limit and args.limit < len(requests):
        requests = requests[:args.limit]
        print(f"  truncated to first {args.limit} for dry-run")

    adapter = get_adapter(provider)
    out_path = Path(args.output) if args.output else Path(
        f"./batch_{provider}_dryrun.jsonl"
    )
    bid = adapter.submit(requests, dry_run=True, dry_run_path=out_path)
    print(f"Dry-run batch_id (synthetic): {bid}")
    print(f"JSONL body written to: {out_path}")
    print(f"  inspect first 1-2 lines to verify shape:")
    with open(out_path) as f:
        for _ in range(2):
            line = f.readline()
            if not line:
                break
            print(f"    {line.strip()[:300]}…")
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    provider = args.provider or infer_provider(args.model)
    requests = build_requests(
        Path(args.prompts_dir), run_id=args.run_id, model=args.model,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )
    print(f"Built {len(requests)} BatchRequests for "
          f"model={args.model}, provider={provider}")

    adapter = get_adapter(provider)

    # Chunk if necessary (canon_unified needs 2 chunks under Anthropic's
    # 256MB cap; fits in 1 for canon_no_distractor). The optional
    # --max-mb-per-chunk override lets sandbox callers force smaller
    # chunks so each upload fits within a per-call timeout budget.
    from .batch_common import chunk_requests
    if provider == "anthropic":
        from .batch_anthropic import _request_to_anthropic_dict
        size_fn = lambda r: len(json.dumps(_request_to_anthropic_dict(r))) + 1
        max_count = 100_000
        max_bytes = 256 * 1024 * 1024
    elif provider == "openai":
        from .batch_openai import _request_to_openai_jsonl_record
        size_fn = lambda r: len(json.dumps(_request_to_openai_jsonl_record(r))) + 1
        max_count = 50_000
        max_bytes = 200 * 1024 * 1024
    else:
        from .batch_gemini import _request_to_gemini_jsonl_record
        size_fn = lambda r: len(json.dumps(_request_to_gemini_jsonl_record(r))) + 1
        max_count = 50_000
        max_bytes = 100 * 1024 * 1024

    if getattr(args, "max_mb_per_chunk", None):
        cap = int(args.max_mb_per_chunk * 1024 * 1024)
        if cap < max_bytes:
            max_bytes = cap
            print(f"  (override: max bytes per chunk = {cap:,} = "
                  f"{args.max_mb_per_chunk} MB)")

    chunks = chunk_requests(
        requests, max_count=max_count, max_bytes=max_bytes,
        bytes_per_request_fn=size_fn,
    )
    print(f"Split into {len(chunks)} chunk(s): "
          f"{[len(c) for c in chunks]}")

    # Resume / per-chunk flow. The manifest accumulates batch_ids
    # across calls; each invocation submits chunks not yet logged.
    leaf = Path(args.prompts_dir).name
    manifest_dir = Path("./batch_manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.run_id}__{leaf}.json"
    existing = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {}
    )
    batch_ids: list[str] = list(existing.get("batch_ids") or [])
    already = len(batch_ids)
    if already:
        print(f"  manifest exists — {already} chunk(s) already submitted")

    target_idx = getattr(args, "chunk_index", None)
    if target_idx is not None:
        if target_idx < 0 or target_idx >= len(chunks):
            print(f"ERROR: --chunk-index {target_idx} out of range "
                  f"[0, {len(chunks)})")
            return 2
        to_submit = [(target_idx, chunks[target_idx])]
    else:
        to_submit = [(i, c) for i, c in enumerate(chunks) if i >= already]

    for i, chunk in to_submit:
        suffix = f"  [chunk {i+1}/{len(chunks)}]" if len(chunks) > 1 else ""
        print(f"Submitting {len(chunk)} requests…{suffix}")
        bid = adapter.submit(chunk, dry_run=False)
        print(f"  BATCH_ID: {bid}")
        # Append + persist immediately so a mid-run timeout doesn't
        # lose the batch_id from a chunk that did succeed.
        if target_idx is not None:
            # Rewrite at exact index — leave gap as None if needed.
            while len(batch_ids) <= i:
                batch_ids.append(None)
            batch_ids[i] = bid
        else:
            batch_ids.append(bid)
        existing.update({
            "batch_id": next((b for b in batch_ids if b), None),
            "batch_ids": batch_ids,
            "provider": provider,
            "model": args.model,
            "run_id": args.run_id,
            "prompts_dir": str(Path(args.prompts_dir).resolve()),
            "n_requests": len(requests),
            "n_chunks": len(chunks),
            "submitted_at": (existing.get("submitted_at")
                             or datetime.now(timezone.utc).isoformat()),
            "last_chunk_submitted_at": datetime.now(timezone.utc).isoformat(),
        })
        manifest_path.write_text(json.dumps(existing, indent=2))

    print(f"Manifest: {manifest_path}")
    print()
    submitted = sum(1 for b in batch_ids if b)
    print(f"Manifest now has {submitted}/{len(chunks)} chunks submitted:")
    for i, bid in enumerate(batch_ids):
        print(f"  chunk {i}: {bid or '(pending)'}")
    if submitted < len(chunks):
        print()
        print(f"To submit the remaining {len(chunks) - submitted} chunk(s), "
              f"re-run the same command (manifest will resume).")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    # Accept either a literal batch_id or a manifest stem (e.g.
    # "canon_haiku_20260501_120000__canon_unified") that resolves to
    # one or more chunked batch_ids.
    manifest = _load_manifest(args.batch_id)
    if manifest and manifest.get("batch_ids"):
        ids = manifest["batch_ids"]
        provider = args.provider or manifest.get("provider") or "anthropic"
    else:
        ids = [args.batch_id]
        provider = args.provider or _provider_from_manifest(args.batch_id) or "anthropic"
    adapter = get_adapter(provider)
    out = []
    for bid in ids:
        status = adapter.poll(bid)
        out.append({
            "batch_id": bid,
            "provider": provider,
            "state": status.state,
            "n_total": status.n_total,
            "n_succeeded": status.n_succeeded,
            "n_failed": status.n_failed,
            "n_pending": status.n_pending,
            "raw": status.raw,
        })
    print(json.dumps(out if len(out) > 1 else out[0], indent=2, default=str))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.batch_id)
    provider = args.provider or (manifest or {}).get("provider") or "anthropic"
    model = args.model or (manifest or {}).get("model") or "?"
    run_id = args.run_id or (manifest or {}).get("run_id") or "?"
    prompts_dir_str = args.prompts_dir or (manifest or {}).get("prompts_dir")
    prompts_dir = Path(prompts_dir_str) if prompts_dir_str else None
    # Preset defaults to the prompts_dir leaf name so the fetched
    # results land under data/runs/<preset>/...  (matches the layout
    # canon_no_distractor/canon_unified expects).
    preset = args.preset or (prompts_dir.name if prompts_dir else "unknown")
    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"./data/runs/{preset}/{model.replace('/', '_')}/{run_id}"
    )

    # Collect batch_ids — single id or chunked list from manifest.
    if manifest and manifest.get("batch_ids"):
        ids = manifest["batch_ids"]
    else:
        ids = [args.batch_id]

    adapter = get_adapter(provider)
    results: list = []
    for bid in ids:
        print(f"Fetching results for {bid} ({provider})…")
        chunk = adapter.fetch_results(bid)
        print(f"  fetched {len(chunk)} results")
        results.extend(chunk)

    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = len(results) - n_ok
    print(f"  status: {n_ok} ok / {n_err} non-ok")

    out_path = out_dir / "results.tsv"
    write_results_tsv(
        results,
        out_path=out_path, model=model,
        run_id=run_id, condition=preset, prompts_dir=prompts_dir,
    )
    print(f"  wrote {out_path}")

    # Cost + raw-I/O logging via the shared OpenRouterClient. Every
    # row gets its own line in api_logs/costs.csv, tagged with
    # provider="<anthropic|openai|gemini>_batch" so it's distinguishable
    # from real-time spend in audit queries. Batch discount (50%) is
    # applied to the live OR pricing.
    try:
        from .openrouter_client import OpenRouterClient
        client = OpenRouterClient(run_id=f"batch_{run_id}")
        total = client.log_batch_results(
            results, model=model, provider=provider,
        )
        print(f"  logged {len(results)} cost rows → "
              f"api_logs/costs.csv  (batch total ${total:.2f})")
    except Exception as e:
        print(f"  WARN: cost logging failed ({type(e).__name__}: {e}). "
              f"Results written, but costs.csv not updated.")

    return 0


def _load_manifest(batch_id_or_stem: str) -> dict[str, Any] | None:
    """Load a manifest by either:
      - exact stem ``<run_id>__<preset>``  (filename match, multi-chunk)
      - a batch_id contained in any manifest's `batch_ids` list
    """
    manifest_dir = Path("./batch_manifests")
    direct = manifest_dir / f"{batch_id_or_stem}.json"
    if direct.exists():
        return json.loads(direct.read_text())
    # Otherwise scan manifests for a matching batch_id in batch_ids.
    if manifest_dir.exists():
        for p in manifest_dir.glob("*.json"):
            try:
                m = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            if batch_id_or_stem in (m.get("batch_ids") or [m.get("batch_id")]):
                return m
    return None


def _provider_from_manifest(batch_id: str) -> str | None:
    m = _load_manifest(batch_id)
    return m.get("provider") if m else None


def main() -> int:
    # Auto-load .env so the per-provider adapters can pick up keys
    # without a separate sourcing step. The OpenRouterClient does this
    # eagerly on construction, but the batch adapters use the provider
    # SDKs directly and would otherwise see only the shell environment.
    try:
        from .openrouter_client import _load_dotenv
        _load_dotenv(Path(__file__).resolve().parent.parent)
    except Exception:
        pass

    p = argparse.ArgumentParser(
        description="USVB batch runner — submit/poll/fetch provider batch jobs."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=["anthropic", "openai", "gemini"],
                        help="Override auto-detection from model id.")
    common.add_argument("--model", help="Model id (e.g. claude-haiku-4-5-20251001).")
    common.add_argument("--run-id", help="Run id tag for cost log + custom_ids.")

    p_dry = sub.add_parser("dry-run", parents=[common], help="Build JSONL without submitting.")
    p_dry.add_argument("--prompts-dir", required=True)
    p_dry.add_argument("--output", help="Where to write the dry-run JSONL.")
    p_dry.add_argument("--limit", type=int, default=10,
                       help="Truncate to first N requests for dry-run readability.")
    p_dry.add_argument("--max-tokens", type=int, default=10000)
    p_dry.add_argument("--temperature", type=float, default=1.0)
    p_dry.set_defaults(func=cmd_dry_run)

    p_sub = sub.add_parser("submit", parents=[common], help="Submit a batch.")
    p_sub.add_argument("--prompts-dir", required=True)
    p_sub.add_argument("--max-tokens", type=int, default=10000)
    p_sub.add_argument("--temperature", type=float, default=1.0)
    p_sub.add_argument("--max-mb-per-chunk", type=float, default=None,
                       help="Optional cap on per-chunk JSONL size in MB. "
                            "Useful when running from a sandbox with a "
                            "per-call timeout budget.")
    p_sub.add_argument("--chunk-index", type=int, default=None,
                       help="Submit only the Nth chunk (0-indexed). When "
                            "omitted, submits all chunks not yet in the "
                            "manifest's batch_ids list.")
    p_sub.set_defaults(func=cmd_submit)

    p_st = sub.add_parser("status", parents=[common], help="Poll a batch.")
    p_st.add_argument("batch_id")
    p_st.set_defaults(func=cmd_status)

    p_fe = sub.add_parser("fetch", parents=[common], help="Fetch finished results.")
    p_fe.add_argument("batch_id")
    p_fe.add_argument("--preset")
    p_fe.add_argument("--out-dir")
    p_fe.add_argument("--prompts-dir",
                      help="Source prompts dir for metadata join (defaults "
                           "to the path stored in the manifest).")
    p_fe.set_defaults(func=cmd_fetch)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
