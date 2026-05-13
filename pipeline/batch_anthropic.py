"""
batch_anthropic.py — Anthropic Messages Batches adapter.
=========================================================

Wraps `anthropic.messages.batches` for USVB canon runs. 50% discount
on the real-time price; 24-hour completion SLA in practice (often
much less).

Usage (typical):

    from pipeline.batch_common import BatchRequest, make_custom_id
    from pipeline.batch_anthropic import (
        AnthropicBatchAdapter, build_requests_from_prompts,
    )

    requests = build_requests_from_prompts(
        prompts_dir="generated/canon_unified",
        run_id="canon_unified_20260501",
        model="claude-haiku-4-5-20251001",
    )
    adapter = AnthropicBatchAdapter()
    batch_id = adapter.submit(requests, dry_run=False)
    while not (status := adapter.poll(batch_id)).is_done():
        print(status); time.sleep(60)
    results = adapter.fetch_results(batch_id)

API details: https://docs.anthropic.com/en/api/creating-message-batches
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .batch_common import (
    BatchRequest, BatchResult, BatchStatus, write_jsonl, read_jsonl,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Anthropic batches take `system` as a separate field (same as the
    real-time API). Pull it out of the message list."""
    system = None
    out: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return system, out


def _request_to_anthropic_dict(req: BatchRequest) -> dict[str, Any]:
    """Translate a BatchRequest to the Anthropic batch entry shape:

        {
          "custom_id": str,
          "params": {
            "model": str,
            "max_tokens": int,
            "system": str (optional),
            "messages": [{"role": "user"|"assistant", "content": str}, ...],
            ...other model_params
          }
        }
    """
    system, msgs = _split_system(req.messages)
    params: dict[str, Any] = {
        "model": req.model,
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "messages": msgs,
    }
    if system:
        params["system"] = system
    # Pass-through for any extra params (top_k, stop_sequences, etc.)
    for k, v in req.extra_params.items():
        params.setdefault(k, v)
    return {"custom_id": req.custom_id, "params": params}


def _result_to_or_shaped_response(individual: Any) -> tuple[str, dict[str, Any] | None, str | None, int, int]:
    """Convert an Anthropic MessageBatchIndividualResponse to the
    (status, OR-shaped response dict, error, in_tok, out_tok) tuple.

    Status mapping:
      result.type == "succeeded" → ("ok", {...}, None, ...)
      result.type == "errored"   → ("error", None, repr(error), 0, 0)
      result.type == "canceled"  → ("cancelled", None, "canceled by API", 0, 0)
      result.type == "expired"   → ("expired", None, "expired before processing", 0, 0)
    """
    res = individual.result
    rtype = getattr(res, "type", None)
    if rtype == "succeeded":
        msg = res.message
        text = "".join(
            getattr(b, "text", "") for b in msg.content
            if getattr(b, "type", None) == "text"
        )
        usage = msg.usage
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        return "ok", {
            "id": getattr(msg, "id", ""),
            "model": getattr(msg, "model", ""),
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": getattr(msg, "stop_reason", "") or "",
            }],
            "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
        }, None, in_tok, out_tok
    if rtype == "errored":
        err = getattr(res, "error", None)
        return "error", None, f"{type(err).__name__ if err else 'AnthropicBatchError'}: {err}", 0, 0
    if rtype == "canceled":
        return "cancelled", None, "canceled by API", 0, 0
    if rtype == "expired":
        return "expired", None, "expired before processing window closed", 0, 0
    return "error", None, f"Unknown result type: {rtype!r}", 0, 0


# ──────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────
class AnthropicBatchAdapter:
    """Submits + polls + fetches Anthropic Messages Batches.

    Uses the synchronous SDK. The async SDK is also available but the
    batch lifecycle (submit-then-poll-with-sleep) is fundamentally
    synchronous from our perspective, and using the sync client keeps
    the dependency surface minimal.
    """

    PROVIDER_NAME = "anthropic"

    # Anthropic enforces:
    #   max 100k requests per batch
    #   max 256 MB total request body size
    # We split larger lists into chunks before submission.
    MAX_REQUESTS_PER_BATCH = 100_000
    MAX_TOTAL_BYTES = 256 * 1024 * 1024

    def __init__(self, api_key: str | None = None):
        # Lazy client init: nothing here needs the SDK or a key.
        # The actual client is built on first API call so dry-run
        # paths can run without anthropic installed or the key set.
        self._api_key = api_key
        self._client = None
        self._anthropic = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic SDK not installed. "
                "Run: pip install anthropic --break-system-packages"
            ) from e
        self._anthropic = anthropic
        key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Put it in .env or env."
            )
        self._client = anthropic.Anthropic(api_key=key)

    # ───── submission ────────────────────────────────────────────────
    def estimate_request_bytes(self, requests: list[BatchRequest]) -> int:
        """Rough estimate of the JSONL body size for sanity-checking
        against MAX_TOTAL_BYTES."""
        return sum(len(json.dumps(_request_to_anthropic_dict(r))) for r in requests)

    def submit(
        self,
        requests: list[BatchRequest],
        *,
        dry_run: bool = False,
        dry_run_path: Path | None = None,
    ) -> str:
        """Submit a batch. Returns the batch_id.

        If `dry_run=True`, writes the would-be-submitted JSONL to
        `dry_run_path` (default `./batch_anthropic_dryrun.jsonl`) and
        returns a fake id `"DRYRUN-<n>"` instead of hitting the API.
        """
        if not requests:
            raise ValueError("Empty request list")
        if len(requests) > self.MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"{len(requests)} requests exceeds Anthropic batch cap "
                f"of {self.MAX_REQUESTS_PER_BATCH}. Split before calling."
            )
        body_bytes = self.estimate_request_bytes(requests)
        if body_bytes > self.MAX_TOTAL_BYTES:
            raise ValueError(
                f"Estimated body size {body_bytes:,} bytes > "
                f"Anthropic cap of {self.MAX_TOTAL_BYTES:,}. Split before calling."
            )
        # Verify all custom_ids unique
        cids = [r.custom_id for r in requests]
        if len(cids) != len(set(cids)):
            raise ValueError("Duplicate custom_ids in request list — must be unique within a batch.")

        anthropic_requests = [_request_to_anthropic_dict(r) for r in requests]

        if dry_run:
            out = dry_run_path or Path("./batch_anthropic_dryrun.jsonl")
            write_jsonl(out, anthropic_requests)
            print(f"[dry-run] would submit {len(anthropic_requests)} requests "
                  f"({body_bytes:,} body bytes) → wrote {out}")
            return f"DRYRUN-{len(anthropic_requests)}"

        # Convert dict requests to SDK Request objects. The SDK accepts
        # plain dicts in 0.97+; passing as-is works.
        self._ensure_client()
        batch = self._client.messages.batches.create(requests=anthropic_requests)
        return batch.id

    # ───── polling ───────────────────────────────────────────────────
    def poll(self, batch_id: str) -> BatchStatus:
        """Snapshot of progress. State mapping:

          processing   → "in_progress"
          ended        → "ended"
          canceling    → "in_progress"
          errored      → "errored"
          expired      → "ended"  (some requests may still have results)
        """
        if batch_id.startswith("DRYRUN-"):
            n = int(batch_id.split("-", 1)[1])
            return BatchStatus(state="ended", n_total=n, n_succeeded=n)
        self._ensure_client()
        b = self._client.messages.batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        n_proc = int(getattr(counts, "processing", 0) or 0) if counts else 0
        n_ok = int(getattr(counts, "succeeded", 0) or 0) if counts else 0
        n_err = int(getattr(counts, "errored", 0) or 0) if counts else 0
        n_can = int(getattr(counts, "canceled", 0) or 0) if counts else 0
        n_exp = int(getattr(counts, "expired", 0) or 0) if counts else 0
        n_total = n_proc + n_ok + n_err + n_can + n_exp
        status_map = {
            "in_progress": "in_progress",
            "canceling": "in_progress",
            "ended": "ended",
            "errored": "errored",
        }
        state = status_map.get(getattr(b, "processing_status", "in_progress"), "in_progress")
        return BatchStatus(
            state=state,
            n_total=n_total,
            n_succeeded=n_ok,
            n_failed=n_err + n_can + n_exp,
            n_pending=n_proc,
            raw={
                "processing_status": getattr(b, "processing_status", ""),
                "ended_at": str(getattr(b, "ended_at", "")),
            },
        )

    def wait_until_done(
        self, batch_id: str, *, poll_every_s: int = 60, max_wait_s: int = 24 * 3600
    ) -> BatchStatus:
        """Block until the batch is in a terminal state or timeout.
        Caller should generally NOT use this in the same shell that
        launched the batch — it's fine for test scripts but for
        production use, separate submit/poll/fetch into different
        invocations so the launch shell can exit."""
        start = time.time()
        last = None
        while time.time() - start < max_wait_s:
            last = self.poll(batch_id)
            if last.is_done():
                return last
            time.sleep(poll_every_s)
        if last is None:
            last = self.poll(batch_id)
        return last

    # ───── result fetch ──────────────────────────────────────────────
    def fetch_results(self, batch_id: str) -> list[BatchResult]:
        """Fetch all results. Streams server-side. The returned list
        preserves API order, which matches submission order."""
        if batch_id.startswith("DRYRUN-"):
            return []
        self._ensure_client()
        out: list[BatchResult] = []
        for individual in self._client.messages.batches.results(batch_id):
            cid = getattr(individual, "custom_id", "") or ""
            status, response, error, in_tok, out_tok = _result_to_or_shaped_response(individual)
            out.append(BatchResult(
                custom_id=cid,
                status=status,
                response=response,
                error=error,
                input_tokens=in_tok,
                output_tokens=out_tok,
            ))
        return out


# ──────────────────────────────────────────────────────────────────────
# Convenience: build BatchRequests from a generated/ prompts dir
# ──────────────────────────────────────────────────────────────────────
def build_requests_from_prompts(
    prompts_dir: str | Path,
    *,
    run_id: str,
    model: str,
    max_tokens: int = 10000,
    temperature: float = 1.0,
) -> list[BatchRequest]:
    """Walk a `generated/<preset>/` directory and emit one
    BatchRequest per .json file. Each prompt file is expected to have:

        {
          "system_prompt": str,
          "user_message": str,
          "metadata": {
            "scenario_id": str,
            "evidence_variant": str,
            "permutation": str,
            ...
          }
        }
    """
    from .batch_common import make_custom_id

    p = Path(prompts_dir)
    if not p.exists():
        raise FileNotFoundError(f"prompts_dir not found: {p}")

    requests: list[BatchRequest] = []
    for f in sorted(p.glob("*.json")):
        # Skip the renderer's manifest sidecar — it lives alongside
        # the prompt files but isn't itself a prompt.
        if f.name == "manifest.json":
            continue
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        meta = d.get("metadata", {})
        # A real prompt file must carry a scenario_id and evidence
        # variant in its metadata. Anything missing → not a prompt.
        if "scenario_id" not in meta or "evidence_variant" not in meta:
            continue
        sid = meta["scenario_id"]
        variant = meta["evidence_variant"]
        perm = str(meta.get("permutation") or "0")
        # Differentiate multi-axis cells (length / placement / draw / scenario-perm)
        # so the custom_id stays unique within a preset.
        for k in ("draw_idx", "length_idx", "placement_idx", "scenario_perm"):
            v = meta.get(k)
            if v is not None:
                perm = f"{perm}-{k[0]}{v}"
        custom_id = make_custom_id(run_id, sid, variant, perm)
        # Build OR/OpenAI-shaped messages list.
        msgs: list[dict[str, str]] = []
        sysp = d.get("system_prompt", "")
        if sysp:
            msgs.append({"role": "system", "content": sysp})
        msgs.append({"role": "user", "content": d.get("user_message", "")})
        requests.append(BatchRequest(
            custom_id=custom_id,
            model=model,
            messages=msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        ))
    return requests
