"""
batch_openai.py — OpenAI Batch API adapter.
============================================

Wraps OpenAI's Batch API for USVB canon runs. 50% discount on the
real-time price; 24-hour completion window.

Lifecycle differs slightly from Anthropic's batches: OpenAI requires
uploading a JSONL file first via the Files API, then referring to it
in the batch creation call.

Usage:

    from pipeline.batch_openai import OpenAIBatchAdapter

    adapter = OpenAIBatchAdapter()
    batch_id = adapter.submit(requests, dry_run=False)
    while not (status := adapter.poll(batch_id)).is_done():
        print(status); time.sleep(60)
    results = adapter.fetch_results(batch_id)

API details: https://platform.openai.com/docs/guides/batch

NOTE: requires the `openai` Python SDK. Install with
``pip install openai --break-system-packages``.
"""

from __future__ import annotations

import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from .batch_common import (
    BatchRequest, BatchResult, BatchStatus, write_jsonl,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _request_to_openai_jsonl_record(req: BatchRequest) -> dict[str, Any]:
    """Translate a BatchRequest to one row of OpenAI's batch JSONL.

    Format:
        {
          "custom_id": "...",
          "method": "POST",
          "url": "/v1/chat/completions",
          "body": {
            "model": "...",
            "messages": [...],
            "max_tokens": int,
            "temperature": float,
            ...
          }
        }
    """
    # Strip the OpenRouter "openai/" prefix if present — Batch API
    # expects bare ids ("gpt-4.1", "gpt-5-nano", etc.).
    model = req.model
    if model.startswith("openai/"):
        model = model[len("openai/"):]
    body: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
    }
    # GPT-5 family quirks (verified 2026-05-01 against gpt-5-mini):
    #   * `max_tokens` is rejected — must use `max_completion_tokens`.
    #     We send `max_completion_tokens` for any gpt-5* model and
    #     fall back to `max_tokens` for older 4.x ids.
    #   * `temperature` is locked to the default 1.0; non-1.0 values
    #     return "Unsupported value". We omit temperature for gpt-5*
    #     when it's 1.0 (the default) and pass it through otherwise so
    #     the API will surface a clear error rather than us silently
    #     swallowing a non-default request.
    is_gpt5 = model.startswith("gpt-5")
    if is_gpt5:
        body["max_completion_tokens"] = req.max_tokens
        if abs(req.temperature - 1.0) > 1e-9:
            body["temperature"] = req.temperature  # will error out — informative
    else:
        body["max_tokens"] = req.max_tokens
        body["temperature"] = req.temperature
    body.update(req.extra_params)
    return {
        "custom_id": req.custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _openai_response_line_to_result(line: dict[str, Any]) -> BatchResult:
    """Convert one line of the output JSONL to a BatchResult.

    Output line shape:
        {
          "id": "...",
          "custom_id": "...",
          "response": {
             "status_code": 200,
             "request_id": "...",
             "body": {
               "id": "...", "model": "...", "choices": [...],
               "usage": {"prompt_tokens": int, "completion_tokens": int, ...}
             }
          },
          "error": null
        }
    """
    cid = line.get("custom_id", "")
    resp = line.get("response") or {}
    err = line.get("error")
    if err:
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return BatchResult(custom_id=cid, status="error", error=str(msg))
    body = resp.get("body") if isinstance(resp, dict) else None
    if not body:
        return BatchResult(custom_id=cid, status="error", error="empty response body")
    usage = body.get("usage", {}) or {}
    in_tok = int(usage.get("prompt_tokens", 0) or 0)
    out_tok = int(usage.get("completion_tokens", 0) or 0)
    # Body already has `choices`/`usage` in the shape downstream code expects.
    return BatchResult(
        custom_id=cid,
        status="ok",
        response=body,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


# ──────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────
class OpenAIBatchAdapter:
    """Submit + poll + fetch OpenAI Batch jobs.

    OpenAI's batch limits:
      * 50 000 requests per file
      * 200 MB per file
      * 50 000 enqueued tokens per organization tier (varies; OpenAI
        admin console shows actual cap)

    For USVB canon (~6 366 unified prompts), one file fits comfortably.
    """

    PROVIDER_NAME = "openai"
    MAX_REQUESTS_PER_FILE = 50_000
    MAX_FILE_BYTES = 200 * 1024 * 1024
    BATCH_ENDPOINT = "/v1/chat/completions"
    COMPLETION_WINDOW = "24h"

    def __init__(self, api_key: str | None = None):
        # Lazy client init — see batch_anthropic for rationale.
        self._api_key = api_key
        self._client = None
        self._openai = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "openai SDK not installed. "
                "Run: pip install openai --break-system-packages"
            ) from e
        self._openai = openai
        key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Put it in .env or env."
            )
        self._client = openai.OpenAI(api_key=key)

    # ───── submission ────────────────────────────────────────────────
    def estimate_request_bytes(self, requests: list[BatchRequest]) -> int:
        return sum(
            len(json.dumps(_request_to_openai_jsonl_record(r))) + 1  # +newline
            for r in requests
        )

    def submit(
        self,
        requests: list[BatchRequest],
        *,
        dry_run: bool = False,
        dry_run_path: Path | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Submit a batch. Returns the batch_id (or "DRYRUN-N" if
        dry_run=True).

        OpenAI requires a 2-step process: upload file → create batch.
        Both happen inside this call.
        """
        if not requests:
            raise ValueError("Empty request list")
        if len(requests) > self.MAX_REQUESTS_PER_FILE:
            raise ValueError(
                f"{len(requests)} requests exceeds OpenAI per-file cap "
                f"of {self.MAX_REQUESTS_PER_FILE}. Split before calling."
            )
        body_bytes = self.estimate_request_bytes(requests)
        if body_bytes > self.MAX_FILE_BYTES:
            raise ValueError(
                f"Estimated file size {body_bytes:,} bytes > "
                f"OpenAI cap of {self.MAX_FILE_BYTES:,}."
            )
        cids = [r.custom_id for r in requests]
        if len(cids) != len(set(cids)):
            raise ValueError("Duplicate custom_ids — must be unique within a batch.")

        records = [_request_to_openai_jsonl_record(r) for r in requests]

        if dry_run:
            out = dry_run_path or Path("./batch_openai_dryrun.jsonl")
            write_jsonl(out, records)
            print(f"[dry-run] would submit {len(records)} requests "
                  f"({body_bytes:,} body bytes) → wrote {out}")
            return f"DRYRUN-{len(records)}"

        self._ensure_client()
        # Build JSONL in memory and upload via Files API.
        # OpenAI's Files SDK accepts a tuple `(filename, file-like, mime)`.
        buf = BytesIO()
        for i, rec in enumerate(records):
            if i > 0:
                buf.write(b"\n")
            buf.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        buf.seek(0)
        uploaded = self._client.files.create(
            file=("usvb_batch.jsonl", buf, "application/jsonl"),
            purpose="batch",
        )

        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint=self.BATCH_ENDPOINT,
            completion_window=self.COMPLETION_WINDOW,
            metadata=metadata or {},
        )
        return batch.id

    # ───── polling ───────────────────────────────────────────────────
    def poll(self, batch_id: str) -> BatchStatus:
        """Snapshot status. OpenAI states:

          validating → in_progress
          in_progress → in_progress
          finalizing → in_progress
          completed → ended
          failed → errored
          expired → ended (some output may exist)
          cancelling → in_progress
          cancelled → cancelled
        """
        if batch_id.startswith("DRYRUN-"):
            n = int(batch_id.split("-", 1)[1])
            return BatchStatus(state="ended", n_total=n, n_succeeded=n)
        self._ensure_client()
        b = self._client.batches.retrieve(batch_id)
        counts = getattr(b, "request_counts", None)
        n_total = int(getattr(counts, "total", 0) or 0) if counts else 0
        n_ok = int(getattr(counts, "completed", 0) or 0) if counts else 0
        n_err = int(getattr(counts, "failed", 0) or 0) if counts else 0
        n_pending = max(0, n_total - n_ok - n_err)
        status_map = {
            "validating": "in_progress",
            "in_progress": "in_progress",
            "finalizing": "in_progress",
            "cancelling": "in_progress",
            "completed": "ended",
            "expired": "ended",
            "failed": "errored",
            "cancelled": "cancelled",
        }
        api_state = getattr(b, "status", "in_progress")
        return BatchStatus(
            state=status_map.get(api_state, "in_progress"),
            n_total=n_total,
            n_succeeded=n_ok,
            n_failed=n_err,
            n_pending=n_pending,
            raw={
                "status": api_state,
                "output_file_id": getattr(b, "output_file_id", None),
                "error_file_id": getattr(b, "error_file_id", None),
                "expires_at": getattr(b, "expires_at", None),
            },
        )

    def wait_until_done(
        self, batch_id: str, *, poll_every_s: int = 60, max_wait_s: int = 24 * 3600
    ) -> BatchStatus:
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
        """Fetch all results, including errored ones from the error
        file if present."""
        if batch_id.startswith("DRYRUN-"):
            return []
        self._ensure_client()
        b = self._client.batches.retrieve(batch_id)
        results: list[BatchResult] = []

        out_id = getattr(b, "output_file_id", None)
        if out_id:
            content = self._client.files.content(out_id)
            text = content.read().decode("utf-8") if hasattr(content, "read") else content.text
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                results.append(_openai_response_line_to_result(json.loads(line)))

        err_id = getattr(b, "error_file_id", None)
        if err_id:
            content = self._client.files.content(err_id)
            text = content.read().decode("utf-8") if hasattr(content, "read") else content.text
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                results.append(_openai_response_line_to_result(json.loads(line)))

        return results
