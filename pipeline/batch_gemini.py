"""
batch_gemini.py — Google Gemini Batch API adapter.
====================================================

Wraps Google's Gemini Batch API for USVB canon runs. 50% discount on
the real-time price; 24-hour completion window.

Two viable backend paths exist for Gemini batch jobs:

  1. The standalone `google-genai` SDK (newer, recommended for
     Generative Language API users): `genai.batches.create(...)`.
  2. The Vertex AI SDK with `BatchPredictionJob` — used when the
     project is on a GCP-billing tenant.

This adapter targets path #1 (`google-genai`) by default since it
matches our existing OR-routed setup. If the project moves to Vertex,
swap the backend at the marked TODO.

Usage:

    from pipeline.batch_gemini import GeminiBatchAdapter

    adapter = GeminiBatchAdapter()
    batch_id = adapter.submit(requests, dry_run=False)
    status = adapter.poll(batch_id)
    results = adapter.fetch_results(batch_id)

API details: https://ai.google.dev/gemini-api/docs/batch-mode

NOTE: requires the `google-genai` Python SDK. Install with
``pip install google-genai --break-system-packages``.
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
def _split_system(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """Gemini takes `systemInstruction` separately from `contents`."""
    system = None
    out: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            out.append(m)
    return system, out


def _messages_to_gemini_contents(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Translate OpenAI-shaped messages to Gemini `contents` shape:

        [
          {"role": "user", "parts": [{"text": "..."}]},
          {"role": "model", "parts": [{"text": "..."}]},
          ...
        ]

    Note Gemini uses "model" instead of "assistant".
    """
    out = []
    for m in messages:
        role = m.get("role", "")
        if role == "system":
            continue  # handled separately
        gemini_role = "user" if role == "user" else "model"
        out.append({
            "role": gemini_role,
            "parts": [{"text": m.get("content", "")}],
        })
    return out


def _request_to_gemini_jsonl_record(req: BatchRequest) -> dict[str, Any]:
    """Translate a BatchRequest to one row of Gemini's batch JSONL.

    Format (Generative Language API):
        {
          "key": "...",
          "request": {
            "contents": [...],
            "systemInstruction": {"parts": [{"text": "..."}]},
            "generationConfig": {
              "temperature": ..., "maxOutputTokens": ...
            }
          }
        }
    """
    system, _ = _split_system(req.messages)
    contents = _messages_to_gemini_contents(req.messages)
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    body.update(req.extra_params)
    return {"key": req.custom_id, "request": body}


def _gemini_response_to_result(line: dict[str, Any]) -> BatchResult:
    """Convert one line of the Gemini output JSONL to a BatchResult.

    Output line shape:
        {
          "key": "...",
          "response": {
            "candidates": [
              {"content": {"parts": [{"text": "..."}], "role": "model"},
               "finishReason": "STOP"}
            ],
            "usageMetadata": {
              "promptTokenCount": int, "candidatesTokenCount": int,
              "totalTokenCount": int
            }
          },
          "error": null
        }
    """
    cid = line.get("key", "")
    err = line.get("error")
    if err:
        return BatchResult(custom_id=cid, status="error", error=str(err))
    resp = line.get("response", {}) or {}
    candidates = resp.get("candidates", []) or []
    if not candidates:
        return BatchResult(custom_id=cid, status="error", error="no candidates in response")
    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    usage = resp.get("usageMetadata", {}) or {}
    in_tok = int(usage.get("promptTokenCount", 0) or 0)
    # Gemini 3.x bills thinking tokens at the output rate (per
    # ai.google.dev/gemini-api/docs/pricing — "Output price (including
    # thinking tokens)"). candidatesTokenCount covers only visible
    # output; thoughtsTokenCount covers internal reasoning. Sum both
    # for accurate billing.
    out_tok = int(usage.get("candidatesTokenCount", 0) or 0) + int(usage.get("thoughtsTokenCount", 0) or 0)
    finish = candidates[0].get("finishReason", "")
    or_shaped = {
        "id": cid,
        "model": resp.get("modelVersion", ""),
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": str(finish),
        }],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }
    return BatchResult(
        custom_id=cid,
        status="ok",
        response=or_shaped,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


# ──────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────
class GeminiBatchAdapter:
    """Submit + poll + fetch Gemini batch jobs via the
    `google-genai` SDK.

    Gemini batch lifecycle is similar to OpenAI's:
      1. Upload a JSONL file via `client.files.upload(...)`
      2. Create a batch via `client.batches.create(model=..., src=fileId)`
      3. Poll via `client.batches.get(name)`
      4. Download via `client.files.download(...)` once dest_file is set

    Limits (as of mid-2026 — verify before launch):
      * 100 MB per JSONL file
      * 50 000 requests per file (model-dependent)
      * 24-hour completion window
    """

    PROVIDER_NAME = "gemini"
    MAX_REQUESTS_PER_FILE = 50_000
    MAX_FILE_BYTES = 100 * 1024 * 1024
    COMPLETION_WINDOW_HOURS = 24

    def __init__(self, api_key: str | None = None):
        # Lazy client init — see batch_anthropic for rationale.
        self._api_key = api_key
        self._client = None
        self._genai = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "google-genai SDK not installed. "
                "Run: pip install google-genai --break-system-packages"
            ) from e
        self._genai = genai
        key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) not set."
            )
        self._client = genai.Client(api_key=key)

    # ───── submission ────────────────────────────────────────────────
    def estimate_request_bytes(self, requests: list[BatchRequest]) -> int:
        return sum(
            len(json.dumps(_request_to_gemini_jsonl_record(r))) + 1
            for r in requests
        )

    def submit(
        self,
        requests: list[BatchRequest],
        *,
        dry_run: bool = False,
        dry_run_path: Path | None = None,
        display_name: str | None = None,
    ) -> str:
        if not requests:
            raise ValueError("Empty request list")
        if len(requests) > self.MAX_REQUESTS_PER_FILE:
            raise ValueError(
                f"{len(requests)} > Gemini per-file cap of {self.MAX_REQUESTS_PER_FILE}."
            )
        body_bytes = self.estimate_request_bytes(requests)
        if body_bytes > self.MAX_FILE_BYTES:
            raise ValueError(
                f"Estimated file size {body_bytes:,} bytes > "
                f"Gemini cap of {self.MAX_FILE_BYTES:,}."
            )
        cids = [r.custom_id for r in requests]
        if len(cids) != len(set(cids)):
            raise ValueError("Duplicate custom_ids in request list — must be unique within a batch.")

        # Gemini batches are pinned to a single model. Verify all
        # requests use the same model id.
        models = {r.model for r in requests}
        if len(models) != 1:
            raise ValueError(
                f"Gemini batch requires a single model; got {sorted(models)!r}. "
                f"Split by model before calling."
            )
        model = models.pop()
        # Strip "google/" prefix if present.
        if model.startswith("google/"):
            model = model[len("google/"):]

        records = [_request_to_gemini_jsonl_record(r) for r in requests]

        if dry_run:
            out = dry_run_path or Path("./batch_gemini_dryrun.jsonl")
            write_jsonl(out, records)
            print(f"[dry-run] would submit {len(records)} requests "
                  f"({body_bytes:,} body bytes) → wrote {out}")
            return f"DRYRUN-{len(records)}"

        self._ensure_client()
        # Upload file then create batch. The google-genai SDK file upload
        # API differs across versions; the snippet below targets the
        # version pinned at module-import time. If file upload semantics
        # change, adjust here.
        buf = BytesIO()
        for i, rec in enumerate(records):
            if i > 0:
                buf.write(b"\n")
            buf.write(json.dumps(rec, ensure_ascii=False).encode("utf-8"))
        buf.seek(0)

        # TODO(verify-on-first-run): file-upload API surface of
        # google-genai 0.4+ accepts a path; for in-memory bytes we may
        # need to write to a tempfile. Keep tempfile fallback.
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="wb"
        ) as tf:
            tf.write(buf.getvalue())
            tmp_path = tf.name
        try:
            uploaded = self._client.files.upload(
                file=tmp_path,
                config={
                    "display_name": display_name or "usvb_batch_input.jsonl",
                    # The .jsonl extension isn't recognized by mimetypes
                    # on Linux, so pass it explicitly. Google's batch
                    # endpoint expects "application/jsonl".
                    "mime_type": "application/jsonl",
                },
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        batch_job = self._client.batches.create(
            model=model,
            src=uploaded.name,
            config={
                "display_name": display_name or "usvb_batch",
            },
        )
        return batch_job.name

    # ───── polling ───────────────────────────────────────────────────
    def poll(self, batch_id: str) -> BatchStatus:
        """Snapshot status. Gemini batch states:

          BATCH_STATE_PENDING    → in_progress
          BATCH_STATE_RUNNING    → in_progress
          BATCH_STATE_SUCCEEDED  → ended
          BATCH_STATE_FAILED     → errored
          BATCH_STATE_CANCELLED  → cancelled
          BATCH_STATE_EXPIRED    → ended
        """
        if batch_id.startswith("DRYRUN-"):
            n = int(batch_id.split("-", 1)[1])
            return BatchStatus(state="ended", n_total=n, n_succeeded=n)
        self._ensure_client()
        b = self._client.batches.get(name=batch_id)
        api_state = str(getattr(b, "state", "") or "")
        status_map = {
            "BATCH_STATE_PENDING": "in_progress",
            "BATCH_STATE_RUNNING": "in_progress",
            "BATCH_STATE_SUCCEEDED": "ended",
            "BATCH_STATE_FAILED": "errored",
            "BATCH_STATE_CANCELLED": "cancelled",
            "BATCH_STATE_EXPIRED": "ended",
        }
        # Gemini doesn't always expose per-row counts during run.
        # We surface what's available and leave the rest at 0.
        completed = getattr(b, "completed_request_count", 0) or 0
        failed = getattr(b, "failed_request_count", 0) or 0
        total = getattr(b, "request_count", completed + failed) or (completed + failed)
        return BatchStatus(
            state=status_map.get(api_state, "in_progress"),
            n_total=int(total),
            n_succeeded=int(completed),
            n_failed=int(failed),
            n_pending=max(0, int(total) - int(completed) - int(failed)),
            raw={
                "state": api_state,
                "name": getattr(b, "name", None),
                "dest_file": getattr(b, "dest", None),
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
        """Fetch all results. The output file id is on the batch
        object's `dest.file_name` or `dest_file` field once the job
        is in SUCCEEDED state."""
        if batch_id.startswith("DRYRUN-"):
            return []
        self._ensure_client()
        b = self._client.batches.get(name=batch_id)
        # The SDK exposes the destination file under .dest.file_name
        # (or sometimes .dest depending on version). Try both.
        dest = getattr(b, "dest", None)
        dest_file_name: str | None = None
        if isinstance(dest, str):
            dest_file_name = dest
        elif dest is not None:
            dest_file_name = getattr(dest, "file_name", None) or getattr(dest, "name", None)
        if not dest_file_name:
            raise RuntimeError(
                f"Gemini batch {batch_id} has no destination file. "
                f"State: {getattr(b, 'state', '?')}."
            )

        # Download — same SDK quirk as upload, file/streaming surface
        # has changed across versions.
        content_bytes = self._client.files.download(file=dest_file_name)
        if isinstance(content_bytes, (bytes, bytearray)):
            text = content_bytes.decode("utf-8")
        else:
            # SDK might return a stream-like object
            text = content_bytes.read().decode("utf-8") if hasattr(content_bytes, "read") else str(content_bytes)

        results: list[BatchResult] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            results.append(_gemini_response_to_result(json.loads(line)))
        return results
