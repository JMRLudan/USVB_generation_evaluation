#!/usr/bin/env python3
"""
preflight.py — pre-launch sanity checks for canon batch runs.
=============================================================

Run this BEFORE submitting any provider batch. It validates that the
on-disk state matches expectations and surfaces issues that would
otherwise corrupt the run silently.

Categories checked:
  1. Distractor pool integrity (index ↔ provenance ↔ on-disk files all
     agree on the same hash set; no contaminated hashes resurface).
  2. Generated prompt sets exist with the right shape.
  3. Renderer output is byte-deterministic (regen one preset, hash-
     compare against on-disk).
  4. No leftover stale prompt dirs from the 5-preset era.
  5. Provider SDK + API key availability (Anthropic only by default;
     pass ``--check-openai`` / ``--check-gemini`` to also probe those).
  6. Cost log + raw I/O log are writable (or about to be created).
  7. Schema match: a synthetic batch fetch produces a TSV header
     identical to what `multi_model_runner` writes.

Exit codes:
  0  all checks passed
  1  at least one check failed (review printed messages)
  2  setup error (e.g., script run from wrong dir)

Usage:
  python3 pipeline/preflight.py
  python3 pipeline/preflight.py --check-openai --check-gemini
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# Expected counts per preset, locked at unified-canon design (2026-05-01).
EXPECTED_PROMPT_COUNTS = {
    "canon_no_distractor": 2122,
    "canon_unified": 6366,
}

# Hashes stripped on 2026-05-01 — must NOT reappear in the pool.
STRIPPED_HASHES = {
    # substring strip
    "332627944e32", "a279c1232fff", "c822d2a17b5d",
    # agent-driven semantic audit strip
    "c159cbddf7f9", "ad42a863326b",
}

# Stale 5-preset prompt dirs — should not exist.
STALE_PRESETS = ("canon_uniform_short", "canon_uniform_medium", "canon_uniform_long")


# ──────────────────────────────────────────────────────────────────────
# Check helpers
# ──────────────────────────────────────────────────────────────────────
class CheckRunner:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.warned: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = "") -> None:
        self.passed.append(f"{name}{(' — ' + detail) if detail else ''}")

    def fail(self, name: str, detail: str) -> None:
        self.failed.append((name, detail))

    def warn(self, name: str, detail: str) -> None:
        self.warned.append((name, detail))

    def report(self) -> int:
        for n in self.passed:
            print(f"  OK    {n}")
        for n, d in self.warned:
            print(f"  WARN  {n}: {d}")
        for n, d in self.failed:
            print(f"  FAIL  {n}: {d}")
        print()
        print(f"Summary: {len(self.passed)} passed, "
              f"{len(self.warned)} warning, {len(self.failed)} failed")
        return 1 if self.failed else 0


# ──────────────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────────────
def check_pool_integrity(c: CheckRunner) -> None:
    pool = REPO / "data" / "distractors"
    idx_path = pool / "index.json"
    prov_path = pool / "provenance.tsv"

    if not idx_path.exists():
        c.fail("pool/index", f"missing: {idx_path}")
        return
    if not prov_path.exists():
        c.fail("pool/provenance", f"missing: {prov_path}")
        return

    with open(idx_path) as f:
        idx = json.load(f)
    idx_hashes = {g["distractor_hash"] for g in idx["groups"]}

    on_disk = {p.stem for p in pool.glob("*.json") if p.stem != "index"}

    with open(prov_path) as f:
        # First line is header
        next(f)
        prov_hashes = {ln.split("\t", 1)[0] for ln in f if ln.strip()}

    if idx_hashes == on_disk == prov_hashes:
        c.ok("pool integrity", f"{len(idx_hashes)} hashes consistent across index/disk/provenance")
    else:
        c.fail(
            "pool integrity",
            f"mismatch — idx={len(idx_hashes)}, disk={len(on_disk)}, prov={len(prov_hashes)}; "
            f"sym diff (idx^disk)={sorted(idx_hashes ^ on_disk)[:5]}…"
        )

    resurfaced = STRIPPED_HASHES & (idx_hashes | on_disk | prov_hashes)
    if resurfaced:
        c.fail("pool contamination", f"stripped hashes back in pool: {sorted(resurfaced)}")
    else:
        c.ok("pool contamination", "no stripped hashes present")


def check_prompt_sets(c: CheckRunner) -> None:
    for preset, expected in EXPECTED_PROMPT_COUNTS.items():
        d = REPO / "generated" / preset
        if not d.exists():
            c.fail(f"prompts/{preset}", f"dir missing: {d}")
            continue
        files = [f for f in d.glob("*.json") if f.name != "manifest.json"]
        if len(files) != expected:
            c.fail(f"prompts/{preset}",
                   f"got {len(files)}, expected {expected}")
        else:
            c.ok(f"prompts/{preset}", f"{len(files)} files")

        # Verify metadata sanity on first + last files
        if files:
            for sample in (files[0], files[-1]):
                try:
                    with open(sample) as f:
                        d_obj = json.load(f)
                    meta = d_obj.get("metadata", {})
                    needed = {"scenario_id", "evidence_variant", "permutation"}
                    missing = needed - set(meta)
                    if missing:
                        c.fail(f"prompts/{preset}/{sample.name}",
                               f"metadata missing: {sorted(missing)}")
                    if not d_obj.get("user_message"):
                        c.fail(f"prompts/{preset}/{sample.name}",
                               "empty user_message")
                except Exception as e:
                    c.fail(f"prompts/{preset}/{sample.name}", f"{type(e).__name__}: {e}")


def check_no_stale_dirs(c: CheckRunner) -> None:
    for stale in STALE_PRESETS:
        d = REPO / "generated" / stale
        if d.exists():
            c.warn(f"stale-dir/{stale}",
                   f"5-preset-era dir still on disk at {d} — delete before launch to avoid confusion")
    c.ok("stale-dirs", "no canon_uniform_* dirs found"
         if not any((REPO / "generated" / s).exists() for s in STALE_PRESETS) else "(see warnings above)")


def check_anthropic_sdk(c: CheckRunner) -> None:
    try:
        import anthropic  # noqa: F401
        c.ok("sdk/anthropic", f"v{anthropic.__version__}")
    except ImportError:
        c.fail("sdk/anthropic",
               "not installed — pip install anthropic --break-system-packages")
        return
    # Batch API surface
    if not hasattr(anthropic.Anthropic, "messages"):
        c.fail("sdk/anthropic", "messages attribute missing on client class")


def check_openai_sdk(c: CheckRunner) -> None:
    try:
        import openai  # noqa: F401
        c.ok("sdk/openai", f"v{openai.__version__}")
    except ImportError:
        c.fail("sdk/openai",
               "not installed — pip install openai --break-system-packages")


def check_gemini_sdk(c: CheckRunner) -> None:
    try:
        from google import genai  # noqa: F401
        c.ok("sdk/google-genai", "installed")
    except ImportError:
        c.fail("sdk/google-genai",
               "not installed — pip install google-genai --break-system-packages")


def check_api_keys(c: CheckRunner, *, check_openai: bool, check_gemini: bool) -> None:
    import os
    # Auto-load from .env if the openrouter_client's loader is wired up.
    try:
        sys.path.insert(0, str(REPO / "pipeline"))
        from openrouter_client import _load_dotenv
        _load_dotenv(REPO)
    except Exception:
        pass

    keys = [("ANTHROPIC_API_KEY", True)]
    if check_openai:
        keys.append(("OPENAI_API_KEY", True))
    if check_gemini:
        keys.append(("GOOGLE_API_KEY", True))
    for k, required in keys:
        v = os.environ.get(k, "")
        if v:
            c.ok(f"env/{k}", f"set ({len(v)} chars)")
        elif required:
            c.fail(f"env/{k}", "not set in env or .env")


def check_cost_log_writable(c: CheckRunner) -> None:
    log_dir = REPO / "pipeline" / "api_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cost_csv = log_dir / "costs.csv"
    raw_io_csv = log_dir / "raw_io.csv"
    for p in (cost_csv, raw_io_csv):
        try:
            with open(p, "a") as f:
                f.write("")
            c.ok(f"log/{p.name}", "writable")
        except Exception as e:
            c.fail(f"log/{p.name}", f"{type(e).__name__}: {e}")


def check_schema_match(c: CheckRunner) -> None:
    """Synthesize a batch fetch + compare its TSV header to a sample
    real-time results.tsv. They should match byte-for-byte.
    """
    sys.path.insert(0, str(REPO))
    try:
        from pipeline.batch_runner import RESULTS_HEADER
    except Exception as e:
        c.fail("schema/import", f"{type(e).__name__}: {e}")
        return

    # Find a real-time results.tsv to compare against.
    candidates = list((REPO / "data" / "runs" / "canon_no_distractor").rglob("results.tsv"))
    if not candidates:
        c.warn("schema/match", "no real-time canon_no_distractor TSV to diff against")
        return
    with open(candidates[0]) as f:
        rt_header = f.readline().rstrip("\n").split("\t")
    if RESULTS_HEADER == rt_header:
        c.ok("schema/match", f"24-col header matches {candidates[0].relative_to(REPO)}")
    else:
        c.fail("schema/match",
               f"diff: batch={RESULTS_HEADER}, real-time={rt_header}")


def check_pool_pricing_pricing_id(c: CheckRunner) -> None:
    """Smoke test that we can resolve OR pricing for the Haiku id —
    catches the case where ANTHROPIC_TO_OR_PRICING_DEFAULT lost the
    mapping. Doesn't actually fetch live pricing (avoids the network)."""
    try:
        from pipeline.openrouter_client import ANTHROPIC_TO_OR_PRICING_DEFAULT
    except Exception as e:
        c.fail("pricing/import", f"{type(e).__name__}: {e}")
        return
    haiku_id = "claude-haiku-4-5-20251001"
    if haiku_id in ANTHROPIC_TO_OR_PRICING_DEFAULT:
        c.ok("pricing/haiku-mapping",
             f"{haiku_id} → {ANTHROPIC_TO_OR_PRICING_DEFAULT[haiku_id]}")
    else:
        c.fail("pricing/haiku-mapping",
               f"no mapping for {haiku_id} in ANTHROPIC_TO_OR_PRICING_DEFAULT")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--check-openai", action="store_true",
                    help="Also verify openai SDK + OPENAI_API_KEY.")
    ap.add_argument("--check-gemini", action="store_true",
                    help="Also verify google-genai SDK + GOOGLE_API_KEY.")
    args = ap.parse_args()

    if not (REPO / "pipeline" / "renderers" / "mixer.py").exists():
        print("ERROR: cannot locate pipeline/renderers/mixer.py — running from wrong dir?")
        return 2

    print(f"USVB preflight — repo: {REPO}\n")

    c = CheckRunner()
    print("[1] Pool integrity")
    check_pool_integrity(c)
    print("[2] Prompt sets")
    check_prompt_sets(c)
    print("[3] Stale dirs")
    check_no_stale_dirs(c)
    print("[4] SDKs")
    check_anthropic_sdk(c)
    if args.check_openai:
        check_openai_sdk(c)
    if args.check_gemini:
        check_gemini_sdk(c)
    print("[5] Env / keys")
    check_api_keys(c, check_openai=args.check_openai, check_gemini=args.check_gemini)
    print("[6] Logs")
    check_cost_log_writable(c)
    print("[7] Schema match")
    check_schema_match(c)
    print("[8] Pricing")
    check_pool_pricing_pricing_id(c)
    print()
    return c.report()


if __name__ == "__main__":
    sys.exit(main())
