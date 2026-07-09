#!/usr/bin/env python3
"""
mitnv_status_sweep.py — poll every batch in the four mitigation-arm
manifests (run mitnv_20260708_171704) and print raw API states.

Prints the RAW JobState from the API (not the adapter's mapped state)
to sidestep any state-mapping regression (see JOURNAL 2026-05-19:
BATCH_STATE_*/JOB_STATE_* mismatch made finished batches look stuck).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

try:
    from pipeline.openrouter_client import _load_dotenv
    _load_dotenv(BASE)
except Exception:
    pass

import os  # noqa: E402
from google import genai  # noqa: E402

RUN_ID = "mitnv_20260708_171704"
PRESETS = [
    "canon_no_distractor_mit_sysbottom",
    "canon_no_distractor_mit_querytop",
    "canon_unified_mit_sysbottom",
    "canon_unified_mit_querytop",
]


def main() -> int:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    all_done = True
    for preset in PRESETS:
        man_path = BASE / "batch_manifests" / f"{RUN_ID}__{preset}.json"
        if not man_path.exists():
            print(f"{preset}: NO MANIFEST")
            all_done = False
            continue
        ids = json.loads(man_path.read_text())["batch_ids"]
        states = Counter()
        for bid in ids:
            b = client.batches.get(name=bid)
            raw = str(b.state)
            raw = raw.rsplit(".", 1)[-1] if "." in raw else raw
            states[raw] += 1
        line = ", ".join(f"{k}×{v}" for k, v in sorted(states.items()))
        done = set(states) <= {"JOB_STATE_SUCCEEDED"}
        if not done:
            all_done = False
        print(f"{preset}  [{len(ids)} batches]  {line}"
              f"{'   ✓ ALL SUCCEEDED' if done else ''}")
    print("\nALL_DONE" if all_done else "\nSTILL_RUNNING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
