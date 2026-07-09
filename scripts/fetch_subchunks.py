#!/usr/bin/env python3
"""
fetch_subchunks.py — sandbox-safe resumable fetch for multi-batch manifests.

`batch_runner fetch` pulls every batch in a manifest serially and only
writes results.tsv at the end — with 12+ batches that can exceed the 45s
sandbox call budget and lose all progress. This driver caches each
batch's results to a pickle sidecar as it lands (deadline-based, re-run
until DONE), then assembles the standard results.tsv + cost log.

Usage (re-run until it prints DONE):
    python3 scripts/fetch_subchunks.py --manifest <stem> --preset <preset> \
        [--deadline-s 32]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

try:
    from pipeline.openrouter_client import _load_dotenv
    _load_dotenv(BASE)
except Exception:
    pass

from pipeline.batch_runner import get_adapter, write_results_tsv  # noqa: E402


def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest stem or filename")
    ap.add_argument("--preset", required=True)
    ap.add_argument("--deadline-s", type=float, default=32.0)
    args = ap.parse_args()

    stem = args.manifest.removesuffix(".json")
    man_path = BASE / "batch_manifests" / f"{stem}.json"
    man = json.loads(man_path.read_text())
    ids = [b for b in man["batch_ids"] if b]
    adapter = get_adapter(man["provider"])

    cache_dir = BASE / "batch_manifests" / f"{stem}.fetchcache"
    cache_dir.mkdir(exist_ok=True)

    for i, bid in enumerate(ids):
        cpath = cache_dir / f"{i:03d}.pkl"
        if cpath.exists():
            continue
        if time.time() - t0 > args.deadline_s:
            n = sum(1 for j in range(len(ids)) if (cache_dir / f"{j:03d}.pkl").exists())
            print(f"DEADLINE — {n}/{len(ids)} batches cached. Re-run to continue.")
            return 0
        print(f"fetching batch {i+1}/{len(ids)} ({bid})…", flush=True)
        results = adapter.fetch_results(bid)
        tmp = cpath.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(results, f)
        tmp.rename(cpath)

    # All cached — assemble.
    results = []
    for i in range(len(ids)):
        with open(cache_dir / f"{i:03d}.pkl", "rb") as f:
            results.extend(pickle.load(f))
    n_ok = sum(1 for r in results if r.status == "ok")
    print(f"assembled {len(results)} results: {n_ok} ok / {len(results)-n_ok} non-ok")

    model = man["model"]
    run_id = man["run_id"]
    out_path = (BASE / "data" / "runs" / args.preset /
                model.replace("/", "_") / run_id / "results.tsv")
    write_results_tsv(results, out_path=out_path, model=model, run_id=run_id,
                      condition=args.preset,
                      prompts_dir=Path(man["prompts_dir"]))
    print(f"wrote {out_path}")
    try:
        from pipeline.openrouter_client import OpenRouterClient
        client = OpenRouterClient(run_id=f"batch_{run_id}")
        total = client.log_batch_results(results, model=model,
                                         provider=man["provider"])
        print(f"logged {len(results)} cost rows (batch total ${total:.2f})")
    except Exception as e:  # noqa: BLE001
        print(f"WARN: cost logging failed ({type(e).__name__}: {e})")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
