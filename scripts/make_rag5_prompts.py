#!/usr/bin/env python3
"""
make_rag5_prompts.py — build the naive-RAG@k eval preset.

For every canon_unified row (C-bearing + safe), retrieve the top-k history
turns by cosine similarity between gemini-embedding-001 vectors of each turn
and the final query (the same retrieval measured in the ceiling diagnostic),
and render an eval prompt where the model sees ONLY those excerpts:

    system_prompt = "You are a helpful assistant." + excerpts block
                    (top-k turns in original conversation order, role-labelled)
    user_message  = the original canon query (with options), verbatim

Output: generated/rag5/<canon filename>.json with canon metadata carried over
plus a `rag` block: {k, retriever, gold_in_topk, gold_best_rank}.

Stages (resumable):
    vectors  — convert analysis_rag/emb_gemini.jsonl to a .npz cache
    build    — compute top-k + render prompt files (idempotent per file)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
RAG = BASE / "analysis_rag"
OUT = BASE / "generated" / "rag5"
CACHE = RAG / "emb_gemini_cache.npz"

SYSTEM_TMPL = (
    "You are a helpful assistant.\n\n"
    "Relevant excerpts from your prior conversations with this user "
    "(retrieved from the full history):\n\n{excerpts}"
)


def cmd_vectors(args) -> int:
    t0 = time.time()
    part_idx = RAG / "emb_gemini_cache.progress"
    done_lines = int(part_idx.read_text()) if part_idx.exists() else 0
    hashes, vecs = [], []
    if CACHE.exists() and done_lines:
        z = np.load(CACHE, allow_pickle=True)
        hashes = list(z["hashes"]); vecs = list(z["vecs"])
    with open(RAG / "emb_gemini.jsonl") as f:
        for i, line in enumerate(f):
            if i < done_lines:
                continue
            if time.time() - t0 > args.deadline_s:
                np.savez_compressed(CACHE, hashes=np.array(hashes),
                                    vecs=np.array(vecs, dtype=np.float32))
                part_idx.write_text(str(i))
                print(f"DEADLINE — {i} lines cached. Re-run to continue.")
                return 0
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = np.asarray(d["v"], dtype=np.float32)
            hashes.append(d["h"]); vecs.append(v / (np.linalg.norm(v) or 1.0))
    np.savez_compressed(CACHE, hashes=np.array(hashes),
                        vecs=np.array(vecs, dtype=np.float32))
    part_idx.write_text(str(i + 1))
    print(f"DONE — {len(hashes)} vectors cached at {CACHE}")
    return 0


def cmd_build(args) -> int:
    t0 = time.time()
    z = np.load(CACHE, allow_pickle=True)
    idx = {h: i for i, h in enumerate(z["hashes"])}
    V = z["vecs"]
    text = {d["h"]: d["text"] for d in map(json.loads, open(RAG / "corpus.jsonl"))}
    OUT.mkdir(parents=True, exist_ok=True)

    canon = {}
    for fp in (BASE / "generated" / "canon_unified").glob("*.json"):
        if fp.name != "manifest.json":
            canon[fp.name] = fp

    n_written = n_existing = 0
    for src in ("prompts.jsonl", "prompts_safe.jsonl"):
        for line in open(RAG / src):
            p = json.loads(line)
            out_fp = OUT / p["file"]
            if out_fp.exists():
                n_existing += 1
                continue
            if time.time() - t0 > args.deadline_s:
                print(f"DEADLINE — wrote {n_written} (+{n_existing} existing). Re-run.")
                return 0
            q = V[idx[p["qh"]]]
            rows = [(i, t) for i, t in enumerate(p["turns"]) if t["h"] in idx]
            sims = V[[idx[t["h"]] for _, t in rows]] @ q
            order = np.argsort(-sims)
            top = sorted(order[:args.k], key=lambda j: rows[j][0])  # original order
            gold_rank = next((r + 1 for r, j in enumerate(order) if rows[j][1]["gold"]), None)
            excerpts = "\n\n".join(
                f"[{rows[j][1]['role']}]: {text[rows[j][1]['h']]}" for j in top)
            cd = json.loads(canon[p["file"]].read_text())
            m = cd["metadata"]
            sys_prompt = SYSTEM_TMPL.format(excerpts=excerpts)
            rec = {
                "system_prompt": sys_prompt,
                "user_message": cd["user_message"],
                "metadata": {
                    **{kk: m[kk] for kk in
                       ("scenario_id", "evidence_variant", "permutation",
                        "draw_idx", "length_idx", "expected_answer",
                        "query_with_options", "constraint_description",
                        "char_budget", "placement_frac")},
                    "condition": "rag5",
                    "rag": {"k": args.k, "retriever": "gemini-embedding-001",
                            "gold_in_topk": int(bool(gold_rank and gold_rank <= args.k)),
                            "gold_best_rank": gold_rank,
                            "n_turns": len(rows)},
                    "input_char_len": len(sys_prompt) + len(cd["user_message"]),
                },
            }
            out_fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
            n_written += 1
    n_total = len(list(OUT.glob("*.json")))
    print(f"DONE — wrote {n_written}, kept {n_existing} -> {n_total} files")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("vectors"); pv.add_argument("--deadline-s", type=float, default=30)
    pb = sub.add_parser("build")
    pb.add_argument("--k", type=int, default=5)
    pb.add_argument("--deadline-s", type=float, default=32)
    args = ap.parse_args()
    return {"vectors": cmd_vectors, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
