#!/usr/bin/env python3
"""
rag_embedding_analysis.py — embedding-relevance diagnostic for the RAG
mitigation question: could similarity retrieval over the conversation
history even FIND the constraint evidence, given the final query?

For every C-bearing canon_unified prompt (variants C, A+C, B+C):
  1. parse the rendered history into turns,
  2. locate the gold turns (the C evidence seeds of the prompt's
     permutation, matched as normalized substrings),
  3. embed every turn + the bare query (metadata.query_with_options),
  4. rank turns by cosine similarity to the query and record where the
     best gold turn lands: rank, percentile, hit@{1,3,5,10}, gold vs
     best-distractor similarity.

Stages (all resumable; state under analysis_rag/):
    extract              parse prompts -> prompts.jsonl + corpus.jsonl
    embed  --model M     embed unique turns + queries (checkpointed)
    score  --model M     per-prompt ranks -> scores_<M>.tsv
    report               aggregate all scored models -> summary + HTML

Embedding models (M):
    oai-small  = openai text-embedding-3-small
    oai-large  = openai text-embedding-3-large
    gemini     = gemini-embedding-001
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
csv.field_size_limit(sys.maxsize)

try:
    from pipeline.openrouter_client import _load_dotenv
    _load_dotenv(BASE)
except Exception:
    pass

import os  # noqa: E402

OUT = BASE / "analysis_rag"
PROMPTS_DIR = BASE / "generated" / "canon_unified"
SCENARIOS = BASE / "data" / "scenarios_FINAL.tsv"

TURN_SPLIT = re.compile(
    r"\n(?=\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] (?:User|Assistant):)")
TURN_HEAD = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (User|Assistant): ", re.S)

MODELS = {
    "oai-small": ("openai", "text-embedding-3-small"),
    "oai-large": ("openai", "text-embedding-3-large"),
    "gemini": ("gemini", "gemini-embedding-001"),
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:20]


def load_c_seeds() -> dict:
    seeds = {}
    for r in csv.DictReader(open(SCENARIOS), delimiter="\t"):
        seeds[r["id"]] = [s.strip() for s in
                          r["evidence_set_c_seeds"].split("||")]
    return seeds


def c_indices(permutation: str) -> list[int]:
    """'c0_a1' -> [0]; 'c0_c2' style not used; single c index per perm."""
    return [int(m.group(1)) for m in re.finditer(r"c(\d+)", permutation)]


# ────────────────────────────────────────────────────────────────────
def cmd_extract(_args) -> int:
    OUT.mkdir(exist_ok=True)
    seeds_by_sid = load_c_seeds()
    corpus: dict[str, str] = {}
    n_prompts = n_gold_missing = 0
    with open(OUT / "prompts.jsonl", "w") as pf:
        for fp in sorted(PROMPTS_DIR.glob("*.json")):
            if fp.name == "manifest.json":
                continue
            d = json.loads(fp.read_text())
            m = d["metadata"]
            ev = m["evidence_variant"]
            if "C" not in ev:
                continue
            gold_seeds = [norm(seeds_by_sid[m["scenario_id"]][i])
                          for i in c_indices(m["permutation"])]
            segs = TURN_SPLIT.split(d["system_prompt"])
            turns = []
            for seg in segs:
                h = TURN_HEAD.match(seg)
                if not h:
                    continue
                body = norm(seg[h.end():])
                if not body:
                    continue
                is_gold = any(g in body for g in gold_seeds)
                th = sha(body)
                corpus.setdefault(th, body)
                turns.append({"h": th, "role": h.group(2), "gold": int(is_gold)})
            if not any(t["gold"] for t in turns):
                n_gold_missing += 1
                continue
            q = norm(m["query_with_options"])
            qh = sha(q)
            corpus.setdefault(qh, q)
            pf.write(json.dumps({
                "file": fp.name, "sid": m["scenario_id"], "ev": ev,
                "perm": m["permutation"], "char_budget": m["char_budget"],
                "placement_frac": m["placement_frac"], "qh": qh,
                "turns": turns,
            }) + "\n")
            n_prompts += 1
    with open(OUT / "corpus.jsonl", "w") as cf:
        for h, text in corpus.items():
            cf.write(json.dumps({"h": h, "text": text}) + "\n")
    print(f"(prompts extracted,        {n_prompts})")
    print(f"(prompts with gold missing, {n_gold_missing}  -> excluded)")
    print(f"(unique texts to embed,    {len(corpus)})")
    return 0


# ────────────────────────────────────────────────────────────────────
def _embed_openai(model: str, texts: list[str]) -> list[list[float]]:
    import urllib.request
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": model, "input": texts}).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)
    out = sorted(d["data"], key=lambda x: x["index"])
    return [e["embedding"] for e in out]


def _embed_gemini(model: str, texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    res = client.models.embed_content(
        model=model, contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=1536))
    return [e.values for e in res.embeddings]


def cmd_embed(args) -> int:
    provider, model_id = MODELS[args.model]
    corpus = [json.loads(l) for l in open(OUT / "corpus.jsonl")]
    emb_path = OUT / f"emb_{args.model}.jsonl"
    done = set()
    if emb_path.exists():
        for l in open(emb_path):
            try:
                done.add(json.loads(l)["h"])
            except Exception:
                pass
    todo = [c for c in corpus if c["h"] not in done]
    print(f"{args.model}: {len(done)} embedded, {len(todo)} to go")
    t0 = time.time()
    bs = args.batch_size
    with open(emb_path, "a") as f:
        for i in range(0, len(todo), bs):
            if time.time() - t0 > args.deadline_s:
                print(f"DEADLINE — {len(done)+i} done. Re-run to continue.")
                return 0
            chunk = todo[i:i+bs]
            fn = _embed_openai if provider == "openai" else _embed_gemini
            # gemini per-call limit is 100 inputs
            vecs = fn(model_id, [c["text"][:8000] for c in chunk])
            for c, v in zip(chunk, vecs):
                f.write(json.dumps({"h": c["h"], "v": [round(x, 6) for x in v]}) + "\n")
            f.flush()
    print(f"DONE — all {len(corpus)} texts embedded for {args.model}.")
    return 0


# ────────────────────────────────────────────────────────────────────
def cmd_score(args) -> int:
    import numpy as np
    vecs = {}
    for l in open(OUT / f"emb_{args.model}.jsonl"):
        try:
            d = json.loads(l)
        except json.JSONDecodeError:
            continue  # truncated line from a killed embed call; re-embedded later
        vecs[d["h"]] = np.array(d["v"], dtype=np.float32)
    out_path = OUT / f"scores_{args.model}.tsv"
    with open(out_path, "w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file", "sid", "ev", "perm", "char_budget",
                    "placement_frac", "n_turns", "n_gold", "gold_best_rank",
                    "gold_pctile", "hit1", "hit3", "hit5", "hit10",
                    "gold_sim_best", "distractor_sim_best"])
        for l in open(OUT / "prompts.jsonl"):
            p = json.loads(l)
            q = vecs[p["qh"]]; qn = q / np.linalg.norm(q)
            sims, golds = [], []
            for t in p["turns"]:
                v = vecs[t["h"]]
                s = float(np.dot(qn, v / np.linalg.norm(v)))
                sims.append(s); golds.append(t["gold"])
            order = sorted(range(len(sims)), key=lambda i: -sims[i])
            rank = next(i for i, j in enumerate(order, 1) if golds[j])
            n = len(sims)
            gbest = max(s for s, g in zip(sims, golds) if g)
            dbest = max((s for s, g in zip(sims, golds) if not g), default=float("nan"))
            w.writerow([p["file"], p["sid"], p["ev"], p["perm"],
                        p["char_budget"], round(p["placement_frac"], 4), n,
                        sum(golds), rank, round(1 - (rank - 1) / n, 4),
                        int(rank <= 1), int(rank <= 3), int(rank <= 5),
                        int(rank <= 10), round(gbest, 4), round(dbest, 4)])
    print(f"wrote {out_path}")
    return 0


# ────────────────────────────────────────────────────────────────────
def cmd_report(_args) -> int:
    for m in MODELS:
        p = OUT / f"scores_{m}.tsv"
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        n = len(rows)
        def avg(k): return sum(float(r[k]) for r in rows) / n
        print(f"\n=== {m}  (n={n} prompts) ===")
        print(f"  hit@1 {100*avg('hit1'):5.1f}%   hit@3 {100*avg('hit3'):5.1f}%   "
              f"hit@5 {100*avg('hit5'):5.1f}%   hit@10 {100*avg('hit10'):5.1f}%")
        print(f"  median gold rank: {sorted(int(r['gold_best_rank']) for r in rows)[n//2]}"
              f"   mean pctile: {avg('gold_pctile'):.3f}")
        # by length tercile
        srt = sorted(rows, key=lambda r: int(r["char_budget"]))
        for name, sl in (("short", srt[:n//3]), ("mid", srt[n//3:2*n//3]), ("long", srt[2*n//3:])):
            h5 = 100*sum(float(r["hit5"]) for r in sl)/len(sl)
            print(f"  hit@5 {name:5s} haystacks: {h5:5.1f}%")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    pe = sub.add_parser("embed")
    pe.add_argument("--model", choices=list(MODELS), required=True)
    pe.add_argument("--batch-size", type=int, default=100)
    pe.add_argument("--deadline-s", type=float, default=30.0)
    ps = sub.add_parser("score")
    ps.add_argument("--model", choices=list(MODELS), required=True)
    sub.add_parser("report")
    args = ap.parse_args()
    return {"extract": cmd_extract, "embed": cmd_embed,
            "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
