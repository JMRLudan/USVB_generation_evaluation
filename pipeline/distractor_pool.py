"""
Deduplicated distractor pool + deterministic scenario→distractor assignment.

Replaces the old `data/distractor_turns.json` 140MB file. The new pool lives
in `data/distractors/` as one JSON file per distractor (hash-named) plus
`index.json` and `provenance.tsv`.

Each distractor JSON has schema:
    {
      "distractor_hash": str,                      # 12-hex-char id
      "distractor_domain": str,                    # e.g. "Programming/CS"
      "turns": [ {"timestamp": "YYYY-MM-DD HH:MM:SS",
                  "role": "user" | "assistant",
                  "content": str }, ... ],
      "num_turns": int,
      "num_user_turns": int,
      "provenance": list
    }

Construction invariants (also documented in the top-level README) that
every renderer in this repo MUST satisfy:

  - **No two scenarios share the same distractor within a single draw.**
    The pool currently holds 94 deduplicated distractor groups (after
    the 2026-05-01 USVB-substring strip — distractors that contained
    any verbatim seed text from a scenario were removed to prevent
    contamination). With 85 scenarios, each draw is a 1-to-1
    assignment using an 85-subset of the pool. Pool size is read
    dynamically from index.json, not hard-coded.
  - **Deterministic assignment seed: 4232026.** For draw index `d`, the
    permutation is `random.Random(4232026 + d).sample(all_hashes,
    len(all_hashes))` and scenario `scenarios_sorted[i]` is assigned
    `permutation[i]`.

Multi-distractor stitching (used by `canon_unified`) is implemented in
`renderers/mixer.py` via `n_distractors_per_prompt` and the assembly
helpers in `renderers/assembly.py`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────
# Paths + seed
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DISTRACTORS_DIR = BASE_DIR / "data" / "distractors"
POOL_INDEX = DISTRACTORS_DIR / "index.json"

# Fixed assignment seed. Do NOT change without a corresponding paper note;
# changing this shuffles every renderer's distractor choice and breaks
# reproducibility against prior runs.
ASSIGNMENT_SEED = 4232026


# ──────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Distractor:
    hash: str
    domain: str
    turns: List[Dict]       # each: {timestamp, role, content}
    num_turns: int
    num_user_turns: int


# ──────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────
def load_pool(distractors_dir: Path | None = None) -> Dict[str, Distractor]:
    """Load the deduplicated distractor pool into memory.

    Returns a dict keyed by distractor_hash. Iteration order is not
    guaranteed — consumers that need determinism should
    `sorted(pool.keys())`.
    """
    root = Path(distractors_dir) if distractors_dir else DISTRACTORS_DIR
    idx = root / "index.json"
    if not idx.exists():
        raise FileNotFoundError(
            f"Pool index not found at {idx}. Expected data/distractors/index.json "
            f"listing the deduplicated distractor groups."
        )
    with open(idx) as f:
        index_data = json.load(f)

    pool: Dict[str, Distractor] = {}
    for group in index_data["groups"]:
        h = group["distractor_hash"]
        path = root / f"{h}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Distractor file missing for hash {h}: expected {path}. "
                f"Index claims {len(index_data['groups'])} groups — "
                f"pool directory appears inconsistent."
            )
        with open(path) as f:
            data = json.load(f)
        pool[h] = Distractor(
            hash=data["distractor_hash"],
            domain=data.get("distractor_domain", ""),
            turns=data["turns"],
            num_turns=int(data.get("num_turns", len(data["turns"]))),
            num_user_turns=int(data.get("num_user_turns", 0)),
        )
    return pool


def pool_hashes(pool: Dict[str, Distractor]) -> List[str]:
    """Sorted list of all distractor hashes — canonical order."""
    return sorted(pool.keys())


# ──────────────────────────────────────────────────────────────────────
# Assignment
# ──────────────────────────────────────────────────────────────────────
def assign_distractors(
    scenario_ids: Sequence[str],
    pool: Dict[str, Distractor],
    num_draws: int = 1,
    seed: int = ASSIGNMENT_SEED,
) -> List[Dict[str, str]]:
    """Deterministically assign one distractor to each scenario, per draw.

    Returns a list of length `num_draws`. Each element is a dict
    `{scenario_id: distractor_hash}` giving the assignment for that draw.

    Guarantees:
      - Within each draw, no two scenarios share the same distractor
        (1-to-1 assignment, using an N-subset of the pool where N =
        len(scenarios)).
      - The assignment is reproducible: same (seed, num_draws, pool,
        scenario_ids) → identical output.
      - Across draws, every scenario sees an independent fresh permutation
        of the pool (seeded by `seed + draw_idx`). Distractor usage is
        balanced in expectation as `num_draws` grows.

    Args:
        scenario_ids: iterable of scenario ids. Order does NOT matter —
            the function sorts internally so that the same set of scenarios
            always yields the same assignment regardless of input order.
        pool: the loaded distractor pool (from `load_pool`). Must have
            at least as many distractors as scenarios.
        num_draws: how many independent assignments to produce.
        seed: base RNG seed. Default is the project-standard `4232026`.

    Raises:
        ValueError: if the pool is smaller than the scenario set.
    """
    hashes = pool_hashes(pool)
    scenarios_sorted = sorted(set(scenario_ids))
    n_scen = len(scenarios_sorted)
    n_pool = len(hashes)
    if n_scen > n_pool:
        raise ValueError(
            f"Pool has only {n_pool} distractors but {n_scen} scenarios were "
            f"requested. Can't assign 1-to-1."
        )

    draws: List[Dict[str, str]] = []
    for d in range(num_draws):
        rng = random.Random(seed + d)
        permutation = rng.sample(hashes, n_pool)  # full shuffle of the pool
        assignment = {
            sid: permutation[i] for i, sid in enumerate(scenarios_sorted)
        }
        draws.append(assignment)
    return draws


def assignment_summary(draws: List[Dict[str, str]]) -> Dict:
    """Small stats helper: how many times each distractor was used across draws."""
    counts: Dict[str, int] = {}
    for d in draws:
        for h in d.values():
            counts[h] = counts.get(h, 0) + 1
    used = len(counts)
    vals = list(counts.values()) or [0]
    return {
        "num_draws": len(draws),
        "distractors_used": used,
        "min_uses": min(vals),
        "max_uses": max(vals),
        "mean_uses": sum(vals) / len(vals) if vals else 0,
    }
