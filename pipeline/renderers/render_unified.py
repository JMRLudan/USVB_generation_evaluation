#!/usr/bin/env python3
"""
render_unified.py — the unified canon condition.
==================================================

Single distractor-bearing canon preset (canon_unified): log-uniform char
budget over [3 000, 250 000] paired with uniform [0, 1] placement, both
stratified within scenario. Each row gets:

* a per-cell stratified placement frac in [0, 1] (where to put the
  evidence in the conversation history),
* a per-cell stratified log-uniform char budget over [length_min,
  length_max] (how big the conversation history is).

Both axes are stratified within scenario — every scenario covers the
[0, 1] depth range and the log-length range uniformly. By default each
(scenario, variant, perm) tuple is sampled three times (n_distractor_draws=3),
giving ~6366 prompts total at the 5-variant scenario set.

Thin wrapper over ``mixer.mix()`` with:

    n_distractor_draws        = 3       (3× resample per tuple)
    n_distractors_per_prompt  = 3       (USVB canon convention)
    n_placements              = 1       (uniform_stratified single)
    n_lengths                 = 1
    placement_mode            = "uniform_stratified"
    length_mode               = "log_uniform_stratified"
    length_range              = (3000, 250000)
    include_constraint_inline = False
    c_only                    = False    (full 5-variant: C/A+C/B+C/A/B)
    condition_label           = "canon_unified"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from renderers.mixer import mix  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "generated" / "canon_unified"

DEFAULT_LENGTH_MIN = 3000
DEFAULT_LENGTH_MAX = 250_000
DEFAULT_N_RESAMPLES = 3
DEFAULT_N_DISTRACTORS_PER_PROMPT = 3


def render(
    out_dir: Path,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    n_distractors_per_prompt: int = DEFAULT_N_DISTRACTORS_PER_PROMPT,
    length_min: int = DEFAULT_LENGTH_MIN,
    length_max: int = DEFAULT_LENGTH_MAX,
    c_only: bool = False,
    merge_gap_days: int = 1,
) -> dict:
    return mix(
        out_dir=out_dir,
        n_distractor_draws=n_resamples,
        n_distractors_per_prompt=n_distractors_per_prompt,
        n_placements=1,
        n_lengths=1,
        placement_mode="uniform_stratified",
        length_mode="log_uniform_stratified",
        length_range=(length_min, length_max),
        c_only=c_only,
        merge_gap_days=merge_gap_days,
        condition_label="canon_unified",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument(
        "--n-resamples", type=int, default=DEFAULT_N_RESAMPLES,
        help=(
            "Number of (length, placement) samples per (scenario, variant, "
            "perm) tuple. Default 3 → ~6366 prompts at the 85-scenario "
            "5-variant canon."
        ),
    )
    ap.add_argument(
        "--n-distractors-per-prompt", type=int,
        default=DEFAULT_N_DISTRACTORS_PER_PROMPT,
        help="How many distractor conversations to merge per prompt.",
    )
    ap.add_argument(
        "--length-min", type=int, default=DEFAULT_LENGTH_MIN,
        help="Minimum char budget (log-uniform lower bound).",
    )
    ap.add_argument(
        "--length-max", type=int, default=DEFAULT_LENGTH_MAX,
        help="Maximum char budget (log-uniform upper bound).",
    )
    ap.add_argument(
        "--c-only", action="store_true",
        help="Restrict to C-bearing variants (skip A and B no-C). Off by "
             "default — canon_unified includes all 5 variants.",
    )
    ap.add_argument(
        "--merge-gap-days", type=int, default=1,
        help="Day gap between consecutive merged distractors.",
    )
    args = ap.parse_args()

    manifest = render(
        out_dir=Path(args.out_dir),
        n_resamples=args.n_resamples,
        n_distractors_per_prompt=args.n_distractors_per_prompt,
        length_min=args.length_min,
        length_max=args.length_max,
        c_only=args.c_only,
        merge_gap_days=args.merge_gap_days,
    )
    print(f"Built {manifest['num_prompts']} prompts → {args.out_dir}")
    ic = manifest["input_chars"]
    print(f"  input chars: avg={ic['avg']:,.0f}  min={ic['min']:,}  max={ic['max']:,}")
    if "placement_frac" in manifest:
        pf = manifest["placement_frac"]
        print(
            f"  placement_frac: avg={pf['avg']:.3f}  "
            f"min={pf['min']:.3f}  max={pf['max']:.3f}"
        )
    if "distractor_usage" in manifest:
        du = manifest["distractor_usage"]
        print(
            f"  distractor usage: {du['distractors_used']} unique  "
            f"min={du['min_uses']}  max={du['max_uses']}  mean={du['mean_uses']:.1f}"
        )


if __name__ == "__main__":
    main()
