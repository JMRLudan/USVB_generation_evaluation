# USVB — Design Notes for the Paper

This is framing material for the methods section. The code is what it is;
this document is about how to *describe* it.

---

## The mixer as a parameterized generative function

`pipeline/renderers/mixer.py` exposes a single function `mix()` that maps
a configuration point to a deterministic set of prompts. Every renderer
in this repository is a thin wrapper that supplies a specific config.

The configuration space is a 7-dimensional hypercube:

| Axis | Type | What it controls |
|---|---|---|
| `n_distractor_draws` | `int ≥ 0` | Number of full re-renders of the scenario set. `0` → no distractor path at all. With `≥ 2` each item is sampled multiple times against independent distractor selections (per-row independent length / placement when stratified-sampling those axes too). |
| `n_distractors_per_prompt` | `int ≥ 1` | How many distractor conversations are merged end-to-end into each prompt's history. `1` → classic single-distractor; `≥ 2` → the stitched variant. |
| `n_placements` | `int ≥ 0` | How many evidence insertion points per `(item, draw, length)` cell. |
| `placement_mode` | `{fixed, uniform, uniform_stratified}` | `fixed` uses `placements_list`; `uniform_stratified` produces one stratified placement per item, balanced within scenario. |
| `n_lengths` | `int ≥ 0` | How many char budgets per cell. |
| `lengths_named` / `lengths_list` | `dict[str,int]` / `list[int]` | The actual budgets when `length_mode="fixed"`. |
| `length_mode` | `{fixed, log_uniform_stratified}` | `fixed` consumes `lengths_named`/`lengths_list` as discrete budgets. `log_uniform_stratified` ignores those and samples one char budget per cell on a log-uniform scale over `length_range`, stratified within scenario. |
| `length_range` | `tuple(int, int)` | Min/max char budgets when `length_mode="log_uniform_stratified"`. |
| `include_constraint_inline` | `bool` | Whether the constraint description is folded into the user message vs inserted into history. Always `False` for the two reported conditions. |
| `c_only` | `bool` | Whether to restrict to `C`-bearing variants. The full canon since 2026-05-01 sets this `False` so `A` and `B` no-C variants are enumerated as well. |
| `merge_gap_days` | `int ≥ 0` | Inter-chat timestamp gap when `n_distractors_per_prompt ≥ 2`. |

Plus a fixed `seed` (`4232026`) and a scenario TSV. Given these, `mix()`
is a pure function: same config → same prompts, byte-for-byte.

### The conditions in the paper are points in this space

The two canonical conditions map cleanly onto specific config points:

| Condition | `n_d_draws` | `n_d_per_prompt` | `n_p` | `n_l` | `placement_mode` | `length_mode` | `length_range` | `include_constraint_inline` |
|---|---|---|---|---|---|---|---|---|
| **No-distractor / primary** (`canon_no_distractor`) | 0 | 1 | 0 | 0 | — | — | — | `False` |
| **Unified with-distractor** (`canon_unified`) | 3 | 3 | 1 | 1 | `uniform_stratified` | `log_uniform_stratified` | `(3000, 250000)` | `False` |

Both set `c_only = False` so each enumerates the full 5-variant
scenario set (`C, A+C, B+C, A, B`) — 2 122 distinct
`(scenario, variant, perm)` tuples on the 85-scenario corpus.
`canon_unified` resamples each tuple three times (`n_distractor_draws=3`),
producing 6 366 prompts; each row gets an independent
`(distractor_set, length, placement)` triple with length sampled
log-uniformly on `[3 000, 250 000]` chars and placement uniformly on
`[0, 1]`. `merge_gap_days = 1` everywhere.

Total per-model prompt count: 2 122 + 6 366 = **8 488**.

### Why a single unified with-distractor preset

Earlier iterations of this canon split with-distractor evaluation into
three fixed-length presets (short/medium/long). The unified design
sweeps both length and depth jointly per row. Two reasons drove the
change:

1. **Length is a continuous variable.** Treating it as three discrete
   levels obscures the shape of the degradation curve. With per-row
   log-uniform length sampling and per-row uniform depth, the SR
   surface can be plotted as `SR(length, depth)` rather than three
   superimposed depth curves at fixed lengths.
2. **Per-prompt economics.** The geometric mean of
   log-uniform([3 000, 250 000]) is ≈ 27 000 chars, vs the weighted
   mean of the prior three-tier setup (~115 000 chars). With 6 366
   unified prompts replacing 4 938 across three tiers, the unified
   preset uses ~30 % fewer total tokens at higher per-scenario
   coverage.

Earlier iterations of the canon used a fixed-grid condition (5 fixed
depths × 3 haystacks) and a three-tier `canon_uniform_short/medium/long`
preset. Both have been retired in favor of the unified preset and are
no longer wired into the codebase. They can be reproduced by calling
`mixer.mix()` directly with the appropriate fixed-mode parameters if a
back-compat replication is needed.

Any new condition — ablations, rebuttal experiments, robustness checks —
is defined the same way: specify the point in the hypercube, call
`mix()`, commit the call site. Nothing is bespoke.

---

## Two layers of permutation

It is worth distinguishing these explicitly in the paper, because
"permutation" gets overloaded otherwise.

**Layer 1 — scenario-level seed permutations.** Each scenario in
`scenarios_FINAL.tsv` supplies up to three C-seeds (evidence sentences
supporting the safety constraint), three A-seeds (supporting option A),
and three B-seeds (supporting option B). The `A+C` variant enumerates
the Cartesian product of the relevant seed indices — `c0_a0`, `c0_a1`, …,
`c2_a2` — producing up to 9 permutations per `(scenario, variant)` pair.
The `A` and `B` no-C variants enumerate just their own seed-index range
(up to 3 perms each). With the validated filter, 85 scenarios × 5
variants × ~5 average perms ≈ **2 122 items**, broken down as 248 (`C`)
+ 699 (`A+C`) + 699 (`B+C`) + 238 (`A`) + 238 (`B`). This layer is
*upstream* of the mixer and identical across every condition.

**Layer 2 — mixer-level permutations.** Given an `(scenario_id,
variant, scenario_perm)` item, the mixer decides:

- Which distractor hash(es) to pull from the distractor pool (per-draw,
  per-slot shuffle; balanced usage). The pool size is read at runtime
  from `data/distractors/index.json` (94 conversations as of the
  shipped canon).
- At what normalized depth(s) to insert evidence (fixed list or
  stratified sample).
- Against which char budget(s) to truncate.

Every point in the config hypercube specifies how the mixer permutes
these — typically producing multiple prompts per item (e.g. the grid
sweep emits 5 depths × 3 haystacks = 15 prompts per item).

The two layers compose independently. The paper should refer to them by
distinct terms (we suggest "scenario permutations" vs "placement /
distractor configuration") to avoid confusion.

---

## Reproducibility claim

For any `(scenario_id, evidence_variant, scenario_perm, mix_config,
seed)`, the produced prompt is byte-identical on re-run. Concretely:

- Scenario loading is deterministic (TSV row order → sorted
  enumeration).
- Seed-permutation enumeration is deterministic (Cartesian product in
  fixed axis order).
- Distractor assignment is deterministic: per-slot
  `Random(seed + draw_idx + slot × stride).sample(pool, |pool|)`,
  then round-robin over items in sorted-triple order, with a
  reject-sample pass on same-item collisions.
- Stratified placements are derived from
  `sha256(scenario_id|variant|perm|draw_idx|length_idx|bin_idx)`; the
  per-scenario stratified placement assignment additionally uses
  `random.Random(seed ^ _det_seed("scenario_strat", sid))` to shuffle
  bin midpoints over cells, where `_det_seed` is a sha256-based
  deterministic hash (Python's built-in `hash()` on strings is
  process-randomized unless PYTHONHASHSEED is fixed, which silently
  broke cross-process reproducibility before 2026-05-01).
- Stratified log-uniform char budgets (when
  `length_mode="log_uniform_stratified"`) come from log-equal-width
  bin midpoints over `length_range`, then shuffled with
  `random.Random(seed ^ _det_seed("scenario_strat_loglen", sid))`.
  Length and placement use distinct shuffles, so the two axes are
  independent per cell.
- Timestamp shifts (for `n_distractors_per_prompt ≥ 2`) are computed
  from each distractor's own `min` / `max` timestamps — no wall-clock
  dependency.

No component touches a random generator that isn't seeded from these
inputs. This is verifiable by diffing two separate renders.

---

## Fluency rendering on the with-distractor path

Inserted evidence seeds and stitched distractor chats are wrapped with
short framing phrases so the assembled prompt reads as one coherent
multi-day dialogue rather than three pasted-together topics. This
applies only to the canon_unified path (`legacy_fluency=False`, the
default); `canon_no_distractor` is unaffected.

| Where | What |
|---|---|
| First user turn of each evidence-seed pair | Prepended with `EVIDENCE_PREFIXES[seed_idx % 2]`: `"Just wanted to randomly say this, a short acknowledgement would suffice - "` for the first seed, `"Another random note, Just acknowledge and move on - "` for the second. |
| First user turn after the inserted evidence block | Prepended with `RESUMPTION_PREFIXES[insert_pair_idx % 2]`: `"Going back to earlier - "` or `"Resuming the earlier topic - "`. Skipped when insertion lands at the end of the haystack (no following turn to resume into). |
| Boundary between stitched distractor chats | A `---` divider line emitted by `turns_to_text` above the first turn of every non-first chat (when `n_distractors_per_prompt ≥ 2`). |

All three pieces are deterministic: the prefix choices are pure
functions of indices already present in the row, no extra RNG.

---

## Char-budget semantics: floor, not ceiling

`truncate_pairs_to_budget` keeps a contiguous prefix of pairs whose
running char total **reaches or just exceeds** the requested
`char_budget`. The most recently added pair is the one that crossed
the line. If the entire pair list fits below the budget, all pairs
are kept.

This is the project's **only** allowed truncation direction
("keep-beginning, drop-end") — it just stops one pair later than a
strict ceiling rule would. The floor semantics prevent an
empty-haystack pathology at the low end of canon_unified's
log-uniform char range: a 1.6K per-chat budget split across three
stitched distractors could under a strict-ceiling rule produce zero
pairs per chat, leaving the model with effectively no haystack. Floor
semantics guarantees ≥1 pair per non-empty chat. At medium and high
budgets the difference is a single pair of overshoot, dwarfed by
haystack size.

---

## What this design *does not* control

- **The distractor pool content.** The distractor conversations in
  `data/distractors/` (94 in the shipped canon) are fixed artifacts —
  synthesized upstream and deduplicated. The mixer picks from the pool
  but does not generate
  new chat content. Consequence: phrases like "Got it." or "I see."
  that appear organically inside distractors show up in rendered
  prompts, regardless of what `DEFAULT_ACKS` is set to. Evidence-ack
  text is the only dialogue the mixer itself authors.
- **The scoring semantics.** Rules for SR and Choice Correct live in
  `SCORING.md` and `pipeline/eval_pipeline.py:JUDGE_SYSTEM`;
  they are orthogonal to prompt construction.
- **The judge.** Held fixed at gemini-3-flash-preview per project convention,
  with the v2 prompt (3-field schema + REASONING preamble + 4-valued
  RECOMMENDATION). A different judge is a different benchmark.

---

## Suggested phrasing for the methods section

> We define prompt generation as a deterministic function
> `mix(config, seed) → {prompts}` over a seven-axis configuration space.
> The three reported conditions are specific points in this space (see
> Table N). The unified with-distractor condition samples haystack
> length log-uniformly over `[3 000, 250 000]` chars and constraint
> placement uniformly over `[0, 1]`, jointly per row, stratified within
> scenario. Every prompt is fully specified by its `(scenario_id,
> evidence_variant, scenario_perm, mix_config, seed)` tuple; all code
> and artifacts needed to reproduce the reported prompt bytes are
> released at [repo URL].

The canonical config points are direct CLI invocations in the
`pipeline/renderers/render_no_distractor.py` and
`pipeline/renderers/render_unified.py` wrappers. A table in the appendix
listing the exact `mix()` argument values per reported row is probably
worth including.
