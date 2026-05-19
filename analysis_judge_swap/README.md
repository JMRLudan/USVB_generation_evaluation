# analysis_judge_swap

Judge-swap robustness analysis. Re-judges the canonical body-roster runs
(currently judged by `gemini-3-flash-preview`) with a second judge,
`claude-haiku-4-5-20251001`, and reports cross-judge SR agreement
per (model, condition).

This is a sibling of `analysis_ablation/`. All artifacts produced here
are self-contained — judged TSVs land in `results/` inside this folder,
not in the canonical `data/runs/` tree.

## Layout

```
analysis_judge_swap/
├── README.md                       this file
├── __init__.py
├── run_haiku_rejudge.py            submit / status / fetch via Anthropic Batches
├── compute_cross_judge_table.py    derived analysis: per-(model, cond) dual-judge SR
├── batch_manifests/                per-batch metadata (one JSON per (subject, preset))
└── results/
    ├── canon_no_distractor/<subject_model>/<subject_run_id>/results.tsv
    └── canon_unified/<subject_model>/<subject_run_id>/results.tsv
```

The `results/` schema matches `data/runs/<preset>/<subject>/<run_id>/results.tsv`
exactly (v2 judge schema with `recommendation` / `constraint_mentioned` /
`sufficiently_modified` / `reasoning`), but with `claude-haiku-4-5-20251001`
as the judge instead of `gemini-3-flash-preview`. Subject fields
(`raw_response`, `input_tokens`, `output_tokens`, etc.) are preserved
verbatim from the canonical run.

## How to run

```bash
# 1. Estimate cost before submitting anything (no API calls).
python3 -m analysis_judge_swap.run_haiku_rejudge estimate-cost

# 2. Submit one (subject, preset). Subject must already have a row in
#    data/runs/<preset>/<subject>/<run_id>/results.tsv.
python3 -m analysis_judge_swap.run_haiku_rejudge submit \
    --preset canon_no_distractor --model-dir claude-opus-4-7

# 2'. Submit all subjects in a preset at once (use with care — this
#     submits N batches in parallel).
python3 -m analysis_judge_swap.run_haiku_rejudge submit \
    --preset canon_no_distractor --all-models

# 3. Poll status.
python3 -m analysis_judge_swap.run_haiku_rejudge status \
    --preset canon_no_distractor --all-models

# 4. Fetch + write judged TSVs once batches complete.
python3 -m analysis_judge_swap.run_haiku_rejudge fetch \
    --preset canon_no_distractor --all-models

# 5. Compute the cross-judge comparison table.
python3 -m analysis_judge_swap.compute_cross_judge_table
```

## Conventions

- Subject inference is **not** re-run. Only the judge call is re-issued
  against the existing `raw_response`.
- Judge prompt is `pipeline.eval_pipeline.JUDGE_SYSTEM` (the canonical
  v2 prompt), identical to what `gemini-3-flash-preview` saw.
- Judge model: `claude-haiku-4-5-20251001` at `temperature=0.0`,
  `max_tokens=4096`, via Anthropic Messages Batches API
  (50% batch discount; ~24h SLA).
- Output schema matches the canonical `data/runs/` TSV column-for-column
  so downstream analysis code can point at either tree interchangeably.
