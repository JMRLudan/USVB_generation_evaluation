# analysis_ablation

Reasoning-ablation contrasts that extend the paper's Appendix I from N=2
paired pairs (Qwen3.5-9B reasoning-on/off, Sonnet 4.6 thinking-on/off)
to N=6 by adding four more. Sibling of `analysis_judge_swap/`; all
artifacts are self-contained — judged TSVs land in `results/` inside
this folder, not in the canonical `data/runs/` tree.

## Ablation rows added (vs paper Table 8 body-roster baselines)

| Model | Mode | ND | WD | Δ vs default |
|---|---|---:|---:|---|
| Claude Haiku 4.5 | + `thinking={"type":"enabled","budget_tokens":4096}` | 73.8 | 32.6 | +18.5 / +9.7 (gap widens +8.8) |
| Gemini 3 Flash | + `thinkingConfig.thinkingBudget=24576` | 90.7 | 83.5 | −1.6 / −2.1 (no-op; see note below) |
| Gemini 3.1 Flash-Lite | + `thinkingConfig.thinkingBudget=24576` | 92.1 | 73.7 | +5.4 / +16.9 (gap closes −11.5) |
| GPT-5-mini | + `reasoning.effort="high"` | 79.8 | 60.8 | −3.1 / +12.2 (gap closes −15.3) |

Match the Sonnet 4.6 thinking-on row from the paper's Appendix I to
form a 3-rung Anthropic ladder (Haiku → Sonnet → Opus *not run, cost*).
Match Qwen3.5-9B reasoning-on/off paired row from the paper to form
the two open-source budget-axis points.

## Layout

```
analysis_ablation/
├── README.md                              this file
├── __init__.py
├── run_canon_haiku_thinking.py            Anthropic Messages Batch runner
├── run_canon_gemini_thinking.py           Google AI Files Batch runner (--model + --level)
├── run_canon_gemini_realtime.py           Native google.genai async runner (unused in final)
├── run_judge_batch.py                     Gemini-batch judge for ablation TSVs
├── smoke_test_haiku_thinking.py           Smoke smoke recovered from git fc21e35
└── results/
    ├── canon_no_distractor/<model-dir>/<run_id>/results.tsv
    └── canon_unified/<model-dir>/<run_id>/results.tsv
```

The `results/` schema matches `data/runs/<preset>/<subject>/<run_id>/results.tsv`
exactly (v2 judge schema: `recommendation` / `constraint_mentioned` /
`sufficiently_modified` / `explanation` + derived booleans). Subjects via
each model's batch endpoint; judges via `analysis_ablation/run_judge_batch.py`
(Gemini Batch with `gemini-3-flash-preview`, temperature=0).

Bulk TSVs are also packaged in the GitHub release tarball
`usvb-data-ablation-v1.tar.gz` for users who want to download just
the data without checking out the full repo. The tarball also bundles
`analysis_judge_swap/results/`.

## Methodological note — Gemini Flash no-op

`thinkingBudget=24576` is **not honored** on the Gemini batch endpoint
for `gemini-3-flash-preview`: the empirical output-token distribution
under the override (ND mean 1,060; WD mean 1,277) is statistically
indistinguishable from the body-roster default (ND 1,050; WD 1,276).
The batch endpoint silently demotes max-budget values to dynamic.
For Gemini 3.1 Flash-Lite the override **is** honored (output tokens
3.1× / 2.9× default) because Flash-Lite defaults to near-minimal
thinking, so the ceiling actually unlocks more headroom.

The semantic `thinkingLevel="high"` (which works on the real-time
endpoint) is rejected by the batch endpoint with `JOB_STATE_FAILED`
and no error message. See JOURNAL 2026-05-19 for the full empirical
narrative.

## Cross-ablation pattern

Three patterns emerge across the N=6 paired contrasts (paper's 2 +
this directory's 4):

- **Parallel lift, gap stable or widens** — Haiku 4.5 (+8.8 gap),
  Qwen3.5-9B (−1.5 ≈ stable). Small / less-vigilant models.
- **Gap closes** — Sonnet 4.6 (−5.1), Flash-Lite (−11.5),
  GPT-5-mini (−15.3). Already-vigilant frontier and already-thinking
  models hit with explicit effort or budget pushes.
- **No-op** — Gemini 3 Flash thinkingBudget=24576. Diagnostic finding.
