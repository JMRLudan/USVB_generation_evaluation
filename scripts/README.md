# USVB scripts — reproduction recipe

The `scripts/` directory holds four utilities for running, monitoring,
summarizing, and publishing USVB canon evaluations.

```
scripts/
├── README.md                # this file
├── run_canon.sh             # canonical single-model launcher (3 presets in parallel)
├── status.sh                # auto-discover progress tracker
├── per_model_card.py        # post-run quick-summary card (also called by run_canon.sh)
└── build_data_tarball.sh    # package canon results + prompts for the GitHub release
```

## Prerequisites

1. Python 3.10+ with the runtime deps installed —
   `pip install -r requirements.txt` from the repo root.
2. API keys at repo root: copy `.env.example` to `.env` and fill in your
   values (`GEMINI_API_KEY` is required for the judge; `OPENROUTER_API_KEY`
   is required for any non-Anthropic subject model; `ANTHROPIC_API_KEY` for
   Anthropic subjects).
3. The 85-scenario canon prompts in `generated/canon_{no_distractor,unified}/`
   (extracted from the data tarball — see top-level `README.md`).

## Run a single model (canonical recipe)

```bash
bash scripts/run_canon.sh --model <openrouter-slug>
```

Examples:

```bash
# Anthropic / OpenAI / Gemini frontier — system defaults
bash scripts/run_canon.sh --model anthropic/claude-sonnet-4.6
bash scripts/run_canon.sh --model openai/gpt-5
bash scripts/run_canon.sh --model google/gemini-3-flash-preview

# Open-source comparators
bash scripts/run_canon.sh --model openai/gpt-oss-20b
bash scripts/run_canon.sh --model deepseek/deepseek-v4-pro
```

Each invocation runs both canon presets (canon_no_distractor and
canon_unified) in parallel, judges each row inline with gemini-3-flash,
and prints a per-model card on completion.

## Reproduce the open-source roster + ablation

```bash
# Qwen no-thinking ladder (5-point scaling curve)
for m in qwen/qwen3.5-9b qwen/qwen3.5-27b qwen/qwen3.5-35b-a3b \
         qwen/qwen3.5-122b-a10b qwen/qwen3.5-397b-a17b; do
    bash scripts/run_canon.sh --model "$m" --reasoning off --model-tag reasoning-off
done

# 9b reasoning ablation (paired with the no-thinking 9b run above)
bash scripts/run_canon.sh --model qwen/qwen3.5-9b --reasoning default --model-tag reasoning-on

# Open-source comparators (system-default reasoning)
for m in openai/gpt-oss-20b openai/gpt-oss-120b deepseek/deepseek-v4-pro; do
    bash scripts/run_canon.sh --model "$m" --reasoning default
done
```

For exact inference parameters per model (temperature, max_tokens,
reasoning settings), see `INFERENCE.md` at repo root.

## Reproduce the frontier roster (Anthropic / OpenAI / Google)

These were originally launched via batch APIs for cost. The
`run_canon.sh` real-time path produces identical output rows; choose
based on your budget and concurrency tolerance.

```bash
# Anthropic frontier
bash scripts/run_canon.sh --model anthropic/claude-sonnet-4.6
bash scripts/run_canon.sh --model anthropic/claude-opus-4.7

# OpenAI
bash scripts/run_canon.sh --model openai/gpt-5-mini
bash scripts/run_canon.sh --model openai/gpt-5
bash scripts/run_canon.sh --model openai/gpt-5.5

# Google Gemini
bash scripts/run_canon.sh --model google/gemini-3.1-flash-lite-preview
bash scripts/run_canon.sh --model google/gemini-3-flash-preview
bash scripts/run_canon.sh --model google/gemini-3.1-pro-preview
```

## Track progress

```bash
bash scripts/status.sh             # one-shot snapshot
bash scripts/status.sh --loop      # auto-refresh every 15s
```

Auto-discovers every model dir under `data/runs/canon_*/` — no
configuration needed.

## Inspect results

```bash
# Spawn the viewer (Flask app on port 5057)
python3 viewer/app.py

# Per-model card from the command line
python3 scripts/per_model_card.py --model qwen/qwen3.5-9b
python3 scripts/per_model_card.py --all
```

The viewer renders all per-model and per-preset metrics. The Frontier
tab's "Baseline vs vigilance" chart shows all models in the roster
grouped by stage.

## Flags reference (`run_canon.sh`)

| Flag | Default | Notes |
|---|---|---|
| `--model` | (required) | OR slug, e.g. `qwen/qwen3.5-9b` |
| `--reasoning` | `default` | `default` / `off` / `low`. `default` = no override (system default). `off` = `reasoning.enabled=false`. `low` = `reasoning.effort=low` (legacy). |
| `--model-tag` | (none) | Suffix appended to dir name only. Use to keep multiple runs of one model under separate viewer dirs (e.g. `reasoning-on` vs `reasoning-off`). |
| `--concurrency` | 100 | Per-preset OR concurrency. Drop to 30–50 if you hit account-tier rate limits frequently. |
| `--max-tokens` | 30000 | Per-call output cap. Reasoning models with thinking on can want 10K+ tokens; 30K leaves headroom. |

## What this does not cover

- Building the canon prompt set. See `pipeline/renderers/` and
  `DESIGN.md`. The repo ships pre-rendered prompts in `generated/`;
  re-render only if you change the scenarios or the distractor pool.
- Batch-API submission. The original frontier and XL runs used
  `pipeline/batch_runner.py` directly against the Anthropic, OpenAI,
  and Gemini Batch APIs (50% pricing). The real-time path documented
  here produces output rows with the same schema. See
  `pipeline/batch_runner.py --help` for batch usage.
