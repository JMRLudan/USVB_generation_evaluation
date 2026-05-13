# Viewer

A read-only Flask app that visualizes canon results from
`data/runs/{canon_no_distractor,canon_unified}/`. Nothing in this viewer
calls any LLM API.

## Tabs

- **Charts** — headline per-model summary metrics.
- **Scenarios** — per-scenario rollup view.
- **Results** — per-row eval results with subject/judge fields.
- **Prompts** — browse the rendered prompt files.
- **SR Surface** — 2D `SR(length, depth)` surface over `canon_unified`.
- **Frontier** — frontier-model comparison view.

## Launch

```bash
python3 viewer/app.py
```

Then open <http://127.0.0.1:5057>.
