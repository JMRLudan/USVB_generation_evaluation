#!/usr/bin/env python3
"""
viewer/app.py — Local Flask app for the USVB canon_unified analysis.
====================================================================

Focused single-purpose viewer. Surfaces only canon_unified results.
Earlier multi-condition/multi-renderer analysis lived in this file
through 2026-05-01; that code has been archived alongside the
non-unified data under ``data/archive_*``.

Quickstart:
    pip install flask
    python viewer/app.py                  # http://127.0.0.1:5057
    python viewer/app.py --port 8080      # different port

Endpoints:
  /                                  serve index.html
  /api/models                        models with canon_unified data
  /api/results/summary               overall metrics for one (model, run)
  /api/results/depth_curve           STAT vs placement_frac
  /api/results/length_curve          STAT vs char_budget (log-binned)
  /api/results/variant_length_curves multi-line: C / A+C / B+C across length
  /api/results/length_depth_surface  optional 2D surface (for export)
  /api/results/reload                clear caches
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    from flask import Flask, jsonify, request, send_from_directory, abort
except ImportError:
    print("Flask is required. pip install flask", file=sys.stderr)
    raise


# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────
VIEWER_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIEWER_DIR.parent
GENERATED_DIR = REPO_ROOT / "generated"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
SCENARIOS_TSV = DATA_DIR / "scenarios_FINAL.tsv"

CONDITION = "canon_unified"

# Increase CSV field-size cap so long raw_response cells don't fail to load.
csv.field_size_limit(sys.maxsize)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# STAT codes recognized by the chart endpoints. Each maps to:
#   (numerator-flag-name, denom-flag-name, applicable-variants)
# where flag-names index the row's bool columns built in _load_run.
# Variants are restricted at query time so the user doesn't get
# nonsense plots (e.g. SR on an A-only variant).
STAT_DEFS: dict[str, dict[str, Any]] = {
    "SR":  {
        "label": "Scenario Reliability (SR)",
        "num": "_vigilance",  "denom": "_vig_set",
        "valid_variants": {"C", "A+C", "B+C"},
    },
    "CM":  {
        "label": "Constraint Mentioned",
        "num": "_cm",         "denom": "_cm_set",
        "valid_variants": {"C", "A+C", "B+C", "A", "B"},
    },
}

C_BEARING = ("C", "A+C", "B+C")


# ──────────────────────────────────────────────────────────────────────
# Caches
# ──────────────────────────────────────────────────────────────────────
_RESULTS_CACHE: dict[tuple[str, str], dict] = {}
_PROMPT_META_CACHE: dict[str, dict] = {}
_SCENARIOS_CACHE: dict[str, dict] = {}


# ──────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────
def _load_scenarios() -> dict[str, dict]:
    if "scenarios" in _SCENARIOS_CACHE:
        return _SCENARIOS_CACHE["scenarios"]
    out: dict[str, dict] = {}
    if SCENARIOS_TSV.exists():
        with open(SCENARIOS_TSV) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r.get("status", "").lower() == "reject":
                    continue
                out[r["id"]] = r
    _SCENARIOS_CACHE["scenarios"] = out
    return out


def _load_prompt_meta() -> dict[tuple[str, str, str], dict]:
    """Walk generated/canon_unified and build a (sid, ev, perm-with-suffixes)
    → metadata lookup. The perm encoding mirrors what the runner writes
    into results.tsv: ``"<base_perm>-d<draw>-l<length>"``.
    """
    if "map" in _PROMPT_META_CACHE:
        return _PROMPT_META_CACHE["map"]
    out: dict[tuple[str, str, str], dict] = {}
    cond_dir = GENERATED_DIR / CONDITION
    if cond_dir.exists():
        for jf in cond_dir.glob("*.json"):
            if jf.name == "manifest.json":
                continue
            try:
                with open(jf) as f:
                    d = json.load(f)
            except Exception:
                continue
            md = d.get("metadata") or {}
            sid = md.get("scenario_id")
            ev = md.get("evidence_variant")
            base_perm = md.get("permutation")
            if not (sid and ev and base_perm is not None):
                continue
            full_perm = str(base_perm)
            di = md.get("draw_idx")
            li = md.get("length_idx")
            if di is not None:
                full_perm += f"-d{di}"
            if li is not None:
                full_perm += f"-l{li}"
            key = (sid, ev, full_perm)
            if key not in out:
                out[key] = md
    _PROMPT_META_CACHE["map"] = out
    return out


def _coerce_bool(v: Any) -> bool:
    return v in ("1", 1, True, "True", "true")


def _load_run(model: str, run_id: str, preset: str = CONDITION) -> dict:
    """Load + parse a single results.tsv. Cached.

    `preset` defaults to canon_unified (the headline view) but can be
    overridden by Scenario-tab endpoints to load canon_no_distractor
    instead. Cache key includes preset so flipping presets in the UI
    doesn't poison the canon_unified cache.
    """
    key = (model, run_id, preset)
    if key in _RESULTS_CACHE:
        return _RESULTS_CACHE[key]

    tsv = RUNS_DIR / preset / model.replace("/", "_") / run_id / "results.tsv"
    if not tsv.exists():
        return {"rows": [], "fields": [], "tsv": str(tsv)}

    scenarios = _load_scenarios()
    pm = _load_prompt_meta()

    rows: list[dict] = []
    with open(tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        for r in reader:
            r["_is_error"] = (r.get("raw_response") or "").startswith(("ERROR", '"ERROR'))
            # v2 SR is read from `vigilance_refuse_only` (refuse-only
            # success path). The `vigilance` column on legacy TSVs is
            # read as a fallback for back-compat.
            r["_vigilance"] = _coerce_bool(r.get("vigilance_refuse_only")
                                           or r.get("vigilance"))
            r["_abstained"] = _coerce_bool(r.get("abstained"))
            r["_vig_set"] = ((r.get("vigilance_refuse_only") not in ("", None)
                              or r.get("vigilance") not in ("", None))
                             and not r["_is_error"])
            cm = (r.get("constraint_mentioned") or "").strip().upper()
            r["_cm_set"] = cm in ("YES", "NO")
            r["_cm"] = (cm == "YES")
            for k in ("input_tokens", "output_tokens",
                      "judge_input_tokens", "judge_output_tokens"):
                try:
                    r["_" + k] = int(r.get(k, 0) or 0)
                except (TypeError, ValueError):
                    r["_" + k] = 0
            sid = r.get("scenario_id", "")
            sc = scenarios.get(sid, {})
            full = sc.get("domain", "") or ""
            prefix = full.split("—", 1)[0].strip() if "—" in full else full.strip()
            r["_domain_full"] = full
            r["_domain_pre"] = prefix
            r["_risk_level"] = sc.get("risk_level", "")
            perm_raw = r.get("permutation", "")
            md = pm.get((sid, r.get("evidence_variant", ""), perm_raw), {})
            if not md and "-d" not in perm_raw:
                # Fallback: an early run.py code path wrote the BARE
                # permutation (`c0_a0`) instead of the suffixed
                # form (`c0_a0-d0-l0`). For canon_unified, the 3 draw
                # variants share the base perm, so we can't pick the exact
                # resample. Use the first matching draw_idx=0 / length_idx=0
                # entry as a best-effort approximation. Charts based on
                # placement_frac / char_budget will be approximate for
                # such runs (a footnote in the writeup is appropriate).
                fallback_key = (sid, r.get("evidence_variant",""), f"{perm_raw}-d0-l0")
                md = pm.get(fallback_key, {})
                if md:
                    r["_perm_join_approximate"] = True
            try:
                r["placement_frac"] = float(md["placement_frac"]) if "placement_frac" in md else None
            except (TypeError, ValueError):
                r["placement_frac"] = None
            try:
                r["char_budget"] = int(md.get("char_budget") or 0) or None
            except (TypeError, ValueError):
                r["char_budget"] = None
            r["distractor_domains"] = md.get("distractor_domains") or []
            r["n_distractor_pairs"] = md.get("n_distractor_pairs")
            rows.append(r)

    out = {"rows": rows, "fields": fields, "tsv": str(tsv)}
    _RESULTS_CACHE[key] = out
    return out


def _list_models() -> list[dict[str, Any]]:
    """Discover (model_dir, run_id) pairs under data/runs/canon_unified/.
    For each model, surface the most recent run as the default.
    """
    cond_dir = RUNS_DIR / CONDITION
    out: list[dict[str, Any]] = []
    if not cond_dir.exists():
        return out
    for model_dir in sorted(cond_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        runs = []
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            tsv = run_dir / "results.tsv"
            if not tsv.exists():
                continue
            mtime = tsv.stat().st_mtime
            # Quick row count
            with open(tsv) as f:
                n_rows = sum(1 for _ in f) - 1
            runs.append({
                "run_id": run_dir.name,
                "n_rows": n_rows,
                "mtime": mtime,
            })
        if not runs:
            continue
        runs.sort(key=lambda x: x["mtime"], reverse=True)
        latest = runs[0]
        out.append({
            "model": model_dir.name,
            "model_display": model_dir.name.replace("_", "/"),
            "latest_run_id": latest["run_id"],
            "latest_n_rows": latest["n_rows"],
            "all_runs": [r["run_id"] for r in runs],
        })
    return out


_VALID_PRESETS = (
    "canon_no_distractor", "canon_unified",
    # naive-prompt mitigation arms (2026-07-08)
    "canon_unified_mit_sysbottom", "canon_unified_mit_querytop",
    "canon_no_distractor_mit_sysbottom", "canon_no_distractor_mit_querytop",
    "canon_unified_mit_systop", "canon_no_distractor_mit_systop",
    # memory-conditioned eval presets (2026-07-08)
    "memcond_profile", "memcond_persn", "memcond_safety",
)


def _req_preset() -> str:
    """Read + validate the `preset` query param (default canon_unified)."""
    p = request.args.get("preset", CONDITION)
    return p if p in _VALID_PRESETS else CONDITION


def _resolve_run_id(model: str, run_id: str | None, preset: str = CONDITION) -> str | None:
    """If run_id is missing, pick the latest run for this (model, preset).

    For presets other than the headline canon_unified, we walk the
    on-disk dir directly since `_list_models()` is canon_unified-only.
    """
    if run_id:
        return run_id
    if preset == CONDITION:
        # Fast path: use the cached _list_models()
        for m in _list_models():
            if m["model"] == model:
                return m["latest_run_id"]
        return None
    # Slow path: scan the preset's dir
    model_dir = RUNS_DIR / preset / model.replace("/", "_")
    if not model_dir.exists():
        return None
    runs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and (d / "results.tsv").exists()],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return runs[0].name if runs else None


# ──────────────────────────────────────────────────────────────────────
# canon_no_distractor reference baseline
# ──────────────────────────────────────────────────────────────────────
_NO_DIST_CACHE: dict[str, list[dict]] = {}


def _load_no_dist_run(model: str) -> list[dict] | None:
    """Read the latest canon_no_distractor results.tsv for this model
    from ``data/runs/canon_no_distractor/<model>/...``.
    """
    if model in _NO_DIST_CACHE:
        return _NO_DIST_CACHE[model]
    model_fs = model.replace("/", "_")

    root = RUNS_DIR / "canon_no_distractor"
    candidates: list[Path] = []
    if root.exists():
        for tsv in root.rglob("results.tsv"):
            # parent = <run_id>; parent.parent = <model_dir>
            if tsv.parent.parent.name == model_fs:
                candidates.append(tsv)

    if not candidates:
        _NO_DIST_CACHE[model] = []
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    tsv = candidates[0]
    rows: list[dict] = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["_is_error"] = (r.get("raw_response") or "").startswith(("ERROR", '"ERROR'))
            r["_vigilance"] = _coerce_bool(r.get("vigilance_refuse_only") or r.get("vigilance"))
            r["_abstained"] = _coerce_bool(r.get("abstained"))
            r["_vig_set"] = ((r.get("vigilance_refuse_only") not in ("", None)
                              or r.get("vigilance") not in ("", None))
                             and not r["_is_error"])
            cm = (r.get("constraint_mentioned") or "").strip().upper()
            r["_cm_set"] = cm in ("YES", "NO")
            r["_cm"] = (cm == "YES")
            rows.append(r)
    _NO_DIST_CACHE[model] = rows
    return rows


def _no_dist_baseline(model: str, stat: str,
                      variant: str | None = None) -> dict[str, Any]:
    """Compute the canon_no_distractor STAT % for the given model
    (and optionally a single variant).

    Returns a dict with `value_pct` (float|None), `n` (int), and
    `available` (bool). When restricted to variants where the STAT
    is valid (mirroring the canon_unified chart filter)."""
    rows = _load_no_dist_run(model)
    if not rows:
        return {"available": False, "value_pct": None, "n": 0}
    valid = STAT_DEFS[stat]["valid_variants"]
    if variant is not None:
        valid = {variant} & valid
    sub = [r for r in rows
           if not r["_is_error"]
           and r.get("evidence_variant") in valid]
    sd = STAT_DEFS[stat]
    num = sum(1 for r in sub if r[sd["num"]])
    den = sum(1 for r in sub if r[sd["denom"]])
    return {
        "available": True,
        "value_pct": round(100 * num / den, 2) if den else None,
        "n": den,
    }


# ──────────────────────────────────────────────────────────────────────
# Stat helpers
# ──────────────────────────────────────────────────────────────────────
def _stat_for_rows(stat: str, rows: list[dict]) -> tuple[int, int]:
    """Return (numerator, denominator) for a STAT given a row slice
    (instance-averaged / micro-averaged)."""
    sd = STAT_DEFS[stat]
    num = sum(1 for r in rows if r[sd["num"]])
    den = sum(1 for r in rows if r[sd["denom"]])
    return num, den


def _macro_avg_pct(rows: list[dict], num_field: str, denom_field: str) -> float | None:
    """Scenario-macro-averaged percentage:
        1. Group rows by scenario_id.
        2. Compute per-scenario rate = sum(num)/sum(denom) for each scenario
           (skip scenarios with denom == 0).
        3. Return the simple mean of per-scenario rates (×100).

    This is what the project uses as the "scenario-averaged" or
    "macro-averaged" headline metric — gives equal weight to every
    scenario regardless of how many rows it has, so a scenario with
    75 rows doesn't dominate one with 50.

    Returns None if no scenario contributes.
    """
    by_scen: dict[str, dict[str, int]] = {}
    for r in rows:
        sid = r.get("scenario_id") or "?"
        s = by_scen.setdefault(sid, {"num": 0, "den": 0})
        if r[denom_field]:
            s["den"] += 1
            if r[num_field]:
                s["num"] += 1
    rates = [s["num"] / s["den"] for s in by_scen.values() if s["den"] > 0]
    if not rates:
        return None
    return 100 * sum(rates) / len(rates)


def _stat_macro_for_rows(stat: str, rows: list[dict]) -> float | None:
    """Scenario-macro-averaged version of _stat_for_rows.
    Returns the percentage directly (or None when no scenario contributes)."""
    sd = STAT_DEFS[stat]
    return _macro_avg_pct(rows, sd["num"], sd["denom"])


def _filter_rows_for_stat(rows: list[dict], stat: str,
                          variant: str | None = None) -> list[dict]:
    """Restrict to rows where the STAT is defined.
    Optionally further restrict by an exact variant name."""
    sd = STAT_DEFS[stat]
    valid = sd["valid_variants"]
    out = []
    for r in rows:
        if r["_is_error"]:
            continue
        v = r.get("evidence_variant")
        if v not in valid:
            continue
        if variant is not None and v != variant:
            continue
        out.append(r)
    return out


# ──────────────────────────────────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(VIEWER_DIR / "static"))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ───── data discovery ─────────────────────────────────────────────────
@app.route("/api/models")
def api_models():
    return jsonify({
        "condition": CONDITION,
        "models": _list_models(),
    })


@app.route("/api/results/reload", methods=["POST", "GET"])
def api_results_reload():
    n = len(_RESULTS_CACHE)
    p = len(_PROMPT_META_CACHE)
    s = len(_SCENARIOS_CACHE)
    _RESULTS_CACHE.clear()
    _PROMPT_META_CACHE.clear()
    _SCENARIOS_CACHE.clear()
    return jsonify({"ok": True, "cleared": {
        "results": n, "prompt_meta": p, "scenarios": s,
    }})


# ───── summary ────────────────────────────────────────────────────────
@app.route("/api/results/summary")
def api_results_summary():
    """Overall metrics for a (model, run) pair."""
    model = request.args.get("model", "")
    preset = _req_preset()
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    if not (model and run_id):
        abort(400, description="missing model and/or run_id")
    data = _load_run(model, run_id, preset)
    rows = [r for r in data["rows"] if not r["_is_error"]]
    if not rows:
        abort(404, description=f"no rows for {model}/{run_id}")

    out: dict[str, Any] = {
        "model": model, "run_id": run_id,
        "n_rows_total": len(data["rows"]),
        "n_rows_ok": len(rows),
        "tsv": data["tsv"],
    }

    # Overall + per-variant counts and metric values.
    by_v: dict[str, dict[str, int]] = {}
    for r in rows:
        v = r.get("evidence_variant", "?")
        e = by_v.setdefault(v, {"n": 0,
                                "vig": 0, "vig_set": 0,
                                "cm": 0, "cm_set": 0,
                                "abstain": 0})
        e["n"] += 1
        if r["_vig_set"]:
            e["vig_set"] += 1
            if r["_vigilance"]: e["vig"] += 1
            if r["_abstained"]: e["abstain"] += 1
        if r["_cm_set"]:
            e["cm_set"] += 1
            if r["_cm"]: e["cm"] += 1

    def pct(n, d): return round(100 * n / d, 2) if d else None

    # ─── Per-variant metrics — scenario-macro-averaged so seed/resample
    # balances don't overweight some scenarios. For each variant we
    # restrict rows to that variant and macro-average over scenarios.
    variants_summary = []
    for v in ("C", "A+C", "B+C", "A", "B"):
        v_rows = [r for r in rows if r.get("evidence_variant") == v]
        if not v_rows: continue
        variants_summary.append({
            "variant": v,
            "n": len(v_rows),
            "SR":  _macro_avg_pct(v_rows, "_vigilance",   "_vig_set"),
            "CM":  _macro_avg_pct(v_rows, "_cm",          "_cm_set"),
            "abstain": _macro_avg_pct(v_rows, "_abstained","_vig_set"),
        })
    out["variants"] = variants_summary

    # ─── Overall — macro-averaged within the STAT's valid_variants subset
    all_rows = rows
    out["overall"] = {}
    for stat in STAT_DEFS:
        sub = _filter_rows_for_stat(all_rows, stat)
        out["overall"][stat] = _stat_macro_for_rows(stat, sub)

    return jsonify(out)


# ───── chart 1: depth curve ───────────────────────────────────────────
@app.route("/api/results/depth_curve")
def api_results_depth_curve():
    """Bin by placement_frac decile. Returns x = depth midpoint, y = STAT %.
    `variant` may pin a single variant (default: aggregate over the STAT's
    valid variants).
    """
    model = request.args.get("model", "")
    preset = _req_preset()
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    stat = request.args.get("stat", "SR").upper()
    variant = request.args.get("variant") or None
    try:
        n_bins = int(request.args.get("n_bins", "10"))
    except ValueError:
        n_bins = 10
    n_bins = max(2, min(50, n_bins))

    if stat not in STAT_DEFS:
        abort(400, description=f"unknown stat {stat!r}")
    if not (model and run_id):
        abort(400, description="missing model/run_id")

    data = _load_run(model, run_id, preset)
    if not data["rows"]:
        abort(404, description=f"no rows for {model}/{run_id}")

    rows = _filter_rows_for_stat(data["rows"], stat, variant=variant)
    rows = [r for r in rows if r.get("placement_frac") is not None]
    if not rows:
        return jsonify({
            "stat": stat, "variant": variant,
            "n_bins": n_bins, "n_total": 0, "bins": [],
            "warn": "no rows have a placement_frac for this STAT/variant",
        })

    bins = [{
        "frac_lo": i / n_bins, "frac_hi": (i + 1) / n_bins,
        "frac_mid": (i + 0.5) / n_bins,
        "num": 0, "den": 0, "n": 0,
    } for i in range(n_bins)]
    sd = STAT_DEFS[stat]
    for r in rows:
        idx = int(r["placement_frac"] * n_bins)
        if idx >= n_bins: idx = n_bins - 1
        b = bins[idx]
        b["n"] += 1
        if r[sd["denom"]]:
            b["den"] += 1
            if r[sd["num"]]: b["num"] += 1
    for b in bins:
        b["value_pct"] = round(100 * b["num"] / b["den"], 2) if b["den"] else None
    return jsonify({
        "model": model, "run_id": run_id,
        "stat": stat, "stat_label": sd["label"],
        "variant": variant,
        "n_bins": n_bins, "n_total": len(rows),
        "bins": bins,
        "baseline_no_distractor": _no_dist_baseline(model, stat, variant),
    })


# ───── chart 2: length curve ──────────────────────────────────────────
@app.route("/api/results/length_curve")
def api_results_length_curve():
    """Bin by char_budget on a log scale. Returns x = log10(midpoint),
    y = STAT %.
    """
    model = request.args.get("model", "")
    preset = _req_preset()
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    stat = request.args.get("stat", "SR").upper()
    variant = request.args.get("variant") or None
    try:
        n_bins = int(request.args.get("n_bins", "10"))
    except ValueError:
        n_bins = 10
    n_bins = max(2, min(50, n_bins))

    if stat not in STAT_DEFS:
        abort(400, description=f"unknown stat {stat!r}")
    if not (model and run_id):
        abort(400, description="missing model/run_id")

    data = _load_run(model, run_id, preset)
    if not data["rows"]:
        abort(404, description=f"no rows for {model}/{run_id}")

    rows = _filter_rows_for_stat(data["rows"], stat, variant=variant)
    rows = [r for r in rows if r.get("char_budget")]
    if not rows:
        return jsonify({
            "stat": stat, "variant": variant,
            "n_bins": n_bins, "n_total": 0, "bins": [],
            "warn": "no rows have a char_budget for this STAT/variant",
        })

    cb_min = max(1, min(r["char_budget"] for r in rows))
    cb_max = max(r["char_budget"] for r in rows)
    log_min, log_max = math.log10(cb_min), math.log10(cb_max)
    span = log_max - log_min if log_max > log_min else 1.0

    bins = [{
        "log10_lo": log_min + span * i / n_bins,
        "log10_hi": log_min + span * (i + 1) / n_bins,
        "n": 0, "num": 0, "den": 0,
    } for i in range(n_bins)]
    for b in bins:
        b["log10_mid"] = (b["log10_lo"] + b["log10_hi"]) / 2
        b["chars_lo"] = 10 ** b["log10_lo"]
        b["chars_hi"] = 10 ** b["log10_hi"]
        b["chars_mid"] = 10 ** b["log10_mid"]

    sd = STAT_DEFS[stat]
    for r in rows:
        log_cb = math.log10(r["char_budget"])
        idx = int((log_cb - log_min) / span * n_bins)
        if idx >= n_bins: idx = n_bins - 1
        if idx < 0: idx = 0
        b = bins[idx]
        b["n"] += 1
        if r[sd["denom"]]:
            b["den"] += 1
            if r[sd["num"]]: b["num"] += 1
    for b in bins:
        b["value_pct"] = round(100 * b["num"] / b["den"], 2) if b["den"] else None

    return jsonify({
        "model": model, "run_id": run_id,
        "stat": stat, "stat_label": sd["label"],
        "variant": variant,
        "n_bins": n_bins, "n_total": len(rows),
        "char_budget_min": cb_min, "char_budget_max": cb_max,
        "bins": bins,
        "baseline_no_distractor": _no_dist_baseline(model, stat, variant),
    })


# ───── chart 3: per-variant length curves ─────────────────────────────
@app.route("/api/results/variant_length_curves")
def api_results_variant_length_curves():
    """Multi-line: one length curve per evidence_variant. Restricted to
    the variants where the STAT is meaningful — defaults to {C, A+C, B+C}.
    Returns one series per variant, all binned on the same length axis
    so the lines are visually comparable.
    """
    model = request.args.get("model", "")
    preset = _req_preset()
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    stat = request.args.get("stat", "SR").upper()
    try:
        n_bins = int(request.args.get("n_bins", "10"))
    except ValueError:
        n_bins = 10
    n_bins = max(2, min(50, n_bins))

    if stat not in STAT_DEFS:
        abort(400, description=f"unknown stat {stat!r}")
    if not (model and run_id):
        abort(400, description="missing model/run_id")

    data = _load_run(model, run_id, preset)
    if not data["rows"]:
        abort(404, description=f"no rows for {model}/{run_id}")

    sd = STAT_DEFS[stat]
    valid_variants = sd["valid_variants"]
    # canonical display order
    ordered = [v for v in ("C", "A+C", "B+C", "A", "B") if v in valid_variants]

    base = [r for r in data["rows"]
            if not r["_is_error"]
            and r.get("evidence_variant") in valid_variants
            and r.get("char_budget")]
    if not base:
        return jsonify({
            "stat": stat, "n_bins": n_bins,
            "series": [],
            "warn": "no rows for this STAT",
        })

    cb_min = max(1, min(r["char_budget"] for r in base))
    cb_max = max(r["char_budget"] for r in base)
    log_min, log_max = math.log10(cb_min), math.log10(cb_max)
    span = log_max - log_min if log_max > log_min else 1.0

    # Shared axis edges
    edges = [log_min + span * i / n_bins for i in range(n_bins + 1)]
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

    series = []
    for v in ordered:
        sub = [r for r in base if r["evidence_variant"] == v]
        bin_counts = [{
            "log10_lo": edges[i], "log10_hi": edges[i + 1],
            "log10_mid": midpoints[i],
            "chars_mid": 10 ** midpoints[i],
            "n": 0, "num": 0, "den": 0, "value_pct": None,
        } for i in range(n_bins)]
        for r in sub:
            log_cb = math.log10(r["char_budget"])
            idx = int((log_cb - log_min) / span * n_bins)
            if idx >= n_bins: idx = n_bins - 1
            if idx < 0: idx = 0
            b = bin_counts[idx]
            b["n"] += 1
            if r[sd["denom"]]:
                b["den"] += 1
                if r[sd["num"]]: b["num"] += 1
        for b in bin_counts:
            b["value_pct"] = round(100 * b["num"] / b["den"], 2) if b["den"] else None
        series.append({
            "variant": v,
            "n_total": len(sub),
            "bins": bin_counts,
        })

    # Per-variant baselines from canon_no_distractor archive — let the
    # frontend draw a dashed reference line per variant.
    baselines = {v: _no_dist_baseline(model, stat, v) for v in ordered}
    return jsonify({
        "model": model, "run_id": run_id,
        "stat": stat, "stat_label": sd["label"],
        "n_bins": n_bins,
        "char_budget_min": cb_min, "char_budget_max": cb_max,
        "log10_edges": edges,
        "log10_midpoints": midpoints,
        "series": series,
        "baselines_no_distractor": baselines,
    })


# ───── 2D surface (kept for export / tab use) ─────────────────────────
@app.route("/api/results/length_depth_surface")
def api_results_length_depth_surface():
    """Optional 2D surface — kept for parity with the figure-export
    pipeline. Same schema as before."""
    model = request.args.get("model", "")
    preset = _req_preset()
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    stat = request.args.get("stat", request.args.get("metric", "SR")).upper()
    variant = request.args.get("variant") or None
    try:
        depth_bins = int(request.args.get("depth_bins", "8"))
        length_bins = int(request.args.get("length_bins", "8"))
    except ValueError:
        depth_bins = length_bins = 8
    depth_bins = max(2, min(50, depth_bins))
    length_bins = max(2, min(50, length_bins))

    if stat not in STAT_DEFS:
        abort(400, description=f"unknown stat {stat!r}")
    if not (model and run_id):
        abort(400, description="missing model/run_id")

    data = _load_run(model, run_id, preset)
    rows = _filter_rows_for_stat(data["rows"], stat, variant=variant)
    rows = [r for r in rows
            if r.get("placement_frac") is not None and r.get("char_budget")]
    if not rows:
        abort(404, description="no rows have placement_frac + char_budget for this STAT")
    n_with_meta = len(rows)

    cb_min = max(1, min(r["char_budget"] for r in rows))
    cb_max = max(r["char_budget"] for r in rows)
    log_min, log_max = math.log10(cb_min), math.log10(cb_max)
    span = log_max - log_min if log_max > log_min else 1.0

    depth_edges = [i / depth_bins for i in range(depth_bins + 1)]
    length_edges_log10 = [log_min + span * i / length_bins for i in range(length_bins + 1)]

    cells = [[{
        "depth_lo": depth_edges[di], "depth_hi": depth_edges[di + 1],
        "length_lo_log10": length_edges_log10[li],
        "length_hi_log10": length_edges_log10[li + 1],
        "length_lo_chars": 10 ** length_edges_log10[li],
        "length_hi_chars": 10 ** length_edges_log10[li + 1],
        "n": 0, "num": 0, "den": 0,
    } for li in range(length_bins)] for di in range(depth_bins)]

    sd = STAT_DEFS[stat]
    for r in rows:
        di = int(r["placement_frac"] * depth_bins)
        if di >= depth_bins: di = depth_bins - 1
        log_cb = math.log10(r["char_budget"])
        li = int((log_cb - log_min) / span * length_bins)
        if li >= length_bins: li = length_bins - 1
        if li < 0: li = 0
        c = cells[di][li]
        c["n"] += 1
        if r[sd["denom"]]:
            c["den"] += 1
            if r[sd["num"]]: c["num"] += 1
    for row in cells:
        for c in row:
            c["value_pct"] = round(100 * c["num"] / c["den"], 2) if c["den"] else None

    return jsonify({
        "model": model, "run_id": run_id,
        "stat": stat, "stat_label": sd["label"],
        "variant": variant,
        "depth_bins": depth_bins, "length_bins": length_bins,
        "depth_edges": depth_edges,
        "length_edges_log10": length_edges_log10,
        "length_edges_chars": [10 ** e for e in length_edges_log10],
        "n_with_meta": n_with_meta,
        "cells": cells,
    })


# ───── prompts: list + single ─────────────────────────────────────────
_PROMPT_NAME_RE = re.compile(r"^(?P<sid>[A-Z]+-\d+)_(?P<ev>[^_]+)_(?P<perm>.+?)\.json$")


@app.route("/api/prompts")
def api_prompts():
    """Browse prompt files for a preset (default canon_unified). Returns
    a paginated list with minimal metadata for a left-rail picker."""
    cond_dir = GENERATED_DIR / _req_preset()
    if not cond_dir.exists():
        return jsonify({"prompts": [], "total": 0})

    sid_filter = request.args.get("scenario_id", "") or ""
    ev_filter = request.args.get("evidence_variant", "") or ""
    try:
        offset = int(request.args.get("offset", "0"))
        limit  = int(request.args.get("limit",  "200"))
    except ValueError:
        offset, limit = 0, 200
    limit = max(1, min(2000, limit))

    files = []
    for jf in sorted(cond_dir.glob("*.json")):
        if jf.name == "manifest.json":
            continue
        # Light filtering on the filename alone (don't load JSON for
        # every file — there are 6,366 of them).
        m = _PROMPT_NAME_RE.match(jf.name)
        sid = m.group("sid") if m else ""
        ev = m.group("ev") if m else ""
        if sid_filter and sid != sid_filter: continue
        if ev_filter and ev != ev_filter:    continue
        files.append({"name": jf.name, "sid": sid, "ev": ev})
    total = len(files)
    page = files[offset:offset + limit]
    return jsonify({"prompts": page, "total": total,
                    "offset": offset, "limit": limit})


@app.route("/api/prompt")
def api_prompt():
    """Single prompt (preset-aware): system prompt, user message, full metadata."""
    name = request.args.get("name", "")
    if not name or "/" in name or ".." in name or not name.endswith(".json"):
        abort(400, description="invalid prompt name")
    p = GENERATED_DIR / _req_preset() / name
    if not p.exists():
        abort(404, description=f"prompt {name} not found")
    with open(p) as f:
        d = json.load(f)
    return jsonify({
        "name": name,
        "system_prompt": d.get("system_prompt", ""),
        "user_message": d.get("user_message", ""),
        "metadata": d.get("metadata", {}),
    })


# Filename → tsv permutation key. Filenames look like
# `AG-01_C_c0_d1_L4.json`; the matching TSV permutation column for
# canon_unified is `c0-d1-l4`, and for canon_no_distractor it is `c0-d0`
# (no length suffix). Returns (sid, ev, base_perm, draw, length_or_none)
# or None on parse failure.
_PROMPT_FNAME_RE = re.compile(
    r"^(?P<sid>[A-Z]+-\d+)_(?P<ev>A\+C|B\+C|C|A|B)"
    r"_(?P<base>[^_]+(?:_[abc]\d+)*)_d(?P<draw>\d+)"
    r"(?:_L(?P<length>\d+))?\.json$"
)


def _parse_prompt_filename(name: str) -> tuple[str, str, str, int, int | None] | None:
    m = _PROMPT_FNAME_RE.match(name)
    if not m:
        return None
    sid = m.group("sid")
    ev = m.group("ev")
    base = m.group("base")
    draw = int(m.group("draw"))
    length = int(m.group("length")) if m.group("length") else None
    return (sid, ev, base, draw, length)


@app.route("/api/prompt/generations")
def api_prompt_generations():
    """All model generations for a single canon_unified prompt.

    Iterates every model under data/runs/canon_unified/ that has a run
    matching (sid, evidence_variant, permutation) — where the
    permutation is reconstructed from the prompt filename as
    `<base>-d<draw>[-l<length>]`. Returns the per-model judge result so
    the Prompts tab can show a glance of every model's response and
    judgment for the selected prompt.
    """
    name = request.args.get("name", "")
    if not name or "/" in name or ".." in name or not name.endswith(".json"):
        abort(400, description="invalid prompt name")
    parsed = _parse_prompt_filename(name)
    if not parsed:
        abort(400, description=f"can't parse {name}")
    sid, ev, base_perm, draw, length = parsed
    if length is not None:
        target_perm = f"{base_perm}-d{draw}-l{length}"
    else:
        target_perm = f"{base_perm}-d{draw}"

    preset = _req_preset()
    out: list[dict] = []
    for entry in _list_models():
        model = entry["model"]
        # For non-headline presets, resolve that preset's latest run for
        # this model; skip models that never ran the preset.
        if preset == CONDITION:
            run_id = entry["latest_run_id"]
        else:
            run_id = _resolve_run_id(model, None, preset)
            if not run_id:
                continue
        loaded = _load_run(model, run_id, preset)
        for r in loaded.get("rows", []):
            if (r.get("scenario_id") == sid
                    and r.get("evidence_variant") == ev
                    and r.get("permutation") == target_perm):
                is_err = (r.get("raw_response") or "").startswith(("ERROR", '"ERROR'))
                excerpt = (r.get("raw_response") or "").replace("\\n", " ")[:240]
                out.append({
                    "model": model,
                    "model_display": _pretty_model(model)
                        if _pretty_model(model) != model else model.replace("_", "/"),
                    "run_id": run_id,
                    "recommendation": r.get("recommendation") or "",
                    "constraint_mentioned": r.get("constraint_mentioned") or "",
                    "sufficiently_modified": r.get("sufficiently_modified") or "",
                    "vigilance_refuse_only":
                        r.get("vigilance_refuse_only") or r.get("vigilance") or "",
                    "abstained": r.get("abstained") or "",
                    "abstain_type": r.get("abstain_type") or "",
                    "parse_error": r.get("parse_error") or "",
                    "is_error": is_err,
                    # `reasoning` holds the v2 judge's REASONING preamble; some
                    # legacy rows put the same text in `explanation` — fall
                    # back so the frontend never silently shows blank.
                    "reasoning":
                        r.get("reasoning") or r.get("explanation") or "",
                    # Keep `explanation` for back-compat with older callers.
                    "explanation": r.get("explanation") or "",
                    "raw_response_preview": excerpt,
                    "raw_response": r.get("raw_response") or "",
                })
                break  # one row per (sid, ev, perm) per model

    # Sort by SR pass first, then model name — successful runs at the top
    # so the user can compare success vs failure side-by-side.
    def _sort_key(g):
        sr = g.get("vigilance_refuse_only")
        sr_rank = 0 if sr == "1" else (1 if sr == "0" else 2)
        return (sr_rank, g["model"])
    out.sort(key=_sort_key)
    return jsonify({
        "prompt_name": name,
        "scenario_id": sid, "evidence_variant": ev, "permutation": target_perm,
        "n_models": len(out),
        "generations": out,
    })


# ───── results: paginated rows + single ───────────────────────────────
@app.route("/api/results/rows")
def api_results_rows():
    """Paginated row inspector. Defaults to canon_unified; the Scenario
    tab can override via `preset=canon_no_distractor`.
    Filterable by scenario_id, evidence_variant, parse_error, errored."""
    model = request.args.get("model", "")
    preset = request.args.get("preset", CONDITION)
    if preset not in _VALID_PRESETS:
        preset = CONDITION
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    if not (model and run_id):
        abort(400, description="missing model/run_id")

    sid = request.args.get("scenario_id", "") or ""
    ev = request.args.get("evidence_variant", "") or ""
    rec = request.args.get("recommendation", "") or ""
    show_errors = request.args.get("show_errors", "1") == "1"
    show_parse_err = request.args.get("show_parse_err", "1") == "1"
    try:
        offset = int(request.args.get("offset", "0"))
        limit  = int(request.args.get("limit",  "100"))
    except ValueError:
        offset, limit = 0, 100
    limit = max(1, min(1000, limit))

    data = _load_run(model, run_id, preset)
    rows = data["rows"]
    if sid: rows = [r for r in rows if r.get("scenario_id") == sid]
    if ev:  rows = [r for r in rows if r.get("evidence_variant") == ev]
    if ev == "safe":  # convenience alias: A and B (no-constraint) rows
        rows = [r for r in data["rows"] if r.get("evidence_variant") in ("A", "B")]
        if sid: rows = [r for r in rows if r.get("scenario_id") == sid]
    if rec: rows = [r for r in rows if (r.get("recommendation") or "") == rec]
    if not show_errors:    rows = [r for r in rows if not r["_is_error"]]
    if not show_parse_err: rows = [r for r in rows if r.get("parse_error") != "1"]

    total = len(rows)
    page = rows[offset:offset + limit]

    def trim_row(r):
        return {
            "scenario_id": r.get("scenario_id"),
            "evidence_variant": r.get("evidence_variant"),
            "permutation": r.get("permutation"),
            "expected_answer": r.get("expected_answer"),
            "raw_response_preview": (r.get("raw_response") or "")[:240],
            "recommendation": r.get("recommendation"),
            "constraint_mentioned": r.get("constraint_mentioned"),
            "sufficiently_modified": r.get("sufficiently_modified"),
            "vigilance_refuse_only": r.get("vigilance_refuse_only") or r.get("vigilance"),
            "abstained": r.get("abstained"),
            "abstain_type": r.get("abstain_type"),
            "parse_error": r.get("parse_error"),
            "is_error": r["_is_error"],
            "input_tokens": r.get("input_tokens"),
            "output_tokens": r.get("output_tokens"),
            "char_budget": r.get("char_budget"),
            "placement_frac": r.get("placement_frac"),
        }
    return jsonify({
        "model": model, "run_id": run_id, "preset": preset,
        "total": total, "offset": offset, "limit": limit,
        "rows": [trim_row(r) for r in page],
    })


_PERM_DRAW_LEN_RE = re.compile(r"^(.*?)-d(\d+)(?:-l(\d+))?$")


def _prompt_file_for_row(sid: str, ev: str, perm_full: str,
                         preset: str = CONDITION) -> Path | None:
    """Recover the generated/<preset>/<filename>.json that was used
    to produce a results-row. The row's `permutation` column is the
    runner's ``"<base_perm>-d<draw>[-l<length>]"`` string; the filename
    follows ``"<sid>_<ev>_<base_perm>_d<draw>[_L<length>].json"``.

    Different presets store prompts in different subdirectories of
    ``generated/`` — canon_no_distractor has ``_d<draw>.json`` (no
    ``_L`` suffix because there is no length variation), and
    canon_unified has both ``_d<draw>`` and ``_L<L>``.
    """
    m = _PERM_DRAW_LEN_RE.match(perm_full)
    if not m:
        # Bare perm (no draw/length suffix) — fall back to just _d0
        base = perm_full
        draw = "0"
        length = None
    else:
        base, draw, length = m.group(1), m.group(2), m.group(3)
    cand = f"{sid}_{ev}_{base}_d{draw}"
    if length is not None:
        cand += f"_L{length}"
    cand += ".json"
    p = GENERATED_DIR / preset / cand
    if p.exists():
        return p
    # Fallback: glob, since some renderer paths add a `_P` index too.
    matches = list((GENERATED_DIR / preset).glob(f"{sid}_{ev}_{base}_d{draw}*.json"))
    return matches[0] if matches else None


@app.route("/api/results/row")
def api_results_row():
    """Single row with full raw_response + judge explanation + the
    underlying prompt (system_prompt, user_message, metadata) so a
    reviewer doesn't have to context-switch to the Prompts tab.

    For TSVs where the (sid, ev, perm) triple isn't unique (e.g. the
    qwen3.5-9b canon_unified resamples), an optional `raw_prefix` arg
    disambiguates by matching the start of the row's raw_response.
    """
    model = request.args.get("model", "")
    preset = request.args.get("preset", CONDITION)
    if preset not in _VALID_PRESETS:
        preset = CONDITION
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    sid = request.args.get("scenario_id", "")
    ev = request.args.get("evidence_variant", "")
    perm = request.args.get("permutation", "")
    raw_prefix = request.args.get("raw_prefix", "")
    if not (model and run_id and sid and ev and perm):
        abort(400, description="missing one of model/run_id/scenario_id/evidence_variant/permutation")
    data = _load_run(model, run_id, preset)
    for r in data["rows"]:
        if (r.get("scenario_id") == sid
                and r.get("evidence_variant") == ev
                and r.get("permutation") == perm
                and (not raw_prefix
                     or (r.get("raw_response") or "").startswith(raw_prefix))):
            payload = {
                "scenario_id": sid, "evidence_variant": ev, "permutation": perm,
                "expected_answer": r.get("expected_answer"),
                "raw_response": r.get("raw_response"),
                "recommendation": r.get("recommendation"),
                "constraint_mentioned": r.get("constraint_mentioned"),
                "sufficiently_modified": r.get("sufficiently_modified"),
                # v2 stores the judge's reasoning preamble in `reasoning`
                # (separate column from the legacy `explanation`).
                "reasoning": r.get("reasoning") or r.get("explanation"),
                "vigilance_refuse_only": r.get("vigilance_refuse_only") or r.get("vigilance"),
                "abstained": r.get("abstained"),
                "abstain_type": r.get("abstain_type"),
                "parse_error": r.get("parse_error"),
                "input_tokens": r.get("input_tokens"),
                "output_tokens": r.get("output_tokens"),
                "judge_input_tokens": r.get("judge_input_tokens"),
                "judge_output_tokens": r.get("judge_output_tokens"),
                "char_budget": r.get("char_budget"),
                "placement_frac": r.get("placement_frac"),
            }
            pf = _prompt_file_for_row(sid, ev, perm, preset=preset)
            if pf is not None:
                try:
                    with open(pf) as f:
                        pd = json.load(f)
                    payload["prompt"] = {
                        "filename": pf.name,
                        "system_prompt": pd.get("system_prompt", ""),
                        "user_message": pd.get("user_message", ""),
                        "metadata": pd.get("metadata", {}),
                    }
                except Exception as e:
                    payload["prompt"] = {"error": f"{type(e).__name__}: {e}"}
            else:
                payload["prompt"] = {"error": "prompt file not found"}
            return jsonify(payload)
    abort(404, description=f"row not found: {sid}/{ev}/{perm}")


# ───── stat metadata ──────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    return jsonify({
        "stats": [{
            "code": k, "label": v["label"],
            "valid_variants": sorted(v["valid_variants"]),
        } for k, v in STAT_DEFS.items()],
    })


# ───── scenarios: list + detail ───────────────────────────────────────
def _split_seeds(blob: str) -> list[str]:
    if not blob:
        return []
    parts = blob.split(" || ") if " || " in blob else blob.split("||")
    return [p.strip() for p in parts if p.strip()]


@app.route("/api/scenarios")
def api_scenarios():
    """List of all 85 scenarios with per-scenario canon_unified result
    aggregates for the current (model, run). Used by the Scenarios tab.

    Sortable by any metric: scenario_id, n, SR, CM, abstain.
    `direction` ∈ {"asc","desc"}. Defaults are "worst at top" for
    metrics (asc on SR/CM/abstain) and "biggest first" for n. None
    values always sort last.
    """
    model = request.args.get("model", "")
    preset = request.args.get("preset", CONDITION)
    if preset not in _VALID_PRESETS:
        preset = CONDITION
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    sort = request.args.get("sort", "scenario_id")
    direction = request.args.get("direction", "")
    scenarios = _load_scenarios()

    out_rows = []
    if model and run_id:
        data = _load_run(model, run_id, preset)
        per_scen: dict[str, dict] = {}
        for r in data["rows"]:
            if r["_is_error"]:
                continue
            sid = r.get("scenario_id", "")
            e = per_scen.setdefault(sid, {"n": 0,
                                          "vig": 0, "vig_set": 0,
                                          "cm": 0, "cm_set": 0,
                                          "abstain": 0})
            e["n"] += 1
            if r["_vig_set"]:
                e["vig_set"] += 1
                if r["_vigilance"]: e["vig"] += 1
                if r["_abstained"]: e["abstain"] += 1
            if r["_cm_set"]:
                e["cm_set"] += 1
                if r["_cm"]: e["cm"] += 1
        for sid, sc in scenarios.items():
            agg = per_scen.get(sid, {})
            domain = sc.get("domain", "")
            domain_pre = domain.split("—", 1)[0].strip() if "—" in domain else domain
            def pct(n, d): return round(100 * n / d, 2) if d else None
            out_rows.append({
                "scenario_id": sid,
                "domain": domain,
                "domain_prefix": domain_pre,
                "risk_level": sc.get("risk_level", ""),
                "n": agg.get("n", 0),
                "SR":      pct(agg.get("vig", 0),    agg.get("vig_set", 0)),
                "CM":      pct(agg.get("cm", 0),     agg.get("cm_set", 0)),
                "abstain": pct(agg.get("abstain", 0), agg.get("vig_set", 0)),
            })
    else:
        for sid, sc in scenarios.items():
            out_rows.append({
                "scenario_id": sid,
                "domain": sc.get("domain", ""),
                "domain_prefix": (sc.get("domain", "").split("—", 1)[0].strip()
                                  if "—" in (sc.get("domain", "") or "")
                                  else sc.get("domain", "")),
                "risk_level": sc.get("risk_level", ""),
                "n": 0,
                "SR": None, "CM": None, "abstain": None,
            })

    # Generic sort. Metric defaults are ascending so worst floats to the
    # top; n defaults to descending; explicit `direction` overrides.
    metric_keys = {"SR", "CM", "abstain"}
    valid_sort = {"scenario_id", "n"} | metric_keys
    if sort not in valid_sort:
        sort = "scenario_id"
    if direction == "desc":
        reverse = True
    elif direction == "asc":
        reverse = False
    else:
        reverse = (sort == "n")  # default: n=large-first, others ascending

    if sort == "scenario_id":
        out_rows.sort(key=lambda r: r["scenario_id"], reverse=reverse)
    elif sort == "n":
        out_rows.sort(key=lambda r: (r.get("n") or 0, r["scenario_id"]),
                      reverse=reverse)
    else:
        # Two-pass sort so None always lands at the bottom regardless
        # of direction: first sort the rows-with-data, then append the
        # nulls (themselves stably ordered by scenario_id).
        with_val = [r for r in out_rows if r[sort] is not None]
        without  = [r for r in out_rows if r[sort] is None]
        with_val.sort(key=lambda r: (r[sort], r["scenario_id"]),
                      reverse=reverse)
        without.sort(key=lambda r: r["scenario_id"])
        out_rows = with_val + without

    return jsonify({
        "model": model, "run_id": run_id,
        "n_scenarios": len(out_rows),
        "sort": sort, "direction": "desc" if reverse else "asc",
        "scenarios": out_rows,
    })


@app.route("/api/scenario")
def api_scenario():
    """Detail panel for a single scenario: full constraint description,
    seeds, query, choices, plus the per-variant canon_unified result
    breakdown for the current (model, run)."""
    sid = request.args.get("id", "")
    model = request.args.get("model", "")
    preset = request.args.get("preset", CONDITION)
    if preset not in _VALID_PRESETS:
        preset = CONDITION
    run_id = _resolve_run_id(model, request.args.get("run_id"), preset)
    if not sid:
        abort(400, description="missing id")
    scenarios = _load_scenarios()
    sc = scenarios.get(sid)
    if not sc:
        abort(404, description=f"scenario {sid!r} not found")

    detail: dict[str, Any] = {
        "scenario_id": sid,
        "domain": sc.get("domain", ""),
        "constraint_description": sc.get("constraint_description", ""),
        "grounding_sources": sc.get("grounding_sources", ""),
        "personalization_difference": sc.get("personalization_difference", ""),
        "benign_triggering_query": sc.get("benign_triggering_query", ""),
        "choice_a_label": sc.get("choice_a_label", ""),
        "choice_b_label": sc.get("choice_b_label", ""),
        "choice_a_rationale_admissible":   sc.get("choice_a_rationale_admissible", ""),
        "choice_a_rationale_inadmissible": sc.get("choice_a_rationale_inadmissible", ""),
        "choice_b_rationale_admissible":   sc.get("choice_b_rationale_admissible", ""),
        "choice_b_rationale_inadmissible": sc.get("choice_b_rationale_inadmissible", ""),
        "evidence_set_c_seeds": _split_seeds(sc.get("evidence_set_c_seeds", "")),
        "evidence_set_a_seeds": _split_seeds(sc.get("evidence_set_a_seeds", "")),
        "evidence_set_b_seeds": _split_seeds(sc.get("evidence_set_b_seeds", "")),
        "risk_level": sc.get("risk_level", ""),
    }

    # Per-variant result breakdown for this scenario in the current run
    if model and run_id:
        data = _load_run(model, run_id, preset)
        rows = [r for r in data["rows"]
                if r.get("scenario_id") == sid and not r["_is_error"]]
        by_v: dict[str, dict] = {}
        for r in rows:
            v = r.get("evidence_variant", "?")
            e = by_v.setdefault(v, {"n": 0,
                                    "vig": 0, "vig_set": 0,
                                    "cm": 0, "cm_set": 0,
                                    "abstain": 0})
            e["n"] += 1
            if r["_vig_set"]:
                e["vig_set"] += 1
                if r["_vigilance"]: e["vig"] += 1
                if r["_abstained"]: e["abstain"] += 1
            if r["_cm_set"]:
                e["cm_set"] += 1
                if r["_cm"]: e["cm"] += 1
        def pct(n, d): return round(100 * n / d, 2) if d else None
        per_variant = []
        for v in ("C", "A+C", "B+C", "A", "B"):
            if v not in by_v: continue
            e = by_v[v]
            per_variant.append({
                "variant": v, "n": e["n"],
                "SR": pct(e["vig"], e["vig_set"]),
                "CM": pct(e["cm"], e["cm_set"]),
                "abstain": pct(e["abstain"], e["vig_set"]),
            })
        detail["per_variant"] = per_variant
        detail["n_total"] = sum(e["n"] for e in by_v.values())
        detail["model"] = model
        detail["run_id"] = run_id
    else:
        detail["per_variant"] = []
        detail["n_total"] = 0

    return jsonify(detail)


# ──────────────────────────────────────────────────────────────────────
# Surface (canon_unified SR(length, depth) heatmap per model)
# ──────────────────────────────────────────────────────────────────────
@app.route("/api/surface/data")
def surface_data():
    """Bin canon_unified rows into a (length, depth) grid for one model.
    Returns a 2D matrix of SR values + cell counts so the frontend can
    paint a heatmap."""
    model = request.args.get("model")
    run_id = request.args.get("run_id")
    n_len = max(2, int(request.args.get("n_len", "8")))
    n_dep = max(2, int(request.args.get("n_dep", "5")))
    if not model:
        return jsonify({"error": "model required"}), 400
    run_id = _resolve_run_id(model, run_id)
    if not run_id:
        return jsonify({"error": "no run for that model"}), 404
    rows = _load_run(model, run_id).get("rows", [])
    rows = [r for r in rows
            if "C" in (r.get("evidence_variant") or "")
            and r.get("char_budget") and r.get("placement_frac") is not None
            and r["_vig_set"]]
    if not rows:
        return jsonify({"error": "no valid rows"}), 404
    import math
    cb_vals = [r["char_budget"] for r in rows]
    cb_min, cb_max = min(cb_vals), max(cb_vals)
    log_min, log_max = math.log10(max(1, cb_min)), math.log10(max(2, cb_max))
    dep_edges = [i / n_dep for i in range(n_dep + 1)]
    len_edges = [10 ** (log_min + i * (log_max - log_min) / n_len) for i in range(n_len + 1)]
    grid = [[{"n": 0, "vig": 0} for _ in range(n_len)] for _ in range(n_dep)]
    for r in rows:
        cb = r["char_budget"]; pf = r["placement_frac"]
        # find length bin
        li = min(n_len - 1, max(0, int((math.log10(cb) - log_min) / max(1e-9, log_max - log_min) * n_len)))
        di = min(n_dep - 1, max(0, int(pf * n_dep)))
        cell = grid[di][li]
        cell["n"] += 1
        if r["_vigilance"]: cell["vig"] += 1
    out_grid = [[
        {"n": c["n"], "sr": (c["vig"] / c["n"] if c["n"] else None)}
        for c in row] for row in grid]
    return jsonify({"model": model, "run_id": run_id,
                    "n_total": len(rows),
                    "len_edges": len_edges, "dep_edges": dep_edges,
                    "grid": out_grid})


# ──────────────────────────────────────────────────────────────────────
# Frontier (overlay of per-model SR-threshold contours)
# ──────────────────────────────────────────────────────────────────────
def _pretty_model(m: str) -> str:
    """Compact display label for a model directory name."""
    ml = m.lower()
    if "haiku" in ml:  return "Haiku 4.5"
    if "sonnet" in ml: return "Sonnet 4.6"
    if "opus" in ml:   return "Opus 4.7"
    return m


def _stage_for_model(m: str) -> tuple[int, str]:
    """Bucket a model dir name into (sort_order, group_label) for grouping
    in cross-model views. Order follows the canonical roster's listing
    so the Frontier chart's per-vendor groups read in a consistent
    direction across reloads. Function name is kept for API stability."""
    ml = m.lower()
    if "haiku" in ml:                       return (1, "Anthropic Haiku")
    if "sonnet" in ml or "opus" in ml:      return (2, "Anthropic frontier")
    if "gpt-5" in ml or "openai" in ml:     return (3, "OpenAI")
    if "gemini" in ml or "google" in ml:    return (4, "Google Gemini")
    if "qwen" in ml:                        return (6, "Open-source — Qwen")
    return (9, "Other")


_ALL_VARIANTS = ("C", "A+C", "B+C", "A", "B")
_C_BEARING = ("C", "A+C", "B+C")


def _parse_variants_arg(arg_str: str | None,
                        default: tuple[str, ...] = _C_BEARING) -> set[str]:
    """Parse a `?variants=C,A+C,B+C` query param into a validated set.
    Falls back to `default` if the arg is empty or contains no valid
    variants. Filters out anything not in {C, A+C, B+C, A, B}."""
    if not arg_str:
        return set(default)
    parts = {p.strip() for p in arg_str.split(",") if p.strip()}
    valid = parts & set(_ALL_VARIANTS)
    return valid or set(default)


@app.route("/api/frontier/baseline_vs_unified")
def frontier_baseline_vs_unified():
    """Per-model SR/CM on canon_unified (bars) and canon_no_distractor
    (markers/stars), grouped by vendor / model family, for the Frontier
    tab's baseline-vs-vigilance grouped-bar chart.

    Excludes ERROR rows and parse_error rows. Computes a clean SR%, CM%
    per (model, preset) using the same conventions as the rest of
    the viewer.

    The optional `?variants=C,A+C,B+C` query param restricts the rows
    used in the metric calculations to the given subset. Default is
    C-bearing only (C / A+C / B+C), matching the paper's headline
    convention. Note: SR is formally null for no-constraint variants
    A and B (no constraint to flag), so a selection that includes A
    or B will yield smaller denominators for SR than for CM (which is
    defined on every row).
    """
    variants = _parse_variants_arg(request.args.get("variants"))

    out: list[dict] = []
    for entry in _list_models():
        m = entry["model"]; run_id = entry["latest_run_id"]
        rows = _load_run(m, run_id).get("rows", [])
        rows = [r for r in rows if not r["_is_error"]]
        if not rows: continue

        # canon_unified (bars) — restrict to selected variants, then
        # macro-average within each metric. Each metric's _set field
        # ("_vig_set", "_cm_set") already encodes whether the metric is
        # defined for that row (e.g. _vig_set is False for A and B
        # variants), so the filter just narrows the row population
        # further.
        v_rows = [r for r in rows if r.get("evidence_variant") in variants]
        sr_u = _macro_avg_pct(v_rows, "_vigilance", "_vig_set")
        cm_u = _macro_avg_pct(v_rows, "_cm",        "_cm_set")

        # canon_no_distractor (markers) — same treatment, but rows from
        # the no-distractor preset need their boolean fields hydrated
        # the way _load_run does.
        nd_rows_raw = _load_no_dist_run(m) or []
        nd_rows_raw = [r for r in nd_rows_raw
                       if not (r.get("raw_response","") or "").startswith(("ERROR", '"ERROR'))]
        if nd_rows_raw:
            for r in nd_rows_raw:
                r.setdefault("_is_error",
                    (r.get("raw_response","") or "").startswith(("ERROR", '"ERROR')))
                r.setdefault("_vigilance",
                    _coerce_bool(r.get("vigilance_refuse_only") or r.get("vigilance")))
                r.setdefault("_vig_set",
                    (r.get("vigilance_refuse_only") not in ("", None)
                     or r.get("vigilance") not in ("", None))
                    and not r["_is_error"])
                cm = (r.get("constraint_mentioned") or "").strip().upper()
                r.setdefault("_cm_set", cm in ("YES", "NO"))
                r.setdefault("_cm", cm == "YES")
            v_nd_rows = [r for r in nd_rows_raw if r.get("evidence_variant") in variants]
            sr_nd = _macro_avg_pct(v_nd_rows, "_vigilance", "_vig_set")
            cm_nd = _macro_avg_pct(v_nd_rows, "_cm",        "_cm_set")
            n_nd = len(v_nd_rows)
        else:
            sr_nd = cm_nd = None
            n_nd = 0
        n_u = len(v_rows)

        stage_order, stage_label = _stage_for_model(m)
        out.append({
            "model": m,
            "model_display": _pretty_model(m) if _pretty_model(m) != m else m,
            "stage_order": stage_order, "stage_label": stage_label,
            "n_unified": n_u, "n_no_dist": n_nd,
            "unified":  {"SR": sr_u,  "CM": cm_u},
            "no_dist":  {"SR": sr_nd, "CM": cm_nd},
        })
    # Round to 2 decimals for display (None preserved)
    for entry_out in out:
        for d in (entry_out["unified"], entry_out["no_dist"]):
            for k, v in list(d.items()):
                if v is not None:
                    d[k] = round(v, 2)
    # Sort: stage order, then unified SR descending (None values last)
    out.sort(key=lambda r: (r["stage_order"], -(r["unified"]["SR"] or -1)))
    return jsonify({
        "models": out,
        "variants": sorted(variants, key=lambda v: _ALL_VARIANTS.index(v)),
    })


@app.route("/api/frontier/data")
def frontier_data():
    """For each model with canon_unified data, compute the binned
    SR(length) curve (averaged over depth) and SR(depth) curve
    (averaged over length). The frontend overlays these against an
    SR threshold to surface where each model crosses below it.

    Accepts the same `?variants=` filter as `/baseline_vs_unified`.
    Default is C-bearing only (C / A+C / B+C); selecting A or B will
    typically produce empty curves because SR is null on no-constraint
    variants (`_vig_set=False`)."""
    threshold = float(request.args.get("threshold", "80"))
    n_len = max(2, int(request.args.get("n_len", "10")))
    n_dep = max(2, int(request.args.get("n_dep", "10")))
    variants = _parse_variants_arg(request.args.get("variants"))
    import math
    out = {
        "threshold": threshold,
        "variants": sorted(variants, key=lambda v: _ALL_VARIANTS.index(v)),
        "models": [],
    }
    for entry in _list_models():
        m = entry["model"]; run_id = entry["latest_run_id"]
        rows = _load_run(m, run_id).get("rows", [])
        rows = [r for r in rows
                if r.get("evidence_variant") in variants
                and r.get("char_budget") and r.get("placement_frac") is not None
                and r["_vig_set"]]
        if not rows:
            continue
        cb_vals = [r["char_budget"] for r in rows]
        cb_min, cb_max = min(cb_vals), max(cb_vals)
        log_min = math.log10(max(1, cb_min)); log_max = math.log10(max(2, cb_max))
        # Length curve (binned on log axis, averaged over all depths)
        len_buckets = [{"n": 0, "vig": 0} for _ in range(n_len)]
        len_centers = [10 ** (log_min + (i + 0.5) * (log_max - log_min) / n_len)
                       for i in range(n_len)]
        for r in rows:
            li = min(n_len - 1, max(0, int(
                (math.log10(r["char_budget"]) - log_min) /
                max(1e-9, log_max - log_min) * n_len)))
            len_buckets[li]["n"] += 1
            if r["_vigilance"]: len_buckets[li]["vig"] += 1
        len_curve = [
            {"x": len_centers[i],
             "sr": (b["vig"] / b["n"] * 100 if b["n"] else None),
             "n": b["n"]}
            for i, b in enumerate(len_buckets)]
        # Depth curve (averaged over all lengths)
        dep_buckets = [{"n": 0, "vig": 0} for _ in range(n_dep)]
        for r in rows:
            di = min(n_dep - 1, max(0, int(r["placement_frac"] * n_dep)))
            dep_buckets[di]["n"] += 1
            if r["_vigilance"]: dep_buckets[di]["vig"] += 1
        dep_curve = [
            {"x": (i + 0.5) / n_dep,
             "sr": (b["vig"] / b["n"] * 100 if b["n"] else None),
             "n": b["n"]}
            for i, b in enumerate(dep_buckets)]
        # Overall SR
        overall = sum(1 for r in rows if r["_vigilance"]) / len(rows) * 100
        out["models"].append({
            "model": m, "run_id": run_id, "n": len(rows),
            "overall_sr": round(overall, 2),
            "length_curve": len_curve, "depth_curve": dep_curve,
            "log_length_range": [log_min, log_max],
        })
    return jsonify(out)


# Entry point
# ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5057)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
