#!/usr/bin/env python3
"""Aggregate the feature-source x budget mechanism analysis into one artifact.

Combines, per history budget (stratified five-borough test split):

- the budget-matched frequency ranker and the seed-42 mixed-budget generalist
  (from analysis_results/formal_main_table.json);
- the static_only and history_only ablation runs (per-L eval JSONs under
  runs_results/feature_source/);
- three-seed generalist Recall@20% at L=14/90/full (seeds 42/43/44) with
  mean/std, plus optional direct-BCE / STGCN full-history seed values.

Writes analysis_results/feature_source_budget_spectrum.json. Fails loudly if
any required run is missing so the paper never cites a partial analysis.

Usage:
    python tools/aggregate_feature_source_budget.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FS = ROOT / "runs_results" / "feature_source"
OUT = ROOT / "analysis_results" / "feature_source_budget_spectrum.json"
BUDGETS = [14, 30, 90, 180, 365, 0]
SEED_BUDGETS = [14, 90, 0]


def eval_metric(path: Path, key: str) -> float:
    doc = json.loads(path.read_text())
    return float(doc["test_metrics"][key])


def main() -> int:
    formal = json.loads((ROOT / "analysis_results" / "formal_main_table.json").read_text())
    spec = {int(row["L"]): row for row in formal["headline_spectrum_six"]}

    curves = {
        "ranker": {str(L): {"R20": spec[L]["ranker"]["R20"], "N20": spec[L]["ranker"]["N20"]}
                   for L in BUDGETS},
        "generalist_seed42": {
            str(L): {"R20": spec[L]["generalist"]["R20"], "N20": spec[L]["generalist"]["N20"]}
            for L in BUDGETS},
    }
    missing = []
    for name in ("static_only_s42", "history_only_s42"):
        per_L = {}
        for L in BUDGETS:
            path = FS / name / f"eval_L{L}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            per_L[str(L)] = {"R20": eval_metric(path, "es_recall_at_20"),
                             "N20": eval_metric(path, "es_ndcg_at_20")}
        curves[name] = per_L

    seeds = {"generalist": {}}
    for L in SEED_BUDGETS:
        values = {42: spec[L]["generalist"]["R20"]}
        for seed in (43, 44):
            path = FS / f"generalist_s{seed}" / f"eval_L{L}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            values[seed] = eval_metric(path, "es_recall_at_20")
        if len(values) == 3:
            vals = [values[s] for s in (42, 43, 44)]
            seeds["generalist"][str(L)] = {
                "values": values,
                "mean": statistics.fmean(vals),
                "std": statistics.stdev(vals),
            }
    for name in ("direct_bce_s43", "direct_bce_s44", "stgcn_s43", "stgcn_s44"):
        path = FS / name / "eval_L0.json"
        if path.exists():
            seeds.setdefault(name.rsplit("_s", 1)[0], {})[
                "full_" + name.rsplit("_s", 1)[1]] = eval_metric(path, "es_recall_at_20")

    if missing:
        print("missing eval artifacts:")
        for m in missing:
            print(" -", m)
        return 1

    doc = {
        "schema_version": 1,
        "scope": "NYC five boroughs, stratified allocation, test split, Recall@20%/NDCG@20%",
        "budgets": BUDGETS,
        "curves": curves,
        "seeds": seeds,
        "provenance": {
            "ranker_and_seed42": "analysis_results/formal_main_table.json",
            "ablations": ["runs_results/feature_source/static_only_s42",
                          "runs_results/feature_source/history_only_s42"],
            "seed_runs": ["runs_results/feature_source/generalist_s43",
                          "runs_results/feature_source/generalist_s44"],
        },
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for L in BUDGETS:
        row = {k: curves[k].get(str(L), {}).get("R20") for k in
               ("ranker", "static_only_s42", "history_only_s42", "generalist_seed42")}
        print(L, {k: (round(v, 4) if v is not None else None) for k, v in row.items()})
    for L, s in seeds["generalist"].items():
        print(f"seed mean@{L}: {s['mean']:.4f} +- {s['std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
