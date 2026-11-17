#!/usr/bin/env python3
"""Aggregate control experiments: attribution ablation and seed robustness.

Collects, from the run artefacts produced by tools/run_review3_chain.sh:

  * no-graph ablation (diffusion_steps=0, mixed-budget, seed 42):
    stratified Recall@20% at L=14/90/full and the citywide raw-convention
    spectrum at the same budgets;
  * three-seed citywide Recall@20% (seeds 42/43/44) for New York and Chicago
    at L=14/90/full, with mean and standard deviation across seeds.

Seed-42 citywide values are read from the frozen cross-city spectra so the
three-seed statistics reuse the published anchor exactly.

Outputs:
  - analysis_results/review3_controls.json

Usage:
    PYTHONPATH=. python tools/aggregate_review3_controls.py
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AR = os.path.join(ROOT, "analysis_results")
RR = os.path.join(ROOT, "runs_results")
BUDGETS = ("14", "90", "0")


def r20(node: dict) -> float:
    return float(node.get("es_recall_at_20", node["es_recall"]))


def spectrum_generalist(path: str) -> dict:
    doc = json.load(open(path, encoding="utf-8"))
    return {b: r20(doc["by_budget"][b]["generalist"]) for b in BUDGETS}


def eval_r20(path: str) -> float:
    doc = json.load(open(path, encoding="utf-8"))
    return r20(doc["test_metrics"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(AR, "review3_controls.json"))
    args = parser.parse_args()

    ng_dir = os.path.join(RR, "review_controls", "no_graph_mixed_s42")
    no_graph = {
        "stratified": {b: eval_r20(os.path.join(ng_dir, f"eval_L{b}.json")) for b in BUDGETS},
        "citywide": spectrum_generalist(os.path.join(ng_dir, "citywide_spectrum.json")),
    }

    seeds = {}
    for city, s42_path, seed_dirs in (
        ("nyc", os.path.join(AR, "cross_city_spectrum_nyc_citywide.json"),
         [os.path.join(RR, "feature_source", f"generalist_s{s}") for s in (43, 44)]),
        ("chicago", os.path.join(AR, "cross_city_spectrum_chicago_citywide.json"),
         [os.path.join(RR, "chicago_formal", f"generalist_s{s}") for s in (43, 44)]),
    ):
        per_budget = {}
        for b in BUDGETS:
            values = {"42": spectrum_generalist(s42_path)[b]}
            for d in seed_dirs:
                seed = os.path.basename(d).rsplit("_s", 1)[-1]
                values[seed] = spectrum_generalist(
                    os.path.join(d, "citywide_spectrum.json"))[b]
            vals = [values[k] for k in ("42", "43", "44")]
            mean = sum(vals) / 3.0
            std = (sum((v - mean) ** 2 for v in vals) / 2.0) ** 0.5
            per_budget[b] = {
                "values": {k: round(v, 4) for k, v in values.items()},
                "mean": round(mean, 4),
                "std": round(std, 4),
            }
        seeds[city] = per_budget

    document = {
        "schema_version": 1,
        "question": ("Control runs: does graph message passing carry the "
                      "short-history gain, and how stable are citywide "
                      "margins across training seeds?"),
        "no_graph_mixed_s42": {k: {b: round(v, 4) for b, v in kv.items()}
                               for k, kv in no_graph.items()},
        "citywide_seeds_42_43_44": seeds,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(document, indent=1)[:1200])
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
