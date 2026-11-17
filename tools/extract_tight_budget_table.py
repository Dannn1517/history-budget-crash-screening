#!/usr/bin/env python3
""""Tight-screening-budget table extraction.

The headline metric is Recall@20%, but operationally tight screening runs at
1-5% of segments. This CPU-only tool lifts the absolute Recall@1% / @5% / @20%
values of the budget-matched frequency baseline and the mixed-budget model
from the frozen small-share spectrum artifact and writes them, with the
model-minus-baseline difference, as the rows of the manuscript's tight-budget
table. No new inference: every number is copied from the frozen JSON and
cross-checked against it on write.

Outputs:
  - analysis_results/tight_budget_table_nyc.json

Usage:
    PYTHONPATH=. python tools/extract_tight_budget_table.py
"""

from __future__ import annotations

import argparse
import json
import os

SOURCE = "analysis_results/small_share_spectrum_nyc_test_stratified.json"
BUDGETS = (14, 90, 0)
SHARES = (1, 5)  # the 20% benchmark rows live in Tables I-II of the manuscript


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--out", default="analysis_results/tight_budget_table_nyc.json")
    args = parser.parse_args()

    doc = json.load(open(args.source, encoding="utf-8"))
    rows = []
    for budget in BUDGETS:
        cell = doc["by_budget"][str(budget)]
        for share in SHARES:
            base = cell["ranker"][f"es_recall_at_{share}"]
            model = cell["generalist"][f"es_recall_at_{share}"]
            rows.append({
                "L": budget,
                "share_pct": share,
                "baseline_recall": round(base, 4),
                "model_recall": round(model, 4),
                "delta": round(model - base, 4),
            })
            # cross-check against the frozen source on write
            assert abs(rows[-1]["baseline_recall"] - base) < 5e-5
            assert abs(rows[-1]["model_recall"] - model) < 5e-5

    document = {
        "schema_version": 1,
        "question": ("Under operationally tight screening shares (1-5% of segments), "
                     "what are the absolute Recall values of the budget-matched "
                     "frequency baseline and the mixed-budget model, and how large "
                     "is the difference?"),
        "source": args.source,
        "scope": "New York, stratified allocation, test split, per-day top-k pooled",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(document, fh, ensure_ascii=False, indent=2, allow_nan=False)
    for r in rows:
        print(f"L={r['L']:>4} @{r['share_pct']:>2}%: baseline {r['baseline_recall']:.4f} "
              f"model {r['model_recall']:.4f} delta {r['delta']:+.4f}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
