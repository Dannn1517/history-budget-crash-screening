"""Audit post-filter crash-to-road matching metadata in the released labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/crashes_daily_labels.parquet"),
    )
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[25.0, 50.0, 75.0])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    columns = [
        "id_date",
        "y_crash_count",
        "y_injured_sum",
        "y_killed_sum",
        "meta_snap_dist_m",
        "meta_nearest_tie_count",
    ]
    labels = pd.read_parquet(args.labels, columns=columns)
    labels["id_date"] = pd.to_datetime(labels["id_date"])
    labels = labels[
        (labels["id_date"] >= pd.Timestamp(args.start_date))
        & (labels["id_date"] <= pd.Timestamp(args.end_date))
    ]
    if labels.empty:
        raise ValueError("No label rows remain after date filtering.")
    if labels[["meta_snap_dist_m", "meta_nearest_tie_count"]].isna().any().any():
        raise ValueError("Matching metadata contain missing values.")

    distances = labels["meta_snap_dist_m"]
    tie_rows = labels["meta_nearest_tie_count"] > 1
    event_risk = (
        labels["y_crash_count"]
        + 2 * labels["y_injured_sum"]
        + 3 * labels["y_killed_sum"]
    )
    audit = {
        "scope": "post-filter aggregated link-day rows; not pre-filter event retention",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "link_day_rows": int(len(labels)),
        "matched_crashes": int(labels["y_crash_count"].sum()),
        "event_risk": {
            "construction": "y_crash_count + 2*y_injured_sum + 3*y_killed_sum",
            "legacy_y_risk_column_used": False,
            "sum": float(event_risk.sum()),
            "max_link_day": float(event_risk.max()),
        },
        "distance_m": {
            "mean": float(distances.mean()),
            "median": float(distances.median()),
            "p90": float(distances.quantile(0.90)),
            "p95": float(distances.quantile(0.95)),
            "p99": float(distances.quantile(0.99)),
            "max": float(distances.max()),
        },
        "nearest_ties": {
            "rows_above_one": int(tie_rows.sum()),
            "row_share": float(tie_rows.mean()),
            "max_count": int(labels["meta_nearest_tie_count"].max()),
        },
        "post_filter_threshold_row_share": {
            f"le_{threshold:g}m": float((distances <= threshold).mean())
            for threshold in args.thresholds
        },
    }
    rendered = json.dumps(audit, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
