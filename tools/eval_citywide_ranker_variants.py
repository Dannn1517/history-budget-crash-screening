#!/usr/bin/env python3
"""Citywide frequency-ranker standardization sensitivity .

The frozen cross-city spectrum evaluates the citywide ranker with per-partition
train-fitted z-scores (shared full-history statistics) whose scores are
concatenated across boroughs before Top-k selection. This tool recomputes the
budget-L ranker under alternative score conventions on identical test windows:

  raw        - unstandardized exclusive trailing-mean crash frequency,
               concatenated citywide (the "raw historical frequency" baseline);
  z_shared   - per-partition z-scores with shared full-history train stats
               (reproduces the frozen convention; sanity anchor);
  stratified - within-borough per-window Top-k (affine-invariant in the score,
               reported once for reference).

Windows, budgets and metrics follow the formal protocol exactly
(``build_time_splits`` + ``update_det_eval_state`` with binary occurrence
labels), so the z_shared numbers must match the frozen
``cross_city_spectrum_*`` ranker rows up to rounding.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from main_baseline import (
    capture_git_provenance,
    finalize_det_eval_state,
    init_det_eval_state,
    is_space_metric_key,
    parse_hr_ratios,
    update_det_eval_state,
)
from utils.data.loading import (
    build_time_splits,
    load_borough_name_order,
    load_mask,
    load_nyc_zinb_data,
)
from utils.data.priors import build_exclusive_cumulative_mean, build_trailing_exclusive_mean


def parse_number_list(text: str, cast=float) -> list:
    values = [cast(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("number list cannot be empty")
    return values


def _metric_dict(state: dict) -> dict:
    raw = finalize_det_eval_state(state)
    return {f"es_{key}": float(value) for key, value in raw.items() if is_space_metric_key(key)}


def evaluate_variant(
        y_event: np.ndarray,
        scores: np.ndarray,
        starts: list[int],
        history_len: int,
        horizon: int,
        hr_ratios: list[float],
        borough_indices: dict[str, np.ndarray] | None = None,
) -> dict:
    """Per-window Top-k evaluation of one score matrix [N, T].

    Without ``borough_indices`` the whole city is one allocation pool per
    window (citywide scope); with it, each borough is its own pool (stratified
    scope, pooled hits/denominators).
    """
    state = init_det_eval_state()
    pools = [None] if borough_indices is None else list(borough_indices.values())
    for start in starts:
        origin = int(start) + int(history_len)
        labels_all = y_event[:, origin:origin + horizon].astype(np.float32, copy=False)
        # The ranker scores every horizon day with the forecast-origin frequency
        # value (same convention as batch["freq_origin"] broadcast in the
        # cross-city spectrum tool).
        score_all = np.broadcast_to(
            scores[:, origin:origin + 1].astype(np.float32, copy=False), labels_all.shape
        )
        for pool in pools:
            idx = slice(None) if pool is None else pool
            labels = labels_all[idx]
            score = score_all[idx]
            update_det_eval_state(
                state,
                labels[None, :, :],
                labels[None, :, :],
                np.zeros_like(labels)[None, :, :],
                hr_ratios,
                score=score[None, :, :],
            )
    return _metric_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--budgets", default="14,30,90,180,365,0")
    parser.add_argument("--hr_ratio", default="0.05,0.1,0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis_results/citywide_ranker_variants_nyc.json")
    args = parser.parse_args()

    budgets = parse_number_list(args.budgets, int)
    hr_ratios = parse_hr_ratios(args.hr_ratio)

    started_at = time.perf_counter()
    provenance = capture_git_provenance()
    _x_static, y_risk, y_crash, _calendar, _weather, _borough, dates = load_nyc_zinb_data(
        args.data_dir, pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    )
    y_event = (y_risk > 0).astype(np.int8)
    train_starts, val_starts, test_starts = build_time_splits(
        y_event.shape[1], args.history_len, args.horizon
    )
    train_end = int(y_event.shape[1] * (8 / 12))
    borough_names = load_borough_name_order(args.data_dir)
    borough_indices = {
        name: np.flatnonzero(load_mask(args.data_dir, name, "", y_event.shape[0]))
        for name in borough_names
    }

    # Shared-statistics standardization: every budget-L channel is z-scored with
    # the FULL-history channel's train-period stats per borough (matching
    # standardize_freq_variants(shared=True, ref_index=-1) in bundle building).
    freq_full = build_exclusive_cumulative_mean(y_crash)
    shared_stats = {}
    for name, indices in borough_indices.items():
        fit_values = freq_full[indices, :train_end]
        mean = float(fit_values.mean())
        std = float(fit_values.std())
        shared_stats[name] = (mean, std if std > 0 else 1.0)

    results = {}
    for lookback in budgets:
        freq_raw = (
            build_exclusive_cumulative_mean(y_crash)
            if lookback == 0
            else build_trailing_exclusive_mean(y_crash, lookback)
        )
        freq_z = np.empty_like(freq_raw)
        for name, indices in borough_indices.items():
            mean, std = shared_stats[name]
            freq_z[indices] = (freq_raw[indices] - mean) / std
        entry = {
            "raw_citywide": evaluate_variant(
                y_event, freq_raw, test_starts, args.history_len, args.horizon, hr_ratios
            ),
            "z_shared_citywide": evaluate_variant(
                y_event, freq_z, test_starts, args.history_len, args.horizon, hr_ratios
            ),
            "z_shared_stratified": evaluate_variant(
                y_event, freq_z, test_starts, args.history_len, args.horizon, hr_ratios,
                borough_indices=borough_indices,
            ),
        }
        results[str(lookback)] = entry
        print(
            f"L={lookback or 'full'} raw_citywide R20={entry['raw_citywide']['es_recall_at_20']:.4f} "
            f"z_citywide R20={entry['z_shared_citywide']['es_recall_at_20']:.4f} "
            f"stratified R20={entry['z_shared_stratified']['es_recall_at_20']:.4f}"
        )

    document = {
        "schema_version": 1,
        "question": (
            "Does the citywide ranker's per-partition standardization convention change "
            "the allocation-scope conclusions (raw vs z-scored frequency)?"
        ),
        "protocol": {
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "boroughs": borough_names,
            "history_len": args.history_len,
            "horizon": args.horizon,
            "budgets": budgets,
            "test_windows": len(test_starts),
            "hr_ratio": hr_ratios,
            "seed": args.seed,
            "run": 1,
            "standardization": (
                "z_shared: per-borough z-scores with full-history-channel train stats "
                f"(train_end={train_end}); raw: unstandardized exclusive trailing mean"
            ),
            **provenance,
        },
        "by_budget": results,
        "wall_seconds": time.perf_counter() - started_at,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
