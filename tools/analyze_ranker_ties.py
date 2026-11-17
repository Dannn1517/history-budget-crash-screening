#!/usr/bin/env python3
"""Tie structure of the zero-training frequency ranker.

At short budgets most segments have no crash in the readable archive, so the
ranker's score is exactly zero for a large majority of roads and the top-k set
is partly filled by tied scores resolved in segment order. This tool
quantifies, per budget on the test windows:

  * zero_fraction        - share of segments with frequency score 0;
  * tie_determined_at_20 - share of the top-20% quota drawn from the tied
                           group at the selection boundary;
  * ranker Recall@20% under three tie-breaking rules:
      deterministic (segment order, the frozen convention),
      random        (uniform random within tie groups, R repeats),
      static_prior  (static logistic score as secondary key, i.e. a
                     road-attribute prior breaks the zero ties).

Reported for both allocation scopes. Answers whether the short-history margin
is sensitive to the tie convention and how much of the budget contraction
coincides with the disappearance of zero ties.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from main_baseline import capture_git_provenance
from tools.eval_simple_supervised_baselines import (
    fit_logistic_models,
    sample_static_training_rows,
)
from sklearn.preprocessing import StandardScaler
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


def rank_orders(score: np.ndarray, rule: str, prior: np.ndarray | None,
                rng: np.random.Generator | None) -> np.ndarray:
    if rule == "deterministic":
        return np.argsort(-score, kind="mergesort")
    if rule == "random":
        return np.lexsort((rng.random(score.size), -score))
    if rule == "static_prior":
        return np.lexsort((-prior, -score))
    raise ValueError(rule)


def recall_at_share(score: np.ndarray, labels: np.ndarray, share: float,
                    rule: str, prior: np.ndarray | None,
                    rng: np.random.Generator | None) -> tuple[float, float]:
    """(hits, positives) for one pool, one window, all horizon days."""
    order = rank_orders(score, rule, prior, rng)
    k = max(1, int(np.ceil(share * score.size)))
    selected = order[:k]
    hits = float(labels[selected].sum())
    return hits, float(labels.sum())


def tie_stats(score: np.ndarray, share: float) -> tuple[float, float]:
    """(zero_fraction, tie-determined share of the top-k quota)."""
    zero_fraction = float(np.mean(score == 0))
    k = max(1, int(np.ceil(share * score.size)))
    order = np.argsort(-score, kind="mergesort")
    boundary = score[order[k - 1]]
    n_gt = int(np.sum(score > boundary))
    tie_determined = (k - n_gt) / float(k)
    return zero_fraction, tie_determined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--budgets", default="14,30,90,180,365,0")
    parser.add_argument("--share", type=float, default=0.20)
    parser.add_argument("--random_repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis_results/ranker_tie_structure_nyc.json")
    args = parser.parse_args()

    budgets = parse_number_list(args.budgets, int)
    started_at = time.perf_counter()
    provenance = capture_git_provenance()
    x_static, y_risk, y_crash, _cal, _wx, _bor, dates = load_nyc_zinb_data(
        args.data_dir, pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    )
    y_event = (y_risk > 0).astype(np.int8)
    _train_starts, _val_starts, test_starts = build_time_splits(
        y_event.shape[1], args.history_len, args.horizon
    )
    borough_names = load_borough_name_order(args.data_dir)
    borough_indices = {
        name: np.flatnonzero(load_mask(args.data_dir, name, "", y_event.shape[0]))
        for name in borough_names
    }

    # static road-attribute prior for tie-breaking (same sampler/fit as the
    # simple-baseline tool, fixed C=0.01, horizon-0 score as node prior)
    rng = np.random.default_rng(args.seed)
    cap = max(1, int(np.ceil(200000 / len(borough_names))))
    parts = [
        sample_static_training_rows(x_static, y_event, idx, _train_starts,
                                    args.history_len, args.horizon, cap, rng)
        for idx in borough_indices.values()
    ]
    x_train = np.concatenate([p[0] for p in parts], axis=0)
    y_train = np.concatenate([p[1] for p in parts], axis=0)
    scaler = StandardScaler().fit(x_train)
    models = fit_logistic_models(scaler.transform(x_train), y_train, 0.01, 300, args.seed)
    static_prior = models[0].predict_proba(scaler.transform(x_static))[:, 1]

    results = {}
    for lookback in budgets:
        freq = (
            build_exclusive_cumulative_mean(y_crash)
            if lookback == 0
            else build_trailing_exclusive_mean(y_crash, lookback)
        )
        entry = {"zero_fraction": [],
                 "tie_determined_at_share_citywide": [],
                 "tie_determined_at_share_stratified": []}
        rules = {}
        for scope, pools in (("stratified", list(borough_indices.values())),
                             ("citywide", [np.arange(y_event.shape[0])])):
            accum = {rule: [0.0, 0.0] for rule in
                     ("deterministic", "static_prior", *[f"random_{r}" for r in range(args.random_repeats)])}
            for start in test_starts:
                origin = int(start) + args.history_len
                labels_all = y_event[:, origin:origin + args.horizon]
                score_all = freq[:, origin]
                for pool in pools:
                    s = score_all[pool]
                    lab = labels_all[pool]
                    zf, td = tie_stats(s, args.share)
                    if scope == "citywide":
                        entry["zero_fraction"].append(zf)
                        entry["tie_determined_at_share_citywide"].append(td)
                    else:
                        entry["tie_determined_at_share_stratified"].append(td)
                    for rule in accum:
                        r = None
                        if rule.startswith("random_"):
                            r = np.random.default_rng(args.seed + int(rule.split("_")[1]))
                        h, p = recall_at_share(s, lab, args.share,
                                               "random" if r is not None else rule,
                                               static_prior[pool], r)
                        accum[rule][0] += h
                        accum[rule][1] += p
            det = accum["deterministic"][0] / max(accum["deterministic"][1], 1)
            stat = accum["static_prior"][0] / max(accum["static_prior"][1], 1)
            rand = np.array([
                accum[f"random_{r}"][0] / max(accum[f"random_{r}"][1], 1)
                for r in range(args.random_repeats)
            ])
            rules[scope] = {
                "recall_at_20_deterministic": det,
                "recall_at_20_random_mean": float(rand.mean()),
                "recall_at_20_random_std": float(rand.std(ddof=1)),
                "recall_at_20_static_prior": stat,
            }
        entry["rules"] = rules
        entry["zero_fraction_mean"] = float(np.mean(entry["zero_fraction"]))
        entry["tie_determined_citywide_mean"] = float(
            np.mean(entry.pop("tie_determined_at_share_citywide")))
        entry["tie_determined_stratified_mean"] = float(
            np.mean(entry.pop("tie_determined_at_share_stratified")))
        entry.pop("zero_fraction")
        results[str(lookback)] = entry
        print(
            f"L={lookback or 'full'}: zero_frac={entry['zero_fraction_mean']:.3f} "
            f"tie_det@20% strat={entry['tie_determined_stratified_mean']:.3f} "
            f"city={entry['tie_determined_citywide_mean']:.3f} | strat R20 "
            f"det={rules['stratified']['recall_at_20_deterministic']:.4f} "
            f"rand={rules['stratified']['recall_at_20_random_mean']:.4f}"
            f"±{rules['stratified']['recall_at_20_random_std']:.4f} "
            f"static={rules['stratified']['recall_at_20_static_prior']:.4f}"
        )

    document = {
        "schema_version": 1,
        "question": (
            "How much of the short-history ranker behaviour and margin is determined "
            "by zero-score ties and the tie-breaking convention?"
        ),
        "protocol": {
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "boroughs": borough_names,
            "history_len": args.history_len,
            "horizon": args.horizon,
            "budgets": budgets,
            "share": args.share,
            "test_windows": len(test_starts),
            "random_repeats": args.random_repeats,
            "seed": args.seed,
            "run": 1,
            "static_prior": "logistic C=0.01 on 200k sampled train rows, horizon-0 score",
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
