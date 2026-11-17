#!/usr/bin/env python3
"""Low-cost statistical baselines for limited-history event retrieval.

The script tunes each lightweight baseline on the validation split and reports
one test result. Scores at forecast origin use every observed history day,
including the final day of the 14-day model input window.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from main_baseline import (
    finalize_det_eval_state,
    init_det_eval_state,
    is_space_metric_key,
    parse_hr_ratios,
    update_det_eval_state,
)
from utils.data.loading import build_time_splits, load_mask, load_nyc_zinb_data
from utils.data.priors import build_exclusive_cumulative_mean, build_trailing_exclusive_mean
from utils.data.statistical_baselines import (
    build_exclusive_ewma,
    build_induced_random_walk,
    effective_ewma_weight_mass,
    effective_history_days,
    estimate_group_event_priors,
    posterior_group_rate,
)


def parse_number_list(text: str, cast=float) -> list:
    values = [cast(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("number list cannot be empty")
    return values


def evaluate_origin_scores(
        y_risk: np.ndarray,
        origin_scores: np.ndarray,
        start_indices: list[int],
        history_len: int,
        horizon: int,
        hr_ratios: list[float],
) -> dict:
    if origin_scores.shape != (y_risk.shape[0], len(start_indices)):
        raise ValueError(
            f"origin_scores must have shape {(y_risk.shape[0], len(start_indices))}, "
            f"got {origin_scores.shape}"
        )
    state = init_det_eval_state()
    for column, start in enumerate(start_indices):
        origin = start + history_len
        labels = (y_risk[:, origin:origin + horizon] > 0).astype(np.float32)
        score = np.broadcast_to(origin_scores[:, column, None], labels.shape).astype(np.float32)
        update_det_eval_state(
            state,
            labels[None, :, :],
            labels[None, :, :],
            np.zeros_like(labels)[None, :, :],
            hr_ratios,
            score=score[None, :, :],
        )
    raw = finalize_det_eval_state(state)
    return {f"es_{key}": float(value) for key, value in raw.items() if is_space_metric_key(key)}


def evaluate_candidate_grid(
        candidates: Dict[str, tuple[np.ndarray, np.ndarray]],
        y_risk: np.ndarray,
        val_starts: list[int],
        test_starts: list[int],
        history_len: int,
        horizon: int,
        hr_ratios: list[float],
        selection_metric: str,
) -> dict:
    validation = {}
    for name, (val_scores, _test_scores) in candidates.items():
        validation[name] = evaluate_origin_scores(
            y_risk, val_scores, val_starts, history_len, horizon, hr_ratios
        )
    selected = max(
        candidates,
        key=lambda name: (validation[name][selection_metric], -list(candidates).index(name)),
    )
    test = evaluate_origin_scores(
        y_risk,
        candidates[selected][1],
        test_starts,
        history_len,
        horizon,
        hr_ratios,
    )
    return {
        "selected": selected,
        "selection_metric": selection_metric,
        "validation_candidates": validation,
        "test": test,
    }


def load_highway_groups(data_dir: str, node_indices: np.ndarray) -> np.ndarray:
    node_df = pd.read_parquet(
        os.path.join(data_dir, "line_graph", "node_id_map.parquet")
    ).sort_values("node_idx")
    links_df = pd.read_parquet(
        os.path.join(data_dir, "links_meta.parquet"),
        columns=["id_geom_hash", "meta_highway_norm"],
    )
    merged = node_df[["node_idx", "id_geom_hash"]].merge(
        links_df, on="id_geom_hash", how="left", validate="one_to_one"
    ).sort_values("node_idx")
    groups = merged["meta_highway_norm"].fillna("unknown").astype(str).to_numpy()
    return groups[node_indices]


def history_rate(values: np.ndarray, lookback: int) -> np.ndarray:
    if lookback > 0:
        return build_trailing_exclusive_mean(values, lookback)
    return build_exclusive_cumulative_mean(values)


def select_origins(matrix: np.ndarray, starts: Iterable[int], history_len: int) -> np.ndarray:
    origins = np.asarray(list(starts), dtype=np.int64) + int(history_len)
    return np.ascontiguousarray(matrix[:, origins], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lookbacks", default="14,30,90,180,0")
    parser.add_argument("--half_lives", default="3,7,14,30,90")
    parser.add_argument("--neighbor_weights", default="0.1,0.25,0.5,0.75,1.0")
    parser.add_argument("--prior_strengths", default="1,7,14,30,90,180,365,1000000")
    parser.add_argument("--hr_ratio", default="0.05,0.10,0.20")
    parser.add_argument("--selection_metric", default="es_ndcg_at_20")
    parser.add_argument("--mask_borough", default="Manhattan")
    parser.add_argument("--out", default="analysis_results/history_statistical_baselines_gate.json")
    args = parser.parse_args()

    if args.mask_borough.strip().lower() in ("", "all"):
        raise ValueError("This low-cost gate expects one borough; run pooled formal evaluation only after it passes.")

    lookbacks = parse_number_list(args.lookbacks, int)
    half_lives = parse_number_list(args.half_lives, float)
    neighbor_weights = parse_number_list(args.neighbor_weights, float)
    prior_strengths = parse_number_list(args.prior_strengths, float)
    hr_ratios = parse_hr_ratios(args.hr_ratio)

    _x_static, y_risk_full, y_crash_full, _calendar, _weather, _borough, dates = load_nyc_zinb_data(
        args.data_dir, pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    )
    mask = load_mask(args.data_dir, args.mask_borough, "", y_risk_full.shape[0])
    node_indices = np.where(mask)[0]
    y_risk = np.ascontiguousarray(y_risk_full[node_indices])
    y_crash = np.ascontiguousarray(y_crash_full[node_indices])
    events = (y_crash > 0).astype(np.float32)
    del y_risk_full, y_crash_full

    train_starts, val_starts, test_starts = build_time_splits(
        y_risk.shape[1], args.history_len, args.horizon
    )
    train_end = max(start + args.history_len + args.horizon for start in train_starts)
    highway_groups = load_highway_groups(args.data_dir, node_indices)
    group_prior = estimate_group_event_priors(events, highway_groups, train_end=train_end)
    edge_index = np.load(os.path.join(args.data_dir, "line_graph", "edge_index.npy"))
    random_walk = build_induced_random_walk(edge_index, node_indices)

    val_origins = np.asarray(val_starts, dtype=np.int64) + args.history_len
    test_origins = np.asarray(test_starts, dtype=np.int64) + args.history_len
    observed_days = effective_history_days(y_risk.shape[1], 0)
    results = {}

    for lookback in lookbacks:
        count_rate_full = history_rate(y_crash, lookback)
        event_rate_full = history_rate(events, lookback)
        count_val = select_origins(count_rate_full, val_starts, args.history_len)
        count_test = select_origins(count_rate_full, test_starts, args.history_len)
        event_val = select_origins(event_rate_full, val_starts, args.history_len)
        event_test = select_origins(event_rate_full, test_starts, args.history_len)
        del count_rate_full, event_rate_full

        frequency_count = {
            "validation": evaluate_origin_scores(
                y_risk, count_val, val_starts, args.history_len, args.horizon, hr_ratios
            ),
            "test": evaluate_origin_scores(
                y_risk, count_test, test_starts, args.history_len, args.horizon, hr_ratios
            ),
        }
        frequency_event = {
            "validation": evaluate_origin_scores(
                y_risk, event_val, val_starts, args.history_len, args.horizon, hr_ratios
            ),
            "test": evaluate_origin_scores(
                y_risk, event_test, test_starts, args.history_len, args.horizon, hr_ratios
            ),
        }

        ewma_candidates = {}
        for half_life in half_lives:
            ewma_full = build_exclusive_ewma(events, half_life=half_life, lookback=lookback)
            ewma_candidates[f"half_life={half_life:g}"] = (
                select_origins(ewma_full, val_starts, args.history_len),
                select_origins(ewma_full, test_starts, args.history_len),
            )
            del ewma_full
        ewma = evaluate_candidate_grid(
            ewma_candidates, y_risk, val_starts, test_starts, args.history_len,
            args.horizon, hr_ratios, args.selection_metric,
        )

        recency_group_validation = {}
        recency_group_selected = None
        recency_group_selected_test = None
        recency_group_best_value = float("-inf")
        for half_life in half_lives:
            name = f"half_life={half_life:g}"
            ewma_val, ewma_test = ewma_candidates[name]
            weight_mass = effective_ewma_weight_mass(
                y_risk.shape[1], half_life=half_life, lookback=lookback
            )
            mass_val = weight_mass[val_origins]
            mass_test = weight_mass[test_origins]
            for strength in prior_strengths:
                candidate_name = f"half_life={half_life:g},strength={strength:g}"
                candidate_val = posterior_group_rate(
                    ewma_val, mass_val, group_prior, strength
                )
                metrics = evaluate_origin_scores(
                    y_risk, candidate_val, val_starts, args.history_len, args.horizon, hr_ratios
                )
                recency_group_validation[candidate_name] = metrics
                value = metrics[args.selection_metric]
                if value > recency_group_best_value:
                    recency_group_best_value = value
                    recency_group_selected = candidate_name
                    recency_group_selected_test = posterior_group_rate(
                        ewma_test, mass_test, group_prior, strength
                    )
        assert recency_group_selected is not None and recency_group_selected_test is not None
        recency_group_eb = {
            "selected": recency_group_selected,
            "selection_metric": args.selection_metric,
            "validation_candidates": recency_group_validation,
            "test": evaluate_origin_scores(
                y_risk, recency_group_selected_test, test_starts,
                args.history_len, args.horizon, hr_ratios,
            ),
        }

        neighbor_val = random_walk.dot(event_val).astype(np.float32, copy=False)
        neighbor_test = random_walk.dot(event_test).astype(np.float32, copy=False)
        neighbor_candidates = {
            f"weight={weight:g}": (
                (1.0 - weight) * event_val + weight * neighbor_val,
                (1.0 - weight) * event_test + weight * neighbor_test,
            )
            for weight in neighbor_weights
        }
        neighbor = evaluate_candidate_grid(
            neighbor_candidates, y_risk, val_starts, test_starts, args.history_len,
            args.horizon, hr_ratios, args.selection_metric,
        )

        days = effective_history_days(y_risk.shape[1], lookback)
        days_val = days[val_origins]
        days_test = days[test_origins]
        group_candidates = {
            f"strength={strength:g}": (
                posterior_group_rate(event_val, days_val, group_prior, strength),
                posterior_group_rate(event_test, days_test, group_prior, strength),
            )
            for strength in prior_strengths
        }
        road_type_eb = evaluate_candidate_grid(
            group_candidates, y_risk, val_starts, test_starts, args.history_len,
            args.horizon, hr_ratios, args.selection_metric,
        )

        key = str(lookback)
        results[key] = {
            "frequency_count": frequency_count,
            "frequency_event": frequency_event,
            "ewma": ewma,
            "neighbor_pool": neighbor,
            "road_type_empirical_bayes": road_type_eb,
            "recency_road_type_empirical_bayes": recency_group_eb,
        }
        print(
            f"L={lookback or 'full'} | "
            f"freq R20/N20={frequency_count['test']['es_recall_at_20']:.4f}/"
            f"{frequency_count['test']['es_ndcg_at_20']:.4f} | "
            f"EWMA({ewma['selected']})={ewma['test']['es_recall_at_20']:.4f}/"
            f"{ewma['test']['es_ndcg_at_20']:.4f} | "
            f"neighbor({neighbor['selected']})={neighbor['test']['es_recall_at_20']:.4f}/"
            f"{neighbor['test']['es_ndcg_at_20']:.4f} | "
            f"road-type EB({road_type_eb['selected']})="
            f"{road_type_eb['test']['es_recall_at_20']:.4f}/"
            f"{road_type_eb['test']['es_ndcg_at_20']:.4f} | "
            f"recency+road-type EB({recency_group_eb['selected']})="
            f"{recency_group_eb['test']['es_recall_at_20']:.4f}/"
            f"{recency_group_eb['test']['es_ndcg_at_20']:.4f}"
        )

    document = {
        "schema_version": 1,
        "question": "Can lightweight statistical shrinkage explain the limited-history model gain?",
        "protocol": {
            "borough": args.mask_borough,
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "history_len": args.history_len,
            "horizon": args.horizon,
            "lookbacks": lookbacks,
            "selection_metric": args.selection_metric,
            "score_time": "forecast origin; includes the final observed input day",
            "group_prior": "training-period event rate pooled by meta_highway_norm",
            "run": 1,
        },
        "global_beta_binomial_note": (
            "With identical observed days for every road, a single global Beta-Binomial prior "
            "is a monotone transform of event count and therefore has exactly the same Top-k ranking."
        ),
        "results_by_L": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(document, file, indent=2, ensure_ascii=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
