#!/usr/bin/env python3
"""Independently trained simple supervised baselines .

Unlike the feature-source input-masking ablation (which zeroes inputs of an
already trained generalist), every model here is trained from scratch on its
own feature subset under the same temporal split, the same history-budget
constraint and the same per-forecast-window Top-k evaluation:

  * static-only logistic / MLP  -- public static road features only;
  * static + history fusion     -- validation-selected linear fusion of the
    z-scored static score and the budget-L exclusive trailing-mean frequency
    channel (the same truncated archive the ranker reads).

Both allocation scopes are reported: borough-stratified per-window Top-k and
unconstrained citywide same-day Top-k (per-partition z-scored scores
concatenated across boroughs, matching the cross-city spectrum convention).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from main_baseline import (
    capture_git_provenance,
    finalize_det_eval_state,
    init_det_eval_state,
    is_space_metric_key,
    parse_hr_ratios,
    update_det_eval_state,
)
from tools.eval_static_logistic_baseline import sample_static_training_rows, zscore_columns
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


def evaluate_scores(
        y_event: np.ndarray,
        borough_indices: dict[str, np.ndarray],
        starts: list[int],
        history_len: int,
        horizon: int,
        hr_ratios: list[float],
        static_scores: dict[str, np.ndarray],
        history_frequency: np.ndarray | None = None,
        static_weight: float = 1.0,
) -> tuple[dict, dict, dict[str, dict]]:
    """Evaluate static or static/history fused scores.

    Returns (stratified_pooled, citywide, per_borough) metric dicts. The
    citywide scope concatenates per-partition z-scored scores across boroughs
    before Top-k selection (one citywide road budget per forecast window).
    """
    pooled = init_det_eval_state()
    city = init_det_eval_state()
    local_states = {name: init_det_eval_state() for name in borough_indices}
    static_z = {name: zscore_columns(static_scores[name]) for name in borough_indices}
    for start in starts:
        origin = int(start) + int(history_len)
        city_scores, city_labels = [], []
        for name, indices in borough_indices.items():
            labels = y_event[indices, origin:origin + horizon].astype(np.float32, copy=False)
            if history_frequency is None:
                score = static_z[name]
            else:
                history_z = zscore_columns(history_frequency[indices, origin])
                score = static_weight * static_z[name] + (1.0 - static_weight) * history_z
            for state in (local_states[name], pooled):
                update_det_eval_state(
                    state,
                    labels[None, :, :],
                    labels[None, :, :],
                    np.zeros_like(labels)[None, :, :],
                    hr_ratios,
                    score=score[None, :, :],
                )
            city_scores.append(score)
            city_labels.append(labels)
        city_score = np.concatenate(city_scores, axis=0)
        city_label = np.concatenate(city_labels, axis=0)
        update_det_eval_state(
            city,
            city_label[None, :, :],
            city_label[None, :, :],
            np.zeros_like(city_label)[None, :, :],
            hr_ratios,
            score=city_score[None, :, :],
        )
    return (
        _metric_dict(pooled),
        _metric_dict(city),
        {name: _metric_dict(state) for name, state in local_states.items()},
    )


def fit_logistic_models(x_train, y_train, c_value, max_iter, seed):
    models = []
    for step in range(y_train.shape[1]):
        target = y_train[:, step]
        if np.unique(target).size != 2:
            raise ValueError(f"horizon step {step} does not contain both classes")
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            solver="lbfgs",
            max_iter=int(max_iter),
            random_state=int(seed + step),
        )
        model.fit(x_train, target)
        models.append(model)
    return models


def fit_mlp_models(x_train, y_train, hidden, max_iter, seed, neg_ratio):
    """Per-horizon MLP classifiers on class-balanced row subsamples.

    sklearn 1.6 MLPClassifier has no class_weight support, so balance by
    keeping every positive row and ``neg_ratio`` randomly drawn negatives per
    positive (documented in the output artifact).
    """
    models = []
    for step in range(y_train.shape[1]):
        target = y_train[:, step]
        pos = np.flatnonzero(target == 1)
        neg = np.flatnonzero(target == 0)
        if pos.size == 0 or neg.size == 0:
            raise ValueError(f"horizon step {step} does not contain both classes")
        rng = np.random.default_rng(int(seed + step))
        if neg_ratio > 0 and neg.size > neg_ratio * pos.size:
            neg = rng.choice(neg, size=int(neg_ratio * pos.size), replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        model = MLPClassifier(
            hidden_layer_sizes=tuple(int(v) for v in hidden),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=int(max_iter),
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=5,
            random_state=int(seed + step),
        )
        model.fit(x_train[idx], target[idx])
        models.append(model)
    return models


def predict_positive(models, features: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(features)[:, 1] for model in models], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--model", default="logistic", choices=["logistic", "mlp"])
    parser.add_argument("--lookbacks", default="14,30,90,0")
    parser.add_argument("--c_values", default="0.01,0.1,1,10")
    parser.add_argument("--mlp_hidden", default="64,64")
    parser.add_argument("--mlp_max_iter", type=int, default=40)
    parser.add_argument("--mlp_neg_ratio", type=int, default=10)
    parser.add_argument("--static_weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--hr_ratio", default="0.05,0.1,0.2")
    parser.add_argument("--selection_metric", default="es_ndcg_at_20")
    parser.add_argument("--train_max_samples", type=int, default=200000)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    lookbacks = parse_number_list(args.lookbacks, int)
    static_weights = parse_number_list(args.static_weights, float)
    c_values = parse_number_list(args.c_values, float)
    hr_ratios = parse_hr_ratios(args.hr_ratio)
    hidden = parse_number_list(args.mlp_hidden, int)
    if any(weight < 0 or weight > 1 for weight in static_weights):
        raise ValueError("static_weights must lie in [0, 1]")
    out_path = args.out or (
        f"analysis_results/simple_supervised_{args.model}_baseline.json"
    )

    started_at = time.perf_counter()
    provenance = capture_git_provenance()
    x_static, y_risk, y_crash, _calendar, _weather, _borough, dates = load_nyc_zinb_data(
        args.data_dir, pd.Timestamp(args.start_date), pd.Timestamp(args.end_date)
    )
    y_event = (y_risk > 0).astype(np.int8)
    train_starts, val_starts, test_starts = build_time_splits(
        y_event.shape[1], args.history_len, args.horizon
    )
    borough_names = load_borough_name_order(args.data_dir)
    borough_indices = {
        name: np.flatnonzero(load_mask(args.data_dir, name, "", y_event.shape[0]))
        for name in borough_names
    }

    rng = np.random.default_rng(args.seed)
    per_borough_cap = (
        0
        if args.train_max_samples <= 0
        else max(1, int(np.ceil(args.train_max_samples / len(borough_names))))
    )
    train_parts = [
        sample_static_training_rows(
            x_static,
            y_event,
            indices,
            train_starts,
            args.history_len,
            args.horizon,
            per_borough_cap,
            rng,
        )
        for indices in borough_indices.values()
    ]
    x_train = np.concatenate([part[0] for part in train_parts], axis=0)
    y_train = np.concatenate([part[1] for part in train_parts], axis=0)
    if x_train.shape[0] > args.train_max_samples > 0:
        keep = rng.choice(x_train.shape[0], size=args.train_max_samples, replace=False)
        x_train = x_train[keep]
        y_train = y_train[keep]
    feature_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)
    print(
        f"[{args.model}] training samples={x_train.shape[0]} features={x_train.shape[1]} "
        f"positive_rates={y_train.mean(axis=0).tolist()}"
    )

    models_by_label: dict[str, list] = {}
    scores_by_label: dict[str, dict[str, np.ndarray]] = {}
    validation_by_label: dict[str, dict] = {}
    training_detail: dict[str, dict] = {}
    if args.model == "logistic":
        for c_value in c_values:
            label = f"C={c_value:g}"
            models = fit_logistic_models(x_train_scaled, y_train, c_value, args.max_iter, args.seed)
            scores = {
                name: predict_positive(models, feature_scaler.transform(x_static[indices]))
                for name, indices in borough_indices.items()
            }
            metrics, _, _ = evaluate_scores(
                y_event, borough_indices, val_starts, args.history_len, args.horizon,
                hr_ratios, scores,
            )
            models_by_label[label] = models
            scores_by_label[label] = scores
            validation_by_label[label] = metrics
            training_detail[label] = {
                "selected_iterations_by_horizon": [int(m.n_iter_[0]) for m in models],
                "selected_converged": all(int(m.n_iter_[0]) < args.max_iter for m in models),
            }
            print(f"{label} val R20/N20={metrics['es_recall_at_20']:.4f}/{metrics['es_ndcg_at_20']:.4f}")
        selected = max(
            validation_by_label,
            key=lambda label: (
                validation_by_label[label][args.selection_metric],
                -c_values.index(float(label[2:])),
            ),
        )
    else:
        label = f"mlp_hidden={'x'.join(str(v) for v in hidden)}"
        models = fit_mlp_models(
            x_train_scaled, y_train, hidden, args.mlp_max_iter, args.seed, args.mlp_neg_ratio
        )
        scores = {
            name: predict_positive(models, feature_scaler.transform(x_static[indices]))
            for name, indices in borough_indices.items()
        }
        metrics, _, _ = evaluate_scores(
            y_event, borough_indices, val_starts, args.history_len, args.horizon,
            hr_ratios, scores,
        )
        models_by_label[label] = models
        scores_by_label[label] = scores
        validation_by_label[label] = metrics
        training_detail[label] = {
            "n_iter_by_horizon": [int(m.n_iter_) for m in models],
            "neg_ratio": args.mlp_neg_ratio,
            "balanced_subsampling": "all positives + neg_ratio x negatives per horizon step",
        }
        selected = label
        print(f"{label} val R20/N20={metrics['es_recall_at_20']:.4f}/{metrics['es_ndcg_at_20']:.4f}")

    models = models_by_label[selected]
    static_scores = scores_by_label[selected]
    static_test, static_test_city, static_test_by_borough = evaluate_scores(
        y_event, borough_indices, test_starts, args.history_len, args.horizon,
        hr_ratios, static_scores,
    )

    fusion_results = {}
    for lookback in lookbacks:
        history_frequency = (
            build_exclusive_cumulative_mean(y_crash)
            if lookback == 0
            else build_trailing_exclusive_mean(y_crash, lookback)
        )
        validation = {}
        for weight in static_weights:
            label = f"static_weight={weight:g}"
            metrics, _, _ = evaluate_scores(
                y_event, borough_indices, val_starts, args.history_len, args.horizon,
                hr_ratios, static_scores, history_frequency, weight,
            )
            validation[label] = metrics
        selected_weight_label = max(
            validation,
            key=lambda label: (
                validation[label][args.selection_metric],
                -static_weights.index(float(label.split("=")[1])),
            ),
        )
        weight = float(selected_weight_label.split("=")[1])
        test_metrics, test_city, test_by_borough = evaluate_scores(
            y_event, borough_indices, test_starts, args.history_len, args.horizon,
            hr_ratios, static_scores, history_frequency, weight,
        )
        fusion_results[str(lookback)] = {
            "validation_candidates": validation,
            "selected": selected_weight_label,
            "test_stratified": test_metrics,
            "test_citywide": test_city,
            "test_by_borough": test_by_borough,
        }
        print(
            f"L={lookback or 'full'} {selected_weight_label} "
            f"test R20 stratified={test_metrics['es_recall_at_20']:.4f} "
            f"citywide={test_city['es_recall_at_20']:.4f}"
        )
        del history_frequency

    document = {
        "schema_version": 1,
        "question": (
            "Can independently trained supervised models on static features (and their "
            "fusion with budget-L crash frequency) match the learned spatiotemporal model?"
        ),
        "protocol": {
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "boroughs": borough_names,
            "evaluation": (
                "per-forecast-window Top-k; borough-stratified pooled + unconstrained "
                "citywide (per-partition z-scored scores concatenated)"
            ),
            "features": "public static road features only; history enters only through explicit fusion",
            "history_len": args.history_len,
            "horizon": args.horizon,
            "lookbacks": lookbacks,
            "selection_metric": args.selection_metric,
            "train_max_samples": args.train_max_samples,
            "seed": args.seed,
            "run": 1,
            **provenance,
        },
        "training": {
            "samples": int(x_train.shape[0]),
            "feature_dim": int(x_train.shape[1]),
            "feature_scaler": "StandardScaler fitted on sampled training rows only",
            "positive_rates_by_horizon": y_train.mean(axis=0).tolist(),
            "model": args.model,
            "selected": selected,
            "candidates": training_detail,
            "validation_candidates": validation_by_label,
        },
        "static_only_test_stratified": static_test,
        "static_only_test_citywide": static_test_city,
        "static_only_test_by_borough": static_test_by_borough,
        "fusion_by_L": fusion_results,
        "wall_seconds": time.perf_counter() - started_at,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
