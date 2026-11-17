#!/usr/bin/env python3
"""Evaluate static logistic risk and validation-tuned history fusion.

The logistic component uses only public road-level static features. Historical
crash frequency is added only in the explicit fusion, so the two information
sources remain identifiable. All rankings are evaluated per forecast window
and borough before pooling, matching the neural-model protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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


def zscore_columns(values: np.ndarray) -> np.ndarray:
    """Standardize node scores independently for each forecast horizon."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("values must have shape [nodes] or [nodes, horizon]")
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    return np.divide(array - mean, std, out=np.zeros_like(array), where=std > 0)


def sample_static_training_rows(
    x_static: np.ndarray,
    y_event: np.ndarray,
    node_indices: np.ndarray,
    train_starts: list[int],
    history_len: int,
    horizon: int,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly sample node-window pairs from one borough."""
    node_indices = np.asarray(node_indices, dtype=np.int64)
    starts = np.asarray(train_starts, dtype=np.int64)
    if node_indices.size == 0 or starts.size == 0:
        raise ValueError("node_indices and train_starts must be non-empty")
    total = int(node_indices.size * starts.size)
    count = total if max_samples <= 0 else min(total, int(max_samples))
    flat = rng.choice(total, size=count, replace=False)
    flat.sort()
    start_positions = flat // node_indices.size
    local_nodes = flat % node_indices.size
    global_nodes = node_indices[local_nodes]
    origins = starts[start_positions] + int(history_len)
    targets = np.stack(
        [y_event[global_nodes, origins + step] for step in range(horizon)],
        axis=1,
    ).astype(np.int8, copy=False)
    return np.ascontiguousarray(x_static[global_nodes]), np.ascontiguousarray(targets)


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
) -> tuple[dict, dict[str, dict]]:
    """Evaluate static or static/history fused scores with borough-stratified Top-k."""
    pooled = init_det_eval_state()
    local_states = {name: init_det_eval_state() for name in borough_indices}
    for name, indices in borough_indices.items():
        static_z = zscore_columns(static_scores[name])
        for start in starts:
            origin = int(start) + int(history_len)
            labels = y_event[indices, origin:origin + horizon].astype(np.float32, copy=False)
            if history_frequency is None:
                score = static_z
            else:
                history_z = zscore_columns(history_frequency[indices, origin])
                score = static_weight * static_z + (1.0 - static_weight) * history_z
            for state in (local_states[name], pooled):
                update_det_eval_state(
                    state,
                    labels[None, :, :],
                    labels[None, :, :],
                    np.zeros_like(labels)[None, :, :],
                    hr_ratios,
                    score=score[None, :, :],
                )
    return _metric_dict(pooled), {name: _metric_dict(state) for name, state in local_states.items()}


def fit_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    max_iter: int,
    seed: int,
) -> list[LogisticRegression]:
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


def predict_static(models: list[LogisticRegression], features: np.ndarray) -> np.ndarray:
    return np.stack([model.predict_proba(features)[:, 1] for model in models], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--start_date", default="2021-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--lookbacks", default="14,30")
    parser.add_argument("--c_values", default="0.01,0.1,1,10")
    parser.add_argument("--static_weights", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--hr_ratio", default="0.05,0.10,0.20")
    parser.add_argument("--selection_metric", default="es_ndcg_at_20")
    parser.add_argument("--train_max_samples", type=int, default=200000)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="analysis_results/static_logistic_history_fusion_gate.json")
    parser.add_argument("--artifact", default="")
    args = parser.parse_args()

    lookbacks = parse_number_list(args.lookbacks, int)
    c_values = parse_number_list(args.c_values, float)
    static_weights = parse_number_list(args.static_weights, float)
    hr_ratios = parse_hr_ratios(args.hr_ratio)
    if any(weight < 0 or weight > 1 for weight in static_weights):
        raise ValueError("static_weights must lie in [0, 1]")

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
        f"Static logistic training samples={x_train.shape[0]} features={x_train.shape[1]} "
        f"positive_rates={y_train.mean(axis=0).tolist()}"
    )

    validation_by_c = {}
    model_candidates = {}
    score_candidates = {}
    for c_value in c_values:
        label = f"C={c_value:g}"
        models = fit_models(x_train_scaled, y_train, c_value, args.max_iter, args.seed)
        scores = {
            name: predict_static(models, feature_scaler.transform(x_static[indices]))
            for name, indices in borough_indices.items()
        }
        metrics, _ = evaluate_scores(
            y_event,
            borough_indices,
            val_starts,
            args.history_len,
            args.horizon,
            hr_ratios,
            scores,
        )
        validation_by_c[label] = metrics
        model_candidates[label] = models
        score_candidates[label] = scores
        print(f"{label} val R20/N20={metrics['es_recall_at_20']:.4f}/{metrics['es_ndcg_at_20']:.4f}")
    selected_c = max(
        validation_by_c,
        key=lambda label: (validation_by_c[label][args.selection_metric], -c_values.index(float(label[2:]))),
    )
    models = model_candidates[selected_c]
    static_scores = score_candidates[selected_c]
    static_test, static_test_by_borough = evaluate_scores(
        y_event,
        borough_indices,
        test_starts,
        args.history_len,
        args.horizon,
        hr_ratios,
        static_scores,
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
            validation[label], _ = evaluate_scores(
                y_event,
                borough_indices,
                val_starts,
                args.history_len,
                args.horizon,
                hr_ratios,
                static_scores,
                history_frequency,
                weight,
            )
        selected_weight = max(
            validation,
            key=lambda label: (
                validation[label][args.selection_metric],
                -static_weights.index(float(label.split("=")[1])),
            ),
        )
        weight = float(selected_weight.split("=")[1])
        test_metrics, test_by_borough = evaluate_scores(
            y_event,
            borough_indices,
            test_starts,
            args.history_len,
            args.horizon,
            hr_ratios,
            static_scores,
            history_frequency,
            weight,
        )
        fusion_results[str(lookback)] = {
            "validation_candidates": validation,
            "selected": selected_weight,
            "test": test_metrics,
            "test_by_borough": test_by_borough,
        }
        print(
            f"L={lookback or 'full'} {selected_weight} "
            f"test R20/N20={test_metrics['es_recall_at_20']:.4f}/{test_metrics['es_ndcg_at_20']:.4f}"
        )
        del history_frequency

    artifact_path = ""
    if args.artifact:
        artifact = Path(args.artifact)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"models": models, "feature_scaler": feature_scaler, "selected_c": selected_c},
            artifact,
        )
        artifact_path = str(artifact)

    document = {
        "schema_version": 1,
        "question": "Can static built-environment propensity plus limited crash history match learned spatiotemporal models?",
        "protocol": {
            "date_range": [str(dates[0].date()), str(dates[-1].date())],
            "boroughs": borough_names,
            "evaluation": "borough-stratified Top-k per forecast window, pooled hits/denominators",
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
            "c_values": c_values,
            "selected": selected_c,
            "selected_iterations_by_horizon": [int(model.n_iter_[0]) for model in models],
            "selected_converged": all(int(model.n_iter_[0]) < args.max_iter for model in models),
            "max_iter": args.max_iter,
            "validation_candidates": validation_by_c,
        },
        "static_only_test": static_test,
        "static_only_test_by_borough": static_test_by_borough,
        "fusion_by_L": fusion_results,
        "artifact_path": artifact_path,
        "wall_seconds": time.perf_counter() - started_at,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
