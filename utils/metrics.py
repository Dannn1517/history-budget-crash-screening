from __future__ import annotations

import numpy as np


def init_metric_acc() -> dict:
    return {
        name: {"sum": 0.0, "count": 0.0}
        for name in [
            "mae_all", "rmse_all",
            "mae_pos", "rmse_pos",
            "zr",
            "picp_all", "mpiw_all",
            "picp_pos", "mpiw_pos",
            "is_all", "is_pos",
            "precision", "recall", "map", "ndcg", "ndcg_g",
        ]
    }


def finalize_metrics(acc: dict) -> dict:
    metrics = {}
    for name, agg in acc.items():
        if agg["count"] > 0:
            metrics[name] = agg["sum"] / agg["count"]
        else:
            metrics[name] = float("nan")
    if acc["rmse_all"]["count"] > 0:
        metrics["rmse_all"] = float(np.sqrt(acc["rmse_all"]["sum"] / acc["rmse_all"]["count"]))
    if acc["rmse_pos"]["count"] > 0:
        metrics["rmse_pos"] = float(np.sqrt(acc["rmse_pos"]["sum"] / acc["rmse_pos"]["count"]))
    return metrics


def target_only_selection_rate(
        y_target: np.ndarray,
        y_event: np.ndarray,
        score: np.ndarray,
        top_frac: float,
) -> float:
    if top_frac <= 0.0 or top_frac > 1.0:
        raise ValueError("top_frac must be in the range (0, 1].")
    target = np.asarray(y_target).reshape(-1)
    event = np.asarray(y_event).reshape(-1)
    ranking_score = np.asarray(score).reshape(-1)
    if target.shape != event.shape or target.shape != ranking_score.shape:
        raise ValueError("y_target, y_event, and score must have the same flattened shape.")
    if ranking_score.size == 0:
        return float("nan")

    k = max(1, int(np.ceil(float(top_frac) * ranking_score.size)))
    top_indices = np.argsort(-ranking_score, kind="mergesort")[:k]
    target_only = (target[top_indices] > 0) & (event[top_indices] <= 0)
    return float(np.mean(target_only))
