from __future__ import annotations

from typing import Tuple

import numpy as np


def standardize_time_features(calendar: np.ndarray, weather: np.ndarray, train_end: int) -> Tuple[np.ndarray, np.ndarray]:
    cal_mean = calendar[:train_end].mean(axis=0)
    cal_std = calendar[:train_end].std(axis=0)
    cal_std = np.where(cal_std == 0, 1.0, cal_std)
    calendar = (calendar - cal_mean) / cal_std

    w_mean = weather[:, :train_end, :].mean(axis=(0, 1))
    w_std = weather[:, :train_end, :].std(axis=(0, 1))
    w_std = np.where(w_std == 0, 1.0, w_std)
    weather = (weather - w_mean) / w_std
    return calendar, weather


def standardize_node_time_feature(
        X: np.ndarray,
        train_end: int,
        fit_node_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, float, float]:
    fit_values = X[:, :train_end]
    if fit_node_mask is not None:
        mask = np.asarray(fit_node_mask, dtype=bool)
        if mask.ndim != 1 or mask.shape[0] != X.shape[0]:
            raise ValueError("fit_node_mask must have one flag per node.")
        if not bool(mask.any()):
            raise ValueError("fit_node_mask must retain at least one node.")
        fit_values = fit_values[mask]
    mean = float(fit_values.mean())
    std = float(fit_values.std())
    if std == 0.0:
        std = 1.0
    X = (X - mean) / std
    return X.astype(np.float32), mean, std


def standardize_freq_variants(
        variants: list,
        train_end: int,
        shared_stats: bool = False,
        ref_index: int = -1,
        fit_node_mask: np.ndarray | None = None,
) -> Tuple[list, list]:
    """Standardize a list of [N, T] frequency variants.

    Per-variant stats (default) equalize first-order statistics but make the
    zero-mass position budget-dependent, letting a model implicitly identify
    the lookback budget. ``shared_stats=True`` instead standardizes every
    variant with the reference variant's train statistics (default: the last
    one, conventionally full history), so channel values keep one physical
    meaning across budgets and budget is not inferable from the zero position.
    """
    if not variants:
        return [], []
    if not shared_stats:
        stats = []
        out = []
        for v in variants:
            arr, mean, std = standardize_node_time_feature(
                v.copy(), train_end, fit_node_mask=fit_node_mask
            )
            out.append(arr)
            stats.append({"mean": mean, "std": std})
        return out, stats
    ref = variants[ref_index]
    _, ref_mean, ref_std = standardize_node_time_feature(
        ref.copy(), train_end, fit_node_mask=fit_node_mask
    )
    if ref_std == 0.0:
        ref_std = 1.0
    out = [((v - ref_mean) / ref_std).astype(np.float32) for v in variants]
    stats = [{"mean": ref_mean, "std": ref_std, "shared": True} for _ in variants]
    return out, stats
