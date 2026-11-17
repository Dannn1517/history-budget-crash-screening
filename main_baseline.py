from __future__ import division

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import gamma as gamma_dist, nbinom, norm

from utils.data import (
    build_baseline_bundle,
    build_spatial_block_holdout,
    build_time_splits,
    build_time_splits_by_idx,
    load_borough_name_order,
    load_mask,
    load_nyc_zinb_data,
    load_static_column_names,
    load_utm_centroids,
    load_expansion_features,
    parse_expansion_groups,
    parse_date_arg,
    set_seed,
    standardize_time_features,
    static_zero_indices,
)
from utils.data.priors import parse_lookback_mix
from utils.metrics import finalize_metrics, init_metric_acc
from utils.tasks import normalize_hist_source_mode, normalize_target_mode, should_warn_distribution_mismatch, uses_diffused_target
from models.deterministic_baselines import DCRNNBaseline, STGATBaseline, STGCNBaseline
from models.probabilistic_baselines import (
    STEventOccurrenceBaseline,
    STGaussianGNNBaseline,
    STGammaBaseline,
    STDAERHurdleGammaBaseline,
    STDAERHurdleLogNormalBaseline,
    STHurdleGammaBaseline,
    STHurdleLogNormalBaseline,
    STHurdleTweedieBaseline,
    STLogNormalBaseline,
    STNBGNNBaseline,
    STTDBaseline,
    STZINBBaseline,
    STZITDBaseline,
    gamma_mean,
    gamma_nll_loss,
    daer_hurdle_gamma_loss,
    daer_hurdle_lognormal_loss,
    event_aligned_hurdle_gamma_loss,
    event_aligned_hurdle_lognormal_loss,
    cumulative_hazard_nll_loss,
    event_occurrence_loss,
    gaussian_nll_loss,
    hurdle_gamma_mean,
    hurdle_gamma_nll_loss,
    hurdle_lognormal_mean,
    hurdle_lognormal_nll_loss,
    hurdle_tweedie_conditional_nll_loss,
    hurdle_tweedie_mean,
    lognormal_mean,
    lognormal_nll_loss,
    negative_binomial_nll_loss,
    negative_binomial_mean,
    nb_zero_prob_mu_theta,
    sample_hurdle_gamma,
    sample_hurdle_lognormal,
    tweedie_interval_from_params_mc,
    tweedie_mean,
    tweedie_nll_loss,
    tweedie_p0_from_params,
    zero_inflated_tweedie_interval_from_params_mc,
    zero_inflated_tweedie_mean,
    zero_inflated_tweedie_nll_loss,
    zero_inflated_tweedie_p0_from_params,
    zero_inflated_nb_mean,
    zero_inflated_nb_nll_loss,
)
from utils.utils import (
    interval_score,
    interval_score_conditional_ypos,
    mpiw,
    picp,
    picp_conditional_ypos,
    zinb_p0_from_params,
    zinb_interval_LU_from_params_scipy,
)

SHANGHAI_TZ = timezone(timedelta(hours=8))
DEFAULT_HR_RATIOS = (0.05, 0.10, 0.20)
PRIMARY_HR_RATIO = 0.20
DAER_MODEL_KINDS = {"daer_hurdle_gamma", "daer_hurdle_lognormal"}
EVENT_ALIGNED_MODEL_KINDS = {"event_aligned_hurdle_gamma", "event_aligned_hurdle_lognormal"}
COLOR_OUTPUT_ENABLED = False

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_BLUE = "\033[94m"
ANSI_CYAN = "\033[96m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_MAGENTA = "\033[95m"


def configure_color_output(mode: str):
    global COLOR_OUTPUT_ENABLED
    if mode == "always":
        COLOR_OUTPUT_ENABLED = True
    elif mode == "never":
        COLOR_OUTPUT_ENABLED = False
    else:
        COLOR_OUTPUT_ENABLED = bool(sys.stdout.isatty() and "NO_COLOR" not in os.environ)


def color_text(text: str, color: str = "", bold: bool = False) -> str:
    if not COLOR_OUTPUT_ENABLED:
        return text
    prefix = f"{ANSI_BOLD if bold else ''}{color}"
    return f"{prefix}{text}{ANSI_RESET}"


def print_status(text: str, color: str = "", bold: bool = False, **kwargs):
    print(color_text(text, color=color, bold=bold), **kwargs)

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def parse_hr_ratios(value) -> List[float]:
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw_parts = list(value)
    else:
        raw_parts = [value]

    ratios: List[float] = []
    for raw in raw_parts:
        ratio = float(raw)
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError("--hr_ratio values must be in the range (0, 1].")
        if not any(np.isclose(ratio, existing, rtol=0.0, atol=1e-12) for existing in ratios):
            ratios.append(ratio)
    if not ratios:
        raise ValueError("--hr_ratio must contain at least one value.")
    return ratios


def parse_cumulative_horizons(value) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        raw_parts = list(value)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw_parts = [part.strip() for part in text.split(",") if part.strip()]

    horizons: List[int] = []
    for raw in raw_parts:
        if isinstance(raw, bool) or (
                not isinstance(raw, (str, int, np.integer))
        ):
            raise ValueError(f"cumulative horizons must be integers, got {raw!r}.")
        try:
            horizon = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cumulative horizons must be integers, got {raw!r}.") from exc
        if horizon < 1:
            raise ValueError("cumulative horizons must be positive integers.")
        horizons.append(horizon)
    for left, right in zip(horizons, horizons[1:]):
        if right <= left:
            raise ValueError("cumulative horizons must be strictly increasing and unique.")
    return tuple(horizons)


def primary_hr_ratio(hr_ratios: List[float]) -> float:
    for ratio in hr_ratios:
        if np.isclose(ratio, PRIMARY_HR_RATIO, rtol=0.0, atol=1e-12):
            return float(ratio)
    return float(hr_ratios[-1])


def hr_ratio_suffix(ratio: float) -> str:
    percent = ratio * 100.0
    percent_text = f"{percent:.6g}".replace(".", "p")
    return f"at_{percent_text}"


def hr_ratio_label(ratio: float) -> str:
    return f"{ratio * 100.0:.6g}%"


def hr_ratios_from_metrics(metrics: Dict[str, float]) -> List[float]:
    ratios = []
    for prefix in ("ts_recall_at_", "es_recall_at_"):
        for key in metrics:
            if not key.startswith(prefix):
                continue
            percent_text = key[len(prefix):].replace("p", ".")
            try:
                ratio = float(percent_text) / 100.0
            except ValueError:
                continue
            if ratio not in ratios:
                ratios.append(ratio)
    if ratios:
        return sorted(ratios)
    if "ts_recall" in metrics or "es_recall" in metrics:
        return [PRIMARY_HR_RATIO]
    return list(DEFAULT_HR_RATIOS)


def ranking_metric_key(space: str, metric: str, ratio: float, metrics: Dict[str, float]) -> str:
    key = f"{space}_{metric}_{hr_ratio_suffix(ratio)}"
    if key in metrics:
        return key
    legacy_key = f"{space}_{metric}"
    if np.isclose(ratio, PRIMARY_HR_RATIO, rtol=0.0, atol=1e-12) and legacy_key in metrics:
        return legacy_key
    return key


def format_metrics_with_na(prefix: str, metrics: Dict[str, float]) -> str:
    order = [
        ("MAE_all", "mae_all"),
        ("RMSE_all", "rmse_all"),
        ("MAE_pos", "mae_pos"),
        ("RMSE_pos", "rmse_pos"),
    ]
    for ratio in hr_ratios_from_metrics(metrics):
        label = hr_ratio_label(ratio)
        order.extend([
            (f"TS_Recall@{label}", ranking_metric_key("ts", "recall", ratio, metrics)),
            (f"ES_Recall@{label}", ranking_metric_key("es", "recall", ratio, metrics)),
            (f"TS_NDCG@{label}", ranking_metric_key("ts", "ndcg", ratio, metrics)),
            (f"ES_NDCG@{label}", ranking_metric_key("es", "ndcg", ratio, metrics)),
            (f"TargetOnly@{label}", f"target_only_rate_{hr_ratio_suffix(ratio)}"),
        ])
    order.extend([
        ("PICP_all", "picp_all"),
        ("MPIW_all", "mpiw_all"),
        ("PICP_pos", "picp_pos"),
        ("MPIW_pos", "mpiw_pos"),
        ("IS_all", "is_all"),
        ("IS_pos", "is_pos"),
    ])
    parts = [color_text(prefix, ANSI_GREEN, bold=True)]
    for label, key in order:
        value = metrics.get(key, float("nan"))
        metric_text = f"{label}={'N/A' if not np.isfinite(value) else f'{value:.4f}'}"
        if label.startswith(("MAE", "RMSE")):
            color = ANSI_CYAN
        elif label.startswith("TS_"):
            color = ANSI_BLUE
        elif label.startswith("ES_"):
            color = ANSI_GREEN
        elif label.startswith("TargetOnly"):
            color = ANSI_YELLOW
        else:
            color = ANSI_MAGENTA
        parts.append(color_text(metric_text, color))
    return " | ".join(parts)


def checkpoint_selection_score(metrics: Dict[str, float], selection_metric: str, epoch: int) -> float:
    if selection_metric == "last":
        return float(epoch)
    metric_key = {
        "es_recall": "es_recall",
        "es_ndcg": "es_ndcg",
        "ts_recall": "ts_recall",
        "ts_ndcg": "ts_ndcg",
        "mae_all": "mae_all",
    }[selection_metric]
    value = float(metrics.get(metric_key, float("nan")))
    if not np.isfinite(value):
        return float("-inf")
    return -value if selection_metric == "mae_all" else value


def init_det_eval_state() -> dict:
    return {
        "acc": init_metric_acc(),
        "hr_ratios": None,
        "recall_hits": None,
        "recall_denoms": None,
        "ndcg_sums": None,
        "ndcg_counts": None,
        "target_only_sums": None,
        "target_only_counts": None,
    }


def seen_node_mask_for_evaluation(batch: dict, seen_nodes_only: bool) -> torch.Tensor | None:
    """Return the shared per-batch mask used for leakage-free validation.

    Node-held-out datasets emit a constant ``node_weight`` vector for every
    sample. Training already uses it as a loss mask; checkpoint selection must
    use the same seen-node population. Final test evaluation intentionally
    keeps all nodes unless the caller explicitly requests this mask.
    """
    if not seen_nodes_only or "node_weight" not in batch:
        return None
    weights = batch["node_weight"]
    if weights.ndim == 1:
        mask = weights > 0
    elif weights.ndim == 2:
        if not bool(torch.all(weights == weights[:1])):
            raise ValueError("node_weight must be identical for every sample in an evaluation batch.")
        mask = weights[0] > 0
    else:
        raise ValueError("node_weight must have shape [nodes] or [batch, nodes].")
    if not bool(mask.any()):
        raise ValueError("Seen-node validation requires at least one unmasked node.")
    return mask


def build_rank_orders(score: np.ndarray) -> Tuple[np.ndarray, ...]:
    score_array = np.asarray(score)
    if score_array.ndim != 3:
        raise ValueError("Ranking score must have shape [batch, nodes, horizon].")
    return tuple(
        np.argsort(
            -np.asarray(score_array[:, :, horizon_idx]).reshape(-1),
            kind="mergesort",
        )
        for horizon_idx in range(score_array.shape[-1])
    )


def update_det_eval_state(
        state: dict,
        y_true: np.ndarray,
        y_rank: np.ndarray,
        pred: np.ndarray,
        hr_ratio,
        score: Optional[np.ndarray] = None,
        rank_orders: Optional[Tuple[np.ndarray, ...]] = None,
):
    acc = state["acc"]
    count = y_true.size
    err = y_true - pred
    acc["mae_all"]["sum"] += float(np.sum(np.abs(err)))
    acc["mae_all"]["count"] += count
    acc["rmse_all"]["sum"] += float(np.sum(err ** 2))
    acc["rmse_all"]["count"] += count

    pos_mask = y_true > 0
    pos_count = int(np.sum(pos_mask))
    if pos_count > 0:
        err_pos = err[pos_mask]
        acc["mae_pos"]["sum"] += float(np.sum(np.abs(err_pos)))
        acc["mae_pos"]["count"] += pos_count
        acc["rmse_pos"]["sum"] += float(np.sum(err_pos ** 2))
        acc["rmse_pos"]["count"] += pos_count

    horizon = y_rank.shape[-1]
    score_for_rank = pred if score is None else score
    if rank_orders is None:
        rank_orders = build_rank_orders(score_for_rank)
    if len(rank_orders) != horizon:
        raise ValueError("rank_orders horizon does not match y_rank.")
    hr_ratios = parse_hr_ratios(hr_ratio)
    if state["recall_hits"] is None:
        state["hr_ratios"] = hr_ratios
        shape = (len(hr_ratios), horizon)
        state["recall_hits"] = np.zeros(shape, dtype=np.float64)
        state["recall_denoms"] = np.zeros(shape, dtype=np.float64)
        state["ndcg_sums"] = np.zeros(shape, dtype=np.float64)
        state["ndcg_counts"] = np.zeros(shape, dtype=np.float64)
    elif len(hr_ratios) != len(state["hr_ratios"]) or any(
            not np.isclose(left, right, rtol=0.0, atol=1e-12)
            for left, right in zip(hr_ratios, state["hr_ratios"])
    ):
        raise ValueError("hr_ratio values must stay unchanged within one evaluation state.")

    for t in range(horizon):
        y_t = y_rank[:, :, t]
        y_flat = np.asarray(y_t).reshape(-1)
        event = y_flat > 0
        total_pos = int(np.sum(event))
        order_full = np.asarray(rank_orders[t])
        if order_full.size != y_flat.size:
            raise ValueError("rank_orders size does not match the flattened ranking target.")
        max_k = max(max(1, int(np.ceil(float(ratio) * y_flat.size))) for ratio in hr_ratios)
        discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
        discount_prefix = np.cumsum(discounts)
        for ratio_idx, ratio in enumerate(hr_ratios):
            k = max(1, int(np.ceil(float(ratio) * y_flat.size)))
            order = order_full[:k]
            if total_pos > 0:
                hits = int(np.sum(event[order]))
                state["recall_hits"][ratio_idx, t] += hits
                state["recall_denoms"][ratio_idx, t] += total_pos

                relevance = event[order].astype(np.float64)
                dcg = float(np.sum(relevance * discounts[:k]))
                ideal_count = min(total_pos, k)
                idcg = float(discount_prefix[ideal_count - 1])
                ndcg_val = dcg / idcg
                state["ndcg_sums"][ratio_idx, t] += ndcg_val
                state["ndcg_counts"][ratio_idx, t] += 1


def update_target_only_eval_state(
        state: dict,
        y_target: np.ndarray,
        y_event: np.ndarray,
        score: np.ndarray,
        hr_ratio,
        rank_orders: Optional[Tuple[np.ndarray, ...]] = None,
):
    hr_ratios = parse_hr_ratios(hr_ratio)
    horizon = y_target.shape[-1]
    if rank_orders is None:
        rank_orders = build_rank_orders(score)
    if len(rank_orders) != horizon:
        raise ValueError("rank_orders horizon does not match target-only labels.")
    if state["target_only_sums"] is None:
        state["target_only_sums"] = np.zeros((len(hr_ratios), horizon), dtype=np.float64)
        state["target_only_counts"] = np.zeros((len(hr_ratios), horizon), dtype=np.float64)

    for horizon_idx in range(horizon):
        target_step = y_target[:, :, horizon_idx]
        event_step = y_event[:, :, horizon_idx]
        target_flat = np.asarray(target_step).reshape(-1)
        event_flat = np.asarray(event_step).reshape(-1)
        order_full = np.asarray(rank_orders[horizon_idx])
        if order_full.size != target_flat.size or target_flat.size != event_flat.size:
            raise ValueError("rank_orders size does not match target-only labels.")
        for ratio_idx, ratio in enumerate(hr_ratios):
            k = max(1, int(np.ceil(float(ratio) * target_flat.size)))
            top_indices = order_full[:k]
            target_only = (target_flat[top_indices] > 0) & (event_flat[top_indices] <= 0)
            state["target_only_sums"][ratio_idx, horizon_idx] += float(np.mean(target_only))
            state["target_only_counts"][ratio_idx, horizon_idx] += 1


def finalize_det_eval_state(state: dict) -> dict:
    metrics = finalize_metrics(state["acc"])
    for retired_key in ("zr", "precision", "map", "ndcg_g"):
        metrics.pop(retired_key, None)
    hr_ratios = state.get("hr_ratios") or [PRIMARY_HR_RATIO]
    if state["recall_hits"] is not None:
        for ratio_idx, ratio in enumerate(hr_ratios):
            recall_t = np.divide(
                state["recall_hits"][ratio_idx],
                state["recall_denoms"][ratio_idx],
                out=np.full_like(state["recall_hits"][ratio_idx], np.nan),
                where=state["recall_denoms"][ratio_idx] > 0,
            )
            metrics[f"recall_{hr_ratio_suffix(ratio)}"] = float(np.nanmean(recall_t))
    if state["ndcg_sums"] is not None:
        for ratio_idx, ratio in enumerate(hr_ratios):
            ndcg_t = np.divide(
                state["ndcg_sums"][ratio_idx],
                state["ndcg_counts"][ratio_idx],
                out=np.full_like(state["ndcg_sums"][ratio_idx], np.nan),
                where=state["ndcg_counts"][ratio_idx] > 0,
            )
            metrics[f"ndcg_{hr_ratio_suffix(ratio)}"] = float(np.nanmean(ndcg_t))
    if state["target_only_sums"] is not None:
        for ratio_idx, ratio in enumerate(hr_ratios):
            target_only_t = np.divide(
                state["target_only_sums"][ratio_idx],
                state["target_only_counts"][ratio_idx],
                out=np.full_like(state["target_only_sums"][ratio_idx], np.nan),
                where=state["target_only_counts"][ratio_idx] > 0,
            )
            metrics[f"target_only_rate_{hr_ratio_suffix(ratio)}"] = float(np.nanmean(target_only_t))

    main_ratio = primary_hr_ratio(hr_ratios)
    main_suffix = hr_ratio_suffix(main_ratio)
    metrics["recall"] = float(metrics.get(f"recall_{main_suffix}", float("nan")))
    metrics["ndcg"] = float(metrics.get(f"ndcg_{main_suffix}", float("nan")))
    return metrics


def point_zero_score_from_mean(mean_pred: np.ndarray) -> np.ndarray:
    mean_pred = np.clip(np.asarray(mean_pred, dtype=np.float64), a_min=0.0, a_max=None)
    return 1.0 / (1.0 + mean_pred)


def is_space_metric_key(key: str) -> bool:
    return key in {"recall", "ndcg"} or key.startswith("recall_at_") or key.startswith("ndcg_at_")


def merge_space_metrics(target_metrics: dict, event_metrics: dict) -> dict:
    merged = dict(target_metrics)
    for key in sorted({key for key in target_metrics.keys() if is_space_metric_key(key)}):
        merged[f"ts_{key}"] = float(target_metrics.get(key, float("nan")))
        merged[f"es_{key}"] = float(event_metrics.get(key, float("nan")))
    return merged


def build_point_rank_score(rank_score: str, mean_pred: np.ndarray) -> np.ndarray:
    if rank_score == "mean":
        return np.asarray(mean_pred, dtype=np.float64)
    if rank_score == "p_nonzero":
        zero_score = point_zero_score_from_mean(mean_pred)
        return 1.0 - zero_score
    raise ValueError(f"Unsupported rank_score: {rank_score}")


def build_prob_rank_score(
        rank_score: str,
        model_kind: str,
        mean_pred: np.ndarray,
        q_np: Optional[np.ndarray] = None,
        n_np: Optional[np.ndarray] = None,
        p_np: Optional[np.ndarray] = None,
        pi_np: Optional[np.ndarray] = None,
        mu_np: Optional[np.ndarray] = None,
        alpha_np: Optional[np.ndarray] = None,
        phi_np: Optional[np.ndarray] = None,
        rho_np: Optional[np.ndarray] = None,
        loc_np: Optional[np.ndarray] = None,
        scale_np: Optional[np.ndarray] = None,
) -> np.ndarray:
    if rank_score == "mean":
        return np.asarray(mean_pred, dtype=np.float64)

    if rank_score != "p_nonzero":
        raise ValueError(f"Unsupported rank_score: {rank_score}")

    if model_kind == "gamma":
        raise ValueError("st_gamma only supports rank_score=mean.")
    if model_kind == "lognormal":
        raise ValueError("st_lognormal only supports rank_score=mean.")
    if model_kind == "zinb":
        return 1.0 - zinb_p0_from_params(n_np, p_np, pi_np)
    if model_kind == "nb":
        n_np = np.clip(np.asarray(n_np, dtype=np.float64), a_min=1e-8, a_max=None)
        p_np = np.clip(np.asarray(p_np, dtype=np.float64), a_min=1e-8, a_max=1.0 - 1e-8)
        return 1.0 - np.power(p_np, n_np)
    if model_kind == "gaussian":
        loc_np = np.asarray(loc_np, dtype=np.float64)
        scale_np = np.clip(np.asarray(scale_np, dtype=np.float64), a_min=1e-8, a_max=None)
        return 1.0 - norm.cdf(0.0, loc=loc_np, scale=scale_np)
    if model_kind == "hurdle_gamma":
        return np.clip(np.asarray(q_np, dtype=np.float64), a_min=0.0, a_max=1.0)
    if model_kind == "hurdle_lognormal":
        return np.clip(np.asarray(q_np, dtype=np.float64), a_min=0.0, a_max=1.0)
    if model_kind == "hurdle_tweedie":
        return np.clip(np.asarray(q_np, dtype=np.float64), a_min=0.0, a_max=1.0)
    if model_kind == "tweedie":
        return 1.0 - tweedie_p0_from_params(mu_np, phi_np, rho_np)
    if model_kind == "zitd":
        return 1.0 - zero_inflated_tweedie_p0_from_params(pi_np, mu_np, phi_np, rho_np)

    raise ValueError(f"Unsupported probabilistic model kind for rank score: {model_kind}")


def update_prob_eval_state(
        state: dict,
        y_true: np.ndarray,
        y_rank: np.ndarray,
        pred_mean: np.ndarray,
        hr_ratio: float,
        interval_alpha: float,
        score: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        rank_orders: Optional[Tuple[np.ndarray, ...]] = None,
):
    update_det_eval_state(
        state,
        y_true,
        y_rank,
        pred_mean,
        hr_ratio,
        score=score,
        rank_orders=rank_orders,
    )

    acc = state["acc"]
    count = y_true.size
    pos_mask = y_true > 0
    pos_count = int(np.sum(pos_mask))

    if lower is None or upper is None:
        return

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    mpiw_val = mpiw(lower, upper)
    acc["mpiw_all"]["sum"] += mpiw_val * count
    acc["mpiw_all"]["count"] += count

    picp_val = picp(y_true, lower, upper)
    acc["picp_all"]["sum"] += picp_val * count
    acc["picp_all"]["count"] += count

    is_all_val = interval_score(y_true, lower, upper, alpha=interval_alpha)
    acc["is_all"]["sum"] += is_all_val * count
    acc["is_all"]["count"] += count

    if pos_count > 0:
        picp_pos = picp_conditional_ypos(y_true, lower, upper)
        if np.isfinite(picp_pos):
            acc["picp_pos"]["sum"] += picp_pos * pos_count
            acc["picp_pos"]["count"] += pos_count

        width_pos = (upper - lower)[pos_mask]
        if width_pos.size > 0:
            acc["mpiw_pos"]["sum"] += float(np.mean(width_pos)) * pos_count
            acc["mpiw_pos"]["count"] += pos_count

        is_pos_val = interval_score_conditional_ypos(y_true, lower, upper, alpha=interval_alpha)
        if np.isfinite(is_pos_val):
            acc["is_pos"]["sum"] += is_pos_val * pos_count
            acc["is_pos"]["count"] += pos_count


def update_deterministic_space_eval_states(
        target_state: dict,
        event_state: dict,
        y_target: np.ndarray,
        y_target_rank: np.ndarray,
        y_event: np.ndarray,
        y_event_rank: np.ndarray,
        pred: np.ndarray,
        hr_ratio,
        score: np.ndarray,
):
    rank_orders = build_rank_orders(score)
    update_det_eval_state(
        target_state,
        y_target,
        y_target_rank,
        pred,
        hr_ratio,
        score=score,
        rank_orders=rank_orders,
    )
    update_det_eval_state(
        event_state,
        y_event,
        y_event_rank,
        pred,
        hr_ratio,
        score=score,
        rank_orders=rank_orders,
    )
    update_target_only_eval_state(
        target_state,
        y_target,
        y_event,
        score,
        hr_ratio,
        rank_orders=rank_orders,
    )


def update_probabilistic_space_eval_states(
        target_state: dict,
        event_state: dict,
        y_target: np.ndarray,
        y_target_rank: np.ndarray,
        y_event: np.ndarray,
        y_event_rank: np.ndarray,
        pred_mean: np.ndarray,
        hr_ratio,
        interval_alpha: float,
        target_score: np.ndarray,
        event_score: Optional[np.ndarray] = None,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
):
    if event_score is None:
        event_score = target_score
    target_orders = build_rank_orders(target_score)
    event_orders = target_orders if event_score is target_score else build_rank_orders(event_score)
    update_prob_eval_state(
        target_state,
        y_target,
        y_target_rank,
        pred_mean,
        hr_ratio,
        interval_alpha=interval_alpha,
        score=target_score,
        lower=lower,
        upper=upper,
        rank_orders=target_orders,
    )
    update_det_eval_state(
        event_state,
        y_event,
        y_event_rank,
        pred_mean,
        hr_ratio,
        score=event_score,
        rank_orders=event_orders,
    )
    update_target_only_eval_state(
        target_state,
        y_target,
        y_event,
        event_score,
        hr_ratio,
        rank_orders=event_orders,
    )


def evaluate_prediction_groups(
        groups: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        hr_ratio: float,
        rank_score: str,
        zr_tau: float,
) -> dict:
    target_state = init_det_eval_state()
    event_state = init_det_eval_state()
    for y_true, y_rank, y_event, y_event_rank, pred in groups:
        arrays = tuple(np.asarray(value) for value in (y_true, y_rank, y_event, y_event_rank, pred))
        if any(value.ndim != 3 for value in arrays):
            shapes = [value.shape for value in arrays]
            raise ValueError(f"Statistical prediction groups must have shape [windows, nodes, horizon]: {shapes}")
        if len({value.shape for value in arrays}) != 1:
            raise ValueError("Statistical prediction group arrays must have identical shapes.")
        # Each forecast window has its own road-selection budget. Flattening the
        # window axis would incorrectly create one global Top-k across dates.
        for window_idx in range(pred.shape[0]):
            window = tuple(value[window_idx:window_idx + 1] for value in arrays)
            score = build_point_rank_score(rank_score, window[-1])
            update_deterministic_space_eval_states(
                target_state,
                event_state,
                window[0],
                window[1],
                window[2],
                window[3],
                window[4],
                hr_ratio,
                score,
            )
    return merge_space_metrics(finalize_det_eval_state(target_state), finalize_det_eval_state(event_state))


@torch.no_grad()
def evaluate_graph_model_by_borough(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        device: torch.device,
        hr_ratio: float,
        use_priors: bool,
        rank_score: str,
        zr_tau: float,
) -> Dict[str, dict]:
    borough_metrics: Dict[str, dict] = {}
    for bundle in bundles:
        borough_metrics[bundle["name"]] = evaluate_graph_model(
            model=model,
            bundles=[bundle],
            split=split,
            device=device,
            hr_ratio=hr_ratio,
            use_priors=use_priors,
            rank_score=rank_score,
            zr_tau=zr_tau,
        )
    return borough_metrics


def gaussian_interval_from_params(loc: np.ndarray, scale: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    z_value = float(norm.ppf(1.0 - alpha / 2.0))
    lower = np.clip(loc - z_value * scale, a_min=0.0, a_max=None)
    upper = np.clip(loc + z_value * scale, a_min=0.0, a_max=None)
    return lower, upper


def nb_interval_from_params(n: np.ndarray, p: np.ndarray, alpha: float, eps: float = 1e-6) -> Tuple[
    np.ndarray, np.ndarray]:
    n = np.clip(np.asarray(n, dtype=np.float64), eps, None)
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    lower = nbinom.ppf(alpha / 2.0, n, p)
    upper = nbinom.ppf(1.0 - alpha / 2.0, n, p)
    lower = np.clip(np.nan_to_num(lower, nan=0.0, posinf=0.0), a_min=0.0, a_max=None)
    upper = np.clip(np.nan_to_num(upper, nan=0.0, posinf=0.0), a_min=0.0, a_max=None)
    return lower, upper


def gamma_interval_from_params(
        mu: np.ndarray,
        alpha_param: np.ndarray,
        alpha: float,
        eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.clip(np.asarray(mu, dtype=np.float64), eps, None)
    alpha_param = np.clip(np.asarray(alpha_param, dtype=np.float64), eps, None)
    scale = mu / alpha_param
    lower = gamma_dist.ppf(alpha / 2.0, a=alpha_param, scale=scale)
    upper = gamma_dist.ppf(1.0 - alpha / 2.0, a=alpha_param, scale=scale)
    lower = np.clip(np.nan_to_num(lower, nan=0.0, posinf=0.0), a_min=0.0, a_max=None)
    upper = np.clip(np.nan_to_num(upper, nan=0.0, posinf=0.0), a_min=0.0, a_max=None)
    return lower, upper



def hurdle_gamma_interval_from_samples(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha_param: torch.Tensor,
        alpha: float,
        num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_hurdle_gamma(
        q_logits=q_logits,
        mu=mu,
        alpha=alpha_param,
        n_samples=num_samples,
    ).detach().cpu().numpy()
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def hurdle_lognormal_interval_from_samples(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        alpha: float,
        num_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_hurdle_lognormal(
        q_logits=q_logits,
        mu_log=mu_log,
        sigma_log=sigma_log,
        n_samples=num_samples,
    ).detach().cpu().numpy()
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def lognormal_interval_from_params(
        mu_log: np.ndarray,
        sigma_log: np.ndarray,
        alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    z_low = float(norm.ppf(alpha / 2.0))
    z_high = float(norm.ppf(1.0 - alpha / 2.0))
    mu_log = np.asarray(mu_log, dtype=np.float64)
    sigma_log = np.clip(np.asarray(sigma_log, dtype=np.float64), a_min=1e-8, a_max=None)
    lower = np.exp(np.clip(mu_log + sigma_log * z_low, a_min=-20.0, a_max=20.0))
    upper = np.exp(np.clip(mu_log + sigma_log * z_high, a_min=-20.0, a_max=20.0))
    return lower, upper


def hurdle_tweedie_interval_from_params_approx(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    q = torch.sigmoid(q_logits).detach().cpu().numpy()
    mu_np = mu.detach().cpu().numpy()
    phi_np = phi.detach().cpu().numpy()
    rho_np = rho.detach().cpu().numpy()

    eps = 1e-8
    q = np.clip(q, 0.0, 1.0)
    mu_np = np.clip(mu_np, eps, None)
    phi_np = np.clip(phi_np, eps, None)
    rho_np = np.clip(rho_np, 1.0 + eps, 2.0 - eps)

    p0 = np.clip(tweedie_p0_from_params(mu_np, phi_np, rho_np), 0.0, 1.0 - 1e-10)
    denom = np.clip(1.0 - p0, eps, None)
    var = phi_np * np.power(mu_np, rho_np)
    cond_mean = mu_np / denom
    cond_second_moment = (var + mu_np * mu_np) / denom
    cond_var = np.maximum(cond_second_moment - cond_mean * cond_mean, eps)

    gamma_shape = np.maximum((cond_mean * cond_mean) / cond_var, eps)
    gamma_scale = np.maximum(cond_var / np.maximum(cond_mean, eps), eps)

    zero_mass = 1.0 - q
    q_safe = np.clip(q, eps, 1.0)
    target_lo = alpha / 2.0
    target_hi = 1.0 - alpha / 2.0

    lower = np.zeros_like(mu_np, dtype=np.float64)
    upper = np.zeros_like(mu_np, dtype=np.float64)

    lo_mask = target_lo > zero_mass
    hi_mask = target_hi > zero_mass
    if np.any(lo_mask):
        u_lo = np.clip((target_lo - zero_mass[lo_mask]) / q_safe[lo_mask], 0.0, 1.0)
        lower[lo_mask] = gamma_dist.ppf(u_lo, a=gamma_shape[lo_mask], scale=gamma_scale[lo_mask])
    if np.any(hi_mask):
        u_hi = np.clip((target_hi - zero_mass[hi_mask]) / q_safe[hi_mask], 0.0, 1.0)
        upper[hi_mask] = gamma_dist.ppf(u_hi, a=gamma_shape[hi_mask], scale=gamma_scale[hi_mask])

    lower = np.nan_to_num(lower, nan=0.0, posinf=0.0, neginf=0.0)
    upper = np.nan_to_num(upper, nan=0.0, posinf=0.0, neginf=0.0)
    return lower, upper


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Check the server PyTorch/CUDA environment."
            )
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def start_resource_tracking(model: torch.nn.Module, device: torch.device) -> dict:
    synchronize_device(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "started_at": time.perf_counter(),
        "train_seconds": 0.0,
        "validation_seconds": 0.0,
        "test_seconds": 0.0,
        "borough_test_seconds": 0.0,
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def timed_section_start(device: torch.device) -> float:
    synchronize_device(device)
    return time.perf_counter()


def timed_section_elapsed(device: torch.device, started_at: float) -> float:
    synchronize_device(device)
    return float(time.perf_counter() - started_at)


def finalize_resource_tracking(tracker: dict, device: torch.device) -> dict:
    synchronize_device(device)
    runtime = {
        key: value
        for key, value in tracker.items()
        if key != "started_at"
    }
    runtime["wall_seconds"] = float(time.perf_counter() - tracker["started_at"])
    runtime["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return runtime


def print_progress(
        label: str,
        processed: int,
        total: int,
        started_at: float,
        device: torch.device,
):
    synchronize_device(device)
    elapsed = time.perf_counter() - started_at
    avg_seconds = elapsed / max(processed, 1)
    eta_seconds = avg_seconds * max(total - processed, 0)
    print_status(
        f"{label} | {processed}/{total} ({processed / max(total, 1):.1%}) | "
        f"avg={avg_seconds:.3f}s/batch | elapsed={format_elapsed(elapsed)} | "
        f"ETA={format_elapsed(eta_seconds)}",
        color=ANSI_BLUE,
        flush=True,
    )


def iter_bundle_batches(
        bundles: List[dict],
        split: str,
        device: torch.device,
        progress_interval: int = 0,
        progress_label: str = "",
):
    total = sum(len(bundle[f"{split}_graph_loader"]) for bundle in bundles)
    processed = 0
    started_at = time.perf_counter()
    for bundle in bundles:
        for batch in bundle[f"{split}_graph_loader"]:
            yield bundle, batch
            processed += 1
            if progress_interval > 0 and (
                    processed % progress_interval == 0 or processed == total
            ):
                print_progress(progress_label, processed, total, started_at, device)
    if progress_label:
        synchronize_device(device)
        elapsed = time.perf_counter() - started_at
        print_status(
            f"{progress_label} completed | batches={processed} | elapsed={format_elapsed(elapsed)}",
            color=ANSI_BLUE,
            flush=True,
        )


def print_runtime_device(device: torch.device, requested_device: str):
    print_status(
        f"Runtime | PyTorch={torch.__version__} | requested_device={requested_device} | "
        f"resolved_device={device}",
        color=ANSI_CYAN,
        bold=True,
        flush=True,
    )
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        total_memory_gb = properties.total_memory / (1024 ** 3)
        capability_tag = f"sm_{properties.major}{properties.minor}"
        compiled_arches = torch.cuda.get_arch_list()
        print_status(
            f"CUDA | available=True | torch_cuda={torch.version.cuda} | "
            f"device_index={index} | gpu={properties.name} | "
            f"capability={properties.major}.{properties.minor} | "
            f"memory={total_memory_gb:.1f} GiB",
            color=ANSI_CYAN,
            flush=True,
        )
        print_status(
            f"CUDA architectures | required={capability_tag} | "
            f"compiled={','.join(compiled_arches) if compiled_arches else 'unknown'}",
            color=ANSI_CYAN,
            flush=True,
        )
        if compiled_arches and capability_tag not in compiled_arches:
            print_status(
                f"Warning: this PyTorch build does not list native support for {capability_tag}. "
                "RTX 5090 requires a CUDA/PyTorch build with Blackwell support; "
                "upgrade the server environment before formal training.",
                color=ANSI_YELLOW,
                bold=True,
                flush=True,
            )
        try:
            probe = torch.ones(1, device=device)
            probe = probe + 1.0
            synchronize_device(device)
            if float(probe.cpu().item()) != 2.0:
                raise RuntimeError("CUDA probe returned an unexpected value.")
        except Exception as exc:
            raise RuntimeError(
                "CUDA runtime probe failed before data loading. "
                "Check whether the server PyTorch build supports the installed GPU architecture."
            ) from exc
        print_status("CUDA runtime probe | passed", color=ANSI_GREEN, flush=True)
    else:
        print_status(
            f"CUDA | available={torch.cuda.is_available()} | torch_cuda={torch.version.cuda} | "
            "training_on_cpu=True",
            color=ANSI_YELLOW,
            flush=True,
        )


def capture_git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {
            "git_commit_at_start": commit,
            "git_dirty_at_start": bool(status.strip()),
            "git_tracked_dirty_at_start": bool(tracked_status.strip()),
            "git_tracked_dirty_files_at_start": [
                line[3:] for line in tracked_status.splitlines()[:50]
            ],
            "git_untracked_count_at_start": len(untracked.splitlines()),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit_at_start": None,
            "git_dirty_at_start": None,
            "git_tracked_dirty_at_start": None,
            "git_tracked_dirty_files_at_start": None,
            "git_untracked_count_at_start": None,
        }


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device=device, non_blocking=device.type == "cuda") if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def to_numpy_block(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_group_a_graph_input(batch: dict, bundle: dict) -> torch.Tensor:
    return batch["x"].contiguous()


def build_group_ab_graph_input(batch: dict, bundle: dict) -> torch.Tensor:
    return batch["x"].contiguous()


def build_group_a_xgb_features(sample: dict, bundle: dict) -> np.ndarray:
    return to_numpy_block(sample["x"]).astype(np.float32, copy=False)


def build_group_ab_xgb_features(sample: dict, bundle: dict) -> np.ndarray:
    return to_numpy_block(sample["x"]).astype(np.float32, copy=False)


def build_graph_input(batch: dict, bundle: dict, use_priors: bool) -> torch.Tensor:
    if use_priors:
        return build_group_ab_graph_input(batch, bundle)
    return build_group_a_graph_input(batch, bundle)


def build_xgb_features(sample: dict, bundle: dict, use_priors: bool) -> np.ndarray:
    if use_priors:
        return build_group_ab_xgb_features(sample, bundle)
    return build_group_a_xgb_features(sample, bundle)


def get_graph_input_dim(bundle: dict, use_priors: bool) -> int:
    return bundle["graph_input_dim"]


def forward_graph_model(model: torch.nn.Module, batch: dict, bundle: dict, x_batch: torch.Tensor):
    return model(x_batch, bundle["A_q"], bundle["A_h"])


def build_bundles(args, device: torch.device) -> List[dict]:
    target_mode = normalize_target_mode(args.target_mode)
    hist_source_mode = normalize_hist_source_mode(args.hist_source_mode)
    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)

    X_static, Y_risk, Y_crash, calendar, weather, borough_id_arr, dates = load_nyc_zinb_data(
        args.data_dir,
        start_date,
        end_date,
    )
    edge_index = np.load(os.path.join(args.data_dir, "line_graph", "edge_index.npy")).astype(np.int64)
    node_df = pd.read_parquet(os.path.join(args.data_dir, "line_graph", "node_id_map.parquet")).sort_values("node_idx")
    expansion_groups = parse_expansion_groups(args.expansion_features)
    expansion_dir = args.expansion_dir or os.path.join(args.data_dir, "expansion")

    X_static_full = X_static.copy()
    zero_prefixes = [
        p.strip() for p in str(getattr(args, "static_zero_columns", "")).split(",") if p.strip()
    ]
    if zero_prefixes:
        col_names = load_static_column_names(args.data_dir)
        if len(col_names) != X_static_full.shape[1]:
            raise ValueError(
                f"static column list ({len(col_names)}) does not match X_static width "
                f"({X_static_full.shape[1]})."
            )
        zidx = static_zero_indices(col_names, zero_prefixes)
        if not zidx:
            raise ValueError(f"--static_zero_columns matched no columns: {zero_prefixes}")
        X_static_full[:, zidx] = 0.0
        print_status(
            f"Static zero-out: {len(zidx)}/{len(col_names)} columns zeroed "
            f"(prefixes {zero_prefixes}); dims preserved, standardized space.",
            color=ANSI_CYAN,
        )
    Y_risk_raw = Y_risk.copy()
    Y_hist_full = Y_risk_raw.copy()
    Y_crash_full = Y_crash.copy()
    borough_id_arr_full = borough_id_arr.copy()

    num_nodes, total_steps = Y_hist_full.shape
    if total_steps < args.history_len + args.horizon:
        raise ValueError("Not enough time steps for the given history_len and horizon.")
    if args.mask_borough.strip().lower() == "all" and args.mask_path:
        raise ValueError("mask_path is not supported when mask_borough=all.")

    raw_zi_rate = np.sum(Y_hist_full == 0) / float(num_nodes * total_steps)
    if hist_source_mode == "target":
        hist_source_label = "Y_target"
    elif hist_source_mode == "both":
        if uses_diffused_target(target_mode):
            hist_source_label = "raw Y_risk + Y_target"
        else:
            hist_source_label = "raw Y_risk (both collapsed under target_mode=risk)"
    else:
        hist_source_label = "raw Y_risk"
    print_status(f"Target mode: {target_mode} | Y_hist source: {hist_source_label}", color=ANSI_CYAN)
    print_status(
        f"Raw Y_risk ZI rate: {raw_zi_rate:.4f} | Y_hist mean:{Y_hist_full.mean():.4f} | "
        f"Y_hist[>0] mean:{Y_hist_full[Y_hist_full > 0].mean():.4f} | "
        f"Borough: {args.mask_borough} | Nodes count: {num_nodes} | Date: {args.start_date} ~ {args.end_date}",
        color=ANSI_CYAN,
    )
    print_status(f"Rank score: {args.rank_score}", color=ANSI_CYAN)
    if args.model in {
        "stg", "stnb", "stzinb", "sttd", "stzitd", "st_gamma", "st_lognormal",
        "st_hurdle_gamma", "st_hurdle_lognormal", "st_daer_hurdle_gamma",
        "st_daer_hurdle_lognormal", "st_event_aligned_hurdle_gamma",
        "st_event_aligned_hurdle_lognormal", "st_hurdle_tweedie",
    }:
        print_status(
            f"Probabilistic backbones | temporal={args.temporal_backbone} | spatial={args.spatial_backbone}",
            color=ANSI_CYAN,
        )

    train_end_override = int(getattr(args, "train_end_idx", -1))
    val_end_override = int(getattr(args, "val_end_idx", -1))
    if train_end_override >= 0 or val_end_override >= 0:
        if train_end_override < 0 or val_end_override < 0:
            raise ValueError("--train_end_idx and --val_end_idx must be provided together (>= 0).")
        train_end = train_end_override
        val_end = val_end_override
        calendar, weather = standardize_time_features(calendar, weather, train_end)
        train_s, val_s, test_s = build_time_splits_by_idx(
            total_steps, args.history_len, args.horizon, train_end, val_end
        )
        print_status(
            f"Explicit day-index split: train_end={train_end} val_end={val_end} "
            f"({dates[train_end - 1].date() if train_end > 0 else '-'} | "
            f"{dates[val_end - 1].date() if val_end > 0 else '-'})",
            color=ANSI_CYAN,
        )
    else:
        train_end = int(total_steps * (8 / 12))
        calendar, weather = standardize_time_features(calendar, weather, train_end)
        train_s, val_s, test_s = build_time_splits(total_steps, args.history_len, args.horizon)
    skip_val_eval = bool(int(getattr(args, "skip_val_eval", 0)))
    skip_test_eval = bool(int(getattr(args, "skip_test_eval", 0)))
    if len(train_s) == 0:
        raise ValueError("Time split resulted in empty train subset.")
    if len(val_s) == 0 and not skip_val_eval:
        raise ValueError("Time split resulted in empty val subset (or pass --skip_val_eval 1).")
    if len(test_s) == 0 and not skip_test_eval:
        raise ValueError("Time split resulted in empty test subset (or pass --skip_test_eval 1).")

    hetero_choices = parse_lookback_mix(getattr(args, "hist_hetero_lookbacks", ""))
    hetero_lookbacks_full = None
    if hetero_choices:
        rng = np.random.default_rng(int(args.seed))
        hetero_lookbacks_full = np.asarray(hetero_choices, dtype=np.int64)[
            rng.integers(len(hetero_choices), size=num_nodes)
        ]
        print_status(
            f"Heterogeneous per-road history budgets from {hetero_choices} "
            f"(seed={args.seed}); mean={hetero_lookbacks_full.mean():.1f}d",
            color=ANSI_CYAN,
        )

    holdout_fraction = float(getattr(args, "node_holdout_fraction", 0.0))
    holdout_mask_full = None
    if holdout_fraction > 0:
        if not 0 < holdout_fraction < 1:
            raise ValueError("--node_holdout_fraction must be in (0, 1).")
        holdout_seed = int(getattr(args, "node_holdout_seed", -1))
        if holdout_seed < 0:
            holdout_seed = int(args.seed)
        holdout_mode = str(getattr(args, "node_holdout_mode", "random"))
        if holdout_mode == "spatial_block":
            block_size_m = float(getattr(args, "spatial_block_size_m", 1000.0))
            centroids = load_utm_centroids(args.data_dir, num_nodes)
            holdout_mask_full = build_spatial_block_holdout(
                centroids, borough_id_arr_full, holdout_fraction, block_size_m, holdout_seed,
            )
        elif holdout_mode == "random":
            block_size_m = 0.0
            rng = np.random.default_rng(holdout_seed)
            holdout_mask_full = rng.random(num_nodes) < holdout_fraction
        else:
            raise ValueError("--node_holdout_mode must be 'random' or 'spatial_block'.")
        print_status(
            f"Node-held-out regime ({holdout_mode}"
            f"{f', block {block_size_m:.0f} m' if holdout_mode == 'spatial_block' else ''}): "
            f"{int(holdout_mask_full.sum())}/{num_nodes} segments "
            f"({holdout_mask_full.mean():.1%}) train and evaluate with zero crash history "
            f"(zero-history endpoint); their labels are excluded from training losses. "
            f"holdout_seed={holdout_seed}",
            color=ANSI_CYAN,
        )

    all_mode = args.mask_borough.strip().lower() == "all"
    bundles = []
    if all_mode:
        borough_names = load_borough_name_order(args.data_dir)
        print_status(
            f"Training mode: all-city round-robin across {len(borough_names)} borough subgraphs",
            color=ANSI_BLUE,
            bold=True,
        )
        target_names = borough_names
    else:
        print_status(f"Training mode: single borough | {args.mask_borough}", color=ANSI_BLUE, bold=True)
        target_names = [args.mask_borough]

    for borough_name in target_names:
        sub_mask = load_mask(args.data_dir, borough_name, "" if all_mode else args.mask_path, num_nodes)
        sub_hashes = node_df.loc[sub_mask, "id_geom_hash"].to_numpy()
        X_dynamic_sub, dynamic_names, dynamic_stats = load_expansion_features(
            expansion_dir=expansion_dir,
            node_hashes=sub_hashes,
            dates=dates,
            groups=expansion_groups,
            train_end=train_end,
            mobility_lag_days=int(args.mobility_lag_days),
        )
        if dynamic_names:
            print_status(
                f"[{borough_name}] expansion features: {','.join(expansion_groups)} "
                f"({len(dynamic_names)} channels, mobility_lag_days={int(args.mobility_lag_days)}"
                f"{' ORACLE same-day' if int(args.mobility_lag_days) == 0 else ''})",
                color=ANSI_CYAN,
            )
        bundle = build_baseline_bundle(
            borough_name=borough_name,
            mask=sub_mask,
            X_static_full=X_static_full,
            X_dynamic_sub=X_dynamic_sub,
            dynamic_feature_names=dynamic_names,
            dynamic_feature_stats=dynamic_stats,
            Y_hist_full=Y_hist_full,
            Y_risk_full=Y_risk_raw,
            Y_crash_full=Y_crash_full,
            calendar=calendar,
            weather=weather,
            borough_id_arr_full=borough_id_arr_full,
            edge_index_full=edge_index,
            args=args,
            train_s=train_s,
            val_s=val_s,
            test_s=test_s,
            train_end=train_end,
            device=device,
            hetero_lookbacks_full=hetero_lookbacks_full,
            holdout_mask_full=holdout_mask_full,
        )
        bundles.append(bundle)

    first_graph_dim = bundles[0]["graph_input_dim"]
    first_xgb_dim = bundles[0]["xgb_input_dim"]
    for bundle in bundles[1:]:
        if bundle["graph_input_dim"] != first_graph_dim:
            raise ValueError("Graph input dimensions are inconsistent across boroughs.")
        if bundle["xgb_input_dim"] != first_xgb_dim:
            raise ValueError("XGBoost input dimensions are inconsistent across boroughs.")

    total_train_label_count = sum(bundle["train_label_count"] for bundle in bundles)
    if total_train_label_count > 0:
        overall_train_label_zero_rate = sum(
            bundle["train_label_zero_rate"] * bundle["train_label_count"] for bundle in bundles
        ) / total_train_label_count
        scope = "all-city" if all_mode else bundles[0]["name"]
        print(
            f"Training label ZI ({scope}, target_mode={target_mode}): "
            f"{overall_train_label_zero_rate:.4f}"
        )
    return bundles


@torch.no_grad()
def evaluate_graph_model(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        device: torch.device,
        hr_ratio: float,
        use_priors: bool,
        rank_score: str,
        zr_tau: float,
        progress_interval: int = 0,
        progress_label: str = "",
) -> dict:
    target_state = init_det_eval_state()
    event_state = init_det_eval_state()
    model.eval()
    for bundle, batch in iter_bundle_batches(
            bundles,
            split,
            device,
            progress_interval=progress_interval,
            progress_label=progress_label,
    ):
        batch = move_batch_to_device(batch, device)
        x_batch = build_graph_input(batch, bundle, use_priors=use_priors)
        pred = forward_graph_model(model, batch, bundle, x_batch).detach().cpu().numpy()
        score = build_point_rank_score(rank_score, pred)
        y_np = batch["y"].detach().cpu().numpy()
        y_rank_np = batch["y_rank"].detach().cpu().numpy()
        y_event_np = batch["y_event"].detach().cpu().numpy()
        y_event_rank_np = batch["y_event_rank"].detach().cpu().numpy()
        update_deterministic_space_eval_states(
            target_state,
            event_state,
            y_np,
            y_rank_np,
            y_event_np,
            y_event_rank_np,
            pred,
            hr_ratio,
            score,
        )
    return merge_space_metrics(finalize_det_eval_state(target_state), finalize_det_eval_state(event_state))


@torch.no_grad()
def evaluate_probabilistic_model(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        device: torch.device,
        hr_ratio: float,
        use_priors: bool,
        model_kind: str,
        interval_alpha: float,
        rank_score: str,
        zr_tau: float,
        interval_samples: int,
        interval_seed: int,
        daer_rank_branch: str = "q_es",
        compute_intervals: bool = True,
        progress_interval: int = 0,
        progress_label: str = "",
) -> dict:
    target_state = init_det_eval_state()
    event_state = init_det_eval_state()
    event_bce_sum = 0.0
    event_brier_sum = 0.0
    event_value_count = 0
    model.eval()
    split_seed_offset = {"val": 0, "test": 100_003}.get(split, 200_003)
    interval_rng = np.random.default_rng(interval_seed + split_seed_offset)

    for bundle, batch in iter_bundle_batches(
            bundles,
            split,
            device,
            progress_interval=progress_interval,
            progress_label=progress_label,
    ):
        batch = move_batch_to_device(batch, device)
        x_batch = build_graph_input(batch, bundle, use_priors=use_priors)
        y_np = batch["y"].detach().cpu().numpy()
        y_rank_np = batch["y_rank"].detach().cpu().numpy()
        y_event_np = batch["y_event"].detach().cpu().numpy()
        y_event_rank_np = batch["y_event_rank"].detach().cpu().numpy()

        def update_event_calibration(event_logits: torch.Tensor):
            nonlocal event_bce_sum, event_brier_sum, event_value_count
            event_labels = (batch["y_event"] > 0).float()
            event_probabilities = torch.sigmoid(event_logits)
            event_bce_sum += float(F.binary_cross_entropy_with_logits(
                event_logits, event_labels, reduction="sum"
            ).cpu().item())
            event_brier_sum += float(torch.square(event_probabilities - event_labels).sum().cpu().item())
            event_value_count += event_labels.numel()

        if model_kind == "gaussian":
            loc_pred, scale_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = loc_pred.detach().cpu().numpy()
            loc_np = pred_mean
            scale_np = scale_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(
                rank_score,
                model_kind="gaussian",
                mean_pred=pred_mean,
                loc_np=loc_np,
                scale_np=scale_np,
            )
            lower, upper = gaussian_interval_from_params(pred_mean, scale_np, alpha=interval_alpha)
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "gamma":
            mu_pred, alpha_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = gamma_mean(mu_pred).detach().cpu().numpy()
            mu_np = mu_pred.detach().cpu().numpy()
            alpha_np = alpha_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(
                "mean",
                model_kind="gamma",
                mean_pred=pred_mean,
                mu_np=mu_np,
                alpha_np=alpha_np,
            )
            lower, upper = gamma_interval_from_params(mu_np, alpha_np, alpha=interval_alpha)
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "lognormal":
            mu_log_pred, sigma_log_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = lognormal_mean(mu_log_pred, sigma_log_pred).detach().cpu().numpy()
            mu_log_np = mu_log_pred.detach().cpu().numpy()
            sigma_log_np = sigma_log_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(
                "mean",
                model_kind="lognormal",
                mean_pred=pred_mean,
            )
            lower, upper = lognormal_interval_from_params(mu_log_np, sigma_log_np, alpha=interval_alpha)
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "hurdle_gamma":
            q_logits_pred, mu_pred, alpha_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = hurdle_gamma_mean(q_logits_pred, mu_pred).detach().cpu().numpy()
            q_np = torch.sigmoid(q_logits_pred).detach().cpu().numpy()
            mu_np = mu_pred.detach().cpu().numpy()
            alpha_np = alpha_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="hurdle_gamma", mean_pred=pred_mean,
                                             q_np=q_np, mu_np=mu_np, alpha_np=alpha_np)
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_gamma_interval_from_samples(
                q_logits=q_logits_pred,
                mu=mu_pred,
                alpha_param=alpha_pred,
                alpha=interval_alpha,
                num_samples=interval_samples,
            )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "event_aligned_hurdle_gamma":
            q_logits_pred, mu_pred, alpha_pred = forward_graph_model(model, batch, bundle, x_batch)
            update_event_calibration(q_logits_pred)
            pred_mean = hurdle_gamma_mean(q_logits_pred, mu_pred).detach().cpu().numpy()
            decision_score_np = q_logits_pred.detach().cpu().numpy()
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_gamma_interval_from_samples(
                    q_logits=q_logits_pred,
                    mu=mu_pred,
                    alpha_param=alpha_pred,
                    alpha=interval_alpha,
                    num_samples=interval_samples,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, pred_mean,
                event_score=decision_score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "daer_hurdle_gamma":
            q_target_logits, q_event_logits, mu_pred, alpha_pred = forward_graph_model(
                model, batch, bundle, x_batch
            )
            update_event_calibration(q_event_logits)
            pred_mean = hurdle_gamma_mean(q_target_logits, mu_pred).detach().cpu().numpy()
            decision_logits = q_target_logits if daer_rank_branch == "q_ts" else q_event_logits
            decision_score_np = decision_logits.detach().cpu().numpy()
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_gamma_interval_from_samples(
                    q_logits=q_target_logits,
                    mu=mu_pred,
                    alpha_param=alpha_pred,
                    alpha=interval_alpha,
                    num_samples=interval_samples,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, pred_mean,
                event_score=decision_score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "hurdle_lognormal":
            q_logits_pred, mu_log_pred, sigma_log_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = hurdle_lognormal_mean(q_logits_pred, mu_log_pred, sigma_log_pred).detach().cpu().numpy()
            q_np = torch.sigmoid(q_logits_pred).detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="hurdle_lognormal", mean_pred=pred_mean,
                                             q_np=q_np)
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_lognormal_interval_from_samples(
                    q_logits=q_logits_pred,
                    mu_log=mu_log_pred,
                    sigma_log=sigma_log_pred,
                    alpha=interval_alpha,
                    num_samples=interval_samples,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "event_aligned_hurdle_lognormal":
            q_logits_pred, mu_log_pred, sigma_log_pred = forward_graph_model(model, batch, bundle, x_batch)
            update_event_calibration(q_logits_pred)
            pred_mean = hurdle_lognormal_mean(q_logits_pred, mu_log_pred, sigma_log_pred).detach().cpu().numpy()
            decision_score_np = q_logits_pred.detach().cpu().numpy()
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_lognormal_interval_from_samples(
                    q_logits=q_logits_pred,
                    mu_log=mu_log_pred,
                    sigma_log=sigma_log_pred,
                    alpha=interval_alpha,
                    num_samples=interval_samples,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, pred_mean,
                event_score=decision_score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "daer_hurdle_lognormal":
            q_target_logits, q_event_logits, mu_log_pred, sigma_log_pred = forward_graph_model(
                model, batch, bundle, x_batch
            )
            update_event_calibration(q_event_logits)
            pred_mean = hurdle_lognormal_mean(
                q_target_logits, mu_log_pred, sigma_log_pred
            ).detach().cpu().numpy()
            decision_logits = q_target_logits if daer_rank_branch == "q_ts" else q_event_logits
            decision_score_np = decision_logits.detach().cpu().numpy()
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_lognormal_interval_from_samples(
                    q_logits=q_target_logits,
                    mu_log=mu_log_pred,
                    sigma_log=sigma_log_pred,
                    alpha=interval_alpha,
                    num_samples=interval_samples,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, pred_mean,
                event_score=decision_score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "hurdle_tweedie":
            q_logits_pred, mu_pred, phi_pred, rho_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = hurdle_tweedie_mean(q_logits_pred, mu_pred, phi_pred, rho_pred).detach().cpu().numpy()
            q_np = torch.sigmoid(q_logits_pred).detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="hurdle_tweedie", mean_pred=pred_mean,
                                             q_np=q_np)
            lower = upper = None
            if compute_intervals:
                lower, upper = hurdle_tweedie_interval_from_params_approx(
                    q_logits=q_logits_pred,
                    mu=mu_pred,
                    phi=phi_pred,
                    rho=rho_pred,
                    alpha=interval_alpha,
                )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "nb":
            n_pred, p_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = negative_binomial_mean(n_pred, p_pred).detach().cpu().numpy()
            n_np = n_pred.detach().cpu().numpy()
            p_np = p_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="nb", mean_pred=pred_mean,
                                             n_np=n_np, p_np=p_np)
            lower, upper = nb_interval_from_params(n_np, p_np, alpha=interval_alpha)
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "zinb":
            n_pred, p_pred, pi_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = zero_inflated_nb_mean(n_pred, p_pred, pi_pred).detach().cpu().numpy()
            n_np = n_pred.detach().cpu().numpy()
            p_np = p_pred.detach().cpu().numpy()
            pi_np = pi_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="zinb", mean_pred=pred_mean,
                                             n_np=n_np, p_np=p_np, pi_np=pi_np)
            lower, upper = zinb_interval_LU_from_params_scipy(n_np, p_np, pi_np, alpha=interval_alpha)
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "tweedie":
            mu_pred, phi_pred, rho_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = tweedie_mean(mu_pred).detach().cpu().numpy()
            mu_np = mu_pred.detach().cpu().numpy()
            phi_np = phi_pred.detach().cpu().numpy()
            rho_np = rho_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="tweedie", mean_pred=pred_mean,
                                             mu_np=mu_np, phi_np=phi_np, rho_np=rho_np)
            lower, upper = tweedie_interval_from_params_mc(
                mu_np, phi_np, rho_np, alpha=interval_alpha, num_samples=interval_samples, rng=interval_rng,
            )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        if model_kind == "zitd":
            pi_pred, mu_pred, phi_pred, rho_pred = forward_graph_model(model, batch, bundle, x_batch)
            pred_mean = zero_inflated_tweedie_mean(pi_pred, mu_pred).detach().cpu().numpy()
            pi_np = pi_pred.detach().cpu().numpy()
            mu_np = mu_pred.detach().cpu().numpy()
            phi_np = phi_pred.detach().cpu().numpy()
            rho_np = rho_pred.detach().cpu().numpy()
            score_np = build_prob_rank_score(rank_score, model_kind="zitd", mean_pred=pred_mean,
                                             pi_np=pi_np, mu_np=mu_np, phi_np=phi_np, rho_np=rho_np)
            lower, upper = zero_inflated_tweedie_interval_from_params_mc(
                pi_np, mu_np, phi_np, rho_np, alpha=interval_alpha, num_samples=interval_samples,
                rng=interval_rng,
            )
            update_probabilistic_space_eval_states(
                target_state, event_state, y_np, y_rank_np, y_event_np, y_event_rank_np,
                pred_mean, hr_ratio, interval_alpha, score_np, lower=lower, upper=upper,
            )
            continue

        raise ValueError(f"Unsupported probabilistic model kind: {model_kind}")

    metrics = merge_space_metrics(finalize_det_eval_state(target_state), finalize_det_eval_state(event_state))
    if event_value_count > 0:
        metrics["event_bce"] = event_bce_sum / event_value_count
        metrics["event_brier"] = event_brier_sum / event_value_count
    return metrics


@torch.no_grad()
def evaluate_probabilistic_model_by_borough(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        device: torch.device,
        hr_ratio: float,
        use_priors: bool,
        model_kind: str,
        interval_alpha: float,
        rank_score: str,
        zr_tau: float,
        interval_samples: int,
        interval_seed: int,
        daer_rank_branch: str = "q_es",
        compute_intervals: bool = True,
) -> Dict[str, dict]:
    borough_metrics: Dict[str, dict] = {}
    for bundle in bundles:
        borough_metrics[bundle["name"]] = evaluate_probabilistic_model(
            model=model,
            bundles=[bundle],
            split=split,
            device=device,
            hr_ratio=hr_ratio,
            use_priors=use_priors,
            model_kind=model_kind,
            interval_alpha=interval_alpha,
            rank_score=rank_score,
            zr_tau=zr_tau,
            interval_samples=interval_samples,
            interval_seed=interval_seed,
            daer_rank_branch=daer_rank_branch,
            compute_intervals=compute_intervals,
        )
    return borough_metrics


def accumulate_prob_stat(stat_store: Dict[str, dict], name: str, value: torch.Tensor):
    detached = value.detach()
    current = stat_store.get(name)
    if current is None:
        stat_store[name] = {
            "sum": detached,
            "min": detached,
            "max": detached,
            "count": 1,
        }
        return
    current["sum"] = current["sum"] + detached
    current["min"] = torch.minimum(current["min"], detached)
    current["max"] = torch.maximum(current["max"], detached)
    current["count"] += 1


def record_prob_output_stats(stat_store: Dict[str, dict], outputs: tuple, model_kind: str):
    def append_stat(name: str, value: torch.Tensor):
        accumulate_prob_stat(stat_store, name, value)

    if model_kind == "gaussian":
        loc, scale = outputs
        append_stat("loc_mean", loc.mean())
        append_stat("loc_max", loc.max())
        append_stat("scale_mean", scale.mean())
        append_stat("scale_min", scale.min())
        append_stat("scale_max", scale.max())
        return

    if model_kind == "gamma":
        mu, alpha = outputs
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("alpha_mean", alpha.mean())
        append_stat("alpha_max", alpha.max())
        return

    if model_kind == "lognormal":
        mu_log, sigma_log = outputs
        append_stat("mu_log_mean", mu_log.mean())
        append_stat("mu_log_min", mu_log.min())
        append_stat("mu_log_max", mu_log.max())
        append_stat("sigma_log_mean", sigma_log.mean())
        append_stat("sigma_log_min", sigma_log.min())
        append_stat("sigma_log_max", sigma_log.max())
        return

    if model_kind in {"hurdle_gamma", "event_aligned_hurdle_gamma"}:
        q_logits, mu, alpha = outputs
        q_prob = torch.sigmoid(q_logits)
        append_stat("q_mean", q_prob.mean())
        append_stat("q_min", q_prob.min())
        append_stat("q_max", q_prob.max())
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("alpha_mean", alpha.mean())
        append_stat("alpha_max", alpha.max())
        return

    if model_kind == "daer_hurdle_gamma":
        q_target_logits, q_event_logits, mu, alpha = outputs
        q_target = torch.sigmoid(q_target_logits)
        q_event = torch.sigmoid(q_event_logits)
        append_stat("q_target_mean", q_target.mean())
        append_stat("q_target_min", q_target.min())
        append_stat("q_target_max", q_target.max())
        append_stat("q_event_mean", q_event.mean())
        append_stat("q_event_min", q_event.min())
        append_stat("q_event_max", q_event.max())
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("alpha_mean", alpha.mean())
        append_stat("alpha_max", alpha.max())
        return

    if model_kind in {"hurdle_lognormal", "event_aligned_hurdle_lognormal"}:
        q_logits, mu_log, sigma_log = outputs
        q_prob = torch.sigmoid(q_logits)
        append_stat("q_mean", q_prob.mean())
        append_stat("q_min", q_prob.min())
        append_stat("q_max", q_prob.max())
        append_stat("mu_log_mean", mu_log.mean())
        append_stat("mu_log_min", mu_log.min())
        append_stat("mu_log_max", mu_log.max())
        append_stat("sigma_log_mean", sigma_log.mean())
        append_stat("sigma_log_min", sigma_log.min())
        append_stat("sigma_log_max", sigma_log.max())
        return

    if model_kind == "daer_hurdle_lognormal":
        q_target_logits, q_event_logits, mu_log, sigma_log = outputs
        q_target = torch.sigmoid(q_target_logits)
        q_event = torch.sigmoid(q_event_logits)
        append_stat("q_target_mean", q_target.mean())
        append_stat("q_target_min", q_target.min())
        append_stat("q_target_max", q_target.max())
        append_stat("q_event_mean", q_event.mean())
        append_stat("q_event_min", q_event.min())
        append_stat("q_event_max", q_event.max())
        append_stat("mu_log_mean", mu_log.mean())
        append_stat("mu_log_min", mu_log.min())
        append_stat("mu_log_max", mu_log.max())
        append_stat("sigma_log_mean", sigma_log.mean())
        append_stat("sigma_log_min", sigma_log.min())
        append_stat("sigma_log_max", sigma_log.max())
        return

    if model_kind == "hurdle_tweedie":
        q_logits, mu, phi, rho = outputs
        q_prob = torch.sigmoid(q_logits)
        append_stat("q_mean", q_prob.mean())
        append_stat("q_min", q_prob.min())
        append_stat("q_max", q_prob.max())
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("phi_mean", phi.mean())
        append_stat("phi_min", phi.min())
        append_stat("phi_max", phi.max())
        append_stat("rho_mean", rho.mean())
        append_stat("rho_min", rho.min())
        append_stat("rho_max", rho.max())
        return

    if model_kind == "nb":
        n, p = outputs
        append_stat("n_mean", n.mean())
        append_stat("n_max", n.max())
        append_stat("p_mean", p.mean())
        append_stat("p_min", p.min())
        append_stat("p_max", p.max())
        return

    if model_kind == "zinb":
        n, p, pi = outputs
        append_stat("n_mean", n.mean())
        append_stat("n_max", n.max())
        append_stat("p_mean", p.mean())
        append_stat("p_min", p.min())
        append_stat("p_max", p.max())
        append_stat("pi_mean", pi.mean())
        append_stat("pi_min", pi.min())
        append_stat("pi_max", pi.max())
        return

    if model_kind == "tweedie":
        mu, phi, rho = outputs
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("phi_mean", phi.mean())
        append_stat("phi_min", phi.min())
        append_stat("phi_max", phi.max())
        append_stat("rho_mean", rho.mean())
        append_stat("rho_min", rho.min())
        append_stat("rho_max", rho.max())
        return

    if model_kind == "zitd":
        pi, mu, phi, rho = outputs
        append_stat("pi_mean", pi.mean())
        append_stat("pi_min", pi.min())
        append_stat("pi_max", pi.max())
        append_stat("mu_mean", mu.mean())
        append_stat("mu_max", mu.max())
        append_stat("phi_mean", phi.mean())
        append_stat("phi_min", phi.min())
        append_stat("phi_max", phi.max())
        append_stat("rho_mean", rho.mean())
        append_stat("rho_min", rho.min())
        append_stat("rho_max", rho.max())
        return

    raise ValueError(f"Unsupported probabilistic model kind: {model_kind}")


def format_prob_output_stats(stat_store: Dict[str, dict], model_kind: str, skipped_batches: int) -> str:
    if not stat_store:
        return f"{model_kind}_params | skipped_batches={skipped_batches} | no_valid_batches"

    materialized_stats = {}

    def mean_stat(name: str) -> float:
        cache_key = ("mean", name)
        if cache_key not in materialized_stats:
            entry = stat_store.get(name)
            materialized_stats[cache_key] = (
                float((entry["sum"] / entry["count"]).cpu().item())
                if entry is not None
                else float("nan")
            )
        return materialized_stats[cache_key]

    def min_stat(name: str) -> float:
        cache_key = ("min", name)
        if cache_key not in materialized_stats:
            entry = stat_store.get(name)
            materialized_stats[cache_key] = (
                float(entry["min"].cpu().item()) if entry is not None else float("nan")
            )
        return materialized_stats[cache_key]

    def max_stat(name: str) -> float:
        cache_key = ("max", name)
        if cache_key not in materialized_stats:
            entry = stat_store.get(name)
            materialized_stats[cache_key] = (
                float(entry["max"].cpu().item()) if entry is not None else float("nan")
            )
        return materialized_stats[cache_key]

    if model_kind == "gaussian":
        return (
            f"gaussian_params | loc_mean={mean_stat('loc_mean'):.4f} | loc_max={max_stat('loc_max'):.4f} | "
            f"scale_mean={mean_stat('scale_mean'):.4f} | scale_min={min_stat('scale_min'):.4f} | "
            f"scale_max={max_stat('scale_max'):.4f} | skipped_batches={skipped_batches}"
        )

    if model_kind == "gamma":
        return (
            f"gamma_params | mu_mean={mean_stat('mu_mean'):.4f} | mu_max={max_stat('mu_max'):.4f} | "
            f"alpha_mean={mean_stat('alpha_mean'):.4f} | alpha_max={max_stat('alpha_max'):.4f} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind == "lognormal":
        return (
            f"lognormal_params | mu_log_mean={mean_stat('mu_log_mean'):.4f} | "
            f"mu_log_min={min_stat('mu_log_min'):.4f} | mu_log_max={max_stat('mu_log_max'):.4f} | "
            f"sigma_log_mean={mean_stat('sigma_log_mean'):.4f} | "
            f"sigma_log_min={min_stat('sigma_log_min'):.4f} | sigma_log_max={max_stat('sigma_log_max'):.4f} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind in {"hurdle_gamma", "event_aligned_hurdle_gamma"}:
        label = "event_aligned_hurdle_gamma" if model_kind == "event_aligned_hurdle_gamma" else "hurdle_gamma"
        event_part = (
            f" | event_cls={mean_stat('event_cls'):.4f} | event_rank={mean_stat('event_rank'):.4f}"
            if model_kind == "event_aligned_hurdle_gamma"
            else ""
        )
        return (
            f"{label}_params | q_mean={mean_stat('q_mean'):.4f} | q_min={min_stat('q_min'):.4f} | "
            f"q_max={max_stat('q_max'):.4f} | mu_mean={mean_stat('mu_mean'):.4f} | "
            f"mu_max={max_stat('mu_max'):.4f} | alpha_mean={mean_stat('alpha_mean'):.4f} | "
            f"alpha_max={max_stat('alpha_max'):.4f} | occ_nll={mean_stat('occ_nll'):.4f} | "
            f"pos_nll={mean_stat('pos_nll'):.4f}{event_part} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind in {"hurdle_lognormal", "event_aligned_hurdle_lognormal"}:
        label = (
            "event_aligned_hurdle_lognormal"
            if model_kind == "event_aligned_hurdle_lognormal"
            else "hurdle_lognormal"
        )
        event_part = (
            f" | event_cls={mean_stat('event_cls'):.4f} | event_rank={mean_stat('event_rank'):.4f}"
            if model_kind == "event_aligned_hurdle_lognormal"
            else ""
        )
        return (
            f"{label}_params | q_mean={mean_stat('q_mean'):.4f} | q_min={min_stat('q_min'):.4f} | "
            f"q_max={max_stat('q_max'):.4f} | mu_log_mean={mean_stat('mu_log_mean'):.4f} | "
            f"mu_log_min={min_stat('mu_log_min'):.4f} | mu_log_max={max_stat('mu_log_max'):.4f} | "
            f"sigma_log_mean={mean_stat('sigma_log_mean'):.4f} | sigma_log_min={min_stat('sigma_log_min'):.4f} | "
            f"sigma_log_max={max_stat('sigma_log_max'):.4f} | occ_nll={mean_stat('occ_nll'):.4f} | "
            f"pos_nll={mean_stat('pos_nll'):.4f}{event_part} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind == "daer_hurdle_gamma":
        return (
            f"daer_hurdle_gamma_params | "
            f"q_ts_mean={mean_stat('q_target_mean'):.4f} | q_ts_min={min_stat('q_target_min'):.4f} | "
            f"q_ts_max={max_stat('q_target_max'):.4f} | q_es_mean={mean_stat('q_event_mean'):.4f} | "
            f"q_es_min={min_stat('q_event_min'):.4f} | q_es_max={max_stat('q_event_max'):.4f} | "
            f"mu_mean={mean_stat('mu_mean'):.4f} | mu_max={max_stat('mu_max'):.4f} | "
            f"alpha_mean={mean_stat('alpha_mean'):.4f} | alpha_max={max_stat('alpha_max'):.4f} | "
            f"ts_loss={mean_stat('ts_loss'):.4f} | event_cls={mean_stat('event_cls'):.4f} | "
            f"event_rank={mean_stat('event_rank'):.4f} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind == "daer_hurdle_lognormal":
        return (
            f"daer_hurdle_lognormal_params | "
            f"q_ts_mean={mean_stat('q_target_mean'):.4f} | q_ts_min={min_stat('q_target_min'):.4f} | "
            f"q_ts_max={max_stat('q_target_max'):.4f} | q_es_mean={mean_stat('q_event_mean'):.4f} | "
            f"q_es_min={min_stat('q_event_min'):.4f} | q_es_max={max_stat('q_event_max'):.4f} | "
            f"mu_log_mean={mean_stat('mu_log_mean'):.4f} | sigma_log_mean={mean_stat('sigma_log_mean'):.4f} | "
            f"ts_loss={mean_stat('ts_loss'):.4f} | event_cls={mean_stat('event_cls'):.4f} | "
            f"event_rank={mean_stat('event_rank'):.4f} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind == "hurdle_tweedie":
        return (
            f"hurdle_tweedie_params | q_mean={mean_stat('q_mean'):.4f} | q_min={min_stat('q_min'):.4f} | "
            f"q_max={max_stat('q_max'):.4f} | mu_mean={mean_stat('mu_mean'):.4f} | "
            f"mu_max={max_stat('mu_max'):.4f} | phi_mean={mean_stat('phi_mean'):.4f} | "
            f"phi_min={min_stat('phi_min'):.4f} | phi_max={max_stat('phi_max'):.4f} | "
            f"rho_mean={mean_stat('rho_mean'):.4f} | rho_min={min_stat('rho_min'):.4f} | "
            f"rho_max={max_stat('rho_max'):.4f} | occ_nll={mean_stat('occ_nll'):.4f} | "
            f"pos_nll={mean_stat('pos_nll'):.4f} | skipped_batches={skipped_batches}"
        )

    if model_kind == "nb":
        return (
            f"nb_params | n_mean={mean_stat('n_mean'):.4f} | n_max={max_stat('n_max'):.4f} | "
            f"p_mean={mean_stat('p_mean'):.4f} | p_min={min_stat('p_min'):.4f} | "
            f"p_max={max_stat('p_max'):.4f} | skipped_batches={skipped_batches}"
        )

    if model_kind == "zinb":
        return (
            f"zinb_params | n_mean={mean_stat('n_mean'):.4f} | n_max={max_stat('n_max'):.4f} | "
            f"p_mean={mean_stat('p_mean'):.4f} | p_min={min_stat('p_min'):.4f} | "
            f"p_max={max_stat('p_max'):.4f} | pi_mean={mean_stat('pi_mean'):.4f} | "
            f"pi_min={min_stat('pi_min'):.4f} | pi_max={max_stat('pi_max'):.4f} | skipped_batches={skipped_batches}"
        )

    if model_kind == "tweedie":
        return (
            f"tweedie_params | mu_mean={mean_stat('mu_mean'):.4f} | mu_max={max_stat('mu_max'):.4f} | "
            f"phi_mean={mean_stat('phi_mean'):.4f} | phi_min={min_stat('phi_min'):.4f} | "
            f"phi_max={max_stat('phi_max'):.4f} | rho_mean={mean_stat('rho_mean'):.4f} | "
            f"rho_min={min_stat('rho_min'):.4f} | rho_max={max_stat('rho_max'):.4f} | "
            f"skipped_batches={skipped_batches}"
        )

    if model_kind == "zitd":
        return (
            f"zitd_params | pi_mean={mean_stat('pi_mean'):.4f} | pi_min={min_stat('pi_min'):.4f} | "
            f"pi_max={max_stat('pi_max'):.4f} | mu_mean={mean_stat('mu_mean'):.4f} | "
            f"mu_max={max_stat('mu_max'):.4f} | phi_mean={mean_stat('phi_mean'):.4f} | "
            f"phi_min={min_stat('phi_min'):.4f} | phi_max={max_stat('phi_max'):.4f} | "
            f"rho_mean={mean_stat('rho_mean'):.4f} | rho_min={min_stat('rho_min'):.4f} | "
            f"rho_max={max_stat('rho_max'):.4f} | skipped_batches={skipped_batches}"
        )

    raise ValueError(f"Unsupported probabilistic model kind: {model_kind}")


@torch.no_grad()
def evaluate_event_occurrence_model(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        device: torch.device,
        hr_ratio,
        use_priors: bool,
        progress_interval: int = 0,
        progress_label: str = "",
        eval_horizon: int = 0,
        seen_nodes_only: bool = False,
) -> dict:
    state = init_det_eval_state()
    bce_sum = 0.0
    brier_sum = 0.0
    value_count = 0
    model.eval()
    for bundle, batch in iter_bundle_batches(
            bundles,
            split,
            device,
            progress_interval=progress_interval,
            progress_label=progress_label,
    ):
        batch = move_batch_to_device(batch, device)
        x_batch = build_graph_input(batch, bundle, use_priors=use_priors)
        logits = forward_graph_model(model, batch, bundle, x_batch)
        labels = (batch["y_event"] > 0).float()
        y_event_rank = batch["y_event_rank"]
        node_mask = seen_node_mask_for_evaluation(batch, seen_nodes_only)
        if node_mask is not None:
            logits = logits[:, node_mask, :]
            labels = labels[:, node_mask, :]
            y_event_rank = y_event_rank[:, node_mask, :]
        if eval_horizon > 0:
            logits = logits[..., :eval_horizon]
            labels = labels[..., :eval_horizon]
            y_event_rank = y_event_rank[..., :eval_horizon]
        probabilities = torch.sigmoid(logits)
        bce_sum += float(torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="sum"
        ).cpu().item())
        brier_sum += float(torch.square(probabilities - labels).sum().cpu().item())
        value_count += labels.numel()
        update_det_eval_state(
            state,
            labels.cpu().numpy(),
            y_event_rank.cpu().numpy(),
            probabilities.cpu().numpy(),
            hr_ratio,
            score=logits.cpu().numpy(),
        )

    raw_metrics = finalize_det_eval_state(state)
    metrics = {
        f"es_{key}": float(value)
        for key, value in raw_metrics.items()
        if is_space_metric_key(key)
    }
    metrics["event_bce"] = bce_sum / max(value_count, 1)
    metrics["event_brier"] = brier_sum / max(value_count, 1)
    return metrics


def iter_interleaved_training_batches(bundles: List[dict]):
    if len(bundles) == 1:
        bundle = bundles[0]
        for batch in bundle["train_graph_loader"]:
            yield bundle, batch
        return

    iterators = [iter(bundle["train_graph_loader"]) for bundle in bundles]
    active_indices = list(range(len(bundles)))
    while active_indices:
        bundle_idx = random.choice(active_indices)
        try:
            batch = next(iterators[bundle_idx])
        except StopIteration:
            active_indices.remove(bundle_idx)
            continue
        yield bundles[bundle_idx], batch


def evaluate_event_by_lookback(
        model: torch.nn.Module,
        bundles: List[dict],
        split: str,
        args,
        device: torch.device,
        progress_label: str = "",
        seen_nodes_only: bool = False,
) -> dict:
    """Evaluate an event model once per pinned history-budget loader."""
    loaders_key = f"{split}_loaders_by_L"
    by_L = {}
    for L in bundles[0][loaders_key]:
        views = []
        for bundle in bundles:
            view = dict(bundle)
            view[f"{split}_graph_loader"] = bundle[loaders_key][L]
            views.append(view)
        by_L[L] = evaluate_event_occurrence_model(
            model,
            views,
            split=split,
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            progress_interval=args.progress_interval,
            progress_label=f"{progress_label} L={L}" if progress_label else "",
            eval_horizon=args.event_eval_horizon,
            seen_nodes_only=seen_nodes_only,
        )
    return by_L


def mean_metric_dicts(metric_dicts: List[dict]) -> dict:
    if not metric_dicts:
        return {}
    return {
        key: float(np.mean([d[key] for d in metric_dicts]))
        for key in metric_dicts[0]
    }


def train_event_occurrence_baseline(
        model: torch.nn.Module,
        args,
        bundles: List[dict],
        device: torch.device,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resource_tracker = start_resource_tracking(model, device)
    skip_val_eval = bool(int(getattr(args, "skip_val_eval", 0)))
    skip_test_eval = bool(int(getattr(args, "skip_test_eval", 0)))
    if skip_val_eval and args.selection_metric != "last":
        raise ValueError("--skip_val_eval 1 requires --selection_metric last (M2 boundary retraining).")
    best_score = float("-inf")
    best_state = None
    best_val_metrics = None
    selected_epoch = None
    epoch_history = []
    validation_seen_nodes_only = any(bundle.get("holdout_count", 0) > 0 for bundle in bundles)

    print_status(
        "Direct event baseline | same encoder as DAER | "
        f"loss={args.event_loss} | selection={args.selection_metric} | "
        f"supervision={getattr(args, 'event_supervision_space', 'raw')}",
        color=ANSI_YELLOW,
        bold=True,
    )
    total_train_batches = sum(len(bundle["train_graph_loader"]) for bundle in bundles)
    for epoch in range(1, args.epochs + 1):
        train_started_at = time.perf_counter()
        model.train()
        loss_sums = {"loss": 0.0, "event_cls": 0.0, "event_rank": 0.0, "event_cumulative": 0.0}
        valid_batches = 0
        processed_batches = 0
        for bundle, batch in iter_interleaved_training_batches(bundles):
            batch = move_batch_to_device(batch, device)
            x_batch = build_graph_input(batch, bundle, use_priors=args.use_priors)
            pos_weight = None
            if args.event_occ_weighting == "auto" and bundle["train_sup_pos_count"] > 0:
                ratio = bundle["train_sup_neg_count"] / float(bundle["train_sup_pos_count"])
                pos_weight = torch.tensor(
                    min(ratio, float(args.event_occ_pos_weight_cap)),
                    device=device,
                )
            optimizer.zero_grad(set_to_none=True)
            logits = forward_graph_model(model, batch, bundle, x_batch)
            primary_logits = logits
            primary_y_event = batch["y_sup"]
            primary_y_target = batch["y"]
            if args.event_primary_horizon > 0:
                primary_logits = logits[..., :args.event_primary_horizon]
                primary_y_event = batch["y_sup"][..., :args.event_primary_horizon]
                primary_y_target = batch["y"][..., :args.event_primary_horizon]
            loss_dict = event_occurrence_loss(
                primary_logits,
                primary_y_event,
                loss_kind=args.event_loss,
                pos_weight=pos_weight,
                focal_gamma=args.event_focal_gamma,
                rank_lambda=args.event_rank_lambda,
                rank_max_pairs=args.event_rank_max_pairs,
                rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                y_target=primary_y_target,
                discrepancy_base_margin=args.event_discrepancy_base_margin,
                discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                rank_ratios=args.hr_ratio,
                boundary_margin=args.event_boundary_margin,
                sample_weight=batch.get("node_weight"),
            )
            loss = loss_dict["loss"]
            if args.event_cumulative_lambda > 0.0:
                event_cumulative = cumulative_hazard_nll_loss(
                    logits,
                    batch["y_sup"],
                    horizons=args.event_cumulative_horizons,
                    pos_weight_cap=args.event_cumulative_pos_weight_cap,
                )
                loss = loss + float(args.event_cumulative_lambda) * event_cumulative
            else:
                event_cumulative = torch.zeros((), device=loss.device, dtype=loss.dtype)
            loss_dict["loss"] = loss
            loss_dict["event_cumulative"] = event_cumulative
            if not bool(torch.isfinite(loss)):
                processed_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            for name in loss_sums:
                loss_sums[name] += float(loss_dict[name].detach().cpu().item())
            valid_batches += 1
            processed_batches += 1
            if args.progress_interval > 0 and (
                    processed_batches % args.progress_interval == 0
                    or processed_batches == total_train_batches
            ):
                print_progress(
                    f"Epoch {epoch:02d} train",
                    processed_batches,
                    total_train_batches,
                    train_started_at,
                    device,
                )

        synchronize_device(device)
        train_elapsed = time.perf_counter() - train_started_at
        resource_tracker["train_seconds"] += train_elapsed
        print_status(
            f"Epoch {epoch:02d} training completed | valid_batches={valid_batches} | "
            f"elapsed={format_elapsed(train_elapsed)}",
            color=ANSI_BLUE,
            flush=True,
        )

        validation_started_at = timed_section_start(device)
        val_metrics_by_L = None
        if skip_val_eval:
            val_metrics = {}
        elif bundles[0].get("val_loaders_by_L"):
            val_metrics_by_L = evaluate_event_by_lookback(
                model,
                bundles,
                "val",
                args,
                device,
                progress_label=f"Epoch {epoch:02d} validation",
                seen_nodes_only=validation_seen_nodes_only,
            )
            val_metrics = mean_metric_dicts(list(val_metrics_by_L.values()))
        else:
            val_metrics = evaluate_event_occurrence_model(
                model,
                bundles,
                split="val",
                device=device,
                hr_ratio=args.hr_ratio,
                use_priors=args.use_priors,
                progress_interval=args.progress_interval,
                progress_label=f"Epoch {epoch:02d} validation",
                eval_horizon=args.event_eval_horizon,
                seen_nodes_only=validation_seen_nodes_only,
            )
        resource_tracker["validation_seconds"] += timed_section_elapsed(device, validation_started_at)
        averages = {name: value / max(valid_batches, 1) for name, value in loss_sums.items()}
        epoch_entry = {
            "epoch": epoch,
            "train": dict(averages),
            "valid_batches": valid_batches,
            "val_metrics": dict(val_metrics),
        }
        if val_metrics_by_L is not None:
            epoch_entry["val_metrics_by_L"] = {str(L): dict(m) for L, m in val_metrics_by_L.items()}
        epoch_history.append(epoch_entry)
        if skip_val_eval:
            print(
                f"Epoch {epoch:02d} | train_loss={averages['loss']:.4f} | "
                f"event_cls={averages['event_cls']:.4f} | event_rank={averages['event_rank']:.4f} | "
                f"event_cumulative={averages['event_cumulative']:.4f} | val=SKIPPED (selection=last)"
            )
        else:
            print(
                f"Epoch {epoch:02d} | train_loss={averages['loss']:.4f} | "
                f"event_cls={averages['event_cls']:.4f} | event_rank={averages['event_rank']:.4f} | "
                f"event_cumulative={averages['event_cumulative']:.4f} | "
                f"{format_metrics_with_na('val', val_metrics)} | "
                f"event_bce={val_metrics['event_bce']:.4f} | event_brier={val_metrics['event_brier']:.4f}"
            )
        if val_metrics_by_L is not None:
            print(
                "  val_by_L "
                + " | ".join(
                    f"L={L}: R@20={m['es_recall_at_20']:.4f} N@20={m['es_ndcg_at_20']:.4f}"
                    for L, m in val_metrics_by_L.items()
                )
            )
        if val_metrics_by_L is not None:
            score = float(np.mean([
                checkpoint_selection_score(m, args.selection_metric, epoch)
                for m in val_metrics_by_L.values()
            ]))
        else:
            score = checkpoint_selection_score(val_metrics, args.selection_metric, epoch)
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_metrics)
            selected_epoch = epoch

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced by the event-only baseline.")
    model.load_state_dict(best_state)
    test_started_at = timed_section_start(device)
    test_metrics_by_L = None
    test_metrics_by_borough = None
    if skip_test_eval:
        test_metrics = {}
        print_status("Test evaluation SKIPPED (--skip_test_eval 1)", color=ANSI_CYAN)
    elif bundles[0].get("test_loaders_by_L"):
        test_metrics_by_L = evaluate_event_by_lookback(
            model,
            bundles,
            "test",
            args,
            device,
            progress_label="Test evaluation",
        )
        test_metrics = mean_metric_dicts(list(test_metrics_by_L.values()))
    else:
        test_metrics = evaluate_event_occurrence_model(
            model,
            bundles,
            split="test",
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            progress_interval=args.progress_interval,
            progress_label="Test evaluation",
            eval_horizon=args.event_eval_horizon,
        )
    resource_tracker["test_seconds"] += timed_section_elapsed(device, test_started_at)
    if not skip_test_eval:
        print(format_metrics_with_na("test", test_metrics))
        print(f"test | event_bce={test_metrics['event_bce']:.4f} | event_brier={test_metrics['event_brier']:.4f}")
    if test_metrics_by_L is not None:
        print(
            "  test_by_L "
            + " | ".join(
                f"L={L}: R@20={m['es_recall_at_20']:.4f} N@20={m['es_ndcg_at_20']:.4f}"
                for L, m in test_metrics_by_L.items()
            )
        )

    if len(bundles) > 1 and not skip_test_eval:
        borough_test_started_at = timed_section_start(device)
        test_metrics_by_borough = {
            bundle["name"]: evaluate_event_occurrence_model(
                model,
                [bundle],
                split="test",
                device=device,
                hr_ratio=args.hr_ratio,
                use_priors=args.use_priors,
                eval_horizon=args.event_eval_horizon,
            )
            for bundle in bundles
        }
        resource_tracker["borough_test_seconds"] += timed_section_elapsed(device, borough_test_started_at)

    ckpt_name = f"{args.model}_{args.event_loss}_best.pt"
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "selected_epoch": selected_epoch,
            "selection_metric": args.selection_metric,
            "epoch_history": epoch_history,
        },
        ckpt_path,
    )
    print_status(f"Saved checkpoint: {ckpt_path}", color=ANSI_GREEN)
    result = {
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": ckpt_path,
        "selected_epoch": selected_epoch,
        "selection_metric": args.selection_metric,
        "epoch_history": epoch_history,
        "resource_usage": finalize_resource_tracking(resource_tracker, device),
    }
    if test_metrics_by_L is not None:
        result["test_metrics_by_L"] = {str(L): m for L, m in test_metrics_by_L.items()}
    if test_metrics_by_borough is not None:
        result["test_metrics_by_borough"] = test_metrics_by_borough
    return result

def train_graph_baseline(
        model: torch.nn.Module,
        args,
        bundles: List[dict],
        device: torch.device,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resource_tracker = start_resource_tracking(model, device)
    event_supervision = args.graph_supervision == "event_bce"
    validation_seen_nodes_only = any(bundle.get("holdout_count", 0) > 0 for bundle in bundles)

    def compute_train_loss(bundle: dict, batch: dict, prediction: torch.Tensor) -> torch.Tensor:
        if not event_supervision:
            return F.mse_loss(prediction, batch["y"])
        pos_weight = None
        if args.event_occ_weighting == "auto" and bundle["train_event_pos_count"] > 0:
            ratio = bundle["train_event_neg_count"] / float(bundle["train_event_pos_count"])
            pos_weight = torch.tensor(
                min(ratio, float(args.event_occ_pos_weight_cap)),
                device=device,
            )
        return event_occurrence_loss(
            prediction,
            batch["y_event"],
            loss_kind=args.event_loss,
            pos_weight=pos_weight,
            focal_gamma=args.event_focal_gamma,
            rank_lambda=args.event_rank_lambda,
            rank_max_pairs=args.event_rank_max_pairs,
            rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
            y_target=batch["y"],
            discrepancy_base_margin=args.event_discrepancy_base_margin,
            discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
            rank_ratios=args.hr_ratio,
            boundary_margin=args.event_boundary_margin,
            sample_weight=batch.get("node_weight"),
        )["loss"]

    print_status(
        f"Graph baseline | supervision={args.graph_supervision} | "
        f"loss={args.event_loss if event_supervision else 'mse'} | selection={args.selection_metric}",
        color=ANSI_YELLOW,
        bold=True,
    )

    best_val = float("-inf")
    best_state = None
    best_val_metrics = None
    selected_epoch = None
    epoch_history = []
    total_train_batches = sum(len(bundle["train_graph_loader"]) for bundle in bundles)

    for epoch in range(1, args.epochs + 1):
        train_started_at = time.perf_counter()
        processed_batches = 0
        model.train()
        train_losses = []
        if len(bundles) > 1:
            iterators = [iter(bundle["train_graph_loader"]) for bundle in bundles]
            active_bundle_indices = list(range(len(bundles)))
            while active_bundle_indices:
                bundle_idx = random.choice(active_bundle_indices)
                bundle = bundles[bundle_idx]
                try:
                    batch = next(iterators[bundle_idx])
                except StopIteration:
                    active_bundle_indices.remove(bundle_idx)
                    continue

                batch = move_batch_to_device(batch, device)
                x_batch = build_graph_input(batch, bundle, use_priors=args.use_priors)
                optimizer.zero_grad(set_to_none=True)
                pred = forward_graph_model(model, batch, bundle, x_batch)
                loss = compute_train_loss(bundle, batch, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                processed_batches += 1
                if args.progress_interval > 0 and (
                        processed_batches % args.progress_interval == 0
                        or processed_batches == total_train_batches
                ):
                    print_progress(
                        f"Epoch {epoch:02d} train",
                        processed_batches,
                        total_train_batches,
                        train_started_at,
                        device,
                    )
        else:
            bundle = bundles[0]
            for batch in bundle["train_graph_loader"]:
                batch = move_batch_to_device(batch, device)
                x_batch = build_graph_input(batch, bundle, use_priors=args.use_priors)
                optimizer.zero_grad(set_to_none=True)
                pred = forward_graph_model(model, batch, bundle, x_batch)
                loss = compute_train_loss(bundle, batch, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                processed_batches += 1
                if args.progress_interval > 0 and (
                        processed_batches % args.progress_interval == 0
                        or processed_batches == total_train_batches
                ):
                    print_progress(
                        f"Epoch {epoch:02d} train",
                        processed_batches,
                        total_train_batches,
                        train_started_at,
                        device,
                    )

        synchronize_device(device)
        resource_tracker["train_seconds"] += time.perf_counter() - train_started_at
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        validation_started_at = timed_section_start(device)
        if event_supervision:
            val_metrics = evaluate_event_occurrence_model(
                model,
                bundles,
                split="val",
                device=device,
                hr_ratio=args.hr_ratio,
                use_priors=args.use_priors,
                progress_interval=args.progress_interval,
                progress_label=f"Epoch {epoch:02d} validation",
                seen_nodes_only=validation_seen_nodes_only,
            )
        else:
            val_metrics = evaluate_graph_model(
                model,
                bundles,
                split="val",
                device=device,
                hr_ratio=args.hr_ratio,
                use_priors=args.use_priors,
                rank_score=args.rank_score,
                zr_tau=args.zr_tau,
                progress_interval=args.progress_interval,
                progress_label=f"Epoch {epoch:02d} validation",
            )
        resource_tracker["validation_seconds"] += timed_section_elapsed(device, validation_started_at)
        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_batches": len(train_losses),
            "val_metrics": dict(val_metrics),
        })
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | {format_metrics_with_na('val', val_metrics)}")
        if event_supervision:
            print(
                f"Epoch {epoch:02d} | event_bce={val_metrics['event_bce']:.4f} | "
                f"event_brier={val_metrics['event_brier']:.4f}"
            )

        selection_score = checkpoint_selection_score(val_metrics, args.selection_metric, epoch)
        if selection_score > best_val:
            best_val = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_metrics)
            selected_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    test_started_at = timed_section_start(device)
    if event_supervision:
        test_metrics = evaluate_event_occurrence_model(
            model,
            bundles,
            split="test",
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            progress_interval=args.progress_interval,
            progress_label="Test evaluation",
        )
    else:
        test_metrics = evaluate_graph_model(
            model,
            bundles,
            split="test",
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            rank_score=args.rank_score,
            zr_tau=args.zr_tau,
            progress_interval=args.progress_interval,
            progress_label="Test evaluation",
        )
    resource_tracker["test_seconds"] += timed_section_elapsed(device, test_started_at)
    print(format_metrics_with_na("test", test_metrics))
    if event_supervision:
        print(f"test | event_bce={test_metrics['event_bce']:.4f} | event_brier={test_metrics['event_brier']:.4f}")

    test_metrics_by_borough = None
    if len(bundles) > 1:
        borough_test_started_at = timed_section_start(device)
        if event_supervision:
            test_metrics_by_borough = {
                bundle["name"]: evaluate_event_occurrence_model(
                    model,
                    [bundle],
                    split="test",
                    device=device,
                    hr_ratio=args.hr_ratio,
                    use_priors=args.use_priors,
                )
                for bundle in bundles
            }
        else:
            test_metrics_by_borough = evaluate_graph_model_by_borough(
                model=model,
                bundles=bundles,
                split="test",
                device=device,
                hr_ratio=args.hr_ratio,
                use_priors=args.use_priors,
                rank_score=args.rank_score,
                zr_tau=args.zr_tau,
            )
        resource_tracker["borough_test_seconds"] += timed_section_elapsed(device, borough_test_started_at)
        for borough_name, borough_metrics in test_metrics_by_borough.items():
            print(format_metrics_with_na(f"test [{borough_name}]", borough_metrics))

    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "selected_epoch": selected_epoch,
            "selection_metric": args.selection_metric,
            "epoch_history": epoch_history,
        },
        ckpt_path,
    )
    print_status(f"Saved checkpoint: {ckpt_path}", color=ANSI_GREEN)
    result = {
        "val_metrics": best_val_metrics if best_val_metrics is not None else val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": ckpt_path,
        "selected_epoch": selected_epoch,
        "selection_metric": args.selection_metric,
        "epoch_history": epoch_history,
        "resource_usage": finalize_resource_tracking(resource_tracker, device),
    }
    if test_metrics_by_borough is not None:
        result["test_metrics_by_borough"] = test_metrics_by_borough
    return result


def train_probabilistic_baseline(
        model: torch.nn.Module,
        args,
        bundles: List[dict],
        device: torch.device,
        model_kind: str,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    resource_tracker = start_resource_tracking(model, device)
    best_val = float("-inf")
    best_state = None
    best_val_metrics = None
    selected_epoch = None
    epoch_history = []
    hurdle_pos_weight = None
    if model_kind in {"hurdle_gamma", "hurdle_lognormal", "hurdle_tweedie"} and args.hurdle_occ_pos_weight > 0:
        hurdle_pos_weight = torch.tensor(float(args.hurdle_occ_pos_weight), device=device)
    if model_kind == "hurdle_tweedie":
        print(
            "Note: st_hurdle_tweedie test intervals now use a deterministic Gamma moment-matching approximation."
        )
    hurdle_event_aux = model_kind in {"hurdle_lognormal", "hurdle_gamma"} and args.hurdle_event_occ_lambda > 0
    hurdle_event_weighting = (
        model_kind in {"hurdle_lognormal", "hurdle_gamma"}
        and args.hurdle_event_occ_weighting == "auto"
        and hurdle_event_aux
    )
    daer_event_weighting = (
        model_kind in DAER_MODEL_KINDS
        and args.daer_event_occ_weighting == "auto"
        and args.daer_event_lambda > 0.0
    )
    event_aligned_event_weighting = (
        model_kind in EVENT_ALIGNED_MODEL_KINDS
        and args.event_aligned_event_occ_weighting == "auto"
        and args.event_aligned_event_lambda > 0.0
    )
    if hurdle_event_aux:
        print(
            f"Note: st_{model_kind} adds an auxiliary event-space BCE on z_event = 1[raw risk > 0] "
            f"with lambda={args.hurdle_event_occ_lambda:.3f}."
        )
    if hurdle_event_weighting:
        print(
            f"Note: st_{model_kind} auxiliary event-space BCE uses auto pos_weight "
            f"with cap={args.hurdle_event_occ_pos_weight_cap:.2f}."
        )
    if model_kind in {
        "hurdle_gamma",
        "hurdle_lognormal",
        "daer_hurdle_gamma",
        "daer_hurdle_lognormal",
        "event_aligned_hurdle_gamma",
        "event_aligned_hurdle_lognormal",
    }:
        print_status(
            f"Hurdle positive-loss normalization={args.hurdle_positive_normalization}",
            color=ANSI_YELLOW,
        )
    if model_kind in DAER_MODEL_KINDS:
        print_status(
            "DAER-HG objective | "
            f"event_lambda={args.daer_event_lambda:.3f} | event_loss={args.event_loss} | "
            f"rank_branch={args.daer_rank_branch}",
            color=ANSI_YELLOW,
        )
    if model_kind in EVENT_ALIGNED_MODEL_KINDS:
        print_status(
            "Event-aligned Hurdle objective | single occurrence branch | "
            f"event_lambda={args.event_aligned_event_lambda:.3f} | event_loss={args.event_loss} | "
            "ES ranking=q_logit",
            color=ANSI_YELLOW,
        )

    total_train_batches = sum(len(bundle["train_graph_loader"]) for bundle in bundles)
    for epoch in range(1, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_started_at = time.perf_counter()
        processed_train_batches = 0
        model.train()
        train_loss_sum = None
        valid_batch_count = 0
        skipped_batches = 0
        stat_store: Dict[str, dict] = {}
        print_status(
            f"Epoch {epoch:02d} training started | batches={total_train_batches} | device={device}",
            color=ANSI_BLUE,
            bold=True,
            flush=True,
        )

        def complete_train_batch():
            nonlocal processed_train_batches
            processed_train_batches += 1
            if args.progress_interval > 0 and (
                    processed_train_batches % args.progress_interval == 0
                    or processed_train_batches == total_train_batches
            ):
                print_progress(
                    f"Epoch {epoch:02d} train",
                    processed_train_batches,
                    total_train_batches,
                    train_started_at,
                    device,
                )

        if len(bundles) > 1:
            iterators = [iter(bundle["train_graph_loader"]) for bundle in bundles]
            active_bundle_indices = list(range(len(bundles)))
            while active_bundle_indices:
                bundle_idx = random.choice(active_bundle_indices)
                bundle = bundles[bundle_idx]
                try:
                    batch = next(iterators[bundle_idx])
                except StopIteration:
                    active_bundle_indices.remove(bundle_idx)
                    continue

                batch = move_batch_to_device(batch, device)
                x_batch = build_graph_input(batch, bundle, use_priors=args.use_priors)
                y_batch = batch["y"]
                z_occ_batch = (batch["y_event"] > 0).float() if hurdle_event_aux else None
                hurdle_event_pos_weight = None
                if hurdle_event_weighting or daer_event_weighting or event_aligned_event_weighting:
                    pos_count = bundle["train_event_pos_count"]
                    neg_count = bundle["train_event_neg_count"]
                    if pos_count > 0:
                        weight_cap = (
                            args.daer_event_occ_pos_weight_cap
                            if daer_event_weighting
                            else (
                                args.event_aligned_event_occ_pos_weight_cap
                                if event_aligned_event_weighting
                                else args.hurdle_event_occ_pos_weight_cap
                            )
                        )
                        weight_value = min(neg_count / float(pos_count), float(weight_cap))
                        hurdle_event_pos_weight = torch.tensor(weight_value, device=device)
                optimizer.zero_grad(set_to_none=True)
                outputs = forward_graph_model(model, batch, bundle, x_batch)
                outputs_finite = torch.stack([torch.isfinite(out).all() for out in outputs]).all()
                if not bool(outputs_finite):
                    skipped_batches += 1
                    complete_train_batch()
                    continue
                if model_kind == "gaussian":
                    loss = gaussian_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "gamma":
                    loss = gamma_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "lognormal":
                    loss = lognormal_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "hurdle_gamma":
                    loss_dict = hurdle_gamma_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        z_occ=z_occ_batch,
                        event_occ_lambda=args.hurdle_event_occ_lambda,
                        pos_weight=hurdle_pos_weight,
                        event_pos_weight=hurdle_event_pos_weight,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "hurdle_lognormal":
                    loss_dict = hurdle_lognormal_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        z_occ=z_occ_batch,
                        event_occ_lambda=args.hurdle_event_occ_lambda,
                        pos_weight=hurdle_pos_weight,
                        event_pos_weight=hurdle_event_pos_weight,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "event_aligned_hurdle_gamma":
                    loss_dict = event_aligned_hurdle_gamma_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_logits=outputs[0],
                        mu=outputs[1],
                        alpha=outputs[2],
                        event_lambda=args.event_aligned_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "event_aligned_hurdle_lognormal":
                    loss_dict = event_aligned_hurdle_lognormal_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_logits=outputs[0],
                        mu_log=outputs[1],
                        sigma_log=outputs[2],
                        event_lambda=args.event_aligned_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "daer_hurdle_gamma":
                    loss_dict = daer_hurdle_gamma_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_target_logits=outputs[0],
                        q_event_logits=outputs[1],
                        mu=outputs[2],
                        alpha=outputs[3],
                        event_lambda=args.daer_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "daer_hurdle_lognormal":
                    loss_dict = daer_hurdle_lognormal_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_target_logits=outputs[0],
                        q_event_logits=outputs[1],
                        mu_log=outputs[2],
                        sigma_log=outputs[3],
                        event_lambda=args.daer_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "hurdle_tweedie":
                    loss_dict = hurdle_tweedie_conditional_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        outputs[3],
                        pos_weight=hurdle_pos_weight,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "nb":
                    loss = negative_binomial_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "zinb":
                    loss = zero_inflated_nb_nll_loss(y_batch, outputs[0], outputs[1], outputs[2])
                elif model_kind == "tweedie":
                    loss = tweedie_nll_loss(y_batch, outputs[0], outputs[1], outputs[2])
                elif model_kind == "zitd":
                    loss = zero_inflated_tweedie_nll_loss(y_batch, outputs[0], outputs[1], outputs[2], outputs[3])
                else:
                    raise ValueError(f"Unsupported probabilistic model kind: {model_kind}")
                if not bool(torch.isfinite(loss)):
                    skipped_batches += 1
                    complete_train_batch()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                record_prob_output_stats(stat_store, outputs, model_kind)
                if model_kind in {"hurdle_gamma", "hurdle_lognormal", "hurdle_tweedie"}:
                    accumulate_prob_stat(stat_store, "occ_nll", loss_dict["occ_nll"])
                    accumulate_prob_stat(stat_store, "pos_nll", loss_dict["pos_nll"])
                elif model_kind in EVENT_ALIGNED_MODEL_KINDS:
                    for stat_name in (
                            "ts_loss",
                            "event_cls",
                            "event_rank",
                            "occ_nll",
                            "pos_nll",
                    ):
                        accumulate_prob_stat(stat_store, stat_name, loss_dict[stat_name])
                elif model_kind in DAER_MODEL_KINDS:
                    for stat_name in (
                            "ts_loss",
                            "event_cls",
                            "event_rank",
                    ):
                        accumulate_prob_stat(stat_store, stat_name, loss_dict[stat_name])
                detached_loss = loss.detach()
                train_loss_sum = detached_loss if train_loss_sum is None else train_loss_sum + detached_loss
                valid_batch_count += 1
                complete_train_batch()
        else:
            bundle = bundles[0]
            for batch in bundle["train_graph_loader"]:
                batch = move_batch_to_device(batch, device)
                x_batch = build_graph_input(batch, bundle, use_priors=args.use_priors)
                y_batch = batch["y"]
                z_occ_batch = (batch["y_event"] > 0).float() if hurdle_event_aux else None
                hurdle_event_pos_weight = None
                if hurdle_event_weighting or daer_event_weighting or event_aligned_event_weighting:
                    pos_count = bundle["train_event_pos_count"]
                    neg_count = bundle["train_event_neg_count"]
                    if pos_count > 0:
                        weight_cap = (
                            args.daer_event_occ_pos_weight_cap
                            if daer_event_weighting
                            else (
                                args.event_aligned_event_occ_pos_weight_cap
                                if event_aligned_event_weighting
                                else args.hurdle_event_occ_pos_weight_cap
                            )
                        )
                        weight_value = min(neg_count / float(pos_count), float(weight_cap))
                        hurdle_event_pos_weight = torch.tensor(weight_value, device=device)
                optimizer.zero_grad(set_to_none=True)
                outputs = forward_graph_model(model, batch, bundle, x_batch)
                outputs_finite = torch.stack([torch.isfinite(out).all() for out in outputs]).all()
                if not bool(outputs_finite):
                    skipped_batches += 1
                    complete_train_batch()
                    continue
                if model_kind == "gaussian":
                    loss = gaussian_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "gamma":
                    loss = gamma_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "lognormal":
                    loss = lognormal_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "hurdle_gamma":
                    loss_dict = hurdle_gamma_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        z_occ=z_occ_batch,
                        event_occ_lambda=args.hurdle_event_occ_lambda,
                        pos_weight=hurdle_pos_weight,
                        event_pos_weight=hurdle_event_pos_weight,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "hurdle_lognormal":
                    loss_dict = hurdle_lognormal_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        z_occ=z_occ_batch,
                        event_occ_lambda=args.hurdle_event_occ_lambda,
                        pos_weight=hurdle_pos_weight,
                        event_pos_weight=hurdle_event_pos_weight,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "event_aligned_hurdle_gamma":
                    loss_dict = event_aligned_hurdle_gamma_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_logits=outputs[0],
                        mu=outputs[1],
                        alpha=outputs[2],
                        event_lambda=args.event_aligned_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "event_aligned_hurdle_lognormal":
                    loss_dict = event_aligned_hurdle_lognormal_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_logits=outputs[0],
                        mu_log=outputs[1],
                        sigma_log=outputs[2],
                        event_lambda=args.event_aligned_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "daer_hurdle_gamma":
                    loss_dict = daer_hurdle_gamma_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_target_logits=outputs[0],
                        q_event_logits=outputs[1],
                        mu=outputs[2],
                        alpha=outputs[3],
                        event_lambda=args.daer_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "daer_hurdle_lognormal":
                    loss_dict = daer_hurdle_lognormal_loss(
                        y_target=y_batch,
                        y_event=batch["y_event"],
                        q_target_logits=outputs[0],
                        q_event_logits=outputs[1],
                        mu_log=outputs[2],
                        sigma_log=outputs[3],
                        event_lambda=args.daer_event_lambda,
                        event_pos_weight=hurdle_event_pos_weight,
                        event_loss_kind=args.event_loss,
                        event_focal_gamma=args.event_focal_gamma,
                        event_rank_lambda=args.event_rank_lambda,
                        event_rank_max_pairs=args.event_rank_max_pairs,
                        event_rank_hard_negative_fraction=args.event_rank_hard_negative_fraction,
                        event_discrepancy_base_margin=args.event_discrepancy_base_margin,
                        event_discrepancy_target_margin_scale=args.event_discrepancy_target_margin_scale,
                        event_rank_ratios=args.hr_ratio,
                        event_boundary_margin=args.event_boundary_margin,
                        positive_loss_normalization=args.hurdle_positive_normalization,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "hurdle_tweedie":
                    loss_dict = hurdle_tweedie_conditional_nll_loss(
                        y_batch,
                        outputs[0],
                        outputs[1],
                        outputs[2],
                        outputs[3],
                        pos_weight=hurdle_pos_weight,
                    )
                    loss = loss_dict["loss"]
                elif model_kind == "nb":
                    loss = negative_binomial_nll_loss(y_batch, outputs[0], outputs[1])
                elif model_kind == "zinb":
                    loss = zero_inflated_nb_nll_loss(y_batch, outputs[0], outputs[1], outputs[2])
                elif model_kind == "tweedie":
                    loss = tweedie_nll_loss(y_batch, outputs[0], outputs[1], outputs[2])
                elif model_kind == "zitd":
                    loss = zero_inflated_tweedie_nll_loss(y_batch, outputs[0], outputs[1], outputs[2], outputs[3])
                else:
                    raise ValueError(f"Unsupported probabilistic model kind: {model_kind}")
                if not bool(torch.isfinite(loss)):
                    skipped_batches += 1
                    complete_train_batch()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)
                optimizer.step()
                record_prob_output_stats(stat_store, outputs, model_kind)
                if model_kind in {"hurdle_gamma", "hurdle_lognormal", "hurdle_tweedie"}:
                    accumulate_prob_stat(stat_store, "occ_nll", loss_dict["occ_nll"])
                    accumulate_prob_stat(stat_store, "pos_nll", loss_dict["pos_nll"])
                elif model_kind in EVENT_ALIGNED_MODEL_KINDS:
                    for stat_name in (
                            "ts_loss",
                            "event_cls",
                            "event_rank",
                            "occ_nll",
                            "pos_nll",
                    ):
                        accumulate_prob_stat(stat_store, stat_name, loss_dict[stat_name])
                elif model_kind in DAER_MODEL_KINDS:
                    for stat_name in (
                            "ts_loss",
                            "event_cls",
                            "event_rank",
                    ):
                        accumulate_prob_stat(stat_store, stat_name, loss_dict[stat_name])
                detached_loss = loss.detach()
                train_loss_sum = detached_loss if train_loss_sum is None else train_loss_sum + detached_loss
                valid_batch_count += 1
                complete_train_batch()

        synchronize_device(device)
        train_elapsed = time.perf_counter() - train_started_at
        resource_tracker["train_seconds"] += train_elapsed
        print_status(
            f"Epoch {epoch:02d} training completed | valid_batches={valid_batch_count} | "
            f"skipped_batches={skipped_batches} | elapsed={format_elapsed(train_elapsed)}",
            color=ANSI_BLUE,
            flush=True,
        )
        train_loss = (
            float((train_loss_sum / valid_batch_count).cpu().item())
            if train_loss_sum is not None and valid_batch_count > 0
            else float("nan")
        )
        print_status(
            format_prob_output_stats(stat_store, model_kind, skipped_batches),
            color=ANSI_YELLOW,
        )
        validation_started_at = timed_section_start(device)
        val_metrics = evaluate_probabilistic_model(
            model=model,
            bundles=bundles,
            split="val",
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            model_kind=model_kind,
            interval_alpha=args.interval_alpha,
            rank_score=args.rank_score,
            zr_tau=args.zr_tau,
            interval_samples=(
                args.mc_interval_samples
                if model_kind in {
                    "hurdle_gamma", "hurdle_lognormal", "event_aligned_hurdle_gamma",
                    "event_aligned_hurdle_lognormal", "daer_hurdle_gamma", "daer_hurdle_lognormal",
                    "hurdle_tweedie"
                }
                else args.tweedie_interval_samples
            ),
            interval_seed=args.seed,
            daer_rank_branch=args.daer_rank_branch,
            compute_intervals=(
                model_kind not in {
                    "hurdle_gamma", "hurdle_lognormal", "event_aligned_hurdle_gamma",
                    "event_aligned_hurdle_lognormal", "daer_hurdle_gamma", "daer_hurdle_lognormal",
                    "hurdle_tweedie"
                }
            ),
            progress_interval=args.progress_interval,
            progress_label=f"Epoch {epoch:02d} validation",
        )
        resource_tracker["validation_seconds"] += timed_section_elapsed(device, validation_started_at)
        epoch_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_batches": valid_batch_count,
            "skipped_batches": skipped_batches,
            "val_metrics": dict(val_metrics),
        })
        print(
            f"{color_text(f'Epoch {epoch:02d} | train_loss={train_loss:.4f}', ANSI_GREEN, bold=True)} | "
            f"{format_metrics_with_na('val', val_metrics)}"
        )

        selection_score = checkpoint_selection_score(val_metrics, args.selection_metric, epoch)
        if selection_score > best_val:
            best_val = selection_score
            best_state = copy.deepcopy(model.state_dict())
            best_val_metrics = dict(val_metrics)
            selected_epoch = epoch
        synchronize_device(device)
        print_status(
            f"Epoch {epoch:02d} completed | total_elapsed="
            f"{format_elapsed(time.perf_counter() - epoch_started_at)}",
            color=ANSI_GREEN,
            bold=True,
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_started_at = timed_section_start(device)
    test_metrics = evaluate_probabilistic_model(
        model=model,
        bundles=bundles,
        split="test",
        device=device,
        hr_ratio=args.hr_ratio,
        use_priors=args.use_priors,
        model_kind=model_kind,
        interval_alpha=args.interval_alpha,
        rank_score=args.rank_score,
        zr_tau=args.zr_tau,
        interval_samples=(
            args.mc_interval_samples
            if model_kind in {
                "hurdle_gamma", "hurdle_lognormal", "event_aligned_hurdle_gamma",
                "event_aligned_hurdle_lognormal", "daer_hurdle_gamma", "daer_hurdle_lognormal",
                "hurdle_tweedie"
            }
            else args.tweedie_interval_samples
        ),
        interval_seed=args.seed,
        daer_rank_branch=args.daer_rank_branch,
        compute_intervals=True,
        progress_interval=args.progress_interval,
        progress_label="Test evaluation",
    )
    resource_tracker["test_seconds"] += timed_section_elapsed(device, test_started_at)
    print(format_metrics_with_na("test", test_metrics))

    test_metrics_by_borough = None
    if len(bundles) > 1:
        borough_test_started_at = timed_section_start(device)
        test_metrics_by_borough = evaluate_probabilistic_model_by_borough(
            model=model,
            bundles=bundles,
            split="test",
            device=device,
            hr_ratio=args.hr_ratio,
            use_priors=args.use_priors,
            model_kind=model_kind,
            interval_alpha=args.interval_alpha,
            rank_score=args.rank_score,
            zr_tau=args.zr_tau,
            interval_samples=(
                args.mc_interval_samples
                if model_kind in {
                    "hurdle_gamma", "hurdle_lognormal", "event_aligned_hurdle_gamma",
                    "event_aligned_hurdle_lognormal", "daer_hurdle_gamma", "daer_hurdle_lognormal",
                    "hurdle_tweedie"
                }
                else args.tweedie_interval_samples
            ),
            interval_seed=args.seed,
            daer_rank_branch=args.daer_rank_branch,
            compute_intervals=True,
        )
        resource_tracker["borough_test_seconds"] += timed_section_elapsed(device, borough_test_started_at)
        for borough_name, borough_metrics in test_metrics_by_borough.items():
            print(format_metrics_with_na(f"test [{borough_name}]", borough_metrics))

    checkpoint_suffix = (
        f"_{args.event_loss}"
        if model_kind in DAER_MODEL_KINDS | EVENT_ALIGNED_MODEL_KINDS
        else ""
    )
    if (
        model_kind
        in {
            "hurdle_gamma",
            "hurdle_lognormal",
            "daer_hurdle_gamma",
            "daer_hurdle_lognormal",
            "event_aligned_hurdle_gamma",
            "event_aligned_hurdle_lognormal",
        }
        and args.hurdle_positive_normalization != "standard"
    ):
        checkpoint_suffix += f"_{args.hurdle_positive_normalization}"
    ckpt_path = os.path.join(args.out_dir, f"{args.model}{checkpoint_suffix}_best.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "selected_epoch": selected_epoch,
            "selection_metric": args.selection_metric,
            "epoch_history": epoch_history,
        },
        ckpt_path,
    )
    print_status(f"Saved checkpoint: {ckpt_path}", color=ANSI_GREEN)
    result = {
        "val_metrics": best_val_metrics if best_val_metrics is not None else val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": ckpt_path,
        "selected_epoch": selected_epoch,
        "selection_metric": args.selection_metric,
        "epoch_history": epoch_history,
        "resource_usage": finalize_resource_tracking(resource_tracker, device),
    }
    if test_metrics_by_borough is not None:
        result["test_metrics_by_borough"] = test_metrics_by_borough
    return result


def run_stgcn_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STGCNBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        cheb_order=args.cheb_order,
        dropout=args.dropout,
        output_activation=args.graph_output_activation,
    ).to(device=device)
    return train_graph_baseline(model, args, bundles, device)


def run_stgat_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STGATBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        heads=args.gat_heads,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        output_activation=args.graph_output_activation,
    ).to(device=device)
    return train_graph_baseline(model, args, bundles, device)


def run_dcrnn_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = DCRNNBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        diffusion_steps=args.diffusion_steps,
        dropout=args.dropout,
        output_activation=args.graph_output_activation,
    ).to(device=device)
    return train_graph_baseline(model, args, bundles, device)


def run_stg_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STGaussianGNNBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        history_len=args.history_len,
        kernel_size=args.kernel_size,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        min_scale=args.gauss_min_scale,
        max_scale=args.gauss_max_scale,
        max_loc=args.gauss_max_loc,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="gaussian")


def run_st_event_occurrence_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STEventOccurrenceBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
    ).to(device=device)
    return train_event_occurrence_baseline(model, args, bundles, device)



def run_st_hurdle_gamma_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STHurdleGammaBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_mu=args.hurdle_max_mu,
        max_alpha=args.hurdle_gamma_max_alpha,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="hurdle_gamma")


def run_st_event_aligned_hurdle_gamma_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STHurdleGammaBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_mu=args.hurdle_max_mu,
        max_alpha=args.hurdle_gamma_max_alpha,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="event_aligned_hurdle_gamma")


def run_st_daer_hurdle_gamma_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STDAERHurdleGammaBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_mu=args.hurdle_max_mu,
        max_alpha=args.hurdle_gamma_max_alpha,
    ).to(device=device)
    return train_probabilistic_baseline(
        model,
        args,
        bundles,
        device,
        model_kind="daer_hurdle_gamma",
    )



def run_st_gamma_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STGammaBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_mu=args.hurdle_max_mu,
        max_alpha=args.hurdle_gamma_max_alpha,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="gamma")


def run_st_lognormal_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STLogNormalBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_sigma=args.hurdle_lognormal_max_sigma,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="lognormal")


def run_st_hurdle_lognormal_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STHurdleLogNormalBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_sigma=args.hurdle_lognormal_max_sigma,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="hurdle_lognormal")


def run_st_event_aligned_hurdle_lognormal_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STHurdleLogNormalBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_sigma=args.hurdle_lognormal_max_sigma,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="event_aligned_hurdle_lognormal")


def run_st_daer_hurdle_lognormal_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STDAERHurdleLogNormalBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_sigma=args.hurdle_lognormal_max_sigma,
    ).to(device=device)
    return train_probabilistic_baseline(
        model,
        args,
        bundles,
        device,
        model_kind="daer_hurdle_lognormal",
    )


def run_st_hurdle_tweedie_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STHurdleTweedieBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        kernel_size=args.kernel_size,
        temp_layers=args.hurdle_temp_layers,
        spatial_layers=args.hurdle_spatial_layers,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        mu_max=args.tweedie_mu_max,
        phi_min=args.tweedie_phi_min,
        phi_max=args.tweedie_phi_max,
        rho_eps=args.tweedie_rho_eps,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="hurdle_tweedie")


def run_stnb_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STNBGNNBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        history_len=args.history_len,
        kernel_size=args.kernel_size,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_n=args.count_max_n,
        min_prob=args.count_min_prob,
        max_prob=args.count_max_prob,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="nb")


def run_stzinb_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STZINBBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        history_len=args.history_len,
        kernel_size=args.kernel_size,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        max_n=args.count_max_n,
        min_prob=args.count_min_prob,
        max_prob=args.count_max_prob,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="zinb")


def run_sttd_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STTDBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        history_len=args.history_len,
        kernel_size=args.kernel_size,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        mu_max=args.tweedie_mu_max,
        phi_min=args.tweedie_phi_min,
        phi_max=args.tweedie_phi_max,
        rho_eps=args.tweedie_rho_eps,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="tweedie")


def run_stzitd_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    model = STZITDBaseline(
        input_dim=get_graph_input_dim(bundles[0], use_priors=args.use_priors),
        hidden_dim=args.hidden_dim,
        horizon=args.horizon,
        history_len=args.history_len,
        kernel_size=args.kernel_size,
        diffusion_order=args.diffusion_steps,
        dropout=args.dropout,
        temporal_backbone=args.temporal_backbone,
        spatial_backbone=args.spatial_backbone,
        gat_heads=args.gat_heads,
        mu_max=args.tweedie_mu_max,
        phi_min=args.tweedie_phi_min,
        phi_max=args.tweedie_phi_max,
        rho_eps=args.tweedie_rho_eps,
        min_prob=args.count_min_prob,
        max_prob=args.count_max_prob,
    ).to(device=device)
    return train_probabilistic_baseline(model, args, bundles, device, model_kind="zitd")


def predict_ha_dataset(bundle: dict, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = bundle[f"{split}_ha_dataset"]
    preds, y_true, y_rank, y_event, y_event_rank = [], [], [], [], []
    for sample in dataset:
        y_hist = to_numpy_block(sample["y_hist"]).squeeze(-1)
        mean_pred = y_hist.mean(axis=1, keepdims=True)
        pred = np.repeat(mean_pred, repeats=dataset.p, axis=1).astype(np.float32)
        preds.append(pred)
        y_true.append(to_numpy_block(sample["y"]))
        y_rank.append(to_numpy_block(sample["y_rank"]))
        y_event.append(to_numpy_block(sample["y_event"]))
        y_event_rank.append(to_numpy_block(sample["y_event_rank"]))
    return (
        np.stack(y_true, axis=0),
        np.stack(y_rank, axis=0),
        np.stack(y_event, axis=0),
        np.stack(y_event_rank, axis=0),
        np.stack(preds, axis=0),
    )

# HA
def run_ha_baseline(args, bundles: List[dict]) -> dict:
    started_at = time.perf_counter()
    if args.use_priors:
        print("HA ignores --use_priors and still uses only Y_hist.")
    val_started_at = time.perf_counter()
    val_groups = [predict_ha_dataset(bundle, "val") for bundle in bundles]
    val_metrics = evaluate_prediction_groups(
        val_groups,
        hr_ratio=args.hr_ratio,
        rank_score=args.rank_score,
        zr_tau=args.zr_tau,
    )
    val_seconds = time.perf_counter() - val_started_at
    test_started_at = time.perf_counter()
    test_groups = [predict_ha_dataset(bundle, "test") for bundle in bundles]
    test_metrics = evaluate_prediction_groups(
        test_groups,
        hr_ratio=args.hr_ratio,
        rank_score=args.rank_score,
        zr_tau=args.zr_tau,
    )
    test_seconds = time.perf_counter() - test_started_at
    print(format_metrics_with_na("val", val_metrics))
    print(format_metrics_with_na("test", test_metrics))
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": "",
        "resource_usage": {
            "train_seconds": 0.0,
            "validation_seconds": val_seconds,
            "test_seconds": test_seconds,
            "borough_test_seconds": 0.0,
            "total_parameters": None,
            "trainable_parameters": None,
            "wall_seconds": time.perf_counter() - started_at,
            "peak_cuda_memory_bytes": None,
        },
    }


def allocate_xgb_capacities(bundles: List[dict], total_cap: int) -> List[Optional[int]]:
    if total_cap <= 0:
        return [None for _ in bundles]

    bundle_points = [len(bundle["train_xgb_dataset"]) * bundle["num_nodes"] for bundle in bundles]
    total_points = max(sum(bundle_points), 1)
    caps = [max(1, int(round(total_cap * points / total_points))) for points in bundle_points]
    diff = total_cap - sum(caps)
    if diff != 0:
        caps[-1] += diff
    return [max(1, cap) for cap in caps]


def build_xgb_training_data(
        bundle: dict,
        cap: Optional[int],
        rng: np.random.Generator,
        use_priors: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    dataset = bundle["train_xgb_dataset"]
    num_windows = len(dataset)
    nodes_per_window = bundle["num_nodes"]
    if cap is not None:
        nodes_per_window = max(1, min(bundle["num_nodes"], int(math.ceil(cap / max(num_windows, 1)))))

    features, targets = [], []
    for idx in range(num_windows):
        sample = dataset[idx]
        y_np = to_numpy_block(sample["y"])

        if nodes_per_window < bundle["num_nodes"]:
            node_idx = rng.choice(bundle["num_nodes"], size=nodes_per_window, replace=False)
            y_np = y_np[node_idx]
            sample = {key: value[node_idx] for key, value in sample.items()}

        features.append(build_xgb_features(sample, bundle, use_priors=use_priors))
        targets.append(y_np.astype(np.float32, copy=False))

    x_all = np.concatenate(features, axis=0)
    y_all = np.concatenate(targets, axis=0)

    if cap is not None and x_all.shape[0] > cap:
        keep_idx = rng.choice(x_all.shape[0], size=cap, replace=False)
        x_all = x_all[keep_idx]
        y_all = y_all[keep_idx]
    return x_all, y_all


def predict_xgb_dataset(
        bundle: dict,
        split: str,
        models: List[XGBRegressor],
        use_priors: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset = bundle[f"{split}_xgb_dataset"]
    preds, y_true, y_rank, y_event, y_event_rank = [], [], [], [], []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        features = build_xgb_features(sample, bundle, use_priors=use_priors)
        pred_steps = [model.predict(features) for model in models]
        pred = np.stack(pred_steps, axis=1).astype(np.float32, copy=False)
        pred = np.clip(pred, a_min=0.0, a_max=None)
        preds.append(pred)
        y_true.append(to_numpy_block(sample["y"]))
        y_rank.append(to_numpy_block(sample["y_rank"]))
        y_event.append(to_numpy_block(sample["y_event"]))
        y_event_rank.append(to_numpy_block(sample["y_event_rank"]))
    return (
        np.stack(y_true, axis=0),
        np.stack(y_rank, axis=0),
        np.stack(y_event, axis=0),
        np.stack(y_event_rank, axis=0),
        np.stack(preds, axis=0),
    )

# xgboost
def run_xgboost_baseline(args, bundles: List[dict]) -> dict:
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed in the current environment.")

    started_at = time.perf_counter()
    train_started_at = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    caps = allocate_xgb_capacities(bundles, args.xgb_train_max_samples)

    x_train_parts, y_train_parts = [], []
    for bundle, cap in zip(bundles, caps):
        x_part, y_part = build_xgb_training_data(bundle, cap=cap, rng=rng, use_priors=args.use_priors)
        x_train_parts.append(x_part)
        y_train_parts.append(y_part)

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    print(f"XGBoost training samples: {x_train.shape[0]} | feature_dim: {x_train.shape[1]}")

    models = []
    for horizon_idx in range(args.horizon):
        reg = XGBRegressor(
            objective=args.xgb_objective,
            n_estimators=args.xgb_n_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            tree_method="hist",
            n_jobs=args.xgb_n_jobs,
            random_state=args.seed + horizon_idx,
        )
        reg.fit(x_train, y_train[:, horizon_idx])
        models.append(reg)
    train_seconds = time.perf_counter() - train_started_at

    val_started_at = time.perf_counter()
    val_groups = [predict_xgb_dataset(bundle, "val", models, use_priors=args.use_priors) for bundle in bundles]
    val_metrics = evaluate_prediction_groups(
        val_groups,
        hr_ratio=args.hr_ratio,
        rank_score=args.rank_score,
        zr_tau=args.zr_tau,
    )
    validation_seconds = time.perf_counter() - val_started_at
    test_started_at = time.perf_counter()
    test_groups = [predict_xgb_dataset(bundle, "test", models, use_priors=args.use_priors) for bundle in bundles]
    test_metrics = evaluate_prediction_groups(
        test_groups,
        hr_ratio=args.hr_ratio,
        rank_score=args.rank_score,
        zr_tau=args.zr_tau,
    )
    test_seconds = time.perf_counter() - test_started_at

    print(format_metrics_with_na("val", val_metrics))
    print(format_metrics_with_na("test", test_metrics))

    artifact_path = os.path.join(args.out_dir, f"{args.model}_models.joblib")
    joblib.dump({"models": models, "args": vars(args)}, artifact_path)
    print(f"Saved XGBoost models: {artifact_path}")
    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifact_path": artifact_path,
        "resource_usage": {
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "test_seconds": test_seconds,
            "borough_test_seconds": 0.0,
            "total_parameters": None,
            "trainable_parameters": None,
            "wall_seconds": time.perf_counter() - started_at,
            "peak_cuda_memory_bytes": None,
        },
    }


def aggregate_metric_dicts(metric_dicts: List[dict]) -> dict:
    keys = sorted({key for metrics in metric_dicts for key in metrics.keys()})
    aggregated = {}
    for key in keys:
        raw_values = [metrics.get(key, float("nan")) for metrics in metric_dicts]
        finite_values = [
            float(value)
            for value in raw_values
            if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value)
        ]
        if finite_values:
            values_arr = np.asarray(finite_values, dtype=np.float64)
            variance = float(np.var(values_arr, ddof=1)) if values_arr.size > 1 else 0.0
            aggregated[key] = {
                "mean": float(np.mean(values_arr)),
                "variance": variance,
                "count": int(values_arr.size),
                "values": finite_values,
            }
        else:
            aggregated[key] = {
                "mean": float("nan"),
                "variance": float("nan"),
                "count": 0,
                "values": [],
            }
    return aggregated


def aggregate_borough_metric_dicts(results: List[dict]) -> Optional[dict]:
    borough_names = sorted({
        borough_name
        for result in results
        for borough_name in result.get("test_metrics_by_borough", {}).keys()
    })
    if not borough_names:
        return None

    aggregated = {}
    for borough_name in borough_names:
        borough_runs = [
            result["test_metrics_by_borough"][borough_name]
            for result in results
            if borough_name in result.get("test_metrics_by_borough", {})
        ]
        aggregated[borough_name] = aggregate_metric_dicts(borough_runs)
    return aggregated


def aggregated_means_for_display(metrics: dict) -> dict:
    return {
        key: value.get("mean", float("nan")) if isinstance(value, dict) else value
        for key, value in metrics.items()
    }


def aggregate_repeated_results(results: List[dict], seeds: List[int]) -> dict:
    aggregated = {
        "val_metrics": aggregate_metric_dicts([result["val_metrics"] for result in results]),
        "test_metrics": aggregate_metric_dicts([result["test_metrics"] for result in results]),
        "artifact_path": results[-1].get("artifact_path", ""),
        "repeat": {
            "num_runs": len(results),
            "seeds": seeds,
            "variance_ddof": 1,
        },
        "runs": [
            {
                "run_index": idx + 1,
                "seed": seeds[idx],
                "val_metrics": result["val_metrics"],
                "test_metrics": result["test_metrics"],
                "test_metrics_by_borough": result.get("test_metrics_by_borough"),
                "artifact_path": result.get("artifact_path", ""),
                "selected_epoch": result.get("selected_epoch"),
                "selection_metric": result.get("selection_metric"),
                "resource_usage": result.get("resource_usage"),
            }
            for idx, result in enumerate(results)
        ],
    }
    borough_metrics = aggregate_borough_metric_dicts(results)
    if borough_metrics is not None:
        aggregated["test_metrics_by_borough"] = borough_metrics
    return aggregated


def run_baseline_with_repeats(args, bundles: List[dict], device: torch.device) -> dict:
    if args.repeat_runs == 1:
        return dispatch_baseline(args, bundles, device)

    results = []
    seeds = []
    base_seed = int(args.seed)
    for run_idx in range(int(args.repeat_runs)):
        run_seed = base_seed + run_idx
        seeds.append(run_seed)
        run_args = copy.deepcopy(args)
        run_args.seed = run_seed
        print_status(
            f"Repeat run {run_idx + 1}/{args.repeat_runs} | seed={run_seed}",
            color=ANSI_MAGENTA,
            bold=True,
        )
        set_seed(run_seed)
        results.append(dispatch_baseline(run_args, bundles, device))

    aggregated = aggregate_repeated_results(results, seeds)
    print(format_metrics_with_na("val_mean", aggregated_means_for_display(aggregated["val_metrics"])))
    print(format_metrics_with_na("test_mean", aggregated_means_for_display(aggregated["test_metrics"])))
    if "test_metrics_by_borough" in aggregated:
        for borough_name, borough_metrics in aggregated["test_metrics_by_borough"].items():
            print(format_metrics_with_na(f"test_mean [{borough_name}]", aggregated_means_for_display(borough_metrics)))
    return aggregated


def build_unique_summary_path(out_dir: str, model_name: str) -> str:
    timestamp = datetime.now(SHANGHAI_TZ).strftime("%m%d%H%M")
    base_name = f"{model_name}_summary_{timestamp}"
    summary_path = os.path.join(out_dir, f"{base_name}.json")
    if not os.path.exists(summary_path):
        return summary_path

    suffix = 1
    while True:
        summary_path = os.path.join(out_dir, f"{base_name}_{suffix:02d}.json")
        if not os.path.exists(summary_path):
            return summary_path
        suffix += 1


def save_run_summary(args, result: dict):
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = build_unique_summary_path(args.out_dir, args.model)

    def sanitize_for_json(value):
        if isinstance(value, dict):
            return {k: sanitize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [sanitize_for_json(v) for v in value]
        if isinstance(value, tuple):
            return [sanitize_for_json(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        return value

    summary = {
        "model": args.model,
        "args": vars(args),
        "val_metrics": result["val_metrics"],
        "test_metrics": result["test_metrics"],
        "artifact_path": result["artifact_path"],
    }
    if "test_metrics_by_borough" in result:
        summary["test_metrics_by_borough"] = result["test_metrics_by_borough"]
    if "test_metrics_by_L" in result:
        summary["test_metrics_by_L"] = result["test_metrics_by_L"]
    if "repeat" in result:
        summary["repeat"] = result["repeat"]
    if "runs" in result:
        summary["runs"] = result["runs"]
    if "selected_epoch" in result:
        summary["selected_epoch"] = result["selected_epoch"]
    if "selection_metric" in result:
        summary["selection_metric"] = result["selection_metric"]
    if "validation_scope" in result:
        summary["validation_scope"] = result["validation_scope"]
    if "epoch_history" in result:
        summary["epoch_history"] = result["epoch_history"]
    if "resource_usage" in result:
        summary["resource_usage"] = result["resource_usage"]
    while True:
        try:
            with open(summary_path, "x", encoding="utf-8") as f:
                json.dump(sanitize_for_json(summary), f, ensure_ascii=False, indent=2)
            break
        except FileExistsError:
            summary_path = build_unique_summary_path(args.out_dir, args.model)
    print_status(f"Saved summary: {summary_path}", color=ANSI_GREEN, bold=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="st_daer_hurdle_lognormal",
        choices=[
            "ha",
            "xgboost",
            "stgcn",
            "stgat",
            "dcrnn",
            "st_gamma",
            "st_lognormal",
            "st_hurdle_gamma",
            "st_hurdle_lognormal",
            "st_event_occurrence",
            "st_event_aligned_hurdle_gamma",
            "st_event_aligned_hurdle_lognormal",
            "st_daer_hurdle_gamma",
            "st_daer_hurdle_lognormal",
            "st_hurdle_tweedie",
            "stg",
            "stnb",
            "stzinb",
            "sttd",
            "stzitd",
        ],help="ha,stgcn,stgat,stzinb,stzitd,st_hurdle_gamma"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument(
        "--expansion_features",
        type=str,
        default="none",
        help="新增动态特征组：none/all，或 construction,311,mobility 的逗号组合。",
    )
    parser.add_argument(
        "--expansion_dir",
        type=str,
        default="",
        help="新增动态特征目录；默认使用 <data_dir>/expansion。",
    )
    parser.add_argument(
        "--mobility_lag_days",
        type=int,
        default=60,
        help=(
            "TLC mobility 可用滞后天数：特征日 t 仅使用 trip day <= t-lag 的数据，lag 前填 0。"
            "正式协议固定 60（TLC 按月发布、约滞后两个月）；0=同日 retrospective oracle，"
            "只允许作为标注的敏感性实验，不得进入正式 D arm、价值矩阵或部署结论。"
        ),
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join("runs_results", "models"))##
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_runs", type=int, default=1, help="重复训练次数；审稿修改阶段默认仅运行一次。") ###
    parser.add_argument("--epochs", type=int, default=6) ###
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--history_len", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--mask_borough", type=str, default="all", help="Bronx/Brooklyn/Manhattan/Queens/Staten Island/all")
    parser.add_argument("--mask_path", type=str, default="")
    parser.add_argument("--start_date", type=str, default="2021-01-01") ###
    parser.add_argument("--end_date", type=str, default="2024-12-31")
    # 先验特征
    parser.add_argument("--use_priors", default=True, help="是否启用先验特征组。")
    parser.add_argument("--risk_diffuse", default=False, help="兼容旧参数；输入不再使用 X_propensity，如需扩散标签请使用 --target_mode=risk_diffuse。")
    parser.add_argument("--use_hist_freq_prior", type=int, default=1, choices=[0, 1], help="是否启用历史频次先验。") ###
    parser.add_argument(
        "--hist_prior_lookback",
        type=int,
        default=0,
        help="历史频次先验的滚动窗口天数；0 表示使用全部历史（冷启动 gate 用 30/90 等截断）。",
    )
    parser.add_argument(
        "--hist_prior_lookback_mix",
        type=str,
        default="",
        help="逗号分隔的多历史预算混合训练（如 14,30,90,180,0；0 表示全历史）。"
        "训练时每个窗口随机采样预算，评估按预算分别报告。当前仅支持 st_event_occurrence。",
    )
    parser.add_argument(
        "--hist_budget_channel",
        type=int,
        default=0,
        choices=[0, 1],
        help="混合预算训练时是否加入显式预算条件通道（log(L)/log(T)）；要求 --hist_prior_lookback_mix。",
    )
    parser.add_argument(
        "--hist_prior_shared_stats",
        type=int,
        default=0,
        choices=[0, 1],
        help="混合预算训练时各预算变体是否共享全历史标准化统计量（消除按分布位置推断预算的可能）；"
        "要求 --hist_prior_lookback_mix。",
    )
    parser.add_argument(
        "--hist_hetero_lookbacks",
        type=str,
        default="",
        help="按路段异质历史预算：逗号分隔候选值（如 14,30,90,180,0），每条路段按 seed 固定抽取一个，"
        "同一批次内不同路段持有不同有效历史长度。与 --hist_prior_lookback/--hist_prior_lookback_mix 互斥。",
    )
    parser.add_argument(
        "--node_holdout_fraction",
        type=float,
        default=0.0,
        help="node-held-out 比例：被抽中路段的全部事故历史通道置零（零历史端点，"
        "注意与本仓库表示完整历史的 L=0 不同），其标签从训练损失中剔除；"
        "评估时保留在排名集合内。模拟台账再标识/新记录路段。",
    )
    parser.add_argument(
        "--node_holdout_seed",
        type=int,
        default=-1,
        help="node-held-out 抽样种子；-1 表示使用 --seed。",
    )
    parser.add_argument(
        "--node_holdout_mode",
        type=str,
        default="random",
        choices=["random", "spatial_block"],
        help="node-held-out 抽取方式：random=逐路段独立抽样；spatial_block=按 borough×UTM 网格抽出"
        "（同区同格路段共享状态；抽中的格子未必相邻，不设缓冲区，跨格图边保留）。",
    )
    parser.add_argument(
        "--spatial_block_size_m",
        type=float,
        default=1000.0,
        help="spatial_block 模式的网格边长（米，UTM18N）。",
    )
    parser.add_argument("--use_structural_prior", type=int, default=1, choices=[0, 1], help="是否启用结构先验。")
    parser.add_argument(
        "--feature_ablation",
        type=str,
        default="none",
        choices=["none", "static_only", "history_only"],
        help="维度保持的特征组消融（mechanism analysis）：static_only 仅保留静态/结构/动态/日历/天气通道（置零事故历史通道），history_only 仅保留事故历史通道（置零静态/结构/动态/日历/天气）。",
    )
    parser.add_argument(
        "--use_recency_prior",
        type=int,
        default=0,
        choices=[0, 1],
        help="是否启用事故间隔先验（距上次事故天数，提供 14 天窗口之外的长间隔信息）。",
    )
    parser.add_argument(
        "--use_static_features",
        type=int,
        default=1,
        choices=[0, 1],
        help="贡献拆分消融：是否使用公开静态道路特征（置零保留维度）。",
    )
    parser.add_argument(
        "--static_zero_columns",
        type=str,
        default="",
        help="细粒度静态置零：逗号分隔列名前缀（如 x_poi_,x_node_），命中列在标准化空间置零、维度保留；"
        "用于漂移敏感通道的泄漏边界敏感性分析。",
    )
    parser.add_argument(
        "--use_event_window",
        type=int,
        default=1,
        choices=[0, 1],
        help="贡献拆分消融：是否使用 14 天原始事件历史窗口（置零保留维度）。",
    )
    parser.add_argument(
        "--use_weather_calendar",
        type=int,
        default=1,
        choices=[0, 1],
        help="贡献拆分消融：是否使用天气与日历特征（置零保留维度）。",
    )
    # 风险扩散设置
    parser.add_argument("--diffuse_alpha", type=float, default=0.3)
    parser.add_argument("--diffuse_k", type=int, default=2, help="风险扩散的空间跳数。")
    parser.add_argument("--diffuse_weight", type=float, default=0.5)
    parser.add_argument("--diffuse_norm", type=str, default="none", choices=["none", "row"])
    parser.add_argument("--risk_diffuse_zero_thresh", type=float, default=0.2, help="risk_diffuse 小于该阈值时置零。")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--color_output",
        type=str,
        default="auto",
        choices=["auto", "always", "never"],
        help="终端颜色：auto 仅在交互终端启用，always 强制启用，never 禁用。",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=500,
        help="训练、验证和测试每隔多少个 batch 输出进度与 ETA；0 表示关闭。",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="es_recall",
        choices=["es_recall", "es_ndcg", "ts_recall", "ts_ndcg", "mae_all", "last"],
        help="验证集检查点选择指标；last 表示使用最后一轮。",
    )
    parser.add_argument(
        "--train_end_idx",
        type=int,
        default=-1,
        help="M2 rolling-origin：训练/Scaler 的显式日索引上界（目标日 < 该索引的窗口进训练集）；-1 表示沿用 8/12 比例。与 --val_end_idx 成对提供。",
    )
    parser.add_argument(
        "--val_end_idx",
        type=int,
        default=-1,
        help="M2 rolling-origin：验证窗口目标日上界（显式日索引）；等于 --train_end_idx 时验证集为空（需 --skip_val_eval 1）。",
    )
    parser.add_argument(
        "--skip_val_eval",
        type=int,
        default=0,
        choices=[0, 1],
        help="跳过逐 epoch 验证评估；仅允许与 --selection_metric last 搭配（boundary 重训固定 e* epoch 取末轮）。",
    )
    parser.add_argument(
        "--skip_test_eval",
        type=int,
        default=0,
        choices=[0, 1],
        help="跳过训练器内置 test 评估（由独立评估 harness 负责逐窗口缓存与块级汇总时使用）。",
    )
    parser.add_argument(
        "--hr_ratio",
        type=str,
        default="0.05,0.10,0.20",
        help="热点检测 top-k 比例；可传单值如 0.2，或逗号列表如 0.05,0.10,0.20。",
    )
    parser.add_argument("--rank_score", type=str, default="mean", choices=["mean", "p_nonzero"], help="排序指标使用的分数。") ###
    parser.add_argument(
        "--graph_supervision",
        type=str,
        default="target_mse",
        choices=["target_mse", "event_bce"],
        help="STGCN/DCRNN/STGAT use either processed target-risk MSE or raw-event objectives.",
    )
    parser.add_argument("--zr_tau", type=float, default=0.01)
    parser.add_argument("--target_mode", type=str, default="risk_diffuse", choices=["risk", "risk_diffuse"], help="训练目标，risk 或 risk_diffuse。") ###
    parser.add_argument("--hist_source_mode", type=str, default="raw_risk", choices=["raw_risk", "target", "both"], help="历史输入使用 raw risk、Y_target，或两者同时输入。") ###

    parser.add_argument("--temporal_backbone", type=str, default="tcn", choices=["tcn", "gru"], help="概率模型时序骨干：TCN 或 GRU。")
    parser.add_argument("--spatial_backbone", type=str, default="diffusion", choices=["diffusion", "gat"], help="概率模型空间骨干：DiffusionConv 或 GAT。")

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--cheb_order", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--diffusion_steps", type=int, default=2,
                        help="扩散卷积阶数；0 表示关闭图消息传递（空间层退化为逐节点线性，no-graph 消融）。")
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--interval_alpha", type=float, default=0.05, help="区间指标使用的显著性水平 alpha。")
    parser.add_argument("--gauss_min_scale", type=float, default=0.1)
    parser.add_argument("--gauss_max_scale", type=float, default=5.0)
    parser.add_argument("--gauss_max_loc", type=float, default=10.0)

    # hurdle 模型配置
    parser.add_argument("--hurdle_temp_layers", type=int, default=3)
    parser.add_argument("--hurdle_spatial_layers", type=int, default=3)
    parser.add_argument("--hurdle_max_mu", type=float, default=20.0)
    parser.add_argument("--hurdle_gamma_max_alpha", type=float, default=50.0)
    parser.add_argument("--hurdle_lognormal_max_sigma", type=float, default=3.0)
    parser.add_argument(
        "--hurdle_positive_normalization",
        type=str,
        default="standard",
        choices=["standard", "balanced"],
        help=(
            "standard averages the complete hurdle NLL over all observations; "
            "balanced averages the conditional positive-density term over positives."
        ),
    )
    parser.add_argument("--hurdle_event_occ_weighting", type=str, default="auto", choices=["none", "auto"], help="连续 hurdle 模型的 event 分支是否自动加权。")
    parser.add_argument("--hurdle_event_occ_pos_weight_cap", type=float, default=20.0, help="event 分支正样本权重上限。")
    parser.add_argument("--hurdle_event_occ_lambda", type=float, default=0.5, help="event 辅助 BCE 的损失权重。") ###
    parser.add_argument("--hurdle_occ_pos_weight", type=float, default=0, help="hurdle 主 occurrence BCE 的正样本权重；0 表示不加权。")

    # 事件预测目标（direct event baseline 与 DAER event 分支共用）
    parser.add_argument(
        "--event_loss",
        type=str,
        default="weighted_bce",
        choices=["weighted_bce", "focal", "pairwise", "hybrid", "discrepancy_hybrid", "boundary_hybrid", "boundary"],
        help="事件目标：点式 BCE、Focal、通用成对排序、跨标签空间差异排序，或 Top-k 边界交换排序。",
    )
    parser.add_argument("--event_occ_weighting", type=str, default="auto", choices=["none", "auto"])
    parser.add_argument("--event_occ_pos_weight_cap", type=float, default=20.0)
    parser.add_argument("--event_focal_gamma", type=float, default=2.0)
    parser.add_argument("--event_rank_lambda", type=float, default=1.0)
    parser.add_argument("--event_rank_max_pairs", type=int, default=4096)
    parser.add_argument("--event_rank_hard_negative_fraction", type=float, default=0.5)
    parser.add_argument(
        "--event_discrepancy_base_margin",
        type=float,
        default=0.0,
        help="TS-positive/ES-negative 差异 pair 的固定 margin。",
    )
    parser.add_argument(
        "--event_discrepancy_target_margin_scale",
        type=float,
        default=0.5,
        help="差异 pair margin 随归一化 target risk 增加的幅度。",
    )
    parser.add_argument(
        "--event-boundary-margin",
        type=float,
        default=0.0,
        help="Top-k 边界交换排序损失的固定 margin。",
    )
    parser.add_argument(
        "--event-primary-horizon",
        type=int,
        default=0,
        help="直接 event baseline 主目标使用的未来天数；0 表示使用全部输出天数。",
    )
    parser.add_argument(
        "--event-cumulative-horizons",
        type=str,
        default="",
        help="逗号分隔的累计 hazard 辅助监督天数，如 1,3,7,14；空表示关闭。",
    )
    parser.add_argument(
        "--event-cumulative-lambda",
        type=float,
        default=0.0,
        help="累计 hazard 辅助损失的权重；0 表示关闭。",
    )
    parser.add_argument(
        "--event-cumulative-pos-weight-cap",
        type=float,
        default=20.0,
        help="累计 hazard 辅助损失的正样本权重上限。",
    )
    parser.add_argument(
        "--event-eval-horizon",
        type=int,
        default=0,
        help="直接 event baseline 评估时使用的未来天数；0 表示使用全部输出天数。",
    )
    parser.add_argument(
        "--event_supervision_space",
        type=str,
        default="raw",
        choices=["raw", "target"],
        help=(
            "事件监督标签来源：raw 使用原始事故空间，target 使用二值化后的目标空间"
            "（要求 --target_mode=risk_diffuse）。评估始终在原始事故空间进行。"
        ),
    )

    # DAER-HG 配置
    parser.add_argument("--daer_event_lambda", type=float, default=0.5, help="独立 event occurrence BCE 权重。")
    parser.add_argument(
        "--daer_event_occ_weighting",
        type=str,
        default="auto",
        choices=["none", "auto"],
        help="DAER event occurrence BCE 是否自动使用正样本权重。",
    )
    parser.add_argument(
        "--daer_event_occ_pos_weight_cap",
        type=float,
        default=20.0,
        help="DAER event occurrence BCE 正样本权重上限。",
    )
    parser.add_argument(
        "--daer_rank_branch",
        type=str,
        default="q_es",
        choices=["q_es", "q_ts"],
        help="DAER 在 event-space 热点排序中使用的 occurrence 分支。",
    )
    parser.add_argument(
        "--event_aligned_event_lambda",
        type=float,
        default=0.5,
        help="single-branch event-aligned Hurdle 的 event occurrence BCE 权重。",
    )
    parser.add_argument(
        "--event_aligned_event_occ_weighting",
        type=str,
        default="auto",
        choices=["none", "auto"],
        help="single-branch event-aligned Hurdle event BCE 是否自动使用正样本权重。",
    )
    parser.add_argument(
        "--event_aligned_event_occ_pos_weight_cap",
        type=float,
        default=20.0,
        help="single-branch event-aligned Hurdle event BCE 正样本权重上限。",
    )
    parser.add_argument("--mc_interval_samples", type=int, default=100)
    parser.add_argument("--count_max_n", type=float, default=50.0)
    parser.add_argument("--count_min_prob", type=float, default=1e-4)
    parser.add_argument("--count_max_prob", type=float, default=0.9999)
    parser.add_argument("--tweedie_mu_max", type=float, default=20.0)
    parser.add_argument("--tweedie_phi_min", type=float, default=1e-3)
    parser.add_argument("--tweedie_phi_max", type=float, default=10.0)
    parser.add_argument("--tweedie_rho_eps", type=float, default=1e-3)
    parser.add_argument("--tweedie_interval_samples", type=int, default=100)

    parser.add_argument("--xgb_train_max_samples", type=int, default=200000)
    parser.add_argument("--xgb_n_estimators", type=int, default=300)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    parser.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    parser.add_argument("--xgb_n_jobs", type=int, default=4)
    parser.add_argument("--xgb_objective", type=str, default="count:poisson")
    return parser


def dispatch_baseline(args, bundles: List[dict], device: torch.device) -> dict:
    if args.model == "ha":
        return run_ha_baseline(args, bundles)
    if args.model == "xgboost":
        return run_xgboost_baseline(args, bundles)
    if args.model == "stgcn":
        return run_stgcn_baseline(args, bundles, device)
    if args.model == "stgat":
        return run_stgat_baseline(args, bundles, device)
    if args.model == "dcrnn":
        return run_dcrnn_baseline(args, bundles, device)
    if args.model == "stg":
        return run_stg_baseline(args, bundles, device)
    if args.model == "st_event_occurrence":
        return run_st_event_occurrence_baseline(args, bundles, device)
    if args.model == "st_hurdle_gamma":
        return run_st_hurdle_gamma_baseline(args, bundles, device)
    if args.model == "st_event_aligned_hurdle_gamma":
        return run_st_event_aligned_hurdle_gamma_baseline(args, bundles, device)
    if args.model == "st_daer_hurdle_gamma":
        return run_st_daer_hurdle_gamma_baseline(args, bundles, device)
    if args.model == "st_gamma":
        return run_st_gamma_baseline(args, bundles, device)
    if args.model == "st_lognormal":
        return run_st_lognormal_baseline(args, bundles, device)

    if args.model == "st_hurdle_lognormal":
        return run_st_hurdle_lognormal_baseline(args, bundles, device)
    if args.model == "st_event_aligned_hurdle_lognormal":
        return run_st_event_aligned_hurdle_lognormal_baseline(args, bundles, device)
    if args.model == "st_daer_hurdle_lognormal":
        return run_st_daer_hurdle_lognormal_baseline(args, bundles, device)
    if args.model == "st_hurdle_tweedie":
        return run_st_hurdle_tweedie_baseline(args, bundles, device)

    if args.model == "stnb":
        return run_stnb_baseline(args, bundles, device)
    if args.model == "stzinb":
        return run_stzinb_baseline(args, bundles, device)
    if args.model == "sttd":
        return run_sttd_baseline(args, bundles, device)
    if args.model == "stzitd":
        return run_stzitd_baseline(args, bundles, device)
    raise ValueError(f"Unsupported model: {args.model}")


def _to_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def main():
    parser = build_parser()
    args = parser.parse_args()
    for key, value in capture_git_provenance().items():
        setattr(args, key, value)
    args.command_at_start = list(sys.argv)
    configure_color_output(args.color_output)

    args.use_priors = _to_bool_flag(args.use_priors)
    args.risk_diffuse = _to_bool_flag(args.risk_diffuse)
    args.use_hist_freq_prior = _to_bool_flag(args.use_hist_freq_prior)
    args.use_structural_prior = _to_bool_flag(args.use_structural_prior)
    args.target_mode = normalize_target_mode(args.target_mode)
    args.hist_source_mode = normalize_hist_source_mode(args.hist_source_mode)
    args.hr_ratio = parse_hr_ratios(args.hr_ratio)

    if args.model in {"st_gamma", "st_lognormal"} and args.rank_score != "mean":
        print(f"Note: {args.model} does not define p_nonzero; rank_score is forced to mean.")
        args.rank_score = "mean"
    if (
            args.model in {
                "st_daer_hurdle_gamma",
                "st_daer_hurdle_lognormal",
                "st_event_aligned_hurdle_gamma",
                "st_event_aligned_hurdle_lognormal",
            }
            and args.rank_score != "mean"
    ):
        print("Note: this model uses predictive mean for TS ranking and occurrence logits for ES ranking.")
        args.rank_score = "mean"
    if args.repeat_runs < 1:
        raise ValueError("--repeat_runs must be >= 1.")
    if args.model == "st_event_occurrence" and args.selection_metric not in {"es_recall", "es_ndcg", "last"}:
        raise ValueError("st_event_occurrence supports --selection_metric es_recall, es_ndcg, or last.")
    if (
            args.model in {"stgcn", "stgat", "dcrnn"}
            and args.graph_supervision == "event_bce"
            and args.selection_metric not in {"es_recall", "es_ndcg", "last"}
    ):
        raise ValueError("Event-supervised graph baselines support es_recall, es_ndcg, or last selection.")
    if args.model in {"stgcn", "stgat", "dcrnn"}:
        args.graph_output_activation = "identity" if args.graph_supervision == "event_bce" else "softplus"
    if args.progress_interval < 0:
        raise ValueError("--progress_interval must be >= 0.")
    if args.daer_event_lambda < 0.0:
        raise ValueError("DAER loss weight must be non-negative.")
    if args.event_aligned_event_lambda < 0.0:
        raise ValueError("event-aligned loss weight must be non-negative.")
    if args.event_focal_gamma < 0.0:
        raise ValueError("--event_focal_gamma must be non-negative.")
    if args.event_rank_lambda < 0.0:
        raise ValueError("--event_rank_lambda must be non-negative.")
    if args.event_rank_max_pairs < 1:
        raise ValueError("--event_rank_max_pairs must be >= 1.")
    if not 0.0 <= args.event_rank_hard_negative_fraction <= 1.0:
        raise ValueError("--event_rank_hard_negative_fraction must be in [0, 1].")
    if args.hist_prior_lookback < 0:
        raise ValueError("--hist_prior_lookback must be >= 0 (0 means full history).")
    lookback_mix = parse_lookback_mix(args.hist_prior_lookback_mix)
    if lookback_mix:
        if not args.use_hist_freq_prior:
            raise ValueError("--hist_prior_lookback_mix requires --use_hist_freq_prior 1.")
        if args.hist_prior_lookback > 0:
            raise ValueError("--hist_prior_lookback_mix and --hist_prior_lookback are mutually exclusive.")
        if args.model != "st_event_occurrence":
            raise ValueError("--hist_prior_lookback_mix is currently supported only for st_event_occurrence.")
    if args.hist_budget_channel and not lookback_mix:
        raise ValueError("--hist_budget_channel requires --hist_prior_lookback_mix.")
    if args.hist_prior_shared_stats and not lookback_mix:
        raise ValueError("--hist_prior_shared_stats requires --hist_prior_lookback_mix.")
    hetero_choices = parse_lookback_mix(args.hist_hetero_lookbacks)
    if hetero_choices:
        if not args.use_hist_freq_prior:
            raise ValueError("--hist_hetero_lookbacks requires --use_hist_freq_prior 1.")
        if args.hist_prior_lookback > 0 or lookback_mix:
            raise ValueError(
                "--hist_hetero_lookbacks is mutually exclusive with --hist_prior_lookback and --hist_prior_lookback_mix."
            )
        if args.model != "st_event_occurrence":
            raise ValueError("--hist_hetero_lookbacks is currently supported only for st_event_occurrence.")
    if args.event_discrepancy_base_margin < 0.0:
        raise ValueError("--event_discrepancy_base_margin must be non-negative.")
    if args.event_discrepancy_target_margin_scale < 0.0:
        raise ValueError("--event_discrepancy_target_margin_scale must be non-negative.")
    if not np.isfinite(args.event_boundary_margin) or args.event_boundary_margin < 0.0:
        raise ValueError("--event-boundary-margin must be finite and non-negative.")

    args.event_cumulative_horizons = parse_cumulative_horizons(args.event_cumulative_horizons)
    if not np.isfinite(args.event_cumulative_lambda) or args.event_cumulative_lambda < 0.0:
        raise ValueError("--event-cumulative-lambda must be finite and non-negative.")
    if not np.isfinite(args.event_cumulative_pos_weight_cap) or args.event_cumulative_pos_weight_cap < 1.0:
        raise ValueError("--event-cumulative-pos-weight-cap must be finite and >= 1.")
    if args.event_cumulative_horizons and max(args.event_cumulative_horizons) > args.horizon:
        raise ValueError("--event-cumulative-horizons cannot exceed --horizon.")
    if args.event_cumulative_lambda > 0.0 and not args.event_cumulative_horizons:
        raise ValueError("--event-cumulative-lambda > 0 requires non-empty --event-cumulative-horizons.")
    if args.event_primary_horizon < 0 or args.event_primary_horizon > args.horizon:
        raise ValueError("--event-primary-horizon must be 0 or within --horizon.")
    if args.event_eval_horizon < 0 or args.event_eval_horizon > args.horizon:
        raise ValueError("--event-eval-horizon must be 0 or within --horizon.")
    if args.event_supervision_space == "target":
        if not uses_diffused_target(args.target_mode):
            raise ValueError(
                "--event_supervision_space=target requires --target_mode=risk_diffuse; "
                "with target_mode=risk the target space equals the raw event space."
            )
        if args.model != "st_event_occurrence":
            raise ValueError(
                "--event_supervision_space=target is currently supported only for "
                "st_event_occurrence (and st_gcrst via main_latent_state.py)."
            )

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.risk_diffuse:
        print(
            "Note: baseline inputs no longer consume X_propensity/risk_diffuse. "
            "Use --target_mode=risk_diffuse to train on diffused Y_risk labels."
        )
    if args.use_priors:
        if not args.use_hist_freq_prior and not args.use_structural_prior:
            print("use_priors enabled, but the available baseline priors are disabled: only Group A will be used.")
        elif uses_diffused_target(args.target_mode):
            if args.hist_source_mode == "target":
                print("target_mode=risk_diffuse: labels and history inputs are both built from diffused Y_risk.")
            elif args.hist_source_mode == "both":
                print("target_mode=risk_diffuse: labels are built from diffused Y_risk; history inputs concatenate raw Y_risk and diffused Y_risk.")
            else:
                print("target_mode=risk_diffuse: labels are built by diffusing Y_risk; history inputs still use raw Y_risk.")
        elif args.hist_source_mode == "both":
            print("target_mode=risk: Y_target equals raw Y_risk, so dual history inputs collapse to a single raw-risk history channel.")
    if should_warn_distribution_mismatch(args.model, args.target_mode):
        print(
            f"Warning: model {args.model} is running with target_mode={args.target_mode}. "
            "The diffused target may be non-integer/continuous, so count-based distribution assumptions are weaker."
        )
    device = resolve_device(args.device)
    if (
            args.model in {
                "st_daer_hurdle_gamma",
                "st_daer_hurdle_lognormal",
                "st_event_aligned_hurdle_gamma",
                "st_event_aligned_hurdle_lognormal",
                "st_event_occurrence",
            }
            and args.device == "auto"
            and device.type != "cuda"
    ):
        raise RuntimeError(
            "DAER/event-aligned Hurdle resolved --device auto to CPU. To prevent an accidental multi-hour run, "
            "fix the server CUDA environment or pass --device cpu explicitly."
        )
    print_runtime_device(device, args.device)
    bundles = build_bundles(args, device=device)
    result = run_baseline_with_repeats(args, bundles, device)
    save_run_summary(args, result)


if __name__ == "__main__":
    main()
