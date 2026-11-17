from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def parse_lookback_mix(value) -> tuple[int, ...]:
    """Parse a comma-separated lookback list like "14,30,90,0" (0 = full history)."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        parts = [int(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return ()
        parts = [int(p.strip()) for p in text.split(",") if p.strip()]
    if any(p < 0 for p in parts):
        raise ValueError("lookback values must be >= 0 (0 means full history)")
    if len(set(parts)) != len(parts):
        raise ValueError(f"duplicate lookback values: {parts}")
    return tuple(parts)


def build_exclusive_cumulative_mean(Y: np.ndarray) -> np.ndarray:
    Y = Y.astype(np.float32, copy=False)
    out = np.zeros_like(Y, dtype=np.float32)
    if Y.shape[1] <= 1:
        return out
    cumsum = np.cumsum(Y, axis=1, dtype=np.float32)
    denom = np.arange(1, Y.shape[1], dtype=np.float32)
    out[:, 1:] = cumsum[:, :-1] / denom[None, :]
    return out


def build_trailing_exclusive_mean(Y: np.ndarray, lookback: int) -> np.ndarray:
    """Exclusive trailing-window mean: freq[t] = mean(Y[max(0, t-L):t]).

    Cold-start counterpart of build_exclusive_cumulative_mean: at day t only the
    most recent ``lookback`` observed days are available, so the frequency prior
    degenerates to a short-window rate. Day t itself is always excluded; early
    days average over however many days have actually been observed.
    """
    Y = Y.astype(np.float32, copy=False)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    out = np.zeros_like(Y, dtype=np.float32)
    T = Y.shape[1]
    if T <= 1:
        return out
    cumsum = np.cumsum(Y, axis=1, dtype=np.float32)
    t_idx = np.arange(1, T)
    lo = np.maximum(t_idx - lookback, 0)
    window_sum = cumsum[:, t_idx - 1] - np.where(lo > 0, cumsum[:, lo - 1], 0.0)
    denom = (t_idx - lo).astype(np.float32)
    out[:, 1:] = window_sum / denom[None, :]
    return out


def build_per_road_trailing_mean(Y: np.ndarray, lookbacks: np.ndarray) -> np.ndarray:
    """Per-road exclusive trailing-window mean with road-specific lookbacks.

    ``lookbacks[i]`` is road i's effective history budget in days; 0 means full
    history. Models heterogeneous archive depth across roads (re-segmented or
    newly onboarded segments have short budgets while established ones keep
    long histories) within a single training batch.
    """
    Y = Y.astype(np.float32, copy=False)
    lookbacks = np.asarray(lookbacks, dtype=np.int64)
    if lookbacks.shape != (Y.shape[0],):
        raise ValueError("lookbacks must have one value per road")
    if np.any(lookbacks < 0):
        raise ValueError("lookbacks must be >= 0 (0 means full history)")
    N, T = Y.shape
    out = np.zeros((N, T), dtype=np.float32)
    if T <= 1:
        return out
    cumsum = np.cumsum(Y, axis=1, dtype=np.float64)
    t_idx = np.arange(1, T)[None, :]
    eff = np.where(lookbacks[:, None] > 0, lookbacks[:, None], T)
    lo = np.maximum(t_idx - eff, 0)
    prev_sum = np.take_along_axis(cumsum, np.clip(lo - 1, 0, None), axis=1)
    window_sum = cumsum[:, :-1] - np.where(lo > 0, prev_sum, 0.0)
    denom = (t_idx - lo).astype(np.float64)
    out[:, 1:] = (window_sum / denom).astype(np.float32)
    return out


def build_days_since_last_event(Y: np.ndarray, cap: int = 365) -> np.ndarray:
    """Days since the most recent event before each day (exclusive, capped).

    out[t] = min(t - last_event_t, cap) for the latest event strictly before t;
    ``cap`` also codes "no event observed so far". Carries long-gap recency
    information that a short history window cannot see.
    """
    Y = np.asarray(Y)
    if cap < 1:
        raise ValueError("cap must be >= 1")
    events = (Y > 0).astype(np.int8, copy=False)
    out = np.full(Y.shape, float(cap), dtype=np.float32)
    T = Y.shape[1]
    for t in range(1, T):
        prev = out[:, t - 1] + 1.0
        has_event = events[:, t - 1] > 0
        prev[has_event] = 1.0
        out[:, t] = np.minimum(prev, float(cap))
    return out


def diffuse_risk_time_space(
        Y: np.ndarray,
        edge_index: np.ndarray,
        alpha: float = 0.5,
        k: int = 3,
        hop_weight: float = 0.5,
        undirected: bool = True,
        normalize: str = "row",
        zero_thresh: float = 0.2,
) -> np.ndarray:
    Y = np.asarray(Y)
    edge_index = np.asarray(edge_index)

    if Y.ndim != 2:
        raise ValueError(f"Y must have shape [N, T], got {Y.shape}")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}")
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    if hop_weight < 0:
        raise ValueError(f"hop_weight must be >= 0, got {hop_weight}")
    if normalize not in ("row", "none"):
        raise ValueError("normalize must be 'row' or 'none'")
    if zero_thresh < 0:
        raise ValueError(f"zero_thresh must be >= 0, got {zero_thresh}")

    Y = Y.astype(np.float32, copy=False)
    N, T = Y.shape

    Yt = np.empty((N, T), dtype=np.float32)
    Yt[:, 0] = Y[:, 0]
    for t in range(1, T):
        Yt[:, t] = alpha * Y[:, t] + (1.0 - alpha) * Yt[:, t - 1]

    if k == 0 or hop_weight == 0.0:
        if not np.isfinite(Yt).all():
            raise ValueError("Temporal smoothing produced NaN/Inf values.")
        return Yt

    src0 = edge_index[0].astype(np.int64, copy=False)
    dst0 = edge_index[1].astype(np.int64, copy=False)
    valid = (src0 >= 0) & (src0 < N) & (dst0 >= 0) & (dst0 < N)
    src0 = src0[valid]
    dst0 = dst0[valid]

    if undirected:
        src = np.concatenate([src0, dst0])
        dst = np.concatenate([dst0, src0])
    else:
        src, dst = src0, dst0

    A = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(N, N),
        dtype=np.float32,
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0
    A.setdiag(0.0)
    A.eliminate_zeros()

    if normalize == "row":
        deg = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
        inv_deg = np.zeros_like(deg, dtype=np.float32)
        nz = deg > 0
        inv_deg[nz] = 1.0 / deg[nz]
        P = sp.diags(inv_deg).dot(A).tocsr()
    else:
        P = A.tocsr()

    out = Yt.copy()
    walk = Yt.copy()
    total_w = 1.0
    for step in range(1, k + 1):
        walk = P.dot(walk).astype(np.float32, copy=False)
        w = float(hop_weight) ** step
        out += w * walk
        total_w += w
    out /= total_w

    if zero_thresh > 0:
        out = out.copy()
        out[out < float(zero_thresh)] = 0.0

    if not np.isfinite(out).all():
        raise ValueError("Risk diffusion produced NaN/Inf values.")
    return out.astype(np.float32, copy=False)
