from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_exclusive_ewma(
        values: np.ndarray,
        half_life: float,
        lookback: int = 0,
) -> np.ndarray:
    """Exclusive causal EWMA, optionally truncated to the latest ``lookback`` days.

    ``out[:, t]`` uses only ``values[:, :t]``. A zero lookback means all
    available history. The result is normalized by the available weight mass.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [N, T], got {values.shape}")
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    if lookback < 0:
        raise ValueError("lookback must be >= 0")

    n_nodes, n_steps = values.shape
    out = np.zeros((n_nodes, n_steps), dtype=np.float32)
    if n_steps <= 1:
        return out

    decay = float(0.5 ** (1.0 / half_life))
    remove_weight = float(decay ** lookback) if lookback > 0 else 0.0
    # Use float64 internally and reset empty windows exactly. With a long time
    # series, repeated subtraction in float32 leaves tiny residues after the
    # final event exits the window; those residues can spuriously break ties
    # among the many all-zero roads and leak an ordering from older history.
    running = np.zeros(n_nodes, dtype=np.float64)
    active_nonzero = np.zeros(n_nodes, dtype=np.int32)
    weight_mass = 0.0
    for t in range(1, n_steps):
        running *= decay
        running += values[:, t - 1]
        active_nonzero += values[:, t - 1] != 0
        weight_mass = decay * weight_mass + 1.0
        if lookback > 0 and t > lookback:
            running -= remove_weight * values[:, t - lookback - 1]
            active_nonzero -= values[:, t - lookback - 1] != 0
            weight_mass -= remove_weight
            running[active_nonzero == 0] = 0.0
        out[:, t] = running / max(weight_mass, np.finfo(np.float32).eps)
    return out


def effective_history_days(n_steps: int, lookback: int) -> np.ndarray:
    """Number of observations available to an exclusive history feature."""
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if lookback < 0:
        raise ValueError("lookback must be >= 0")
    days = np.arange(n_steps, dtype=np.float32)
    if lookback > 0:
        days = np.minimum(days, float(lookback))
    return days


def effective_ewma_weight_mass(n_steps: int, half_life: float, lookback: int = 0) -> np.ndarray:
    """Total pseudo-observation weight behind each exclusive EWMA value."""
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    if lookback < 0:
        raise ValueError("lookback must be >= 0")
    decay = float(0.5 ** (1.0 / half_life))
    remove_weight = float(decay ** lookback) if lookback > 0 else 0.0
    result = np.zeros(n_steps, dtype=np.float32)
    running = 0.0
    for t in range(1, n_steps):
        running = decay * running + 1.0
        if lookback > 0 and t > lookback:
            running -= remove_weight
        result[t] = running
    return result


def estimate_group_event_priors(
        events: np.ndarray,
        group_ids: np.ndarray,
        train_end: int,
        prior_strength: float = 1.0,
) -> np.ndarray:
    """Estimate one smoothed training-period event rate per node's group."""
    events = np.asarray(events, dtype=np.float32)
    group_ids = np.asarray(group_ids)
    if events.ndim != 2:
        raise ValueError(f"events must have shape [N, T], got {events.shape}")
    if group_ids.shape != (events.shape[0],):
        raise ValueError("group_ids must have shape [N]")
    if not 0 < train_end <= events.shape[1]:
        raise ValueError("train_end must lie in (0, T]")
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    _, inverse = np.unique(group_ids.astype(str), return_inverse=True)
    node_hits = events[:, :train_end].sum(axis=1, dtype=np.float64)
    group_hits = np.bincount(inverse, weights=node_hits)
    group_nodes = np.bincount(inverse).astype(np.float64)
    group_exposure = group_nodes * float(train_end)
    global_rate = float(node_hits.sum() / max(group_exposure.sum(), 1.0))
    group_rate = (
        group_hits + float(prior_strength) * global_rate
    ) / np.maximum(group_exposure + float(prior_strength), 1.0)
    return group_rate[inverse].astype(np.float32)


def posterior_group_rate(
        observed_rate: np.ndarray,
        observed_days: np.ndarray,
        group_prior: np.ndarray,
        prior_strength: float,
) -> np.ndarray:
    """Combine node history and a group prior as a Beta-Binomial posterior mean."""
    observed_rate = np.asarray(observed_rate, dtype=np.float32)
    observed_days = np.asarray(observed_days, dtype=np.float32)
    group_prior = np.asarray(group_prior, dtype=np.float32)
    if observed_rate.ndim != 2:
        raise ValueError("observed_rate must have shape [N, S]")
    if observed_days.shape != (observed_rate.shape[1],):
        raise ValueError("observed_days must have shape [S]")
    if group_prior.shape != (observed_rate.shape[0],):
        raise ValueError("group_prior must have shape [N]")
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    strength = float(prior_strength)
    denom = observed_days[None, :] + strength
    numer = observed_rate * observed_days[None, :] + strength * group_prior[:, None]
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)


def build_induced_random_walk(edge_index: np.ndarray, node_indices: np.ndarray) -> sp.csr_matrix:
    """Build an undirected row-normalized adjacency on an induced node subset."""
    edge_index = np.asarray(edge_index)
    node_indices = np.asarray(node_indices, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_indices.ndim != 1:
        raise ValueError("node_indices must have shape [N]")
    if node_indices.size == 0:
        return sp.csr_matrix((0, 0), dtype=np.float32)

    max_node = int(max(edge_index.max(initial=-1), node_indices.max(initial=-1)))
    global_to_local = np.full(max_node + 1, -1, dtype=np.int64)
    global_to_local[node_indices] = np.arange(node_indices.size, dtype=np.int64)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    valid = (
        (src >= 0) & (src <= max_node) & (dst >= 0) & (dst <= max_node)
        & (global_to_local[src] >= 0) & (global_to_local[dst] >= 0)
    )
    src_local = global_to_local[src[valid]]
    dst_local = global_to_local[dst[valid]]
    src_bi = np.concatenate([src_local, dst_local])
    dst_bi = np.concatenate([dst_local, src_local])
    adjacency = sp.coo_matrix(
        (np.ones(src_bi.size, dtype=np.float32), (src_bi, dst_bi)),
        shape=(node_indices.size, node_indices.size),
        dtype=np.float32,
    ).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float32)
    inv_degree = np.divide(1.0, degree, out=np.zeros_like(degree), where=degree > 0)
    return sp.diags(inv_degree).dot(adjacency).tocsr()
