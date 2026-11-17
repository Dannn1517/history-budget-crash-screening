from __future__ import division

import numpy as np
import scipy.sparse as sp


rand = np.random.RandomState(0)


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if sp.issparse(A):
        diag = A.diagonal()
        if np.all(diag == 0):
            A = A + sp.eye(A.shape[0], dtype=np.float32, format="coo")
        D = np.array(A.sum(axis=1)).reshape((-1,))
        D[D <= 10e-5] = 10e-5  # Prevent infs
        d_inv_sqrt = np.reciprocal(np.sqrt(D))
        d_mat = sp.diags(d_inv_sqrt)
        return (d_mat.dot(A).dot(d_mat)).tocoo()
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))  # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5  # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    if sp.issparse(adj_mx):
        adj_mx = adj_mx.tocoo()
        d = np.array(adj_mx.sum(1)).reshape((-1,))
        d_inv = np.power(d, -1, where=d > 0)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
        return random_walk_mx
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def zr_from_mean_threshold(y_true, pred_mean, tau=0.5):
    y_true = np.asarray(y_true).reshape(-1)
    pred_mean = np.asarray(pred_mean).reshape(-1)

    pred0 = pred_mean < tau
    true0 = y_true == 0

    total_samples = y_true.shape[0]
    if total_samples == 0:
        return np.nan

    return float(np.sum(true0 & pred0) / total_samples)

def zinb_p0_from_params(n, p, pi, eps=1e-8):
    n = np.clip(n, eps, None)
    p = np.clip(p, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)
    return pi + (1 - pi) * (p ** n)

###############################################################################################
# Uncertainty Quantification
def zinb_interval_LU_from_params_scipy(n, p, pi, alpha=0.1, eps=1e-8, integer_bounds=True):
    """
    解析分位数（需要 scipy）
    """
    import numpy as np
    from scipy.stats import nbinom

    n = _to_numpy(n).astype(np.float64)
    p = _to_numpy(p).astype(np.float64)
    pi = _to_numpy(pi).astype(np.float64)

    n = np.clip(n, eps, None)
    p = np.clip(p, eps, 1 - eps)
    pi = np.clip(pi, 0.0, 1.0)

    q_lo = alpha / 2.0
    q_hi = 1.0 - alpha / 2.0

    def zinb_ppf(q):
        # F_ZINB(k)=pi+(1-pi)*F_NB(k)
        # 若 q <= F(0)=pi+(1-pi)*p^n，则分位数=0
        p0 = pi + (1 - pi) * (p ** n)
        out = np.zeros(np.broadcast(n, p, pi).shape, dtype=np.float64)

        mask = q > p0
        if np.any(mask):
            one_minus_pi = np.clip(1 - pi, 1e-12, 1.0)
            q_nb = (q - pi) / one_minus_pi
            q_nb = np.clip(q_nb, 0.0, 1.0)
            out[mask] = nbinom.ppf(q_nb[mask], n[mask], p[mask])
        return out

    L = zinb_ppf(q_lo)
    U = zinb_ppf(q_hi)

    if integer_bounds:
        L = np.floor(L)
        U = np.ceil(U)

    return L, U


def picp(truth, L, U):
    """
    PICP = mean( 1{ L <= y <= U } )
    取值范围 [0,1]，越接近标称覆盖率(1-alpha)越好
    """
    y = _to_numpy(truth)
    L = _to_numpy(L)
    U = _to_numpy(U)
    return float(np.mean((y >= L) & (y <= U)))


def mpiw(L, U):
    """
    MPIW = mean(U - L)
    取值范围 [0, +inf)，越小越“尖锐”，但要配合 PICP 一起看
    """
    L = _to_numpy(L)
    U = _to_numpy(U)
    return float(np.mean(U - L))


def interval_score(truth, L, U, alpha=0.1):
    y = _to_numpy(truth)
    L = _to_numpy(L)
    U = _to_numpy(U)
    width = U - L
    under = np.clip(L - y, a_min=0.0, a_max=None)
    over = np.clip(y - U, a_min=0.0, a_max=None)
    return float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))


def picp_conditional_ypos(truth, L, U):
    y = _to_numpy(truth).reshape(-1)
    L = _to_numpy(L).reshape(-1)
    U = _to_numpy(U).reshape(-1)
    mask = y > 0
    if np.sum(mask) == 0:
        return np.nan
    return float(np.mean((y[mask] >= L[mask]) & (y[mask] <= U[mask])))


def interval_score_conditional_ypos(truth, L, U, alpha=0.1):
    y = _to_numpy(truth).reshape(-1)
    L = _to_numpy(L).reshape(-1)
    U = _to_numpy(U).reshape(-1)
    mask = y > 0
    if np.sum(mask) == 0:
        return np.nan
    width = U[mask] - L[mask]
    under = np.clip(L[mask] - y[mask], a_min=0.0, a_max=None)
    over = np.clip(y[mask] - U[mask], a_min=0.0, a_max=None)
    return float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))


###############################################################################################
# 分类指标
def _select_top_k_mask(score, top_frac=0.2, uncertainty=None, uncertainty_threshold="mean"):
    score = _to_numpy(score).reshape(-1)
    N = score.shape[0]
    k = max(1, int(np.ceil(top_frac * N)))
    order = np.argsort(-score, kind="mergesort")
    top_idx = order[:k]
    selected = np.zeros(N, dtype=bool)
    selected[top_idx] = True
    if uncertainty is not None:
        u = _to_numpy(uncertainty).reshape(-1)
        assert u.shape[0] == N
        if uncertainty_threshold == "mean":
            thr = float(np.mean(u))
        elif uncertainty_threshold == "median":
            thr = float(np.median(u))
        else:
            thr = float(uncertainty_threshold)
        selected = selected & (u <= thr)
    return selected


def precision_at_k(truth, risk_score, uncertainty=None, top_frac=0.2, uncertainty_threshold="mean"):
    """
    Precision@k%:
      1) truth 定义事件：y>0 为正
      2) risk_score 用于排序
      3) 取 top k%，precision = hits / selected
    """
    y = _to_numpy(truth).reshape(-1)
    score = _to_numpy(risk_score).reshape(-1)
    assert y.shape[0] == score.shape[0]
    event = (y > 0)
    selected = _select_top_k_mask(score, top_frac, uncertainty, uncertainty_threshold)
    denom = np.sum(selected)
    if denom == 0:
        return np.nan, selected
    hits = np.sum(event & selected)
    return float(hits / denom), selected


def map_at_k(truth, risk_score, top_frac=0.2):
    """
    MAP@k% for a single ranked list (binary relevance: y>0).
    """
    y = _to_numpy(truth).reshape(-1)
    score = _to_numpy(risk_score).reshape(-1)
    assert y.shape[0] == score.shape[0]
    N = score.shape[0]
    k = max(1, int(np.ceil(top_frac * N)))
    order = np.argsort(-score, kind="mergesort")[:k]
    rel = (y[order] > 0).astype(np.float64)
    total_pos = int(np.sum(y > 0))
    if total_pos == 0:
        return np.nan
    denom = min(total_pos, k)
    if denom == 0:
        return np.nan
    cumsum = np.cumsum(rel)
    precision_at_i = cumsum / (np.arange(k) + 1.0)
    ap = np.sum(precision_at_i * rel) / float(denom)
    return float(ap)


def ndcg_at_k(truth, risk_score, top_frac=0.2):
    """
    NDCG@k% for a single ranked list (binary relevance: y>0).
    """
    y = _to_numpy(truth).reshape(-1)
    score = _to_numpy(risk_score).reshape(-1)
    assert y.shape[0] == score.shape[0]
    N = score.shape[0]
    k = max(1, int(np.ceil(top_frac * N)))
    order = np.argsort(-score, kind="mergesort")[:k]
    rel = (y[order] > 0).astype(np.float64)
    total_pos = int(np.sum(y > 0))
    if total_pos == 0:
        return np.nan
    denom = min(total_pos, k)
    if denom == 0:
        return np.nan
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel * discounts))
    idcg = float(np.sum(discounts[:denom]))
    if idcg == 0:
        return np.nan
    return float(dcg / idcg)


def ndcg_at_k_graded(truth, risk_score, top_frac=0.2):
    """
    NDCG@k% for a single ranked list (graded relevance: log1p(y)).
    """
    y = _to_numpy(truth).reshape(-1)
    score = _to_numpy(risk_score).reshape(-1)
    assert y.shape[0] == score.shape[0]
    N = score.shape[0]
    k = max(1, int(np.ceil(top_frac * N)))
    rel = np.log1p(np.maximum(y, 0.0))
    if np.sum(rel) == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")[:k]
    rel_ranked = rel[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel_ranked * discounts))
    rel_ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(rel_ideal * discounts))
    if idcg == 0:
        return np.nan
    return float(dcg / idcg)
