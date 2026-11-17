from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from utils import calculate_random_walk_matrix, get_normalized_adj


def make_undirected_edge_index(edge_index: np.ndarray) -> np.ndarray:
    src, dst = edge_index
    rev = np.stack([dst, src], axis=0)
    return np.concatenate([edge_index, rev], axis=1)


def apply_mask_to_edge_index(edge_index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.all():
        return edge_index
    node_idx = np.where(mask)[0]
    mapper = -np.ones(mask.shape[0], dtype=np.int64)
    mapper[node_idx] = np.arange(node_idx.shape[0])
    src, dst = edge_index
    keep = mask[src] & mask[dst]
    src_new = mapper[src[keep]]
    dst_new = mapper[dst[keep]]
    return np.stack([src_new, dst_new], axis=0)


def scipy_to_torch_sparse(mat: sp.spmatrix, device: torch.device) -> torch.Tensor:
    coo = mat.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data.astype(np.float32)
    i = torch.from_numpy(indices).long()
    v = torch.from_numpy(values).float()
    return torch.sparse_coo_tensor(i, v, coo.shape, device=device).coalesce()


def compute_structural_node_features(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    if num_nodes <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if edge_index.size == 0:
        return np.zeros((num_nodes, 2), dtype=np.float32)

    src, dst = edge_index
    A = sp.coo_matrix(
        (np.ones(src.shape[0], dtype=np.float32), (src, dst)),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    A.sum_duplicates()
    A.data[:] = 1.0

    degree = np.asarray(A.sum(axis=1)).reshape(-1).astype(np.float32)
    degree_centrality = degree / float(max(num_nodes - 1, 1))

    pr = np.full(num_nodes, 1.0 / max(num_nodes, 1), dtype=np.float64)
    degree64 = degree.astype(np.float64)
    inv_degree = np.zeros_like(degree64)
    nonzero = degree64 > 0
    inv_degree[nonzero] = 1.0 / degree64[nonzero]
    P = sp.diags(inv_degree).dot(A).tocsr().astype(np.float64)
    dangling = ~nonzero
    alpha = 0.85
    teleport = (1.0 - alpha) / max(num_nodes, 1)
    for _ in range(100):
        dangling_mass = pr[dangling].sum()
        pr_next = alpha * (P.T @ pr + dangling_mass / max(num_nodes, 1)) + teleport
        if np.abs(pr_next - pr).sum() < 1e-8:
            pr = pr_next
            break
        pr = pr_next

    features = np.stack([degree_centrality, pr.astype(np.float32)], axis=1)
    feat_mean = features.mean(axis=0, keepdims=True)
    feat_std = features.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std == 0, 1.0, feat_std)
    return ((features - feat_mean) / feat_std).astype(np.float32)


def build_diffusion_supports(edge_index_undirected: np.ndarray, num_nodes: int, device: torch.device):
    A = sp.coo_matrix(
        (
            np.ones(edge_index_undirected.shape[1], dtype=np.float32),
            (edge_index_undirected[0], edge_index_undirected[1]),
        ),
        shape=(num_nodes, num_nodes),
    )
    A_wave = get_normalized_adj(A)
    A_q = calculate_random_walk_matrix(A_wave).T
    A_h = calculate_random_walk_matrix(A_wave.T).T

    if sp.issparse(A_q):
        A_q = scipy_to_torch_sparse(A_q, device)
        A_h = scipy_to_torch_sparse(A_h, device)
    else:
        A_q = torch.from_numpy(A_q.astype("float32")).to(device=device)
        A_h = torch.from_numpy(A_h.astype("float32")).to(device=device)
    return A_q, A_h

