from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.deterministic_baselines import (
    CausalTemporalConv,
    DiffusionConv,
    SparseGraphAttentionLayer,
    _edge_index_from_supports,
)

EPS = 1e-6
LOGNORMAL_EXP_CLAMP = 20.0


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    log2 = -0.6931471805599453
    return torch.where(
        x < log2,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def negative_binomial_nll_loss(
        y: torch.Tensor,
        n: torch.Tensor,
        p: torch.Tensor,
        y_mask: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y = y.clamp_min(0.0)
    n = n.clamp_min(eps)
    p = p.clamp(eps, 1.0 - eps)
    log_prob = (
            torch.lgamma(n + y)
            - torch.lgamma(n)
            - torch.lgamma(y + 1.0)
            + n * torch.log(p)
            + y * torch.log1p(-p)
    )
    nll = -log_prob
    if y_mask is not None:
        nll = nll * y_mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def zero_inflated_nb_nll_loss(
        y: torch.Tensor,
        n: torch.Tensor,
        p: torch.Tensor,
        pi: torch.Tensor,
        y_mask: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y = y.clamp_min(0.0)
    n = n.clamp_min(eps)
    p = p.clamp(eps, 1.0 - eps)
    pi = pi.clamp(eps, 1.0 - eps)

    log_nb = (
            torch.lgamma(n + y)
            - torch.lgamma(n)
            - torch.lgamma(y + 1.0)
            + n * torch.log(p)
            + y * torch.log1p(-p)
    )
    zero_mass = pi + (1.0 - pi) * torch.exp(n * torch.log(p))
    log_prob_zero = torch.log(zero_mass.clamp_min(eps))
    log_prob_pos = torch.log1p(-pi) + log_nb
    log_prob = torch.where(y <= 0.0, log_prob_zero, log_prob_pos)

    nll = -log_prob
    if y_mask is not None:
        nll = nll * y_mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def gaussian_nll_loss(
        y: torch.Tensor,
        loc: torch.Tensor,
        scale: torch.Tensor,
        y_mask: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y = y.clamp_min(0.0)
    scale = scale.clamp_min(eps)
    dist = torch.distributions.Normal(loc, scale)
    nll = -dist.log_prob(y)
    if y_mask is not None:
        nll = nll * y_mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def negative_binomial_mean(n: torch.Tensor, p: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    n = n.clamp_min(eps)
    return n * (1.0 - p) / p


def zero_inflated_nb_mean(n: torch.Tensor, p: torch.Tensor, pi: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    pi = pi.clamp(eps, 1.0 - eps)
    return (1.0 - pi) * negative_binomial_mean(n, p, eps=eps)


def log_nb_pmf_mu_theta(
        y: torch.Tensor,
        mu: torch.Tensor,
        theta: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.float()
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)
    log_theta_mu = torch.log(theta + mu + eps)
    return (
            torch.lgamma(y + theta)
            - torch.lgamma(theta)
            - torch.lgamma(y + 1.0)
            + theta * (torch.log(theta + eps) - log_theta_mu)
            + y * (torch.log(mu + eps) - log_theta_mu)
    )


def nb_zero_prob_mu_theta(mu: torch.Tensor, theta: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    mu = mu.clamp_min(eps)
    theta = theta.clamp_min(eps)
    return torch.exp(theta * (torch.log(theta + eps) - torch.log(theta + mu + eps)))


def log_gamma_pdf_mu_alpha(
        y: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.clamp_min(eps)
    mu = mu.clamp_min(eps)
    alpha = alpha.clamp_min(eps)
    return (
            alpha * (torch.log(alpha) - torch.log(mu))
            - torch.lgamma(alpha)
            + (alpha - 1.0) * torch.log(y)
            - alpha * y / mu
    )


def hurdle_gamma_nll_loss(
        y: torch.Tensor,
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        z_occ: torch.Tensor | None = None,
        event_occ_lambda: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        event_pos_weight: torch.Tensor | None = None,
        positive_loss_normalization: str = "standard",
        eps: float = EPS,
        reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    y = y.float().clamp_min(0.0)
    z = (y > 0).float()

    occ_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        occ_kwargs["pos_weight"] = pos_weight
    occ_nll = F.binary_cross_entropy_with_logits(q_logits, z, **occ_kwargs)

    safe_y = torch.where(z > 0, y, torch.ones_like(y))
    log_pdf = log_gamma_pdf_mu_alpha(safe_y, mu=mu, alpha=alpha, eps=eps)
    pos_nll = -log_pdf
    total = occ_nll + z * pos_nll
    aux_occ = torch.zeros_like(total)
    if z_occ is not None and event_occ_lambda > 0.0:
        event_kwargs = {"reduction": "none"}
        if event_pos_weight is not None:
            event_kwargs["pos_weight"] = event_pos_weight
        aux_occ = F.binary_cross_entropy_with_logits(q_logits, z_occ.float(), **event_kwargs)
        total = total + event_occ_lambda * aux_occ

    if reduction == "sum":
        return {
            "loss": total.sum(),
            "occ_nll": occ_nll.sum(),
            "pos_nll": (z * pos_nll).sum(),
            "aux_occ": aux_occ.sum() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                       device=y.device),
        }
    if reduction == "none":
        return {
            "loss": total,
            "occ_nll": occ_nll,
            "pos_nll": z * pos_nll,
            "aux_occ": aux_occ,
        }
    if positive_loss_normalization == "standard":
        mean_loss = total.mean()
    elif positive_loss_normalization == "balanced":
        mean_loss = occ_nll.mean() + (z * pos_nll).sum() / z.sum().clamp_min(1.0)
        if z_occ is not None and event_occ_lambda > 0.0:
            mean_loss = mean_loss + float(event_occ_lambda) * aux_occ.mean()
    else:
        raise ValueError(
            "positive_loss_normalization must be 'standard' or 'balanced', "
            f"got {positive_loss_normalization!r}"
        )
    return {
        "loss": mean_loss,
        "occ_nll": occ_nll.mean(),
        "pos_nll": (z * pos_nll).sum() / z.sum().clamp_min(1.0),
        "aux_occ": aux_occ.mean() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                    device=y.device),
    }


def hurdle_gamma_mean(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    return q * mu.clamp_min(eps)


def gamma_nll_loss(
        y: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y_safe = y.float().clamp_min(eps)
    nll = -log_gamma_pdf_mu_alpha(y_safe, mu=mu, alpha=alpha, eps=eps)
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def gamma_mean(
        mu: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    return mu.clamp_min(eps)


@torch.no_grad()
def sample_hurdle_gamma(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        n_samples: int = 100,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    mu = mu.clamp_min(eps)
    alpha = alpha.clamp_min(eps)
    beta = alpha / mu
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=beta)

    shape = (int(n_samples),) + tuple(q.shape)
    bern = torch.bernoulli(q.expand(shape))
    gamma_samples = gamma_dist.sample((int(n_samples),))
    return bern * gamma_samples


def log_lognormal_pdf(
        y: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.clamp_min(eps)
    sigma_log = sigma_log.clamp_min(eps)
    log_y = torch.log(y)
    z = (log_y - mu_log) / sigma_log
    return (
            -torch.log(y)
            - torch.log(sigma_log)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * z * z
    )


def hurdle_lognormal_nll_loss(
        y: torch.Tensor,
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        z_occ: torch.Tensor | None = None,
        event_occ_lambda: float = 0.0,
        pos_weight: torch.Tensor | None = None,
        event_pos_weight: torch.Tensor | None = None,
        positive_loss_normalization: str = "standard",
        eps: float = EPS,
        reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    y = y.float().clamp_min(0.0)
    z = (y > 0).float()

    occ_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        occ_kwargs["pos_weight"] = pos_weight
    occ_nll = F.binary_cross_entropy_with_logits(q_logits, z, **occ_kwargs)

    safe_y = torch.where(z > 0, y, torch.ones_like(y))
    log_pdf = log_lognormal_pdf(safe_y, mu_log=mu_log, sigma_log=sigma_log, eps=eps)
    pos_nll = -log_pdf
    total = occ_nll + z * pos_nll
    aux_occ = torch.zeros_like(total)
    if z_occ is not None and event_occ_lambda > 0.0:
        event_kwargs = {"reduction": "none"}
        if event_pos_weight is not None:
            event_kwargs["pos_weight"] = event_pos_weight
        aux_occ = F.binary_cross_entropy_with_logits(q_logits, z_occ.float(), **event_kwargs)
        total = total + event_occ_lambda * aux_occ

    if reduction == "sum":
        return {
            "loss": total.sum(),
            "occ_nll": occ_nll.sum(),
            "pos_nll": (z * pos_nll).sum(),
            "aux_occ": aux_occ.sum() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                       device=y.device),
        }
    if reduction == "none":
        return {
            "loss": total,
            "occ_nll": occ_nll,
            "pos_nll": z * pos_nll,
            "aux_occ": aux_occ,
        }
    if positive_loss_normalization == "standard":
        mean_loss = total.mean()
    elif positive_loss_normalization == "balanced":
        mean_loss = occ_nll.mean() + (z * pos_nll).sum() / z.sum().clamp_min(1.0)
        if z_occ is not None and event_occ_lambda > 0.0:
            mean_loss = mean_loss + float(event_occ_lambda) * aux_occ.mean()
    else:
        raise ValueError(
            "positive_loss_normalization must be 'standard' or 'balanced', "
            f"got {positive_loss_normalization!r}"
        )
    return {
        "loss": mean_loss,
        "occ_nll": occ_nll.mean(),
        "pos_nll": (z * pos_nll).sum() / z.sum().clamp_min(1.0),
        "aux_occ": aux_occ.mean() if z_occ is not None and event_occ_lambda > 0.0 else torch.tensor(0.0,
                                                                                                    device=y.device),
    }


def hurdle_lognormal_mean(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    exp_term = torch.clamp(mu_log + 0.5 * sigma_log * sigma_log, max=LOGNORMAL_EXP_CLAMP)
    cond_mean = torch.exp(exp_term)
    return q * cond_mean


def pairwise_event_ranking_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_pairs: int = 4096,
        hard_negative_fraction: float = 0.5,
) -> torch.Tensor:
    """Sample positive-negative pairs and optimize their relative event scores."""
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1.")
    hard_negative_fraction = float(hard_negative_fraction)
    if not 0.0 <= hard_negative_fraction <= 1.0:
        raise ValueError("hard_negative_fraction must be in [0, 1].")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1, labels.shape[-1]) > 0
    horizon_losses = []
    for horizon_idx in range(flat_logits.shape[-1]):
        step_scores = flat_logits[:, horizon_idx]
        step_labels = flat_labels[:, horizon_idx]
        positive_scores = step_scores[step_labels]
        negative_scores = step_scores[~step_labels]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            continue

        pair_count = min(int(max_pairs), max(positive_scores.numel(), negative_scores.numel()))
        positive_idx = torch.randint(positive_scores.numel(), (pair_count,), device=logits.device)
        sampled_positive = positive_scores[positive_idx]

        hard_count = min(
            int(round(pair_count * hard_negative_fraction)),
            negative_scores.numel(),
        )
        negative_parts = []
        if hard_count > 0:
            negative_parts.append(torch.topk(negative_scores, k=hard_count, sorted=False).values)
        random_count = pair_count - hard_count
        if random_count > 0:
            random_idx = torch.randint(negative_scores.numel(), (random_count,), device=logits.device)
            negative_parts.append(negative_scores[random_idx])
        sampled_negative = torch.cat(negative_parts, dim=0)
        if sampled_negative.numel() > 1:
            sampled_negative = sampled_negative[torch.randperm(sampled_negative.numel(), device=logits.device)]

        horizon_losses.append(F.softplus(sampled_negative - sampled_positive).mean())

    if not horizon_losses:
        return logits.sum() * 0.0
    return torch.stack(horizon_losses).mean()


def cross_space_discrepancy_ranking_loss(
        logits: torch.Tensor,
        y_event: torch.Tensor,
        y_target: torch.Tensor,
        max_pairs: int = 4096,
        base_margin: float = 0.0,
        target_margin_scale: float = 0.5,
        eps: float = EPS,
) -> torch.Tensor:
    """Rank raw events above proxy-positive/raw-event-negative road segments."""
    if logits.shape != y_event.shape or logits.shape != y_target.shape:
        raise ValueError("logits, y_event, and y_target must have identical shapes.")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1.")
    if base_margin < 0.0 or target_margin_scale < 0.0:
        raise ValueError("Discrepancy margins must be non-negative.")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_events = (y_event.reshape(-1, y_event.shape[-1]) > 0)
    flat_targets = y_target.reshape(-1, y_target.shape[-1]).float().clamp_min(0.0)
    horizon_losses = []
    for horizon_idx in range(flat_logits.shape[-1]):
        step_scores = flat_logits[:, horizon_idx]
        event_mask = flat_events[:, horizon_idx]
        target_values = flat_targets[:, horizon_idx]
        discrepancy_mask = (~event_mask) & (target_values > 0)
        positive_scores = step_scores[event_mask]
        negative_scores = step_scores[discrepancy_mask]
        negative_targets = target_values[discrepancy_mask]
        if positive_scores.numel() == 0 or negative_scores.numel() == 0:
            continue

        pair_count = min(int(max_pairs), max(positive_scores.numel(), negative_scores.numel()))
        positive_idx = torch.randint(positive_scores.numel(), (pair_count,), device=logits.device)
        sampled_positive = positive_scores[positive_idx]

        if negative_scores.numel() >= pair_count:
            negative_idx = torch.topk(negative_targets, k=pair_count, sorted=False).indices
        else:
            negative_idx = torch.randint(negative_scores.numel(), (pair_count,), device=logits.device)
        sampled_negative = negative_scores[negative_idx]
        sampled_target = negative_targets[negative_idx]
        target_scale = negative_targets.max().clamp_min(float(eps))
        adaptive_margin = (
            float(base_margin)
            + float(target_margin_scale) * (sampled_target / target_scale).clamp(0.0, 1.0)
        )
        horizon_losses.append(
            F.softplus(adaptive_margin + sampled_negative - sampled_positive).mean()
        )

    if not horizon_losses:
        return logits.sum() * 0.0
    return torch.stack(horizon_losses).mean()


def _bound_cross_pair_sides(
        n_neg: int,
        n_pos: int,
        max_pairs: int,
) -> Tuple[int, int]:
    """Bound two boundary sets so their cross product never exceeds max_pairs.

    The two sides are truncated symmetrically around a square budget: each side
    is first capped at ``isqrt(max_pairs)`` elements, then the negative side is
    kept as-is while the positive side is limited to ``max_pairs // n_neg_used``
    so the broadcast cross product is always ``<= max_pairs``.
    """
    n_neg = int(n_neg)
    n_pos = int(n_pos)
    if n_neg * n_pos <= max_pairs:
        return n_neg, n_pos
    side = int(math.isqrt(max_pairs))
    neg_use = min(n_neg, side)
    pos_use = min(n_pos, max_pairs // neg_use)
    return neg_use, max(1, pos_use)


def multi_budget_boundary_ranking_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        top_k_ratios=(0.05, 0.10, 0.20),
        max_pairs: int = 4096,
        margin: float = 0.0,
) -> torch.Tensor:
    """Deterministic multi-budget Top-k boundary-swap event ranking loss.

    For every horizon and every budget ratio the batch/node scores are sorted in
    descending order (using a detached ordering, so the hard-example mining is
    deterministic). Negatives inside the exact ``ceil(ratio * n)`` prefix are
    the "selected negatives" and positives outside that prefix are the "missed
    positives". Only the examples nearest the boundary are kept on each side
    (lowest-score negatives inside, highest-score positives outside) and paired
    cross-wise so the number of pairs never exceeds ``max_pairs``. Each pair is
    penalized by ``softplus(margin + negative - positive)`` and all valid
    horizon/budget groups are averaged.
    """
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes.")
    if logits.dim() != 3:
        raise ValueError("logits and labels must have shape [batch, nodes, horizon].")
    if not bool(torch.isfinite(labels).all()) or bool((labels < 0).any()):
        raise ValueError("labels must be finite and non-negative.")

    if isinstance(top_k_ratios, (float, int)):
        ratios = (float(top_k_ratios),)
    else:
        ratios = tuple(top_k_ratios)
    if not ratios:
        raise ValueError("top_k_ratios must contain at least one ratio.")
    ratios = [float(ratio) for ratio in ratios]
    for ratio in ratios:
        if not math.isfinite(ratio) or not (0.0 < ratio < 1.0):
            raise ValueError("top_k_ratios must be finite and strictly in (0, 1).")

    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer.")

    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be finite and non-negative.")

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1, labels.shape[-1])
    horizon_count = flat_logits.shape[-1]

    group_losses: list[torch.Tensor] = []
    for horizon_idx in range(horizon_count):
        scores = flat_logits[:, horizon_idx]
        step_labels = flat_labels[:, horizon_idx] > 0
        n = scores.numel()
        if n == 0:
            continue
        order = torch.argsort(scores.detach(), dim=0, descending=True)
        sorted_labels = step_labels[order]
        for ratio in ratios:
            k = int(math.ceil(ratio * n))
            if k >= n:
                continue
            inside = sorted_labels[:k]
            outside = sorted_labels[k:]
            neg_inside_idx = torch.nonzero(~inside, as_tuple=False).reshape(-1)
            pos_outside_idx = torch.nonzero(outside, as_tuple=False).reshape(-1)
            if neg_inside_idx.numel() == 0 or pos_outside_idx.numel() == 0:
                continue
            # Nearest boundary: negatives inside with the highest sorted index
            # (lowest scores), positives outside with the lowest sorted index
            # (highest scores).
            neg_order = neg_inside_idx.flip(0)
            pos_order = pos_outside_idx + k

            neg_use, pos_use = _bound_cross_pair_sides(
                neg_order.numel(), pos_order.numel(), int(max_pairs)
            )
            selected_neg = scores[order[neg_order[:neg_use]]]
            selected_pos = scores[order[pos_order[:pos_use]]]
            diffs = selected_neg[:, None] - selected_pos[None, :]
            group_losses.append(F.softplus(margin + diffs).mean())

    if not group_losses:
        return logits.sum() * 0.0
    return torch.stack(group_losses).mean()


def event_occurrence_loss(
        logits: torch.Tensor,
        y_event: torch.Tensor,
        loss_kind: str = "weighted_bce",
        pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 2.0,
        rank_lambda: float = 1.0,
        rank_max_pairs: int = 4096,
        rank_hard_negative_fraction: float = 0.5,
        y_target: torch.Tensor | None = None,
        discrepancy_base_margin: float = 0.0,
        discrepancy_target_margin_scale: float = 0.5,
        rank_ratios=(0.05, 0.10, 0.20),
        boundary_margin: float = 0.0,
        sample_weight: torch.Tensor = None,
) -> dict[str, torch.Tensor]:
    """Event classification/ranking objective shared by direct and DAER baselines."""
    if loss_kind not in {
        "weighted_bce",
        "focal",
        "pairwise",
        "hybrid",
        "discrepancy_hybrid",
        "boundary_hybrid",
        "boundary",
    }:
        raise ValueError(f"Unsupported event loss: {loss_kind}")
    if focal_gamma < 0.0:
        raise ValueError("focal_gamma must be non-negative.")
    if rank_lambda < 0.0:
        raise ValueError("rank_lambda must be non-negative.")
    if discrepancy_base_margin < 0.0 or discrepancy_target_margin_scale < 0.0:
        raise ValueError("Discrepancy margins must be non-negative.")
    if not math.isfinite(float(boundary_margin)) or boundary_margin < 0.0:
        raise ValueError("boundary_margin must be finite and non-negative.")

    labels = (y_event > 0).float()
    bce_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        bce_kwargs["pos_weight"] = pos_weight
    pointwise = F.binary_cross_entropy_with_logits(logits, labels, **bce_kwargs)

    if sample_weight is not None:
        if loss_kind != "weighted_bce":
            raise NotImplementedError("sample_weight masking currently supports weighted_bce only.")
        denom = (sample_weight.sum() * logits.shape[-1]).clamp(min=1.0)
        classification = (pointwise * sample_weight.unsqueeze(-1)).sum() / denom
    elif loss_kind == "focal":
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
        classification = ((1.0 - p_t).pow(float(focal_gamma)) * pointwise).mean()
    elif loss_kind in {"weighted_bce", "hybrid", "discrepancy_hybrid", "boundary_hybrid"}:
        classification = pointwise.mean()
    else:
        classification = logits.sum() * 0.0

    if loss_kind in {"pairwise", "hybrid"}:
        ranking = pairwise_event_ranking_loss(
            logits,
            labels,
            max_pairs=rank_max_pairs,
            hard_negative_fraction=rank_hard_negative_fraction,
        )
    elif loss_kind == "discrepancy_hybrid":
        if y_target is None:
            raise ValueError("discrepancy_hybrid requires y_target.")
        ranking = cross_space_discrepancy_ranking_loss(
            logits,
            y_event,
            y_target,
            max_pairs=rank_max_pairs,
            base_margin=discrepancy_base_margin,
            target_margin_scale=discrepancy_target_margin_scale,
        )
    elif loss_kind in {"boundary_hybrid", "boundary"}:
        ranking = multi_budget_boundary_ranking_loss(
            logits,
            labels,
            top_k_ratios=rank_ratios,
            max_pairs=rank_max_pairs,
            margin=boundary_margin,
        )
    else:
        ranking = logits.sum() * 0.0

    total = classification + float(rank_lambda) * ranking
    return {
        "loss": total,
        "event_cls": classification,
        "event_rank": ranking,
    }


def cumulative_hazard_nll_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        horizons=(1, 3, 7, 14),
        pos_weight_cap: float = 20.0,
) -> torch.Tensor:
    """Cumulative-hazard auxiliary objective derived from daily event logits.

    For each requested horizon ``h`` the target is whether any event occurs in
    the first ``h`` days and the predicted probability is the cumulative hazard
    ``1 - prod_{d<=h}(1 - sigmoid(logit_d))``. The loss is a weighted binary NLL
    computed in log-survival space, averaged over the requested horizons. Each
    horizon's positive weight is derived from its own labels as ``neg/pos``,
    floored at one and capped by ``pos_weight_cap``. Horizons whose batch has no
    positives contribute a connected zero so gradients still flow.
    """
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have identical shapes.")
    if logits.dim() != 3:
        raise ValueError("logits and labels must have shape [batch, nodes, horizon].")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite.")
    if not bool(torch.isfinite(labels).all()) or bool((labels < 0).any()):
        raise ValueError("labels must be finite and non-negative.")

    horizon = logits.shape[-1]
    if isinstance(horizons, int):
        horizon_list = [horizons]
    else:
        try:
            raw_horizons = list(horizons)
        except TypeError as exc:
            raise ValueError("horizons must be an int or an iterable of ints.") from exc
        horizon_list = []
        for h in raw_horizons:
            if isinstance(h, bool) or not isinstance(h, (int, np.integer)):
                raise ValueError(f"cumulative horizons must be integers, got {h!r}.")
            horizon_list.append(int(h))
    if not horizon_list:
        raise ValueError("horizons must contain at least one horizon.")
    for h in horizon_list:
        if h < 1 or h > horizon:
            raise ValueError("cumulative horizons must be within [1, horizon].")
    if len(set(horizon_list)) != len(horizon_list):
        raise ValueError("cumulative horizons must be unique.")
    if not math.isfinite(float(pos_weight_cap)) or float(pos_weight_cap) < 1.0:
        raise ValueError("pos_weight_cap must be finite and >= 1.")

    event = (labels > 0).float()
    log_survival = -F.softplus(logits)
    log_cum_survival = torch.cumsum(log_survival, dim=-1)
    cum_any = (torch.cumsum(event, dim=-1) > 0).float()

    horizon_losses: list[torch.Tensor] = []
    for h in horizon_list:
        log_survival_h = log_cum_survival[..., h - 1]
        z = cum_any[..., h - 1]
        pos = z.sum()
        if float(pos) <= 0.0:
            horizon_losses.append(logits.sum() * 0.0)
            continue
        neg = z.numel() - pos
        pos_weight = torch.clamp(neg / pos, min=1.0, max=float(pos_weight_cap))
        log_p = torch.log(-torch.expm1(log_survival_h))
        log_1mp = log_survival_h
        nll = -(z * log_p + (1.0 - z) * log_1mp)
        weights = torch.where(z > 0.5, pos_weight, torch.ones_like(z))
        horizon_losses.append((nll * weights).mean())

    return torch.stack(horizon_losses).mean()


def daer_hurdle_lognormal_loss(
        y_target: torch.Tensor,
        y_event: torch.Tensor,
        q_target_logits: torch.Tensor,
        q_event_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        event_lambda: float = 0.5,
        event_pos_weight: torch.Tensor | None = None,
        event_loss_kind: str = "weighted_bce",
        event_focal_gamma: float = 2.0,
        event_rank_lambda: float = 1.0,
        event_rank_max_pairs: int = 4096,
        event_rank_hard_negative_fraction: float = 0.5,
        event_discrepancy_base_margin: float = 0.0,
        event_discrepancy_target_margin_scale: float = 0.5,
        event_rank_ratios=(0.05, 0.10, 0.20),
        event_boundary_margin: float = 0.0,
        positive_loss_normalization: str = "standard",
) -> dict[str, torch.Tensor]:
    target_loss = hurdle_lognormal_nll_loss(
        y_target,
        q_target_logits,
        mu_log,
        sigma_log,
        positive_loss_normalization=positive_loss_normalization,
    )
    event_loss = event_occurrence_loss(
        q_event_logits,
        y_event,
        loss_kind=event_loss_kind,
        pos_weight=event_pos_weight,
        focal_gamma=event_focal_gamma,
        rank_lambda=event_rank_lambda,
        rank_max_pairs=event_rank_max_pairs,
        rank_hard_negative_fraction=event_rank_hard_negative_fraction,
        y_target=y_target,
        discrepancy_base_margin=event_discrepancy_base_margin,
        discrepancy_target_margin_scale=event_discrepancy_target_margin_scale,
        rank_ratios=event_rank_ratios,
        boundary_margin=event_boundary_margin,
    )
    total = (
            target_loss["loss"]
            + float(event_lambda) * event_loss["loss"]
    )
    return {
        "loss": total,
        "ts_loss": target_loss["loss"],
        "occ_nll": target_loss["occ_nll"],
        "pos_nll": target_loss["pos_nll"],
        "event_cls": event_loss["event_cls"],
        "event_rank": event_loss["event_rank"],
    }


def event_aligned_hurdle_lognormal_loss(
        y_target: torch.Tensor,
        y_event: torch.Tensor,
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        event_lambda: float = 0.5,
        event_pos_weight: torch.Tensor | None = None,
        event_loss_kind: str = "weighted_bce",
        event_focal_gamma: float = 2.0,
        event_rank_lambda: float = 1.0,
        event_rank_max_pairs: int = 4096,
        event_rank_hard_negative_fraction: float = 0.5,
        event_discrepancy_base_margin: float = 0.0,
        event_discrepancy_target_margin_scale: float = 0.5,
        event_rank_ratios=(0.05, 0.10, 0.20),
        event_boundary_margin: float = 0.0,
        positive_loss_normalization: str = "standard",
) -> dict[str, torch.Tensor]:
    target_loss = hurdle_lognormal_nll_loss(
        y_target,
        q_logits,
        mu_log,
        sigma_log,
        positive_loss_normalization=positive_loss_normalization,
    )
    event_loss = event_occurrence_loss(
        q_logits,
        y_event,
        loss_kind=event_loss_kind,
        pos_weight=event_pos_weight,
        focal_gamma=event_focal_gamma,
        rank_lambda=event_rank_lambda,
        rank_max_pairs=event_rank_max_pairs,
        rank_hard_negative_fraction=event_rank_hard_negative_fraction,
        y_target=y_target,
        discrepancy_base_margin=event_discrepancy_base_margin,
        discrepancy_target_margin_scale=event_discrepancy_target_margin_scale,
        rank_ratios=event_rank_ratios,
        boundary_margin=event_boundary_margin,
    )
    total = target_loss["loss"] + float(event_lambda) * event_loss["loss"]
    return {
        "loss": total,
        "ts_loss": target_loss["loss"],
        "occ_nll": target_loss["occ_nll"],
        "pos_nll": target_loss["pos_nll"],
        "event_cls": event_loss["event_cls"],
        "event_rank": event_loss["event_rank"],
    }


def daer_hurdle_gamma_loss(
        y_target: torch.Tensor,
        y_event: torch.Tensor,
        q_target_logits: torch.Tensor,
        q_event_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        event_lambda: float = 0.5,
        event_pos_weight: torch.Tensor | None = None,
        event_loss_kind: str = "weighted_bce",
        event_focal_gamma: float = 2.0,
        event_rank_lambda: float = 1.0,
        event_rank_max_pairs: int = 4096,
        event_rank_hard_negative_fraction: float = 0.5,
        event_discrepancy_base_margin: float = 0.0,
        event_discrepancy_target_margin_scale: float = 0.5,
        event_rank_ratios=(0.05, 0.10, 0.20),
        event_boundary_margin: float = 0.0,
        positive_loss_normalization: str = "standard",
) -> dict[str, torch.Tensor]:
    target_loss = hurdle_gamma_nll_loss(
        y_target,
        q_target_logits,
        mu,
        alpha,
        positive_loss_normalization=positive_loss_normalization,
    )
    event_loss = event_occurrence_loss(
        q_event_logits,
        y_event,
        loss_kind=event_loss_kind,
        pos_weight=event_pos_weight,
        focal_gamma=event_focal_gamma,
        rank_lambda=event_rank_lambda,
        rank_max_pairs=event_rank_max_pairs,
        rank_hard_negative_fraction=event_rank_hard_negative_fraction,
        y_target=y_target,
        discrepancy_base_margin=event_discrepancy_base_margin,
        discrepancy_target_margin_scale=event_discrepancy_target_margin_scale,
        rank_ratios=event_rank_ratios,
        boundary_margin=event_boundary_margin,
    )
    total = target_loss["loss"] + float(event_lambda) * event_loss["loss"]
    return {
        "loss": total,
        "ts_loss": target_loss["loss"],
        "occ_nll": target_loss["occ_nll"],
        "pos_nll": target_loss["pos_nll"],
        "event_cls": event_loss["event_cls"],
        "event_rank": event_loss["event_rank"],
    }


def event_aligned_hurdle_gamma_loss(
        y_target: torch.Tensor,
        y_event: torch.Tensor,
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        alpha: torch.Tensor,
        event_lambda: float = 0.5,
        event_pos_weight: torch.Tensor | None = None,
        event_loss_kind: str = "weighted_bce",
        event_focal_gamma: float = 2.0,
        event_rank_lambda: float = 1.0,
        event_rank_max_pairs: int = 4096,
        event_rank_hard_negative_fraction: float = 0.5,
        event_discrepancy_base_margin: float = 0.0,
        event_discrepancy_target_margin_scale: float = 0.5,
        event_rank_ratios=(0.05, 0.10, 0.20),
        event_boundary_margin: float = 0.0,
        positive_loss_normalization: str = "standard",
) -> dict[str, torch.Tensor]:
    target_loss = hurdle_gamma_nll_loss(
        y_target,
        q_logits,
        mu,
        alpha,
        positive_loss_normalization=positive_loss_normalization,
    )
    event_loss = event_occurrence_loss(
        q_logits,
        y_event,
        loss_kind=event_loss_kind,
        pos_weight=event_pos_weight,
        focal_gamma=event_focal_gamma,
        rank_lambda=event_rank_lambda,
        rank_max_pairs=event_rank_max_pairs,
        rank_hard_negative_fraction=event_rank_hard_negative_fraction,
        y_target=y_target,
        discrepancy_base_margin=event_discrepancy_base_margin,
        discrepancy_target_margin_scale=event_discrepancy_target_margin_scale,
        rank_ratios=event_rank_ratios,
        boundary_margin=event_boundary_margin,
    )
    total = target_loss["loss"] + float(event_lambda) * event_loss["loss"]
    return {
        "loss": total,
        "ts_loss": target_loss["loss"],
        "occ_nll": target_loss["occ_nll"],
        "pos_nll": target_loss["pos_nll"],
        "event_cls": event_loss["event_cls"],
        "event_rank": event_loss["event_rank"],
    }


def lognormal_nll_loss(
        y: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y_safe = y.float().clamp_min(eps)
    nll = -log_lognormal_pdf(y_safe, mu_log=mu_log, sigma_log=sigma_log, eps=eps)
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def lognormal_mean(
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
) -> torch.Tensor:
    exp_term = torch.clamp(mu_log + 0.5 * sigma_log * sigma_log, max=LOGNORMAL_EXP_CLAMP)
    return torch.exp(exp_term)


@torch.no_grad()
def sample_hurdle_lognormal(
        q_logits: torch.Tensor,
        mu_log: torch.Tensor,
        sigma_log: torch.Tensor,
        n_samples: int = 100,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    sigma_log = sigma_log.clamp_min(eps)
    shape = (int(n_samples),) + tuple(q.shape)
    bern = torch.bernoulli(q.expand(shape))
    eps_std = torch.randn(shape, device=q.device, dtype=q.dtype)
    lognormal_samples = torch.exp(
        torch.clamp(mu_log.unsqueeze(0) + sigma_log.unsqueeze(0) * eps_std, min=-LOGNORMAL_EXP_CLAMP,
                    max=LOGNORMAL_EXP_CLAMP)
    )
    return bern * lognormal_samples


def tweedie_log_p0_torch(
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    mu = mu.clamp_min(eps)
    phi = phi.clamp_min(eps)
    rho = rho.clamp(1.0 + eps, 2.0 - eps)
    lam = torch.pow(mu, 2.0 - rho) / (phi * (2.0 - rho))
    return -lam


def tweedie_positive_loglik_lower_bound(
        y: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    y = y.clamp_min(eps)
    mu = mu.clamp_min(eps)
    phi = phi.clamp_min(eps)
    rho = rho.clamp(1.0 + eps, 2.0 - eps)

    alpha = (2.0 - rho) / (1.0 - rho)
    jmax = torch.pow(y, 2.0 - rho) / ((2.0 - rho) * phi)

    term_exp = (
                       y * torch.pow(mu, 1.0 - rho) / (1.0 - rho)
                       - torch.pow(mu, 2.0 - rho) / (2.0 - rho)
               ) / phi
    term_lb = -torch.log(jmax * torch.sqrt((-alpha) * y) + eps) + jmax * (alpha - 1.0)
    return term_exp + term_lb


def hurdle_tweedie_conditional_nll_loss(
        y: torch.Tensor,
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> dict[str, torch.Tensor]:
    y = y.float().clamp_min(0.0)
    z = (y > 0).float()

    occ_kwargs = {"reduction": "none"}
    if pos_weight is not None:
        occ_kwargs["pos_weight"] = pos_weight
    occ_nll = F.binary_cross_entropy_with_logits(q_logits, z, **occ_kwargs)

    log_p0 = tweedie_log_p0_torch(mu=mu, phi=phi, rho=rho, eps=eps)
    log_one_minus_p0 = log1mexp(log_p0)

    safe_y = torch.where(z > 0, y, torch.ones_like(y))
    td_pos_lb = tweedie_positive_loglik_lower_bound(
        y=safe_y,
        mu=mu,
        phi=phi,
        rho=rho,
        eps=eps,
    )
    cond_pos_loglik_lb = td_pos_lb - log_one_minus_p0
    pos_nll = -cond_pos_loglik_lb
    total = occ_nll + z * pos_nll
    aux_occ = torch.zeros_like(total)

    if reduction == "sum":
        return {
            "loss": total.sum(),
            "occ_nll": occ_nll.sum(),
            "pos_nll": (z * pos_nll).sum(),
            "aux_occ": torch.tensor(0.0, device=y.device),
        }
    if reduction == "none":
        return {
            "loss": total,
            "occ_nll": occ_nll,
            "pos_nll": z * pos_nll,
            "aux_occ": aux_occ,
        }
    return {
        "loss": total.mean(),
        "occ_nll": occ_nll.mean(),
        "pos_nll": (z * pos_nll).sum() / z.sum().clamp_min(1.0),
        "aux_occ": torch.tensor(0.0, device=y.device),
    }


def hurdle_tweedie_mean(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    p0 = torch.exp(tweedie_log_p0_torch(mu=mu, phi=phi, rho=rho, eps=eps))
    cond_mean = mu.clamp_min(eps) / (1.0 - p0).clamp_min(eps)
    return q * cond_mean


@torch.no_grad()
def sample_hurdle_tweedie(
        q_logits: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        n_samples: int = 100,
        eps: float = EPS,
) -> torch.Tensor:
    q = torch.sigmoid(q_logits).clamp(eps, 1.0 - eps)
    mu = mu.clamp_min(eps)
    phi = phi.clamp_min(eps)
    rho = rho.clamp(1.0 + eps, 2.0 - eps)

    lam = torch.pow(mu, 2.0 - rho) / (phi * (2.0 - rho))
    alpha = (2.0 - rho) / (rho - 1.0)
    theta = phi * (rho - 1.0) * torch.pow(mu, rho - 1.0)
    alpha = alpha.clamp_min(eps)
    theta = theta.clamp_min(eps)

    shape = (int(n_samples),) + tuple(q.shape)
    bern = torch.bernoulli(q.expand(shape))
    lam_expand = lam.unsqueeze(0).expand(shape)
    counts = torch.poisson(lam_expand)
    positive_gate = bern > 0
    pos_count_mask = positive_gate & (counts <= 0)
    while pos_count_mask.any():
        counts[pos_count_mask] = torch.poisson(lam_expand[pos_count_mask])
        pos_count_mask = positive_gate & (counts <= 0)

    samples = torch.zeros(shape, device=q.device, dtype=q.dtype)
    gamma_mask = counts > 0
    if gamma_mask.any():
        gamma_shape = counts * alpha.unsqueeze(0)
        gamma_rate = (1.0 / theta).unsqueeze(0).expand_as(gamma_shape)
        gamma_dist = torch.distributions.Gamma(
            concentration=gamma_shape[gamma_mask],
            rate=gamma_rate[gamma_mask],
        )
        samples[gamma_mask] = gamma_dist.sample()
    return samples


def tweedie_nll_loss(
        y: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        y_mask: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y = y.clamp_min(0.0)
    mu = mu.clamp_min(eps)
    phi = phi.clamp_min(eps)
    rho = rho.clamp(1.0 + eps, 2.0 - eps)

    y_pos = y.clamp_min(eps)
    one_minus_rho = 1.0 - rho
    two_minus_rho = 2.0 - rho

    term1 = (
                    y_pos * torch.pow(mu, one_minus_rho) / one_minus_rho
                    - torch.pow(mu, two_minus_rho) / two_minus_rho
            ) / phi
    alpha = two_minus_rho / one_minus_rho
    j_max = torch.pow(y_pos, two_minus_rho) / (two_minus_rho * phi)
    log_term = torch.log(j_max * torch.sqrt((-alpha).clamp_min(eps)) * y_pos + eps)
    log_prob_pos = term1 - log_term + j_max * (alpha - 1.0)
    log_prob_zero = -torch.pow(mu, two_minus_rho) / (phi * two_minus_rho)
    log_prob = torch.where(y <= 0.0, log_prob_zero, log_prob_pos)

    nll = -log_prob
    if y_mask is not None:
        nll = nll * y_mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def zero_inflated_tweedie_nll_loss(
        y: torch.Tensor,
        pi: torch.Tensor,
        mu: torch.Tensor,
        phi: torch.Tensor,
        rho: torch.Tensor,
        y_mask: torch.Tensor | None = None,
        eps: float = EPS,
        reduction: str = "mean",
) -> torch.Tensor:
    y = y.clamp_min(0.0)
    pi = pi.clamp(eps, 1.0 - eps)
    mu = mu.clamp_min(eps)
    phi = phi.clamp_min(eps)
    rho = rho.clamp(1.0 + eps, 2.0 - eps)

    y_pos = y.clamp_min(eps)
    one_minus_rho = 1.0 - rho
    two_minus_rho = 2.0 - rho

    term1 = (
                    y_pos * torch.pow(mu, one_minus_rho) / one_minus_rho
                    - torch.pow(mu, two_minus_rho) / two_minus_rho
            ) / phi
    alpha = two_minus_rho / one_minus_rho
    j_max = torch.pow(y_pos, two_minus_rho) / (two_minus_rho * phi)
    log_term = torch.log(j_max * torch.sqrt((-alpha).clamp_min(eps)) * y_pos + eps)
    log_prob_td_pos = term1 - log_term + j_max * (alpha - 1.0)
    log_prob_td_zero = -torch.pow(mu, two_minus_rho) / (phi * two_minus_rho)

    log_prob_pos = torch.log1p(-pi) + log_prob_td_pos
    log_prob_zero = torch.logaddexp(torch.log(pi), torch.log1p(-pi) + log_prob_td_zero)
    log_prob = torch.where(y <= 0.0, log_prob_zero, log_prob_pos)

    nll = -log_prob
    if y_mask is not None:
        nll = nll * y_mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


def tweedie_mean(mu: torch.Tensor) -> torch.Tensor:
    return mu.clamp_min(EPS)


def zero_inflated_tweedie_mean(pi: torch.Tensor, mu: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    pi = pi.clamp(eps, 1.0 - eps)
    return (1.0 - pi) * tweedie_mean(mu)


def _to_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def tweedie_p0_from_params(mu, phi, rho, eps: float = 1e-8) -> np.ndarray:
    mu_np = np.clip(_to_numpy(mu), eps, None)
    phi_np = np.clip(_to_numpy(phi), eps, None)
    rho_np = np.clip(_to_numpy(rho), 1.0 + eps, 2.0 - eps)
    lam = np.power(mu_np, 2.0 - rho_np) / (phi_np * (2.0 - rho_np))
    lam = np.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
    return np.exp(-lam)


def zero_inflated_tweedie_p0_from_params(pi, mu, phi, rho, eps: float = 1e-8) -> np.ndarray:
    pi_np = np.clip(_to_numpy(pi), eps, 1.0 - eps)
    td_zero = tweedie_p0_from_params(mu, phi, rho, eps=eps)
    return pi_np + (1.0 - pi_np) * td_zero


def _tweedie_cp_gamma_params(mu, phi, rho, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu_np = np.clip(_to_numpy(mu), eps, None)
    phi_np = np.clip(_to_numpy(phi), eps, None)
    rho_np = np.clip(_to_numpy(rho), 1.0 + eps, 2.0 - eps)
    lam = np.power(mu_np, 2.0 - rho_np) / (phi_np * (2.0 - rho_np))
    lam = np.nan_to_num(lam, nan=0.0, posinf=1e6, neginf=0.0)
    lam = np.clip(lam, 0.0, 1e6)
    alpha = (2.0 - rho_np) / (rho_np - 1.0)
    theta = phi_np * (rho_np - 1.0) * np.power(mu_np, rho_np - 1.0)
    theta = np.nan_to_num(theta, nan=eps, posinf=1e6, neginf=eps)
    theta = np.clip(theta, eps, 1e6)
    return lam, alpha, theta


def sample_tweedie_from_params(
        mu,
        phi,
        rho,
        num_samples: int = 100,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    mu_np = _to_numpy(mu)
    phi_np = _to_numpy(phi)
    rho_np = _to_numpy(rho)

    shape = np.broadcast(mu_np, phi_np, rho_np).shape
    mu_np = np.broadcast_to(mu_np, shape)
    phi_np = np.broadcast_to(phi_np, shape)
    rho_np = np.broadcast_to(rho_np, shape)

    lam, alpha, theta = _tweedie_cp_gamma_params(mu_np, phi_np, rho_np)
    flat_size = int(np.prod(shape))
    lam_flat = lam.reshape(flat_size)
    alpha_flat = alpha.reshape(flat_size)
    theta_flat = theta.reshape(flat_size)

    counts = rng.poisson(lam=lam_flat[None, :], size=(num_samples, flat_size))
    samples = np.zeros((num_samples, flat_size), dtype=np.float64)
    mask = counts > 0
    if np.any(mask):
        shape_param = counts * alpha_flat[None, :]
        scale_param = np.broadcast_to(theta_flat[None, :], shape_param.shape)
        samples[mask] = rng.gamma(shape=shape_param[mask], scale=scale_param[mask])
    return samples.reshape((num_samples,) + shape)


def tweedie_interval_from_params_mc(
        mu,
        phi,
        rho,
        alpha: float = 0.05,
        num_samples: int = 100,
        rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_tweedie_from_params(mu, phi, rho, num_samples=num_samples, rng=rng)
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


def sample_zero_inflated_tweedie_from_params(
        pi,
        mu,
        phi,
        rho,
        num_samples: int = 100,
        rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    pi_np = np.clip(_to_numpy(pi), EPS, 1.0 - EPS)
    mu_np = _to_numpy(mu)
    phi_np = _to_numpy(phi)
    rho_np = _to_numpy(rho)

    shape = np.broadcast(pi_np, mu_np, phi_np, rho_np).shape
    pi_np = np.broadcast_to(pi_np, shape)
    mu_np = np.broadcast_to(mu_np, shape)
    phi_np = np.broadcast_to(phi_np, shape)
    rho_np = np.broadcast_to(rho_np, shape)

    lam, alpha, theta = _tweedie_cp_gamma_params(mu_np, phi_np, rho_np)
    flat_size = int(np.prod(shape))
    pi_flat = pi_np.reshape(flat_size)
    lam_flat = lam.reshape(flat_size)
    alpha_flat = alpha.reshape(flat_size)
    theta_flat = theta.reshape(flat_size)

    extra_zero = rng.random((num_samples, flat_size)) < pi_flat[None, :]
    counts = rng.poisson(lam=lam_flat[None, :], size=(num_samples, flat_size))
    counts[extra_zero] = 0

    samples = np.zeros((num_samples, flat_size), dtype=np.float64)
    mask = counts > 0
    if np.any(mask):
        shape_param = counts * alpha_flat[None, :]
        scale_param = np.broadcast_to(theta_flat[None, :], shape_param.shape)
        samples[mask] = rng.gamma(shape=shape_param[mask], scale=scale_param[mask])
    return samples.reshape((num_samples,) + shape)


def zero_inflated_tweedie_interval_from_params_mc(
        pi,
        mu,
        phi,
        rho,
        alpha: float = 0.05,
        num_samples: int = 100,
        rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    samples = sample_zero_inflated_tweedie_from_params(pi, mu, phi, rho, num_samples=num_samples, rng=rng)
    lower = np.quantile(samples, alpha / 2.0, axis=0)
    upper = np.quantile(samples, 1.0 - alpha / 2.0, axis=0)
    return lower, upper


class TemporalEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            dropout: float,
            temporal_backbone: str = "tcn",
            gru_layers: int = 3,
    ):
        super().__init__()
        self.temporal_backbone = str(temporal_backbone)
        self.hidden_dim = int(hidden_dim)
        if self.temporal_backbone == "tcn":
            self.block1 = CausalTemporalConv(input_dim, hidden_dim, kernel_size, dropout)
            self.block2 = CausalTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout)
            self.block3 = CausalTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout)
            self.gru = None
        elif self.temporal_backbone == "gru":
            self.block1 = None
            self.block2 = None
            self.block3 = None
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=max(1, int(gru_layers)),
                batch_first=True,
                dropout=float(dropout) if int(gru_layers) > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
        else:
            raise ValueError(f"Unsupported temporal_backbone: {temporal_backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_backbone == "tcn":
            h = x.permute(0, 3, 1, 2).contiguous()
            h = self.block1(h)
            h = self.block2(h)
            h = self.block3(h)
            return h[:, :, :, -1].permute(0, 2, 1).contiguous()

        batch_size, num_nodes, history_len, feat_dim = x.shape
        x_seq = x.reshape(batch_size * num_nodes, history_len, feat_dim)
        h_seq, _ = self.gru(x_seq)
        h_last = self.dropout(h_seq[:, -1, :])
        return h_last.view(batch_size, num_nodes, self.hidden_dim)


class GATSpatialBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        inter_dim = max(1, hidden_dim // max(1, int(heads)))
        self.gat1 = SparseGraphAttentionLayer(hidden_dim, inter_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = SparseGraphAttentionLayer(inter_dim * max(1, int(heads)), hidden_dim, heads=1, dropout=dropout,
                                              concat=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        h = self.dropout(self.gat1(h, edge_index))
        h = self.dropout(self.gat2(h, edge_index))
        return h


class SpatialHistoryEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            history_len: int,
            hidden_dim: int,
            diffusion_order: int,
            dropout: float,
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
    ):
        super().__init__()
        self.spatial_backbone = str(spatial_backbone)
        flat_dim = int(input_dim) * int(history_len)
        self.dropout = nn.Dropout(dropout)
        if self.spatial_backbone == "diffusion":
            self.conv1 = DiffusionConv(flat_dim, hidden_dim, diffusion_order, num_supports=2)
            self.conv2 = DiffusionConv(hidden_dim, hidden_dim, diffusion_order, num_supports=2)
            self.conv3 = DiffusionConv(hidden_dim, hidden_dim, diffusion_order, num_supports=2)
            self.gat_blocks = None
        elif self.spatial_backbone == "gat":
            self.conv1 = None
            self.conv2 = None
            self.conv3 = None
            self.gat_blocks = nn.ModuleList(
                [
                    GATSpatialBlock(flat_dim if idx == 0 else hidden_dim, hidden_dim, gat_heads, dropout)
                    for idx in range(3)
                ]
            )
        else:
            raise ValueError(f"Unsupported spatial_backbone: {spatial_backbone}")

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, history_len, feat_dim = x.shape
        h = x.reshape(batch_size, num_nodes, history_len * feat_dim)
        if self.spatial_backbone == "diffusion":
            supports = [A_q, A_h]
            h = self.dropout(F.relu(self.conv1(h, supports)))
            h = self.dropout(F.relu(self.conv2(h, supports)))
            h = self.dropout(F.relu(self.conv3(h, supports)))
            return h

        edge_index = _edge_index_from_supports(A_q, A_h)
        for block in self.gat_blocks:
            h = block(h, edge_index)
        return h


class GaussianHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            min_scale: float = 0.1,
            max_scale: float = 5.0,
            max_loc: float = 10.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.loc_proj = nn.Linear(hidden_dim, horizon)
        self.scale_proj = nn.Linear(hidden_dim, horizon)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_loc = float(max_loc)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loc = F.softplus(self.loc_proj(h)).clamp(0.0, self.max_loc)
        scale = (F.softplus(self.scale_proj(h)) + self.eps).clamp(self.min_scale, self.max_scale)
        return loc, scale


class NBHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_n: float = 50.0,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
            eps: float = EPS,
    ):
        super().__init__()
        self.n_proj = nn.Linear(hidden_dim, horizon)
        self.p_proj = nn.Linear(hidden_dim, horizon)
        self.max_n = float(max_n)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = (F.softplus(self.n_proj(h)) + self.eps).clamp(self.eps, self.max_n)
        p = torch.sigmoid(self.p_proj(h)).clamp(self.min_prob, self.max_prob)
        return n, p


class ZINBHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_n: float = 50.0,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
            eps: float = EPS,
    ):
        super().__init__()
        self.n_proj = nn.Linear(hidden_dim, horizon)
        self.p_proj = nn.Linear(hidden_dim, horizon)
        self.pi_proj = nn.Linear(hidden_dim, horizon)
        self.max_n = float(max_n)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = (F.softplus(self.n_proj(h)) + self.eps).clamp(self.eps, self.max_n)
        p = torch.sigmoid(self.p_proj(h)).clamp(self.min_prob, self.max_prob)
        pi = torch.sigmoid(self.pi_proj(h)).clamp(self.min_prob, self.max_prob)
        return n, p, pi


class HurdleGammaHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, horizon)
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.alpha_proj = nn.Linear(hidden_dim, horizon)
        self.max_mu = float(max_mu)
        self.max_alpha = float(max_alpha)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_logits = self.q_proj(h)
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.max_mu)
        alpha = (F.softplus(self.alpha_proj(h)) + self.eps).clamp(self.eps, self.max_alpha)
        return q_logits, mu, alpha


class DAERHurdleGammaHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_target_proj = nn.Linear(hidden_dim, horizon)
        self.q_event_proj = nn.Linear(hidden_dim, horizon)
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.alpha_proj = nn.Linear(hidden_dim, horizon)
        self.max_mu = float(max_mu)
        self.max_alpha = float(max_alpha)
        self.eps = float(eps)

    def forward(
            self,
            h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_target_logits = self.q_target_proj(h)
        q_event_logits = self.q_event_proj(h)
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.max_mu)
        alpha = (F.softplus(self.alpha_proj(h)) + self.eps).clamp(self.eps, self.max_alpha)
        return q_target_logits, q_event_logits, mu, alpha


class GammaHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.alpha_proj = nn.Linear(hidden_dim, horizon)
        self.max_mu = float(max_mu)
        self.max_alpha = float(max_alpha)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.max_mu)
        alpha = (F.softplus(self.alpha_proj(h)) + self.eps).clamp(self.eps, self.max_alpha)
        return mu, alpha


class HurdleLogNormalHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_sigma: float = 2.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, horizon)
        self.mu_log_proj = nn.Linear(hidden_dim, horizon)
        self.sigma_log_proj = nn.Linear(hidden_dim, horizon)
        self.max_sigma = float(max_sigma)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_logits = self.q_proj(h)
        mu_log = self.mu_log_proj(h)
        sigma_log = (F.softplus(self.sigma_log_proj(h)) + self.eps).clamp(self.eps, self.max_sigma)
        return q_logits, mu_log, sigma_log


class DAERHurdleLogNormalHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_sigma: float = 2.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_target_proj = nn.Linear(hidden_dim, horizon)
        self.q_event_proj = nn.Linear(hidden_dim, horizon)
        self.mu_log_proj = nn.Linear(hidden_dim, horizon)
        self.sigma_log_proj = nn.Linear(hidden_dim, horizon)
        self.max_sigma = float(max_sigma)
        self.eps = float(eps)

    def forward(
            self,
            h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_target_logits = self.q_target_proj(h)
        q_event_logits = self.q_event_proj(h)
        mu_log = self.mu_log_proj(h)
        sigma_log = (F.softplus(self.sigma_log_proj(h)) + self.eps).clamp(self.eps, self.max_sigma)
        return q_target_logits, q_event_logits, mu_log, sigma_log


class LogNormalHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            max_sigma: float = 2.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.mu_log_proj = nn.Linear(hidden_dim, horizon)
        self.sigma_log_proj = nn.Linear(hidden_dim, horizon)
        self.max_sigma = float(max_sigma)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_log = self.mu_log_proj(h)
        sigma_log = (F.softplus(self.sigma_log_proj(h)) + self.eps).clamp(self.eps, self.max_sigma)
        return mu_log, sigma_log


class HurdleTweedieHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
            eps: float = EPS,
    ):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, horizon)
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.phi_proj = nn.Linear(hidden_dim, horizon)
        self.rho_proj = nn.Linear(hidden_dim, horizon)
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.rho_eps = float(rho_eps)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_logits = self.q_proj(h)
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.mu_max)
        phi = (F.softplus(self.phi_proj(h)) + self.eps).clamp(self.phi_min, self.phi_max)
        rho = torch.sigmoid(self.rho_proj(h))
        rho = rho * (1.0 - 2.0 * self.rho_eps) + 1.0 + self.rho_eps
        rho = rho.clamp(1.0 + self.rho_eps, 2.0 - self.rho_eps)
        return q_logits, mu, phi, rho


class SharedHurdleEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            kernel_size: int,
            temp_layers: int,
            spatial_layers: int,
            diffusion_order: int,
            dropout: float,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temporal_backbone = str(temporal_backbone)
        self.spatial_backbone = str(spatial_backbone)
        if self.temporal_backbone == "tcn":
            self.input_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
            self.temporal_blocks = nn.ModuleList(
                [CausalTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout) for _ in range(temp_layers)]
            )
            self.temporal_gru = None
        elif self.temporal_backbone == "gru":
            self.input_proj = None
            self.temporal_blocks = None
            self.temporal_gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=max(1, int(temp_layers)),
                batch_first=True,
                dropout=float(dropout) if int(temp_layers) > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unsupported temporal_backbone: {temporal_backbone}")

        if self.spatial_backbone == "diffusion":
            self.spatial_blocks = nn.ModuleList(
                [DiffusionConv(hidden_dim, hidden_dim, diffusion_order, num_supports=2) for _ in range(spatial_layers)]
            )
            self.spatial_gat_blocks = None
        elif self.spatial_backbone == "gat":
            self.spatial_blocks = None
            self.spatial_gat_blocks = nn.ModuleList(
                [GATSpatialBlock(hidden_dim, hidden_dim, gat_heads, dropout) for _ in range(spatial_layers)]
            )
        else:
            raise ValueError(f"Unsupported spatial_backbone: {spatial_backbone}")
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def encode_sequence(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        """Return the causal spatiotemporal representation at every history step."""
        batch_size, num_nodes, history_len, feat_dim = x.shape
        if self.temporal_backbone == "tcn":
            h = x.permute(0, 3, 1, 2).contiguous()
            h = self.input_proj(h)
            for block in self.temporal_blocks:
                h = block(h)
            h = h.permute(0, 3, 2, 1).contiguous()
        else:
            x_seq = x.reshape(batch_size * num_nodes, history_len, feat_dim)
            h_seq, _ = self.temporal_gru(x_seq)
            h_seq = self.dropout(h_seq)
            h = h_seq.view(batch_size, num_nodes, history_len, self.hidden_dim).permute(0, 2, 1, 3).contiguous()

        if self.spatial_backbone == "diffusion":
            supports = [A_q, A_h]
            for gconv in self.spatial_blocks:
                h_bt = h.reshape(batch_size * history_len, h.size(2), self.hidden_dim)
                h_bt = self.dropout(F.relu(gconv(h_bt, supports)))
                h = h_bt.view(batch_size, history_len, h.size(2), self.hidden_dim)
        else:
            edge_index = _edge_index_from_supports(A_q, A_h)
            for gat in self.spatial_gat_blocks:
                h_bt = h.reshape(batch_size * history_len, h.size(2), self.hidden_dim)
                h_bt = gat(h_bt, edge_index)
                h_bt = self.dropout(h_bt)
                h = h_bt.view(batch_size, history_len, h.size(2), self.hidden_dim)

        return h

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        h = self.encode_sequence(x, A_q, A_h)
        h_last = h[:, -1, :, :]
        h_mean = h.mean(dim=1)
        return self.norm(0.5 * (h_last + h_mean))


class TweedieHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
            eps: float = EPS,
    ):
        super().__init__()
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.phi_proj = nn.Linear(hidden_dim, horizon)
        self.rho_proj = nn.Linear(hidden_dim, horizon)
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.rho_eps = float(rho_eps)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.mu_max)
        phi = (F.softplus(self.phi_proj(h)) + self.eps).clamp(self.phi_min, self.phi_max)
        rho = torch.sigmoid(self.rho_proj(h))
        rho = rho * (1.0 - 2.0 * self.rho_eps) + 1.0 + self.rho_eps
        rho = rho.clamp(1.0 + self.rho_eps, 2.0 - self.rho_eps)
        return mu, phi, rho


class ZITDHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            horizon: int,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
            eps: float = EPS,
    ):
        super().__init__()
        self.pi_proj = nn.Linear(hidden_dim, horizon)
        self.mu_proj = nn.Linear(hidden_dim, horizon)
        self.phi_proj = nn.Linear(hidden_dim, horizon)
        self.rho_proj = nn.Linear(hidden_dim, horizon)
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.rho_eps = float(rho_eps)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.eps = float(eps)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = torch.sigmoid(self.pi_proj(h)).clamp(self.min_prob, self.max_prob)
        mu = (F.softplus(self.mu_proj(h)) + self.eps).clamp(self.eps, self.mu_max)
        phi = (F.softplus(self.phi_proj(h)) + self.eps).clamp(self.phi_min, self.phi_max)
        rho = torch.sigmoid(self.rho_proj(h))
        rho = rho * (1.0 - 2.0 * self.rho_eps) + 1.0 + self.rho_eps
        rho = rho.clamp(1.0 + self.rho_eps, 2.0 - self.rho_eps)
        return pi, mu, phi, rho


class STGaussianGNNBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            history_len: int,
            kernel_size: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            min_scale: float = 0.1,
            max_scale: float = 5.0,
            max_loc: float = 10.0,
    ):
        super().__init__()
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.max_loc = float(max_loc)
        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, kernel_size, dropout, temporal_backbone=temporal_backbone
        )
        self.spatial_encoder = SpatialHistoryEncoder(
            input_dim,
            history_len,
            hidden_dim,
            diffusion_order,
            dropout,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.temporal_head = GaussianHead(
            hidden_dim, horizon, min_scale=min_scale, max_scale=max_scale, max_loc=max_loc
        )
        self.spatial_head = GaussianHead(
            hidden_dim, horizon, min_scale=min_scale, max_scale=max_scale, max_loc=max_loc
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_hidden = self.temporal_encoder(x)
        spatial_hidden = self.spatial_encoder(x, A_q, A_h)

        loc_t, scale_t = self.temporal_head(temporal_hidden)
        loc_s, scale_s = self.spatial_head(spatial_hidden)

        loc = 0.5 * (loc_t + loc_s)
        scale = 0.5 * (scale_t + scale_s)
        loc = loc.clamp(0.0, self.max_loc)
        scale = scale.clamp(self.min_scale, self.max_scale)
        return loc, scale


class STNBGNNBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            history_len: int,
            kernel_size: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_n: float = 50.0,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
    ):
        super().__init__()
        self.max_n = float(max_n)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, kernel_size, dropout, temporal_backbone=temporal_backbone
        )
        self.spatial_encoder = SpatialHistoryEncoder(
            input_dim,
            history_len,
            hidden_dim,
            diffusion_order,
            dropout,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.temporal_head = NBHead(
            hidden_dim, horizon, max_n=max_n, min_prob=min_prob, max_prob=max_prob
        )
        self.spatial_head = NBHead(
            hidden_dim, horizon, max_n=max_n, min_prob=min_prob, max_prob=max_prob
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temporal_hidden = self.temporal_encoder(x)
        spatial_hidden = self.spatial_encoder(x, A_q, A_h)

        n_t, p_t = self.temporal_head(temporal_hidden)
        n_s, p_s = self.spatial_head(spatial_hidden)

        n = 0.5 * (n_t + n_s)
        p = 0.5 * (p_t + p_s)
        n = n.clamp(EPS, self.max_n)
        p = p.clamp(self.min_prob, self.max_prob)
        return n, p


class STZINBBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            history_len: int,
            kernel_size: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_n: float = 50.0,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
    ):
        super().__init__()
        self.max_n = float(max_n)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, kernel_size, dropout, temporal_backbone=temporal_backbone
        )
        self.spatial_encoder = SpatialHistoryEncoder(
            input_dim,
            history_len,
            hidden_dim,
            diffusion_order,
            dropout,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.temporal_head = ZINBHead(
            hidden_dim, horizon, max_n=max_n, min_prob=min_prob, max_prob=max_prob
        )
        self.spatial_head = ZINBHead(
            hidden_dim, horizon, max_n=max_n, min_prob=min_prob, max_prob=max_prob
        )

    def forward(
            self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_hidden = self.temporal_encoder(x)
        spatial_hidden = self.spatial_encoder(x, A_q, A_h)

        n_t, p_t, pi_t = self.temporal_head(temporal_hidden)
        n_s, p_s, pi_s = self.spatial_head(spatial_hidden)

        n = 0.5 * (n_t + n_s)
        p = 0.5 * (p_t + p_s)
        pi = 0.5 * (pi_t + pi_s)
        n = n.clamp(EPS, self.max_n)
        p = p.clamp(self.min_prob, self.max_prob)
        pi = pi.clamp(self.min_prob, self.max_prob)
        return n, p, pi


class STTDBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            history_len: int,
            kernel_size: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
    ):
        super().__init__()
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.rho_eps = float(rho_eps)
        self.temporal_encoder = TemporalEncoder(
            input_dim, hidden_dim, kernel_size, dropout, temporal_backbone=temporal_backbone
        )
        self.spatial_encoder = SpatialHistoryEncoder(
            input_dim,
            history_len,
            hidden_dim,
            diffusion_order,
            dropout,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.temporal_head = TweedieHead(
            hidden_dim, horizon, mu_max=mu_max, phi_min=phi_min, phi_max=phi_max, rho_eps=rho_eps
        )
        self.spatial_head = TweedieHead(
            hidden_dim, horizon, mu_max=mu_max, phi_min=phi_min, phi_max=phi_max, rho_eps=rho_eps
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_hidden = self.temporal_encoder(x)
        spatial_hidden = self.spatial_encoder(x, A_q, A_h)

        mu_t, phi_t, rho_t = self.temporal_head(temporal_hidden)
        mu_s, phi_s, rho_s = self.spatial_head(spatial_hidden)

        mu = 0.5 * (mu_t + mu_s)
        phi = 0.5 * (phi_t + phi_s)
        rho = 0.5 * (rho_t + rho_s)
        mu = mu.clamp(EPS, self.mu_max)
        phi = phi.clamp(self.phi_min, self.phi_max)
        rho = rho.clamp(1.0 + self.rho_eps, 2.0 - self.rho_eps)
        return mu, phi, rho


class STZITDBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            history_len: int,
            kernel_size: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
            min_prob: float = 1e-4,
            max_prob: float = 0.9999,
    ):
        super().__init__()
        self.mu_max = float(mu_max)
        self.phi_min = float(phi_min)
        self.phi_max = float(phi_max)
        self.rho_eps = float(rho_eps)
        self.min_prob = float(min_prob)
        self.max_prob = float(max_prob)
        self.temporal_encoder = TemporalEncoder(input_dim, hidden_dim, kernel_size, dropout,
                                                temporal_backbone=temporal_backbone)
        self.spatial_encoder = SpatialHistoryEncoder(input_dim, history_len, hidden_dim, diffusion_order, dropout,
                                                     spatial_backbone=spatial_backbone, gat_heads=gat_heads)
        self.temporal_head = ZITDHead(
            hidden_dim,
            horizon,
            mu_max=mu_max,
            phi_min=phi_min,
            phi_max=phi_max,
            rho_eps=rho_eps,
            min_prob=min_prob,
            max_prob=max_prob,
        )
        self.spatial_head = ZITDHead(
            hidden_dim,
            horizon,
            mu_max=mu_max,
            phi_min=phi_min,
            phi_max=phi_max,
            rho_eps=rho_eps,
            min_prob=min_prob,
            max_prob=max_prob,
        )

    def forward(
            self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        temporal_hidden = self.temporal_encoder(x)
        spatial_hidden = self.spatial_encoder(x, A_q, A_h)

        pi_t, mu_t, phi_t, rho_t = self.temporal_head(temporal_hidden)
        pi_s, mu_s, phi_s, rho_s = self.spatial_head(spatial_hidden)

        pi = 0.5 * (pi_t + pi_s)
        mu = 0.5 * (mu_t + mu_s)
        phi = 0.5 * (phi_t + phi_s)
        rho = 0.5 * (rho_t + rho_s)
        pi = pi.clamp(self.min_prob, self.max_prob)
        mu = mu.clamp(EPS, self.mu_max)
        phi = phi.clamp(self.phi_min, self.phi_max)
        rho = rho.clamp(1.0 + self.rho_eps, 2.0 - self.rho_eps)
        return pi, mu, phi, rho


class STEventOccurrenceBaseline(nn.Module):
    """Direct event-only baseline using the same encoder as the Hurdle models."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.event_head = nn.Linear(hidden_dim, horizon)

    def forward(
            self,
            x: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> torch.Tensor:
        return self.event_head(self.encoder(x, A_q, A_h))


class STHurdleGammaBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = HurdleGammaHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_mu=max_mu,
            max_alpha=max_alpha,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STDAERHurdleGammaBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = DAERHurdleGammaHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_mu=max_mu,
            max_alpha=max_alpha,
            eps=eps,
        )

    def forward(
            self,
            x: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STGammaBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_mu: float = 20.0,
            max_alpha: float = 50.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = GammaHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_mu=max_mu,
            max_alpha=max_alpha,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STHurdleLogNormalBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_sigma: float = 2.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = HurdleLogNormalHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_sigma=max_sigma,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STDAERHurdleLogNormalBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_sigma: float = 2.0,
            eps: float = EPS,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = DAERHurdleLogNormalHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_sigma=max_sigma,
            eps=eps,
        )

    def forward(
            self,
            x: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STLogNormalBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            max_sigma: float = 2.0,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = LogNormalHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            max_sigma=max_sigma,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)


class STHurdleTweedieBaseline(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 2,
            spatial_layers: int = 2,
            diffusion_order: int = 2,
            dropout: float = 0.2,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            mu_max: float = 20.0,
            phi_min: float = 1e-3,
            phi_max: float = 10.0,
            rho_eps: float = 1e-3,
    ):
        super().__init__()
        self.encoder = SharedHurdleEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            temp_layers=temp_layers,
            spatial_layers=spatial_layers,
            diffusion_order=diffusion_order,
            dropout=dropout,
            temporal_backbone=temporal_backbone,
            spatial_backbone=spatial_backbone,
            gat_heads=gat_heads,
        )
        self.head = HurdleTweedieHead(
            hidden_dim=hidden_dim,
            horizon=horizon,
            mu_max=mu_max,
            phi_min=phi_min,
            phi_max=phi_max,
            rho_eps=rho_eps,
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x, A_q, A_h)
        return self.head(h)
