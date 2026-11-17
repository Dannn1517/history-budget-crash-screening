"""Reliability-Gated Historical Shrinkage (RGHS).

Learned monotone empirical-Bayes shrinkage for road-level crash retrieval under
limited history: the per-road empirical frequency score is shrunk toward a
static-feature population prior with a gate driven by observable archive depth,
plus a constrained temporal residual. At sufficient archive depth the gate
saturates to 1 and the residual is penalized to 0, so the model's ranking
asymptotically falls back to the zero-training frequency ranker.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.probabilistic_baselines import SharedHurdleEncoder


@dataclass
class RGHSOutput:
    logits: torch.Tensor            # [B, N, H] occurrence logits
    gate: torch.Tensor              # [B, N, 1] reliability weight w in (0,1)
    population_score: torch.Tensor  # [B, N, 1] static prior branch s_i
    residual: torch.Tensor          # [B, N, H] masked temporal residual
    residual_raw: torch.Tensor      # [B, N, H] unmasked residual head output
    residual_mask: torch.Tensor     # [B, N, 1] structural fallback mask m(n)
    fused_score: torch.Tensor       # [B, N, 1] w*freq + (1-w)*s


class ReliabilityGatedHistoricalShrinkage(nn.Module):
    def __init__(
            self,
            input_dim: int,
            static_dim: int,
            hidden_dim: int = 64,
            horizon: int = 3,
            kernel_size: int = 3,
            temp_layers: int = 3,
            spatial_layers: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            population_hidden: int = 64,
            log_n_ref: float = 5.7,
            init_gate_slope: float = 1.0,
            init_gate_bias: float = 0.0,
            init_fused_scale: float = 1.0,
            init_mask_slope: float = 1.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.log_n_ref = float(log_n_ref)
        self.residual_encoder = SharedHurdleEncoder(
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
        self.residual_head = nn.Linear(hidden_dim, horizon)
        self.population = nn.Sequential(
            nn.Linear(static_dim, population_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(population_hidden, 1),
        )
        # gate slope kept positive via softplus reparameterization
        self.gate_slope_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_gate_slope)))
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(init_gate_bias)))
        self.fused_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_fused_scale)))
        )
        # structural fallback mask slope: m(n) = sigmoid(c * (log_n_ref - log n))
        self.mask_slope_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_mask_slope)))
        )

    def gate_slope(self) -> torch.Tensor:
        return F.softplus(self.gate_slope_raw)

    def fused_scale(self) -> torch.Tensor:
        return F.softplus(self.fused_scale_raw)

    def mask_slope(self) -> torch.Tensor:
        return F.softplus(self.mask_slope_raw)

    def forward(
            self,
            x: torch.Tensor,
            a_q: torch.Tensor,
            a_h: torch.Tensor,
            static: torch.Tensor,
            freq_origin: torch.Tensor,
            log_n: torch.Tensor,
    ) -> RGHSOutput:
        """x [B,N,T,C]; static [B,N,S]; freq_origin [B,N]; log_n [B,N]."""
        encoded = self.residual_encoder(x, a_q, a_h)
        residual_raw = self.residual_head(encoded)
        population_score = self.population(static)
        centered = (log_n.unsqueeze(-1) - self.log_n_ref)
        gate = torch.sigmoid(self.gate_slope() * centered + self.gate_bias)
        # structural fallback: mask -> 0 as archive depth grows, so the residual
        # branch vanishes at deep archives without needing a global penalty
        mask = torch.sigmoid(-self.mask_slope() * centered)
        residual = residual_raw * mask
        fused = gate * freq_origin.unsqueeze(-1) + (1.0 - gate) * population_score
        logits = self.fused_scale() * fused + residual
        return RGHSOutput(
            logits=logits,
            gate=gate,
            population_score=population_score,
            residual=residual,
            residual_raw=residual_raw,
            residual_mask=mask,
            fused_score=fused,
        )


class ReliabilityGatedSafeFallback(nn.Module):
    """Generalist expert with a monotone, asymptotically safe history fallback.

    Unlike :class:`ReliabilityGatedHistoricalShrinkage`, this variant preserves
    the full event-occurrence Generalist at low archive depth and only blends in
    the frequency expert as reliability grows.  At gate=1 its road ranking is
    exactly the forecast-origin frequency ranking.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            horizon: int = 3,
            kernel_size: int = 3,
            temp_layers: int = 3,
            spatial_layers: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            log_n_ref: float = 5.7,
            init_gate_slope: float = 6.0,
            init_gate_bias: float = 0.0,
            init_fused_scale: float = 1.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.log_n_ref = float(log_n_ref)
        self.generalist_encoder = SharedHurdleEncoder(
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
        self.generalist_head = nn.Linear(hidden_dim, horizon)
        self.gate_slope_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_gate_slope)))
        )
        self.gate_bias = nn.Parameter(torch.tensor(float(init_gate_bias)))
        self.fused_scale_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(init_fused_scale)))
        )
        self.history_bias = nn.Parameter(torch.tensor(0.0))

    def gate_slope(self) -> torch.Tensor:
        return F.softplus(self.gate_slope_raw)

    def fused_scale(self) -> torch.Tensor:
        return F.softplus(self.fused_scale_raw)

    def mask_slope(self) -> torch.Tensor:
        # Exposed for common diagnostics; 1-gate is the actual neural mask.
        return self.gate_slope()

    def forward(
            self,
            x: torch.Tensor,
            a_q: torch.Tensor,
            a_h: torch.Tensor,
            static: torch.Tensor,
            freq_origin: torch.Tensor,
            log_n: torch.Tensor,
    ) -> RGHSOutput:
        del static  # kept in the common interface; static features are in x
        encoded = self.generalist_encoder(x, a_q, a_h)
        generalist_logits = self.generalist_head(encoded)
        centered = log_n.unsqueeze(-1) - self.log_n_ref
        gate = torch.sigmoid(self.gate_slope() * centered + self.gate_bias)
        history_logits = (
            self.fused_scale() * freq_origin.unsqueeze(-1) + self.history_bias
        ).expand_as(generalist_logits)
        neural_mask = 1.0 - gate
        neural_contribution = neural_mask * generalist_logits
        logits = neural_contribution + gate * history_logits
        return RGHSOutput(
            logits=logits,
            gate=gate,
            population_score=generalist_logits,
            residual=neural_contribution,
            residual_raw=generalist_logits,
            residual_mask=neural_mask,
            fused_score=history_logits,
        )


def _inverse_softplus(value: float) -> float:
    import math

    return math.log(math.expm1(value))


def rghs_loss(
        output: RGHSOutput,
        event_target: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        log_n: torch.Tensor | None = None,
        n_ref_days: float = 365.0,
        fallback_weight: float = 0.1,
        gate_fallback_weight: float = 0.1,
        sample_weight: torch.Tensor = None,
) -> dict[str, torch.Tensor]:
    """Weighted BCE plus asymptotic-fallback penalties at deep archives.

    fallback:      E[ Delta^2 * 1{n >= n_ref} ]
    gate_fallback: E[ (1-w)^2 * 1{n >= n_ref} ]
    sample_weight: optional per-node [B, N] mask excluding held-out nodes from the loss.
    """
    import math

    labels = (event_target > 0).to(dtype=output.logits.dtype)
    if sample_weight is not None:
        pointwise = F.binary_cross_entropy_with_logits(
            output.logits, labels, pos_weight=pos_weight, reduction="none"
        )
        denom = (sample_weight.sum() * output.logits.shape[-1]).clamp(min=1.0)
        bce = (pointwise * sample_weight.unsqueeze(-1)).sum() / denom
    else:
        bce = F.binary_cross_entropy_with_logits(output.logits, labels, pos_weight=pos_weight)
    loss = bce
    residual_pen = torch.zeros((), dtype=output.logits.dtype, device=output.logits.device)
    gate_pen = torch.zeros((), dtype=output.logits.dtype, device=output.logits.device)
    if log_n is not None and (fallback_weight > 0 or gate_fallback_weight > 0):
        deep = (log_n >= math.log(float(n_ref_days))).to(dtype=output.logits.dtype)
        deep_mask = deep.unsqueeze(-1)
        deep_count = deep.sum().clamp(min=1.0)
        if fallback_weight > 0:
            horizon = output.residual.shape[-1]
            residual_pen = (output.residual.square() * deep_mask).sum() / (deep_count * horizon)
            loss = loss + float(fallback_weight) * residual_pen
        if gate_fallback_weight > 0:
            gate_pen = ((1.0 - output.gate).square() * deep_mask).sum() / deep_count
            loss = loss + float(gate_fallback_weight) * gate_pen
    return {
        "loss": loss,
        "bce": bce,
        "residual_penalty": residual_pen,
        "gate_penalty": gate_pen,
    }
