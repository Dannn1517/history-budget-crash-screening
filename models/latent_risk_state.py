"""Graph-conditioned latent risk-state model for sparse crash occurrence.

Unlike a hurdle occurrence head, the binary state in this module is not the
observed crash indicator.  Historical observations update a persistent latent
state through a differentiable Bayesian filter; future states are propagated
without observing their crash outcomes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.probabilistic_baselines import SharedHurdleEncoder, log_lognormal_pdf


EPS = 1e-6
TRANSITION_MODES = {"graph_dynamic", "input_dynamic", "static"}
MIN_POSITIVE_GRAPH_SCALE = 1e-6

MIN_SEVERITY_SIGMA = 1e-3
MAX_SEVERITY_MEAN = 50.0
INIT_SEVERITY_MEAN = 2.0
INIT_SEVERITY_GAP = 0.2
INIT_SEVERITY_SIGMA = 0.5


def _logit(probability: float) -> float:
    """Map a probability in (0, 1) to its log-odds."""
    return math.log(probability / (1.0 - probability))


def _inverse_softplus(value: float) -> float:
    """Inverse of softplus for a strictly positive value."""
    return math.log(math.expm1(value))


def prior_corrected_logits(
        logits: torch.Tensor,
        pos_weight: torch.Tensor | float | None,
) -> torch.Tensor:
    """Shift logits by log(pos_weight) so weighted BCE optimizes generative odds.

    ``binary_cross_entropy_with_logits(logits, target, pos_weight=w)`` is
    minimized at ``logits = log-odds + log(w)``.  Passing ``logits + log(w)``
    therefore lets the model's own logits represent the unweighted generative
    log odds while still enjoying positive weighting during optimization.
    ``None`` and ``1.0`` are no-ops.
    """
    if pos_weight is None:
        return logits
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    if not bool(torch.all(torch.isfinite(weight))) or not bool(torch.all(weight > 0)):
        raise ValueError("pos_weight must be positive and finite.")
    if bool(torch.all(weight == 1.0)):
        return logits
    return logits + torch.log(weight)


@dataclass
class LatentRiskStateOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    history_probabilities: torch.Tensor
    history_logits: torch.Tensor
    history_state_posterior: torch.Tensor
    forecast_state_prior: torch.Tensor
    transition_entry: torch.Tensor
    transition_persistence: torch.Tensor
    emission_low: torch.Tensor
    emission_high: torch.Tensor
    severity_mean_low: torch.Tensor
    severity_mean_high: torch.Tensor
    severity_sigma: torch.Tensor
    forecast_emission_low: torch.Tensor
    forecast_emission_high: torch.Tensor
    expected_risk: torch.Tensor


def apply_graph_support(support: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Apply one graph support to batched scalar node values [B, N]."""
    if values.ndim != 2:
        raise ValueError(f"values must have shape [B, N], got {tuple(values.shape)}")
    if support.is_sparse:
        return torch.sparse.mm(support, values.transpose(0, 1)).transpose(0, 1)
    return torch.matmul(values, support.transpose(0, 1))


def bayesian_state_update(
        prior_high: torch.Tensor,
        emission_low: torch.Tensor,
        emission_high: torch.Tensor,
        observation: torch.Tensor,
        eps: float = EPS,
) -> torch.Tensor:
    """Posterior P(S=high | binary observation) for a two-state emission."""
    prior_high = prior_high.clamp(eps, 1.0 - eps)
    emission_low = emission_low.clamp(eps, 1.0 - eps)
    emission_high = emission_high.clamp(eps, 1.0 - eps)
    observation = observation.to(dtype=prior_high.dtype)

    log_like_high = observation * torch.log(emission_high) + (1.0 - observation) * torch.log1p(
        -emission_high
    )
    log_like_low = observation * torch.log(emission_low) + (1.0 - observation) * torch.log1p(
        -emission_low
    )
    log_joint = torch.stack(
        [torch.log1p(-prior_high) + log_like_low, torch.log(prior_high) + log_like_high],
        dim=-1,
    )
    return torch.softmax(log_joint, dim=-1)[..., 1]


class GraphConditionedLatentRiskState(nn.Module):
    """Two-state neural filter with graph-conditioned, time-varying transitions."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            horizon: int,
            kernel_size: int = 3,
            temp_layers: int = 3,
            spatial_layers: int = 3,
            diffusion_order: int = 2,
            dropout: float = 0.1,
            temporal_backbone: str = "tcn",
            spatial_backbone: str = "diffusion",
            gat_heads: int = 4,
            transition_mode: str = "graph_dynamic",
            initial_state_probability: float = 0.05,
            initial_entry_probability: float = 0.005,
            initial_persistence_probability: float = 0.90,
            initial_low_emission_probability: float = 0.002,
            initial_high_emission_probability: float = 0.02,
            initial_graph_scale: float = 0.05,
            severity_sigma_max: float = 3.0,
    ):
        super().__init__()
        if transition_mode not in TRANSITION_MODES:
            raise ValueError(
                f"Unsupported transition_mode={transition_mode!r}; expected one of {sorted(TRANSITION_MODES)}"
            )
        initial_state_probability = float(initial_state_probability)
        initial_entry_probability = float(initial_entry_probability)
        initial_persistence_probability = float(initial_persistence_probability)
        initial_low_emission_probability = float(initial_low_emission_probability)
        initial_high_emission_probability = float(initial_high_emission_probability)
        initial_graph_scale = float(initial_graph_scale)
        severity_sigma_max = float(severity_sigma_max)

        if not (0.0 < initial_state_probability < 1.0):
            raise ValueError(
                "initial_state_probability must lie strictly between 0 and 1; "
                f"got {initial_state_probability}"
            )
        if not (0.0 < initial_entry_probability < initial_persistence_probability < 1.0):
            raise ValueError(
                "initial entry/persistence probabilities must satisfy "
                "0 < entry < persistence < 1; "
                f"got entry={initial_entry_probability}, persistence={initial_persistence_probability}"
            )
        if not (0.0 < initial_low_emission_probability < initial_high_emission_probability < 1.0):
            raise ValueError(
                "initial emission probabilities must satisfy 0 < low < high < 1; "
                f"got low={initial_low_emission_probability}, high={initial_high_emission_probability}"
            )
        if not (initial_graph_scale >= 0.0):
            raise ValueError(
                f"initial_graph_scale must be non-negative; got {initial_graph_scale}"
            )
        if not (severity_sigma_max > 0.0 and math.isfinite(severity_sigma_max)):
            raise ValueError(
                f"severity_sigma_max must be positive and finite; got {severity_sigma_max}"
            )

        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.transition_mode = str(transition_mode)
        self.severity_sigma_max = severity_sigma_max
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
        self.sequence_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.initial_state = nn.Linear(hidden_dim, 1)
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, hidden_dim))

        entry_logit = _logit(initial_entry_probability)
        persistence_fraction = (
            (initial_persistence_probability - initial_entry_probability)
            / (1.0 - initial_entry_probability)
        )
        persistence_delta_logit = _logit(persistence_fraction)

        if transition_mode == "static":
            self.static_entry_logit = nn.Parameter(torch.tensor(entry_logit))
            self.static_persistence_delta_logit = nn.Parameter(
                torch.tensor(persistence_delta_logit)
            )
            self.transition_network = None
        else:
            self.register_parameter("static_entry_logit", None)
            self.register_parameter("static_persistence_delta_logit", None)
            self.transition_network = nn.Sequential(
                nn.Linear(hidden_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            final_transition = self.transition_network[-1]
            nn.init.zeros_(final_transition.weight)
            with torch.no_grad():
                final_transition.bias.zero_()
                final_transition.bias[0] = entry_logit
                final_transition.bias[1] = persistence_delta_logit

        if transition_mode == "graph_dynamic":
            graph_scale = initial_graph_scale
            if graph_scale <= 0.0:
                graph_scale = MIN_POSITIVE_GRAPH_SCALE
            graph_scale_raw = _inverse_softplus(graph_scale)
            self.graph_entry_scale_raw = nn.Parameter(torch.tensor(graph_scale_raw))
            self.graph_persistence_scale_raw = nn.Parameter(torch.tensor(graph_scale_raw))
        else:
            self.register_parameter("graph_entry_scale_raw", None)
            self.register_parameter("graph_persistence_scale_raw", None)

        self.emission_low = nn.Linear(hidden_dim, 1)
        self.emission_gap = nn.Linear(hidden_dim, 1)
        self.severity_low_mean = nn.Linear(hidden_dim, 1)
        self.severity_mean_gap = nn.Linear(hidden_dim, 1)
        self.severity_log_sigma = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.initial_state.weight)
        nn.init.constant_(self.initial_state.bias, _logit(initial_state_probability))

        nn.init.zeros_(self.emission_low.weight)
        nn.init.constant_(self.emission_low.bias, _logit(initial_low_emission_probability))
        nn.init.zeros_(self.emission_gap.weight)
        emission_gap_logit = (
            _logit(initial_high_emission_probability) - _logit(initial_low_emission_probability)
        )
        nn.init.constant_(self.emission_gap.bias, _inverse_softplus(emission_gap_logit))

        nn.init.zeros_(self.severity_low_mean.weight)
        nn.init.constant_(self.severity_low_mean.bias, _inverse_softplus(INIT_SEVERITY_MEAN))
        nn.init.zeros_(self.severity_mean_gap.weight)
        nn.init.constant_(self.severity_mean_gap.bias, _inverse_softplus(INIT_SEVERITY_GAP))
        nn.init.zeros_(self.severity_log_sigma.weight)
        nn.init.constant_(self.severity_log_sigma.bias, _inverse_softplus(INIT_SEVERITY_SIGMA))

    def _neighbor_state(
            self,
            state: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (apply_graph_support(A_q, state) + apply_graph_support(A_h, state))

    def _transition(
            self,
            previous_high: torch.Tensor,
            context: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.transition_mode == "static":
            entry_logit = self.static_entry_logit.expand_as(previous_high)
            persistence_delta_logit = self.static_persistence_delta_logit.expand_as(previous_high)
        else:
            transition_input = torch.cat([context, previous_high.unsqueeze(-1)], dim=-1)
            raw = self.transition_network(transition_input)
            entry_logit = raw[..., 0]
            persistence_delta_logit = raw[..., 1]
            if self.transition_mode == "graph_dynamic":
                neighbor_high = self._neighbor_state(previous_high, A_q, A_h)
                entry_logit = entry_logit + F.softplus(self.graph_entry_scale_raw) * neighbor_high
                persistence_delta_logit = (
                    persistence_delta_logit
                    + F.softplus(self.graph_persistence_scale_raw) * neighbor_high
                )

        p_low_to_high = torch.sigmoid(entry_logit)
        persistence_fraction = torch.sigmoid(persistence_delta_logit)
        p_high_to_high = p_low_to_high + (1.0 - p_low_to_high) * persistence_fraction
        next_high = (1.0 - previous_high) * p_low_to_high + previous_high * p_high_to_high
        return next_high.clamp(EPS, 1.0 - EPS), p_low_to_high, p_high_to_high

    def _emissions(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low_logit = self.emission_low(context).squeeze(-1)
        high_logit = low_logit + F.softplus(self.emission_gap(context).squeeze(-1))
        return torch.sigmoid(low_logit), torch.sigmoid(high_logit)

    def _severity_parameters(
            self,
            context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low_mean = F.softplus(self.severity_low_mean(context).squeeze(-1)).clamp(
            EPS, MAX_SEVERITY_MEAN
        )
        gap = F.softplus(self.severity_mean_gap(context).squeeze(-1)).clamp(
            0.0, MAX_SEVERITY_MEAN
        )
        high_mean = (low_mean + gap).clamp(max=MAX_SEVERITY_MEAN)
        sigma = F.softplus(self.severity_log_sigma(context).squeeze(-1)).clamp(
            MIN_SEVERITY_SIGMA, self.severity_sigma_max
        )
        return low_mean, high_mean, sigma

    @staticmethod
    def _mixture_probability(
            state_high: torch.Tensor,
            emission_low: torch.Tensor,
            emission_high: torch.Tensor,
    ) -> torch.Tensor:
        probability = (1.0 - state_high) * emission_low + state_high * emission_high
        return probability.clamp(EPS, 1.0 - EPS)

    def forward(
            self,
            x: torch.Tensor,
            A_q: torch.Tensor,
            A_h: torch.Tensor,
    ) -> LatentRiskStateOutput:
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, N, T, F], got {tuple(x.shape)}")
        history_observation = (x[..., 0] > 0).to(dtype=x.dtype).permute(0, 2, 1)

        encoder_input = x.clone()
        shifted_event = torch.zeros_like(x[..., 0])
        shifted_event[..., 1:] = x[..., :-1, 0]
        encoder_input[..., 0] = shifted_event

        history_context = self.sequence_norm(self.encoder.encode_sequence(encoder_input, A_q, A_h))
        history_len = history_context.size(1)

        state_high = torch.sigmoid(self.initial_state(history_context[:, 0]).squeeze(-1))
        history_probabilities = []
        history_posteriors = []
        transition_entries = []
        transition_persistences = []
        emission_lows = []
        emission_highs = []

        for step in range(history_len):
            context = history_context[:, step]
            if step > 0:
                state_high, p01, p11 = self._transition(state_high, context, A_q, A_h)
                transition_entries.append(p01)
                transition_persistences.append(p11)
            low, high = self._emissions(context)
            probability = self._mixture_probability(state_high, low, high)
            state_high = bayesian_state_update(
                state_high,
                low,
                high,
                history_observation[:, step],
            )
            history_probabilities.append(probability)
            history_posteriors.append(state_high)
            emission_lows.append(low)
            emission_highs.append(high)

        forecast_context_base = self.context_norm(
            0.5 * (history_context[:, -1] + history_context.mean(dim=1))
        )
        forecast_probabilities = []
        forecast_states = []
        forecast_emission_lows = []
        forecast_emission_highs = []
        severity_mean_lows = []
        severity_mean_highs = []
        severity_sigmas = []
        expected_risks = []
        for step in range(self.horizon):
            context = forecast_context_base + self.horizon_embedding[step].view(1, 1, -1)
            state_high, p01, p11 = self._transition(state_high, context, A_q, A_h)
            low, high = self._emissions(context)
            probability = self._mixture_probability(state_high, low, high)
            low_mean, high_mean, sigma = self._severity_parameters(context)
            expected_risk = (1.0 - state_high) * low * low_mean + state_high * high * high_mean
            forecast_probabilities.append(probability)
            forecast_states.append(state_high)
            forecast_emission_lows.append(low)
            forecast_emission_highs.append(high)
            severity_mean_lows.append(low_mean)
            severity_mean_highs.append(high_mean)
            severity_sigmas.append(sigma)
            expected_risks.append(expected_risk)
            transition_entries.append(p01)
            transition_persistences.append(p11)
            emission_lows.append(low)
            emission_highs.append(high)

        probabilities = torch.stack(forecast_probabilities, dim=-1)
        logits = torch.logit(probabilities, eps=EPS)
        history_probabilities = torch.stack(history_probabilities, dim=-1)
        history_logits = torch.logit(history_probabilities, eps=EPS)
        return LatentRiskStateOutput(
            logits=logits,
            probabilities=probabilities,
            history_probabilities=history_probabilities,
            history_logits=history_logits,
            history_state_posterior=torch.stack(history_posteriors, dim=-1),
            forecast_state_prior=torch.stack(forecast_states, dim=-1),
            transition_entry=torch.stack(transition_entries, dim=-1),
            transition_persistence=torch.stack(transition_persistences, dim=-1),
            emission_low=torch.stack(emission_lows, dim=-1),
            emission_high=torch.stack(emission_highs, dim=-1),
            severity_mean_low=torch.stack(severity_mean_lows, dim=-1),
            severity_mean_high=torch.stack(severity_mean_highs, dim=-1),
            severity_sigma=torch.stack(severity_sigmas, dim=-1),
            forecast_emission_low=torch.stack(forecast_emission_lows, dim=-1),
            forecast_emission_high=torch.stack(forecast_emission_highs, dim=-1),
            expected_risk=torch.stack(expected_risks, dim=-1),
        )

    def graph_scales(self) -> dict[str, float | None]:
        if self.transition_mode != "graph_dynamic":
            return {"entry": None, "persistence": None}
        return {
            "entry": float(F.softplus(self.graph_entry_scale_raw).detach().cpu().item()),
            "persistence": float(F.softplus(self.graph_persistence_scale_raw).detach().cpu().item()),
        }


def _state_conditioned_severity_nll(
        severity_target: torch.Tensor,
        event_prob_low: torch.Tensor,
        event_prob_high: torch.Tensor,
        state_high: torch.Tensor,
        severity_mean_low: torch.Tensor,
        severity_mean_high: torch.Tensor,
        severity_sigma: torch.Tensor,
) -> torch.Tensor:
    """Mean LogNormal mixture NLL over positive severity observations.

    The conditional-on-event state weights are proportional to
    ``(1 - q) * event_prob_low`` and ``q * event_prob_high`` where ``q`` is the
    high-state prior.  Densities are mixed in log space for numerical stability.
    Returns a finite zero when there are no positive observations.
    """
    positive = severity_target > 0
    if not bool(positive.any()):
        return (
            severity_mean_low.sum() + severity_mean_high.sum() + severity_sigma.sum()
        ) * 0.0

    event_prob_low = event_prob_low.clamp(EPS, 1.0 - EPS)
    event_prob_high = event_prob_high.clamp(EPS, 1.0 - EPS)
    state_high = state_high.clamp(EPS, 1.0 - EPS)
    event_prob = (
        (1.0 - state_high) * event_prob_low + state_high * event_prob_high
    ).clamp(EPS, 1.0 - EPS)

    log_w_low = torch.log1p(-state_high) + torch.log(event_prob_low) - torch.log(event_prob)
    log_w_high = torch.log(state_high) + torch.log(event_prob_high) - torch.log(event_prob)

    sigma_log = severity_sigma.clamp(min=EPS)
    mu_log_low = torch.log(severity_mean_low.clamp(min=EPS)) - 0.5 * sigma_log * sigma_log
    mu_log_high = torch.log(severity_mean_high.clamp(min=EPS)) - 0.5 * sigma_log * sigma_log

    log_pdf_low = log_lognormal_pdf(severity_target, mu_log_low, sigma_log)
    log_pdf_high = log_lognormal_pdf(severity_target, mu_log_high, sigma_log)

    log_mixture = torch.logsumexp(
        torch.stack([log_w_low + log_pdf_low, log_w_high + log_pdf_high], dim=-1),
        dim=-1,
    )
    return -log_mixture[positive].mean()


def latent_risk_state_loss(
        output: LatentRiskStateOutput,
        event_target: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        prior_correction: bool = True,
        history_event_target: torch.Tensor | None = None,
        history_weight: float = 0.1,
        severity_weight: float = 0.1,
        emission_gap_min: float = 0.02,
        transition_gap_min: float = 0.10,
        separation_weight: float = 0.01,
        persistence_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    if history_weight < 0 or not math.isfinite(float(history_weight)):
        raise ValueError(f"history_weight must be finite and non-negative; got {history_weight}")
    if severity_weight < 0 or not math.isfinite(float(severity_weight)):
        raise ValueError(f"severity_weight must be finite and non-negative; got {severity_weight}")
    if event_target.shape != output.logits.shape:
        raise ValueError(
            "event_target shape "
            f"{tuple(event_target.shape)} does not match logits shape {tuple(output.logits.shape)}."
        )
    for name, tensor in (
            ("forecast_emission_low", output.forecast_emission_low),
            ("forecast_emission_high", output.forecast_emission_high),
            ("forecast_state_prior", output.forecast_state_prior),
            ("severity_mean_low", output.severity_mean_low),
            ("severity_mean_high", output.severity_mean_high),
            ("severity_sigma", output.severity_sigma),
    ):
        if tensor.shape != event_target.shape:
            raise ValueError(
                f"output.{name} shape {tuple(tensor.shape)} does not match "
                f"event_target shape {tuple(event_target.shape)}."
            )

    target = (event_target > 0).to(dtype=output.logits.dtype)
    logits = output.logits
    if prior_correction:
        logits = prior_corrected_logits(logits, pos_weight)
    event_nll = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
    )

    if history_weight > 0:
        if history_event_target is None:
            raise ValueError("history_event_target is required when history_weight > 0.")
        history_target = (history_event_target > 0).to(dtype=output.history_logits.dtype)
        if history_target.shape != output.history_logits.shape:
            raise ValueError(
                "history_event_target shape "
                f"{tuple(history_target.shape)} does not match history_logits shape "
                f"{tuple(output.history_logits.shape)}."
            )
        history_logits = output.history_logits
        if prior_correction:
            history_logits = prior_corrected_logits(history_logits, pos_weight)
        history_nll = F.binary_cross_entropy_with_logits(
            history_logits,
            history_target,
            pos_weight=pos_weight,
        )
    else:
        history_nll = torch.zeros((), dtype=output.logits.dtype, device=output.logits.device)

    severity_nll = _state_conditioned_severity_nll(
        event_target.to(dtype=output.logits.dtype),
        output.forecast_emission_low,
        output.forecast_emission_high,
        output.forecast_state_prior,
        output.severity_mean_low,
        output.severity_mean_high,
        output.severity_sigma,
    )

    emission_gap = output.emission_high - output.emission_low
    transition_gap = output.transition_persistence - output.transition_entry
    emission_separation = F.relu(float(emission_gap_min) - emission_gap).mean()
    transition_persistence = F.relu(float(transition_gap_min) - transition_gap).mean()
    loss = (
        event_nll
        + float(history_weight) * history_nll
        + float(severity_weight) * severity_nll
        + float(separation_weight) * emission_separation
        + float(persistence_weight) * transition_persistence
    )
    return {
        "loss": loss,
        "event_nll": event_nll,
        "history_nll": history_nll,
        "severity_nll": severity_nll,
        "emission_separation": emission_separation,
        "transition_persistence": transition_persistence,
    }
