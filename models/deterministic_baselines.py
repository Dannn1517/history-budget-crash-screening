from __future__ import annotations

import math
from typing import Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_output_activation(value: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "softplus":
        return F.softplus(value)
    if activation == "identity":
        return value
    raise ValueError(f"Unsupported output activation: {activation}")


def _support_mm(support: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if support.is_sparse:
        support = support.coalesce()
        return torch.stack([torch.sparse.mm(support, x_i) for x_i in x], dim=0)
    return torch.einsum("nm,bmf->bnf", support, x)


def _average_support(A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
    if A_q.is_sparse:
        avg = (A_q.coalesce() + A_h.coalesce()).coalesce()
        return torch.sparse_coo_tensor(
            avg.indices(),
            avg.values() * 0.5,
            avg.size(),
            device=avg.device,
            dtype=avg.dtype,
        ).coalesce()
    return 0.5 * (A_q + A_h)


def _edge_index_from_supports(A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
    if A_q.is_sparse:
        idx = torch.cat([A_q.coalesce().indices(), A_h.coalesce().indices()], dim=1)
        num_nodes = A_q.size(0)
        device = idx.device
    else:
        dense_mask = (A_q != 0) | (A_h != 0)
        idx = torch.nonzero(dense_mask, as_tuple=False).t().contiguous()
        num_nodes = A_q.size(0)
        device = A_q.device

    loop = torch.arange(num_nodes, device=device, dtype=torch.long)
    loop = torch.stack([loop, loop], dim=0)
    idx = torch.cat([idx.long(), loop], dim=1)
    return torch.unique(idx, dim=1)


class CausalTemporalConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.filter_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.kernel_size))
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, self.kernel_size))
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        filt = self.filter_conv(x_pad)
        gate = torch.sigmoid(self.gate_conv(x_pad))
        y = filt * gate + self.residual(x)
        return self.dropout(F.relu(y))


class DiffusionConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, order: int, num_supports: int = 1):
        super().__init__()
        self.order = int(order)
        self.num_supports = int(num_supports)
        self.linear = nn.Linear(in_dim * (1 + self.order * self.num_supports), out_dim)

    def forward(self, x: torch.Tensor, supports: Iterable[torch.Tensor]) -> torch.Tensor:
        if self.order <= 0:
            # order=0 即关闭图消息传递：仅逐节点线性混合（no-graph 消融对照）。
            return self.linear(x)
        x_terms: List[torch.Tensor] = [x]
        for support in supports:
            x0 = x
            x1 = _support_mm(support, x0)
            x_terms.append(x1)
            prev_prev, prev = x0, x1
            for _ in range(2, self.order + 1):
                current = 2.0 * _support_mm(support, prev) - prev_prev
                x_terms.append(current)
                prev_prev, prev = prev, current
        return self.linear(torch.cat(x_terms, dim=-1))


class STGCNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        kernel_size: int = 3,
        cheb_order: int = 2,
        dropout: float = 0.1,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.output_activation = str(output_activation)
        self.temp1 = CausalTemporalConv(input_dim, hidden_dim, kernel_size, dropout)
        self.spatial = DiffusionConv(hidden_dim, hidden_dim, cheb_order, num_supports=1)
        self.temp2 = CausalTemporalConv(hidden_dim, hidden_dim, kernel_size, dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        support = _average_support(A_q, A_h)
        h = x.permute(0, 3, 1, 2).contiguous()
        h = self.temp1(h)

        spatial_steps = []
        for step in range(h.size(-1)):
            h_t = h[:, :, :, step].permute(0, 2, 1).contiguous()
            h_t = self.spatial(h_t, [support])
            spatial_steps.append(h_t.permute(0, 2, 1).unsqueeze(-1))
        h = torch.cat(spatial_steps, dim=-1)

        h = self.temp2(h)
        h_last = h[:, :, :, -1].permute(0, 2, 1).contiguous()
        return _apply_output_activation(self.head(h_last), self.output_activation)


class SparseGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.heads = int(heads)
        self.concat = bool(concat)
        self.dropout = float(dropout)

        self.proj = nn.Linear(self.in_dim, self.heads * self.out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.attn_dst = nn.Parameter(torch.empty(self.heads, self.out_dim))
        self.bias = nn.Parameter(torch.zeros(self.heads * self.out_dim if self.concat else self.out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        nn.init.zeros_(self.bias)

    def _single_head(self, z: torch.Tensor, edge_index: torch.Tensor, head: int) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        logits = (z[:, src, :] * self.attn_src[head]).sum(dim=-1)
        logits = logits + (z[:, dst, :] * self.attn_dst[head]).sum(dim=-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        outputs = []
        num_nodes = z.size(1)
        for batch_idx in range(z.size(0)):
            att = torch.sparse_coo_tensor(
                edge_index,
                logits[batch_idx],
                (num_nodes, num_nodes),
                device=z.device,
                dtype=z.dtype,
            ).coalesce()
            att = torch.sparse.softmax(att, dim=1)
            values = F.dropout(att.values(), p=self.dropout, training=self.training)
            att = torch.sparse_coo_tensor(att.indices(), values, att.size(), device=z.device, dtype=z.dtype).coalesce()
            outputs.append(torch.sparse.mm(att, z[batch_idx]))
        return torch.stack(outputs, dim=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        proj = self.proj(x).view(x.size(0), x.size(1), self.heads, self.out_dim)
        head_outputs = [self._single_head(proj[:, :, head, :], edge_index, head) for head in range(self.heads)]

        if self.concat:
            out = torch.cat(head_outputs, dim=-1)
        else:
            out = torch.mean(torch.stack(head_outputs, dim=0), dim=0)
        out = out + self.bias
        return F.elu(out)


class STGATBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        heads: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.output_activation = str(output_activation)
        per_head_dim = max(1, hidden_dim // max(1, heads))
        gat_hidden_dim = per_head_dim * max(1, heads)

        self.temp = CausalTemporalConv(input_dim, gat_hidden_dim, kernel_size, dropout)
        self.gat1 = SparseGraphAttentionLayer(gat_hidden_dim, per_head_dim, heads=heads, dropout=dropout, concat=True)
        self.gat2 = SparseGraphAttentionLayer(gat_hidden_dim, gat_hidden_dim, heads=1, dropout=dropout, concat=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(gat_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        edge_index = _edge_index_from_supports(A_q, A_h)
        h = x.permute(0, 3, 1, 2).contiguous()
        h = self.temp(h)
        h = h[:, :, :, -1].permute(0, 2, 1).contiguous()
        h = self.gat1(h, edge_index)
        h = self.dropout(h)
        h = self.gat2(h, edge_index)
        h = self.dropout(h)
        return _apply_output_activation(self.head(h), self.output_activation)


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, diffusion_steps: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        concat_dim = input_dim + self.hidden_dim
        self.gate_conv = DiffusionConv(concat_dim, 2 * self.hidden_dim, diffusion_steps, num_supports=2)
        self.update_conv = DiffusionConv(concat_dim, self.hidden_dim, diffusion_steps, num_supports=2)

    def forward(self, x: torch.Tensor, state: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, state], dim=-1)
        gates = torch.sigmoid(self.gate_conv(combined, [A_q, A_h]))
        reset, update = torch.chunk(gates, chunks=2, dim=-1)
        candidate = torch.cat([x, reset * state], dim=-1)
        candidate = torch.tanh(self.update_conv(candidate, [A_q, A_h]))
        return update * state + (1.0 - update) * candidate


class DCRNNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        diffusion_steps: int = 2,
        dropout: float = 0.0,
        output_activation: str = "softplus",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.dropout = nn.Dropout(dropout)
        self.output_activation = str(output_activation)
        self.encoder_cell = DCGRUCell(input_dim, hidden_dim, diffusion_steps)
        self.decoder_cell = DCGRUCell(1, hidden_dim, diffusion_steps)
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, A_q: torch.Tensor, A_h: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _, _ = x.shape
        state = x.new_zeros(batch_size, num_nodes, self.hidden_dim)

        for step in range(x.size(2)):
            state = self.encoder_cell(x[:, :, step, :], state, A_q, A_h)

        state = self.dropout(state)
        decoder_input = x[:, :, -1, :1]
        outputs = []
        for _ in range(self.horizon):
            state = self.decoder_cell(decoder_input, state, A_q, A_h)
            state = self.dropout(state)
            decoder_input = _apply_output_activation(self.proj(state), self.output_activation)
            outputs.append(decoder_input)
        return torch.cat(outputs, dim=-1)
