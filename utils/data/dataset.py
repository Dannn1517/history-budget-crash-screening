from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class _BaseWindowDataset(Dataset):
    def __init__(
            self,
            Y_hist_raw: np.ndarray,
            Y_hist_target: np.ndarray,
            Y_target: np.ndarray,
            Y_rank: np.ndarray,
            Y_event: np.ndarray,
            hist_source_mode: str,
            history_len: int,
            horizon: int,
            start_indices: List[int],
            Y_sup: np.ndarray = None,
    ):
        super().__init__()
        self.Y_hist_raw = np.ascontiguousarray(Y_hist_raw.astype(np.float32, copy=False))
        self.Y_hist_target = np.ascontiguousarray(Y_hist_target.astype(np.float32, copy=False))
        self.Y_target = np.ascontiguousarray(Y_target.astype(np.float32, copy=False))
        self.Y_rank = np.ascontiguousarray(Y_rank.astype(np.float32, copy=False))
        self.Y_event = np.ascontiguousarray(Y_event.astype(np.float32, copy=False))
        if Y_sup is None:
            Y_sup = Y_event
        self.Y_sup = np.ascontiguousarray(Y_sup.astype(np.float32, copy=False))
        self.hist_source_mode = str(hist_source_mode)
        self.t = int(history_len)
        self.p = int(horizon)
        self.start_indices = start_indices
        self.num_nodes = int(self.Y_target.shape[0])

    def __len__(self) -> int:
        return len(self.start_indices)

    def _window_history(self, s: int) -> np.ndarray:
        raw_hist = self.Y_hist_raw[:, s:s + self.t, None]
        if self.hist_source_mode == "raw_risk":
            return raw_hist
        target_hist = self.Y_hist_target[:, s:s + self.t, None]
        if self.hist_source_mode == "target":
            return target_hist
        if self.hist_source_mode == "both":
            return np.concatenate([raw_hist, target_hist], axis=-1)
        raise ValueError(f"Unsupported hist_source_mode: {self.hist_source_mode}")

    def _window_targets(self, idx: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s = self.start_indices[idx]
        y_hist = self._window_history(s)
        y_fut = self.Y_target[:, s + self.t:s + self.t + self.p]
        y_rank_fut = self.Y_rank[:, s + self.t:s + self.t + self.p]
        y_event_fut = self.Y_event[:, s + self.t:s + self.t + self.p]
        y_sup_fut = self.Y_sup[:, s + self.t:s + self.t + self.p]
        return s, y_hist, y_fut, y_rank_fut, y_event_fut, y_sup_fut

    def _rghs_inputs(self, s: int, k: int = None) -> dict[str, np.ndarray]:
        """Reliability-gate inputs at the forecast origin (all observed, causal).

        freq_origin: frequency channel value at day s+t (exclusive cumulative/trailing
        mean over data up to s+t-1), identical to the forecast_origin ranker score.
        log_n: log archive depth per road, min(lookback, s+t); lookback 0 uses s+t.
        """
        origin = s + self.t
        if self.X_hist_freq_variants is not None:
            if k is None:
                k = self._select_lookback_index()
            freq_origin = self.X_hist_freq_variants[k][:, origin]
            lookback = self.variant_lookbacks[k] if self.variant_lookbacks is not None else 0
        else:
            freq_origin = self.X_hist_freq[:, origin]
            lookback = self.single_lookback
        if np.ndim(lookback) == 0:
            n = np.full(self.num_nodes, min(int(lookback) if lookback > 0 else origin, origin), dtype=np.float32)
        else:
            n = np.minimum(np.asarray(lookback, dtype=np.float32), float(origin))
            n = np.where(np.asarray(lookback) > 0, n, float(origin))
        if self.node_holdout is not None:
            # held-out nodes have no crash archive at all: zero-history endpoint
            freq_origin = np.where(self.node_holdout, 0.0, freq_origin).astype(np.float32)
            n = np.where(self.node_holdout, 0.0, n).astype(np.float32)
        return {
            "freq_origin": freq_origin.astype(np.float32),
            "log_n": np.log(np.maximum(n, 1.0)).astype(np.float32),
        }


class HABaselineDataset(_BaseWindowDataset):
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        _, y_hist, y_fut, y_rank_fut, y_event_fut, y_sup_fut = self._window_targets(idx)
        return {
            "y_hist": torch.from_numpy(np.ascontiguousarray(y_hist)).float(),
            "y": torch.from_numpy(np.ascontiguousarray(y_fut)).float(),
            "y_rank": torch.from_numpy(np.ascontiguousarray(y_rank_fut)).float(),
            "y_event": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
            "y_event_rank": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
            "y_sup": torch.from_numpy(np.ascontiguousarray(y_sup_fut)).float(),
        }


class GraphBaselineDataset(_BaseWindowDataset):
    def __init__(
            self,
            Y_hist_raw: np.ndarray,
            Y_hist_target: np.ndarray,
            Y_target: np.ndarray,
            Y_rank: np.ndarray,
            Y_event: np.ndarray,
            X_hist_freq: np.ndarray,
            X_static_public: np.ndarray,
            X_structural: np.ndarray,
            X_dynamic: np.ndarray,
            calendar: np.ndarray,
            weather_by_node: np.ndarray,
            hist_source_mode: str,
            history_len: int,
            horizon: int,
            start_indices: List[int],
            use_priors: bool,
            Y_sup: np.ndarray = None,
            X_hist_freq_variants: np.ndarray = None,
            mix_lookbacks: bool = False,
            lookback_index: int = 0,
            X_recency: np.ndarray = None,
            budget_channel_values: Sequence[float] = None,
            variant_lookbacks: Sequence[int] = None,
            single_lookback: int = 0,
            emit_rghs: bool = False,
            node_holdout: np.ndarray = None,
            feature_ablation: str = "none",
    ):
        super().__init__(
            Y_hist_raw=Y_hist_raw,
            Y_hist_target=Y_hist_target,
            Y_target=Y_target,
            Y_rank=Y_rank,
            Y_event=Y_event,
            hist_source_mode=hist_source_mode,
            history_len=history_len,
            horizon=horizon,
            start_indices=start_indices,
            Y_sup=Y_sup,
        )
        self.X_hist_freq = np.ascontiguousarray(X_hist_freq.astype(np.float32, copy=False))
        self.X_static_public = np.ascontiguousarray(X_static_public.astype(np.float32, copy=False))
        self.X_structural = np.ascontiguousarray(X_structural.astype(np.float32, copy=False))
        self.X_dynamic = np.ascontiguousarray(X_dynamic.astype(np.float32, copy=False))
        self.calendar = np.ascontiguousarray(calendar.astype(np.float32, copy=False))
        self.weather_by_node = np.ascontiguousarray(weather_by_node.astype(np.float32, copy=False))
        self.use_priors = bool(use_priors)
        if X_hist_freq_variants is not None:
            variants = np.ascontiguousarray(X_hist_freq_variants.astype(np.float32, copy=False))
            if variants.ndim != 3 or variants.shape[1:] != self.X_hist_freq.shape:
                raise ValueError(
                    "X_hist_freq_variants must have shape [K, N, T] matching X_hist_freq [N, T]."
                )
            self.X_hist_freq_variants = variants
        else:
            self.X_hist_freq_variants = None
        self.mix_lookbacks = bool(mix_lookbacks)
        self.lookback_index = int(lookback_index)
        if X_recency is not None:
            recency = np.ascontiguousarray(X_recency.astype(np.float32, copy=False))
            if recency.shape != self.X_hist_freq.shape:
                raise ValueError("X_recency must have shape [N, T] matching X_hist_freq.")
            self.X_recency = recency
        else:
            self.X_recency = None
        if budget_channel_values is not None:
            if self.X_hist_freq_variants is None:
                raise ValueError("budget_channel_values requires X_hist_freq_variants.")
            values = np.asarray(list(budget_channel_values), dtype=np.float32)
            if values.shape != (self.X_hist_freq_variants.shape[0],):
                raise ValueError("budget_channel_values must have one value per lookback variant.")
            self.budget_channel_values = values
        else:
            self.budget_channel_values = None
        self.variant_lookbacks = (
            tuple(int(v) for v in variant_lookbacks) if variant_lookbacks is not None else None
        )
        if self.variant_lookbacks is not None and self.X_hist_freq_variants is not None:
            if len(self.variant_lookbacks) != self.X_hist_freq_variants.shape[0]:
                raise ValueError("variant_lookbacks must have one value per variant.")
        if isinstance(single_lookback, np.ndarray):
            self.single_lookback = np.asarray(single_lookback, dtype=np.int64)
        else:
            self.single_lookback = int(single_lookback)
        self.emit_rghs = bool(emit_rghs)
        if node_holdout is not None:
            ho = np.asarray(node_holdout, dtype=bool)
            if ho.shape != (self.num_nodes,):
                raise ValueError("node_holdout must have one flag per node.")
            self.node_holdout = ho
        else:
            self.node_holdout = None
        self.feature_ablation = str(feature_ablation)
        if self.feature_ablation not in ("none", "static_only", "history_only"):
            raise ValueError(
                "feature_ablation must be one of none/static_only/history_only, got "
                f"{self.feature_ablation!r}."
            )
        if self.feature_ablation == "static_only":
            # Mechanism analysis: keep only static/contextual inputs
            # (static, structural, dynamic/expansion, calendar, weather) and
            # zero the crash-history-derived inputs (raw history window,
            # frequency prior channel, budget channel).
            # Zero IN PLACE: the arrays are shared across every dataset of a
            # bundle, and per-dataset copies OOM-killed the 30 GB host.
            self.Y_hist_raw[:] = 0.0
            self.X_hist_freq[:] = 0.0
            if self.X_hist_freq_variants is not None:
                self.X_hist_freq_variants[:] = 0.0
        elif self.feature_ablation == "history_only":
            # Mechanism analysis: keep only crash-history-derived inputs
            # (raw history window, frequency prior channel); zero
            # static/structural/dynamic/calendar/weather inputs.
            # Dimensions are preserved (zero-out ablation).
            self.X_static_public[:] = 0.0
            self.X_structural[:] = 0.0
            self.X_dynamic[:] = 0.0
            self.calendar[:] = 0.0
            self.weather_by_node[:] = 0.0

    def _select_lookback_index(self) -> int:
        if self.mix_lookbacks:
            return int(np.random.randint(self.X_hist_freq_variants.shape[0]))
        return self.lookback_index

    def _window_hist_freq(self, s: int, k: int = None) -> np.ndarray:
        if self.X_hist_freq_variants is None:
            return self.X_hist_freq[:, s:s + self.t, None]
        if k is None:
            k = self._select_lookback_index()
        return self.X_hist_freq_variants[k][:, s:s + self.t, None]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s, y_hist, y_fut, y_rank_fut, y_event_fut, y_sup_fut = self._window_targets(idx)
        x_static_t = np.broadcast_to(
            self.X_static_public[:, None, :],
            (self.num_nodes, self.t, self.X_static_public.shape[1]),
        )
        cal_hist = self.calendar[s:s + self.t, :]
        cal_t = np.broadcast_to(cal_hist[None, :, :], (self.num_nodes, self.t, cal_hist.shape[1]))
        w_hist = self.weather_by_node[:, s:s + self.t, :]
        k = self._select_lookback_index() if self.X_hist_freq_variants is not None else None
        x_hist_freq = self._window_hist_freq(s, k)
        if not self.use_priors:
            x_hist_freq = np.zeros_like(x_hist_freq, dtype=np.float32)

        parts = [y_hist]
        if self.use_priors:
            parts.append(x_hist_freq)
        if self.use_priors and self.budget_channel_values is not None and k is not None:
            parts.append(
                np.full(
                    (self.num_nodes, self.t, 1),
                    self.budget_channel_values[k],
                    dtype=np.float32,
                )
            )
        if self.use_priors and self.X_recency is not None:
            parts.append(self.X_recency[:, s:s + self.t, None])
        if x_static_t.shape[-1] > 0:
            parts.append(x_static_t)
        if self.use_priors and self.X_structural.shape[1] > 0:
            x_structural_t = np.broadcast_to(
                self.X_structural[:, None, :],
                (self.num_nodes, self.t, self.X_structural.shape[1]),
            )
            parts.append(x_structural_t)
        if self.X_dynamic.shape[2] > 0:
            parts.append(self.X_dynamic[:, s:s + self.t, :])
        if cal_t.shape[-1] > 0:
            parts.append(cal_t)
        if w_hist.shape[-1] > 0:
            parts.append(w_hist)
        x = np.concatenate(parts, axis=-1)

        sample = {
            "x": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "y": torch.from_numpy(np.ascontiguousarray(y_fut)).float(),
            "y_rank": torch.from_numpy(np.ascontiguousarray(y_rank_fut)).float(),
            "y_event": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
            "y_event_rank": torch.from_numpy(np.ascontiguousarray(y_event_fut)).float(),
            "y_sup": torch.from_numpy(np.ascontiguousarray(y_sup_fut)).float(),
        }
        if self.emit_rghs:
            rghs = self._rghs_inputs(s, k)
            sample["freq_origin"] = torch.from_numpy(np.ascontiguousarray(rghs["freq_origin"])).float()
            sample["log_n"] = torch.from_numpy(np.ascontiguousarray(rghs["log_n"])).float()
            sample["x_static"] = torch.from_numpy(np.ascontiguousarray(self.X_static_public)).float()
        if self.node_holdout is not None:
            sample["node_weight"] = torch.from_numpy(
                np.ascontiguousarray((~self.node_holdout).astype(np.float32))
            )
        return sample


class XGBBaselineDataset(_BaseWindowDataset):
    def __init__(
            self,
            Y_hist_raw: np.ndarray,
            Y_hist_target: np.ndarray,
            Y_target: np.ndarray,
            Y_rank: np.ndarray,
            Y_event: np.ndarray,
            X_hist_freq: np.ndarray,
            X_static_public: np.ndarray,
            X_structural: np.ndarray,
            X_dynamic: np.ndarray,
            calendar: np.ndarray,
            weather_by_node: np.ndarray,
            hist_source_mode: str,
            history_len: int,
            horizon: int,
            start_indices: List[int],
            use_priors: bool,
            Y_sup: np.ndarray = None,
    ):
        super().__init__(
            Y_hist_raw=Y_hist_raw,
            Y_hist_target=Y_hist_target,
            Y_target=Y_target,
            Y_rank=Y_rank,
            Y_event=Y_event,
            hist_source_mode=hist_source_mode,
            history_len=history_len,
            horizon=horizon,
            start_indices=start_indices,
            Y_sup=Y_sup,
        )
        self.X_hist_freq = np.ascontiguousarray(X_hist_freq.astype(np.float32, copy=False))
        self.X_static_public = np.ascontiguousarray(X_static_public.astype(np.float32, copy=False))
        self.X_structural = np.ascontiguousarray(X_structural.astype(np.float32, copy=False))
        self.X_dynamic = np.ascontiguousarray(X_dynamic.astype(np.float32, copy=False))
        self.calendar = np.ascontiguousarray(calendar.astype(np.float32, copy=False))
        self.weather_by_node = np.ascontiguousarray(weather_by_node.astype(np.float32, copy=False))
        self.use_priors = bool(use_priors)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        s = self.start_indices[idx]
        y_hist_flat = self._window_history(s).transpose(0, 2, 1).reshape(self.num_nodes, -1)
        y_fut = self.Y_target[:, s + self.t:s + self.t + self.p]
        y_rank_fut = self.Y_rank[:, s + self.t:s + self.t + self.p]
        y_event_fut = self.Y_event[:, s + self.t:s + self.t + self.p]
        y_sup_fut = self.Y_sup[:, s + self.t:s + self.t + self.p]
        num_nodes = self.num_nodes
        cal_hist = self.calendar[s:s + self.t, :]
        cal_flat = np.broadcast_to(cal_hist.reshape(1, -1), (num_nodes, cal_hist.size)).copy()
        weather_flat = self.weather_by_node[:, s:s + self.t, :].reshape(num_nodes, -1)

        parts = [y_hist_flat]
        if self.use_priors:
            x_hist_freq_flat = self.X_hist_freq[:, s:s + self.t]
            parts.append(x_hist_freq_flat)
        if self.X_static_public.shape[1] > 0:
            parts.append(self.X_static_public)
        if self.use_priors and self.X_structural.shape[1] > 0:
            parts.append(self.X_structural)
        if self.X_dynamic.shape[2] > 0:
            parts.append(self.X_dynamic[:, s:s + self.t, :].reshape(num_nodes, -1))
        if cal_flat.shape[1] > 0:
            parts.append(cal_flat)
        if weather_flat.shape[1] > 0:
            parts.append(weather_flat)
        x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

        return {
            "x": x,
            "y": y_fut.astype(np.float32, copy=False),
            "y_rank": y_rank_fut.astype(np.float32, copy=False),
            "y_event": y_event_fut.astype(np.float32, copy=False),
            "y_event_rank": y_event_fut.astype(np.float32, copy=False),
            "y_sup": y_sup_fut.astype(np.float32, copy=False),
        }
