from __future__ import annotations

import json
import os
import random
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_date_arg(s: str) -> Optional[pd.Timestamp]:
    if not s:
        return None
    return pd.to_datetime(s)


def build_time_splits_by_idx(
        T: int,
        history_len: int,
        horizon: int,
        train_end: int,
        val_end: int,
):
    """Time splits from explicit day-index boundaries (exclusive target ends).

    train: windows whose 3 target days end at or before day index train_end-1;
    val: targets fully inside [train_end, val_end); test: targets starting at
    or after val_end. Same acceptance rules as build_time_splits, which
    delegates here with ratio-derived indices.
    """
    if not 0 < train_end <= T:
        raise ValueError(f"train_end must fall in (0, T]; got {train_end} with T={T}")
    if not train_end <= val_end <= T:
        raise ValueError(f"require train_end <= val_end <= T; got {train_end}, {val_end}, {T}")
    max_start = T - (history_len + horizon)
    all_s = list(range(0, max_start + 1))

    train_s, val_s, test_s = [], [], []
    for s in all_s:
        tgt_start = s + history_len
        tgt_end = s + history_len + horizon
        if tgt_end <= train_end:
            train_s.append(s)
        elif (tgt_start >= train_end) and (tgt_end <= val_end):
            val_s.append(s)
        elif tgt_start >= val_end and tgt_end <= T:
            test_s.append(s)
    return train_s, val_s, test_s


def build_time_splits(T: int, history_len: int, horizon: int, train_ratio: float = 8 / 12, val_ratio: float = 2 / 12):
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    return build_time_splits_by_idx(T, history_len, horizon, train_end, val_end)


def load_mask(data_dir: str, mask_borough: str, mask_path: str, num_nodes: int) -> np.ndarray:
    if mask_path:
        path = mask_path
    else:
        name = mask_borough.strip().lower()
        if name in ("all", ""):
            return np.ones(num_nodes, dtype=bool)
        borough_map = {
            "bronx": "mask_borough_bronx.npz",
            "brooklyn": "mask_borough_brooklyn.npz",
            "manhattan": "mask_borough_manhattan.npz",
            "queens": "mask_borough_queens.npz",
            "staten island": "mask_borough_staten_island.npz",
            "staten_island": "mask_borough_staten_island.npz",
        }
        if name not in borough_map:
            # generic fallback for non-NYC datasets (e.g. Chicago super-regions):
            # mask_borough_<name>.npz inside the dataset's line_graph directory
            generic = os.path.join(
                data_dir, "line_graph", f"mask_borough_{name.replace(' ', '_').replace('-', '_')}.npz"
            )
            if not os.path.exists(generic):
                raise ValueError(f"Unknown mask_borough: {mask_borough}")
            path = generic
        else:
            path = os.path.join(data_dir, "line_graph", borough_map[name])

    z = np.load(path)
    key = "mask.npy" if "mask.npy" in z.files else z.files[0]
    mask = z[key].astype(bool)
    if mask.shape[0] != num_nodes:
        raise ValueError(f"Mask size mismatch: {mask.shape[0]} vs {num_nodes}")
    return mask


def load_nyc_zinb_data(
        data_dir: str,
        start_date: Optional[pd.Timestamp],
        end_date: Optional[pd.Timestamp],
):
    node_map_path = os.path.join(data_dir, "line_graph", "node_id_map.parquet")
    links_path = os.path.join(data_dir, "links_with_features.parquet")
    labels_path = os.path.join(data_dir, "crashes_daily_labels.parquet")
    weather_path = os.path.join(data_dir, "weather", "nyc_weather_daily.parquet")
    calendar_path = os.path.join(data_dir, "calendar", "calendar_day.parquet")
    scaler_path = os.path.join(data_dir, "links_features_scaler.json")
    borough_map_path = os.path.join(data_dir, "line_graph", "borough_id_mapping.json")

    node_df = pd.read_parquet(node_map_path).sort_values("node_idx")
    num_nodes = len(node_df)

    links_df = pd.read_parquet(links_path)
    static_cols = [c for c in links_df.columns if c.startswith("x_")]
    if "m_pop_imputed" in links_df.columns:
        static_cols.append("m_pop_imputed")
    static_df = node_df.merge(links_df[["id_geom_hash"] + static_cols], on="id_geom_hash", how="left")
    if static_df[static_cols].isnull().any().any():
        static_df[static_cols] = static_df[static_cols].fillna(0.0)
    X_static = static_df[static_cols].to_numpy(dtype=np.float32)

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler = json.load(f)
    col_to_idx = {c: i for i, c in enumerate(static_cols)}
    for col, stats in scaler.get("features", {}).items():
        if col in col_to_idx:
            idx = col_to_idx[col]
            std = stats.get("std", 1.0)
            if std == 0:
                std = 1.0
            X_static[:, idx] = (X_static[:, idx] - stats.get("mean", 0.0)) / std

    labels_df = pd.read_parquet(labels_path)
    weather_df = pd.read_parquet(weather_path)
    calendar_df = pd.read_parquet(calendar_path)

    for df in (labels_df, weather_df, calendar_df):
        df["id_date"] = pd.to_datetime(df["id_date"])

    min_date = max(calendar_df["id_date"].min(), weather_df["id_date"].min(), labels_df["id_date"].min())
    max_date = min(calendar_df["id_date"].max(), weather_df["id_date"].max(), labels_df["id_date"].max())
    if start_date is not None:
        min_date = max(min_date, start_date)
    if end_date is not None:
        max_date = min(max_date, end_date)
    if min_date > max_date:
        raise ValueError(f"Invalid date range after intersection: {min_date} > {max_date}")

    calendar_df = calendar_df[(calendar_df["id_date"] >= min_date) & (calendar_df["id_date"] <= max_date)].sort_values(
        "id_date")
    weather_df = weather_df[(weather_df["id_date"] >= min_date) & (weather_df["id_date"] <= max_date)]
    labels_df = labels_df[(labels_df["id_date"] >= min_date) & (labels_df["id_date"] <= max_date)]

    dates = calendar_df["id_date"].tolist()
    T = len(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    cal_cols = [c for c in calendar_df.columns if c.startswith("x_cal_")]
    calendar = calendar_df[cal_cols].to_numpy(dtype=np.float32)

    with open(borough_map_path, "r", encoding="utf-8") as f:
        borough_map = json.load(f)["borough_to_id"]
    weather_cols = [c for c in weather_df.columns if c.startswith("x_wth_")]
    weather = np.zeros((len(borough_map), T, len(weather_cols)), dtype=np.float32)
    for borough_name, borough_id in borough_map.items():
        sub = weather_df[weather_df["id_borough"] == borough_name].set_index("id_date")
        sub = sub.reindex(dates)
        if sub[weather_cols].isnull().any().any():
            raise ValueError(f"Weather missing values for {borough_name}")
        weather[borough_id] = sub[weather_cols].to_numpy(dtype=np.float32)

    crash_col = "y_crash_count"
    inj_col = "y_injured_sum"
    kill_col = "y_killed_sum"
    missing_cols = [c for c in (crash_col, inj_col, kill_col) if c not in labels_df.columns]
    if missing_cols:
        raise ValueError(f"Missing label columns: {missing_cols}. Available: {labels_df.columns.tolist()}")

    node_map = pd.Series(node_df["node_idx"].values, index=node_df["id_geom_hash"]).to_dict()
    labels_df["node_idx"] = labels_df["id_geom_hash"].map(node_map)
    labels_df["date_idx"] = labels_df["id_date"].map(date_to_idx)
    if labels_df["node_idx"].isnull().any() or labels_df["date_idx"].isnull().any():
        raise ValueError("Label rows contain unknown node or date")
    # Reconstruct the paper's event-space label from auditable count columns.
    # The parquet's legacy y_risk column follows an older transformation and is intentionally unused.
    risk = (
        labels_df[crash_col].fillna(0).astype(np.int64)
        + 2 * labels_df[inj_col].fillna(0).astype(np.int64)
        + 3 * labels_df[kill_col].fillna(0).astype(np.int64)
    )
    crash = labels_df[crash_col].fillna(0).astype(np.int64)
    Y_risk = np.zeros((num_nodes, T), dtype=np.float32)
    Y_crash = np.zeros((num_nodes, T), dtype=np.float32)
    Y_risk[labels_df["node_idx"].values, labels_df["date_idx"].values] = risk.to_numpy(dtype=np.float32)
    Y_crash[labels_df["node_idx"].values, labels_df["date_idx"].values] = crash.to_numpy(dtype=np.float32)

    borough_id_arr = np.load(os.path.join(data_dir, "line_graph", "borough_id.npy")).astype(np.int64)
    if borough_id_arr.shape[0] != num_nodes:
        raise ValueError("borough_id.npy size mismatch")

    return X_static, Y_risk, Y_crash, calendar, weather, borough_id_arr, dates


def load_borough_name_order(data_dir: str) -> List[str]:
    mapping_path = os.path.join(data_dir, "line_graph", "borough_id_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        borough_to_id = json.load(f)["borough_to_id"]
    return [name for name, _ in sorted(borough_to_id.items(), key=lambda kv: kv[1])]


def load_static_column_names(data_dir: str) -> List[str]:
    """Static feature column names in the exact order load_nyc_zinb_data assembles them."""
    links_df = pd.read_parquet(os.path.join(data_dir, "links_with_features.parquet"))
    static_cols = [c for c in links_df.columns if c.startswith("x_")]
    if "m_pop_imputed" in links_df.columns:
        static_cols.append("m_pop_imputed")
    return static_cols


def static_zero_indices(col_names: List[str], prefixes: List[str]) -> List[int]:
    """Column positions whose name starts with any of the given prefixes."""
    cleaned = [p.strip() for p in prefixes if p.strip()]
    return [j for j, c in enumerate(col_names) if any(c.startswith(p) for p in cleaned)]


def load_utm_centroids(data_dir: str, num_nodes: int) -> np.ndarray:
    """UTM18N segment centroids in node_idx order: [num_nodes, 2] (x, y) metres."""
    node_df = pd.read_parquet(
        os.path.join(data_dir, "line_graph", "node_id_map.parquet")
    ).sort_values("node_idx")
    meta = pd.read_parquet(
        os.path.join(data_dir, "links_meta.parquet"),
        columns=["id_geom_hash", "geo_centroid_x_utm", "geo_centroid_y_utm"],
    )
    df = node_df.merge(meta, on="id_geom_hash", how="left")
    if len(df) != num_nodes:
        raise ValueError(f"centroid table has {len(df)} rows, expected {num_nodes} nodes.")
    coords = df[["geo_centroid_x_utm", "geo_centroid_y_utm"]].to_numpy(dtype=np.float64)
    if np.isnan(coords).any():
        raise ValueError("centroid table contains missing coordinates for some nodes.")
    return coords


def build_spatial_block_holdout(
        centroids: np.ndarray,
        borough_id: np.ndarray,
        fraction: float,
        block_size_m: float = 1000.0,
        seed: int = 42,
) -> np.ndarray:
    """Whole-grid-cell holdout mask for a block-grouped zero-history regime.

    Blocks are borough-by-cell strata built from axis-aligned UTM cells of
    `block_size_m` metres. Within each borough, cells are shuffled under `seed`
    and taken greedily until the held-out node count reaches `fraction` of that
    borough's nodes. A cell is uniform within a borough, but a boundary cell
    shared by two boroughs may have different status in the two borough
    subgraphs. Selected strata need not form one contiguous region; no spatial
    buffer is imposed. Graph edges crossing cell boundaries are retained, so
    neighbouring seen-segment information remains observable.
    """
    if centroids.shape[0] != borough_id.shape[0]:
        raise ValueError("centroids and borough_id must cover the same nodes.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1).")
    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive.")
    bx = np.floor(centroids[:, 0] / float(block_size_m)).astype(np.int64)
    by = np.floor(centroids[:, 1] / float(block_size_m)).astype(np.int64)
    bx -= bx.min()
    by -= by.min()
    span = int(by.max()) + 1
    keys = bx * np.int64(span) + by
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(borough_id), dtype=bool)
    for b in np.unique(borough_id):
        idx = np.where(borough_id == b)[0]
        uniq, inverse, counts = np.unique(keys[idx], return_inverse=True, return_counts=True)
        target = int(round(fraction * len(idx)))
        if target <= 0:
            continue
        chosen: list[int] = []
        acc = 0
        for j in rng.permutation(len(uniq)):
            if acc >= target:
                break
            chosen.append(int(j))
            acc += int(counts[j])
        selected = np.zeros(len(uniq), dtype=bool)
        selected[chosen] = True
        mask[idx[selected[inverse]]] = True
    return mask
