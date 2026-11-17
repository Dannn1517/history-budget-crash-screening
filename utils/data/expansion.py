from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


EXPANSION_COLUMNS = {
    "construction": [
        "x_construction_active",
        "x_construction_count_1d",
        "x_construction_count_7d",
        "x_construction_neighbor_count_7d",
    ],
    "311": [
        "x_311_roadissue_count_1d",
        "x_311_roadissue_count_7d",
        "x_311_neighbor_roadissue_count_7d",
        "x_311_pothole_count_1d",
        "x_311_pothole_count_7d",
        "x_311_signal_count_1d",
        "x_311_signal_count_7d",
        "x_311_streetlight_count_1d",
        "x_311_streetlight_count_7d",
        "x_311_streetcond_count_7d",
        "x_311_sign_count_7d",
        "x_311_sidewalk_count_7d",
        "x_311_snowice_count_7d",
    ],
    "mobility": [
        "x_mobility_pickups_1d",
        "x_mobility_dropoffs_1d",
        "x_mobility_trip_count_7d",
        "x_mobility_change_1d",
        "m_mobility_no_zone",
    ],
}


def parse_expansion_groups(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    else:
        raw = [str(part).strip().lower() for part in value if str(part).strip()]
    if not raw or raw == ["none"]:
        return ()
    if "none" in raw:
        raise ValueError("Expansion feature 'none' cannot be combined with other groups")
    if "all" in raw:
        if len(raw) != 1:
            raise ValueError("Expansion feature 'all' cannot be combined with other groups")
        return tuple(EXPANSION_COLUMNS)
    unknown = sorted(set(raw) - set(EXPANSION_COLUMNS))
    if unknown:
        raise ValueError(f"Unknown expansion feature groups: {unknown}")
    return tuple(group for group in EXPANSION_COLUMNS if group in raw)


def _date_indices(values, first_date: np.datetime64, n_dates: int) -> np.ndarray:
    dates = pd.to_datetime(np.asarray(values)).values.astype("datetime64[D]")
    indices = (dates - first_date).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= n_dates):
        raise ValueError("Expansion feature rows fall outside the requested date range")
    return indices


def _load_sparse_group(
        path: Path,
        node_hashes: np.ndarray,
        dates: Sequence[pd.Timestamp],
        columns: list[str],
) -> np.ndarray:
    first_date = np.datetime64(pd.Timestamp(dates[0]).date())
    last_date = pd.Timestamp(dates[-1]).strftime("%Y-%m-%d")
    first_date_str = pd.Timestamp(dates[0]).strftime("%Y-%m-%d")
    hash_values = pa.array(node_hashes.tolist())
    predicate = (
        ds.field("id_geom_hash").isin(hash_values)
        & (ds.field("id_date") >= first_date_str)
        & (ds.field("id_date") <= last_date)
    )
    table = ds.dataset(path, format="parquet").to_table(
        columns=["id_geom_hash", "id_date", *columns], filter=predicate
    )
    output = np.zeros((len(node_hashes), len(dates), len(columns)), dtype=np.float32)
    if table.num_rows == 0:
        return output
    node_idx = pc.index_in(table["id_geom_hash"], value_set=hash_values).to_numpy()
    if np.any(node_idx < 0):
        raise ValueError(f"{path.name} contains nodes outside the requested node set")
    date_idx = _date_indices(table["id_date"].to_numpy(), first_date, len(dates))
    key_pairs = np.stack([node_idx, date_idx], axis=1)
    if len(np.unique(key_pairs, axis=0)) != len(key_pairs):
        raise ValueError(f"{path.name} contains duplicate node-date keys")
    for feature_idx, column in enumerate(columns):
        output[node_idx, date_idx, feature_idx] = table[column].to_numpy(
            zero_copy_only=False
        ).astype(np.float32, copy=False)
    return output


def _load_mobility(
        expansion_dir: Path,
        node_hashes: np.ndarray,
        dates: Sequence[pd.Timestamp],
        availability_lag_days: int = 0,
) -> np.ndarray:
    if availability_lag_days < 0:
        raise ValueError("availability_lag_days must be >= 0")
    mapping = pd.read_parquet(expansion_dir / "links_mobility_zone.parquet").set_index("id_geom_hash")
    mapping = mapping.reindex(node_hashes)
    if mapping["m_mobility_no_zone"].isna().any():
        raise ValueError("Mobility mapping does not cover every requested node")

    columns = EXPANSION_COLUMNS["mobility"][:-1]
    zone_daily = pd.read_parquet(
        expansion_dir / "taxi_zone_daily_mobility.parquet",
        columns=["id_taxi_zone", "id_date", *columns],
    )
    zone_daily["id_date"] = pd.to_datetime(zone_daily["id_date"])
    wanted_dates = pd.DatetimeIndex(dates)
    # TLC publishes monthly with roughly a two-month delay: on feature day d
    # the newest legally available record is trip day d - lag. The lag is
    # resolved on REAL calendar dates, so a request window starting mid-source
    # still reads its true t-lag history; zero-fill happens only where the
    # source table genuinely has no record for the trip day (e.g. before
    # 2021-01-01). The static no-zone mask channel is deliberately unshifted.
    if availability_lag_days > 0:
        if len(wanted_dates) == 0:
            raise ValueError("dates must not be empty when availability_lag_days > 0")
        expected = pd.date_range(wanted_dates[0], wanted_dates[-1], freq="D")
        if len(expected) != len(wanted_dates) or not (wanted_dates == expected).all():
            raise ValueError("mobility availability lag requires a contiguous daily date axis")
        trip_dates = wanted_dates - pd.Timedelta(days=availability_lag_days)
    else:
        trip_dates = wanted_dates
    zones = np.sort(zone_daily["id_taxi_zone"].unique())
    zone_index = {zone: idx for idx, zone in enumerate(zones)}
    date_index = pd.Series(np.arange(len(trip_dates)), index=trip_dates)
    zone_daily = zone_daily[zone_daily["id_date"].isin(date_index.index)]
    dense = np.zeros((len(zones), len(trip_dates), len(columns)), dtype=np.float32)
    if len(zone_daily):
        zone_idx = zone_daily["id_taxi_zone"].map(zone_index).to_numpy(dtype=np.int64)
        date_idx = zone_daily["id_date"].map(date_index).to_numpy(dtype=np.int64)
        for feature_idx, column in enumerate(columns):
            dense[zone_idx, date_idx, feature_idx] = zone_daily[column].to_numpy(dtype=np.float32)

    output = np.zeros((len(node_hashes), len(dates), len(columns) + 1), dtype=np.float32)
    has_zone = mapping["m_mobility_no_zone"].to_numpy(dtype=np.uint8) == 0
    mapped_zone = mapping.loc[has_zone, "id_taxi_zone"].map(zone_index)
    if mapped_zone.isna().any():
        raise ValueError("Mobility zone-day table is missing a mapped taxi zone")
    output[has_zone, :, :-1] = dense[mapped_zone.to_numpy(dtype=np.int64)]
    output[:, :, -1] = mapping["m_mobility_no_zone"].to_numpy(dtype=np.float32)[:, None]
    return output


def _transform_and_standardize(
        values: np.ndarray,
        names: list[str],
        train_end: int,
) -> tuple[np.ndarray, dict[str, dict[str, float | str]]]:
    if not 0 < train_end <= values.shape[1]:
        raise ValueError("train_end must fall within the dynamic feature date axis")
    stats: dict[str, dict[str, float | str]] = {}
    for idx, name in enumerate(names):
        if name == "x_mobility_change_1d":
            values[:, :, idx] = np.sign(values[:, :, idx]) * np.log1p(np.abs(values[:, :, idx]))
            transform = "signed_log1p"
        elif name == "m_mobility_no_zone":
            transform = "identity"
        else:
            values[:, :, idx] = np.log1p(np.maximum(values[:, :, idx], 0.0))
            transform = "log1p"
        train_values = values[:, :train_end, idx]
        mean = float(np.mean(train_values, dtype=np.float64))
        std = float(np.std(train_values, dtype=np.float64))
        if std == 0.0:
            std = 1.0
        values[:, :, idx] = (values[:, :, idx] - mean) / std
        stats[name] = {"transform": transform, "mean": mean, "std": std}
    return values, stats


def load_expansion_features(
        expansion_dir: str | Path,
        node_hashes: Sequence[str],
        dates: Sequence[pd.Timestamp],
        groups: str | Iterable[str],
        train_end: int,
        mobility_lag_days: int = 60,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float | str]]]:
    """Load expansion feature blocks, transform and train-window standardize.

    ``mobility_lag_days`` is the TLC availability lag: feature day t carries
    zone-day trip aggregates of day t - lag (pre-lag days zero-filled). The
    formal protocol fixes lag=60; lag=0 is the same-day retrospective ORACLE
    and is only permitted as a labelled sensitivity, never in formal arms.
    """
    selected = parse_expansion_groups(groups)
    hashes = np.asarray(node_hashes, dtype=object)
    if len(set(hashes.tolist())) != len(hashes):
        raise ValueError("Requested node hashes must be unique")
    if not selected:
        return np.zeros((len(hashes), len(dates), 0), dtype=np.float32), [], {}

    root = Path(expansion_dir)
    blocks: list[np.ndarray] = []
    names: list[str] = []
    for group in selected:
        columns = EXPANSION_COLUMNS[group]
        if group == "mobility":
            block = _load_mobility(root, hashes, dates, availability_lag_days=int(mobility_lag_days))
        else:
            block = _load_sparse_group(
                root / f"links_daily_{group}.parquet", hashes, dates, columns
            )
        blocks.append(block)
        names.extend(columns)
    values = np.concatenate(blocks, axis=2) if len(blocks) > 1 else blocks[0]
    values, stats = _transform_and_standardize(values, names, train_end)
    if "mobility" in selected:
        for name in EXPANSION_COLUMNS["mobility"][:-1]:
            stats[name]["availability_lag_days"] = int(mobility_lag_days)
        stats[EXPANSION_COLUMNS["mobility"][-1]]["availability_lag_days"] = 0  # static mask, unshifted
    return values, names, stats
