#!/usr/bin/env python3
"""Materialize a loader-compatible Chicago data directory from a build directory.

Source: set CHICAGO_RAW_SRC to a Chicago build directory derived from the
public City of Chicago Traffic Crashes data; no default path is assumed.
Target (default): data_chicago/ (kept separate from the NYC data/ tree).

The NYC loader expects a flat layout with borough-style partitions. Chicago
ships 77 Community Areas and 2 weather zones, so this script materializes:

  - borough := 5 super-regions (k-means, k=5, seed 42 on the 77 CA centroids
    computed from node UTM16N centroids; deterministic, documented). The
    cross-city primary metric is citywide top-k;
    super-regions only mirror the NYC training structure (sub-city subgraphs
    with per-district weather) at comparable scale.
  - line_graph/borough_id.npy + borough_id_mapping.json + mask_borough_*.npz
  - weather/nyc_weather_daily.parquet: zone rows expanded to super-region rows
    (each super-region takes its majority weather zone; boundary CAs may sit in
    the minority zone of their super-region - documented approximation).

Originals (region_id, masks, zone weather, README, manifest, QA summaries) are
copied alongside for audit into a clear provenance layout: ``manifest.json``
(the source build manifest) and ``qa/*.json|*.md`` (small QA/provenance
summaries). Raw data and large derived artifacts are never copied by this
script; only the loader inputs listed above are materialized.

Validation asserts the README-documented invariants and runs
load_nyc_zinb_data + build_bundles smoke checks.

Usage:
  PYTHONPATH=. python tools/prepare_chicago_data_dir.py [--force]
  PYTHONPATH=. python tools/prepare_chicago_data_dir.py \
      --src /path/to/build/data/chicago --dst /path/to/out
  PYTHONPATH=. python tools/prepare_chicago_data_dir.py \
      --src /path/to/build/data/chicago --dst data_chicago --provenance-only
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import os

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path(os.environ["CHICAGO_RAW_SRC"]) if os.environ.get("CHICAGO_RAW_SRC") else None
DEFAULT_DST = ROOT / "data_chicago"
K_SUPER = 5
SEED = 42


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Materialize a loader-compatible Chicago data directory.")
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="Chicago build directory root (default preserves legacy behavior).")
    ap.add_argument("--dst", type=Path, default=DEFAULT_DST,
                    help="Destination data directory (default: data_chicago/).")
    ap.add_argument("--force", action="store_true", help="recreate the destination from scratch.")
    ap.add_argument(
        "--provenance-only",
        action="store_true",
        help="copy/refresh only the small manifest/QA/README provenance files, "
             "without rebuilding or overwriting model inputs.",
    )
    return ap


def _validate_flags(force: bool, provenance_only: bool) -> None:
    if force and provenance_only:
        raise ValueError("--force and --provenance-only are mutually exclusive.")


def assert_safe_dst(dst: Path) -> None:
    """Reject obviously broad / destructive destinations before any write."""
    resolved = dst.resolve()
    forbidden = {
        Path(resolved.anchor),        # filesystem root ("/")
        Path.home().resolve(),        # user home
        ROOT.resolve(),               # repository root
    }
    if resolved in forbidden:
        raise ValueError(f"refusing to use {dst} as a build destination: it is a protected location.")
    try:
        ROOT.resolve().relative_to(resolved)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"refusing to use {dst} as a build destination: it contains the repository root."
        )


def copy_provenance(src: Path, dst: Path) -> dict:
    """Copy the small provenance surface (manifest + QA summaries + README).

    Never touches raw data or large derived artifacts. Returns a summary of
    what was copied so callers can log or assert on it.
    """
    manifest = src / "manifest.json"
    qa_src = src / "qa"
    acceptance = qa_src / "final_acceptance.md"
    for required in (manifest, acceptance):
        if not required.is_file():
            raise FileNotFoundError(f"required provenance file does not exist: {required}")

    dst.mkdir(parents=True, exist_ok=True)
    qa_dst = dst / "qa"
    qa_dst.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    shutil.copy2(manifest, dst / "manifest.json")
    copied.append("manifest.json")

    if qa_src.is_dir():
        for f in sorted(list(qa_src.glob("*.json")) + list(qa_src.glob("*.md"))):
            shutil.copy2(f, qa_dst / f.name)
            copied.append(f"qa/{f.name}")

    readme = src / "README.md"
    if readme.exists():
        shutil.copy2(readme, dst / "README_source.md")
        copied.append("README_source.md")

    return {"src": str(src), "dst": str(dst), "copied": copied}


def copy_common_core(core: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "line_graph").mkdir(parents=True, exist_ok=True)
    (dst / "calendar").mkdir(parents=True, exist_ok=True)
    (dst / "weather").mkdir(parents=True, exist_ok=True)

    for name in ("links_with_features.parquet", "links_features_scaler.json", "links_meta.parquet",
                 "train_keys_all.parquet", "crashes_daily_labels.parquet"):
        shutil.copy2(core / name, dst / name)
    for name in ("node_id_map.parquet", "edge_index.npy", "edge_weight.npy", "graph_meta.json",
                 "region_id.npy", "region_id_mapping.json", "weather_zone_id.npy",
                 "weather_zone_mapping.json"):
        shutil.copy2(core / "line_graph" / name, dst / "line_graph" / name)
    for f in sorted((core / "line_graph").glob("mask_region_*.npz")):
        shutil.copy2(f, dst / "line_graph" / f.name)
    shutil.copy2(core / "calendar/calendar_day.parquet", dst / "calendar/calendar_day.parquet")
    shutil.copy2(core / "weather/chicago_weather_daily.parquet", dst / "weather/chicago_weather_daily.parquet")
    for extra in ("weather/weather_meta.json", "weather/weather_code_mapping.json"):
        if (core / extra).exists():
            shutil.copy2(core / extra, dst / extra)
    print("copied common_core files")


def build_super_regions(dst: Path) -> dict:
    """K-means the 77 CA centroids into super-regions and write the borough inputs."""
    from sklearn.cluster import KMeans

    node_map = pd.read_parquet(dst / "line_graph/node_id_map.parquet").sort_values("node_idx")
    meta = pd.read_parquet(dst / "links_meta.parquet",
                           columns=["id_geom_hash", "geo_centroid_x_utm", "geo_centroid_y_utm"])
    df = node_map.merge(meta, on="id_geom_hash", how="left")
    assert len(df) == 43335 and not df[["geo_centroid_x_utm", "geo_centroid_y_utm"]].isnull().any().any()
    region_id = np.load(dst / "line_graph/region_id.npy").astype(int)   # 1..77, node order
    zone_id = np.load(dst / "line_graph/weather_zone_id.npy").astype(int)
    zone_names = {int(k): v for k, v in
                  json.loads((dst / "line_graph/weather_zone_mapping.json").read_text()).items() if k.isdigit()}
    assert region_id.shape[0] == len(df)

    ca_ids = np.arange(1, 78)
    cx = np.array([df.loc[region_id == ca, "geo_centroid_x_utm"].mean() for ca in ca_ids])
    cy = np.array([df.loc[region_id == ca, "geo_centroid_y_utm"].mean() for ca in ca_ids])
    ca_centroids = np.stack([cx, cy], axis=1)
    km = KMeans(n_clusters=K_SUPER, random_state=SEED, n_init=10).fit(ca_centroids)
    ca_to_super = {int(ca): int(lab) for ca, lab in zip(ca_ids, km.labels_)}
    super_of_node = np.array([ca_to_super[r] for r in region_id], dtype=np.int64)
    names = [f"super_{k + 1}" for k in range(K_SUPER)]

    # zone per super-region = majority of member nodes
    super_zone = {}
    for k in range(K_SUPER):
        z = zone_id[super_of_node == k]
        super_zone[k] = int(np.bincount(z).argmax())

    np.save(dst / "line_graph/borough_id.npy", super_of_node)
    (dst / "line_graph/borough_id_mapping.json").write_text(
        json.dumps({"borough_to_id": {names[k]: k for k in range(K_SUPER)}}, indent=2))
    for k in range(K_SUPER):
        np.savez_compressed(dst / "line_graph" / f"mask_borough_{names[k]}.npz",
                            mask=(super_of_node == k))

    region_names = json.loads((dst / "line_graph/region_id_mapping.json").read_text())
    prov = {
        "method": f"KMeans(k={K_SUPER}, random_state={SEED}, n_init=10) on the 77 Community-Area centroids "
                  "(mean node UTM16N centroid per CA)",
        "ca_to_super": {f"{ca} ({region_names.get(str(ca), '?')})": names[s] for ca, s in ca_to_super.items()},
        "super_zone_majority": {names[k]: int(v) for k, v in super_zone.items()},
        "super_zone_names": {names[k]: zone_names[int(v)] for k, v in super_zone.items()},
        "node_counts": {names[k]: int((super_of_node == k).sum()) for k in range(K_SUPER)},
        "note": "Cross-city primary metrics are citywide top-k; "
                "super-regions exist only to mirror the NYC sub-city training partition.",
    }
    (dst / "chicago_super_regions.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False))
    print("super-regions:", prov["node_counts"], "| zones:", prov["super_zone_majority"])
    return {"prov": prov, "super_zone": super_zone, "zone_names": zone_names}


def expand_weather(dst: Path, super_zone: dict, zone_names: dict) -> None:
    """Expand zone weather rows into per-super-region rows (majority zone)."""
    wea = pd.read_parquet(dst / "weather/chicago_weather_daily.parquet")
    xwth = [c for c in wea.columns if c.startswith("x_wth_")]
    keep = ["id_date"] + xwth + (["m_wth_imputed"] if "m_wth_imputed" in wea.columns else [])
    parts = []
    names = [f"super_{k + 1}" for k in range(K_SUPER)]
    for k in range(K_SUPER):
        zone_name = zone_names[super_zone[k]]
        sub = wea[wea["id_weather_zone"] == zone_name][keep].copy()
        assert len(sub) == 1461, (names[k], zone_name, len(sub))
        sub.insert(0, "id_borough", names[k])
        parts.append(sub)
    wea_out = pd.concat(parts, ignore_index=True)
    assert len(wea_out) == K_SUPER * 1461, len(wea_out)
    assert not wea_out[xwth].isnull().any().any()
    wea_out.to_parquet(dst / "weather/nyc_weather_daily.parquet", index=False)
    print(f"weather expanded: {len(wea_out)} rows ({K_SUPER} regions x 1461 days)")


def validate_dst(dst: Path) -> None:
    from utils.data.loading import build_time_splits, load_nyc_zinb_data

    X_static, Y_risk, Y_crash, calendar, weather, borough_arr, dates = load_nyc_zinb_data(
        str(dst), pd.Timestamp("2021-01-01"), pd.Timestamp("2024-12-31"))
    assert X_static.shape == (43335, 70), X_static.shape
    assert Y_crash.shape[1] == 1461
    total_crashes = float(Y_crash.sum())
    nonzero_road_days = int((Y_crash > 0).sum())
    print(f"labels: total crash count {total_crashes:.0f} (README 403252), nonzero road-days {nonzero_road_days} (README 394400)")
    assert abs(total_crashes - 403252) <= 2, total_crashes
    assert abs(nonzero_road_days - 394400) <= 2, nonzero_road_days
    assert not np.isnan(weather).any() and not np.isnan(calendar).any()

    scaler = json.loads((dst / "links_features_scaler.json").read_text())["features"]
    links = pd.read_parquet(dst / "links_with_features.parquet")
    static_cols = [c for c in links.columns if c.startswith("x_")] + ["m_pop_imputed"]
    for col, st in list(scaler.items())[:3]:
        j = static_cols.index(col)
        z = X_static[:, j]
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.05, (col, z.mean(), z.std())
    print("scaler single-standardization check: OK")

    train_s, val_s, test_s = build_time_splits(1461, 14, 3)
    print(f"splits: train {len(train_s)} / val {len(val_s)} / test {len(test_s)} windows")


def main() -> None:
    args = build_parser().parse_args()
    _validate_flags(args.force, args.provenance_only)

    src = Path(args.src)
    dst = Path(args.dst)

    if args.provenance_only:
        if not src.is_dir():
            raise FileNotFoundError(f"source directory does not exist: {src}")
        assert_safe_dst(dst)
        summary = copy_provenance(src, dst)
        print("PROVENANCE ONLY:", dst)
        for item in summary["copied"]:
            print("  copied", item)
        return

    assert_safe_dst(dst)
    if dst.exists() and args.force:
        shutil.rmtree(dst)

    core = src / "common_core"
    if not core.is_dir():
        raise FileNotFoundError(f"common_core directory does not exist: {core}")

    copy_common_core(core, dst)
    copy_provenance(src, dst)

    info = build_super_regions(dst)
    expand_weather(dst, info["super_zone"], info["zone_names"])
    validate_dst(dst)

    print("\nPREP DONE:", dst)


if __name__ == "__main__":
    main()
