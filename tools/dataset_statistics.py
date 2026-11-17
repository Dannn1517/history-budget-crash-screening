from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUT_DIR = ROOT_DIR / "analysis_results" / "dataset_stats"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.data.graph_ops import apply_mask_to_edge_index
from utils.data.loading import load_borough_name_order, load_mask, load_nyc_zinb_data
from utils.data.priors import diffuse_risk_time_space


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute dataset statistics for the NYC road-level benchmark.")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--start_date", type=str, default="2021-01-01")
    parser.add_argument("--end_date", type=str, default="2024-12-31")
    parser.add_argument("--diffuse_alpha", type=float, default=0.3)
    parser.add_argument("--diffuse_k", type=int, default=2)
    parser.add_argument("--diffuse_weight", type=float, default=0.5)
    parser.add_argument("--diffuse_norm", type=str, default="none", choices=["row", "none"])
    parser.add_argument("--risk_diffuse_zero_thresh", type=float, default=0.2)
    return parser


def classify_link_feature_columns(columns: List[str]) -> Dict[str, List[str]]:
    road_cols = [
        c
        for c in columns
        if c.startswith("x_highway_")
        or c.startswith("x_access_")
        or c.startswith("x_junction_")
        or c in {
            "x_bridge",
            "x_tunnel",
            "x_oneway_flag",
            "x_width_m",
            "x_maxspeed_kmh",
            "x_lanes_num",
            "x_length_m",
            "x_curvature_log1p",
        }
    ]
    poi_cols = [c for c in columns if c.startswith("x_poi_")]
    socio_cols = [c for c in columns if c.startswith("x_pop_") or c == "m_pop_imputed"]
    topology_cols = [c for c in columns if c.startswith("x_node_")]
    return {
        "road_cols": road_cols,
        "poi_cols": poi_cols,
        "socio_cols": socio_cols,
        "topology_cols": topology_cols,
    }


def format_ratio(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def format_float(x: float) -> str:
    return f"{float(x):.4f}"


def compute_region_stats(
    name: str,
    Y_risk: np.ndarray,
    Y_crash: np.ndarray,
    Y_target: np.ndarray,
    line_graph_edges: int,
) -> dict:
    node_count = int(Y_risk.shape[0])
    accident_node_ratio = float(np.mean(np.any(Y_crash > 0, axis=1))) if node_count > 0 else 0.0
    avg_degree = float((2.0 * float(line_graph_edges)) / float(node_count)) if node_count > 0 else 0.0
    return {
        "region": name,
        "road_segments": node_count,
        "line_graph_edges": int(line_graph_edges),
        "avg_degree": avg_degree,
        "accident_node_ratio": accident_node_ratio,
        "days": int(Y_risk.shape[1]),
        "road_day_observations": int(Y_risk.size),
        "total_crashes": int(np.sum(Y_crash)),
        "event_zero_rate": float(np.mean(Y_risk == 0)),
        "target_zero_rate": float(np.mean(Y_target == 0)),
        "event_mean_risk": float(np.mean(Y_risk)),
        "target_mean_risk": float(np.mean(Y_target)),
    }


def build_source_rows(
    region_name: str,
    node_count: int,
    n_days: int,
    crash_total: int,
    poi_categories: List[str],
    feature_groups: Dict[str, List[str]],
    calendar_dim: int,
    weather_dim: int,
) -> List[dict]:
    road_feature_channels = len(feature_groups["road_cols"]) + len(feature_groups["topology_cols"])
    poi_channels = len(feature_groups["poi_cols"])
    socio_channels = len(feature_groups["socio_cols"])
    return [
        {
            "Regions": region_name,
            "Data": "Crash Counts",
            "Source": "NYC Open Data",
            "Classes": "1",
            "Observation Count": int(crash_total),
            "Count Definition": "Total crash counts within the study period",
        },
        {
            "Regions": region_name,
            "Data": "Point of Interest (road-level aggregated)",
            "Source": "OpenStreetMap via OSMnx",
            "Classes": f"{len(poi_categories)} categories / {poi_channels} channels",
            "Observation Count": int(node_count),
            "Count Definition": "Number of road-link observations carrying POI-derived features",
        },
        {
            "Regions": region_name,
            "Data": "Road-network Features",
            "Source": "OpenStreetMap via OSMnx",
            "Classes": str(road_feature_channels),
            "Observation Count": int(node_count),
            "Count Definition": "Number of road-link observations",
        },
        {
            "Regions": region_name,
            "Data": "Date / Calendar",
            "Source": "Historical calendar data",
            "Classes": str(calendar_dim),
            "Observation Count": int(n_days),
            "Count Definition": "Daily observations",
        },
        {
            "Regions": region_name,
            "Data": "Meteorological Characteristics",
            "Source": "Open-Meteo",
            "Classes": str(weather_dim),
            "Observation Count": int(n_days),
            "Count Definition": "Daily borough-level observations",
        },
        {
            "Regions": region_name,
            "Data": "Socio-demographic Characteristics",
            "Source": "NYC Open Data",
            "Classes": str(socio_channels),
            "Observation Count": int(node_count),
            "Count Definition": "Number of road-link observations carrying socio-demographic features",
        },
    ]


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    dataset_df: pd.DataFrame,
    source_df: pd.DataFrame,
    poi_categories: List[str],
    feature_groups: Dict[str, List[str]],
) -> None:
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        headers = [str(c) for c in df.columns]
        rows = [[str(v) for v in row] for row in df.itertuples(index=False, name=None)]
        sep = ["---"] * len(headers)
        table = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        table.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(table)

    md_dataset = dataset_df.copy()
    md_dataset["avg_degree"] = md_dataset["avg_degree"].map(format_float)
    md_dataset["accident_node_ratio"] = md_dataset["accident_node_ratio"].map(format_ratio)
    md_dataset["event_zero_rate"] = md_dataset["event_zero_rate"].map(format_ratio)
    md_dataset["target_zero_rate"] = md_dataset["target_zero_rate"].map(format_ratio)
    md_dataset["event_mean_risk"] = md_dataset["event_mean_risk"].map(format_float)
    md_dataset["target_mean_risk"] = md_dataset["target_mean_risk"].map(format_float)
    lines = [
        "# Dataset Statistics",
        "",
        "## 统计配置",
        "",
        f"- 时间范围：{args.start_date} ~ {args.end_date}",
        "- 风险定义：`raw risk = crash_count + 2 * injured + 3 * killed`",
        (
            f"- 风险扩散参数：alpha={args.diffuse_alpha}, k={args.diffuse_k}, "
            f"hop_weight={args.diffuse_weight}, normalize={args.diffuse_norm}, "
            f"zero_thresh={args.risk_diffuse_zero_thresh}"
        ),
        "",
        "## 表 A. 数据集与任务统计",
        "",
        dataframe_to_markdown(md_dataset),
        "",
        "## 表 B. 数据来源与观测统计（按 borough）",
        "",
        dataframe_to_markdown(source_df),
        "",
        "## 注释",
        "",
        "- `event-space` 对应原始风险场 `raw risk`。",
        "- `target-space` 对应按当前默认参数构造的 `risk diffuse`。",
        "- `Point of Interest` 与 `Socio-demographic Characteristics` 已预聚合到 road-link 级别，因此其 `Observation Count` 记为承载该模态的路段观测数。",
        (
            f"- POI 类别包括：{', '.join(poi_categories)}。"
            f" 当前 POI 派生特征总通道数为 {len(feature_groups['poi_cols'])}。"
        ),
        (
            f"- 路网特征总通道数为 {len(feature_groups['road_cols']) + len(feature_groups['topology_cols'])}，"
            f"其中链路属性 {len(feature_groups['road_cols'])} 个，端点/拓扑属性 {len(feature_groups['topology_cols'])} 个。"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_static, Y_risk_full, Y_crash_full, calendar, weather, borough_id_arr, dates = load_nyc_zinb_data(
        str(data_dir),
        pd.to_datetime(args.start_date),
        pd.to_datetime(args.end_date),
    )
    edge_index_full = np.load(data_dir / "line_graph" / "edge_index.npy")
    borough_names = load_borough_name_order(str(data_dir))

    Y_target_full = diffuse_risk_time_space(
        Y_risk_full,
        edge_index_full,
        alpha=args.diffuse_alpha,
        k=args.diffuse_k,
        hop_weight=args.diffuse_weight,
        undirected=True,
        normalize=args.diffuse_norm,
        zero_thresh=args.risk_diffuse_zero_thresh,
    ).astype(np.float32)

    links_df = pd.read_parquet(data_dir / "links_with_features.parquet")
    feature_groups = classify_link_feature_columns(links_df.columns.tolist())
    calendar_dim = int(sum(c.startswith("x_cal_") for c in calendar.shape and pd.read_parquet(data_dir / "calendar" / "calendar_day.parquet").columns))
    calendar_df = pd.read_parquet(data_dir / "calendar" / "calendar_day.parquet")
    calendar_dim = int(sum(c.startswith("x_cal_") for c in calendar_df.columns))
    weather_df = pd.read_parquet(data_dir / "weather" / "nyc_weather_daily.parquet")
    weather_dim = int(sum(c.startswith("x_wth_") for c in weather_df.columns))
    poi_categories = sorted(
        {
            c.replace("x_poi_count_", "").replace("x_poi_logcount_", "").replace("x_poi_per_km_", "")
            for c in feature_groups["poi_cols"]
        }
    )

    dataset_rows = [
        compute_region_stats(
            "All NYC",
            Y_risk_full,
            Y_crash_full,
            Y_target_full,
            int(edge_index_full.shape[1]),
        )
    ]
    source_rows: List[dict] = []
    num_nodes = int(Y_risk_full.shape[0])
    n_days = int(Y_risk_full.shape[1])
    for borough_name in borough_names:
        mask = load_mask(str(data_dir), borough_name, "", num_nodes)
        idx = np.where(mask)[0]
        edge_index_sub = apply_mask_to_edge_index(edge_index_full, mask)
        Y_risk_sub = Y_risk_full[idx].copy()
        Y_crash_sub = Y_crash_full[idx].copy()
        Y_target_sub = diffuse_risk_time_space(
            Y_risk_sub,
            edge_index_sub,
            alpha=args.diffuse_alpha,
            k=args.diffuse_k,
            hop_weight=args.diffuse_weight,
            undirected=True,
            normalize=args.diffuse_norm,
            zero_thresh=args.risk_diffuse_zero_thresh,
        ).astype(np.float32)
        stats_row = compute_region_stats(
            borough_name,
            Y_risk_sub,
            Y_crash_sub,
            Y_target_sub,
            int(edge_index_sub.shape[1]),
        )
        dataset_rows.append(stats_row)
        source_rows.extend(
            build_source_rows(
                region_name=borough_name,
                node_count=int(idx.size),
                n_days=n_days,
                crash_total=stats_row["total_crashes"],
                poi_categories=poi_categories,
                feature_groups=feature_groups,
                calendar_dim=calendar_dim,
                weather_dim=weather_dim,
            )
        )

    dataset_df = pd.DataFrame(dataset_rows)
    source_df = pd.DataFrame(source_rows)

    totals_ok = int(dataset_df.loc[dataset_df["region"] != "All NYC", "road_segments"].sum()) == int(
        dataset_df.loc[dataset_df["region"] == "All NYC", "road_segments"].iloc[0]
    )
    crash_totals_ok = int(dataset_df.loc[dataset_df["region"] != "All NYC", "total_crashes"].sum()) == int(
        dataset_df.loc[dataset_df["region"] == "All NYC", "total_crashes"].iloc[0]
    )

    metadata = {
        "time_range": {"start_date": args.start_date, "end_date": args.end_date, "n_days": n_days},
        "diffusion": {
            "alpha": args.diffuse_alpha,
            "k": args.diffuse_k,
            "hop_weight": args.diffuse_weight,
            "normalize": args.diffuse_norm,
            "zero_thresh": args.risk_diffuse_zero_thresh,
        },
        "sources": {
            "roads": "OpenStreetMap via OSMnx",
            "poi": "OpenStreetMap via OSMnx",
            "calendar": "Historical calendar data",
            "weather": "Open-Meteo",
            "collisions": "NYC Open Data",
            "socio_demographics": "NYC Open Data",
        },
        "feature_groups": {
            "road_link_channels": len(feature_groups["road_cols"]),
            "road_topology_channels": len(feature_groups["topology_cols"]),
            "road_total_channels": len(feature_groups["road_cols"]) + len(feature_groups["topology_cols"]),
            "poi_channels": len(feature_groups["poi_cols"]),
            "poi_categories": poi_categories,
            "calendar_channels": calendar_dim,
            "weather_channels": weather_dim,
            "socio_channels": len(feature_groups["socio_cols"]),
        },
        "consistency_checks": {
            "borough_road_segments_sum_to_all_city": totals_ok,
            "borough_total_crashes_sum_to_all_city": crash_totals_ok,
        },
    }

    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "dataset_statistics": dataset_df.to_dict(orient="records"),
                "source_statistics": source_df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    dataset_df.to_csv(out_dir / "dataset_statistics.csv", index=False, encoding="utf-8")
    source_df.to_csv(out_dir / "data_sources_by_borough.csv", index=False, encoding="utf-8")
    write_markdown(
        out_dir / "dataset_statistics.md",
        args=args,
        dataset_df=dataset_df,
        source_df=source_df,
        poi_categories=poi_categories,
        feature_groups=feature_groups,
    )

    print(f"Output directory: {out_dir}")
    print("Dataset statistics:")
    print(dataset_df.to_string(index=False))
    print("\nConsistency checks:")
    print(json.dumps(metadata["consistency_checks"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
