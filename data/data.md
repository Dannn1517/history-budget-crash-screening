道路以"id_geom_hash"为主键

# data/crashes_daily_labels.parquet
按道路-日期聚合的事故数、受伤数、死亡数、PDO 数和匹配审计字段。

- 正式实验在加载时使用 `y_crash_count + 2*y_injured_sum + 3*y_killed_sum` 重构 event-space risk。
- `y_risk` 是旧流程遗留字段，当前训练与评估均不读取；不要直接用于论文公式或新实验。

## Target-space 标签提醒

- 默认参数：`alpha=0.3`、`k=2`、`hop_weight=0.5`、`zero_thresh=0.2`。
- 空间扩散使用无自环的二值 line-graph 邻接矩阵，保留 0-hop 原信号，并除以 hop 权重和；默认 `normalize=none`，不做度归一化。

# data/links_meta.parquet
道路meta元信息 辅助可视化

# data/links_with_features.parquet
路网特征（node节点特征、links链路特征、poi特征、人口普查分布特征等）

# data/train_keys_all.parquet
geom_hash




## 静态特征缩放契约（2026-09-11 起）

`links_with_features.parquet` 存储**原始未标准化**的连续特征值；
`links_features_scaler.json` 记录训练键上的 mean/std；
唯一一次标准化由训练端 loader（`utils/data/loading.py: load_nyc_zinb_data`）执行。
指示变量列（x_highway_*、x_access_*、x_bridge、x_tunnel、x_oneway_flag、节点指示位）与 m_* 掩码列不进 scaler。
2026-04-20 之前的历史导出违反此契约（导出端已 z-score、loader 又标准化一次）；
本仓库发布的数据均为按此契约修复后的导出。
