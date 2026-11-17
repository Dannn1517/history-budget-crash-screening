# Chicago 数据集（data_chicago/）

> 主键 `id_geom_hash`（无向物理路段，UTM16N WKT 0.001m 方向不变 SHA-1）× `id_date`（'YYYY-MM-DD' 字符串）。
> 时间窗 2021-01-01 ~ 2024-12-31（1,461 天）；建模域 = `m_analysis_eligible=1` 的 **43,335** 条链路
> （= `common_core/train_keys_all.parquet` = line graph 节点）。

## 两个视图

### common_core/（跨城主视图，与 NYC common_core 同构）
| 文件 | 内容 |
|---|---|
| `links_meta.parquet` | 完整路网 49,812 条（审计用）：id_/meta_/geo_ 列 + 6 个 eligibility 原因列 |
| `links_with_features.parquet` | **eligible 43,335 条 × 70 固定列**（附录 A 顺序；RAW 连续值） |
| `links_with_features_raw.parquet` | 同上（单一标准化契约下两者一致；保留显式 raw 档） |
| `links_features_scaler.json` | 连续 34 列的 mean/std（在全部 eligible=training keys 上拟合）；**loader 只标准化一次** |
| `train_keys_all.parquet` | eligible 链路 geom_hash（与 NYC 同名同构） |
| `crashes_events.parquet` | 403,252 条事件（含链路映射、snap 距离、fatal/nonfatal/pdo） |
| `crashes_daily_labels.parquet` | 394,400 个非零 road-day：y_crash_count/y_injured_sum/y_killed_sum/y_pdo_count/y_risk + meta |
| `calendar/calendar_day.parquet` | 1,461 天 × 11 列（dow 0=Sun、weekend、US federal holiday、doy sin/cos） |
| `weather/chicago_weather_daily.parquet` | 2 zone（ohare/midway）× 1,461 天 × 20 个 x_wth_ 通道 + m_wth_imputed |
| `weather/weather_meta.json` / `weather_code_mapping.json` | 口径与 weather_code 索引映射 |
| `line_graph/` | node_id_map（按 hash 升序）、edge_index/weight（116,262 边）、region_id(1-77 CA)、weather_zone_id(0/1)、mask_region_01..77.npz、graph_meta.json（相对路径） |

### chicago_full/（数据丰富版，不与 NYC common_core 混作公平对照）
所有文件为稀疏 road×day（仅非零行；缺失=0），主键对齐 common_core：
| 文件 | 特征列 |
|---|---|
| `links_daily_construction.parquet` | x_construction_{active,count_1d,count_7d,neighbor_count_7d} |
| `links_daily_excavation.parquet` | x_excavation_{active,count_1d,count_7d,neighbor_count_7d} |
| `links_daily_311.parquet` | x_311_{roadissue,pothole,signal,streetlight}_{1d,7d 按列} + streetcond/sign/sidewalk/snowice_7d + neighbor_roadissue_7d（13 列） |
| `links_daily_mobility.parquet` | x_mobility_{pickups_1d,dropoffs_1d,trip_count_7d,change_1d}（CA 级 proxy 广播到链路；dense） |
| `links_daily_traffic.parquet` | x_traffic_volume_{1d,log1p,mean_7d,change_1d} + m_traffic_{observed,imputed}_1d（dense；ADT 实际观测 2022-07-28 起，此前全为分层中位数填补且 mask=0） |

### auxiliary/（不进模型）
- `traffic_count_segment_link_map.parquet`：ADT segment×direction → link 匹配（重叠长度）
- `traffic_tracker_segment_day.parquet`：segment×日 速度/公交计数留档（2023-10 ~ 2024-02 结构性缺口）

## 标签与风险

- `y_risk = y_crash_count + 2*y_injured_sum + 3*y_killed_sum`（便捷审计列；训练端应从计数重构）
- `y_injured_sum` = 非致命受伤（Chicago `injuries_total` 含死亡，已减 fatal）
- 缺失 road-day 标签语义为 0
- 日期切分：train 2021-01-15~2023-09-01 / val 2023-09-02~2024-05-01 / test 2024-05-02~2024-12-31；T=14, H=3

## 防时间泄漏

所有动态特征只用 `<= t` 信息：permit 需 issued<=t 且区间覆盖 t；dig 有效开始 = max(requestdate, digdate)；
311 只用 created_date；滚动窗 [t-6,t]；矩阵在 2020-12-01 起预热网格上计算后切片导出。
抽查重算见 `qa/temporal_leakage.json`（construction/excavation/311/traffic 各 300 样本 0 失配）。

## 标准化契约（与修复后 NYC 管线一致）

`links_with_features.parquet` 存 **RAW** 连续值；loader 读取 `links_features_scaler.json` 后
**只标准化一次**（one-hot/flag/mask 不标准化）。NYC→Chicago zero-shot 时对 Chicago raw 列应用 NYC scaler。

## 与 NYC 数据集的列级对齐

- 静态 70 列名与顺序完全一致（附录 A）；未见类别补 0/进 other，词表冻结
- `crashes_daily_labels` 列集合与 NYC 相同（y_* + meta_*）
- 日历列同构；天气通道同为 20 个 x_wth_（NYC 按 5 borough，Chicago 按 2 weather zone，
  经 `line_graph/weather_zone_id.npy` 映射到链路）
- 动态特征列名与 NYC expansion 完全一致（x_construction_*/x_311_*/x_mobility_*），
  Chicago 额外提供 x_excavation_* 与 x_traffic_volume_*
