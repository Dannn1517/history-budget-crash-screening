# Chicago 数据集最终验收 (final_acceptance.md)

- 门槛总数: 30 | PASS: 30 | FAIL: 0
- 总体结论: **PASS**

| 门槛 | 结果 | 详情 |
|---|---|---|
| 6.1 manifest 源记录完整 | PASS | missing=[] incomplete=[] |
| 6.1 id_geom_hash 全表唯一 | PASS | n=49,812 |
| 6.1 node_idx 连续且按 hash 升序一一对应 | PASS | n=43,335 |
| 6.1 road-day 文件主键唯一且属于 common core | PASS |  |
| 6.2 eligible 几何 100% 有效 | PASS |  |
| 6.2 eligible 质心 100% 在边界内 | PASS | outside=0 |
| 6.2 eligible 无 private/no access | PASS |  |
| 6.2 eligible 无 motorway/motorway_link | PASS |  |
| 6.2 centerline 20m 长度保留率 >= 90% | PASS | 99.31% |
| 6.2 line graph isolated < 0.5% | PASS | 0.127% |
| 6.3 下载总数 >= 400,000 | PASS | 439,991 |
| 6.3 经纬度完整率 >= 98.5% | PASS | 98.86% |
| 6.3 75m 保留率 >= 95% (vs 有效+非私人+界内) | PASS | 96.61% |
| 6.3 标签非负整数 | PASS |  |
| 6.3 road-day 汇总 == 事件表总数 | PASS |  |
| 6.3 y_risk 公式正确 | PASS |  |
| 6.3 y_injured_sum 不含 fatality (nonfatal>=0 且 = total-fatal) | PASS |  |
| 6.3 labels 主键唯一 | PASS |  |
| 6.4 静态列名/列序/数量与附录 A 一致 | PASS | got 71 cols, expect 71 |
| 6.4 静态特征无 NaN/Inf | PASS |  |
| 6.4 主特征文件为 RAW（未预标准化） | PASS |  |
| 6.4 calendar 每日完整 | PASS |  |
| 6.4 weather 双 zone 每日完整 | PASS | zones=2 |
| 6.4 每 link 恰有一个 region/weather zone | PASS |  |
| 6.4 特征清单已导出 | PASS | core=70, full=31 |
| 6.4 时间泄漏抽查（重算一致 + rolling 单测） | PASS | {"rolling_boundary_unit_test": true, "construction_recheck_bad": 0, "excavation_recheck_bad": 0, "sr311_recheck_bad": 0, |
| 6.4 动态源统计已导出 | PASS |  |
| 6.5 smoke: 100 links x 60 days 窗口/图batch/前向反向 | PASS | windows=44, sub_edges=413 |
| 6.5 日期切分无重叠且覆盖有效 | PASS | train=960, val=243, test=244 |
| 6.5 全量 1 epoch dry-run（标签索引） | PASS | {'train': 257826, 'val': 64392, 'test': 69506} |