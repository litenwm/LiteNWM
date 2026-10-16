# Universal dual-domain raw scorer v2：训练与严格复核结果

## 结论

`universal_dualdomain_raw_v2` 已按固定 K17、CEM=false 的协议完成全量训练与复核。主模型只使用 action、DINOv3 static、raw LTWM rollout/epistemic、spacing 和 domain 信息；不把 scorer-v2 的 ATE/RTP 预测混入主特征。

完整 D 臂在第二个 RECON fresh H4-100 确认集上通过全部部署门槛，但在两个完全 route-disjoint fresh H4-100 合并后的 200 窗口统计上没有通过安全门。因此当前证据支持“显著提高轨迹质量上限”，不支持“已经得到跨 route、跨域可部署的 0.90 dual precision scorer”。SCAND 目前只有历史上已打开标签的 spent-development H4-100，没有可称为 fresh outer 的独立 route/bag 池。

## 方法名称

- A，legacy balanced support+raw：旧的 3,024-state 双域模型；support 只作尺度感知 OOD guard，raw scorer 决定动作质量。
- B，full shared raw：使用完整 12,781 个 train-fit state，共享 raw trunk，并使用单一风险门。
- C，domain-conditioned raw：B 加同一 checkpoint 内的 domain-conditioned normalization/outcome heads，以及 RECON/SCAND 各自 scorer-cal 风险门。
- D，setwise/listwise risk：C 加 K17 setwise/listwise ranking、异方差/ensemble uncertainty 和保守 dual/harm gate；这是当前主方案。
- native scorer-v2：冻结的三 seed scorer-v2 贪心 fixed-pool 基线。
- safe-rank：后续 dual-safe 排序微调；跨域结果不稳定，保留为失败消融，不作为主方案。

## RECON 第二个 fresh H4-100 确认集

25 条此前从未使用的 route，每条 4 个窗口。所有方法的 `selected_index` 均在首次打开未来位姿前冻结。表中每个误差格为 mean / p90，单位为米。

| 方法 | Promotions | Coverage | Dual precision | Wilson lower | Harm | ATE | RPE | Endpoint | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent | 0 | 0.0000 | — | — | — | 0.1125 / 0.1940 | 0.0539 / 0.0902 | 0.1643 / 0.3002 | 0.1436 / 0.2563 |
| Native scorer-v2 | 98 | 0.9800 | 0.5408 | 0.4425 | 0.4592 | 0.1063 / 0.1708 | 0.0504 / 0.0740 | 0.1515 / 0.2585 | 0.1334 / 0.2214 |
| A: legacy balanced support+raw | 69 | 0.6900 | 0.8116 | 0.7039 | 0.1884 | 0.0968 / 0.1664 | 0.0477 / 0.0805 | 0.1391 / 0.2537 | 0.1222 / 0.2176 |
| B: full shared raw | 88 | 0.8800 | 0.8636 | 0.7766 | 0.1364 | 0.0958 / 0.1666 | 0.0476 / 0.0780 | 0.1374 / 0.2566 | 0.1208 / 0.2247 |
| C: domain-conditioned raw | 92 | 0.9200 | 0.8152 | 0.7239 | 0.1848 | 0.0965 / 0.1664 | 0.0479 / 0.0753 | 0.1382 / 0.2646 | 0.1216 / 0.2251 |
| **D: setwise/listwise risk** | **57** | **0.5700** | **0.9123** | **0.8106** | **0.0877** | **0.0879 / 0.1416** | **0.0441 / 0.0683** | **0.1245 / 0.2205** | **0.1099 / 0.1919** |
| Post-hoc pool oracle | 100 | 1.0000 | 1.0000 | 0.9630 | 0.0000 | 0.0703 / 0.1294 | 0.0396 / 0.0611 | 0.0952 / 0.1970 | 0.0852 / 0.1719 |

D 在该批次满足：dual ≥ 0.90、Wilson ≥ 0.80、coverage ≥ 0.40、harm ≤ 0.10，且四项 mean 和 p90 都优于 incumbent 与 native scorer-v2。

## 两个 RECON fresh H4-100 合并的稳健性统计

两个 cohort 之间 record、trajectory、route 均零重叠；B/C/D/native/safe-rank 都在各自标签打开前冻结。A 只在第二批预冻结，因此不放入 pooled-200，避免事后补算冒充严格结果。

| 方法 | Promotions | Coverage | Dual precision | Wilson lower | Harm | ATE mean / p90 | RPE mean / p90 | Endpoint mean / p90 | Objective mean / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent | 0 | 0.0000 | — | — | — | 0.1086 / 0.1987 | 0.0528 / 0.0893 | 0.1583 / 0.3026 | 0.1384 / 0.2613 |
| Native scorer-v2 | 193 | 0.9650 | 0.5699 | 0.4994 | 0.4301 | 0.0988 / 0.1728 | 0.0482 / 0.0740 | 0.1404 / 0.2585 | 0.1238 / 0.2214 |
| B: full shared raw | 184 | 0.9200 | 0.8152 | 0.7529 | 0.1848 | 0.0927 / 0.1681 | 0.0467 / 0.0752 | 0.1330 / 0.2547 | 0.1169 / 0.2196 |
| C: domain-conditioned raw | 190 | 0.9500 | 0.8053 | 0.7431 | 0.1947 | 0.0926 / 0.1680 | 0.0466 / 0.0733 | 0.1321 / 0.2543 | 0.1163 / 0.2197 |
| **D: setwise/listwise risk** | **108** | **0.5400** | **0.8426** | **0.7623** | **0.1574** | **0.0888 / 0.1516** | **0.0446 / 0.0711** | **0.1264 / 0.2405** | **0.1114 / 0.2048** |
| Safe-rank failed ablation | 117 | 0.5850 | 0.8462 | 0.7699 | 0.1538 | 0.0868 / 0.1548 | 0.0440 / 0.0717 | 0.1236 / 0.2405 | 0.1088 / 0.2053 |

D 相对 incumbent 的 route-clustered paired-bootstrap 95% CI：ATE −0.0266 到 −0.0137 m，RPE −0.0113 到 −0.0056 m，Endpoint −0.0429 到 −0.0222 m，Objective −0.0364 到 −0.0189 m。四项质量增益均稳定小于 0，但安全门未通过。

## SCAND H4-100 spent-development 复核

该集合的 outcomes 在更早实验中已打开，只能作为 spent-development replay。没有使用它重新训练、调阈值或早停。

| 方法 | Promotions | Coverage | Dual precision | Wilson lower | Harm | ATE mean / p90 | RPE mean / p90 | Endpoint mean / p90 | Objective mean / p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Incumbent | 0 | 0.0000 | — | — | — | 0.2303 / 0.5807 | 0.1099 / 0.2515 | 0.3302 / 0.8419 | 0.2903 / 0.7339 |
| Native scorer-v2 | 6 | 0.0600 | 0.6667 | 0.3000 | 0.3333 | 0.2262 / 0.5807 | 0.1087 / 0.2515 | 0.3237 / 0.8419 | 0.2847 / 0.7339 |
| A: legacy balanced support+raw | 53 | 0.5300 | 0.7358 | 0.6042 | 0.2642 | 0.2141 / 0.5525 | 0.1053 / 0.2263 | 0.3055 / 0.8025 | 0.2690 / 0.6980 |
| B: full shared raw | 86 | 0.8600 | 0.7209 | 0.6183 | 0.2791 | 0.2125 / 0.5368 | 0.1053 / 0.2281 | 0.3009 / 0.8440 | 0.2655 / 0.7066 |
| C: domain-conditioned raw | 76 | 0.7600 | 0.7500 | 0.6422 | 0.2500 | 0.2141 / 0.5257 | 0.1055 / 0.2340 | 0.3046 / 0.7831 | 0.2684 / 0.6724 |
| **D: setwise/listwise risk** | **88** | **0.8800** | **0.7159** | **0.6142** | **0.2841** | **0.2044 / 0.4821** | **0.1040 / 0.2154** | **0.2891 / 0.7483** | **0.2552 / 0.6470** |

D 给出最好的 learned mean/p90，但 SCAND dual/harm 明显不合格。由于没有未使用且 route/bag 隔离的 SCAND 外测池，不能把该表称为 fresh outer。

## 数据、模型与协议审计

- 完整 train-fit：RECON 5,292 + SCAND 7,489 = 12,781 states；K17 共 217,277 candidates。
- scorer-cal：RECON 410 + SCAND 1,213；按 route/family 做 group-disjoint cross-fitting。
- 主模型 raw 输入维度 815；scorer-v2 ATE/RTP 仅作为独立 native baseline，不进入 B/C/D 主输入。
- RECON 与 SCAND spacing 分别为 0.25 m 和 0.38 m；support 只作物理 guard/OOD 判断。
- 两批 RECON fresh manifest 都来自 split-lock test，且只用 RGB 文件名选取；候选、特征、模型、support、policy、selected_index 均在未来位姿前冻结。
- 全程 fixed-pool only，CEM=false，cem-cal 未打开。

## 关键产物

- 训练结果：`outputs/universal_dualdomain_raw_v2/training_v2/result.json`
- 第二批严格 RECON 结果：`outputs/universal_dualdomain_raw_v2/recon_fresh_outer_confirm_v2/result.json`
- RECON pooled-200：`outputs/universal_dualdomain_raw_v2/recon_fresh_outer_pooled_v2.json`
- SCAND spent replay：`outputs/universal_dualdomain_raw_v2/scand_spent_replay_v2/result.json`

关键 SHA：training result `f898c4bb8b8261a87906e170133bede3475ae9ec69267fc70344b154ebbb3f3c`；第二批 manifest `3869547dcea03a23942b3087bd473d036efe977a9cc358ad1fd7bae524d9e6b4`；K17 pool `9d9addc94e2a040dcfa9a30861388817e50322eb8a796bf740e10d515e7b447f`；label-free features `1c3edc8cbfa49b77974090c839c86c1b549ba83ead042e738f90ed5a76455202`；prelabel selected_index `774d3bdbb077230e4a53788026e9cb7916531bdae093b2b8936d199f2bc307d3`；pooled result `f592f4b2a1245a99fea95bc25dd3505c924ad464fc76a144b17366a61bec0411`。

## 当前判断与下一步

当前瓶颈不是“缺少 latent”：D 已经包含三 seed raw LTWM rollout/epistemic latent statistics。真正的瓶颈是跨 route、尤其 SCAND 的 challenger 排序与风险校准稳定性。CEM 只会扩大候选分布，现阶段会放大 gate 风险，仍不应开启。

下一阶段应优先做 route-robust/domain-shift calibration：扩大完全未使用的 SCAND route/bag outer 资源；在不查看 outer 标签的条件下改进 group-DRO、route-level uncertainty 和 coverage-constrained conformal gate；然后重新冻结新的双域 outer，再决定是否启用 CEM。
