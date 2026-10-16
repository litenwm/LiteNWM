# 当前发布版本

源码直接复制本工作区当前文件，不从旧 ZIP、旧 Git commit 或服务器导出副本还原。
本包的 `manifests/source_release.json` 固定每个交付文件的实际哈希。

## 默认模型

已有封存记录的三域 V70 模型：RECON + SCAND + SACSoN。
对应原始导出 `v70_tridomain_latest_20260729`，以及匹配的三域 ATE-primary scorer。
统一离线入口默认要求 seed1701，完整研究管线采用 1701/1702/1703 三成员。
模型哈希在 `manifests/latest_weights.json`；权重文件不在发布包中。

此版本 split-lock SHA-256：

```text
66294673129bd20355e3aef55757bdf47bd915b54881c8c3ccfb01e057ca5614
```

`data/split_lock/` 中旧双域锁不能作为上述三域锁使用。正式三域复现须另备匹配的原始锁与数据元信息。

## 最新代码入口

| 用途 | 当前代码 |
| --- | --- |
| 安装和离线评分 | `examples/run_pipeline.py` |
| 三域公平 V70 archive replay | `tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py` |
| 三域 scorer 训练 | `tools/tridomain_v70/train_scorer_v2_tridomain.py` |
| 三域 selector 训练 | `tools/tridomain_v70/train_proposal_aware_cem_risk_tridomain.py` |
| 三域划分、编码、世界模型训练 | `tools/tridomain_v71/` |
| 常驻加载与规划参考 | `deploy/v70_go2_orin/v70_persistent_planner.py` |
| 单目标实机历史入口 | `deploy/v70_go2_orin/run_v70_single_goal.bash` |
| 当前无动作复用实机发布入口 | `robot/run_go2.bash`，默认 `v70_live_nav_fast.py` |

`v70_persistent_planner.py` 的实际依赖仍为 `v70_cem_tridomain_v4_shadow.py`；不因为文件名有 v5/v6 就随意改换调用链。原始部署 amendment 仅是历史记录，实际源码哈希以本次文件清单为准。

三域 selector 的原封存状态为 `RELAXED_FEASIBLE_V1`：RECON/SCAND 严格校准通过，SACSoN 采用原先固定的放宽合同。本次整理未重新训练或改变模型/选择规则。

## 三域研究脚本运行条件

archive replay 在导入时读取以下环境变量，必须先准备对应外部资产：

```bash
export TRIDOMAIN_V70_RESULT=/absolute/path/result.json
export TRIDOMAIN_V70_PROTOCOL_PREREG=/absolute/path/fair_eval_protocol_preregistration_v1.json
export TRIDOMAIN_V70_CHECKPOINT_ROOT=/absolute/path/feature_checkpoint_root
export TRIDOMAIN_RUNTIME_DATASET=scand
python tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py --help
```

校准记录、seal 与 preregistration 之间有哈希绑定，移动路径后需按原程序正确重建绑定，不能只改其中一个 JSON。旧脚本 `tools/prepare_local_tridomain_v70_runtime.py` 保留作迁移逻辑参考，含原机器路径，运行前必须配置。

公开机器狗分支已补齐，常驻 planner 为 `v70_persistent_planner_fast.py`，`robot/assets_manifest.json` 绑定同一三域模型哈希。执行策略为每个新规划最多一次 H1，详见 `robot/OPTIMIZATIONS.md`，不能沿用旧实机执行性能结论。
