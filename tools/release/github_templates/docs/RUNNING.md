# 运行与复现

## 无权重示例

```bash
python examples/run_pipeline.py --demo
```

CPU 上运行小型随机 LTWM/scorer，完成 H4 前向计算。没有数据下载、checkpoint 下载或硬件访问。

## 正式离线输入

`configs/inference.example.json` 中的路径相对于**配置文件所在目录**解析。复制为 `inference.local.json` 后调整路径。支持绝对路径。

输入 NPZ 的三个键和形状：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| `history_tokens` | `[B,4,256,384]` | 由冻结 DINOv3 + 项目适配器产生的历史 tokens，时间从早到晚 |
| `goal_tokens` | `[B,256,384]` | 同一编码链得到的目标图像 tokens |
| `candidate_actions` | `[B,K,4,3]` | K 个候选的四步局部连续 `[dx,dy,dyaw]` 动作 |

存储为 float16 或 float32；入口转换为 float32。动作坐标、时间间隔和尺度必须遵循产生权重时的协议，不要直接把速度指令当作这里的位移。输入必须来自同一套编码器/adapter 和标定，不能填入其他视觉特征。

已有数组可写为：

```python
np.savez_compressed("inputs.npz", history_tokens=history,
                    goal_tokens=goal, candidate_actions=actions)
```

## 最新三域训练入口

```bash
python tools/tridomain_v71/build_tridomain_split.py --help
python tools/tridomain_v71/encode_dinov3_adapted_tridomain.py --help
python tools/tridomain_v71/train_litetokenformer_wm_v2_tridomain.py --help
python tools/tridomain_v70/train_scorer_v2_tridomain.py --help
python tools/tridomain_v70/train_proposal_aware_cem_risk_tridomain.py --help
```

准备与 `docs/LATEST_VERSION.md` 所述三域协议匹配的划分、数据和编码器。下面双域命令是旧实验的可复现参考，不是当前默认训练配置。

## 原双域数据编码和训练参考

下面所有命令在仓库根目录执行。`--help` 查看参数；先安装对应可选依赖。

```bash
python tools/recon/process_scand_bags_rosbags.py --help
python tools/recon/encode_dinov3_adapted.py --help
python tools/recon/train_litetokenformer_wm_v2_4ctx.py --help
python tools/recon/eval_litetokenformer_wm_v2_4ctx.py --help
python tools/scoring/cache_trajectory_quality_features_v2.py --help
python tools/scoring/train_trajectory_quality_scorer_v2.py --help
```

编码示例（先准备真实数据和 adapter）：

```bash
python tools/recon/encode_dinov3_adapted.py \
  --trajectory-root datasets/recon \
  --output-root latents/recon \
  --strict-splits data/split_lock/recon_strict_splits.json \
  --split-lock data/split_lock/split_lock.json \
  --dataset recon \
  --model-dir external/dinov3-vitl16-pretrain-lvd1689m \
  --adapter-checkpoint checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt \
  --adapter-parent-split-lock PATH_TO_ADAPTER_PARENT_SPLIT_LOCK.json \
  --include-splits train,val --device cuda
```

原 adapter 的 parent split lock 与扩展划分不同；parent lock 必须另行提供并通过验证，不能关闭验证来冒充正式复现。冻结划分元数据在 `data/split_lock/`；RGB、ROS bags、latent shards 均需另备。

训练 H4 LTWM（完整训练参数依原实验记录配置）：

```bash
python tools/recon/train_litetokenformer_wm_v2_4ctx.py \
  --latent-root latents/recon --scand-latent-root latents/scand \
  --split-lock data/split_lock/split_lock.json \
  --history-frames 4 --future-frames 4 \
  --checkpoint-dir checkpoints/local/ltwm --report-dir outputs/ltwm \
  --seed 1701
```

scorer 训练需先固定 proposal，再独立生成 pose 标签、缓存特征，并按 train-fit/scorer-cal 等严格划分训练；单独执行训练脚本不会自动生成上游数据。见 `docs/PROTOCOL.md`。

## v70 与历史实验

`tools/tridomain_v70/run_tridomain_cem_v70_archive_replay.py` 是三域研究入口。
运行前必须配置环境变量，完整条件见 `docs/LATEST_VERSION.md`。
`tools/planning/run_tridomain_cem_tail_v70_runtime_v2.py` 是部署使用的辅助模块。
需另备对应 selector/Arm-D 权重、support envelopes、calibration/training result 和 proposal 数据。
本代码包排除了所有 `.npz/.npy` 数组，包括这些学习到的 support 资产。

`tools/release/build_v70_*` 为原历史资产包构建器，可能打包权重且依赖旧服务器文件。
本次交付由 `tools/release/build_github_source_release.py` 构建，专门排除权重。
`docs/LOCAL_ASSETS.md` 的原始内容保存在 `docs/archive/`，当前入口见本文件。

## 实机机器狗

最终代码放在 `robot/`。当前仅预留 README；作者填入实现和设备依赖后，再补充设备型号、通信配置、传感器映射与启动步骤。
现有 `deploy/v70_go2_orin/` 保留为历史研究参考，其 README 描述的是原始含权重设备包，并非当前源码包内容。
