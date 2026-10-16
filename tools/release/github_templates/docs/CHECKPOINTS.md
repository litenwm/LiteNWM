# 权重准备

此发布包不含项目权重、外部权重、训练状态或学习得到的 support 数组。
源码不能自动还原已训练模型的数值。需要作者另行提供权重下载渠道或自行训练。

离线评分入口需要两个匹配的项目 checkpoint：

| 文件 | checkpoint 中必须存在的键 |
| --- | --- |
| H4 LTWM `best.pt` | `config`（`horizon=4`）、`model`、`split_lock_sha256` |
| scorer-v2 `best.pt` | `model_config`、`model`、`split_lock_sha256` |

当前默认配置对应已封存的最新三域模型，接收方本地放置路径：

```text
checkpoints/latest/ltwm_expansion/seed1701/best.pt
checkpoints/latest/scorer_v2/seed1701/best.pt
```

正式比较应使用同版本、同种子、同数据划分的模型；不能混用 H6、highres-dual 或其他 scorer 架构。统一入口会严格加载 state dict 并检查 split-lock 一致性。

RGB 编码还需要冻结的 DINOv3-L/16 与适配器：

```text
external/dinov3-vitl16-pretrain-lvd1689m/
checkpoints/project/dinov3_adapter/adapter_pilot_v1_final.pt
```

`manifests/latest_weights.json` 记录当前三域 LTWM、配套 scorer、selector 和 support 的期望路径与 SHA-256，**只含元数据**。配置中的哈希防止误用旧权重；scorer 内的 world-model 哈希绑定也会检查。
`manifests/checkpoints.json` 保留原始双域权重索引，仅供历史追溯。

三种子集成、NoMaD、v70 Arm-D/selector、support envelope、calibration result 等额外依赖须独立获取，不能仅凭双域两个 checkpoint 复现这些路径。相关研究脚本保留在 `tools/` 和 `deploy/`。

训练 checkpoint 包含训练元数据，统一入口使用 `torch.load(..., weights_only=False)`，仅加载可信项目来源的文件。
