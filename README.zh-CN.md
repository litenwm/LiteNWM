# LiteNWM

**Efficient Latent World Models for Onboard Visual Navigation in the Wild**

[项目网站](https://litenwm.github.io/) · [论文](https://arxiv.org/abs/2610.12368) · [代码仓库](https://github.com/litenwm/LiteNWM) · [配套权重](https://github.com/litenwm/LiteNWM/releases/tag/v1.0.0)

arXiv 预印本。

项目结项代码发布包：DINOv3 图像表征 → 冻结适配器 → LiteTokenFormer 世界模型 → H4 轨迹质量评分 → 可选 CEM 优化。

源码来自当前工作区最新文件（含尚未提交的修改）；正式评分配置默认绑定已封存的 **2026-07-29 三域 V70** LTWM 与其配套 scorer。不是旧的 2026-07-19 双域归档。更新的训练/探索脚本同样保留，但不凭文件名版本号替换已经封存的模型。版本入口见 `docs/LATEST_VERSION.md`。

**本仓库不包含任何模型权重。** 安装后可直接运行无权重示例；正式推理需要另行准备匹配的项目权重与输入数据。`robot/` 已提供无动作复用的 Go2/Orin 部署、CPU 执行策略测试、保存图像正式规划、资产安装与硬件检查。配套权重需单独下载，见 [机器狗部署](robot/README.md)。

## 安装并运行

在仓库根目录执行。Python 3.10 或更高版本；CPU 即可运行示例。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU 环境先安装 CPU 版 PyTorch；GPU 环境按对应 CUDA 环境安装 PyTorch。
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python examples/run_pipeline.py --demo --output outputs/demo.json
```

Windows PowerShell 的虚拟环境激活命令为 `.venv\Scripts\Activate.ps1`。

示例使用小尺寸、随机初始化的真实 LTWM 和 scorer 网络，以及合成输入。输出未来 token 的形状 `[1,3,4,256,384]` 和三个候选轨迹的 ATE/RTP 评分。这只检查程序能运行，不能作为训练效果或实验结果。

可选依赖：

```bash
# DINOv3 编码、ROS bag 数据处理、分析绘图
python -m pip install -e '.[encoder,data,analysis,dev]'
```

NoMaD 和 NWM 等外部方法需要各自的代码、权重及依赖环境，见 `EXTERNAL_DEPENDENCIES.md`。Jetson/ROS 实机环境需单独配置，不采用上述 CPU 安装命令。

## 机器狗与执行策略测试

```bash
python robot/smoke_test.py
python -m pytest -q tests/robot
```

普通电脑可先测试执行状态机和“每个新规划只消费一次 H1”。Go2 上先使用 `bash robot/run_go2.bash shadow /path/goal.json`，验收后切换 live；不会自动 ARM。环境与停机步骤见 [robot/README.md](robot/README.md)，保留的优化与移除的动作复用见 [robot/OPTIMIZATIONS.md](robot/OPTIMIZATIONS.md)。

## 用正式权重运行

1. 按 `docs/CHECKPOINTS.md` 准备匹配的 H4 LTWM 和 scorer-v2 权重。
2. 按 `docs/RUNNING.md` 准备适配后的历史/目标 tokens 和动作候选 NPZ。
3. 复制配置，修改本地路径，再执行：

```bash
cp configs/inference.example.json configs/inference.local.json
python examples/run_pipeline.py --config configs/inference.local.json \
  --device cpu --output outputs/prediction.json
```

此入口输出单成员模型评分，不执行 CEM、v70 选择策略或实机控制。v70/三域的实验入口和所需额外资产见 `docs/RUNNING.md`。

## 目录

| 目录 | 内容 |
| --- | --- |
| `src/recon_latents/` | DINOv3 适配器、世界模型、损失和数据接口 |
| `src/scoring/` | 轨迹评分、指标、集成和校准 |
| `src/planning/` | CEM、约束及外部 proposal 接口 |
| `examples/` | 统一无权重示例与正式离线评分入口 |
| `configs/` | 研究配置与可修改的本地配置模板 |
| `tools/` | 数据处理、训练、评估及历史实验脚本 |
| `tests/` | 原项目测试和协议检查 |
| `data/`, `data_splits/` | 数据划分与身份元数据，无原始图像或 latent shards |
| `docs/` | 运行说明、协议和历史研究记录 |
| `deploy/` | 当前 Go2/Orin 执行、安全桥、常驻 planner 与历史研究入口 |
| `robot/` | 公开入口、预检、资产安装、离线测试与部署说明 |
| `manifests/` | 本代码包文件哈希及原始权重哈希记录 |

历史脚本和研究记录保留其原始路径与实验版本；它们不是开箱即用的入口。不要直接套用其中的服务器绝对路径。原始 `pipeline_contract.json` 描述 2026-07-19 的双域 H4 协议，不能代表后来所有实验版本。

## 检查

```bash
# 无需依赖、数据或权重：检查交付文件的完整性
python tools/release/verify_repository.py
# 核心 CPU 测试
python -m pytest -q tests/recon_latents/test_ltwm_horizon.py \
  tests/scoring/test_trajectory_quality_scorer_v2.py \
  tests/planning/test_trajectory_constraints.py
```

完整研究测试集中的部分检查需要未打包的实验资产或外部环境，不能用核心测试通过代替完整实验复现。源码包检查应在刚解压的目录运行；自行修改源码后哈希变化属于预期行为。

## 发布与论文对应

代码仓库为 https://github.com/litenwm/LiteNWM 。模型资产放在 v1.0.0 Release，下载 `litenwm_go2_assets.tar` 与 `asset_release.json`，核对 SHA-256 后按机器狗部署说明安装。

论文原实机实验与当前无动作复用部署版分开记录。后者没有新的物理运动、成功率或板端时延测量；不能直接套用网站数字。完整论文表格复现仍需另备原始数据、匹配的三域 split-lock、候选池及实验配置，详见 [论文与发布对应说明](docs/PAPER_RELEASE_MAP.md)。

## License / 使用许可

除另有说明的第三方材料外，本项目原创代码与文档，以及作者单独发布并明确纳入本许可的项目训练权重，采用 **Creative Commons Attribution-NonCommercial 4.0 International（CC BY-NC 4.0，署名—非商业性）**。完整标准条款见 [LICENSE.md](LICENSE.md)。

允许在许可条件下进行非商业用途的复制、修改和分享；分享时须保留适当署名与许可说明，并注明修改。商业用途未获本许可授权，须另行取得相关权利人的许可。非商业性的判断以标准条款为准，不等同于“仅限大学或科研机构”。

第三方代码、预训练模型、数据和其他外部材料不因本声明而改变原许可；具体来源与适用范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本许可不将论文或独立项目网站重新授权。科研使用请引用 [LiteNWM 论文](https://arxiv.org/abs/2610.12368)；这一引用请求不增加标准许可之外的限制。
