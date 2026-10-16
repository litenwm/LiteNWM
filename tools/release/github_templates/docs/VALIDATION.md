# 发布验证记录

本次源码包验证环境：Python 3.10，PyTorch 2.13.0.dev20260501+cu130，NumPy 2.2.6，pytest 8.4.2。前向验证使用 CPU。
该环境来自原工作区，不等同于对所有新机器和稳定版依赖组合的保证。

已完成：

- 安装包 wheel 构建和独立 target 目录安装成功，从工作区之外导入运行成功。
- 无权重 demo：合成输入 + 随机小模型，H4 未来形状 `[1,3,4,256,384]`，输出均为有限值。
- 正式三域 seed1701 LTWM/scorer：使用本地现有真实权重、合成输入完成 CPU 前向计算，未来形状 `[1,2,4,256,384]`；权重仍未复制到发布包。
- 将配置中 LTWM 哈希换成错误值，入口按预期拒绝加载。
- 核心 H4、scorer、动作约束、运行合同及部署纯逻辑测试；缺少外部 coherent-turn support 数组时，依赖该数组的测试明确跳过。
- 发布 ZIP 完整性、Python 语法解析、权重/数组排除和源码文件哈希检查。

复查命令：

```bash
python tools/release/verify_repository.py
python examples/run_pipeline.py --demo
python -m pytest -q tests/recon_latents/test_ltwm_horizon.py \
  tests/scoring/test_trajectory_quality_scorer_v2.py \
  tests/planning/test_trajectory_constraints.py \
  tests/planning/test_v70_go2_runtime_contract.py tests/deploy
```

未执行：从原始数据重训、完整数据集效果复现、GPU/Jetson 实机运行。合成输入前向验证不证明训练效果。
