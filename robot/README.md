# Unitree Go2 / Jetson Orin 部署

原创部署代码与明确按本项目许可发布的训练权重采用 [CC BY-NC 4.0](../LICENSE.md)，商业用途需另行授权。第三方代码与预训练权重保留各自许可，见 [第三方说明](../THIRD_PARTY_NOTICES.md)。

本目录提供 CPU 执行策略测试、保存图像的正式规划、环境检查、资产安装和实机启动入口，调用 `deploy/v70_go2_orin/v70_live_nav_fast.py`。

**每个新规划最多执行一次 H1。** 无有效新规划则零速；旧 H1 重放、H2/H4 缓冲、规划空档搭桥和旧动作视觉丢失继续运动已移除。视觉特征与 NoMaD 条件缓存仍保留。详见 [优化与边界](OPTIMIZATIONS.md)。改动后的执行策略没有新的实机成功率/时延测量，不能直接沿用原演示结果。

## 普通电脑测试

按根 README 安装核心环境后：

```bash
python examples/run_pipeline.py --demo
python robot/smoke_test.py
python -m pytest -q tests/robot tests/deploy/test_v70_closed_loop_core.py tests/planning/test_safe_bridge_core.py
```

无需权重、ROS、SDK 或机器狗；网络 demo 为随机权重，执行 demo 为真实动作状态机，均不证明模型精度。

## 安装正式资产

源码不含权重。从 [v1.0.0 Release](https://github.com/litenwm/LiteNWM/releases/tag/v1.0.0) 下载配套 `litenwm_go2_assets.tar` 与 `asset_release.json`，核对压缩包 SHA-256 后：

```bash
python robot/install_assets.py --archive /absolute/path/litenwm_go2_assets.tar
python robot/preflight.py --assets
```

安装器只接收普通文件进入 `checkpoints/`、`runtime/`、`external/`，核对 `robot/assets_manifest.json` 的 SHA-256，生成相对路径的 `configs/local_assets.json`。资产含 DINOv3、adapter、NoMaD、三成员 LTWM/scorer/selector、support、冻结校准 JSON、NoMaD 最小外部源码及其许可证。LTWM/scorer 哈希与 `manifests/latest_weights.json` 相同。校准 JSON 保持原样；checkpoint 按文件名定位并核对哈希。

作者可从已验证机器狗备份重建：

```bash
python robot/export_assets.py --backup-archive /path/LiteNWM_robot_code_weights_full.zip \
  --output /outside/source/checkout/litenwm_go2_assets.tar
```

只上传源码而不提供配套资产，别人只能运行无权重测试。资产目录已排除 Git；外部源码与模型依各自许可使用。

## RGB 离线正式规划（无 ROS/控制）

安装适配电脑的 PyTorch/torchvision，以及 `transformers==4.57.0`、`diffusers==0.35.1`、`opencv-python`、`efficientnet-pytorch==0.7.1`、`einops`、`safetensors`。提供四张时间排列的相机图像、目标和位姿 JSON：

```json
[{"receive_time":0.0,"x":0.0,"y":0.0,"yaw":0.0},
 {"receive_time":0.2,"x":0.02,"y":0.0,"yaw":0.0},
 {"receive_time":0.4,"x":0.04,"y":0.0,"yaw":0.0},
 {"receive_time":0.6,"x":0.06,"y":0.0,"yaw":0.0}]
```

米/弧度。上述只是格式示例，必须使用与图像对齐的真实位姿。

```bash
python robot/offline_plan.py --history h0.jpg h1.jpg h2.jpg h3.jpg --goal goal.jpg \
  --poses poses.json --device cuda --cem-iterations 1 --output outputs/robot_plan.json
```

可用 `--device cpu`，但大模型较慢。输出选择/拒绝、动作和计时，不发送运动指令。

## 板端环境与预检

参考环境为 Go2 EDU/Orin NX、Ubuntu 22.04、ROS 2 Humble、Python 3.10。按 [ROS 官方说明](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)、[Unitree SDK](https://github.com/unitreerobotics/unitree_sdk2_python)、[NVIDIA Jetson PyTorch](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) 配置各自环境。保留厂商 torch/torchvision，不执行根 README 的 CPU 安装命令。原板端为 torch 2.5.0a0、torchvision 0.20.0a0、NumPy 1.24.4。

```bash
python -m pip install -r robot/requirements-jetson.txt
export V70_PYTHON=/path/to/your/venv/bin/python
export V70_INTERFACE=enP8p1s0                 # 实际网卡
export V70_EXPECTED_IP=192.168.123.18        # 该网卡实际地址
export V70_ROS_SETUP=/opt/ros/humble/setup.bash
export V70_SENSOR_SETUP=/path/unitree_ros2/install/setup.bash
export V70_ODOM_TOPIC=/utlidar/robot_odom
export V70_CLOUD_TOPIC=/utlidar/cloud_base
source "$V70_ROS_SETUP"
source "$V70_SENSOR_SETUP"
```

不必在板端安装根 PyPI 包；启动器设置仓库/`src` 路径。点云必须为机体坐标系、米制。代码不依赖 OpenTrackVLA、作者遥控服务或 UI。安全桥、planner、SportClient 分离；执行器使用 Fast DDS，避免同进程 CycloneDDS 冲突。

## Shadow / live / 停止

复制 `robot/goal.example.json`，填入当前机器狗拍摄的 goal 图片路径。未经真实标定保持 `arrival_calibrated=false`，不能为绕过检查把它改为 true。

```bash
touch /tmp/go2_nav_stop
python robot/preflight.py --assets --hardware --route /path/goal.json --mode shadow
bash robot/run_go2.bash shadow /path/goal.json
```

shadow 取真实传感器、跑模型，桥和执行器保持 dry-run；STOP 必须存在、ARM 必须不存在。发布启动器不启用主动扫描运动。首次实机使用先检查感知、零速关停及物理停止控制。

live 使用真实标定的自动到达，或明确由操作者终止。未标定单目标使用后者：

```bash
export V70_OPERATOR_POSTURE=standing
export V70_OPERATOR_STOP_ONLY=1
export V70_OPERATOR_STOP_ACK=I_WILL_STOP_THE_ROBOT_MANUALLY
bash robot/run_go2.bash live /path/goal.json
# 报告 ready 后，在同环境的另一个终端显式 ARM：
bash deploy/v70_go2_orin/arm_v70_live.bash I_UNDERSTAND_THIS_WILL_MOVE_THE_ROBOT
# 停止：
bash deploy/v70_go2_orin/disarm_v70_live.bash
```

Ctrl+C 会关停；启动器不会自动 ARM 或移除 STOP。日志默认 `~/litenwm_runs/<UTC时间>`，可设置 `V70_OUTPUT_DIR`。ROS/CUDA/TensorRT 与本次无动作复用版本的实际运动需由接收方在机器狗上验收。

常驻模型、内容缓存、固定池快速通道、条件 CEM、异步图片记录和人工停止模式下的零速 settle 预规划默认开启，自动到达模式关闭。运动中预规划、TF32、TensorRT 默认关闭，可按 `OPTIMIZATIONS.md` 启用。

TensorRT 引擎不作为通用权重跨设备复制。`export_tensorrt_onnx.py` 提供原手工双线性 warp 的导出；在接收方 JetPack/TensorRT 构建、核对结果后设置 `V70_LTWM_TRT=1`。不可用或输出非法时回退当前规划的 PyTorch evaluator，不回退旧动作。

可选 TensorRT 导出/构建，在目标 Orin 环境安装对应 ONNX/TensorRT 工具后执行：

```bash
for seed in 1701 1702 1703; do
  V70_LTWM_TRT=0 python robot/export_tensorrt_onnx.py --seed "$seed"
done
python robot/build_tensorrt.py
```

先分别设置 `V70_LTWM_TRT=0` 和 `=1`，对同一四幅图像/位姿执行 `offline_plan.py`，比较拒绝状态、候选来源、H1/H4 动作与规划耗时，再决定是否在 live 使用。构建脚本不会把原机器人旧引擎的验证结果冒充当前硬件结果。

跨平台 checkpoint 元数据由 `planning/checkpoint_io.py` 转换为可读取的纯路径类型，解决 Linux 权重在 Windows 上的 PosixPath 反序列化问题；不改权重字节，也不全局替换 pathlib。全 pickle 加载只面向已核对 SHA 的可信作者权重。

本次验证记录见 `validation_report.json`：53 项核心/执行策略及模拟启动测试通过，34 个正式资产安装和 SHA 核对通过，真实权重的完整 CPU RGB 规划与 CLI 运行通过；测试输入为合成图像/位姿，不代表实机模型精度。未验证本次改动的真实运动或可选 TensorRT 构建。

无机器狗连接时的详细检查与边界见 [OFFLINE_READINESS.md](OFFLINE_READINESS.md)，包括启动参数修复、模拟 ROS/SDK 节点测试、middleware 与 ARM 目录准备。模拟通过不等于真机运动已验收。
