# 无机器狗连接时的真机入口检查

结论：可以确认源码、真实权重、启动参数与控制节点的离线/模拟可执行性；不能把它当作当前无动作复用版本的真机运动验收。

## 这次新增的检查与修复

1. 公共启动器在 `set -u` 下 source ROS：修复 optional tracing 变量未设置导致提前退出的问题，source 后恢复原 nounset 状态。
2. 自动到达 live 模式开启 async 图片但没有 save 图片：现在同时传入 `--save-plan-images --async-plan-images`，满足节点真实参数检查。
3. 修复人工停止专用预规划默认开关误传给自动到达模式的问题；两种模式按参数合同区分。
4. 使用真实 Bash 启动器，替换子进程为仅记录 argv 的空进程，覆盖 shadow、自动到达 live、人工停止 live；确认网卡/topic/零速参数传递，退出保留 STOP、不创建 ARM。
5. 使用真实 `V70LiveNavNode`、`SafeBridgeV5`、`MotionExecutorV2`，仅模拟 ROS 消息/Node API、相机和 SportClient。检查初始化、接受新 H1、拒绝后零速、重复 sequence、STOP、取消 ARM、过期命令、延迟旧速度、正常关停。
6. 硬件 preflight 改为实际导入依赖，而不是只看包名存在；检查 ROS middleware 安装、CUDA、SDK channel/video 模块和 live ARM 目录可写。
7. 使用真实 frozen 资产在 Windows/CPU 完成严格加载和完整 RGB 规划；修复 Linux checkpoint 路径元数据跨平台读取。34 个资产逐文件 SHA 已核对。

模拟测试不加载真实 DDS 网络，不调用实际 SportClient；不能确认实际 topic 频率、坐标系、视频 RPC、lease、CUDA/JetPack ABI、障碍数据或实体步态响应。TensorRT 默认关闭，构建与数值一致性仍需目标硬件验证。

## 接收方环境补充

ROS bridge 使用 CycloneDDS，执行器使用 Fast DDS；两种 ROS middleware 均应安装。对应 Humble 包包括 `ros-humble-rmw-cyclonedds-cpp`、`ros-humble-rmw-fastrtps-cpp`、`ros-humble-sensor-msgs-py`。

原机器已有 `/run/go2-safety` 可写目录，新机器不一定有。可显式选择自己的 ARM 路径，在运行前准备：

```bash
export GO2_MOTION_ARM_FILE="$HOME/.local/state/litenwm/ARMED"
mkdir -p "$(dirname "$GO2_MOTION_ARM_FILE")"
touch /tmp/go2_nav_stop
```

所有终端使用相同的 STOP/ARM 环境变量。preflight 不会自动创建 ARM，live 启动也不会自动解锁。

首次重新接触设备，仍需按 README 先运行 shadow，检查真实图像、odom/cloud 及 status；之后仅做一次小幅 H1 与 STOP/取消 ARM 验收。只有这一步通过，才能确认该版本在那台具体机器狗上的实际运动正常。

最新集中测试：53 项通过，包含上述真实启动脚本和生产节点边界模拟。这里只证明程序层面检查通过，不对未连接设备给出“真机实测通过”的结论。
