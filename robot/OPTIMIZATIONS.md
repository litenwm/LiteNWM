# 公开机器狗版本的部署优化

初次核对的 `src/` 39 个核心文件与已保存部署版一致；本次另外加入 checkpoint 路径元数据的跨平台加载，NoMaD 后端仅调整加载调用，不改变模型计算，模型对应三域 V70 封存权重。不包含作者场景的目标图、示教路线或遥控服务。

| 优化 | 发布默认 | 实际作用与边界 |
| --- | --- | --- |
| 常驻模型 | 开启 | DINOv3、adapter、NoMaD、三成员 LTWM/scorer/selector 只加载一次 |
| DINO 图像内容缓存 | 开启 | 用图像 SHA 缓存未变化的特征；保留 float16 工件边界再转 float32，不缓存控制动作 |
| NoMaD 条件编码缓存 | 开启 | 同观测/目标的当前候选重采样可共享编码，候选仍重新采样、评分、过门 |
| 固定池快速通道 | 开启 | 当前 K=17 池有足够合格且可执行候选才跳过 CEM，不无条件采纳 NoMaD |
| 条件 CEM / 早停 | 开启，配置预算默认 1 | 按当前 gate/support/archive 做 refinement，预算可设 1–3 |
| 异步图片记录 | 开启 | worker 写图，重复诊断图像可硬链接，减少控制线程等待 |
| settle 预规划 | 人工停止 live 开启；自动到达关闭 | 前一步完成、机体零速稳定期间提前算新计划，不能搭桥运动 |
| rolling 预规划 | 关闭，可开 | 当前合法 H1 执行期间取新图算新计划，到端点检查年龄/漂移、SE(2) rebase 再交接；没新计划则零速 |
| DINO TF32 | 关闭，可开 | 可改变浮点误差的性能选项，需要在接收方硬件复核 |
| LTWM TensorRT | 关闭，可开 | K17/H4 固定 IO，核对引擎/checkpoint SHA；失败或非法输出改用当前规划的 PyTorch evaluator |
| 相机独立进程 | 开启 | JPEG 完整性检查、有界重启，隔离坏帧和视频阻塞 |
| ROS / SDK 进程隔离 | 开启 | planner、安全桥、执行器分离；执行器用 Fast DDS，避免 CycloneDDS 绑定冲突 |

相关环境开关见 `run_go2.bash`。可选 `V70_ROLLING_PLAN_PREFETCH=1`、`V70_DINO_TF32=1`、`V70_LTWM_TRT=1`；当前拒绝后重新取图并规划的恢复仍走完整模型和 gate。

## 已去掉的动作复用

- 拒绝规划、路线偏离、执行失败和端点等待时旧 H1 重放；删除 `_start_last_valid_action_replay` 和旧动作记忆。
- `_start_inflight_forward_bridge`：规划空档的前向搭桥。
- `_start_abstention_forward_reposition`：拒绝后合成的小幅前进。
- `_start_target_loss_forward_continuation` 和 visual-recheck 中基于旧 H1 的继续运动。
- H2/H4 后续动作缓冲交接；每次只执行当前输出的第一步。

`FreshPlanGate` 只存最后消费的 sequence，不存动作。拒绝、STOP/ARM 关闭、动作非法、sequence 已消费或过时，均不能启动新动作。新规划恰好得到相同数值不算旧动作重放：它仍经过新的感知、评分、执行门和独立 sequence。

主动作 follower 的 `start()` 仅剩一个调用点，位于一次性消费门后；测试检查这一集成约束。`v70_live_nav.py` 是同一公开节点的兼容入口，不能再启动旧策略。

发布启动器同时把 `V70_PLANNED_NEUTRAL_MOVE` 默认设为 0：等待/拒绝时走 StopMove，而非用 Move(0,0,0) 维持步态。已选 H1 内部的速度映射与闭环跟随仍保留。

安全桥、LiDAR 地面/障碍过滤、输入超时、里程计闭环、StopMove、STOP/ARM、关停清理保留。速度下限、转向稳定等待和有界转向辅助是把当前 H1 映射为实体运动的控制适配，不是旧动作重放。

## 结果边界

不重训、不改模型权重、不重新调 learned gate。发布启动器默认关闭 aggressive-navigation 与主动扫描。若用户提供自己的视觉 route profile，可推进中间目标，仍需每轮重新选择 H1。

旧长程演示有动作重放、示教 route intent 和工程恢复，不能用于证明此公开策略的成功率、速度或轨迹。CPU、资产与模型加载测试不能替代新的 Go2 运动验收。本次仅在电脑整理，没有启动机器人运动。

视觉-only 复查现在只请求新的完整模型规划，不再恢复缓存计划。启动脚本已覆盖 shadow、自动到达 live、人工停止 live 的实际参数校验，见 OFFLINE_READINESS.md。
