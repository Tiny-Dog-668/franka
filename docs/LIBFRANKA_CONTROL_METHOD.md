# 当前 libfranka 机械臂控制方法

本文记录 `0802_DR simactuator GPU` 真机部署当前实际采用的机械臂控制链路。对应入口为：

```bash
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 250
```

核心配置位于：

```text
configs/e2e_bundle_real_exported_0802_dr_simactuator_gpu.json
```

## 1. 总体结构

当前机械臂不是由 Python 以 30 Hz 直接发送关节命令，也不是每个 policy step 调用一次
`franky.JointMotion`。机械臂由一个 C++ 原生 worker 持续占有 libfranka FCI 控制会话：

```text
RealSense + TorchScript policy（Python，30 Hz，GPU）
                     │
                     │  基座坐标系 TCP 增量、generation、心跳
                     ▼
             ABI-7 共享内存
                     │
                     ▼
       DLS 逆运动学（C++ worker，60 Hz）
                     │
                     │  带符号关节速度参考
                     ▼
   有速度/加速度/跃度约束的轨迹生成器（1 kHz）
                     │
                     │  franka::JointPositions
                     ▼
 libfranka Robot::control + JointImpedance（FCI，1 kHz）
                     │
                     ▼
                 Franka Panda
```

相关实现：

- Python 调度与日志：`franka_sim2real/streaming_server9.py`
- 共享内存和 worker 生命周期：`franka_sim2real/server9_ipc.py`
- 动作缩放、历史和相机输入：`franka_sim2real/streaming.py`
- 1 kHz 原生控制：`native/franka_server9_streaming_worker.cpp`
- worker 构建：`scripts/build_franka_server9_worker.sh`

## 2. 为什么使用 libfranka 0.17.0

当前机器人是 robot server version 9。`pylibfranka 0.21.1` 面向 server version 10，不能直接
用于这台机器人。因此项目编译一个小型 C++ worker，并链接安装在 `franky` wheel 内的
`libfranka 0.17.0` 动态库。

这仍然是官方 libfranka 的 `Robot::control` 接口。`franky` 在当前部署中的作用是：

- 为构建过程提供兼容 server 9 的 libfranka 0.17.0 动态库；
- 通过 `franky.Gripper` 异步控制夹爪；
- **不负责机械臂的 1 kHz policy 轨迹控制**。

## 3. 启动流程

一次真机 session 按以下顺序启动：

1. 加载 GPU TorchScript 和 metadata，校验模型哈希、输入输出维度及 RMA 部署契约。
2. 启动原生 server9 worker；worker 连接机器人、加载模型并读取一次状态。
3. 检查初始关节位置、关节速度、TCP 位姿、夹爪宽度、机器人模式和错误状态。
4. 启动 RealSense，并在进入 FCI 控制前完成相机 warmup 和第一次 policy 推理。
5. 打印第 0 步建议动作，等待操作员输入 `y/yes`。
6. worker 开启一个连续的 `Robot::control` session；整个 rollout 不重启控制器。
7. Python 以绝对 30 Hz deadline 提交 policy action，worker 以 60 Hz 更新 DLS，并在每个
   1 kHz FCI callback 中生成连续关节位置命令。
8. 正常结束或 Python 请求停止时，worker 先受控减速到零，再发送 `MotionFinished`；worker
   自身检测到 watchdog、机器人或 FCI 异常时则中止控制，由 libfranka/机器人安全机制接管。

`--steps` 由操作员指定，RMA 路径当前没有 150 步硬限制。

## 4. Policy 动作如何转换

模型输出四维归一化动作：

```text
[dx, dy, dz, gripper]
```

首先裁剪到 `[-1, 1]`。默认 commissioning 限制为 `0.1`：XYZ 三维采用统一比例缩放，避免
逐维裁剪改变笛卡尔方向；夹爪维度单独裁剪。

当前物理尺度是：

```text
[0.05 m, 0.05 m, 0.05 m, 0.01 m]
```

因此未使用 `--allow-full-scale` 时，单个 policy action 的最大命令为：

```text
XYZ：每个方向峰值最多 0.005 m
夹爪总宽度：每步最多变化 0.001 m
```

XYZ 使用机器人基座坐标系 `robot_root`，不执行旧 blocking 路径中的 Y/Z 符号翻转。

Python 从共享内存读取当前 `O_T_EE`，保持当前姿态方向不变，只给平移部分加上 policy 增量：

```text
target_translation = current_translation + [dx, dy, dz]
target_rotation    = current_rotation
```

该绝对 TCP 目标和递增的 action generation 一起写入共享内存。

## 5. 30 Hz 动作锁存与 60 Hz DLS

每个被接受的 policy action 必须在 worker 中执行恰好两个 60 Hz DLS tick，对应训练时的
`decimation=2`：

```text
1 个 30 Hz policy action = 2 个 60 Hz DLS substep
```

worker 只有在上一代动作已经得到两个 DLS tick 后，才锁存下一代 generation。generation 可以
跳号，因为过期 policy 结果会被丢弃。

末步验收使用 ABI-7 的：

```text
latched_action_generation
latched_action_tick_count
```

它不再依赖容量有限的 control trace，因此长 rollout 不会因为 trace 写满而产生“最后动作未执行”
的假超时。正常 trace 容量目前为 1024 个 60 Hz tick。

DLS 使用末端 Jacobian 求解：

```text
delta_q = Jᵀ (J Jᵀ + lambda² I)⁻¹ pose_error
```

当前阻尼参数：

```text
dls_lambda = 0.01
```

如果下一代动作没有按时到达，当前动作执行完两个 DLS tick 后不会继续追逐旧 TCP 目标，而是把
关节速度参考平滑降到零。

## 6. sim_actuator_velocity 控制律

当前 GPU 配置使用：

```text
control_law = sim_actuator_velocity
reference_velocity_gain = 5.0
maximum_ik_reference_delta_rad = 0.25
```

它将 DLS 关节增量转换成带符号关节速度参考：

```text
dq_reference = reference_velocity_gain * clamp(delta_q)
```

`gain=5.0` 来自 Isaac Lab implicit PD 的 `stiffness / damping = 400 / 80`。与旧的
`joint_position_pursuit` 相比，该控制律保留动作幅值：小 policy action 产生较小速度，大动作产生
较大速度，而不是统一饱和到相同关节速度。

1 kHz callback 不直接把速度参考阶跃发送给机器人。它使用临界阻尼速度跟踪器，根据上一周期
FCI 状态 `q_d`、`dq_d`、`ddq_d` 生成下一条位置命令，同时逐周期限制：

```text
maximum_joint_velocities     = [1, 1, 1, 1, 1.5, 1.5, 1.5] rad/s
maximum_joint_accelerations  = [7, 7, 7, 7, 10, 10, 10] rad/s²
maximum_joint_jerks          = [1500, 1500, 1500, 1500, 2000, 2000, 2000] rad/s³
```

最终通过以下接口连续返回：

```cpp
franka::JointPositions(q_command)
```

控制模式为：

```cpp
franka::ControllerMode::kJointImpedance
```

当前关节阻抗为：

```text
[800, 800, 800, 600, 600, 400, 400]
```

### 当前关节限位边界

worker 会检查当前关节位置没有超出 Panda 硬限位，并计算带 `0.05 rad` margin 的
`q_dls_target`。但是在当前 `sim_actuator_velocity` 分支中，真正驱动机械臂的是由原始 DLS
`delta_q` 生成的速度参考；带 margin 的 `q_dls_target` 主要用于 trace/状态记录，并没有直接裁剪
该速度参考。实际保护仍包括速度、加速度、跃度包络、工作空间检查、Franka 内部保护和硬关节状态
检查。靠近关节边界运行前，应补充速度参考方向上的 soft-limit 约束。

## 7. 夹爪控制

夹爪不进入 1 kHz arm callback。Python 使用独立的 `franky.Gripper` 和异步队列：

- policy 输出的是总夹爪宽度增量；
- 目标宽度限制在 `[0, 0.08] m`；
- 当前速度为 `0.03 m/s`；
- 已有命令执行时，只保留最新的排队目标；
- arm session 停止或异常时调用 gripper stop。

夹爪的 `command`、`poll` 耗时会写入 timing 日志，便于定位 Python 调度停顿。

## 8. Deadline 和 watchdog

30 Hz policy 周期约为 `33.33 ms`。每个动作同时检查：

- policy 相机读取、输入构造和 GPU 推理耗时是否超过一个 policy 周期；
- 动作决策时刻是否已经落后绝对 deadline 一个完整周期；
- 单次 policy 耗时是否超过 `policy_watchdog_s = 0.25 s`。

错过周期但未触发 watchdog 的动作会被安全丢弃，不会随后补执行。终端进度会显示：

```text
Streaming progress: 60/250 steps (accepted=58, deadline_misses=2)
```

进度默认每 30 步打印一次，并在最后一步额外打印。

原生 worker 还检查：

- FCI callback 周期是否超过 `control_watchdog_s = 0.05 s`；
- Python parent/policy heartbeat 是否超时；
- robot state 是否包含 NaN/Inf；
- 当前关节是否越过 Panda 硬限位；
- TCP 目标是否位于配置工作空间；
- robot 是否报告 active error。

当前工作空间为：

```text
x: [0.20, 0.65] m
y: [-0.30, 0.30] m
z: [0.05, 0.45] m
```

`control_command_success_rate` 仅作为诊断值记录。FCI 通信错误由 libfranka 自己抛出的
`ControlException` 处理。

## 9. 停止流程

正常结束、Python 清理流程或 Ctrl+C 向仍在运行的 worker 发出 stop 时，worker 不会立即在运动中
发送 `MotionFinished`。当前受控停止过程为：

1. 将关节速度参考设置为精确零；
2. 使用与正常动作相同的速度/加速度/跃度约束平滑制动；
3. 等待 `q_command == q_d`，且命令速度和加速度归零；
4. 要求实测最大关节速度不超过 `0.005 rad/s`；
5. 上述状态连续保持 100 个 1 kHz 周期；
6. 返回 `franka::MotionFinished`。

停止过程最长允许 2 秒。若不能在 2 秒内满足条件，worker 将中止并报告错误，而不是强行结束
运动生成器。

如果是 worker 自己检测到 FCI callback 超时、heartbeat 超时、机器人 active error、非法状态、
workspace 或 IK 错误，它会从控制 callback 抛出异常并标记 `ABORTED`，不会先走上述 2 秒正常
停止流程。libfranka 和机器人控制器负责这种故障路径的停止保护。

## 10. 共享内存和进程职责

ABI-7 共享内存使用 sequence lock 传递命令和状态。Python 负责：

- policy、相机、action history 和 deadline；
- session 确认、日志、夹爪队列；
- 发送目标、generation 和心跳；
- 监控 worker 状态并在异常时清理资源。

C++ worker 负责：

- 独占 libfranka `Robot::control`；
- 60 Hz DLS、两 tick 动作生命周期；
- 1 kHz 连续轨迹和全部 arm 指令；
- FCI、workspace、状态和 heartbeat watchdog；
- 受控停止以及 ControlException 诊断日志。

共享内存 ABI 版本变更后必须重新构建 worker，否则启动时会报告 ABI mismatch。

## 11. 构建与离线验证

以下操作不会连接或移动机器人：

```bash
./scripts/build_franka_server9_worker.sh
dist/franka_server9/franka_server9_streaming_worker --version
dist/franka_server9/franka_server9_streaming_worker --layout
dist/franka_server9/franka_server9_streaming_worker --self-test
.venv/bin/python -m unittest tests.test_server9_streaming tests.test_streaming
```

当前正确版本应显示：

```text
franka-server9-streaming-worker 0.17.0 abi-7
server9 worker self-test: PASS
```

## 12. 真机运行顺序

先准备和检查 FCI 主机：

```bash
sudo ./scripts/fci/prepare_fci_host.sh --interface enp3s0 --cpu 21
sudo ./scripts/fci/test_fci_network.sh --interface enp3s0 --cpu 21 --host 172.16.0.2
```

然后按风险从低到高运行：

```bash
# 不连接硬件
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --validate-only

# 读取机器人和相机，但不发送运动
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 5 --preview-only

# 只检查连续 FCI hold，不加载 policy、不控制夹爪
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --streaming-check

# 真机短会话
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 5

# 长会话，由操作员选择步数
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 250
```

默认不要直接添加 `--yes`。进入真机 session 前应检查第 0 步打印的 XYZ 位移、夹爪目标和当前
TCP，并保证工作空间清空且急停可用。

## 13. 运行产物和故障定位

每次运行写入：

```text
runs/<timestamp>_e2e_bundle_real_exported_0802_dr_simactuator_gpu/
```

主要文件：

- `rollout.jsonl`：每个 policy step 的输入、输出、动作是否被接受及逐步 timing；
- `control_trace.jsonl`：60 Hz 实测关节状态、目标和 generation；
- `timing_summary.json`：deadline miss、最大 policy/夹爪/latch/循环耗时；
- `fci_error_trace.jsonl`：发生 libfranka ControlException 前的官方 1 kHz 记录；
- `fci_error_audit.json`：根据 FCI 记录重建的速度、加速度、跃度峰值和违规项；
- `summary.json`：本次 session 汇总；
- `rgb/`：按配置保存的 policy RGB 图像。

正常运行时重点检查：

```text
policy_deadline_misses = 0
absolute_policy_deadline_misses = 0
cleanup_errors 不存在或为 []
aborted 不存在或为 false
```

跟踪质量分析：

```bash
python scripts/diagnostics/analyze_streaming_tracking.py --latest 1
```

## 14. 当前方法的关键限制

- policy、相机和夹爪调度仍运行在非实时 Python 进程中；1 kHz arm loop 已隔离到 C++。
- DLS 是阻尼伪逆，不包含完整避障、奇异性度量或 null-space 姿态优化。
- `sim_actuator_velocity` 的速度参考当前没有直接使用 joint-limit margin 做方向约束。
- 工作空间只检查锁存的 TCP 目标，不等价于整条机器人连杆避障。
- 超过训练 episode 的长 rollout 可以执行，但模型行为是否仍在训练分布内需要单独验证。
- RealSense、GPU、Linux 调度和夹爪状态读取仍可能造成 policy deadline miss；过期动作会被丢弃。

## 15. Oracle 物体位置消融入口

可以绕过视觉位置/contact 预测，直接测试同一个 actor_core 和本文件描述的 libfranka 控制链路：

```bash
python scripts/policy/run_exported_0802_dr_simactuator_oracle_gpu.py \
  --cube-position-root 0.50 0.00 0.026 \
  --rma-contact-source zero \
  --steps 250
```

该位置是物体中心在 `robot_root` 下的 XYZ 米制坐标，并在整个 session 内固定。Oracle 模式仍读取
并保存 RealSense 图片，但推理不会调用视觉网络；因此图片只用于实验记录，不影响动作。该入口
默认在 CPU 上运行 actor_core，本机实测约 `0.2 ms/step`；CUDA 正常时可显式添加 `--gpu`。
