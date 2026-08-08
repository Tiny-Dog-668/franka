# 具体架构

本文件描述 `/home/td/franka` 真机部署工具链的模块职责、控制链路、数据流和接口契约。面向需要修改代码的人，而不是只想跑命令的操作者（后者见 [`../README.md`](../README.md)）。

维护规则见 [`../AGENT.md`](../AGENT.md) 第 9 节。改动模块边界、数据流、共享内存 ABI 或模型契约后必须同步更新本文件。

> 本文件描述的是**仓库根目录的真机部署轨道**。`TacEx/` 是独立 Git 仓库的 Isaac Lab 训练轨道，其架构见 `TacEx/docs/ARCHITECTURE.md`。两者的唯一耦合点是导出的 TorchScript 权重及其 metadata 契约（见第 6 节）。

## 1. 全局视图

工具链把 Isaac Lab 中训练的 RMA 策略搬到真机上执行，核心难点是把 30 Hz 的策略输出安全地转换成 1 kHz 的 FCI 关节指令。

```text
┌─────────────────────── 训练侧（TacEx/，独立仓库） ────────────────────────┐
│  Isaac Lab 环境  →  skrl PPO Teacher  →  Student 蒸馏  →  TorchScript 导出 │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │  checkpoint/*.pt + *.json（契约 metadata）
                                 ▼
┌─────────────────────── 部署侧（本仓库） ─────────────────────────────────┐
│                                                                          │
│  configs/*.json ──► BundleDeployConfig ──► 契约校验 / 初始状态门禁        │
│                                                                          │
│  RealSense D435 ──► crop+resize 224x224 ──┐                              │
│  GelSight Mini（可选） ───────────────────┤                              │
│  机器人状态 ──► proprio_obs[15] ──────────┼──► BundleTorchScriptPolicy    │
│  动作历史 ──► action_history[4] ──────────┘         │  30 Hz             │
│                                                     ▼                    │
│                                          mean_actions[4] ∈ [-1,1]        │
│                                                     │                    │
│                          clip + action_adapter.scales                    │
│                                                     ▼                    │
│                                     [dx, dy, dz, gripper]（米，基座系）   │
│                                                     │                    │
│                  ┌──────────────────────────────────┴────────┐           │
│                  ▼ control_mode=streaming                    ▼ blocking  │
│         DLS IK（60 Hz，Python）                      RealFrankaEnv        │
│                  │                                   （franky            │
│                  ▼  ABI-8 共享内存                     CartesianMotion）  │
│         原生 worker（C++，1 kHz FCI）                        │            │
│                  │                                           │            │
│                  └──────────────┬────────────────────────────┘            │
│                                 ▼                                        │
│                        Franka Panda（FCI server v9）                     │
│                                                                          │
│  产物 ──► runs/<timestamp>_<run_name>/                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 2. 目录结构

| 路径 | 作用 |
| --- | --- |
| `franka_sim2real/` | 运行时库：配置、相机、策略、控制、IPC、标定、类型 |
| `franka_sim2real/envs/` | 机器人后端抽象：`base.py`、`franka_real.py`（franky blocking）、`mock_sim.py` |
| `franka_sim2real/calibration/` | eye-to-hand 标定数学 |
| `scripts/policy/` | 策略部署 CLI 入口与视觉头验证工具 |
| `scripts/robot/` | 状态读取、回零、手动小幅运动 |
| `scripts/camera/` | RealSense 预览、crop 采样、GelSight 启动 |
| `scripts/calibration/` | 标定采集/求解、AprilTag 位姿验证 |
| `scripts/diagnostics/` | 力传感器、方向、streaming 跟踪质量诊断 |
| `scripts/fci/` | FCI 主机网络与 CPU 亲和准备（需 root） |
| `native/` | 1 kHz libfranka C++ worker 源码 |
| `configs/` | 每个部署版本一份 JSON，历史配置不覆盖 |
| `checkpoint/` | 模型权重与 metadata（`*.pt` 被 Git 忽略，`*.json` 保留） |
| `tests/` | 标准库 `unittest` 离线测试 |
| `tools/system/` | 实时内核、网络、主机验证脚本 |
| `third_party/upstream_libfranka/` | vendored libfranka + pylibfranka（Git submodule） |
| `third_party/pylibfranka_streaming_patch/` | pylibfranka 0.21.1 的 streaming 补丁包 |
| `docs/` | 本文件及各专题文档 |
| `assets/` | AprilTag 打印资源 |
| `TacEx/` | Isaac Lab 训练子项目（独立 Git 仓库，主仓库中 untracked） |
| `runs/`、`artifacts/`、`dist/`、`.venv/` | 运行产物与本地环境，均被 Git 忽略 |

## 3. 配置模型

所有部署行为由一份 JSON 描述，`load_bundle_config()` 解析为 `BundleDeployConfig` 数据类（`franka_sim2real/e2e_bundle.py`）。

| 配置段 | 数据类 | 关键字段 |
| --- | --- | --- |
| 顶层 | `BundleDeployConfig` | `robot_ip`、`realtime`（`enforce`/`ignore`）、`control_mode`（`blocking`/`streaming`）、`speed`、夹爪参数、`async_gripper_commands` |
| `camera` | `BundleCameraConfig` | `source`、`serial`、分辨率、`warmup_frames`、`enable_crop` 与 crop 四元组 |
| `tactile_camera` | `BundleTactileCameraConfig` | `enabled`、`left_device`/`right_device`、分辨率、`first_frame_timeout_s` |
| `action_adapter` | `BundleActionAdapterConfig` | `labels`、`scales`、`clip_low`/`clip_high`、`gripper_mode` |
| `runner` | `BundleRunnerConfig` | `steps`、`log_dir`、`run_name` |
| `model` | `BundleModelConfig` | `model_path`、`metadata_path`、`device`、`history_source`/`history_scale`/`history_delay_steps`、`enforce_policy_contract` |
| `initial_state` | `BundleInitialStateConfig` | `enforce`、目标关节角/TCP/夹爪宽度及各自容差、`required_robot_mode` |
| `streaming` | `BundleStreamingConfig` | `backend`、频率、`control_law`、增益与三阶限幅、`joint_impedance`、watchdog |
| `streaming.collision_behavior` | `BundleCollisionBehaviorConfig` | 关节力矩与笛卡尔力的上下阈值 |
| `workspace` | `WorkspaceLimits`（`types.py`） | `minimum`/`maximum` |

`control_mode` 只接受 `blocking` 和 `streaming`；`streaming.backend` 只接受 `async_position` 和 `server9_joint_position`，其它值在校验阶段直接报错。

**版本演进约定**：每个新部署版本新增一份 `configs/e2e_bundle_real_exported_<日期><变体>.json`，绝不覆盖旧配置。这保证任何历史实验都能原样复现。

## 4. 入口层

### 4.1 统一 CLI

`scripts/policy/run_exported_0711.py` 是**唯一**的部署实现，导出 `REPO_ROOT` 和 `main(default_config=...)`。其余 `run_exported_*.py` 每个只有约 10 行：

```python
from run_exported_0711 import REPO_ROOT, main

DEFAULT_CONFIG = REPO_ROOT / "configs" / "e2e_bundle_real_exported_0726.json"

if __name__ == "__main__":
    raise SystemExit(main(default_config=DEFAULT_CONFIG))
```

新增版本入口时保持这个形式，通用逻辑一律改在 `run_exported_0711.py` 或 `franka_sim2real/` 中。

### 4.2 运行模式

`main()` 解析参数后按下列层次决定行为，这也是强制的验收顺序：

| 模式 | 参数 | 连接机器人 | 连接相机 | 发送指令 |
| --- | --- | --- | --- | --- |
| 契约校验 | `--validate-only` | 否 | 否 | 否 |
| 无运动预览 | `--preview-only` | 是（只读） | 是 | 否 |
| 通信检查 | `--streaming-check` | 是 | 是 | 建链但不运动 |
| 单步 | `--steps 1` | 是 | 是 | **是，需人工输入 `y`/`yes`** |
| 多步逐步确认 | `--confirm-each-step` | 是 | 是 | 是（blocking 路径专用） |
| 多步连续 | `--steps N --yes` | 是 | 是 | 是 |

`--streaming-check` 会强制走 streaming 分支，即使配置写的是 blocking。streaming 路径不支持 `--confirm-each-step`。

### 4.3 分发

`run_bundle_deploy()`（`e2e_bundle.py:1803`）按 `control_mode` 分发；streaming 再由 `run_streaming_bundle_deploy()`（`streaming.py:854`）按 `backend` 二次分发：

```text
run_bundle_deploy
├── control_mode == "streaming" 或 --streaming-check
│     └── run_streaming_bundle_deploy
│           ├── backend == "server9_joint_position" → run_server9_streaming_bundle_deploy（原生 worker）
│           └── backend == "async_position"         → pylibfranka async_position（需 server v10）
└── control_mode == "blocking" → RealFrankaEnv（franky）
```

当前真机是 FCI server version 9，`async_position` 路径依赖 pylibfranka 0.21.1 / server v10，**在本机不可用**；实际生产路径是 `server9_joint_position`。

## 5. 感知层

### 5.1 RealSense 主相机

`RealSenseRGBCamera` 采集 640x480 BGR，经 `_apply_camera_crop()` 裁剪后 resize 到 224x224 RGB。crop 参数是**模型契约的一部分**，不同训练批次不同：

| 版本 | crop |
| --- | --- |
| 0726 | x[80:560]、y[0:480] |
| 0801 及之后 | `left=100, top=34, width=400, height=398` |

crop 改错等于给模型喂了分布外输入，症状是策略输出方向系统性偏移，而不是明显报错。用 `scripts/camera/preview_camera_crop.py` 可视化确认。

`LatestFrameCamera` 在 streaming 路径下维护后台取帧线程，保证 30 Hz 策略 tick 总能拿到最新一帧而不被相机 IO 阻塞。`StaticRGBCamera` 用于离线用图片替代相机。

### 5.2 GelSight 触觉（0808）

`GelSightPairCamera` 通过 OpenCV 打开左右两个 GelSight Mini（device 0 和 6，3280x2464@25），仅在配置中 `tactile_camera.enabled` 为 true 时启用。目前只有 `configs/e2e_bundle_real_exported_0808_gelsight.json` 打开。

### 5.3 相机装配

`PolicyCameraRig` 把主相机和可选触觉相机组合成策略实际需要的输入集合，屏蔽上层对具体相机类型的感知。

## 6. 策略层与模型契约

`BundleTorchScriptPolicy` 加载 TorchScript 权重和同名 JSON metadata，并在启动时做强校验。

当前主力模型 `checkpoint/0802_DR_gpu/rma_student_dr_sim2real_gpu.json` 的契约：

| 项 | 值 |
| --- | --- |
| `kind` / `version` | `tacex_rma_student_torchscript` / `6` |
| 输入顺序 | `wrist_rgb` → `proprio_obs` → `action_history` |
| `wrist_rgb` | `[224, 224, 3]`，uint8 RGB，NHWC |
| `proprio_obs` | `[15]` |
| `action_history` | `[4]` |
| 输出 | `mean_actions[4]`，界 `[-1, 1]` |
| Actor 特征 | 30 维：`normalized_proprio_obs[15]` + `normalized_action_history[4]` + `normalized_cube_position_root[3]` + `normalized_gripper_position_root_from_fk[3]` + `normalized_cube_minus_gripper_position_root[3]` + `left_right_cube_finger_contact[2]` |
| 位置参考系 | `robot_root`（Franka 基座） |
| 归一化 | 关节上下限、关节速度尺度、夹爪中心/尺度、`history_scale = [0.025, 0.025, 0.025, 0.005]`、cube/gripper/target 的 center 与 scale |
| 内部 FK | Actor 内部用 `proprio_obs` 的关节角做 Panda FK 求指尖中点，`panda_link7→link8 = 0.107 m`，`hand→指尖中点 = 0.1034 m` |

模型**内部**从 `wrist_rgb` 预测方块在基座系的 XYZ 和左右手指接触概率，不预测物体朝向。

`enforce_policy_contract: true` 时，`_validate_policy_contract()` 和 `_validate_tacex_rma_student_contract()` 会校验输入输出维度、metadata 版本、SHA256 和动作维度是否与 `action_adapter` 一致；`validate_bundle_artifacts()` 是 `--validate-only` 的实现，完全不连硬件。

`ActionHistoryBuffer` 按 `history_source`、`history_scale` 和 `history_delay_steps` 维护动作历史；默认用 `processed_action` 且延迟 1 步，与训练时的时序对齐。

## 7. 动作映射

```text
mean_actions[4]
  │  clip_policy_action：按 action_adapter.clip_low/clip_high 裁剪到 [-1, 1]
  ▼
  │  raw_action_to_robot_action：逐维乘 action_adapter.scales = [0.05, 0.05, 0.05, 0.01]
  ▼
RobotAction(dx, dy, dz, gripper)   # 单位：米
```

**坐标系约定（极易出错）**：

- **streaming 路径**：XYZ 是机器人基座坐标系，**不做**任何取反。
- **legacy blocking 路径**：沿用早期实现的 Y/Z 取反。

两者不能混用。`gripper_mode: delta_width` 表示第 4 维是夹爪**总宽度**的增量，不是单指位移。

## 8. 控制层

### 8.1 Blocking 路径（0711 / 0712 / 0726）

`RealFrankaEnv`（`envs/franka_real.py`）基于 franky，每个策略 tick 下发一次相对 `CartesianMotion` 并等待完成，夹爪用 `franky.Gripper`。实现简单、便于逐步确认，但每步都有起停，无法连续运动，且时序不受控。

### 8.2 Streaming 路径（0801 及之后）

分成三个频率层，每层职责严格分离：

| 层 | 频率 | 位置 | 职责 |
| --- | --- | --- | --- |
| 策略层 | 30 Hz | Python，`streaming.py` | 采图、推理、动作裁剪与缩放、TCP 目标锁存 |
| IK 层 | 60 Hz | Python，`streaming.py` | 阻尼最小二乘 IK，输出关节增量 |
| 控制层 | 1000 Hz | C++，`native/franka_server9_streaming_worker.cpp` | 速度跟踪、三阶限幅、FCI 读写、停止时序 |

IK 层核心函数：

- `pose_error(current_pose, target_pose)`：位姿误差，旋转部分由 `_rotation_vector()` 转成旋转向量。
- `dls_joint_delta(jacobian, pose_error, damping)`：阻尼最小二乘解，`damping` 来自 `dls_lambda`（默认 0.01）。
- `latch_tcp_target(current_pose, executed_xyz)`：锁存目标，避免误差在多个 tick 间累积漂移。
- `_apply_joint_limits(q_target, margin)`：按 `joint_limit_margin_rad` 留出关节限位余量。

夹爪走 `AsyncGripperQueue`，在独立线程执行，不阻塞 1 kHz 控制回路。

策略结果带时效性检查：`policy_result_is_timely()` 结合 `policy_watchdog_s`（0.25 s）判断，超时的结果被丢弃，worker 端的 generation 编号因此允许跳号。

### 8.3 控制律

C++ worker 支持两种控制律（`enum class ControlLaw`）：

| 值 | 名称 | 说明 |
| --- | --- | --- |
| 0 | `joint_position_pursuit` | 把参考点放在实测位置前方一个固定关节增量处。`libfranka::limitRate` 读到的隐含速度是 `增量 / 周期`，任何可用增量都会饱和到 `maximum_joint_velocities`，导致关节速度与动作幅值无关，且逐关节 clamp 会旋转笛卡尔方向（0802 参考位姿实测偏差 54.5°）。**已不推荐**。 |
| 1 | `sim_actuator_velocity` | 复现 Isaac Lab implicit PD 的稳态 `qd = (stiffness / damping) * dq_ik`。`FRANKA_PANDA_HIGH_PD_CFG` 是 400/80，故 `reference_velocity_gain = 5.0`，饱和动作对应 0.25 m/s TCP 速度即 8.33 mm/步。关节速度与动作成正比。**当前默认**。 |

`sim_actuator_velocity` 下，1 kHz 回调用**临界阻尼速度跟踪器**把速度参考转成位置指令，逐周期限制速度、加速度和跃度。速度参考可以任意缩小、清零或反向，跟踪器仍保证轨迹导数连续——这修复了早期把 `abs(qd_ref)` 当作 `limitRate` 动态硬上限导致的 `joint_motion_generator_velocity_discontinuity`。

因为控制律是比例的，`commissioning_action_limit` 是**唯一**且线性的调速旋钮，按 `0.1 → 0.3 → 0.6 → 1.0` 逐级放开，对应 0.83 → 2.5 → 5.0 → 8.33 mm/步。关节限幅是安全包络，不是调速手段：一旦饱和，DLS 的幅值信息就被丢弃，方向也会被旋转。

详细推导见 [`LIBFRANKA_CONTROL_METHOD.md`](LIBFRANKA_CONTROL_METHOD.md)。

### 8.4 停止时序

`MotionFinished` 必须发在一个**指令速度和加速度精确为零**的指令上。停止路径与动作路径共用同一个临界阻尼、有界跃度的速度跟踪器，参考设为零；只有当一步精确归零所需的加速度和跃度都在配置包络内时才落到 bit-exact zero。随后要求 `q_command == q_d`、`dq_d` 与 `ddq_d` 均为零、实测关节速度不超过 0.005 rad/s，**连续保持 100 个 1 kHz 周期**后才发送 `MotionFinished`。

这段逻辑修复过真实 FCI 报错（实测第 6 关节跃度约 7654 rad/s³ 对上限 2000，加速度约 11.9 rad/s² 对上限 10），不要改动。`--self-test` 会从正负多个入口速度扫描完整停止过程逐周期断言。

## 9. 共享内存 IPC

Python 与 C++ worker 通过共享内存通信，布局由 `franka_sim2real/server9_ipc.py` 的 `SharedData`（ctypes `Structure`）和 C++ 侧同名结构体共同定义。

| 项 | 值 |
| --- | --- |
| ABI 版本 | **8**（`ABI_VERSION` 与 `kAbiVersion` 必须一致，否则 worker 启动即拒绝） |
| 主要结构 | `SharedData`（命令、状态、序号、控制律、限幅）、`TraceEntry`（控制 trace）、`FciLogEntry`（FCI 日志） |
| Python 侧封装 | `Server9SharedMemory`（映射与读写）、`Server9Worker`（进程生命周期）、`WorkerSnapshot`（一次快照） |
| 布局自检 | Python `validate_ctypes_layout()`；C++ 侧 `offsetof` 静态断言（如 `offsetof(SharedData, control_law) == 520`） |

**修改布局的完整步骤**：改 `SharedData` → 同步 C++ 结构体与 `offsetof` 断言 → 两侧版本号同时加一 → `./scripts/build_franka_server9_worker.sh` → 跑 `tests/test_server9_streaming.py`。漏掉任何一步都会在真机上表现为难以定位的数据错位。

`streaming_server9.py` 负责 worker 的启动、`_wait_running()` / `_wait_generation_ticks()` 同步、`snapshot_to_observation()` 状态转换，以及 `_audit_fci_log()` 在异常时把 FCI 报错前的官方命令日志写入 `runs/`。

## 10. 安全机制

按执行顺序排列，任何一层失败都会中止：

1. **离线契约校验**（`--validate-only`）：模型输入输出维度、metadata 版本、SHA256、动作维度与配置一致性。
2. **streaming 契约校验**（`validate_streaming_contract`）：频率、增益、限幅、backend 与控制律取值合法性。
3. **初始状态门禁**（`evaluate_initial_state`）：关节角、关节速度、TCP 平移与朝向、夹爪宽度、机器人模式为 `Idle`、无 FCI 错误、夹爪未夹持。`enforce: true` 时不满足直接拒绝启动。门禁失败的正确处理是用 `scripts/robot/go_to_zero_pose.py` 回到训练初始姿态，不是放宽容差。
4. **人工确认**：单步模式打印拟执行的位移与夹爪宽度，只有输入 `y` 或 `yes` 才运动。`--allow-full-scale` 禁止与 `--yes` 组合，且始终要求一次会话确认。
5. **调试期动作限幅**（`commissioning_action_limit`）：默认把归一化动作限制在 `[-0.10, 0.10]`。
6. **工作空间限制**（`_validate_workspace_target`）：目标 TCP 超出 `workspace` 立即中止。
7. **关节限位余量**（`_apply_joint_limits`）：留出 `joint_limit_margin_rad`。
8. **三阶限幅**：worker 逐周期限制关节速度、加速度、跃度。
9. **FCI 碰撞阈值**：`collision_behavior` 显式配置关节力矩 20/40 Nm、笛卡尔力 10~25 N。
10. **双 watchdog**：`policy_watchdog_s`（0.25 s，策略结果时效）与 `control_watchdog_s`（0.05 s，控制回路存活）。
11. **安全停止**：`_safe_stop()` 在异常路径上停控制并清空夹爪队列。

## 11. 运行产物

每次运行写入 `runs/<timestamp>_<run_name>/`（被 Git 忽略，可能含真机图像）：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 本次实际生效的完整配置 |
| `rollout.jsonl` | 逐步 observation、action、timing；oracle 模式下还含 `model_input.rma_actor_input` |
| `summary.json` | 运行汇总 |
| `rgb/step_XXXX.png` | 模型实际看到的 RGB |
| `control_trace.jsonl` | streaming 专有，逐周期控制 trace |
| `timing_summary.json` | streaming 专有，时序统计 |

streaming 路径在控制期间**只缓存不落盘**，`stop_control()` 之后才一次性写出，避免磁盘 IO 干扰 1 kHz 回路。`scripts/diagnostics/analyze_streaming_tracking.py` 读取这些 trace 计算 ratio、slope 和方向误差，理想值分别接近 1、1、0。

## 12. 标定链路

固定 D435 的 eye-to-hand 标定：

```text
collect_eye_to_hand.py（棋盘格 11x8 内角点、15 mm 方格，连续采集 25 个姿态）
        │
        ▼
franka_sim2real/calibration/hand_eye.py: solve_eye_to_hand()
        │
        ▼
configs/eye_to_hand_d435_215322076207.json
        │
        ▼
live_apriltag_cube_pose.py（tag36h11 ID 0，40 mm；50 mm 方块中心位于 Tag -Z 方向 25 mm）
evaluate_policy_vs_apriltag.py（把 AprilTag 位姿与策略视觉头预测对比）
```

流程细节见 [`HAND_EYE_CALIBRATION.md`](HAND_EYE_CALIBRATION.md)。首次使用必须先跑 `--camera-check` 和 `--dry-run`。

## 13. 构建产物

| 产物 | 构建命令 | 输出 |
| --- | --- | --- |
| 原生 worker | `./scripts/build_franka_server9_worker.sh` | `dist/franka_server9/franka_server9_streaming_worker`（`g++ -std=c++17`，链接 franky wheel 内的 `libfranka 0.17.0`，构建后自动跑 `--version` 与 `--self-test`） |
| pylibfranka 补丁 wheel | `./scripts/build_pylibfranka_wheel.sh [--install]` | `dist/pylibfranka/pylibfranka_streaming_patch-0.21.1.1-*.whl`（需 `pybind11>=3,<4` 与已安装的 pylibfranka 0.21.1） |

`dist/` 被 Git 忽略，换机器或改动 `native/` 后必须重新构建。

## 14. 环境与依赖

| 项 | 值 |
| --- | --- |
| Python | 3.10，本机 `.venv`（无 `requirements.txt`，环境不可复现，见 `AGENT.md` 第 8 节） |
| 关键包 | `torch 2.12.1+cu132`、`numpy 2.2.6`、`opencv-python 4.13.0.92`、`franky-control 1.1.3`、`pylibfranka 0.21.1`、`pylibfranka-streaming-patch 0.21.1.1`、`pyrealsense2 2.57.7` |
| C++ | libfranka 0.17.0（经 franky，server v9）；libfranka 0.21.1（pylibfranka，server v10，本机机器人不支持） |
| 内核 | 实时内核；streaming 部署要求 `realtime: "enforce"`，只读诊断可用 `ignore` |
| 主机准备 | `tools/system/setup_franka_realtime_root.sh`、`scripts/fci/prepare_fci_host.sh --interface enp3s0 --cpu 21` |

## 15. 相关文档

| 文档 | 内容 |
| --- | --- |
| [`../AGENT.md`](../AGENT.md) | Agent 维护约束、安全红线、验证规则 |
| [`CHANGE_HISTORY.md`](CHANGE_HISTORY.md) | 修改历史时间线 |
| [`LIBFRANKA_CONTROL_METHOD.md`](LIBFRANKA_CONTROL_METHOD.md) | 当前控制链路权威说明 |
| [`PYLIBFRANKA_STREAMING.md`](PYLIBFRANKA_STREAMING.md) | streaming 架构、server9 vs server10、FCI 主机准备 |
| [`SMOOTH_STREAMING_ARCHITECTURE.md`](SMOOTH_STREAMING_ARCHITECTURE.md) | 0802 smooth 变体的速度 profile |
| [`HAND_EYE_CALIBRATION.md`](HAND_EYE_CALIBRATION.md) | eye-to-hand 标定流程 |
| [`E2E_BUNDLE_DEPLOY.md`](E2E_BUNDLE_DEPLOY.md) | 早期 bundled policy 部署（5 维动作） |
| [`SIM2REAL_FRAMEWORK.md`](SIM2REAL_FRAMEWORK.md) | 通用 sim/real backend 切换验证框架 |
| [`22_04_realtime_workflow.txt`](22_04_realtime_workflow.txt)、[`after_realtime_boot_commands.txt`](after_realtime_boot_commands.txt) | 实时内核安装与重启后验证 |
| [`GITHUB_UPLOAD.md`](GITHUB_UPLOAD.md) | 仓库上传步骤 |
