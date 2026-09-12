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
| `scripts/real_rlpd/` | RLPD 的契约校验、专家/online 采集、离线标注与训练入口 |
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
| `real_rlpd/` | 独立 PyTorch RLPD：策略 adapter、4D residual、Replay、ensemble SAC、teleop |
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
| `tactile_camera` | `BundleTactileCameraConfig` | `enabled`、运行时 `auto_discover`、`left_device`/`right_device`、分辨率、`first_frame_timeout_s` |
| `action_adapter` | `BundleActionAdapterConfig` | `labels`、`scales`、`clip_low`/`clip_high`、`gripper_mode` |
| `runner` | `BundleRunnerConfig` | `steps`、`log_dir`、`run_name` |
| `model` | `BundleModelConfig` | `model_path`、`metadata_path`、`device`、`history_source`/`history_scale`/`history_delay_steps`、`enforce_policy_contract`；0809 XY RMA 另有 `rma_contact_force_source` |
| `initial_state` | `BundleInitialStateConfig` | `enforce`、目标关节角/TCP/夹爪宽度及各自容差、`required_robot_mode` |
| `streaming` | `BundleStreamingConfig` | `backend`、频率、`control_law`、增益与三阶限幅、`joint_impedance`、watchdog |
| `streaming.collision_behavior` | `BundleCollisionBehaviorConfig` | 关节力矩与笛卡尔力的上下阈值 |
| `workspace` | `WorkspaceLimits`（`types.py`） | `minimum`/`maximum` |

`control_mode` 只接受 `blocking` 和 `streaming`；`streaming.backend` 只接受 `async_position` 和 `server9_joint_position`，其它值在校验阶段直接报错。

**版本演进约定**：每个新部署版本新增一份 `configs/e2e_bundle_real_exported_<日期><变体>.json`，绝不覆盖旧配置。这保证任何历史实验都能原样复现。

0809 XY RMA Student 的输入还包含 `contact_force_n[2]`。真机没有独立的左右指尖力读数，配置
`rma_contact_force_source: "gripper_is_grasped"` 因而只把 Franka gripper 的单一
`is_grasped` 标志映射为 `[1,1]` 或 `[0,0]`，以复现 Actor 使用的“双侧均超过 1 N”二值特征。
这是用于效果验证的近似，不能用腕部 `O_F_ext_hat_K` 代替，也不能视为左右指尖力测量；运行记录会保存
实际送入的二元输入。`"zeros"` 只适用于无接触基线测试。

0809 Direct-Action RMA Student 只输入 `wrist_rgb[224,224,3]`、`proprio_obs[15]` 和
`action_history[4]`，没有运行时接触力或其他 privileged input。部署层按 metadata kind
`tacex_rma_direct_action_student_torchscript` v1 单独校验 SHA、输入顺序、无接触输入、模型契约、
历史动作、相机 crop 和四维动作适配；streaming 路径仍使用机器人基座系 XYZ。每个 checkpoint 的
metadata 必须记录独立的 CPU/GPU 一致性验证，严格校验才允许 `cuda:0` 部署：0010000 使用九组仿真
rollout 输入（最大绝对误差 `0.000737`），0040000 使用四个 episode 的 12 组输入（最大绝对误差
`0.000704`），0060000 使用同样的 12 组输入（最大绝对误差 `0.000450`），0100000 使用同样的
12 组输入（最大绝对误差 `0.000749`）；四者容差均为 `0.001`。
动作尺度暂与其共用 teacher 的 0809 配置保持一致；direct metadata 未携带完整训练环境动作契约，
部署前仍须从训练/export manifest 复核。

0809 X040-Wide Direct-Action Student 使用独立 metadata kind
`tacex_rma_x040_wide_direct_action_torchscript` v1。它同样只接受 RGB、本体状态和物理 action history，
训练时 cube XYZ 仅用于 Teacher 与辅助 position loss，绝不能作为真机输入。契约固定动作尺度
`[0.05,0.05,0.05,0.01]`、训练初始关节状态、D435 crop `(100,34,400,398)`，并要求真机工作空间包含
训练 reset 方块范围 `x=[0.32,0.48]`、`y=[-0.10,0.10]`、`z=0.026`；这只是覆盖校验，不会扩展工作空间。
该 checkpoint 使用 12 组离线 rollout 输入完成 CPU/GPU 一致性验证，最大绝对误差 `0.000722`，容差 `0.001`。
若只用 `forward_with_position()` 做相机/AprilTag 位置诊断，checkpoint metadata 仍必须携带其训练时
`normalization.cube_position_center`、`cube_position_scale` 和 `position_frame`，以便把归一化输出还原为
`robot_root` 米单位；这不等价于完成可进入真机部署的完整契约验证。

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

统一入口还负责操作者错误呈现：默认把部署异常归类为中文的故障含义和安全处理建议，并保留未改写的原始错误；
`--debug` 才会重新抛出完整 Python traceback。该层只改变终端诊断文本，不修改动作、频率、看门狗、初始状态
门禁或 native worker 通信协议。server9 动作锁存超时的原始错误额外包含等待时长、worker 状态/错误码、控制
周期数、IK tick 数和当前动作已执行 tick 数，便于区分 worker 调度问题与策略推理问题。

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

### 4.4 HIL Residual BC 运行时扩展

统一 CLI 的 `--hil` / `--hil-speed-m-s` 只形成 `HILSettings` 运行时对象，不写入既有部署 JSON。
该模式仅允许 `server9_joint_position` streaming；blocking、async backend 和 `--streaming-check`
在连接硬件前拒绝。`--validate-only` 会验证这些组合但不创建键盘窗口；Pygame 只在真正进入 HIL
preview 或控制会话时由 `franka_sim2real/hil.py` 懒加载。

Pygame 独立线程仅维护焦点和 `Space/W/S/A/D/R/F` 按键状态，不持有机器人对象，也不发送命令。
主策略循环在实际 30 Hz 发送边界读取一次快照，因此保留下一步 policy 预计算时，不会把 Space 状态
提前一拍。焦点丢失会清空全部按键；Escape 或窗口关闭通过主循环现有异常清理路径请求安全停止。

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

`GelSightPairCamera` 通过 OpenCV 打开左右两个 GelSight Mini（3280x2464@25），仅在配置中
`tactile_camera.enabled` 为 true 时启用。默认严格使用配置的 `left_device/right_device`。

统一 CLI 的 `--auto-gelsight` 只在本次运行中设置 `auto_discover=true`。相机创建前，
`gelsight_devices.py` 枚举 `/sys/class/video4linux/video*`，要求设备名称包含 GelSight 且 UVC
`index=0`，从而跳过 RealSense、普通 webcam 和同一 UVC 设备的辅助流。恰好两路时按当前 video 编号
排序绑定为 left/right，并打印编号、label、serial；零路、一路或多于两路均 fail closed。解析后的实际
`left_device/right_device` 会进入该次 server9 运行产物的 `config.json`。自动选择不能识别物理安装方向，
设备换边后必须先运行 `scripts/camera/gelsight_start.py` 预览确认。

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

HIL 下 Actor 仍在每步执行，得到 `base_action[4]`。键盘方向先归一化，再按
`hil_speed_m_s / policy_frequency_hz` 得到基座系米增量，除以既有 XYZ scales 后成为限幅前归一化
`human_xyz`。Space 按住时选择 `[human_xyz, base_action[3]]`，否则选择 base action。Residual BC 标签
定义在同一限幅前归一化层：介入时为 `human_xyz - base_action[:3]`，未介入时为零。

### 6.1 独立 Residual BC 训练

`franka_sim2real/residual_bc.py` 与 `scripts/training/train_residual_bc.py` 是纯离线模块，不被现有部署
runner 导入。数据扫描以 `policy_action_accepted` 为唯一执行过滤条件，并逐 NPZ 校验 human-base 标签关系。
数据切分单位是完整 episode，避免连续视觉帧跨 train/validation/test 泄漏。

冻结的 0814 TorchScript 通过 `encode_visual`、`tactile_encoder` 和 `normalizer` 构造原 action head 的
1043 维输入，再与采集时保存的 `base_action[4]` 拼接。训练网络为 `1047→256→128→3`，只输出
pre-limit normalized XYZ residual；base Student、gripper、安全限幅和控制链均不属于训练参数。特征抽取
阶段还会用原 `action_head` 重算 base action 并与日志比对，checkpoint 则记录 base-model SHA256，防止
策略错配。为复现部署数值，DataLoader 可以批量读取 NPZ，但 TorchScript 特征推理固定为 batch=1；
这避免 CUDA CNN 在大 batch 下选择不同 kernel 后造成可观测的 action/feature 漂移。训练产物默认仍是
独立 artifact；需要通过下一节的显式 runtime 参数才能进入 streaming 部署路径。

### 6.2 Residual BC server9 部署

`ResidualDeploySettings` 是 CLI 创建的临时运行时对象，不进入部署 JSON。`ResidualPolicyRuntime` 加载
TorchScript head 和训练 metadata，强校验 kind/version、1043+4→3 契约、base gripper 来源及训练记录的
base-model SHA。当前只允许 0814 GelSight size-buckets Student。

`BundleTorchScriptPolicy.predict()` 先执行原 base forward，再用相同的 batch=1 输入张量通过公开的
visual/tactile encoders 和 normalizer 复现 actor feature。Residual head 输出先乘 runtime scale、做逐轴
cap，再执行 `final_xyz=base_xyz+applied_residual_xyz`；第 4 维直接复制 base gripper。返回后的 final raw
action 不走任何旁路，仍由 server9 的 `clip_streaming_action`、history、workspace、DLS 和 FCI 处理。

Residual 推理信息进入每步 JSONL/NPZ；deadline miss 不更新 history。CLI 禁止 residual 与 HIL 等含义不清
的动作选择组合，真机模式还要求独立 enable flag 且禁止无交互 `--yes`。默认不开启时 policy 构造、推理、
日志和依赖路径保持原状。

### 6.3 0912 Frozen-Encoder Direct BC

`real_rlpd/direct_bc.py` 与 `scripts/training/train_0912_direct_bc.py` 从 0912 policy-specific expert Replay
读取 `state[:1043]` 和最终 `executed_action[4]`。Replay contract 必须绑定
`gelsight_reference_progress_three_frame_v1`、准确的 base model SHA、1043D feature 和 4D action；动作
超过现有 `±0.1` commissioning limit 时失败即拒绝。数据按完整 episode 分为 train/validation/test，并在
validation/test 中各保留成功和未成功 episode，避免连续帧泄漏。默认保留专家动作的真实采样分布；可选
`--balanced-sampling` 才按 hold、XYZ-only、gripper-only、XYZ+gripper 四类反频率采样。

BC head 是 `1043→256→256→4`，以 `executed_action/0.1` 为标签使用 Smooth-L1；feature normalizer 只由
train episode 拟合并冻结。导出的 `FrozenEncoderDirectBCPolicy` 内嵌原 0912 encoder，一次前向计算三帧
视觉、双 GelSight、proprio/history feature；新的 action head 输出经 `tanh×0.1` 成为四维 direct action，
原 base action 不参与组合。原 position/contact 辅助输出继续供 rollout 日志使用，不驱动动作或终止。

部署使用独立 `configs/e2e_bundle_real_exported_0912_direct_bc.json` 和
`scripts/policy/run_exported_0912_direct_bc.py`。metadata 的 `deployment_variant=frozen_encoder_direct_bc`
触发额外 provenance 校验，固定 feature/action 维度、base/replay SHA 关系和 `0.1` action limit。之后仍经过
标准 `clip_streaming_action`、动作尺度、workspace、DLS、碰撞阈值和 server9 worker；不新增运行时旁路，
共享内存 ABI 不变。

Direct BC 在 RTX 5060 上会分别为首次全零与首次非零图像触发 CUDA/TorchScript 惰性初始化。因此
`warm_up_bundle_policy()` 在首个计时 policy boundary 和控制接管前，对该 deployment variant 执行
`0/127` 两组确定性图像预热；普通策略仍仅预热全零输入。预热耗时记录到
`timing.policy_warmup`，不计入实时 deadline，也不通过放宽 watchdog 掩盖首帧初始化。

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

server9 的夹爪走 `ProcessGripperQueue`：独立 `spawn` 进程独占 Hand API，避免 native binding 持有 GIL 时
阻塞 30 Hz Python 策略循环。普通非 server9 streaming 仍使用线程版 `AsyncGripperQueue`。两者都不进入
1 kHz C++ 控制回路。

策略结果带时效性检查：`policy_result_is_timely()` 结合 `policy_watchdog_s`（0.25 s）判断，超时的结果被丢弃，worker 端的 generation 编号因此允许跳号。

HIL 选择动作复用完全相同的 `clip_streaming_action`、commissioning limit、action scales、workspace、
DLS 和 FCI 路径。仅当 worker 按期接受动作时才执行
`ActionHistoryBuffer.update(selected_raw, selected_limited)`；因此下一次模型输入中的 processed
history 是最终限幅后的实际接受组合动作。deadline miss 令机械臂继续 hold/brake，且不更新 history；
夹爪则保持最后一个已接受的速度意图，避免把单帧推理抖动误判成松键。只有超过
`policy_watchdog_s` 仍没有接受新动作时，夹爪才进入 hold。

`scripts/diagnostics/test_franka_gripper_latency.py` 是独立的 Hand 时序诊断入口。默认只校验并打印参数，
只有显式 `--run-hardware` 且人工输入 `y`/`yes` 后才构造 `franky.Gripper`；它从不构造 `Robot`，因此不连接
或移动机械臂。真机测试将每条夹爪运动限制在 10 mm、0.05 m/s 内，可分别测有界 move 中的同步
`stop()` 和有限目标自然完成，并把原始耗时保存到 `runs/gripper_latency/`。该入口用于定位 Hand/franky
延迟，不属于策略控制链路，也不改变 `ProcessGripperQueue` 行为。

`native/franka_gripper_latency.cpp` 提供同条件的原生对照：它链接 `franky-control 1.1.3` 随附的
`libfranka 0.17.0`，由一个 C++ 线程阻塞执行 `Gripper::move()`，主线程延时后在同一个线程安全对象上调用
`Gripper::stop()`。因此与 Python 报告对比时保持 Hand、libfranka 版本、目标和速度不变，只去掉 franky
future/包装层。`scripts/build_franka_gripper_latency.sh` 构建后自动运行纯离线 self-test 与 dry-run；原生
程序仍需显式 `--run-hardware` 和 `y`/`yes` 才连接 Hand，使用相同的 10 mm/0.05 m/s 硬上限。

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
3. **初始状态门禁**（`evaluate_initial_state`）：关节角、关节速度、TCP 平移与朝向、夹爪宽度、机器人模式为 `Idle`、无 FCI 错误、夹爪未夹持。`enforce: true` 时不满足直接拒绝启动。门禁失败的正确处理是用 `scripts/robot/go_to_zero_pose.py` 回到训练初始姿态，不是放宽容差；该脚本在移动前拒绝 `max_width <= 0` 的未 homing 夹爪，必须先显式运行单独的 gripper homing。回零默认使用一条连续 `JointMotion`，避免多段轨迹在 motion generator 边界重新起停；显式 `--staged` 模式则在每段后确认实测关节速度已归零才发送下一段。
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
| `rollout.jsonl` | 逐步 observation、action、timing；0809 XY RMA 额外记录实际 `contact_force_n`；Direct-Action 的该字段为 `null`；oracle 模式下还含 `model_input.rma_actor_input` |
| `summary.json` | 运行汇总 |
| `rgb/step_XXXX.png` | 模型实际看到的 RGB |
| `control_trace.jsonl` | streaming 专有，逐周期控制 trace |
| `timing_summary.json` | streaming 专有，时序统计 |
| `step_data/step_XXXX.npz` | `--hil` 强制生成；精确 RGB/视觉历史、GelSight/reference、proprio、action history 与 HIL 标签 |

HIL 的 `rollout.jsonl` 仍沿用既有逐步记录，仅追加 `episode_id`、`step_id`、`intervention`、
`base_action`、可空 `human_action` 和 `residual_target_xyz`。既有 `raw_action` 表示最终选择的限幅前动作，
`limited_action` 表示限幅后候选，`executed_action` 只在 worker 接受时非空；它是被接受的控制命令，
不是测量到的 TCP 实际位移。NPZ 同步保存标签，非介入步不创建 `human_action` 数组，并保存
`policy_action_accepted` 供训练过滤 deadline miss。不开 `--hil` 时 JSONL 和依赖加载行为保持原样。
HIL preview 没有 worker 动作发送，其样本也明确记录 `policy_action_accepted=false`；仅保留候选动作与
Residual 数据流用于检查。

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
record_cube_pose_dataset.py（只采集 D435 静态 RGB burst，不连接 Franka）
        │
        ▼
evaluate_policy_vs_apriltag.py（离线将 AprilTag 位姿与策略视觉头预测对比，输出 RMSE/中位/P95/偏差）
```

`RMAPolicyCubePredictor` 仅用于这一只读诊断链路。除旧 RMA Student 的视觉 adaptation head 外，也支持
X040-Wide 的 `forward_with_position(wrist_rgb, proprio_obs, action_history)`：该导出方法的 position
branch 实际只读取 RGB，评估器仅提供零值本体/历史张量以满足 TorchScript 签名，并丢弃 action 输出。
还原后的 XYZ 使用 metadata 中 `cube_position * scale + center`，坐标系为 `robot_root`。X040-Wide
没有 contact head，CSV 使用 `NaN`，摘要明确 `contact_prediction_available: false`。上述输出绝不能进入
部署 observation 或控制路径。

流程细节见 [`HAND_EYE_CALIBRATION.md`](HAND_EYE_CALIBRATION.md)。首次使用必须先跑 `--camera-check` 和 `--dry-run`。

除几何标定外，`measure_camera_noise.py` 标定相机的辐射特性，供仿真侧对齐图像噪声：

```text
measure_camera_noise.py（场景静止，逐 (exposure, gain) 工况连拍）
        │  复用 e2e_bundle 的 RealSenseRGBCamera 与 _apply_camera_crop，
        │  默认在模型输入域（crop -> 双线性 224x224）统计，与仿真加噪的域一致
        ▼
逐像素时间统计 -> 扣除整帧亮度漂移 -> 按亮度分箱取 sigma 中位数
        │
        ▼
runs/<时间戳>_camera_noise/{noise_report.json, noise_lut.csv, mean_*.png, sigma_*.png}
```

输出的 `sigma_lut` 与 `affine_fit`（`sigma^2 = a*mu + b`，a 为散粒噪声项、b 为读出噪声项）供 `TacEx/` 侧把同方差高斯噪声替换为信号相关噪声。该脚本只读相机，不连接机器人。模型输入尺寸从 bundle config 的 `model.metadata_path` → `input_signature.wrist_rgb` 解析，不硬编码。

### 12.1 Real-world Residual SAC

Real-RL 是 server9 的可选旁路，不改变 native worker 或安全控制：

```text
0814 frozen feature[1043] + base_action[4]
        ├─ Actor → unit residual XYZ → ±2 mm 映射 → base XYZ 相加
        │                                      ↓
        │                       既有 limit/workspace/DLS/FCI
        └─ Replay + AprilTag relative XYZ/height → privileged Q1/Q2（仅离线训练）
```

`collect` 启动前用 AprilTag 完成静止高度预检，随后关闭检测线程。每个真实 policy deadline 到达时，
控制路径直接快照 `LatestFrameCamera` 的最新 raw packet；该 packet 独立于上一周期提前计算的模型输入帧，
从而使下一边界帧通常位于上一动作之后。30 Hz 路径同时向 SQLite WAL 后台 writer 提交尚未标注的
transition skeleton。
机械臂 worker 停止后才压缩原始帧、逐帧离线检测（上一帧 ROI 放大优先、整帧 fallback）并原子回写奖励。
唯一训练门是 Replay v2 的
`trainable/trainable_reason`，Normalizer、high-watermark、UTD 和 sampler 均查询同一字段。
相机帧与命令使用 monotonic `capture_timestamp/action_timestamp`，只有 accepted 且满足
`tag_t <= action_time < tag_t1` 的 transition 可训练；Tag 缺失、过期、复用或不包围 action 时仍记录。
Deadline miss 不更新 history，且不进入训练。`train` 不连接硬件，Actor state 为 1047D，Critic 额外读取
relative XYZ＋height 4D privileged state；height 明确定义为
`object_z_in_base - initial_object_z_in_base`。

Normalizer 固定由最早 1000 条 trainable Replay 拟合：只经验归一化 policy feature[1043]，base
action[4] 按固定 contract 原样（仅 contract clip）传入，privileged[4] 单独经验归一化。首次训练使用固定
`bootstrap_updates`，后续才以新增 trainable transition 数乘 UTD（默认 2，限制 1–4）。随机 warmup
使用有时间相关性的 AR(1) Gaussian。Replay 还保存 safety 前后 action、是否介入及其 4D normalized
L2 幅度。checkpoint 校验失败时 residual=0 且默认在 worker 启动前拒绝 episode；只有显式 fallback
开关才允许继续采 base-only 数据。

server9 夹爪使用单 owner 子进程：子进程内才构造 `franky.Gripper`，`move_async/grasp_async`、future
`wait/get`、hold 后的 `gripper.state` 和同步 `stop` 都在该进程串行执行。`franky 1.1.x` 的
`stop_async` 会先等待当前异步动作，不能用于抢占；因此 owner 按官方中断模式直接调用 `stop()`。反向命令
在 stop 返回后立即启动，不等待只用于 hold 重定位的状态读取。policy `command()` 只把带 generation
的最新目标宽度写入共享内存，`poll()` 只检查本地缓存错误；因此 native Hand 调用即使持有 GIL，也只会
阻塞 owner 子进程。物体阻挡 close 后自动转换为 force grasp 的状态机不变，停止时先停机械臂 worker，
再通知 Hand owner 在同一子进程执行 stop，避免并发调用。

单个 RealSense pipeline 同时保留原始 640×480 packet 给离线 AprilTag，并将既有 crop/resize 224×224
交给策略。Replay 通过 `run_dir/rollout_step` 关联现有 RGB、原始边界帧、GelSight、JSONL 和 step NPZ。
Real-RL 数据根目录是 `real_rl_logs/`，不与通用部署的 `runs/` 混放。

### 12.2 多策略 RLPD 4D residual

RLPD 与 12.1 的旧 XYZ Residual SAC 完全分开，当前通过 adapter 支持四种 frozen base policy：0814
单帧 GelSight、0911 Progress 单帧 GelSight 的 `encode_visual()`，以及 0823 和 0912 Progress 三帧
GelSight 的 `encode_visual_features()`。四条路径都在
原模型的 `action_head` 之前得到完全相同的 feature contract：

```text
wrist visual[512] + left tactile[256] + right tactile[256]
  + normalized proprio[15] + normalized history[4] = frozen feature[1043]
                                    │
base policy raw[4] -> commissioning limiter -> base_limited[4]
                                    │
                                    ├─ Actor state[1047] -> unit residual[4]
                                    │                         × (2 × commissioning_limit)
                                    └───────────────────────── +
                                                              │
                            最终既有 clip/workspace/DLS/FCI <-┘
```

`2 × commissioning_limit` 使任意两个已限幅 4D 动作之差都能表示，因此 expert 模式可以让人工 XYZ＋
gripper 完全接管，同时保留 base action 作为 shadow 输入。组合发生在 base commissioning limiter 之后，
候选动作随后再次经过相同的最终 limiter；RLPD 路径禁止 full-scale。XYZ 仍是机器人基座系，物理尺度仍为
`[0.05,0.05,0.05] m`，gripper delta-width 尺度仍为 `0.01 m`。初始状态门禁、碰撞阈值、watchdog、
停止时序、native worker 和 ABI-8 均未改变。

RLPD config schema v2 先按策略日期建立命名空间，再拆成策略无关格式的专家源、policy-specific 派生
expert Replay，以及 online rollout 三层：

```text
real_rlpd_data/<policy>/expert/<episode>/        # 专家源 episode
real_rlpd_data/<policy>/derived/offline_expert.sqlite3
real_rlpd_data/<policy>/rollout/<episode>/       # rollout episode
real_rlpd_data/<policy>/rollout/online_policy.sqlite3
```

专家源由 `expert_dataset.py` 写入，保存同步的当前腕部 RGB、左右 GelSight 当前帧与每回合固定参考帧、
proprio、action history、机器人观测，以及基座系 XYZ＋gripper delta-width 的专家绝对物理动作。源契约
明确排除 base feature、base action 和 residual target；shadow model 的 kind/SHA 只作为采集 provenance。
因此同一任务的专家源可以由另一 adapter 重新编码。采集时仍同步写入当前 base-policy 的 offline SQLite，
它是为现有 trainer 准备的派生缓存；当前尚无从历史源 episode 批量重建该缓存的独立 CLI。
通用 rollout JSON 在 expert takeover 时记录 `expert_requested_action`、`expert_limited_action` 和按键快照，
不写 `unit_residual_action`/`residual_normalized_action`；后二者只属于 online residual policy 记录。

Replay schema v1 继续按 base-policy contract 隔离派生 expert 与 online policy 两个 SQLite 文件。契约包括
adapter ID、policy kind、模型 SHA、metadata SHA、1047D state、4D residual、动作尺度和 commissioning
limit；不同策略、不同模型或不同限幅不能直接共用派生 Replay/checkpoint。每条 transition 保存当前/下一
state、base/executed/residual action、TCP 位置、action timestamp 及离线 reward 字段。控制期间 collector
只缓存 transition 和专家源数组；worker 和 Hand owner 停止后才写源 episode，并以单个事务写库，避免 I/O
进入 30 Hz 路径。原始 D435 边界帧沿用 Real-RL 的离线 AprilTag 流程，只有 accepted 且满足
`tag_t <= action_time < tag_t1` 的样本标为 trainable。
操作者在启动确认处取消，或 episode 没有写入任何 transition 时，CLI 将其作为空采集正常结束并跳过
AprilTag 离线标注；空 run 没有原始边界帧，也不会被误报成旧版 224×224-only 数据。
标注器通过 `RewardFunction` 接口调用由 `reward_kind` 选择的实现。旧的
`apriltag_reach_lift_success` 保持原 reach/lift 增量及 residual-action penalty；0912 可选的
`apriltag_x040_observable_absolute_v1` 使用仿真同权重的绝对归一化 reach/lift、一次性成功奖励、最终执行
动作幅值/变化、掉落和工具最低点桌面间隙。后者用 `+17.1 mm` 将最低点 TCP 转为 GelSight 接触面中点；
contact force、非法碰撞分类和 upright gate 因当前真机标签不可观测而不伪造。reward 参数属于
Replay/checkpoint contract，公式或阈值不同的数据和 checkpoint 不得混用。
collector 在 AprilTag preflight 前校验 Replay contract，不兼容时不会等待检测或启动控制。

PyTorch learner 是 asymmetric SAC：Actor 只看 1047D state；10 个 Q 都额外看 object-relative XYZ＋height
4D privileged state，Critic 隐层带 LayerNorm。每次 target 随机选择 2 个 Q 取 minimum，online 数据存在
时 batch 严格按 offline/online 50:50 采样。一个 update group 包含 UTD=20 个 Critic update，随后只做
一次 Actor 和 temperature update。首次训练默认 1000 group；后续只在 episode 已结束且离线标签完成后，
按 checkpoint high-watermark 之后新增的 trainable online transition 更新。Normalizer 首次由 expert
offline 数据拟合并随 checkpoint 冻结。部署 checkpoint 的模型/metadata/action contract 不匹配时，在
worker 启动前 fail closed；随机 residual 还要求单独的显式开关。
trainer 默认每 10 个 update group 输出当前进度、累计 group、elapsed/ETA 和 Critic/Actor/temperature
指标；`--progress-interval` 只控制日志频率，不进入算法、Replay 或 checkpoint contract。

`configs/real_rlpd_0814.json`、`configs/real_rlpd_0823.json`、
`configs/real_rlpd_0911_progress.json` 与 `configs/real_rlpd_0912_progress.json` 分别使用 `0814/`、`0823/`、
`0911/`、`0912/` 数据命名空间；其 `expert/`、`derived/`、`rollout/` 和
`checkpoints/real_rlpd/<policy>/` 相互隔离，不会迁移或读取旧
`real_rl_logs/replay.sqlite3` 或 `real_rlpd_runs/`。`rlpd/` 只是用户引入的参考实现，生产路径不 import 它。

0911 另建 `tacex_rma_gelsight_size_buckets_progress_student_torchscript` kind 和
`gelsight_reference_progress_single_frame_v1` adapter ID，避免与 0814 的 Replay/checkpoint 合并。其 metadata
记录 Progress reward/terminal-success、GelSight v7 几何和真实历史帧行为审计。审计中的动作饱和与
辅助接触输出塌缩告警继续保留；操作者确认饱和符合简单仿真任务预期后，运行时将其标记为
`direct_and_rlpd`，允许标准 base-policy 部署与 RLPD。RLPD 专家模式是四维人工全接管；online 模式必须
先通过 residual checkpoint 的 model/metadata/reward 完整契约校验。其离线成功边界为抬升 `0.035 m`
且连续检测 5 帧，派生 Replay 与 0814/0823 均不兼容。
metadata 将训练时的 150 policy-step horizon 记录为 provenance，不将其用作直接部署或 RLPD 的运行时
步数上限。RLPD 的 AprilTag 检测仍是停止后离线标注，所以提前成功需要操作者或外部监控停止，不使用
已发生塌缩的辅助 contact logits 作为终止信号。

0911 裸 policy 复用标准部署 CLI，运行步数由 `--steps` 设置。动作尺度、`±0.1` commissioning limit、workspace、
初始状态门禁、碰撞阈值、坐标系和首次 proposed action 人工确认均未改变。`±0.1` normalized envelope
对应每步单轴最大 `5 mm` XYZ 和 `1 mm` gripper-width 增量；由于裸部署没有闭环成功检测，运行期间由
操作者负责在成功或异常时停止。

0911 的 GelSight v7 最低点距 `panda_hand` 为 `0.1563 m`，减去 `O_T_EE` 已包含的 `0.1034 m` 后，
workspace/日志附加偏移为 `tool_tcp_offset_ee_m.z=0.0529 m`。该值不改变 IK 命令点或动作坐标系。

0823 的工具最低点距 `panda_hand` 为 `0.1613 m`；`O_T_EE` 已包含 `0.1034 m` 的 flange-to-EE 平移，
因此 workspace/日志使用的附加 `tool_tcp_offset_ee_m.z` 是 `0.0579 m`。该偏移不改变 IK 命令点或策略
动作坐标系。0823 workspace z 下界现为 `0.01 m`，与测试和 RLPD 真机门禁一致。

0912 X040 Progress 三帧 Student 继续复用 0823 的三帧缓存、双 GelSight 固定 reference 和三输出
TorchScript 路径，但通过 metadata `task` 与旧的无 task 产物区分几何。0912 使用当前 v7 的
`0.1563 m` 最低点和 `tool_tcp_offset_ee_m.z=0.0529 m`；旧 0814/0815/0823 三帧产物继续使用
`0.0579 m`。辅助接触概率和方块根坐标仅记录到 rollout，不参与动作或自动终止；150 步是训练
horizon provenance，直接部署可覆盖步数并由操作者负责停止。

0912 RLPD 在相同三帧 feature API 上注册独立的 `gelsight_reference_progress_three_frame_v1` adapter，
与 0823 的 `gelsight_reference_three_frame_v1` 分离；派生 Replay、online Replay 和 checkpoint 使用
`0912_progress` 命名空间。reward/离线成功标注沿用 Progress 契约的 `0.035 m` 抬升阈值和连续 5 次检测，
不会读取 0823 或 0911 的 policy-specific Replay/checkpoint。

`configs/real_rlpd_0912_progress_observable_absolute.json` 是独立实验契约，不覆盖上述旧配置；它使用
`offline_expert_observable_absolute_v1.sqlite3`、`online_policy_observable_absolute_v1.sqlite3` 和
`checkpoints/real_rlpd/0912_progress_observable_absolute_v1/`。`rebuild-reward-replay` 可校验除 reward 外
完全相同的 policy/action contract，复制旧 Replay 的派生 state/action 后清空旧标签，再从各 episode 的
原始 D435 边界帧重新计算 privileged state 和新 reward；目标存在时拒绝覆盖。
缺少 episode 目录或原始边界帧的历史记录保留为 non-trainable 并报告 `skipped`，不伪造新 reward。

`scripts/diagnostics/compare_sim_real_images.py` 是纯离线的外观对齐分析，不碰硬件：

```text
真机：runs/<部署运行>/rgb/ 或 runs/<时间戳>_camera_noise/mean_*.png
仿真：TacEx collect_rma_student_rollouts.py 的 episode NPZ（键 wrist_rgb，[T,224,224,3] uint8，已过 DR）
        │  两侧都必须是模型输入域，尺寸不同直接报错
        ▼
亮度直方图 / 分位数 / 明暗分区通道平衡 / Wasserstein-1 与 KS 距离
        │
        ▼
runs/<时间戳>_sim_real_image_stats/{image_stats.json, comparison.png}
```

Wasserstein-1 的单位是 DN，含义是"仿真整体平均需要平移多少 DN 才能对上真机"。两份同场景真机数据之间的 W1 约 10 DN，可作为该指标的本底。

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
