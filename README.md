# Franka Control and Sim2Real Toolkit

Franka 真机控制、RealSense 图像采集、TorchScript policy 部署和基础诊断工具。

> 真机运行前请清空工作空间、准备急停，并先使用 `--preview-only` 或单步确认。

## 目录

- [快速开始](#快速开始)
- [常用命令速查](#常用命令速查)
- [常用机器人操作](#常用机器人操作)
- Policy 部署
  - [0809 Direct-Action Policy 部署](#0809-direct-action-policy-部署)
  - [0809 X040-Wide Direct-Action Policy 部署](#0809-x040-wide-direct-action-policy-部署)
  - [0809 XY Policy 部署](#0809-xy-policy-部署)
  - [0726 Policy 部署](#0726-policy-部署)
  - [0801 Policy Streaming 部署](#0801-policy-streaming-部署)
  - [HIL Residual BC 数据采集](#hil-residual-bc-数据采集)
  - [0802 比例控制律部署（sim_actuator_velocity）](#0802-比例控制律部署sim_actuator_velocity)
  - [Policy 真机测试（0711 默认入口）](#policy-真机测试0711-默认入口)
- [相机和对齐](#相机和对齐)
- [常用诊断](#常用诊断)
- [重要配置](#重要配置)
- [输出日志](#输出日志)
- [0912 Frozen-Encoder Direct BC](#0912-frozen-encoder-direct-bc)
- [目录结构](#目录结构)
- [安全原则](#安全原则)

## 快速开始

```bash
cd /home/td/franka
source .venv/bin/activate
export FRANKA_ROBOT_IP=172.16.0.2
```

主要依赖包括 `franky`、`pylibfranka`、`torch`、`pyrealsense2`、`numpy` 和 `pillow`。
`.venv`、模型权重和运行日志不会提交到 Git，需要在本机单独准备。

检查主机和机器人连接：

```bash
bash tools/system/verify_franka_host.sh
bash tools/system/check_franka_network.sh 172.16.0.2
python scripts/robot/read_franka_state.py --ip 172.16.0.2
```

## 常用命令速查

下面按任务归纳最常用的命令，详细说明和参数见后续对应章节。所有命令默认在
`/home/td/franka` 下、已 `source .venv/bin/activate` 且已 `export FRANKA_ROBOT_IP=172.16.0.2`。

### 环境与连接检查

| 目的 | 命令 |
| --- | --- |
| 检查主机 realtime/权限 | `bash tools/system/verify_franka_host.sh` |
| 检查网络到机器人 | `bash tools/system/check_franka_network.sh 172.16.0.2` |
| 读取一次机器人状态 | `python scripts/robot/read_franka_state.py --ip 172.16.0.2` |

### 机器人基础操作

| 目的 | 命令 |
| --- | --- |
| 连续读取状态 | `python scripts/robot/read_franka_state.py --ip 172.16.0.2 --count 0 --interval 1` |
| 回到 policy 初始姿态 | `python scripts/robot/go_to_zero_pose.py --ip 172.16.0.2 --realtime ignore --speed 0.3 --transit-joint-impedance 1500 1500 1500 1200 1200 1000 1000` |
| 松开夹爪 | `python scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --gripper-open --gripper-speed 0.03` |
| 小幅 Cartesian 位移 | `python scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dx 0.01 --speed 0.02 --realtime ignore` |
| 键盘遥操作并采集真机轨迹 | `python scripts/robot/teleop_record_real_trajectory.py --ip 172.16.0.2` |

### Policy 部署（通用验收顺序）

先离线校验，再逐级放开，每一步确认无误后再进入下一步。把 `<runner>` 换成下表中对应日期的入口：

```bash
python scripts/policy/<runner> --validate-only          # 1. 不连接硬件的契约校验
python scripts/policy/<runner> --steps 1 --preview-only  # 2. 只读状态和相机，不发送运动
python scripts/policy/<runner> --streaming-check         # 3. streaming 入口的额外时序检查
python scripts/policy/<runner> --steps 1                 # 4. 单步真机（需手动确认，勿加 --yes）
python scripts/policy/<runner> --steps 15 --confirm-each-step  # 5. 多步逐步确认
```

部署入口遇到配置、相机、GPU、初始状态或 streaming 控制异常时，会打印“中文说明”“建议处理”和未改写的
“原始错误”。需要保留完整 Python 调用栈供开发排查时，在原命令末尾加 `--debug`；这不会改变控制参数或
安全门禁。

| 部署版本 | 入口脚本 | 说明 |
| --- | --- | --- |
| 0711（默认） | `run_exported_0711.py` | 基础 blocking 部署入口 |
| 0726 | `run_exported_0726.py` | 严格训练契约动作尺度 |
| 0801 | `run_exported_0801.py` | server v9 原生 streaming backend |
| 0802 比例控制律 | `run_exported_0802_dr_simactuator.py` | `sim_actuator_velocity`，速度与动作成正比 |
| 0802 GPU | `run_exported_0802_dr_simactuator_gpu.py` | portable v6 DR，`cuda:0` |
| 0803 heatmap | `run_exported_0803_dr_heatmap.py` | 需 `--model` / `--metadata` |
| 0809 Direct-Action | `run_exported_0809_direct_action.py` | 仅 RGB、本体状态和动作历史 |
| 0809 X040-Wide Direct-Action | `run_exported_0809_x040_wide_direct_action.py` | X040-Wide XYZ、无接触输入 |
| 0809 XY | `run_exported_0809_xy_only.py` | XY RMA，需运行时抓取状态 |
| 0823 GelSight | `run_exported_0823_gelsight.py` | 修正 161.3 mm 夹爪几何的三帧 RGB＋双 GelSight Student |
| 0912 GelSight Progress | `run_exported_0912_gelsight_progress.py` | 当前 156.3 mm 夹爪几何的三帧 Progress Student |
| 0912 Frozen-Encoder Direct BC | `run_exported_0912_direct_bc.py` | 冻结 0912 encoder、直接模仿专家 4D 动作 |

> `--streaming-check` 仅 streaming 配置支持（包括 0801/0802/0823/0912）；`0726` 只有 `--validate-only` / `--preview-only` / `--steps`。

0823 GelSight 策略使用三帧腕部 RGB、左右 GelSight 当前帧和每回合固定参考帧。部署配置将物理
workspace 检查点放在相对 `O_T_EE` 的 `+57.9 mm`，对应
`panda_hand→最低点 161.3 mm - F_T_EE 103.4 mm`；它只改变安全检查点，不改变 IK 控制帧。
默认关闭逐帧 RGB 落盘以降低 30 Hz 控制期间的 I/O 压力，结构化 rollout 写入
`real_policy_logs/`。首次部署必须依次运行：

```bash
.venv/bin/python scripts/policy/run_exported_0823_gelsight.py --validate-only
.venv/bin/python scripts/policy/run_exported_0823_gelsight.py --steps 1 --preview-only --auto-gelsight
.venv/bin/python scripts/policy/run_exported_0823_gelsight.py --streaming-check
.venv/bin/python scripts/policy/run_exported_0823_gelsight.py --steps 1 --auto-gelsight
```

单步运动会要求人工确认，首次验收不要加 `--yes`、`--allow-full-scale` 或提高
`--action-limit`。确认动作方向、最低点 workspace 和夹爪行为正确后，再逐渐增加步数。

0912 Progress 策略沿用三帧腕部 RGB、双 GelSight 当前帧和每回合固定参考帧，辅助接触概率和
方块根坐标只写入 rollout 日志，不参与真机控制。当前 +21 mm 夹爪的
`panda_hand→最低点` 为 `156.3 mm`，因此 workspace 检查点相对 `O_T_EE` 使用 `+52.9 mm`。
默认 150 步仅对应训练 horizon；通过 `--steps 1000` 可延长运行，但没有视觉成功自动停止，必须由
操作者持续监控。首次部署依次执行：

```bash
.venv/bin/python scripts/policy/run_exported_0912_gelsight_progress.py --validate-only
.venv/bin/python scripts/policy/run_exported_0912_gelsight_progress.py --steps 1 --preview-only --auto-gelsight
.venv/bin/python scripts/policy/run_exported_0912_gelsight_progress.py --streaming-check
.venv/bin/python scripts/policy/run_exported_0912_gelsight_progress.py --steps 1 --auto-gelsight
.venv/bin/python scripts/policy/run_exported_0912_gelsight_progress.py --steps 1000 --auto-gelsight
```

首次运动不要使用 `--yes` 或 `--allow-full-scale`；确认单步方向、夹爪动作和最低点安全检查均正确后，
再执行多步运行。

### 相机、对齐与视觉验证

| 目的 | 命令 |
| --- | --- |
| 预览相机原图/crop/模型输入 | `python scripts/camera/preview_camera_crop.py --open` |
| 实时查看 RMA 物体位置预测 | `python scripts/policy/monitor_rma_object_position.py --device cuda:0 --exposure 80 --gain 64` |
| 用真值验证位置预测 | `python scripts/policy/validate_rma_object_position.py --realsense --ground-truth 0.50 0.00 0.026` |
| 实时 AprilTag 方块位姿 | `python scripts/calibration/live_apriltag_cube_pose.py --ground-truth 0.50 0.00 0.026` |

### 诊断

| 目的 | 命令 |
| --- | --- |
| 静止外力/力矩 | `python scripts/diagnostics/test_franka_force_sensor.py --ip 172.16.0.2` |
| action 坐标方向 | `python scripts/diagnostics/diagnose_franka_directions.py --robot-ip 172.16.0.2 --cases dx+ dx-` |
| streaming 跟踪质量 | `python scripts/diagnostics/analyze_streaming_tracking.py --latest 1` |
| 对比仿真/真机图像 | `python scripts/diagnostics/compare_sim_real_images.py --real <rgb 目录> --sim <episode.npz>` |

## 常用机器人操作

读取一次状态：

```bash
python scripts/robot/read_franka_state.py --ip 172.16.0.2
```

连续读取状态：

```bash
python scripts/robot/read_franka_state.py --ip 172.16.0.2 --count 0 --interval 1
```

松开夹爪：
```
python scripts/robot/minimal_franka_move.py \
  --ip 172.16.0.2 \
  --gripper-open \
  --gripper-speed 0.03
```

回到 policy 初始关节姿态和夹爪宽度：

```bash
python scripts/robot/go_to_zero_pose.py \
  --ip 172.16.0.2 \
  --realtime ignore \
  --speed 0.1 \
  --transit-joint-impedance 1500 1500 1500 1200 1200 1000 1000 \
  --gripper-speed 0.3
```

默认使用一条连续 `JointMotion` 回到初始姿态，避免旧的多段轨迹在段间重新启动 motion generator 而触发
速度/加速度不连续 reflex。只有需要兼容旧方式时才显式加 `--staged`；该模式会在每段后确认实测关节速度
已归零，未归零则拒绝发送下一段。`--single-step` 会自动选择分段模式。

回零脚本会在**发送任何机械臂或夹爪运动前**读取夹爪宽度和 `max_width`。如果 `max_width` 为 0，
表示夹爪未 homing 或没有有效行程，脚本会拒绝执行，绝不会把 0.04 m 目标夹紧到零宽度。确认夹爪内
没有物体、手或线缆后，先单独执行 homing：

```bash
python scripts/robot/minimal_franka_move.py \
  --ip 172.16.0.2 \
  --gripper-homing
```

若只需要回到机械臂初始关节姿态，显式加 `--skip-gripper`。

执行推理

```bash
python scripts/policy/run_exported_0803_dr_heatmap.py \
  --model checkpoint/0803_dr3/rma_student_e2e_student_0040000.pt \
  --metadata checkpoint/0803_dr3/rma_student_e2e_student_0040000.json \
  --device cuda:0 \
  --exposure 100 \
  --gain 64 \
  --steps 1000 \
  --yes
```

Policy 部署入口也支持 RealSense 颜色相机控制。指定 `--exposure` 或 `--gain` 会在相机
预热前关闭自动曝光；使用 `--auto-exposure` 可恢复自动曝光，且不能与两个手动参数同时使用。

## 0809 Direct-Action Policy 部署

0809 Direct-Action RMA Student 只接收 RGB、本体状态和动作历史，不接收 `contact_force_n`；不要把
XY 版的夹爪状态近似套用到此策略。配置保持 D435 `640x480@30`、crop `(100,34,400,398)`、四维
`[dx, dy, dz, gripper]` 与既有 streaming 安全限幅。专用配置使用 `cuda:0`；已用九组仿真
rollout 输入完成 CPU/GPU 一致性验证，最大绝对误差为
`0.000737`（容差 `0.001`）。

动作尺度沿用与该 checkpoint 共用 teacher 的 0809 配置 `[0.05, 0.05, 0.05, 0.01]`；direct
metadata 本身未带完整训练环境动作契约，部署前仍应从训练/export manifest 核实这一约定。

先进行不连接硬件的契约校验：

```bash
.venv/bin/python scripts/policy/run_exported_0809_direct_action.py --validate-only
```

真机验收顺序不变：`--preview-only` → `--streaming-check` → 人工确认的 `--steps 1`。首次单步不要
加入 `--yes`，也不要使用 `--allow-full-scale`。该 checkpoint 与 XY 版共用 teacher，但不能假定两者
输出动作相同；保持 `commissioning_action_limit: 0.1` 直到完成真机单步验收。

## 0809 X040-Wide Direct-Action Policy 部署

X040-Wide Student 同样只接收 RGB、本体状态和动作历史；训练时的 cube XYZ 只是 Teacher/辅助 loss 标签，
不得在真机端传入。训练方块范围为机器人基座系 `x=[0.32,0.48] m`、`y=[-0.10,0.10] m`，已被配置中既有
安全工作空间覆盖；配置不会为此扩大工作空间。动作尺度保持训练确认的 `[0.05, 0.05, 0.05, 0.01]`，并强制
同一初始关节状态、D435 `640x480@30` crop `(100,34,400,398)` 和物理 action history。已完成 CPU/GPU
一致性验证，最大绝对误差为 `0.000722`（容差 `0.001`）。

```bash
.venv/bin/python scripts/policy/run_exported_0809_x040_wide_direct_action.py --validate-only
```

验收顺序必须为 `--preview-only`、`--streaming-check`、人工确认的 `--steps 1`，之后才可多步运行；首次单步
不要添加 `--yes`，并保持 `commissioning_action_limit: 0.1`。

## 0809 XY Policy 部署

0809 XY RMA policy 需要运行时抓取状态。部署配置使用 Franka gripper 的 `is_grasped` 作为近似：
抓住时送入 `[1, 1]`，否则送入 `[0, 0]`；这不是左右指尖力测量，不能以腕部外力替代。先进行不连接
硬件的契约校验：

```bash
.venv/bin/python scripts/policy/run_exported_0809_xy_only.py --device cpu --validate-only
```

真机验收仍按安全顺序执行：先 `--preview-only`，再 `--streaming-check`，最后在操作者确认后使用
`--steps 1`。不要为首次单步加入 `--yes`。

streaming 会每 30 步（及最后一步）在终端打印累计 accepted/deadline misses、模型输入的 RGB 尺寸、
`action_history`、`proprio_obs`、`contact_force_n`，以及 raw/实际动作和 timing；完整数值仍写入运行目录。

执行小幅 Cartesian 位移：

```bash
python scripts/robot/minimal_franka_move.py \
  --ip 172.16.0.2 \
  --dx 0.01 \
  --speed 0.02 \
  --realtime ignore
```

`minimal_franka_move.py` 还支持 `--dy`、`--dz`、`--roll`、`--pitch`、`--yaw` 和夹爪命令，执行前会要求确认。

### 键盘遥操作采集真实轨迹

下面的脚本会连续保存 D435 原图、按键动作和机械臂本体状态。所有记录以主机
`host_monotonic_ns` 为共同时间轴；结束时会写出 `alignment.jsonl`，为每个动作/状态标记
时间上最近的一帧 RGB。它采用**单次按键、小步、阻塞式** Cartesian 位移，避免按住按键产生不可控连续运动。

```bash
python scripts/robot/teleop_record_real_trajectory.py \
  --ip 172.16.0.2 \
  --step-m 0.005 \
  --speed 0.05
```

请从真实终端运行（而非 IDE 的 Output 面板），并确保急停可用。启动时会要求确认；先用
`--preview-only` 检查相机、按键和落盘。按键为：`w/s` X 正/负、`a/d` Y 正/负、`r/f` Z
正/负、`j/l` yaw 正/负、`o/c` 夹爪开/合，`p` 立即记录一条状态，`m` 插入时间标记，`q` 正常结束。
默认工作空间与真机配置一致：`x=[0.2, 0.65]`、`y=[-0.3, 0.3]`、`z=[0.0, 0.45]` m；可通过
`--workspace-min` / `--workspace-max` 修改。

每次运行写入 `runs/<timestamp>_real_teleop_trajectory/`：`rgb/`（图片）、`frames.jsonl`
（相机时间戳）、`states.jsonl`（本体状态）、`actions.jsonl`（请求与实际发送的动作）、
`markers.jsonl` 和 `alignment.jsonl`。RealSense 设备时间戳也会保留在 `frames.jsonl`，但跨流对齐应使用
主机 monotonic 时间轴，因为设备时钟与主机时钟的原点不同。

## 0726 Policy 部署

0726 的部署入口和严格契约配置已经准备好：

- 入口：`scripts/policy/run_exported_0726.py`
- 配置：`configs/e2e_bundle_real_exported_0726.json`
- 模型：`checkpoint/0726/exported/policy_actor_e2e_best_agent.pt`
- 相机：D435 `215322076207`，`640x480 -> x[80:560], y[0:480] -> 224x224`
- 动作历史：逐维缩放 `[0.05, 0.05, 0.05, 0.01]`，延迟 1 步

先执行完全不连接硬件的校验：

```bash
python scripts/policy/run_exported_0726.py --validate-only
```

再执行无运动预览；该命令会读取机器人状态和相机，但不会发送机械臂、夹爪或
homing 命令：

```bash
python scripts/policy/run_exported_0726.py --steps 1 --preview-only
```

如果初始状态门禁失败，先确认工作空间安全，再使用上面的
`scripts/robot/go_to_zero_pose.py` 回到训练初始姿态。单步真机测试：

```bash
python scripts/policy/run_exported_0726.py --steps 1
```

脚本会先打印拟执行的 Cartesian 位移和夹爪宽度，只有输入 `y` 或 `yes` 才会运动。
0726 严格使用训练契约的最大动作尺度：XYZ 每步最多 `0.05 m`，夹爪总宽度每步最多
变化 `0.01 m`，因此第一次必须检查打印值，不能直接使用 `--yes`。

单步方向和图像对齐确认后，再进行多步、逐步确认：

```bash
python scripts/policy/run_exported_0726.py \
  --steps 15 \
  --confirm-each-step
```

## 0801 Policy Streaming 部署

0801 配置默认使用兼容 robot server version 9 的原生 backend：Python 运行
30 Hz policy，原生 worker 运行 60 Hz DLS IK 和 1 kHz FCI read/write，并强制
realtime。首次使用先构建 worker（构建过程不会连接机器人；详情见
`docs/PYLIBFRANKA_STREAMING.md`）：

```bash
./scripts/build_franka_server9_worker.sh
```

按下面顺序验收；`--validate-only` 完全不连接硬件：

```bash
python scripts/policy/run_exported_0801.py --validate-only
python scripts/policy/run_exported_0801.py --steps 150 --preview-only
python scripts/policy/run_exported_0801.py --streaming-check
python scripts/policy/run_exported_0801.py --steps 1
```

默认会把归一化动作限制在 `[-0.10, 0.10]`。只有 commissioning 的方向、工作空间、
timing 和 watchdog 全部通过后，才使用 `--allow-full-scale`；该选项禁止与 `--yes`
组合，并始终要求一次人工会话确认：

```bash
python scripts/policy/run_exported_0801.py --steps 1 --allow-full-scale
```

Streaming 的 XYZ 是机器人基座坐标系，且不使用 legacy blocking 路径的 Y/Z 取反。
控制期间只缓存数据，`stop_control()` 后才写入 RGB、rollout、control trace 和 timing。

## HIL Residual BC 数据采集

采用 `[dx,dy,dz,gripper]` contract 的 server9 streaming 策略可加 `--hil`。Pygame 窗口只在
HIL 模式下加载；若环境缺少依赖，先运行：

```bash
.venv/bin/python -m pip install pygame
```

启动后聚焦 HIL 窗口并按 Enter 就绪。每个 30 Hz 边界仍先计算 base policy；按住 Space 时，
`W/S`、`A/D`、`R/F` 分别控制基座系 `+X/-X`、`+Y/-Y`、`+Z/-Z`，夹爪继续使用 base policy。
松开 Space 或窗口失焦会立即恢复 base policy，Escape/关闭窗口会请求安全停止。相反方向抵消，
斜向输入归一化；按住 Space 但不按方向键表示人工 XYZ 为零。

```bash
.venv/bin/python scripts/policy/run_exported_0814_gelsight.py \
  --device cuda:0 \
  --steps 150 \
  --auto-gelsight \
  --hil \
  --hil-speed-m-s 0.05
```

人工速度先按 `speed / policy_frequency` 转为每步米增量，再除以既有 XYZ action scales，得到
限幅前归一化 `human_action`。最终选择的动作仍统一经过 commissioning limit、action scaling、
workspace、DLS IK 和 FCI；没有绕过安全链。仅在 worker 接受该步时，用最终限幅后的选择动作更新
`action_history`，deadline miss 保持 hold 且不更新 history。

`--hil` 强制保存 `step_data/*.npz`。`rollout.jsonl` 每步增加 `episode_id`、`step_id`、
`intervention`、`base_action`、`human_action`、`residual_target_xyz`；`raw_action` 是最终选择的限幅前
动作，`limited_action` 是 contract/commissioning limit 后候选，`executed_action` 只在 worker 接受时
非空。Residual 标签是限幅前归一化层的 `human_action[:3] - base_action[:3]`，无介入时固定为零。
训练时应使用 `policy_action_accepted` 过滤 deadline-miss 样本。
HIL preview 不向 worker 发送动作，因此其 `policy_action_accepted` 同样为 `false`，不能混入执行样本。

### 独立 Residual BC 训练

`scripts/training/train_residual_bc.py` 读取现有 `step_data/*.npz`，冻结采集数据时使用的 0814
TorchScript Student，并复现送入原 `action_head` 的 1043 维融合特征：视觉 512、左右 GelSight
各 256、归一化 proprio 15、归一化 action history 4。独立 MLP 的输入是该特征与日志中的
`base_action[4]`，输出仅为限幅前归一化 `residual_xyz[3]`；夹爪仍来自 base policy。

下面的命令会自动跳过没有 step NPZ 的空 run，按 episode 名排序，将最后一条作为 test、倒数第二条
作为 validation，其余作为 train。相邻 timestep 不会被随机拆到不同数据集：

```bash
.venv/bin/python scripts/training/train_residual_bc.py \
  --runs-root runs \
  --run-pattern '20260822_23*_e2e_bundle_real_exported_0814_gelsight' \
  --base-model checkpoint/0814_gelsight/gelsight_reference_student_latest.pt \
  --output-dir checkpoint/residual_bc_0823 \
  --device cuda:0 \
  --epochs 100 \
  --batch-size 64
```

也可重复指定 `--run-dir`、`--validation-run`、`--test-run` 覆盖自动发现和切分。训练只接收
`policy_action_accepted=true` 的样本，并在优化前用冻结模型重算每个 `base_action`；误差超过
`--base-action-tolerance` 会停止，防止把某个 base policy 的 residual 标签用于另一 checkpoint。
NPZ 可按 `--feature-batch-size` 批量读取，但冻结 policy 始终按真机相同的 batch=1 抽取特征，避免
CUDA 卷积在较大 batch 下产生数值漂移并改变 Residual MLP 的输入分布。
训练默认按 normal、人工 hold、人工 moving 三组做逆频率采样，使用 SmoothL1 loss 和 early stopping。

输出目录必须为空，产物包括：

- `residual_bc_best.pt`：权重、结构、数据切分、base-model SHA 和验证契约；
- `residual_bc_best.ts`：独立 Residual MLP TorchScript；
- `metadata.json`：验证/test 指标及每组样本数量；
- `training_history.json`：逐 epoch 训练记录。

该脚本只训练和离线评估，不会连接相机或机器人，也不会自动把 residual 接入真机控制。部署时仍须先做
离线 replay，并保证 `base_xyz + residual_xyz` 之后继续经过现有完整安全链。

### Residual BC 部署接口

0814 server9 策略可用 `--residual-model` 加载独立 head。运行时先正常计算 base action，再从同一模型输入
复现 1043 维 Actor feature，得到 predicted residual；依次乘 `--residual-scale`、按
`--residual-max-abs` 做逐轴 cap，并只叠加到 base XYZ。base gripper 原样保留，组合动作随后仍统一进入
既有 action contract、commissioning limit、action scaling、workspace、DLS IK 和 FCI。

先做完全离线验证：

```bash
.venv/bin/python scripts/policy/run_exported_0814_gelsight.py \
  --device cuda:0 \
  --steps 1 \
  --residual-model checkpoint/residual_bc_0823/residual_bc_best.ts \
  --residual-scale 0.25 \
  --residual-max-abs 0.02 \
  --validate-only
```

再做有相机/机器人状态读取、但不发送运动命令的 preview：

```bash
.venv/bin/python scripts/policy/run_exported_0814_gelsight.py \
  --device cuda:0 \
  --steps 150 \
  --auto-gelsight \
  --residual-model checkpoint/residual_bc_0823/residual_bc_best.ts \
  --residual-scale 0.25 \
  --residual-max-abs 0.02 \
  --preview-only
```

真机 residual motion 额外要求 `--enable-residual-control`，禁止 `--yes`，仍会显示首步组合动作并要求
人工输入 y/yes。第一次只运行一步：

```bash
.venv/bin/python scripts/policy/run_exported_0814_gelsight.py \
  --device cuda:0 \
  --steps 1 \
  --auto-gelsight \
  --residual-model checkpoint/residual_bc_0823/residual_bc_best.ts \
  --residual-scale 0.25 \
  --residual-max-abs 0.02 \
  --enable-residual-control
```

默认 residual cap 是每轴归一化 `0.1`；当前 `residual_bc_0823` 在 normal test 上输出偏大，因此首次验收
建议显式使用上面的 scale `0.25` 和 cap `0.02`，不要直接使用默认值。Residual v1 只支持采集它的 0814
GelSight checkpoint，并校验 base-model SHA；禁止与 HIL、streaming-check、blocking、async backend、RMA
输入 override 或 `optimize_for_inference` 组合。

Residual 模式强制保存 step NPZ。`rollout.jsonl`/NPZ 记录 `base_action`、
`predicted_residual_xyz`、`applied_residual_xyz`、最终 `raw_action`、限幅后/accepted 动作、scale、cap 和
residual SHA；下一步 action history 使用真正被接受的最终组合动作。

真机验收依次执行 `--hil --preview-only`、不带 HIL 的 `--streaming-check`、`--steps 1 --hil`，
确认方向、Space 释放、日志和 history 后再运行多步。`--hil` 不支持 blocking、async backend 或
`--streaming-check`；`--hil --validate-only` 只做离线契约验证，不启动 Pygame 或硬件。

`--auto-gelsight` 会在打开触觉相机前读取 `/sys/class/video4linux`，只保留名称含 GelSight 且
UVC `index=0` 的主图像流，从而排除普通 webcam、RealSense 和每台 GelSight 的第二个 metadata 流。
它要求恰好发现两路，打印设备号、名称和序列号后，按当前设备号顺序绑定为 left/right；数量不等于二时
直接停止。设备物理位置改变后应先用下面的自动预览确认左右顺序：

```bash
.venv/bin/python scripts/camera/gelsight_start.py
```

不加 `--auto-gelsight` 时仍严格使用部署 JSON 中的 `left_device/right_device`，用于复现实验或显式覆盖。

## 0802 比例控制律部署（sim_actuator_velocity）

当前原生 libfranka 控制链路的独立说明见
[`docs/LIBFRANKA_CONTROL_METHOD.md`](docs/LIBFRANKA_CONTROL_METHOD.md)。

`joint_position_pursuit`（0801/0802 的默认律）把参考点放在实测位置前方一个固定关节增量处，
`libfranka::limitRate` 读到的隐含速度是 `增量 / 周期`，任何可用增量都会让它饱和到
`maximum_joint_velocities`。结果是关节速度与策略动作幅值无关，且逐关节 clamp 会旋转笛卡尔方向
——在 0802 参考位姿上实测把 `[-0.577, 0.577, -0.577]` 变成 `[-0.104, 0.991, 0.088]`，偏了 54.5°。

`sim_actuator_velocity` 复现 Isaac Lab implicit PD 的稳态 `qd = (stiffness / damping) * dq_ik`，
关节速度与动作成正比。`FRANKA_PANDA_HIGH_PD_CFG` 是 400/80，所以
`reference_velocity_gain = 5.0`，饱和动作对应 `5 × 0.05 = 0.25 m/s` 的 TCP 速度，
即 8.33 mm/步，而不是 50 mm/步。

实现先计算带符号的关节速度参考 `qd_ref = gain * dq_ik`；`dq_ik` 只受
`maximum_ik_reference_delta_rad` 这个宽限幅保护，正常训练尺度下不生效。1 kHz FCI callback
使用临界阻尼速度跟踪器把参考转换成位置指令，并逐周期限制关节速度、加速度和跃度。

早期实现曾把 `abs(qd_ref)` 直接用作 `libfranka::limitRate` 的动态硬速度上限，这在参考减小时
并不安全：若当前 `dq_d` 已经高于新上限，`limitRate` 会为了立即进入新包络而跳到最大减速度。
例如 7 rad/s² 的单周期跳变相当于 7000 rad/s³，超过配置的 1500 rad/s³，FCI 会报
`joint_motion_generator_velocity_discontinuity`；同一个动作的第二次 DLS 重算或一次 policy
deadline miss 都可能触发。现在速度参考可以任意缩小、清零或反向，跟踪器仍保证轨迹导数连续。
worker 的 `--self-test` 会复现真实日志中的第二 tick 参考缩小、stale action 和反向指令序列，
逐周期断言速度、加速度、跃度均不越界。generation 编号允许因丢弃的策略结果而跳号。

### 停止时序（既有 bug，与控制律无关）

`MotionFinished` 必须发在一个**指令速度和加速度精确为零**的指令上。早期停止路径改用
`limitRate(q_target=q_d)` 刹车，但 ABI-7 的真实 FCI 日志证明它在较高入口速度穿越零点时仍会
产生离散尖峰：实测第 6 关节跃度约 7654 rad/s³（配置上限 2000），并出现约 11.9 rad/s²
的加速度（配置上限 10）。错误在停止过零阶段产生，到 `MotionFinished` 才上报，所以表面看起来
像结束条件问题。

停止路径现在与动作路径共用临界阻尼、有界跃度的速度跟踪器，参考设为零；只有当一步精确归零
所需的加速度和跃度都在配置包络内时才落到 bit-exact zero。随后要求 `q_command == q_d`，且
`dq_d`、`ddq_d` 都为零、实测关节速度不超过 0.005 rad/s，连续保持 100 个 1 kHz 周期后才
发送 `MotionFinished`。`--self-test` 会从正负多个入口速度扫描完整停止过程，逐周期断言速度、
加速度、跃度不越界并在 2 s watchdog 内精确停止。

该律、完整单步会话的 FCI 异常前官方命令日志，以及不依赖有限 trace 的动作 tick 验收，
需要 abi-7 的 worker，先重新构建：

```bash
./scripts/build_franka_server9_worker.sh   # 应打印 abi-7 与 self-test: PASS
```

验收顺序与其他入口一致：

```bash
python scripts/policy/run_exported_0802_dr_simactuator.py --validate-only
python scripts/policy/run_exported_0802_dr_simactuator.py --steps 150 --preview-only
python scripts/policy/run_exported_0802_dr_simactuator.py --streaming-check
python scripts/policy/run_exported_0802_dr_simactuator.py --steps 1
```

使用 portable v6 DR TorchScript 在第一张 CUDA GPU 上推理时，改用专用入口；该入口默认
加载 `checkpoint/0802_DR_gpu/rma_student_dr_sim2real_gpu.pt` 并使用 `cuda:0`：

```bash
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --validate-only
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 5 --preview-only
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 5
```

### RMA 真实位置 / 零接触消融实验

下面的独立入口绕过 `vision_encoder + adaptation_head`，直接将物体中心在 Franka 基座
`robot_root` 下的真实 XYZ 输入同一个 frozen `actor_core`，并始终输入左右 contact `[0, 0]`。
该路径默认使用 CPU（本机实测约 0.2 ms/步，远低于 33.3 ms budget），不依赖 CUDA：

```bash
python scripts/policy/run_exported_0802_dr_simactuator_oracle_gpu.py \
  --cube-position-root 0.50 0.00 0.026 \
  --steps 5 --preview-only

python scripts/policy/run_exported_0802_dr_simactuator_oracle_gpu.py \
  --cube-position-root 0.50 0.00 0.026 \
  --steps 5
```

位置单位为米，表示物体中心而非目标 TCP；该值在一次 session 内保持不变，移动物体后需要用新
坐标重新启动。训练 XY 范围是 `x=[0.45,0.55] m`、`y=[-0.05,0.05] m`，建议先在此范围内测试。
每步真正送入 actor 的 `cube_position_root`、`contact_state` 和 `vision_bypassed` 会保存到
`rollout.jsonl` 的 `model_input.rma_actor_input`。

也可以只替换一个视觉分支，组成对照实验：

```bash
# 真实位置 + 视觉 contact
python scripts/policy/run_exported_0802_dr_simactuator_oracle_gpu.py \
  --cube-position-root 0.50 0.00 0.026 --rma-contact-source vision --steps 5

# 视觉位置 + 恒零 contact
python scripts/policy/run_exported_0802_dr_simactuator_oracle_gpu.py \
  --rma-position-source vision --rma-contact-source zero --steps 5
```

恒零 contact 适合验证到达阶段；发生真实手指接触后，它不再符合 Teacher 的输入契约，因此仅凭
该模式抓取失败不能判定 actor 或低层控制无效。

### 验证 RMA 物体位置预测

实时查看相机视觉头输出（只连接 RealSense，不连接或移动机器人）：

```bash
python scripts/policy/monitor_rma_object_position.py \
  --device cuda:0 \
  --exposure 80 \
  --gain 64 \
  --print-every 1 \
  --save-rgb-every 30
```

`--exposure` 和 `--gain` 使用 RealSense Viewer 中显示的数值；指定其中任意一个都会关闭
自动曝光。脚本默认打开实时窗口，画面是实际送入模型的裁剪后 224x224 RGB，并叠加原始/EMA
平滑后的 `robot_root` XYZ（窗口用 mm，CSV/终端用 m）、左右 contact 概率、推理耗时和
当前曝光设置。按窗口中的 `q` 或
`Esc`（也可在终端按 `Ctrl+C`）停止。若只需要终端输出，可加 `--no-display`；要恢复自动曝光，
使用 `--auto-exposure`，且不要同时传 `--exposure` 或 `--gain`。

终端也会逐帧打印预测结果：

```text
frame=000120 xyz_m=[+0.5031, -0.0182, +0.0415] \
ema_m=[+0.5028, -0.0180, +0.0412] contact=[0.032, 0.041] ...
```

完整数据自动保存到 `runs/*_rma_object_position_live/predictions.csv`；`--save-rgb-every`
保存的是没有文字叠加的模型输入图。这些是模型预测值，没有真实位置时不能据此判断精度。

0802 RMA Student 会从 RGB 内部预测方块在机器人基座（`robot_root`）坐标系中的
位置 XYZ，但当前 policy 不预测物体朝向。下面的命令只连接 RealSense，不连接或移动机器人；
把方块的实测基座坐标填入 `--ground-truth`，脚本会采集 30 帧并输出 XYZ MAE/RMSE、
3D 误差分位数和 10 mm 阈值通过率：

```bash
python scripts/policy/validate_rma_object_position.py \
  --realsense \
  --ground-truth 0.50 0.00 0.026 \
  --frames 30 \
  --max-error-mm 10
```

也可以验证已经按模型相机契约处理好的 224x224 图片或包含 `wrist_rgb` 和
`rma_cube_pos` 的 NPZ：

```bash
python scripts/policy/validate_rma_object_position.py \
  --input path/to/frame.png \
  --ground-truth 0.50 0.00 0.026

python scripts/policy/validate_rma_object_position.py \
  --input path/to/simulation_samples.npz
```

批量实测数据可使用 CSV manifest，列名为 `image,x_m,y_m,z_m`，图片相对路径以
manifest 所在目录为基准。原始 640x480 图片会自动使用配置中的 crop/resize；已经是
224x224 的图片默认直接送入模型。结果写入 `runs/*_rma_object_position_validation/`。
没有真值时脚本只报告预测均值和帧间抖动，并明确标记 `NO_GROUND_TRUTH`，不能据此判断精度。

贴好仓库生成的 `tag36h11 / ID 0` 后，可以实时检测 40 mm AprilTag，并同时用
policy 训练外参和已验收 eye-to-hand 外参计算 50 mm 方块中心：

```bash
python scripts/calibration/live_apriltag_cube_pose.py \
  --ground-truth 0.50 0.00 0.026
```

使用 `checkpoint/0803_dr4` 的 RMA 视觉头与 AprilTag 实时对照时，显式指定该导出模型：

```bash
.venv/bin/python scripts/calibration/live_apriltag_cube_pose.py \
  --policy-config configs/e2e_bundle_real_exported_0803_dr_heatmap.json \
  --policy-model checkpoint/0803_dr4/rma_student_e2e_student_0100000.pt \
  --policy-metadata checkpoint/0803_dr4/rma_student_e2e_student_0100000.json \
  --policy-device cuda:0 \
  --ground-truth 0.50 0.00 0.026
```

这条命令会同时显示 AprilTag 解算位置、0803_dr4 policy 的物块位置预测及二者误差；它仅连接
D435 相机，不连接或移动 Franka。

X040-Wide 的位置头采用不同的 `forward_with_position()` 导出接口，现已可用同一条 AprilTag
链路做**离线精度评估**。先在训练范围内手工摆放方块（建议覆盖 `x=0.32~0.48 m`、
`y=-0.10~0.10 m` 的至少 5 个静态位置），每个位置按 `s` 录制一段画面，最后按 `q`：

```bash
.venv/bin/python scripts/calibration/record_cube_pose_dataset.py --burst-frames 60
```

然后将输出目录替换到下列命令中。它只读取已录制的图片，不打开相机、不连接或移动 Franka；结果中的
`position_summary.csv` 和 `summary.json` 给出相对于 AprilTag 真值的 XYZ RMSE、中位 3D 误差、P95
和轴向偏差。X040-Wide 不训练 contact head，因此 contact 列会明确为 `NaN` / 不适用。

```bash
.venv/bin/python scripts/calibration/evaluate_policy_vs_apriltag.py \
  runs/<时间>_cube_pose_dataset \
  --policy-config configs/e2e_bundle_real_exported_0809_x040_wide_direct_action.json \
  --policy-device cuda:0 \
  --family tag36h11 --id 2 \
  --marker-length-m 0.038 \
  --tag-to-object 0 0 -0.025
```

`--tag-to-object 0 0 -0.025` 只适用于 50 mm 方块、38 mm 标签居中贴在一个表面且标签 +Z 朝外的情况；
标签位置或方块尺寸不同，必须先按实际几何修改该偏移，不能把几何误差当成 policy 误差。

`--ground-truth` 必须替换为独立测量的方块中心基座坐标；如果暂时没有真值，可以省略。
默认假设 50x50 mm 标签纸与方块表面对齐，因此方块中心位于 Tag 的 `-Z` 方向 25 mm。
程序只连接 D435，不连接或移动 Franka。实时窗口中按 `s` 保存截图、`c` 清空平滑窗口、
`q` 或 `Esc` 退出；逐帧完整位姿写入 `detections.csv`，统计结果写入 `summary.json`。

要比较不同手动曝光下的位置误差，可以固定 gain 后自动扫描。下面每个曝光采集 120 帧，
分别保存原始结果，并生成汇总 `exposure_results.csv` 和 `exposure_report.json`：

```bash
python scripts/calibration/sweep_apriltag_exposure.py \
  --exposures 40 60 80 100 120 160 200 \
  --gain 64 \
  --frames-per-exposure 120
```

没有独立真值时，排名使用 policy 预测与 AprilTag 标定位置之间的 3D RMSE；若传入
`--ground-truth X Y Z`，排名改用 policy 相对该真值的 3D RMSE。默认只有 AprilTag 检出率
达到 50% 的曝光才参与最佳曝光选择。

因为控制律是比例的，`commissioning_action_limit` 现在真正线性地控制速度，可以作为唯一的调速旋钮
按 `0.1 → 0.3 → 0.6 → 1.0` 逐级放开，对应 0.83 → 2.5 → 5.0 → 8.33 mm/步。
关节限幅是安全包络，不是调速手段：一旦它们饱和，DLS 的幅值信息就被丢弃，方向也会被旋转。

每级之后用下面的脚本确认跟踪质量，`ratio` 和 `slope` 应接近 1，`direction error` 应接近 0：

```bash
python scripts/diagnostics/analyze_streaming_tracking.py --latest 1
```

## Policy 真机测试（0711 默认入口）

默认入口使用：

- 配置：`configs/e2e_bundle_real_exported_0711.json`
- 模型：`checkpoint/exported_0711/exported/policy_actor_e2e_best_agent.pt`
- 输入：`action_history[4]`、`proprio_obs[15]`、`wrist_rgb[224,224,3]`
- 输出：`[dx, dy, dz, gripper]`

### 1. 只推理，不运动

```bash
python scripts/policy/run_exported_0711.py \
  --steps 1 \
  --preview-only
```

此模式仍会连接机器人读取状态并读取 RealSense，但不会发送机械臂、夹爪或 homing 命令。

### 2. 单步真机测试

```bash
python scripts/policy/run_exported_0711.py --steps 1
```

脚本会打印 proposed action，只有输入 `y` 或 `yes` 后才会运动。

### 3. 多步测试，每步确认

```bash
python scripts/policy/run_exported_0711.py \
  --steps 15 \
  --confirm-each-step
```

### 使用其他模型配置

```bash
python scripts/policy/run_exported_0711.py \
  --config configs/你的配置.json \
  --robot-ip 172.16.0.2 \
  --steps 1 \
  --preview-only
```

开始前确认终端打印的 `Model:` 路径正确。当前 runner 会将 raw action 裁剪到 `[-1,1]`，再按配置中的 `action_adapter.scales` 映射为真机动作。

## 相机和对齐

预览 RealSense 原图、crop 和模型输入：

```bash
python scripts/camera/preview_camera_crop.py --open
```

采集机器人初始状态和相机对齐基准（只读，不运动）：

```bash
python scripts/calibration/capture_real_alignment_reference.py \
  --config configs/e2e_bundle_real_exported_0711.json \
  --ip 172.16.0.2 \
  --realtime ignore \
  --frames 10
```

固定 D435 的 eye-to-hand 标定支持一次确认后连续采集 25 个求解姿态：

```bash
python scripts/calibration/collect_eye_to_hand.py --phase calibration --continuous
```

首次使用前必须先执行 `--camera-check` 和 `--dry-run`；完整流程与续采方法见
`docs/HAND_EYE_CALIBRATION.md`。

测量相机的信号相关噪声模型 `sigma(mu)`，用于把仿真的图像噪声对齐到真机（只读相机，
不连接机器人）：

```bash
python scripts/calibration/measure_camera_noise.py \
  --exposures 100 \
  --gains 16 32 64 \
  --frames 200
```

执行期间视野必须完全静止，机械臂、手和光源都不能动，否则运动会被当成噪声。运动像素
占比超过 `--maximum-motion-fraction` 时脚本会判定该工况作废并以非零码退出。结果写到
`runs/<时间戳>_camera_noise/`，其中 `noise_lut.csv` 是按像素亮度分箱的 `sigma` 查找表，
`noise_report.json` 还包含 `sigma^2 = a*mu + b` 的拟合系数和暗区单独统计。默认在模型
输入域（crop 后 224x224）测量，`--domain raw` 可切到 640x480 原始分辨率。

## 常用诊断

检查静止状态下的外力和力矩：

```bash
python scripts/diagnostics/test_franka_force_sensor.py \
  --ip 172.16.0.2 \
  --baseline-samples 20 \
  --threshold-n 0.5
```

小幅运动中采集 wrench：

```bash
python scripts/diagnostics/test_franka_force_during_motion.py \
  --ip 172.16.0.2 \
  --dx 0.005 \
  --cycles 1 \
  --speed 0.03 \
  --contact-force-threshold 6 \
  --raw-output
```

诊断 action 坐标方向：

```bash
python scripts/diagnostics/diagnose_franka_directions.py \
  --robot-ip 172.16.0.2 \
  --cases dx+ dx- dy+ dy- dz+ dz-
```

所有会运动的诊断都应从小位移和单次确认开始。

比较仿真与真机策略输入图像的外观差异（纯离线，不碰硬件）：

```bash
python scripts/diagnostics/compare_sim_real_images.py \
  --real runs/20260808_223325_e2e_bundle_real_exported_0803_dr_heatmap/rgb \
  --sim /path/to/episode_0000.npz
```

两侧都必须是模型输入域（224x224）的图像，尺寸不一致会直接报错。真机侧用部署运行的
`rgb/` 目录，或 `measure_camera_noise.py` 输出的 `mean_*.png`；仿真侧用 `TacEx/` 的
`collect_rma_student_rollouts.py` 产出的 episode NPZ（键 `wrist_rgb`，已含 DR 后处理）。
输出写到 `runs/<时间戳>_sim_real_image_stats/`，包含亮度分位数、明暗分区的通道平衡，
以及亮度直方图的 Wasserstein-1 距离（单位 DN，即仿真整体平均需平移多少才能对上真机）。
两份同场景真机数据之间的 W1 约 10 DN，可作为判断差距是否显著的本底。

## 重要配置

主要部署配置位于 `configs/`。常改字段：

- `robot_ip`：机器人地址。
- `speed`：机械臂 relative dynamics factor。
- `camera.*`：RealSense 分辨率、预热帧和 crop。
- `runner.steps`：rollout 步数。
- `action_adapter.scales`：每步动作尺度。
- `workspace.minimum/maximum`：允许的 Cartesian 工作空间。
- `model.model_path`、`metadata_path`：TorchScript 权重与 metadata。

## 输出日志

每次运行会在 `runs/<timestamp>_<run_name>/` 保存：

- `config.json`：本次实际配置。
- `rollout.jsonl`：逐步 observation、action 和 timing。
- `summary.json`：完整运行汇总。
- `rgb/step_XXXX.png`：模型实际看到的 RGB 图像。
- Streaming 另存 `control_trace.jsonl` 和 `timing_summary.json`，且仅在停止控制后落盘。
- HIL 另强制保存 `step_data/step_XXXX.npz`，其中包含精确模型输入和 Residual BC 标签。

`runs/` 默认被 Git 忽略，因为其中可能包含真机图像和大量实验数据。

## Real-world Residual SAC

0814 视觉＋双 GelSight 策略支持独立的 XYZ Residual SAC。Base Student、gripper、DLS、FCI 和既有
安全限幅保持不变；AprilTag 只用于 reward 和 privileged Critic，不进入部署 Actor。

先做无硬件校验：

```bash
.venv/bin/python scripts/real_rl/run_residual_sac.py validate \
  --config configs/real_residual_sac_0814.json --device cuda:0
```

Base-only 或小幅随机 warmup（二选一）。随机 warmup 使用配置中的 AR(1) 相关高斯噪声，默认
`rho=0.9`、`sigma=0.5 mm`、裁剪到 `±1 mm`，不再逐拍 IID 跳变：

```bash
.venv/bin/python scripts/real_rl/run_residual_sac.py collect \
  --config configs/real_residual_sac_0814.json --steps 300 --device cuda:0 \
  --auto-gelsight --zero-residual --enable-real-rl-control

.venv/bin/python scripts/real_rl/run_residual_sac.py collect \
  --config configs/real_residual_sac_0814.json --steps 300 --device cuda:0 \
  --auto-gelsight --warmup-random-residual --enable-real-rl-control
```

首次离线训练固定执行配置中的 `bootstrap_updates`（默认 1000），不使用 `N×UTD`；之后才按
checkpoint high-watermark 之后新增的 `trainable` transition 计算更新量，默认 UTD=2：

```bash
.venv/bin/python scripts/real_rl/run_residual_sac.py train \
  --config configs/real_residual_sac_0814.json --device cuda:0 --utd-ratio 2
```

Real-RL 的运行目录和 Replay 分别位于 `real_rl_logs/<时间戳>_*/` 与
`real_rl_logs/replay.sqlite3`，不再写入通用 `runs/`；checkpoint 仍位于
`checkpoints/real_residual_sac/`。真机 collect 永远要求 `--enable-real-rl-control` 和交互确认；
先使用 `--preview-only`，再进行单步和小批量验收。checkpoint 缺失、损坏或 contract/SHA 不匹配时
默认拒绝 episode 启动；只有明确接受 base-only fallback 数据时才能额外给
`--allow-checkpoint-fallback-collect`。

Replay v2 用统一的 `trainable/trainable_reason` 判定训练样本，并保存同一 monotonic 时钟域的
`capture_timestamp` 与 action timestamp；严格要求 `tag_t <= action_time < tag_t1`。同时记录
`pre_safety_action`、`post_safety_action`、`safety_intervened` 和归一化 4D action contract 下的 L2
`intervention_magnitude`。物体高度固定定义为
`object_z_in_base - initial_object_z_in_base`。Normalizer 仅对 1043D policy feature 和 4D privileged
state 做经验归一化；base action 使用固定 action contract，不做 z-score。

AprilTag 仅在启动前做 15 次静止预检；预检通过后检测线程立即停止。每个真实策略 deadline 到达时，
系统独立快照相机最新的 640×480 raw packet；它与提前算好的 policy frame 分开审计。机械臂控制停止后
才写入 `raw_rgb/boundary_XXXX.png`，再按帧离线检测、
计算 reward 并回写 Replay。检测先在上一帧 Tag 周围的扩大 ROI 搜索，失败自动退回整帧；结果位于
`offline_apriltag_labels.jsonl` 和 `offline_apriltag_report.json`。如采集完成后标签阶段被中断，可单独重跑：

```bash
.venv/bin/python scripts/real_rl/run_residual_sac.py label \
  --config configs/real_residual_sac_0814.json \
  --run-dir real_rl_logs/<采集目录>
```

server9 Real-RL 的 Franka Hand 由独立 `spawn` 进程独占：`move_async/grasp_async`、future 完成查询、
`gripper.state` 和 `stop` 都不在主进程执行。30 Hz 策略线程只向共享内存写入最新目标宽度并检查缓存错误，
所以 Hand native binding 即使持有 Python GIL，也不会冻结机械臂策略循环。
缺少 capture/action timestamp 的 Replay v1 不会被自动升级或伪造成可训练数据；若已有 v1 数据库，
请在配置中使用新的 Replay 路径重新采集，旧库会以 schema mismatch 明确拒绝。

如果需要区分 Hand 的 `stop()` RPC 延迟与有限宽度运动耗时，可先运行完全离线的参数预览：

```bash
.venv/bin/python scripts/diagnostics/test_franka_gripper_latency.py
```

确认夹爪内没有物体、手指运动范围无人后，才显式启用只连接夹爪的 6 mm 有界对比测试；脚本不会连接或
移动机械臂，并要求输入 `y`/`yes`：

```bash
.venv/bin/python scripts/diagnostics/test_franka_gripper_latency.py \
  --run-hardware --mode compare --direction open \
  --travel-mm 6 --speed 0.03 --stop-after-ms 50
```

测试入口硬限制单次行程不超过 10 mm、速度不超过 0.05 m/s，并把逐次 `move_async()`、`stop()`、
自然完成和 Hand 状态读取耗时写入 `runs/gripper_latency/*.json`。它只诊断当前生产使用的 `franky`
路径，不会修改 RLPD 控制方式。

如果 Python 结果确认延迟存在，再构建并运行同版本的原生对照。构建脚本链接 `franky-control 1.1.3`
自带的 `libfranka 0.17.0`，先自动执行不连接硬件的 self-test 和 dry-run：

```bash
bash scripts/build_franka_gripper_latency.sh
```

随后在相同清场和人工确认条件下执行原生 C++ 测试：

```bash
dist/gripper_latency/franka_gripper_latency \
  --run-hardware --mode compare --direction open \
  --travel-mm 6 --speed 0.03 --stop-after-ms 50
```

该程序在一个线程执行阻塞 `Gripper::move()`，另一个线程在指定延迟后调用同一线程安全 Gripper 对象的
`stop()`，从而去掉 `franky` future/包装层，同时保持 libfranka 版本、目标、速度和动作范围一致。报告写入
`runs/gripper_latency/*_libfranka_0_17.json`。

## RLPD 残差强化学习

`real_rlpd/` 是独立于旧 Real-RL 的 PyTorch RLPD 实现。当前适配 0814 单帧、0823 三帧、0911
Progress 单帧和 0912 Progress 三帧 GelSight 策略，默认仍使用 0823。专家源数据保存同步视觉/触觉、机器人观测和专家绝对动作，不绑定某个
base policy；每个 base policy 的派生 expert Replay、online Replay 和 checkpoint 仍由模型与 metadata
SHA 隔离。
Actor 输入是 frozen base-policy feature 1043D 加限幅后的 base action 4D，输出 XYZ＋gripper 4D residual。
Critic 额外读取 AprilTag object-relative XYZ＋height。默认使用 10 个 Q、随机取 2 个最小 Q、Critic
LayerNorm、offline/online 50:50、UTD=20；每组做 20 次 Critic 更新和 1 次 Actor/temperature 更新。
完整模块、操作和扩展说明见 [`real_rlpd/README.md`](real_rlpd/README.md)。

先做完全离线的 0823 模型与契约校验：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0823.json --device cuda:0
```

0911 Progress 使用新的 artifact kind、adapter ID 和独立数据目录，不修改或复用 0814/0823 的
Replay/checkpoint。该 checkpoint 的真实历史帧离线审计发现动作饱和和触觉辅助接触输出塌缩；操作者已
确认饱和是简单仿真任务中预期的 bang-bang 最优行为，因此 metadata 保留审计告警，同时设置
`motion_authorization=direct_and_rlpd`，允许标准裸 policy 多步部署和 RLPD 流程。

```bash
.venv/bin/python scripts/policy/run_exported_0911_gelsight_progress.py --validate-only
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0911_progress.json --device cuda:0
```

0912 Progress 使用三帧 feature API、独立的 `gelsight_reference_progress_three_frame_v1` adapter ID 和
`0912_progress` 数据/checkpoint 命名空间，同时沿用 Progress 成功条件：抬升 `0.035 m` 且连续检测 5 帧。

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0912_progress.json --device cuda:0
```

若使用与仿真可观测项对齐的绝对 reach/lift 奖励，请改用独立配置；它不会覆盖原 0912 Replay/checkpoint：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0912_progress_observable_absolute.json --device cpu
```

具体公式、旧专家 Replay 的非破坏性重建命令和不可观测项说明见
[`real_rlpd/README.md`](real_rlpd/README.md)。

把下文命令中的 `configs/real_rlpd_0823.json` 替换为
`configs/real_rlpd_0911_progress.json` 或 `configs/real_rlpd_0912_progress.json`，即可执行对应 Progress
policy 的专家采集、训练和 online 采集；checkpoint 分别写入 `checkpoints/real_rlpd/0911_progress/`
和 `checkpoints/real_rlpd/0912_progress/`。Progress reward 的成功条件按训练契约设置为
抬升 `0.035 m` 且连续检测 5 帧。训练 horizon 的 150 步只作为 provenance 记录，不是运行时上限；
当前在线检测仍是离线标注，因此成功后需要操作者或外部监控停止。

裸 policy 使用标准部署入口，`--steps` 可由操作者设置且没有 150 步硬上限。动作尺度、`±0.1`
commissioning limit、workspace、
初始状态门禁、碰撞阈值和首次 proposed action 人工确认均保持不变。首次真机测试仍按 preview、
streaming check、单步、再多步的顺序进行；不要传 `--yes`：

```bash
.venv/bin/python scripts/policy/run_exported_0911_gelsight_progress.py \
  --steps 300 --auto-gelsight --preview-only

.venv/bin/python scripts/policy/run_exported_0911_gelsight_progress.py \
  --steps 300 --streaming-check

.venv/bin/python scripts/policy/run_exported_0911_gelsight_progress.py \
  --steps 1 --auto-gelsight

.venv/bin/python scripts/policy/run_exported_0911_gelsight_progress.py \
  --steps 300 --auto-gelsight
```

`±0.1` normalized envelope 对应每个 policy step 的 XYZ 单轴最多 `5 mm`、夹爪总宽度最多变化 `1 mm`。
由于成功检测不在裸部署闭环内，多步运行期间必须由操作者监控并在成功、异常接触或偏离预期时停止。

专家数据是全人工接管：base policy 只做 shadow，窗口聚焦后按 Enter arm，`W/S`、`A/D`、`J/K`
控制基座系 XYZ，`U/I` 连续开合夹爪；窗口失焦、ESC 或关闭窗口都会安全停止。先 preview，再
streaming check，再单步：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-expert \
  --config configs/real_rlpd_0823.json --device cuda:0 --steps 150 \
  --auto-gelsight --preview-only

.venv/bin/python scripts/real_rlpd/run_rlpd.py streaming-check \
  --config configs/real_rlpd_0823.json --device cuda:0 --enable-streaming-check

.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-expert \
  --config configs/real_rlpd_0823.json --device cuda:0 --steps 1 \
  --auto-gelsight --enable-expert-control
```

采够至少 1000 条完成离线标注的 expert transition 后，首次训练默认执行 1000 个 update group：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py train \
  --config configs/real_rlpd_0823.json --device cuda:0
```

随后用同一策略的 checkpoint 采 online episode。确定性 residual 是默认值；随机策略必须同时显式给出
`--stochastic --enable-stochastic-control`。每个 episode 停止并完成离线标注后再运行一次 `train`，它只按
checkpoint high-watermark 之后新增的 trainable online transition 安排 update group，训练不会与真机控制并发：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-online \
  --config configs/real_rlpd_0823.json --device cuda:0 --steps 1 \
  --checkpoint checkpoints/real_rlpd/0823/latest.pt --auto-gelsight \
  --enable-rlpd-control

.venv/bin/python scripts/real_rlpd/run_rlpd.py train \
  --config configs/real_rlpd_0823.json --device cuda:0
```

online 动作在原 base policy 的 commissioning limiter 之后组成 residual，再经过原有最终裁剪；专家控制
保存的是人工绝对动作，同时生成当前 policy-specific residual Replay。
不允许 `--allow-full-scale`。动作尺度、workspace、initial-state gate、DLS/FCI 与 server9 ABI 都没有放宽。
采集时 30 Hz 控制线程只缓存 transition，worker 停止后才单事务写 SQLite；AprilTag reward 也只在停止后
离线标注。每个策略日期使用独立目录，例如专家 episode 写入 `real_rlpd_data/0823/expert/`，rollout 写入
`real_rlpd_data/0823/rollout/`；新增 0913 时使用对应的 `real_rlpd_data/0913/`，不会混放。0823 的附加物理
TCP 偏移已确认为 `0.0579 m`，workspace z 下界为 `0.01 m`；该偏移仅用于工具
最低点的 workspace 检查和日志，不改变策略动作坐标系或 IK 命令点。

## 0912 Frozen-Encoder Direct BC

该实验路径复用并冻结 0912 三帧 RGB、双 GelSight、proprio/history encoder，只训练
`1043→256→256→4` 的 direct-action BC head。标签是专家最终执行的四维 normalized action，训练时除以
固定 `0.1`，部署 artifact 再乘回 `0.1`；它不把 base action 加到输出上。现有 0912 base/RLPD 配置和
checkpoint 均保持不变。

重新训练要求输出目录为空：

```bash
.venv/bin/python scripts/training/train_0912_direct_bc.py \
  --device cuda:0 --epochs 100 --batch-size 256 --patience 15
```

当前产物和独立部署入口：

```text
checkpoint/0912_direct_bc/direct_bc_best.pt
checkpoint/0912_direct_bc/direct_bc_policy.pt
checkpoint/0912_direct_bc/direct_bc_policy.json
configs/e2e_bundle_real_exported_0912_direct_bc.json
scripts/policy/run_exported_0912_direct_bc.py
```

真机验收必须保持标准顺序，不使用 `--yes` 或 `--allow-full-scale`：

```bash
.venv/bin/python scripts/policy/run_exported_0912_direct_bc.py --validate-only
.venv/bin/python scripts/policy/run_exported_0912_direct_bc.py \
  --steps 1 --preview-only --auto-gelsight
.venv/bin/python scripts/policy/run_exported_0912_direct_bc.py --streaming-check
.venv/bin/python scripts/policy/run_exported_0912_direct_bc.py \
  --steps 1 --auto-gelsight
```

该入口沿用 `±0.1` commissioning limit；单步 XYZ/夹爪宽度上限仍是 `5 mm / 1 mm`。当前已完成离线
训练和 artifact 校验；第一次 preview 在 step 0 被 CUDA 惰性初始化安全拒绝，以下预热修复后的 preview、
streaming check 和真机单步仍需操作者执行。

Direct BC 启动时会在控制接管前打印 `Policy CUDA warmup completed before control`。它会依次覆盖全零和
固定非零图像的 CUDA 惰性初始化，约需 1 秒但不会发送动作；之后首个真实帧才进入 250 ms policy
watchdog。若仍在 step 0 超时，请保留该行的两次预热耗时和 run 目录，不要增大 watchdog。

## 目录结构

- `scripts/robot/`：状态读取、回零和直接控制。
- `scripts/policy/`：TorchScript policy 部署入口。
- `scripts/camera/`：RealSense 和 GelSight 工具。
- `scripts/calibration/`：真机视觉与状态对齐。
- `scripts/diagnostics/`：力传感器、运动和方向诊断。
- `scripts/real_rl/`：Residual SAC 的校验、真机采集和离线训练入口。
- `scripts/real_rlpd/`：RLPD 专家/online 采集、标注和 episode 间训练入口。
- `real_rlpd/`：多 base-policy adapter、4D residual、ensemble SAC 与分离 Replay。
- `real_rlpd/direct_bc.py`：0912 frozen-feature Direct BC 数据、训练、模型与导出工具。
- `franka_sim2real/`：运行时、真实机器人 backend、安全限制和日志。
- `configs/`：部署配置。
- `tools/system/`：网络、realtime 和主机检查。
- `docs/`：详细设计与旧流程说明。

## 安全原则

- 先检查机器人状态和 camera crop。
- policy rollout 前先回到训练对应的初始姿态。
- 新模型先运行 `--preview-only`，再做单步测试。
- 检查 proposed action、模型路径、动作尺度和 workspace 限制。
- 不要仅依赖软件裁剪；始终保持人员远离机械臂并准备急停。

详细说明见：

- `docs/E2E_BUNDLE_DEPLOY.md`
- `docs/SIM2REAL_FRAMEWORK.md`
- `docs/HAND_EYE_CALIBRATION.md`：固定 D435 与 Franka 的 eye-to-hand 标定流程。
- `docs/GITHUB_UPLOAD.md`：GitHub 仓库创建与上传步骤。
