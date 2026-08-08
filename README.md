# Franka Control and Sim2Real Toolkit

Franka 真机控制、RealSense 图像采集、TorchScript policy 部署和基础诊断工具。

> 真机运行前请清空工作空间、准备急停，并先使用 `--preview-only` 或单步确认。

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
  --speed 0.3 \
  --max-step-rad 0.3 \
  --gripper-speed 0.3
```

执行推理

```
python scripts/policy/run_exported_0802_dr_simactuator_gpu.py   
  --model checkpoint/0803_dr3/rma_student_e2e_student_0040000.pt   
  --device cuda:0   
  --steps 1000 
  --yes
```
执行小幅 Cartesian 位移：

```bash
python scripts/robot/minimal_franka_move.py \
  --ip 172.16.0.2 \
  --dx 0.01 \
  --speed 0.02 \
  --realtime ignore
```

`minimal_franka_move.py` 还支持 `--dy`、`--dz`、`--roll`、`--pitch`、`--yaw` 和夹爪命令，执行前会要求确认。

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

`--ground-truth` 必须替换为独立测量的方块中心基座坐标；如果暂时没有真值，可以省略。
默认假设 50x50 mm 标签纸与方块表面对齐，因此方块中心位于 Tag 的 `-Z` 方向 25 mm。
程序只连接 D435，不连接或移动 Franka。实时窗口中按 `s` 保存截图、`c` 清空平滑窗口、
`q` 或 `Esc` 退出；逐帧完整位姿写入 `detections.csv`，统计结果写入 `summary.json`。

因为控制律是比例的，`commissioning_action_limit` 现在真正线性地控制速度，可以作为唯一的调速旋钮
按 `0.1 → 0.3 → 0.6 → 1.0` 逐级放开，对应 0.83 → 2.5 → 5.0 → 8.33 mm/步。
关节限幅是安全包络，不是调速手段：一旦它们饱和，DLS 的幅值信息就被丢弃，方向也会被旋转。

每级之后用下面的脚本确认跟踪质量，`ratio` 和 `slope` 应接近 1，`direction error` 应接近 0：

```bash
python scripts/diagnostics/analyze_streaming_tracking.py --latest 1
```

## Policy 真机测试

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

`runs/` 默认被 Git 忽略，因为其中可能包含真机图像和大量实验数据。

## 目录

- `scripts/robot/`：状态读取、回零和直接控制。
- `scripts/policy/`：TorchScript policy 部署入口。
- `scripts/camera/`：RealSense 和 GelSight 工具。
- `scripts/calibration/`：真机视觉与状态对齐。
- `scripts/diagnostics/`：力传感器、运动和方向诊断。
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
