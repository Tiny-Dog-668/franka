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

回到 policy 初始关节姿态和夹爪宽度：

```bash
python scripts/robot/go_to_zero_pose.py \
  --ip 172.16.0.2 \
  --realtime ignore \
  --speed 0.1 \
  --max-step-rad 0.1 \
  --gripper-speed 0.03
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
