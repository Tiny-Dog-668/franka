# Franka Control and Sim2Real Toolkit

这个目录是一个围绕 `franky` 搭起来的 Franka 控制工作区，同时包含基于 TorchScript 的 sim2real 抓取策略部署流程。

## 目录内容

- `scripts/robot/`
  - 最小运动示例、状态读取、零位姿记录和零位姿恢复工具。
- `scripts/camera/`
  - RealSense crop 预览和 GelSight 视频辅助脚本。
- `scripts/policy/`
  - TorchScript bundle 部署入口，以及较早的 sim2real 验证入口。
- `scripts/diagnostics/`
  - 方向诊断、力传感器 baseline 检查、运动中力采样和画图工具。
- `tools/system/`
  - 网络、realtime kernel、CPU performance 和主机检查脚本。
- `docs/`
  - 部署说明、框架说明和 realtime 工作流检查清单。
- `assets/`
  - 静态图片和其他小型本地测试资源。
- `artifacts/`
  - 离线 wheel、wheel bundle 和其他构建产物。
- `third_party/`
  - 为参考保留的上游 vendored 源码。
- `franka_sim2real/`
  - 共用 sim2real runtime、真实机器人 backend、action mapping、安全逻辑和日志。
- `configs/e2e_bundle_real_example.json`
  - 真实机器人主部署配置。
- `configs/e2e_bundle_real_exported_0711.json`
  - `exported_0711` cube-grasp 模型的当前真实机器人配置。
- `deploy_bundle_e2e/`
  - TorchScript policy bundle 及其 metadata。
- `checkpoint/exported_0711/exported/`
  - 当前 cube-grasp TorchScript 模型和 metadata。
- `docs/E2E_BUNDLE_DEPLOY.md`
  - TorchScript bundle 部署路径的更详细说明。
- `docs/SIM2REAL_FRAMEWORK.md`
  - 较早的通用 sim2real scaffold 说明。
- `artifacts/franky_dist/`
  - 多个 Python 版本的 `franky` wheel 构建产物。
- `assets/test_static_image.png`
  - dry-run 和 crop-preview 检查用的静态 RGB 图片。

## 环境

运行任何脚本前，先启用项目虚拟环境：

```bash
source /home/td/franka/.venv/bin/activate
```

主要 Python 依赖已经安装在 `.venv` 中，包括：

- `franky`
- `pylibfranka`
- `torch`
- `pyrealsense2`
- `pillow`

## 机器人 IP

下面的示例默认使用：

```bash
172.16.0.2
```

可以通过 `--ip` 或 `--robot-ip` 显式传入，也可以导出环境变量：

```bash
export FRANKA_ROBOT_IP=172.16.0.2
```

## 1. 最小直接控制

最快的直接移动方式是：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dx 0.01 --speed 0.02 --realtime ignore
```

示例：

只沿 `x` 移动：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dx 0.01 --speed 0.02 --realtime ignore
```

只沿 `y` 移动：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dy 0.01 --speed 0.02 --realtime ignore
```

只沿 `z` 移动：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dz 0.01 --speed 0.02 --realtime ignore
```

只做 yaw 旋转：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --yaw 5 --speed 0.02 --realtime ignore
```

做 roll 或 pitch 旋转：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --roll 2 --speed 0.02 --realtime ignore
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --pitch 2 --speed 0.02 --realtime ignore
```

Gripper 示例：

```bash
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --gripper-homing
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --gripper-width 0.05 --gripper-speed 0.03
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --grasp-width 0.03 --gripper-speed 0.03 --gripper-force 20
python3 /home/td/franka/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --gripper-open --gripper-speed 0.03
```

注意：

- `--roll`、`--pitch` 和 `--yaw` 的单位是 degree。
- `--speed` 是 relative dynamics factor，不是直接的 `m/s` 速度。
- `--realtime ignore` 适合普通桌面系统；但真实 Franka 运动仍然依赖稳定的 FCI 通信。
- 脚本在发送 arm 或 gripper 命令前都会要求确认。

## 2. 读取机器人状态

读取一次：

```bash
python3 /home/td/franka/scripts/robot/read_franka_state.py --ip 172.16.0.2
```

连续读取：

```bash
python3 /home/td/franka/scripts/robot/read_franka_state.py --ip 172.16.0.2 --count 0 --interval 1.0
```

脚本会打印：

- 关节位置和速度
- TCP translation 和 quaternion
- external wrench
- 当前和上一次 motion errors

## 3. 保存并复用零位姿

读取机器人当前位姿，并打印可复制的常量：

```bash
python3 /home/td/franka/scripts/robot/read_current_zero_pose.py --ip 172.16.0.2 --include-gripper
```

输出包括：

- 当前 `tcp_translation`
- 当前 `tcp_quaternion`
- 当前 joint positions
- 当前 gripper width
- `TARGET_TRANSLATION`
- `TARGET_QUATERNION`
- `TARGET_GRIPPER_WIDTH`

这些常量可用于记录和诊断当前真实位姿。当前
`scripts/robot/go_to_zero_pose.py` 使用仿真环境给出的 7 维初始关节角，
并在运动完成后打印对应的真实 TCP 位姿。

将 arm 和 gripper 移动到 `exported_0711` 的仿真初始状态：

```bash
python3 /home/td/franka/scripts/robot/go_to_zero_pose.py \
  --ip 172.16.0.2 \
  --realtime ignore \
  --speed 0.1 \
  --max-step-rad 0.1 \
  --gripper-speed 0.03
```

只移动 arm 回零位姿，保持 gripper 不变：

```bash
python3 /home/td/franka/scripts/robot/go_to_zero_pose.py --ip 172.16.0.2 --skip-gripper
```

目前 `scripts/robot/go_to_zero_pose.py` 已设置为 `exported_0711` 使用的仿真初始关节姿态，
夹爪初始总开口宽度为 `0.04 m`。

仿真初始关节角为：

```text
degree: [-17.649, -6.608, 17.296, -130.236, 2.348, 123.894, 43.211]
radian: [-0.308033, -0.115331, 0.301872, -2.273047, 0.040980, 2.162358, 0.754174]
```

脚本使用分段 `JointMotion`，完成后会打印实际 joint position、joint velocity、
关节误差和真实 TCP pose。真实 policy rollout 前应先完成这一步，以对齐
`proprio_obs` 中的关节位置、关节速度和夹爪宽度。

## 4. 力和运动诊断

### A. 读取相对 baseline 的 force/torque

用这个脚本在机器人静止时检查 `O_F_ext_hat_K`。脚本会先采集一段空载 baseline，然后打印相对 baseline 的 force 和 torque 变化：

```bash
python3 /home/td/franka/scripts/diagnostics/test_franka_force_sensor.py \
  --ip 172.16.0.2 \
  --baseline-samples 20 \
  --interval 0.1 \
  --threshold-n 0.5
```

常用选项：

- `--count 100`
  - 采集固定数量的 live samples；`0` 表示持续采集。
- `--no-dynamic-baseline`
  - 固定使用初始 baseline。
- `--baseline-alpha 0.02`
  - 无接触时 dynamic baseline 的更新率。
- `--threshold-n 0.5`
  - 相对 baseline 的 force norm 超过该值时标记样本。
- `--compare-jacobian`
  - 同时用 `tau_ext_hat_filtered` 和 `zero_jacobian` 反解末端 wrench，和 `O_F_ext_hat_K` 并排打印。
- `--jacobian-frame stiffness --jacobian-mode full --jacobian-damping 0.01`
  - 默认使用 stiffness frame，对齐 `O_F_ext_hat_K`；`force` 模式只估计 `[Fx,Fy,Fz]`。

### B. 小幅运动中采样 force

用这个脚本估计机器人单向小幅运动时产生的 force/torque 测量伪影。默认不会自动往返，`--cycles` 表示连续做几次同方向 forward motion：

```bash
python3 /home/td/franka/scripts/diagnostics/test_franka_force_during_motion.py \
  --ip 172.16.0.2 \
  --dx 0.005 \
  --cycles 1 \
  --speed 0.03 \
  --sample-interval 0.02 \
  --contact-force-threshold 6.0 \
  --raw-output
```

脚本会在 `runs/..._force_during_motion/` 下创建带时间戳的 run 目录，包含：

- `samples.csv`
- `summary.json`
- `force_plot_data.csv`
- `plots/*.png`
  - 自动生成的标注图，包含 direction、speed、起始 TCP、实际位移、contact threshold、max force/torque 等信息。

重要选项：

- `--raw-output`
  - 终端实时输出只显示原始 `O_F_ext_hat_K` force/torque 和 TCP，不显示 baseline delta。
- `--compare-jacobian`
  - 在终端和 `samples.csv` 中增加 Jacobian 反解结果；CSV 列名前缀为 `j_`，并包含 `j_minus_o_*` 差值列。
- `--jacobian-frame stiffness --jacobian-mode full --jacobian-damping 0.01`
  - 默认用 `zero_jacobian(Frame.Stiffness)` 和阻尼最小二乘从 `tau_ext_hat_filtered` 估计 wrench。
- `--baseline-mode linear`
  - 用 endpoint interpolation 补偿位置相关的 baseline drift。
- `--baseline-mode exp --baseline-exp-tau 0.2`
  - 用指数曲线在起点/终点 baseline 之间过渡；`tau` 越小，越快靠近终点 baseline。
- `--baseline-mode start`
  - 固定使用起点 baseline。
- `--motion-frequency 1.0`
  - 限制 motion command 的启动频率。
- `--return-motion`
  - 每次 forward motion 后再执行反向 motion；默认不启用。
- `--stop-on-contact`
  - 检测到 contact 后调用 `robot.stop()`。
- `--no-auto-plot`
  - 只保存 CSV/JSON，不自动生成图片。
- `--plot-components norm fx fy fz`
  - 指定自动画哪些 force component。

### C. 比较不同位置的 x/y/z 平移 wrench 响应

用这个脚本回答：同样的 x/y/z 末端平移，在不同 Cartesian position 上是否产生类似的 6D force/torque 读数。

```bash
python3 /home/td/franka/scripts/diagnostics/test_franka_wrench_translation_grid.py \
  --ip 172.16.0.2 \
  --axes x y z \
  --probe-distance 0.005 \
  --cycles 2 \
  --speed 0.03 \
  --abort-on-motion-error
```

默认情况下，脚本把启动时的 TCP pose 当作 `center`，并测试附近 offset：

- `center = [0, 0, 0]`
- `x_plus = [0.03, 0, 0]`
- `x_minus = [-0.03, 0, 0]`
- `y_plus = [0, 0.03, 0]`
- `y_minus = [0, -0.03, 0]`
- `z_plus = [0, 0, 0.03]`

也可以手动指定测试位置：

```bash
python3 /home/td/franka/scripts/diagnostics/test_franka_wrench_translation_grid.py \
  --ip 172.16.0.2 \
  --position center=0,0,0 \
  --position front=0.04,0,0 \
  --position left=0,0.04,0 \
  --position up=0,0,0.03
```

输出 run 位于 `runs/..._wrench_translation_grid/`，包含：

- `samples.csv`
  - 每个采样点的 raw wrench，以及相对 baseline 的 `dfx/dfy/dfz/dtx/dty/dtz`
- `summary.json`
  - 每次 motion 的 mean delta、带符号的 max-absolute delta、max force norm 和 max torque norm

如果某个位姿接近 singularity，libfranka 可能会拒绝 Cartesian motion，并报 `cannot start at singular pose`。默认情况下，脚本会在第一次被拒绝后停止后续测试，仍然保存部分结果，并把错误写入 `summary.json`。

### D. 绘制 force-during-motion 曲线

运行 `test_franka_force_during_motion.py` 后，可以从保存的 `force_plot_data.csv` 生成曲线图：

```bash
python3 /home/td/franka/scripts/diagnostics/plot_franka_force_motion.py \
  /home/td/franka/runs/<RUN_DIR> \
  --component norm \
  --x progress \
  --baseline-source motion
```

画图选项：

- `--component norm`、`fx`、`fy` 或 `fz`
- `--x progress` 或 `time`
- `--baseline-source motion`、`exp` 或 `settled`
- `--show`
  - 保存后交互式显示图像。

默认保存到 `<RUN_DIR>/plots/`。

### E. 诊断 action 方向

用这个脚本执行小幅带符号运动，并保存执行前后的机器人 observation 和 RealSense 图片：

```bash
python3 /home/td/franka/scripts/diagnostics/diagnose_franka_directions.py \
  --robot-ip 172.16.0.2 \
  --cases dx+ dx- dy+ dy- dz+ dz- yaw+ yaw-
```

每个 case 执行前都需要手动确认。结果会保存到 `runs/..._direction_diagnosis/`。

## 5. 端到端 sim2real policy

### A. 当前 `exported_0711` cube-grasp 模型

当前主要真机入口：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0711.py \
  --steps 15
```

脚本默认在第一步打印 action 并确认一次；确认后自动执行剩余步骤。调试阶段建议每步确认：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0711.py \
  --steps 15 \
  --confirm-each-step
```

模型文件：

```text
checkpoint/exported_0711/exported/policy_actor_e2e_best_agent.pt
```

输入签名：

```text
action_history: [1, 4] float32
proprio_obs:    [1, 15] float32
wrist_rgb:      [1, 224, 224, 3] uint8
```

`proprio_obs` 的顺序为 7 维 joint position、7 维 joint velocity 和总夹爪宽度。
输出为 4 维 raw actor mean：

```text
[dx, dy, dz, gripper]
```

当前部署会先把 raw action clip 到 `[-1, 1]`，再映射为每步最大 5 mm 的
XYZ 位移和每步最大 2 mm 的夹爪宽度增量。metadata 中的训练标称频率为
60 Hz，但当前 arm backend 仍是阻塞式 `CartesianMotion`，实测约 3.7–3.8 Hz。

### B. 较早的 bundle 和 `exported_0402`

bundle policy runner：

```bash
python3 /home/td/franka/scripts/policy/run_e2e_bundle.py \
  --config /home/td/franka/configs/e2e_bundle_real_example.json \
  --robot-ip 172.16.0.2
```

如果使用 `exported_0402/` 下的新模型：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0402.py \
  --robot-ip 172.16.0.2
```

这个 runner 会自动选择 `exported_0402/` 中最新的 `policy_actor_e2e_agent_*.pt`。

如果想记录每个真实 step 的图片、proprio inputs、observations 和 actions：

```bash
python3 /home/td/franka/scripts/policy/record_exported_0402_rollout.py \
  --robot-ip 172.16.0.2 \
  --steps 10 \
  --confirm-step
```

runner 会：

1. 连接 Franka arm 和 gripper
2. 读取机器人状态
3. 从 RealSense 或静态图片读取 RGB 输入
4. 构造模型输入：
   - `action_history`
   - `proprio_obs`
   - `wrist_rgb`
5. 运行 TorchScript inference
6. 将模型 action 映射为真实机器人命令
7. 可选地执行该 step
8. 把日志写到 `runs/...`

## 6. `exported_0711` 推荐安全测试流程

### A. Preview only，不运动

先用这个模式。它会跑完整 observation 和 inference 流程，但不会移动机器人：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0711.py \
  --steps 1 \
  --preview-only
```

`--preview-only` 不发送 arm、gripper 或自动 gripper homing 命令，但仍会连接
机器人读取 proprio，并从 RealSense 获取图像。

### B. 用静态图片代替 live camera

在不接 RealSense 或不想使用 live camera 时，可以用这个验证 policy 路径：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0711.py \
  --image /home/td/franka/assets/test_static_image.png \
  --steps 1 \
  --preview-only
```

`exported_0402` runner 使用纯白图片：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0402.py \
  --robot-ip 172.16.0.2 \
  --image /home/td/franka/assets/input_white_640x480.png \
  --steps 1 \
  --preview-only
```

`exported_0402` runner 使用纯黑图片：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0402.py \
  --robot-ip 172.16.0.2 \
  --image /home/td/franka/assets/input_black_640x480.png \
  --steps 1 \
  --preview-only
```

### C. 带确认的一步真实运动

只有在 preview 看起来合理后再运行：

```bash
python3 /home/td/franka/scripts/policy/run_exported_0711.py \
  --steps 1
```

runner 会打印 proposed action，并等待输入：

```text
y / yes
```

确认后才会真正移动机器人。

### D. 预览 camera crop

用这个脚本保存和检查：

- raw camera frame
- 输入模型前的 cropped frame
- side-by-side comparison image

Live RealSense capture：

```bash
python3 /home/td/franka/scripts/camera/preview_camera_crop.py --open
```

### 采集真机视觉与初始状态对齐基准

下面的脚本只读取 Franka 和 D435，不会发送机械臂、夹爪运动或夹爪 homing 命令：

```bash
python3 /home/td/franka/scripts/calibration/capture_real_alignment_reference.py \
  --config /home/td/franka/configs/e2e_bundle_real_exported_0711.json \
  --ip 172.16.0.2 \
  --realtime ignore \
  --frames 10
```

输出目录为 `runs/<timestamp>_real_alignment_reference/`，其中包括：

- D435 原始、中值、crop 后和 policy `224x224` RGB 图像；
- D435 原始、crop 后和缩放后的 `fx/fy/cx/cy`、FOV、畸变参数；
- 实际曝光、增益、白平衡等受支持的 color sensor 参数；
- 采集前后的 Franka 关节位置、速度、TCP 位姿、夹爪宽度和错误状态；
- 拍摄期间的最大关节漂移与 TCP 平移漂移；
- 汇总文件 `alignment_reference.json` 和三阶段图像对比 `comparison.png`。

如果夹爪没有连接或不需要读取夹爪状态，可以增加 `--skip-gripper`。

静态图片 preview：

```bash
python3 /home/td/franka/scripts/camera/preview_camera_crop.py \
  --image /home/td/franka/assets/test_static_image.png \
  --open
```

脚本会在新的 `runs/..._camera_crop_preview/` 目录下保存：

- `raw.png`
- `cropped.png`
- `comparison.png`
- `crop_metadata.json`

## 7. 部署配置

主要配置文件：

- [configs/e2e_bundle_real_exported_0711.json](/home/td/franka/configs/e2e_bundle_real_exported_0711.json)
- [configs/e2e_bundle_real_example.json](/home/td/franka/configs/e2e_bundle_real_example.json)
- [configs/e2e_bundle_real_exported_0402.json](/home/td/franka/configs/e2e_bundle_real_exported_0402.json)

最重要的字段：

- `speed`
  - Franka arm 的 relative dynamics factor。
- `camera.source`
  - `realsense` 或 `image`
- `camera.enable_crop`
  - RGB 图片输入模型前是否 crop。
- `camera.crop_left`、`camera.crop_top`、`camera.crop_width`、`camera.crop_height`
  - crop box，crop 后会 resize 回模型输入分辨率。
- `camera.warmup_frames`
  - RealSense 启动后丢弃的预热帧数；0711 配置为 30，用于等待自动曝光稳定。
- `runner.steps`
  - rollout step 数量。
- `action_adapter.scales`
  - 每个 step 的物理 action scale。
- `async_gripper_commands`
  - 是否异步执行夹爪命令。0711 配置为 `true`，避免每步等待约 0.7 秒。
- `gripper_command_tolerance_m`
  - 目标宽度与当前宽度足够接近时跳过冗余夹爪命令。

默认 5-D bundle 的 action 含义：

- `action[0] -> dx`
- `action[1] -> dy`
- `action[2] -> dz`
- `action[3] -> yaw_deg`
- `action[4] -> gripper`

对 `run_exported_0711.py` 和当前 `exported_0402` bundle 来说，action 是 4-D：

- `action[0] -> dx`
- `action[1] -> dy`
- `action[2] -> dz`
- `action[3] -> gripper`

当前配置中：

- `dx/dy/dz` 单位是每 step 的 meters
- `yaw` 单位是每 step 的 degrees
- `gripper` 是每 step 的 width delta，单位 meters

0711 的异步夹爪会逐步累计期望宽度；若上一条命令尚未完成，只保留最新排队目标。
因此某一步 `robot_action.gripper_width` 可能领先于同一步 observation 中的实际宽度，
rollout 结束时 backend 会等待最后一个排队目标完成。

## 8. 计时日志

sim2real runner 现在会记录每个 step 的 timing，包括：

- `image_read_ms`
- `policy_infer_ms`
- `action_map_ms`
- `arm_move_ms`
- `gripper_move_ms`
- `settle_ms`
- `state_read_ms`
- `step_total_ms`

这些信息会出现在 run output JSON 的：

```json
"info": {
  "timing": { ... }
}
```

这有助于判断瓶颈来自：

- image capture
- model inference
- arm motion
- gripper motion
- settle delay

### 当前 0711 实测频率

启用异步夹爪和单次 gripper-state 读取后，最近一次 15-step rollout 的平均耗时约为：

```text
arm_move_ms:      172.5
gripper_move_ms:    0.01
settle_ms:          50.3
state_read_ms:      12.4
image_read_ms:       3.9
policy_infer_ms:    20.4
```

完整 policy 周期约 265 ms，即约 3.8 Hz。当前主要瓶颈是阻塞式
`robot.move(CartesianMotion(...))`：每发送一个小位移，都要等待 arm 完成后才能
计算下一步。仅优化现有阻塞架构很难稳定超过约 5 Hz；若要接近训练的 60 Hz，
需要另行实现基于 `pylibfranka.start_cartesian_velocity_control()` 的连续控制循环，
并先确认训练环境中 action 是位置增量还是速度命令。

## 9. 输出日志

每次 run 都会在下面目录创建一个新目录：

- `runs/`

典型内容：

- `config.json`
- `rollout.jsonl`
- `summary.json`
- `rgb/step_XXXX.png`

`rollout.jsonl` 每一步会记录：

- `raw_action`
- `clipped_action`
- `robot_action`
- `action_history` 和 `proprio_obs`
- 执行前后 observation
- `motion_executed` 和 motion errors
- timing
- 异步夹爪的 pending、queued 和 completed 状态

由 `record_exported_0402_rollout.py` 创建的 recorded rollouts 还包含：

- `step_data/step_XXXX.npz`

每个 `step_data` 文件包含：

- `action_history`
- `proprio_obs`
- `wrist_rgb`
- `raw_action`
- `clipped_action`

由 `test_franka_force_during_motion.py` 创建的 force 诊断 run 包含：

- `samples.csv`
- `summary.json`
- `force_plot_data.csv`
- `plots/*.png`

由 `test_franka_wrench_translation_grid.py` 创建的 translation-grid wrench 测试包含：

- `samples.csv`
- `summary.json`

由 `diagnose_franka_directions.py` 创建的方向诊断 run 包含：

- `summary.json`
- `NN_<case>/before.png`
- `NN_<case>/after.png`
- `NN_<case>/result.json`

## 10. 重要安全提示

这个工作区包含一些保护措施：

- preview-only mode
- 每 step 手动确认
- `--confirm-each-step`
- action clipping
- workspace clamping
- gripper width clamping

但这些仍然不能保证真实机器人绝对安全。任何真实运动前：

- 清空工作空间
- 保持 action 很小
- 优先一次只跑一个 step
- 执行前检查 proposed action
- 从 `--preview-only` 开始

## 11. 相关文档

- [docs/E2E_BUNDLE_DEPLOY.md](/home/td/franka/docs/E2E_BUNDLE_DEPLOY.md)
- [docs/SIM2REAL_FRAMEWORK.md](/home/td/franka/docs/SIM2REAL_FRAMEWORK.md)
- [deploy_bundle_e2e/README.md](/home/td/franka/deploy_bundle_e2e/README.md)
