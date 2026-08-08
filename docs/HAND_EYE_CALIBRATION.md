# Franka–D435 Eye-to-Hand 标定

本工具标定固定在机器人外部的 D435 RGB 相机。标定板由 Franka 夹爪夹持，
最终输出 `T_base_color`：

```text
p_base = T_base_color * p_color
```

结果属于 D435 的 RGB color optical frame（X 向右、Y 向下、Z 向前）。
如果三维点来自深度光学坐标系，必须先使用 RealSense 内部外参变换到彩色
光学坐标系，或先将深度对齐到彩色流。

## 固定配置

- D435 序列号：`215322076207`
- 彩色流：`640x480 RGB8 @ 30 Hz`
- 棋盘格：`11x8` 个内部角点
- 单格边长：`0.015 m`
- 求解样本：25 个
- 独立验证样本：5 个

配置位于：

```text
configs/eye_to_hand_d435_215322076207.json
```

## 运行前检查

1. 清空机器人、标定板和线缆可能经过的空间，急停保持可用。
2. 确认标定板被夹爪可靠夹紧，运动期间禁止任何夹爪命令。
3. 确认相机支架不动；标定完成后移动相机必须重新标定。
4. 将 TCP 摆到棋盘完整、清晰可见的中心姿态。
5. 确认棋盘最低边缘与桌面/障碍物有足够间距。
6. 关闭 RealSense Viewer。相机不能同时被 Viewer 和采集程序占用。

采集程序会读取夹爪宽度，但代码中不调用 `homing`、`open`、`move` 或
`grasp`。

## 1. 检查相机和棋盘

```bash
source /home/td/franka/.venv/bin/activate

python scripts/calibration/collect_eye_to_hand.py \
  --camera-check
```

必须看到正确的相机序列号，且 10 帧中能稳定检测到棋盘。没有图形桌面时可加
`--no-window`。

## 2. 轨迹 dry-run

```bash
python scripts/calibration/collect_eye_to_hand.py \
  --phase calibration \
  --dry-run
```

dry-run 会连接相机、机器人和夹爪并读取状态，但不会发送机械臂或夹爪运动
命令。它会打印 25 个求解目标和 5 个验证目标，并检查 TCP workspace 和约
220 mm 工具包络。

## 3. 采集求解样本

推荐首次使用逐姿态确认模式：

```bash
python scripts/calibration/collect_eye_to_hand.py \
  --phase calibration
```

每个姿态执行前都必须输入 `y` 或 `yes`。程序采用绝对目标，并把轨迹分成不
超过 `10 mm/5 deg` 的小段。所有倾斜姿态都会先把 TCP 抬高 30 mm。

每个姿态采集 10 帧原始 RGB，保存角点、PnP、机器人状态和夹爪状态。失败时：

- `r`：保持当前姿态并重新采集；
- `s`：跳过，数据集保持未完成；
- `a`：停止并请求 `robot.stop()`。

如果当前中心姿态已经通过 dry-run、运动空间已清空，并且线缆、标定板夹持和
急停均已确认，可用连续模式一次确认后自动采集全部 25 个姿态：

```bash
python scripts/calibration/collect_eye_to_hand.py \
  --phase calibration \
  --continuous
```

连续模式保留所有 workspace、工具包络、机器人状态、夹爪宽度、TCP 静止度和
重投影误差检查。单个姿态采集失败时默认自动重试，最多 3 次；仍失败则调用
`robot.stop()` 并保留数据集供 `--resume` 续采。可用
`--max-capture-attempts N` 调整次数。阶段完成后机械臂自动返回本次标定中心。

中断后使用输出的数据集目录继续：

```bash
python scripts/calibration/collect_eye_to_hand.py \
  --phase calibration \
  --continuous \
  --resume runs/YYYYMMDD_HHMMSS_eye_to_hand
```

已接受的样本永远不会被覆盖。

## 4. 初步离线求解

```bash
python scripts/calibration/solve_eye_to_hand.py \
  --dataset runs/YYYYMMDD_HHMMSS_eye_to_hand
```

没有独立验证样本时，报告会包含五种算法的诊断候选，但
`accepted=false`，不会发出可部署的顶层 `T_base_color`。

如果需要在采集尚未完成时用当前所有已接受样本做诊断拟合，可以使用：

```bash
python scripts/calibration/solve_eye_to_hand.py \
  --dataset runs/YYYYMMDD_HHMMSS_eye_to_hand \
  --use-available
```

该模式至少需要 3 个已接受的求解样本，并使用当前所有可用的
求解和验证样本。在没有满足配置的 25+5 数量时，报告会将位姿标记为
`provisional=true` 且 `deployment_ready=false`，用于诊断而不是最终部署。

如需人工排除已知错误样本：

```bash
python scripts/calibration/solve_eye_to_hand.py \
  --dataset runs/YYYYMMDD_HHMMSS_eye_to_hand \
  --exclude calibration_005_raised_center
```

排除样本会导致数量检查失败，因此主要用于诊断，不会静默形成可部署结果。

## 5. 采集独立验证样本

保持相机和棋盘夹持完全不变：

```bash
python scripts/calibration/collect_eye_to_hand.py \
  --phase validation \
  --resume runs/YYYYMMDD_HHMMSS_eye_to_hand
```

验证阶段包含 5 个未参与拟合的组合姿态。

## 6. 生成最终报告

再次运行求解：

```bash
python scripts/calibration/solve_eye_to_hand.py \
  --dataset runs/YYYYMMDD_HHMMSS_eye_to_hand
```

报告保存为：

```text
DATASET/reports/<timestamp>/calibration_report.json
DATASET/reports/<timestamp>/calibration_report.md
```

只有以下条件全部满足时，顶层结果才会包含 `T_base_color`：

- 25 个求解样本和 5 个验证样本均有效；
- 每个样本重投影 RMS 不超过 `0.8 px`；
- 至少 3 种 OpenCV 方法有效并在 `5 mm/1 deg` 内形成共识；
- 5 个独立验证姿态的平移误差均不超过 `5 mm`，旋转误差均不超过 `1 deg`。

失败报告仍保留所有候选与残差，用于定位角点翻转、夹板滑动、相机移动或
姿态数据方向错误，但不会自动修改任何部署或 TacEx 配置。

## 变换方向

每个样本记录：

```text
T_base_gripper
T_color_target
```

eye-to-hand 求解先计算：

```text
T_gripper_base = inverse(T_base_gripper)
```

然后调用 OpenCV `calibrateHandEye`，将返回值解释为 `T_base_color`。验证
方程是：

```text
T_gripper_base * T_base_color * T_color_target = T_gripper_target
```

右侧在所有姿态中应保持恒定。
