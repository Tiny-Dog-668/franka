# 修改历史

本文件按时间倒序记录 `/home/td/franka` 真机部署轨道的代码、配置和文档变更。维护规则见 [`../AGENT.md`](../AGENT.md) 第 9 节。

## 关于本文件的数据来源

主仓库的 Git 正式历史只有 3 条提交，全部集中在 2026-07-12，之后所有工作都留在工作区未提交。因此本文件的时间线由三部分拼成：

| 来源 | 可靠性 |
| --- | --- |
| Git 提交（`git log`） | 确定 |
| 工作区文件修改时间（`find -printf '%T'`） | 确定的是**最后一次修改时间**，不是首次创建时间；同一天的多个文件按 mtime 排序，反映的是当天工作顺序的近似 |
| 配置、metadata、README 和专题文档中记录的技术结论 | 确定（有代码或实测依据） |

被 Git 忽略的目录（`runs/`、`.venv/`、`dist/`、`artifacts/`、`*.pt`）不在本文件覆盖范围内。`TacEx/` 是独立 Git 仓库，其变更记录在 `TacEx/docs/DEVLOG.md`。

## 追加条目的格式

每次实质性改动后在“变更时间线”顶部追加一节：

```markdown
## YYYY-MM-DD 一句话主题

**变更**：改了什么。
**动机**：为什么改。
**影响**：涉及的模块、张量维度、共享内存 ABI、已有 checkpoint 或安全参数；无影响就写“无”。
**验证**：实际运行过的命令和结果；未运行的验证明确写“未运行”。
**文件**：实际创建或修改的路径。
```

不得把未执行的实验或未验证的推测写成事实。

---

# 变更时间线

## 2026-08-08 GelSight 触觉输入与 AprilTag 位姿验证

**变更**：新增 0808 GelSight 部署配置与入口，配置中首次启用 `tactile_camera`（左右 device 0/6，3280x2464@25，`first_frame_timeout_s: 10`），模型指向 `checkpoint/0808/rma_gelsight_student_latest.pt`。工作空间 z 下限从 0.02 放宽到 0.01。同日更新了 `e2e_bundle.py`、`streaming.py`、`streaming_server9.py`、统一入口 `run_exported_0711.py`，以及原生 worker 源码；新增 AprilTag 方块位姿实时验证工具与对应测试。

**影响**：`BundleTactileCameraConfig` 与 `GelSightPairCamera` 进入主部署路径；触觉输入只对 0808 配置生效，其余配置行为不变。工作空间下限放宽属于安全参数变动。

**验证**：单元测试 77 项，1 项 ERROR（见下方“已知问题”条目）。GelSight 策略的真机验收状态**待确认**。

**文件**：`configs/e2e_bundle_real_exported_0808_gelsight.json`、`scripts/policy/run_exported_0808_gelsight.py`、`scripts/camera/gelsight_start.py`、`scripts/calibration/live_apriltag_cube_pose.py`、`tests/test_live_apriltag_cube_pose.py`、`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、`scripts/policy/run_exported_0711.py`、`tests/test_e2e_bundle_runtime.py`、`native/franka_server9_streaming_worker.cpp`

## 2026-08-03 共享内存 ABI 定稿、heatmap 变体与位姿数据集

**变更**：`franka_sim2real/server9_ipc.py` 定稿为 `ABI_VERSION = 8`，含 `SharedData`、`TraceEntry`、`FciLogEntry` 结构与 `validate_ctypes_layout()` 布局自检；补齐 `tests/test_server9_streaming.py` 和 `tests/test_streaming.py`。新增 0803 DR heatmap 变体入口与配置。`franka_sim2real/types.py` 扩展外力/力矩报告（`external_wrench_norms`、`format_error_wrench_report`）。新增方块位姿数据集录制与「策略预测 vs AprilTag」对比脚本，以及 reflex wrench 录制工具。

**影响**：ABI 从 7 升到 8，Python 与 C++ 两侧必须同版本，旧的 worker 二进制不兼容，需重新构建。

**验证**：ABI 布局由 `tests/test_server9_streaming.py` 与 C++ 侧 `offsetof` 静态断言双向校验。

**文件**：`franka_sim2real/server9_ipc.py`、`franka_sim2real/types.py`、`tests/test_server9_streaming.py`、`tests/test_streaming.py`、`configs/e2e_bundle_real_exported_0803_dr_heatmap.json`、`configs/e2e_bundle_real_exported_0802_dr_simactuator_gpu.json`、`configs/e2e_bundle_real_exported_0802_dr_simactuator_oracle_gpu.json`、`scripts/policy/run_exported_0803_dr_heatmap.py`、`scripts/calibration/record_cube_pose_dataset.py`、`scripts/calibration/evaluate_policy_vs_apriltag.py`、`scripts/robot/record_franka_reflex_wrench.py`

## 2026-08-02 sim_actuator_velocity 控制律、eye-to-hand 标定与 RMA 视觉头验证

单日内完成的大批改动，按当天工作顺序分组。

### 控制律替换（本仓库最关键的一次修正）

**变更**：新增 `sim_actuator_velocity` 控制律并设为默认，取代 `joint_position_pursuit`。新律复现 Isaac Lab implicit PD 的稳态关系 `qd = (stiffness / damping) * dq_ik`；`FRANKA_PANDA_HIGH_PD_CFG` 是 400/80，故 `reference_velocity_gain = 5.0`。1 kHz 回调改用临界阻尼速度跟踪器把速度参考转成位置指令，逐周期限制速度、加速度、跃度。

**动机**：`joint_position_pursuit` 把参考点放在实测位置前方一个固定关节增量处，`libfranka::limitRate` 读到的隐含速度是 `增量 / 周期`，任何可用增量都会饱和到 `maximum_joint_velocities`。后果有两个：关节速度与策略动作幅值完全无关；逐关节 clamp 会旋转笛卡尔方向——在 0802 参考位姿上实测把 `[-0.577, 0.577, -0.577]` 变成 `[-0.104, 0.991, 0.088]`，偏了 54.5°。

**影响**：饱和动作对应的 TCP 速度从 50 mm/步降到 8.33 mm/步。`commissioning_action_limit` 从此成为线性且唯一的调速旋钮（`0.1 → 0.3 → 0.6 → 1.0` 对应 0.83 → 2.5 → 5.0 → 8.33 mm/步）。旧配置保留，未被覆盖。

**验证**：worker `--self-test` 复现第二 tick 参考缩小、stale action 和反向指令序列，逐周期断言速度、加速度、跃度不越界。真机侧用 `analyze_streaming_tracking.py` 检查 ratio、slope 接近 1，方向误差接近 0。

### 停止时序修复

**变更**：停止路径改为与动作路径共用同一个临界阻尼、有界跃度的速度跟踪器，参考设为零；只有当一步精确归零所需的加速度和跃度都在配置包络内时才落到 bit-exact zero。随后要求 `q_command == q_d`、`dq_d` 与 `ddq_d` 为零、实测关节速度不超过 0.005 rad/s，连续保持 100 个 1 kHz 周期后才发 `MotionFinished`。

**动机**：早期停止路径用 `limitRate(q_target=q_d)` 刹车，真实 FCI 日志证明它在较高入口速度穿越零点时仍产生离散尖峰——实测第 6 关节跃度约 7654 rad/s³（上限 2000），加速度约 11.9 rad/s²（上限 10）。错误在过零阶段产生但到 `MotionFinished` 才上报，表面看像结束条件问题。

**影响**：这是既有 bug 的修复，与控制律选择无关。

**验证**：`--self-test` 从正负多个入口速度扫描完整停止过程，逐周期断言并检查 2 s watchdog 内精确停止。

### 早期不安全实现的回退

**变更**：移除「把 `abs(qd_ref)` 直接用作 `libfranka::limitRate` 动态硬速度上限」的做法。

**动机**：参考减小时该做法不安全。若当前 `dq_d` 已高于新上限，`limitRate` 会为立即进入新包络而跳到最大减速度；7 rad/s² 的单周期跳变相当于 7000 rad/s³，超过配置的 1500 rad/s³，触发 `joint_motion_generator_velocity_discontinuity`。同一动作的第二次 DLS 重算或一次 policy deadline miss 都可能触发。

**影响**：现在速度参考可以任意缩小、清零或反向，跟踪器仍保证轨迹导数连续。

### eye-to-hand 标定链路

**变更**：新增 `franka_sim2real/calibration/hand_eye.py`（棋盘格 11x8 内角点、15 mm 方格的 eye-to-hand 求解）、采集脚本（支持一次确认后连续采集 25 个姿态）、求解脚本、标定结果配置和单元测试；配套 `docs/HAND_EYE_CALIBRATION.md`。同时新增 AprilTag 打印资源生成和基准标识工具。

**验证**：`tests/test_hand_eye_calibration.py`。首次使用要求先跑 `--camera-check` 和 `--dry-run`。

### RMA 视觉头验证与消融

**变更**：新增 `monitor_rma_object_position.py`（实时窗口显示模型输入的 224x224 RGB、叠加原始/EMA 平滑的 `robot_root` XYZ、左右 contact 概率、推理耗时与曝光设置）和 `validate_rma_object_position.py`（对 RealSense、单张图片、NPZ 或 CSV manifest 计算 XYZ MAE/RMSE、3D 误差分位数和 10 mm 阈值通过率）。新增 oracle 消融入口，绕过 `vision_encoder + adaptation_head`，把真实物体 XYZ 直接送入同一个 frozen `actor_core`，contact 恒为 `[0, 0]`；支持只替换单个视觉分支做对照。

**影响**：oracle 路径默认走 CPU（实测约 0.2 ms/步，远低于 33.3 ms 预算），不依赖 CUDA。恒零 contact 适合验证到达阶段；发生真实手指接触后不再符合 Teacher 输入契约，因此仅凭该模式抓取失败不能判定 actor 或低层控制无效。

**验证**：`tests/test_rma_object_position_validation.py`、`tests/test_gpu_policy_deployment.py`。无真值时脚本明确标记 `NO_GROUND_TRUTH`，不据此判断精度。

### 其它

**变更**：新增 0802 DR / smooth / simactuator / simactuator_gpu 四个配置与入口；GPU 入口默认加载 `checkpoint/0802_DR_gpu/rma_student_dr_sim2real_gpu.pt` 并用 `cuda:0`。新增 `analyze_streaming_tracking.py` 跟踪质量分析。0801 配置正式落盘并配套 FCI 主机准备脚本（`--interface enp3s0 --cpu 21`）与 `docs/PYLIBFRANKA_STREAMING.md`。更新 `go_to_zero_pose.py`。

**文件**：`franka_sim2real/calibration/hand_eye.py`、`configs/e2e_bundle_real_exported_0801.json`、`configs/e2e_bundle_real_exported_0802_dr.json`、`configs/e2e_bundle_real_exported_0802_dr_smooth.json`、`configs/e2e_bundle_real_exported_0802_dr_simactuator.json`、`configs/eye_to_hand_d435_215322076207.json`、`scripts/policy/run_exported_0802_dr*.py`、`scripts/policy/monitor_rma_object_position.py`、`scripts/policy/validate_rma_object_position.py`、`scripts/calibration/collect_eye_to_hand.py`、`scripts/calibration/solve_eye_to_hand.py`、`scripts/calibration/generate_apriltag_printable.py`、`scripts/calibration/identify_fiducial_marker.py`、`scripts/diagnostics/analyze_streaming_tracking.py`、`scripts/fci/prepare_fci_host.sh`、`scripts/fci/test_fci_network.sh`、`scripts/robot/go_to_zero_pose.py`、`docs/PYLIBFRANKA_STREAMING.md`、`docs/HAND_EYE_CALIBRATION.md`、`docs/LIBFRANKA_CONTROL_METHOD.md`、`docs/SMOOTH_STREAMING_ARCHITECTURE.md`、`tests/test_hand_eye_calibration.py`、`tests/test_gpu_policy_deployment.py`、`tests/test_rma_object_position_validation.py`

## 2026-08-01 Streaming 架构落地：原生 server9 worker

**变更**：从 blocking 控制切换到分层 streaming 控制。新增 0801 入口，以及两个构建脚本：`build_franka_server9_worker.sh` 构建 1 kHz libfranka C++ worker（`g++ -std=c++17`，链接 franky wheel 内的 `libfranka 0.17.0`，构建后自动跑 `--version` 与 `--self-test`）；`build_pylibfranka_wheel.sh` 构建 pylibfranka 0.21.1 的 streaming 补丁 wheel。

**动机**：blocking 路径每个策略 tick 下发一次相对 `CartesianMotion` 并等待完成，每步都有起停，无法连续运动且时序不受控。

**影响**：控制拆成三层——Python 30 Hz 策略、Python 60 Hz DLS IK、C++ 1 kHz FCI 读写，强制 `realtime: "enforce"`。streaming 路径的 XYZ 是机器人基座坐标系，**不使用** legacy blocking 路径的 Y/Z 取反；两条路径的坐标约定从此不同。控制期间只缓存数据，`stop_control()` 后才写 RGB、rollout、control trace 和 timing。归一化动作默认限制在 `[-0.10, 0.10]`，`--allow-full-scale` 禁止与 `--yes` 组合且始终要求一次人工会话确认。

**待确认记录**：当前机器人是 FCI robot server version 9，`async_position` backend 依赖 pylibfranka 0.21.1 / server v10，本机不可用，实际生产路径是 `server9_joint_position`。

**文件**：`scripts/policy/run_exported_0801.py`、`scripts/build_franka_server9_worker.sh`、`scripts/build_pylibfranka_wheel.sh`、`native/franka_server9_streaming_worker.cpp`、`third_party/pylibfranka_streaming_patch/`

## 2026-07-26 0726 严格契约部署与相机 crop 采样

**变更**：新增 0726 入口与配置，动作从 5 维收敛到 4 维 `[dx, dy, dz, gripper]`，相机 crop 为 `640x480 → x[80:560], y[0:480] → 224x224`，动作历史逐维缩放 `[0.05, 0.05, 0.05, 0.01]`、延迟 1 步。新增 D435 crop 样本采集脚本。

**影响**：0726 严格使用训练契约的最大动作尺度（XYZ 每步最多 0.05 m，夹爪总宽度每步最多变化 0.01 m），因此第一次必须检查打印值，不能直接用 `--yes`。

**文件**：`configs/e2e_bundle_real_exported_0726.json`、`scripts/policy/run_exported_0726.py`、`scripts/camera/capture_d435_crop_samples.py`

## 2026-07-25 标定包骨架

**变更**：创建 `franka_sim2real/calibration/` 包。

**文件**：`franka_sim2real/calibration/__init__.py`

## 2026-07-12 Git 正式历史（全部三条提交）

| 提交 | 作者 | 说明 |
| --- | --- | --- |
| `d8b944f` | ZGD | `Initial commit: Franka sim2real toolkit` |
| `7683a11` | ZGD | `docs: simplify README` |
| `75702fd` | ZGD | `feat: harden 0712 real-policy deployment` |

**变更**：仓库首次纳入 Git 并推送到 `origin/main`。同日新增 0712 严格契约配置与入口，以及 `docs/GITHUB_UPLOAD.md`。

**影响**：这是分支 `main` 目前的 HEAD。**此后所有工作（0726 至 0808）均未提交**，只存在于工作区。

**文件**：`configs/e2e_bundle_real_exported_0712.json`、`scripts/policy/run_exported_0712.py`、`docs/GITHUB_UPLOAD.md`

## 2026-07-11 0711 基线部署与真机后端

**变更**：建立 0711 部署基线：`franka_sim2real/config.py`、基于 franky 的 `envs/franka_real.py`（`RealFrankaEnv`），以及 0711 配置。输入契约为 `action_history[4]`、`proprio_obs[15]`、`wrist_rgb[224,224,3]`，输出 `[dx, dy, dz, gripper]`。新增视觉对齐验证与真机对齐基准采集（只读，不运动）脚本。

**影响**：`scripts/policy/run_exported_0711.py` 从此成为所有部署入口的**唯一实现**，后续版本脚本只有约 10 行，import 它的 `REPO_ROOT` 和 `main` 并指定 `DEFAULT_CONFIG`。这个约定沿用至今。

**文件**：`franka_sim2real/config.py`、`franka_sim2real/envs/franka_real.py`、`configs/e2e_bundle_real_exported_0711.json`、`scripts/policy/run_exported_0711.py`、`scripts/policy/validate_visual_alignment.py`、`scripts/calibration/capture_real_alignment_reference.py`

## 2026-05-05 力与力矩诊断工具集

**变更**：新增静止状态外力检查、小幅运动中 wrench 采集、wrench 平移网格测试和绘图脚本；新增机器人姿态工具（`go_to_safe_joint_pose.py`、`orient_current_pose_down.py`）与手动小幅位移工具 `minimal_franka_move.py`（支持 `--dx/--dy/--dz/--roll/--pitch/--yaw` 和夹爪命令，执行前要求确认）。

**文件**：`scripts/diagnostics/test_franka_force_sensor.py`、`test_franka_force_during_motion.py`、`test_franka_wrench_translation_grid.py`、`plot_franka_force_motion.py`、`scripts/robot/go_to_safe_joint_pose.py`、`orient_current_pose_down.py`、`minimal_franka_move.py`

## 2026-04-02 0402 导出策略部署

**变更**：首个「导出策略 + 专用配置 + 专用入口」组合，附 rollout 录制与 NPZ 检查工具。这确立了「每个训练批次一份配置一个入口」的演进方式。

**文件**：`configs/e2e_bundle_real_exported_0402.json`、`scripts/policy/run_exported_0402.py`、`record_exported_0402_rollout.py`、`inspect_step_npz.py`

## 2026-04-01 E2E bundle 部署框架与实时内核工具

**变更**：建立 bundled TorchScript 部署框架（`franka_sim2real/runner.py`、`scripts/policy/run_e2e_bundle.py`）与 sim2real 验证入口。新增相机 crop 预览、动作方向诊断、机器人状态读取脚本。新增实时内核安装与主机验证脚本。写出首批文档：`E2E_BUNDLE_DEPLOY.md`、`SIM2REAL_FRAMEWORK.md`、`22_04_realtime_workflow.txt`、`after_realtime_boot_commands.txt`。

**记录的环境事实**：Ubuntu 22.04 + `5.15.0-1032-realtime` 内核；`setup_franka_realtime_root.sh` 安装 `ubuntu-realtime`、配置 `@realtime` group、PAM limits 和 performance systemd unit。

**文件**：`franka_sim2real/runner.py`、`scripts/policy/run_e2e_bundle.py`、`scripts/policy/sim2real_validate.py`、`scripts/camera/preview_camera_crop.py`、`scripts/diagnostics/diagnose_franka_directions.py`、`scripts/robot/read_franka_state.py`、`read_current_zero_pose.py`、`tools/system/setup_franka_realtime_root.sh`、`verify_franka_host.sh`、`set_default_generic_kernel.sh`、`configs/e2e_bundle_real_example.json`、`configs/e2e_bundle_real_agent_{100000,150000,200000}.json`、`docs/E2E_BUNDLE_DEPLOY.md`、`docs/SIM2REAL_FRAMEWORK.md`

## 2026-03-27 ~ 2026-03-28 仓库骨架

**变更**：建立 `franka_sim2real` 包骨架：`envs/`（`base.py`、`mock_sim.py` 运动学 mock）、`safety.py`、`metrics.py`、`policies.py`、`example_policy.py`。新增网络检查与 CPU performance 脚本，以及 sim2real 验证和 Isaac Sim TorchScript 示例配置。

**文件**：`franka_sim2real/__init__.py`、`envs/base.py`、`envs/mock_sim.py`、`envs/__init__.py`、`safety.py`、`metrics.py`、`policies.py`、`example_policy.py`、`tools/system/check_franka_network.sh`、`franka-set-performance.sh`、`configs/sim2real_validation_example.json`、`configs/isaacsim_torchscript_example.json`

---

# 长期未决事项

这些不是某一次变更，而是跨越多个版本仍未解决的问题。修复后请移到时间线中并从这里删除。

| 事项 | 状态 | 说明 |
| --- | --- | --- |
| 0726 至 0808 的工作全部未提交 | 未决 | `git status --short` 显示大量 untracked 与 modified；`TacEx/` 整个目录 untracked |
| `tests/test_streaming.py` 有 1 个 ERROR | 未决 | `test_rma_streaming_contract_allows_longer_operator_selected_run` 报 `AttributeError: 'types.SimpleNamespace' object has no attribute 'metadata'`。测试桩缺属性，非生产代码缺陷 |
| `README.md` 中的 ABI 版本号过期 | 未决 | 多处写 “abi-7”，实际 `ABI_VERSION` 与 `kAbiVersion` 均为 8 |
| 根仓库无依赖清单 | 未决 | 没有 `requirements.txt` / `pyproject.toml`，依赖只在本机 `.venv`，环境无法复现 |
| `TacEx/` 的版本管理方式 | 待确认 | 是否以 submodule 或独立仓库形式正式纳入 |
| 0808 GelSight 策略真机验收 | 待确认 | `checkpoint/0808/rma_gelsight_student_latest.pt` 是否已完成真机验收 |
