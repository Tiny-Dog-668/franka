# AGENT.md

本文件是 `/home/td/franka` 仓库的长期维护约束。任何自动化助手或协作者在修改代码、配置或文档前，必须先阅读本文件，并优先遵守这里的项目事实、工作步骤、安全红线和验证规则。

配套文档：

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)：具体架构（模块职责、数据流、控制链路、接口契约）。
- [`docs/CHANGE_HISTORY.md`](docs/CHANGE_HISTORY.md)：修改历史（按时间线记录代码、配置、文档变更）。
- [`README.md`](README.md)：面向操作者的命令手册。

> 本仓库直接驱动一台真实的 Franka Panda 机械臂。任何错误的动作尺度、坐标方向或初始状态判断都可能造成硬件损坏或人身伤害。**宁可拒绝执行，也不要猜测。**

## 1. 项目定位

这是一个 **Franka Panda 真机 sim2real 部署工具链**，把在 Isaac Lab 中训练的 RMA 策略导出为 TorchScript，在真机上以 30 Hz 推理 / 60 Hz IK / 1 kHz FCI 的分层频率执行。

仓库是**双轨结构**，两条线的约束不同：

| 轨道 | 路径 | 内容 | Git 状态 |
| --- | --- | --- | --- |
| 真机部署（本文件管辖） | 仓库根目录 | `franka_sim2real/`、`scripts/`、`native/`、`configs/`、`tests/` | 主仓库 `/home/td/franka/.git` |
| 仿真训练 | `TacEx/` | Isaac Sim / Isaac Lab + skrl PPO 训练与 TorchScript 导出 | **独立 Git 仓库**，在主仓库中处于 untracked 状态，另有自己的 `TacEx/AGENTS.md` |

修改 `TacEx/` 下的内容时遵守 `TacEx/AGENTS.md`；修改根目录内容时遵守本文件。不要把两边的规则互相套用，也不要在一次改动中同时跨越两个仓库边界，除非用户明确要求。

## 2. 已确认的项目事实

以下均从代码、配置或实测确认，不是从文件名推断。

| 项目 | 值 | 依据 |
| --- | --- | --- |
| 机器人 | Franka Panda 7-DOF，FCI robot server version **9** | `docs/PYLIBFRANKA_STREAMING.md`、`docs/LIBFRANKA_CONTROL_METHOD.md` |
| 机器人 IP | `172.16.0.2` | 所有 `configs/*.json` 的 `robot_ip` |
| FCI 网卡 / CPU | `enp3s0` / CPU 21 | `scripts/fci/prepare_fci_host.sh`、config `streaming.server9_control_cpu` |
| 相机 | Intel RealSense D435，序列号 `215322076207`，640x480@30 | config `camera` |
| 模型输入 crop | `left=100, top=34, width=400, height=398` → resize 224x224 | 0802 及之后的 config `camera` |
| 触觉 | GelSight Mini 双相机，device 0 / 6，3280x2464@25，仅 0808 配置启用 | `configs/e2e_bundle_real_exported_0808_gelsight.json` 的 `tactile_camera` |
| 当前主力策略 | RMA Student TorchScript，metadata `kind: tacex_rma_student_torchscript`，`version: 6` | `checkpoint/0802_DR_gpu/rma_student_dr_sim2real_gpu.json` |
| 策略输入 | `wrist_rgb[224,224,3] uint8 NHWC`、`proprio_obs[15]`、`action_history[4]`，顺序固定 | 同上 `input_order` / `input_signature` |
| 策略输出 | `mean_actions[4]` = `[dx, dy, dz, gripper]`，界 `[-1, 1]` | 同上 `output_signature`、`actor_mean_bounds` |
| 动作尺度 | `[0.05, 0.05, 0.05, 0.01]`（米） | config `action_adapter.scales` |
| 控制频率 | policy 30 Hz / IK 60 Hz / FCI 1000 Hz | config `streaming`、`native/franka_server9_streaming_worker.cpp` |
| 共享内存 ABI | **8** | `franka_sim2real/server9_ipc.py:ABI_VERSION`、`native/...cpp:kAbiVersion` |
| 控制律 | `joint_position_pursuit`（0）与 `sim_actuator_velocity`（1，当前默认） | `native/...cpp:enum class ControlLaw` |
| 工作空间 | XYZ `[0.2, -0.3, 0.02] ~ [0.65, 0.3, 0.45]`（0808 下限 z=0.01） | config `workspace` |
| 训练物体范围 | x ∈ [0.45, 0.55] m，y ∈ [-0.05, 0.05] m，z ≈ 0.026 m | `README.md`、metadata `cube_position_center` |
| Python 环境 | 本机 `.venv`（Python 3.10），无 `requirements.txt` / `pyproject.toml` | 仓库根目录实测 |
| 测试现状 | 77 个 unittest，**1 个 ERROR**（见第 8 节） | `.venv/bin/python -m unittest discover -s tests` 实测 |

## 3. 每次工作前必须执行的步骤

1. 阅读 `AGENT.md`（本文件）。
2. 阅读 `README.md` 中与任务相关的章节。
3. 阅读 `docs/ARCHITECTURE.md` 中对应模块的说明。
4. 运行 `git status --short`，识别用户已有的未提交改动。**本仓库长期存在大量未提交改动，这是正常状态，不是需要清理的问题。**
5. 运行 `git log --oneline -n 10` 了解已提交历史（注意：正式历史很短，大部分演进只存在于工作区和 `docs/CHANGE_HISTORY.md`）。
6. 用 `rg` 定位相关文件，不要只根据文件名判断功能。
7. 阅读相关入口脚本、配置文件和被调用模块的实际实现。
8. 确认动作维度、观测 key、坐标系约定、频率和限幅来自实际代码或 metadata，而不是记忆或推断。
9. 输出对当前实现的简短理解，列出计划创建或修改的文件。
10. 说明改动可能影响的模块、张量维度、共享内存 ABI 和已有 checkpoint。
11. 对无法从仓库确认的内容明确写“待确认”，不要编造。
12. 用户限制修改范围时，只修改允许的文件。
13. 改完后同步更新 `docs/ARCHITECTURE.md` 和 `docs/CHANGE_HISTORY.md`，并报告实际执行过的验证命令。

## 4. 真机安全红线

这些规则优先于任何其他目标，包括用户要求的“快一点”。

- **不要自作主张让机器人运动。** 只有用户明确要求执行真机命令时才运行会发送控制指令的脚本。
- **验收顺序不可跳过**：`--validate-only`（不连硬件）→ `--preview-only`（只读状态和图像）→ `--streaming-check`（通信检查）→ `--steps 1`（单步，需人工输入 `y`/`yes`）→ 多步。
- **不要在未验收的情况下加 `--yes`**，不要建议用户加 `--yes` 来跳过第一次单步确认。
- **不要放宽安全参数**：`commissioning_action_limit`、`workspace.minimum/maximum`、`maximum_joint_velocities/accelerations/jerks`、`collision_behavior`、`initial_state.enforce` 和各类 tolerance，只能在用户明确要求且说明理由时修改，且必须在回复中标注影响。
- **不要关闭初始状态门禁**（`initial_state.enforce: false`）。门禁失败时正确做法是用 `scripts/robot/go_to_zero_pose.py` 回到训练初始姿态，不是改容差。
- **不要改动坐标系约定**：streaming 路径的 XYZ 是机器人基座坐标系且**不做** legacy blocking 路径的 Y/Z 取反。混淆两者会让机械臂朝反方向运动。
- **不要修改停止时序逻辑**（`MotionFinished` 只能发在指令速度、加速度精确为零且已连续保持 100 个 1 kHz 周期之后）。这是修复过真实 FCI 报错的代码，见 `README.md` 的“停止时序”一节。
- 改动 `native/franka_server9_streaming_worker.cpp` 后必须重新构建并跑 `--self-test`，通过后才能上真机。
- 任何声称“已验证安全”的说法，必须对应实际运行过的命令和输出。

## 5. 项目地图

| 模块 | 文件路径 | 关键类或函数 | 作用 |
| --- | --- | --- | --- |
| Agent 维护约束 | `AGENT.md` | 本文件 | 工作流程、安全红线、项目地图、验证与文档规则 |
| 操作手册 | `README.md` | 文档主体 | 快速开始、各版本部署命令、诊断和安全原则 |
| 部署核心 | `franka_sim2real/e2e_bundle.py` | `BundleDeployConfig`、`load_bundle_config`、`BundleTorchScriptPolicy`、`evaluate_initial_state`、`run_bundle_deploy` | 配置模型、契约校验、相机、TorchScript 推理、blocking rollout |
| 动作映射 | `franka_sim2real/e2e_bundle.py` | `clip_policy_action`、`raw_action_to_robot_action`、`ActionHistoryBuffer` | raw action 裁剪到 `[-1,1]` 后按 `action_adapter.scales` 映射，维护动作历史 |
| 相机 | `franka_sim2real/e2e_bundle.py` | `RealSenseRGBCamera`、`GelSightPairCamera`、`PolicyCameraRig`、`StaticRGBCamera`、`_apply_camera_crop` | RealSense / GelSight 采集与 crop-resize 契约 |
| Streaming 调度 | `franka_sim2real/streaming.py` | `run_streaming_bundle_deploy`、`dls_joint_delta`、`pose_error`、`validate_streaming_contract`、`AsyncGripperQueue` | DLS IK、契约校验、异步夹爪、artifact 落盘 |
| Server9 编排 | `franka_sim2real/streaming_server9.py` | `run_server9_streaming_bundle_deploy`、`snapshot_to_observation`、`_audit_fci_log` | 原生 worker 生命周期、FCI 日志审计与诊断输出 |
| 共享内存 IPC | `franka_sim2real/server9_ipc.py` | `ABI_VERSION`、`SharedData`、`Server9SharedMemory`、`Server9Worker`、`validate_ctypes_layout` | Python 与 C++ worker 之间的 ABI-8 布局与进程管理 |
| 原生 1 kHz 控制 | `native/franka_server9_streaming_worker.cpp` | `kAbiVersion`、`enum class ControlLaw`、`main` | libfranka FCI 回调、速度跟踪器、限幅、停止时序、`--self-test` |
| 数据类型 | `franka_sim2real/types.py` | `RobotAction`、`RobotObservation`、`WorkspaceLimits`、`external_wrench_norms` | 动作/观测结构与外力报告 |
| 真机 backend | `franka_sim2real/envs/franka_real.py` | `RealFrankaEnv` | 基于 franky 的 blocking 控制路径 |
| Mock backend | `franka_sim2real/envs/mock_sim.py` | `MockFrankaSimEnv` | 不连硬件的运动学 mock |
| 手眼标定 | `franka_sim2real/calibration/hand_eye.py` | `solve_eye_to_hand` 等 | 固定 D435 的 eye-to-hand 求解 |
| 统一 CLI 入口 | `scripts/policy/run_exported_0711.py` | `build_parser`、`main(default_config=...)` | 所有 policy 部署脚本的唯一实现 |
| 各版本入口 | `scripts/policy/run_exported_*.py` | `DEFAULT_CONFIG` | 每个只有约 10 行，import `run_exported_0711.main` 并指定默认配置 |
| 视觉头验证 | `scripts/policy/monitor_rma_object_position.py`、`validate_rma_object_position.py` | `main` | 实时/离线检查 RMA 预测的物体位置 |
| 回零 | `scripts/robot/go_to_zero_pose.py` | `main` | 回到训练初始关节角与夹爪宽度 |
| 机器人基础操作 | `scripts/robot/read_franka_state.py`、`minimal_franka_move.py` | `main` | 状态读取与小幅手动位移 |
| 标定工具 | `scripts/calibration/collect_eye_to_hand.py`、`solve_eye_to_hand.py`、`live_apriltag_cube_pose.py` | `main` | 采集、求解、AprilTag 位姿验证 |
| 跟踪质量分析 | `scripts/diagnostics/analyze_streaming_tracking.py` | `main` | 从 `runs/` 的 control trace 计算 ratio / slope / 方向误差 |
| 力诊断 | `scripts/diagnostics/test_franka_force_sensor.py`、`test_franka_force_during_motion.py` | `main` | 静止与运动中的 wrench 检查 |
| 构建脚本 | `scripts/build_franka_server9_worker.sh`、`build_pylibfranka_wheel.sh` | — | 构建原生 worker 与 pylibfranka streaming 补丁 wheel |
| FCI 主机准备 | `scripts/fci/prepare_fci_host.sh`、`test_fci_network.sh` | — | 网卡、IRQ、CPU 亲和与实时性准备（需 root） |
| 主机检查 | `tools/system/verify_franka_host.sh`、`check_franka_network.sh`、`setup_franka_realtime_root.sh` | — | 实时内核、网络与主机配置 |
| 部署配置 | `configs/e2e_bundle_real_exported_*.json` | — | 每个版本一份，不覆盖旧配置 |
| 测试 | `tests/test_streaming.py`、`test_server9_streaming.py`、`test_e2e_bundle_runtime.py` 等 | 标准库 `unittest` | 离线契约、ABI 布局与数值测试 |
| 运行产物 | `runs/<timestamp>_<run_name>/` | `config.json`、`rollout.jsonl`、`summary.json`、`rgb/`、`control_trace.jsonl`、`timing_summary.json` | 每次运行的完整记录，被 Git 忽略 |

## 6. 代码修改规则

- 只修改完成当前任务所必需的文件；不得未经确认重构无关模块。
- 不得删除已有功能、历史配置、checkpoint 路径记录或文档中的历史信息。
- **新版本部署一律新增 `configs/` 文件和对应的 `run_exported_*.py` 入口，不覆盖仍在使用的配置。** 这是本仓库既有的演进方式（0711 → 0712 → 0726 → 0801 → 0802 → 0803 → 0808）。
- 新入口脚本保持现有极简形式：import `run_exported_0711` 的 `REPO_ROOT` 和 `main`，只声明 `DEFAULT_CONFIG`。所有通用逻辑改在 `run_exported_0711.py` 或 `franka_sim2real/` 中。
- 修改 `SharedData` 布局时必须同时更新 `franka_sim2real/server9_ipc.py` 的 `ABI_VERSION` 和 `native/franka_server9_streaming_worker.cpp` 的 `kAbiVersion`、同步 `offsetof` 断言，并重新构建 worker。两边版本号不一致时 worker 会直接拒绝启动。
- 不得随意更改动作维度、观测 key、归一化参数或坐标系约定；这些由 checkpoint metadata 的契约决定，改动会使已有 checkpoint 失效。
- 遇到已有未提交改动时，默认认为是用户的工作；不要回滚，不要使用 destructive git 命令，不要主动 commit。
- 复杂张量操作注明输入/输出维度；坐标系转换注明约定；限幅与增益注明物理含义和出处。
- 不要留下临时调试代码、硬编码的本机绝对路径（配置文件中已有的 `/home/td/franka/...` 除外）、设备号或步数。
- 注释和文档使用中文，代码标识符使用英文；沿用所在文件的既有风格。

## 7. 验证规则

只使用从实际代码确认过的命令。**除非用户明确要求，不要声称未运行的验证已经通过。**

| 验证目标 | 命令 | 是否连硬件 |
| --- | --- | --- |
| 全部单元测试 | `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` | 否 |
| Streaming 相关测试 | `.venv/bin/python -m unittest tests.test_server9_streaming tests.test_streaming tests.test_e2e_bundle_runtime` | 否 |
| Python 语法检查 | `.venv/bin/python -m compileall franka_sim2real scripts tests` | 否 |
| 配置与模型契约校验 | `python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --validate-only` | 否 |
| 原生 worker 构建与自测 | `./scripts/build_franka_server9_worker.sh` | 否（构建后自动跑 `--version` 和 `--self-test`） |
| worker 单独自测 | `dist/franka_server9/franka_server9_streaming_worker --self-test` | 否 |
| 共享内存布局 | `dist/franka_server9/franka_server9_streaming_worker --layout` | 否 |
| 主机与网络 | `bash tools/system/verify_franka_host.sh`、`bash tools/system/check_franka_network.sh 172.16.0.2` | 只读网络 |
| 机器人状态 | `python scripts/robot/read_franka_state.py --ip 172.16.0.2` | 是（只读） |
| 无运动预览 | `python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 150 --preview-only` | 是（只读状态和相机） |
| Streaming 通信检查 | `python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --streaming-check` | 是（不运动） |
| 单步真机 | `python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --steps 1` | **是，会运动，需人工确认** |
| 跟踪质量 | `python scripts/diagnostics/analyze_streaming_tracking.py --latest 1` | 否（读 `runs/`） |

## 8. 已知问题与待确认

| 类型 | 内容 | 位置 |
| --- | --- | --- |
| 已知失败 | `tests/test_streaming.py::ContractAndSchedulingTests::test_rma_streaming_contract_allows_longer_operator_selected_run` 报 `AttributeError: 'types.SimpleNamespace' object has no attribute 'metadata'`。是测试桩缺少 `metadata` 属性，不是生产代码缺陷；修复应在测试侧补齐桩对象。 | `tests/test_streaming.py:259`、`franka_sim2real/streaming.py:292` |
| 文档过期 | `README.md` 多处写 “abi-7”，实际 `ABI_VERSION` 与 `kAbiVersion` 均为 **8**。以代码为准。 | `README.md`、`franka_sim2real/server9_ipc.py:17`、`native/...cpp:33` |
| 工程债 | 0801 之后的大量代码、配置、文档和全部 `docs/` 新增文件仍未提交；`TacEx/` 整个目录 untracked。 | `git status --short` |
| 依赖管理 | 根仓库没有 `requirements.txt` / `pyproject.toml`，依赖只存在于本机 `.venv`，无法复现环境。 | 仓库根目录 |
| 待确认 | 是否计划把 `TacEx/` 以 submodule 或独立仓库形式正式纳入版本管理。 | — |
| 待确认 | `checkpoint/0808/rma_gelsight_student_latest.pt` 对应的 GelSight 策略是否已完成真机验收。 | `configs/e2e_bundle_real_exported_0808_gelsight.json` |

## 9. 文档维护规则

- `README.md` 维护面向操作者的命令与验收顺序。
- `docs/ARCHITECTURE.md` 维护模块职责、目录结构、控制链路、数据流、接口契约和产物布局。**改动模块边界、数据流、ABI 或契约后必须同步更新。**
- `docs/CHANGE_HISTORY.md` 按时间倒序记录代码、配置和文档变更。**每次实质性改动后追加条目**，格式见该文件顶部说明。不得把未执行的实验或未验证的结论写成事实。
- 专题文档各司其职，不要重复：`docs/LIBFRANKA_CONTROL_METHOD.md`（当前控制链路权威说明）、`docs/PYLIBFRANKA_STREAMING.md`（streaming 架构与主机准备）、`docs/SMOOTH_STREAMING_ARCHITECTURE.md`、`docs/HAND_EYE_CALIBRATION.md`、`docs/E2E_BUNDLE_DEPLOY.md`、`docs/SIM2REAL_FRAMEWORK.md`、`docs/GITHUB_UPLOAD.md`。
- 不确定的信息写“待确认”，不要用文件名或目录名推断结论。

## 10. 每次任务结束时的输出格式

### 修改文件

列出实际创建或修改的文件路径。

### 已确认信息

列出从代码、配置、日志或实测确认的关键事实。

### 待确认

列出仍无法从仓库确认的信息。

### 验证情况

列出实际运行过的命令和结果；未运行的验证必须明确说明未运行，尤其是任何真机相关操作。

### 安全影响

说明改动是否触及动作尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI。

### 建议提交信息

给出一条 Conventional Commits 风格的建议 commit message（仓库既有历史使用 `feat:`、`docs:`）。
