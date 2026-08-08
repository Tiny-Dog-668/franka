# AGENT.md

本文件是 `/home/td/franka` 的长期维护约束。修改代码、配置或文档前必须先读，并优先于其它判断。

配套：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)（架构与数据流）、[`docs/CHANGE_HISTORY.md`](docs/CHANGE_HISTORY.md)（修改历史）、[`README.md`](README.md)（操作命令）。

> 本仓库直接驱动一台真实的 Franka Panda。错误的动作尺度、坐标方向或初始状态判断会造成硬件损坏或人身伤害。**宁可拒绝执行，也不要猜测。**

## 1. 项目定位

把 Isaac Lab 训练的 RMA 策略导出为 TorchScript，在真机上以 30 Hz 推理 / 60 Hz IK / 1 kHz FCI 的分层频率执行。

仓库是双轨的：**根目录**是真机部署，由本文件管辖；**`TacEx/`** 是 Isaac Lab 训练，属于独立 Git 仓库，遵守它自己的 `TacEx/AGENTS.md`。两边规则不要互相套用，也不要在一次改动中同时跨越两个仓库边界。

## 2. 不要猜的事实

以下均从代码或配置确认。需要用到时按“依据”列复查，不要凭记忆。

| 项 | 值 | 依据 |
| --- | --- | --- |
| 机器人 | Franka Panda 7-DOF，FCI robot server version **9** | `docs/PYLIBFRANKA_STREAMING.md` |
| 机器人 IP / 网卡 / 控制 CPU | `172.16.0.2` / `enp3s0` / 21 | config `robot_ip`、`streaming.server9_control_cpu` |
| 相机 | RealSense D435 `215322076207`，640x480@30 | config `camera` |
| 模型输入 crop | `left=100, top=34, width=400, height=398` → 224x224（0801 之前不同） | config `camera` |
| 触觉 | GelSight Mini 双相机 device 0/6，仅 0808 配置启用 | config `tactile_camera` |
| 当前主力策略 | RMA Student TorchScript，`kind: tacex_rma_student_torchscript`，`version: 6` | `checkpoint/0802_DR_gpu/*.json` |
| 策略输入 | `wrist_rgb[224,224,3] uint8 NHWC`、`proprio_obs[15]`、`action_history[4]`，顺序固定 | 同上 `input_order` |
| 策略输出 | `mean_actions[4]` = `[dx, dy, dz, gripper]`，界 `[-1, 1]` | 同上 `output_signature` |
| 动作尺度 | `[0.05, 0.05, 0.05, 0.01]` 米 | config `action_adapter.scales` |
| 共享内存 ABI | **8**，Python 与 C++ 两侧必须一致 | `server9_ipc.py:ABI_VERSION`、`native/*.cpp:kAbiVersion` |
| 控制律 | `sim_actuator_velocity`（当前默认）、`joint_position_pursuit`（已不推荐） | `native/*.cpp:enum class ControlLaw` |
| 工作空间 | XYZ `[0.2, -0.3, 0.02] ~ [0.65, 0.3, 0.45]`（0808 下限 z=0.01） | config `workspace` |
| Python 环境 | 本机 `.venv`（3.10），无依赖清单 | 仓库根目录 |

## 3. 工作流程

1. 读本文件、`README.md` 相关章节和 `docs/ARCHITECTURE.md` 对应模块。
2. 跑 `git status --short` 和 `git log --oneline -n 10`，识别用户已有的未提交改动。
3. 阅读实际实现，不要根据文件名推断功能。动作维度、观测 key、坐标系、频率、限幅一律以代码或 checkpoint metadata 为准。
4. 动手前说明当前理解、计划修改的文件，以及可能影响的模块、张量维度、共享内存 ABI 和已有 checkpoint。
5. 无法从仓库确认的写“待确认”，不要编造。用户限制范围时只改允许的文件。
6. 改完同步更新 `docs/ARCHITECTURE.md` 和 `docs/CHANGE_HISTORY.md`，并报告实际跑过的验证命令。

## 4. 真机安全红线

优先于任何其它目标，包括“快一点”。

- **不要自作主张让机器人运动。** 只有用户明确要求时才运行会发送控制指令的脚本。
- **验收顺序不可跳过**：`--validate-only`（不连硬件）→ `--preview-only`（只读）→ `--streaming-check`（不运动）→ `--steps 1`（需人工输入 `y`/`yes`）→ 多步。不要加 `--yes` 跳过第一次单步确认，也不要建议用户这么做。
- **不要放宽安全参数**：`commissioning_action_limit`、`workspace`、`maximum_joint_velocities/accelerations/jerks`、`collision_behavior`、`initial_state.enforce` 及各类 tolerance。用户明确要求时才改，并在回复中标注影响。
- **不要关闭初始状态门禁。** 门禁失败的正确做法是用 `scripts/robot/go_to_zero_pose.py` 回到训练初始姿态，不是放宽容差。
- **不要改动坐标系约定**：streaming 路径的 XYZ 是机器人基座系且**不做** legacy blocking 路径的 Y/Z 取反。混淆两者会让机械臂朝反方向运动。
- **不要修改停止时序**：`MotionFinished` 只能发在指令速度和加速度精确为零、且已连续保持 100 个 1 kHz 周期之后。这是修复过真实 FCI 报错的代码。
- 改动 `native/franka_server9_streaming_worker.cpp` 后必须重新构建并通过 `--self-test` 才能上真机。
- 任何“已验证安全”的说法都必须对应实际运行过的命令和输出。

## 5. 从哪里下手

完整模块地图见 `docs/ARCHITECTURE.md`，这里只给入口：

| 要改什么 | 从哪读起 |
| --- | --- |
| 部署行为、相机、策略加载、契约校验 | `franka_sim2real/e2e_bundle.py` + 对应 `configs/*.json` |
| 任何部署 CLI 的行为 | `scripts/policy/run_exported_0711.py`（唯一实现，其余入口只有约 10 行） |
| Streaming 控制 | `streaming.py`（调度与 DLS IK）→ `streaming_server9.py`（worker 编排）→ `server9_ipc.py`（ABI）→ `native/*.cpp`（1 kHz 回调） |
| 模型契约与归一化 | `checkpoint/<版本>/*.json` |
| 标定 | `franka_sim2real/calibration/hand_eye.py` + `scripts/calibration/` |

## 6. 代码修改规则

- 只改完成任务必需的文件；不得未经确认重构无关模块，不得删除历史配置、checkpoint 记录或文档中的历史信息。
- **新版本部署一律新增 `configs/` 文件和对应入口，不覆盖仍在使用的配置**（0711 → 0712 → 0726 → 0801 → 0802 → 0803 → 0808 都是这么演进的）。新入口保持极简：import `run_exported_0711` 的 `REPO_ROOT` 和 `main`，只声明 `DEFAULT_CONFIG`；通用逻辑改在 `run_exported_0711.py` 或 `franka_sim2real/`。
- 改 `SharedData` 布局必须同时更新两侧版本号和 `offsetof` 断言并重新构建 worker，否则 worker 拒绝启动。
- 不得随意更改动作维度、观测 key、归一化参数或坐标系约定；这些由 checkpoint metadata 契约决定，改动会使已有 checkpoint 失效。
- 遇到已有未提交改动时默认认为是用户的工作：不要回滚，不要用 destructive git 命令，不要主动 commit。
- 不要留下临时调试代码、硬编码路径（配置中已有的绝对路径除外）、设备号或步数。
- 注释和文档用中文，标识符用英文，沿用所在文件的既有风格。

## 7. 验证命令

只用从代码确认过的命令。**不要声称未运行的验证已经通过。**

| 目标 | 命令 | 硬件 |
| --- | --- | --- |
| 单元测试 | `.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` | 否 |
| 语法检查 | `.venv/bin/python -m compileall franka_sim2real scripts tests` | 否 |
| 配置与模型契约 | `python scripts/policy/run_exported_0802_dr_simactuator_gpu.py --validate-only` | 否 |
| 构建 worker 并自测 | `./scripts/build_franka_server9_worker.sh`（自动跑 `--version` 和 `--self-test`） | 否 |
| 跟踪质量 | `python scripts/diagnostics/analyze_streaming_tracking.py --latest 1` | 否 |
| 主机与网络 | `bash tools/system/verify_franka_host.sh`、`check_franka_network.sh 172.16.0.2` | 只读 |
| 机器人状态 | `python scripts/robot/read_franka_state.py --ip 172.16.0.2` | 只读 |
| 无运动预览 / 通信检查 | 上面 GPU 入口加 `--steps 150 --preview-only` 或 `--streaming-check` | 只读，不运动 |
| 单步真机 | 上面 GPU 入口加 `--steps 1` | **会运动，需人工确认** |

## 8. 已知问题

完整清单见 `docs/CHANGE_HISTORY.md` 的「长期未决事项」。会立刻遇到的两条：

- `tests/test_streaming.py::test_rma_streaming_contract_allows_longer_operator_selected_run` 报 `AttributeError: ... no attribute 'metadata'`。**这是既有失败**，测试桩缺属性而非生产代码缺陷，不要误判成自己改坏了。
- `README.md` 多处写 “abi-7”，实际两侧均为 **8**。以代码为准。

## 9. 文档维护

- `docs/ARCHITECTURE.md`：模块职责、数据流、契约、ABI、产物布局。改动模块边界、数据流、ABI 或契约后必须同步更新。
- `docs/CHANGE_HISTORY.md`：每次实质性改动后按该文件顶部的格式追加条目。不得把未执行的实验或未验证的推测写成事实。
- `README.md`：面向操作者的命令与验收顺序。专题文档（`LIBFRANKA_CONTROL_METHOD.md`、`PYLIBFRANKA_STREAMING.md`、`HAND_EYE_CALIBRATION.md` 等）各司其职，不要重复。

## 10. 任务结束时必须输出

- **修改文件**：实际创建或修改的路径。
- **已确认 / 待确认**：从代码或实测确认的关键事实；仍无法确认的明确列出。
- **验证情况**：实际跑过的命令和结果；未运行的明确说明未运行，真机操作尤其如此。
- **安全影响**：是否触及动作尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI；无则写“无”。
- **建议提交信息**：一条 Conventional Commits 风格的 message（仓库使用 `feat:`、`docs:`）。
