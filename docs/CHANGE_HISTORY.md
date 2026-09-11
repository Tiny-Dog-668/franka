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

## 2026-09-11 0911 Progress policy 隔离与 RLPD 接入

**变更**：为 0911 checkpoint 新增独立 Progress metadata kind、部署配置/入口和 RLPD 配置；新增
`gelsight_reference_progress_single_frame_v1` adapter，并按 model/metadata/reward contract 隔离
`0911_progress` 的派生 Replay、rollout 与 checkpoint。旧 0814/0823 kind、配置和 adapter 保持不变。

**动机**：0911 来自 terminal-success Progress 环境，但原导出 metadata 与普通 Size-Buckets 共用 kind；
同时真实历史帧审计显示动作饱和与辅助接触预测塌缩，不能把它当作已批准的裸 base policy。

**影响**：0911 直接真机运动在 worker 启动前拒绝，只允许离线校验/preview、RLPD 专家全接管或已校验
RLPD checkpoint。动作仍为 4D、尺度仍为 `[0.05,0.05,0.05,0.01]`，commissioning limit、workspace
下界、初始状态门禁、坐标系、停止时序和 ABI-8 均未放宽；v7 工具最低点附加偏移新增为 `0.0529 m`。

**验证**：RLPD、bundle 与部署 CLI 相关单测 71 项通过；0911 GPU `--validate-only` 与 RLPD GPU
`validate` 通过，1043D feature 能复现 base action；0814/0823 原入口 GPU `--validate-only` 均通过。
0814/0823 原 RLPD GPU `validate` 也均通过且保持原 adapter ID。compileall 与 `git diff --check` 通过。
未连接相机或 Franka，未执行真机运动。

**文件**：`checkpoint/0911/gelsight_reference_progress_student_100000.json`、
`configs/e2e_bundle_real_exported_0911_gelsight_progress.json`、`configs/real_rlpd_0911_progress.json`、
`scripts/policy/run_exported_0911_gelsight_progress.py`、`franka_sim2real/e2e_bundle.py`、
`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、`real_rlpd/adapters.py`、
`tests/test_e2e_bundle_runtime.py`、`tests/test_real_rlpd.py`、`README.md`、`real_rlpd/README.md`、
`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-09-09 0823 RLPD 策略支持与专家/rollout 数据分层

**变更**：确认 0823 三帧 GelSight 策略的 workspace 工具点附加偏移为 `0.0579 m`，并将该配置的
workspace z 下界从 `0.00 m` 收紧到 `0.01 m`。RLPD 默认入口切换到 0823，配置 schema 升至 v2；专家
episode 写入共享的 `real_rlpd_data/expert/`，rollout 写入 policy-specific
`real_rlpd_data/rollout/0823/`。新增策略无关专家源格式，保存同步腕部 RGB、左右 GelSight 当前帧和固定
参考帧、机器人观测、proprio/action history 及专家绝对物理动作；base feature/action/residual 只进入
`real_rlpd_data/derived/0823/offline_expert.sqlite3` 派生缓存。专家与 rollout 目录相同或互相嵌套时配置
校验会拒绝启动。专家进度日志也改为显示绝对 requested/limited action，不再读取不存在的 residual 字段。

**动机**：同一个真机任务的遥操示范不应永久绑定采集时使用的 base policy；同时专家数据和策略 rollout
需要物理隔离，避免后续采样时混淆。0823 夹爪最低点距 `panda_hand` 为 `0.1613 m`，而 `O_T_EE` 已包含
`0.1034 m`，所以附加 workspace 偏移是两者之差 `0.0579 m`。旧 `0.027408 m` 会把工具最低点向上少算
`0.030492 m`。

**影响**：0823 的动作维度、动作尺度、commissioning limit、基座系方向、IK 命令点、初始状态门禁、
碰撞阈值、watchdog、停止时序和 ABI-8 不变；workspace 下界只收紧、不放宽。0814/0815 配置使用
`0.027408 m` 的历史 episode 不会被删除，且仍可用于离线分析，但其旧观测并不包含本次新格式要求的
完整同步触觉源，不能无损转换为新的策略无关专家源。当前会在采集时同步生成所选 policy 的派生 Replay；
从历史专家源批量重建其他 policy Replay 的独立 CLI 仍待实现。

**验证**：0823 实际 TorchScript 已在 CPU 和 `cuda:0` 完成离线 artifact、三帧视觉/双 GelSight 输入、1043D feature、
4D action 和 RLPD contract 校验；RLPD 12 项及 RLPD/bundle/streaming/server9 相关 88 项测试通过，
compileall 与 `git diff --check` 通过。全仓 236 项中 235 项通过；唯一失败是既有
`test_live_apriltag_cube_pose` 仍指向已不存在的 `runs/.../calibration_report.json`，实际配置使用
`runs_old/...`，与本次改动无关。历史日志审计中，48 份 0823 run config 全部记录 `0.0579 m` 偏移和
`0.01 m` workspace z 下界；36 份旧 0814 RLPD run config 记录 `0.027408 m`。未连接相机或 Franka，
未执行真机运动。

**文件**：`configs/e2e_bundle_real_exported_0823_gelsight.json`、`configs/real_rlpd_0814.json`、
`configs/real_rlpd_0823.json`、`real_rlpd/config.py`、`real_rlpd/expert_dataset.py`、
`real_rlpd/collector.py`、`real_rlpd/teleop.py`、`scripts/real_rlpd/run_rlpd.py`、
`franka_sim2real/streaming_server9.py`、`tests/test_real_rlpd.py`、`.gitignore`、`README.md`、
`real_rlpd/README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-09-09 0814 RLPD 成功抬升阈值调整为 5 cm

**变更**：将 0814 RLPD 的 `reward.success_height_m` 从 `0.015 m` 调整为 `0.05 m`；连续 3 帧确认、
成功奖励大小及 reach/lift/action-penalty 公式保持不变，并增加配置契约断言。collector 现在先校验
Replay contract，再执行 15 次有效 AprilTag 检测的 preflight，不兼容时立即拒绝。

**动机**：让抓取成功代表物体已经稳定抬升到 5 cm，而不是仅离开初始高度 1.5 cm。

**影响**：只改变 0814 离线 AprilTag 成功边界、reward contract 和 preflight/契约检查顺序；不改变 0823、
动作尺度、速度、限幅、workspace、initial-state gate、碰撞阈值、停止时序、FCI worker 或共享内存 ABI。
旧 `0.015 m` Replay 和 checkpoint 与新配置不兼容，必须隔离或重新标注/迁移，不能静默混用。

**验证**：RLPD 单元测试、配置加载和 `git diff --check` 通过。轨迹 `20260909_182558` 重新标注后，成功
边界由 431 推迟到 484，最大抬升 `0.092151 m`，在新阈值下仍判定成功；trainable transition 从 425
变为 476。现有 offline Replay 已迁移到 5 cm contract，迁移前 SQLite 备份保留在同目录。未连接相机或
Franka，未执行真机运动。

**文件**：`configs/real_rlpd_0814.json`、`real_rlpd/collector.py`、`tests/test_real_rlpd.py`、`real_rlpd/README.md`、
`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-09-09 RLPD 夹爪抢占停止与 deadline 容错

**变更**：RLPD 连续夹爪执行器不再使用 `franky.Gripper.stop_async()` 中断 endpoint move，改为在隔离的
Hand owner 进程中直接调用同步 `stop()`；反向动作在 stop 返回后立即启动，只有静止 hold 才读取实测宽度
并重置逻辑累计起点。单帧被丢弃的 policy 结果不再直接生成夹爪 hold；机械臂仍立即 hold，夹爪保持最后
一个已接受的速度意图，持续超过既有 `policy_watchdog_s` 后才停止。timing artifact 将停止调用字段收敛为
`maximum_stop_call_ms`，并新增 `gripper_policy_watchdog_holds`。

**动机**：真机运行 `20260909_164427` 测得指令发布到 Hand owner 仅 `1.58–3.53 ms`、`move_async()`
仅 `0.05–0.07 ms`，但 `stop_async()` 调用本身阻塞 `1319.59 ms`。`franky 1.1.x` 的
`setCurrentFuture()` 会先等待已有异步动作，因此 `stop_async()` 不能抢占正在运行的 `move_async()`；一次
45 ms 的孤立 policy miss 又被旧逻辑误判为松键，最终造成后续反向命令约 802 ms 的可见延迟。

**影响**：RLPD Replay 中的 30 Hz、每步 1 mm gripper delta、动作限幅、速度上限、力、workspace、机械臂
deadline hold、history 接受规则、FCI worker 和共享内存 ABI 均不变。连续丢帧仍受现有 0.25 s policy
watchdog 保护。普通非 RLPD/非 servo 夹爪目标队列不变。

**验证**：`git diff --check` 与 Python compileall 通过；gripper、server9、RLPD、streaming 共 55 项测试
通过。mock 明确令 `stop_async()` 抛错，确认抢占路径只调用 `stop()`；新增 45 ms 单帧 miss 不触发 stop、
250 ms watchdog 到期只触发一次 hold 的测试。未执行新的真机运动，实际 stop RPC 延迟需下一次采集确认。

**文件**：`franka_sim2real/gripper_process.py`、`franka_sim2real/streaming_server9.py`、
`tests/test_gripper_process.py`、`tests/test_server9_streaming.py`、`real_rlpd/README.md`、
`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-09-09 RLPD 专家遥操键位调整

**变更**：将 RLPD 专家遥操的 Z 轴按键从 `R/F` 改为 `J/K`，夹爪连续打开/关闭从 `O/C` 改为
`U/I`；同步更新 Pygame 监听、窗口提示、测试和操作文档。旧 HIL Residual BC 的按键不变。

**影响**：只改变显式 RLPD expert 模式的人工输入映射；动作尺度、坐标系、安全限幅、workspace、
initial-state gate、DLS/FCI、停止时序和 ABI 均不变。

**验证**：RLPD 单元测试与 `git diff --check` 通过；未连接相机或 Franka，未执行真机运动。

**文件**：`real_rlpd/teleop.py`、`tests/test_real_rlpd.py`、`real_rlpd/README.md`、`README.md`、
`docs/CHANGE_HISTORY.md`

## 2026-09-09 多策略真机 RLPD 4D residual

**变更**：新增独立 `real_rlpd/` PyTorch 实现和统一 CLI，按 RLPD 采用 10-Q ensemble、随机 min-2、
Critic LayerNorm、offline/online 50:50 和 UTD=20。Actor 使用 frozen policy feature 1043D＋已限幅
base action 4D，并输出 XYZ＋gripper 4D residual；Critic 额外读取 AprilTag object-relative XYZ＋height。
新增 0814 单帧 GelSight 与 0823 三帧 GelSight adapter、各自独立的 expert/online Replay 和 checkpoint
配置。专家采集使用 30 Hz Pygame 全接管遥操，base policy 只做 shadow；online 采集只允许有效 checkpoint，
随机 residual 需要额外显式开关。训练严格在 episode 之间运行，首次默认 1000 update group，后续按新增
trainable online transition 更新。控制期间 transition 只保存在内存，worker 停止后才以单事务写 SQLite，
AprilTag reward 继续使用原始边界帧离线标注。旧 Real-RL Replay 不迁移。新增目录内 README，集中记录
模块边界、算法/动作契约、专家按键、完整验收命令、Replay 时序和扩展新 policy/reward 的要求。

**动机**：把用户引入的 `rlpd/` 参考实现思路适配到现有 server9 真机链路，同时支持不同 base policy、
高质量人工 expert 数据和不改变最终真机安全包络的 residual 学习。

**影响**：RLPD 是显式、互斥的新路径；不开启时旧部署、HIL、Residual BC 和旧 Real-RL 保持原行为。
residual 在 base commissioning limiter 之后叠加，再经过原有最终裁剪；RLPD 禁止 full-scale。动作物理尺度、
基座系 XYZ、workspace、initial-state gate、碰撞阈值、DLS/FCI、停止时序、native worker 与共享内存 ABI
均未修改。运行数据新增到 Git 忽略的 `real_rlpd_runs/`。

**验证**：0814 与 0823 实际 TorchScript 均在 CPU 完成离线 artifact/feature contract 及完整 RLPD
post-limit tick 校验；临时 Replay 的 1-group 训练、原子 checkpoint 和重新加载 smoke test 通过。新增 RLPD
8 项单元测试通过。RLPD/streaming/server9/bundle 相关 83 项中 82 项通过，唯一失败是既有 0823 配置
`workspace.minimum.z=0.00` 与既有测试期望 `0.01` 不一致。全量 230 项中 228 项通过；另一个既有错误是
live AprilTag 测试仍指向已迁移的 `runs/` 标定路径。compileall 与 `git diff --check` 通过。未连接相机或
Franka，未运行 preview、streaming-check 或任何真机运动。

**待确认**：0823 base config 的 `workspace.minimum.z` 实际为 `0.00 m`，但其既有测试和变更记录要求
`0.01 m`。本次未擅自修改安全参数；RLPD CLI 在该矛盾解决前拒绝 0823 真机运动，离线 validate 和
preview 不受影响。

**文件**：`real_rlpd/`、`scripts/real_rlpd/run_rlpd.py`、`configs/real_rlpd_0814.json`、
`configs/real_rlpd_0823.json`、`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、
`franka_sim2real/streaming_server9.py`、`tests/test_real_rlpd.py`、`.gitignore`、`README.md`、
`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-23 修正几何的 0823 三帧 GelSight 真机部署

**变更**：新增 0823 三帧 RGB＋双 GelSight TorchScript 的独立配置和入口；沿用 4D 动作、30 Hz、
首帧重复历史和固定触觉参考帧，并将物理 workspace 点设为相对 `O_T_EE` 的 `+57.9 mm`。
**影响**：不修改模型权重、观测或动作尺度；旧三帧配置的 `27.408 mm` 安全点会被契约校验拒绝。
**验证**：CPU artifact SHA/contract/dummy inference 通过，相关 54 项回归测试通过；当前自动化环境无法
初始化 CUDA，故 GPU 校验未运行。未连接相机或 Franka。文件：`configs/e2e_bundle_real_exported_0823_gelsight.json`、
`scripts/policy/run_exported_0823_gelsight.py`、`franka_sim2real/e2e_bundle.py`。

## 2026-08-23 Real-RL CUDA checkpoint RNG 恢复兼容

**变更**：恢复 Residual SAC checkpoint 时，将 `cuda_rng_state_all` 中的状态 buffer 显式搬回 CPU 后再调用
`torch.cuda.set_rng_state_all()`。

**动机**：训练续跑通过 `map_location=cuda:0` 加载 checkpoint 时，序列化的 RNG ByteTensor 也会被搬到
GPU，而 PyTorch RNG 恢复接口要求 CPU ByteTensor，导致续训在任何 gradient update 前报
`TypeError: RNG state must be a torch.ByteTensor`。

**影响**：只修复 CUDA checkpoint 的随机数状态恢复；不改变权重、优化器、Normalizer、Replay
high-watermark、UTD 计划或动作契约。失败的续训没有覆盖已有 checkpoint。

**验证**：新增映射后 RNG state 回到 CPU 的单元测试；Real-RL 20 项测试通过；现有 `latest.pt` 在
`cuda:0` 上恢复通过，保持 `update_count=1000`、`replay_high_watermark=4253`。未执行 gradient update。

**文件**：`franka_sim2real/real_rl/residual_sac.py`、`tests/test_real_rl.py`、
`docs/CHANGE_HISTORY.md`

## 2026-08-23 Real-RL Hand API 进程隔离

**变更**：server9 Real-RL 将 Franka Hand owner 从 Python 后台线程迁移到独立 `spawn` 进程。主进程通过
共享内存发布带 generation 的最新目标宽度，子进程独占构造和调用 `franky.Gripper`，并通过单向状态通道
返回宽度、抓取状态和错误。命令继续 latest-only 合并，不建立过期动作队列；blocked close 转 force grasp
和安全停止语义保持不变。

**动机**：`20260823_133347` 中，Hand 状态缓存更新的 9 个 step 全部对应主循环约 `100–132 ms` 停顿，
相关连续组造成 23/27 次 deadline miss。虽然 Hand API 已不在 policy 线程调用，但这一一对应关系表明 native
binding 很可能在后台调用期间持有进程级 Python GIL；线程隔离无法消除这种阻塞，进程隔离可以。

**影响**：仅 server9 的 Hand owner 和进程生命周期改变；机械臂 C++ worker、FCI 共享内存 ABI、策略模型、
夹爪速度/力/宽度目标、动作尺度、workspace、安全门禁和 Replay schema 均不变。普通非 server9 streaming
仍保留原 `AsyncGripperQueue`。

**验证**：新增无硬件 `spawn` 测试，以 120 ms 忙等模拟 Hand 调用持有 GIL，确认父进程 `command/poll`
仍低于 20 ms；同时覆盖 blocked-close 转 force grasp 和失败错误回传。相关 gripper、streaming、server9
测试通过；未连接机器人执行真机运动。

**文件**：`franka_sim2real/gripper_process.py`、`franka_sim2real/streaming_server9.py`、
`tests/test_gripper_process.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-23 Real-RL 夹爪非阻塞化与动作边界采帧

**变更**：将 Franka Hand 的 `move_async/grasp_async`、future `get` 和 `gripper.state` 全部移入专属 owner
线程；30 Hz policy 线程的 `command/poll` 仅更新合并目标和读取缓存错误。保留 blocked-close 自动转为
force grasp 的逻辑。Real-RL 不再把提前推理时取得的 raw frame 当作 transition 边界，而是在每个真实
policy deadline 独立读取 `LatestFrameCamera` 最新 packet；rollout 同时保留 policy frame 和 reward boundary
frame 元数据，离线标注优先读取后者，旧日志仍可按旧字段重标。

**动机**：`20260823_132048` 中 Hand 完成查询反复阻塞 `77–99 ms`，产生 32 次 deadline miss；同时下一
policy inference 在动作后立即读取的 latest frame 实际平均早于动作 `28.5 ms`，导致 300 条 transition
只有 2 条可训练。

**影响**：不改变 Hand 力、目标宽度、受阻接触语义、机械臂控制、安全限幅、Replay schema 或模型输入。
新采集的 `model_input.offline_apriltag_boundary` 明确记录 reward 图像路径、形状、相机 sequence 和 monotonic
时间戳。

**验证**：streaming、server9、Real-RL、Residual runtime 和 HIL 相关 75 项单元测试通过；包含后台 Hand
API owner、受阻转 grasp、错误传播、动作边界 packet 选择及离线字段优先级测试。CPU Real-RL validate
通过。全量 208 项中 207 项通过，唯一失败仍是既有 live AprilTag 工具指向已迁移的标定路径。未连接
机器人或相机执行真机验证。

**文件**：`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、
`franka_sim2real/real_rl/offline_apriltag.py`、`tests/test_streaming.py`、`tests/test_server9_streaming.py`、
`tests/test_real_rl.py`、`README.md`、`docs/ARCHITECTURE.md`

## 2026-08-23 0814 GelSight 夹爪速度调整

**变更**：将 `e2e_bundle_real_exported_0814_gelsight.json` 的夹爪移动速度从 `0.03 m/s` 提高到
`0.20 m/s`；夹爪力保持 `20 N`，机械臂速度、动作限幅、workspace、DLS 和 FCI 参数不变。

**动机**：Real-RL 采集 `20260823_131030` 中，夹爪异步命令完成期间出现最长约 `99 ms` 的 command/poll
延迟，并在其后形成连续 policy deadline miss；提高物理开合速度可缩短长距离宽度调整的持续时间。

**影响**：使用该 0814 GelSight bundle 配置的部署和 Real-RL 采集均使用 `0.20 m/s`；其它 bundle 配置
不变。

**验证**：运行 Real-RL CPU `validate` 和相关配置/streaming 单元测试；未连接机器人执行夹爪运动。

**文件**：`configs/e2e_bundle_real_exported_0814_gelsight.json`、`docs/CHANGE_HISTORY.md`

## 2026-08-23 Franka Real-world Residual SAC

**变更**：为 0814 视觉＋双 GelSight Student 新增独立 Real-RL 模块和 `validate/collect/train` 入口。
Actor 使用 frozen policy feature 1043D＋base action 4D，仅输出每轴最大 ±2 mm 的 XYZ residual；Q1/Q2
从第一版读取 AprilTag object-relative XYZ＋height 4D privileged state。新增 SQLite WAL Replay、后台
latest-only AprilTag/Replay writer、progress/lift/success/action reward、双 Q SAC、自动 entropy、冻结
normalizer、UTD high-watermark、原子 checkpoint 和 deterministic Actor artifact。Warmup 支持 residual=0，
也支持 σ=0.5 mm、裁剪 ±1 mm、默认相关系数 0.9 的 AR(1) 随机 residual。Replay v2 新增统一
`trainable/trainable_reason`、严格 `tag_t <= action_time < tag_t1` 的 capture/action 时间契约、基座系
物体高度命名，以及 safety 前后 action、介入标志和介入幅度。Normalizer 改为只经验归一化 1043D
policy feature 与 4D privileged state，base action 固定 contract、不做 z-score；首次训练使用独立
`bootstrap_updates`，后续才使用 UTD。checkpoint 校验失败默认在 worker 启动前拒绝 episode，显式
`--allow-checkpoint-fallback-collect` 才可用零 residual 继续采集。随后将 reward 标注改为离线逐帧模式：
AprilTag 只在启动前预检，控制期间保存原始边界帧，停止 worker 后使用扩大 ROI＋整帧 fallback 检测并
回写 Replay；新增可恢复重跑的 `label` 命令。Real-RL run 和 Replay 移至独立 `real_rl_logs/` 根目录。

**影响**：只新增 server9 的显式 Real-RL 路径。Base Student、gripper、action contract、commissioning limit、
workspace、DLS、FCI、native worker、IPC ABI 和 deadline-miss/history 规则未修改。不开 Real-RL 时不会加载
AprilTag/OpenCV collector，现有 HIL 与 Residual BC 行为保持不变。

**验证**：Real-RL、Residual BC、streaming、server9、bundle 等 46 项相关测试通过；compileall、
0814 CPU `validate` 通过实际 TorchScript SHA、1043D feature/base action 复现和
Real-RL 零 residual smoke test，未连接相机或机器人；缺失 checkpoint 的 validate 已验证 fail closed。
全量 205 项测试中 204 项通过；唯一失败是既有
`live_apriltag_cube_pose.DEFAULT_CALIBRATION_REPORT` 仍指向已从 `runs/` 移到 `runs_old/` 的报告，与本改动无关。
未执行真机 preview 或运动。

**文件**：`franka_sim2real/real_rl/`、`franka_sim2real/policy_features.py`、
`scripts/real_rl/run_residual_sac.py`、`configs/real_residual_sac_0814.json`、
`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、
`franka_sim2real/streaming_server9.py`、`tests/test_real_rl.py`、`README.md`、`docs/ARCHITECTURE.md`

## 2026-08-22 GelSight 双相机运行时自动枚举

**变更**：新增公共 GelSight V4L2 枚举模块，并让相机预览与 policy 部署共用同一筛选逻辑。统一部署 CLI
新增 `--auto-gelsight`：相机启动前读取 sysfs，只选择名称包含 GelSight 且 UVC `index=0` 的主图像流，
要求恰好两路，然后按当前 video 编号绑定 left/right，并打印设备号、label 和 serial。实际解析后的设备号
覆盖本次运行时配置；未修改历史部署 JSON，不加开关时仍使用其中的显式设备号。

**动机**：Linux 的 `/dev/videoN` 会随 USB 连接和其它相机枚举变化。0814 配置中的旧 `4/6` 在当前机器
已变为 GelSight `10/12`，手动修改编号容易误选 RealSense、webcam 或同一 UVC 设备的辅助流。

**影响**：只改变显式请求 `--auto-gelsight` 的相机选择步骤；模型输入、左右张量顺序、图像尺寸、参考帧、
HIL、动作、控制、安全参数、native worker 和共享内存 ABI 均未修改。自动模式不能从图像推断物理左右，
换插口或交换安装位置后仍须先预览确认；发现数量不等于二时 fail closed，且不会尝试打开候选相机。

**验证**：当前 sysfs 只读枚举得到 `/dev/video10`（serial `2G7W6P6N`）和 `/dev/video12`
（serial `2G7KGVER`），选择结果为 left 10/right 12；全量 unittest 171 项通过，compileall 与
`git diff --check` 通过；0814 CUDA 的 `--auto-gelsight --hil --validate-only` 通过且未打开相机或连接
Franka。未运行 GelSight 实时预览或真机策略运动。

**文件**：`franka_sim2real/gelsight_devices.py`、`franka_sim2real/e2e_bundle.py`、
`scripts/camera/gelsight_start.py`、`scripts/policy/run_exported_0711.py`、
`tests/test_gelsight_devices.py`、`tests/test_run_exported_cli.py`、`README.md`、
`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`、`order.txt`

## 2026-08-22 server9 HIL Residual BC 数据采集

**变更**：统一 policy CLI 新增 `--hil` 与 `--hil-speed-m-s`。新增懒加载 Pygame 的焦点键盘线程，
在 server9 的 30 Hz 实际发送边界采样 Space 与 `W/S/A/D/R/F`，介入时用基座系人工 XYZ 替换
Actor XYZ，同时保留 Actor gripper。Actor 在介入期间仍每步推理；选择动作继续经过既有 contract、
commissioning limit、action scaling、workspace、DLS IK 和 FCI。HIL 强制复用 step NPZ，并在既有
JSONL/NPZ 中追加 base/human/executed/intervention/residual 与 accepted 标签。

**动机**：采集 observation/policy feature 与 base action 对应的人工 XYZ Residual BC 标签，同时保证
人工接管不绕过已经验收的真机控制与安全链，并保证下一步 action history 使用实际接受的人工组合动作。

**影响**：仅增加 server9 streaming 的可选运行时路径；不开 `--hil` 时不加载 Pygame，现有日志结构和
运行行为保持不变。未修改 PPO/Student、checkpoint、部署 JSON、动作维度/尺度、commissioning limit、
workspace、DLS、FCI、native worker、共享内存 ABI、碰撞阈值、坐标系或停止时序。blocking、async backend
与 `--hil --streaming-check` 会在硬件资源创建前拒绝。

**验证**：`.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` 共 165 项通过；
`.venv/bin/python -m compileall -q franka_sim2real scripts tests` 与 `git diff --check` 通过；
`.venv/bin/python scripts/policy/run_exported_0814_gelsight.py --device cuda:0 --steps 150 --hil --validate-only`
通过 TorchScript SHA、输入输出、CUDA dummy inference 和 streaming contract 校验，且未启动 Pygame、相机或
Franka。未执行 HIL preview、streaming-check 或任何真机运动测试。

**文件**：`franka_sim2real/hil.py`、`franka_sim2real/e2e_bundle.py`、
`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、
`scripts/policy/run_exported_0711.py`、`tests/test_hil.py`、`tests/test_run_exported_cli.py`、
`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-23 独立 HIL Residual BC 训练模块

**变更**：新增纯离线 Residual BC 数据扫描、episode-level split、冻结 0814 Actor 特征提取、三组平衡
采样、XYZ Residual MLP、SmoothL1 训练、early stopping、分组指标和 TorchScript/checkpoint 导出。训练前
重算并校验日志 base action，产物绑定 base-model SHA256。

**影响**：不修改 PPO/Student 权重、部署配置、streaming、安全或机器人控制。训练模块不连接硬件，导出
的 residual head 也不会自动进入真机路径。

**文件**：`franka_sim2real/residual_bc.py`、`scripts/training/train_residual_bc.py`、
`tests/test_residual_bc.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-23 Residual BC server9 部署接口

**变更**：统一 policy CLI 新增 residual model/metadata、scale、逐轴 cap 和显式真机 enable 参数。运行时绑定
训练记录的 base SHA，复现 0814 的 1043 维 actor feature，只叠加 XYZ 并保留 base gripper；JSONL/NPZ
记录 predicted/applied residual 和最终动作。

**安全**：默认 cap 为 0.1；真机禁止 `--yes` 并要求 `--enable-residual-control`，禁止 HIL、blocking、
async、streaming-check、RMA override 和 optimized base graph。最终动作继续经过原 safety/control pipeline。
当前模型 normal residual 偏大，文档首次验收使用 scale 0.25、cap 0.02。

**文件**：`franka_sim2real/residual_runtime.py`、`franka_sim2real/e2e_bundle.py`、
`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、
`scripts/policy/run_exported_0711.py`、`tests/test_residual_runtime.py`、
`tests/test_run_exported_cli.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 X040-Wide 0030000 位置评估 metadata

**变更**：为 `checkpoint/0809_30000/rma_x040_wide_student_0030000.json` 补齐训练导出时遗漏的
`normalization`。数值由同一 distillation run 的 X040-Wide 训练源码与该 TorchScript 的 normalizer buffers
交叉确认，包含 cube position 的 center `[0.4,0,0.026]`、scale `[0.08,0.1,0.1]` 及 `robot_root` 坐标系。

**影响**：使相机/AprilTag 位置评估器能够将 0030000 的归一化 position head 输出还原为米单位。没有添加
CPU/GPU 一致性记录、完整部署模型契约或新真机入口，故该 metadata 补全本身不构成真机部署验收；未触及动作
尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI。

**验证**：已校验 metadata 中 TorchScript SHA-256 为 `79aea3ec6e26c59d7595f298942d03982279189a84d3fdaa37c3e7ae28b23538`，并在 CPU 加载模型读取 normalizer buffers；实时相机测试由操作者后续执行，未连接 D435 或 Franka。

**文件**：`checkpoint/0809_30000/rma_x040_wide_student_0030000.json`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 X040-Wide 位置预测 AprilTag 离线评估

**变更**：`RMAPolicyCubePredictor` 支持读取 X040-Wide 导出模型的 `forward_with_position()` 视觉辅助 position head；新增对输出 `[N,4]` actions 与 `[N,3]` normalized XYZ 的形状校验，并按 metadata 的 cube center/scale 还原 `robot_root` 米单位坐标。离线 AprilTag 评估摘要写入 policy kind、坐标系和 contact 可用性；X040-Wide 的无 contact head 以 `NaN` / `false` 明确表示。README 补充静态多位置采集与离线 RMSE/中位/P95 评估命令。

**影响**：仅改变相机/文件只读诊断路径。没有变更部署模型输入、动作尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI；位置辅助输出仍不会进入真实机器人控制。

**验证**：已在 CPU 加载 X040-Wide TorchScript 并直接调用 `forward_with_position()`，确认输出 actions `[2,4]`、normalized position `[2,3]` 均为有限值；`tests.test_live_apriltag_cube_pose` 6 项通过，`compileall scripts/calibration tests` 通过，GPU 批量预测 `[3,3]` 全为有限值且 contact `[3,2]` 全为 `NaN`。使用已有 20260803 静态数据集的 252 帧（203 帧检出 Tag、19 个有效位置）在 RTX 5060 上完成离线评估：位置中位 3D 误差 `37.1 mm`、P95 `44.3 mm`、中位偏差 `[-28.8,+16.4,+14.5] mm`（policy - AprilTag）。该数据仅覆盖 X040 范围中的 `x=0.4035~0.4573 m`、`y=-0.0382~0.0211 m`，不能代表完整训练范围。未连接 D435 或 Franka。

**文件**：`scripts/calibration/live_apriltag_cube_pose.py`、`scripts/calibration/evaluate_policy_vs_apriltag.py`、`tests/test_live_apriltag_cube_pose.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 0809 X040-Wide Direct-Action 真机部署适配

**变更**：新增 X040-Wide 专用配置 `e2e_bundle_real_exported_0809_x040_wide_direct_action.json` 与入口 `run_exported_0809_x040_wide_direct_action.py`；部署运行时识别 `tacex_rma_x040_wide_direct_action_torchscript` v1，并对三输入/四输出、无 contact/GelSight 输入、TorchScript SHA、normalization、模型结构、训练初始关节状态、D435 crop、动作/历史尺度及训练方块范围的工作空间覆盖执行 fail-closed 校验。为 metadata 补齐由训练环境和导出模型确定的模型/环境契约，并写入 CUDA 验证记录。

**动机**：X040-Wide Student 不是旧 Direct-Action checkpoint：Teacher 只使用 cube XYZ、无接触输入，方块中心为 `x=0.40 m` 且 reset 范围为 `x±0.08 m`、`y±0.10 m`。原始 exporter JSON 未携带部署所需的环境/模型契约和 CUDA 验证，通用路径无法安全判断它能否进入真机 streaming。

**影响**：新配置保持现有 `[0.05,0.05,0.05,0.01]` 动作尺度、`commissioning_action_limit: 0.1`、工作空间、初始状态门禁、碰撞阈值、基座系 XYZ、30/60/1000 Hz、停止时序与共享内存 ABI；只额外确认既有工作空间覆盖训练方块范围，未放宽任何安全参数。

**验证**：训练环境源文件确认 30 Hz、动作尺度、初始关节、D435 crop 及 X040-Wide cube range；CPU `--validate-only` 通过；12 组离线 rollout 输入经部署运行时 CPU 与 RTX 5060 CUDA 推理，全部输出有限 `[4]`，最大绝对误差 `0.0007221698760986328 < 0.001`。未连接相机或机器人，未执行 preview、streaming-check 或运动测试。

**文件**：`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、`configs/e2e_bundle_real_exported_0809_x040_wide_direct_action.json`、`scripts/policy/run_exported_0809_x040_wide_direct_action.py`、`checkpoint/0809_extend_xy_10000/rma_x040_wide_student_latest.json`、`tests/test_e2e_bundle_runtime.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 0809 Direct-Action 0100000 的 CUDA 一致性验证

**变更**：为 `checkpoint/0809_direct_100000/rma_direct_action_student_student_0100000.json` 写入 CUDA 验证记录：`available: true`、最大 CPU/GPU 输出绝对误差 `0.00074923038482666016`、容差 `0.001`。

**动机**：Direct-Action GPU 部署契约要求每个 TorchScript checkpoint 独立记录成功的 CUDA 一致性验证；0100000 缺少该记录，无法安全进入 CUDA 部署。

**影响**：0100000 可以通过 CUDA 部署契约；不改变模型权重 SHA、输入输出签名、动作尺度、限幅、工作空间、初始状态门禁、坐标系、停止时序或共享内存 ABI。

**验证**：先以 CPU 运行 `--validate-only` 通过 SHA、输入输出与 Direct-Action 契约校验；再使用 `sim_rollout/0809_rollout/episode_0000.npz` 至 `episode_0003.npz` 的开头、中段、末段共 12 组 `wrist_rgb`、`proprio_obs`、`action_history` 输入，经部署运行时 `BundleTorchScriptPolicy.predict()` 分别在 CPU 与 RTX 5060 CUDA 上推理；输出均为有限 `[4]`，最大绝对误差小于 `0.001`。未连接相机或机器人。

**文件**：`checkpoint/0809_direct_100000/rma_direct_action_student_student_0100000.json`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 回零关节轨迹的连续执行与段间静止门禁

**变更**：`go_to_zero_pose.py` 默认从当前关节姿态到 policy 初始姿态只发送一条连续 `franky.JointMotion`；旧的多段方式改为显式 `--staged`，并在起动前和每段结束后确认实测关节速度不超过既有 `--velocity-tolerance-rad-s`，否则拒绝发送下一条轨迹。对 libfranka 的 joint motion 速度/加速度不连续 reflex 添加中文处理提示。

**动机**：真机在默认 7 段回零的第 4 段报告 `Motion finished commanded, but the robot is still moving`，同时触发 `joint_motion_generator_velocity_discontinuity` 和 `joint_motion_generator_acceleration_discontinuity`。每段都是一次独立 motion generator 会话，段间重新起停无法保证连续轨迹导数。

**影响**：默认回零路径仍是相同的当前关节位置到固定 policy 初始关节位置的关节空间直线，但由 Franka motion generator 在单一会话内连续规划。未放宽 speed、关节限位、工作空间、初始状态门禁、停止时序或共享内存 ABI；外力、碰撞或 FCI 反射仍会中止，不自动重试。

**验证**：新增离线 mock 测试确认默认只构造/发送一条 `JointMotion`，并覆盖不连续 reflex 的中文提示；未连接机器人或夹爪，未执行真机回零。

**文件**：`scripts/robot/go_to_zero_pose.py`、`tests/test_go_to_zero_pose.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 0809 Direct-Action 0060000 的 CUDA 一致性验证

**变更**：为 `checkpoint/0809_direct_60000/rma_direct_action_student_student_0060000.json` 写入 CUDA 验证记录：`available: true`、最大 CPU/GPU 输出绝对误差 `0.00044992566108703613`、容差 `0.001`。

**动机**：Direct-Action GPU 部署契约要求每个 TorchScript checkpoint 独立记录成功的 CUDA 一致性验证；0060000 缺少该记录，无法安全进入 CUDA 部署。

**影响**：0060000 可以通过 CUDA 部署契约；不改变模型权重 SHA、输入输出签名、动作尺度、限幅、工作空间、初始状态门禁、坐标系、停止时序或共享内存 ABI。

**验证**：先以 CPU 运行 `--validate-only` 通过 SHA、输入输出与 Direct-Action 契约校验；再使用 `sim_rollout/0809_rollout/episode_0000.npz` 至 `episode_0003.npz` 的开头、中段、末段共 12 组 `wrist_rgb`、`proprio_obs`、`action_history` 输入，经部署运行时 `BundleTorchScriptPolicy.predict()` 分别在 CPU 与 RTX 5060 CUDA 上推理；输出均为有限 `[4]`，最大绝对误差小于 `0.001`。未连接相机或机器人。

**文件**：`checkpoint/0809_direct_60000/rma_direct_action_student_student_0060000.json`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 0809 Direct-Action 0040000 的 CUDA 一致性验证

**变更**：为 `checkpoint/0809_direct_40000/rma_direct_action_student_student_0040000.json` 写入 CUDA 验证记录：`available: true`、最大 CPU/GPU 输出绝对误差 `0.0007036328315734863`、容差 `0.001`。

**动机**：Direct-Action GPU 部署契约要求每个 TorchScript checkpoint 独立记录成功的 CUDA 一致性验证；0040000 缺少该记录，因此即使 CUDA 可用也会被安全校验拒绝。

**影响**：0040000 可以通过 CUDA 部署契约；不改变模型权重 SHA、输入输出签名、动作尺度、限幅、工作空间、初始状态门禁、坐标系、停止时序或共享内存 ABI。

**验证**：使用 `sim_rollout/0809_rollout/episode_0000.npz` 至 `episode_0003.npz` 的开头、中段、末段共 12 组 `wrist_rgb`、`proprio_obs`、`action_history` 输入，经部署运行时 `BundleTorchScriptPolicy.predict()` 分别在 CPU 与 RTX 5060 CUDA 上推理；输出均为有限 `[4]`，最大绝对误差小于 `0.001`。未连接相机或机器人。

**文件**：`checkpoint/0809_direct_40000/rma_direct_action_student_student_0040000.json`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 部署异常的中文操作者提示

**变更**：统一 `run_exported_*` 入口默认将部署异常输出为中文故障含义、建议处理和原始错误；新增 `--debug` 用于保留完整 Python traceback。server9 动作锁存超时原始错误补充等待时长、worker 状态/错误码、控制周期、IK tick 和当前动作 tick 诊断。

**动机**：原先的 Python traceback 只能显示 `server9 worker did not latch policy action` 等底层英文异常，操作者难以判断该检查是在保护什么、下一步应进行无运动通信检查还是恢复机器人。

**影响**：仅改变部署 CLI 的错误呈现和错误文本；正常运行路径、动作尺度、限幅、工作空间、初始状态门禁、坐标系、停止时序和共享内存 ABI 均未改变。

**验证**：新增离线单元测试覆盖动作锁存超时、初始状态门禁和 `--debug` 参数；真机、相机和 native worker 未连接或执行。

**文件**：`scripts/policy/run_exported_0711.py`、`franka_sim2real/server9_ipc.py`、`tests/test_run_exported_cli.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 回零脚本的夹爪 homing 门禁

**变更**：`go_to_zero_pose.py` 在任何机械臂或夹爪运动之前读取夹爪状态；`max_width <= 0`、状态非有限、当前宽度或目标 0.04 m 超出报告行程时直接拒绝。夹爪 homing 仍须由操作者显式调用 `minimal_franka_move.py --gripper-homing`，不会自动执行。

**动机**：未 homing 的 Franka gripper 报告 `width=0`、`max_width=0` 时，旧脚本仍发送 0.04 m move，底层可能将其收敛为接近零宽度并闭合夹爪。

**影响**：对有效、已 homing 的夹爪行为不变；无效夹爪状态下不再发送任何机械臂或夹爪运动。未触及策略动作尺度、工作空间、初始状态门禁容差、坐标系、停止时序或共享内存 ABI。

**验证**：`tests/test_go_to_zero_pose.py` 覆盖未 homing 拒绝、行程外拒绝和正常通过；未连接机器人或夹爪。

**文件**：`scripts/robot/go_to_zero_pose.py`、`tests/test_go_to_zero_pose.py`、`README.md`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`

## 2026-08-09 0809 Direct-Action RMA Student 真机部署适配

**变更**：新增 `tacex_rma_direct_action_student_torchscript` v1 的严格部署契约、0809 Direct-Action 配置与入口。模型固定为 RGB、本体状态、动作历史三个输入；streaming 路径不构造或传入 `contact_force_n`。九组仿真 rollout 输入的 CPU/GPU 验证通过后，metadata 记录最大误差 `0.000737`（容差 `0.001`），专用配置使用 `cuda:0`。

**动机**：该 checkpoint 与 XY Student 同为四维动作，但其 TorchScript 签名不含 XY Student 所需的双侧接触输入；通用 legacy contract 又要求缺失的 `policy_contract`，无法安全进入 streaming 部署。

**影响**：配置沿用共用 teacher 的 `[0.05, 0.05, 0.05, 0.01]` 动作尺度，但 direct metadata 未带完整训练环境动作契约，部署前仍须以训练/export manifest 复核。未改变 commissioning 限幅、工作空间、初始状态门禁、基座系 XYZ 约定、30/60/1000 Hz 频率、停止时序或共享内存 ABI。

**验证**：`.venv/bin/python scripts/policy/run_exported_0809_direct_action.py --validate-only` 通过，已校验 SHA-256、三输入/四输出契约并完成 CPU dummy inference；九组 rollout 的 CPU/GPU 最大绝对误差为 `0.000737`，RTX 5060 上单帧 GPU 推理均值为 1.55 ms；`Exported0809DirectActionConfigTests` 3 项通过；`.venv/bin/python -m compileall franka_sim2real scripts tests` 通过；完整 `unittest discover` 共 117 项，仅有已知的 `test_streaming.py::test_rma_streaming_contract_allows_longer_operator_selected_run` 测试桩缺少 `metadata` 的 ERROR。未连接相机或机器人，未执行 preview、streaming-check 或运动测试。

**文件**：`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、`configs/e2e_bundle_real_exported_0809_direct_action.json`、`scripts/policy/run_exported_0809_direct_action.py`、`tests/test_e2e_bundle_runtime.py`、`docs/ARCHITECTURE.md`、`docs/CHANGE_HISTORY.md`、`README.md`

## 2026-08-09 0809 XY RMA Student 真机部署适配

**变更**：新增 0809 配置与入口；部署运行时识别 `tacex_rma_xy_student_torchscript` v8，并将 metadata 要求的 `contact_force_n[2]` 传给 TorchScript。配置显式选择 `rma_contact_force_source: "gripper_is_grasped"`，把真机夹爪单一抓取标志映射为 `[1,1]` 或 `[0,0]`。server9 streaming 每 30 步和末步打印累计接受/超时、实际 policy 输入与输出值。

**动机**：0809 模型把视觉定位改为 XY，并移除了视觉接触预测；其 Actor 仍需要双侧接触力阈值形成的抓取二值特征。现有部署只接受旧 RMA 输入顺序，无法加载该模型。

**影响**：不改变四维动作、动作尺度、工作空间、初始状态门禁、坐标系、streaming 频率或共享内存 ABI。夹爪标志不是左右指尖力读数，禁止以腕部外力替代；该近似仅供效果测试，真实力传感器接入前不应宣称与训练契约等价。

**验证**：`.venv/bin/python -m compileall franka_sim2real scripts tests` 通过；`.venv/bin/python scripts/policy/run_exported_0809_xy_only.py --device cpu --validate-only` 通过，已校验 SHA-256、v8 输入契约并完成 CPU dummy inference；完整 `unittest discover` 共 113 项，仅有已记录的 `test_streaming.py::test_rma_streaming_contract_allows_longer_operator_selected_run` 测试桩缺少 `metadata` 的既有 ERROR。未连接相机或机器人，未执行 preview/streaming/运动测试。

**文件**：`franka_sim2real/e2e_bundle.py`、`franka_sim2real/streaming.py`、`franka_sim2real/streaming_server9.py`、`configs/e2e_bundle_real_exported_0809_xy_only.json`、`scripts/policy/run_exported_0809_xy_only.py`、`tests/test_e2e_bundle_runtime.py`、`docs/ARCHITECTURE.md`、`README.md`

## 2026-08-09 仿真与真机图像统计对比脚本

**变更**：新增 `scripts/diagnostics/compare_sim_real_images.py`，比较仿真与真机策略输入图像的亮度分布、明暗分区通道平衡，以及亮度直方图的 Wasserstein-1 和 Kolmogorov-Smirnov 距离。两侧来源都支持 PNG 文件、图像目录和 NPZ（默认键 `wrist_rgb`），要求尺寸一致，否则报错提示改用模型输入域的图像。输出 `image_stats.json` 与六面板对比图 `comparison.png`。

**动机**：2026-08-08 的噪声标定否定了"仿真噪声太小"的假设，同时暴露出真机图像极暗且双峰（p50 = 6 DN，74% 像素低于 16 DN），并且存在 Franka 状态指示灯造成的绿色偏色（暗区 G/R = 3.97，亮区 G/R = 1.29）。仿真侧是 4800–6200 K 的中性白 DomeLight，白平衡随机化只有 ±0.04，量级完全不匹配。需要一个可重复的工具来量化这个差距，而不是靠看图判断。

**影响**：纯新增的离线分析，不连接机器人也不打开相机，不读写任何配置或 checkpoint。未触及动作尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI。

**验证**：`.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` 共 108 项，仅剩本文件“长期未决事项”中已记录的 `test_streaming.py` 既有 ERROR。`compileall` 通过。用两份真机数据（`runs/20260808_223325_.../rgb/` 300 张 与 `runs/20260809_001349_camera_noise/mean_e100_g64.png`）跑通了完整链路，包括绘图；两者 W1 = 9.7 DN、KS = 0.054，可作为该指标的本底参考。**与仿真的实际对比未运行**：`TacEx/` 侧目前没有已保存的 rollout NPZ，需要先跑 `collect_rma_student_rollouts.py`。

**文件**：`scripts/diagnostics/compare_sim_real_images.py`、`tests/test_compare_sim_real_images.py`、`docs/ARCHITECTURE.md`、`README.md`

## 2026-08-08 相机传感器噪声标定脚本

**变更**：新增 `scripts/calibration/measure_camera_noise.py`，在场景静止的前提下用逐像素时间统计测量 D435 的信号相关噪声模型 `sigma(mu)`，并拟合 `sigma^2 = a*mu + b`。默认在模型输入域（复用 `e2e_bundle` 的 `RealSenseRGBCamera` 与 `_apply_camera_crop`，crop 后双线性缩放）统计，可用 `--domain raw` 切到 640x480。LUT 使用扣除整帧亮度漂移之后的标准差，把工频闪烁和光源热漂移与传感器噪声分开；报告中同时保留未扣除的数值。输出 `noise_report.json`、`noise_lut.csv` 和均值/标准差可视化图，并打印实测值与仿真噪声上界的倍率对照。

**动机**：仿真侧的图像噪声是同方差的（全图共用一个 std），而真实传感器噪声随像素亮度变化，在高增益下的暗区尤其显著；本场景背景接近全黑，这一段正是差异最大的区域。要把仿真噪声改成信号相关模型，先要有真机实测曲线，不能猜。

**影响**：纯新增，只读相机。未触及动作尺度、限幅、工作空间、初始状态门禁、坐标系约定、停止时序或共享内存 ABI。模型输入尺寸从 bundle config 的 `model.metadata_path` → `input_signature.wrist_rgb` 解析，未硬编码 224。脚本内 `--simulation-noise-std` 的默认值 0.006 引自 `TacEx/` 侧配置，仅用于打印对照，可用 CLI 覆盖，不构成跨仓库依赖。

**验证**：`.venv/bin/python -m unittest discover -s tests -p 'test_*.py'` 共 96 项，仅剩本文件“长期未决事项”中已记录的 `test_streaming.py` 既有 ERROR。`.venv/bin/python -m compileall franka_sim2real scripts tests` 通过。`measure_camera_noise.py --help` 通过。**真机采集未运行**：测量结果只有在操作者确认视野静止后才有意义，需人工布置场景后执行。

**文件**：`scripts/calibration/measure_camera_noise.py`、`tests/test_measure_camera_noise.py`、`docs/ARCHITECTURE.md`、`README.md`

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
