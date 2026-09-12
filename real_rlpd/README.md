# Real-Robot RLPD Residual RL

本目录是在现有 Franka server9 真机部署链路上实现的独立 PyTorch RLPD 模块。它参考根目录
`rlpd/` 的训练思路，但不在生产代码中 import 或依赖该参考仓库。

当前支持：

- 0814 单帧 RGB＋双 GelSight policy；
- 0823 三帧 RGB＋双 GelSight policy；
- 0911 Progress 单帧 RGB＋双 GelSight policy（支持直接部署和 RLPD base）；
- 0912 Progress 三帧 RGB＋双 GelSight policy（独立 Replay/checkpoint contract）；
- XYZ＋gripper 四维 residual action；
- 人工专家全接管的 offline 数据采集；
- policy checkpoint 控制的 online 数据采集；
- offline/online Replay 混合训练；
- AprilTag 离线 reward 和 privileged critic state。

当前默认配置和命令入口使用 **0823 三帧策略**。0814 仍保留用于检查旧数据和旧 checkpoint。0911
使用独立的 `gelsight_reference_progress_single_frame_v1` adapter ID；0912 使用
`gelsight_reference_progress_three_frame_v1`，通过 metadata task 与旧 0823 三帧策略隔离。0911 metadata 保留离线行为审计告警，并在
操作者确认饱和动作符合简单仿真任务预期后允许直接部署和 RLPD，二者仍使用同一套动作与安全契约。

## 1. 核心数据流

```text
Frozen base policy
  ├─ visual feature                  512D
  ├─ left/right tactile feature  256D+256D
  ├─ normalized proprio              15D
  └─ normalized action history        4D
                                      ────
                                feature 1043D

base raw action[4]
       │
       ▼
原 commissioning limiter
       │
       ├──────────────┐
       │              ▼
       │       RLPD Actor state = feature[1043] + base_limited[4]
       │              │
       │              ▼
       │       unit residual[4] ∈ [-1, 1]
       │              │
       └──────────────┴─ post-limit residual composition
                      │
                      ▼
             原有最终 action clip
                      │
                      ▼
        workspace → DLS IK → 1 kHz FCI worker
```

Actor state 是 1047D。Critic 除了 Actor state 和 residual action，还读取由 AprilTag 离线标注的
4D privileged state：

```text
[object_x - tcp_x, object_y - tcp_y, object_z - tcp_z, object_height]
```

AprilTag 信息不会送入部署 Actor。

专家采集额外保存一份策略无关的源 episode：同步腕部 RGB、左右 GelSight 当前帧与固定参考帧、机器人
观测、动作历史、proprio，以及人工专家的绝对物理动作。base feature、base action 和 residual target
只写入当前策略的派生 Replay，不作为专家源数据契约。因而同一任务的专家源 episode 可以用于后续策略
adapter 重新计算特征和 residual，而无需把某个 base policy 的输出当成永久标签。

## 2. Residual action 契约

动作顺序固定为：

```text
[dx, dy, dz, gripper_delta_width]
```

物理动作尺度保持 base policy 原契约：

```text
[0.05, 0.05, 0.05, 0.01] m
```

Residual 在 base action 已经过 commissioning limiter 后叠加：

```text
residual_normalized = unit_residual * (2 * commissioning_action_limit)
candidate_action = base_limited_action + residual_normalized
executed_action = existing_final_clip(candidate_action)
```

`2 × commissioning_action_limit` 可以表达任意两个已限幅动作之间的差，因此专家模式能够完全接管
XYZ 和 gripper。最终命令仍会经过原有安全裁剪；RLPD 路径禁止 full-scale。

XYZ 使用 Franka 基座坐标系，不执行 legacy blocking 路径的 Y/Z 取反。

## 3. RLPD 默认算法

默认参数位于 `configs/real_rlpd_*.json`：

| 参数 | 默认值 |
| --- | --- |
| Actor/Critic hidden dims | `[256, 256]` |
| Q ensemble | 10 |
| Target 随机最小 Q 数量 | 2 |
| Critic LayerNorm | 开启 |
| Offline/online batch 比例 | 50/50 |
| UTD ratio | 20 |
| Batch size | 256 |
| Discount | 0.99 |
| Target update `tau` | 0.005 |
| 初始离线预训练 | 1000 update groups |

一个 update group 包含 20 次 Critic update，随后只执行一次 Actor update 和一次 temperature update。
首次训练只使用 expert offline Replay；生成初始 checkpoint 后，后续训练才按 50/50 混合 offline 和
online 数据。

训练不会与机器人控制并发。每个 online episode 停止并完成离线标注后，再单独运行 `train`。

## 4. 目录结构

| 文件 | 作用 |
| --- | --- |
| `config.py` | RLPD、teleop、residual 和算法配置 |
| `adapters.py` | 不同 frozen base policy 的 1043D feature adapter |
| `action.py` | post-commissioning residual 组成和 expert residual target |
| `runtime.py` | checkpoint 校验、Actor 推理和 fail-closed 部署 |
| `teleop.py` | 30 Hz Pygame 四维专家遥操 |
| `expert_dataset.py` | 策略无关的专家观测与绝对动作源 episode |
| `collector.py` | streaming boundary/action 配对、专家源数据和派生 Replay 缓存 |
| `replay.py` | offline/online SQLite Replay 与 policy contract |
| `labeler.py` | 原始 D435 边界帧的离线 AprilTag 标注 |
| `reward.py` | 可替换的 reward 接口及当前 AprilTag reward |
| `learner.py` | asymmetric ensemble SAC learner |
| `trainer.py` | offline pretrain、混合采样、续训和原子 checkpoint |

统一命令入口是：

```text
scripts/real_rlpd/run_rlpd.py
```

## 5. 专家源数据、派生 Replay 与 rollout 隔离

数据按策略日期隔离，每个命名空间内再分为三层：

- `real_rlpd_data/<policy>/expert/`：该任务采集的策略无关专家源 episode；
- `real_rlpd_data/<policy>/derived/`：针对该 frozen base policy 生成的专家 residual Replay；
- `real_rlpd_data/<policy>/rollout/`：该 residual checkpoint 控制产生的 rollout episode 和 online Replay。

默认布局为：

```text
real_rlpd_data/
├── 0912/
│   ├── expert/<timestamp>_expert_.../
│   │   ├── expert_manifest.json
│   │   ├── expert_boundaries.jsonl
│   │   ├── expert_actions.jsonl
│   │   ├── expert_observations/
│   │   └── raw_rgb/
│   ├── derived/offline_expert.sqlite3
│   └── rollout/
│       ├── <timestamp>_rollout_.../
│       └── online_policy.sqlite3
└── 0913/...
├── rollout/0911_progress/
│   ├── <timestamp>_rollout_.../
│   └── online_policy.sqlite3
└── rollout/0912_progress/
    ├── <timestamp>_rollout_.../
    └── online_policy.sqlite3

checkpoints/real_rlpd/0823/
checkpoints/real_rlpd/0911_progress/
checkpoints/real_rlpd/0912_progress/
```

专家源的 `expert_action_physical_delta_m` 是 canonical 标签，顺序为基座系 XYZ 加 gripper 总宽度增量；
其中不保存 `base_policy_feature`、`base_policy_action` 或 `residual_action`。采集时仍会同步生成当前策略的
offline SQLite，供当前版本直接训练；它是可重建的派生缓存，不是专家源数据本体。

每个派生 Replay 和 checkpoint 都绑定以下契约：

- adapter ID 和 policy kind；
- base model SHA256；
- metadata SHA256；
- feature/state/action 维度；
- commissioning limit 和动作尺度；
- residual composition；
- reward kind 和 reward 参数。

因此 0814/0823/0911 Progress 的派生 Replay、online Replay 和 checkpoint 不能直接混用，但新的专家源 episode 可以
复用。旧 `real_rl_logs/replay.sqlite3` 和 `real_rlpd_runs/` 不会自动迁移；新 `real_rlpd_data/` 与模型
checkpoint 已被根目录 `.gitignore` 忽略。

## 6. 强制安全验收顺序

本仓库会驱动真实 Franka。首次使用新配置或新 checkpoint 时不得跳过以下顺序：

1. 离线 `validate`；
2. `--preview-only`，只读机器人和相机，不发送运动命令；
3. `streaming-check`，建立 server9 控制链路但保持零运动；
4. `--steps 1`，查看 proposed action 后人工输入 `y` 或 `yes`；
5. 小批量 episode；
6. 确认日志、deadline miss、动作方向和 workspace 后再增加步数。

不要关闭 initial-state gate，不要放宽 commissioning limit、workspace、碰撞阈值或 watchdog。人员应远离
机械臂，并始终准备使用急停。

## 7. 离线验证

0823（当前默认）：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0823.json \
  --device cuda:0
```

如需检查旧 0814 契约：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0814.json \
  --device cuda:0
```

0911 Progress：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0911_progress.json \
  --device cuda:0
```

0911 后续专家采集、训练和 online 采集命令与 0823 相同，只需替换 config；其派生数据和 checkpoint
自动写入 `0911_progress` 独立目录。成功标注使用抬升 `0.035 m`、连续 5 次有效 AprilTag 检测，和
Progress 训练终止契约一致。训练时的 150 step horizon 仅作为 provenance，不限制直接部署或 RLPD 的
`--steps`；AprilTag reward 当前仍在控制结束后离线标注，因此成功时由操作者或外部监控停止，而不是
依赖 base policy 的辅助 contact logits。

0912 Progress：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0912_progress.json \
  --device cuda:0
```

0912 后续命令使用同一入口并指定 `configs/real_rlpd_0912_progress.json`；派生 Replay、online Replay
和 checkpoint 均位于 `0912_progress` 独立命名空间，reward 同样使用 `0.035 m` 和连续 5 次检测。

0912 另提供 `configs/real_rlpd_0912_progress_observable_absolute.json`。它保留原配置和数据，使用独立的
`apriltag_x040_observable_absolute_v1` reward、Replay 和 checkpoint 路径。先用 CPU 验证：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py validate \
  --config configs/real_rlpd_0912_progress_observable_absolute.json \
  --device cpu
```

`validate` 不连接机器人和相机。它会加载实际 TorchScript，校验 policy metadata、feature API、模型哈希，
并验证从 1043D feature 重新计算的 base action 与模型输出一致。

## 8. 专家遥操采集

### 8.1 按键

Pygame 窗口获得焦点后按 Enter arm：

| 按键 | 动作 |
| --- | --- |
| `W` / `S` | `+X` / `-X` |
| `A` / `D` | `+Y` / `-Y` |
| `J` / `K` | `+Z` / `-Z` |
| `U` / `I` | 连续打开 / 关闭 gripper |
| 无按键 | 保持当前位置和夹爪宽度 |
| 失去窗口焦点、ESC、关闭窗口 | 停止 episode |

XYZ 对角输入会归一化，避免多轴同时按下时提高合速度。默认遥操速度为 XYZ `0.05 m/s`、gripper
`0.03 m/s`，再按 30 Hz 转换为每步 delta。

RLPD 仍逐步累计并记录 gripper delta；真机执行层会将连续同方向的步进转换为一段同速运动。明确松键
时使用同步 `stop()` 抢占运动并用实测宽度重置累计起点；反向时在 stop 返回后立即启动反向运动，不等待
额外状态读取。被丢弃的单帧 policy 结果不会被误判为松键，只有持续超过 `policy_watchdog_s` 没有接受
新动作才会触发夹爪 hold。这样不会把 Franka Hand 每个具有固定开销的 `move` 动作串行排队，也不会
改变 Replay 中的 30 Hz 步进动作。

每次运行的 `timing_summary.json` 还会写入 `gripper_owner_timing`，其中分别记录 policy 发布到运动/
停止请求的延迟、`move_async`/`stop` API 调用耗时和 hold 后的状态读取耗时；`events` 保留对应的
monotonic 时间戳，便于区分主循环、IPC 和 Franka Hand 服务延迟。

### 8.2 Preview

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-expert \
  --config configs/real_rlpd_0823.json \
  --device cuda:0 \
  --steps 150 \
  --auto-gelsight \
  --preview-only
```

Preview 不写专家源数据或 expert Replay。

### 8.3 Streaming check

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py streaming-check \
  --config configs/real_rlpd_0823.json \
  --device cuda:0 \
  --enable-streaming-check
```

### 8.4 第一次单步采集

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-expert \
  --config configs/real_rlpd_0823.json \
  --device cuda:0 \
  --steps 1 \
  --auto-gelsight \
  --enable-expert-control
```

真机 expert episode 同时要求 `--enable-expert-control` 和启动前的交互确认。没有这两层授权不会开始
运动。专家源观测和 GelSight 数据总会保存；`--save-debug-artifacts` 只控制额外的通用逐步 PNG/NPZ
调试产物。

## 9. 离线 AprilTag 标注

采集开始前会连续取得配置要求数量的有效 AprilTag detection，用中位数建立初始物体高度。预检完成后
检测线程立即关闭，控制期间不会运行 AprilTag 检测。

每个真实 30 Hz action boundary 保存独立的原始 D435 frame 和 monotonic timestamp。worker 停止后，
CLI 默认自动完成 AprilTag 检测、reward 计算及 Replay 回写。只有 accepted 且满足下式的 transition 才会
成为 trainable：

```text
tag_t.capture_timestamp <= action_timestamp < tag_t1.capture_timestamp
```

如使用了 `--defer-offline-label`，稍后手动运行：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py label \
  --config configs/real_rlpd_0823.json \
  --role offline \
  --run-dir real_rlpd_data/0823/expert/<运行目录>
```

当前 reward kind 是 `apriltag_reach_lift_success`：

```text
k_reach * (distance_t - distance_t1)
+ k_lift * (height_t1 - height_t)
+ success_bonus
- k_action * ||unit_residual||²
```

当前 0823 配置在物体相对 episode 初始高度严格超过 `0.05 m`，并连续检测到 3 帧时判定成功；
成功奖励仍只写入触发成功的 transition。该阈值属于 Replay/checkpoint contract，不能与旧阈值的数据或
checkpoint 静默混用。

0912 可观测绝对奖励按 transition 的下一状态计算：

```text
2.5 * (1 - tanh(distance_to_gelpad_midpoint / 0.1))
+ 2.5 * clip(reset_relative_height / 0.035, 0, 1)
+ 1000 * success_event
- 0.05 * mean(executed_action^2)
- 0.05 * mean((executed_action - previous_executed_action)^2)
- 10 * drop_event
- 10 * (physical_tool_clearance < 0.010 m)
```

物理工具 TCP 是夹爪最低点；reward 用固定基座系 `+0.0171 m` 偏移还原 GelSight 接触面中点。成功事件为
相对初始高度达到 `0.035 m` 并连续检测 5 帧。掉落事件在曾达到 `0.020 m` 后首次低于 `0.005 m` 时触发。
当前数据没有可靠的左右接触力、非法碰撞分类和方块姿态，因此仿真的 contact acquisition、15 N 超力、
illegal collision 和 upright success gate 明确不在此可观测版本内；模型 contact head 不参与 reward。

已有 0912 派生 Replay 不能直接换 reward，但可从旧 Replay 非破坏性复制 policy/action/state 字段，然后用
原始边界帧重新标注。目标文件已存在时命令会拒绝覆盖：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py rebuild-reward-replay \
  --config configs/real_rlpd_0912_progress_observable_absolute.json \
  --device cpu --role offline \
  --source-replay real_rlpd_data/0912/derived/offline_expert.sqlite3
```

源 Replay 中已找不到 episode 目录或原始边界帧的旧记录会保留为不可训练条目，并列在 `skipped` 中；它们
不会阻止其余完整 episode 完成重建。

## 10. 初始离线训练

至少准备配置中 `minimum_offline_transitions` 条 trainable expert transition，默认是 1000 条：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py train \
  --config configs/real_rlpd_0823.json \
  --device cuda:0
```

训练启动时打印数据量和 schedule；默认在第 1 个 group、之后每 10 个 group 以及最后一个 group 打印
百分比、累计 `update_group`、elapsed、ETA、`critic_loss`、`q_mean`、`actor_loss`、temperature 和 entropy。
可用 `--progress-interval N` 调整打印间隔，例如：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py train \
  --config configs/real_rlpd_0912_progress_observable_absolute.json \
  --device cuda:0 --progress-interval 10
```

首次执行默认进行 1000 个 offline update group，并生成：

```text
checkpoints/real_rlpd/0823/latest.pt
checkpoints/real_rlpd/0823/actor_latest.ts
checkpoints/real_rlpd/0823/group_XXXXXXXX.pt
```

`latest.pt` 包含 Actor、Q ensemble、target Q、temperature、优化器、frozen normalizer、随机数状态、
policy contract 和 online Replay high-watermark。文件使用临时文件加 `os.replace()` 原子更新。

## 11. Online episode 与 episode 间更新

先 preview checkpoint：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-online \
  --config configs/real_rlpd_0823.json \
  --device cuda:0 \
  --checkpoint checkpoints/real_rlpd/0823/latest.pt \
  --steps 150 \
  --auto-gelsight \
  --preview-only
```

完成验收后第一次仍使用单步：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py collect-online \
  --config configs/real_rlpd_0823.json \
  --device cuda:0 \
  --checkpoint checkpoints/real_rlpd/0823/latest.pt \
  --steps 1 \
  --auto-gelsight \
  --enable-rlpd-control
```

默认执行 deterministic residual。若确实需要 stochastic residual，必须同时添加两个参数：

```text
--stochastic --enable-stochastic-control
```

checkpoint 缺失、损坏、policy SHA/metadata SHA、reward 或动作契约不匹配时，会在 worker 启动前拒绝
episode，不提供 base-only fallback。

每个 online episode 完成并成功标注后运行：

```bash
.venv/bin/python scripts/real_rlpd/run_rlpd.py train \
  --config configs/real_rlpd_0823.json \
  --device cuda:0
```

存在 `latest.pt` 时会自动续训。默认 update group 数等于 checkpoint high-watermark 之后新增的 trainable
online transition 数；每个 batch 严格混合 50% expert offline 和 50% policy online 数据。

## 12. Replay 写入时序

控制期间 collector 只在内存中缓存 transition，不执行逐步 SQLite commit。正常结束或异常停止时，资源
顺序为：

```text
停止机械臂 worker 和 gripper
        ↓
写专家源 episode；单事务写入当前策略的派生 transitions
        ↓
写 raw RGB、rollout、timing 和 control trace
        ↓
运行离线 AprilTag label
```

Deadline miss 会被记录，但该 action 不更新 history，也不会被标为 trainable。

## 13. 扩展新 base policy

新增 policy 时不要直接复用已有 adapter ID。应在 `adapters.py` 中：

1. 为新 metadata `kind` 注册新的稳定 adapter ID；
2. 明确调用该模型导出的 visual/tactile/normalizer API；
3. 拼出 action head 实际使用的 feature；
4. 校验 feature 维度；
5. 用 `tanh(action_head(feature))` 重新计算 base action；
6. 与原模型输出逐元素比较；
7. 增加真实 TorchScript 的 CPU contract 测试；
8. 使用独立的派生 Replay、rollout 和 checkpoint 路径；专家源目录保持共享。

不得根据模型文件名猜测输入顺序、feature 维度、归一化或动作尺度，必须以 TorchScript API 和 metadata
为准。

## 14. 扩展 reward

Reward 实现在 `reward.py` 中，通过 `RewardFunction` 接口调用。新增任务时：

1. 新增一个稳定的 `reward_kind`；
2. 实现 `compute(RewardInput) -> RewardOutput`；
3. 在 `build_reward()` 中注册；
4. 扩展配置验证与单元测试；
5. 使用新的 Replay/checkpoint 路径，避免与旧 reward 数据混用。

不需要修改 Actor、Critic、collector 或 Replay schema。

## 15. 当前已知限制

- 只支持 0814 和 0823 两种 GelSight policy kind；
- 不迁移旧 Real-RL Replay；
- 当前采集会生成专家源 episode 和所选策略的派生 Replay，但尚未提供从历史专家源批量重建另一策略
  Replay 的独立 CLI；
- 训练读取完整 Replay 到内存，适合当前真机数据规模；
- AprilTag 丢失或 timestamp 不满足边界契约的样本会保留，但不可训练；
- 0823 的物理 TCP 额外偏移为 `0.0579 m`，workspace z 下界为 `0.01 m`；这是 workspace/日志用的
  工具最低点，不会改写策略动作坐标系或 IK 的 `O_T_EE` 命令点。

## 16. 离线测试

```bash
.venv/bin/python -m unittest tests.test_real_rlpd -v
.venv/bin/python -m unittest \
  tests.test_real_rlpd \
  tests.test_streaming \
  tests.test_server9_streaming
.venv/bin/python -m compileall -q franka_sim2real real_rlpd scripts tests
git diff --check
```

这些命令不应连接或驱动机器人。
