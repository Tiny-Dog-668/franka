# Franka Sim2Real Validation Framework

This scaffold gives you one shared evaluation API for:

- `sim`: a light mock simulator for fast policy dry-runs
- `real`: the real Franka arm and gripper through `franky`

It is intentionally small and safe enough to start validating policy wiring, logging, action scaling, and sim-to-real rollout flow before you commit to a full simulator stack such as MuJoCo or Isaac.

## Layout

- `franka_sim2real/config.py`: config dataclasses and JSON loading
- `franka_sim2real/types.py`: shared action and observation structures
- `franka_sim2real/policies.py`: built-in policies and policy adapters
- `franka_sim2real/example_policy.py`: a minimal policy adapter example
- `franka_sim2real/safety.py`: action clipping and workspace limiting
- `franka_sim2real/envs/mock_sim.py`: kinematic mock sim backend
- `franka_sim2real/envs/franka_real.py`: real Franka backend
- `franka_sim2real/runner.py`: rollout runner and JSONL logging
- `scripts/policy/sim2real_validate.py`: CLI entrypoint
- `configs/sim2real_validation_example.json`: example config
- `configs/isaacsim_torchscript_example.json`: example config for an Isaac Sim `.pt` policy

## What it validates today

- One policy can run against both `sim` and `real`
- Actions use the same interface: `dx`, `dy`, `dz`, `yaw_deg`, optional `gripper_width`
- Observations use the same interface on both backends
- Workspace and action deltas are clamped before execution
- Every step is logged to `runs/.../episodes.jsonl`
- Run summaries are saved to `runs/.../summary.json`

## Run a mock sim rollout

```bash
source /home/td/franka/.venv/bin/activate
python3 /home/td/franka/scripts/policy/sim2real_validate.py \
  --config /home/td/franka/configs/sim2real_validation_example.json \
  --backend sim
```

## Run a real-robot rollout

Edit the config first, then:

```bash
source /home/td/franka/.venv/bin/activate
python3 /home/td/franka/scripts/policy/sim2real_validate.py \
  --config /home/td/franka/configs/sim2real_validation_example.json \
  --backend real \
  --robot-ip 172.16.0.2
```

The CLI will ask for confirmation before it sends commands to the real robot.

## Policy options

Supported `policy.kind` values:

- `goal_tracking`: simple hand-coded controller for smoke tests
- `zero`: no-op policy
- `random`: random action policy
- `scripted`: replay actions from `policy.scripted_actions`
- `python_callable`: import `module:function` and call it every step
- `torchscript`: load a TorchScript `.pt` policy

For `python_callable`, the function signature is:

```python
def predict(observation_dict, step_index, episode_index, config_dict):
    return [ax, ay, az, ayaw, agripper]
```

If `policy.action_mode` is `normalized`, the vector is interpreted in `[-1, 1]` and scaled using the configured safety limits.

## Isaac Sim `.pt` policies

If your policy comes from Isaac Sim and is a real TorchScript `.pt` export, you can point the config to it:

```json
{
  "policy": {
    "kind": "torchscript",
    "checkpoint_path": "/home/td/franka/policies/policy.pt",
    "device": "cpu",
    "observation_keys": [
      "joint_positions",
      "joint_velocities",
      "tcp_translation",
      "tcp_quaternion",
      "external_wrench",
      "gripper_width"
    ],
    "expects_batch_dim": true
  }
}
```

Then run:

```bash
source /home/td/franka/.venv/bin/activate
python3 /home/td/franka/scripts/policy/sim2real_validate.py \
  --config /home/td/franka/configs/isaacsim_torchscript_example.json \
  --backend real \
  --robot-ip 172.16.0.2
```

Important:

- `torch` must be installed in the active Python environment.
- `checkpoint_path` must point to a TorchScript model that can be loaded with `torch.jit.load`.
- `observation_keys` must match the observation order used during training. This is the most important sim2real wiring detail.
- If your `.pt` is only a training checkpoint and not a TorchScript export, the framework still needs your network definition before it can run inference.

## Important limitation

The current `sim` backend is a mock kinematic simulator, not a contact-rich physics engine. It is designed to validate:

- policy I/O shape
- rollout plumbing
- logging
- safety clipping
- backend switching

If you want real end-to-end contact sim2real for grasping or insertion, the next step should be plugging the same interface into MuJoCo, Isaac Lab, or another physics simulator.
