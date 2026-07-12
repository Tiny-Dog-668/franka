# End-to-End TorchScript Bundle Deploy

This deploy path is specialized for the bundled policy in `deploy_bundle_e2e/`.

It feeds the model exactly the three inputs declared in the bundle metadata:

- `action_history`: `[1, 5]`
- `proprio_obs`: `[1, 15]`
- `wrist_rgb`: `[1, 480, 640, 3]`

The deploy runner then converts the 5-D raw model output into Franka commands using a configurable action adapter.

## Files

- `deploy_bundle_e2e/policy_actor_e2e.pt`: bundled TorchScript actor
- `deploy_bundle_e2e/policy_actor_e2e.json`: metadata for signatures
- `deploy_bundle_e2e/test_policy.py`: standalone smoke test
- `franka_sim2real/e2e_bundle.py`: deployment runtime
- `scripts/policy/run_e2e_bundle.py`: CLI entrypoint
- `configs/e2e_bundle_real_example.json`: deployment config

## Action alignment used here

This deploy path treats the 5 policy outputs as the normalized control vector:

- `dx`
- `dy`
- `dz`
- `yaw_deg`
- `gripper`

That means the physical semantics are aligned as:

- action `0`: normalized Cartesian `dx`
- action `1`: normalized Cartesian `dy`
- action `2`: normalized Cartesian `dz`
- action `3`: normalized yaw delta about tool Z
- action `4`: normalized gripper open/close command

The bundle metadata confirms only the shape, not the physical semantics, so this alignment is an inference from the training task (`TacEx-Sim2Real-Grasp-v0`), the 15-D proprio layout, the 5-D action history, and common Isaac/Sim grasp-control conventions.

## Important runtime behavior

The TorchScript model outputs raw actor means, and those means are not guaranteed to lie in `[-1, 1]`. In practice this bundle already emits values outside that range.

To match the usual PPO deployment path more closely, the runner now:

1. clips the raw 5-D policy output to `action_adapter.clip_low/high`
2. uses the clipped normalized action for scaling into real robot deltas
3. feeds the clipped normalized action back into `action_history` by default

The shipped config therefore uses:

- `action_adapter.clip_low = [-1, -1, -1, -1, -1]`
- `action_adapter.clip_high = [1, 1, 1, 1, 1]`
- `model.history_source = "clipped_action"`
- `action_adapter.gripper_mode = "delta_width"`

The shipped config now keeps the gripper on the same incremental-control pattern as `dx/dy/dz`: action `4` is treated as a normalized jaw-width delta, with `1.0 -> +0.004 m` per step and `-1.0 -> -0.004 m` per step.

## Run the bundle smoke test

```bash
source /home/td/franka/.venv/bin/activate
python3 /home/td/franka/deploy_bundle_e2e/test_policy.py --runs 1 --warmup 0
```

## Run on the real robot

```bash
source /home/td/franka/.venv/bin/activate
python3 /home/td/franka/scripts/policy/run_e2e_bundle.py \
  --config /home/td/franka/configs/e2e_bundle_real_example.json \
  --robot-ip 172.16.0.2
```

The runner will:

1. Connect to Franka arm and gripper
2. Open the D435 color stream
3. Build `proprio_obs` from:
   - 7 joint positions
   - 7 joint velocities
   - 1 gripper width
4. Build `action_history` from the previous clipped normalized 5-D action
5. Run TorchScript inference
6. Clip the normalized action, then map it into a safe Franka command
7. Log observations, raw actions, mapped actions, and RGB frames under `runs/...`

If the D435 is not connected yet, you can temporarily switch the config to:

```json
"camera": {
  "source": "image",
  "image_path": "/absolute/path/to/a/640x480_rgb.png"
}
```

This is useful for validating the TorchScript bundle, observation packing, and action mapping before you move to the live camera.

## Current limits

- The deploy loop currently uses only the RGB color stream from D435
- The action mapping is conservative by default
- I could not read the original `TacEx` training repo from this workstation, so the 5-D semantics are still an informed deployment inference rather than a verbatim copy of `sim2real_grasp_env.py`
- If you later recover the original environment code and it defines the 5th action differently, update `action_adapter.gripper_mode` first, then `labels/scales` if needed
