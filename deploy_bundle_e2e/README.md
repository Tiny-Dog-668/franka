# Sim2Real Grasp Deploy Bundle

Contents:

- `policy_actor_e2e.pt`: end-to-end TorchScript policy
- `policy_actor_e2e.json`: input/output metadata
- `test_policy.py`: standalone smoke test

Expected runtime inputs:

- `action_history`: shape `[N, 5]`, `float32`
- `proprio_obs`: shape `[N, 15]`, `float32`
- `wrist_rgb`: shape `[N, 480, 640, 3]`, `uint8`

Quick test:

```bash
python test_policy.py
```

Test with an image:

```bash
python test_policy.py --image /absolute/path/to/rgb_image.png
```

Optional dependencies:

- `torch`
- `numpy`
- `pillow` only if you use `--image`
