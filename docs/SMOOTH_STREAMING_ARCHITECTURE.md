# Smooth 0802 streaming architecture

The 0802 policy contract remains unchanged: policy at 30 Hz emits a relative TCP XYZ action,
and DLS updates the joint reference at 60 Hz.  The native libfranka worker remains in one
continuous 1 kHz `Robot::control` session for the rollout.

```text
30 Hz Student policy -> 60 Hz DLS q_ref -> 1 kHz limitRate q_command -> joint impedance
```

Unlike a `JointMotion` call for every policy action, the 1 kHz trajectory state (`q_d`, `dq_d`,
and `ddq_d`) is never reset between policy actions.  The configured velocity, acceleration, and
jerk limits are used both while tracking and during the controlled stop at the end of a rollout.

`run_exported_0802_dr_smooth.py` uses a dedicated configuration.  Its first commissioning
profile is intentionally slower than the legacy worker:

- velocity: `[0.25, 0.25, 0.25, 0.25, 0.50, 0.50, 0.50]` rad/s
- acceleration: `[2, 2, 2, 2, 3, 3, 3]` rad/s²
- jerk: `[100, 100, 100, 100, 150, 150, 150]` rad/s³

These are real-robot trajectory limits, not the simulator PD gains.  The policy action-history
input still records the actual 30 Hz physical TCP/gripper increment.
