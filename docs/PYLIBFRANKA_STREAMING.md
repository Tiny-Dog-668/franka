# Streaming control architecture

## Server version 9: 0801 deployment path

`pylibfranka 0.21.1` speaks server/protocol version 10 and cannot connect to a
server-version-9 robot.  The 0801 deployment therefore builds a small native
worker against the `libfranka 0.17.0` library shipped with the installed
`franky` wheel.  This is still the official libfranka control interface; it
does not use an emulated Python control loop.

The canonical backend name is:

```json
"backend": "server9_joint_position"
```

For the current workstation, `server9_control_cpu` is set to `21`, the CPU
currently serving `enp3s0`'s MSI interrupt. The worker is launched through
`taskset` on that CPU before libfranka upgrades its callback thread to realtime
priority. Re-check the IRQ assignment after reboot or a NIC driver change.

The ownership boundary is intentionally narrow:

```text
policy + camera (Python, 30 Hz)
    -> absolute TCP target in shared memory
native worker (C++, 60 Hz)
    -> custom DLS inverse kinematics, bounded by workspace, joint limits,
       and maximum_joint_target_delta_rad
libfranka Robot::control callback (C++, 1 kHz)
    -> official franka::limitRate with commissioned joint-speed limits
    -> JointPositions under ControllerMode::kJointImpedance
FCI / robot
    -> high-frequency joint impedance (PD-like) tracking
```

The policy never sends a 30 Hz FCI command.  It changes a target at 30 Hz;
the 1 kHz libfranka callback supplies every FCI packet and the robot's joint
impedance controller tracks the smooth joint trajectory.  `franky` is a
wrapper around libfranka, not a separate high-frequency controller.

`franka::limitRate` is applied explicitly with the configured commissioning
velocity limits and libfranka's official joint acceleration/jerk limits.  The
worker disables libfranka's second built-in limiter and low-pass filter, so
there is exactly one, inspectable limiter.  On stop it first commands a
rate-limited deceleration to the current desired joint position, waits for ten
settled 1 kHz cycles, and only then returns `franka::MotionFinished`.  This
avoids ending a motion generator while its trajectory is still moving.

`minimum_command_success_rate` is deliberately not a deployment safety
threshold and has been removed from the configuration.  The official
communication test may report an average below 0.9 as a diagnostic, but FCI
communication errors are handled by libfranka's own control exception.  A
communication-constraint violation must be fixed at the network/real-time
layer before policy deployment; changing an application threshold cannot make
that safe.

The custom parts are only the application-specific DLS IK, workspace checks,
shared-memory handoff, and parent/policy watchdogs.  Their residual risks are
IK singularity/model mismatch, a stale target held until the watchdog expires,
and process scheduling/network packet loss.  The worker bounds shared-memory
reads in the real-time callback and retains the last coherent command rather
than spinning there.

Build and offline checks do not connect to the robot:

```bash
./scripts/build_franka_server9_worker.sh
dist/franka_server9/franka_server9_streaming_worker --layout
dist/franka_server9/franka_server9_streaming_worker --self-test
.venv/bin/python -m unittest tests.test_server9_streaming tests.test_streaming tests.test_e2e_bundle_runtime
```

Before any real FCI session, prepare the host as root using the auditable
script below. It sets the official `performance` CPU governor, disables dynamic
IRQ migration, pins the actual NIC MSI IRQ(s), and disables packet aggregation
offloads on the dedicated FCI interface:

```bash
sudo ./scripts/fci/prepare_fci_host.sh --interface enp3s0 --cpu 21
```

This does not move the robot. It changes host-wide scheduling/network settings;
use `--restore` to return to Ubuntu's `ondemand` governor and re-enable
`irqbalance`.

Check the prepared host without opening an FCI connection or moving the robot:

```bash
sudo ./scripts/fci/test_fci_network.sh --interface enp3s0 --cpu 21 --host 172.16.0.2
```

The test sends 10,000 1200-byte packets at 1 kHz from the same CPU affinity
and FIFO priority used by the worker. It exits non-zero if any latency reaches
1 ms, because FCI must also fit callback computation into that cycle.

## Server version 10

The version-10 path remains `async_position` and uses the separate
`pylibfranka` sidecar described by its build scripts.  It is not compatible
with a server-version-9 robot.
