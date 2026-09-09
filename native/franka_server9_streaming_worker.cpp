#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <franka/control_types.h>
#include <franka/exception.h>
#include <franka/model.h>
#include <franka/rate_limiting.h>
#include <franka/robot.h>
#include <franka/robot_state.h>
#include <research_interface/robot/service_types.h>

namespace {

constexpr uint64_t kMagic = 0x46524B4139535452ULL;  // "FRKA9STR"
constexpr uint32_t kAbiVersion = 11;
constexpr size_t kTraceCapacity = 1024;
constexpr size_t kFciLogCapacity = 1024;
constexpr size_t kCommandReadRetries = 8;
constexpr double kStopVelocityThresholdRadS = 0.005;
// The FCI validates the commanded trajectory when MotionFinished is raised, so
// the settle test has to inspect the command about to be sent rather than how
// slowly the arm happens to be moving. franka::limitRate overshoots zero while
// landing and rings inside a +/-0.013 rad/s band, so any tolerance inside that
// band cannot distinguish a stopped command from a ringing one. It does converge
// to a bit-exact zero, within 163 ms from the fastest velocity the commissioned
// envelope allows, so these epsilons only absorb floating point dust.
constexpr double kStopCommandDeltaEpsilonRad = 1e-12;
constexpr double kStopCommandVelocityEpsilonRadS = 1e-9;
constexpr double kStopCommandAccelerationEpsilonRadS2 = 1e-6;
constexpr uint64_t kStopSettledCycles = 100;
constexpr double kStopTimeoutS = 2.0;
constexpr uint64_t kDlsTicksPerPolicyAction = 2;
constexpr double kControlPeriodS = 1e-3;
// Critically damped bandwidth for tracking the simulator-derived joint velocity
// reference. Acceleration and jerk are bounded separately below, so this only
// controls how quickly a reachable reference is approached.
constexpr double kVelocityTrackingOmegaRadS = 100.0;
constexpr std::array<double, 7> kJointLower{
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973};
constexpr std::array<double, 7> kJointUpper{
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973};

enum class Command : uint32_t { kWait = 0, kStart = 1, kStop = 2 };
// kJointPositionPursuit chases a reference a fixed joint delta ahead of the
// measured position. libfranka::limitRate then reads an implied velocity of
// delta/period, which for any usable delta saturates the velocity limit, so
// the joint speed is the velocity limit regardless of how large the policy
// action was. Isaac Lab's implicit PD instead settles at
// qd = (stiffness / damping) * dq_ik, which is proportional to the action.
// kSimActuatorVelocity reproduces that proportional law.
enum class ControlLaw : uint64_t { kJointPositionPursuit = 0, kSimActuatorVelocity = 1 };
enum class ImpedanceMode : uint64_t { kJoint = 0, kCartesian = 1 };
enum class Status : uint32_t {
  kInitializing = 0,
  kReady = 1,
  kRunning = 2,
  kStopped = 3,
  kAborted = 4,
};
enum class ErrorCode : int32_t {
  kNone = 0,
  kConfiguration = 1,
  kConnection = 2,
  kControl = 3,
  kParentWatchdog = 4,
  kPolicyWatchdog = 5,
  kRobotError = 6,
  kInvalidState = 8,
  kWorkspace = 9,
  kIk = 10,
};

struct TraceEntry {
  uint64_t tick_index;
  uint64_t monotonic_ns;
  uint64_t action_generation;
  double q[7];
  double q_target[7];
  double command_success_rate;
};

// Exact state/command pairs retained by libfranka immediately before a
// ControlException. Unlike the normal 60 Hz trace, RobotCommand is the command
// that FCI actually received after libfranka's conversion path.
struct FciLogEntry {
  double q_command[7];
  double q_d[7];
  double dq_d[7];
  double ddq_d[7];
  double dq[7];
  double command_success_rate;
};

struct SharedData {
  uint64_t magic;
  uint32_t abi_version;
  uint32_t struct_size;

  alignas(8) uint64_t command_seq;
  uint64_t parent_heartbeat_ns;
  uint64_t policy_heartbeat_ns;
  uint64_t action_generation;
  uint32_t command;
  uint32_t streaming_check;
  double delta_xyz[3];
  double target_pose[16];
  double maximum_joint_velocities[7];
  double joint_impedance[7];
  double cartesian_impedance[6];
  uint64_t impedance_mode;
  double maximum_joint_accelerations[7];
  double maximum_joint_jerks[7];
  double dls_lambda;
  double maximum_joint_target_delta_rad;
  double joint_limit_margin_rad;
  double policy_watchdog_s;
  double control_watchdog_s;
  double workspace_minimum[3];
  double workspace_maximum[3];
  double tool_tcp_offset_ee[3];
  uint64_t control_law;
  // Gain applied to the DLS joint delta to obtain a joint velocity. Set it to
  // the simulator's stiffness/damping ratio to match the trained dynamics.
  double reference_velocity_gain;
  // Reference limiting on the DLS delta before the gain, so a large pose error
  // cannot demand an unbounded velocity. Set it wide enough not to bind at the
  // trained action scale, otherwise it discards the DLS magnitude again.
  double maximum_ik_reference_delta_rad;

  alignas(8) uint64_t state_seq;
  uint64_t worker_heartbeat_ns;
  uint64_t control_cycle_count;
  uint64_t ik_tick_count;
  uint64_t latched_action_generation;
  uint64_t latched_action_tick_count;
  uint64_t trace_count;
  uint32_t status;
  int32_t error_code;
  uint32_t robot_mode;
  uint32_t has_errors;
  double command_success_rate;
  double q[7];
  double dq[7];
  double O_T_EE[16];
  double F_T_EE[16];
  double external_wrench[6];
  double q_target[7];
  char error_message[256];
  TraceEntry trace[kTraceCapacity];
  uint64_t fci_log_count;
  FciLogEntry fci_log[kFciLogCapacity];
  uint32_t collision_behavior_enabled;
  uint32_t collision_behavior_reserved;
  double lower_torque_thresholds[7];
  double upper_torque_thresholds[7];
  double lower_force_thresholds[6];
  double upper_force_thresholds[6];
};

struct CommandSnapshot {
  uint64_t parent_heartbeat_ns{};
  uint64_t policy_heartbeat_ns{};
  uint64_t action_generation{};
  Command command{Command::kWait};
  bool streaming_check{};
  std::array<double, 3> delta_xyz{};
  std::array<double, 16> target_pose{};
};

std::atomic<bool> g_signal_stop{false};

uint64_t monotonicNs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

uint64_t atomicLoad(const uint64_t* value) {
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

void atomicStore(uint64_t* target, uint64_t value) {
  __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

void signalHandler(int) {
  g_signal_stop.store(true, std::memory_order_relaxed);
}

bool finiteArray(const double* values, size_t size) {
  for (size_t index = 0; index < size; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

bool validThresholdPair(const double* lower, const double* upper, size_t size) {
  if (!finiteArray(lower, size) || !finiteArray(upper, size)) {
    return false;
  }
  for (size_t index = 0; index < size; ++index) {
    if (lower[index] <= 0.0 || upper[index] <= 0.0 || lower[index] > upper[index]) {
      return false;
    }
  }
  return true;
}

bool shouldLatchAction(uint64_t pending_generation,
                       uint64_t latched_generation,
                       uint64_t dls_ticks_on_generation) {
  return pending_generation != 0 && pending_generation != latched_generation &&
         (latched_generation == 0 || dls_ticks_on_generation >= kDlsTicksPerPolicyAction);
}

bool shouldRunDls(bool hold_only,
                  uint64_t latched_generation,
                  uint64_t dls_ticks_on_generation) {
  // Before the worker has latched an explicit parent action, q_d is the only
  // safe target. A TCP pose sampled before the control loop started can differ
  // slightly from the live robot state and must not create a first correction.
  return !hold_only && latched_generation != 0 &&
         dls_ticks_on_generation < kDlsTicksPerPolicyAction;
}

std::string activeRobotErrorMessage(const franka::RobotState& state) {
  std::ostringstream message;
  message << "robot reported an active error: " << state.current_errors
          << "; last_motion_errors: " << state.last_motion_errors;
  return message.str();
}

bool tryReadCommand(const SharedData& shared, CommandSnapshot* result) {
  for (size_t attempt = 0; attempt < kCommandReadRetries; ++attempt) {
    const uint64_t before = atomicLoad(&shared.command_seq);
    if ((before & 1U) != 0U) {
      continue;
    }
    CommandSnapshot candidate;
    candidate.parent_heartbeat_ns = atomicLoad(&shared.parent_heartbeat_ns);
    candidate.policy_heartbeat_ns = atomicLoad(&shared.policy_heartbeat_ns);
    candidate.action_generation = shared.action_generation;
    candidate.command = static_cast<Command>(shared.command);
    candidate.streaming_check = shared.streaming_check != 0U;
    std::copy(std::begin(shared.delta_xyz), std::end(shared.delta_xyz),
              candidate.delta_xyz.begin());
    std::copy(std::begin(shared.target_pose), std::end(shared.target_pose),
              candidate.target_pose.begin());
    const uint64_t after = atomicLoad(&shared.command_seq);
    if (before == after && (after & 1U) == 0U) {
      *result = candidate;
      return true;
    }
  }
  return false;
}

CommandSnapshot readCommand(const SharedData& shared) {
  CommandSnapshot result;
  while (!tryReadCommand(shared, &result)) {
    std::this_thread::yield();
  }
  return result;
}

void beginStateWrite(SharedData& shared) {
  const uint64_t sequence = atomicLoad(&shared.state_seq);
  atomicStore(&shared.state_seq, (sequence & 1U) == 0U ? sequence + 1U : sequence + 2U);
}

void endStateWrite(SharedData& shared) {
  const uint64_t sequence = atomicLoad(&shared.state_seq);
  atomicStore(&shared.state_seq, (sequence & 1U) != 0U ? sequence + 1U : sequence + 2U);
}

void copyState(SharedData& shared,
               const franka::RobotState& state,
               const std::array<double, 7>& q_target,
               Status status,
               ErrorCode error_code,
               const std::string& error_message,
               uint64_t latched_action_generation = 0,
               uint64_t latched_action_tick_count = 0) {
  beginStateWrite(shared);
  shared.latched_action_generation = latched_action_generation;
  shared.latched_action_tick_count = latched_action_tick_count;
  shared.status = static_cast<uint32_t>(status);
  shared.error_code = static_cast<int32_t>(error_code);
  shared.robot_mode = static_cast<uint32_t>(state.robot_mode);
  shared.has_errors = static_cast<bool>(state.current_errors) ? 1U : 0U;
  shared.command_success_rate = state.control_command_success_rate;
  std::copy(state.q.begin(), state.q.end(), std::begin(shared.q));
  std::copy(state.dq.begin(), state.dq.end(), std::begin(shared.dq));
  std::copy(state.O_T_EE.begin(), state.O_T_EE.end(), std::begin(shared.O_T_EE));
  std::copy(state.F_T_EE.begin(), state.F_T_EE.end(), std::begin(shared.F_T_EE));
  std::copy(state.O_F_ext_hat_K.begin(), state.O_F_ext_hat_K.end(),
            std::begin(shared.external_wrench));
  std::copy(q_target.begin(), q_target.end(), std::begin(shared.q_target));
  std::memset(shared.error_message, 0, sizeof(shared.error_message));
  std::strncpy(shared.error_message, error_message.c_str(), sizeof(shared.error_message) - 1U);
  endStateWrite(shared);
}

std::array<double, 3> translation(const std::array<double, 16>& pose) {
  return {pose[12], pose[13], pose[14]};
}

std::array<double, 3> toolTcpTranslation(const std::array<double, 16>& pose,
                                         const SharedData& shared) {
  // O_T_EE is column-major. The offset is local to EE, never a fixed world-Z
  // correction.
  return {
      pose[12] + pose[0] * shared.tool_tcp_offset_ee[0] +
          pose[4] * shared.tool_tcp_offset_ee[1] + pose[8] * shared.tool_tcp_offset_ee[2],
      pose[13] + pose[1] * shared.tool_tcp_offset_ee[0] +
          pose[5] * shared.tool_tcp_offset_ee[1] + pose[9] * shared.tool_tcp_offset_ee[2],
      pose[14] + pose[2] * shared.tool_tcp_offset_ee[0] +
          pose[6] * shared.tool_tcp_offset_ee[1] + pose[10] * shared.tool_tcp_offset_ee[2],
  };
}

bool insideWorkspace(const std::array<double, 3>& position, const SharedData& shared) {
  for (size_t axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(position[axis]) || position[axis] < shared.workspace_minimum[axis] ||
        position[axis] > shared.workspace_maximum[axis]) {
      return false;
    }
  }
  return true;
}

std::array<double, 3> rotationVector(const std::array<double, 16>& current,
                                     const std::array<double, 16>& target) {
  double error_rotation[3][3]{};
  for (size_t row = 0; row < 3; ++row) {
    for (size_t column = 0; column < 3; ++column) {
      for (size_t inner = 0; inner < 3; ++inner) {
        error_rotation[row][column] += target[row + inner * 4] * current[column + inner * 4];
      }
    }
  }
  const double cosine = std::clamp(
      (error_rotation[0][0] + error_rotation[1][1] + error_rotation[2][2] - 1.0) * 0.5,
      -1.0, 1.0);
  const double angle = std::acos(cosine);
  const std::array<double, 3> skew{
      error_rotation[2][1] - error_rotation[1][2],
      error_rotation[0][2] - error_rotation[2][0],
      error_rotation[1][0] - error_rotation[0][1]};
  if (angle < 1e-8) {
    return {0.5 * skew[0], 0.5 * skew[1], 0.5 * skew[2]};
  }
  const double sine = std::sin(angle);
  if (std::abs(sine) < 1e-8) {
    throw std::runtime_error("orientation error is singular");
  }
  const double scale = angle / (2.0 * sine);
  return {scale * skew[0], scale * skew[1], scale * skew[2]};
}

std::array<double, 6> poseError(const std::array<double, 16>& current,
                                const std::array<double, 16>& target) {
  const auto orientation = rotationVector(current, target);
  return {target[12] - current[12], target[13] - current[13], target[14] - current[14],
          orientation[0], orientation[1], orientation[2]};
}

std::array<double, 6> solve6x6(double matrix[6][6], const std::array<double, 6>& rhs) {
  double augmented[6][7]{};
  for (size_t row = 0; row < 6; ++row) {
    for (size_t column = 0; column < 6; ++column) {
      augmented[row][column] = matrix[row][column];
    }
    augmented[row][6] = rhs[row];
  }
  for (size_t pivot = 0; pivot < 6; ++pivot) {
    size_t best = pivot;
    for (size_t row = pivot + 1; row < 6; ++row) {
      if (std::abs(augmented[row][pivot]) > std::abs(augmented[best][pivot])) {
        best = row;
      }
    }
    if (std::abs(augmented[best][pivot]) < 1e-12) {
      throw std::runtime_error("DLS solve is singular");
    }
    if (best != pivot) {
      for (size_t column = pivot; column < 7; ++column) {
        std::swap(augmented[pivot][column], augmented[best][column]);
      }
    }
    const double divisor = augmented[pivot][pivot];
    for (size_t column = pivot; column < 7; ++column) {
      augmented[pivot][column] /= divisor;
    }
    for (size_t row = 0; row < 6; ++row) {
      if (row == pivot) {
        continue;
      }
      const double factor = augmented[row][pivot];
      for (size_t column = pivot; column < 7; ++column) {
        augmented[row][column] -= factor * augmented[pivot][column];
      }
    }
  }
  std::array<double, 6> solution{};
  for (size_t row = 0; row < 6; ++row) {
    solution[row] = augmented[row][6];
  }
  return solution;
}

std::array<double, 7> dlsDelta(const franka::Model& model,
                               const franka::RobotState& state,
                               const std::array<double, 16>& target_pose,
                               double damping) {
  const auto jacobian = model.zeroJacobian(franka::Frame::kEndEffector, state);
  const auto error = poseError(state.O_T_EE, target_pose);
  if (!finiteArray(jacobian.data(), jacobian.size()) || !finiteArray(error.data(), error.size())) {
    throw std::runtime_error("DLS input contains NaN or Inf");
  }
  double regularized[6][6]{};
  for (size_t row = 0; row < 6; ++row) {
    for (size_t column = 0; column < 6; ++column) {
      for (size_t joint = 0; joint < 7; ++joint) {
        regularized[row][column] += jacobian[row + joint * 6] * jacobian[column + joint * 6];
      }
    }
    regularized[row][row] += damping * damping;
  }
  const auto solved = solve6x6(regularized, error);
  std::array<double, 7> delta{};
  for (size_t joint = 0; joint < 7; ++joint) {
    double value = 0.0;
    for (size_t axis = 0; axis < 6; ++axis) {
      value += jacobian[axis + joint * 6] * solved[axis];
    }
    if (!std::isfinite(value)) {
      throw std::runtime_error("DLS result contains NaN or Inf");
    }
    delta[joint] = value;
  }
  return delta;
}

std::array<double, 7> clampToJointLimits(const std::array<double, 7>& q,
                                         const std::array<double, 7>& delta,
                                         double maximum_delta,
                                         double joint_margin) {
  std::array<double, 7> result{};
  for (size_t joint = 0; joint < 7; ++joint) {
    const double limited = std::clamp(delta[joint], -maximum_delta, maximum_delta);
    result[joint] = std::clamp(q[joint] + limited, kJointLower[joint] + joint_margin,
                               kJointUpper[joint] - joint_margin);
  }
  return result;
}

std::array<double, 7> dlsTarget(const franka::Model& model,
                                const franka::RobotState& state,
                                const std::array<double, 16>& target_pose,
                                double damping,
                                double maximum_joint_target_delta,
                                double joint_margin) {
  return clampToJointLimits(state.q, dlsDelta(model, state, target_pose, damping),
                            maximum_joint_target_delta, joint_margin);
}

// Signed steady-state joint velocity from Isaac Lab's implicit-PD relation
// qd = (stiffness / damping) * dq_ik. This is a reference, not a time-varying
// hard velocity bound: lowering a hard bound below the current dq_d makes
// libfranka::limitRate violate its own jerk limit in order to enter that bound.
std::array<double, 7> simActuatorVelocityReference(const std::array<double, 7>& delta,
                                                   double gain,
                                                   double maximum_reference_delta,
                                                   const double* maximum_joint_velocities) {
  std::array<double, 7> reference{};
  for (size_t joint = 0; joint < 7; ++joint) {
    const double limited =
        std::clamp(delta[joint], -maximum_reference_delta, maximum_reference_delta);
    reference[joint] = std::clamp(gain * limited, -maximum_joint_velocities[joint],
                                  maximum_joint_velocities[joint]);
  }
  return reference;
}

std::array<double, 7> trackVelocityReference(
    const std::array<double, 7>& reference,
    const std::array<double, 7>& q_d,
    const std::array<double, 7>& dq_d,
    const std::array<double, 7>& ddq_d,
    const double* maximum_joint_velocities,
    const std::array<double, 7>& maximum_joint_acceleration,
    const std::array<double, 7>& maximum_joint_jerk) {
  std::array<double, 7> command{};
  for (size_t joint = 0; joint < 7; ++joint) {
    // Critical damping in velocity space. The jerk clamp makes every reference
    // step (including a sign reversal or stale-action zero) a valid continuation
    // of the previous FCI trajectory.
    const double requested_jerk =
        kVelocityTrackingOmegaRadS * kVelocityTrackingOmegaRadS *
            (reference[joint] - dq_d[joint]) -
        2.0 * kVelocityTrackingOmegaRadS * ddq_d[joint];
    const double limited_jerk = std::clamp(
        requested_jerk, -maximum_joint_jerk[joint], maximum_joint_jerk[joint]);
    const double acceleration = std::clamp(
        ddq_d[joint] + limited_jerk * kControlPeriodS,
        -maximum_joint_acceleration[joint], maximum_joint_acceleration[joint]);
    double velocity = dq_d[joint] + acceleration * kControlPeriodS;
    if (reference[joint] == 0.0) {
      // Land on an exact zero in finite time once doing so fits inside both the
      // acceleration and jerk envelopes. This avoids an asymptotic tail while
      // preserving a valid continuation of the previous FCI trajectory.
      const double exact_stop_acceleration = -dq_d[joint] / kControlPeriodS;
      const double exact_stop_jerk =
          (exact_stop_acceleration - ddq_d[joint]) / kControlPeriodS;
      if (std::abs(exact_stop_acceleration) <= maximum_joint_acceleration[joint] &&
          std::abs(exact_stop_jerk) <= maximum_joint_jerk[joint] &&
          // The following zero-velocity cycle must also be able to remove the
          // landing acceleration without exceeding jerk.
          std::abs(exact_stop_acceleration / kControlPeriodS) <=
              maximum_joint_jerk[joint]) {
        velocity = 0.0;
      }
    }
    // A critically damped step does not overshoot its bounded reference. Keep a
    // numerical guard for non-ideal initial FCI states without changing the
    // normal trajectory.
    velocity = std::clamp(velocity, -maximum_joint_velocities[joint],
                          maximum_joint_velocities[joint]);
    command[joint] = q_d[joint] + velocity * kControlPeriodS;
  }
  return command;
}

double maximumAbsolute(const std::array<double, 7>& values) {
  double maximum = 0.0;
  for (double value : values) {
    maximum = std::max(maximum, std::abs(value));
  }
  return maximum;
}

// True when q_command holds q_d, so the commanded velocity and acceleration are
// both zero and MotionFinished is a valid continuation of the trajectory.
// Signalling MotionFinished on any other command makes the FCI reject it with
// joint_motion_generator_velocity_discontinuity.
bool isZeroVelocityCommand(const std::array<double, 7>& q_command,
                           const std::array<double, 7>& q_d,
                           const std::array<double, 7>& dq_d,
                           const std::array<double, 7>& ddq_d) {
  for (size_t joint = 0; joint < 7; ++joint) {
    if (std::abs(q_command[joint] - q_d[joint]) > kStopCommandDeltaEpsilonRad ||
        std::abs(dq_d[joint]) > kStopCommandVelocityEpsilonRadS ||
        std::abs(ddq_d[joint]) > kStopCommandAccelerationEpsilonRadS2) {
      return false;
    }
  }
  return true;
}

void validateCurrentState(const franka::RobotState& state, const SharedData& shared) {
  if (!finiteArray(state.q.data(), state.q.size()) ||
      !finiteArray(state.O_T_EE.data(), state.O_T_EE.size())) {
    throw std::runtime_error("robot state contains NaN or Inf");
  }
  for (size_t joint = 0; joint < 7; ++joint) {
    if (state.q[joint] < kJointLower[joint] || state.q[joint] > kJointUpper[joint]) {
      throw std::runtime_error("current joint position is outside Panda limits");
    }
  }
  if (!insideWorkspace(toolTcpTranslation(state.O_T_EE, shared), shared)) {
    throw std::runtime_error("current physical tool TCP is outside configured workspace");
  }
}

void appendTrace(SharedData& shared,
                 uint64_t tick,
                 uint64_t action_generation,
                 const franka::RobotState& state,
                 const std::array<double, 7>& q_target) {
  if (shared.trace_count >= kTraceCapacity) {
    return;
  }
  TraceEntry& entry = shared.trace[shared.trace_count];
  entry.tick_index = tick;
  entry.monotonic_ns = monotonicNs();
  entry.action_generation = action_generation;
  std::copy(state.q.begin(), state.q.end(), std::begin(entry.q));
  std::copy(q_target.begin(), q_target.end(), std::begin(entry.q_target));
  entry.command_success_rate = state.control_command_success_rate;
  ++shared.trace_count;
}

void copyFciExceptionLog(SharedData& shared, const std::vector<franka::Record>& log) {
  const size_t count = std::min(log.size(), kFciLogCapacity);
  const size_t start = log.size() - count;
  for (size_t index = 0; index < count; ++index) {
    const franka::Record& record = log[start + index];
    FciLogEntry& entry = shared.fci_log[index];
    std::copy(record.command.joint_positions.q.begin(),
              record.command.joint_positions.q.end(), std::begin(entry.q_command));
    std::copy(record.state.q_d.begin(), record.state.q_d.end(), std::begin(entry.q_d));
    std::copy(record.state.dq_d.begin(), record.state.dq_d.end(), std::begin(entry.dq_d));
    std::copy(record.state.ddq_d.begin(), record.state.ddq_d.end(), std::begin(entry.ddq_d));
    std::copy(record.state.dq.begin(), record.state.dq.end(), std::begin(entry.dq));
    entry.command_success_rate = record.state.control_command_success_rate;
  }
  shared.fci_log_count = count;
}

class Mapping {
 public:
  explicit Mapping(const std::string& path) {
    fd_ = open(path.c_str(), O_RDWR);
    if (fd_ < 0) {
      throw std::runtime_error("failed to open shared-memory file");
    }
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size != static_cast<off_t>(sizeof(SharedData))) {
      close(fd_);
      throw std::runtime_error("shared-memory file has the wrong size");
    }
    void* address = mmap(nullptr, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (address == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("failed to map shared-memory file");
    }
    shared_ = static_cast<SharedData*>(address);
  }

  ~Mapping() {
    if (shared_ != nullptr) {
      munmap(shared_, sizeof(SharedData));
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  SharedData& get() { return *shared_; }

 private:
  int fd_{-1};
  SharedData* shared_{nullptr};
};

int run(const std::string& robot_ip, const std::string& shared_path) {
  Mapping mapping(shared_path);
  SharedData& shared = mapping.get();
  if (shared.magic != kMagic || shared.abi_version != kAbiVersion ||
      shared.struct_size != sizeof(SharedData)) {
    throw std::runtime_error("shared-memory ABI mismatch");
  }
  if (!finiteArray(shared.maximum_joint_velocities, 7) ||
      !finiteArray(shared.maximum_joint_accelerations, 7) ||
      !finiteArray(shared.maximum_joint_jerks, 7) ||
      !finiteArray(shared.tool_tcp_offset_ee, 3) || shared.dls_lambda <= 0.0 ||
      !std::isfinite(shared.maximum_joint_target_delta_rad) ||
      shared.maximum_joint_target_delta_rad <= 0.0 || shared.control_watchdog_s <= 0.0 ||
      shared.policy_watchdog_s <= 0.0) {
    throw std::runtime_error("invalid streaming configuration");
  }
  const ControlLaw control_law = static_cast<ControlLaw>(shared.control_law);
  if (control_law != ControlLaw::kJointPositionPursuit &&
      control_law != ControlLaw::kSimActuatorVelocity) {
    throw std::runtime_error("unknown control_law");
  }
  const ImpedanceMode impedance_mode = static_cast<ImpedanceMode>(shared.impedance_mode);
  if (impedance_mode != ImpedanceMode::kJoint &&
      impedance_mode != ImpedanceMode::kCartesian) {
    throw std::runtime_error("unknown impedance_mode");
  }
  if (control_law == ControlLaw::kSimActuatorVelocity) {
    if (!std::isfinite(shared.reference_velocity_gain) || shared.reference_velocity_gain <= 0.0 ||
        shared.reference_velocity_gain > 100.0) {
      throw std::runtime_error("reference_velocity_gain must be finite and in (0, 100]");
    }
    if (!std::isfinite(shared.maximum_ik_reference_delta_rad) ||
        shared.maximum_ik_reference_delta_rad <= 0.0 ||
        shared.maximum_ik_reference_delta_rad > 0.5) {
      throw std::runtime_error("maximum_ik_reference_delta_rad must be finite and in (0, 0.5]");
    }
  }
  const bool set_joint_impedance = finiteArray(shared.joint_impedance, 7);
  if (!set_joint_impedance) {
    for (const double value : shared.joint_impedance) {
      if (!std::isnan(value)) {
        throw std::runtime_error(
            "joint_impedance must be either seven finite values or disabled (all NaN)");
      }
    }
  } else {
    for (const double value : shared.joint_impedance) {
      if (value <= 0.0 || value > 14250.0) {
        throw std::runtime_error("joint_impedance values must be in (0, 14250]");
      }
    }
  }
  const bool set_cartesian_impedance = finiteArray(shared.cartesian_impedance, 6);
  if (!set_cartesian_impedance) {
    for (const double value : shared.cartesian_impedance) {
      if (!std::isnan(value)) {
        throw std::runtime_error(
            "cartesian_impedance must be either six finite values or disabled (all NaN)");
      }
    }
  } else {
    for (size_t axis = 0; axis < 6; ++axis) {
      const double lower = axis < 3 ? 10.0 : 1.0;
      const double upper = axis < 3 ? 3000.0 : 300.0;
      if (shared.cartesian_impedance[axis] < lower ||
          shared.cartesian_impedance[axis] > upper) {
        throw std::runtime_error(
            "cartesian_impedance must be [10, 3000] N/m for XYZ and [1, 300] Nm/rad for rotation");
      }
    }
  }
  if (impedance_mode == ImpedanceMode::kJoint && set_cartesian_impedance) {
    throw std::runtime_error("cartesian_impedance requires impedance_mode='cartesian'");
  }
  if (impedance_mode == ImpedanceMode::kCartesian && !set_cartesian_impedance) {
    throw std::runtime_error("cartesian_impedance is required in Cartesian impedance mode");
  }
  if (impedance_mode == ImpedanceMode::kCartesian && set_joint_impedance) {
    throw std::runtime_error("joint_impedance must be disabled in Cartesian impedance mode");
  }
  if (shared.collision_behavior_enabled > 1U) {
    throw std::runtime_error("collision_behavior_enabled must be zero or one");
  }
  const bool set_collision_behavior = shared.collision_behavior_enabled == 1U;
  if (set_collision_behavior &&
      (!validThresholdPair(shared.lower_torque_thresholds,
                           shared.upper_torque_thresholds, 7) ||
       !validThresholdPair(shared.lower_force_thresholds,
                           shared.upper_force_thresholds, 6))) {
    throw std::runtime_error(
        "collision thresholds must be finite, positive, and lower <= upper");
  }
  for (size_t joint = 0; joint < 7; ++joint) {
    if (shared.maximum_joint_velocities[joint] <= 0.0 ||
        shared.maximum_joint_accelerations[joint] <= 0.0 ||
        shared.maximum_joint_jerks[joint] <= 0.0) {
      throw std::runtime_error("joint trajectory limits must be positive");
    }
  }

  franka::RobotState state{};
  std::array<double, 7> q_command{};
  ErrorCode error_code = ErrorCode::kNone;
  std::string error_message;
  bool control_started = false;
  try {
    franka::Robot robot(robot_ip, franka::RealtimeConfig::kEnforce, kFciLogCapacity);
    if (set_collision_behavior) {
      std::array<double, 7> lower_torque{};
      std::array<double, 7> upper_torque{};
      std::array<double, 6> lower_force{};
      std::array<double, 6> upper_force{};
      std::copy(std::begin(shared.lower_torque_thresholds),
                std::end(shared.lower_torque_thresholds), lower_torque.begin());
      std::copy(std::begin(shared.upper_torque_thresholds),
                std::end(shared.upper_torque_thresholds), upper_torque.begin());
      std::copy(std::begin(shared.lower_force_thresholds),
                std::end(shared.lower_force_thresholds), lower_force.begin());
      std::copy(std::begin(shared.upper_force_thresholds),
                std::end(shared.upper_force_thresholds), upper_force.begin());
      robot.setCollisionBehavior(lower_torque, upper_torque, lower_force, upper_force);
    }
    if (set_joint_impedance) {
      std::array<double, 7> joint_impedance{};
      std::copy(std::begin(shared.joint_impedance), std::end(shared.joint_impedance),
                joint_impedance.begin());
      robot.setJointImpedance(joint_impedance);
    }
    if (set_cartesian_impedance) {
      std::array<double, 6> cartesian_impedance{};
      std::copy(std::begin(shared.cartesian_impedance), std::end(shared.cartesian_impedance),
                cartesian_impedance.begin());
      robot.setCartesianImpedance(cartesian_impedance);
    }
    franka::Model model = robot.loadModel();
    state = robot.readOnce();
    // Joint-position control must begin from the last commanded joint target,
    // not the measured position. libfranka documents this as q_c; for a joint
    // motion generator q_c is represented by q_d in RobotState. Using q can
    // create a velocity/jerk discontinuity at the controller handover.
    q_command = state.q_d;
    validateCurrentState(state, shared);
    copyState(shared, state, q_command, Status::kReady, ErrorCode::kNone, "");

    while (!g_signal_stop.load(std::memory_order_relaxed)) {
      const CommandSnapshot command = readCommand(shared);
      atomicStore(&shared.worker_heartbeat_ns, monotonicNs());
      if (command.command == Command::kStart) {
        break;
      }
      if (command.command == Command::kStop) {
        copyState(shared, state, q_command, Status::kStopped, ErrorCode::kNone, "");
        return 0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (g_signal_stop.load(std::memory_order_relaxed)) {
      copyState(shared, state, q_command, Status::kStopped, ErrorCode::kNone, "signal stop");
      return 0;
    }

    std::array<double, 7> q_dls_target = q_command;
    std::array<double, 7> dq_reference{};
    std::array<double, 7> upper_joint_velocity{};
    std::array<double, 7> lower_joint_velocity{};
    std::array<double, 7> maximum_joint_acceleration{};
    std::array<double, 7> maximum_joint_jerk{};
    for (size_t joint = 0; joint < 7; ++joint) {
      upper_joint_velocity[joint] = shared.maximum_joint_velocities[joint];
      lower_joint_velocity[joint] = -shared.maximum_joint_velocities[joint];
      maximum_joint_acceleration[joint] = shared.maximum_joint_accelerations[joint];
      maximum_joint_jerk[joint] = shared.maximum_joint_jerks[joint];
    }
    std::array<double, 16> target_pose = state.O_T_EE;
    uint64_t latched_generation = 0;
    uint64_t dls_ticks_on_generation = 0;
    uint64_t control_cycles = 0;
    uint64_t ik_ticks = 0;
    double ik_accumulator_s = 0.0;
    bool stop_requested = false;
    uint64_t stop_settled_cycles = 0;
    double stop_elapsed_s = 0.0;
    CommandSnapshot last_command = readCommand(shared);
    copyState(shared, state, q_command, Status::kRunning, ErrorCode::kNone, "");

    control_started = true;
    robot.control(
        [&](const franka::RobotState& callback_state,
            franka::Duration callback_period) -> franka::JointPositions {
      state = callback_state;
      const double raw_period_s = callback_period.toSec();
      const double period_s = std::max(1e-6, raw_period_s);
      const uint64_t now_ns = monotonicNs();
      atomicStore(&shared.worker_heartbeat_ns, now_ns);
      ++control_cycles;
      shared.control_cycle_count = control_cycles;

      CommandSnapshot candidate_command;
      if (tryReadCommand(shared, &candidate_command)) {
        last_command = candidate_command;
      }
      const CommandSnapshot& command = last_command;
      if (command.command == Command::kStop || g_signal_stop.load(std::memory_order_relaxed)) {
        stop_requested = true;
      }
      if (stop_requested) {
        // Use the same bounded-jerk velocity tracker as the active control law.
        // limitRate(q_target=q_d) rings while crossing zero on real hardware and
        // produced the velocity/acceleration discontinuities captured by the
        // ABI-7 FCI log.
        q_dls_target = state.q_d;
        dq_reference.fill(0.0);
        q_command = trackVelocityReference(
            dq_reference, state.q_d, state.dq_d, state.ddq_d,
            shared.maximum_joint_velocities, maximum_joint_acceleration,
            maximum_joint_jerk);
        stop_elapsed_s += period_s;
        if (isZeroVelocityCommand(q_command, state.q_d, state.dq_d, state.ddq_d) &&
            maximumAbsolute(state.dq) <= kStopVelocityThresholdRadS) {
          ++stop_settled_cycles;
        } else {
          stop_settled_cycles = 0;
        }
        const bool settled = stop_settled_cycles >= kStopSettledCycles;
        copyState(shared, state, q_command, settled ? Status::kStopped : Status::kRunning,
                  ErrorCode::kNone, "");
        if (settled) {
          return franka::MotionFinished(franka::JointPositions(q_command));
        }
        if (stop_elapsed_s > kStopTimeoutS) {
          error_code = ErrorCode::kControl;
          throw std::runtime_error("stop trajectory did not settle within two seconds");
        }
        return franka::JointPositions(q_command);
      }
      if (period_s > shared.control_watchdog_s) {
        error_code = ErrorCode::kControl;
        throw std::runtime_error("FCI control period exceeded watchdog");
      }
      // The parent heartbeat is emitted by a non-realtime Python supervision
      // thread.  It must not be stricter than the policy watchdog: a normal
      // Python scheduler stall could otherwise abort a healthy 1 kHz FCI
      // callback before the policy watchdog has declared the policy stale.
      // Keep control_watchdog_s above as the strict callback-period guard.
      const double parent_watchdog_s =
          std::max(shared.control_watchdog_s, shared.policy_watchdog_s);
      const uint64_t parent_watchdog_ns =
          static_cast<uint64_t>(parent_watchdog_s * 1e9);
      if (now_ns - command.parent_heartbeat_ns > parent_watchdog_ns) {
        error_code = ErrorCode::kParentWatchdog;
        throw std::runtime_error("parent control heartbeat exceeded watchdog");
      }
      if (!command.streaming_check) {
        const uint64_t policy_watchdog_ns =
            static_cast<uint64_t>(shared.policy_watchdog_s * 1e9);
        if (now_ns - command.policy_heartbeat_ns > policy_watchdog_ns) {
          error_code = ErrorCode::kPolicyWatchdog;
          throw std::runtime_error("policy heartbeat exceeded watchdog");
        }
      }
      if (static_cast<bool>(state.current_errors)) {
        error_code = ErrorCode::kRobotError;
        throw std::runtime_error(activeRobotErrorMessage(state));
      }
      const bool hold_only = command.streaming_check;
      try {
        validateCurrentState(state, shared);
      } catch (...) {
        error_code = ErrorCode::kInvalidState;
        throw;
      }

      // Do not create a DLS target until the parent has explicitly supplied an
      // action. libfranka's control loop owns all realtime packet-loss and
      // motion-generator error handling; success rate is diagnostic data, not
      // a per-cycle policy-session watchdog.
      if (hold_only) {
        q_command = state.q_d;
        q_dls_target = state.q_d;
        dq_reference.fill(0.0);
      }

      ik_accumulator_s += period_s;
      if (ik_accumulator_s + 1e-9 >= 1.0 / 60.0) {
        ik_accumulator_s -= 1.0 / 60.0;
        if (!hold_only &&
            shouldLatchAction(command.action_generation, latched_generation,
                              dls_ticks_on_generation)) {
          if (!finiteArray(command.delta_xyz.data(), command.delta_xyz.size()) ||
              !finiteArray(command.target_pose.data(), command.target_pose.size())) {
            error_code = ErrorCode::kInvalidState;
            throw std::runtime_error("action target contains NaN or Inf");
          }
          target_pose = command.target_pose;
          if (!insideWorkspace(toolTcpTranslation(target_pose, shared), shared)) {
            error_code = ErrorCode::kWorkspace;
            throw std::runtime_error("latched physical tool TCP target is outside configured workspace");
          }
          latched_generation = command.action_generation;
          dls_ticks_on_generation = 0;
        }
        if (shouldRunDls(hold_only, latched_generation, dls_ticks_on_generation)) {
          try {
            if (control_law == ControlLaw::kSimActuatorVelocity) {
              // The reference limit is wide enough not to bind at the trained
              // action scale, so the DLS magnitude survives and the Cartesian
              // direction is preserved. The proportional part of the law lives
              // entirely in the signed velocity reference.
              const auto delta = dlsDelta(model, state, target_pose, shared.dls_lambda);
              dq_reference = simActuatorVelocityReference(
                  delta, shared.reference_velocity_gain,
                  shared.maximum_ik_reference_delta_rad, shared.maximum_joint_velocities);
              q_dls_target = clampToJointLimits(state.q, delta,
                                                shared.maximum_ik_reference_delta_rad,
                                                shared.joint_limit_margin_rad);
            } else {
              q_dls_target = dlsTarget(model, state, target_pose, shared.dls_lambda,
                                       shared.maximum_joint_target_delta_rad,
                                       shared.joint_limit_margin_rad);
            }
          } catch (...) {
            error_code = ErrorCode::kIk;
            throw;
          }
        } else if (!hold_only) {
          // A Student action represents exactly two 60 Hz simulation substeps.
          // If the next 30 Hz result is late, smoothly brake the signed velocity
          // reference to zero instead of chasing the stale TCP target.
          q_dls_target = state.q_d;
          dq_reference.fill(0.0);
        }
        appendTrace(shared, ik_ticks, latched_generation, state, q_dls_target);
        ++ik_ticks;
        ++dls_ticks_on_generation;
        shared.ik_tick_count = ik_ticks;
        copyState(shared, state, q_dls_target, Status::kRunning, ErrorCode::kNone, "",
                  latched_generation,
                  latched_generation == 0 ? 0 : dls_ticks_on_generation);
      }

      // The 60 Hz loop owns the target/reference; the FCI callback emits a
      // smooth 1 kHz trajectory under the commissioned velocity, acceleration,
      // and jerk envelope.
      if (control_law == ControlLaw::kSimActuatorVelocity) {
        q_command = trackVelocityReference(
            dq_reference, state.q_d, state.dq_d, state.ddq_d,
            shared.maximum_joint_velocities, maximum_joint_acceleration,
            maximum_joint_jerk);
      } else {
        q_command = franka::limitRate(
            upper_joint_velocity, lower_joint_velocity, maximum_joint_acceleration,
            maximum_joint_jerk, q_dls_target, state.q_d, state.dq_d, state.ddq_d);
      }
      return franka::JointPositions(q_command);
    }, impedance_mode == ImpedanceMode::kCartesian
           ? franka::ControllerMode::kCartesianImpedance
           : franka::ControllerMode::kJointImpedance,
       false, franka::kMaxCutoffFrequency);
    copyState(shared, state, q_command, Status::kStopped, ErrorCode::kNone, "");
    return 0;
  } catch (const franka::ControlException& exception) {
    copyFciExceptionLog(shared, exception.log);
    if (!exception.log.empty()) {
      state = exception.log.back().state;
    }
    error_message = exception.what();
    if (error_code == ErrorCode::kNone) {
      error_code = control_started ? ErrorCode::kControl : ErrorCode::kConnection;
    }
    copyState(shared, state, q_command, Status::kAborted, error_code, error_message);
    return 1;
  } catch (const std::exception& exception) {
    error_message = exception.what();
    if (error_code == ErrorCode::kNone) {
      error_code = control_started ? ErrorCode::kControl : ErrorCode::kConnection;
    }
    copyState(shared, state, q_command, Status::kAborted, error_code, error_message);
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "franka-server9-streaming-worker 0.18.0 abi-" << kAbiVersion << "\n";
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--layout") {
    std::cout << "{\"shared_size\":" << sizeof(SharedData)
              << ",\"trace_size\":" << sizeof(TraceEntry)
              << ",\"fci_log_size\":" << sizeof(FciLogEntry)
              << ",\"command_seq\":" << offsetof(SharedData, command_seq)
              << ",\"state_seq\":" << offsetof(SharedData, state_seq)
              << ",\"trace\":" << offsetof(SharedData, trace)
              << ",\"fci_log\":" << offsetof(SharedData, fci_log)
              << ",\"collision_behavior_enabled\":"
              << offsetof(SharedData, collision_behavior_enabled) << "}\n";
    return 0;
  }
  if (argc == 2 && std::string(argv[1]) == "--self-test") {
    double matrix[6][6]{};
    std::array<double, 6> rhs{1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    for (size_t index = 0; index < 6; ++index) {
      matrix[index][index] = 1.0 + 0.01 * 0.01;
    }
    const auto solved = solve6x6(matrix, rhs);
    for (size_t index = 0; index < 6; ++index) {
      if (std::abs(solved[index] - rhs[index] / 1.0001) > 1e-12) {
        std::cerr << "DLS linear solve self-test failed\n";
        return 1;
      }
    }
    if (sizeof(SharedData) != 444024 || sizeof(TraceEntry) != 144 ||
        sizeof(FciLogEntry) != 288 || offsetof(SharedData, fci_log) != 148896 ||
        offsetof(SharedData, control_law) != 600 || offsetof(SharedData, state_seq) != 624 ||
        offsetof(SharedData, trace) != 1432 ||
        offsetof(SharedData, collision_behavior_enabled) != 443808) {
      std::cerr << "shared-memory layout self-test failed\n";
      return 1;
    }
    // The signed reference must scale with the DLS delta. The legacy law's
    // constant envelope is what collapses every magnitude onto one speed.
    const std::array<double, 7> velocity_envelope{2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
    const auto small = simActuatorVelocityReference({0.01, 0, 0, 0, 0, 0, 0}, 5.0, 0.25,
                                                    velocity_envelope.data());
    const auto large = simActuatorVelocityReference({-0.05, 0, 0, 0, 0, 0, 0}, 5.0, 0.25,
                                                    velocity_envelope.data());
    if (std::abs(small[0] - 0.05) > 1e-12 || std::abs(large[0] + 0.25) > 1e-12) {
      std::cerr << "sim actuator velocity-reference gain self-test failed\n";
      return 1;
    }
    const auto reference_limited = simActuatorVelocityReference(
        {0.4, 0, 0, 0, 0, 0, 0}, 5.0, 0.25, velocity_envelope.data());
    if (std::abs(reference_limited[0] - 1.25) > 1e-12) {
      std::cerr << "sim actuator reference limiting self-test failed\n";
      return 1;
    }
    const auto envelope_clamped = simActuatorVelocityReference(
        {-0.25, 0, 0, 0, 0, 0, 0}, 20.0, 0.25, velocity_envelope.data());
    if (std::abs(envelope_clamped[0] + 2.0) > 1e-12) {
      std::cerr << "sim actuator velocity envelope self-test failed\n";
      return 1;
    }

    // Exercise the exact class of transitions seen on hardware: a reference
    // shrinks on the second DLS tick, becomes stale, and later changes sign.
    // Every emitted position must remain inside the configured velocity,
    // acceleration, and jerk envelope.
    const std::array<double, 7> accel{7, 7, 7, 7, 10, 10, 10};
    const std::array<double, 7> jerk{1500, 1500, 1500, 1500, 2000, 2000, 2000};
    const std::array<double, 7> max_velocity{1, 1, 1, 1, 1.5, 1.5, 1.5};
    std::array<double, 7> q_d{}, dq_d{}, ddq_d{}, reference{};
    const std::array<std::pair<double, int>, 7> segments{{
        {0.0200316, 17}, {0.0190542, 17}, {0.0, 100}, {-0.068, 100},
        {0.25, 100}, {-0.25, 150}, {0.0, 200},
    }};
    for (const auto& segment : segments) {
      reference.fill(segment.first);
      for (int cycle = 0; cycle < segment.second; ++cycle) {
        const auto command = trackVelocityReference(
            reference, q_d, dq_d, ddq_d, max_velocity.data(), accel, jerk);
        for (size_t joint = 0; joint < 7; ++joint) {
          const double velocity = (command[joint] - q_d[joint]) / kControlPeriodS;
          const double acceleration = (velocity - dq_d[joint]) / kControlPeriodS;
          const double current_jerk = (acceleration - ddq_d[joint]) / kControlPeriodS;
          if (!std::isfinite(velocity) ||
              std::abs(velocity) > max_velocity[joint] + 1e-9 ||
              std::abs(acceleration) > accel[joint] + 1e-6 ||
              std::abs(current_jerk) > jerk[joint] + 1e-3) {
            std::cerr << "velocity-reference tracking self-test failed: reference="
                      << segment.first << " velocity=" << velocity
                      << " acceleration=" << acceleration << " jerk=" << current_jerk << "\n";
            return 1;
          }
          q_d[joint] = command[joint];
          dq_d[joint] = velocity;
          ddq_d[joint] = acceleration;
        }
      }
    }
    if (maximumAbsolute(dq_d) > 1e-6 || maximumAbsolute(ddq_d) > 1e-4) {
      std::cerr << "velocity-reference tracking did not settle after stale action\n";
      return 1;
    }
    // The production stop path uses the same tracker with a zero reference.
    // Scan both signs and the entire commissioned range, requiring every
    // emitted derivative to stay bounded and the command to land exactly at
    // zero before MotionFinished can be signalled.
    for (const double entry_velocity :
         {-1.0, -0.68, -0.25, -0.068, -0.02, 0.02, 0.068, 0.25, 0.68, 1.0}) {
      std::array<double, 7> q_d{}, dq_d{}, ddq_d{}, zero_reference{};
      for (size_t joint = 0; joint < 7; ++joint) {
        dq_d[joint] = entry_velocity;
      }
      uint64_t settled_cycles = 0;
      bool finished = false;
      for (int cycle = 0; cycle < static_cast<int>(kStopTimeoutS / 1e-3); ++cycle) {
        const auto command = trackVelocityReference(
            zero_reference, q_d, dq_d, ddq_d, max_velocity.data(), accel, jerk);
        settled_cycles =
            isZeroVelocityCommand(command, q_d, dq_d, ddq_d) ? settled_cycles + 1 : 0;
        for (size_t joint = 0; joint < 7; ++joint) {
          const double velocity = (command[joint] - q_d[joint]) / kControlPeriodS;
          const double acceleration = (velocity - dq_d[joint]) / kControlPeriodS;
          const double current_jerk = (acceleration - ddq_d[joint]) / kControlPeriodS;
          if (std::abs(velocity) > max_velocity[joint] + 1e-9 ||
              std::abs(acceleration) > accel[joint] + 1e-6 ||
              std::abs(current_jerk) > jerk[joint] + 1e-3) {
            std::cerr << "bounded stop self-test violated trajectory envelope at entry "
                      << entry_velocity << ": velocity=" << velocity
                      << " acceleration=" << acceleration << " jerk=" << current_jerk << "\n";
            return 1;
          }
          ddq_d[joint] = acceleration;
          dq_d[joint] = velocity;
          q_d[joint] = command[joint];
        }
        if (settled_cycles >= kStopSettledCycles) {
          finished = true;
          break;
        }
      }
      if (!finished) {
        std::cerr << "stop settle self-test did not settle within " << kStopTimeoutS
                  << " s from entry " << entry_velocity << "\n";
        return 1;
      }
    }
    if (!shouldLatchAction(1, 0, 0) || shouldLatchAction(2, 1, 0) ||
        shouldLatchAction(2, 1, 1) || !shouldLatchAction(2, 1, 2)) {
      std::cerr << "two-DLS-ticks-per-action self-test failed\n";
      return 1;
    }
    if (shouldRunDls(true, 1, 0) || shouldRunDls(false, 0, 0) ||
        !shouldRunDls(false, 1, 0) || !shouldRunDls(false, 1, 1) ||
        shouldRunDls(false, 1, 2)) {
      std::cerr << "two-tick DLS action lifetime self-test failed\n";
      return 1;
    }
    if (std::abs(maximumAbsolute({-0.01, 0.02, -0.03, 0.04, -0.05, 0.06, -0.07}) -
                 0.07) > 1e-12) {
      std::cerr << "joint velocity settle self-test failed\n";
      return 1;
    }
    std::cout << "server9 worker self-test: PASS\n";
    return 0;
  }
  if (argc != 3) {
    std::cerr << "usage: franka_server9_streaming_worker ROBOT_IP SHARED_MEMORY_PATH\n";
    return 2;
  }
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);
  try {
    return run(argv[1], argv[2]);
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return 2;
  }
}
