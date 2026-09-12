#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <franka/gripper.h>
#include <franka/gripper_state.h>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kDefaultTravelM = 0.006;
constexpr double kDefaultSpeedMps = 0.03;
constexpr double kDefaultStopAfterS = 0.05;
constexpr double kMaximumTravelM = 0.010;
constexpr double kMaximumSpeedMps = 0.05;
constexpr double kMaximumStopAfterS = 0.20;
constexpr int kMaximumRepeats = 5;

struct Options {
  bool run_hardware{false};
  bool self_test{false};
  bool help{false};
  std::string ip{"172.16.0.2"};
  std::string mode{"compare"};
  std::string direction{"open"};
  double travel_m{kDefaultTravelM};
  double speed_mps{kDefaultSpeedMps};
  double stop_after_s{kDefaultStopAfterS};
  double natural_timeout_s{5.0};
  int repeats{1};
  std::optional<std::filesystem::path> output;
};

struct StateSample {
  double width_m{};
  double max_width_m{};
  bool is_grasped{};
  double read_ms{};
};

struct Trial {
  std::string kind;
  std::string direction;
  StateSample before;
  StateSample after;
  double target_width_m{};
  double commanded_travel_m{};
  double speed_mps{};
  double stop_after_ms{};
  double stop_call_ms{};
  double move_call_ms{};
  double nominal_motion_ms{};
  bool move_result{};
  bool stop_result{};
};

double elapsedMs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

double parseDouble(const std::string& flag, const std::string& value) {
  size_t parsed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " requires a number");
  }
  if (parsed != value.size() || !std::isfinite(result)) {
    throw std::invalid_argument(flag + " requires a finite number");
  }
  return result;
}

int parseInt(const std::string& flag, const std::string& value) {
  size_t parsed = 0;
  int result = 0;
  try {
    result = std::stoi(value, &parsed);
  } catch (const std::exception&) {
    throw std::invalid_argument(flag + " requires an integer");
  }
  if (parsed != value.size()) {
    throw std::invalid_argument(flag + " requires an integer");
  }
  return result;
}

std::string requireValue(int argc, char** argv, int& index, const std::string& flag) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(flag + " requires a value");
  }
  return argv[++index];
}

Options parseOptions(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--run-hardware") {
      options.run_hardware = true;
    } else if (flag == "--self-test") {
      options.self_test = true;
    } else if (flag == "--help" || flag == "-h") {
      options.help = true;
    } else if (flag == "--ip") {
      options.ip = requireValue(argc, argv, index, flag);
    } else if (flag == "--mode") {
      options.mode = requireValue(argc, argv, index, flag);
    } else if (flag == "--direction") {
      options.direction = requireValue(argc, argv, index, flag);
    } else if (flag == "--travel-mm") {
      options.travel_m = parseDouble(flag, requireValue(argc, argv, index, flag)) / 1000.0;
    } else if (flag == "--speed") {
      options.speed_mps = parseDouble(flag, requireValue(argc, argv, index, flag));
    } else if (flag == "--stop-after-ms") {
      options.stop_after_s =
          parseDouble(flag, requireValue(argc, argv, index, flag)) / 1000.0;
    } else if (flag == "--natural-timeout-s") {
      options.natural_timeout_s = parseDouble(flag, requireValue(argc, argv, index, flag));
    } else if (flag == "--repeats") {
      options.repeats = parseInt(flag, requireValue(argc, argv, index, flag));
    } else if (flag == "--output") {
      options.output = std::filesystem::path(requireValue(argc, argv, index, flag));
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }
  return options;
}

void validateOptions(const Options& options) {
  if (options.mode != "stop" && options.mode != "natural" && options.mode != "compare") {
    throw std::invalid_argument("--mode must be stop, natural, or compare");
  }
  if (options.direction != "open" && options.direction != "close") {
    throw std::invalid_argument("--direction must be open or close");
  }
  if (!std::isfinite(options.travel_m) || options.travel_m < 0.001 ||
      options.travel_m > kMaximumTravelM) {
    throw std::invalid_argument("--travel-mm must be in [1, 10]");
  }
  if (!std::isfinite(options.speed_mps) || options.speed_mps < 0.005 ||
      options.speed_mps > kMaximumSpeedMps) {
    throw std::invalid_argument("--speed must be in [0.005, 0.05]");
  }
  if (!std::isfinite(options.stop_after_s) || options.stop_after_s < 0.010 ||
      options.stop_after_s > kMaximumStopAfterS) {
    throw std::invalid_argument("--stop-after-ms must be in [10, 200]");
  }
  if ((options.mode == "stop" || options.mode == "compare") &&
      options.stop_after_s >= 0.8 * options.travel_m / options.speed_mps) {
    throw std::invalid_argument(
        "--stop-after-ms must be below 80% of the nominal move duration");
  }
  if (!std::isfinite(options.natural_timeout_s) || options.natural_timeout_s < 0.5 ||
      options.natural_timeout_s > 10.0) {
    throw std::invalid_argument("--natural-timeout-s must be in [0.5, 10.0]");
  }
  if (options.repeats < 1 || options.repeats > kMaximumRepeats) {
    throw std::invalid_argument("--repeats must be in [1, 5]");
  }
}

double boundedTarget(double width_m,
                     double max_width_m,
                     const std::string& direction,
                     double travel_m) {
  const double target = width_m + (direction == "open" ? travel_m : -travel_m);
  if (target < 0.0 || target > max_width_m) {
    std::ostringstream message;
    message << "cannot move " << direction << " by " << travel_m * 1000.0
            << " mm from " << width_m * 1000.0 << " mm; reported range is [0, "
            << max_width_m * 1000.0 << "] mm";
    throw std::invalid_argument(message.str());
  }
  return target;
}

StateSample readState(const franka::Gripper& gripper) {
  const auto started = Clock::now();
  const franka::GripperState state = gripper.readOnce();
  const auto completed = Clock::now();
  StateSample sample{state.width, state.max_width, state.is_grasped,
                     elapsedMs(started, completed)};
  if (!std::isfinite(sample.width_m) || !std::isfinite(sample.max_width_m) ||
      sample.max_width_m <= 0.0 || sample.width_m < 0.0 ||
      sample.width_m > sample.max_width_m) {
    throw std::runtime_error("Hand reported invalid state; homing may be required");
  }
  return sample;
}

struct MoveCall {
  std::thread thread;
  std::atomic<bool> entered{false};
  std::atomic<bool> done{false};
  bool result{false};
  double call_ms{};
  std::exception_ptr error;
};

void startMove(MoveCall& call,
               const franka::Gripper& gripper,
               double target_width_m,
               double speed_mps) {
  call.thread = std::thread([&call, &gripper, target_width_m, speed_mps]() {
    call.entered.store(true, std::memory_order_release);
    const auto started = Clock::now();
    try {
      call.result = gripper.move(target_width_m, speed_mps);
    } catch (...) {
      call.error = std::current_exception();
    }
    call.call_ms = elapsedMs(started, Clock::now());
    call.done.store(true, std::memory_order_release);
  });
  while (!call.entered.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

void joinMove(MoveCall& call) {
  if (call.thread.joinable()) {
    call.thread.join();
  }
  if (call.error) {
    std::rethrow_exception(call.error);
  }
}

Trial runStopTrial(const franka::Gripper& gripper,
                   const std::string& direction,
                   double travel_m,
                   double speed_mps,
                   double stop_after_s) {
  Trial trial;
  trial.kind = "stop";
  trial.direction = direction;
  trial.before = readState(gripper);
  trial.target_width_m =
      boundedTarget(trial.before.width_m, trial.before.max_width_m, direction, travel_m);
  trial.commanded_travel_m = std::abs(trial.target_width_m - trial.before.width_m);
  trial.speed_mps = speed_mps;
  trial.stop_after_ms = stop_after_s * 1000.0;
  trial.nominal_motion_ms = trial.commanded_travel_m / speed_mps * 1000.0;

  MoveCall move;
  startMove(move, gripper, trial.target_width_m, speed_mps);
  std::this_thread::sleep_for(std::chrono::duration<double>(stop_after_s));
  if (move.done.load(std::memory_order_acquire)) {
    joinMove(move);
    throw std::runtime_error(
        "bounded move completed before stop; reduce --stop-after-ms or increase travel");
  }

  std::exception_ptr stop_error;
  const auto stop_started = Clock::now();
  try {
    trial.stop_result = gripper.stop();
  } catch (...) {
    stop_error = std::current_exception();
  }
  trial.stop_call_ms = elapsedMs(stop_started, Clock::now());
  joinMove(move);
  if (stop_error) {
    std::rethrow_exception(stop_error);
  }
  trial.move_result = move.result;
  if (!trial.stop_result) {
    throw std::runtime_error("libfranka Gripper::stop() reported failure");
  }
  trial.move_call_ms = move.call_ms;
  trial.after = readState(gripper);
  return trial;
}

Trial runNaturalTrial(const franka::Gripper& gripper,
                      double target_width_m,
                      double speed_mps,
                      double timeout_s) {
  Trial trial;
  trial.kind = "natural";
  trial.before = readState(gripper);
  if (target_width_m < 0.0 || target_width_m > trial.before.max_width_m) {
    throw std::invalid_argument("natural target is outside the reported Hand range");
  }
  trial.target_width_m = target_width_m;
  trial.direction = target_width_m > trial.before.width_m ? "open" : "close";
  trial.commanded_travel_m = std::abs(target_width_m - trial.before.width_m);
  if (trial.commanded_travel_m < 1e-6 ||
      trial.commanded_travel_m > kMaximumTravelM + 1e-9) {
    throw std::invalid_argument("natural move must be within (0, 10] mm");
  }
  trial.speed_mps = speed_mps;
  trial.nominal_motion_ms = trial.commanded_travel_m / speed_mps * 1000.0;

  MoveCall move;
  startMove(move, gripper, target_width_m, speed_mps);
  const auto deadline = Clock::now() + std::chrono::duration<double>(timeout_s);
  bool timeout = false;
  while (!move.done.load(std::memory_order_acquire)) {
    if (Clock::now() >= deadline) {
      timeout = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  if (timeout) {
    try {
      gripper.stop();
    } catch (...) {
      joinMove(move);
      throw;
    }
  }
  joinMove(move);
  if (timeout) {
    throw std::runtime_error("finite Hand move exceeded --natural-timeout-s");
  }
  trial.move_result = move.result;
  trial.move_call_ms = move.call_ms;
  if (!trial.move_result) {
    throw std::runtime_error("libfranka Gripper::move() reported failure");
  }
  trial.after = readState(gripper);
  return trial;
}

std::string jsonString(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      output << '\\';
    }
    output << character;
  }
  output << '"';
  return output.str();
}

void writeState(std::ostream& output, const StateSample& state) {
  output << "{\"width_m\":" << state.width_m << ",\"max_width_m\":"
         << state.max_width_m << ",\"is_grasped\":"
         << (state.is_grasped ? "true" : "false") << ",\"read_ms\":"
         << state.read_ms << '}';
}

std::filesystem::path defaultOutputPath() {
  const std::time_t now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  std::ostringstream name;
  name << std::put_time(&local, "%Y%m%d_%H%M%S") << "_libfranka_0_17.json";
  return std::filesystem::path("runs") / "gripper_latency" / name.str();
}

void writeReport(const std::filesystem::path& path,
                 const Options& options,
                 const StateSample& initial,
                 const std::vector<Trial>& trials) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("cannot open report path: " + path.string());
  }
  output << std::setprecision(12);
  output << "{\n  \"schema_version\":1,\n  \"backend\":\"libfranka_cpp\",\n"
         << "  \"libfranka_version\":\"0.17.0\",\n  \"robot_ip\":"
         << jsonString(options.ip) << ",\n  \"arm_connection_or_motion\":false,\n"
         << "  \"parameters\":{\"mode\":" << jsonString(options.mode)
         << ",\"direction\":" << jsonString(options.direction)
         << ",\"travel_m\":" << options.travel_m << ",\"speed_m_s\":"
         << options.speed_mps << ",\"stop_after_s\":" << options.stop_after_s
         << ",\"repeats\":" << options.repeats << "},\n  \"initial_state\":";
  writeState(output, initial);
  output << ",\n  \"trials\":[\n";
  for (size_t index = 0; index < trials.size(); ++index) {
    const Trial& trial = trials[index];
    output << "    {\"kind\":" << jsonString(trial.kind)
           << ",\"direction\":" << jsonString(trial.direction) << ",\"before\":";
    writeState(output, trial.before);
    output << ",\"target_width_m\":" << trial.target_width_m
           << ",\"commanded_travel_m\":" << trial.commanded_travel_m
           << ",\"speed_m_s\":" << trial.speed_mps
           << ",\"nominal_motion_ms\":" << trial.nominal_motion_ms
           << ",\"move_call_ms\":" << trial.move_call_ms
           << ",\"move_result\":" << (trial.move_result ? "true" : "false");
    if (trial.kind == "stop") {
      output << ",\"stop_after_ms\":" << trial.stop_after_ms
             << ",\"stop_call_ms\":" << trial.stop_call_ms
             << ",\"stop_result\":" << (trial.stop_result ? "true" : "false");
    } else {
      output << ",\"stop_after_ms\":null,\"stop_call_ms\":null,\"stop_result\":null";
    }
    output << ",\"after\":";
    writeState(output, trial.after);
    output << ",\"measured_travel_m\":"
           << std::abs(trial.after.width_m - trial.before.width_m)
           << ",\"final_error_m\":" << trial.after.width_m - trial.target_width_m
           << '}';
    if (index + 1 != trials.size()) {
      output << ',';
    }
    output << '\n';
  }
  output << "  ]\n}\n";
}

void printHelp() {
  std::cout
      << "Usage: franka_gripper_latency [options]\n\n"
      << "Defaults to a dry run; it never constructs franka::Robot.\n\n"
      << "  --run-hardware              Connect to Hand and execute bounded motion\n"
      << "  --ip ADDRESS                Default: 172.16.0.2\n"
      << "  --mode stop|natural|compare Default: compare\n"
      << "  --direction open|close      Default: open\n"
      << "  --travel-mm MM              Range: 1..10; default: 6\n"
      << "  --speed M_S                 Range: 0.005..0.05; default: 0.03\n"
      << "  --stop-after-ms MS          Range: 10..200; default: 50\n"
      << "  --repeats N                 Range: 1..5; default: 1\n"
      << "  --natural-timeout-s S       Range: 0.5..10; default: 5\n"
      << "  --output PATH               JSON report path\n"
      << "  --self-test                 Offline validation only\n";
}

void printPlan(const Options& options) {
  std::cout << "Native Franka Hand latency diagnostic\n"
            << "  backend: libfranka C++ 0.17.0 (same library version as franky 1.1.3)\n"
            << "  arm connection/motion: disabled\n"
            << "  mode: " << options.mode << "\n"
            << "  first direction: " << options.direction << "\n"
            << "  bounded travel: " << options.travel_m * 1000.0
            << " mm (hard cap 10 mm)\n"
            << "  speed: " << options.speed_mps << " m/s (hard cap 0.05 m/s)\n"
            << "  stop request delay: " << options.stop_after_s * 1000.0 << " ms\n"
            << "  repeats: " << options.repeats << '\n';
}

void runSelfTest() {
  Options defaults;
  validateOptions(defaults);
  if (std::abs(boundedTarget(0.04, 0.08, "open", 0.006) - 0.046) > 1e-12 ||
      std::abs(boundedTarget(0.04, 0.08, "close", 0.006) - 0.034) > 1e-12) {
    throw std::runtime_error("boundedTarget self-test failed");
  }
  bool rejected = false;
  try {
    boundedTarget(0.078, 0.08, "open", 0.006);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected) {
    throw std::runtime_error("physical-range self-test failed");
  }
  std::cout << "Self-test passed (no hardware connection).\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    if (options.help) {
      printHelp();
      return 0;
    }
    validateOptions(options);
    if (options.self_test) {
      if (options.run_hardware) {
        throw std::invalid_argument("--self-test cannot be combined with --run-hardware");
      }
      runSelfTest();
      return 0;
    }
    printPlan(options);
    if (!options.run_hardware) {
      std::cout << "Dry run only: no Hand connection and no movement were performed.\n"
                << "Add --run-hardware only after removing objects and clearing the fingers.\n";
      return 0;
    }

    std::cout << "Remove all objects, keep hands clear, and type y/yes to run this "
                 "gripper-only test: ";
    std::string answer;
    std::getline(std::cin, answer);
    if (answer != "y" && answer != "yes") {
      std::cout << "Cancelled; no Hand connection or movement was performed.\n";
      return 1;
    }

    const franka::Gripper gripper(options.ip);
    const StateSample initial = readState(gripper);
    if (initial.is_grasped) {
      throw std::runtime_error("Hand reports is_grasped=true; remove the object first");
    }

    std::vector<Trial> trials;
    for (int repeat = 0; repeat < options.repeats; ++repeat) {
      std::cout << "Running trial " << repeat + 1 << '/' << options.repeats << "...\n";
      const StateSample start = readState(gripper);
      const double first_target = boundedTarget(start.width_m, start.max_width_m,
                                                options.direction, options.travel_m);
      if (options.mode == "stop" || options.mode == "compare") {
        Trial trial = runStopTrial(gripper, options.direction, options.travel_m,
                                   options.speed_mps, options.stop_after_s);
        std::cout << "  stop(): " << trial.stop_call_ms << " ms; blocking move(): "
                  << trial.move_call_ms << " ms; width " << trial.before.width_m * 1000.0
                  << " -> " << trial.after.width_m * 1000.0 << " mm\n";
        trials.push_back(trial);
      }
      if (options.mode == "natural") {
        trials.push_back(runNaturalTrial(gripper, first_target, options.speed_mps,
                                         options.natural_timeout_s));
      } else if (options.mode == "compare") {
        const StateSample current = readState(gripper);
        if (std::abs(current.width_m - start.width_m) >= 1e-6) {
          Trial trial = runNaturalTrial(gripper, start.width_m, options.speed_mps,
                                        options.natural_timeout_s);
          std::cout << "  natural return move(): " << trial.move_call_ms << " ms for "
                    << trial.commanded_travel_m * 1000.0 << " mm\n";
          trials.push_back(trial);
        } else {
          std::cout << "  natural return skipped: no measurable travel\n";
        }
      }
    }

    const std::filesystem::path output = options.output.value_or(defaultOutputPath());
    writeReport(output, options, initial, trials);
    std::cout << "Saved report to: " << std::filesystem::absolute(output) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native gripper latency test failed: " << error.what() << '\n';
    return 1;
  }
}
