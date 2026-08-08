// Copyright (c) 2026 Franka Robotics GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#include <memory>
#include <tuple>

#include <franka/async_control/async_position_control_handler.hpp>
#include <franka/async_control/target_status.hpp>
#include <franka/exception.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace franka {

auto AsyncPositionControlHandler::readOnce() -> RobotState {
  if (control_status_ == TargetStatus::kAborted || active_robot_control_ == nullptr) {
    throw InvalidOperationException("Control interface is in aborted state.");
  }

  try {
    std::tie(current_robot_state_, std::ignore) = active_robot_control_->readOnce();
    return current_robot_state_;
  } catch (...) {
    control_status_ = TargetStatus::kAborted;
    throw;
  }
}

}  // namespace franka

PYBIND11_MODULE(_native, module) {
  py::enum_<franka::TargetStatus>(module, "TargetStatus")
      .value("kIdle", franka::TargetStatus::kIdle)
      .value("kExecuting", franka::TargetStatus::kExecuting)
      .value("kTargetReached", franka::TargetStatus::kTargetReached)
      .value("kAborted", franka::TargetStatus::kAborted);

  module.def(
      "read_once",
      [](const std::shared_ptr<franka::AsyncPositionControlHandler>& handler) {
        return handler->readOnce();
      },
      py::arg("handler"),
      "Read RobotState from an active AsyncPositionControlHandler.");
}
