#!/usr/bin/env bash
set -euo pipefail

ROBOT_IP="${1:-172.16.0.2}"
INTERNET_TEST_IP="${2:-1.1.1.1}"

echo "== Interfaces =="
ip -br addr
echo

echo "== Route to robot (${ROBOT_IP}) =="
robot_route="$(ip route get "${ROBOT_IP}")"
echo "${robot_route}"
echo

echo "== Route to internet test IP (${INTERNET_TEST_IP}) =="
if internet_route="$(ip route get "${INTERNET_TEST_IP}" 2>/dev/null)"; then
  echo "${internet_route}"
else
  echo "No route to ${INTERNET_TEST_IP}"
fi
echo

if command -v nmcli >/dev/null 2>&1; then
  echo "== NetworkManager devices =="
  nmcli device status
  echo
fi

if [[ "${robot_route}" == *" dev enp3s0 "* ]]; then
  echo "Robot route looks good: ${ROBOT_IP} is going via enp3s0."
else
  echo "Warning: ${ROBOT_IP} is not routing via enp3s0."
  echo "Check cable connection and IP settings before running robot motion."
fi

