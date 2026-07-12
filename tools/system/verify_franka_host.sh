#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "== Kernel =="
uname -a
if [[ -f /sys/kernel/realtime ]]; then
  echo -n "/sys/kernel/realtime: "
  cat /sys/kernel/realtime
else
  echo "/sys/kernel/realtime: missing"
fi

echo
echo "== CPU =="
if command -v powerprofilesctl >/dev/null 2>&1; then
  echo -n "powerprofilesctl: "
  powerprofilesctl get || true
fi
echo -n "scaling_governor: "
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown
echo -n "energy_performance_preference: "
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo unknown

echo
echo "== Realtime Limits =="
echo -n "groups: "
groups
echo -n "ulimit -r: "
ulimit -r
echo -n "ulimit -e: "
ulimit -e

echo
echo "== Network =="
ip route get 172.16.0.2 || true
ip -br addr show enp3s0 || true

echo
echo "== Franka Read Test =="
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/robot/read_franka_state.py" --ip 172.16.0.2 --count 1 --realtime ignore || true
else
  echo "Python venv not found at ${ROOT_DIR}/.venv"
fi
