#!/usr/bin/env bash
set -euo pipefail

shopt -s nullglob

if command -v powerprofilesctl >/dev/null 2>&1; then
  powerprofilesctl set performance || true
fi

for cpu_dir in /sys/devices/system/cpu/cpu[0-9]*; do
  gov_file="${cpu_dir}/cpufreq/scaling_governor"
  epp_file="${cpu_dir}/cpufreq/energy_performance_preference"

  if [[ -w "${gov_file}" ]]; then
    if grep -qx 'performance' "${cpu_dir}/cpufreq/scaling_available_governors" 2>/dev/null; then
      echo performance >"${gov_file}" || true
    fi
  fi

  if [[ -w "${epp_file}" ]]; then
    echo performance >"${epp_file}" || true
  fi
done

exit 0
