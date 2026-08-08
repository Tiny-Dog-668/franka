#!/usr/bin/env bash
# Measure FCI-level ICMP load from the same CPU affinity and FIFO policy used
# by the native control callback. This test never opens an FCI connection.
set -euo pipefail

usage() {
  echo "Usage: sudo $0 --interface IFACE --cpu CPU --host FCI_IP [--count N]" >&2
  exit 2
}

interface=""
cpu=""
host=""
count=10000
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interface) interface="${2:-}"; shift 2 ;;
    --cpu) cpu="${2:-}"; shift 2 ;;
    --host) host="${2:-}"; shift 2 ;;
    --count) count="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ ${EUID} -eq 0 && -n "${interface}" && -n "${host}" && "${cpu}" =~ ^[0-9]+$ && "${count}" =~ ^[1-9][0-9]*$ ]] || usage
[[ -d "/sys/class/net/${interface}" && -d "/sys/devices/system/cpu/cpu${cpu}" ]] || usage

log_file="$(mktemp /tmp/franka_fci_ping.XXXXXX)"
trap 'rm -f "${log_file}"' EXIT

echo "Testing ${host} via ${interface}, CPU ${cpu}, SCHED_FIFO 80, ${count} packets..."
chrt -f 80 taskset --cpu-list "${cpu}" \
  ping -I "${interface}" -i 0.001 -D -c "${count}" -s 1200 "${host}" > "${log_file}"
tail -5 "${log_file}"

awk '
/bytes from/ {
  for (i = 1; i <= NF; ++i) {
    if ($i ~ /^time=/) { split($i, parts, "="); latency_ms = parts[2] + 0.0 }
  }
  samples++
  if (latency_ms >= 0.5) { over_half_ms++ }
  if (latency_ms >= 1.0) { over_one_ms++ }
}
END {
  printf "samples=%d latency>=0.5ms=%d latency>=1.0ms=%d\n", samples, over_half_ms, over_one_ms
  exit (samples == 0 || over_one_ms != 0)
}' "${log_file}"
