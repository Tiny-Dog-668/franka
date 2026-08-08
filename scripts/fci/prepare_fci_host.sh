#!/usr/bin/env bash
# Prepare a dedicated Ethernet interface for Franka FCI. Run only on the
# workstation directly connected to the robot's Control port.
set -euo pipefail

usage() {
  echo "Usage: sudo $0 --interface IFACE --cpu CPU [--restore]" >&2
  exit 2
}

interface=""
cpu=""
restore=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interface) interface="${2:-}"; shift 2 ;;
    --cpu) cpu="${2:-}"; shift 2 ;;
    --restore) restore=true; shift ;;
    *) usage ;;
  esac
done

[[ ${EUID} -eq 0 && -n "${interface}" && "${cpu}" =~ ^[0-9]+$ ]] || usage
[[ -d "/sys/class/net/${interface}" ]] || { echo "Unknown interface: ${interface}" >&2; exit 1; }
[[ -d "/sys/devices/system/cpu/cpu${cpu}" ]] || { echo "Unknown CPU: ${cpu}" >&2; exit 1; }

if ${restore}; then
  for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    [[ -w "${governor}" ]] && echo ondemand > "${governor}"
  done
  systemctl enable --now irqbalance
  echo "Restored ondemand CPU governor and irqbalance. NIC IRQ affinity remains pinned until reboot."
  exit 0
fi

for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -w "${governor}" ]] && echo performance > "${governor}"
done

# irqbalance may rewrite /proc/irq/*/smp_affinity_list at any time, so stop it
# before assigning every MSI IRQ exposed by the dedicated FCI NIC.
systemctl disable --now irqbalance
for irq_path in /sys/class/net/"${interface}"/device/msi_irqs/*; do
  [[ -e "${irq_path}" ]] || continue
  irq="${irq_path##*/}"
  echo "${cpu}" > "/proc/irq/${irq}/smp_affinity_list"
done

# FCI uses small 1 kHz UDP packets; aggregation adds latency and jitter.
ethtool -K "${interface}" gro off gso off tso off lro off || true
ip link set dev "${interface}" mtu 1500

echo "Prepared ${interface} for FCI on CPU ${cpu}:"
for governor in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  printf '%s=' "${governor%/scaling_governor}"; cat "${governor}"
done
for irq_path in /sys/class/net/"${interface}"/device/msi_irqs/*; do
  [[ -e "${irq_path}" ]] || continue
  irq="${irq_path##*/}"
  printf 'IRQ %s CPU=' "${irq}"; cat "/proc/irq/${irq}/effective_affinity_list"
done
