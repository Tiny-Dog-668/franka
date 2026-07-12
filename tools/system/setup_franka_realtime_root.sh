#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run this script with sudo:"
  echo "  sudo bash $0"
  exit 1
fi

if [[ $# -gt 1 ]]; then
  echo "Usage: sudo bash $0 [username]"
  exit 1
fi

TARGET_USER="${1:-${SUDO_USER:-}}"
if [[ -z "${TARGET_USER}" ]]; then
  echo "Could not determine target username."
  echo "Pass it explicitly, for example:"
  echo "  sudo bash $0 td"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[1/7] Installing realtime kernel and supporting packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ubuntu-realtime cpufrequtils

echo "[2/7] Creating realtime group if needed..."
getent group realtime >/dev/null || groupadd realtime

echo "[3/7] Adding ${TARGET_USER} to realtime group..."
usermod -a -G realtime "${TARGET_USER}"

echo "[4/7] Installing PAM limits for realtime scheduling..."
cat >/etc/security/limits.d/99-franka-realtime.conf <<'EOF'
@realtime soft rtprio 99
@realtime soft priority 99
@realtime soft memlock 102400
@realtime hard rtprio 99
@realtime hard priority 99
@realtime hard memlock 102400
EOF

echo "[5/7] Installing CPU performance helper..."
install -m 0755 "${SCRIPT_DIR}/franka-set-performance.sh" /usr/local/sbin/franka-set-performance.sh

echo "[6/7] Installing systemd unit for performance mode..."
cat >/etc/systemd/system/franka-performance.service <<'EOF'
[Unit]
Description=Set CPU performance mode for Franka FCI
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/franka-set-performance.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable franka-performance.service
systemctl start franka-performance.service || true

echo "[7/7] Applying performance mode immediately..."
/usr/local/sbin/franka-set-performance.sh || true

echo
echo "Setup finished."
echo "Important next steps:"
echo "  1. Reboot the machine."
echo "  2. In GRUB, boot the new realtime kernel."
echo "  3. Log out and log back in once after reboot."
echo "  4. Run: ${ROOT_DIR}/tools/system/verify_franka_host.sh"
echo "  5. Then test motion with:"
echo "     source ${ROOT_DIR}/.venv/bin/activate"
echo "     python3 ${ROOT_DIR}/scripts/robot/minimal_franka_move.py --ip 172.16.0.2 --dz 0.005 --speed 0.02 --realtime ignore"
