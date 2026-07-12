#!/usr/bin/env bash
set -euo pipefail

TARGET='Advanced options for Ubuntu>Ubuntu, with Linux 6.8.0-106-generic'
GRUB_FILE='/etc/default/grub'
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Please run as root:"
  echo "  sudo bash ${ROOT_DIR}/tools/system/set_default_generic_kernel.sh"
  exit 1
fi

if ! grep -q "menuentry 'Ubuntu, with Linux 6.8.0-106-generic'" /boot/grub/grub.cfg; then
  echo "Target kernel entry not found in /boot/grub/grub.cfg"
  exit 1
fi

backup="${GRUB_FILE}.bak_codex_$(date +%Y%m%d_%H%M%S)"
cp "${GRUB_FILE}" "${backup}"

if grep -q '^GRUB_DEFAULT=' "${GRUB_FILE}"; then
  sed -i "s|^GRUB_DEFAULT=.*|GRUB_DEFAULT=\"${TARGET}\"|" "${GRUB_FILE}"
else
  printf '\nGRUB_DEFAULT="%s"\n' "${TARGET}" >> "${GRUB_FILE}"
fi

update-grub

echo "Updated ${GRUB_FILE}"
echo "Backup saved to ${backup}"
echo "Default boot target is now:"
echo "  ${TARGET}"
echo
echo "Reboot to apply:"
echo "  sudo reboot"
