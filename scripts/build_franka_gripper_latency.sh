#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
source_root="${repo_root}/dist/build_deps/libfranka-0.17.0"
output_dir="${repo_root}/dist/gripper_latency"
output_path="${output_dir}/franka_gripper_latency"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

if [[ ! -f "${source_root}/include/franka/gripper.h" ]]; then
  git clone --depth 1 --branch 0.17.0 --recurse-submodules --shallow-submodules \
    https://github.com/frankarobotics/libfranka.git "${source_root}"
fi

site_packages="$("${python_bin}" -c 'import pathlib, franky; print(pathlib.Path(franky.__file__).resolve().parent.parent)')"
franky_lib_dir="${site_packages}/franky_control.libs"
libfranka_library="$(find "${franky_lib_dir}" -maxdepth 1 -type f -name 'libfranka-*.so.0.17.0' -print -quit)"
if [[ -z "${libfranka_library}" ]]; then
  echo "franky does not contain the required libfranka 0.17.0 library" >&2
  exit 1
fi

mkdir -p "${output_dir}"
g++ -O2 -DNDEBUG -std=c++17 -pthread \
  -I"${source_root}/include" \
  -I"${source_root}/common/include" \
  "${repo_root}/native/franka_gripper_latency.cpp" \
  "${libfranka_library}" \
  -Wl,--disable-new-dtags,-rpath,"${franky_lib_dir}" \
  -o "${output_path}"

"${output_path}" --self-test
"${output_path}"
echo "Built: ${output_path}"
