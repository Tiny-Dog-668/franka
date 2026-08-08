#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vendored_root="${repo_root}/third_party/upstream_libfranka"
source_dir="${repo_root}/third_party/pylibfranka_streaming_patch"
wheel_dir="${repo_root}/dist/pylibfranka"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 1
fi

if [[ ! -f "${vendored_root}/common/include/research_interface/robot/service_types.h" ]]; then
  git -C "${vendored_root}" submodule update --init --depth 1 common
fi

installed_version="$("${python_bin}" -c 'import pylibfranka; print(pylibfranka.__version__)')"
if [[ "${installed_version}" != "0.21.1" ]]; then
  echo "Expected installed pylibfranka 0.21.1, got ${installed_version}" >&2
  exit 1
fi

site_packages="$("${python_bin}" -c 'import pathlib, pylibfranka; print(pathlib.Path(pylibfranka.__file__).resolve().parent.parent)')"
libfranka_library="$(find "${site_packages}/pylibfranka.libs" -maxdepth 1 -type f -name 'libfranka-*.so.0.21.1' -print -quit)"
if [[ -z "${libfranka_library}" ]]; then
  echo "The installed pylibfranka wheel does not contain libfranka 0.21.1" >&2
  exit 1
fi

python_include_dir="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
python_include_dirs="${python_include_dir}"
if [[ ! -f "${python_include_dir}/Python.h" ]]; then
  python_dev_root="${repo_root}/dist/build_deps/python3.10-dev"
  python_header_dir="${python_dev_root}/usr/include/python3.10"
  python_include_root="${python_dev_root}/usr/include"
  if [[ ! -f "${python_header_dir}/Python.h" ]]; then
    download_dir="$(mktemp -d)"
    mkdir -p "${python_dev_root}"
    (
      cd "${download_dir}"
      apt-get download libpython3.10-dev
      dpkg-deb -x libpython3.10-dev_*.deb "${python_dev_root}"
    )
  fi
  python_include_dirs="${python_header_dir}:${python_include_root}"
fi

mkdir -p "${wheel_dir}"
PYLIBFRANKA_VENDORED_ROOT="${vendored_root}" \
PYLIBFRANKA_LIBRARY="${libfranka_library}" \
PYLIBFRANKA_PYTHON_INCLUDE_DIRS="${python_include_dirs}" \
  "${python_bin}" -m pip wheel --no-deps --wheel-dir "${wheel_dir}" "${source_dir}"

if [[ "${1:-}" == "--install" ]]; then
  wheel_path="$(find "${wheel_dir}" -maxdepth 1 -type f -name 'pylibfranka_streaming_patch-0.21.1.1-*.whl' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -z "${wheel_path}" ]]; then
    echo "Built wheel was not found in ${wheel_dir}" >&2
    exit 1
  fi
  "${python_bin}" -m pip install --force-reinstall --no-deps "${wheel_path}"
  "${python_bin}" -c 'import pylibfranka as p; from pylibfranka_streaming_patch import install; install(p); assert p.__version__ == "0.21.1"; assert hasattr(p.AsyncPositionControlHandler, "read_once"); assert hasattr(p, "TargetStatus"); print("pylibfranka streaming API: PASS")'
fi
