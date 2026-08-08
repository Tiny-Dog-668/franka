from __future__ import annotations

import os
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent
LIBFRANKA_ROOT = Path(os.environ["PYLIBFRANKA_VENDORED_ROOT"]).resolve()
LIBFRANKA_LIBRARY = Path(os.environ["PYLIBFRANKA_LIBRARY"]).resolve()
PYTHON_INCLUDE_DIRS = [
    path for path in os.environ.get("PYLIBFRANKA_PYTHON_INCLUDE_DIRS", "").split(os.pathsep) if path
]

if not LIBFRANKA_LIBRARY.is_file():
    raise RuntimeError(f"Bundled pylibfranka library does not exist: {LIBFRANKA_LIBRARY}")


extension = Pybind11Extension(
    "pylibfranka_streaming_patch._native",
    ["src/native.cpp"],
    include_dirs=[
        str(LIBFRANKA_ROOT / "include"),
        str(LIBFRANKA_ROOT / "common" / "include"),
        *PYTHON_INCLUDE_DIRS,
    ],
    extra_objects=[str(LIBFRANKA_LIBRARY)],
    runtime_library_dirs=["$ORIGIN/../pylibfranka.libs"],
    cxx_std=17,
)


setup(ext_modules=[extension], cmdclass={"build_ext": build_ext})
