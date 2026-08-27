#!/usr/bin/env bash
# Out-of-tree rebuild of the fs_io_C extension (CPU-only, no CMake, no GPU).
#
# Mirrors CMakeLists.txt: define_extension_target(fs_io_C ... USE_SABI 3.11
# WITH_SOABI) -> raw CPython C-API, stable ABI >= 3.11, suffix .abi3.so.
# One aarch64 build serves every node on the same libc/arch.
#
# Usage:
#   ./build_fs_io.sh [out.so]            # default out: ./fs_io_C.abi3.so
#   PYTHON=/path/to/python ./build_fs_io.sh
#
# Install (safe while the server runs; the mapped old .so stays in memory,
# the new file takes effect at the next controlled restart):
#   cp fs_io_C.abi3.so <venv>/lib/python3.12/site-packages/vllm/fs_io_C.abi3.so
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"
OUT="${1:-fs_io_C.abi3.so}"
INC="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
g++ -O2 -shared -fPIC -std=c++17 -DPy_LIMITED_API=0x030b0000 \
  -I"$INC" fs_io.cpp -o "$OUT"
echo "built $OUT"
