#!/usr/bin/env bash
# Out-of-tree rebuild of the fs_io_C extension (CPU-only, no CMake, no GPU).
#
# Mirrors CMakeLists.txt: define_extension_target(fs_io_C ... USE_SABI 3.11
# WITH_SOABI) -> raw CPython C-API, stable ABI >= 3.11, suffix .abi3.so.
# Needed on a checkout installed with VLLM_USE_PRECOMPILED=1: that install
# drops the wheel's fs_io_C.abi3.so into vllm/ and never compiles csrc/.
# On aarch64 the CRC32 instructions are enabled (-march=armv8-a+crc).
#
# Usage:
#   csrc/build_fs_io.sh [out.so]        # default out: vllm/fs_io_C.abi3.so
#   PYTHON=/path/to/python csrc/build_fs_io.sh
#
# Rebuild while the server is stopped; a running process keeps the old .so
# mapped and only the next start picks up the new file.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
OUT="${1:-vllm/fs_io_C.abi3.so}"
INC="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
ARCH_FLAGS=""
[ "$(uname -m)" = aarch64 ] && ARCH_FLAGS="-march=armv8-a+crc"
# macOS (local development only) resolves the Python symbols at load time.
[ "$(uname -s)" = Darwin ] && ARCH_FLAGS="$ARCH_FLAGS -undefined dynamic_lookup"
# shellcheck disable=SC2086 # ARCH_FLAGS holds separate flags or is empty.
g++ -O2 -shared -fPIC -std=c++17 -DPy_LIMITED_API=0x030b0000 $ARCH_FLAGS \
  -I"$INC" csrc/fs_io.cpp -o "$OUT"
echo "built $OUT"
