#!/usr/bin/env bash
# Build TDMD.
#
# This is a CPU-only convenience script. For CUDA, MPI, FP64, or benchmark
# builds, use cmake directly. See docs/04-development/build-and-run.md.
#
# Usage:
#   ./scripts/build.sh                  # default Release build (CPU-only)
#   ./scripts/build.sh Debug            # debug build
#   ./scripts/build.sh Release clean    # clean rebuild
#
# For CUDA/MPI builds, use cmake directly:
#   cmake -B build -S . -G Ninja -DCMAKE_BUILD_TYPE=Release \
#       -DTDMD_ENABLE_CUDA=ON -DTDMD_BUILD_TESTS=ON
#   cmake --build build -j
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_TYPE="${1:-Release}"
CLEAN="${2:-}"

BUILD_DIR="${ROOT_DIR}/build"

if [ "${CLEAN}" = "clean" ]; then
  echo "[build] cleaning ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

echo "[build] type=${BUILD_TYPE} (CPU-only; use cmake directly for CUDA/MPI)"
echo "[build] root=${ROOT_DIR}"

cmake -B "${BUILD_DIR}" -S "${ROOT_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DTDMD_BUILD_TESTS=ON

cmake --build "${BUILD_DIR}" -j

# Symlink compile_commands.json to root for clangd / IDE
ln -sf "${BUILD_DIR}/compile_commands.json" "${ROOT_DIR}/compile_commands.json"

echo "[build] done -> ${BUILD_DIR}"
