#!/usr/bin/env bash
# Run TDMD unit tests.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "[test] no build dir, running build first"
  "${ROOT_DIR}/scripts/build.sh"
fi

ctest --test-dir "${BUILD_DIR}" --output-on-failure "$@"
