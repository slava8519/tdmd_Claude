#!/usr/bin/env bash
# Format C++/CUDA sources with clang-format and Python with black (if installed).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "[format] clang-format on src/ tests/"
find src tests -type f \( -name "*.hpp" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.h" \) \
  -exec clang-format -i --style=file {} +

if command -v black >/dev/null 2>&1; then
  echo "[format] black on verifylab/ scripts/"
  black --quiet verifylab/ scripts/ 2>/dev/null || true
fi

echo "[format] done"
