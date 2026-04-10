#!/usr/bin/env bash
# Run VerifyLab physics-validation cases.
# Usage:
#   ./scripts/run-verifylab.sh                      # all cases, mixed mode
#   ./scripts/run-verifylab.sh --mode fp64          # all cases, fp64 mode
#   ./scripts/run-verifylab.sh --case two-atoms-morse
#   ./scripts/run-verifylab.sh --suite fast
#   ./scripts/run-verifylab.sh --suite slow
#
# --mode selects the build directory and precision tolerance column.
# Other flags are passed through to verifylab/runners/run_all.py.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="mixed"
RUNNER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --mode=*)
      MODE="${1#--mode=}"
      shift
      ;;
    *)
      RUNNER_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${MODE}" in
  mixed) BUILD_DIR="${ROOT_DIR}/build-mixed" ;;
  fp64)  BUILD_DIR="${ROOT_DIR}/build-fp64"  ;;
  *)
    echo "[verifylab] unknown --mode '${MODE}' (expected mixed|fp64)" >&2
    exit 2
    ;;
esac

TDMD_BIN="${BUILD_DIR}/tdmd_standalone"
if [ ! -x "${TDMD_BIN}" ]; then
  echo "[verifylab] tdmd_standalone not built at ${TDMD_BIN}" >&2
  echo "[verifylab] build first: cmake --build ${BUILD_DIR} --target tdmd_standalone" >&2
  exit 2
fi

export TDMD_BIN
export TDMD_MODE="${MODE}"

echo "[verifylab] mode=${MODE}  bin=${TDMD_BIN}"
cd "${ROOT_DIR}"
python3 verifylab/runners/run_all.py "${RUNNER_ARGS[@]}"
