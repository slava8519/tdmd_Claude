#!/usr/bin/env bash
# Run VerifyLab physics-validation cases.
# Usage:
#   ./scripts/run-verifylab.sh                 # all cases in 'fast' suite
#   ./scripts/run-verifylab.sh --suite full    # all cases
#   ./scripts/run-verifylab.sh --case nve-drift
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -x "${ROOT_DIR}/build/tdmd_standalone" ]; then
  echo "[verifylab] tdmd_standalone not built; running build"
  "${ROOT_DIR}/scripts/build.sh"
fi

cd "${ROOT_DIR}"
python3 verifylab/runners/run_all.py "$@"
