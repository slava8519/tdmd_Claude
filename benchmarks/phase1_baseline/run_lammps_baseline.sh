#!/usr/bin/env bash
# Run LAMMPS-GPU on the small baseline system for A/B comparison.
# Usage: ./run_lammps_baseline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_DIR="${SCRIPT_DIR}/lammps_small"
OUT_DIR="${CASE_DIR}/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

# Check that small.data exists (LAMMPS input references ../small.data).
if [ ! -f "${SCRIPT_DIR}/small.data" ]; then
  echo "Error: small.data not found. Run ./generate_data.sh first."
  exit 1
fi

# Default: LAMMPS built in third_party/lammps/build/
LMP_BIN="${LMP_BIN:-$(cd "$(dirname "$0")/../.." && pwd)/third_party/lammps/build/lmp}"

if ! command -v "${LMP_BIN}" &>/dev/null; then
  echo "Error: LAMMPS binary '${LMP_BIN}' not found."
  echo "Set LMP_BIN to the path of your LAMMPS executable."
  exit 1
fi

echo "=== LAMMPS Phase 1 baseline: small (4000 atoms) ==="
echo "Input:   ${CASE_DIR}/in.baseline"
echo "Output:  ${OUT_DIR}"
echo ""

# Plain run with GPU package.
echo "--- Step 1: plain run ---"
cd "${CASE_DIR}"
"${LMP_BIN}" -in in.baseline -log "${OUT_DIR}/lammps.log" \
    -sf gpu -pk gpu 1 \
    2>&1 | tee "${OUT_DIR}/lammps.stdout"
cd "${SCRIPT_DIR}"
echo ""

# nsys trace for timeline comparison.
echo "--- Step 2: nsys profile ---"
if command -v nsys &>/dev/null; then
  cd "${CASE_DIR}"
  nsys profile \
      --output="${OUT_DIR}/trace" \
      --trace=cuda,osrt \
      --force-overwrite=true \
      --stats=true \
      "${LMP_BIN}" -in in.baseline -log "${OUT_DIR}/nsys.log" \
          -sf gpu -pk gpu 1 \
      2>&1 | tee "${OUT_DIR}/nsys.stdout"
  cd "${SCRIPT_DIR}"
else
  echo "nsys not found in PATH, skipping trace."
fi

echo ""
echo "=== Results in ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
