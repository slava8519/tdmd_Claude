#!/usr/bin/env bash
# Generate FCC Cu data files for Phase 1 benchmarks.
# Uses existing tools/gen_fcc_data.py.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN="${SCRIPT_DIR}/../../tools/gen_fcc_data.py"

if [ ! -f "${GEN}" ]; then
  echo "Error: gen_fcc_data.py not found at ${GEN}"
  exit 1
fi

# tiny: symlink to existing test data (256 atoms, 4x4x4)
TINY_SRC="${SCRIPT_DIR}/../../tests/data/cu_fcc_256.data"
if [ -f "${TINY_SRC}" ]; then
  ln -sf "${TINY_SRC}" "${SCRIPT_DIR}/tiny.data" 2>/dev/null || \
    cp "${TINY_SRC}" "${SCRIPT_DIR}/tiny.data"
  echo "tiny:   256 atoms (symlink to tests/data/cu_fcc_256.data)"
else
  python3 "${GEN}" --elem Cu --a 3.615 --nx 4 --ny 4 --nz 4 -o "${SCRIPT_DIR}/tiny.data"
  echo "tiny:   256 atoms (generated)"
fi

# small: 4000 atoms (10x10x10)
python3 "${GEN}" --elem Cu --a 3.615 --nx 10 --ny 10 --nz 10 -o "${SCRIPT_DIR}/small.data"
echo "small:  4000 atoms"

# medium: 32000 atoms (20x20x20)
python3 "${GEN}" --elem Cu --a 3.615 --nx 20 --ny 20 --nz 20 -o "${SCRIPT_DIR}/medium.data"
echo "medium: 32000 atoms"

echo ""
echo "Generated data files:"
ls -lh "${SCRIPT_DIR}"/*.data
