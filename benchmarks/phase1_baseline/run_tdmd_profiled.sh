#!/usr/bin/env bash
# Run TDMD benchmark on a Phase 1 baseline system, capture nsys + ncu traces.
# Usage: ./run_tdmd_profiled.sh <tiny|small|medium> [extra bench args...]
set -euo pipefail

SIZE="${1:?Usage: run_tdmd_profiled.sh <tiny|small|medium> [extra args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA="${SCRIPT_DIR}/${SIZE}.data"
OUT_DIR="${SCRIPT_DIR}/results/${SIZE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

# Find benchmark binary. Check common build locations.
BENCH_BIN=""
for candidate in \
    "${SCRIPT_DIR}/../../build/benchmarks/bench_pipeline_scheduler" \
    "${SCRIPT_DIR}/../../build/benchmarks/phase1_baseline/bench_pipeline_scheduler" \
    "${SCRIPT_DIR}/../../build/bench_pipeline_scheduler"; do
  if [ -x "${candidate}" ]; then
    BENCH_BIN="${candidate}"
    break
  fi
done

if [ -z "${BENCH_BIN}" ]; then
  echo "Error: bench_pipeline_scheduler not found. Build with -DTDMD_BUILD_BENCHMARKS=ON"
  exit 1
fi

if [ ! -f "${DATA}" ]; then
  echo "Error: ${DATA} not found. Run ./generate_data.sh first."
  exit 1
fi

# Default steps per size.
case "${SIZE}" in
  tiny)   STEPS=2000 ;;
  small)  STEPS=1000 ;;
  medium) STEPS=500  ;;
  *)      STEPS=1000 ;;
esac

echo "=== TDMD Phase 1 benchmark: ${SIZE} ==="
echo "Data:    ${DATA}"
echo "Steps:   ${STEPS}"
echo "Output:  ${OUT_DIR}"
echo ""

# 1. Plain run — timesteps/s and telemetry JSON.
echo "--- Step 1: plain run ---"
"${BENCH_BIN}" --data "${DATA}" --steps "${STEPS}" --warmup 100 \
    --output "${OUT_DIR}/results.json" "$@" \
    2>"${OUT_DIR}/plain.stderr"
cat "${OUT_DIR}/plain.stderr"
echo ""

# 2. nsys trace — full timeline, kernel launches.
echo "--- Step 2: nsys profile ---"
if command -v nsys &>/dev/null; then
  nsys profile \
      --output="${OUT_DIR}/trace" \
      --trace=cuda,osrt \
      --force-overwrite=true \
      --stats=true \
      "${BENCH_BIN}" --data "${DATA}" --steps "${STEPS}" --warmup 100 "$@" \
      2>&1 | tee "${OUT_DIR}/nsys.stdout"
  echo ""
else
  echo "nsys not found in PATH, skipping trace."
fi

# 3. ncu per-kernel profile (optional, slow on large systems).
echo "--- Step 3: ncu profile (10 launches) ---"
if command -v ncu &>/dev/null; then
  ncu \
      --target-processes all \
      --kernel-name-base demangled \
      --launch-count 10 \
      --set full \
      --export "${OUT_DIR}/ncu_profile" \
      --force-overwrite \
      "${BENCH_BIN}" --data "${DATA}" --steps 50 --warmup 10 "$@" \
      2>&1 | tee "${OUT_DIR}/ncu.stdout"
  echo ""
else
  echo "ncu not found in PATH, skipping kernel profiling."
fi

echo ""
echo "=== Results in ${OUT_DIR} ==="
ls -la "${OUT_DIR}"
