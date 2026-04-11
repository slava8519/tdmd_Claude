#!/usr/bin/env bash
# Phase C LAMMPS-vs-TDMD throughput sweep.
#
# Runs TDMD and LAMMPS sequentially on the exact same 6 cells
# (Morse/EAM × tiny/small/medium) with 7-second pauses between each
# run, honoring CLAUDE.md §4 (GPU-exclusive, no parallel runs).
#
# Layout of the sweep:
#   For each (pot, size) in tiny/small/medium × morse/eam:
#     TDMD run 1 → sleep 7
#     TDMD run 2 → sleep 7
#     TDMD run 3 → sleep 7
#     LAMMPS run 1 → sleep 7
#     LAMMPS run 2 → sleep 7
#     LAMMPS run 3 → sleep 7
#
# Interleaving TDMD and LAMMPS *within* each cell keeps them in the
# same thermal/clock-boost regime, so the reported ratios are immune
# to the cross-session drift we documented in feat-eam-pipeline-results.md.
#
# TDMD numbers come from the same `bench_pipeline_scheduler` binary used
# in the FEAT-EAM sweep, so the protocol is identical. LAMMPS "Loop time
# of X.XX on 1 procs for N steps" is parsed from the SECOND `run` block
# (the first `run` is warmup, then `reset_timestep 0`, then the measured
# block).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TDMD_BIN="${TDMD_BIN:-${REPO_ROOT}/build-mixed/benchmarks/bench_pipeline_scheduler}"
LMP_BIN="${LMP_BIN:-${REPO_ROOT}/third_party/lammps/build/lmp}"
EAM_FILE="${REPO_ROOT}/tests/data/Cu_mishin1.eam.alloy"

# Per-size TDMD data files and step counts — must match the FEAT-EAM
# benchmark protocol exactly, otherwise these numbers cannot be
# cross-referenced against feat-eam-pipeline-results.md.
data_file_for() {
  case "$1" in
    tiny)   echo "${REPO_ROOT}/tests/data/cu_fcc_256.data" ;;
    small)  echo "${REPO_ROOT}/benchmarks/phase1_baseline/small.data" ;;
    medium) echo "${REPO_ROOT}/benchmarks/phase1_baseline/medium.data" ;;
  esac
}
steps_for()  { case "$1" in tiny) echo 5000 ;; small) echo 2000 ;; medium) echo 1000 ;; esac; }
warmup_for() { case "$1" in tiny) echo 500  ;; small) echo 200  ;; medium) echo 100  ;; esac; }

OUT_DIR="${SCRIPT_DIR}/results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

[ -x "${TDMD_BIN}" ] || { echo "TDMD_BIN not found: ${TDMD_BIN}"; exit 1; }
[ -x "${LMP_BIN}"  ] || { echo "LMP_BIN not found: ${LMP_BIN}"; exit 1; }

PAUSE=7
RUNS=3

run_tdmd() {
  local pot=$1 size=$2 run_idx=$3
  local label="tdmd_${pot}_${size}_run${run_idx}"
  local data
  data=$(data_file_for "${size}")
  local extra=()
  if [ "${pot}" = "eam" ]; then
    extra=(--potential eam --eam "${EAM_FILE}")
  fi
  echo "[$(date +%H:%M:%S)] TDMD ${pot} ${size} run ${run_idx}"
  "${TDMD_BIN}" \
      --scheduler fast_pipeline \
      --data "${data}" \
      --steps "$(steps_for "${size}")" \
      --warmup "$(warmup_for "${size}")" \
      "${extra[@]}" \
      --output "${OUT_DIR}/${label}.json" \
      > "${OUT_DIR}/${label}.stdout" 2>&1 || {
        echo "FAIL ${label}"; return 1; }
}

run_lammps() {
  local pot=$1 size=$2 run_idx=$3
  local label="lammps_${pot}_${size}_run${run_idx}"
  local input="${SCRIPT_DIR}/${pot}_${size}.in"
  echo "[$(date +%H:%M:%S)] LAMMPS ${pot} ${size} run ${run_idx}"
  (
    cd "${SCRIPT_DIR}"
    "${LMP_BIN}" \
        -in "${input}" \
        -log "${OUT_DIR}/${label}.log" \
        -sf gpu -pk gpu 1 \
        > "${OUT_DIR}/${label}.stdout" 2>&1
  ) || {
    echo "FAIL ${label}"; return 1; }
}

cells=(
  "morse tiny"
  "morse small"
  "morse medium"
  "eam tiny"
  "eam small"
  "eam medium"
)

for cell in "${cells[@]}"; do
  pot=${cell%% *}
  size=${cell##* }
  echo ""
  echo "=== ${pot} ${size} ==="
  for r in $(seq 1 ${RUNS}); do
    run_tdmd "${pot}" "${size}" "${r}"
    sleep ${PAUSE}
  done
  for r in $(seq 1 ${RUNS}); do
    run_lammps "${pot}" "${size}" "${r}"
    sleep ${PAUSE}
  done
done

echo ""
echo "=== done ==="
echo "results in: ${OUT_DIR}"
ls "${OUT_DIR}" | head -40
