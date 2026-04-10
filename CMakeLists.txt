# TDMD — Time Decomposition Molecular Dynamics
# Root CMake build
cmake_minimum_required(VERSION 3.24)

project(tdmd
  VERSION 0.0.1
  DESCRIPTION "Time-Decomposition Molecular Dynamics engine"
  LANGUAGES CXX
)

# ---------------- Build options ----------------
option(TDMD_BUILD_TESTS   "Build unit tests"           ON)
option(TDMD_ENABLE_CUDA   "Enable CUDA kernels"        OFF)  # turned ON at M2
option(TDMD_ENABLE_MPI    "Enable MPI ring comm"       OFF)  # turned ON at M5
set(TDMD_PRECISION "mixed" CACHE STRING "Precision mode: mixed (LAMMPS-style) or fp64 (all-double reference)")
set_property(CACHE TDMD_PRECISION PROPERTY STRINGS mixed fp64)
option(TDMD_BUILD_BENCHMARKS "Build benchmark tools"  OFF)
option(TDMD_DETERMINISTIC "Force deterministic mode"   OFF)
option(TDMD_DETERMINISTIC_REDUCE "Use ID-ordered reductions for bit-reproducibility (ADR 0010)" OFF)

# Backward compat: -DTDMD_FP64=ON → TDMD_PRECISION=fp64
if(DEFINED TDMD_FP64 AND TDMD_FP64)
  message(WARNING "TDMD_FP64 is deprecated. Use -DTDMD_PRECISION=fp64 instead.")
  set(TDMD_PRECISION "fp64" CACHE STRING "" FORCE)
endif()

# Validate TDMD_PRECISION value
if(NOT TDMD_PRECISION STREQUAL "mixed" AND NOT TDMD_PRECISION STREQUAL "fp64")
  message(FATAL_ERROR "TDMD_PRECISION must be 'mixed' or 'fp64', got '${TDMD_PRECISION}'")
endif()

# ---------------- C++ standard ----------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Emit compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerFlags)

# ---------------- Optional CUDA ----------------
if(TDMD_ENABLE_CUDA)
  # Set architectures BEFORE enable_language(CUDA), so CMake doesn't
  # auto-detect a wrong default.
  include(CUDAConfig)
  enable_language(CUDA)
endif()

# ---------------- Optional MPI ----------------
if(TDMD_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

# ---------------- Subdirectories ----------------
add_subdirectory(src)

if(TDMD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(TDMD_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

# ---------------- Status summary ----------------
message(STATUS "")
message(STATUS "===== TDMD build configuration =====")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:             ${TDMD_ENABLE_CUDA}")
message(STATUS "  MPI:              ${TDMD_ENABLE_MPI}")
message(STATUS "  Precision:        ${TDMD_PRECISION}")
message(STATUS "  Deterministic:    ${TDMD_DETERMINISTIC}")
message(STATUS "  Det. reductions:  ${TDMD_DETERMINISTIC_REDUCE}")
message(STATUS "  Tests:            ${TDMD_BUILD_TESTS}")
message(STATUS "  Benchmarks:       ${TDMD_BUILD_BENCHMARKS}")
message(STATUS "====================================")
message(STATUS "")
