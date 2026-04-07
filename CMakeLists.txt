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
option(TDMD_FP64          "Build in FP64-only mode"    OFF)
option(TDMD_DETERMINISTIC "Force deterministic mode"   OFF)

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
  enable_language(CUDA)
  include(CUDAConfig)
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

# ---------------- Status summary ----------------
message(STATUS "")
message(STATUS "===== TDMD build configuration =====")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA:             ${TDMD_ENABLE_CUDA}")
message(STATUS "  MPI:              ${TDMD_ENABLE_MPI}")
message(STATUS "  FP64 only:        ${TDMD_FP64}")
message(STATUS "  Deterministic:    ${TDMD_DETERMINISTIC}")
message(STATUS "  Tests:            ${TDMD_BUILD_TESTS}")
message(STATUS "====================================")
message(STATUS "")
