# CUDA configuration for TDMD.
# Currently a stub — filled in at M2 when CUDA kernels land.

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Dev box: RTX 5080 = sm_120 (Blackwell).
# Also target a couple of common architectures for portability.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;120")
endif()

message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
