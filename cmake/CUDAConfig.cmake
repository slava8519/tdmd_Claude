# CUDA configuration for TDMD.
# RTX 5080 = sm_120 (Blackwell). CUDA 12.6 doesn't know sm_120 natively,
# so we use compute_90 PTX which the driver JITs for sm_120.

if(NOT CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Generate native code for sm_89 (Ada) + sm_90 (Hopper),
# plus PTX for compute_90 (forward-compatible with Blackwell sm_120 via JIT).
# Set unconditionally BEFORE enable_language(CUDA) to prevent auto-detection.
set(CMAKE_CUDA_ARCHITECTURES "89-real;90")

# Allow CUDA files to include host headers.
# Suppress warnings from CUDA runtime headers (old-style casts etc.).
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler -Wno-old-style-cast")

message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
