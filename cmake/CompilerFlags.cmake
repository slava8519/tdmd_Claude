# Common compiler flags for TDMD.
# Kept deliberately conservative: warnings as errors, no fancy sanitizers by default.

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

add_library(tdmd_flags INTERFACE)

target_compile_features(tdmd_flags INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tdmd_flags INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wconversion
    -Wsign-conversion
    -Wnull-dereference
    -Wdouble-promotion
    -Wformat=2
    -fno-omit-frame-pointer
  )
endif()

if(TDMD_ENABLE_CUDA)
  target_compile_definitions(tdmd_flags INTERFACE TDMD_ENABLE_CUDA=1)
endif()
if(TDMD_FP64)
  target_compile_definitions(tdmd_flags INTERFACE TDMD_FP64=1)
endif()
if(TDMD_DETERMINISTIC)
  target_compile_definitions(tdmd_flags INTERFACE TDMD_DETERMINISTIC=1)
endif()
