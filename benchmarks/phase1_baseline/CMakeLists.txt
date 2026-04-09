# benchmarks/phase1_baseline/CMakeLists.txt
# Phase 1 measurement tools for ADR 0005.
# Builds only when -DTDMD_BUILD_BENCHMARKS=ON and CUDA is enabled.

if(NOT TDMD_ENABLE_CUDA)
  message(WARNING "phase1_baseline benchmarks require CUDA; skipping")
  return()
endif()

add_executable(bench_pipeline_scheduler bench_pipeline_scheduler.cu)
target_link_libraries(bench_pipeline_scheduler PRIVATE
  tdmd::core tdmd::io tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::scheduler
)
set_target_properties(bench_pipeline_scheduler PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmarks"
)
