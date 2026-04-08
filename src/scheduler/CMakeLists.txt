# scheduler module — zone state machine, TD pipeline coordinator
if(TDMD_ENABLE_CUDA AND TDMD_ENABLE_MPI)
  add_library(tdmd_scheduler STATIC
    sequential_scheduler.cpp
    stream_pool.cu
    pipeline_scheduler.cu
    distributed_pipeline_scheduler.cu
    hybrid_pipeline_scheduler.cu
  )
  set_target_properties(tdmd_scheduler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(tdmd_scheduler PUBLIC tdmd::core tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator tdmd::comm)
elseif(TDMD_ENABLE_CUDA)
  add_library(tdmd_scheduler STATIC
    sequential_scheduler.cpp
    stream_pool.cu
    pipeline_scheduler.cu
  )
  set_target_properties(tdmd_scheduler PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  target_link_libraries(tdmd_scheduler PUBLIC tdmd::core tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator)
else()
  add_library(tdmd_scheduler STATIC
    sequential_scheduler.cpp
  )
  target_link_libraries(tdmd_scheduler PUBLIC tdmd::core tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator)
endif()
target_include_directories(tdmd_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
add_library(tdmd::scheduler ALIAS tdmd_scheduler)
