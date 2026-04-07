# core module — header-only on CPU, adds .cu sources when CUDA is enabled.

if(TDMD_ENABLE_CUDA)
  add_library(tdmd_core_impl STATIC
    device_system_state.cu
  )
  target_include_directories(tdmd_core_impl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
  target_link_libraries(tdmd_core_impl PUBLIC tdmd_flags)
  set_target_properties(tdmd_core_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
  add_library(tdmd::core ALIAS tdmd_core_impl)
else()
  add_library(tdmd_core_headers INTERFACE)
  target_include_directories(tdmd_core_headers INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
  )
  target_link_libraries(tdmd_core_headers INTERFACE tdmd_flags)
  add_library(tdmd::core ALIAS tdmd_core_headers)
endif()
