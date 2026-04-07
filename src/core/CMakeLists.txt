# core module — no dependencies, header-only at M0
add_library(tdmd_core_headers INTERFACE)
target_include_directories(tdmd_core_headers INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(tdmd_core_headers INTERFACE tdmd_flags)
add_library(tdmd::core ALIAS tdmd_core_headers)
