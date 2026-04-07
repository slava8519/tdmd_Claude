# standalone driver
add_executable(tdmd_standalone tdmd_main.cpp)
target_link_libraries(tdmd_standalone PRIVATE tdmd::core)
set_target_properties(tdmd_standalone PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
