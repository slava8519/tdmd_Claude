# scheduler module — zone state machine, TD pipeline coordinator
add_library(tdmd_scheduler STATIC
  sequential_scheduler.cpp
)
target_include_directories(tdmd_scheduler PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_scheduler PUBLIC tdmd::core tdmd::domain tdmd::neighbors tdmd::potentials tdmd::integrator)
add_library(tdmd::scheduler ALIAS tdmd_scheduler)
