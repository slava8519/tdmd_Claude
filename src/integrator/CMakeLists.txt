# integrator module — time integration
add_library(tdmd_integrator STATIC velocity_verlet.cpp)
target_include_directories(tdmd_integrator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_integrator PUBLIC tdmd::core)
add_library(tdmd::integrator ALIAS tdmd_integrator)
