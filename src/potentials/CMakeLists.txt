# potentials module — pair and many-body force evaluation
add_library(tdmd_potentials STATIC force_compute.cpp eam_alloy.cpp)
target_include_directories(tdmd_potentials PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_potentials PUBLIC tdmd::core tdmd::neighbors)
add_library(tdmd::potentials ALIAS tdmd_potentials)
