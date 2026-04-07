# neighbors module — Verlet list with skin
add_library(tdmd_neighbors STATIC neighbor_list.cpp)
target_include_directories(tdmd_neighbors PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(tdmd_neighbors PUBLIC tdmd::core tdmd::domain)
add_library(tdmd::neighbors ALIAS tdmd_neighbors)
